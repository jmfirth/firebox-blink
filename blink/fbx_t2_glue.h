#ifndef BLINK_FBX_T2_GLUE_H_
#define BLINK_FBX_T2_GLUE_H_

/*
 * Firebox ELF-perf Phase 2 — Tier 2 §13.4: Blink ↔ host glue.
 *
 * This header is the C-side of the Tier 2 wire contract.  It declares the
 * five host-import symbols Blink calls into when escalating a Tier 1 block
 * to Tier 2 dispatch, plus the small set of Blink-internal helpers
 * (`Fbxt2TryEscalate`, `Fbxt2HotnessThreshold`, `Fbxt2Enabled`) that
 * `ExecuteBlock` uses to drive the §6.4 fast-path described in
 * `work/tracks/elf-performance/phase-2-hot-path-caching/tier-2-spec.md`.
 *
 * Why the host-import shims are declared `extern "C"` with weak default
 * bodies in `fbx_t2_glue.c`:
 *
 *   - When Blink is compiled to wasm (production target), undefined
 *     `extern` symbols become wasm imports.  The wasmer-backed firebox
 *     runtime registers the corresponding host functions on the Blink
 *     module's import object (see `crates/firebox-wasix/src/t2_bridge.rs`
 *     `T2Bridge::build_host_imports` for the bridge side — these are
 *     the C-side counterparts).  Spec §11.1 + §11.2.
 *
 *   - When Blink is compiled native (for unit tests, the
 *     ExecuteBlock/escalate path needs to be exercisable on the dev
 *     machine), strong overrides supplied by the test bench replace
 *     the weak stubs at link time.  This is how `fbx_t2_e2e_test.c`
 *     drives the dispatch path with a synthetic Tier 2 engine.
 *
 *   - When Blink is compiled native WITHOUT a test bench supplying
 *     overrides (e.g., `blink/blink` standalone), the weak stubs are
 *     called.  They behave as "T2 permanently disabled": instantiate
 *     returns -1, dispatch returns 1 (bailout), the rest are no-ops.
 *     This means `Fbxt2TryEscalate` always fails fast, no escalation
 *     ever sticks, and ExecuteBlock degrades cleanly to the Tier 1
 *     path.  Standalone Blink keeps working exactly as before #589.
 *
 * The five shims' signatures MUST match byte-for-byte the wasm-host
 * functions registered by `T2Bridge::build_host_imports` on the
 * firebox-wasix side AND the `extern "C"` shims at
 * `crates/firebox-wasix/src/t2_bridge.rs:782-859`.  Drift between any
 * of these surfaces is the failure mode #591 reconciled — keep them
 * lock-stepped.
 *
 * AGENTS.md invariant 4: this header is identical across native +
 * wasm builds.  The only thing that varies is which copy of the
 * `fbx_t2_*` symbols the linker resolves to.
 */

#include <stdbool.h>

#include "blink/types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Machine;
struct FbxTcBlock;
struct FbxT2BlockCtx;

/* ────────────────────────────────────────────────────────────────────────── */
/* Host-import shims — declared `extern` so the linker resolves them to     */
/* either (a) wasm imports when Blink is compiled to wasm, (b) test         */
/* overrides when fbx_t2_e2e_test.c is linked in, or (c) the weak defaults  */
/* in fbx_t2_glue.c when Blink is built standalone.                         */
/*                                                                          */
/* Signatures pin-point identical to                                        */
/* crates/firebox-wasix/src/t2_bridge.rs:782-859 — DO NOT drift without    */
/* a parallel update on the Rust side (spec §11.2 + §591 reconciliation). */
/*                                                                          */
/* `FBX_T2_HOST_IMPORT` decoration (#607): on wasm builds we attach        */
/* `__attribute__((import_module("fbx")))` so wasi-sdk/clang emits the     */
/* externs as wasm imports under the `"fbx"` module name (spec §11.2),    */
/* not the default `"env"` module.  Without this decoration wasi-sdk      */
/* placed undecorated externs under `"env"` and the weak local defaults   */
/* in `fbx_t2_glue.c` won the link silently — see                          */
/* `class_lesson_wasm_import_module_name_drift_between_spec_and_undecorated_extern`
 * (#604/#607 witness).  On native builds the decoration is empty so the */
/* three-way resolution from `class_lesson_weak_symbol_default_stub_for_three_way_host_shim_resolution`
 * still applies (test-bench strong override > weak default).             */
/* ────────────────────────────────────────────────────────────────────────── */

#ifdef __wasm__
#define FBX_T2_HOST_IMPORT __attribute__((import_module("fbx")))
#else
#define FBX_T2_HOST_IMPORT
#endif

/* Instantiate a wasm module fragment.  Returns a funcref index (>=0) or
 * -1 on failure (instantiation error, OOM, cap reached, T2 disabled).
 *
 * firebox#719 ABI: `block_ctx` points at a fully-built `FbxT2BlockCtx`
 * that the CALLER (Blink's `Fbxt2TryEscalate`) allocated + populated in
 * the GUEST's own heap.  On the wasm target a C pointer IS the i32
 * guest-memory offset, so the bridge receives `block_ctx` as the
 * `block_ctx_ptr` it records alongside the funcref and replays as the 2nd
 * `translated_block(m_ptr, block_ctx_ptr)` argument at every dispatch.
 *
 * This supersedes the pre-#719 `(const u64 *consts, u32 nconsts)` pair,
 * where the bridge copied the consts into a bridge-owned scratch memory.
 * That scratch memory is gone: the T2 module now imports the guest's REAL
 * (shared) linear memory, so both `m_ptr` (the `Machine` offset) and
 * `block_ctx` resolve against the SAME bytes the Tier-1 interpreter reads
 * (spec §5.2).  Writing a ctx at a bridge-chosen offset would clobber live
 * guest data — hence guest ownership.
 *
 * `block_ctx` MUST outlive every dispatch of the returned funcref (the
 * wasm reads it at dispatch time, not instantiate time).  The caller
 * stashes it on the block (`FbxTcBlock::t2_block_ctx`).  Layout + indexing
 * rules live in `blink/fbx_t2_block_ctx.h`; `block_ctx->nconsts` MUST be
 * ≤ `FBX_T2_BLOCK_CTX_MAX_CONSTS` (the caller's allocator enforces this).
 * `block_ctx` may be NULL only for the legacy zero-ctx path (the wasm
 * loads then read whatever is at offset 0 — used by tests, not production). */
FBX_T2_HOST_IMPORT
int fbx_t2_instantiate(u64 sys_id, const u8 *wasm_bytes, u32 wasm_len,
                       const struct FbxT2BlockCtx *block_ctx);

/* firebox#794 ASYNC ESCALATION — the host may dispatch a block's ~20 ms
 * cranelift compile to a BACKGROUND thread instead of blocking the guest hot
 * path, returning this sentinel to mean "compile in flight; stay Tier-1 and
 * re-attempt on a later hit."  Distinct from -1 ("terminal failure; never
 * escalate this block").  Keep in sync with `FBX_T2_INSTANTIATE_PENDING` in
 * `crates/firebox-wasix/src/t2_bridge.rs`.  This removes the run-1 cold-compile
 * cost that made T2-on a regression on short memory/branch-heavy workloads: a
 * short workload finishes at interpreter speed while the compile warms the
 * cache for the next hit / next run.  A host that compiles synchronously
 * (FBX_T2_ASYNC_COMPILE=0, or the browser host with no thread pool) never
 * returns this value, so the retry machinery below stays dormant there. */
#define FBX_T2_INSTANTIATE_PENDING (-2)

/* Re-attempt cadence for a PENDING block (firebox#794).  A self-loop is
 * dispatched once PER ITERATION, so retrying on every hit would re-run the
 * lift+emit pipeline millions of times inside one ~20 ms compile window.
 * Instead, re-attempt only every FBX_T2_PENDING_RETRY_STRIDE hits (tracked via
 * `FbxTcBlock::t2_retry_at_hits`), and give up after FBX_T2_PENDING_MAX_ATTEMPTS
 * polls (the compile, if it ever lands, still warms the cache for the next run).
 * Tuned so a deep compute loop catches its ready compile within a poll or two
 * (→ the ~10x win) while a short scan exhausts neither budget before the
 * workload ends (→ no escalation, no run-1 regression — the desired outcome for
 * the memory/branch-heavy class). */
#define FBX_T2_PENDING_RETRY_STRIDE 8192u
#define FBX_T2_PENDING_MAX_ATTEMPTS 64u

/* Dispatch the translated block.  Returns the wasm function's exit code
 * (0 = normal completion, 1 = bailout to Tier 1, 2 = host-call escape).
 * Spec §5.1 (translated_block return-value convention). */
FBX_T2_HOST_IMPORT
int fbx_t2_dispatch(u64 sys_id, i32 funcref, i32 m_ptr);

/* Drop all translated modules for this System.  Called from
 * FbxTcInvalidate (§7.1).  Epoch-aware on the host side — in-flight
 * dispatches keep their module alive until they return (spec §Q4). */
FBX_T2_HOST_IMPORT
void fbx_t2_drop_all(u64 sys_id);

/* Resolve an indirect-call target to a translated funcref, or -1 if
 * the target isn't a translated block.  v0.1 §Q2 option A treats this
 * as block-end-with-bailout, so the runtime path doesn't call this in
 * v0.1 — kept here so the wire contract is complete and v0.2 can light
 * it up without re-revving the import shape. */
FBX_T2_HOST_IMPORT
int fbx_t2_resolve_indirect(u64 sys_id, u64 target_pc);

/* Populate `*modules` and `*bytes` with the current cache occupancy
 * for this System.  Either pointer may be NULL to skip that field. */
FBX_T2_HOST_IMPORT
void fbx_t2_get_stats(u64 sys_id, u32 *modules, u64 *bytes);

/* ────────────────────────────────────────────────────────────────────────── */
/* Blink-internal helpers — these stay inside Blink and never need to be    */
/* overridden.  They orchestrate the §6.4 fast-path on top of the shims.    */
/* ────────────────────────────────────────────────────────────────────────── */

/* The Tier 2 hotness threshold (spec §6.1, §Q2).  Reads
 * `FBX_T2_HOTNESS_THRESHOLD` env var on first call; caches afterwards.
 * Default 100 (spec §Q2, calibrated from the Tier 1 v0.1 bench corpus). */
u32 Fbxt2HotnessThreshold(void);

/* Tier 2 kill-switch (spec §6.6).  Returns false when `FIREBOX_T2=0`
 * (or `=off`, `=false`, `=no`) is set in the environment.  Default
 * true — Tier 2 is on by default once a hot block escalates.  Cached
 * on first call. */
bool Fbxt2Enabled(void);

/* Attempt to escalate a Tier 1 block to Tier 2.  Walks the spec §6.2
 * pipeline: lift → emit_wasm → fbx_t2_instantiate → stash funcref on
 * the block.  Sets `b->t2_attempted = 1` whether escalation succeeded
 * or bailed; on success also sets `b->t2_funcref` to the host-returned
 * funcref (>=0).  Safe to call multiple times — the latch + funcref
 * stashing make subsequent calls a fast no-op. */
void Fbxt2TryEscalate(struct Machine *m, struct FbxTcBlock *b);

/* Dispatch a Tier-2-translated block.  Wraps `fbx_t2_dispatch` with the
 * §6.4 exit-code semantics.  Returns:
 *   0 — block dispatched + completed normally; caller returns
 *   1 — bailout: caller should fall through to the Tier 1 entries[]
 *       walk (the cached translation refused the block at runtime;
 *       Tier 1 has the canonical behavior)
 *   2 — host-call escape: caller returns; outer dispatcher resumes
 * The funcref MUST be the value previously stashed onto `b` by
 * `Fbxt2TryEscalate` (we don't validate it here for hot-path cost). */
int Fbxt2Dispatch(struct Machine *m, struct FbxTcBlock *b);

/* Test/diagnostic helper — resets the cached `Fbxt2HotnessThreshold`
 * and `Fbxt2Enabled` values so a subsequent call re-reads the env.
 * Production code does not need this; only the e2e test calls it. */
void Fbxt2ResetEnvCacheForTest(void);

#ifdef __cplusplus
}
#endif

#endif /* BLINK_FBX_T2_GLUE_H_ */
