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
 * -1 on failure (instantiation error, OOM, cap reached, T2 disabled). */
FBX_T2_HOST_IMPORT
int fbx_t2_instantiate(u64 sys_id, const u8 *wasm_bytes, u32 wasm_len);

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
