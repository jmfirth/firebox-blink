/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2026 Justin Firth (Firebox)                                        │
│                                                                              │
│ Phase 2 — Tier 2 §13.4: Blink ↔ host glue for the Tier 2 dispatch path.      │
│                                                                              │
│ Two things live here:                                                        │
│                                                                              │
│   1. Weak-default definitions of the five `fbx_t2_*` host-import shims.      │
│      These are the symbols Blink's wasm build emits as imports + the         │
│      symbols the native `blink` binary's linker resolves to the weak         │
│      bodies in this file.  See `fbx_t2_glue.h` for the design rationale.    │
│                                                                              │
│   2. The Blink-internal Tier-2 helpers `Fbxt2TryEscalate`,                   │
│      `Fbxt2Dispatch`, `Fbxt2HotnessThreshold`, `Fbxt2Enabled`.  These        │
│      compose lift → emit_wasm → fbx_t2_instantiate into the spec §6.2        │
│      pipeline + the §6.4 fast-path entry point that `ExecuteBlock`          │
│      calls.                                                                  │
│                                                                              │
│ Cross-reference:                                                             │
│   - blink/threadedcode.h FbxTcBlock — extended in #589 with t2_funcref +    │
│     t2_attempted fields                                                      │
│   - blink/threadedcode.c ExecuteBlock — the §6.4 fast-path patch site       │
│   - blink/fbx_ir.h + blink/fbx_wasm_emit.h — the producers this consumes    │
│   - crates/firebox-wasix/src/t2_bridge.rs — the Rust side of the wire       │
│   - work/tracks/elf-performance/phase-2-hot-path-caching/tier-2-spec.md     │
│     §5.4 (host-import contract), §6 (runtime escalation), §11 (host-API),  │
│     §13.4 (this task)                                                       │
╚─────────────────────────────────────────────────────────────────────────────*/

#include "blink/fbx_t2_glue.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "blink/fbx_ir.h"
#include "blink/fbx_t2_block_ctx.h"
#include "blink/fbx_wasm_emit.h"
#include "blink/machine.h"
#include "blink/threadedcode.h"

/* ────────────────────────────────────────────────────────────────────────── */
/* Trace flag — `FIREBOX_T2_TRACE=escalations,bailouts,verbose,all` per      */
/* spec §11.2 table.  Categories not in that table are silently dropped       */
/* (see `class_lesson_trace_silence_can_be_category_omission`).               */
/*                                                                            */
/* #668 closes the `verbose`-token silent-drop recurrence: previously the     */
/* token fell through to the silent-drop branch with no flag wired, so the    */
/* new [t2 synth-fail] line had no observer.  Also removes the dead toklen==12*/
/* compare-against-11-chars branch (always false, leftover from #667 probe).  */
/* ────────────────────────────────────────────────────────────────────────── */

static int g_t2_trace_init = 0;
static int g_t2_trace_escalations = 0;
static int g_t2_trace_bailouts = 0;
static int g_t2_trace_verbose = 0;
/* `dispatches` not currently used by this file — would fire on every Tier 2
 * dispatch, voluminous.  Bridge-side `t2_bridge.rs` owns that category. */

static void T2EnsureTraceFlags(void) {
  const char *raw;
  const char *tok;
  size_t toklen;
  if (g_t2_trace_init) return;
  g_t2_trace_init = 1;
  raw = getenv("FIREBOX_T2_TRACE");
  if (!raw || !*raw) return;
  /* Walk comma-separated tokens. */
  while (*raw) {
    tok = raw;
    while (*raw && *raw != ',') ++raw;
    toklen = (size_t)(raw - tok);
    while (toklen && (tok[0] == ' ' || tok[0] == '\t')) { ++tok; --toklen; }
    while (toklen && (tok[toklen - 1] == ' ' || tok[toklen - 1] == '\t')) --toklen;
    if (toklen == 11 && !strncmp(tok, "escalations", 11)) {
      g_t2_trace_escalations = 1;
    } else if (toklen == 8 && !strncmp(tok, "bailouts", 8)) {
      g_t2_trace_bailouts = 1;
    } else if (toklen == 7 && !strncmp(tok, "verbose", 7)) {
      g_t2_trace_verbose = 1;
    } else if (toklen == 3 && !strncmp(tok, "all", 3)) {
      g_t2_trace_escalations = 1;
      g_t2_trace_bailouts = 1;
      g_t2_trace_verbose = 1;
    }
    /* Unknown tokens silently dropped — matches the bridge-side parser. */
    if (*raw == ',') ++raw;
  }
}

static void T2TraceLine(int enabled, const char *fmt, ...) {
  char buf[200];
  va_list ap;
  int n;
  if (!enabled) return;
  va_start(ap, fmt);
  n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (n > 0) (void)write(2, buf, (size_t)n);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Cached env flags.  Both are read once on first call; the test bench        */
/* uses `Fbxt2ResetEnvCacheForTest` to force a re-read between scenarios.    */
/* ────────────────────────────────────────────────────────────────────────── */

static int g_t2_threshold_cached = 0;
static u32 g_t2_threshold = 100; /* spec §Q2 default */

static int g_t2_enabled_cached = 0;
static bool g_t2_enabled = true;

u32 Fbxt2HotnessThreshold(void) {
  if (!g_t2_threshold_cached) {
    const char *e = getenv("FBX_T2_HOTNESS_THRESHOLD");
    g_t2_threshold_cached = 1;
    if (e && *e) {
      long v = strtol(e, NULL, 10);
      if (v > 0 && v <= 1000000L) {
        g_t2_threshold = (u32)v;
      }
    }
  }
  return g_t2_threshold;
}

bool Fbxt2Enabled(void) {
  if (!g_t2_enabled_cached) {
    const char *e = getenv("FIREBOX_T2");
    g_t2_enabled_cached = 1;
    if (e && *e) {
      if (!strcmp(e, "0") || !strcmp(e, "off") ||
          !strcmp(e, "false") || !strcmp(e, "no")) {
        g_t2_enabled = false;
      }
    }
  }
  return g_t2_enabled;
}

void Fbxt2ResetEnvCacheForTest(void) {
  g_t2_threshold_cached = 0;
  g_t2_threshold = 100;
  g_t2_enabled_cached = 0;
  g_t2_enabled = true;
  g_t2_trace_init = 0;
  g_t2_trace_escalations = 0;
  g_t2_trace_bailouts = 0;
  g_t2_trace_verbose = 0;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Escalation pipeline — spec §6.2.                                           */
/* ────────────────────────────────────────────────────────────────────────── */

void Fbxt2TryEscalate(struct Machine *m, struct FbxTcBlock *b) {
  struct FbxIrBlock *ir;
  struct FbxWasmBuffer buf;
  u64 sys_id;
  int funcref;
  /* #635: per-block constants table.  Built from a one-shot walk of the IR
   * BEFORE emit (the emit walker dispatches through the same allocation
   * helpers, getting the same slot indices).  The bridge copies these into
   * the per-block FbxT2BlockCtx at instantiate time. */
  u64 consts[FBX_T2_BLOCK_CTX_MAX_CONSTS];
  u32 nconsts = 0;
  /* The latch — spec §6.1 idempotence.  Set FIRST so a racing thread that
   * also sees `hits >= threshold` short-circuits without re-running the
   * pipeline.  The leak-on-race is harmless per spec (the loser's funcref
   * leaks but doesn't corrupt). */
  if (b->t2_attempted) return;
  b->t2_attempted = 1;

  T2EnsureTraceFlags();

  if (!Fbxt2Enabled()) {
    T2TraceLine(g_t2_trace_escalations,
                "[t2 escalate] sys=%p pc=%#llx outcome=disabled\n",
                (void *)m->system, (unsigned long long)b->start_pc);
    return;
  }

  /* Step 1 — lift Tier 1 block to IR. */
  ir = fbx_ir_lift(b);
  if (!ir) {
    T2TraceLine(g_t2_trace_escalations,
                "[t2 escalate] sys=%p pc=%#llx outcome=lift_bailed\n",
                (void *)m->system, (unsigned long long)b->start_pc);
    return;
  }

  /* Step 2a — build the per-block consts[] table (#635). */
  if (!fbx_ir_build_block_ctx_consts(ir, consts,
                                     FBX_T2_BLOCK_CTX_MAX_CONSTS,
                                     &nconsts)) {
    fbx_ir_free(ir);
    T2TraceLine(g_t2_trace_escalations,
                "[t2 escalate] sys=%p pc=%#llx outcome=ctx_overflow\n",
                (void *)m->system, (unsigned long long)b->start_pc);
    return;
  }

  /* Step 2b — synthesise wasm bytes (constant-free at hoisted sites).
   *
   * #668: route through the with_reason variant so the verbose channel can
   * emit a per-opcode [t2 synth-fail] line.  The legacy [t2 escalate]
   * outcome=synth_failed line is preserved on the escalations channel for
   * back-compat with #667's raw logs + grep-pipelines.  Reason+opcode are
   * meaningful only on a 0 return; the with_reason API guarantees they fall
   * back to OK/0 on success. */
  fbx_wasm_buffer_init(&buf);
  {
    enum FbxIrEmitFailReason ir_fail_reason = FBX_IR_EMIT_OK;
    u8 ir_fail_opcode = 0;
    if (!fbx_ir_emit_wasm_with_reason(ir, b, &buf, &ir_fail_reason,
                                      &ir_fail_opcode)) {
      fbx_ir_free(ir);
      fbx_wasm_buffer_free(&buf);
      T2TraceLine(g_t2_trace_verbose,
                  "[t2 synth-fail] sys=%p pc=%#llx opcode=0x%02x reason=%s\n",
                  (void *)m->system, (unsigned long long)b->start_pc,
                  (unsigned)ir_fail_opcode,
                  fbx_ir_fail_reason_name(ir_fail_reason));
      T2TraceLine(g_t2_trace_escalations,
                  "[t2 escalate] sys=%p pc=%#llx outcome=synth_failed\n",
                  (void *)m->system, (unsigned long long)b->start_pc);
      return;
    }
  }

  /* Step 3 — build the per-block FbxT2BlockCtx in the GUEST's own heap
   * (firebox#719).  Pre-#719 the bridge copied consts[] into a
   * bridge-owned scratch memory; now the T2 module imports the guest's
   * REAL (shared) linear memory, so the ctx must live in guest memory at
   * an address the guest controls (a bridge-chosen offset would clobber
   * live guest data).  The allocation is stashed on the block and outlives
   * every dispatch of the funcref (the wasm reads it at dispatch time);
   * it is freed in FbxTcInvalidate alongside the funcref drop. */
  {
    struct FbxT2BlockCtx *ctx;
    u32 ci;
    ctx = (struct FbxT2BlockCtx *)malloc(sizeof(struct FbxT2BlockCtx));
    if (!ctx) {
      fbx_ir_free(ir);
      fbx_wasm_buffer_free(&buf);
      T2TraceLine(g_t2_trace_escalations,
                  "[t2 escalate] sys=%p pc=%#llx outcome=ctx_alloc_failed\n",
                  (void *)m->system, (unsigned long long)b->start_pc);
      return;
    }
    ctx->version = FBX_T2_BLOCK_CTX_VERSION;
    ctx->nconsts = nconsts;
    for (ci = 0; ci < FBX_T2_BLOCK_CTX_MAX_CONSTS; ++ci) {
      ctx->consts[ci] = (ci < nconsts) ? consts[ci] : 0;
    }

    /* Step 4 — instantiate via host import; pass the guest-heap ctx
     * pointer (== the i32 guest-memory offset on the wasm target). */
    sys_id = (u64)(uintptr_t)m->system; /* spec §11.3: sys_id = host-side &System */
    funcref = fbx_t2_instantiate(sys_id, buf.data, (u32)buf.len, ctx);
    fbx_ir_free(ir);
    fbx_wasm_buffer_free(&buf);
    if (funcref < 0) {
      free(ctx);
      T2TraceLine(g_t2_trace_escalations,
                  "[t2 escalate] sys=%p pc=%#llx outcome=instantiate_failed\n",
                  (void *)m->system, (unsigned long long)b->start_pc);
      return;
    }

    /* Step 5 — stash funcref + ctx on the block.  Future ExecuteBlock
     * calls route through Fbxt2Dispatch instead of walking entries[].
     * The ctx pointer is retained so it stays live for every dispatch. */
    b->t2_block_ctx = ctx;
  }
  b->t2_funcref = funcref;
  T2TraceLine(g_t2_trace_escalations,
              "[t2 escalate] sys=%p pc=%#llx outcome=success funcref=%d\n",
              (void *)m->system, (unsigned long long)b->start_pc, funcref);
}

int Fbxt2Dispatch(struct Machine *m, struct FbxTcBlock *b) {
  u64 sys_id = (u64)(uintptr_t)m->system;
  /* `m_ptr` is the linear-memory offset of the Machine struct.  When Blink
   * runs inside wasm (production), truncating the `struct Machine *` to i32
   * yields the correct wasm offset into Blink's OWN linear memory — and
   * firebox#719 makes the T2 module import THAT SAME (shared) memory, so a
   * translated block's `m_ptr + reg_offset` load/store (§13.5 REG_GET/SET,
   * LOAD/STORE) lands in the real `Machine` the Tier-1 interpreter uses.
   * (Before #719 the bridge gave the T2 module an isolated scratch memory,
   * so this offset — typically several MB — trapped OOB on the first
   * access; that was the #709 root cause.)  For the native test bench it's
   * a meaningless integer the synthetic dispatcher echoes back, which is
   * fine for the integer-only opcodes those tests exercise. */
  i32 m_ptr = (i32)(intptr_t)m;
  int exit_code;
#ifdef __wasm__
  /* firebox#786 — GUEST-SIDE DISPATCH.  Instead of crossing to the host via
   * the `fbx_t2_dispatch` import (a wasm->host re-entry that pays the ~270 ns
   * wasmer coroutine-stack-switch crossing on EVERY dispatch, #694), call the
   * translated block IN-GUEST through `__indirect_function_table`.  The host's
   * `fbx_t2_instantiate` co-instantiates the block into Blink's OWN store and
   * places its `translated_block` export into Blink's (exported, growable)
   * function table, returning the TABLE INDEX as `t2_funcref`.  Casting that
   * index to a C function pointer and calling it compiles (clang wasm ABI) to
   * `call_indirect __indirect_function_table` with the type immediate for the
   * #635 block ABI `(i32 m_ptr, i32 block_ctx_ptr) -> (i32 exit)` — a pure
   * in-instance call with NO coroutine tax (the tax is only for ENTERING wasm
   * from the host).  The guest supplies `block_ctx_ptr` itself: `t2_block_ctx`
   * is the guest-heap FbxT2BlockCtx pointer (firebox#719), and
   * `(i32)(intptr_t)t2_block_ctx` is byte-identical to the `block_ctx_ptr` the
   * host computed under the old host-dispatch path.  `t2_funcref >= 0` is
   * guaranteed by the ExecuteBlock fast-path gate (threadedcode.c). */
  {
    typedef int (*FbxT2BlockFn)(i32, i32);
    i32 ctx_ptr = (i32)(intptr_t)b->t2_block_ctx;
    FbxT2BlockFn fn = (FbxT2BlockFn)(uintptr_t)(u32)b->t2_funcref;
    exit_code = fn(m_ptr, ctx_ptr);
  }
#else
  /* Native test bench: no in-guest table; keep the host-dispatch shim (the
   * synthetic dispatcher echoes integer-only opcodes for the unit tests). */
  exit_code = fbx_t2_dispatch(sys_id, b->t2_funcref, m_ptr);
#endif
  if (exit_code == 1 || exit_code == 2) {
    T2EnsureTraceFlags();
    T2TraceLine(g_t2_trace_bailouts,
                "[t2 bailout] sys=%#llx pc=%#llx funcref=%d exit=%d\n",
                (unsigned long long)sys_id,
                (unsigned long long)b->start_pc,
                b->t2_funcref, exit_code);
  }
  return exit_code;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Weak default stubs — see fbx_t2_glue.h "Why" block for rationale.          */
/*                                                                            */
/* These are weak so that:                                                    */
/*   - Native test benches that supply strong overrides win at link time.    */
/*   - Standalone native Blink falls through to "T2 permanently disabled".   */
/*                                                                            */
/* They are gated on `!defined(__wasm__)` because on wasm targets the         */
/* header declarations carry `__attribute__((import_module("fbx")))` so       */
/* wasm-ld emits these symbols as wasm imports under the `"fbx"` module       */
/* (spec §11.2).  If we ALSO defined the weak bodies for wasm targets, the    */
/* static linker would resolve the externs locally to the weak defs and       */
/* skip the import emission entirely — exactly the #604 failure mode this     */
/* task (#607) closes.  See                                                   */
/* `class_lesson_wasm_import_module_name_drift_between_spec_and_undecorated_extern`
 * for the underlying pattern.  The three-way resolution from                 */
/* `class_lesson_weak_symbol_default_stub_for_three_way_host_shim_resolution`
 * is preserved: (a) wasm → wasm imports (this file emits nothing), (b)       */
/* native test bench → strong overrides win, (c) standalone native → weak     */
/* defaults below.                                                             */
/* ────────────────────────────────────────────────────────────────────────── */

#ifndef __wasm__

__attribute__((weak))
int fbx_t2_instantiate(u64 sys_id, const u8 *wasm_bytes, u32 wasm_len,
                       const struct FbxT2BlockCtx *block_ctx) {
  (void)sys_id;
  (void)wasm_bytes;
  (void)wasm_len;
  (void)block_ctx;
  return -1; /* T2 disabled — no engine wired in */
}

__attribute__((weak))
int fbx_t2_dispatch(u64 sys_id, i32 funcref, i32 m_ptr) {
  (void)sys_id;
  (void)funcref;
  (void)m_ptr;
  return 1; /* bailout — Tier 1 must take the block */
}

__attribute__((weak))
void fbx_t2_drop_all(u64 sys_id) {
  (void)sys_id;
}

__attribute__((weak))
int fbx_t2_resolve_indirect(u64 sys_id, u64 target_pc) {
  (void)sys_id;
  (void)target_pc;
  return -1; /* unresolved — v0.1 §Q2 option A: bailout to Tier 1 */
}

__attribute__((weak))
void fbx_t2_get_stats(u64 sys_id, u32 *modules, u64 *bytes) {
  (void)sys_id;
  if (modules) *modules = 0;
  if (bytes) *bytes = 0;
}

#endif /* !__wasm__ */
