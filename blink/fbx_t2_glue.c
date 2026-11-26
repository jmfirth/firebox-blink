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

#include "blink/atomic.h"
#include "blink/fbx_ir.h"
#include "blink/fbx_t2_block_ctx.h"
#include "blink/fbx_wasm_emit.h"
#include "blink/machine.h"
#include "blink/threadedcode.h"

#ifdef __wasm__
/* firebox#FSZ — Blink's OWN Tier-2 dispatch table.
 *
 * Guest-side dispatch (#786) used to `call_indirect` through
 * `__indirect_function_table` with the host appending one slot per hot block.
 * That only works while that table is Blink's alone, i.e. the static link.
 * Under Route C (a -pie thin main over libc.so) the table is IMPORTED: it is the
 * dynamic linker's, `--export-table` is refused, and the linker allocates
 * indices in it in lockstep across every instance group.  The host therefore
 * found no table to append to, fell back to host dispatch, and handed back a
 * HOST registry index that this file then `call_indirect`ed into the LINKER's
 * table: blink32 died on "indirect call type mismatch", blink64 panicked the
 * whole runtime before it got that far.
 *
 * A second table that Blink defines and exports gives T2 slots nobody else
 * allocates in, in both link shapes, with no linker coordination.  It is not
 * reachable from C (clang cannot name a second table from a function-pointer
 * call), so the table and its one caller are declared here in assembly.  The
 * host (`crates/firebox-wasix/src/t2_bridge.rs`) binds the `fbx_t2_table`
 * export in preference to `__indirect_function_table`.  On wasm64 a defined
 * table is a table64, so its index is i64 there; the block ABI is unchanged:
 * `(i32 m_ptr, i32 block_ctx_ptr) -> (i32 exit)`. */
#ifdef __wasm64__
#define FBX_T2_TABLE_INDEX_TYPE "i64"
#else
#define FBX_T2_TABLE_INDEX_TYPE "i32"
#endif
__asm__("  .tabletype fbx_t2_table, funcref\n"
        "  .globl fbx_t2_table\n"
        "fbx_t2_table:\n"
        "  .globl fbx_t2_call\n"
        "  .type fbx_t2_call,@function\n"
        "fbx_t2_call:\n"
        "  .functype fbx_t2_call (" FBX_T2_TABLE_INDEX_TYPE ", i32, i32) -> (i32)\n"
        "  local.get 1\n"
        "  local.get 2\n"
        "  local.get 0\n"
        "  call_indirect fbx_t2_table, (i32, i32) -> (i32)\n"
        "  end_function\n");
int fbx_t2_call(uintptr_t index, i32 m_ptr, i32 block_ctx_ptr);
#endif

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

/* #794 increment 3 — selective-escalation policy mode (measurement knob; the
 * winning mode becomes the default).  Decides, from a block's IR structure,
 * whether escalating it is PROFITABLE — coverage made grep escalate 94
 * memory/branch-heavy blocks that lose to the interpreter (2.4×); only blocks
 * the compiled form actually beats T1 on (the tight compute self-loops) should
 * escalate.  Modes:
 *   0 = all (current behaviour — escalate every covered+hot block)
 *   1 = self-loop only (the dispatch-amortizing shape)
 *   2 = no-memory only (refuse LOAD/STORE/PUSH/POP/CALL/RET — the MMU losers)
 *   3 = self-loop AND no-memory */
static int g_t2_selective_cached = 0;
static int g_t2_selective_mode = 0;
/* firebox#JJJ — Tier-2 region knobs; see Fbxt2RegionEnv. */
static int g_t2_region_cached = 0;
static bool g_t2_region = false;
static bool g_t2_region_count = false;
static u32 g_t2_region_verify_every = 0;

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

/* ────────────────────────────────────────────────────────────────────────── */
/* firebox#WTT — the ONE boolean grammar for `FIREBOX_*`, mirrored from      */
/* crates/firebox-paths/src/lib.rs `classify_bool`.                          */
/*                                                                            */
/* This used to be a bare deny-list (`0|off|false|no`, case-SENSITIVE,        */
/* untrimmed, default ON) while the host read the same variable through an    */
/* allow-list (`1|true|yes|on`, case-INsensitive, trimmed, default OFF).      */
/* Two complementary parsers agree on the eight recognised spellings and      */
/* disagree on EVERY other string: MEASURED over 19 values, `banana`, `2`,    */
/* `00`, `OFF`, `False`, ` 0 ` and `enabled` were OFF host-side and ON here,  */
/* silently. Invariant 4 calls that silent divergence a regression, and the   */
/* failure mode is the bad one — a mistyped kill-switch reads as ON.          */
/*                                                                            */
/* ASCII trim and ASCII case-folding, deliberately: the Rust half matches     */
/* this exactly rather than using its Unicode-aware `trim`, because a         */
/* grammar one side can express and the other cannot is the same defect       */
/* wearing a rarer input.                                                     */
/*                                                                            */
/* UNSPECIFIED is kept distinct from OFF so each side can keep the default    */
/* firebox#ZVJ ruled for it (ON here, OFF for the host's legacy field)        */
/* WITHOUT either side resolving an unrecognised value into that default —    */
/* which would reproduce the divergence, since the defaults are what differ.  */
/* ────────────────────────────────────────────────────────────────────────── */

enum FbxBoolEnv {
  kFbxBoolUnspecified,
  kFbxBoolOn,
  kFbxBoolOff,
  kFbxBoolInvalid,
};

static int FbxIsAsciiSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' ||
         c == '\r';
}

static enum FbxBoolEnv FbxClassifyBoolEnv(const char *raw) {
  char buf[8];
  const char *beg;
  const char *end;
  size_t n;
  size_t i;
  if (!raw) return kFbxBoolUnspecified;
  beg = raw;
  while (*beg && FbxIsAsciiSpace(*beg)) ++beg;
  end = beg + strlen(beg);
  while (end > beg && FbxIsAsciiSpace(end[-1])) --end;
  n = (size_t)(end - beg);
  if (!n) return kFbxBoolUnspecified;
  /* Longer than the longest recognised spelling ("false"), so it cannot be
     one — and must not silently fall through to a default. */
  if (n >= sizeof(buf)) return kFbxBoolInvalid;
  for (i = 0; i < n; ++i) {
    char c = beg[i];
    if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    buf[i] = c;
  }
  buf[n] = '\0';
  if (!strcmp(buf, "1") || !strcmp(buf, "true") || !strcmp(buf, "yes") ||
      !strcmp(buf, "on")) {
    return kFbxBoolOn;
  }
  if (!strcmp(buf, "0") || !strcmp(buf, "false") || !strcmp(buf, "no") ||
      !strcmp(buf, "off")) {
    return kFbxBoolOff;
  }
  return kFbxBoolInvalid;
}

bool Fbxt2Enabled(void) {
  if (!g_t2_enabled_cached) {
    const char *e = getenv("FIREBOX_T2");
    g_t2_enabled_cached = 1;
    switch (FbxClassifyBoolEnv(e)) {
      case kFbxBoolOn:
        g_t2_enabled = true;
        break;
      case kFbxBoolOff:
        g_t2_enabled = false;
        break;
      case kFbxBoolUnspecified:
        /* firebox#ZVJ, ruled 2026-07-27: Tier 2 is default-ON. */
        break;
      case kFbxBoolInvalid: {
        /* Refuse loudly. The host half (`firebox_paths::env_bool_strict`)
           writes the same sentence and exits with the same status; a
           silent pick here would be a false success either way, and a
           warning-only line is how this family of no-ops survived before
           (firebox#617, firebox#ZVJ). */
        char msg[256];
        int len = snprintf(msg, sizeof(msg),
                           "firebox: FIREBOX_T2=%s is not a recognized boolean "
                           "value (expected 1/true/yes/on or 0/false/no/off)\n",
                           e);
        if (len > 0) {
          (void)write(2, msg, (size_t)(len < (int)sizeof(msg) ? len
                                                              : (int)sizeof(msg) - 1));
        }
        exit(2);
      }
    }
  }
  return g_t2_enabled;
}

static int Fbxt2SelectiveMode(void) {
  if (!g_t2_selective_cached) {
    const char *e = getenv("FBX_T2_SELECTIVE");
    g_t2_selective_cached = 1;
    if (e && *e) {
      long v = strtol(e, NULL, 10);
      if (v >= 0 && v <= 3) g_t2_selective_mode = (int)v;
    }
  }
  return g_t2_selective_mode;
}

/* #794 increment 3b — runtime-profitability DE-ESCALATION.  A block is
 * profitable only if it does enough work per dispatch to amortize the dispatch
 * cost; a self-loop reports "full budget of work" via exit code 3.  A block
 * that NEVER fills the budget in its first FBX_T2_DEESCALATE_AFTER dispatches
 * (short loop or non-self-loop) is reverted to Tier 1.
 *
 * Default OFF (FBX_T2_DEESCALATE=1 enables).  This mechanism is arch-correct
 * and works (it reverts the unprofitable blocks), but #794 MEASURED that it
 * does NOT fix the grep T2-on regression: even reverting EVERY escalated block
 * to Tier 1 leaves grep ~3.6 s (vs t2off 1.5 s), because grep's loss is the
 * one-time runtime ESCALATION/COMPILE overhead (~110 ms/block: lift→emit→
 * cranelift-compile→guest co-instantiate), which is SUNK at escalation and
 * irrecoverable by de-escalation — NOT per-dispatch block-execution loss
 * (T2≈T1/dispatch, so de-escalating saves ~0 and the bookkeeping makes grep
 * slightly worse).  De-escalation helps only workloads whose blocks are
 * genuine per-dispatch losers; grep is escalation-overhead-bound.  The real
 * grep fix is reducing escalation cost (compiled-block cache hit-rate / faster
 * codegen) or T4 AOT (compile at build time → zero runtime escalation).  Kept
 * opt-in as a correct, banked mechanism. */
#define FBX_T2_DEESCALATE_AFTER 8u
static int g_t2_deescalate_cached = 0;
static bool g_t2_deescalate = false;
static bool Fbxt2DeescalateEnabled(void) {
  if (!g_t2_deescalate_cached) {
    const char *e = getenv("FBX_T2_DEESCALATE");
    g_t2_deescalate_cached = 1;
    if (e && *e && (!strcmp(e, "1") || !strcmp(e, "on") ||
                    !strcmp(e, "true") || !strcmp(e, "yes"))) {
      g_t2_deescalate = true;
    }
  }
  return g_t2_deescalate;
}

void Fbxt2ResetEnvCacheForTest(void) {
  g_t2_threshold_cached = 0;
  g_t2_threshold = 100;
  g_t2_enabled_cached = 0;
  g_t2_enabled = true;
  g_t2_selective_cached = 0;
  g_t2_selective_mode = 0;
  g_t2_deescalate_cached = 0;
  g_t2_deescalate = false;
  g_t2_region_cached = 0;
  g_t2_region = false;
  g_t2_region_count = false;
  g_t2_region_verify_every = 0;
  g_t2_trace_init = 0;
  g_t2_trace_escalations = 0;
  g_t2_trace_bailouts = 0;
  g_t2_trace_verbose = 0;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Escalation pipeline — spec §6.2.                                           */
/* ────────────────────────────────────────────────────────────────────────── */


/* ────────────────────────────────────────────────────────────────────────── */
/* firebox#JJJ — Tier-2 regions.                                             */
/*                                                                            */
/* When a block escalates, try first to compile it together with the hot     */
/* Tier-1 blocks it statically reaches (depth-first, fall-through first, so  */
/* the layout follows the fall-through chain) into one region module         */
/* (fbx_ir_emit_region_wasm).  The region is attached to the escalating      */
/* block only; it is entered there and exits to Tier 1 everywhere else.  A   */
/* block that cannot head a region of at least two members takes the         */
/* per-block path unchanged.                                                  */
/*                                                                            */
/*   FBX_T2_REGION=1          enable (default off while this is measured)    */
/*   FBX_T2_REGION_VERIFY=N   replay every Nth region dispatch through        */
/*                            FbxInterpretOne and compare registers, flags   */
/*                            and ip; report at teardown                      */
/*   FBX_T2_REGION_COUNT=1    count guest instructions run inside regions     */
/* ────────────────────────────────────────────────────────────────────────── */

#define FBX_T2_REGION_BUDGET 4096u
/* firebox#T3R2: store-log entries a verifying region carries (its ctx is
 * allocated with the log behind it; see FBX_REGION_EMIT_LOG). */
#define FBX_T2_REGION_LOG_CAP 1024u

struct Fbxt2RegionCtx {
  struct FbxT2BlockCtx ctx; /* first: the module reads it */
  u8 log[FBX_T2_REGION_LOG_CAP][16];
};
/* FbxT2BlockState::region.  ABANDONED: the region compile stayed pending for
 * every attempt; the block takes the per-block path, with fresh attempts. */
#define FBX_T2_REGION_NONE 0u
#define FBX_T2_REGION_INSTALLED 1u
#define FBX_T2_REGION_ABANDONED 2u

static struct {
  _Atomic(unsigned long long) built, members, calls, insns;
  _Atomic(unsigned long long) verified, verify_insns, mismatches, tick;
  _Atomic(unsigned long long) verify_stores;
} g_t2_region_stats;

static bool Fbxt2RegionBoolEnv(const char *name, const char *e) {
  switch (FbxClassifyBoolEnv(e)) {
    case kFbxBoolOn:
      return true;
    case kFbxBoolOff:
    case kFbxBoolUnspecified:
      return false;
    default: {
      char msg[256];
      int len = snprintf(msg, sizeof(msg),
                         "firebox: %s=%s is not a recognized boolean value "
                         "(expected 1/true/yes/on or 0/false/no/off)\n",
                         name, e);
      if (len > 0) {
        (void)write(2, msg, (size_t)(len < (int)sizeof(msg) ? len
                                                            : (int)sizeof(msg) - 1));
      }
      exit(2);
    }
  }
}

static void Fbxt2RegionEnv(void) {
  const char *e;
  if (g_t2_region_cached) return;
  g_t2_region = Fbxt2RegionBoolEnv("FBX_T2_REGION", getenv("FBX_T2_REGION"));
  g_t2_region_count =
      Fbxt2RegionBoolEnv("FBX_T2_REGION_COUNT", getenv("FBX_T2_REGION_COUNT"));
  e = getenv("FBX_T2_REGION_VERIFY");
  if (e && *e) {
    char *end = NULL;
    long v = strtol(e, &end, 10);
    if (!end || *end || v < 0 || v > 1000000000L) {
      (void)write(2, "firebox: FBX_T2_REGION_VERIFY must be a count\n", 46);
      exit(2);
    }
    g_t2_region_verify_every = (u32)v;
  }
  g_t2_region_cached = 1;
}

static bool Fbxt2RegionCounting(void) {
  return g_t2_region_count || g_t2_region_verify_every != 0;
}

void Fbxt2RegionReport(void) {
  char buf[400];
  int n;
  /* Only when asked for counts: the line goes to the guest's stderr. */
  if (!g_t2_region_cached || !g_t2_region || !Fbxt2RegionCounting()) return;
  n = snprintf(buf, sizeof(buf),
               "[t2 region] built=%llu members=%llu calls=%llu insns=%llu "
               "verified=%llu verify_insns=%llu verify_stores=%llu "
               "mismatches=%llu\n",
               (unsigned long long)g_t2_region_stats.built,
               (unsigned long long)g_t2_region_stats.members,
               (unsigned long long)g_t2_region_stats.calls,
               (unsigned long long)g_t2_region_stats.insns,
               (unsigned long long)g_t2_region_stats.verified,
               (unsigned long long)g_t2_region_stats.verify_insns,
               (unsigned long long)g_t2_region_stats.verify_stores,
               (unsigned long long)g_t2_region_stats.mismatches);
  if (n > 0) (void)write(2, buf, (size_t)(n < (int)sizeof(buf) ? n : (int)sizeof(buf) - 1));
}

/* Returns 1 when the escalation was handled here (a region was installed, or
 * its compile is pending and the latch was reopened), 0 to take the
 * per-block path. */
/* How far a lift covers its block: the BAILOUT's PC, or the block's end. */
static u64 Fbxt2LiftCover(const struct FbxIrBlock *ir, const struct FbxTcBlock *b) {
  if (ir->ninsts && ir->insts[ir->ninsts - 1].opcode == FBX_IR_OP_BAILOUT) {
    return ir->insts[ir->ninsts - 1].imm;
  }
  return b->end_pc;
}

static int Fbxt2TryRegion(struct Machine *m, struct FbxTcBlock *b,
                          const struct FbxIrBlock *pb_ir,
                          struct FbxT2BlockState *st) {
  /* firebox#T3R2: members are lifted for a region (fbx_ir_lift_region), the
   * entry included; the per-block lift the caller holds is not a member's. */
  struct FbxIrBlock *entry_ir = fbx_ir_lift_region(b);
  const struct FbxIrBlock *irs[FBX_REGION_MAX_MEMBERS];
  const struct FbxTcBlock *tcs[FBX_REGION_MAX_MEMBERS];
  struct FbxIrBlock *owned[FBX_REGION_MAX_MEMBERS];
  u64 stack[2u * FBX_REGION_MAX_MEMBERS + 2u];
  u64 succ[2];
  u32 sp = 0, n = 0, i, k, ns;
  u32 min_hits = Fbxt2HotnessThreshold() / 2u;
  struct FbxWasmBuffer buf;
  struct FbxT2BlockCtx *ctx;
  int funcref, handled = 0;
  enum FbxIrEmitFailReason why = FBX_IR_EMIT_OK;
  u8 why_op = 0;
  int logging = g_t2_region_verify_every != 0;
  if (!entry_ir) return 0;
  if (!fbx_ir_region_member_ok(entry_ir, b, &why, &why_op)) {
    T2TraceLine(g_t2_trace_verbose,
                "[t2 region] pc=%#llx entry_refused reason=%s op=0x%02x\n",
                (unsigned long long)b->start_pc, fbx_ir_fail_reason_name(why),
                (unsigned)why_op);
    fbx_ir_free(entry_ir);
    return 0;
  }
  irs[0] = entry_ir;
  tcs[0] = b;
  owned[0] = entry_ir;
  n = 1;
  ns = fbx_ir_region_successors(entry_ir, b, succ);
  for (k = ns; k-- > 0;) stack[sp++] = succ[k];
  while (sp > 0 && n < FBX_REGION_MAX_MEMBERS) {
    u64 pc = stack[--sp];
    struct FbxTcBlock *t;
    struct FbxIrBlock *ir;
    int seen = 0;
    for (i = 0; i < n; ++i) seen |= tcs[i]->start_pc == pc;
    if (seen) continue;
    t = FbxTcLookup(m->system, pc);
    if (!t || t->hits < min_hits) continue;
    ir = fbx_ir_lift_region(t);
    if (!ir) continue;
    if (!fbx_ir_region_member_ok(ir, t, NULL, NULL)) {
      fbx_ir_free(ir);
      continue;
    }
    irs[n] = ir;
    tcs[n] = t;
    owned[n] = ir;
    ++n;
    ns = fbx_ir_region_successors(ir, t, succ);
    for (k = ns; k-- > 0;) {
      if (sp < sizeof(stack) / sizeof(stack[0])) stack[sp++] = succ[k];
    }
  }
  /* firebox#T3R2: a block no other member joins is still a region when the
   * per-block path cannot take it, or would cover less of it (its lift
   * BAILs earlier: a form only regions lower). */
  if (n < 2 && fbx_ir_per_block_gate_ok(pb_ir, b) &&
      Fbxt2LiftCover(pb_ir, b) >= Fbxt2LiftCover(entry_ir, b)) {
    T2TraceLine(g_t2_trace_verbose, "[t2 region] pc=%#llx alone\n",
                (unsigned long long)b->start_pc);
    goto done;
  }
  fbx_wasm_buffer_init(&buf);
  if (!fbx_ir_emit_region_wasm(irs, tcs, n,
                               (Fbxt2RegionCounting() ? FBX_REGION_EMIT_COUNT
                                                      : 0) |
                                   (logging ? FBX_REGION_EMIT_LOG : 0),
                               FBX_T2_REGION_BUDGET, &buf)) {
    fbx_wasm_buffer_free(&buf);
    T2TraceLine(g_t2_trace_verbose, "[t2 region] pc=%#llx emit_failed\n",
                (unsigned long long)b->start_pc);
    goto done;
  }
  {
    size_t sz = logging ? sizeof(struct Fbxt2RegionCtx)
                        : sizeof(struct FbxT2BlockCtx);
    ctx = (struct FbxT2BlockCtx *)calloc(1, sz);
    if (!ctx || (u64)(uintptr_t)m + sizeof(*m) > 0x100000000ull ||
        (u64)(uintptr_t)ctx + sz > 0x100000000ull) {
      /* firebox#FSZ: the block ABI's i32 pointers must not truncate. */
      free(ctx);
      fbx_wasm_buffer_free(&buf);
      goto done;
    }
  }
  ctx->version = FBX_T2_BLOCK_CTX_VERSION;
  ctx->nconsts = 1; /* consts[0]: the instruction count, when counting */
  if (logging) {
    /* consts[1..3]: the store log (FBX_REGION_EMIT_LOG) */
    ctx->nconsts = 4;
    ctx->consts[1] = (u64)(uintptr_t)((struct Fbxt2RegionCtx *)ctx)->log;
    ctx->consts[2] = 0;
    ctx->consts[3] = FBX_T2_REGION_LOG_CAP;
  }
  funcref = fbx_t2_instantiate((u64)(uintptr_t)m->system, buf.data,
                               (u32)buf.len, ctx);
  fbx_wasm_buffer_free(&buf);
  if (funcref == FBX_T2_INSTANTIATE_PENDING) {
    /* Same bounded retry as the per-block path (see Fbxt2TryEscalate). */
    free(ctx);
    T2TraceLine(g_t2_trace_escalations,
                "[t2 escalate] sys=%p pc=%#llx outcome=region_pending "
                "members=%u attempts=%u\n",
                (void *)m->system, (unsigned long long)b->start_pc,
                (unsigned)n, (unsigned)st->pending_attempts + 1u);
    if (++st->pending_attempts < FBX_T2_PENDING_MAX_ATTEMPTS) {
      st->attempted = 0;
      st->retry_at_hits = b->hits + FBX_T2_PENDING_RETRY_STRIDE;
      handled = 1;
    } else {
      /* Out of attempts (membership may have kept changing the bytes): stop
       * trying regions here and give the per-block path its own attempts. */
      st->region = FBX_T2_REGION_ABANDONED;
      st->pending_attempts = 0;
    }
    goto done;
  }
  if (funcref < 0) {
    free(ctx);
    goto done;
  }
  st->block_ctx = ctx;
  st->funcref = funcref;
  st->region = FBX_T2_REGION_INSTALLED;
  atomic_fetch_add_explicit(&g_t2_region_stats.built, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&g_t2_region_stats.members, n,
                            memory_order_relaxed);
  T2TraceLine(g_t2_trace_escalations,
              "[t2 escalate] sys=%p pc=%#llx outcome=region members=%u "
              "funcref=%d\n",
              (void *)m->system, (unsigned long long)b->start_pc, (unsigned)n,
              funcref);
  handled = 1;
done:
  for (i = 0; i < n; ++i) fbx_ir_free(owned[i]);
  return handled;
}

#ifdef __wasm__
/* Run the region, then replay the same number of guest instructions from the
 * same starting state through FbxInterpretOne, and compare.  The guest state
 * a region can change is the general registers, flags, ip and (firebox#T3R2)
 * the memory its stores wrote.  A verifying region logs every store's host
 * offset, width and overwritten bytes (FBX_REGION_EMIT_LOG); the region's
 * final bytes at each logged address are read, the stores are undone in
 * reverse, the interpreter runs, and those bytes are compared too.  This
 * catches a wrong value or a wrong address; a store the region failed to
 * make at all, to an address it never wrote, is not seen here (the whole-
 * program differentials are what see that).  The interpreter's result is
 * kept. */
static u64 Fbxt2RegionReadN(u64 off, u32 width) {
  u64 v = 0;
  memcpy(&v, (const void *)(uintptr_t)off, width);
  return v;
}

static int Fbxt2RegionVerifiedCall(struct Machine *m,
                                   struct FbxT2BlockState *st, i32 m_ptr) {
  struct FbxT2BlockCtx *ctx = (struct FbxT2BlockCtx *)st->block_ctx;
  struct Fbxt2RegionCtx *rc_ctx = (struct Fbxt2RegionCtx *)st->block_ctx;
  u8 pre_weg[16][8], post_weg[16][8];
  static _Thread_local u64 post_mem[FBX_T2_REGION_LOG_CAP];
  u32 pre_flags, post_flags;
  u64 pre_ip, post_ip, n, k, nlog;
  int rc, mem_bad = -1;
  memcpy(pre_weg, m->weg, sizeof pre_weg);
  pre_flags = m->flags;
  pre_ip = m->ip;
  ctx->consts[0] = 0;
  ctx->consts[2] = 0;
  rc = fbx_t2_call((uintptr_t)(u32)st->funcref, m_ptr,
                   (i32)(intptr_t)st->block_ctx);
  n = ctx->consts[0];
  nlog = ctx->consts[2];
  if (nlog > FBX_T2_REGION_LOG_CAP) nlog = FBX_T2_REGION_LOG_CAP; /* cannot */
  for (k = 0; k < nlog; ++k) {
    u32 off, w;
    memcpy(&off, rc_ctx->log[k], 4);
    memcpy(&w, rc_ctx->log[k] + 4, 4);
    post_mem[k] = Fbxt2RegionReadN(off, w);
  }
  for (k = nlog; k-- > 0;) {
    u32 off, w;
    memcpy(&off, rc_ctx->log[k], 4);
    memcpy(&w, rc_ctx->log[k] + 4, 4);
    memcpy((void *)(uintptr_t)off, rc_ctx->log[k] + 8, w); /* undo */
  }
  memcpy(post_weg, m->weg, sizeof post_weg);
  post_flags = m->flags;
  post_ip = m->ip;
  memcpy(m->weg, pre_weg, sizeof pre_weg);
  m->flags = pre_flags;
  m->ip = pre_ip;
  for (k = 0; k < n; ++k) FbxInterpretOne(m);
  for (k = 0; k < nlog && mem_bad < 0; ++k) {
    u32 off, w;
    memcpy(&off, rc_ctx->log[k], 4);
    memcpy(&w, rc_ctx->log[k] + 4, 4);
    if (Fbxt2RegionReadN(off, w) != post_mem[k]) mem_bad = (int)k;
  }
  atomic_fetch_add_explicit(&g_t2_region_stats.verified, 1,
                            memory_order_relaxed);
  atomic_fetch_add_explicit(&g_t2_region_stats.verify_insns, n,
                            memory_order_relaxed);
  atomic_fetch_add_explicit(&g_t2_region_stats.verify_stores, nlog,
                            memory_order_relaxed);
  if ((rc != 0 && rc != 1) || memcmp(post_weg, m->weg, sizeof post_weg) ||
      post_flags != m->flags || post_ip != m->ip || mem_bad >= 0) {
    unsigned long long nm = atomic_fetch_add_explicit(
        &g_t2_region_stats.mismatches, 1, memory_order_relaxed);
    /* A region that disagreed once is not run again: back to Tier 1 for
     * this block (latch kept, no re-escalation). */
    st->funcref = -1;
    if (nm < 16) {
      char line[200];
      int len;
      u32 r;
      len = snprintf(line, sizeof(line),
                     "[t2 region MISMATCH] entry=%#llx n=%llu rc=%d ip "
                     "region=%#llx interp=%#llx flags region=%#x interp=%#x\n",
                     (unsigned long long)pre_ip, (unsigned long long)n, rc,
                     (unsigned long long)post_ip, (unsigned long long)m->ip,
                     post_flags, m->flags);
      if (len > 0) (void)write(2, line, (size_t)len);
      for (r = 0; r < 16; ++r) {
        u64 a, c;
        memcpy(&a, post_weg[r], 8);
        memcpy(&c, m->weg[r], 8);
        if (a == c) continue;
        len = snprintf(line, sizeof(line),
                       "[t2 region MISMATCH]   r%u region=%#llx interp=%#llx\n",
                       r, (unsigned long long)a, (unsigned long long)c);
        if (len > 0) (void)write(2, line, (size_t)len);
      }
      if (mem_bad >= 0) {
        u32 off, w;
        memcpy(&off, rc_ctx->log[mem_bad], 4);
        memcpy(&w, rc_ctx->log[mem_bad] + 4, 4);
        len = snprintf(line, sizeof(line),
                       "[t2 region MISMATCH]   store %d off=%#x w=%u "
                       "region=%#llx interp=%#llx\n",
                       mem_bad, off, w, (unsigned long long)post_mem[mem_bad],
                       (unsigned long long)Fbxt2RegionReadN(off, w));
        if (len > 0) (void)write(2, line, (size_t)len);
      }
    }
  }
  return rc == 1 ? 1 : 0; /* ip agreed: Tier 1 resumes there either way */
}
#endif

void Fbxt2TryEscalate(struct Machine *m, struct FbxTcBlock *b,
                      struct FbxT2BlockState *st) {
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
  /* The latch — spec §6.1 idempotence.  Set FIRST so this Machine does not
   * re-run the pipeline for this block.  firebox#HYS made the latch
   * PER-INSTANCE along with the funcref it guards: it used to be shared, which
   * meant the first thread to escalate latched every OTHER thread out of ever
   * minting a translation in its own table — while leaving behind the funcref
   * those threads would then dispatch.  Each instance now earns, and pays for,
   * its own translation.  firebox#794: a PENDING async-compile outcome CLEARS
   * this latch again (the only path that does) so the block re-attempts on a
   * later hit; every terminal outcome leaves it set. */
  if (st->attempted) return;
  st->attempted = 1;

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

  /* #794 increment 3 — selective escalation.  Profile the lifted block + apply
   * the policy mode.  Coverage (inc 1) made memory/branch-heavy blocks escalate
   * and LOSE to the interpreter (grep 2.4×); only blocks the compiled form
   * actually beats T1 on should escalate.  The profile line (verbose) lets us
   * SEE the loser distribution; the mode gate refuses the unprofitable ones. */
  {
    u32 pi;
    int self_loop, has_mem = 0, n_mem = 0, n_alu = 0;
    u8 last = FBX_IR_OP_PC_MARK;
    int mode, refuse = 0;
    for (pi = 0; pi < ir->ninsts; ++pi) {
      u8 op = ir->insts[pi].opcode;
      if (op == FBX_IR_OP_PC_MARK) continue;
      last = op;
      switch (op) {
        case FBX_IR_OP_LOAD: case FBX_IR_OP_STORE: case FBX_IR_OP_PUSH:
        case FBX_IR_OP_POP: case FBX_IR_OP_CALL_DIRECT: case FBX_IR_OP_RET:
          has_mem = 1; n_mem++; break;
        case FBX_IR_OP_ADD: case FBX_IR_OP_SUB: case FBX_IR_OP_AND:
        case FBX_IR_OP_OR: case FBX_IR_OP_XOR: case FBX_IR_OP_IMUL:
          n_alu++; break;
        default: break;
      }
    }
    self_loop = (last == FBX_IR_OP_BRANCH_COND);
    T2TraceLine(g_t2_trace_verbose,
                "[t2 profile] pc=%#llx ninsts=%u self_loop=%d n_mem=%d "
                "n_alu=%d\n",
                (unsigned long long)b->start_pc, (unsigned)ir->ninsts,
                self_loop, n_mem, n_alu);
    mode = Fbxt2SelectiveMode();
    if (mode == 1 && !self_loop) refuse = 1;
    else if (mode == 2 && has_mem) refuse = 1;
    else if (mode == 3 && (!self_loop || has_mem)) refuse = 1;
    if (refuse) {
      fbx_ir_free(ir);
      T2TraceLine(g_t2_trace_escalations,
                  "[t2 escalate] sys=%p pc=%#llx outcome=not_profitable\n",
                  (void *)m->system, (unsigned long long)b->start_pc);
      return;
    }
  }

  /* firebox#JJJ — a region headed by this block, when enabled. */
  Fbxt2RegionEnv();
  if (g_t2_region && st->region != FBX_T2_REGION_ABANDONED &&
      Fbxt2TryRegion(m, b, ir, st)) {
    fbx_ir_free(ir);
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
    /* firebox#FSZ — the block ABI carries `m_ptr` and `block_ctx_ptr` as i32
     * linear offsets and the block imports a 32-bit memory view.  A wasm64
     * Blink's memory is not bounded at 4 GiB (Route C blink64: 8 GiB), so a
     * Machine or ctx allocated above it would be TRUNCATED and the block would
     * read and write some other object — silently.  Stay Tier 1 instead. */
    if ((u64)(uintptr_t)m + sizeof(*m) > 0x100000000ull ||
        (u64)(uintptr_t)ctx + sizeof(*ctx) > 0x100000000ull) {
      free(ctx);
      fbx_ir_free(ir);
      fbx_wasm_buffer_free(&buf);
      T2TraceLine(g_t2_trace_escalations,
                  "[t2 escalate] sys=%p pc=%#llx outcome=ptr_above_4g\n",
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
    if (funcref == FBX_T2_INSTANTIATE_PENDING) {
      /* firebox#794 async escalation — the host dispatched this block's compile
       * to its background pool; it is not ready yet.  Free the per-attempt ctx
       * (a fresh one is built on the next poll), then arrange a BOUNDED retry:
       * clear the terminal latch and push the next attempt out by
       * FBX_T2_PENDING_RETRY_STRIDE hits — UNLESS we've already polled
       * FBX_T2_PENDING_MAX_ATTEMPTS times, in which case give up (leave the
       * latch set).  Giving up is harmless: the background compile still lands
       * in the cache, so a later run (or a hotter sibling block) starts warm.
       * The whole branch is dead on a synchronous host (it never returns
       * PENDING), so non-async behavior is byte-identical. */
      free(ctx);
      if (++st->pending_attempts >= FBX_T2_PENDING_MAX_ATTEMPTS) {
        T2TraceLine(g_t2_trace_escalations,
                    "[t2 escalate] sys=%p pc=%#llx outcome=pending_gave_up\n",
                    (void *)m->system, (unsigned long long)b->start_pc);
      } else {
        st->attempted = 0; /* reopen — the ONLY path that clears the latch */
        st->retry_at_hits = b->hits + FBX_T2_PENDING_RETRY_STRIDE;
        T2TraceLine(g_t2_trace_escalations,
                    "[t2 escalate] sys=%p pc=%#llx outcome=pending attempts=%u\n",
                    (void *)m->system, (unsigned long long)b->start_pc,
                    (unsigned)st->pending_attempts);
      }
      return;
    }
    if (funcref < 0) {
      free(ctx);
      T2TraceLine(g_t2_trace_escalations,
                  "[t2 escalate] sys=%p pc=%#llx outcome=instantiate_failed\n",
                  (void *)m->system, (unsigned long long)b->start_pc);
      return;
    }

    /* Step 5 — stash funcref + ctx on THIS MACHINE's state for the block
     * (firebox#HYS; it used to be the block itself).  Future ExecuteBlock
     * calls on this Machine route through Fbxt2Dispatch instead of walking
     * entries[].  The ctx pointer is retained so it stays live for every
     * dispatch, and is freed with this map entry. */
    st->block_ctx = ctx;
  }
  st->funcref = funcref;
  T2TraceLine(g_t2_trace_escalations,
              "[t2 escalate] sys=%p pc=%#llx outcome=success funcref=%d\n",
              (void *)m->system, (unsigned long long)b->start_pc, funcref);
}

int Fbxt2Dispatch(struct Machine *m, struct FbxTcBlock *b,
                  struct FbxT2BlockState *st) {
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
    /* firebox#FSZ: through Blink's own `fbx_t2_table`, never the (possibly
     * linker-owned) `__indirect_function_table` — see the table's comment. */
    i32 ctx_ptr = (i32)(intptr_t)st->block_ctx;
    if (st->region == FBX_T2_REGION_INSTALLED && g_t2_region_verify_every &&
        atomic_fetch_add_explicit(&g_t2_region_stats.tick, 1,
                                  memory_order_relaxed) %
                g_t2_region_verify_every == 0) {
      exit_code = Fbxt2RegionVerifiedCall(m, st, m_ptr);
    } else {
      if (st->region == FBX_T2_REGION_INSTALLED && g_t2_region_verify_every) {
        /* firebox#T3R2: this region logs its stores; an unverified call
         * starts from an empty log too. */
        ((struct FbxT2BlockCtx *)st->block_ctx)->consts[2] = 0;
      }
      exit_code = fbx_t2_call((uintptr_t)(u32)st->funcref, m_ptr, ctx_ptr);
      if (st->region == FBX_T2_REGION_INSTALLED && g_t2_region_count) {
        atomic_fetch_add_explicit(&g_t2_region_stats.calls, 1,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(
            &g_t2_region_stats.insns,
            ((struct FbxT2BlockCtx *)st->block_ctx)->consts[0],
            memory_order_relaxed);
      }
    }
  }
#else
  /* Native test bench: no in-guest table; keep the host-dispatch shim (the
   * synthetic dispatcher echoes integer-only opcodes for the unit tests). */
  exit_code = fbx_t2_dispatch(sys_id, st->funcref, m_ptr);
#endif
  /* #794 increment 3b — runtime-profitability feedback + de-escalation.  Exit
   * code 3 (a self-loop that exhausted the full iteration budget = a deep,
   * profitable loop) is counted and mapped to 0 so ExecuteBlock's control flow
   * is unchanged.  A block that never fills the budget in its first
   * FBX_T2_DEESCALATE_AFTER dispatches (short loop / non-self-loop) does too
   * little work per dispatch to amortize the dispatch cost → revert it to
   * Tier 1 (t2_funcref = -1; the t2_attempted latch stays set so it is not
   * re-escalated → no oscillation).  This is the fix for the coverage-driven
   * grep T2-on regression (#794): the headroom loop fills the budget on its
   * first dispatch and is KEPT; grep's short/straight-line blocks never fill
   * and revert. */
  if (exit_code == 3) {
    if (st->filled != 0xFFFFFFFFu) ++st->filled;
    exit_code = 0;
  }
  if (st->dispatches != 0xFFFFFFFFu) ++st->dispatches;
  /* A region never returns 3 (it has no single self-loop to fill), so the
   * fill-based de-escalation would revert every region: exempt them. */
  if (Fbxt2DeescalateEnabled() && st->region != FBX_T2_REGION_INSTALLED &&
      st->dispatches == FBX_T2_DEESCALATE_AFTER && st->filled == 0) {
    st->funcref = -1; /* revert to Tier 1; latch kept (no re-escalation) */
    T2EnsureTraceFlags();
    T2TraceLine(g_t2_trace_escalations,
                "[t2 deescalate] sys=%p pc=%#llx dispatches=%u "
                "(never filled budget — unprofitable)\n",
                (void *)m->system, (unsigned long long)b->start_pc,
                (unsigned)st->dispatches);
  }
  if (exit_code == 1 || exit_code == 2) {
    T2EnsureTraceFlags();
    T2TraceLine(g_t2_trace_bailouts,
                "[t2 bailout] sys=%#llx pc=%#llx funcref=%d exit=%d\n",
                (unsigned long long)sys_id,
                (unsigned long long)b->start_pc,
                st->funcref, exit_code);
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
