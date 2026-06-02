/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2026 Justin Firth (Firebox)                                        │
│                                                                              │
│ Phase 2 — Tier 1 of the ELF-performance track: threaded-code dispatcher.     │
│                                                                              │
│ Motivated by Phase 1.4's load-bearing falsification: per-instruction         │
│ FbxThunksMaybeDispatch in JitlessDispatch cost +14.8% on busybox-awk-10k     │
│ (pure-ALU workload, no thunks fired).  Fix: move dispatch out of the         │
│ per-instruction path.  Compile basic blocks to flat (Op_fn_ptr, args)        │
│ arrays cached by start-PC; replay via a tight loop.  Decode happens once     │
│ per block.                                                                   │
│                                                                              │
│ See blink/threadedcode.h for the design header + Q1-Q6 verdicts.             │
╚─────────────────────────────────────────────────────────────────────────────*/
#include "blink/threadedcode.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "blink/assert.h"
#include "blink/builtin.h"
#include "blink/endian.h"
#include "blink/fbx_t2_glue.h"
#include "blink/log.h"
#include "blink/machine.h"
#include "blink/rde.h"
#include "blink/thunks.h"
#include "blink/x86.h"

#include <stdarg.h>

/* ────────────────────────────────────────────────────────────────────────── */
/* Trace flag — lit when FIREBOX_TC_TRACE=1 or 2 in the environment.          */
/*                                                                            */
/* FIREBOX_TC_TRACE=1 — legacy structural trace (init / compile / invalidate  */
/*                     events to stderr).                                     */
/* FIREBOX_TC_TRACE=2 — per-opcode host-time profile (NEW for §13.5/§Q7).     */
/*                     Instruments ExecuteBlock to measure cumulative host    */
/*                     time spent in each Mopcode handler, then dumps a       */
/*                     ranked table to stderr at process exit.  Used by       */
/*                     the §13.5 dispatch to resolve §Q7 — "which 30          */
/*                     opcodes" — by host-time spent on the bench corpus,     */
/*                     not by raw frequency (which §3.1's hints capture).     */
/* ────────────────────────────────────────────────────────────────────────── */

static int g_tc_trace = -1;        /* -1 = uninit; 0 = off; 1 = structural; */
                                   /* 2 = +per-opcode profile.              */

/* §Q7 per-opcode profile.  Mopcode is at most 12 bits in Blink's encoding   */
/* (see blink/rde.h's Mopcode accessor); a 4096-entry table covers it        */
/* densely.  Two parallel u64 arrays keep the hot path cache-tight (load     */
/* one bucket, accumulate count, accumulate nanoseconds; no struct stride). */
#define FBX_TC_PROFILE_BUCKETS 4096u
static u64 g_op_count_by_mop[FBX_TC_PROFILE_BUCKETS];
static u64 g_op_ns_by_mop[FBX_TC_PROFILE_BUCKETS];
static u64 g_op_total_count;
static u64 g_op_total_ns;
static int g_op_dump_registered;   /* atexit() install latch */

static void EnsureTcTraceFlag(void) {
  if (g_tc_trace == -1) {
    const char *e = getenv("FIREBOX_TC_TRACE");
    if (!e || !*e || *e == '0') {
      g_tc_trace = 0;
    } else if (e[0] == '2') {
      /* "2", "2\n", "2 " — all mean level 2.  Be lenient on trailing
       * whitespace because guest env-var passthrough may strip or add
       * terminators inconsistently. */
      g_tc_trace = 2;
    } else {
      g_tc_trace = 1;
    }
  }
}

static void TcTraceLine(const char *fmt, ...) {
  char buf[160];
  va_list ap;
  int n;
  if (!g_tc_trace) return;
  va_start(ap, fmt);
  n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (n > 0) (void)write(2, buf, (size_t)n);
}

/* §Q7 profile dump — called via atexit() when FIREBOX_TC_TRACE=2.  Emits a  */
/* machine-parseable, sorted-descending-by-ns table to stderr.               */
/*                                                                          */
/* Format (one header line + one data line per opcode with non-zero count): */
/*   FBX_PROFILE start total_count=N total_ns=M                              */
/*   FBX_PROFILE op mop=0xNNN count=N ns=M pct_count=PP.PP pct_ns=QQ.QQ      */
/*   ...                                                                    */
/*   FBX_PROFILE end                                                        */
/*                                                                          */
/* Output is parseable by awk for the measurement step.  Sort is O(N²) over */
/* 4096 buckets — acceptable for a one-shot exit dump.                      */
static void FbxTcProfileDump(void) {
  /* Sort buckets descending by ns.  We sort indices into a parallel array  */
  /* to keep g_op_ns_by_mop's mop→ns mapping stable for the report.         */
  u32 idx[FBX_TC_PROFILE_BUCKETS];
  u32 n_used = 0;
  u32 i, j;
  char buf[200];
  int n;
  for (i = 0; i < FBX_TC_PROFILE_BUCKETS; ++i) {
    if (g_op_count_by_mop[i] > 0) {
      idx[n_used++] = i;
    }
  }
  /* Selection sort descending by ns; bounded by n_used ≤ ~256 typical. */
  for (i = 0; i + 1 < n_used; ++i) {
    u32 best = i;
    for (j = i + 1; j < n_used; ++j) {
      if (g_op_ns_by_mop[idx[j]] > g_op_ns_by_mop[idx[best]]) best = j;
    }
    if (best != i) {
      u32 tmp = idx[i];
      idx[i] = idx[best];
      idx[best] = tmp;
    }
  }
  n = snprintf(buf, sizeof(buf),
               "FBX_PROFILE start total_count=%llu total_ns=%llu\n",
               (unsigned long long)g_op_total_count,
               (unsigned long long)g_op_total_ns);
  if (n > 0) (void)write(2, buf, (size_t)n);
  for (i = 0; i < n_used; ++i) {
    u32 m = idx[i];
    u64 c = g_op_count_by_mop[m];
    u64 ns = g_op_ns_by_mop[m];
    /* Compute pct as integer hundredths to avoid float dependency.       */
    u64 pct_count_x100 = g_op_total_count
        ? (c * 10000ull) / g_op_total_count : 0;
    u64 pct_ns_x100 = g_op_total_ns
        ? (ns * 10000ull) / g_op_total_ns : 0;
    n = snprintf(buf, sizeof(buf),
                 "FBX_PROFILE op mop=0x%03x count=%llu ns=%llu "
                 "pct_count=%llu.%02llu pct_ns=%llu.%02llu\n",
                 (unsigned)m,
                 (unsigned long long)c,
                 (unsigned long long)ns,
                 (unsigned long long)(pct_count_x100 / 100ull),
                 (unsigned long long)(pct_count_x100 % 100ull),
                 (unsigned long long)(pct_ns_x100 / 100ull),
                 (unsigned long long)(pct_ns_x100 % 100ull));
    if (n > 0) (void)write(2, buf, (size_t)n);
  }
  n = snprintf(buf, sizeof(buf), "FBX_PROFILE end\n");
  if (n > 0) (void)write(2, buf, (size_t)n);
}

/* Read the monotonic clock as nanoseconds since an arbitrary epoch.        */
/* Used by the §Q7 profile to measure per-opcode host time spent.            */
/* On wasm32-wasi-threads this resolves to wasix's clock_time_get; on       */
/* the native build it resolves to CLOCK_MONOTONIC.                          */
static inline u64 FbxNowNs(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
  return (u64)ts.tv_sec * 1000000000ull + (u64)ts.tv_nsec;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Tuning knobs.  Single-source-of-truth at the top of the file so a future   */
/* perf retune doesn't have to grep across files.                             */
/* ────────────────────────────────────────────────────────────────────────── */

/* Initial bucket count — power of 2 for cheap masking.  Sized so that a
 * typical busybox grep run (estimated 5000-15000 unique blocks) fits with
 * load factor ~1.0.  Grow handled by the load-factor check below. */
#define FBX_TC_INIT_BUCKETS 16384

/* Maximum instructions per cached block.  Higher = more savings amortising
 * the dispatch overhead; lower = less wasted work on never-re-executed
 * blocks.  Picked to comfortably exceed musl's longest known basic block
 * (~80 ops in memchr's word-loop) while bounding pathological cases. */
#define FBX_TC_MAX_BLOCK_ENTRIES 128

/* Total cap on live entries before we trigger a global flush.  Sized to
 * keep cache memory under ~50 MB (FBX_TC_MAX_BLOCK_ENTRIES bound means
 * each entry is sizeof(FbxTcEntry) ~ 80 bytes → ~50 MB at 640k entries).
 * Set to 0 to disable the cap entirely (test-only — production guests
 * could blow up RAM without this). */
#define FBX_TC_DEFAULT_ENTRY_CAP 640000

/* Bucket-array growth trigger: when block_count / nbuckets > 4, double
 * the bucket array.  Keeps lookups O(1) amortised. */
#define FBX_TC_LOAD_FACTOR_NUMERATOR 4

/* ────────────────────────────────────────────────────────────────────────── */
/* External symbols imported from machine.c — the opcode dispatch table and   */
/* its accessor.  We use GetOp() so we benefit from the existing fall-through */
/* paths (XLAT switch for opcodes >= ARRAYLEN(kNexgen32e)).                   */
/* ────────────────────────────────────────────────────────────────────────── */

extern nexgen32e_f GetOp(long op);

/* ────────────────────────────────────────────────────────────────────────── */
/* Hash function — just the high bits of the PC mixed with the low.  Guest    */
/* code is 16-byte aligned on most function entries; we mix the high half to  */
/* avoid clustering everything in the same bucket.                            */
/* ────────────────────────────────────────────────────────────────────────── */

static inline u32 HashPc(u64 pc, u32 mask) {
  /* Splittable64 mix — cheap, scrambles low bits well. */
  pc ^= pc >> 33;
  pc *= 0xff51afd7ed558ccdULL;
  pc ^= pc >> 33;
  return (u32)pc & mask;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Init / teardown.                                                           */
/* ────────────────────────────────────────────────────────────────────────── */

static bool TcParseEnabled(void) {
  const char *e = getenv("FIREBOX_TC");
  if (!e || !*e) return true;            /* default ON */
  if (!strcmp(e, "0") || !strcmp(e, "off") ||
      !strcmp(e, "false") || !strcmp(e, "no")) {
    return false;
  }
  return true;
}

/* SIGTERM/SIGINT handler — emits the §Q7 profile dump before terminating.
 * Without this, a bench wrapped in `timeout` sends SIGTERM, the process dies
 * without running atexit handlers, and the ranking data is lost.  Async-signal
 * safety: FbxTcProfileDump uses only snprintf into stack buffers + write(2);
 * both are AS-safe.  After dumping we restore the default handler and re-raise
 * so the parent sees the original termination semantics. */
static void FbxTcSignalDump(int sig) {
  FbxTcProfileDump();
  signal(sig, SIG_DFL);
  raise(sig);
}

void FbxTcInit(struct System *sys) {
  EnsureTcTraceFlag();
  /* Register the §Q7 profile dump exactly once.  We can't use a static
   * initialiser because atexit() isn't constexpr; gate on a flag.  We also
   * install SIGTERM/SIGINT handlers so a `timeout`-wrapped run still dumps
   * the ranking before terminating.  SIGQUIT/SIGABRT/SIGSEGV are NOT trapped
   * — those carry information (core dump, abort message) the developer needs
   * to see uncorrupted by the dump's writes. */
  if (g_tc_trace == 2 && !g_op_dump_registered) {
    atexit(FbxTcProfileDump);
    signal(SIGTERM, FbxTcSignalDump);
    signal(SIGINT, FbxTcSignalDump);
    g_op_dump_registered = 1;
  }
  if (sys->tc.initialised) return;
  sys->tc.enabled = TcParseEnabled() ? 1 : 0;
  sys->tc.entry_cap = FBX_TC_DEFAULT_ENTRY_CAP;
  if (sys->tc.enabled) {
    sys->tc.nbuckets = FBX_TC_INIT_BUCKETS;
    sys->tc.buckets = (struct FbxTcBlock **)calloc(
        sys->tc.nbuckets, sizeof(struct FbxTcBlock *));
    if (!sys->tc.buckets) {
      /* OOM — degrade gracefully to legacy dispatch.  Not fatal. */
      sys->tc.enabled = 0;
      sys->tc.nbuckets = 0;
    }
  }
  sys->tc.block_count = 0;
  sys->tc.entry_count = 0;
  sys->tc.initialised = 1;
  if (g_tc_trace) {
    TcTraceLine("[tc] init: enabled=%d nbuckets=%u entry_cap=%u\n",
                sys->tc.enabled, sys->tc.nbuckets, sys->tc.entry_cap);
  }
}

static void FreeBlockChain(struct FbxTcBlock *b) {
  while (b) {
    struct FbxTcBlock *next = b->next;
    /* firebox#719: free the guest-owned per-block FbxT2BlockCtx (NULL
     * unless the block escalated to Tier 2).  Allocated in
     * Fbxt2TryEscalate; outlives every dispatch of t2_funcref and is
     * reclaimed here when the block itself is dropped (FbxTcInvalidate /
     * SMC).  The matching funcref's host-side module is dropped via
     * fbx_t2_drop_all on the same invalidate path. */
    free(b->t2_block_ctx);
    free(b->entries);
    free(b);
    b = next;
  }
}

void FbxTcReset(struct System *sys) {
  u32 i;
  if (!sys->tc.initialised) return;
  if (sys->tc.buckets) {
    for (i = 0; i < sys->tc.nbuckets; ++i) {
      FreeBlockChain(sys->tc.buckets[i]);
      sys->tc.buckets[i] = NULL;
    }
  }
  sys->tc.block_count = 0;
  sys->tc.entry_count = 0;
}

void FbxTcInvalidate(struct System *sys) {
  if (!sys->tc.initialised) return;
  if (g_tc_trace) {
    TcTraceLine("[tc] invalidate: %u blocks / %u entries dropped\n",
                sys->tc.block_count, sys->tc.entry_count);
  }
  /* TODO(v0.2): RCU-style deferred reclamation for the multi-thread case.
   * Today's eager free is safe only when no other Machine in this System
   * is mid-ExecuteBlock — true for the v0.1 invalidate paths (LoadElf
   * runs on a fresh exec with no peer machines; SMC is rare on real
   * workloads).  If multi-threaded guests start triggering SMC under
   * load, ExecuteBlock could read freed memory.  Mitigation v0.2:
   * stamp a per-System epoch counter; ExecuteBlock captures epoch on
   * entry; FbxTcInvalidate bumps the epoch + parks free() in an RCU
   * grace queue. */
  FbxTcReset(sys);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Lookup + insertion.                                                        */
/* ────────────────────────────────────────────────────────────────────────── */

struct FbxTcBlock *FbxTcLookup(struct System *sys, u64 pc) {
  u32 idx;
  struct FbxTcBlock *b;
  if (!sys->tc.initialised || !sys->tc.buckets) return NULL;
  idx = HashPc(pc, sys->tc.nbuckets - 1);
  for (b = sys->tc.buckets[idx]; b; b = b->next) {
    if (b->start_pc == pc) return b;
  }
  return NULL;
}

static void InsertBlock(struct System *sys, struct FbxTcBlock *b) {
  u32 idx = HashPc(b->start_pc, sys->tc.nbuckets - 1);
  b->next = sys->tc.buckets[idx];
  sys->tc.buckets[idx] = b;
  ++sys->tc.block_count;
  sys->tc.entry_count += b->nentries;
}

void FbxTcGetStats(struct System *sys, u32 *blocks, u32 *entries) {
  if (blocks) *blocks = sys->tc.initialised ? sys->tc.block_count : 0;
  if (entries) *entries = sys->tc.initialised ? sys->tc.entry_count : 0;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Thunk lookup helper — finds a registered thunk at the given guest PC.      */
/* Returns the trampoline pointer or NULL.  Linear scan; the thunk table is   */
/* tiny (<= 5 entries) and only consulted at compile time, not on the hot    */
/* loop, so the cost is fully amortised.                                      */
/* ────────────────────────────────────────────────────────────────────────── */

static void (*LookupThunkAt(struct System *sys, u64 pc))(struct Machine *) {
  const struct FbxThunks *t = &sys->thunks;
  int i;
  if (!t->count) return NULL;
  if (pc < t->min_pc || pc > t->max_pc) return NULL;
  for (i = 0; i < t->count; ++i) {
    if (t->entries[i].pc == pc) return t->entries[i].trampoline;
  }
  return NULL;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Block compilation — decode forward from `start_pc` until we hit a          */
/* branch/precious/serializing op (block-end) OR a registered thunk PC       */
/* (terminates block; thunk takes over) OR an entry-cap overflow.             */
/*                                                                            */
/* We mirror JitlessDispatch's logic exactly: LoadInstruction populates       */
/* m->xedd, we read rde/disp/uimm0/length and stash them; advance the         */
/* decode PC by length and loop.                                              */
/*                                                                            */
/* Returns the freshly-allocated block on success, NULL on:                   */
/*   - LoadInstruction raised a fault (m gets longjmp'd out before return,   */
/*     so we never see this case; defensive only)                             */
/*   - malloc failure (degrade to legacy dispatch)                            */
/*   - entry-cap reached and we'd have produced a zero-entry block            */
/* ────────────────────────────────────────────────────────────────────────── */

extern void LoadInstruction(struct Machine *m, u64 pc);
extern int ClassifyOp(u64 rde) pureconst;

/* Defensive: cap a single block's compile work to avoid pathological        */
/* "decode 100k linear ops before any branch" cases (synthetic; real x86     */
/* code branches every <100 ops on average).                                  */
static struct FbxTcBlock *CompileBlock(struct Machine *m, u64 start_pc) {
  struct FbxTcBlock *b;
  struct FbxTcEntry *staging;
  u32 n = 0;
  u64 pc = start_pc;
  staging = (struct FbxTcEntry *)calloc(FBX_TC_MAX_BLOCK_ENTRIES,
                                        sizeof(struct FbxTcEntry));
  if (!staging) return NULL;
  while (n < FBX_TC_MAX_BLOCK_ENTRIES) {
    void (*thunk)(struct Machine *);
    u64 rde;
    i64 disp;
    u64 uimm0;
    u8 oplen;
    int opclass;
    /* Phase 1 thunk takes priority — if a thunk is registered at this PC,
     * emit a single THUNK entry and end the block.  The thunk's trampoline
     * is the function-body replacement (it performs the libc op + ret),
     * so anything after it in the same "block" never executes naturally. */
    if ((thunk = LookupThunkAt(m->system, pc))) {
      staging[n].ip = pc;
      staging[n].rde = 0;
      staging[n].disp = 0;
      staging[n].uimm0 = 0;
      staging[n].fn = (void *)thunk;
      staging[n].oplen = 0;
      staging[n].kind = FBX_TC_KIND_THUNK;
      memset(&staging[n].xedd, 0, sizeof(staging[n].xedd));
      ++n;
      break;
    }
    /* Decode the next instruction at `pc`.  LoadInstruction populates
     * m->xedd, and on fault it longjmp's via HaltMachine, so we never
     * return from a failed decode here. */
    LoadInstruction(m, pc);
    rde = m->xedd->op.rde;
    disp = m->xedd->op.disp;
    uimm0 = m->xedd->op.uimm0;
    oplen = Oplength(rde);
    /* Safety: refuse zero-length ops (would loop forever). */
    if (oplen == 0) {
      free(staging);
      return NULL;
    }
    /* Populate this entry. */
    staging[n].ip = pc;
    staging[n].rde = rde;
    staging[n].disp = disp;
    staging[n].uimm0 = uimm0;
    staging[n].fn = (void *)GetOp(Mopcode(rde));
    staging[n].oplen = oplen;
    staging[n].kind = FBX_TC_KIND_NORMAL;
    /* Copy the full xedd so dispatchers reading m->xedd directly work. */
    staging[n].xedd = *m->xedd;
    ++n;
    /* Block ends on branching/precious/serializing ops.  We INCLUDE the
     * terminating op in the block (we still need to call its handler);
     * the handler will mutate m->ip itself (jumps, calls) and our
     * post-loop check sees the divergence. */
    opclass = ClassifyOp(rde);
    if (opclass != 0 /* kOpNormal */) {
      break;
    }
    /* Advance to the next decode PC.  If this op overlaps a page
     * boundary, terminate the block — re-decoding across page seams
     * is rare and not worth the extra complexity for v0.1. */
    {
      u64 next = pc + oplen;
      if ((pc & ~(u64)4095) != ((next - 1) & ~(u64)4095)) {
        /* Op crossed a page; safer to end block. */
        break;
      }
      pc = next;
    }
  }
  if (n == 0) {
    free(staging);
    return NULL;
  }
  b = (struct FbxTcBlock *)calloc(1, sizeof(struct FbxTcBlock));
  if (!b) {
    free(staging);
    return NULL;
  }
  /* Trim allocation to actual size. */
  b->entries = (struct FbxTcEntry *)realloc(
      staging, n * sizeof(struct FbxTcEntry));
  if (!b->entries) {
    /* realloc-shrink can theoretically fail; fall back to keeping the
     * over-large alloc. */
    b->entries = staging;
  }
  b->start_pc = start_pc;
  b->end_pc = staging[n - 1].ip + staging[n - 1].oplen;
  b->page = start_pc & ~(u64)4095;
  b->nentries = n;
  b->hits = 0;
  /* Tier 2 §6.3: fresh block has no translation.  -1 (NOT 0) is the
   * "no funcref" sentinel — 0 is a valid funcref. */
  b->t2_funcref = -1;
  b->t2_attempted = 0;
  b->t2_pending_attempts = 0; /* #794 async escalation */
  b->t2_retry_at_hits = 0;    /* #794 async escalation */
  b->t2_dispatches = 0;       /* #794 3b — de-escalation feedback */
  b->t2_filled = 0;
  b->next = NULL;
  return b;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Hot path — the dispatcher.                                                 */
/* ────────────────────────────────────────────────────────────────────────── */

static void ExecuteBlock(struct Machine *m, struct FbxTcBlock *b) {
  u32 i;
  /* §6.4 bailout-resume (firebox#735): the entries[] index Tier 1 resumes
   * at.  0 for the common path (no T2 funcref, or exit=0/2 already
   * returned).  On a T2 bailout (exit=1) we resume at the entry whose
   * pc == m->ip — see the resume-scan below. */
  u32 resume_idx = 0;
  ++b->hits;

  /* Tier 2 §6.4 fast path — if we have a Tier-2-translated funcref,
   * dispatch through it instead of walking entries[].  Bailout (exit=1)
   * RESUMES the Tier 1 entries[] walk at m->ip (firebox#735, below);
   * host-call escape (exit=2) and normal completion (exit=0) both return
   * immediately (the dispatched module has already advanced m->ip).
   *
   * Why resume-at-m->ip on exit=1 (firebox#735, replaces the prior
   * re-walk-from-index-0): a BAILOUT terminator emits
   * `EmitStoreIp(bailout_pc); return 1`, where `bailout_pc` is the guest
   * PC of the FIRST instruction the T2 block could not lower (the lifter
   * emits BAILOUT at `e->ip` for that entry — fbx_ir_lift.c).  A
   * mid-block translation-miss bailout (the §13.5 inline software-MMU
   * path — fbx_ir_emit_wasm.c EmitGuestVaToHostOffsetOrBailout) likewise
   * stores the PC of the FAILING memory op and returns 1, having
   * committed nothing for that op (translation precedes every state
   * mutation).  In BOTH cases the T2 block has committed exactly the
   * ops at PCs strictly BELOW m->ip; the ops at m->ip and after were NOT
   * run.  So Tier 1 must resume at the entry whose pc == m->ip — running
   * exactly the not-yet-committed suffix.  The pre-#735 code re-walked
   * from index 0, RE-EXECUTING every committed op (REG_SET / ALU / stack)
   * a second time → guest state diverged → SIGSEGV.  #719 had to refuse
   * any BAILOUT-after-commit block to dodge this; #735 fixes the handoff
   * so those blocks (and the §13.5 memory-operand blocks) are sound.
   *
   * We deliberately do NOT clear `b->t2_funcref` — the translation is
   * still valid for the common case; a bailout on one dispatch (e.g. a
   * cold-TLB miss) isn't evidence of pervasive corruption, and the next
   * hit may translate cleanly. */
  if (b->t2_funcref >= 0) {
    int exit_code = Fbxt2Dispatch(m, b);
    if (exit_code == 0 || exit_code == 2) {
      /* Normal completion or host-call escape — both leave m->ip in
       * the right state for the outer dispatch loop. */
      return;
    }
    /* exit_code == 1: bailout to Tier 1.  Resume at the entry whose
     * pc == m->ip (the bailout PC the T2 block stored).  Linear scan —
     * blocks are short (typically < 16 entries; the spec §13.4 bounded
     * size); a pc→index map would be premature.  If no entry matches
     * (should not happen — the bailout PC always equals some entry's ip
     * by construction in the lifter), fall back to index 0, which is the
     * pre-#735 behavior — safe for the no-committed-prefix case the #719
     * gate still guarantees as a backstop, and loud via the assert. */
    {
#ifndef FBX735_DIAG_NO_RESUME_AT_IP
      u32 j;
      u64 ip = m->ip;
      int found = 0;
      for (j = 0; j < b->nentries; ++j) {
        if (b->entries[j].ip == ip) {
          resume_idx = j;
          found = 1;
          break;
        }
      }
      unassert(found || b->nentries == 0);
      (void)found;
#endif
#ifdef FBX735_DIAG_RESUME_TRACE
      {
        static _Atomic(long) dbg_n = 0;
        long n = atomic_fetch_add_explicit(&dbg_n, 1, memory_order_relaxed);
        /* firebox#738: log ONLY the dangerous resumes — a mid-block resume
         * (resume_idx != 0) means the T2 block committed a prefix before
         * bailing, the exact double-commit window. Start-of-block resumes are
         * sound and flood the trace. Also flag found=0 (scan miss). */
        if ((resume_idx != 0 || !found) && n < 2000) {
          fprintf(stderr,
                  "[t735 resume] ip=%#llx start=%#llx end=%#llx nent=%u "
                  "resume_idx=%u found=%d e0_ip=%#llx\n",
                  (unsigned long long)ip, (unsigned long long)b->start_pc,
                  (unsigned long long)b->end_pc, b->nentries, resume_idx, found,
                  (unsigned long long)b->entries[0].ip);
        }
      }
#endif
    }
  }

  for (i = resume_idx; i < b->nentries; ++i) {
    struct FbxTcEntry *e = &b->entries[i];
    if (e->kind == FBX_TC_KIND_THUNK) {
      /* Thunk replaces the function body + emulates ret.  After the
       * call m->ip points to the return PC; control returns to Actor's
       * main loop.  Thunks are accounted at a synthetic mop (0xFFF) so
       * they don't pollute the §Q7 ranking with libc-internal weight. */
      if (g_tc_trace == 2) {
        u64 t0 = FbxNowNs();
        ((void (*)(struct Machine *))e->fn)(m);
        {
          u64 dt = FbxNowNs() - t0;
          g_op_count_by_mop[0xFFFu] += 1;
          g_op_ns_by_mop[0xFFFu] += dt;
          g_op_total_count += 1;
          g_op_total_ns += dt;
        }
      } else {
        ((void (*)(struct Machine *))e->fn)(m);
      }
      goto post_tier1;
    }
    /* Mirror JitlessDispatch's per-op sequence: oplen for fault rewind,
     * advance ip, set m->xedd to the cached decode, call the handler,
     * commit any stash, clear oplen.  See machine.c lines 2099-2118.
     *
     * §Q7 profile mode (FIREBOX_TC_TRACE=2) wraps the handler call with
     * a CLOCK_MONOTONIC measurement.  The branch is a single global
     * load + compare-against-immediate; in non-trace mode the cost is
     * ~negligible (mispredict-free since g_tc_trace is set once at
     * init and never written again on the hot path).  In trace=2 mode
     * the measurement adds ~50-200ns per opcode on macOS aarch64 — a
     * very large tax (often >50% of run time on tight loops) but
     * acceptable for a one-shot ranking measurement.  Acceptance:
     * the RANKING is what we trust, not absolute ns figures. */
    if (g_tc_trace == 2) {
      u64 mop = Mopcode(e->rde);
      u64 bucket = mop & (FBX_TC_PROFILE_BUCKETS - 1u);
      u64 t0 = FbxNowNs();
      m->oplen = (u8)e->oplen;
      m->ip += e->oplen;
      m->xedd = &e->xedd;
      ((void (*)(struct Machine *, u64, i64, u64))e->fn)(
          m, e->rde, e->disp, e->uimm0);
      if (m->stashaddr) CommitStash(m);
      m->oplen = 0;
      {
        u64 dt = FbxNowNs() - t0;
        g_op_count_by_mop[bucket] += 1;
        g_op_ns_by_mop[bucket] += dt;
        g_op_total_count += 1;
        g_op_total_ns += dt;
      }
    } else {
      m->oplen = (u8)e->oplen;
      m->ip += e->oplen;
      m->xedd = &e->xedd;
      ((void (*)(struct Machine *, u64, i64, u64))e->fn)(
          m, e->rde, e->disp, e->uimm0);
      if (m->stashaddr) CommitStash(m);
      m->oplen = 0;
    }
    /* A branching op may have mutated m->ip; the next iteration's PC
     * may no longer match our cached layout.  We don't enforce this —
     * branches are ALWAYS the last entry in their block (CompileBlock
     * ends on kOpBranching), so the loop naturally exits after them. */
  }

post_tier1:
  /* Tier 2 §6.4 escalation — opportunistically try to escalate AFTER the Tier 1
   * dispatch returns.  Cost is amortised: only when the block has been hit
   * `FBX_T2_HOTNESS_THRESHOLD` times, and (for a TERMINAL outcome) only once —
   * t2_attempted latches at success or permanent failure.
   *
   * firebox#794 async escalation: a host that compiles in the background
   * returns PENDING (not terminal), leaving t2_attempted clear so we re-attempt.
   * The `b->hits >= b->t2_retry_at_hits` guard bounds the retry cadence:
   * Fbxt2TryEscalate sets `t2_retry_at_hits = hits + FBX_T2_PENDING_RETRY_STRIDE`
   * on each PENDING outcome, so a self-loop (dispatched once per iteration)
   * polls every STRIDE hits instead of re-lifting on every one.  t2_retry_at_hits
   * is 0 for a fresh block, so the FIRST attempt still fires exactly at the
   * threshold; a synchronous host never returns PENDING, so the guard is inert
   * there (the block latches terminal on the first attempt as before). */
  if (!b->t2_attempted && b->hits >= Fbxt2HotnessThreshold() &&
      b->hits >= b->t2_retry_at_hits) {
    Fbxt2TryEscalate(m, b);
  }
}

bool FbxTcMaybeDispatch(struct Machine *m) {
  struct System *sys = m->system;
  struct FbxTcBlock *b;
  u64 ip;
  /* Lazy init — first dispatch initialises the cache. */
  if (!sys->tc.initialised) {
    FbxTcInit(sys);
  }
  if (!sys->tc.enabled) return false;
  /* SMC handshake: if Blink's icache was invalidated (memorymalloc.c
   * sets opcache->invalidated on mmap/munmap that overlaps executable
   * pages), our cache is stale too — flush it.  We check m's opcache
   * (per-machine) rather than walking all machines; the invariant we
   * need is "if THIS thread saw the invalidation, the cache it's about
   * to read from must already be flushed".  The other-thread flush
   * runs lazily when each thread hits its own dispatch. */
  if (atomic_load_explicit(&m->opcache->invalidated, memory_order_acquire)) {
    /* Leave the actual icache flush to LoadInstruction's normal path
     * (which clears `invalidated`) — but flush TC eagerly. */
    FbxTcInvalidate(sys);
  }
  ip = m->ip;
  b = FbxTcLookup(sys, ip);
  if (!b) {
    /* Entry-cap check — flush if we'd exceed the cap.  Crude but
     * deterministic; LRU eviction is a v0.2 concern. */
    if (sys->tc.entry_cap && sys->tc.entry_count >= sys->tc.entry_cap) {
      if (g_tc_trace) {
        TcTraceLine("[tc] entry cap reached (%u) — flushing\n",
                    sys->tc.entry_count);
      }
      FbxTcReset(sys);
    }
    b = CompileBlock(m, ip);
    if (!b) return false;
    InsertBlock(sys, b);
    if (g_tc_trace) {
      TcTraceLine("[tc] compiled block @ %#llx: %u entries (%llu..%llu)\n",
                  (unsigned long long)b->start_pc, b->nentries,
                  (unsigned long long)b->start_pc,
                  (unsigned long long)b->end_pc);
    }
  }
  ExecuteBlock(m, b);
  return true;
}
