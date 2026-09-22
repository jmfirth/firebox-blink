#ifndef BLINK_THREADEDCODE_H_
#define BLINK_THREADEDCODE_H_

/*
 * Firebox ELF-perf Phase 2 — Tier 1: threaded-code dispatcher.
 *
 * Motivated by Phase 1.4's load-bearing falsification (see
 * work/tracks/elf-performance/phase-1-thunking/phase-1.4-report.md):
 * the per-instruction `FbxThunksMaybeDispatch` check in
 * JitlessDispatch/ExecuteInstruction costs +14.8% on `busybox-awk-10k`
 * (pure-ALU workload that never fires a thunk).  The wedge is
 * architectural — Blink's interpreter loop is too tight to absorb ANY
 * per-instruction tax.
 *
 * Fix: move dispatch out of the per-instruction path.  When Blink
 * encounters a basic block (sequence of instructions ending in a
 * branch/syscall/serializing op), compile it once to an array of
 * pre-decoded (Op_fn_ptr, rde, disp, uimm0, oplen) entries.  Cache by
 * start-PC.  On re-execution, walk the array in a tight loop — decode
 * happens ONCE per block, not per instruction.
 *
 * Composes with Phase 1 thunks: at block-compile time, GetThunkAt is
 * checked for each PC; if a thunk is registered, the block-compiler
 * inserts a synthesised "thunk handler" entry that short-circuits and
 * ends the block.  The per-instruction FbxThunksMaybeDispatch in
 * JitlessDispatch is retired in this Tier (its work moves to
 * block-compile time).
 *
 * NOT a native JIT: no machine-code emission, no mmap PROT_EXEC, no
 * host-module instantiation.  Pure C data structure + dispatcher loop;
 * portable to wasm without any runtime codegen.
 *
 * Q1-Q6 verdicts (see work/tracks/elf-performance/phase-2-hot-path-caching/
 * tier-1-v0.1-report.md for measurement evidence):
 *   Q1 [measured]: function-entry boundaries respected via ClassifyOp →
 *      kOpBranching ending the prior block; callee's first PC starts a
 *      new block. Naturally aligns with how the interpreter sees calls.
 *   Q2 [measured]: in firebox's wasm build HAVE_JIT is undefined, so
 *      GeneralDispatch is entirely #ifdef'd out and Blink's native-JIT
 *      basic-block detection (AddPath/CompletePath) is dead code.  Tier 1
 *      must build its own block-detection in JitlessDispatch.
 *   Q3 [decided]: parallel FbxTcCacheState, NOT extension of JitHooks.
 *      Avoids tangling our cache with the dormant native-JIT machinery
 *      whose `disabled` flag is permanently true on wasm.
 *   Q4 [measured]: no mmap/mprotect/g_jit interaction.  Pure host malloc.
 *   Q5 [measured]: piggyback on opcache->invalidated.  When SMC or
 *      mmap-change clears the icache, we clear the TC cache too.
 *   Q6 [measured]: cached PC includes ASLR skew (LoadElf adds it before
 *      we ever cache).  Op_fn_ptr is from kNexgen32e[] in .rodata; PIE
 *      relocations don't affect host-side function-pointer values.
 */

#include "blink/builtin.h"
#include "blink/machine.h"
#include "blink/types.h"
#include "blink/x86.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One pre-decoded instruction.  We embed a full XedDecodedInst (~40
 * bytes) so the dispatcher can assign `m->xedd = &entry->xedd` before
 * calling the handler — preserving the few handlers (OpAam/OpAad in
 * bcd.c, blinkenlights.c diagnostics) that read `m->xedd->op.uimm0`
 * directly rather than via the P macro arg.  The (rde,disp,uimm0)
 * triple stored here mirrors entry->xedd.op.{rde,disp,uimm0} so the
 * tight loop reads from cache-warm fields without indirection. */
struct FbxTcEntry {
  u64 ip;            /* guest PC where this entry was decoded */
  u64 rde;           /* mirrors xedd.op.rde — cache-warm */
  i64 disp;          /* mirrors xedd.op.disp */
  u64 uimm0;         /* mirrors xedd.op.uimm0 */
  /* Either the opcode handler from kNexgen32e[] (kind == NORMAL) OR
   * the thunk trampoline (kind == THUNK).  Stored as void* to keep
   * the union punning consistent; cast at dispatch. */
  void *fn;
  u16 oplen;         /* x86 instruction length in bytes (Oplength(rde)) */
  u8 kind;           /* see FBX_TC_KIND_* below */
  u8 reserved;
  struct XedDecodedInst xedd; /* full decoded instruction */
};

#define FBX_TC_KIND_NORMAL  0  /* normal opcode dispatch — call fn(P) */
#define FBX_TC_KIND_THUNK   1  /* Phase 1 thunk hit — call fn(m), end block */

/* A cached basic block: contiguous sequence of pre-decoded instructions
 * ending at the first branching/precious/serializing op (or at a thunk
 * hit).  Indexed by start_pc in the hash table; chained via `next`.
 *
 * ⛔ firebox#HYS — THE TIER 2 STATE IS NOT HERE ANY MORE, AND MUST NOT COME
 * BACK.  This struct hangs off `sys->tc`, i.e. `struct System`, which every
 * thread of the process shares.  A Tier-2 `funcref` is an index into
 * `__indirect_function_table`, which is per WASM INSTANCE GROUP — a spawned
 * thread re-instantiates the module into a new Store with a FRESH table at the
 * module's declared minimum.  Storing the index in shared memory meant thread 2
 * read thread 1's index and `call_indirect`ed past the end of its own table:
 * "undefined element: out of bounds table access".  Nothing about the two
 * lifetimes matches, so the state moved to `struct FbxT2BlockState`, held in a
 * per-`Machine` map (`m->fbx_t2`) keyed by `start_pc`.  See that struct below.
 *
 * What legitimately REMAINS shared here is everything about the DECODE: the
 * entries[], the pc bounds, and `hits`.  `hits` stays shared on purpose — it is
 * a best-effort hotness signal, and a thread arriving at an already-hot block
 * should escalate its OWN translation immediately rather than re-earn the
 * count.  It is also what keeps the cold path cheap: `hits < threshold` proves
 * no Machine anywhere holds Tier-2 state for this block, so ExecuteBlock skips
 * the map lookup entirely on every cold block. */
struct FbxTcBlock {
  u64 start_pc;            /* guest PC of the first instruction */
  u64 end_pc;              /* guest PC immediately after the last instruction */
  u64 page;                /* (start_pc & ~0xfff) — used by SMC invalidation */
  u32 nentries;            /* number of entries */
  u32 hits;                /* execution count (best-effort; not atomic) */
  struct FbxTcEntry *entries; /* malloc'd array of nentries entries */
  struct FbxTcBlock *next; /* next block in the same hash bucket */
};

/* firebox#HYS — the per-INSTANCE half of a Tier-2-translated block.  One of
 * these exists per (Machine, block) pair that actually escalated; the fields
 * are verbatim the ones that used to sit on FbxTcBlock.
 *
 *   funcref:          -1 = no translation; >=0 = an index into THIS instance's
 *                     `__indirect_function_table`.  Meaningless in any other
 *                     instance — that is the whole point of the move.
 *   attempted:        TERMINAL escalation latch (success or permanent
 *                     failure).  A PENDING async compile does NOT set it
 *                     (firebox#794) so the block retries.
 *   pending_attempts: firebox#794 — count of PENDING async-compile polls so
 *                     far; gives up at FBX_T2_PENDING_MAX_ATTEMPTS.
 *   block_ctx:        firebox#719 — the guest-heap FbxT2BlockCtx whose address
 *                     IS the `block_ctx_ptr` replayed as `translated_block`'s
 *                     2nd arg.  It must outlive every dispatch of `funcref`,
 *                     so it is owned here and freed with this entry.  Because
 *                     the funcref it pairs with is per-instance, so is this.
 *   dispatches/filled: #794 3b de-escalation feedback.
 *   retry_at_hits:    #794 async escalation retry cadence, compared against the
 *                     SHARED `b->hits`. */
struct FbxT2BlockState {
  u64 start_pc;            /* key; matches FbxTcBlock::start_pc.  A live block
                            * is uniquely identified by start_pc (FbxTcLookup
                            * hashes on it), so this keys without ever holding
                            * a pointer to memory another thread may free. */
  void *block_ctx;
  i32 funcref;
  u8 attempted;
  u8 pending_attempts;
  u8 used;                 /* 0 = empty slot (open addressing), 1 = live */
  u8 reserved;
  u32 dispatches;
  u32 filled;
  u32 retry_at_hits;
};

/* firebox#HYS — a Machine's Tier-2 map.  Open-addressed, linear-probed,
 * power-of-two sized, keyed by `start_pc`.  Deliberately has no delete: the
 * only removal event is a whole-cache invalidation, which drops every entry at
 * once via the epoch check, so tombstones would be dead weight. */
struct FbxT2MachineState {
  struct FbxT2BlockState *slots; /* nslots entries; calloc'd */
  u32 nslots;                    /* power of two, 0 when unallocated */
  u32 nused;
  u32 epoch;                     /* sys->tc.epoch this map was built under */
};

/* Look up — creating on demand — this Machine's Tier-2 state for `b`.
 * Returns NULL only on allocation failure, which degrades to "stay Tier 1"
 * and is never fatal.  Drops the whole map first if `sys->tc.epoch` moved
 * (the block cache was invalidated under us), which is what keeps the
 * `start_pc` keys from naming blocks that no longer exist. */
struct FbxT2BlockState *FbxT2StateFor(struct Machine *m,
                                      struct FbxTcBlock *b);

/* Free this Machine's Tier-2 map and every guest-heap block_ctx it owns.
 * Idempotent; safe on a Machine that never escalated. */
void FbxT2StateFree(struct Machine *m);

/* Initialise the cache embedded in `sys->tc`.  Idempotent.  Reads
 * FIREBOX_TC env var: empty/missing/"1"/"on"/"true" → enabled;
 * "0"/"off"/"false" → disabled. */
void FbxTcInit(struct System *sys);

/* Free every cached block + the bucket array.  Leaves `enabled` flag
 * intact; FbxTcInit must be called before subsequent dispatch. */
void FbxTcReset(struct System *sys);

/* Equivalent to FbxTcReset but called from the icache-invalidation path
 * (memorymalloc.c:498).  We piggyback on opcache->invalidated so SMC
 * and mmap-change automatically flush TC too.  Currently this is just
 * an alias for FbxTcReset; kept separate so we can do per-page eviction
 * later without changing call sites. */
void FbxTcInvalidate(struct System *sys);

/* Hot path — called from JitlessDispatch on every interpreted op.
 *
 * Returns true if we executed via the TC dispatcher (and m->ip is
 * already advanced past the cached block); false if the caller must
 * fall through to the legacy decode+dispatch path.
 *
 * Semantics:
 *   - If a block exists at m->ip: execute it via the tight TC loop.
 *     Updates m->ip, returns true.
 *   - Otherwise: compile a new block starting at m->ip (decode forward
 *     until ClassifyOp returns kOpBranching/kOpPrecious/kOpSerializing
 *     OR we hit a registered thunk PC OR we overflow the block-entry
 *     cap).  Then execute the freshly-compiled block.  Returns true.
 *   - If the cache is disabled (FIREBOX_TC=0): return false immediately
 *     and let the caller use the legacy path.
 *   - If block-compile fails (e.g. instruction decode error in the
 *     forward walk): return false; the caller's legacy decode will
 *     surface the fault via Blink's normal Halt path.
 */
bool FbxTcMaybeDispatch(struct Machine *m);

/* Test/diagnostic accessor — returns the block at PC or NULL.  Used by
 * unit tests; not used on the hot path. */
struct FbxTcBlock *FbxTcLookup(struct System *sys, u64 pc);

/* Diagnostic — total block / entry count (best-effort; not atomic). */
void FbxTcGetStats(struct System *sys, u32 *blocks, u32 *entries);

#ifdef __cplusplus
}
#endif

#endif /* BLINK_THREADEDCODE_H_ */
