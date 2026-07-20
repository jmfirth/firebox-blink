#ifndef BLINK_FBX_IR_H_
#define BLINK_FBX_IR_H_

/*
 * Firebox ELF-perf Phase 2 — Tier 2: hot-path-caching IR.
 *
 * This file defines the IR substrate that §13.2 (wasm synthesis),
 * §13.3 (firebox-wasix bridge), §13.4 (Blink escalation glue), and a
 * future Phase 4 offline ELF walker all consume.  The IR is a
 * linear three-address representation with virtual registers and
 * size-tagged opcodes.  See
 * `work/tracks/elf-performance/phase-2-hot-path-caching/tier-2-spec.md`
 * §3 ("The IR") for the design rationale.
 *
 * Three properties are LOAD-BEARING for the rest of the track:
 *
 *   1. Flat, pointer-free `FbxIrInst` records that serialise via a
 *      plain memcpy.  Phase 4's sidecar artefact format relies on
 *      this.  The `_Static_assert` at the bottom of the file locks
 *      the wire format; any change requires a `FBX_IR_VERSION` bump.
 *
 *   2. Deterministic lifting.  Given the same input bytes + same
 *      blink build, `fbx_ir_lift()` MUST produce a byte-identical
 *      FbxIrBlock.  This is what makes Phase 4 reachability work:
 *      runtime-lifted IR and offline-lifted IR must agree.  vreg
 *      allocation is a monotonic counter; no maps, no hash sets, no
 *      thread-local state, no clocks.  See spec §Q5 and design
 *      decisions §2.
 *
 *   3. Pure data, no profile.  Hit counts and `t2_attempted` latches
 *      live on `struct FbxTcBlock` (Tier 1's runtime cache entry),
 *      NOT on FbxIrBlock.  The IR is what the program IS; profile is
 *      what the program DID.  See design decision 7.
 *
 * Coverage scope for v0.1 / §13.1:
 *
 *   Top-10 opcodes (chosen from the frequency hints in machine.c
 *   lines 1611-2090):
 *
 *     OpMovEvqpGvqp     (0x089)  MOV r/m64, r64           — #1, 22.2%
 *     OpMovGvqpEvqp     (0x08B)  MOV r64, r/m64           — #12, 2.9%
 *     OpMovZvqpIvqp     (0x0B8-BF) MOV reg, imm64         — top-3% combined
 *     OpMovImm          (0x0C7)  MOV r/m, imm32           — 0.16%
 *     OpLeaGvqpM        (0x08D)  LEA r64, m               — 0.8%
 *     OpAlu             (group 1 r/m,r — common; covers the
 *                        ADD/SUB/CMP/AND/OR/XOR family at 0x01-0x39)
 *     OpAlui            (0x083)  group 1 r/m, imm8        — 6.5% (#4)
 *     OpAluTest         (0x085)  TEST r/m, r              — 0.6%
 *     OpJcc             (0x070-7F, 0x180-18F) Jcc rel     — 9%+ combined
 *     OpCallJvds        (0x0E8)  CALL rel32               — 0.4%
 *
 *   Everything else bails out: the lifting pass emits
 *   FBX_IR_BAILOUT at the offending PC and returns the partial block
 *   (or NULL if no usable prefix exists).  The runtime falls back to
 *   Tier 1 dispatch for the bailout op.  Per spec §4.2, the
 *   unsupported tail (~5% of total instructions) is acceptable for
 *   v0.1.
 *
 * NOT covered by §13.1:
 *
 *   - Wasm synthesis (§13.2):       fbx_ir_emit_wasm()
 *   - firebox-wasix host bridge (§13.3): the trait + impl side
 *   - Runtime escalation glue (§13.4): the patch to ExecuteBlock
 *   - Top-30 opcode extension (§13.5)
 *   - Cap/LRU/kill-switch (§13.6)
 *   - Diagnostic mode / property tests (§13.7)
 *
 * What this file gives §13.2 + Phase 4: a stable on-disk IR shape +
 * a lifting pass they can call without re-implementing.
 */

#include <stddef.h>

#include "blink/builtin.h"
#include "blink/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ────────────────────────────────────────────────────────────────────────── */
/* Versioning — the LOAD-BEARING constant for cache-key stability.            */
/*                                                                            */
/* The IR SHA fed to Phase 4's sidecar validation is computed as:             */
/*                                                                            */
/*   sha256(FBX_IR_VERSION || blink_git_sha || serialised_ir)                 */
/*                                                                            */
/* When the lifting pass gains a new opcode (e.g. SSE in v0.2), every block   */
/* that previously bailed out and now lifts produces a different IR — the     */
/* cache must invalidate.  Per spec §Q5 axis 3 (versioned-opcode determinism) */
/* the version constant absorbs that change: bumping FBX_IR_VERSION causes   */
/* every SHA to differ, so stale sidecars miss instead of hit-on-stale.       */
/*                                                                            */
/* Bump this when:                                                            */
/*   - any FBX_IR_OP_* enumerator is added, removed, or renumbered            */
/*   - the FbxIrInst layout changes (size, field order, semantics)            */
/*   - the lifting pass emits ANY new IR for previously-unsupported input     */
/* ────────────────────────────────────────────────────────────────────────── */

/* v1: §13.1 + §13.2 + §13.5 initial (lift+emit for top-30 mod3, dead-flag
 *     elision, no lazy-flag synthesis).
 * v2: §13.5 follow-on (#599) — lazy-flag wasm synthesis for ALU ops
 *     (ADD/SUB/AND/OR/XOR/CMP/TEST × widths 1/2/4/8) + BRANCH_COND emit
 *     for all 16 Jcc condition codes.  The IR shape is byte-identical to
 *     v1 (SET_FLAGS_RAW already carried op-kind in imm; BRANCH_COND
 *     already carried condition code in src1).  The version bump invalidates
 *     stale Phase 4 sidecars compiled against v1's "refuse" semantics so a
 *     fresh emit-side pass runs and the cache reflects the larger covered
 *     surface.  See work/tasks/599-* for the full closure narrative.
 * v3: §13.5c follow-on (#602) — CALL_DIRECT / RET / PUSH / POP wasm
 *     synthesis.  Adds FBX_IR_OP_PUSH and FBX_IR_OP_POP opcodes (new
 *     enumerators at the tail; pinned values preserved).  CALL_DIRECT
 *     and RET already existed; the version bump reflects the emit-side
 *     flip from "refuse" to "synthesize" for these four opcodes plus
 *     the two new PUSH/POP opcodes.  Stale Phase 4 sidecars compiled
 *     against v2's refuse semantics invalidate cleanly so the new emit
 *     pass runs.  See work/tasks/602-* for closure narrative.
 * v4: §13.5d follow-on (#677) — MOV r/m memory-form emit coverage.  The
 *     lifter now emits FBX_IR_OP_LOAD / FBX_IR_OP_STORE for MOV opcodes
 *     0x88/0x89/0x8A/0x8B in the `[base + disp]` addressing form (no SIB
 *     index, not RIP-relative); the emit pass synthesizes the matching
 *     guest-memory i64.load / i64.store.  Previously these blocks bailed
 *     out (reason=mod3_required) and stayed on Tier 1; the version bump
 *     reflects the lifter producing NEW IR for input that previously
 *     emitted only BAILOUT/REG_GET-REG_SET-then-refused.  Stale Phase 4
 *     sidecars compiled against v3 invalidate cleanly.  Index/RIP-relative
 *     forms still BAILOUT (correct-or-refuse).  See work/tasks/677-* for
 *     the closure narrative.
 * v5: §13.5d follow-on (#778) — inline SIB-index LEA emit coverage.  LiftLea
 *     now accepts the `[base + index*scale + disp]` form (it previously bailed
 *     to Tier 1 via the #738 correct-or-refuse): src2 carries the index greg,
 *     the new `scale` field (the renamed offset-20 word) carries the SIB scale,
 *     and EmitLea adds `weg[index]*scale` to the effective address.  The IR
 *     LAYOUT is unchanged (the `_pad2` slot is reused as `scale`, same offset +
 *     size), but the lifter now produces NEW IR (indexed LEAs) for input that
 *     previously emitted BAILOUT, AND the offset-20 word now carries meaning, so
 *     stale v4 sidecars (which assumed that word was always zero) must
 *     invalidate.  RIP-relative / no-base SIB LEAs still BAILOUT.  See
 *     work/tasks/778-* for the closure narrative.
 * v6: #794 (T3/T4 emit-coverage floor) — (1) new FBX_IR_OP_IMUL enumerator +
 *     lift/emit for the truncating IMUL forms (0x69/0x6B/0x0FAF, reg-form);
 *     (2) accumulator-immediate ALU forms (0x04/0x05…0x3C/0x3D) lifted via
 *     LiftAluAccImm; (3) immediate-operand ALU made EMITTABLE — REG_GET with
 *     an IMM src now lowers to a block-ctx IMM-slot load (previously the
 *     coverage gate refused it, so EVERY immediate-operand ALU silently stayed
 *     on Tier 1).  The lifter now produces NEW IR for input that previously
 *     bailed/refused, so stale v5 sidecars must invalidate.  Also corrects a
 *     latent width-4 register-write bug (32-bit writes now zero-extend, x86-64
 *     semantics) — see EmitRegSlotStore.  IMUL carries no flag synthesis;
 *     blocks where its flags are live at a reader refuse (correct-or-refuse).
 *     See work/tasks/794-* for the closure narrative.
 * v7: #CM4 (correctness) — the lifter now appends an explicit BRANCH_TAKEN to
 *     the fall-through PC when a Tier-1 block ends WITHOUT a control-flow op
 *     (blink split the run mid-stream, so the block flows into the next at
 *     end_pc).  Previously such a block lifted to a terminator-less IR; the
 *     emit's un-terminated fallback returned exit=0 without advancing m->ip, so
 *     the guest re-dispatched the SAME block forever (a 100%-CPU wasm wedge that
 *     blocked every elf-perf th=1 bench — the wedge fired only once a hot
 *     fall-through block escalated, which is why the default threshold=100 hid
 *     it).  Fall-through blocks now produce NEW IR (the appended BRANCH_TAKEN),
 *     so stale v6 sidecars/caches must invalidate cleanly.  See work/tasks/CM4-*
 *     for the RCA + closure narrative. */
#define FBX_IR_VERSION 7u

/* ────────────────────────────────────────────────────────────────────────── */
/* IR opcodes.                                                                */
/*                                                                            */
/* The discriminator on FbxIrInst.opcode.  Numeric values are PINNED — never  */
/* renumber an existing entry.  Append new opcodes at the end and bump        */
/* FBX_IR_VERSION.                                                            */
/*                                                                            */
/* Why pinned values: §Q5 axis 2 (cross-build determinism).  Two builds of    */
/* the same blink source must produce byte-identical IR; reordering this enum */
/* would silently change the wire format.                                     */
/* ────────────────────────────────────────────────────────────────────────── */

enum FbxIrOpcode {
  /* Pseudo-op: marks the original guest PC.  Emitted at the start of each
   * lifted x86 instruction.  Runtime BAILOUT uses this to resume Tier 1
   * at the correct PC.  Phase 4 carries it as source-position metadata. */
  FBX_IR_OP_PC_MARK = 0,

  /* Memory access against guest linear memory.  Width in `width`.
   *
   * v0.1 (#677) lowers the `[base + disp]` addressing form only — a
   * single guest base register plus a signed displacement, no SIB
   * index/scale, not RIP-relative.  The full `src2*scale` index term is
   * RESERVED for a later increment (the lifter refuses index/RIP forms
   * via BAILOUT today, so no IR carrying them is ever produced).
   *
   * LOAD  encoding: dst_kind=VREG  dst=dest vreg;
   *                 src1_kind=GREG src1=base greg id;
   *                 src2_kind=NONE (reserved for index);
   *                 imm=signed displacement (i64, sign-extended);
   *                 width=access width (1/2/4/8).  Narrow widths
   *                 zero-extend (matches the x86 MOV/MOVZX load shape;
   *                 sign-extending loads are out of scope here).
   *
   * STORE encoding: dst_kind=NONE;
   *                 src1_kind=GREG src1=base greg id (address);
   *                 src2_kind=VREG src2=value vreg to store;
   *                 imm=signed displacement (i64, sign-extended);
   *                 width=access width (1/2/4/8). */
  FBX_IR_OP_LOAD = 1,    /* dst(vreg) = mem[greg(src1) + (i64)imm]    */
  FBX_IR_OP_STORE = 2,   /* mem[greg(src1) + (i64)imm] = vreg(src2)   */

  /* Guest register file access. */
  FBX_IR_OP_REG_GET = 3, /* dst (vreg) = guest_reg[src1.reg_id]  (width-sized) */
  FBX_IR_OP_REG_SET = 4, /* guest_reg[dst.reg_id] = src1            (width-sized) */

  /* Integer ALU.  Two-operand form (dst = src1 OP src2).  `width` carries
   * the operand width in bytes (1/2/4/8). */
  FBX_IR_OP_ADD = 5,
  FBX_IR_OP_SUB = 6,
  FBX_IR_OP_AND = 7,
  FBX_IR_OP_OR  = 8,
  FBX_IR_OP_XOR = 9,
  FBX_IR_OP_CMP = 10,  /* sets flags only; no dst write */
  FBX_IR_OP_TEST = 11, /* sets flags only; no dst write */

  /* Lazy flag handling — mirrors Blink's interpreter (see spec §Q3). */
  FBX_IR_OP_SET_FLAGS_RAW = 12, /* dst = flag-shadow; carries op kind in imm */
  FBX_IR_OP_GET_FLAG = 13,      /* dst = flag value; imm encodes which flag */

  /* Address computation (LEA): dst = src1 + src2*scale + imm.  No memory
   * access.  `width` is the result width (4 or 8 bytes).
   *
   * Two encodings (the index term is OPTIONAL):
   *   - disp-only (`[base + disp]`): dst_kind=GREG dst=dest greg;
   *     src1_kind=GREG src1=base greg; src2_kind=NONE or IMM (legacy
   *     marker); scale=0; imm=signed displacement.
   *   - indexed (`[base + index*scale + disp]`, #778): dst_kind=GREG;
   *     src1_kind=GREG src1=base greg; src2_kind=GREG src2=index greg;
   *     scale ∈ {1,2,4,8} (the SIB scale, 1<<SibScale); imm=disp.
   * LEA computes an ADDRESS and never dereferences guest memory, so it
   * needs no software-MMU translation even on the wasm32 (non-linear)
   * build — it is pure register arithmetic into the Machine struct. */
  FBX_IR_OP_LEA = 14,

  /* Control flow — block terminators. */
  FBX_IR_OP_BRANCH_TAKEN = 15,  /* unconditional; imm = target PC */
  FBX_IR_OP_BRANCH_COND = 16,   /* conditional; imm = taken PC,
                                   src1 = condition flag id,
                                   src2 = fallthrough PC */
  FBX_IR_OP_CALL_DIRECT = 17,   /* direct call; imm = target PC,
                                   src2 = return PC */
  FBX_IR_OP_CALL_INDIRECT = 18, /* indirect call; src1 = target vreg,
                                   src2 = return PC */
  FBX_IR_OP_RET = 19,           /* return; pops from guest stack */

  /* Block-end bailout — exit the translated path, resume on Tier 1.  Used
   * when the lifting pass encounters an unsupported opcode, a precious op
   * (syscall/interrupt/cpuid/wrmsr/fence), or runs out of room.  `imm`
   * carries the PC to resume at. */
  FBX_IR_OP_BAILOUT = 20,

  /* Host-import call (Phase 1 thunks, syscalls if v0.2 lifts them, etc.).
   * `imm` carries the host-import index (assigned by §13.3 host bridge).
   * Always block-terminating.  NOT emitted in v0.1 (thunk blocks bail
   * out, syscalls bail out); reserved for v0.2+ extension. */
  FBX_IR_OP_CALL_HOST = 21,

  /* #602 — guest-stack push/pop of a guest register.  Mutates RSP and
   * writes/reads the 8-byte word at the new/current top-of-stack.
   *
   * PUSH semantics: RSP -= 8; mem[RSP] = greg[reg_id].
   *   dst_kind = NONE; src1_kind = GREG (the register to push);
   *   width   = operand size (1/2/4/8 — v0.1 only emits 8 for the
   *             x86-64 default operand size).
   *
   * POP semantics: greg[reg_id] = mem[RSP]; RSP += 8.
   *   dst_kind = GREG (the register being loaded);
   *   src1_kind = NONE; width = operand size (8 in v0.1).
   *
   * Both opcodes mutate the guest stack at the always-known address
   * RSP; the v0.1 emitter (§13.5c) lowers them via i64.load/i64.store
   * against the imported wasm linear memory + i32.wrap_i64(RSP).
   * CALL/RET reuse the same memory machinery. */
  FBX_IR_OP_PUSH = 22,
  FBX_IR_OP_POP  = 23,

  /* #794 (T3/T4 emit-coverage floor) — truncating-form integer multiply.
   * Lifts x86 IMUL r,r/m,imm (0x69/0x6B) and IMUL r,r/m (0x0FAF) to a
   * vreg*vreg product, the SAME REG_GET→OP→REG_SET shape as the ALU ops:
   *   dst(vreg) = src1(vreg) * src2(vreg)  (low operand-width bits; i64.mul
   *   + the existing dest-width REG_SET store).
   * Carries NO flags: imul's CF/OF (full-product overflow) + x86-undefined
   * SF/ZF/AF/PF are not synthesized at this increment.  The emit coverage
   * gate refuses any block where imul's flags are LIVE at a reader, so
   * Tier 1 computes them (correct-or-refuse — the #677/#735 pattern). */
  FBX_IR_OP_IMUL = 24,

  /* Sentinel — count of defined opcodes.  Used by validation; not emitted. */
  FBX_IR_OP_LAST_,
};

/* ────────────────────────────────────────────────────────────────────────── */
/* Operand-kind encoding for FbxIrInst.{dst,src1,src2}_kind.                  */
/*                                                                            */
/* Pinned values — see Versioning above.  Same cross-build determinism        */
/* concern.                                                                   */
/* ────────────────────────────────────────────────────────────────────────── */

#define FBX_IR_KIND_NONE 0   /* operand not used */
#define FBX_IR_KIND_VREG 1   /* virtual register; index in .src/.dst */
#define FBX_IR_KIND_GREG 2   /* guest register id; encoding in .src/.dst */
#define FBX_IR_KIND_IMM  3   /* immediate value in .src/.dst (or .imm) */

/* ────────────────────────────────────────────────────────────────────────── */
/* FbxIrInst — a single IR instruction.                                       */
/*                                                                            */
/* Three-address with explicit operand kinds, immediate slot, and per-inst    */
/* width.  Total size 40 bytes (pinned by the static_assert at the bottom of  */
/* the file).  Layout is FLAT (no pointers) so:                               */
/*   - serialisation = memcpy(buf, &inst, sizeof inst)                        */
/*   - deserialisation = memcpy(&inst, buf, sizeof inst)                      */
/*   - sha256 over a block = sha256 over its packed FbxIrInst array bytes     */
/*                                                                            */
/* Field semantics:                                                           */
/*   opcode    : one of FBX_IR_OP_*; the discriminator                        */
/*   width     : operand width in bytes (1/2/4/8); 0 if N/A                   */
/*   *_kind    : FBX_IR_KIND_* tag for each operand                          */
/*   dst       : destination operand (vreg index OR guest-reg id)             */
/*   src1, src2: source operands (vreg, greg, or imm-narrow)                  */
/*   imm       : opcode-specific wide immediate (branch target PC, etc.)      */
/*                                                                            */
/* The `_padN` slots make the layout aligned + reproducible across compilers; */
/* they are NOT used by the lifting pass.  Initialise via the helper          */
/* `fbx_ir_inst_zero()` to ensure the padding bytes are zero so SHA-256 over  */
/* the serialised form is deterministic.                                      */
/* ────────────────────────────────────────────────────────────────────────── */

struct FbxIrInst {
  u8 opcode;     /* FBX_IR_OP_* */
  u8 width;      /* 1/2/4/8 or 0 */
  u8 dst_kind;   /* FBX_IR_KIND_* */
  u8 src1_kind;  /* FBX_IR_KIND_* */
  u8 src2_kind;  /* FBX_IR_KIND_* */
  u8 _pad0;
  u16 _pad1;
  u32 dst;       /* vreg index or greg id */
  u32 src1;      /* vreg index or greg id or narrow imm */
  u32 src2;      /* vreg index or greg id or narrow imm */
  u32 scale;     /* LEA SIB scale ∈ {1,2,4,8}; 0 = no index term (#778).
                  * Occupies the former `_pad2` slot (same offset 20, same
                  * 4-byte size) so the layout — and the 40-byte wire format —
                  * is unchanged; it just gives the alignment pad a name and a
                  * meaning for the indexed LEA.  Zero for every other opcode
                  * (fbx_ir_inst_zero memsets it), so SHA determinism holds. */
  u64 imm;       /* wide immediate (PC, full-width literal, etc.) */
  u64 _pad3;     /* reserved for v0.2 extension without breaking layout */
};

/* ────────────────────────────────────────────────────────────────────────── */
/* FbxIrBlock — a lifted basic block.                                         */
/*                                                                            */
/* `insts` is a contiguous array; `ninsts` is its length.  `flags` carries    */
/* block-level properties (see FBX_IR_BLOCK_FLAG_*).  `cache_key_sha256` is   */
/* the 32-byte digest computed by fbx_ir_block_sha256(); §13.2 / §13.3 / the  */
/* Phase 4 sidecar key the host-side module cache off this value.            */
/* ────────────────────────────────────────────────────────────────────────── */

#define FBX_IR_BLOCK_FLAG_HAS_CALL        0x01
#define FBX_IR_BLOCK_FLAG_HAS_INDIRECT_BR 0x02
#define FBX_IR_BLOCK_FLAG_HAS_MEM_BARRIER 0x04
#define FBX_IR_BLOCK_FLAG_SOURCED_OFFLINE 0x08 /* Phase 4 sidecar emitted it */
#define FBX_IR_BLOCK_FLAG_NON_TRANSLATABLE 0x10 /* lifter bailed; do not escalate */
#define FBX_IR_BLOCK_FLAG_HAS_BAILOUT     0x20 /* terminates with FBX_IR_OP_BAILOUT */

struct FbxIrBlock {
  u64 start_pc;
  u64 end_pc;
  u32 ninsts;
  u16 nvregs;
  u16 flags;
  struct FbxIrInst *insts;
  u8 cache_key_sha256[32];
};

/* ────────────────────────────────────────────────────────────────────────── */
/* Bailout reason — surfaced via flags + via the diagnostic API.  Pinned      */
/* values; cross-build deterministic.                                         */
/* ────────────────────────────────────────────────────────────────────────── */

enum FbxIrBailoutReason {
  FBX_IR_BAILOUT_NONE = 0,
  FBX_IR_BAILOUT_THUNK = 1,           /* Phase 1 thunk; keep on Tier 1 */
  FBX_IR_BAILOUT_SYSCALL = 2,         /* precious op */
  FBX_IR_BAILOUT_SERIALIZING = 3,     /* mfence/cpuid/wrmsr */
  FBX_IR_BAILOUT_UNSUPPORTED = 4,     /* opcode outside top-N coverage */
  FBX_IR_BAILOUT_DECODE_ERROR = 5,    /* unrecognisable rde */
  FBX_IR_BAILOUT_BUDGET_EXHAUSTED = 6 /* lifting-pass inst budget hit */
};

/* ────────────────────────────────────────────────────────────────────────── */
/* Forward declaration — defined in blink/threadedcode.h.  Avoiding a hard    */
/* include here keeps the IR header standalone for Phase 4's offline walker.  */
/* ────────────────────────────────────────────────────────────────────────── */

struct FbxTcBlock;

/* ────────────────────────────────────────────────────────────────────────── */
/* API surface.  Implementation lives in fbx_ir_lift.c.                       */
/* ────────────────────────────────────────────────────────────────────────── */

/* Zero an FbxIrInst including padding — required for SHA determinism. */
void fbx_ir_inst_zero(struct FbxIrInst *inst);

/* Lift a Tier 1 block into an IR block.
 *
 * Returns:
 *   - non-NULL FbxIrBlock pointer on success (full or partial-with-BAILOUT)
 *   - NULL if:
 *       - input is NULL or empty
 *       - the block starts on a thunk entry (caller must NOT escalate;
 *         thunks stay on Tier 1)
 *       - allocation failed
 *   - In the partial-success case, block->flags has
 *     FBX_IR_BLOCK_FLAG_HAS_BAILOUT set and the final instruction is
 *     FBX_IR_OP_BAILOUT.  The caller can still emit wasm for the
 *     translatable prefix; runtime bails out at the BAILOUT PC.
 *
 * The function is pure over its inputs: no global state, no clocks, no
 * thread-local data.  Two calls with the same FbxTcBlock contents produce
 * byte-identical output (verified by fbx_ir_lift_test.c).
 */
struct FbxIrBlock *fbx_ir_lift(const struct FbxTcBlock *tc);

/* Free a block returned from fbx_ir_lift(). */
void fbx_ir_free(struct FbxIrBlock *ir);

/* Compute the cache-key SHA-256 over an IR block.
 *
 * Hash composition:
 *     sha256(FBX_IR_VERSION (u32 LE) || blink_git_sha (32 bytes) ||
 *            start_pc (u64 LE) || end_pc (u64 LE) ||
 *            ninsts (u32 LE) || nvregs (u16 LE) || flags (u16 LE) ||
 *            insts[0..ninsts-1])
 *
 * The blink_git_sha is a build-time constant captured at compile time
 * (FBX_IR_BLINK_GIT_SHA macro; defaults to all-zero if unset).  Per spec
 * §Q5 axis 2 (cross-build determinism) this MUST be deterministic across
 * rebuilds of the same source: it is sourced from blink/buildinfo.h
 * when available, else from the FBX_IR_BLINK_GIT_SHA env var at compile
 * time, else zero.
 *
 * Writes 32 bytes to out_sha256.  Idempotent + side-effect-free.
 */
void fbx_ir_block_sha256(const struct FbxIrBlock *ir, u8 out_sha256[32]);

/* Compute a SHA-256 over an arbitrary byte range.  Exposed so callers
 * (e.g. §13.3 host bridge) can SHA the synthesised wasm output keyed
 * identically.  Public-domain implementation; matches FIPS 180-4. */
void fbx_sha256(const void *data, size_t len, u8 out[32]);

/* Diagnostic: human-readable opcode name (e.g. "ADD", "BRANCH_COND").
 * Returns a static string; never NULL.  Used by tests + future trace. */
const char *fbx_ir_opcode_name(u8 opcode);

/* Diagnostic: human-readable bailout reason name. */
const char *fbx_ir_bailout_reason_name(int reason);

/* ────────────────────────────────────────────────────────────────────────── */
/* Wire-format lock.                                                          */
/*                                                                            */
/* Any compiler that disagrees with this size has different padding rules     */
/* than we expect; bumping FBX_IR_VERSION won't be enough — the on-disk       */
/* shape would be incompatible with Phase 4 sidecars built by the prevailing  */
/* layout.  Failing here is the right behaviour: it forces the layout to be   */
/* re-pinned with explicit pragma pack or rearrangement.                      */
/* ────────────────────────────────────────────────────────────────────────── */

#define FBX_IR_INST_SERIALISED_BYTES 40

#if !defined(__cplusplus)
_Static_assert(sizeof(struct FbxIrInst) == FBX_IR_INST_SERIALISED_BYTES,
               "FbxIrInst on-disk layout drift — Phase 4 sidecars depend on "
               "this exact size.  Fix the struct, do NOT bump the static "
               "assert.");
#endif

#ifdef __cplusplus
}
#endif

#endif /* BLINK_FBX_IR_H_ */
