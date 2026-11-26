/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2026 Justin Firth (Firebox)                                        │
│                                                                              │
│ Phase 2 — Tier 2 §13.1: IR data structures + lifting pass for top-10        │
│ opcodes.  See blink/fbx_ir.h for the design header + spec pointers.          │
│                                                                              │
│ This file is the SUBSTRATE for §13.2 (wasm synthesis), §13.3 (firebox-wasix  │
│ bridge), §13.4 (Blink escalation glue), and a future Phase 4 offline ELF    │
│ walker.  All four read the FbxIrBlock shape defined here.                    │
│                                                                              │
│ Determinism contract (LOAD-BEARING — see spec §Q5):                          │
│   - No global state read or written                                          │
│   - No clocks, no random sources, no thread-local data                       │
│   - vreg counter is per-block + monotonic from zero                          │
│   - No hashtable / hashset / set with unspecified iteration order            │
│   - Padding bytes in FbxIrInst are zeroed via fbx_ir_inst_zero()             │
│                                                                              │
│ Two calls to fbx_ir_lift() over the same FbxTcBlock contents MUST produce    │
│ a byte-identical FbxIrBlock.  fbx_ir_lift_test.c's TestDeterminism gate     │
│ validates this; do not weaken it.                                            │
╚─────────────────────────────────────────────────────────────────────────────*/

#include "blink/fbx_ir.h"

#include <stdlib.h>
#include <string.h>

#include "blink/builtin.h"
#include "blink/rde.h"
#include "blink/threadedcode.h"
#include "blink/types.h"
#include "blink/modrm.h"
#include "blink/x86.h"

/* The blink git SHA the build was produced from.  Optional — if the build
 * system does not provide it, the cache key falls back to all-zero.  Phase
 * 4 sidecars produced by a build with FBX_IR_BLINK_GIT_SHA unset will not
 * round-trip against a build with it set, by design (the spec §Q5 axis 2
 * test catches the mismatch). */
#ifndef FBX_IR_BLINK_GIT_SHA
#define FBX_IR_BLINK_GIT_SHA \
  "0000000000000000000000000000000000000000000000000000000000000000"
#endif

/* Defensive cap on emitted IR insts per block.  FBX_TC_MAX_BLOCK_ENTRIES
 * in threadedcode.c is 128; a single x86 op lifts to at most ~6 IR insts
 * in the v0.1 coverage (PC_MARK + REG_GET + REG_GET + ALU + SET_FLAGS_RAW
 * + REG_SET).  6 × 128 = 768.  We round up + leave headroom. */
#define FBX_IR_LIFT_MAX_INSTS 1024

/* ────────────────────────────────────────────────────────────────────────── */
/* SHA-256.                                                                   */
/*                                                                            */
/* Reference implementation from FIPS 180-4.  Public-domain.  Side-effect-    */
/* free, deterministic, no clock + no global state — required by the spec    */
/* §Q5 cache-key-stability contract.                                          */
/*                                                                            */
/* Tested by fbx_ir_lift_test.c::TestSha256KnownVectors against the           */
/* canonical "abc" + empty-string vectors from FIPS 180-4 §B.                 */
/* ────────────────────────────────────────────────────────────────────────── */

struct Sha256Ctx {
  u32 state[8];
  u64 nbytes;
  u8 buf[64];
  u32 buf_len;
};

static const u32 kSha256K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

static inline u32 RotR32(u32 x, unsigned n) {
  return (x >> n) | (x << (32u - n));
}

static void Sha256Init(struct Sha256Ctx *c) {
  c->state[0] = 0x6a09e667u;
  c->state[1] = 0xbb67ae85u;
  c->state[2] = 0x3c6ef372u;
  c->state[3] = 0xa54ff53au;
  c->state[4] = 0x510e527fu;
  c->state[5] = 0x9b05688cu;
  c->state[6] = 0x1f83d9abu;
  c->state[7] = 0x5be0cd19u;
  c->nbytes = 0;
  c->buf_len = 0;
}

static void Sha256Compress(struct Sha256Ctx *c, const u8 *block) {
  u32 w[64];
  u32 a, b, d, e, f, g, h, t1, t2, ch, maj, s0, s1, ep0, ep1;
  u32 cc;
  unsigned i;
  for (i = 0; i < 16; ++i) {
    w[i] = ((u32)block[i * 4 + 0] << 24) | ((u32)block[i * 4 + 1] << 16) |
           ((u32)block[i * 4 + 2] << 8) | ((u32)block[i * 4 + 3] << 0);
  }
  for (i = 16; i < 64; ++i) {
    s0 = RotR32(w[i - 15], 7) ^ RotR32(w[i - 15], 18) ^ (w[i - 15] >> 3);
    s1 = RotR32(w[i - 2], 17) ^ RotR32(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  a = c->state[0];
  b = c->state[1];
  cc = c->state[2];
  d = c->state[3];
  e = c->state[4];
  f = c->state[5];
  g = c->state[6];
  h = c->state[7];
  for (i = 0; i < 64; ++i) {
    ep1 = RotR32(e, 6) ^ RotR32(e, 11) ^ RotR32(e, 25);
    ch = (e & f) ^ ((~e) & g);
    t1 = h + ep1 + ch + kSha256K[i] + w[i];
    ep0 = RotR32(a, 2) ^ RotR32(a, 13) ^ RotR32(a, 22);
    maj = (a & b) ^ (a & cc) ^ (b & cc);
    t2 = ep0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = cc;
    cc = b;
    b = a;
    a = t1 + t2;
  }
  c->state[0] += a;
  c->state[1] += b;
  c->state[2] += cc;
  c->state[3] += d;
  c->state[4] += e;
  c->state[5] += f;
  c->state[6] += g;
  c->state[7] += h;
}

static void Sha256Update(struct Sha256Ctx *c, const void *data, size_t len) {
  const u8 *p = (const u8 *)data;
  c->nbytes += (u64)len;
  while (len > 0) {
    size_t take = 64 - c->buf_len;
    if (take > len) take = len;
    memcpy(c->buf + c->buf_len, p, take);
    c->buf_len += (u32)take;
    p += take;
    len -= take;
    if (c->buf_len == 64) {
      Sha256Compress(c, c->buf);
      c->buf_len = 0;
    }
  }
}

static void Sha256Final(struct Sha256Ctx *c, u8 out[32]) {
  u64 bit_len = c->nbytes * 8u;
  u8 lenbuf[8];
  unsigned i;
  /* Append the 1-bit + padding. */
  c->buf[c->buf_len++] = 0x80u;
  if (c->buf_len > 56) {
    while (c->buf_len < 64) c->buf[c->buf_len++] = 0;
    Sha256Compress(c, c->buf);
    c->buf_len = 0;
  }
  while (c->buf_len < 56) c->buf[c->buf_len++] = 0;
  /* 64-bit big-endian length in bits. */
  for (i = 0; i < 8; ++i) {
    lenbuf[i] = (u8)(bit_len >> (56 - i * 8));
  }
  memcpy(c->buf + 56, lenbuf, 8);
  Sha256Compress(c, c->buf);
  for (i = 0; i < 8; ++i) {
    out[i * 4 + 0] = (u8)(c->state[i] >> 24);
    out[i * 4 + 1] = (u8)(c->state[i] >> 16);
    out[i * 4 + 2] = (u8)(c->state[i] >> 8);
    out[i * 4 + 3] = (u8)(c->state[i] >> 0);
  }
}

void fbx_sha256(const void *data, size_t len, u8 out[32]) {
  struct Sha256Ctx c;
  Sha256Init(&c);
  Sha256Update(&c, data, len);
  Sha256Final(&c, out);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* FbxIrInst zeroing.                                                         */
/* Padding zeroing is LOAD-BEARING: the SHA-256 hashes the raw bytes of the   */
/* inst array, so unset padding bytes would make two semantically-identical   */
/* lifts produce different SHAs.                                              */
/* ────────────────────────────────────────────────────────────────────────── */

void fbx_ir_inst_zero(struct FbxIrInst *inst) {
  memset(inst, 0, sizeof(*inst));
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Diagnostic name tables.  Static + pinned; no dynamic allocation, no env-   */
/* var reads.                                                                 */
/* ────────────────────────────────────────────────────────────────────────── */

static const char *const kOpcodeNames[FBX_IR_OP_LAST_] = {
    [FBX_IR_OP_PC_MARK] = "PC_MARK",
    [FBX_IR_OP_LOAD] = "LOAD",
    [FBX_IR_OP_STORE] = "STORE",
    [FBX_IR_OP_REG_GET] = "REG_GET",
    [FBX_IR_OP_REG_SET] = "REG_SET",
    [FBX_IR_OP_ADD] = "ADD",
    [FBX_IR_OP_SUB] = "SUB",
    [FBX_IR_OP_AND] = "AND",
    [FBX_IR_OP_OR] = "OR",
    [FBX_IR_OP_XOR] = "XOR",
    [FBX_IR_OP_IMUL] = "IMUL",
    [FBX_IR_OP_CMP] = "CMP",
    [FBX_IR_OP_TEST] = "TEST",
    [FBX_IR_OP_SET_FLAGS_RAW] = "SET_FLAGS_RAW",
    [FBX_IR_OP_GET_FLAG] = "GET_FLAG",
    [FBX_IR_OP_LEA] = "LEA",
    [FBX_IR_OP_BRANCH_TAKEN] = "BRANCH_TAKEN",
    [FBX_IR_OP_BRANCH_COND] = "BRANCH_COND",
    [FBX_IR_OP_CALL_DIRECT] = "CALL_DIRECT",
    [FBX_IR_OP_CALL_INDIRECT] = "CALL_INDIRECT",
    [FBX_IR_OP_RET] = "RET",
    [FBX_IR_OP_BAILOUT] = "BAILOUT",
    [FBX_IR_OP_CALL_HOST] = "CALL_HOST",
    [FBX_IR_OP_PUSH] = "PUSH",
    [FBX_IR_OP_POP] = "POP",
    [FBX_IR_OP_JMP_INDIRECT] = "JMP_INDIRECT",
    [FBX_IR_OP_SHL] = "SHL",
    [FBX_IR_OP_SHR] = "SHR",
    [FBX_IR_OP_SAR] = "SAR",
    [FBX_IR_OP_SEXT] = "SEXT",
    [FBX_IR_OP_CMOV] = "CMOV",
};

const char *fbx_ir_opcode_name(u8 opcode) {
  if (opcode >= FBX_IR_OP_LAST_ || !kOpcodeNames[opcode]) return "UNKNOWN";
  return kOpcodeNames[opcode];
}

const char *fbx_ir_bailout_reason_name(int reason) {
  switch (reason) {
    case FBX_IR_BAILOUT_NONE: return "NONE";
    case FBX_IR_BAILOUT_THUNK: return "THUNK";
    case FBX_IR_BAILOUT_SYSCALL: return "SYSCALL";
    case FBX_IR_BAILOUT_SERIALIZING: return "SERIALIZING";
    case FBX_IR_BAILOUT_UNSUPPORTED: return "UNSUPPORTED";
    case FBX_IR_BAILOUT_DECODE_ERROR: return "DECODE_ERROR";
    case FBX_IR_BAILOUT_BUDGET_EXHAUSTED: return "BUDGET_EXHAUSTED";
    default: return "UNKNOWN";
  }
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Lifting context.                                                           */
/*                                                                            */
/* All state is per-call (heap-allocated by fbx_ir_lift()).  No globals, no   */
/* TLS, no statics-with-mutable-state — see the determinism contract.        */
/* ────────────────────────────────────────────────────────────────────────── */

struct LiftCtx {
  struct FbxIrInst *insts;
  u32 ninsts;
  u32 cap;
  u32 next_vreg;
  u16 flags;
  /* firebox#T3R2: 1 when lifting for a Tier-2 region (fbx_ir_lift_region).
   * The forms only regions lower are lifted in this mode alone; the per-block
   * lift BAILOUTs them exactly as before. */
  int region;
  u64 next_pc; /* PC after the instruction being lifted (RIP-relative) */
  /* firebox#T3R3: the instruction's segment override (Sego), when a region
   * lift may apply it to the memory operand, and whether a lifter did
   * (LiftClassifyAddr).  An override no lifter consumed refuses the
   * instruction (LiftOne). */
  u8 seg;
  int seg_used;
};

static int CtxReserve(struct LiftCtx *ctx, u32 want) {
  u32 need;
  if (want < 8) want = 8;
  need = ctx->ninsts + want;
  if (need <= ctx->cap) return 1;
  if (need > FBX_IR_LIFT_MAX_INSTS) return 0;
  {
    u32 new_cap = ctx->cap ? ctx->cap * 2 : 64;
    void *p;
    while (new_cap < need) new_cap *= 2;
    if (new_cap > FBX_IR_LIFT_MAX_INSTS) new_cap = FBX_IR_LIFT_MAX_INSTS;
    p = realloc(ctx->insts, (size_t)new_cap * sizeof(struct FbxIrInst));
    if (!p) return 0;
    ctx->insts = (struct FbxIrInst *)p;
    ctx->cap = new_cap;
  }
  return 1;
}

static struct FbxIrInst *CtxEmit(struct LiftCtx *ctx) {
  struct FbxIrInst *out;
  if (!CtxReserve(ctx, 1)) return NULL;
  out = &ctx->insts[ctx->ninsts++];
  fbx_ir_inst_zero(out);
  return out;
}

static u32 CtxAllocVreg(struct LiftCtx *ctx) {
  /* Deterministic monotonic counter — required by spec §Q5 determinism. */
  return ctx->next_vreg++;
}

/* PC_MARK emission helper — every lifted x86 op starts with one. */
static int EmitPcMark(struct LiftCtx *ctx, u64 pc) {
  struct FbxIrInst *i = CtxEmit(ctx);
  if (!i) return 0;
  i->opcode = FBX_IR_OP_PC_MARK;
  i->imm = pc;
  return 1;
}

/* BAILOUT emission helper — block-terminating. */
static int EmitBailout(struct LiftCtx *ctx, u64 pc, int reason) {
  struct FbxIrInst *i = CtxEmit(ctx);
  if (!i) return 0;
  i->opcode = FBX_IR_OP_BAILOUT;
  i->imm = pc;
  i->src1_kind = FBX_IR_KIND_IMM;
  i->src1 = (u32)reason;
  ctx->flags |= FBX_IR_BLOCK_FLAG_HAS_BAILOUT;
  return 1;
}

/* firebox#CM4 — unconditional fall-through terminator.  Emitted when a Tier-1
 * block ends WITHOUT a control-flow op (blink split the block mid-stream; it
 * falls through to the next block at `target_pc`).  Lowers exactly like an
 * `jmp target_pc`: the T2 block stores `m->ip = target_pc` and returns exit=0,
 * so the dispatcher continues to the next block — identical to how Tier-1
 * advances past a fall-through boundary. */
static int EmitFallthroughBranch(struct LiftCtx *ctx, u64 target_pc) {
  struct FbxIrInst *i = CtxEmit(ctx);
  if (!i) return 0;
  i->opcode = FBX_IR_OP_BRANCH_TAKEN;
  i->imm = target_pc;
  return 1;
}

/* True iff `op` is a block terminator that advances (or hands off) m->ip.  Used
 * to detect a lifted block that fell off the end of its Tier-1 entries WITHOUT
 * a terminator — the fbx_wasm_emit.h / EmitFunctionBody contract ("every lifted
 * block ends with a terminator") must hold, else the emit's un-terminated
 * fallback returns exit=0 without advancing m->ip and the guest re-dispatches
 * the SAME block forever (firebox#CM4 wedge). */
static int LiftOpIsTerminator(u8 op) {
  switch (op) {
    case FBX_IR_OP_BRANCH_TAKEN:
    case FBX_IR_OP_BRANCH_COND:
    case FBX_IR_OP_CALL_DIRECT:
    case FBX_IR_OP_CALL_INDIRECT:
    case FBX_IR_OP_RET:
    case FBX_IR_OP_BAILOUT:
    case FBX_IR_OP_CALL_HOST:
    case FBX_IR_OP_JMP_INDIRECT:
      return 1;
    default:
      return 0;
  }
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Per-opcode lifters for the top-10.                                         */
/*                                                                            */
/* Each emits 1..N FbxIrInst that describe the x86 op's effect at a level     */
/* the §13.2 synthesis pass can lower to wasm.  At v0.1 the emitted IR is     */
/* approximate (not a full Blink-equivalent); the §13.5 work extends it to    */
/* top-30 with finer modelling.  What matters for §13.1:                      */
/*                                                                            */
/*   - Each top-10 opcode has a deterministic IR shape (same input → same    */
/*     output bytes).                                                         */
/*   - The IR records ENOUGH that the §13.2 lowering can route the op (e.g., */
/*     "this is an ADD between two greg-encoded operands" is sufficient for   */
/*     §13.2 to emit the wasm load + i64.add + store sequence).               */
/*   - Bailouts emit FBX_IR_OP_BAILOUT and terminate the block.               */
/* ────────────────────────────────────────────────────────────────────────── */

/* Operand-width helper.  Mirrors Blink's interpreter:
 *   - byte ops (low bit 0 of opcode) → 1
 *   - Rex.W → 8
 *   - operand-size prefix → 2
 *   - else → 4
 */
static u8 WidthFromRde(u64 rde, int byte_op) {
  if (byte_op) return 1;
  if (Rexw(rde)) return 8;
  if (Osz(rde)) return 2;
  return 4;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Memory-form addressing classification (#677).                              */
/*                                                                            */
/* The v0.1 memory-form MOV emit (§13.5d) lowers exactly ONE addressing      */
/* shape: `[base + disp]` — a single guest base register plus a signed       */
/* displacement, with NO SIB index/scale term and NOT RIP-relative.          */
/* Empirically (work/tasks/677-* EA-form probe) this covers ~85% of the      */
/* MOV r/m memory-form bailouts across the 6-fixture corpus; the index /     */
/* RIP-relative tail is deferred to a later increment via correct-or-refuse  */
/* (the lifter BAILOUTs, exactly as LEA refuses its SIB-index form).         */
/*                                                                            */
/* Returns 1 and writes *out_base_greg if the form is the supported          */
/* `[base + disp]`; returns 0 otherwise (caller must BAILOUT).               */
/*                                                                            */
/* Base-register sourcing:                                                    */
/*   - No SIB byte (ModrmRm != 4): base greg = RexbRm(rde).  Excludes the    */
/*     RIP-relative encoding (mod==0 && ModrmRm==5), which has no base reg.  */
/*   - SIB byte present (ModrmRm == 4): require a base and no index          */
/*     (SibHasBase && !SibHasIndex); base greg = RexbBase(rde).  The         */
/*     no-base SIB form (SibIsAbsolute / disp32-only) and any indexed form   */
/*     are refused.                                                          */
static int ClassifyMemBaseDisp(u64 rde, u32 *out_base_greg) {
  if (IsModrmRegister(rde)) return 0; /* reg-form; caller handles separately */
  if (!SibExists(rde)) {
    /* No SIB.  RIP-relative (mod==0, ModrmRm==5) has no base register. */
    if (ModrmMod(rde) == 0 && ModrmRm(rde) == 5) return 0;
    *out_base_greg = (u32)RexbRm(rde);
    return 1;
  }
  /* SIB byte present.  Support base-only (no index). */
  if (SibHasBase(rde) && !SibHasIndex(rde)) {
    *out_base_greg = (u32)RexbBase(rde);
    return 1;
  }
  return 0; /* indexed or no-base SIB — refuse */
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Operand-shape guard (firebox#RAT).                                         */
/*                                                                            */
/* Every lifter below names a register operand by its 4-bit NUMBER            */
/* (`RexrReg` / `RexbRm`) and a memory operand by `base [+ index*scale] +     */
/* disp`, and the emitter turns a register number into `weg[n]`.  Three x86   */
/* operand shapes mean something those two descriptions cannot express, and   */
/* until this guard the lifter translated all three as if they did not exist: */
/*                                                                            */
/*   1. HIGH-BYTE REGISTERS.  In a byte-sized operand WITHOUT a REX prefix,   */
/*      register numbers 4..7 are AH/CH/DH/BH — byte 1 of rAX/rCX/rDX/rBX —   */
/*      not SPL/BPL/SIL/DIL (blink/breg.c kByteReg, which the interpreter     */
/*      indexes with `Rex << 4 | number`).  Lifted by number, `xor $0x40,%ah` */
/*      became `xor $0x40,%spl`: busybox awk's RSP moved by 0x40 and its next */
/*      `ret` popped a heap pointer into RIP.  `movzbl %dh,%ecx` read SIL and */
/*      simply computed a wrong value.                                        */
/*   2. SEGMENT OVERRIDES.  `mov %gs:(%rsi),%rax` addresses GS.base + rsi     */
/*      (blink/address.c AddSegment); lifted without the base it read — and   */
/*      the store form wrote — a different object.  LEA is exempt: it         */
/*      computes an offset and never applies a segment base.                  */
/*   3. ADDRESS-SIZE OVERRIDE (0x67).  The effective address is truncated to  */
/*      32 bits (blink/modrm.c LoadEffectiveAddress, XED_MODE_LEGACY), for    */
/*      LEA as well as for a real access.                                     */
/*                                                                            */
/* Correct-or-refuse, the #677/#738 pattern: an instruction with any of these */
/* shapes is not lifted, the block gets a BAILOUT at its PC, and Tier 1 —     */
/* which implements all three — executes it.  Modelling them (a byte-offset   */
/* register operand for 1, a segment-base term for 2, an i32 wrap for 3) is   */
/* a coverage increment on top of this, not a precondition for correctness.   */
/*                                                                            */
/* The guard is keyed on the opcode's operand ROLES rather than applied per   */
/* lifter, and an opcode with no entry here is refused outright, so a new     */
/* opcode has to state which of its fields are bytes before LiftOne's switch  */
/* can reach its lifter at all.                                               */
/* ────────────────────────────────────────────────────────────────────────── */

#define FBX_LIFT_UNCLASSIFIED (-1)
#define FBX_LIFT_NO_MODRM  0 /* no ModRM operand                         */
#define FBX_LIFT_HAS_MODRM 1 /* opcode carries a ModRM byte              */
#define FBX_LIFT_REG_BYTE  2 /* ModRM.reg names a byte-sized register    */
#define FBX_LIFT_RM_BYTE   4 /* ModRM.rm, when a register, is byte-sized */
#define FBX_LIFT_NO_SEG    8 /* memory operand ignores segment overrides */

static int LiftOperandRoles(u64 mop) {
  switch (mop) {
    /* Byte ops with two byte registers: MOV, ALU r/m8,r8 and r8,r/m8, TEST. */
    case 0x088: case 0x08A:
    case 0x000: case 0x008: case 0x020: case 0x028: case 0x030: case 0x038:
    case 0x002: case 0x00A: case 0x022: case 0x02A: case 0x032: case 0x03A:
    case 0x084:
      return FBX_LIFT_HAS_MODRM | FBX_LIFT_REG_BYTE | FBX_LIFT_RM_BYTE;
    /* ModRM.reg is an opcode extension (0x80, 0xC6) or a wide register
     * (MOVZX r, r/m8); only r/m is a byte. */
    case 0x080: case 0x0C6: case 0x1B6:
    case 0x0F6: case 0x0FE: /* firebox#T3R2: TEST/NOT/NEG, INC/DEC r/m8 */
    case 0x1BE: /* firebox#T3R3: MOVSX r, r/m8 */
      return FBX_LIFT_HAS_MODRM | FBX_LIFT_RM_BYTE;
    /* Word/dword/qword ModRM ops — no byte register in either field. */
    case 0x089: case 0x08B: case 0x0C7:
    case 0x001: case 0x009: case 0x021: case 0x029: case 0x031: case 0x039:
    case 0x003: case 0x00B: case 0x023: case 0x02B: case 0x033: case 0x03B:
    case 0x085: case 0x1B7: case 0x081: case 0x083:
    case 0x069: case 0x06B: case 0x1AF:
    case 0x0F7: case 0x0FF: /* firebox#T3R2: groups 3 and 5 */
    case 0x0C1: case 0x0D1: case 0x0D3: /* firebox#T3R3: group 2 */
    case 0x063: case 0x1BF: /* firebox#T3R3: MOVSXD, MOVSX r, r/m16 */
    case 0x140: case 0x141: case 0x142: case 0x143: /* firebox#T3R3: CMOVcc */
    case 0x144: case 0x145: case 0x146: case 0x147:
    case 0x148: case 0x149: case 0x14A: case 0x14B:
    case 0x14C: case 0x14D: case 0x14E: case 0x14F:
      return FBX_LIFT_HAS_MODRM;
    case 0x08D:
      return FBX_LIFT_HAS_MODRM | FBX_LIFT_NO_SEG;
    /* No ModRM: MOV reg,imm; accumulator-immediate ALU (AL is number 0, never
     * a high byte); Jcc; CALL/JMP rel; PUSH/POP reg; RET. */
    case 0x0B8: case 0x0B9: case 0x0BA: case 0x0BB:
    case 0x0BC: case 0x0BD: case 0x0BE: case 0x0BF:
    case 0x004: case 0x005: case 0x00C: case 0x00D:
    case 0x024: case 0x025: case 0x02C: case 0x02D:
    case 0x034: case 0x035: case 0x03C: case 0x03D:
    case 0x070: case 0x071: case 0x072: case 0x073:
    case 0x074: case 0x075: case 0x076: case 0x077:
    case 0x078: case 0x079: case 0x07A: case 0x07B:
    case 0x07C: case 0x07D: case 0x07E: case 0x07F:
    case 0x180: case 0x181: case 0x182: case 0x183:
    case 0x184: case 0x185: case 0x186: case 0x187:
    case 0x188: case 0x189: case 0x18A: case 0x18B:
    case 0x18C: case 0x18D: case 0x18E: case 0x18F:
    case 0x0E8: case 0x0E9: case 0x0EB: case 0x0C3:
    case 0x0A8: case 0x0A9: /* firebox#T3R2: TEST AL/rAX, imm */
    case 0x098: /* firebox#T3R3: CWDE/CDQE */
    case 0x050: case 0x051: case 0x052: case 0x053:
    case 0x054: case 0x055: case 0x056: case 0x057:
    case 0x058: case 0x059: case 0x05A: case 0x05B:
    case 0x05C: case 0x05D: case 0x05E: case 0x05F:
      return FBX_LIFT_NO_MODRM;
    default:
      return FBX_LIFT_UNCLASSIFIED;
  }
}

/* True iff byte-register number `reg3` (a raw 3-bit ModRM field) names
 * AH/CH/DH/BH under `rde`: no REX prefix, and bit 2 set. */
static int LiftIsHighByteReg(u64 rde, u32 reg3) {
  return !Rex(rde) && (reg3 & 4u);
}

/* firebox#RAT (second-lineage review) — the LOCK bit of `rde`, read RAW.
 * blink/modrm.h's Lock() is the constant 0 in a build without HAVE_THREADS,
 * but the decoder (blink/x86.c, prefix 0xF0) sets the bit in every build. */
#define FBX_LIFT_RDE_LOCK ((u64)020000000000)

/* True iff the instruction was VEX- or EVEX-encoded.  `rde` does not say so:
 * the decoder folds a VEX prefix into the map/rep/osz/rex fields, so
 * `c5 f8 b6 c0` arrives here with Mopcode 0x1B6 and looks like MOVZX.  The
 * raw bytes do say so: in long mode, C4/C5 (LES/LDS) and 62 (BOUND) are not
 * instructions, so one of them as the first byte after the legacy and REX
 * prefixes can only be the escape.  XOP (8F with map >= 8) needs no arm: its
 * opcode byte is 0x8F, which has no role entry and is refused as unclassified.
 * An entry carrying no bytes (a unit-test fixture built from `rde` alone) is
 * not VEX. */
static int LiftIsVexEncoded(const struct FbxTcEntry *e) {
  unsigned i;
  unsigned n = e->xedd.length;
  if (n > sizeof e->xedd.bytes) n = sizeof e->xedd.bytes;
  for (i = 0; i < n; ++i) {
    u8 b = e->xedd.bytes[i];
    switch (b) {
      case 0x66: case 0x67: case 0x2E: case 0x36: case 0x3E: case 0x26:
      case 0x64: case 0x65: case 0xF0: case 0xF2: case 0xF3:
        continue;
      case 0xC4: case 0xC5: case 0x62:
        return 1;
      default:
        if ((b & 0xF0) == 0x40) continue; /* REX */
        return 0;
    }
  }
  return 0;
}

/* Returns 1 if `rde` uses an operand shape the IR cannot express (see above);
 * the caller must BAILOUT rather than lift. */
static int LiftOperandShapeUnmodelled(u64 rde, u64 mop, int region) {
  int roles = LiftOperandRoles(mop);
  if (roles == FBX_LIFT_UNCLASSIFIED) return 1;
  /* LOCK.  No lifter here models an atomic read-modify-write, and on every
   * form this file CAN lift (register destinations, MOV, LEA, branches) the
   * prefix is #UD — which is Tier 1's to raise.  Refuse them all. */
  if (rde & FBX_LIFT_RDE_LOCK) return 1;
  /* Opcode GROUPS: ModRM.reg selects the instruction, so each admitted group
   * names its members.  C6/C7 are MOV only as /0 — C6 /7 is XABORT, C7 /7 is
   * XBEGIN, /1../6 are reserved.  0x80/0x81/0x83 name theirs in AluSubOp
   * (ADC /2 and SBB /3 are refused there).  No other group opcode (0x82,
   * C0/C1, D0-D3, F6/F7, FE/FF, 0F BA, 8F) has a role entry at all. */
  if ((mop == 0x0C6 || mop == 0x0C7) && ModrmReg(rde) != 0) return 1;
  /* firebox#T3R2: group 3 (F6/F7) as TEST /0, NOT /2, NEG /3 — not the /1
   * TEST alias, not MUL/IMUL/DIV/IDIV; group 4/5 (FE/FF) as INC /0, DEC /1
   * and, for FF, CALL /2 and JMP /4 — not the far forms or PUSH r/m. */
  if ((mop == 0x0F6 || mop == 0x0F7) &&
      !(ModrmReg(rde) == 0 || ModrmReg(rde) == 2 || ModrmReg(rde) == 3)) {
    return 1;
  }
  if (mop == 0x0FE && ModrmReg(rde) > 1) return 1;
  /* firebox#T3R3: group 2 as SHL /4, SHR /5, SAR /7 only (no rotates, no
   * /6 alias), and not at operand size 2. */
  if ((mop == 0x0C1 || mop == 0x0D1 || mop == 0x0D3) &&
      (Osz(rde) ||
       !(ModrmReg(rde) == 4 || ModrmReg(rde) == 5 || ModrmReg(rde) == 7))) {
    return 1;
  }
  if (mop == 0x0FF && !(ModrmReg(rde) <= 2 || ModrmReg(rde) == 4)) return 1;
  /* firebox#T3R3: sign extension into 4 or 8 bytes only (no operand-size
   * prefix: CBW, MOVSX r16), and MOVSXD only with REX.W. */
  if ((mop == 0x063 || mop == 0x098 || mop == 0x1BE || mop == 0x1BF) &&
      Osz(rde)) {
    return 1;
  }
  if (mop == 0x063 && !Rexw(rde)) return 1;
  /* firebox#T3R2: an operand-size prefix makes PUSH/POP/RET move 2 bytes
   * (blink/stack.c kStackOsz) and changes CALL/JMP; the IR models only the
   * 8-byte forms.  (These used to lift at 8 bytes regardless.) */
  if (Osz(rde) &&
      ((mop >= 0x050 && mop <= 0x05F) || mop == 0x0E8 || mop == 0x0C3 ||
       (mop == 0x0FF && (ModrmReg(rde) == 2 || ModrmReg(rde) == 4)))) {
    return 1;
  }
  if (!(roles & FBX_LIFT_HAS_MODRM)) return 0;
  if ((roles & FBX_LIFT_REG_BYTE) &&
      LiftIsHighByteReg(rde, (u32)ModrmReg(rde))) {
    return 1;
  }
  if (IsModrmRegister(rde)) {
    /* firebox#T3R3: a region lifts MOVZX/MOVSX from AH/CH/DH/BH
     * (LiftByteSourceRegion). */
    if (region && (mop == 0x1B6 || mop == 0x1BE)) return 0;
    return (roles & FBX_LIFT_RM_BYTE) &&
           LiftIsHighByteReg(rde, (u32)ModrmRm(rde));
  }
  /* Memory operand. */
  if (Asz(rde)) return 1;
  /* firebox#T3R3: a region lift applies the segment base (LiftClassifyAddr
   * records it in the IR; LiftOne refuses an override nothing consumed). */
  if (Sego(rde) && !(roles & FBX_LIFT_NO_SEG) && !region) return 1;
  return 0;
}

/* Lift MOV r/m, r in the memory-store direction (0x88/0x89, mod != 3):
 *   REG_GET (src reg → vreg) ; STORE (mem[base+disp] = vreg).
 * Returns 1 on success, 0 on alloc failure, -1 to request BAILOUT. */
static int LiftStoreRegion(struct LiftCtx *ctx, u64 rde, i64 disp, u8 width,
                           int imm_src, u64 uimm0);

static int LiftMovStoreMem(struct LiftCtx *ctx, u64 rde, i64 disp,
                           u8 width) {
  u32 base_greg;
  u32 val_vreg;
  struct FbxIrInst *get;
  struct FbxIrInst *st;
  if (ctx->region) return LiftStoreRegion(ctx, rde, disp, width, 0, 0);
  if (!ClassifyMemBaseDisp(rde, &base_greg)) return -1;
  get = CtxEmit(ctx);
  if (!get) return 0;
  val_vreg = CtxAllocVreg(ctx);
  get->opcode = FBX_IR_OP_REG_GET;
  get->width = width;
  get->dst_kind = FBX_IR_KIND_VREG;
  get->dst = val_vreg;
  get->src1_kind = FBX_IR_KIND_GREG;
  get->src1 = (u32)RexrReg(rde);
  st = CtxEmit(ctx);
  if (!st) return 0;
  st->opcode = FBX_IR_OP_STORE;
  st->width = width;
  st->dst_kind = FBX_IR_KIND_NONE;
  st->src1_kind = FBX_IR_KIND_GREG; /* base address register */
  st->src1 = base_greg;
  st->src2_kind = FBX_IR_KIND_VREG; /* value to store */
  st->src2 = val_vreg;
  st->imm = (u64)disp; /* sign-extended displacement */
  return 1;
}

/* Lift MOV r, r/m in the memory-load direction (0x8A/0x8B, mod != 3):
 *   LOAD (vreg = mem[base+disp]) ; REG_SET (dst reg = vreg).
 * Returns 1 on success, 0 on alloc failure, -1 to request BAILOUT. */
static int ClassifyLeaMem(u64 rde, u32 *out_base_greg, u32 *out_index_greg,
                          u32 *out_scale);

/* firebox#T3R2 — one memory operand's address.  The per-block lift accepts
 * ClassifyLeaMem's forms (base, optional index*scale, disp); the region lift
 * also accepts RIP-relative, as an absolute VA (the PC after the instruction
 * plus disp, blink/modrm.c ComputeAddress).  Everything else is refused, and
 * Tier 1 runs it. */
struct LiftAddr {
  int abs;
  u32 base, index, scale; /* scale 0: no index (an absolute VA may have one) */
  u64 va;
  i64 disp;
  u8 seg;
};

/* firebox#T3R3: a region lift also takes the no-base SIB forms, `[disp32]`
 * and `[index*scale + disp32]` (an absolute VA, plus the index), and any
 * segment override (blink/modrm.c LoadEffectiveAddress then AddSegment:
 * the override's base is added to whichever form it is). */
static int LiftClassifyAddr(struct LiftCtx *ctx, u64 rde, i64 disp,
                            struct LiftAddr *a) {
  memset(a, 0, sizeof *a);
  if (IsModrmRegister(rde)) return 0;
  if (!SibExists(rde) && ModrmMod(rde) == 0 && ModrmRm(rde) == 5) {
    if (!ctx->region || !IsRipRelative(rde)) return 0;
    a->abs = 1;
    a->va = ctx->next_pc + (u64)disp;
  } else if (ctx->region && SibExists(rde) && !SibHasBase(rde)) {
    a->abs = 1;
    a->va = (u64)disp;
    if (SibHasIndex(rde)) {
      a->index = ((u32)Rexx(rde) << 3) | (u32)SibIndex(rde);
      a->scale = 1u << (u32)SibScale(rde);
    }
  } else {
    if (!ClassifyLeaMem(rde, &a->base, &a->index, &a->scale)) return 0;
    a->disp = disp;
  }
  a->seg = ctx->seg;
  ctx->seg_used = 1;
  return 1;
}

/* LOAD address encoding: src1 GREG base (+ src2 GREG index, scale) + imm
 * disp; or src1_kind IMM with the absolute VA in imm. */
static void LiftSetLoadAddr(struct FbxIrInst *ld, const struct LiftAddr *a) {
  ld->seg = a->seg;
  if (a->abs) {
    ld->src1_kind = FBX_IR_KIND_IMM;
    ld->src1 = 0;
    ld->imm = a->va;
    if (a->scale != 0) {
      ld->src2_kind = FBX_IR_KIND_GREG;
      ld->src2 = a->index;
      ld->scale = a->scale;
    }
    return;
  }
  ld->src1_kind = FBX_IR_KIND_GREG;
  ld->src1 = a->base;
  if (a->scale != 0) {
    ld->src2_kind = FBX_IR_KIND_GREG;
    ld->src2 = a->index;
    ld->scale = a->scale;
  }
  ld->imm = (u64)a->disp;
}

/* Emit `vreg = LOAD{width} [mem]` for the instruction's ModRM memory
 * operand.  Returns 1 (and *out_vreg), 0 on alloc failure, -1 to BAILOUT. */
static int LiftLoadOperand(struct LiftCtx *ctx, u64 rde, i64 disp, u8 width,
                           u32 *out_vreg) {
  struct LiftAddr a;
  struct FbxIrInst *ld;
  if (!LiftClassifyAddr(ctx, rde, disp, &a)) return -1;
  ld = CtxEmit(ctx);
  if (!ld) return 0;
  *out_vreg = CtxAllocVreg(ctx);
  ld->opcode = FBX_IR_OP_LOAD;
  ld->width = width;
  ld->dst_kind = FBX_IR_KIND_VREG;
  ld->dst = *out_vreg;
  LiftSetLoadAddr(ld, &a);
  return 1;
}

/* firebox#T3R2 — a region STORE: `mem = RexrReg` (MOV r/m, r) or `mem = imm`
 * (MOV r/m, imm).  STORE encoding: src1 GREG base or src1_kind IMM with the
 * absolute VA in imm; src2 VREG value; the optional SIB index rides in dst
 * (GREG) with `scale`; imm is the disp for the register forms. */
static int LiftStoreRegion(struct LiftCtx *ctx, u64 rde, i64 disp, u8 width,
                           int imm_src, u64 uimm0) {
  struct LiftAddr a;
  u32 val_vreg;
  struct FbxIrInst *get;
  struct FbxIrInst *st;
  if (!LiftClassifyAddr(ctx, rde, disp, &a)) return -1;
  get = CtxEmit(ctx);
  if (!get) return 0;
  val_vreg = CtxAllocVreg(ctx);
  get->opcode = FBX_IR_OP_REG_GET;
  get->width = width;
  get->dst_kind = FBX_IR_KIND_VREG;
  get->dst = val_vreg;
  if (imm_src) {
    get->src1_kind = FBX_IR_KIND_IMM;
    get->imm = uimm0;
  } else {
    get->src1_kind = FBX_IR_KIND_GREG;
    get->src1 = (u32)RexrReg(rde);
  }
  st = CtxEmit(ctx);
  if (!st) return 0;
  st->opcode = FBX_IR_OP_STORE;
  st->width = width;
  st->src2_kind = FBX_IR_KIND_VREG;
  st->src2 = val_vreg;
  st->seg = a.seg;
  if (a.abs) {
    st->src1_kind = FBX_IR_KIND_IMM;
    st->imm = a.va;
  } else {
    st->src1_kind = FBX_IR_KIND_GREG;
    st->src1 = a.base;
    st->imm = (u64)a.disp;
  }
  if (a.scale != 0) {
    st->dst_kind = FBX_IR_KIND_GREG;
    st->dst = a.index;
    st->scale = a.scale;
  }
  return 1;
}

/* Emit `vreg = LOAD{load_width} [base (+ index*scale) + disp]` and
 * `REG_SET{set_width} RexrReg = vreg`.
 *
 * firebox#JJJ: the load accepts the SIB-index form `[base + index*scale +
 * disp]` as well as `[base + disp]`.  The addressing classification is
 * ClassifyLeaMem's — the one the indexed LEA already uses — so the same forms
 * are refused (RIP-relative, no-base SIB) and Tier 1 runs them.  The index
 * rides in src2 (GREG) with the SIB scale; src2_kind stays NONE for the
 * base-only form, which the emitter reads as "no index term".
 *
 * A narrower load widened by REG_SET is MOVZX: the emitter's narrow loads are
 * i64.load{8,16}_u, which zero-extend, and REG_SET's width then applies the
 * x86 destination rule (4 zero-extends to 64 bits, 2 merges into the low
 * word). */
static int LiftLoadToReg(struct LiftCtx *ctx, u64 rde, i64 disp,
                         u8 load_width, u8 set_width) {
  u32 val_vreg;
  int rc;
  struct FbxIrInst *set;
  rc = LiftLoadOperand(ctx, rde, disp, load_width, &val_vreg);
  if (rc <= 0) return rc;
  set = CtxEmit(ctx);
  if (!set) return 0;
  set->opcode = FBX_IR_OP_REG_SET;
  set->width = set_width;
  set->dst_kind = FBX_IR_KIND_GREG;
  set->dst = (u32)RexrReg(rde);
  set->src1_kind = FBX_IR_KIND_VREG;
  set->src1 = val_vreg;
  return 1;
}

static int LiftMovLoadMem(struct LiftCtx *ctx, u64 rde, i64 disp,
                          u8 width) {
  return LiftLoadToReg(ctx, rde, disp, width, width);
}

/* ALU sub-op decoder for OpAlui / OpAlu group:
 *   Mopcode 0x83 (OpAlui) → ModrmReg encodes the op:
 *     0=ADD  1=OR  2=ADC  3=SBB  4=AND  5=SUB  6=XOR  7=CMP
 * Return one of FBX_IR_OP_ADD/SUB/AND/OR/XOR/CMP or 0 if not lifted (ADC/SBB).
 */
static u8 AluSubOp(u64 rde) {
  switch (ModrmReg(rde)) {
    case 0: return FBX_IR_OP_ADD;
    case 1: return FBX_IR_OP_OR;
    case 4: return FBX_IR_OP_AND;
    case 5: return FBX_IR_OP_SUB;
    case 6: return FBX_IR_OP_XOR;
    case 7: return FBX_IR_OP_CMP;
    default: return 0; /* ADC/SBB not lifted in v0.1 */
  }
}

/* Lift a MOV r/m, r (opcode 0x89 / 0x88).
 *
 *   modrm.mod==3 (reg-to-reg form): REG_GET (src) + REG_SET (r/m as reg).
 *   modrm.mod!=3 (memory-store form, #677): REG_GET (src) + STORE to
 *     mem[base+disp], for the supported `[base+disp]` addressing shape;
 *     unsupported addressing forms (SIB index, RIP-relative, no-base SIB)
 *     return -1 so the caller emits BAILOUT.
 *
 * Returns 1 on success, 0 on alloc failure, -1 to request BAILOUT. */
static int LiftMovEvqpGvqp(struct LiftCtx *ctx, u64 rde, i64 disp,
                           int byte_op) {
  u8 width = WidthFromRde(rde, byte_op);
  u32 src_vreg;
  struct FbxIrInst *get;
  struct FbxIrInst *set;
  if (!IsModrmRegister(rde)) {
    return LiftMovStoreMem(ctx, rde, disp, width);
  }
  get = CtxEmit(ctx);
  if (!get) return 0;
  src_vreg = CtxAllocVreg(ctx);
  get->opcode = FBX_IR_OP_REG_GET;
  get->width = width;
  get->dst_kind = FBX_IR_KIND_VREG;
  get->dst = src_vreg;
  get->src1_kind = FBX_IR_KIND_GREG;
  get->src1 = (u32)RexrReg(rde);
  set = CtxEmit(ctx);
  if (!set) return 0;
  set->opcode = FBX_IR_OP_REG_SET;
  set->width = width;
  set->dst_kind = FBX_IR_KIND_GREG;
  set->dst = (u32)RexbRm(rde);
  set->src1_kind = FBX_IR_KIND_VREG;
  set->src1 = src_vreg;
  return 1;
}

/* Lift MOV r, r/m (opcode 0x8B / 0x8A) — direction inverted from 0x89.
 *
 *   modrm.mod==3 (reg-to-reg form): REG_GET (r/m as reg) + REG_SET (dst).
 *   modrm.mod!=3 (memory-load form, #677): LOAD mem[base+disp] + REG_SET
 *     (dst), for the supported `[base+disp]` addressing shape;
 *     unsupported addressing forms return -1 so the caller emits BAILOUT.
 *
 * Returns 1 on success, 0 on alloc failure, -1 to request BAILOUT. */
static int LiftMovGvqpEvqp(struct LiftCtx *ctx, u64 rde, i64 disp,
                           int byte_op) {
  u8 width = WidthFromRde(rde, byte_op);
  u32 src_vreg;
  struct FbxIrInst *get;
  struct FbxIrInst *set;
  if (!IsModrmRegister(rde)) {
    return LiftMovLoadMem(ctx, rde, disp, width);
  }
  get = CtxEmit(ctx);
  if (!get) return 0;
  src_vreg = CtxAllocVreg(ctx);
  get->opcode = FBX_IR_OP_REG_GET;
  get->width = width;
  get->dst_kind = FBX_IR_KIND_VREG;
  get->dst = src_vreg;
  get->src1_kind = FBX_IR_KIND_GREG;
  get->src1 = (u32)RexbRm(rde);
  set = CtxEmit(ctx);
  if (!set) return 0;
  set->opcode = FBX_IR_OP_REG_SET;
  set->width = width;
  set->dst_kind = FBX_IR_KIND_GREG;
  set->dst = (u32)RexrReg(rde);
  set->src1_kind = FBX_IR_KIND_VREG;
  set->src1 = src_vreg;
  return 1;
}

/* Lift MOV reg, imm (opcodes 0xB8-0xBF: OpMovZvqpIvqp). */
static int LiftMovZvqpIvqp(struct LiftCtx *ctx, u64 rde, u64 uimm0) {
  u8 width = WidthFromRde(rde, 0);
  struct FbxIrInst *set = CtxEmit(ctx);
  if (!set) return 0;
  set->opcode = FBX_IR_OP_REG_SET;
  set->width = width;
  set->dst_kind = FBX_IR_KIND_GREG;
  set->dst = (u32)RexbSrm(rde);
  set->src1_kind = FBX_IR_KIND_IMM;
  set->imm = uimm0;
  return 1;
}

/* Lift MOV r/m, imm (opcode 0xC6 byte / 0xC7 word: OpMovImm).  Approximate as
 * REG_SET when modrm.mod==3, leave §13.2 to handle the memory store case.
 *
 * firebox#RAT: 0xC6 is the BYTE form.  It was lifted at the 0xC7 width, so
 * `c6 c0 5a` (mov $0x5a,%al) became a 32-bit register write: it replaced all
 * of eAX and, by the 32-bit zero-extension rule, cleared the upper half of
 * rAX as well. */
static int LiftMovImm(struct LiftCtx *ctx, u64 rde, i64 disp, u64 uimm0,
                      int byte_op) {
  u8 width = WidthFromRde(rde, byte_op);
  struct FbxIrInst *set;
  /* firebox#T3R2: the memory form is a store, not a register write.  (It was
   * lifted as `RexbRm = imm`; the per-block gate's mod3 check refused it.) */
  if (!IsModrmRegister(rde)) {
    if (!ctx->region) return -1;
    return LiftStoreRegion(ctx, rde, disp, width, 1, uimm0);
  }
  set = CtxEmit(ctx);
  if (!set) return 0;
  set->opcode = FBX_IR_OP_REG_SET;
  set->width = width;
  set->dst_kind = FBX_IR_KIND_GREG;
  set->dst = (u32)RexbRm(rde);
  set->src1_kind = FBX_IR_KIND_IMM;
  set->imm = uimm0;
  return 1;
}

/* Lift LEA (opcode 0x8D).
 *
 * firebox#738 (ROOT CAUSE — correctness, load-bearing): LEA's emit lowers
 * exactly ONE addressing shape, `[base + disp]` (FBX_IR_OP_LEA = greg base +
 * immediate disp; see EmitLea).  The original lift sourced the base
 * UNCONDITIONALLY as `RexbRm(rde)` and dropped any SIB index register — which
 * is WRONG for every SIB-addressed LEA:
 *   - `lea r14, [r9 + rax]` (SIB byte, base=r9 via RexbBase, index=rax) was
 *     mis-lifted as `r14 = weg[RexbRm] + disp`.  With a SIB byte ModrmRm==4, so
 *     RexbRm decodes reg 4/12 (NOT the SIB base r9), AND the index register rax
 *     was silently dropped → r14 computed from the wrong base with no index →
 *     a garbage pointer.  In the hot strcmp/strcoll string-end computation
 *     (`mov r9,[rsp+disp]; lea r14,[r9+rax]; test …; jns …`) this produced a
 *     -1/out-of-range pointer that Tier 1 then dereferenced → SIGSEGV after
 *     correct output up to the crash (witnessed: sort crashes at line 85 once
 *     the comparison block escalates to T2 at the default hotness threshold).
 * This was masked until #735 enabled the inline wide guest LOAD, because the
 * #719 acceptance gate refused LOAD/STORE → the strcmp blocks (which carry the
 * wide stack LOAD feeding the LEA base) never escalated.  The SESSION-4
 * "wide LOAD + in-block flag-writer + BRANCH_COND" correlation was exactly
 * this block shape: the wide LOAD sources the LEA's base; the flag-writer +
 * Jcc terminate it.  Refusing the wide LOAD OR the flag-reader was byte-clean
 * only because each bailed the SAME block that carried the mis-lifted LEA.
 *
 * FIX (correct-or-refuse, the established #677 pattern): validate the EA is the
 * supported `[base + disp]` via ClassifyMemBaseDisp — which sources the base
 * correctly (RexbBase for the SIB-base form) and REFUSES SIB-index /
 * RIP-relative / no-base forms.  A refused form returns -1 → LiftOne emits a
 * BAILOUT so Tier 1 computes the LEA authoritatively (it already handles every
 * addressing form).  This both fixes the mis-sourced base AND stops dropping
 * the index register; the SIB-index LEA support is a clean follow-on (extend
 * FBX_IR_OP_LEA's emit with an index+scale term, then accept it here). */
/* Classify an LEA memory operand for the inline emitter (#778).               */
/*                                                                            */
/* LEA computes `dst = base + index*scale + disp`; it NEVER dereferences      */
/* guest memory, so (unlike LOAD/STORE) there is no software-MMU concern — it */
/* is pure register arithmetic into the Machine struct.  That is why LEA can  */
/* support the SIB-INDEX form inline while LOAD/STORE/MOV still refuse it      */
/* (their addressing form drives a translated memory access, deferred).       */
/*                                                                            */
/* Supported forms (return 1):                                                */
/*   - No SIB:           base = RexbRm(rde),   no index, scale 0.             */
/*   - SIB base-only:    base = RexbBase(rde), no index, scale 0.            */
/*   - SIB base+index:   base = RexbBase(rde), index = Rexx<<3 | SibIndex,   */
/*                       scale = 1 << SibScale(rde).                          */
/*                                                                            */
/* Refused (return 0 → caller BAILOUTs to Tier 1, which handles every form): */
/*   - RIP-relative (no SIB, mod==0, ModrmRm==5) — no base register.         */
/*   - No-base SIB (SibIsAbsolute / disp32 + optional index, mod==0,         */
/*     SibBase==5): the address has no base GREG.  The index-without-base    */
/*     encoding is rare and would need a 2-term (index*scale + disp) emit     */
/*     the disp-or-base inline path doesn't model; bail rather than widen.   */
/*                                                                            */
/* `*out_scale` is 0 when there is no index term, else the SIB scale.         */
static int ClassifyLeaMem(u64 rde, u32 *out_base_greg, u32 *out_index_greg,
                          u32 *out_scale) {
  *out_index_greg = 0;
  *out_scale = 0;
  if (IsModrmRegister(rde)) return 0; /* LEA must have mod != 3 */
  if (!SibExists(rde)) {
    /* No SIB.  RIP-relative (mod==0, ModrmRm==5) has no base register. */
    if (ModrmMod(rde) == 0 && ModrmRm(rde) == 5) return 0;
    *out_base_greg = (u32)RexbRm(rde);
    return 1;
  }
  /* SIB present.  Require a base register (no-base / absolute forms refused). */
  if (!SibHasBase(rde)) return 0;
  *out_base_greg = (u32)RexbBase(rde);
  if (SibHasIndex(rde)) {
    /* Index register = Rexx<<3 | SibIndex; scale = 1 << SibScale.  Note that
     * SibHasIndex already excludes the canonical no-index encoding
     * (SibIndex==4 && !Rexx), so reg 4 here is the legitimate r12 index. */
    *out_index_greg = ((u32)Rexx(rde) << 3) | (u32)SibIndex(rde);
    *out_scale = 1u << (u32)SibScale(rde);
  }
  return 1;
}

static int LiftLea(struct LiftCtx *ctx, u64 rde, i64 disp) {
  u8 width = WidthFromRde(rde, 0);
  u32 base_greg;
  u32 index_greg;
  u32 scale;
  struct FbxIrInst *lea;
  /* #778 — inline the SIB-INDEX form too: `lea dst, [base + index*scale +
   * disp]`.  #738 made LiftLea correct-or-refuse but the supported set was the
   * disp-only `[base + disp]` shape (it BAILED every indexed LEA to Tier 1).
   * ClassifyLeaMem now accepts the indexed form (the hot strcmp/strcoll
   * `lea r14,[r9+rax]`) and refuses only the no-base / RIP-relative forms.
   * EmitLea adds `weg[index]*scale` to the effective address — no MMU
   * translation (LEA never touches guest memory). */
  if (ctx->region && IsRipRelative(rde)) {
    /* firebox#T3R2: `lea r, [rip + disp]` is a constant. */
    struct FbxIrInst *set = CtxEmit(ctx);
    if (!set) return 0;
    set->opcode = FBX_IR_OP_REG_SET;
    set->width = width;
    set->dst_kind = FBX_IR_KIND_GREG;
    set->dst = (u32)RexrReg(rde);
    set->src1_kind = FBX_IR_KIND_IMM;
    set->imm = ctx->next_pc + (u64)disp;
    return 1;
  }
  if (ctx->region && SibExists(rde) && !SibHasBase(rde)) {
    /* firebox#T3R3: no base, `lea r, [index*scale + disp32]` (or the
     * constant `[disp32]`).  LEA src1_kind NONE: regions only. */
    lea = CtxEmit(ctx);
    if (!lea) return 0;
    if (!SibHasIndex(rde)) {
      lea->opcode = FBX_IR_OP_REG_SET;
      lea->width = width;
      lea->dst_kind = FBX_IR_KIND_GREG;
      lea->dst = (u32)RexrReg(rde);
      lea->src1_kind = FBX_IR_KIND_IMM;
      lea->imm = (u64)disp;
      return 1;
    }
    lea->opcode = FBX_IR_OP_LEA;
    lea->width = width;
    lea->dst_kind = FBX_IR_KIND_GREG;
    lea->dst = (u32)RexrReg(rde);
    lea->src1_kind = FBX_IR_KIND_NONE;
    lea->src2_kind = FBX_IR_KIND_GREG;
    lea->src2 = ((u32)Rexx(rde) << 3) | (u32)SibIndex(rde);
    lea->scale = 1u << (u32)SibScale(rde);
    lea->imm = (u64)disp;
    return 1;
  }
  if (!ClassifyLeaMem(rde, &base_greg, &index_greg, &scale)) {
    return -1;
  }
  lea = CtxEmit(ctx);
  if (!lea) return 0;
  lea->opcode = FBX_IR_OP_LEA;
  lea->width = width;
  lea->dst_kind = FBX_IR_KIND_GREG;
  lea->dst = (u32)RexrReg(rde);
  lea->src1_kind = FBX_IR_KIND_GREG;
  lea->src1 = base_greg;
  if (scale != 0) {
    /* Indexed form: src2 = index greg, scale = SIB scale. */
    lea->src2_kind = FBX_IR_KIND_GREG;
    lea->src2 = index_greg;
    lea->scale = scale;
  } else {
    /* Disp-only form (legacy IMM marker; src2 carries no greg). */
    lea->src2_kind = FBX_IR_KIND_IMM;
  }
  lea->imm = (u64)disp;
  return 1;
}

/* firebox#T3R2 — shared tails for the region-only lifts. */
static int LiftGetReg(struct LiftCtx *ctx, u8 width, u32 greg, u32 *out) {
  struct FbxIrInst *g = CtxEmit(ctx);
  if (!g) return 0;
  *out = CtxAllocVreg(ctx);
  g->opcode = FBX_IR_OP_REG_GET;
  g->width = width;
  g->dst_kind = FBX_IR_KIND_VREG;
  g->dst = *out;
  g->src1_kind = FBX_IR_KIND_GREG;
  g->src1 = greg;
  return 1;
}

static int LiftGetImm(struct LiftCtx *ctx, u8 width, u64 imm, u32 *out) {
  struct FbxIrInst *g = CtxEmit(ctx);
  if (!g) return 0;
  *out = CtxAllocVreg(ctx);
  g->opcode = FBX_IR_OP_REG_GET;
  g->width = width;
  g->dst_kind = FBX_IR_KIND_VREG;
  g->dst = *out;
  g->src1_kind = FBX_IR_KIND_IMM;
  g->imm = imm;
  return 1;
}

/* `op` on two vregs; the result goes to `dst_greg` unless op is CMP/TEST;
 * then SET_FLAGS_RAW of kind `flags_kind` (0: no flags) over `flags_src1`
 * and `flags_src2`. */
static int LiftAluTail(struct LiftCtx *ctx, u8 width, u8 op, u32 lhs, u32 rhs,
                       u32 dst_greg, u8 flags_kind, u32 flags_src1,
                       u32 flags_src2) {
  int writes = op != FBX_IR_OP_CMP && op != FBX_IR_OP_TEST;
  u32 res = 0;
  struct FbxIrInst *alu = CtxEmit(ctx);
  if (!alu) return 0;
  if (writes) res = CtxAllocVreg(ctx);
  alu->opcode = op;
  alu->width = width;
  alu->dst_kind = writes ? FBX_IR_KIND_VREG : FBX_IR_KIND_NONE;
  alu->dst = res;
  alu->src1_kind = FBX_IR_KIND_VREG;
  alu->src1 = lhs;
  alu->src2_kind = FBX_IR_KIND_VREG;
  alu->src2 = rhs;
  if (writes) {
    struct FbxIrInst *set = CtxEmit(ctx);
    if (!set) return 0;
    set->opcode = FBX_IR_OP_REG_SET;
    set->width = width;
    set->dst_kind = FBX_IR_KIND_GREG;
    set->dst = dst_greg;
    set->src1_kind = FBX_IR_KIND_VREG;
    set->src1 = res;
  }
  if (flags_kind) {
    struct FbxIrInst *fl = CtxEmit(ctx);
    if (!fl) return 0;
    fl->opcode = FBX_IR_OP_SET_FLAGS_RAW;
    fl->width = width;
    fl->imm = flags_kind;
    fl->src1_kind = FBX_IR_KIND_VREG;
    fl->src1 = flags_src1;
    fl->src2_kind = FBX_IR_KIND_VREG;
    fl->src2 = flags_src2;
  }
  return 1;
}

/* firebox#T3R2 — the memory-operand ALU forms a region lowers, all of which
 * READ memory only: `op r, [m]` (r is the destination), `cmp/test [m], r`,
 * and `cmp [m], imm`.  The read-modify-write forms (`op [m], r`, `op [m],
 * imm` for every op but CMP/TEST) are refused: Tier 1 runs them. */
static int LiftAluMemRegion(struct LiftCtx *ctx, u64 rde, i64 disp, u8 width,
                            u8 op, int mem_is_lhs, int imm_rhs, u64 uimm0) {
  u32 m, o;
  int rc;
  if (!ctx->region) return -1;
  if (mem_is_lhs && op != FBX_IR_OP_CMP && op != FBX_IR_OP_TEST) return -1;
  rc = LiftLoadOperand(ctx, rde, disp, width, &m);
  if (rc <= 0) return rc;
  if (imm_rhs) {
    if (!LiftGetImm(ctx, width, uimm0, &o)) return 0;
  } else {
    if (!LiftGetReg(ctx, width, (u32)RexrReg(rde), &o)) return 0;
  }
  if (mem_is_lhs) return LiftAluTail(ctx, width, op, m, o, 0, op, m, o);
  return LiftAluTail(ctx, width, op, o, m, (u32)RexrReg(rde), op, o, m);
}

/* firebox#T3R2 — group 3 (F6/F7) /0 TEST r, imm; /2 NOT; /3 NEG; group 4/5
 * (FE/FF) /0 INC, /1 DEC — register forms; FF /2 CALL and /4 JMP, register
 * or memory.  Region lifts only.  Flags are blink/alu.c's: NOT writes none;
 * INC/DEC keep CF; NEG's AF is (x != 0). */
static int LiftUnaryRegion(struct LiftCtx *ctx, u64 rde, i64 disp, u64 uimm0,
                           u64 mop) {
  int byte_op = mop == 0x0F6 || mop == 0x0FE;
  u8 width = WidthFromRde(rde, byte_op);
  u32 greg = (u32)RexbRm(rde);
  u32 x, k;
  int sub = (int)ModrmReg(rde);
  if (!ctx->region) return -1;
  if ((mop == 0x0FF) && (sub == 2 || sub == 4)) {
    struct FbxIrInst *c;
    u32 t;
    if (IsModrmRegister(rde)) {
      if (!LiftGetReg(ctx, 8, greg, &t)) return 0;
    } else {
      int rc = LiftLoadOperand(ctx, rde, disp, 8, &t);
      if (rc <= 0) return rc;
    }
    c = CtxEmit(ctx);
    if (!c) return 0;
    c->opcode = sub == 2 ? FBX_IR_OP_CALL_INDIRECT : FBX_IR_OP_JMP_INDIRECT;
    c->src1_kind = FBX_IR_KIND_VREG;
    c->src1 = t;
    if (sub == 2) {
      c->imm = ctx->next_pc; /* return PC, full width */
      ctx->flags |= FBX_IR_BLOCK_FLAG_HAS_CALL;
    }
    ctx->flags |= FBX_IR_BLOCK_FLAG_HAS_INDIRECT_BR;
    return 1;
  }
  if (!IsModrmRegister(rde)) {
    /* firebox#T3R3: TEST [m], imm only reads memory. */
    if ((mop == 0x0F6 || mop == 0x0F7) && sub == 0) {
      return LiftAluMemRegion(ctx, rde, disp, width, FBX_IR_OP_TEST,
                              /*mem_is_lhs=*/1, /*imm_rhs=*/1, uimm0);
    }
    return -1; /* read-modify-write: Tier 1 */
  }
  if (mop == 0x0F6 || mop == 0x0F7) {
    switch (sub) {
      case 0: /* TEST r, imm */
        if (!LiftGetReg(ctx, width, greg, &x)) return 0;
        if (!LiftGetImm(ctx, width, uimm0, &k)) return 0;
        return LiftAluTail(ctx, width, FBX_IR_OP_TEST, x, k, 0,
                           FBX_IR_OP_TEST, x, k);
      case 2: /* NOT: x ^ all-ones, no flags */
        if (!LiftGetReg(ctx, width, greg, &x)) return 0;
        if (!LiftGetImm(ctx, width, ~(u64)0, &k)) return 0;
        return LiftAluTail(ctx, width, FBX_IR_OP_XOR, x, k, greg, 0, 0, 0);
      case 3: /* NEG: 0 - x */
        if (!LiftGetReg(ctx, width, greg, &x)) return 0;
        if (!LiftGetImm(ctx, width, 0, &k)) return 0;
        return LiftAluTail(ctx, width, FBX_IR_OP_SUB, k, x, greg,
                           FBX_IR_FLAGS_NEG, x, x);
      default:
        return -1;
    }
  }
  if (sub != 0 && sub != 1) return -1;
  if (!LiftGetReg(ctx, width, greg, &x)) return 0;
  if (!LiftGetImm(ctx, width, 1, &k)) return 0;
  return LiftAluTail(ctx, width, sub == 0 ? FBX_IR_OP_ADD : FBX_IR_OP_SUB, x,
                     k, greg, sub == 0 ? FBX_IR_FLAGS_INC : FBX_IR_FLAGS_DEC,
                     x, k);
}

/* firebox#T3R3 — group 2 SHL/SHR/SAR r, 1 (D1) / imm8 (C1) / CL (D3), at
 * widths 4 and 8, register destination.  Region lifts only.  The flags are
 * blink/alu.c's ShlN / ShrN / SarN (kBsu), via SET_FLAGS_RAW over the operand
 * and the unmasked count. */
static int LiftShiftRegion(struct LiftCtx *ctx, u64 rde, u64 uimm0, u64 mop) {
  u8 width = WidthFromRde(rde, 0);
  u32 greg = (u32)RexbRm(rde);
  u32 x, c, z;
  u8 op, kind;
  struct FbxIrInst *sh, *set, *fl;
  if (!ctx->region || !IsModrmRegister(rde)) return -1;
  if (width != 4 && width != 8) return -1;
  switch (ModrmReg(rde)) {
    case 4: op = FBX_IR_OP_SHL; kind = FBX_IR_FLAGS_SHL; break;
    case 5: op = FBX_IR_OP_SHR; kind = FBX_IR_FLAGS_SHR; break;
    case 7: op = FBX_IR_OP_SAR; kind = FBX_IR_FLAGS_SAR; break;
    default: return -1;
  }
  if (!LiftGetReg(ctx, width, greg, &x)) return 0;
  if (mop == 0x0D3) {
    if (!LiftGetReg(ctx, 1, 1u /* CL */, &c)) return 0;
  } else {
    if (!LiftGetImm(ctx, 1, mop == 0x0D1 ? 1u : (uimm0 & 0xFFu), &c)) return 0;
  }
  sh = CtxEmit(ctx);
  if (!sh) return 0;
  z = CtxAllocVreg(ctx);
  sh->opcode = op;
  sh->width = width;
  sh->dst_kind = FBX_IR_KIND_VREG;
  sh->dst = z;
  sh->src1_kind = FBX_IR_KIND_VREG;
  sh->src1 = x;
  sh->src2_kind = FBX_IR_KIND_VREG;
  sh->src2 = c;
  set = CtxEmit(ctx);
  if (!set) return 0;
  set->opcode = FBX_IR_OP_REG_SET;
  set->width = width;
  set->dst_kind = FBX_IR_KIND_GREG;
  set->dst = greg;
  set->src1_kind = FBX_IR_KIND_VREG;
  set->src1 = z;
  fl = CtxEmit(ctx);
  if (!fl) return 0;
  fl->opcode = FBX_IR_OP_SET_FLAGS_RAW;
  fl->width = width;
  fl->imm = kind;
  fl->src1_kind = FBX_IR_KIND_VREG;
  fl->src1 = x;
  fl->src2_kind = FBX_IR_KIND_VREG;
  fl->src2 = c;
  return 1;
}

/* Common body for the immediate-form ALU lift: `greg` = OP(greg, imm) with
 * the eager SET_FLAGS_RAW.  `greg` is the destination/lhs guest register id;
 * `op` must be a non-zero FBX_IR_OP_* (the caller resolves ADC/SBB to a
 * bailout).  REG_GET with an IMM src1 materializes the immediate into a vreg.
 *
 * Shared by the group-1 r/m,imm forms (greg = RexbRm) and the #794
 * accumulator-imm forms (greg = 0 = rAX). */
static int LiftAluImmCommon(struct LiftCtx *ctx, u8 width, u32 greg,
                            u64 uimm0, u8 op) {
  u32 lhs_vreg;
  u32 rhs_vreg;
  u32 res_vreg;
  struct FbxIrInst *get_lhs;
  struct FbxIrInst *set_rhs;
  struct FbxIrInst *alu;
  struct FbxIrInst *set_res;
  struct FbxIrInst *flags;
  get_lhs = CtxEmit(ctx);
  if (!get_lhs) return 0;
  lhs_vreg = CtxAllocVreg(ctx);
  get_lhs->opcode = FBX_IR_OP_REG_GET;
  get_lhs->width = width;
  get_lhs->dst_kind = FBX_IR_KIND_VREG;
  get_lhs->dst = lhs_vreg;
  get_lhs->src1_kind = FBX_IR_KIND_GREG;
  get_lhs->src1 = greg;
  set_rhs = CtxEmit(ctx);
  if (!set_rhs) return 0;
  rhs_vreg = CtxAllocVreg(ctx);
  set_rhs->opcode = FBX_IR_OP_REG_GET;  /* loads imm into a vreg slot */
  set_rhs->width = width;
  set_rhs->dst_kind = FBX_IR_KIND_VREG;
  set_rhs->dst = rhs_vreg;
  set_rhs->src1_kind = FBX_IR_KIND_IMM;
  set_rhs->imm = uimm0;
  alu = CtxEmit(ctx);
  if (!alu) return 0;
  res_vreg = (op == FBX_IR_OP_CMP) ? 0 : CtxAllocVreg(ctx);
  alu->opcode = op;
  alu->width = width;
  alu->dst_kind = (op == FBX_IR_OP_CMP) ? FBX_IR_KIND_NONE : FBX_IR_KIND_VREG;
  alu->dst = res_vreg;
  alu->src1_kind = FBX_IR_KIND_VREG;
  alu->src1 = lhs_vreg;
  alu->src2_kind = FBX_IR_KIND_VREG;
  alu->src2 = rhs_vreg;
  if (op != FBX_IR_OP_CMP) {
    set_res = CtxEmit(ctx);
    if (!set_res) return 0;
    set_res->opcode = FBX_IR_OP_REG_SET;
    set_res->width = width;
    set_res->dst_kind = FBX_IR_KIND_GREG;
    set_res->dst = greg;
    set_res->src1_kind = FBX_IR_KIND_VREG;
    set_res->src1 = res_vreg;
  }
  flags = CtxEmit(ctx);
  if (!flags) return 0;
  flags->opcode = FBX_IR_OP_SET_FLAGS_RAW;
  flags->width = width;
  flags->imm = op; /* records which ALU op set the flags */
  /* #599 (FBX_IR_VERSION 2): also encode the operand vregs so the emit pass
   * can compute the flag update without re-walking IR history.  The encoding
   * is uniform across CMP/non-CMP — emit pass recomputes the i64 result
   * locally from src1/src2/op for the flag-bit derivation. */
  flags->src1_kind = FBX_IR_KIND_VREG;
  flags->src1 = lhs_vreg;
  flags->src2_kind = FBX_IR_KIND_VREG;
  flags->src2 = rhs_vreg;
  return 1;
}

/* Lift ALU group 1 r/m, imm (opcode 0x83 = OpAlui w/ sign-extended imm8;
 * also 0x81 = OpAlui w/ imm32; mopcode 0x080 / 0x081 / 0x082 / 0x083). */
static int LiftAlui(struct LiftCtx *ctx, u64 rde, i64 disp, u64 uimm0,
                    int byte_op) {
  u8 op = AluSubOp(rde);
  if (op == 0) return -1; /* ADC/SBB — bailout */
  /* firebox#T3R2: the memory form is not a register operation.  (It was
   * lifted as one on RexbRm; the per-block gate's mod3 check refused it.)
   * A region lowers `cmp [m], imm`; the rest is read-modify-write. */
  if (!IsModrmRegister(rde)) {
    return LiftAluMemRegion(ctx, rde, disp, WidthFromRde(rde, byte_op), op,
                            /*mem_is_lhs=*/1, /*imm_rhs=*/1, uimm0);
  }
  return LiftAluImmCommon(ctx, WidthFromRde(rde, byte_op), (u32)RexbRm(rde),
                          uimm0, op);
}

/* #794 — lift the accumulator-immediate ALU forms (no modrm byte; the
 * destination/source is always rAX/eAX/AX/AL = greg 0):
 *   0x04 ADD AL,imm8   0x05 ADD eAX,imm32   (+ OR/AND/SUB/XOR/CMP siblings).
 * The assembler picks these short encodings for `add rax, imm` etc. — the
 * exact shape the #794 register loop uses (`add rax, 12345`).  `op` is the
 * resolved FBX_IR_OP_* (caller maps the opcode); byte_op selects AL vs eAX. */
static int LiftAluAccImm(struct LiftCtx *ctx, u64 rde, u64 uimm0, u8 op,
                         int byte_op) {
  return LiftAluImmCommon(ctx, WidthFromRde(rde, byte_op), 0u, uimm0, op);
}

/* #794 (T3/T4 emit-coverage floor) — lift the truncating IMUL forms.
 *
 *   has_imm==1: IMUL r, r/m, imm   (0x69 imm32-sx / 0x6B imm8-sx).
 *               dst = RexrReg; lhs = r/m (RexbRm); rhs = imm.
 *   has_imm==0: IMUL r, r/m        (0x0FAF).
 *               dst = RexrReg; lhs = RexrReg; rhs = r/m (RexbRm).
 *
 * Shape mirrors LiftAlui/LiftAluRR (REG_GET lhs + REG_GET rhs + OP + REG_SET)
 * but with op=FBX_IR_OP_IMUL and — deliberately — NO trailing SET_FLAGS_RAW:
 * imul's CF/OF (full-product overflow) + x86-undefined SF/ZF/AF/PF are not
 * synthesized at this increment.  The emit coverage gate's imul-flag-liveness
 * pass refuses any block where those flags are observed, so Tier 1 computes
 * them (correct-or-refuse).  REG_GET with an IMM src1 materializes the
 * immediate into a vreg (same trick LiftAlui uses).
 *
 * Reg-form (modrm.mod==3) only: the r/m memory-operand form requests a
 * BAILOUT (return -1) — the §13.5 memory path is a later increment. */
static int LiftImul(struct LiftCtx *ctx, u64 rde, int has_imm, u64 uimm0) {
  u8 width = WidthFromRde(rde, 0); /* no byte form for 0x69/0x6B/0x0FAF */
  u32 lhs_vreg;
  u32 rhs_vreg;
  u32 res_vreg;
  struct FbxIrInst *get_lhs;
  struct FbxIrInst *get_rhs;
  struct FbxIrInst *mul;
  struct FbxIrInst *set_res;
  if (!IsModrmRegister(rde)) {
    return -1; /* memory r/m operand — bail to Tier 1 (later increment) */
  }
  get_lhs = CtxEmit(ctx);
  if (!get_lhs) return 0;
  lhs_vreg = CtxAllocVreg(ctx);
  get_lhs->opcode = FBX_IR_OP_REG_GET;
  get_lhs->width = width;
  get_lhs->dst_kind = FBX_IR_KIND_VREG;
  get_lhs->dst = lhs_vreg;
  get_lhs->src1_kind = FBX_IR_KIND_GREG;
  get_lhs->src1 = has_imm ? (u32)RexbRm(rde) : (u32)RexrReg(rde);
  get_rhs = CtxEmit(ctx);
  if (!get_rhs) return 0;
  rhs_vreg = CtxAllocVreg(ctx);
  get_rhs->opcode = FBX_IR_OP_REG_GET;
  get_rhs->width = width;
  get_rhs->dst_kind = FBX_IR_KIND_VREG;
  get_rhs->dst = rhs_vreg;
  if (has_imm) {
    get_rhs->src1_kind = FBX_IR_KIND_IMM;
    get_rhs->imm = uimm0;
  } else {
    get_rhs->src1_kind = FBX_IR_KIND_GREG;
    get_rhs->src1 = (u32)RexbRm(rde);
  }
  mul = CtxEmit(ctx);
  if (!mul) return 0;
  res_vreg = CtxAllocVreg(ctx);
  mul->opcode = FBX_IR_OP_IMUL;
  mul->width = width;
  mul->dst_kind = FBX_IR_KIND_VREG;
  mul->dst = res_vreg;
  mul->src1_kind = FBX_IR_KIND_VREG;
  mul->src1 = lhs_vreg;
  mul->src2_kind = FBX_IR_KIND_VREG;
  mul->src2 = rhs_vreg;
  set_res = CtxEmit(ctx);
  if (!set_res) return 0;
  set_res->opcode = FBX_IR_OP_REG_SET;
  set_res->width = width;
  set_res->dst_kind = FBX_IR_KIND_GREG;
  set_res->dst = (u32)RexrReg(rde);
  set_res->src1_kind = FBX_IR_KIND_VREG;
  set_res->src1 = res_vreg;
  if (ctx->region) {
    /* firebox#T3R3: a region models IMUL's CF/OF (FBX_IR_FLAGS_IMUL). */
    struct FbxIrInst *fl = CtxEmit(ctx);
    if (!fl) return 0;
    fl->opcode = FBX_IR_OP_SET_FLAGS_RAW;
    fl->width = width;
    fl->imm = FBX_IR_FLAGS_IMUL;
    fl->src1_kind = FBX_IR_KIND_VREG;
    fl->src1 = lhs_vreg;
    fl->src2_kind = FBX_IR_KIND_VREG;
    fl->src2 = rhs_vreg;
  }
  return 1;
}

/* Lift ALU group r/m, r (mopcode 0x01 = OpAluAdd, 0x29 = OpAluSub, 0x21 =
 * OpAluAnd, 0x09 = OpAluOr, 0x31 = OpAluXor, 0x39 = OpAluCmp, 0x85 =
 * OpAluTest).  Approximate as two REG_GETs + ALU + (REG_SET unless
 * cmp/test) + SET_FLAGS_RAW.
 *
 * `op_dir` selects the operand-direction encoding:
 *   FBX_ALU_DIR_RM_R: r/m is destination (mopcode 0x01/0x09/0x21/0x29/...).
 *     lhs = RexbRm, rhs = RexrReg, dst = RexbRm.
 *   FBX_ALU_DIR_R_RM: r is destination (mopcode 0x03/0x0B/0x23/0x2B/...).
 *     lhs = RexrReg, rhs = RexbRm, dst = RexrReg.
 *
 * #596 §13.5 added FBX_ALU_DIR_R_RM to lift the mirror-direction variants
 * (0x002/0x003/0x00A/0x00B/0x022/0x023/0x02A/0x02B/0x032/0x033/0x03A/0x03B).
 * The emitted IR shape is byte-identical to the RM_R case once the lhs/rhs
 * vregs are populated — only the source greg ids differ.  TEST has no
 * mirror form in x86 (single encoding 0x84/0x85), so the dir flag is only
 * meaningful for ADD/OR/AND/SUB/XOR/CMP. */
#define FBX_ALU_DIR_RM_R 0
#define FBX_ALU_DIR_R_RM 1

static int LiftAluRR(struct LiftCtx *ctx, u64 rde, i64 disp, u8 op,
                     int byte_op, int op_dir) {
  u8 width = WidthFromRde(rde, byte_op);
  u32 lhs_greg;
  u32 rhs_greg;
  u32 dst_greg;
  u32 lhs_vreg;
  u32 rhs_vreg;
  u32 res_vreg;
  struct FbxIrInst *get_lhs;
  struct FbxIrInst *get_rhs;
  struct FbxIrInst *alu;
  struct FbxIrInst *set_res;
  struct FbxIrInst *flags;
  /* firebox#T3R2: the memory form is not a register operation.  (It was
   * lifted as one on RexbRm; the per-block gate's mod3 check refused it.) */
  if (!IsModrmRegister(rde)) {
    return LiftAluMemRegion(ctx, rde, disp, width, op,
                            op_dir == FBX_ALU_DIR_RM_R, 0, 0);
  }
  if (op_dir == FBX_ALU_DIR_RM_R) {
    lhs_greg = (u32)RexbRm(rde);
    rhs_greg = (u32)RexrReg(rde);
    dst_greg = (u32)RexbRm(rde);
  } else {
    lhs_greg = (u32)RexrReg(rde);
    rhs_greg = (u32)RexbRm(rde);
    dst_greg = (u32)RexrReg(rde);
  }
  get_lhs = CtxEmit(ctx);
  if (!get_lhs) return 0;
  lhs_vreg = CtxAllocVreg(ctx);
  get_lhs->opcode = FBX_IR_OP_REG_GET;
  get_lhs->width = width;
  get_lhs->dst_kind = FBX_IR_KIND_VREG;
  get_lhs->dst = lhs_vreg;
  get_lhs->src1_kind = FBX_IR_KIND_GREG;
  get_lhs->src1 = lhs_greg;
  get_rhs = CtxEmit(ctx);
  if (!get_rhs) return 0;
  rhs_vreg = CtxAllocVreg(ctx);
  get_rhs->opcode = FBX_IR_OP_REG_GET;
  get_rhs->width = width;
  get_rhs->dst_kind = FBX_IR_KIND_VREG;
  get_rhs->dst = rhs_vreg;
  get_rhs->src1_kind = FBX_IR_KIND_GREG;
  get_rhs->src1 = rhs_greg;
  alu = CtxEmit(ctx);
  if (!alu) return 0;
  res_vreg = (op == FBX_IR_OP_CMP || op == FBX_IR_OP_TEST)
                 ? 0
                 : CtxAllocVreg(ctx);
  alu->opcode = op;
  alu->width = width;
  alu->dst_kind = (op == FBX_IR_OP_CMP || op == FBX_IR_OP_TEST)
                      ? FBX_IR_KIND_NONE
                      : FBX_IR_KIND_VREG;
  alu->dst = res_vreg;
  alu->src1_kind = FBX_IR_KIND_VREG;
  alu->src1 = lhs_vreg;
  alu->src2_kind = FBX_IR_KIND_VREG;
  alu->src2 = rhs_vreg;
  if (op != FBX_IR_OP_CMP && op != FBX_IR_OP_TEST) {
    set_res = CtxEmit(ctx);
    if (!set_res) return 0;
    set_res->opcode = FBX_IR_OP_REG_SET;
    set_res->width = width;
    set_res->dst_kind = FBX_IR_KIND_GREG;
    set_res->dst = dst_greg;
    set_res->src1_kind = FBX_IR_KIND_VREG;
    set_res->src1 = res_vreg;
  }
  flags = CtxEmit(ctx);
  if (!flags) return 0;
  flags->opcode = FBX_IR_OP_SET_FLAGS_RAW;
  flags->width = width;
  flags->imm = op;
  /* #599 (FBX_IR_VERSION 2): encode operand vregs so the emit pass can
   * synthesize the lazy-flag wasm without walking back through IR history.
   * See LiftAlui for the encoding rationale + the version-bump comment in
   * blink/fbx_ir.h. */
  flags->src1_kind = FBX_IR_KIND_VREG;
  flags->src1 = lhs_vreg;
  flags->src2_kind = FBX_IR_KIND_VREG;
  flags->src2 = rhs_vreg;
  return 1;
}

/* Lift MOVZX r, r/m {byte,word} — zero-extending move (mopcode 0x1B6 =
 * MOVZX r, r/m8; mopcode 0x1B7 = MOVZX r, r/m16).
 *
 * Register form: REG_GET (narrow width) + REG_SET (wide width).  Memory
 * form (firebox#JJJ): LOAD (narrow width) + REG_SET (wide width), through
 * LiftLoadToReg.  The synthesis
 * pass's existing i64.load{8,16}_u path is unsigned — the implicit zero-
 * extension of the wasm load matches MOVZX semantics exactly.  No new IR
 * opcode and no new emit-side machinery needed.
 *
 * `src_width` is 1 or 2 (byte vs word source).  Destination width comes
 * from REX.W / operand-size prefix per the wider operand (Word vs Dword vs
 * Qword), determined by WidthFromRde(byte_op=0). */
/* firebox#T3R3 — the r/m8 register source of MOVZX/MOVSX into a vreg,
 * zero-extended.  AH/CH/DH/BH (no REX, rm 4..7) is bits 8..15 of register
 * rm & 3: a REG_GET at width 2 and an SHR by 8 (regions only; the operand
 * shape check refuses a high-byte source anywhere else). */
static int LiftByteSourceRegion(struct LiftCtx *ctx, u64 rde, u32 *out) {
  u32 w, c, z;
  struct FbxIrInst *sh;
  if (!LiftIsHighByteReg(rde, (u32)ModrmRm(rde))) {
    return LiftGetReg(ctx, 1, (u32)RexbRm(rde), out) ? 1 : 0;
  }
  if (!ctx->region) return -1;
  if (!LiftGetReg(ctx, 2, (u32)ModrmRm(rde) & 3u, &w)) return 0;
  if (!LiftGetImm(ctx, 1, 8, &c)) return 0;
  sh = CtxEmit(ctx);
  if (!sh) return 0;
  z = CtxAllocVreg(ctx);
  sh->opcode = FBX_IR_OP_SHR;
  sh->width = 4;
  sh->dst_kind = FBX_IR_KIND_VREG;
  sh->dst = z;
  sh->src1_kind = FBX_IR_KIND_VREG;
  sh->src1 = w;
  sh->src2_kind = FBX_IR_KIND_VREG;
  sh->src2 = c;
  *out = z;
  return 1;
}

/* firebox#T3R3 — CMOVcc r, r (0F 40..4F), regions only.  blink/machine.c
 * OpCmovImpl: the destination is written either way (WriteRegister: a
 * 4-byte write zero-extends even when the condition fails).  The memory
 * form, and P/NP (no region predicate for PF), stay with Tier 1. */
static int LiftCmovRegion(struct LiftCtx *ctx, u64 rde, u64 mop) {
  u8 width = WidthFromRde(rde, 0);
  u32 src, old, z;
  struct FbxIrInst *cm, *set;
  if (!ctx->region || !IsModrmRegister(rde)) return -1;
  if ((mop & 0xEu) == 0xAu) return -1;
  if (!LiftGetReg(ctx, width, (u32)RexbRm(rde), &src)) return 0;
  if (!LiftGetReg(ctx, width, (u32)RexrReg(rde), &old)) return 0;
  cm = CtxEmit(ctx);
  if (!cm) return 0;
  z = CtxAllocVreg(ctx);
  cm->opcode = FBX_IR_OP_CMOV;
  cm->width = width;
  cm->dst_kind = FBX_IR_KIND_VREG;
  cm->dst = z;
  cm->src1_kind = FBX_IR_KIND_VREG;
  cm->src1 = src;
  cm->src2_kind = FBX_IR_KIND_VREG;
  cm->src2 = old;
  cm->imm = mop & 0xFu;
  set = CtxEmit(ctx);
  if (!set) return 0;
  set->opcode = FBX_IR_OP_REG_SET;
  set->width = width;
  set->dst_kind = FBX_IR_KIND_GREG;
  set->dst = (u32)RexrReg(rde);
  set->src1_kind = FBX_IR_KIND_VREG;
  set->src1 = z;
  return 1;
}

/* firebox#T3R3 — sign extension, regions only:
 *   0x063 MOVSXD r64, r/m32 (REX.W)      0x1BE MOVSX r, r/m8
 *   0x1BF MOVSX r, r/m16                 0x098 CWDE (EAX = sext AX) and
 *                                              CDQE (RAX = sext EAX)
 * The source (REG_GET, a narrow LOAD, or a high-byte extract), a SEXT to
 * the destination width, a REG_SET. */
static int LiftSextRegion(struct LiftCtx *ctx, u64 rde, i64 disp, u64 mop) {
  u8 dw = WidthFromRde(rde, 0), sw;
  u32 dst, v, z;
  int rc;
  struct FbxIrInst *sx, *set;
  if (!ctx->region || (dw != 4 && dw != 8)) return -1;
  switch (mop) {
    case 0x063: sw = 4; break;
    case 0x1BE: sw = 1; break;
    case 0x1BF: sw = 2; break;
    case 0x098: sw = dw / 2u; break;
    default: return -1;
  }
  if (mop == 0x063 && dw != 8) return -1;
  if (mop == 0x098) {
    dst = 0; /* rAX */
    if (!LiftGetReg(ctx, sw, 0, &v)) return 0;
  } else {
    dst = (u32)RexrReg(rde);
    if (!IsModrmRegister(rde)) {
      rc = LiftLoadOperand(ctx, rde, disp, sw, &v);
      if (rc <= 0) return rc;
    } else if (sw == 1) {
      rc = LiftByteSourceRegion(ctx, rde, &v);
      if (rc <= 0) return rc;
    } else if (!LiftGetReg(ctx, sw, (u32)RexbRm(rde), &v)) {
      return 0;
    }
  }
  sx = CtxEmit(ctx);
  if (!sx) return 0;
  z = CtxAllocVreg(ctx);
  sx->opcode = FBX_IR_OP_SEXT;
  sx->width = dw;
  sx->dst_kind = FBX_IR_KIND_VREG;
  sx->dst = z;
  sx->src1_kind = FBX_IR_KIND_VREG;
  sx->src1 = v;
  sx->imm = sw;
  set = CtxEmit(ctx);
  if (!set) return 0;
  set->opcode = FBX_IR_OP_REG_SET;
  set->width = dw;
  set->dst_kind = FBX_IR_KIND_GREG;
  set->dst = dst;
  set->src1_kind = FBX_IR_KIND_VREG;
  set->src1 = z;
  return 1;
}

static int LiftMovzx(struct LiftCtx *ctx, u64 rde, i64 disp, u8 src_width) {
  u8 dst_width = WidthFromRde(rde, 0);
  u32 src_vreg;
  struct FbxIrInst *get;
  struct FbxIrInst *set;
  /* Defensive: dst must be at least as wide as src.  WidthFromRde with
   * byte_op=0 always returns 2/4/8 (depending on prefix/REX.W); src_width
   * passed in is 1 (for MOVZX r/m8) or 2 (for MOVZX r/m16).  Refuse the
   * pathological case where source is wider than destination — the IR
   * wouldn't be well-defined and a real compiler-emitted x86 wouldn't
   * produce that encoding. */
  if (src_width > dst_width) return -1;
  /* firebox#JJJ: the memory form, `movzx r, byte/word [mem]`, is a narrow
   * zero-extending LOAD.  It is the hottest Tier-2 refusal on rg (the DFA
   * transition loop reads the haystack byte and the byte-class table with
   * it).  `disp` is the decoded displacement. */
  if (!IsModrmRegister(rde)) {
    return LiftLoadToReg(ctx, rde, disp, src_width, dst_width);
  }
  if (src_width == 1 && LiftIsHighByteReg(rde, (u32)ModrmRm(rde))) {
    /* firebox#T3R3: MOVZX from AH/CH/DH/BH, regions only. */
    int rc = LiftByteSourceRegion(ctx, rde, &src_vreg);
    if (rc <= 0) return rc;
    set = CtxEmit(ctx);
    if (!set) return 0;
    set->opcode = FBX_IR_OP_REG_SET;
    set->width = dst_width;
    set->dst_kind = FBX_IR_KIND_GREG;
    set->dst = (u32)RexrReg(rde);
    set->src1_kind = FBX_IR_KIND_VREG;
    set->src1 = src_vreg;
    return 1;
  }
  get = CtxEmit(ctx);
  if (!get) return 0;
  src_vreg = CtxAllocVreg(ctx);
  get->opcode = FBX_IR_OP_REG_GET;
  get->width = src_width;
  get->dst_kind = FBX_IR_KIND_VREG;
  get->dst = src_vreg;
  get->src1_kind = FBX_IR_KIND_GREG;
  get->src1 = (u32)RexbRm(rde);
  set = CtxEmit(ctx);
  if (!set) return 0;
  set->opcode = FBX_IR_OP_REG_SET;
  set->width = dst_width;
  set->dst_kind = FBX_IR_KIND_GREG;
  set->dst = (u32)RexrReg(rde);
  set->src1_kind = FBX_IR_KIND_VREG;
  set->src1 = src_vreg;
  return 1;
}

/* Lift conditional jump (opcodes 0x70-0x7F short, 0x180-0x18F long).
 * `taken_pc` and `fallthrough_pc` come from the lifting pass walker.
 * `cond_flag_id` encodes which condition (low byte of mopcode). */
static int LiftJcc(struct LiftCtx *ctx, u64 mopcode, u64 taken_pc,
                   u64 fallthrough_pc) {
  struct FbxIrInst *br = CtxEmit(ctx);
  if (!br) return 0;
  br->opcode = FBX_IR_OP_BRANCH_COND;
  br->width = 0;
  br->src1_kind = FBX_IR_KIND_IMM;
  br->src1 = (u32)(mopcode & 0xFu); /* low nibble = Jcc condition code */
  br->src2_kind = FBX_IR_KIND_IMM;
  br->src2 = (u32)(fallthrough_pc & 0xFFFFFFFFu);
  br->imm = taken_pc;
  return 1;
}

/* Lift CALL rel32 (opcode 0xE8). */
static int LiftCallJvds(struct LiftCtx *ctx, u64 target_pc, u64 return_pc) {
  struct FbxIrInst *c = CtxEmit(ctx);
  if (!c) return 0;
  c->opcode = FBX_IR_OP_CALL_DIRECT;
  c->src2_kind = FBX_IR_KIND_IMM;
  c->src2 = (u32)(return_pc & 0xFFFFFFFFu);
  c->imm = target_pc;
  ctx->flags |= FBX_IR_BLOCK_FLAG_HAS_CALL;
  return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Top-level lifter.                                                          */
/* ────────────────────────────────────────────────────────────────────────── */

/* Returns:
 *    >0  : opcode lifted; continue (block may end if a branch/call/ret/etc.
 *          was lifted)
 *    0   : alloc failure
 *   -1   : unsupported opcode; caller emits BAILOUT
 */
static int LiftOneOp(struct LiftCtx *ctx, const struct FbxTcEntry *e,
                     u64 next_pc);

static int LiftOne(struct LiftCtx *ctx, const struct FbxTcEntry *e,
                   u64 next_pc) {
  u64 rde = e->rde;
  u64 mop = Mopcode(rde);
  int roles, rc;
  /* firebox#RAT — refuse the operand shapes the IR cannot express BEFORE any
   * lifter reads a register number or an addressing form out of `rde`. */
  if (LiftIsVexEncoded(e)) return -1;
  if (LiftOperandShapeUnmodelled(rde, mop, ctx->region)) return -1;
  /* firebox#T3R3: a region lift carries a memory operand's segment override
   * into the IR.  Only LiftClassifyAddr applies it, so an instruction whose
   * lifter computed its address any other way is refused here. */
  roles = LiftOperandRoles(mop);
  ctx->seg = 0;
  ctx->seg_used = 0;
  if (ctx->region && (roles & FBX_LIFT_HAS_MODRM) &&
      !(roles & FBX_LIFT_NO_SEG) && !IsModrmRegister(rde)) {
    ctx->seg = (u8)Sego(rde);
  }
  rc = LiftOneOp(ctx, e, next_pc);
  if (rc > 0 && ctx->seg && !ctx->seg_used) return -1;
  return rc;
}

static int LiftOneOp(struct LiftCtx *ctx, const struct FbxTcEntry *e,
                     u64 next_pc) {
  u64 rde = e->rde;
  u64 mop = Mopcode(rde);
  switch (mop) {
    /* MOV r/m, r — opcode 0x88 (byte) and 0x89 (word).  #677 routes the
     * memory-store form (mod != 3) through LiftMovStoreMem; the call may
     * return -1 (unsupported addressing) which LiftOne maps to BAILOUT. */
    case 0x088: return LiftMovEvqpGvqp(ctx, rde, e->disp, 1);
    case 0x089: return LiftMovEvqpGvqp(ctx, rde, e->disp, 0);
    /* MOV r, r/m — opcode 0x8A (byte) and 0x8B (word).  #677 routes the
     * memory-load form (mod != 3) through LiftMovLoadMem. */
    case 0x08A: return LiftMovGvqpEvqp(ctx, rde, e->disp, 1);
    case 0x08B: return LiftMovGvqpEvqp(ctx, rde, e->disp, 0);
    /* LEA — opcode 0x8D. */
    case 0x08D: return LiftLea(ctx, rde, e->disp);
    /* MOV reg, imm — opcodes 0xB8-0xBF. */
    case 0x0B8: case 0x0B9: case 0x0BA: case 0x0BB:
    case 0x0BC: case 0x0BD: case 0x0BE: case 0x0BF:
      return LiftMovZvqpIvqp(ctx, rde, e->uimm0);
    /* MOV r/m, imm — opcode 0xC6 (byte) / 0xC7 (word). */
    case 0x0C6: return LiftMovImm(ctx, rde, e->disp, e->uimm0, 1);
    case 0x0C7: return LiftMovImm(ctx, rde, e->disp, e->uimm0, 0);
    /* ALU group r/m, r — ADD/OR/AND/SUB/XOR/CMP at 0x00/01, 0x08/09, etc.
     * Operand-direction = r/m is destination (op_dir=RM_R). */
    case 0x000: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_ADD, 1, FBX_ALU_DIR_RM_R);
    case 0x001: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_ADD, 0, FBX_ALU_DIR_RM_R);
    case 0x008: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_OR,  1, FBX_ALU_DIR_RM_R);
    case 0x009: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_OR,  0, FBX_ALU_DIR_RM_R);
    case 0x020: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_AND, 1, FBX_ALU_DIR_RM_R);
    case 0x021: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_AND, 0, FBX_ALU_DIR_RM_R);
    case 0x028: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_SUB, 1, FBX_ALU_DIR_RM_R);
    case 0x029: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_SUB, 0, FBX_ALU_DIR_RM_R);
    case 0x030: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_XOR, 1, FBX_ALU_DIR_RM_R);
    case 0x031: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_XOR, 0, FBX_ALU_DIR_RM_R);
    case 0x038: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_CMP, 1, FBX_ALU_DIR_RM_R);
    case 0x039: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_CMP, 0, FBX_ALU_DIR_RM_R);
    /* §13.5 mirror-direction ALU group r, r/m — same ops, swapped operand
     * direction (r is destination, r/m is the second source).  Mopcodes
     * 0x02/0x03 (ADD), 0x0A/0x0B (OR), 0x22/0x23 (AND), 0x2A/0x2B (SUB),
     * 0x32/0x33 (XOR), 0x3A/0x3B (CMP).  These show up in the §Q7
     * ranking (top-30: 0x003 at rank 23, 0x03B at rank 19, 0x02B at rank
     * 43, etc.) — lifting them collapses ~4-5% of total host time from
     * "lift-side bailout" to "synthesis-side refused" (the BAILOUT slot
     * was preventing escalation entirely; refusing-after-lift lets
     * Phase 4 reachability still log the IR shape, and lets #597's
     * lazy-flag work pick them up automatically when it lands). */
    case 0x002: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_ADD, 1, FBX_ALU_DIR_R_RM);
    case 0x003: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_ADD, 0, FBX_ALU_DIR_R_RM);
    case 0x00A: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_OR,  1, FBX_ALU_DIR_R_RM);
    case 0x00B: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_OR,  0, FBX_ALU_DIR_R_RM);
    case 0x022: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_AND, 1, FBX_ALU_DIR_R_RM);
    case 0x023: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_AND, 0, FBX_ALU_DIR_R_RM);
    case 0x02A: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_SUB, 1, FBX_ALU_DIR_R_RM);
    case 0x02B: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_SUB, 0, FBX_ALU_DIR_R_RM);
    case 0x032: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_XOR, 1, FBX_ALU_DIR_R_RM);
    case 0x033: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_XOR, 0, FBX_ALU_DIR_R_RM);
    case 0x03A: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_CMP, 1, FBX_ALU_DIR_R_RM);
    case 0x03B: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_CMP, 0, FBX_ALU_DIR_R_RM);
    /* TEST r/m, r — 0x84 / 0x85.  No mirror variant in x86 (TEST is
     * symmetric in its operand encoding). */
    case 0x084: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_TEST, 1, FBX_ALU_DIR_RM_R);
    case 0x085: return LiftAluRR(ctx, rde, e->disp, FBX_IR_OP_TEST, 0, FBX_ALU_DIR_RM_R);
    /* §13.5 MOVZX — zero-extending move.  Top-30 rank 15 (0x1B6) +
     * rank 34 (0x1B7) per §Q7 measurement.  IR shape matches existing
     * REG_GET (narrow) + REG_SET (wide); the synthesis pass's
     * i64.load{8,16}_u is unsigned, matching MOVZX semantics. */
    case 0x1B6: return LiftMovzx(ctx, rde, e->disp, 1);
    case 0x1B7: return LiftMovzx(ctx, rde, e->disp, 2);
    case 0x063: case 0x098: case 0x1BE: case 0x1BF: /* firebox#T3R3 */
      return LiftSextRegion(ctx, rde, e->disp, mop);
    case 0x140: case 0x141: case 0x142: case 0x143: /* firebox#T3R3 */
    case 0x144: case 0x145: case 0x146: case 0x147:
    case 0x148: case 0x149: case 0x14A: case 0x14B:
    case 0x14C: case 0x14D: case 0x14E: case 0x14F:
      return LiftCmovRegion(ctx, rde, mop);
    /* ALU group r/m, imm — 0x80 (byte imm), 0x81 (imm32), 0x83 (imm8 sx). */
    case 0x080: return LiftAlui(ctx, rde, e->disp, e->uimm0, 1);
    case 0x081:
    case 0x083: return LiftAlui(ctx, rde, e->disp, e->uimm0, 0);
    /* #794 — IMUL truncating forms.  3-operand imm (0x69 imm32-sx / 0x6B
     * imm8-sx) → r = r/m * imm; 2-operand (0x0FAF → mopcode 0x1AF) → r =
     * r * r/m.  No flag synthesis (correct-or-refuse at the emit gate). */
    case 0x069:
    case 0x06B: return LiftImul(ctx, rde, /*has_imm=*/1, e->uimm0);
    case 0x1AF: return LiftImul(ctx, rde, /*has_imm=*/0, 0);
    /* #794 — accumulator-immediate ALU forms (no modrm; dst/lhs = rAX).
     * The assembler emits these short encodings for `add rax,imm` etc.
     * Byte form (0x04/0x0C/0x24/0x2C/0x34/0x3C) → byte_op=1; word form
     * (0x05/0x0D/0x25/0x2D/0x35/0x3D) → byte_op=0.  ADC/SBB (0x14/0x15/
     * 0x1C/0x1D) intentionally omitted — same as AluSubOp's exclusion. */
    case 0x004: return LiftAluAccImm(ctx, rde, e->uimm0, FBX_IR_OP_ADD, 1);
    case 0x005: return LiftAluAccImm(ctx, rde, e->uimm0, FBX_IR_OP_ADD, 0);
    case 0x00C: return LiftAluAccImm(ctx, rde, e->uimm0, FBX_IR_OP_OR,  1);
    case 0x00D: return LiftAluAccImm(ctx, rde, e->uimm0, FBX_IR_OP_OR,  0);
    case 0x024: return LiftAluAccImm(ctx, rde, e->uimm0, FBX_IR_OP_AND, 1);
    case 0x025: return LiftAluAccImm(ctx, rde, e->uimm0, FBX_IR_OP_AND, 0);
    case 0x02C: return LiftAluAccImm(ctx, rde, e->uimm0, FBX_IR_OP_SUB, 1);
    case 0x02D: return LiftAluAccImm(ctx, rde, e->uimm0, FBX_IR_OP_SUB, 0);
    case 0x034: return LiftAluAccImm(ctx, rde, e->uimm0, FBX_IR_OP_XOR, 1);
    case 0x035: return LiftAluAccImm(ctx, rde, e->uimm0, FBX_IR_OP_XOR, 0);
    case 0x03C: return LiftAluAccImm(ctx, rde, e->uimm0, FBX_IR_OP_CMP, 1);
    case 0x03D: return LiftAluAccImm(ctx, rde, e->uimm0, FBX_IR_OP_CMP, 0);
    /* Conditional jumps — short (0x70-0x7F) and near (0x180-0x18F). */
    case 0x070: case 0x071: case 0x072: case 0x073:
    case 0x074: case 0x075: case 0x076: case 0x077:
    case 0x078: case 0x079: case 0x07A: case 0x07B:
    case 0x07C: case 0x07D: case 0x07E: case 0x07F:
    case 0x180: case 0x181: case 0x182: case 0x183:
    case 0x184: case 0x185: case 0x186: case 0x187:
    case 0x188: case 0x189: case 0x18A: case 0x18B:
    case 0x18C: case 0x18D: case 0x18E: case 0x18F:
      return LiftJcc(ctx, mop, (u64)((i64)next_pc + e->disp), next_pc);
    /* CALL rel32 — 0xE8. */
    case 0x0E8:
      return LiftCallJvds(ctx, (u64)((i64)next_pc + e->disp), next_pc);
    /* #602 — PUSH reg (0x050-0x057, low 3 bits = Srm reg id; REX.B
     * prefix extends to 0x58-0x5F greg ids via the high bit at
     * kRegRexbSrmMask bit 15).
     *
     * Source of truth for the register id: x86 encodes the register
     * in the LOW 3 BITS OF THE OPCODE itself (mop & 7), plus REX.B (1
     * bit) if a REX prefix is present.  Blink's xed decoder mirrors
     * this into rde's Srm/RexbSrm fields, but the mopcode low nibble
     * is the authoritative source.  Using `(u32)(mop & 0x7)` here
     * (instead of RexbSrm(rde)) makes the lifter robust against test
     * fixtures that don't populate rde's Srm bits explicitly — the
     * mopcode low nibble carries the same information by definition.
     *
     * REX.B-extended registers (r8-r15 for PUSH/POP) come in via rde's
     * top bit of kRegRexbSrmMask (bit 15), which `RexbSrm(rde) & 0x8`
     * captures.  We OR them together for completeness.
     *
     * Width = 8 (x86-64 default operand size; the v0.1 emitter only
     * synthesizes the 8-byte form, matching the §Q7 top-30 ranking
     * where 0x055 = PUSH RBP appears as rank 16). */
    case 0x050: case 0x051: case 0x052: case 0x053:
    case 0x054: case 0x055: case 0x056: case 0x057: {
      struct FbxIrInst *p = CtxEmit(ctx);
      if (!p) return 0;
      p->opcode = FBX_IR_OP_PUSH;
      p->width = 8;
      p->src1_kind = FBX_IR_KIND_GREG;
      p->src1 = (u32)((mop & 0x7) | (RexbSrm(rde) & 0x8));
      return 1;
    }
    /* #602 — POP reg (0x058-0x05F).  Inverse of PUSH; greg[reg_id] = *RSP
     * then RSP += 8.  Same width semantics as PUSH.  Same reg-id
     * extraction strategy: mop low 3 bits OR'd with REX.B from rde. */
    case 0x058: case 0x059: case 0x05A: case 0x05B:
    case 0x05C: case 0x05D: case 0x05E: case 0x05F: {
      struct FbxIrInst *p = CtxEmit(ctx);
      if (!p) return 0;
      p->opcode = FBX_IR_OP_POP;
      p->width = 8;
      p->dst_kind = FBX_IR_KIND_GREG;
      p->dst = (u32)((mop & 0x7) | (RexbSrm(rde) & 0x8));
      return 1;
    }
    /* RET — 0xC3. */
    case 0x0C3: {
      struct FbxIrInst *r = CtxEmit(ctx);
      if (!r) return 0;
      r->opcode = FBX_IR_OP_RET;
      return 1;
    }
    /* Unconditional jumps — 0xE9 (rel32) / 0xEB (rel8). */
    case 0x0E9:
    case 0x0EB: {
      struct FbxIrInst *br = CtxEmit(ctx);
      if (!br) return 0;
      br->opcode = FBX_IR_OP_BRANCH_TAKEN;
      br->imm = (u64)((i64)next_pc + e->disp);
      return 1;
    }
    /* firebox#T3R2 — region lifts only (each returns -1 outside a region). */
    case 0x0F6: case 0x0F7: case 0x0FE: case 0x0FF:
      return LiftUnaryRegion(ctx, rde, e->disp, e->uimm0, mop);
    case 0x0C1: case 0x0D1: case 0x0D3: /* firebox#T3R3 */
      return LiftShiftRegion(ctx, rde, e->uimm0, mop);
    case 0x0A8: case 0x0A9: {
      u32 x, k;
      u8 w = WidthFromRde(rde, mop == 0x0A8);
      if (!ctx->region) return -1;
      if (!LiftGetReg(ctx, w, 0u, &x)) return 0;
      if (!LiftGetImm(ctx, w, e->uimm0, &k)) return 0;
      return LiftAluTail(ctx, w, FBX_IR_OP_TEST, x, k, 0, FBX_IR_OP_TEST, x,
                         k);
    }
    default:
      return -1;
  }
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Public API.                                                                */
/* ────────────────────────────────────────────────────────────────────────── */

static struct FbxIrBlock *LiftBlock(const struct FbxTcBlock *tc, int region);

struct FbxIrBlock *fbx_ir_lift(const struct FbxTcBlock *tc) {
  return LiftBlock(tc, 0);
}

struct FbxIrBlock *fbx_ir_lift_region(const struct FbxTcBlock *tc) {
  return LiftBlock(tc, 1);
}

static struct FbxIrBlock *LiftBlock(const struct FbxTcBlock *tc, int region) {
  struct LiftCtx ctx;
  u32 mark;
  struct FbxIrBlock *blk;
  u32 i;
  /* firebox#CM4 — the PC after the last op we successfully lifted; the target
   * of the synthetic fall-through terminator if the block ends unterminated. */
  u64 fallthrough_pc = 0;
  if (!tc || tc->nentries == 0) return NULL;
  /* Thunk blocks stay on Tier 1 — refuse to lift.  Spec §6.5. */
  if (tc->entries[0].kind == FBX_TC_KIND_THUNK) return NULL;
  memset(&ctx, 0, sizeof ctx);
  ctx.region = region;
  for (i = 0; i < tc->nentries; ++i) {
    const struct FbxTcEntry *e = &tc->entries[i];
    u64 next_pc = e->ip + e->oplen;
    int rc;
    if (e->kind == FBX_TC_KIND_THUNK) {
      /* Mid-block thunk shouldn't happen (Tier 1 ends the block at the thunk),
       * but be defensive: bail out at the thunk's PC. */
      if (!EmitBailout(&ctx, e->ip, FBX_IR_BAILOUT_THUNK)) goto fail;
      break;
    }
    /* Refuse zero-length entries. */
    if (e->oplen == 0) {
      if (!EmitBailout(&ctx, e->ip, FBX_IR_BAILOUT_DECODE_ERROR)) goto fail;
      break;
    }
    /* Precious / serializing ops bail out before the op fires. */
    {
      u64 mop = Mopcode(e->rde);
      /* Syscall / interrupt. */
      if (mop == 0x105 /* OpSyscall */ || mop == 0x0F1 /* int1 */ ||
          mop == 0x0CC /* int3 */ || mop == 0x0CD /* int n */) {
        if (!EmitPcMark(&ctx, e->ip)) goto fail;
        if (!EmitBailout(&ctx, e->ip, FBX_IR_BAILOUT_SYSCALL)) goto fail;
        break;
      }
      /* Serializing — cpuid / wrmsr / mfence / lfence. */
      if (mop == 0x130 /* wrmsr */ || mop == 0x1A2 /* cpuid */ ||
          mop == 0x1AE /* lfence/mfence group */) {
        if (!EmitPcMark(&ctx, e->ip)) goto fail;
        if (!EmitBailout(&ctx, e->ip, FBX_IR_BAILOUT_SERIALIZING)) goto fail;
        ctx.flags |= FBX_IR_BLOCK_FLAG_HAS_MEM_BARRIER;
        break;
      }
    }
    if (!EmitPcMark(&ctx, e->ip)) goto fail;
    mark = ctx.ninsts;
    ctx.next_pc = next_pc;
    /* firebox#T3R2: the region lowerings assume 64-bit mode (stack width,
     * RIP-relative addressing); anything else stays on Tier 1. */
    if (region && Mode(e->rde) != XED_MODE_LONG) {
      rc = -1;
    } else {
      rc = LiftOne(&ctx, e, next_pc);
    }
    if (rc == 0) goto fail; /* alloc */
    if (rc < 0) {
      /* Unsupported opcode — emit BAILOUT and stop.  firebox#T3R2: drop
       * anything the lifter emitted for this instruction before refusing
       * it, so the bailout follows only whole instructions. */
      ctx.ninsts = mark;
      if (!EmitBailout(&ctx, e->ip, FBX_IR_BAILOUT_UNSUPPORTED)) goto fail;
      break;
    }
    /* firebox#CM4 — record the fall-through PC (after this successfully-lifted,
     * non-terminating op) in case the block ends here without a terminator. */
    fallthrough_pc = next_pc;
  }
  /* firebox#CM4 — a Tier-1 block whose entries end WITHOUT a control-flow op
   * (a fall-through — blink split the run mid-stream, e.g. `mov rdi,rax` as its
   * own block) lifts to IR with NO terminator.  The emit's un-terminated
   * fallback (fbx_ir_emit_wasm.c EmitFunctionBody) then returns exit=0 WITHOUT
   * advancing m->ip, so the dispatcher re-enters the SAME block at the
   * unchanged m->ip forever (100%-CPU guest-wasm wedge; blocked every elf-perf
   * th=1 bench).  Restore the "every lifted block ends with a terminator"
   * contract: append an explicit unconditional jump to the fall-through PC so
   * the T2 block advances m->ip to end_pc and control continues to the next
   * block — byte-for-byte the Tier-1 fall-through behaviour. */
  if (ctx.ninsts > 0 &&
      !LiftOpIsTerminator(ctx.insts[ctx.ninsts - 1].opcode)) {
    if (!EmitFallthroughBranch(&ctx, fallthrough_pc)) goto fail;
  }
  /* Allocate the FbxIrBlock + a tightly-sized inst array. */
  blk = (struct FbxIrBlock *)calloc(1, sizeof *blk);
  if (!blk) goto fail;
  if (ctx.ninsts > 0) {
    blk->insts = (struct FbxIrInst *)malloc(ctx.ninsts *
                                            sizeof(struct FbxIrInst));
    if (!blk->insts) {
      free(blk);
      goto fail;
    }
    memcpy(blk->insts, ctx.insts, ctx.ninsts * sizeof(struct FbxIrInst));
  }
  blk->ninsts = ctx.ninsts;
  blk->nvregs = (u16)((ctx.next_vreg > 0xFFFFu) ? 0xFFFFu : ctx.next_vreg);
  blk->flags = ctx.flags;
  blk->start_pc = tc->start_pc;
  blk->end_pc = tc->end_pc;
  if (blk->flags & FBX_IR_BLOCK_FLAG_HAS_BAILOUT) {
    /* A bailout-terminated block can still be partially translated; only
     * mark NON_TRANSLATABLE if the first instruction was the bailout (no
     * translatable prefix).  insts[0] is PC_MARK if any op lifted; check
     * insts[1] for the BAILOUT-only case. */
    if (blk->ninsts <= 2 &&
        (blk->ninsts == 0 ||
         blk->insts[blk->ninsts - 1].opcode == FBX_IR_OP_BAILOUT)) {
      blk->flags |= FBX_IR_BLOCK_FLAG_NON_TRANSLATABLE;
    }
  }
  fbx_ir_block_sha256(blk, blk->cache_key_sha256);
  free(ctx.insts);
  return blk;
fail:
  free(ctx.insts);
  return NULL;
}

void fbx_ir_free(struct FbxIrBlock *ir) {
  if (!ir) return;
  free(ir->insts);
  free(ir);
}

void fbx_ir_block_sha256(const struct FbxIrBlock *ir, u8 out_sha256[32]) {
  struct Sha256Ctx c;
  u32 version_le;
  u8 hdr[8 + 8 + 4 + 2 + 2];
  size_t off = 0;
  Sha256Init(&c);
  version_le = (u32)FBX_IR_VERSION;
  /* version (u32 little-endian). */
  {
    u8 v[4];
    v[0] = (u8)(version_le >> 0);
    v[1] = (u8)(version_le >> 8);
    v[2] = (u8)(version_le >> 16);
    v[3] = (u8)(version_le >> 24);
    Sha256Update(&c, v, 4);
  }
  /* blink git sha — hex string from the build (default 64-char zero pad). */
  Sha256Update(&c, FBX_IR_BLINK_GIT_SHA, sizeof(FBX_IR_BLINK_GIT_SHA) - 1);
  /* block header — pack little-endian into a fixed buffer to keep
   * cross-arch deterministic. */
  hdr[off + 0] = (u8)(ir->start_pc >> 0);
  hdr[off + 1] = (u8)(ir->start_pc >> 8);
  hdr[off + 2] = (u8)(ir->start_pc >> 16);
  hdr[off + 3] = (u8)(ir->start_pc >> 24);
  hdr[off + 4] = (u8)(ir->start_pc >> 32);
  hdr[off + 5] = (u8)(ir->start_pc >> 40);
  hdr[off + 6] = (u8)(ir->start_pc >> 48);
  hdr[off + 7] = (u8)(ir->start_pc >> 56);
  off += 8;
  hdr[off + 0] = (u8)(ir->end_pc >> 0);
  hdr[off + 1] = (u8)(ir->end_pc >> 8);
  hdr[off + 2] = (u8)(ir->end_pc >> 16);
  hdr[off + 3] = (u8)(ir->end_pc >> 24);
  hdr[off + 4] = (u8)(ir->end_pc >> 32);
  hdr[off + 5] = (u8)(ir->end_pc >> 40);
  hdr[off + 6] = (u8)(ir->end_pc >> 48);
  hdr[off + 7] = (u8)(ir->end_pc >> 56);
  off += 8;
  hdr[off + 0] = (u8)(ir->ninsts >> 0);
  hdr[off + 1] = (u8)(ir->ninsts >> 8);
  hdr[off + 2] = (u8)(ir->ninsts >> 16);
  hdr[off + 3] = (u8)(ir->ninsts >> 24);
  off += 4;
  hdr[off + 0] = (u8)(ir->nvregs >> 0);
  hdr[off + 1] = (u8)(ir->nvregs >> 8);
  off += 2;
  hdr[off + 0] = (u8)(ir->flags >> 0);
  hdr[off + 1] = (u8)(ir->flags >> 8);
  off += 2;
  Sha256Update(&c, hdr, off);
  /* insts — bytes of the FbxIrInst array.  Padding-zeroed at emit time. */
  if (ir->ninsts > 0 && ir->insts) {
    /* Each FbxIrInst is hashed field-by-field in little-endian to remove
     * cross-arch struct-layout dependency for SHA-256 determinism (the
     * struct layout is locked by _Static_assert, but a paranoid serializer
     * decouples it from host endianness regardless). */
    u32 i;
    for (i = 0; i < ir->ninsts; ++i) {
      const struct FbxIrInst *p = &ir->insts[i];
      u8 b[FBX_IR_INST_SERIALISED_BYTES];
      size_t k = 0;
      b[k++] = p->opcode;
      b[k++] = p->width;
      b[k++] = p->dst_kind;
      b[k++] = p->src1_kind;
      b[k++] = p->src2_kind;
      b[k++] = p->seg;
      b[k++] = (u8)(p->_pad1 >> 0);
      b[k++] = (u8)(p->_pad1 >> 8);
      b[k++] = (u8)(p->dst >> 0);
      b[k++] = (u8)(p->dst >> 8);
      b[k++] = (u8)(p->dst >> 16);
      b[k++] = (u8)(p->dst >> 24);
      b[k++] = (u8)(p->src1 >> 0);
      b[k++] = (u8)(p->src1 >> 8);
      b[k++] = (u8)(p->src1 >> 16);
      b[k++] = (u8)(p->src1 >> 24);
      b[k++] = (u8)(p->src2 >> 0);
      b[k++] = (u8)(p->src2 >> 8);
      b[k++] = (u8)(p->src2 >> 16);
      b[k++] = (u8)(p->src2 >> 24);
      /* Offset-20 word — formerly `_pad2`, now the LEA SIB `scale` (#778).
       * Hashed in the same position; the FBX_IR_VERSION bump invalidates any
       * sidecar built before the field carried meaning. */
      b[k++] = (u8)(p->scale >> 0);
      b[k++] = (u8)(p->scale >> 8);
      b[k++] = (u8)(p->scale >> 16);
      b[k++] = (u8)(p->scale >> 24);
      b[k++] = (u8)(p->imm >> 0);
      b[k++] = (u8)(p->imm >> 8);
      b[k++] = (u8)(p->imm >> 16);
      b[k++] = (u8)(p->imm >> 24);
      b[k++] = (u8)(p->imm >> 32);
      b[k++] = (u8)(p->imm >> 40);
      b[k++] = (u8)(p->imm >> 48);
      b[k++] = (u8)(p->imm >> 56);
      b[k++] = (u8)(p->_pad3 >> 0);
      b[k++] = (u8)(p->_pad3 >> 8);
      b[k++] = (u8)(p->_pad3 >> 16);
      b[k++] = (u8)(p->_pad3 >> 24);
      b[k++] = (u8)(p->_pad3 >> 32);
      b[k++] = (u8)(p->_pad3 >> 40);
      b[k++] = (u8)(p->_pad3 >> 48);
      b[k++] = (u8)(p->_pad3 >> 56);
      Sha256Update(&c, b, k);
    }
  }
  Sha256Final(&c, out_sha256);
}
