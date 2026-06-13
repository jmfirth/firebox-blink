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

/* Lift MOV r/m, r in the memory-store direction (0x88/0x89, mod != 3):
 *   REG_GET (src reg → vreg) ; STORE (mem[base+disp] = vreg).
 * Returns 1 on success, 0 on alloc failure, -1 to request BAILOUT. */
static int LiftMovStoreMem(struct LiftCtx *ctx, u64 rde, i64 disp,
                           u8 width) {
  u32 base_greg;
  u32 val_vreg;
  struct FbxIrInst *get;
  struct FbxIrInst *st;
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
static int LiftMovLoadMem(struct LiftCtx *ctx, u64 rde, i64 disp,
                          u8 width) {
  u32 base_greg;
  u32 val_vreg;
  struct FbxIrInst *ld;
  struct FbxIrInst *set;
  if (!ClassifyMemBaseDisp(rde, &base_greg)) return -1;
  ld = CtxEmit(ctx);
  if (!ld) return 0;
  val_vreg = CtxAllocVreg(ctx);
  ld->opcode = FBX_IR_OP_LOAD;
  ld->width = width;
  ld->dst_kind = FBX_IR_KIND_VREG;
  ld->dst = val_vreg;
  ld->src1_kind = FBX_IR_KIND_GREG; /* base address register */
  ld->src1 = base_greg;
  ld->imm = (u64)disp; /* sign-extended displacement */
  set = CtxEmit(ctx);
  if (!set) return 0;
  set->opcode = FBX_IR_OP_REG_SET;
  set->width = width;
  set->dst_kind = FBX_IR_KIND_GREG;
  set->dst = (u32)RexrReg(rde);
  set->src1_kind = FBX_IR_KIND_VREG;
  set->src1 = val_vreg;
  return 1;
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

/* Lift MOV r/m, imm (opcode 0xC7: OpMovImm).  Approximate as REG_SET when
 * modrm.mod==3, leave §13.2 to handle the memory store case. */
static int LiftMovImm(struct LiftCtx *ctx, u64 rde, u64 uimm0) {
  u8 width = WidthFromRde(rde, 0);
  struct FbxIrInst *set = CtxEmit(ctx);
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
static int LiftAlui(struct LiftCtx *ctx, u64 rde, u64 uimm0, int byte_op) {
  u8 op = AluSubOp(rde);
  if (op == 0) return -1; /* ADC/SBB — bailout */
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

static int LiftAluRR(struct LiftCtx *ctx, u64 rde, u8 op, int byte_op,
                     int op_dir) {
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
 * mod3-only at v0.1 (see comment on the call site in LiftOne).  The IR
 * shape is REG_GET (narrow width) + REG_SET (wide width).  The synthesis
 * pass's existing i64.load{8,16}_u path is unsigned — the implicit zero-
 * extension of the wasm load matches MOVZX semantics exactly.  No new IR
 * opcode and no new emit-side machinery needed.
 *
 * `src_width` is 1 or 2 (byte vs word source).  Destination width comes
 * from REX.W / operand-size prefix per the wider operand (Word vs Dword vs
 * Qword), determined by WidthFromRde(byte_op=0). */
static int LiftMovzx(struct LiftCtx *ctx, u64 rde, u8 src_width) {
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
static int LiftOne(struct LiftCtx *ctx, const struct FbxTcEntry *e,
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
    case 0x0C6:
    case 0x0C7: return LiftMovImm(ctx, rde, e->uimm0);
    /* ALU group r/m, r — ADD/OR/AND/SUB/XOR/CMP at 0x00/01, 0x08/09, etc.
     * Operand-direction = r/m is destination (op_dir=RM_R). */
    case 0x000: return LiftAluRR(ctx, rde, FBX_IR_OP_ADD, 1, FBX_ALU_DIR_RM_R);
    case 0x001: return LiftAluRR(ctx, rde, FBX_IR_OP_ADD, 0, FBX_ALU_DIR_RM_R);
    case 0x008: return LiftAluRR(ctx, rde, FBX_IR_OP_OR,  1, FBX_ALU_DIR_RM_R);
    case 0x009: return LiftAluRR(ctx, rde, FBX_IR_OP_OR,  0, FBX_ALU_DIR_RM_R);
    case 0x020: return LiftAluRR(ctx, rde, FBX_IR_OP_AND, 1, FBX_ALU_DIR_RM_R);
    case 0x021: return LiftAluRR(ctx, rde, FBX_IR_OP_AND, 0, FBX_ALU_DIR_RM_R);
    case 0x028: return LiftAluRR(ctx, rde, FBX_IR_OP_SUB, 1, FBX_ALU_DIR_RM_R);
    case 0x029: return LiftAluRR(ctx, rde, FBX_IR_OP_SUB, 0, FBX_ALU_DIR_RM_R);
    case 0x030: return LiftAluRR(ctx, rde, FBX_IR_OP_XOR, 1, FBX_ALU_DIR_RM_R);
    case 0x031: return LiftAluRR(ctx, rde, FBX_IR_OP_XOR, 0, FBX_ALU_DIR_RM_R);
    case 0x038: return LiftAluRR(ctx, rde, FBX_IR_OP_CMP, 1, FBX_ALU_DIR_RM_R);
    case 0x039: return LiftAluRR(ctx, rde, FBX_IR_OP_CMP, 0, FBX_ALU_DIR_RM_R);
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
    case 0x002: return LiftAluRR(ctx, rde, FBX_IR_OP_ADD, 1, FBX_ALU_DIR_R_RM);
    case 0x003: return LiftAluRR(ctx, rde, FBX_IR_OP_ADD, 0, FBX_ALU_DIR_R_RM);
    case 0x00A: return LiftAluRR(ctx, rde, FBX_IR_OP_OR,  1, FBX_ALU_DIR_R_RM);
    case 0x00B: return LiftAluRR(ctx, rde, FBX_IR_OP_OR,  0, FBX_ALU_DIR_R_RM);
    case 0x022: return LiftAluRR(ctx, rde, FBX_IR_OP_AND, 1, FBX_ALU_DIR_R_RM);
    case 0x023: return LiftAluRR(ctx, rde, FBX_IR_OP_AND, 0, FBX_ALU_DIR_R_RM);
    case 0x02A: return LiftAluRR(ctx, rde, FBX_IR_OP_SUB, 1, FBX_ALU_DIR_R_RM);
    case 0x02B: return LiftAluRR(ctx, rde, FBX_IR_OP_SUB, 0, FBX_ALU_DIR_R_RM);
    case 0x032: return LiftAluRR(ctx, rde, FBX_IR_OP_XOR, 1, FBX_ALU_DIR_R_RM);
    case 0x033: return LiftAluRR(ctx, rde, FBX_IR_OP_XOR, 0, FBX_ALU_DIR_R_RM);
    case 0x03A: return LiftAluRR(ctx, rde, FBX_IR_OP_CMP, 1, FBX_ALU_DIR_R_RM);
    case 0x03B: return LiftAluRR(ctx, rde, FBX_IR_OP_CMP, 0, FBX_ALU_DIR_R_RM);
    /* TEST r/m, r — 0x84 / 0x85.  No mirror variant in x86 (TEST is
     * symmetric in its operand encoding). */
    case 0x084: return LiftAluRR(ctx, rde, FBX_IR_OP_TEST, 1, FBX_ALU_DIR_RM_R);
    case 0x085: return LiftAluRR(ctx, rde, FBX_IR_OP_TEST, 0, FBX_ALU_DIR_RM_R);
    /* §13.5 MOVZX — zero-extending move.  Top-30 rank 15 (0x1B6) +
     * rank 34 (0x1B7) per §Q7 measurement.  IR shape matches existing
     * REG_GET (narrow) + REG_SET (wide); the synthesis pass's
     * i64.load{8,16}_u is unsigned, matching MOVZX semantics. */
    case 0x1B6: return LiftMovzx(ctx, rde, 1);
    case 0x1B7: return LiftMovzx(ctx, rde, 2);
    /* ALU group r/m, imm — 0x80 (byte imm), 0x81 (imm32), 0x83 (imm8 sx). */
    case 0x080: return LiftAlui(ctx, rde, e->uimm0, 1);
    case 0x081:
    case 0x083: return LiftAlui(ctx, rde, e->uimm0, 0);
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
    default:
      return -1;
  }
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Public API.                                                                */
/* ────────────────────────────────────────────────────────────────────────── */

struct FbxIrBlock *fbx_ir_lift(const struct FbxTcBlock *tc) {
  struct LiftCtx ctx;
  struct FbxIrBlock *blk;
  u32 i;
  if (!tc || tc->nentries == 0) return NULL;
  /* Thunk blocks stay on Tier 1 — refuse to lift.  Spec §6.5. */
  if (tc->entries[0].kind == FBX_TC_KIND_THUNK) return NULL;
  memset(&ctx, 0, sizeof ctx);
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
    rc = LiftOne(&ctx, e, next_pc);
    if (rc == 0) goto fail; /* alloc */
    if (rc < 0) {
      /* Unsupported opcode — emit BAILOUT and stop. */
      if (!EmitBailout(&ctx, e->ip, FBX_IR_BAILOUT_UNSUPPORTED)) goto fail;
      break;
    }
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
      b[k++] = p->_pad0;
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
