/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2026 Justin Firth (Firebox)                                        │
│                                                                              │
│ Unit tests for the Phase-2 Tier-2 §13.1 IR data structures + lifting pass.   │
│ See blink/fbx_ir.h for the spec pointers + design header.                    │
│                                                                              │
│ Coverage:                                                                    │
│   - Wire-format invariants (FbxIrInst size, FBX_IR_VERSION presence)         │
│   - SHA-256 against FIPS 180-4 vectors                                       │
│   - Each top-10 opcode lifts deterministically                              │
│   - Bailout: thunk, syscall, serializing, unsupported                       │
│   - Determinism: same input → same SHA, same bytes                          │
│   - Version-bump invalidation: SHA changes when IR changes                  │
╚─────────────────────────────────────────────────────────────────────────────*/

#include "blink/fbx_ir.h"
#include "blink/rde.h"
#include "blink/threadedcode.h"
#include "blink/types.h"
#include "test/test.h"

#include <string.h>

/* The test harness drives main(); these are required no-ops. */
void SetUp(void) {}
void TearDown(void) {}

/* ────────────────────────────────────────────────────────────────────────── */
/* Wire-format invariants.                                                    */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxIrInst, Sizeof) {
  EXPECT_EQ(40, (i64)sizeof(struct FbxIrInst));
  EXPECT_EQ(40, (i64)FBX_IR_INST_SERIALISED_BYTES);
}

TEST(FbxIrInst, ZeroClearsPadding) {
  struct FbxIrInst a;
  unsigned char *bytes;
  unsigned i;
  /* Fill with sentinel garbage; fbx_ir_inst_zero must wipe it all. */
  memset(&a, 0xAB, sizeof a);
  fbx_ir_inst_zero(&a);
  bytes = (unsigned char *)&a;
  for (i = 0; i < sizeof a; ++i) {
    ASSERT_EQ(0, bytes[i]);
  }
}

/* ────────────────────────────────────────────────────────────────────────── */
/* SHA-256 vectors (FIPS 180-4 §B).                                           */
/* ────────────────────────────────────────────────────────────────────────── */

static void HexExpect(const u8 got[32], const char *expect_hex) {
  static const char kHex[] = "0123456789abcdef";
  char got_hex[65];
  unsigned i;
  for (i = 0; i < 32; ++i) {
    got_hex[i * 2 + 0] = kHex[got[i] >> 4];
    got_hex[i * 2 + 1] = kHex[got[i] & 0xF];
  }
  got_hex[64] = 0;
  EXPECT_STREQ(expect_hex, got_hex);
}

TEST(FbxSha256, EmptyString) {
  u8 out[32];
  fbx_sha256("", 0, out);
  HexExpect(out,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST(FbxSha256, Abc) {
  u8 out[32];
  fbx_sha256("abc", 3, out);
  HexExpect(out,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(FbxSha256, MultiBlock) {
  /* FIPS vector: 448-bit message that pads into a second block. */
  u8 out[32];
  fbx_sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 56,
             out);
  HexExpect(out,
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Lifting pass helpers — build synthetic FbxTcBlock inputs.                  */
/* ────────────────────────────────────────────────────────────────────────── */

/* Build a single-entry FbxTcBlock for the given mopcode + fields.  Returns a
 * heap-allocated block; caller frees via FreeTc(). */
static struct FbxTcBlock *MakeTc(u64 mopcode, u64 rde_extra, u64 uimm0,
                                 i64 disp, u64 ip, u8 oplen, u8 kind) {
  struct FbxTcBlock *b = (struct FbxTcBlock *)calloc(1, sizeof *b);
  b->start_pc = ip;
  b->end_pc = ip + oplen;
  b->page = ip & ~(u64)4095;
  b->nentries = 1;
  b->hits = 0;
  b->entries = (struct FbxTcEntry *)calloc(1, sizeof(struct FbxTcEntry));
  b->entries[0].ip = ip;
  b->entries[0].rde = (mopcode << 050) | rde_extra;
  b->entries[0].uimm0 = uimm0;
  b->entries[0].disp = disp;
  b->entries[0].oplen = oplen;
  b->entries[0].kind = kind;
  b->entries[0].fn = NULL;
  return b;
}

static void FreeTc(struct FbxTcBlock *b) {
  free(b->entries);
  free(b);
}

/* Helper — does the block contain an inst with this opcode anywhere? */
static int BlockHas(const struct FbxIrBlock *ir, u8 opcode) {
  u32 i;
  if (!ir) return 0;
  for (i = 0; i < ir->ninsts; ++i) {
    if (ir->insts[i].opcode == opcode) return 1;
  }
  return 0;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Per-opcode lifting tests.                                                  */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxIrLift, MovEvqpGvqp) {
  /* opcode 0x89: MOV r/m64, r64.  Expect REG_GET + REG_SET. */
  struct FbxTcBlock *tc = MakeTc(0x089, 0, 0, 0, 0x400000, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_PC_MARK));
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_REG_GET));
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_REG_SET));
  EXPECT_EQ(0, BlockHas(ir, FBX_IR_OP_BAILOUT));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, MovGvqpEvqp) {
  /* opcode 0x8B: MOV r64, r/m64. */
  struct FbxTcBlock *tc = MakeTc(0x08B, 0, 0, 0, 0x400000, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_REG_GET));
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_REG_SET));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, MovZvqpIvqp) {
  /* opcode 0xB8: MOV reg, imm.  Expect a single REG_SET with IMM src. */
  struct FbxTcBlock *tc = MakeTc(0x0B8, 0, 0xDEADBEEFCAFEBABEULL, 0,
                                 0x400000, 10, FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  u32 i;
  int saw_set_imm = 0;
  ASSERT_NOTNULL(ir);
  for (i = 0; i < ir->ninsts; ++i) {
    if (ir->insts[i].opcode == FBX_IR_OP_REG_SET &&
        ir->insts[i].src1_kind == FBX_IR_KIND_IMM &&
        ir->insts[i].imm == 0xDEADBEEFCAFEBABEULL) {
      saw_set_imm = 1;
    }
  }
  EXPECT_EQ(1, saw_set_imm);
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, MovImm) {
  /* opcode 0xC7: MOV r/m, imm32. */
  struct FbxTcBlock *tc = MakeTc(0x0C7, 0, 0x12345678, 0, 0x400000, 7,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_REG_SET));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, Lea) {
  /* opcode 0x8D: LEA r64, m. */
  struct FbxTcBlock *tc = MakeTc(0x08D, 0, 0, 0x10, 0x400000, 4,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_LEA));
  EXPECT_EQ(0, BlockHas(ir, FBX_IR_OP_BAILOUT));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, AddRR) {
  /* opcode 0x01: ADD r/m64, r64. */
  struct FbxTcBlock *tc = MakeTc(0x001, 0, 0, 0, 0x400000, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_ADD));
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_SET_FLAGS_RAW));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, SetFlagsRawCarriesOperandsV2) {
  /* #599 (FBX_IR_VERSION 2) — SET_FLAGS_RAW must encode (lhs_vreg, rhs_vreg)
   * into (src1, src2) so the wasm synthesis can compute the flag update
   * without re-walking IR history.  This test guards against regression to
   * v1's "imm-only" encoding.  Both LiftAluRR and LiftAlui have to maintain
   * this contract — covered by ADD r/m, r (LiftAluRR) below + AluiSubImm
   * test variant. */
  struct FbxTcBlock *tc = MakeTc(0x001, 0, 0, 0, 0x400000, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  u32 i;
  int saw_flags = 0;
  ASSERT_NOTNULL(ir);
  for (i = 0; i < ir->ninsts; ++i) {
    if (ir->insts[i].opcode == FBX_IR_OP_SET_FLAGS_RAW) {
      saw_flags = 1;
      EXPECT_EQ(FBX_IR_KIND_VREG, ir->insts[i].src1_kind);
      EXPECT_EQ(FBX_IR_KIND_VREG, ir->insts[i].src2_kind);
      /* op_kind in imm stays. */
      EXPECT_EQ((u64)FBX_IR_OP_ADD, ir->insts[i].imm);
    }
  }
  EXPECT_EQ(1, saw_flags);
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, SetFlagsRawCarriesOperandsV2Alui) {
  /* Same v2 IR contract for LiftAlui (opcode 0x83 group-1 r/m, imm8). */
  u64 rde_extra = ((u64)5) << 0; /* ModrmReg=5 → SUB */
  struct FbxTcBlock *tc = MakeTc(0x083, rde_extra, 0x10, 0, 0x400000, 4,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  u32 i;
  int saw_flags = 0;
  ASSERT_NOTNULL(ir);
  for (i = 0; i < ir->ninsts; ++i) {
    if (ir->insts[i].opcode == FBX_IR_OP_SET_FLAGS_RAW) {
      saw_flags = 1;
      EXPECT_EQ(FBX_IR_KIND_VREG, ir->insts[i].src1_kind);
      EXPECT_EQ(FBX_IR_KIND_VREG, ir->insts[i].src2_kind);
      EXPECT_EQ((u64)FBX_IR_OP_SUB, ir->insts[i].imm);
    }
  }
  EXPECT_EQ(1, saw_flags);
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, AluiSubImm) {
  /* opcode 0x83: group 1 r/m, imm8 — pick SUB via ModrmReg=5. */
  u64 rde_extra = ((u64)5) << 0; /* ModrmReg=5 */
  struct FbxTcBlock *tc = MakeTc(0x083, rde_extra, 0x10, 0, 0x400000, 4,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_SUB));
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_SET_FLAGS_RAW));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, CmpRR) {
  /* opcode 0x39: CMP r/m, r. */
  struct FbxTcBlock *tc = MakeTc(0x039, 0, 0, 0, 0x400000, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_CMP));
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_SET_FLAGS_RAW));
  /* CMP must NOT write back to a guest reg. */
  EXPECT_EQ(0, BlockHas(ir, FBX_IR_OP_REG_SET));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, JeShort) {
  /* opcode 0x74: JE rel8. */
  struct FbxTcBlock *tc = MakeTc(0x074, 0, 0, 0x10, 0x400000, 2,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_BRANCH_COND));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, CallRel32) {
  /* opcode 0xE8: CALL rel32. */
  struct FbxTcBlock *tc = MakeTc(0x0E8, 0, 0, 0x100, 0x400000, 5,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_CALL_DIRECT));
  EXPECT_NE(0, (int)(ir->flags & FBX_IR_BLOCK_FLAG_HAS_CALL));
  fbx_ir_free(ir);
  FreeTc(tc);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Bailout coverage.                                                          */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxIrLift, ThunkBlockRefuses) {
  /* A block whose first entry is a thunk MUST return NULL — Tier 1 owns
   * thunked dispatch and Tier 2 has no business escalating it. */
  struct FbxTcBlock *tc = MakeTc(0, 0, 0, 0, 0x400000, 0,
                                 FBX_TC_KIND_THUNK);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  EXPECT_EQ(0, (i64)(intptr_t)ir);
  FreeTc(tc);
}

TEST(FbxIrLift, SyscallBailout) {
  /* opcode 0x105: OpSyscall.  Must bail before the op fires. */
  struct FbxTcBlock *tc = MakeTc(0x105, 0, 0, 0, 0x400000, 2,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_BAILOUT));
  EXPECT_NE(0, (int)(ir->flags & FBX_IR_BLOCK_FLAG_HAS_BAILOUT));
  /* Final inst must be BAILOUT with reason=SYSCALL. */
  ASSERT_GT(ir->ninsts, 0);
  EXPECT_EQ(FBX_IR_OP_BAILOUT, ir->insts[ir->ninsts - 1].opcode);
  EXPECT_EQ(FBX_IR_BAILOUT_SYSCALL,
            (int)ir->insts[ir->ninsts - 1].src1);
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, UnsupportedOpcodeBailout) {
  /* opcode 0x06D: OpIns (rare; not in top-10). */
  struct FbxTcBlock *tc = MakeTc(0x06D, 0, 0, 0, 0x400000, 1,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_BAILOUT));
  ASSERT_GT(ir->ninsts, 0);
  EXPECT_EQ(FBX_IR_OP_BAILOUT, ir->insts[ir->ninsts - 1].opcode);
  EXPECT_EQ(FBX_IR_BAILOUT_UNSUPPORTED,
            (int)ir->insts[ir->ninsts - 1].src1);
  /* Should also be flagged NON_TRANSLATABLE since there's no translatable
   * prefix (PC_MARK + BAILOUT is the whole block). */
  EXPECT_NE(0, (int)(ir->flags & FBX_IR_BLOCK_FLAG_NON_TRANSLATABLE));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, SerializingBailout) {
  /* opcode 0x1A2: OpCpuid — must bail out. */
  struct FbxTcBlock *tc = MakeTc(0x1A2, 0, 0, 0, 0x400000, 2,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  ASSERT_GT(ir->ninsts, 0);
  EXPECT_EQ(FBX_IR_OP_BAILOUT, ir->insts[ir->ninsts - 1].opcode);
  EXPECT_EQ(FBX_IR_BAILOUT_SERIALIZING,
            (int)ir->insts[ir->ninsts - 1].src1);
  EXPECT_NE(0, (int)(ir->flags & FBX_IR_BLOCK_FLAG_HAS_MEM_BARRIER));
  fbx_ir_free(ir);
  FreeTc(tc);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Determinism — §Q5 axis 1 (same-process determinism).                       */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxIrLift, DeterminismSameInputSameSha) {
  struct FbxTcBlock *tc = MakeTc(0x089, 0, 0, 0, 0x400000, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *a = fbx_ir_lift(tc);
  struct FbxIrBlock *b = fbx_ir_lift(tc);
  ASSERT_NOTNULL(a);
  ASSERT_NOTNULL(b);
  EXPECT_EQ(0, memcmp(a->cache_key_sha256, b->cache_key_sha256, 32));
  EXPECT_EQ(a->ninsts, b->ninsts);
  if (a->ninsts == b->ninsts) {
    EXPECT_EQ(0, memcmp(a->insts, b->insts,
                        (size_t)a->ninsts * sizeof(struct FbxIrInst)));
  }
  fbx_ir_free(a);
  fbx_ir_free(b);
  FreeTc(tc);
}

TEST(FbxIrLift, DifferentInputDifferentSha) {
  /* Pick two opcodes whose IR shapes differ:
   *   MOV r/m, r (0x089)  → REG_GET + REG_SET (2 ALU-side insts)
   *   ADD r/m, r (0x001)  → REG_GET x2 + ADD + REG_SET + SET_FLAGS_RAW
   * The lifted byte streams differ → SHAs MUST differ. */
  struct FbxTcBlock *tc1 = MakeTc(0x089, 0, 0, 0, 0x400000, 3,
                                  FBX_TC_KIND_NORMAL);
  struct FbxTcBlock *tc2 = MakeTc(0x001, 0, 0, 0, 0x400000, 3,
                                  FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *a = fbx_ir_lift(tc1);
  struct FbxIrBlock *b = fbx_ir_lift(tc2);
  ASSERT_NOTNULL(a);
  ASSERT_NOTNULL(b);
  EXPECT_NE(0, memcmp(a->cache_key_sha256, b->cache_key_sha256, 32));
  fbx_ir_free(a);
  fbx_ir_free(b);
  FreeTc(tc1);
  FreeTc(tc2);
}

TEST(FbxIrLift, ShaIsNonzero) {
  struct FbxTcBlock *tc = MakeTc(0x089, 0, 0, 0, 0x400000, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  u8 zero[32];
  memset(zero, 0, 32);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, memcmp(ir->cache_key_sha256, zero, 32));
  fbx_ir_free(ir);
  FreeTc(tc);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Version-bump invalidation — §Q5 axis 3.                                    */
/*                                                                            */
/* We can't bump FBX_IR_VERSION at test time, but we CAN validate the         */
/* equivalent invariant: changing the opcode bytes in an IR (post-lift)       */
/* MUST change the SHA.  This is the "stale-IR detection" path that the      */
/* version bump piggybacks on.                                                */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxIrLift, ChangingInstChangesSha) {
  struct FbxTcBlock *tc = MakeTc(0x089, 0, 0, 0, 0x400000, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  u8 sha_orig[32];
  u8 sha_changed[32];
  ASSERT_NOTNULL(ir);
  ASSERT_GT(ir->ninsts, 1);
  memcpy(sha_orig, ir->cache_key_sha256, 32);
  /* Change one byte in the inst stream + re-hash. */
  ir->insts[1].dst ^= 0x1;
  fbx_ir_block_sha256(ir, sha_changed);
  EXPECT_NE(0, memcmp(sha_orig, sha_changed, 32));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, OpcodeNameNonNull) {
  /* Defensive: opcode_name must never return NULL even for sentinel values. */
  u32 i;
  EXPECT_STREQ("PC_MARK", fbx_ir_opcode_name(FBX_IR_OP_PC_MARK));
  EXPECT_STREQ("BAILOUT", fbx_ir_opcode_name(FBX_IR_OP_BAILOUT));
  EXPECT_STREQ("UNKNOWN", fbx_ir_opcode_name(255));
  for (i = 0; i < FBX_IR_OP_LAST_; ++i) {
    ASSERT_NOTNULL(fbx_ir_opcode_name((u8)i));
  }
}

TEST(FbxIrLift, BailoutReasonName) {
  EXPECT_STREQ("THUNK",
               fbx_ir_bailout_reason_name(FBX_IR_BAILOUT_THUNK));
  EXPECT_STREQ("SYSCALL",
               fbx_ir_bailout_reason_name(FBX_IR_BAILOUT_SYSCALL));
  EXPECT_STREQ("UNSUPPORTED",
               fbx_ir_bailout_reason_name(FBX_IR_BAILOUT_UNSUPPORTED));
  EXPECT_STREQ("UNKNOWN", fbx_ir_bailout_reason_name(99));
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Multi-instruction block — mixed ADD + MOV ending in JE.                    */
/* Validates the full lifting walk (multiple PC_MARKs, branches, etc.).       */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxIrLift, MultiInstBlock) {
  struct FbxTcBlock *b = (struct FbxTcBlock *)calloc(1, sizeof *b);
  struct FbxIrBlock *ir;
  u32 pc_marks = 0;
  u32 i;
  b->start_pc = 0x400000;
  b->page = 0x400000;
  b->nentries = 3;
  b->entries = (struct FbxTcEntry *)calloc(3, sizeof(struct FbxTcEntry));
  /* [0]: ADD r/m, r at 0x400000 (mopcode 0x001). */
  b->entries[0].ip = 0x400000;
  b->entries[0].rde = (u64)0x001 << 050;
  b->entries[0].oplen = 3;
  b->entries[0].kind = FBX_TC_KIND_NORMAL;
  /* [1]: MOV r/m, r at 0x400003 (mopcode 0x089). */
  b->entries[1].ip = 0x400003;
  b->entries[1].rde = (u64)0x089 << 050;
  b->entries[1].oplen = 3;
  b->entries[1].kind = FBX_TC_KIND_NORMAL;
  /* [2]: JE rel8 at 0x400006 (mopcode 0x074) — block-terminating. */
  b->entries[2].ip = 0x400006;
  b->entries[2].rde = (u64)0x074 << 050;
  b->entries[2].disp = 0x10;
  b->entries[2].oplen = 2;
  b->entries[2].kind = FBX_TC_KIND_NORMAL;
  b->end_pc = 0x400008;
  ir = fbx_ir_lift(b);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_ADD));
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_BRANCH_COND));
  for (i = 0; i < ir->ninsts; ++i) {
    if (ir->insts[i].opcode == FBX_IR_OP_PC_MARK) ++pc_marks;
  }
  EXPECT_EQ(3, (i64)pc_marks);
  EXPECT_EQ(0, BlockHas(ir, FBX_IR_OP_BAILOUT));
  fbx_ir_free(ir);
  free(b->entries);
  free(b);
}

/* Empty / NULL input — defensive. */
TEST(FbxIrLift, NullInputReturnsNull) {
  EXPECT_EQ(0, (i64)(intptr_t)fbx_ir_lift(NULL));
}

/* ────────────────────────────────────────────────────────────────────────── */
/* §13.5 — Mirror-direction ALU forms (lift extension).                       */
/*                                                                            */
/* These mopcodes were previously routed to the default-case BAILOUT path     */
/* and the runtime stayed on Tier 1.  After §13.5, the lifter produces       */
/* full IR (REG_GET x2 + ALU + REG_SET? + SET_FLAGS_RAW) and the synthesis    */
/* pass refuses on SET_FLAGS_RAW — runtime still stays on Tier 1, but the     */
/* IR is now available for #597's lazy-flag work and Phase 4 reachability.    */
/* The tests below validate the IR shape, NOT the synthesis (separate test    */
/* file).                                                                     */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxIrLift, MirrorAddRRm) {
  /* opcode 0x003: ADD r, r/m.  Lift to REG_GET x2 + ADD + REG_SET + flags. */
  struct FbxTcBlock *tc = MakeTc(0x003, 0, 0, 0, 0x400000, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_ADD));
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_REG_GET));
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_REG_SET));
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_SET_FLAGS_RAW));
  EXPECT_EQ(0, BlockHas(ir, FBX_IR_OP_BAILOUT));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, MirrorSubRRm) {
  /* opcode 0x02B: SUB r, r/m. */
  struct FbxTcBlock *tc = MakeTc(0x02B, 0, 0, 0, 0x400000, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_SUB));
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_SET_FLAGS_RAW));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, MirrorCmpRRm) {
  /* opcode 0x03B: CMP r, r/m.  CMP must NOT emit REG_SET (flag-only op). */
  struct FbxTcBlock *tc = MakeTc(0x03B, 0, 0, 0, 0x400000, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_CMP));
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_SET_FLAGS_RAW));
  /* CMP r, r/m must not REG_SET — same invariant as 0x39 CMP r/m, r. */
  EXPECT_EQ(0, BlockHas(ir, FBX_IR_OP_REG_SET));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, MirrorXorRRm) {
  /* opcode 0x033: XOR r, r/m. */
  struct FbxTcBlock *tc = MakeTc(0x033, 0, 0, 0, 0x400000, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_XOR));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, MirrorAndRRm) {
  /* opcode 0x023: AND r, r/m. */
  struct FbxTcBlock *tc = MakeTc(0x023, 0, 0, 0, 0x400000, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_AND));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, MirrorOrRRm) {
  /* opcode 0x00B: OR r, r/m. */
  struct FbxTcBlock *tc = MakeTc(0x00B, 0, 0, 0, 0x400000, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_OR));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, MirrorByteAddRRm) {
  /* opcode 0x002: ADD r8, r/m8.  Byte-form mirror.  Width should be 1. */
  struct FbxTcBlock *tc = MakeTc(0x002, 0, 0, 0, 0x400000, 2,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  u32 i;
  int saw_byte_get = 0;
  ASSERT_NOTNULL(ir);
  EXPECT_NE(0, BlockHas(ir, FBX_IR_OP_ADD));
  for (i = 0; i < ir->ninsts; ++i) {
    if (ir->insts[i].opcode == FBX_IR_OP_REG_GET &&
        ir->insts[i].width == 1) {
      saw_byte_get = 1;
    }
  }
  EXPECT_EQ(1, saw_byte_get);
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, MirrorRegisterAssignment) {
  /* Mirror ops swap the source greg ids.  Use RexrReg=3, RexbRm=5; verify:
   *   - For 0x001 (ADD r/m, r): lhs greg = 5 (RexbRm), rhs greg = 3 (RexrReg),
   *     dst greg = 5
   *   - For 0x003 (ADD r, r/m): lhs greg = 3 (RexrReg), rhs greg = 5 (RexbRm),
   *     dst greg = 3
   * rde_extra encodes both RexbRm (bits 7-10, shifted >> 7) and RexrReg
   * (bits 0-3).  See blink/rde.h.
   *   RexrReg=3 → bits 0-3 = 3
   *   RexbRm=5  → bits 7-10 = 5 → 5 << 7 = 0x280 = 0640o (octal)
   */
  u64 rde_extra = (u64)3 | ((u64)5 << 7);
  struct FbxTcBlock *tc_a = MakeTc(0x001, rde_extra, 0, 0, 0x400000, 3,
                                   FBX_TC_KIND_NORMAL);
  struct FbxTcBlock *tc_b = MakeTc(0x003, rde_extra, 0, 0, 0x400000, 3,
                                   FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *a = fbx_ir_lift(tc_a);
  struct FbxIrBlock *b = fbx_ir_lift(tc_b);
  u32 i;
  u32 a_lhs_src = 0xFFFF, a_rhs_src = 0xFFFF, a_dst = 0xFFFF;
  u32 b_lhs_src = 0xFFFF, b_rhs_src = 0xFFFF, b_dst = 0xFFFF;
  int seen = 0;
  ASSERT_NOTNULL(a);
  ASSERT_NOTNULL(b);
  for (i = 0; i < a->ninsts; ++i) {
    if (a->insts[i].opcode == FBX_IR_OP_REG_GET) {
      if (seen == 0) a_lhs_src = a->insts[i].src1;
      else if (seen == 1) a_rhs_src = a->insts[i].src1;
      seen++;
    } else if (a->insts[i].opcode == FBX_IR_OP_REG_SET) {
      a_dst = a->insts[i].dst;
    }
  }
  seen = 0;
  for (i = 0; i < b->ninsts; ++i) {
    if (b->insts[i].opcode == FBX_IR_OP_REG_GET) {
      if (seen == 0) b_lhs_src = b->insts[i].src1;
      else if (seen == 1) b_rhs_src = b->insts[i].src1;
      seen++;
    } else if (b->insts[i].opcode == FBX_IR_OP_REG_SET) {
      b_dst = b->insts[i].dst;
    }
  }
  /* 0x001: RM_R direction; lhs=RexbRm=5, rhs=RexrReg=3, dst=RexbRm=5. */
  EXPECT_EQ(5u, a_lhs_src);
  EXPECT_EQ(3u, a_rhs_src);
  EXPECT_EQ(5u, a_dst);
  /* 0x003: R_RM direction; lhs=RexrReg=3, rhs=RexbRm=5, dst=RexrReg=3. */
  EXPECT_EQ(3u, b_lhs_src);
  EXPECT_EQ(5u, b_rhs_src);
  EXPECT_EQ(3u, b_dst);
  fbx_ir_free(a);
  fbx_ir_free(b);
  FreeTc(tc_a);
  FreeTc(tc_b);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* §13.5 — MOVZX (zero-extending move).                                       */
/*                                                                            */
/* Lift to REG_GET (narrow) + REG_SET (wide).  Synthesis SHOULD handle this   */
/* via the existing i64.load{8,16}_u → i64.store pipeline (separate test).    */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxIrLift, Movzx8) {
  /* opcode 0x1B6: MOVZX r, r/m8.  No REX.W → dst width = 4. */
  struct FbxTcBlock *tc = MakeTc(0x1B6, 0, 0, 0, 0x400000, 4,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  u32 i;
  int saw_byte_get = 0;
  int saw_wide_set = 0;
  ASSERT_NOTNULL(ir);
  EXPECT_EQ(0, BlockHas(ir, FBX_IR_OP_BAILOUT));
  /* Must have exactly one REG_GET with width=1 and one REG_SET with
   * width >= 4 (full register destination). */
  for (i = 0; i < ir->ninsts; ++i) {
    if (ir->insts[i].opcode == FBX_IR_OP_REG_GET) {
      EXPECT_EQ(1, (int)ir->insts[i].width);
      saw_byte_get = 1;
    }
    if (ir->insts[i].opcode == FBX_IR_OP_REG_SET) {
      EXPECT_NE(0, ir->insts[i].width >= 4);
      saw_wide_set = 1;
    }
  }
  EXPECT_EQ(1, saw_byte_get);
  EXPECT_EQ(1, saw_wide_set);
  /* MOVZX must NOT emit ALU ops or flag effects. */
  EXPECT_EQ(0, BlockHas(ir, FBX_IR_OP_SET_FLAGS_RAW));
  EXPECT_EQ(0, BlockHas(ir, FBX_IR_OP_ADD));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, Movzx16) {
  /* opcode 0x1B7: MOVZX r, r/m16. */
  struct FbxTcBlock *tc = MakeTc(0x1B7, 0, 0, 0, 0x400000, 4,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  u32 i;
  int saw_word_get = 0;
  ASSERT_NOTNULL(ir);
  EXPECT_EQ(0, BlockHas(ir, FBX_IR_OP_BAILOUT));
  for (i = 0; i < ir->ninsts; ++i) {
    if (ir->insts[i].opcode == FBX_IR_OP_REG_GET) {
      EXPECT_EQ(2, (int)ir->insts[i].width);
      saw_word_get = 1;
    }
  }
  EXPECT_EQ(1, saw_word_get);
  EXPECT_EQ(0, BlockHas(ir, FBX_IR_OP_SET_FLAGS_RAW));
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxIrLift, Movzx8Rexw) {
  /* opcode 0x1B6 with REX.W set — dst width must be 8 (Qword reg).
   * Per blink/rde.h:  Rexw(x) = ((x & 0100o) >> 6).  Octal 0100 = 0x40 =
   * bit 6.  Set rde bit 6 to force REX.W = 1.  Width helper:
   * WidthFromRde(rde, byte_op=0) → 8 when REX.W is set. */
  u64 rexw_bit = (u64)0x40;
  struct FbxTcBlock *tc = MakeTc(0x1B6, rexw_bit, 0, 0, 0x400000, 4,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  u32 i;
  int saw_qword_set = 0;
  ASSERT_NOTNULL(ir);
  for (i = 0; i < ir->ninsts; ++i) {
    if (ir->insts[i].opcode == FBX_IR_OP_REG_SET) {
      if (ir->insts[i].width == 8) saw_qword_set = 1;
    }
  }
  EXPECT_EQ(1, saw_qword_set);
  fbx_ir_free(ir);
  FreeTc(tc);
}
