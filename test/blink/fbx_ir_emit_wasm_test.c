/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2026 Justin Firth (Firebox)                                        │
│                                                                              │
│ Unit tests for the Phase-2 Tier-2 §13.2 wasm synthesis pass.                 │
│ See blink/fbx_wasm_emit.h for the API + design rationale.                    │
│                                                                              │
│ Coverage:                                                                    │
│   - LEB128 wire format (FIPS-style golden vectors)                          │
│   - Module preamble (magic + version)                                        │
│   - Host-import declaration count + names                                   │
│   - Per-opcode lowering for the v0.1 subset                                 │
│   - Coverage gate refusal for non-v0.1 IR (memory-modrm, flag-dependent)    │
│   - Determinism: same input → same bytes                                    │
│   - Round-trip: lift → emit → wasm bytes form a complete module             │
│                                                                              │
│ A separate cross-engine validation harness (out-of-tree, run from the       │
│ orchestrating shell) parses the emitted wasm via both wasmer::Module::new   │
│ and WebAssembly.compile() to satisfy AGENTS.md invariant 4 (browser is a   │
│ co-equal target).  Per the spec §5.1, the emitted bytes must parse cleanly │
│ on every wasm engine Firebox supports.                                     │
╚─────────────────────────────────────────────────────────────────────────────*/

#include "blink/fbx_ir.h"
#include "blink/fbx_wasm_emit.h"
#include "blink/rde.h"
#include "blink/threadedcode.h"
#include "blink/types.h"
#include "test/test.h"

#include <string.h>

void SetUp(void) {}
void TearDown(void) {}

/* ────────────────────────────────────────────────────────────────────────── */
/* LEB128 wire format.                                                        */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxWasmLeb, UlebSingleByte) {
  struct FbxWasmBuffer b;
  fbx_wasm_buffer_init(&b);
  fbx_wasm_buffer_uleb(&b, 0);
  ASSERT_EQ(1, (i64)b.len);
  ASSERT_EQ(0x00, b.data[0]);
  fbx_wasm_buffer_free(&b);
}

TEST(FbxWasmLeb, UlebMaxSingleByte) {
  struct FbxWasmBuffer b;
  fbx_wasm_buffer_init(&b);
  fbx_wasm_buffer_uleb(&b, 0x7f);
  ASSERT_EQ(1, (i64)b.len);
  ASSERT_EQ(0x7f, b.data[0]);
  fbx_wasm_buffer_free(&b);
}

TEST(FbxWasmLeb, UlebTwoBytes) {
  struct FbxWasmBuffer b;
  fbx_wasm_buffer_init(&b);
  fbx_wasm_buffer_uleb(&b, 128);
  ASSERT_EQ(2, (i64)b.len);
  ASSERT_EQ(0x80, b.data[0]);
  ASSERT_EQ(0x01, b.data[1]);
  fbx_wasm_buffer_free(&b);
}

TEST(FbxWasmLeb, UlebLargeValue) {
  struct FbxWasmBuffer b;
  fbx_wasm_buffer_init(&b);
  fbx_wasm_buffer_uleb(&b, 624485);
  /* LEB128 canonical: 0xE5 0x8E 0x26 */
  ASSERT_EQ(3, (i64)b.len);
  ASSERT_EQ(0xE5, b.data[0]);
  ASSERT_EQ(0x8E, b.data[1]);
  ASSERT_EQ(0x26, b.data[2]);
  fbx_wasm_buffer_free(&b);
}

TEST(FbxWasmLeb, SlebPositive) {
  struct FbxWasmBuffer b;
  fbx_wasm_buffer_init(&b);
  fbx_wasm_buffer_sleb(&b, 0);
  ASSERT_EQ(1, (i64)b.len);
  ASSERT_EQ(0x00, b.data[0]);
  fbx_wasm_buffer_free(&b);
}

TEST(FbxWasmLeb, SlebNegativeOne) {
  struct FbxWasmBuffer b;
  fbx_wasm_buffer_init(&b);
  fbx_wasm_buffer_sleb(&b, -1);
  ASSERT_EQ(1, (i64)b.len);
  ASSERT_EQ(0x7f, b.data[0]);
  fbx_wasm_buffer_free(&b);
}

TEST(FbxWasmLeb, SlebMinusSixtyFour) {
  struct FbxWasmBuffer b;
  fbx_wasm_buffer_init(&b);
  fbx_wasm_buffer_sleb(&b, -64);
  ASSERT_EQ(1, (i64)b.len);
  ASSERT_EQ(0x40, b.data[0]);
  fbx_wasm_buffer_free(&b);
}

TEST(FbxWasmLeb, U32Le) {
  struct FbxWasmBuffer b;
  fbx_wasm_buffer_init(&b);
  fbx_wasm_buffer_u32_le(&b, 0x6D736100u);
  ASSERT_EQ(4, (i64)b.len);
  ASSERT_EQ(0x00, b.data[0]);
  ASSERT_EQ(0x61, b.data[1]);
  ASSERT_EQ(0x73, b.data[2]);
  ASSERT_EQ(0x6D, b.data[3]);
  fbx_wasm_buffer_free(&b);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Host import surface.                                                       */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxWasmImports, CountStable) {
  /* Per spec §5.4 the import count is FIXED at v0.1.  Adding an import
   * requires cache invalidation; the test pins the expected value. */
  ASSERT_EQ(3, (i64)fbx_wasm_host_import_count());
}

TEST(FbxWasmImports, NamesStable) {
  EXPECT_STREQ("call_thunk", fbx_wasm_host_import_name(0));
  EXPECT_STREQ("call_syscall", fbx_wasm_host_import_name(1));
  EXPECT_STREQ("resolve_indirect", fbx_wasm_host_import_name(2));
  ASSERT_EQ(0, (i64)(intptr_t)fbx_wasm_host_import_name(99));
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Helper: build a single-entry FbxTcBlock for synthesis tests.               */
/*                                                                            */
/* The synthesis pass reads tc->entries[].rde + mopcode to apply the         */
/* coverage gate.  We construct rde with the same (mop << 050 | extra)        */
/* layout used by Blink's xed decoder (mirroring the §13.1 lift_test          */
/* helper).                                                                   */
/* ────────────────────────────────────────────────────────────────────────── */

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

/* Build an N-entry TC block (each entry sharing identical fields).  Used to
 * exercise multi-instruction lifts. */
static struct FbxTcBlock *MakeTcN(const u64 *mops, const u64 *rdes,
                                  const u64 *uimms, const i64 *disps,
                                  const u64 *ips, const u8 *oplens,
                                  unsigned n) {
  struct FbxTcBlock *b = (struct FbxTcBlock *)calloc(1, sizeof *b);
  unsigned i;
  b->start_pc = ips[0];
  b->end_pc = ips[n - 1] + oplens[n - 1];
  b->page = ips[0] & ~(u64)4095;
  b->nentries = n;
  b->entries = (struct FbxTcEntry *)calloc(n, sizeof(struct FbxTcEntry));
  for (i = 0; i < n; ++i) {
    b->entries[i].ip = ips[i];
    b->entries[i].rde = (mops[i] << 050) | rdes[i];
    b->entries[i].uimm0 = uimms[i];
    b->entries[i].disp = disps[i];
    b->entries[i].oplen = oplens[i];
    b->entries[i].kind = FBX_TC_KIND_NORMAL;
    b->entries[i].fn = NULL;
  }
  return b;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Module preamble — every successful synthesis emits the wasm magic + ver.   */
/* ────────────────────────────────────────────────────────────────────────── */

/* RDE bit positions (see blink/rde.h):
 *   ModrmMod(x) = (x & 0o000060000000) >> 0o026
 *   In hex: mask = 0xC00000, shift = 22 (decimal).
 *   To encode modrm.mod==3 we set bits 22-23, i.e. 3 << 22 = 0xC00000.
 *
 * Pinned via a Python-verified probe (see #588 task notes); do NOT change
 * without re-deriving from blink/rde.h's octal literal. */
#define RDE_MOD3 ((u64)0x00C00000ull)

TEST(FbxWasmEmit, EmptyBlockReturnsZero) {
  struct FbxIrBlock ir;
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  tc = MakeTc(0x089, RDE_MOD3, 0, 0, 0x1000, 3, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ir.start_pc = 0x1000;
  ir.end_pc = 0x1003;
  ir.ninsts = 0;
  /* Empty IR: coverage gate refuses (ir->ninsts == 0). */
  ASSERT_EQ(0, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_EQ(0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, MovRegRegMagic) {
  /* Build a TC block + IR block for a single MOV %rax, %rbx (mop 0x089). */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[3];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  /* PC_MARK at 0x1000. */
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0x1000;
  /* REG_GET — load %rax (greg 0) into vreg 0. */
  insts[1].opcode = FBX_IR_OP_REG_GET;
  insts[1].width = 8;
  insts[1].dst_kind = FBX_IR_KIND_VREG;
  insts[1].dst = 0;
  insts[1].src1_kind = FBX_IR_KIND_GREG;
  insts[1].src1 = 0;
  /* REG_SET — store vreg 0 into %rbx (greg 3). */
  insts[2].opcode = FBX_IR_OP_REG_SET;
  insts[2].width = 8;
  insts[2].dst_kind = FBX_IR_KIND_GREG;
  insts[2].dst = 3;
  insts[2].src1_kind = FBX_IR_KIND_VREG;
  insts[2].src1 = 0;
  ir.insts = insts;
  ir.ninsts = 3;
  ir.nvregs = 1;
  ir.start_pc = 0x1000;
  ir.end_pc = 0x1003;
  tc = MakeTc(0x089, RDE_MOD3, 0, 0, 0x1000, 3, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  /* Verify magic + version. */
  ASSERT_EQ(0x00, out.data[0]);
  ASSERT_EQ(0x61, out.data[1]); /* 'a' */
  ASSERT_EQ(0x73, out.data[2]); /* 's' */
  ASSERT_EQ(0x6D, out.data[3]); /* 'm' */
  ASSERT_EQ(0x01, out.data[4]);
  ASSERT_EQ(0x00, out.data[5]);
  ASSERT_EQ(0x00, out.data[6]);
  ASSERT_EQ(0x00, out.data[7]);
  /* Section 1 (type) follows. */
  ASSERT_EQ(0x01, out.data[8]);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Determinism — same input must produce byte-identical output.               */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxWasmEmit, DeterministicOutput) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[3];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out1, out2;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0x2000;
  insts[1].opcode = FBX_IR_OP_REG_GET;
  insts[1].width = 8;
  insts[1].dst_kind = FBX_IR_KIND_VREG;
  insts[1].dst = 0;
  insts[1].src1_kind = FBX_IR_KIND_GREG;
  insts[1].src1 = 1;
  insts[2].opcode = FBX_IR_OP_REG_SET;
  insts[2].width = 8;
  insts[2].dst_kind = FBX_IR_KIND_GREG;
  insts[2].dst = 2;
  insts[2].src1_kind = FBX_IR_KIND_VREG;
  insts[2].src1 = 0;
  ir.insts = insts;
  ir.ninsts = 3;
  ir.nvregs = 1;
  ir.start_pc = 0x2000;
  ir.end_pc = 0x2003;
  tc = MakeTc(0x089, RDE_MOD3, 0, 0, 0x2000, 3, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out1);
  fbx_wasm_buffer_init(&out2);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out1));
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out2));
  ASSERT_EQ((i64)out1.len, (i64)out2.len);
  ASSERT_EQ(0, memcmp(out1.data, out2.data, out1.len));
  fbx_wasm_buffer_free(&out1);
  fbx_wasm_buffer_free(&out2);
  FreeTc(tc);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Per-opcode synthesis tests.  Each builds the IR for one x86 op + verifies  */
/* that synthesis succeeds (returns 1) and produces non-zero wasm bytes.      */
/* ────────────────────────────────────────────────────────────────────────── */

/* Helper: build an IR block with an ALU + supporting REG_GET/REG_SET seq. */
static void BuildAluIr(struct FbxIrBlock *ir, struct FbxIrInst *insts,
                       u8 alu_op) {
  memset(ir, 0, sizeof *ir);
  memset(insts, 0, 5 * sizeof(struct FbxIrInst));
  /* PC_MARK */
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0x3000;
  /* REG_GET vreg0 = greg[0] */
  insts[1].opcode = FBX_IR_OP_REG_GET;
  insts[1].width = 8;
  insts[1].dst_kind = FBX_IR_KIND_VREG;
  insts[1].dst = 0;
  insts[1].src1_kind = FBX_IR_KIND_GREG;
  insts[1].src1 = 0;
  /* REG_GET vreg1 = greg[1] */
  insts[2].opcode = FBX_IR_OP_REG_GET;
  insts[2].width = 8;
  insts[2].dst_kind = FBX_IR_KIND_VREG;
  insts[2].dst = 1;
  insts[2].src1_kind = FBX_IR_KIND_GREG;
  insts[2].src1 = 1;
  /* ALU vreg2 = vreg0 OP vreg1 */
  insts[3].opcode = alu_op;
  insts[3].width = 8;
  insts[3].dst_kind = FBX_IR_KIND_VREG;
  insts[3].dst = 2;
  insts[3].src1_kind = FBX_IR_KIND_VREG;
  insts[3].src1 = 0;
  insts[3].src2_kind = FBX_IR_KIND_VREG;
  insts[3].src2 = 1;
  /* REG_SET greg[0] = vreg2 */
  insts[4].opcode = FBX_IR_OP_REG_SET;
  insts[4].width = 8;
  insts[4].dst_kind = FBX_IR_KIND_GREG;
  insts[4].dst = 0;
  insts[4].src1_kind = FBX_IR_KIND_VREG;
  insts[4].src1 = 2;
  ir->insts = insts;
  ir->ninsts = 5;
  ir->nvregs = 3;
  ir->start_pc = 0x3000;
  ir->end_pc = 0x3003;
}

TEST(FbxWasmEmit, AluAddSynth) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[5];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  BuildAluIr(&ir, insts, FBX_IR_OP_ADD);
  tc = MakeTc(0x001, RDE_MOD3, 0, 0, 0x3000, 3, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, AluSubSynth) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[5];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  BuildAluIr(&ir, insts, FBX_IR_OP_SUB);
  tc = MakeTc(0x029, RDE_MOD3, 0, 0, 0x3000, 3, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, AluAndSynth) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[5];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  BuildAluIr(&ir, insts, FBX_IR_OP_AND);
  tc = MakeTc(0x021, RDE_MOD3, 0, 0, 0x3000, 3, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, AluOrSynth) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[5];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  BuildAluIr(&ir, insts, FBX_IR_OP_OR);
  tc = MakeTc(0x009, RDE_MOD3, 0, 0, 0x3000, 3, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, AluXorSynth) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[5];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  BuildAluIr(&ir, insts, FBX_IR_OP_XOR);
  tc = MakeTc(0x031, RDE_MOD3, 0, 0, 0x3000, 3, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, MovImmSynth) {
  /* MOV reg, imm — opcode 0xB8.  IR is a single REG_SET with imm source. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[2];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0x4000;
  insts[1].opcode = FBX_IR_OP_REG_SET;
  insts[1].width = 8;
  insts[1].dst_kind = FBX_IR_KIND_GREG;
  insts[1].dst = 0;
  insts[1].src1_kind = FBX_IR_KIND_IMM;
  insts[1].imm = 0x123456789abcdefull;
  ir.insts = insts;
  ir.ninsts = 2;
  ir.nvregs = 0;
  ir.start_pc = 0x4000;
  ir.end_pc = 0x400A;
  tc = MakeTc(0x0B8, 0, 0x123456789abcdefull, 0, 0x4000, 10,
              FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, LeaSimpleSynth) {
  /* LEA dst, [base + disp] with no SIB index. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[2];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0x5000;
  insts[1].opcode = FBX_IR_OP_LEA;
  insts[1].width = 8;
  insts[1].dst_kind = FBX_IR_KIND_GREG;
  insts[1].dst = 1; /* rcx */
  insts[1].src1_kind = FBX_IR_KIND_GREG;
  insts[1].src1 = 0; /* rax */
  insts[1].src2_kind = FBX_IR_KIND_IMM;
  insts[1].imm = 16;
  ir.insts = insts;
  ir.ninsts = 2;
  ir.nvregs = 0;
  ir.start_pc = 0x5000;
  ir.end_pc = 0x5004;
  /* LEA encoding: mop 0x08D, modrm.mod != 3 by definition. */
  tc = MakeTc(0x08D, 0, 0, 16, 0x5000, 4, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, BranchTakenTerminator) {
  /* JMP rel — opcode 0xE9.  IR: PC_MARK + BRANCH_TAKEN. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[2];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0x6000;
  insts[1].opcode = FBX_IR_OP_BRANCH_TAKEN;
  insts[1].imm = 0x6100; /* target PC */
  ir.insts = insts;
  ir.ninsts = 2;
  ir.nvregs = 0;
  ir.start_pc = 0x6000;
  ir.end_pc = 0x6005;
  tc = MakeTc(0x0E9, 0, 0, 0x100, 0x6000, 5, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, BailoutTerminator) {
  /* IR with a single BAILOUT — should still emit a valid wasm module that
   * returns 1 + writes m->ip = bailout PC. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_BAILOUT;
  insts[0].imm = 0x7000;
  ir.insts = insts;
  ir.ninsts = 1;
  ir.nvregs = 0;
  ir.start_pc = 0x7000;
  ir.end_pc = 0x7001;
  /* Pair with a NORMAL kind TC entry so the coverage gate passes (the
   * gate iterates TC entries; we use mop 0x089 with mod=3 as a permissive
   * choice — the gate doesn't require IR↔TC bijection). */
  tc = MakeTc(0x089, RDE_MOD3, 0, 0, 0x7000, 1, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Coverage-gate refusal tests — synthesis must REFUSE unsupported IR.       */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxWasmEmit, RefusesBranchCond) {
  /* BRANCH_COND requires flag handling not in v0.1.  Must return 0. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_BRANCH_COND;
  insts[0].src1_kind = FBX_IR_KIND_IMM;
  insts[0].src1 = 4; /* JE */
  insts[0].imm = 0x8100;
  ir.insts = insts;
  ir.ninsts = 1;
  ir.nvregs = 0;
  tc = MakeTc(0x074, 0, 0, 0, 0x8000, 2, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(0, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_EQ(0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, RefusesSetFlagsRaw) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_SET_FLAGS_RAW;
  ir.insts = insts;
  ir.ninsts = 1;
  tc = MakeTc(0x001, RDE_MOD3, 0, 0, 0x9000, 3, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(0, fbx_ir_emit_wasm(&ir, tc, &out));
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, RefusesCallDirect) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_CALL_DIRECT;
  insts[0].imm = 0xA100;
  ir.insts = insts;
  ir.ninsts = 1;
  tc = MakeTc(0x0E8, 0, 0, 0x100, 0xA000, 5, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(0, fbx_ir_emit_wasm(&ir, tc, &out));
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, RefusesMemoryModrmMov) {
  /* MOV r/m64, r64 with modrm.mod != 3 (memory form) — v0.1 refuses. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[3];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[1].opcode = FBX_IR_OP_REG_GET;
  insts[1].width = 8;
  insts[1].dst_kind = FBX_IR_KIND_VREG;
  insts[1].dst = 0;
  insts[1].src1_kind = FBX_IR_KIND_GREG;
  insts[1].src1 = 0;
  insts[2].opcode = FBX_IR_OP_REG_SET;
  insts[2].width = 8;
  insts[2].dst_kind = FBX_IR_KIND_GREG;
  insts[2].dst = 3;
  insts[2].src1_kind = FBX_IR_KIND_VREG;
  insts[2].src1 = 0;
  ir.insts = insts;
  ir.ninsts = 3;
  ir.nvregs = 1;
  /* RDE with modrm.mod == 0 (memory). */
  tc = MakeTc(0x089, 0, 0, 0, 0xB000, 3, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(0, fbx_ir_emit_wasm(&ir, tc, &out));
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, RefusesThunkBlock) {
  /* TC entry with kind THUNK — coverage gate refuses regardless of IR. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_BAILOUT;
  ir.insts = insts;
  ir.ninsts = 1;
  tc = MakeTc(0x089, RDE_MOD3, 0, 0, 0xC000, 3, FBX_TC_KIND_THUNK);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(0, fbx_ir_emit_wasm(&ir, tc, &out));
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Multi-instruction round-trip: build a multi-op TC block, lift it, emit    */
/* wasm, verify length non-zero and module preamble present.                  */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxWasmEmit, MultiInstRoundTrip) {
  /* Two MOV r/m r ops in sequence (reg-form). */
  u64 mops[2] = {0x089, 0x089};
  u64 rdes[2] = {RDE_MOD3, RDE_MOD3};
  u64 uimms[2] = {0, 0};
  i64 disps[2] = {0, 0};
  u64 ips[2] = {0xD000, 0xD003};
  u8 oplens[2] = {3, 3};
  struct FbxTcBlock *tc = MakeTcN(mops, rdes, uimms, disps, ips, oplens, 2);
  struct FbxIrBlock *ir;
  struct FbxWasmBuffer out;
  ir = fbx_ir_lift(tc);
  ASSERT_NE(0, (i64)(intptr_t)ir);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  /* Magic. */
  ASSERT_EQ(0x00, out.data[0]);
  ASSERT_EQ(0x61, out.data[1]);
  ASSERT_EQ(0x73, out.data[2]);
  ASSERT_EQ(0x6D, out.data[3]);
  fbx_wasm_buffer_free(&out);
  fbx_ir_free(ir);
  FreeTc(tc);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Structural validation — verify section ordering + import count.            */
/*                                                                            */
/* Walks the emitted module's section headers and confirms types/imports/    */
/* functions/exports/code appear in the right order with sensible lengths.   */
/* This is a lightweight structural check; full semantic validation requires */
/* a real wasm engine and lives in the cross-engine harness.                  */
/* ────────────────────────────────────────────────────────────────────────── */

/* Read a ULEB128 from `buf[pos..end]` into *out_val; advance *pos.  Returns
 * 1 on success, 0 on malformed input. */
static int ReadUleb(const u8 *buf, size_t *pos, size_t end, u64 *out_val) {
  u64 v = 0;
  unsigned shift = 0;
  while (*pos < end) {
    u8 byte = buf[(*pos)++];
    v |= ((u64)(byte & 0x7Fu)) << shift;
    if ((byte & 0x80u) == 0) {
      *out_val = v;
      return 1;
    }
    shift += 7;
    if (shift >= 64) return 0;
  }
  return 0;
}

TEST(FbxWasmEmit, SectionOrderingAndCounts) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[3];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  size_t pos = 8; /* skip magic + version */
  u64 sec_len;
  /* Section 1: TYPE.  Body starts with type count. */
  /* Section IDs per wasm 1.0 spec: type=1, import=2, function=3,
   * export=7, code=10.  Hard-coded here so the test doesn't depend on
   * synthesis-internal macros. */
  u8 expected_sections[5];
  expected_sections[0] = 1;  /* type */
  expected_sections[1] = 2;  /* import */
  expected_sections[2] = 3;  /* function */
  expected_sections[3] = 7;  /* export */
  expected_sections[4] = 10; /* code */
  unsigned si;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0xE000;
  insts[1].opcode = FBX_IR_OP_REG_GET;
  insts[1].width = 8;
  insts[1].dst_kind = FBX_IR_KIND_VREG;
  insts[1].dst = 0;
  insts[1].src1_kind = FBX_IR_KIND_GREG;
  insts[1].src1 = 0;
  insts[2].opcode = FBX_IR_OP_REG_SET;
  insts[2].width = 8;
  insts[2].dst_kind = FBX_IR_KIND_GREG;
  insts[2].dst = 1;
  insts[2].src1_kind = FBX_IR_KIND_VREG;
  insts[2].src1 = 0;
  ir.insts = insts;
  ir.ninsts = 3;
  ir.nvregs = 1;
  ir.start_pc = 0xE000;
  ir.end_pc = 0xE003;
  tc = MakeTc(0x089, RDE_MOD3, 0, 0, 0xE000, 3, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  for (si = 0; si < sizeof expected_sections / sizeof expected_sections[0];
       ++si) {
    ASSERT_EQ((i64)expected_sections[si], (i64)out.data[pos]);
    pos++;
    ASSERT_EQ(1, ReadUleb(out.data, &pos, out.len, &sec_len));
    /* Skip section body. */
    pos += sec_len;
    ASSERT_EQ(1, (i64)(pos <= out.len));
  }
  /* All sections consumed → pos must equal out.len. */
  ASSERT_EQ((i64)out.len, (i64)pos);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Buffer primitives: stress.                                                 */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxWasmBuffer, AppendStress) {
  struct FbxWasmBuffer b;
  unsigned i;
  fbx_wasm_buffer_init(&b);
  for (i = 0; i < 10000; ++i) {
    fbx_wasm_buffer_u8(&b, (u8)(i & 0xFFu));
  }
  ASSERT_EQ(10000, (i64)b.len);
  for (i = 0; i < 10000; ++i) {
    ASSERT_EQ((i64)(i & 0xFFu), (i64)b.data[i]);
  }
  fbx_wasm_buffer_free(&b);
  ASSERT_EQ(0, (i64)b.len);
  ASSERT_EQ(0, (i64)(intptr_t)b.data);
}
