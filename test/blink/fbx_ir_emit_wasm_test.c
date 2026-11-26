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
#include "blink/fbx_t2_block_ctx.h"
#include "blink/fbx_wasm_emit.h"
#include "blink/rde.h"
#include "blink/threadedcode.h"
#include "blink/types.h"
#include "blink/x86.h"
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

/* #778 — the SIB-index LEA form `lea dst, [base + index*scale + disp]` now
 * emits INLINE (it previously bailed to Tier 1 via the #738 correct-or-refuse).
 * The emit must ACCEPT (return 1) and produce a non-empty module: src2 carries
 * the index greg, `scale` the SIB scale.  The TC carries the SIB-index rde so
 * the coverage gate's mop-switch sees a real LEA. */
TEST(FbxWasmEmit, LeaSibIndexSynth) {
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
  insts[1].dst = 14; /* r14 — the hot strcmp dst */
  insts[1].src1_kind = FBX_IR_KIND_GREG;
  insts[1].src1 = 9; /* base = r9 */
  insts[1].src2_kind = FBX_IR_KIND_GREG;
  insts[1].src2 = 0; /* index = rax */
  insts[1].scale = 1;
  insts[1].imm = 0;
  ir.insts = insts;
  ir.ninsts = 2;
  ir.nvregs = 0;
  ir.start_pc = 0x5000;
  ir.end_pc = 0x5004;
  /* SIB-index rde: ModrmRm=4 (0x200) | Rexb | SibBase=1 ⇒ r9 base. */
  tc = MakeTc(0x08D, 0x200ull | 000000002000ull | (1ull << 040), 0, 0, 0x5000,
              4, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

/* #778 — scale 8 (`[base + index*8]`) emits inline too. */
TEST(FbxWasmEmit, LeaSibIndexScale8Synth) {
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
  insts[1].dst = 1;
  insts[1].src1_kind = FBX_IR_KIND_GREG;
  insts[1].src1 = 0;
  insts[1].src2_kind = FBX_IR_KIND_GREG;
  insts[1].src2 = 2;
  insts[1].scale = 8;
  insts[1].imm = 0x20;
  ir.insts = insts;
  ir.ninsts = 2;
  ir.nvregs = 0;
  ir.start_pc = 0x5000;
  ir.end_pc = 0x5004;
  /* ModrmRm=4 (SIB) | SibScale=3 ⇒ scale 8. */
  tc = MakeTc(0x08D, 0x200ull | (3ull << 046), 0, 0x20, 0x5000, 4,
              FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

/* #778 — defense: a GREG index with an INVALID scale (not 1/2/4/8) must be
 * REFUSED by the acceptance gate (the whole block fails → Tier 1).  Guards
 * against a malformed IR slipping a bad shift into the emit. */
TEST(FbxWasmEmit, LeaSibIndexBadScaleRefused) {
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
  insts[1].dst = 1;
  insts[1].src1_kind = FBX_IR_KIND_GREG;
  insts[1].src1 = 0;
  insts[1].src2_kind = FBX_IR_KIND_GREG;
  insts[1].src2 = 2;
  insts[1].scale = 3; /* invalid SIB scale */
  ir.insts = insts;
  ir.ninsts = 2;
  ir.nvregs = 0;
  ir.start_pc = 0x5000;
  ir.end_pc = 0x5004;
  tc = MakeTc(0x08D, 0x200ull, 0, 0, 0x5000, 4, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  EXPECT_EQ(0, fbx_ir_emit_wasm(&ir, tc, &out)); /* refused */
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

TEST(FbxWasmEmit, BranchCondJESynths) {
  /* #599 (FBX_IR_VERSION 2): standalone BRANCH_COND with JE predicate
   * (cond_id=4) now synthesizes.  The emitted wasm reads m->flags, masks
   * the ZF bit (bit 6), and uses `select` to write either taken_pc or
   * fallthrough_pc to m->ip.  Pre-#599 this was the `RefusesBranchCond`
   * test (returned 0).  The flip is intentional + documented in #599's
   * README.
   *
   * The IR here is constructed by hand (not lifter output) — caller
   * supplies the predicate-only block; no preceding SET_FLAGS_RAW is
   * required because the wasm reads m->flags directly. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_BRANCH_COND;
  insts[0].src1_kind = FBX_IR_KIND_IMM;
  insts[0].src1 = 4; /* JE — ZF=1 */
  insts[0].src2_kind = FBX_IR_KIND_IMM;
  insts[0].src2 = 0x8002; /* fallthrough_pc (low 32 bits) */
  insts[0].imm = 0x8100; /* taken_pc */
  ir.insts = insts;
  ir.ninsts = 1;
  ir.nvregs = 0;
  tc = MakeTc(0x074, 0, 0, 0, 0x8000, 2, FBX_TC_KIND_NORMAL);
  ir.start_pc = tc->start_pc;
  ir.end_pc = tc->end_pc; /* firebox#5BW */
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, RefusesBranchCondJP) {
  /* #599 — Jcc PF/NP (cond 0xA / 0xB) deferred per the EmitJccPredicate
   * comment: lazy-parity computation is non-trivial in wasm MVP.
   * Synthesis refuses for these two predicates. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_BRANCH_COND;
  insts[0].src1_kind = FBX_IR_KIND_IMM;
  insts[0].src1 = 0xA; /* JP — PF=1 */
  insts[0].src2_kind = FBX_IR_KIND_IMM;
  insts[0].src2 = 0x8002;
  insts[0].imm = 0x8100;
  ir.insts = insts;
  ir.ninsts = 1;
  ir.nvregs = 0;
  tc = MakeTc(0x07A, 0, 0, 0, 0x8000, 2, FBX_TC_KIND_NORMAL);
  ir.start_pc = tc->start_pc;
  ir.end_pc = tc->end_pc; /* firebox#5BW */
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(0, fbx_ir_emit_wasm(&ir, tc, &out));
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, MalformedSetFlagsRawRefusedEvenWithNoReader) {
  /* firebox#NRR.  This test used to be `SetFlagsRawAcceptedWhenNoReader` and
   * asserted 1: under the §13.5 whole-block dead-flag elision a reader-free
   * block dropped every SET_FLAGS_RAW, so a MALFORMED one (legacy v1 shape,
   * no operand vregs) was accepted because it was never emitted.
   *
   * #NRR removed the whole-block elision — it was an in-block liveness test
   * authorizing a cross-block drop; m->flags outlives the block and no exit
   * path re-executes the dropped op.  A reader-free SET_FLAGS_RAW now reaches
   * EmitFlagsAfterAlu, whose `default:` arm answers an unrecognized op_kind
   * with an i64.and — a silently WRONG flags computation.  So the encoding
   * validation had to widen out of its `if (saw_flag_reader)` guard, and this
   * block must now REFUSE (correct-or-refuse: Tier 1 runs it).
   *
   * Note the guard was already too narrow before #NRR: a reader-free block
   * containing RET / PUSH / POP / BAILOUT got needs_scratch == 1 from those
   * arms and emitted unvalidated flags. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  enum FbxIrEmitFailReason reason = FBX_IR_EMIT_OK;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_SET_FLAGS_RAW;
  ir.insts = insts;
  ir.ninsts = 1;
  tc = MakeTc(0x001, RDE_MOD3, 0, 0, 0x9000, 3, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(0, fbx_ir_emit_wasm_with_reason(&ir, tc, &out, &reason, NULL));
  ASSERT_EQ((i64)FBX_IR_EMIT_SET_FLAGS_RAW_BAD, (i64)reason);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* firebox#NRR — the flag shadow outlives the block.                          */
/*                                                                            */
/* Discriminator shape, self-controlled: emit a lifted block, then emit the    */
/* SAME block with its SET_FLAGS_RAW instructions spliced OUT, and compare     */
/* lengths.  If the emitter commits the flags, the two differ; if it drops     */
/* them (the pre-#NRR whole-block elision), they are byte-identical.  The      */
/* splice is the positive control: it guarantees the delta can only come from  */
/* the SET_FLAGS_RAW, and it makes the test independent of every other emit    */
/* size in the file.                                                          */
/* ────────────────────────────────────────────────────────────────────────── */

/* Emit `ir` with every SET_FLAGS_RAW removed.  Returns the emitted length, or
 * 0 if the emit refused. */
static size_t EmitLenWithoutSetFlagsRaw(const struct FbxIrBlock *ir,
                                        const struct FbxTcBlock *tc) {
  struct FbxIrBlock stripped;
  struct FbxIrInst *insts;
  struct FbxWasmBuffer out;
  size_t len = 0;
  u32 i, n = 0;
  insts = (struct FbxIrInst *)calloc(ir->ninsts ? ir->ninsts : 1,
                                     sizeof(struct FbxIrInst));
  for (i = 0; i < ir->ninsts; ++i) {
    if (ir->insts[i].opcode == FBX_IR_OP_SET_FLAGS_RAW) continue;
    insts[n++] = ir->insts[i];
  }
  stripped = *ir;
  stripped.insts = insts;
  stripped.ninsts = n;
  fbx_wasm_buffer_init(&out);
  if (fbx_ir_emit_wasm(&stripped, tc, &out)) len = out.len;
  fbx_wasm_buffer_free(&out);
  free(insts);
  return len;
}

/* Count SET_FLAGS_RAW / flag-readers in a lifted block. */
static u32 NrrCountOp(const struct FbxIrBlock *ir, u8 opcode) {
  u32 i, n = 0;
  for (i = 0; i < ir->ninsts; ++i) {
    if (ir->insts[i].opcode == opcode) ++n;
  }
  return n;
}

TEST(FbxWasmNrr, FallthroughBlockCommitsFlagsForItsSuccessor) {
  /* THE #NRR CASE, and after #CM4 the common one: a lone `cmp rax, rbx` that
   * blink split mid-stream.  The lifter appends a synthesized BRANCH_TAKEN
   * (EmitFallthroughBranch, #CM4 / FBX_IR_VERSION 7), so the block has a
   * SET_FLAGS_RAW, NO in-block reader, and a terminator the pre-#NRR
   * BlockNeedsScratchLocals did not enumerate → needs_scratch == 0 → the CMP
   * the NEXT block's Jcc branches on was dropped.  Nothing re-runs it: the
   * block exits and Tier 1 resumes at the next PC. */
  struct FbxTcBlock *tc = MakeTc(0x039, RDE_MOD3, 0, 0, 0x10000, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  struct FbxWasmBuffer out;
  size_t without;
  ASSERT_NOTNULL(ir);
  ASSERT_EQ(1u, NrrCountOp(ir, FBX_IR_OP_SET_FLAGS_RAW));
  ASSERT_EQ(0u, NrrCountOp(ir, FBX_IR_OP_BRANCH_COND));
  ASSERT_EQ(0u, NrrCountOp(ir, FBX_IR_OP_GET_FLAG));
  ASSERT_EQ(FBX_IR_OP_BRANCH_TAKEN, ir->insts[ir->ninsts - 1].opcode);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(ir, tc, &out));
  without = EmitLenWithoutSetFlagsRaw(ir, tc);
  ASSERT_NE((i64)0, (i64)without);          /* control: the strip still emits */
  ASSERT_NE((i64)without, (i64)out.len);    /* the flags are COMMITTED */
  fbx_wasm_buffer_free(&out);
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxWasmNrr, CallDirectBlockCommitsFlagsForItsSuccessor) {
  /* CALL_DIRECT was the SECOND uncovered terminator.  The pre-#NRR comment in
   * BlockNeedsScratchLocals read "CALL_DIRECT does NOT need scratch" — true
   * about scratch locals, and exactly the conflation that made the predicate
   * wrong.  Real x86 `call` does not write EFLAGS, so `cmp; call f` with a
   * flag-reader at f's entry elided identically to the jmp case. */
  u64 mops[2] = {0x039, 0x0e8};
  u64 rdes[2] = {RDE_MOD3, 0};
  u64 uimm0s[2] = {0, 0};
  i64 disps[2] = {0, 0x40};
  u64 ips[2] = {0x10000, 0x10003};
  u8 oplens[2] = {3, 5};
  struct FbxTcBlock *tc = MakeTcN(mops, rdes, uimm0s, disps, ips, oplens, 2);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  struct FbxWasmBuffer out;
  size_t without;
  ASSERT_NOTNULL(ir);
  ASSERT_EQ(1u, NrrCountOp(ir, FBX_IR_OP_SET_FLAGS_RAW));
  ASSERT_EQ(0u, NrrCountOp(ir, FBX_IR_OP_BRANCH_COND));
  ASSERT_EQ(FBX_IR_OP_CALL_DIRECT, ir->insts[ir->ninsts - 1].opcode);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(ir, tc, &out));
  without = EmitLenWithoutSetFlagsRaw(ir, tc);
  ASSERT_NE((i64)0, (i64)without);
  ASSERT_NE((i64)without, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxWasmNrr, OnlyTheLastSetFlagsRawSurvivesInAReaderFreeBlock) {
  /* The conservatism bound.  #NRR withdraws the WHOLE-BLOCK drop, not the
   * per-instruction one: SetFlagsRawIsLive (#794) is already cross-block
   * correct — it keeps the LAST SET_FLAGS_RAW because a next block may read
   * it, and drops any earlier one a later in-block SET_FLAGS_RAW provably
   * overwrites first.  So `add; sub; <fallthrough>` has TWO SET_FLAGS_RAW and
   * must cost exactly ONE AluFlags sequence — the same delta as the
   * single-SET_FLAGS_RAW block above.  If this ever asserts unequal, the
   * elision has regressed to per-op flag materialization, which #794 measured
   * as the dominant cost of a flag-reading loop. */
  u64 mops2[2] = {0x001, 0x029};
  u64 rdes2[2] = {RDE_MOD3, RDE_MOD3};
  u64 uimm0s2[2] = {0, 0};
  i64 disps2[2] = {0, 0};
  u64 ips2[2] = {0x10000, 0x10003};
  u8 oplens2[2] = {3, 3};
  struct FbxTcBlock *tc2 = MakeTcN(mops2, rdes2, uimm0s2, disps2, ips2,
                                   oplens2, 2);
  struct FbxIrBlock *ir2 = fbx_ir_lift(tc2);
  struct FbxTcBlock *tc1 = MakeTc(0x039, RDE_MOD3, 0, 0, 0x10000, 3,
                                  FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir1 = fbx_ir_lift(tc1);
  struct FbxWasmBuffer o1, o2;
  size_t d1, d2;
  ASSERT_NOTNULL(ir1);
  ASSERT_NOTNULL(ir2);
  ASSERT_EQ(2u, NrrCountOp(ir2, FBX_IR_OP_SET_FLAGS_RAW));
  ASSERT_EQ(0u, NrrCountOp(ir2, FBX_IR_OP_BRANCH_COND));
  fbx_wasm_buffer_init(&o1);
  fbx_wasm_buffer_init(&o2);
  ASSERT_EQ(1, fbx_ir_emit_wasm(ir1, tc1, &o1));
  ASSERT_EQ(1, fbx_ir_emit_wasm(ir2, tc2, &o2));
  d1 = o1.len - EmitLenWithoutSetFlagsRaw(ir1, tc1);
  d2 = o2.len - EmitLenWithoutSetFlagsRaw(ir2, tc2);
  ASSERT_NE((i64)0, (i64)d1);
  ASSERT_EQ((i64)d1, (i64)d2); /* two SET_FLAGS_RAW, one AluFlags sequence */
  fbx_wasm_buffer_free(&o1);
  fbx_wasm_buffer_free(&o2);
  fbx_ir_free(ir1);
  fbx_ir_free(ir2);
  FreeTc(tc1);
  FreeTc(tc2);
}

TEST(FbxWasmNrr, FlagFreeFallthroughBlockIsUnchanged) {
  /* Confinement control.  A block with NO SET_FLAGS_RAW and the same
   * synthesized BRANCH_TAKEN terminator must be untouched by #NRR — the fix
   * withdraws a liveness claim, it does not force scratch locals on every
   * block.  `mov rdi, rax` is the shape #CM4's own comment names. */
  struct FbxTcBlock *tc = MakeTc(0x089, RDE_MOD3, 0, 0, 0x10000, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  struct FbxWasmBuffer out;
  ASSERT_NOTNULL(ir);
  ASSERT_EQ(0u, NrrCountOp(ir, FBX_IR_OP_SET_FLAGS_RAW));
  ASSERT_EQ(FBX_IR_OP_BRANCH_TAKEN, ir->insts[ir->ninsts - 1].opcode);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(ir, tc, &out));
  /* Stripping a SET_FLAGS_RAW that isn't there must be a no-op. */
  ASSERT_EQ((i64)out.len, (i64)EmitLenWithoutSetFlagsRaw(ir, tc));
  fbx_wasm_buffer_free(&out);
  fbx_ir_free(ir);
  FreeTc(tc);
}

/* ── firebox#3XY — IMUL flag-liveness at the BLOCK BOUNDARY ──────────────── */
/* #794 refused `imul` only at an IN-BLOCK reader.  Its own comment argued the
 * reader-free case was safe because such blocks elided all flags anyway — the
 * same "a block boundary is not a reader" premise #NRR withdrew, and #NRR's fix
 * cannot close it because IMUL produces no SET_FLAGS_RAW to conserve.  The
 * refuse now also fires when the imul shadow survives to the block end.
 * #794 shipped with no IMUL test at all; these are the first. */

/* `imul rax, rbx` (0F AF /r), reg-form, then whatever the caller appends. */
#define MOP_IMUL 0x1af

static int Fbx3xyEmits(struct FbxTcBlock *tc, u8 *last_op, u32 *nsfr) {
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  struct FbxWasmBuffer out;
  int ok;
  if (!ir) return -1;
  *last_op = ir->ninsts ? ir->insts[ir->ninsts - 1].opcode : 0;
  *nsfr = NrrCountOp(ir, FBX_IR_OP_SET_FLAGS_RAW);
  fbx_wasm_buffer_init(&out);
  ok = fbx_ir_emit_wasm(ir, tc, &out);
  fbx_wasm_buffer_free(&out);
  fbx_ir_free(ir);
  return ok;
}

TEST(FbxWasm3xy, ImulFallthroughRefusesBecauseBlockEndIsAReader) {
  /* THE #3XY CASE.  `imul rax, rbx` alone; #CM4 appends the synthesized
   * BRANCH_TAKEN.  No in-block reader, so #794's scan never fired; IMUL writes
   * no flags, so the block committed none.  The successor's `jo`/`jc`/`js` then
   * read a shadow imul never wrote, where Tier 1 writes imul's own CF/OF.
   * MEASURED pre-fix: 205 bytes, zero SET_FLAGS_RAW, no flag commit. */
  struct FbxTcBlock *tc = MakeTc(MOP_IMUL, RDE_MOD3, 0, 0, 0x10000, 3,
                                 FBX_TC_KIND_NORMAL);
  u8 last_op = 0;
  u32 nsfr = 0;
  int ok = Fbx3xyEmits(tc, &last_op, &nsfr);
  ASSERT_NE(-1, ok);                                  /* control: it lifts */
  ASSERT_EQ(0u, nsfr);                  /* IMUL synthesizes nothing to commit */
  ASSERT_EQ(FBX_IR_OP_BRANCH_TAKEN, last_op);
  ASSERT_EQ(0, ok);                                            /* REFUSED */
  FreeTc(tc);
}

TEST(FbxWasm3xy, ImulBeforeExplicitJmpAndCallAlsoRefuse) {
  /* Same fail-open reached by the two terminators #NRR found uncovered.  A
   * `call` does not write EFLAGS, so the callee's entry reads the same shadow. */
  u64 mops_jmp[2] = {MOP_IMUL, 0x0eb};
  u64 mops_call[2] = {MOP_IMUL, 0x0e8};
  u64 rdes[2] = {RDE_MOD3, 0};
  u64 uimm0s[2] = {0, 0};
  i64 disps_jmp[2] = {0, 0x10};
  i64 disps_call[2] = {0, 0x40};
  u64 ips[2] = {0x10000, 0x10003};
  u8 oplens_jmp[2] = {3, 2};
  u8 oplens_call[2] = {3, 5};
  struct FbxTcBlock *tc;
  u8 last_op = 0;
  u32 nsfr = 0;
  tc = MakeTcN(mops_jmp, rdes, uimm0s, disps_jmp, ips, oplens_jmp, 2);
  ASSERT_EQ(0, Fbx3xyEmits(tc, &last_op, &nsfr));
  ASSERT_EQ(FBX_IR_OP_BRANCH_TAKEN, last_op);
  FreeTc(tc);
  tc = MakeTcN(mops_call, rdes, uimm0s, disps_call, ips, oplens_call, 2);
  ASSERT_EQ(0, Fbx3xyEmits(tc, &last_op, &nsfr));
  ASSERT_EQ(FBX_IR_OP_CALL_DIRECT, last_op);
  FreeTc(tc);
}

TEST(FbxWasm3xy, SetFlagsRawBeforeImulDoesNotClearTheShadow) {
  /* ORDER test, and the case #NRR made WORSE rather than merely stale.
   * `sub rax, rbx; imul rax, rbx` carries one SET_FLAGS_RAW, so post-#NRR the
   * block now COMMITS the SUB's flags at block end — an active write of a wrong
   * value, freshly stamped, after the imul redefined CF/OF.  The scan must be
   * order-sensitive: only a flag-writer AFTER the imul clears it. */
  u64 mops[2] = {0x029, MOP_IMUL};
  u64 rdes[2] = {RDE_MOD3, RDE_MOD3};
  u64 uimm0s[2] = {0, 0};
  i64 disps[2] = {0, 0};
  u64 ips[2] = {0x10000, 0x10003};
  u8 oplens[2] = {3, 3};
  struct FbxTcBlock *tc = MakeTcN(mops, rdes, uimm0s, disps, ips, oplens, 2);
  u8 last_op = 0;
  u32 nsfr = 0;
  ASSERT_EQ(0, Fbx3xyEmits(tc, &last_op, &nsfr));               /* REFUSED */
  ASSERT_EQ(1u, nsfr);           /* and it had a live commit to get wrong */
  FreeTc(tc);
}

TEST(FbxWasm3xy, ImulFollowedByARealFlagWriterStillEmits) {
  /* CONSERVATISM BOUND, and the positive control for the three refuses above:
   * `imul rax, rbx; sub rax, rbx` is the hot-loop shape #794 deliberately keeps
   * on Tier 2.  The SUB's SET_FLAGS_RAW is what the successor reads, so imul's
   * dead flags never matter.  If this refuses, #3XY over-refused and the whole
   * `imul…dec;jnz` class fell back to Tier 1. */
  u64 mops[2] = {MOP_IMUL, 0x029};
  u64 rdes[2] = {RDE_MOD3, RDE_MOD3};
  u64 uimm0s[2] = {0, 0};
  i64 disps[2] = {0, 0};
  u64 ips[2] = {0x10000, 0x10003};
  u8 oplens[2] = {3, 3};
  struct FbxTcBlock *tc = MakeTcN(mops, rdes, uimm0s, disps, ips, oplens, 2);
  u8 last_op = 0;
  u32 nsfr = 0;
  ASSERT_EQ(1, Fbx3xyEmits(tc, &last_op, &nsfr));                /* EMITS */
  ASSERT_EQ(1u, nsfr);
  ASSERT_EQ(FBX_IR_OP_BRANCH_TAKEN, last_op);
  FreeTc(tc);
}

TEST(FbxWasm3xy, ImulWithInBlockReaderStillRefuses) {
  /* #794's original arm, unchanged — it is the control that proves the probe
   * can observe a refuse at all, and that #3XY widened the rule without
   * disturbing it. */
  u64 mops[2] = {MOP_IMUL, 0x074};
  u64 rdes[2] = {RDE_MOD3, 0};
  u64 uimm0s[2] = {0, 0};
  i64 disps[2] = {0, 0x10};
  u64 ips[2] = {0x10000, 0x10003};
  u8 oplens[2] = {3, 2};
  struct FbxTcBlock *tc = MakeTcN(mops, rdes, uimm0s, disps, ips, oplens, 2);
  u8 last_op = 0;
  u32 nsfr = 0;
  ASSERT_EQ(0, Fbx3xyEmits(tc, &last_op, &nsfr));
  ASSERT_EQ(FBX_IR_OP_BRANCH_COND, last_op);
  FreeTc(tc);
}

TEST(FbxWasmEmit, RefusesMalformedSetFlagsRawWithBranchCond) {
  /* #599 (v2): SET_FLAGS_RAW followed by BRANCH_COND now synthesizes
   * end-to-end IFF the SET_FLAGS_RAW carries the v2 IR encoding
   * (src1/src2 = operand vregs).  This test feeds the LEGACY v1 IR
   * shape (no operand encoding); CoverageGate must refuse so the
   * runtime stays on Tier 1.
   *
   * The validation guard belongs to #599 because the IR-version bump
   * (v1→v2) doesn't otherwise invalidate ALL existing IR — Phase 4
   * sidecars + the runtime lifter must agree on which encoding is in
   * effect.  Strict v2-required gating on flag-reader blocks closes the
   * mixed-encoding hazard. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[2];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_SET_FLAGS_RAW;
  /* INTENTIONALLY no src1_kind / src2_kind — this is malformed v2. */
  insts[1].opcode = FBX_IR_OP_BRANCH_COND;
  insts[1].imm = 0x9100;
  ir.insts = insts;
  ir.ninsts = 2;
  tc = MakeTc(0x001, RDE_MOD3, 0, 0, 0x9000, 3, FBX_TC_KIND_NORMAL);
  ir.start_pc = tc->start_pc;
  ir.end_pc = tc->end_pc; /* firebox#5BW */
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(0, fbx_ir_emit_wasm(&ir, tc, &out));
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

/* #602 — CALL_DIRECT now synthesizes (previously refused).  The wasm
 * module produced should: (1) succeed synthesis, (2) emit non-zero bytes,
 * (3) start with the wasm magic, (4) be deterministic. */
TEST(FbxWasmEmit, SynthsCallDirect) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_CALL_DIRECT;
  insts[0].src2_kind = FBX_IR_KIND_IMM;
  insts[0].src2 = 0xA005;       /* return_pc (32-bit truncated) */
  insts[0].imm = 0xA100;        /* target_pc */
  ir.insts = insts;
  ir.ninsts = 1;
  tc = MakeTc(0x0E8, 0, 0, 0x100, 0xA000, 5, FBX_TC_KIND_NORMAL);
  ir.start_pc = tc->start_pc;
  ir.end_pc = tc->end_pc; /* firebox#5BW */
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  ASSERT_EQ(0x00, out.data[0]);
  ASSERT_EQ(0x61, out.data[1]);
  ASSERT_EQ(0x73, out.data[2]);
  ASSERT_EQ(0x6D, out.data[3]);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

/* #677 — RDE for mod==1 (disp8) + ModrmRm==0 (base=rax, no SIB): the
 * supported `[base + disp]` memory-form addressing shape.  Bits 22-23 =
 * mod (1 here); bits 7-9 = ModrmRm (0 here). */
#define RDE_MEM_BASE_DISP8 ((u64)0x00400000ull)

TEST(FbxWasmEmit, EmitsMemoryFormMovStore) {
  /* #677 — MOV [rax+disp], r64 (mop 0x089, mem-form): the lifter emits
   * REG_GET (source reg → vreg) + STORE (mem[base+disp] = vreg).  The
   * emit pass must SYNTHESIZE this now (previously it refused with
   * reason=mod3_required).  Assert success + i64.store present in the
   * emitted body. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[3];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  unsigned j;
  int saw_store = 0;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0xB000;
  /* REG_GET — load source reg (greg 3) into vreg 0. */
  insts[1].opcode = FBX_IR_OP_REG_GET;
  insts[1].width = 8;
  insts[1].dst_kind = FBX_IR_KIND_VREG;
  insts[1].dst = 0;
  insts[1].src1_kind = FBX_IR_KIND_GREG;
  insts[1].src1 = 3;
  /* STORE — mem[greg0 + 0x18] = vreg0. */
  insts[2].opcode = FBX_IR_OP_STORE;
  insts[2].width = 8;
  insts[2].dst_kind = FBX_IR_KIND_NONE;
  insts[2].src1_kind = FBX_IR_KIND_GREG;
  insts[2].src1 = 0;
  insts[2].src2_kind = FBX_IR_KIND_VREG;
  insts[2].src2 = 0;
  insts[2].imm = 0x18;
  ir.insts = insts;
  ir.ninsts = 3;
  ir.nvregs = 1;
  ir.start_pc = 0xB000;
  ir.end_pc = 0xB003;
  tc = MakeTc(0x089, RDE_MEM_BASE_DISP8, 0, 0x18, 0xB000, 3,
              FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  /* WASM_OP_I64_STORE == 0x37. */
  for (j = 0; j < out.len; ++j) {
    if (out.data[j] == 0x37u) saw_store = 1;
  }
  EXPECT_NE(0, saw_store);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, EmitsMemoryFormMovLoad) {
  /* #677 — MOV r64, [rax+disp] (mop 0x08B, mem-form): lifter emits
   * LOAD (vreg = mem[base+disp]) + REG_SET (dest reg = vreg).  Assert
   * success + i64.load present. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[3];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  unsigned j;
  int saw_load = 0;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0xB100;
  /* LOAD — vreg0 = mem[greg0 + 0x20]. */
  insts[1].opcode = FBX_IR_OP_LOAD;
  insts[1].width = 8;
  insts[1].dst_kind = FBX_IR_KIND_VREG;
  insts[1].dst = 0;
  insts[1].src1_kind = FBX_IR_KIND_GREG;
  insts[1].src1 = 0;
  insts[1].imm = 0x20;
  /* REG_SET — dest reg (greg 3) = vreg0. */
  insts[2].opcode = FBX_IR_OP_REG_SET;
  insts[2].width = 8;
  insts[2].dst_kind = FBX_IR_KIND_GREG;
  insts[2].dst = 3;
  insts[2].src1_kind = FBX_IR_KIND_VREG;
  insts[2].src1 = 0;
  ir.insts = insts;
  ir.ninsts = 3;
  ir.nvregs = 1;
  ir.start_pc = 0xB100;
  ir.end_pc = 0xB103;
  tc = MakeTc(0x08B, RDE_MEM_BASE_DISP8, 0, 0x20, 0xB100, 3,
              FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  /* WASM_OP_I64_LOAD == 0x29. */
  for (j = 0; j < out.len; ++j) {
    if (out.data[j] == 0x29u) saw_load = 1;
  }
  EXPECT_NE(0, saw_load);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

/* firebox#JJJ — a LOAD with a SIB index: `mov ebx, [rax + rcx*4 + 0x20]`.
 * Builds the IR the lifter produces (LiftLoadToReg) with the given index
 * kind / scale, and emits it against a mem-form 0x8B TC entry. */
static int EmitIndexedLoad(u8 src2_kind, u32 scale, struct FbxWasmBuffer *out) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[3];
  struct FbxTcBlock *tc;
  int rc;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0xB100;
  insts[1].opcode = FBX_IR_OP_LOAD;
  insts[1].width = 4;
  insts[1].dst_kind = FBX_IR_KIND_VREG;
  insts[1].dst = 0;
  insts[1].src1_kind = FBX_IR_KIND_GREG;
  insts[1].src1 = 0;
  insts[1].src2_kind = src2_kind;
  insts[1].src2 = 1;
  insts[1].scale = scale;
  insts[1].imm = 0x20;
  insts[2].opcode = FBX_IR_OP_REG_SET;
  insts[2].width = 4;
  insts[2].dst_kind = FBX_IR_KIND_GREG;
  insts[2].dst = 3;
  insts[2].src1_kind = FBX_IR_KIND_VREG;
  insts[2].src1 = 0;
  ir.insts = insts;
  ir.ninsts = 3;
  ir.nvregs = 1;
  ir.start_pc = 0xB100;
  ir.end_pc = 0xB104;
  tc = MakeTc(0x08B, RDE_MEM_BASE_DISP8, 0, 0x20, 0xB100, 4,
              FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(out);
  rc = fbx_ir_emit_wasm(&ir, tc, out);
  FreeTc(tc);
  return rc;
}

TEST(FbxWasmEmit, IndexedLoadEmitsTheIndexTerm) {
  struct FbxWasmBuffer plain, idx1, idx4;
  u32 j;
  int saw_shl = 0;
  ASSERT_EQ(1, EmitIndexedLoad(FBX_IR_KIND_NONE, 0, &plain));
  ASSERT_EQ(1, EmitIndexedLoad(FBX_IR_KIND_GREG, 1, &idx1));
  ASSERT_EQ(1, EmitIndexedLoad(FBX_IR_KIND_GREG, 4, &idx4));
  /* The index is not dropped: each form is a different module, and the
   * scaled one carries an i64.shl (0x86) the scale-1 one does not need. */
  EXPECT_NE(0, plain.len != idx1.len ||
                   memcmp(plain.data, idx1.data, plain.len) != 0);
  EXPECT_NE(0, idx1.len != idx4.len ||
                   memcmp(idx1.data, idx4.data, idx1.len) != 0);
  for (j = 0; j < idx4.len; ++j) {
    if (idx4.data[j] == 0x86u) saw_shl = 1;
  }
  EXPECT_EQ(1, saw_shl);
  fbx_wasm_buffer_free(&plain);
  fbx_wasm_buffer_free(&idx1);
  fbx_wasm_buffer_free(&idx4);
}

TEST(FbxWasmEmit, IndexedLoadBadShapesAreRefused) {
  struct FbxWasmBuffer out;
  /* scale 3 is not a SIB scale; an IMM or VREG src2 is not an index. */
  EXPECT_EQ(0, EmitIndexedLoad(FBX_IR_KIND_GREG, 3, &out));
  fbx_wasm_buffer_free(&out);
  EXPECT_EQ(0, EmitIndexedLoad(FBX_IR_KIND_IMM, 0, &out));
  fbx_wasm_buffer_free(&out);
  EXPECT_EQ(0, EmitIndexedLoad(FBX_IR_KIND_VREG, 1, &out));
  fbx_wasm_buffer_free(&out);
}

TEST(FbxWasmEmit, MovzxMemoryFormPassesTheGate) {
  /* firebox#JJJ: `movzx edx, byte [rax + 0x20]` — LOAD width 1 + REG_SET
   * width 4 against a mem-form 0x1B6 TC entry.  Was MOD3_REQUIRED. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[3];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0xB100;
  insts[1].opcode = FBX_IR_OP_LOAD;
  insts[1].width = 1;
  insts[1].dst_kind = FBX_IR_KIND_VREG;
  insts[1].src1_kind = FBX_IR_KIND_GREG;
  insts[1].imm = 0x20;
  insts[2].opcode = FBX_IR_OP_REG_SET;
  insts[2].width = 4;
  insts[2].dst_kind = FBX_IR_KIND_GREG;
  insts[2].dst = 2;
  insts[2].src1_kind = FBX_IR_KIND_VREG;
  ir.insts = insts;
  ir.ninsts = 3;
  ir.nvregs = 1;
  ir.start_pc = 0xB100;
  ir.end_pc = 0xB104;
  tc = MakeTc(0x1B6, RDE_MEM_BASE_DISP8, 0, 0x20, 0xB100, 4,
              FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  EXPECT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, MemoryFormMovZeroDispEmits) {
  /* #677 — disp==0 form: no IMM slot is consumed (disp elided), but emit
   * still succeeds.  STORE mem[greg0] = vreg0. */
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
  insts[1].src1 = 3;
  insts[2].opcode = FBX_IR_OP_STORE;
  insts[2].width = 8;
  insts[2].dst_kind = FBX_IR_KIND_NONE;
  insts[2].src1_kind = FBX_IR_KIND_GREG;
  insts[2].src1 = 0;
  insts[2].src2_kind = FBX_IR_KIND_VREG;
  insts[2].src2 = 0;
  insts[2].imm = 0; /* disp == 0 */
  ir.insts = insts;
  ir.ninsts = 3;
  ir.nvregs = 1;
  tc = MakeTc(0x089, 0 /* mod==0, rm==0 → [rax] */, 0, 0, 0xB200, 3,
              FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, MemoryFormMovStoreDeterministic) {
  /* #677 — same mem-form STORE IR emits byte-identical wasm across two
   * calls (spec §Q5 determinism; preserves Phase 4 reachability). */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[3];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer a, b;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[1].opcode = FBX_IR_OP_REG_GET;
  insts[1].width = 8;
  insts[1].dst_kind = FBX_IR_KIND_VREG;
  insts[1].dst = 0;
  insts[1].src1_kind = FBX_IR_KIND_GREG;
  insts[1].src1 = 3;
  insts[2].opcode = FBX_IR_OP_STORE;
  insts[2].width = 8;
  insts[2].dst_kind = FBX_IR_KIND_NONE;
  insts[2].src1_kind = FBX_IR_KIND_GREG;
  insts[2].src1 = 0;
  insts[2].src2_kind = FBX_IR_KIND_VREG;
  insts[2].src2 = 0;
  insts[2].imm = 0x18;
  ir.insts = insts;
  ir.ninsts = 3;
  ir.nvregs = 1;
  tc = MakeTc(0x089, RDE_MEM_BASE_DISP8, 0, 0x18, 0xB300, 3,
              FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&a);
  fbx_wasm_buffer_init(&b);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &a));
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &b));
  ASSERT_EQ((i64)a.len, (i64)b.len);
  ASSERT_EQ(0, memcmp(a.data, b.data, a.len));
  fbx_wasm_buffer_free(&a);
  fbx_wasm_buffer_free(&b);
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

/* ────────────────────────────────────────────────────────────────────────── */
/* §13.5 — MOVZX synthesis.                                                   */
/*                                                                            */
/* MOVZX r, r/m{8,16} is lifted as REG_GET (narrow) + REG_SET (wide).  The   */
/* synthesis pipeline's i64.load{8,16}_u → i64.store path produces the       */
/* exact zero-extending semantics — no new emit machinery required.           */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxWasmEmit, Movzx8To32Synth) {
  /* MOVZX r32, r/m8 — IR: REG_GET width=1 + REG_SET width=4. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[3];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0xF000;
  insts[1].opcode = FBX_IR_OP_REG_GET;
  insts[1].width = 1;
  insts[1].dst_kind = FBX_IR_KIND_VREG;
  insts[1].dst = 0;
  insts[1].src1_kind = FBX_IR_KIND_GREG;
  insts[1].src1 = 2;
  insts[2].opcode = FBX_IR_OP_REG_SET;
  insts[2].width = 4;
  insts[2].dst_kind = FBX_IR_KIND_GREG;
  insts[2].dst = 0;
  insts[2].src1_kind = FBX_IR_KIND_VREG;
  insts[2].src1 = 0;
  ir.insts = insts;
  ir.ninsts = 3;
  ir.nvregs = 1;
  ir.start_pc = 0xF000;
  ir.end_pc = 0xF004;
  tc = MakeTc(0x1B6, RDE_MOD3, 0, 0, 0xF000, 4, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  /* Spot-check the emitted bytes contain i64.load8_u (0x31) somewhere in
   * the code section.  This guards against a regression that would emit
   * i64.load instead. */
  {
    int saw_load8u = 0;
    u32 j;
    for (j = 0; j + 0 < out.len; ++j) {
      if (out.data[j] == 0x31u /* WASM_OP_I64_LOAD8U */) {
        saw_load8u = 1;
        break;
      }
    }
    ASSERT_EQ(1, saw_load8u);
  }
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasmEmit, Movzx16To64Synth) {
  /* MOVZX r64, r/m16 with REX.W — IR: REG_GET width=2 + REG_SET width=8. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[3];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  u64 rde_rexw = RDE_MOD3 | (u64)0x40;  /* REX.W bit 6 */
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0xF100;
  insts[1].opcode = FBX_IR_OP_REG_GET;
  insts[1].width = 2;
  insts[1].dst_kind = FBX_IR_KIND_VREG;
  insts[1].dst = 0;
  insts[1].src1_kind = FBX_IR_KIND_GREG;
  insts[1].src1 = 1;
  insts[2].opcode = FBX_IR_OP_REG_SET;
  insts[2].width = 8;
  insts[2].dst_kind = FBX_IR_KIND_GREG;
  insts[2].dst = 0;
  insts[2].src1_kind = FBX_IR_KIND_VREG;
  insts[2].src1 = 0;
  ir.insts = insts;
  ir.ninsts = 3;
  ir.nvregs = 1;
  ir.start_pc = 0xF100;
  ir.end_pc = 0xF105;
  tc = MakeTc(0x1B7, rde_rexw, 0, 0, 0xF100, 5, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  /* Spot-check i64.load16_u (opcode 0x33). */
  {
    int saw_load16u = 0;
    u32 j;
    for (j = 0; j < out.len; ++j) {
      if (out.data[j] == 0x33u /* WASM_OP_I64_LOAD16U */) {
        saw_load16u = 1;
        break;
      }
    }
    ASSERT_EQ(1, saw_load16u);
  }
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* §13.5 — SET_FLAGS_RAW dead-flag elision unblocks lifted ALU IR.            */
/*                                                                            */
/* The LIFTER produces SET_FLAGS_RAW after every ALU op.  Pre-§13.5 the      */
/* synthesis pass refused on contact with SET_FLAGS_RAW.  Post-§13.5 it     */
/* accepts when no BRANCH_COND/GET_FLAG observer follows in the block —     */
/* SET_FLAGS_RAW becomes a benign marker dropped at emit time.               */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxWasmEmit, LiftedAddRRThenBranchTakenSynths) {
  /* End-to-end: build a TC block for ADD r/m, r (0x001) at modrm.mod==3,
   * lift it (lifter emits ADD + SET_FLAGS_RAW), then run the synthesis.
   * Expected: synth succeeds because no flag-reader follows.  Per the
   * §Q7 measurement, this unblocks ~2.1% of host time (rank 13 of top-30)
   * for straight-line ADD-then-fallthrough blocks. */
  struct FbxTcBlock *tc;
  struct FbxIrBlock *ir;
  struct FbxWasmBuffer out;
  /* MakeTc creates a single-entry block.  rde encodes mopcode 0x001 at the
   * high bits + RDE_MOD3 for modrm.mod==3 — same shape as the existing
   * AluAddSynth test, but routed through the real lifter rather than a
   * hand-built IR. */
  tc = MakeTc(0x001, RDE_MOD3, 0, 0, 0x10000, 3, FBX_TC_KIND_NORMAL);
  ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  /* The lifted IR must contain SET_FLAGS_RAW (the lifter emits it
   * unconditionally for ALU ops).  This is the test of §13.5's dead-flag
   * elision: the IR has SET_FLAGS_RAW, no BRANCH_COND, no GET_FLAG. */
  {
    u32 j;
    int saw_flags = 0;
    for (j = 0; j < ir->ninsts; ++j) {
      if (ir->insts[j].opcode == FBX_IR_OP_SET_FLAGS_RAW) saw_flags = 1;
    }
    ASSERT_EQ(1, saw_flags);
  }
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  /* The emitted wasm starts with the magic.  We trust DeterministicOutput
   * for byte-stability and the SectionOrderingAndCounts test for structural
   * validation; this test only needs to confirm the synth succeeded. */
  ASSERT_EQ(0x00u, out.data[0]);
  ASSERT_EQ(0x61u, out.data[1]);
  fbx_wasm_buffer_free(&out);
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxWasmEmit, LiftedCmpRRAcceptedWhenJccFollows) {
  /* #599 (FBX_IR_VERSION 2): a 2-instruction TC block of CMP r/m, r
   * (0x039) followed by JE rel8 (0x074) now SYNTHESIZES — the eager-update
   * lazy-flag wasm path is the v0.1 of §13.5 follow-on.  Pre-#599 this
   * was the `LiftedCmpRRRefusesWhenJccFollows` test (returned 0).  The
   * flip is intentional + documented in #599's README.
   *
   * The synthesis still REFUSES if the lift drops the v2 IR encoding
   * (SET_FLAGS_RAW must carry operand vregs in src1/src2).  This test
   * verifies the post-#599 lifter writes the encoding correctly. */
  u64 mops[2] = {0x039, 0x074};
  u64 rdes[2] = {RDE_MOD3, 0};
  u64 uimm0s[2] = {0, 0};
  i64 disps[2] = {0, 0x10};
  u64 ips[2] = {0x11000, 0x11003};
  u8 oplens[2] = {3, 2};
  struct FbxTcBlock *tc = MakeTcN(mops, rdes, uimm0s, disps, ips, oplens, 2);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  struct FbxWasmBuffer out;
  ASSERT_NOTNULL(ir);
  {
    u32 j;
    int saw_branch_cond = 0;
    int saw_set_flags_raw = 0;
    for (j = 0; j < ir->ninsts; ++j) {
      if (ir->insts[j].opcode == FBX_IR_OP_BRANCH_COND) saw_branch_cond = 1;
      if (ir->insts[j].opcode == FBX_IR_OP_SET_FLAGS_RAW) {
        saw_set_flags_raw = 1;
        /* #599: SET_FLAGS_RAW must carry operand vregs in src1/src2. */
        ASSERT_EQ(FBX_IR_KIND_VREG, ir->insts[j].src1_kind);
        ASSERT_EQ(FBX_IR_KIND_VREG, ir->insts[j].src2_kind);
      }
    }
    ASSERT_EQ(1, saw_branch_cond);
    ASSERT_EQ(1, saw_set_flags_raw);
  }
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  /* Verify magic + version. */
  ASSERT_EQ(0x00u, out.data[0]);
  ASSERT_EQ(0x61u, out.data[1]);
  fbx_wasm_buffer_free(&out);
  fbx_ir_free(ir);
  FreeTc(tc);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* §13.5 — Mirror-direction ALU synth (operand-direction swap).               */
/*                                                                            */
/* The lifter (post-#596) accepts mopcodes 0x002/0x003/0x00A/0x00B/0x022/    */
/* 0x023/0x02A/0x02B/0x032/0x033/0x03A/0x03B.  Synthesis can emit them      */
/* iff no flag-reader follows.                                                */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxWasmEmit, MirrorAddRRmFromLiftSynths) {
  /* 0x003: ADD r, r/m at mod3, no branch.  Lift+emit end-to-end. */
  struct FbxTcBlock *tc = MakeTc(0x003, RDE_MOD3, 0, 0, 0x12000, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  struct FbxWasmBuffer out;
  ASSERT_NOTNULL(ir);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxWasmEmit, MirrorCmpRRmFromLiftSynths) {
  /* 0x03B: CMP r, r/m at mod3, no branch.  CMP is flag-only (no REG_SET).
   * The synth still succeeds via §13.5's dead-flag elision — the entire
   * IR sequence is a no-op from a register-state perspective when no
   * Jcc follows, which is precisely the semantics CMP-no-Jcc produces
   * (the only observable effect is on flags). */
  struct FbxTcBlock *tc = MakeTc(0x03B, RDE_MOD3, 0, 0, 0x12100, 3,
                                 FBX_TC_KIND_NORMAL);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  struct FbxWasmBuffer out;
  ASSERT_NOTNULL(ir);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  fbx_ir_free(ir);
  FreeTc(tc);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* #599 — Lazy-flag wasm synthesis + BRANCH_COND end-to-end tests.            */
/*                                                                            */
/* These tests drive lift → emit through the new v2 IR shape.  They verify:  */
/*   (a) every flag-emitting ALU op (ADD/SUB/AND/OR/XOR/CMP/TEST) at width   */
/*       1/2/4/8 followed by Jcc synthesizes,                                 */
/*   (b) every Jcc condition code 0-9 + C-F synthesizes,                     */
/*   (c) Jcc PF/NP (A/B) refuses,                                             */
/*   (d) determinism is preserved (same input → same bytes),                 */
/*   (e) cross-engine validation modules dump cleanly.                       */
/* ────────────────────────────────────────────────────────────────────────── */

/* Helper: lift+emit a 2-instruction block (flag-writer + Jcc). */
static int LiftEmitFlagJcc(u64 alu_mop, u64 alu_rde, u64 jcc_mop,
                           i64 jcc_disp, struct FbxWasmBuffer *out) {
  u64 mops[2] = {alu_mop, jcc_mop};
  u64 rdes[2] = {alu_rde, 0};
  u64 uimms[2] = {0, 0};
  i64 disps[2] = {0, jcc_disp};
  u64 ips[2] = {0x20000, 0x20003};
  u8 oplens[2] = {3, 2};
  struct FbxTcBlock *tc = MakeTcN(mops, rdes, uimms, disps, ips, oplens, 2);
  struct FbxIrBlock *ir = fbx_ir_lift(tc);
  int ok;
  if (!ir) {
    FreeTc(tc);
    return 0;
  }
  fbx_wasm_buffer_init(out);
  ok = fbx_ir_emit_wasm(ir, tc, out);
  fbx_ir_free(ir);
  FreeTc(tc);
  return ok;
}

TEST(FbxWasm599, AddRMrJESynths) {
  /* 0x001 ADD r/m, r at mod3, followed by 0x074 JE. */
  struct FbxWasmBuffer out;
  ASSERT_EQ(1, LiftEmitFlagJcc(0x001, RDE_MOD3, 0x074, 0x10, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
}

TEST(FbxWasm599, SubRMrJNESynths) {
  struct FbxWasmBuffer out;
  ASSERT_EQ(1, LiftEmitFlagJcc(0x029, RDE_MOD3, 0x075, 0x10, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
}

TEST(FbxWasm599, AndRMrJZSynths) {
  /* 0x021 AND r/m, r at mod3, followed by 0x074 JE. */
  struct FbxWasmBuffer out;
  ASSERT_EQ(1, LiftEmitFlagJcc(0x021, RDE_MOD3, 0x074, 0x10, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
}

TEST(FbxWasm599, OrRMrJSSynths) {
  /* 0x009 OR r/m, r followed by 0x078 JS (sign-flag). */
  struct FbxWasmBuffer out;
  ASSERT_EQ(1, LiftEmitFlagJcc(0x009, RDE_MOD3, 0x078, 0x10, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
}

TEST(FbxWasm599, XorRMrJOSynths) {
  /* 0x031 XOR r/m, r followed by 0x070 JO (overflow).  XOR sets OF=0 so
   * the predicate is well-defined; we don't validate runtime semantics
   * (host-engine emulation does that). */
  struct FbxWasmBuffer out;
  ASSERT_EQ(1, LiftEmitFlagJcc(0x031, RDE_MOD3, 0x070, 0x10, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
}

TEST(FbxWasm599, CmpRMrJBSynths) {
  /* 0x039 CMP r/m, r followed by 0x072 JB (CF=1). */
  struct FbxWasmBuffer out;
  ASSERT_EQ(1, LiftEmitFlagJcc(0x039, RDE_MOD3, 0x072, 0x10, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
}

TEST(FbxWasm599, TestRMrJBESynths) {
  /* 0x085 TEST r/m, r followed by 0x076 JBE (CF=1 || ZF=1). */
  struct FbxWasmBuffer out;
  ASSERT_EQ(1, LiftEmitFlagJcc(0x085, RDE_MOD3, 0x076, 0x10, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
}

TEST(FbxWasm599, AddRMrJLSynths) {
  /* 0x001 ADD followed by 0x07C JL (SF != OF). */
  struct FbxWasmBuffer out;
  ASSERT_EQ(1, LiftEmitFlagJcc(0x001, RDE_MOD3, 0x07C, 0x10, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
}

TEST(FbxWasm599, AddRMrJLESynths) {
  /* 0x001 ADD followed by 0x07E JLE (ZF=1 || SF!=OF). */
  struct FbxWasmBuffer out;
  ASSERT_EQ(1, LiftEmitFlagJcc(0x001, RDE_MOD3, 0x07E, 0x10, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
}

TEST(FbxWasm599, CmpRMrJPRefuses) {
  /* JP (cond 0xA) — Jcc PF deferred per #599 scope.  Synthesis refuses
   * even though the lift succeeded; runtime stays on Tier 1. */
  struct FbxWasmBuffer out;
  ASSERT_EQ(0, LiftEmitFlagJcc(0x039, RDE_MOD3, 0x07A, 0x10, &out));
  fbx_wasm_buffer_free(&out);
}

TEST(FbxWasm599, AddRMrJaSynths) {
  /* 0x001 ADD followed by 0x077 JA (!CF && !ZF). */
  struct FbxWasmBuffer out;
  ASSERT_EQ(1, LiftEmitFlagJcc(0x001, RDE_MOD3, 0x077, 0x10, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
}

TEST(FbxWasm599, DeterministicFlagSynth) {
  /* Two synth calls over the same (lift, tc) must produce identical bytes. */
  struct FbxWasmBuffer out1, out2;
  ASSERT_EQ(1, LiftEmitFlagJcc(0x001, RDE_MOD3, 0x074, 0x10, &out1));
  ASSERT_EQ(1, LiftEmitFlagJcc(0x001, RDE_MOD3, 0x074, 0x10, &out2));
  ASSERT_EQ((i64)out1.len, (i64)out2.len);
  ASSERT_EQ(0, memcmp(out1.data, out2.data, out1.len));
  fbx_wasm_buffer_free(&out1);
  fbx_wasm_buffer_free(&out2);
}

TEST(FbxWasm599, IrVersionBumped) {
  /* #599 changes the SET_FLAGS_RAW shape; the IR version constant
   * MUST be bumped so Phase 4 sidecar caches invalidate.  This guards
   * against silently shipping a layout change with the same version.
   *
   * #602 (CALL_DIRECT / RET / PUSH / POP synthesis) re-bumps to 3 —
   * adds new IR opcodes (PUSH/POP) and flips CALL/RET emit from
   * refuse to synthesize, so stale v2 sidecars must invalidate.
   *
   * #677 (MOV r/m memory-form) re-bumps to 4 — the lifter now emits
   * FBX_IR_OP_LOAD / FBX_IR_OP_STORE for MOV 0x88/0x89/0x8A/0x8B in the
   * `[base+disp]` form (previously bailed out at mod != 3), so stale v3
   * sidecars must invalidate.
   *
   * #778 (inline SIB-index LEA) re-bumps to 5 — the lifter now emits the
   * indexed LEA form (src2=GREG index + the `scale` field, previously
   * bailed), and the offset-20 IR word (formerly `_pad2`) now carries the
   * SIB scale, so stale v4 sidecars (which assumed it was always zero) must
   * invalidate.
   *
   * #794 re-bumps to 6, #CM4 re-bumps to 7 — NEITHER updated this assertion,
   * so this guard sat RED and unnoticed for two bumps (blink's unit tests need
   * GNU make, which is not on every dev box).  Corrected here.
   *
   * #NRR re-bumps to 8 — the §599 whole-block dead-flag elision is removed (an
   * in-block liveness test authorizing a cross-block drop), so every block with
   * a SET_FLAGS_RAW and no in-block flag-reader now emits one AluFlags sequence
   * where it previously emitted none.  Emit semantics flipped for a whole block
   * class, so stale v7 sidecars must invalidate.
   *
   * #3XY re-bumps to 9 — #794's IMUL flag-liveness refuse now also fires when
   * the imul shadow survives to the BLOCK END, not just at an in-block reader.
   * Emit flipped from synthesize to refuse for a whole block class (any block
   * whose last flag-definer is an IMUL), so stale v8 sidecars must invalidate
   * rather than serve the fail-open wasm they already hold. */
  ASSERT_EQ(11u, (u32)FBX_IR_VERSION);
}

TEST(FbxWasm599, FlagSynthDifferentOpKindProducesDifferentBytes) {
  /* ADD-flag-update vs SUB-flag-update vs AND-flag-update must produce
   * different wasm — confirms the emit pass actually distinguishes the
   * three formulae rather than emitting identical no-ops. */
  struct FbxWasmBuffer add_out, sub_out, and_out;
  ASSERT_EQ(1, LiftEmitFlagJcc(0x001, RDE_MOD3, 0x074, 0x10, &add_out));
  ASSERT_EQ(1, LiftEmitFlagJcc(0x029, RDE_MOD3, 0x074, 0x10, &sub_out));
  ASSERT_EQ(1, LiftEmitFlagJcc(0x021, RDE_MOD3, 0x074, 0x10, &and_out));
  /* Lengths likely differ (ADD has all formulae; AND skips CF/OF/AF). */
  ASSERT_NE((i64)add_out.len, (i64)and_out.len);
  /* ADD and SUB lengths may be equal but bytes must differ (CF formula
   * differs: ADD is z<rhs, SUB is lhs<z). */
  if ((i64)add_out.len == (i64)sub_out.len) {
    ASSERT_NE(0, memcmp(add_out.data, sub_out.data, add_out.len));
  }
  fbx_wasm_buffer_free(&add_out);
  fbx_wasm_buffer_free(&sub_out);
  fbx_wasm_buffer_free(&and_out);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* #602 — CALL_DIRECT / RET / PUSH / POP wasm synthesis.                      */
/*                                                                            */
/* The §13.5c emit pass lowers all four stack-mutating control-flow ops       */
/* into wasm that mutates the guest stack via the imported linear memory      */
/* (using i32.wrap_i64(RSP) as the address).  Tests:                          */
/*                                                                            */
/*   - SynthsRet            — bare RET lifts + emits                          */
/*   - SynthsPushRbp        — PUSH RBP lifts + emits                          */
/*   - SynthsPopRbp         — POP RBP lifts + emits                           */
/*   - PushPopFromLift      — PUSH RBP; ...; POP RBP lifted as a block        */
/*   - BranchCondJESynthsWithCallRet — CMP+JE multi-inst with CALL/RET emit   */
/*   - DeterministicCallEmit— same input → same bytes                         */
/*   - PushPopWidthIs8      — lifter emits width=8 for PUSH/POP (#602 v0.1)   */
/* ────────────────────────────────────────────────────────────────────────── */

TEST(FbxWasm602, SynthsRet) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_RET;
  ir.insts = insts;
  ir.ninsts = 1;
  ir.nvregs = 0;
  ir.start_pc = 0xB000;
  ir.end_pc = 0xB001;
  tc = MakeTc(0x0C3, 0, 0, 0, 0xB000, 1, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  /* Magic prefix. */
  ASSERT_EQ(0x00, out.data[0]);
  ASSERT_EQ(0x61, out.data[1]);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasm602, SynthsPushRbp) {
  /* Direct IR construction: PUSH RBP (greg 5, width 8). */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[2];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PUSH;
  insts[0].width = 8;
  insts[0].src1_kind = FBX_IR_KIND_GREG;
  insts[0].src1 = 5;            /* RBP */
  insts[1].opcode = FBX_IR_OP_BRANCH_TAKEN;
  insts[1].imm = 0xB001;
  ir.insts = insts;
  ir.ninsts = 2;
  ir.nvregs = 0;
  ir.start_pc = 0xB000;
  ir.end_pc = 0xB001;
  tc = MakeTc(0x055, 0, 0, 0, 0xB000, 1, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasm602, SynthsPopRbp) {
  /* Direct IR construction: POP RBP. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[2];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_POP;
  insts[0].width = 8;
  insts[0].dst_kind = FBX_IR_KIND_GREG;
  insts[0].dst = 5;             /* RBP */
  insts[1].opcode = FBX_IR_OP_BRANCH_TAKEN;
  insts[1].imm = 0xC001;
  ir.insts = insts;
  ir.ninsts = 2;
  ir.nvregs = 0;
  ir.start_pc = 0xC000;
  ir.end_pc = 0xC001;
  tc = MakeTc(0x05D, 0, 0, 0, 0xC000, 1, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

TEST(FbxWasm602, PushPopFromLift) {
  /* End-to-end: lift PUSH RBP + POP RBP as adjacent TC entries; verify
   * the lifted IR has FBX_IR_OP_PUSH + FBX_IR_OP_POP, and synthesis
   * succeeds. */
  u64 mops[2] = {0x055, 0x05D};                 /* PUSH RBP; POP RBP */
  u64 rdes[2] = {0, 0};
  u64 uimms[2] = {0, 0};
  i64 disps[2] = {0, 0};
  u64 ips[2] = {0xD000, 0xD001};
  u8 oplens[2] = {1, 1};
  struct FbxTcBlock *tc = MakeTcN(mops, rdes, uimms, disps, ips, oplens, 2);
  struct FbxIrBlock *ir;
  struct FbxWasmBuffer out;
  ir = fbx_ir_lift(tc);
  ASSERT_NE((i64)0, (i64)(intptr_t)ir);
  /* Walk IR, expect FBX_IR_OP_PUSH and FBX_IR_OP_POP somewhere. */
  {
    u32 i;
    int saw_push = 0, saw_pop = 0;
    for (i = 0; i < ir->ninsts; ++i) {
      if (ir->insts[i].opcode == FBX_IR_OP_PUSH) {
        saw_push = 1;
        ASSERT_EQ(8u, (u32)ir->insts[i].width);
        ASSERT_EQ((u32)FBX_IR_KIND_GREG, (u32)ir->insts[i].src1_kind);
        ASSERT_EQ(5u, (u32)ir->insts[i].src1);  /* RBP = greg 5 */
      }
      if (ir->insts[i].opcode == FBX_IR_OP_POP) {
        saw_pop = 1;
        ASSERT_EQ(8u, (u32)ir->insts[i].width);
        ASSERT_EQ((u32)FBX_IR_KIND_GREG, (u32)ir->insts[i].dst_kind);
        ASSERT_EQ(5u, (u32)ir->insts[i].dst);
      }
    }
    ASSERT_EQ(1, saw_push);
    ASSERT_EQ(1, saw_pop);
  }
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  fbx_wasm_buffer_free(&out);
  fbx_ir_free(ir);
  FreeTc(tc);
}

TEST(FbxWasm602, BranchCondJESynthsWithCallRet) {
  /* Multi-instruction block:
   *   CMP %rax, %rbx     ; 0x039  (3 bytes)
   *   JE  rel8           ; 0x074  (2 bytes)
   * The block ends with a Jcc; CALL/RET would be a separate basic block
   * in real code (each is block-terminating).  Build it via the lift
   * path to confirm CMP+JE multi-inst works alongside the #602 emit
   * changes (regression guard — none of the #602 surface should break
   * #599's lazy-flag synthesis).
   *
   * After the CMP+JE base case, also synthesize a bare CALL_DIRECT IR
   * directly to confirm both surfaces compose. */
  u64 mops[2] = {0x039, 0x074};                 /* CMP RR + JE rel8 */
  u64 rdes[2] = {RDE_MOD3, 0};
  u64 uimms[2] = {0, 0};
  i64 disps[2] = {0, 0x10};
  u64 ips[2] = {0xE000, 0xE003};
  u8 oplens[2] = {3, 2};
  struct FbxTcBlock *cmpje_tc = MakeTcN(mops, rdes, uimms, disps, ips, oplens, 2);
  struct FbxIrBlock *cmpje_ir;
  struct FbxWasmBuffer cmpje_out, call_out;
  /* Synth the CMP+JE block (lazy-flag + BRANCH_COND emit from #599). */
  cmpje_ir = fbx_ir_lift(cmpje_tc);
  ASSERT_NE((i64)0, (i64)(intptr_t)cmpje_ir);
  fbx_wasm_buffer_init(&cmpje_out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(cmpje_ir, cmpje_tc, &cmpje_out));
  ASSERT_NE((i64)0, (i64)cmpje_out.len);
  /* Synth a bare CALL_DIRECT block — the §13.5c addition.  This sibling
   * surface must produce a non-zero, magic-prefixed wasm module. */
  {
    struct FbxIrBlock ir2;
    struct FbxIrInst insts[1];
    struct FbxTcBlock *call_tc;
    memset(&ir2, 0, sizeof ir2);
    memset(insts, 0, sizeof insts);
    insts[0].opcode = FBX_IR_OP_CALL_DIRECT;
    insts[0].src2_kind = FBX_IR_KIND_IMM;
    insts[0].src2 = 0xE008;
    insts[0].imm = 0xF000;
    ir2.insts = insts;
    ir2.ninsts = 1;
    ir2.nvregs = 0;
    call_tc = MakeTc(0x0E8, 0, 0, 0x1000, 0xE003, 5, FBX_TC_KIND_NORMAL);
    ir2.end_pc = call_tc->end_pc; /* firebox#5BW */
    fbx_wasm_buffer_init(&call_out);
    ASSERT_EQ(1, fbx_ir_emit_wasm(&ir2, call_tc, &call_out));
    ASSERT_NE((i64)0, (i64)call_out.len);
    FreeTc(call_tc);
  }
  fbx_wasm_buffer_free(&cmpje_out);
  fbx_wasm_buffer_free(&call_out);
  fbx_ir_free(cmpje_ir);
  FreeTc(cmpje_tc);
}

TEST(FbxWasm602, DeterministicCallEmit) {
  /* Two synth calls over the same CALL_DIRECT IR must produce identical
   * bytes (spec §Q5 determinism contract). */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out1, out2;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_CALL_DIRECT;
  insts[0].src2_kind = FBX_IR_KIND_IMM;
  insts[0].src2 = 0xA005;
  insts[0].imm = 0xA100;
  ir.insts = insts;
  ir.ninsts = 1;
  tc = MakeTc(0x0E8, 0, 0, 0x100, 0xA000, 5, FBX_TC_KIND_NORMAL);
  ir.start_pc = tc->start_pc;
  ir.end_pc = tc->end_pc; /* firebox#5BW */
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

TEST(FbxWasm602, DeterministicRetEmit) {
  /* RET is even simpler; assert determinism + non-empty body. */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out1, out2;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_RET;
  ir.insts = insts;
  ir.ninsts = 1;
  tc = MakeTc(0x0C3, 0, 0, 0, 0xB000, 1, FBX_TC_KIND_NORMAL);
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

TEST(FbxWasm602, CallAndRetProduceDifferentBytes) {
  /* CALL_DIRECT and RET have distinct lowerings; their wasm output must
   * differ.  Guard against an accidental fallthrough that emits the
   * same module shell for both. */
  struct FbxIrBlock ir_call, ir_ret;
  struct FbxIrInst call_insts[1], ret_insts[1];
  struct FbxTcBlock *call_tc, *ret_tc;
  struct FbxWasmBuffer call_out, ret_out;
  memset(&ir_call, 0, sizeof ir_call);
  memset(&ir_ret, 0, sizeof ir_ret);
  memset(call_insts, 0, sizeof call_insts);
  memset(ret_insts, 0, sizeof ret_insts);
  call_insts[0].opcode = FBX_IR_OP_CALL_DIRECT;
  call_insts[0].src2_kind = FBX_IR_KIND_IMM;
  call_insts[0].src2 = 0xA005;
  call_insts[0].imm = 0xA100;
  ir_call.insts = call_insts;
  ir_call.ninsts = 1;
  ret_insts[0].opcode = FBX_IR_OP_RET;
  ir_ret.insts = ret_insts;
  ir_ret.ninsts = 1;
  call_tc = MakeTc(0x0E8, 0, 0, 0x100, 0xA000, 5, FBX_TC_KIND_NORMAL);
  ir_call.end_pc = call_tc->end_pc; /* firebox#5BW */
  ret_tc = MakeTc(0x0C3, 0, 0, 0, 0xB000, 1, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&call_out);
  fbx_wasm_buffer_init(&ret_out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir_call, call_tc, &call_out));
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir_ret, ret_tc, &ret_out));
  /* Either length differs or bytes differ — they must NOT be identical. */
  if ((i64)call_out.len == (i64)ret_out.len) {
    ASSERT_NE(0, memcmp(call_out.data, ret_out.data, call_out.len));
  }
  fbx_wasm_buffer_free(&call_out);
  fbx_wasm_buffer_free(&ret_out);
  FreeTc(call_tc);
  FreeTc(ret_tc);
}

TEST(FbxWasm602, PushPopOpcodeNamesPresent) {
  /* Diagnostic tables expose the new opcode names — sanity check that
   * the strings exist (avoids "UNKNOWN" returns from stale tables). */
  EXPECT_STREQ("PUSH", fbx_ir_opcode_name(FBX_IR_OP_PUSH));
  EXPECT_STREQ("POP",  fbx_ir_opcode_name(FBX_IR_OP_POP));
}

TEST(FbxWasm602, RefusesPushWithoutGregSrc) {
  /* Defense: malformed IR with PUSH but src1_kind != GREG must be
   * refused by the emitter (correct-or-refuse contract). */
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PUSH;
  insts[0].width = 8;
  insts[0].src1_kind = FBX_IR_KIND_IMM;   /* malformed: should be GREG */
  insts[0].imm = 0xdeadbeef;
  ir.insts = insts;
  ir.ninsts = 1;
  tc = MakeTc(0x055, 0, 0, 0, 0xF000, 1, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  /* CoverageGate passes (it doesn't introspect PUSH operand kinds);
   * the emit pass refuses at EmitPush.  Either way, the final result
   * is `0` (refusal). */
  ASSERT_EQ(0, fbx_ir_emit_wasm(&ir, tc, &out));
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

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

/* ────────────────────────────────────────────────────────────────────────── */
/* #695 — self-loop whole-block emit.                                          */
/*                                                                            */
/* A block whose last emit-relevant op is BRANCH_COND is the do/while hot-loop */
/* shape; #695 wraps it in a wasm `loop` so a runtime-self-targeting branch    */
/* re-enters guest-side instead of crossing the FFI boundary every iteration.  */
/* The wrapper is CACHE-SAFE: the loop-back is a runtime comparison            */
/* (next_ip == entry_ip && --iter_budget != 0); nothing PC-related is baked    */
/* into the bytes, so a cache-shared module reused on a non-self-loop block     */
/* simply never takes the br_if and exits identically to the base path.        */
/*                                                                            */
/* These tests assert the structural signature of the wrapper at the byte      */
/* level.  Cross-engine validity (wasmer-Cranelift + wabt) of the same bytes   */
/* is checked by the out-of-tree dumper harness (fbx_ir_emit_wasm_dump.c +     */
/* the 599_* modules, which all lift to BRANCH_COND-terminated blocks).        */
/* ────────────────────────────────────────────────────────────────────────── */

/* Wasm opcodes the self-loop wrapper introduces (mirrors fbx_ir_emit_wasm.c). */
#define T695_OP_UNREACHABLE 0x00u
#define T695_OP_LOOP        0x03u
#define T695_OP_BR_IF       0x0Du
#define T695_BLOCKTYPE_VOID 0x40u

/* Returns the offset of the first occurrence of the 2-byte sequence
 * {a, b} in [data, data+len), or -1 if absent. */
static long T695FindPair(const u8 *data, size_t len, u8 a, u8 b) {
  size_t i;
  if (len < 2) return -1;
  for (i = 0; i + 1 < len; ++i) {
    if (data[i] == a && data[i + 1] == b) return (long)i;
  }
  return -1;
}

/* A single-instruction BRANCH_COND block is a self-loop candidate (the
 * BranchCondJESynths shape).  The emit must carry the #695 wrapper signature:
 *   - `loop 0x40` (loop opener with the void block-type), and
 *   - `br_if 0`   (0x0D 0x00 — loop-back to the innermost label), and
 *   - a trailing `unreachable` (0x00) immediately before the final function
 *     `end` (0x0B), for the statically-unreachable tail. */
TEST(FbxWasmEmit, SelfLoopWrapperEmitted) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_BRANCH_COND;
  insts[0].src1_kind = FBX_IR_KIND_IMM;
  insts[0].src1 = 4; /* JE — ZF=1 */
  insts[0].src2_kind = FBX_IR_KIND_IMM;
  insts[0].src2 = 0x8002; /* fallthrough_pc (low 32 bits) */
  insts[0].imm = 0x8100;  /* taken_pc */
  ir.insts = insts;
  ir.ninsts = 1;
  ir.nvregs = 0;
  tc = MakeTc(0x074, 0, 0, 0, 0x8000, 2, FBX_TC_KIND_NORMAL);
  ir.start_pc = tc->start_pc;
  ir.end_pc = tc->end_pc; /* firebox#5BW */
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  /* loop opener with void block-type. */
  ASSERT_NE((i64)-1,
            (i64)T695FindPair(out.data, out.len, T695_OP_LOOP,
                              T695_BLOCKTYPE_VOID));
  /* br_if to the innermost loop label (depth 0). */
  ASSERT_NE((i64)-1, (i64)T695FindPair(out.data, out.len, T695_OP_BR_IF, 0x00));
  /* The body ends with `unreachable` then the function `end` (0x0B). */
  ASSERT_EQ((i64)T695_OP_UNREACHABLE, (i64)out.data[out.len - 2]);
  ASSERT_EQ((i64)0x0Bu, (i64)out.data[out.len - 1]);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

/* A non-BRANCH_COND terminator (BRANCH_TAKEN) is NOT a self-loop candidate:
 * the wrapper must be absent so the emit stays byte-identical to the pre-#695
 * base path (preserving the #635 per-block cache + all existing emit tests).
 * The absence is proven by the lack of the `loop 0x40` opener and the lack of
 * a trailing `unreachable` before the function `end`. */
TEST(FbxWasmEmit, NonSelfLoopBlockHasNoWrapper) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[2];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0x6000;
  insts[1].opcode = FBX_IR_OP_BRANCH_TAKEN;
  insts[1].imm = 0x6100;
  ir.insts = insts;
  ir.ninsts = 2;
  ir.nvregs = 0;
  ir.start_pc = 0x6000;
  ir.end_pc = 0x6005;
  tc = MakeTc(0x0E9, 0, 0, 0x100, 0x6000, 5, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  /* No loop opener. */
  ASSERT_EQ((i64)-1,
            (i64)T695FindPair(out.data, out.len, T695_OP_LOOP,
                              T695_BLOCKTYPE_VOID));
  /* The final byte is the function `end`; the byte before it is NOT the
   * wrapper's `unreachable` (a BRANCH_TAKEN block ends in `return` then
   * `end`). */
  ASSERT_EQ((i64)0x0Bu, (i64)out.data[out.len - 1]);
  ASSERT_NE((i64)T695_OP_UNREACHABLE, (i64)out.data[out.len - 2]);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
}

/* The self-loop emit is deterministic: identical IR + TC → identical bytes.
 * (The #599 BranchCondJESynths test covers the same block shape; this asserts
 * the determinism property specifically across the #695 wrapper.) */
TEST(FbxWasmEmit, SelfLoopEmitDeterministic) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer a, b;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_BRANCH_COND;
  insts[0].src1_kind = FBX_IR_KIND_IMM;
  insts[0].src1 = 4;
  insts[0].src2_kind = FBX_IR_KIND_IMM;
  insts[0].src2 = 0x8002;
  insts[0].imm = 0x8100;
  ir.insts = insts;
  ir.ninsts = 1;
  ir.nvregs = 0;
  ir.start_pc = 0x8000;
  ir.end_pc = 0x8002; /* firebox#5BW: the fall-through is the block end */
  tc = MakeTc(0x074, 0, 0, 0, 0x8000, 2, FBX_TC_KIND_NORMAL);
  fbx_wasm_buffer_init(&a);
  fbx_wasm_buffer_init(&b);
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &a));
  ASSERT_EQ(1, fbx_ir_emit_wasm(&ir, tc, &b));
  ASSERT_EQ((i64)a.len, (i64)b.len);
  ASSERT_EQ(0, memcmp(a.data, b.data, a.len));
  fbx_wasm_buffer_free(&a);
  fbx_wasm_buffer_free(&b);
  FreeTc(tc);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* firebox#JJJ — Tier-2 regions.                                             */
/*                                                                            */
/* The subject is the real hot loop of the shipped rg binary (rg -j1, the    */
/* dense-DFA transition loop, 4x unrolled; MEASURED top blocks 0-3, 9, 10):  */
/* six Tier-1 blocks decoded from the bytes rg executes, at their PCs.       */
/* ────────────────────────────────────────────────────────────────────────── */

struct RgInsn {
  u64 ip;
  u8 len;
  u8 bytes[8];
};

static const struct RgInsn kRgDfa[] = {
    /* block 0x8813b145 */
    {0x8813b145, 6, {0x43, 0x0f, 0xb6, 0x54, 0x25, 0x00}},
    {0x8813b14b, 6, {0x41, 0x0f, 0xb6, 0x54, 0x16, 0x78}},
    {0x8813b151, 2, {0x89, 0xee}},
    {0x8813b153, 3, {0x48, 0x01, 0xd6}},
    {0x8813b156, 3, {0x8b, 0x2c, 0xb1}},
    {0x8813b159, 2, {0x39, 0xc5}},
    {0x8813b15b, 2, {0x76, 0x65}},
    /* block 0x8813b15d */
    {0x8813b15d, 5, {0x49, 0x8d, 0x54, 0x24, 0x03}},
    {0x8813b162, 3, {0x4c, 0x39, 0xfa}},
    {0x8813b165, 2, {0x73, 0x5b}},
    /* block 0x8813b167 */
    {0x8813b167, 6, {0x43, 0x0f, 0xb6, 0x74, 0x25, 0x01}},
    {0x8813b16d, 6, {0x41, 0x0f, 0xb6, 0x74, 0x36, 0x78}},
    {0x8813b173, 3, {0x48, 0x01, 0xee}},
    {0x8813b176, 3, {0x8b, 0x2c, 0xb1}},
    {0x8813b179, 2, {0x39, 0xc5}},
    {0x8813b17b, 2, {0x76, 0x37}},
    /* block 0x8813b17d */
    {0x8813b17d, 6, {0x43, 0x0f, 0xb6, 0x74, 0x25, 0x02}},
    {0x8813b183, 6, {0x41, 0x0f, 0xb6, 0x74, 0x36, 0x78}},
    {0x8813b189, 3, {0x48, 0x01, 0xee}},
    {0x8813b18c, 3, {0x8b, 0x2c, 0xb1}},
    {0x8813b18f, 2, {0x39, 0xc5}},
    {0x8813b191, 2, {0x76, 0x26}},
    /* block 0x8813b193 */
    {0x8813b193, 6, {0x43, 0x0f, 0xb6, 0x74, 0x25, 0x03}},
    {0x8813b199, 6, {0x41, 0x0f, 0xb6, 0x74, 0x36, 0x78}},
    {0x8813b19f, 3, {0x48, 0x01, 0xee}},
    {0x8813b1a2, 3, {0x8b, 0x2c, 0xb1}},
    {0x8813b1a5, 2, {0x39, 0xc5}},
    {0x8813b1a7, 2, {0x76, 0x16}},
    /* block 0x8813b1a9 */
    {0x8813b1a9, 4, {0x49, 0x83, 0xc4, 0x04}},
    {0x8813b1ad, 3, {0x4d, 0x39, 0xfc}},
    {0x8813b1b0, 2, {0x72, 0x93}},
};

/* Decode kRgDfa[first, first+n) into one Tier-1 block. */
static struct FbxTcBlock *DecodeTc(const struct RgInsn *ins, unsigned n) {
  struct FbxTcBlock *b = (struct FbxTcBlock *)calloc(1, sizeof *b);
  unsigned i;
  b->entries = (struct FbxTcEntry *)calloc(n, sizeof(struct FbxTcEntry));
  for (i = 0; i < n; ++i) {
    struct FbxTcEntry *e = &b->entries[i];
    u8 buf[15];
    memset(buf, 0x90, sizeof buf);
    memcpy(buf, ins[i].bytes, ins[i].len);
    if (DecodeInstruction(&e->xedd, buf, sizeof buf, XED_MODE_LONG) ||
        e->xedd.length != ins[i].len) {
      free(b->entries);
      free(b);
      return NULL;
    }
    e->ip = ins[i].ip;
    e->rde = e->xedd.op.rde;
    e->uimm0 = e->xedd.op.uimm0;
    e->disp = e->xedd.op.disp;
    e->oplen = ins[i].len;
    e->kind = FBX_TC_KIND_NORMAL;
  }
  b->start_pc = ins[0].ip;
  b->end_pc = ins[n - 1].ip + ins[n - 1].len;
  b->page = b->start_pc & ~(u64)4095;
  b->nentries = n;
  return b;
}

static int BlockHasOp(const struct FbxIrBlock *ir, u8 op) {
  u32 i;
  for (i = 0; i < ir->ninsts; ++i) {
    if (ir->insts[i].opcode == op) return 1;
  }
  return 0;
}

#define RG_NBLOCKS 6
static const unsigned kRgBlockFirst[RG_NBLOCKS] = {0, 7, 10, 16, 22, 28};
static const unsigned kRgBlockLen[RG_NBLOCKS] = {7, 3, 6, 6, 6, 3};

TEST(FbxRegion, RgDfaLoopMembersAreAdmittedAndEmit) {
  struct FbxTcBlock *tcs[RG_NBLOCKS];
  struct FbxIrBlock *irs[RG_NBLOCKS];
  struct FbxWasmBuffer out;
  u64 succ[2];
  unsigned i;
  u32 j;
  int saw_br_table = 0;
  for (i = 0; i < RG_NBLOCKS; ++i) {
    enum FbxIrEmitFailReason why = FBX_IR_EMIT_OK;
    u8 op = 0;
    tcs[i] = DecodeTc(&kRgDfa[kRgBlockFirst[i]], kRgBlockLen[i]);
    ASSERT_NOTNULL(tcs[i]);
    irs[i] = fbx_ir_lift(tcs[i]);
    ASSERT_NOTNULL(irs[i]);
    EXPECT_EQ(0, BlockHasOp(irs[i], FBX_IR_OP_BAILOUT));
    EXPECT_EQ(1, fbx_ir_region_member_ok(irs[i], tcs[i], &why, &op));
    EXPECT_EQ(FBX_IR_EMIT_OK, why);
  }
  /* Successors, fall-through first: block 0 → (0x8813b15d, 0x8813b1c2). */
  ASSERT_EQ(2, (i64)fbx_ir_region_successors(irs[0], tcs[0], succ));
  EXPECT_EQ((i64)0x8813b15d, (i64)succ[0]);
  EXPECT_EQ((i64)0x8813b1c2, (i64)succ[1]);
  ASSERT_EQ(2, (i64)fbx_ir_region_successors(irs[5], tcs[5], succ));
  EXPECT_EQ((i64)0x8813b1b2, (i64)succ[0]);
  EXPECT_EQ((i64)0x8813b145, (i64)succ[1]);
  fbx_wasm_buffer_init(&out);
  ASSERT_EQ(1, fbx_ir_emit_region_wasm(
                   (const struct FbxIrBlock *const *)irs,
                   (const struct FbxTcBlock *const *)tcs, RG_NBLOCKS, 0,
                   4096, &out));
  ASSERT_NE((i64)0, (i64)out.len);
  for (j = 0; j < out.len; ++j) {
    if (out.data[j] == 0x0Eu) saw_br_table = 1;
  }
  EXPECT_EQ(1, saw_br_table);
  {
    /* Cross-engine validation hook: FBX_REGION_DUMP=<path> writes the
     * module so wasm-tools / wasmer can validate it out of tree. */
    const char *dump = getenv("FBX_REGION_DUMP");
    if (dump && *dump) {
      FILE *f = fopen(dump, "wb");
      if (f) {
        fwrite(out.data, 1, out.len, f);
        fclose(f);
      }
    }
  }
  /* The counting variant is a different module (it stores the count). */
  {
    struct FbxWasmBuffer counted;
    fbx_wasm_buffer_init(&counted);
    ASSERT_EQ(1, fbx_ir_emit_region_wasm(
                     (const struct FbxIrBlock *const *)irs,
                     (const struct FbxTcBlock *const *)tcs, RG_NBLOCKS, 1,
                     4096, &counted));
    EXPECT_NE(0, counted.len != out.len ||
                     memcmp(counted.data, out.data, out.len) != 0);
    fbx_wasm_buffer_free(&counted);
  }
  /* Refusals: duplicate member, zero members, zero budget. */
  {
    const struct FbxIrBlock *dirs[2] = {irs[0], irs[0]};
    const struct FbxTcBlock *dtcs[2] = {tcs[0], tcs[0]};
    struct FbxWasmBuffer o2;
    fbx_wasm_buffer_init(&o2);
    EXPECT_EQ(0, fbx_ir_emit_region_wasm(dirs, dtcs, 2, 0, 4096, &o2));
    EXPECT_EQ(0, fbx_ir_emit_region_wasm(dirs, dtcs, 0, 0, 4096, &o2));
    EXPECT_EQ(0, fbx_ir_emit_region_wasm(dirs, dtcs, 1, 0, 0, &o2));
    fbx_wasm_buffer_free(&o2);
  }
  fbx_wasm_buffer_free(&out);
  for (i = 0; i < RG_NBLOCKS; ++i) {
    fbx_ir_free(irs[i]);
    FreeTc(tcs[i]);
  }
}

/* Lift-and-ask helper: is this one-or-two instruction block a member?  The
 * block is lifted as a region member (fbx_ir_lift_region), as
 * fbx_t2_glue.c's Fbxt2TryRegion lifts it. */
static int RegionMemberOkBytes(const struct RgInsn *ins, unsigned n) {
  struct FbxTcBlock *tc = DecodeTc(ins, n);
  struct FbxIrBlock *ir;
  int ok;
  if (!tc) return -1;
  ir = fbx_ir_lift_region(tc);
  if (!ir) {
    FreeTc(tc);
    return -1;
  }
  ok = fbx_ir_region_member_ok(ir, tc, NULL, NULL);
  fbx_ir_free(ir);
  FreeTc(tc);
  return ok;
}

/* Does the PER-BLOCK lift of this block BAIL? */
static int PerBlockLiftBails(const struct RgInsn *ins, unsigned n) {
  struct FbxTcBlock *tc = DecodeTc(ins, n);
  struct FbxIrBlock *ir;
  int bails;
  if (!tc) return -1;
  ir = fbx_ir_lift(tc);
  if (!ir) {
    FreeTc(tc);
    return -1;
  }
  bails = (ir->flags & FBX_IR_BLOCK_FLAG_HAS_BAILOUT) != 0;
  fbx_ir_free(ir);
  FreeTc(tc);
  return bails;
}

TEST(FbxRegion, MembersThatCannotBeLoweredAreRefused) {
  /* cmp eax, ebx ; jne — admitted. */
  static const struct RgInsn jne[] = {
      {0x1000, 2, {0x39, 0xd8}}, {0x1002, 2, {0x75, 0xfc}}};
  /* firebox#T3R2: stores, stack ops, calls and returns are region members. */
  static const struct RgInsn store[] = { /* mov [rax+8], ebx ; jne */
      {0x1000, 3, {0x89, 0x58, 0x08}}, {0x1003, 2, {0x75, 0xfb}}};
  static const struct RgInsn store_ix[] = { /* mov [r12+rdi], bpl ; jne */
      {0x1000, 4, {0x41, 0x88, 0x2c, 0x3c}}, {0x1004, 2, {0x75, 0xfa}}};
  static const struct RgInsn push[] = { /* push rbx ; jmp */
      {0x1000, 1, {0x53}}, {0x1001, 2, {0xeb, 0xfd}}};
  static const struct RgInsn ret[] = {{0x1000, 1, {0xc3}}};
  static const struct RgInsn callr[] = {{0x1000, 2, {0xff, 0xd0}}}; /* call rax */
  static const struct RgInsn jmpm[] = { /* jmp [rip+0x10] */
      {0x1000, 6, {0xff, 0x25, 0x10, 0x00, 0x00, 0x00}}};
  static const struct RgInsn inc[] = { /* inc rax ; jne */
      {0x1000, 3, {0x48, 0xff, 0xc0}}, {0x1003, 2, {0x75, 0xfb}}};
  static const struct RgInsn neg[] = { /* neg rcx ; jne */
      {0x1000, 3, {0x48, 0xf7, 0xd9}}, {0x1003, 2, {0x75, 0xfb}}};
  static const struct RgInsn cmpm[] = { /* cmp [rax+rcx], r14b ; jne */
      {0x1000, 4, {0x44, 0x38, 0x34, 0x08}}, {0x1004, 2, {0x75, 0xfa}}};
  static const struct RgInsn testi[] = { /* test al, 1 ; jne */
      {0x1000, 2, {0xa8, 0x01}}, {0x1002, 2, {0x75, 0xfc}}};
  /* Refused: */
  static const struct RgInsn jp[] = { /* cmp eax, ebx ; jp — PF/NP */
      {0x1000, 2, {0x39, 0xd8}}, {0x1002, 2, {0x7a, 0xfc}}};
  static const struct RgInsn imul[] = { /* imul eax, ebx ; jmp */
      {0x1000, 3, {0x0f, 0xaf, 0xc3}}, {0x1003, 2, {0xeb, 0xfb}}};
  static const struct RgInsn sys[] = {{0x1000, 2, {0x0f, 0x05}}};
  static const struct RgInsn rmw[] = { /* add [rax], ebx ; jne */
      {0x1000, 2, {0x01, 0x18}}, {0x1002, 2, {0x75, 0xfc}}};
  static const struct RgInsn incm[] = { /* inc dword [rax] ; jne */
      {0x1000, 2, {0xff, 0x00}}, {0x1002, 2, {0x75, 0xfc}}};
  static const struct RgInsn push16[] = { /* push bx (66) ; jmp */
      {0x1000, 2, {0x66, 0x53}}, {0x1002, 2, {0xeb, 0xfc}}};
  static const struct RgInsn mulr[] = { /* mul rbx (F7 /4) ; jmp */
      {0x1000, 3, {0x48, 0xf7, 0xe3}}, {0x1003, 2, {0xeb, 0xfb}}};
  static const struct RgInsn pushm[] = { /* push [rax] (FF /6) ; jmp */
      {0x1000, 2, {0xff, 0x30}}, {0x1002, 2, {0xeb, 0xfc}}};
  static const struct RgInsn lockinc[] = { /* lock inc dword [rax] ; jmp */
      {0x1000, 3, {0xf0, 0xff, 0x00}}, {0x1003, 2, {0xeb, 0xfb}}};
  EXPECT_EQ(1, RegionMemberOkBytes(jne, 2));
  EXPECT_EQ(1, RegionMemberOkBytes(store, 2));
  EXPECT_EQ(1, RegionMemberOkBytes(store_ix, 2));
  EXPECT_EQ(1, RegionMemberOkBytes(push, 2));
  EXPECT_EQ(1, RegionMemberOkBytes(ret, 1));
  EXPECT_EQ(1, RegionMemberOkBytes(callr, 1));
  EXPECT_EQ(1, RegionMemberOkBytes(jmpm, 1));
  EXPECT_EQ(1, RegionMemberOkBytes(inc, 2));
  EXPECT_EQ(1, RegionMemberOkBytes(neg, 2));
  EXPECT_EQ(1, RegionMemberOkBytes(cmpm, 2));
  EXPECT_EQ(1, RegionMemberOkBytes(testi, 2));
  EXPECT_EQ(0, RegionMemberOkBytes(jp, 2));
  EXPECT_EQ(0, RegionMemberOkBytes(imul, 2));
  EXPECT_EQ(0, RegionMemberOkBytes(sys, 1));
  EXPECT_EQ(0, RegionMemberOkBytes(rmw, 2));
  EXPECT_EQ(0, RegionMemberOkBytes(incm, 2));
  EXPECT_EQ(0, RegionMemberOkBytes(push16, 2));
  EXPECT_EQ(0, RegionMemberOkBytes(mulr, 2));
  EXPECT_EQ(0, RegionMemberOkBytes(pushm, 2));
  EXPECT_EQ(0, RegionMemberOkBytes(lockinc, 2));
  /* The region-only forms stay off the per-block path: its lift BAILs. */
  EXPECT_EQ(1, PerBlockLiftBails(store_ix, 2));
  EXPECT_EQ(1, PerBlockLiftBails(callr, 1));
  EXPECT_EQ(1, PerBlockLiftBails(jmpm, 1));
  EXPECT_EQ(1, PerBlockLiftBails(inc, 2));
  EXPECT_EQ(1, PerBlockLiftBails(neg, 2));
  EXPECT_EQ(1, PerBlockLiftBails(cmpm, 2));
  EXPECT_EQ(1, PerBlockLiftBails(testi, 2));
  EXPECT_EQ(1, PerBlockLiftBails(rmw, 2));
  EXPECT_EQ(0, PerBlockLiftBails(store, 2)); /* #677's base+disp store */
}

/* ────────────────────────────────────────────────────────────────────────── */
/* firebox#NDG — a flag definition is dead only if nothing between it and the */
/* next definition can leave the block.                                       */
/* ────────────────────────────────────────────────────────────────────────── */

/* Occurrences of `i32.store align=2 offset=offsetof(Machine, flags)`. */
static unsigned CountFlagsStores(const struct FbxWasmBuffer *w) {
  u8 pat[8];
  unsigned n = 0, k = 0, v = (unsigned)offsetof(struct Machine, flags);
  size_t i;
  pat[k++] = 0x36u;
  pat[k++] = 0x02u;
  do {
    u8 byte = v & 0x7Fu;
    v >>= 7;
    pat[k++] = byte | (v ? 0x80u : 0u);
  } while (v);
  for (i = 0; i + k <= w->len; ++i) {
    if (!memcmp(w->data + i, pat, k)) ++n;
  }
  return n;
}

static unsigned PerBlockFlagsStores(const struct RgInsn *ins, unsigned n) {
  struct FbxTcBlock *tc = DecodeTc(ins, n);
  struct FbxIrBlock *ir;
  struct FbxWasmBuffer out;
  unsigned stores = 0xFFFFu;
  if (!tc) return 0xFFFEu;
  ir = fbx_ir_lift(tc);
  fbx_wasm_buffer_init(&out);
  if (ir && fbx_ir_emit_wasm(ir, tc, &out)) stores = CountFlagsStores(&out);
  fbx_wasm_buffer_free(&out);
  fbx_ir_free(ir);
  free(tc->entries);
  free(tc);
  return stores;
}

TEST(FbxNdg, FlagDefinitionSurvivesALoadThatCanBailOut) {
  /* add rax, rbx ; mov rcx, [rdx+8] ; sub rsi, rdi ; jmp .+2
   * The LOAD can return 1 with m->ip at the mov: Tier 1 resumes there and,
   * if the access faults, the signal frame carries the ADD's flags. */
  static const struct RgInsn across_load[] = {
      {0x1000, 3, {0x48, 0x01, 0xd8}},
      {0x1003, 4, {0x48, 0x8b, 0x4a, 0x08}},
      {0x1007, 3, {0x48, 0x29, 0xfe}},
      {0x100a, 2, {0xeb, 0x00}}};
  /* The control: a register mov cannot leave the block, so the ADD's
   * definition is overwritten unobserved and elided. */
  static const struct RgInsn no_exit[] = {
      {0x1000, 3, {0x48, 0x01, 0xd8}},
      {0x1003, 3, {0x48, 0x89, 0xd1}},
      {0x1006, 3, {0x48, 0x29, 0xfe}},
      {0x1009, 2, {0xeb, 0x00}}};
  EXPECT_EQ(2, (i64)PerBlockFlagsStores(across_load, 4));
  EXPECT_EQ(1, (i64)PerBlockFlagsStores(no_exit, 4));
}

/* ────────────────────────────────────────────────────────────────────────── */
/* firebox#5BW — BRANCH_COND's fall-through and CALL_DIRECT's return PC at    */
/* full width.  The IR's src2 holds only the low 32 bits.                     */
/* ────────────────────────────────────────────────────────────────────────── */

static int ConstsHold(const struct RgInsn *ins, unsigned n, u64 want,
                      u64 forbid) {
  struct FbxTcBlock *tc = DecodeTc(ins, n);
  struct FbxIrBlock *ir;
  u64 consts[FBX_T2_BLOCK_CTX_MAX_CONSTS];
  u32 nconsts = 0, k;
  int saw = 0, bad = 0;
  struct FbxWasmBuffer out;
  if (!tc) return -1;
  ir = fbx_ir_lift(tc);
  fbx_wasm_buffer_init(&out);
  if (!ir || !fbx_ir_emit_wasm(ir, tc, &out) ||
      !fbx_ir_build_block_ctx_consts(ir, consts, FBX_T2_BLOCK_CTX_MAX_CONSTS,
                                     &nconsts)) {
    saw = -2;
  } else {
    for (k = 0; k < nconsts; ++k) {
      saw |= consts[k] == want;
      bad |= consts[k] == forbid;
    }
  }
  fbx_wasm_buffer_free(&out);
  fbx_ir_free(ir);
  free(tc->entries);
  free(tc);
  return saw < 0 ? saw : (saw && !bad);
}

TEST(Fbx5bw, FallThroughAndReturnPcKeepTheHighHalf) {
  /* cmp rax, rbx ; jne .+0x10, at a PC above 4 GiB */
  static const struct RgInsn jcc[] = {
      {0x100001000ull, 3, {0x48, 0x39, 0xd8}},
      {0x100001003ull, 2, {0x75, 0x10}}};
  /* call .+0x100, at a PC above 4 GiB */
  static const struct RgInsn call[] = {
      {0x200002000ull, 5, {0xe8, 0x00, 0x01, 0x00, 0x00}}};
  EXPECT_EQ(1, ConstsHold(jcc, 2, 0x100001005ull, 0x00001005ull));
  EXPECT_EQ(1, ConstsHold(call, 1, 0x200002005ull, 0x00002005ull));
}

TEST(Fbx5bw, BranchCondWhoseSrc2DisagreesWithTheBlockEndIsRefused) {
  static const struct RgInsn jcc[] = {
      {0x100001000ull, 3, {0x48, 0x39, 0xd8}},
      {0x100001003ull, 2, {0x75, 0x10}}};
  struct FbxTcBlock *tc = DecodeTc(jcc, 2);
  struct FbxIrBlock *ir;
  struct FbxWasmBuffer out;
  u32 k;
  ASSERT_NOTNULL(tc);
  ir = fbx_ir_lift(tc);
  ASSERT_NOTNULL(ir);
  fbx_wasm_buffer_init(&out);
  EXPECT_EQ(1, fbx_ir_emit_wasm(ir, tc, &out));
  for (k = 0; k < ir->ninsts; ++k) {
    if (ir->insts[k].opcode == FBX_IR_OP_BRANCH_COND) ir->insts[k].src2 += 1;
  }
  EXPECT_EQ(0, fbx_ir_emit_wasm(ir, tc, &out));
  fbx_wasm_buffer_free(&out);
  fbx_ir_free(ir);
  free(tc->entries);
  free(tc);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* firebox#T3R2 — EXECUTE a region.                                            */
/*                                                                            */
/* The tests above check what a region module looks like; these run one.  A  */
/* small interpreter for the WebAssembly subset the emitter produces runs the */
/* module natively over a byte array standing in for the guest's linear       */
/* memory, holding a Machine, its System (cr3), a 4-level page table and the  */
/* guest pages.  The expected state is independent of the emitter: values by  */
/* hand, flags from blink/alu.c's own Inc64 / Neg64 / Dec8 / Sub8 — the       */
/* functions Tier 1 runs.  An opcode the interpreter does not know fails the  */
/* test rather than being skipped.                                            */
/* ────────────────────────────────────────────────────────────────────────── */

#include "blink/alu.h"
#include "blink/machine.h"

#define RX_MEM (4u << 20)
#define RX_MAXF 4
#define RX_STACK 1024
#define RX_LABELS 512

struct RxFunc {
  const u8 *code, *end; /* body, after the locals */
  u32 nparams, nlocals;
  u32 type;
};

struct RxModule {
  struct RxFunc f[RX_MAXF];
  u32 nimports, nfuncs;
  u32 nparams_of_type[8];
  u8 *mem;
  int err;
  char why[96];
};

static u64 RxUleb(const u8 **p) {
  u64 v = 0;
  unsigned s = 0;
  u8 b;
  do {
    b = *(*p)++;
    v |= (u64)(b & 0x7f) << s;
    s += 7;
  } while (b & 0x80);
  return v;
}

static i64 RxSleb(const u8 **p) {
  i64 v = 0;
  unsigned s = 0;
  u8 b;
  do {
    b = *(*p)++;
    v |= (i64)(b & 0x7f) << s;
    s += 7;
  } while (b & 0x80);
  if (s < 64 && (b & 0x40)) v |= -((i64)1 << s);
  return v;
}

static int RxParse(struct RxModule *x, const u8 *d, size_t n) {
  const u8 *p = d + 8, *e = d + n;
  u32 ntypes = 0, i;
  u32 ftypes[RX_MAXF];
  memset(x->f, 0, sizeof x->f);
  x->nimports = x->nfuncs = 0;
  while (p < e) {
    u8 id = *p++;
    u64 len = RxUleb(&p);
    const u8 *s = p, *se = p + len;
    if (id == 1) {
      ntypes = (u32)RxUleb(&s);
      for (i = 0; i < ntypes && i < 8; ++i) {
        u32 np, nr;
        s++; /* 0x60 */
        np = (u32)RxUleb(&s);
        s += np;
        nr = (u32)RxUleb(&s);
        s += nr;
        x->nparams_of_type[i] = np;
      }
    } else if (id == 2) {
      u32 k = (u32)RxUleb(&s);
      for (i = 0; i < k; ++i) {
        u64 l = RxUleb(&s);
        s += l;
        l = RxUleb(&s);
        s += l;
        if (*s++ == 0) { /* func */
          RxUleb(&s);
          x->nimports++;
        } else { /* memory: limits */
          u8 fl = *s++;
          RxUleb(&s);
          if (fl & 1) RxUleb(&s);
        }
      }
    } else if (id == 3) {
      x->nfuncs = (u32)RxUleb(&s);
      if (x->nfuncs > RX_MAXF) return 0;
      for (i = 0; i < x->nfuncs; ++i) ftypes[i] = (u32)RxUleb(&s);
    } else if (id == 10) {
      u32 k = (u32)RxUleb(&s);
      if (k != x->nfuncs) return 0;
      for (i = 0; i < k; ++i) {
        u64 bl = RxUleb(&s);
        const u8 *b = s, *be = s + bl;
        u32 groups = (u32)RxUleb(&b), g, nl = 0;
        for (g = 0; g < groups; ++g) {
          nl += (u32)RxUleb(&b);
          b++; /* valtype */
        }
        x->f[i].type = ftypes[i];
        x->f[i].nparams = x->nparams_of_type[ftypes[i]];
        x->f[i].nlocals = nl;
        x->f[i].code = b;
        x->f[i].end = be;
        s = be;
      }
    }
    p = se;
  }
  return x->nfuncs > 0;
}

/* Skip one instruction's immediates (after its opcode byte). */
static int RxSkipImm(u8 op, const u8 **p) {
  switch (op) {
    case 0x02: case 0x03: case 0x04:
      (*p)++; /* blocktype: the emitter only uses 0x40 */
      return 1;
    case 0x0C: case 0x0D: case 0x10: case 0x20: case 0x21: case 0x22:
      RxUleb(p);
      return 1;
    case 0x0E: {
      u64 k = RxUleb(p), i;
      for (i = 0; i <= k; ++i) RxUleb(p);
      return 1;
    }
    case 0x41: case 0x42:
      RxSleb(p);
      return 1;
    default:
      if (op >= 0x28 && op <= 0x3E) {
        RxUleb(p);
        RxUleb(p);
      }
      return 1;
  }
}

/* The matching `end` (and `else`) of the block/loop/if whose immediate ends
 * at `p`. */
static const u8 *RxFindEnd(const u8 *p, const u8 *e, const u8 **else_at) {
  int depth = 0;
  *else_at = NULL;
  while (p < e) {
    const u8 *at = p;
    u8 op = *p++;
    if (op == 0x02 || op == 0x03 || op == 0x04) {
      depth++;
    } else if (op == 0x05 && depth == 0) {
      *else_at = at;
    } else if (op == 0x0B) {
      if (depth == 0) return at;
      depth--;
    }
    RxSkipImm(op, &p);
  }
  return NULL;
}

static int RxMem(struct RxModule *x, u64 a, u32 n) {
  if (a + n > RX_MEM || a + n < a) {
    x->err = 1;
    snprintf(x->why, sizeof x->why, "memory access %#llx+%u out of bounds",
             (unsigned long long)a, n);
    return 0;
  }
  return 1;
}

static u64 RxRun(struct RxModule *x, u32 fi, const u64 *args, int depth) {
  struct RxFunc *f = &x->f[fi];
  u64 loc[256], st[RX_STACK];
  struct {
    const u8 *body; /* a loop's first instruction */
    const u8 *end;  /* the block's `end` */
    u32 sp;
    int loop;
  } lab[RX_LABELS];
  u32 sp = 0, nl = 0, i;
  const u8 *p = f->code;
  if (depth > 4 || f->nparams + f->nlocals > 256) {
    x->err = 1;
    return 0;
  }
  memset(loc, 0, sizeof loc);
  for (i = 0; i < f->nparams; ++i) loc[i] = args[i];
#define POP() (st[--sp])
#define PUSH(v) (st[sp++] = (u64)(v))
#define I32(v) ((u64)(u32)(v))
#define FAIL(...)                                  \
  do {                                             \
    x->err = 1;                                    \
    snprintf(x->why, sizeof x->why, __VA_ARGS__);  \
    return 0;                                      \
  } while (0)
  for (;;) {
    u8 op;
    if (x->err) return 0;
    if (p >= f->end || sp >= RX_STACK - 4 || nl >= RX_LABELS - 1) {
      FAIL("ran off the body or the stack");
    }
    op = *p++;
    switch (op) {
      case 0x00:
        FAIL("unreachable executed");
      case 0x02: case 0x03: case 0x04: {
        const u8 *els, *end;
        if (*p++ != 0x40) FAIL("blocktype");
        end = RxFindEnd(p, f->end, &els);
        if (!end) FAIL("no end");
        lab[nl].body = p;
        lab[nl].end = end;
        lab[nl].loop = op == 0x03;
        if (op == 0x04) {
          u32 c = (u32)POP();
          lab[nl].sp = sp;
          nl++;
          if (!c) {
            if (els) {
              p = els + 1;
            } else {
              p = end + 1;
              nl--;
            }
          }
        } else {
          lab[nl].sp = sp;
          nl++;
        }
        break;
      }
      case 0x05: /* the then-arm finished: skip the else-arm */
        if (!nl) FAIL("else outside if");
        p = lab[nl - 1].end + 1;
        nl--;
        break;
      case 0x0B:
        if (nl == 0) return sp ? st[sp - 1] : 0;
        nl--;
        break;
      case 0x0C: case 0x0D: case 0x0E: {
        u64 d;
        if (op == 0x0E) {
          u64 k = RxUleb(&p), j, sel = (u32)POP();
          d = 0;
          for (j = 0; j <= k; ++j) {
            u64 t = RxUleb(&p);
            if (j == sel || (j == k && sel >= k)) d = t;
          }
        } else {
          d = RxUleb(&p);
          if (op == 0x0D && !(u32)POP()) break;
        }
        if (d >= nl) return sp ? st[sp - 1] : 0;
        {
          u32 t = nl - 1u - (u32)d;
          sp = lab[t].sp;
          if (lab[t].loop) {
            nl = t + 1u;
            p = lab[t].body;
          } else {
            nl = t;
            p = lab[t].end + 1;
          }
        }
        break;
      }
      case 0x0F:
        return sp ? st[sp - 1] : 0;
      case 0x10: {
        u32 callee = (u32)RxUleb(&p);
        u64 a[4];
        u32 np, k;
        if (callee < x->nimports || callee - x->nimports >= x->nfuncs) {
          FAIL("call to import %u", callee);
        }
        np = x->f[callee - x->nimports].nparams;
        if (np > 4) FAIL("arity");
        for (k = np; k-- > 0;) a[k] = POP();
        PUSH(RxRun(x, callee - x->nimports, a, depth + 1));
        break;
      }
      case 0x1A: sp--; break;
      case 0x1B: {
        u32 c = (u32)POP();
        u64 b = POP(), a = POP();
        PUSH(c ? a : b);
        break;
      }
      case 0x20: PUSH(loc[RxUleb(&p)]); break;
      case 0x21: loc[RxUleb(&p)] = POP(); break;
      case 0x22: loc[RxUleb(&p)] = st[sp - 1]; break;
      case 0x41: PUSH(I32(RxSleb(&p))); break;
      case 0x42: PUSH(RxSleb(&p)); break;
      default:
        if (op >= 0x28 && op <= 0x3E) {
          u64 off, a, v = 0;
          u32 w;
          RxUleb(&p);
          off = RxUleb(&p);
          switch (op) {
            case 0x28: case 0x35: case 0x36: case 0x3E: w = 4; break;
            case 0x29: case 0x37: w = 8; break;
            case 0x2D: case 0x31: case 0x3A: case 0x3C: w = 1; break;
            case 0x33: case 0x3B: case 0x3D: w = 2; break;
            default: FAIL("memory opcode %#x", op);
          }
          if (op >= 0x36) {
            v = POP();
            a = (u32)POP() + off;
            if (!RxMem(x, a, w)) return 0;
            memcpy(x->mem + a, &v, w);
          } else {
            a = (u32)POP() + off;
            if (!RxMem(x, a, w)) return 0;
            memcpy(&v, x->mem + a, w);
            PUSH(v);
          }
          break;
        }
        {
          u64 b, a, r;
          switch (op) {
            case 0x45: a = POP(); PUSH((u32)a == 0); continue;
            case 0x50: a = POP(); PUSH(a == 0); continue;
            case 0xA7: a = POP(); PUSH(I32(a)); continue;
            case 0xAD: a = POP(); PUSH((u32)a); continue;
            case 0xAC: a = POP(); PUSH((u64)(i64)(i32)a); continue;
            default: break;
          }
          b = POP();
          a = POP();
          switch (op) {
            case 0x46: r = (u32)a == (u32)b; break;
            case 0x47: r = (u32)a != (u32)b; break;
            case 0x48: r = (i32)a < (i32)b; break;
            case 0x49: r = (u32)a < (u32)b; break;
            case 0x4B: r = (u32)a > (u32)b; break;
            case 0x4D: r = (u32)a <= (u32)b; break;
            case 0x4F: r = (u32)a >= (u32)b; break;
            case 0x51: r = a == b; break;
            case 0x52: r = a != b; break;
            case 0x53: r = (i64)a < (i64)b; break;
            case 0x54: r = a < b; break;
            case 0x56: r = a > b; break;
            case 0x58: r = a <= b; break;
            case 0x5A: r = a >= b; break;
            case 0x6A: r = I32(a + b); break;
            case 0x6B: r = I32(a - b); break;
            case 0x6C: r = I32((u32)a * (u32)b); break;
            case 0x71: r = I32(a & b); break;
            case 0x72: r = I32(a | b); break;
            case 0x73: r = I32(a ^ b); break;
            case 0x74: r = I32((u32)a << (b & 31)); break;
            case 0x76: r = I32((u32)a >> (b & 31)); break;
            case 0x7C: r = a + b; break;
            case 0x7D: r = a - b; break;
            case 0x7E: r = a * b; break;
            case 0x83: r = a & b; break;
            case 0x84: r = a | b; break;
            case 0x85: r = a ^ b; break;
            case 0x86: r = a << (b & 63); break;
            case 0x87: r = (u64)((i64)a >> (b & 63)); break;
            case 0x88: r = a >> (b & 63); break;
            default: FAIL("opcode %#x", op);
          }
          PUSH(r);
        }
    }
  }
#undef POP
#undef PUSH
#undef I32
#undef FAIL
}

/* Guest layout inside the fake linear memory. */
#define RX_M 0x10000u      /* struct Machine */
#define RX_S 0x80000u      /* struct System (only cr3 is read) */
#define RX_CTX 0x100000u   /* struct FbxT2BlockCtx + store log */
#define RX_PT 0x200000u    /* PML4, PDPT, PD, PT */
#define RX_FRAME 0x300000u /* guest page frames */
#define RX_DATA_VA 0x0000000000601000ull
#define RX_RO_VA 0x0000000000602000ull
#define RX_STACK_VA 0x0000000000700000ull
#define RX_CODE 0x401000ull

static void RxMap(u8 *mem, u64 va, u32 frame, u64 extra) {
  static const unsigned shifts[3] = {39, 30, 21};
  u32 table = RX_PT, k;
  for (k = 0; k < 3; ++k) {
    u32 idx = (u32)((va >> shifts[k]) & 511);
    u64 e;
    memcpy(&e, mem + table + idx * 8, 8);
    if (!(e & PAGE_V)) {
      e = (u64)(RX_PT + 0x1000u * (k + 1u)) | PAGE_V | PAGE_U | PAGE_RW |
          PAGE_HOST;
      memcpy(mem + table + idx * 8, &e, 8);
    }
    table = (u32)(e & PAGE_TA);
  }
  {
    u64 e = (u64)frame | PAGE_V | PAGE_U | PAGE_HOST | extra;
    memcpy(mem + table + ((va >> 12) & 511) * 8, &e, 8);
  }
}

struct RxRun {
  struct Machine m; /* state before; after the run, the region's result */
  u32 rc;
  int err;
  char why[96];
  u64 log_used;
  u8 frames[3][4096]; /* data, read-only, stack pages after the run */
};

/* Lift the blocks as region members, emit, and run from `m0`. */
static void RxExec(const struct RgInsn *const *blocks, const unsigned *lens,
                   unsigned nb, const struct Machine *m0, int flags,
                   struct RxRun *out) {
  struct FbxTcBlock *tcs[4];
  struct FbxIrBlock *irs[4];
  struct FbxWasmBuffer w;
  struct RxModule x;
  struct Machine mm;
  struct FbxT2BlockCtx ctx;
  u64 args[2], cr3;
  unsigned i;
  memset(out, 0, sizeof *out);
  for (i = 0; i < nb; ++i) {
    tcs[i] = DecodeTc(blocks[i], lens[i]);
    irs[i] = tcs[i] ? fbx_ir_lift_region(tcs[i]) : NULL;
  }
  fbx_wasm_buffer_init(&w);
  if (!fbx_ir_emit_region_wasm((const struct FbxIrBlock *const *)irs,
                               (const struct FbxTcBlock *const *)tcs, nb,
                               flags, 64, &w)) {
    out->err = 1;
    snprintf(out->why, sizeof out->why, "emit refused");
    goto done;
  }
  x.mem = (u8 *)calloc(1, RX_MEM);
  x.err = 0;
  x.why[0] = 0;
  if (!RxParse(&x, w.data, w.len)) {
    out->err = 1;
    snprintf(out->why, sizeof out->why, "parse");
    free(x.mem);
    goto done;
  }
  RxMap(x.mem, RX_DATA_VA, RX_FRAME, PAGE_RW);
  RxMap(x.mem, RX_RO_VA, RX_FRAME + 0x1000u, 0); /* read-only */
  RxMap(x.mem, RX_STACK_VA - 0x1000u, RX_FRAME + 0x2000u, PAGE_RW);
  memcpy(x.mem + RX_FRAME, "region-data:0123456789abcdef", 28);
  cr3 = RX_PT | PAGE_V | PAGE_RW | PAGE_HOST; /* an entry, as AllocatePageTable makes */
  memcpy(x.mem + RX_S + offsetof(struct System, cr3), &cr3, 8);
  mm = *m0;
  mm.system = (struct System *)(uintptr_t)RX_S; /* an offset in x.mem */
  memcpy(x.mem + RX_M, &mm, sizeof mm);
  memset(&ctx, 0, sizeof ctx);
  ctx.nconsts = 4;
  ctx.consts[1] = RX_CTX + sizeof ctx; /* the store log */
  ctx.consts[3] = 64;
  memcpy(x.mem + RX_CTX, &ctx, sizeof ctx);
  args[0] = RX_M;
  args[1] = RX_CTX;
  out->rc = (u32)RxRun(&x, 0, args, 0);
  out->err = x.err;
  memcpy(out->why, x.why, sizeof out->why);
  memcpy(&out->m, x.mem + RX_M, sizeof out->m);
  memcpy(&ctx, x.mem + RX_CTX, sizeof ctx);
  out->log_used = ctx.consts[2];
  memcpy(out->frames, x.mem + RX_FRAME, sizeof out->frames);
  free(x.mem);
done:
  fbx_wasm_buffer_free(&w);
  for (i = 0; i < nb; ++i) {
    fbx_ir_free(irs[i]);
    if (tcs[i]) FreeTc(tcs[i]);
  }
}

static void RxSetG(struct Machine *m, unsigned r, u64 v) { memcpy(m->weg[r], &v, 8); }
static u64 RxG(const struct Machine *m, unsigned r) {
  u64 v;
  memcpy(&v, m->weg[r], 8);
  return v;
}

/* A Machine at its defaults, with empty TLB slots (no page matches them). */
static void RxMachine(struct Machine *m) {
  unsigned k;
  memset(m, 0, sizeof *m);
  for (k = 0; k < 32; ++k) {
    m->tlb[k].page = (i64)0x7ff0000000000000ll;
    m->tlb[k].entry = 0;
  }
  m->flags = 0x202 | (0x5Au << 24);
  RxSetG(m, 0, 0x7f);                   /* rax */
  RxSetG(m, 1, 0x1234);                 /* rcx */
  RxSetG(m, 2, 0x1100);                 /* rdx: dl = 0 */
  RxSetG(m, 3, 0xfffffffffffffffeull);  /* rbx */
  RxSetG(m, 4, RX_STACK_VA);            /* rsp */
  RxSetG(m, 6, RX_DATA_VA);             /* rsi */
  RxSetG(m, 7, 0x10);                   /* rdi */
  m->ip = RX_CODE;
}

/* inc rax; neg rcx; dec dl; mov [rsi+rdi], al; cmp [rsi+rdi], al;
 * push rbx; pop rdx; call F                       — member 0 at RX_CODE
 * F: inc rbx; ret                                  — member 1 */
static const struct RgInsn kRxMain[] = {
    {RX_CODE + 0, 3, {0x48, 0xff, 0xc0}},
    {RX_CODE + 3, 3, {0x48, 0xf7, 0xd9}},
    {RX_CODE + 6, 2, {0xfe, 0xca}},
    {RX_CODE + 8, 3, {0x88, 0x04, 0x3e}},
    {RX_CODE + 11, 3, {0x38, 0x04, 0x3e}},
    {RX_CODE + 14, 1, {0x53}},
    {RX_CODE + 15, 1, {0x5a}},
    {RX_CODE + 16, 5, {0xe8, 0xeb, 0x00, 0x00, 0x00}}, /* -> RX_CODE+0x100 */
};
static const struct RgInsn kRxFn[] = {
    {RX_CODE + 0x100, 3, {0x48, 0xff, 0xc3}},
    {RX_CODE + 0x103, 1, {0xc3}},
};

TEST(FbxRegionExec, ArithmeticStoresStackCallAndReturn) {
  const struct RgInsn *blocks[2] = {kRxMain, kRxFn};
  unsigned lens[2] = {8, 2};
  struct Machine m0, om;
  struct RxRun r;
  u64 rax, rcx, rbx, ret_slot;
  int mode;
  RxMachine(&m0);
  /* The oracle: blink/alu.c, the functions Tier 1 runs, in program order. */
  om = m0;
  rax = (u64)Inc64(&om, RxG(&m0, 0), 0);
  rcx = (u64)Neg64(&om, RxG(&m0, 1), 0);
  (void)Dec8(&om, RxG(&m0, 2) & 0xff, 0);
  (void)Sub8(&om, rax & 0xff, rax & 0xff); /* cmp [mem]=al, al */
  rbx = (u64)Inc64(&om, RxG(&m0, 3), 0);
  for (mode = 0; mode < 2; ++mode) {
    RxExec(blocks, lens, 2, &m0,
           mode ? FBX_REGION_EMIT_COUNT | FBX_REGION_EMIT_LOG : 0, &r);
    ASSERT_EQ(0, r.err);
    EXPECT_EQ(0u, r.rc);
    EXPECT_EQ(RX_CODE + 21, r.m.ip); /* ret popped the call's return PC */
    EXPECT_EQ(rax, RxG(&r.m, 0));
    EXPECT_EQ(rcx, RxG(&r.m, 1));
    EXPECT_EQ(RxG(&m0, 3), RxG(&r.m, 2)); /* push rbx; pop rdx */
    EXPECT_EQ(rbx, RxG(&r.m, 3));
    EXPECT_EQ(RX_STACK_VA, RxG(&r.m, 4));
    EXPECT_EQ(om.flags, r.m.flags);
    EXPECT_EQ(rax & 0xff, r.frames[0][0x10]);
    EXPECT_EQ('5', r.frames[0][0x11]); /* a byte store writes one byte */
    memcpy(&ret_slot, &r.frames[2][4096 - 8], 8);
    EXPECT_EQ(RX_CODE + 21, ret_slot);
    if (mode) {
      EXPECT_EQ(3u, r.log_used); /* the byte store, the push, the call */
    }
  }
}

TEST(FbxRegionExec, UnaryFlagsMatchBlinkAtEveryWidth) {
  /* inc/dec/neg/not on r8 at widths 8,4,2,1, against blink/alu.c.  The
   * values cross the carry, sign, overflow and nibble edges. */
  static const u64 vals[] = {0, 1, 0x7f, 0x80, 0xff, 0x7fff, 0x8000, 0xffff,
                             0x7fffffff, 0x80000000, 0xffffffff, 0xf,
                             0x7fffffffffffffffull, 0x8000000000000000ull,
                             0xffffffffffffffffull, 0x123456789abcdef0ull};
  static const u8 pre[4][2] = {{0x49, 0}, {0x41, 0}, {0x66, 0x41}, {0x41, 0}};
  static const unsigned widths[4] = {8, 4, 2, 1};
  unsigned v, wi, op;
  for (wi = 0; wi < 4; ++wi) {
    for (op = 0; op < 4; ++op) { /* inc, dec, neg, not */
      for (v = 0; v < sizeof vals / sizeof vals[0]; ++v) {
        struct RgInsn b[2];
        const struct RgInsn *blocks[1] = {b};
        unsigned lens[1] = {2}, n = 0, w = widths[wi];
        struct Machine m0, om;
        struct RxRun r;
        u64 x = vals[v], want, mask = w == 8 ? ~0ull : (1ull << (w * 8)) - 1;
        memset(b, 0, sizeof b);
        b[0].ip = RX_CODE;
        b[0].bytes[n++] = pre[wi][0];
        if (pre[wi][1]) b[0].bytes[n++] = pre[wi][1];
        b[0].bytes[n++] = w == 1 ? 0xfe : (op < 2 ? 0xff : 0xf7);
        if (w == 1 && op >= 2) b[0].bytes[n - 1] = 0xf6;
        b[0].bytes[n++] = (u8)(0xc0 | (op == 0 ? 0 : op == 1 ? 1 : op == 2 ? 3 : 2) << 3);
        b[0].len = (u8)n;
        b[1].ip = RX_CODE + n;
        b[1].len = 2;
        b[1].bytes[0] = 0xeb; /* jmp . (an exit: not a member) */
        b[1].bytes[1] = 0xfe;
        lens[0] = 2;
        RxMachine(&m0);
        m0.flags |= 1; /* CF set: INC/DEC must keep it */
        RxSetG(&m0, 8, x);
        om = m0;
        switch (op) {
          case 0: want = (u64)(w == 8 ? Inc64(&om, x, 0) : w == 4 ? Inc32(&om, x, 0) : w == 2 ? Inc16(&om, x, 0) : Inc8(&om, x, 0)); break;
          case 1: want = (u64)(w == 8 ? Dec64(&om, x, 0) : w == 4 ? Dec32(&om, x, 0) : w == 2 ? Dec16(&om, x, 0) : Dec8(&om, x, 0)); break;
          case 2: want = (u64)(w == 8 ? Neg64(&om, x, 0) : w == 4 ? Neg32(&om, x, 0) : w == 2 ? Neg16(&om, x, 0) : Neg8(&om, x, 0)); break;
          default: want = ~x; break;
        }
        want &= mask;
        want = w == 4 ? want : (w == 8 ? want : (x & ~mask) | want);
        RxExec(blocks, lens, 1, &m0, 0, &r);
        ASSERT_EQ(0, r.err);
        EXPECT_EQ(want, RxG(&r.m, 8));
        EXPECT_EQ(om.flags, r.m.flags);
        EXPECT_EQ(RX_CODE + n, r.m.ip);
      }
    }
  }
}

TEST(FbxRegionExec, UntranslatableAccessLeavesWithNothingCommitted) {
  /* The store of member 0 cannot be made: the page is read-only, or not
   * mapped.  The region returns 1 with m->ip AT the store, the instructions
   * before it committed, the store's memory and everything after untouched. */
  const struct RgInsn *blocks[2] = {kRxMain, kRxFn};
  unsigned lens[2] = {8, 2};
  static const u64 bad[2] = {RX_RO_VA, 0x0000000000900000ull};
  struct Machine m0, om;
  struct RxRun r;
  unsigned k;
  for (k = 0; k < 2; ++k) {
    u64 rax, rcx;
    RxMachine(&m0);
    RxSetG(&m0, 6, bad[k]);
    om = m0;
    rax = (u64)Inc64(&om, RxG(&m0, 0), 0);
    rcx = (u64)Neg64(&om, RxG(&m0, 1), 0);
    (void)Dec8(&om, RxG(&m0, 2) & 0xff, 0);
    RxExec(blocks, lens, 2, &m0, 0, &r);
    ASSERT_EQ(0, r.err);
    EXPECT_EQ(1u, r.rc);
    EXPECT_EQ(RX_CODE + 8, r.m.ip);
    EXPECT_EQ(rax, RxG(&r.m, 0));
    EXPECT_EQ(rcx, RxG(&r.m, 1));
    EXPECT_EQ((RxG(&m0, 2) & ~0xffull) | 0xff, RxG(&r.m, 2));
    EXPECT_EQ(RxG(&m0, 3), RxG(&r.m, 3));
    EXPECT_EQ(RX_STACK_VA, RxG(&r.m, 4));
    EXPECT_EQ(om.flags, r.m.flags);
    EXPECT_EQ(0, r.frames[1][0x10]);
  }
}
