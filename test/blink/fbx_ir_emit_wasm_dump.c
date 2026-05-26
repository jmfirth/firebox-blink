/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2026 Justin Firth (Firebox)                                        │
│                                                                              │
│ Cross-engine validation harness for the §13.2 synthesis pass.               │
│                                                                              │
│ Dumps a representative set of synthesised wasm modules to disk so an        │
│ external harness can parse them via both wasmer::Module::new (native) and   │
│ WebAssembly.compile() (browser) per AGENTS.md invariant 4.                  │
│                                                                              │
│ Usage:                                                                       │
│   fbx_ir_emit_wasm_dump <output_dir>                                         │
│                                                                              │
│ Emits:                                                                       │
│   <dir>/mov_reg_reg.wasm    — MOV %rax, %rbx                                 │
│   <dir>/alu_add.wasm        — ADD %rax, %rbx                                 │
│   <dir>/mov_imm.wasm        — MOV %rax, $imm                                 │
│   <dir>/lea.wasm            — LEA %rcx, [%rax + 16]                          │
│   <dir>/branch_taken.wasm   — JMP rel                                        │
│   <dir>/bailout.wasm        — single BAILOUT                                 │
│                                                                              │
│ Exit codes:                                                                  │
│   0 — all modules emitted successfully                                       │
│   1 — synthesis or write failed                                              │
╚─────────────────────────────────────────────────────────────────────────────*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "blink/fbx_ir.h"
#include "blink/fbx_wasm_emit.h"
#include "blink/rde.h"
#include "blink/threadedcode.h"
#include "blink/types.h"

#define RDE_MOD3 ((u64)0x00C00000ull)

static struct FbxTcBlock *MakeTc(u64 mop, u64 rde_extra, u64 uimm0, i64 disp,
                                 u64 ip, u8 oplen) {
  struct FbxTcBlock *b = (struct FbxTcBlock *)calloc(1, sizeof *b);
  b->start_pc = ip;
  b->end_pc = ip + oplen;
  b->page = ip & ~(u64)4095;
  b->nentries = 1;
  b->entries = (struct FbxTcEntry *)calloc(1, sizeof(struct FbxTcEntry));
  b->entries[0].ip = ip;
  b->entries[0].rde = (mop << 050) | rde_extra;
  b->entries[0].uimm0 = uimm0;
  b->entries[0].disp = disp;
  b->entries[0].oplen = oplen;
  b->entries[0].kind = FBX_TC_KIND_NORMAL;
  return b;
}

static void FreeTc(struct FbxTcBlock *b) {
  free(b->entries);
  free(b);
}

static int WriteFile(const char *path, const u8 *data, size_t len) {
  FILE *f = fopen(path, "wb");
  size_t n;
  if (!f) return 0;
  n = fwrite(data, 1, len, f);
  fclose(f);
  return n == len;
}

static int DumpMovRegReg(const char *dir) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[3];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  char path[1024];
  int rc;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0x1000;
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
  ir.start_pc = 0x1000;
  ir.end_pc = 0x1003;
  tc = MakeTc(0x089, RDE_MOD3, 0, 0, 0x1000, 3);
  fbx_wasm_buffer_init(&out);
  if (!fbx_ir_emit_wasm(&ir, tc, &out)) {
    fprintf(stderr, "mov_reg_reg: synthesis refused\n");
    FreeTc(tc);
    return 0;
  }
  snprintf(path, sizeof path, "%s/mov_reg_reg.wasm", dir);
  rc = WriteFile(path, out.data, out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
  if (!rc) fprintf(stderr, "mov_reg_reg: write failed\n");
  return rc;
}

static int DumpAluAdd(const char *dir) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[5];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  char path[1024];
  int rc;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0x3000;
  insts[1].opcode = FBX_IR_OP_REG_GET;
  insts[1].width = 8;
  insts[1].dst_kind = FBX_IR_KIND_VREG;
  insts[1].dst = 0;
  insts[1].src1_kind = FBX_IR_KIND_GREG;
  insts[1].src1 = 0;
  insts[2].opcode = FBX_IR_OP_REG_GET;
  insts[2].width = 8;
  insts[2].dst_kind = FBX_IR_KIND_VREG;
  insts[2].dst = 1;
  insts[2].src1_kind = FBX_IR_KIND_GREG;
  insts[2].src1 = 1;
  insts[3].opcode = FBX_IR_OP_ADD;
  insts[3].width = 8;
  insts[3].dst_kind = FBX_IR_KIND_VREG;
  insts[3].dst = 2;
  insts[3].src1_kind = FBX_IR_KIND_VREG;
  insts[3].src1 = 0;
  insts[3].src2_kind = FBX_IR_KIND_VREG;
  insts[3].src2 = 1;
  insts[4].opcode = FBX_IR_OP_REG_SET;
  insts[4].width = 8;
  insts[4].dst_kind = FBX_IR_KIND_GREG;
  insts[4].dst = 0;
  insts[4].src1_kind = FBX_IR_KIND_VREG;
  insts[4].src1 = 2;
  ir.insts = insts;
  ir.ninsts = 5;
  ir.nvregs = 3;
  ir.start_pc = 0x3000;
  ir.end_pc = 0x3003;
  tc = MakeTc(0x001, RDE_MOD3, 0, 0, 0x3000, 3);
  fbx_wasm_buffer_init(&out);
  if (!fbx_ir_emit_wasm(&ir, tc, &out)) {
    fprintf(stderr, "alu_add: synthesis refused\n");
    FreeTc(tc);
    return 0;
  }
  snprintf(path, sizeof path, "%s/alu_add.wasm", dir);
  rc = WriteFile(path, out.data, out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
  return rc;
}

static int DumpMovImm(const char *dir) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[2];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  char path[1024];
  int rc;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0x4000;
  insts[1].opcode = FBX_IR_OP_REG_SET;
  insts[1].width = 8;
  insts[1].dst_kind = FBX_IR_KIND_GREG;
  insts[1].dst = 0;
  insts[1].src1_kind = FBX_IR_KIND_IMM;
  insts[1].imm = 0xdeadbeefcafef00dULL;
  ir.insts = insts;
  ir.ninsts = 2;
  ir.nvregs = 0;
  ir.start_pc = 0x4000;
  ir.end_pc = 0x400A;
  tc = MakeTc(0x0B8, 0, 0xdeadbeefcafef00dULL, 0, 0x4000, 10);
  fbx_wasm_buffer_init(&out);
  if (!fbx_ir_emit_wasm(&ir, tc, &out)) {
    fprintf(stderr, "mov_imm: synthesis refused\n");
    FreeTc(tc);
    return 0;
  }
  snprintf(path, sizeof path, "%s/mov_imm.wasm", dir);
  rc = WriteFile(path, out.data, out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
  return rc;
}

static int DumpLea(const char *dir) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[2];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  char path[1024];
  int rc;
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
  insts[1].src2_kind = FBX_IR_KIND_IMM;
  insts[1].imm = 16;
  ir.insts = insts;
  ir.ninsts = 2;
  ir.nvregs = 0;
  ir.start_pc = 0x5000;
  ir.end_pc = 0x5004;
  tc = MakeTc(0x08D, 0, 0, 16, 0x5000, 4);
  fbx_wasm_buffer_init(&out);
  if (!fbx_ir_emit_wasm(&ir, tc, &out)) {
    fprintf(stderr, "lea: synthesis refused\n");
    FreeTc(tc);
    return 0;
  }
  snprintf(path, sizeof path, "%s/lea.wasm", dir);
  rc = WriteFile(path, out.data, out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
  return rc;
}

static int DumpBranchTaken(const char *dir) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[2];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  char path[1024];
  int rc;
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
  tc = MakeTc(0x0E9, 0, 0, 0x100, 0x6000, 5);
  fbx_wasm_buffer_init(&out);
  if (!fbx_ir_emit_wasm(&ir, tc, &out)) {
    fprintf(stderr, "branch_taken: synthesis refused\n");
    FreeTc(tc);
    return 0;
  }
  snprintf(path, sizeof path, "%s/branch_taken.wasm", dir);
  rc = WriteFile(path, out.data, out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
  return rc;
}

static int DumpBailout(const char *dir) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  char path[1024];
  int rc;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_BAILOUT;
  insts[0].imm = 0x7000;
  ir.insts = insts;
  ir.ninsts = 1;
  ir.nvregs = 0;
  ir.start_pc = 0x7000;
  ir.end_pc = 0x7001;
  tc = MakeTc(0x089, RDE_MOD3, 0, 0, 0x7000, 1);
  fbx_wasm_buffer_init(&out);
  if (!fbx_ir_emit_wasm(&ir, tc, &out)) {
    fprintf(stderr, "bailout: synthesis refused\n");
    FreeTc(tc);
    return 0;
  }
  snprintf(path, sizeof path, "%s/bailout.wasm", dir);
  rc = WriteFile(path, out.data, out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
  return rc;
}

/* §13.5 — MOVZX r/m8 → r32.  Emits a module with i64.load8_u + i64.store. */
static int DumpMovzx(const char *dir) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[3];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  char path[1024];
  int rc;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_PC_MARK;
  insts[0].imm = 0x8000;
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
  ir.start_pc = 0x8000;
  ir.end_pc = 0x8004;
  tc = MakeTc(0x1B6, RDE_MOD3, 0, 0, 0x8000, 4);
  fbx_wasm_buffer_init(&out);
  if (!fbx_ir_emit_wasm(&ir, tc, &out)) {
    fprintf(stderr, "movzx: synthesis refused\n");
    FreeTc(tc);
    return 0;
  }
  snprintf(path, sizeof path, "%s/movzx_8_to_32.wasm", dir);
  rc = WriteFile(path, out.data, out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
  return rc;
}

/* §13.5 — ADD r/m, r lifted end-to-end (SET_FLAGS_RAW elided as benign). */
static int DumpAluAddFromLift(const char *dir) {
  struct FbxTcBlock *tc;
  struct FbxIrBlock *ir;
  struct FbxWasmBuffer out;
  char path[1024];
  int rc;
  tc = MakeTc(0x001, RDE_MOD3, 0, 0, 0x9000, 3);
  ir = fbx_ir_lift(tc);
  if (!ir) {
    fprintf(stderr, "alu_add_from_lift: lift failed\n");
    FreeTc(tc);
    return 0;
  }
  fbx_wasm_buffer_init(&out);
  if (!fbx_ir_emit_wasm(ir, tc, &out)) {
    fprintf(stderr, "alu_add_from_lift: synthesis refused\n");
    fbx_ir_free(ir);
    FreeTc(tc);
    return 0;
  }
  snprintf(path, sizeof path, "%s/alu_add_from_lift.wasm", dir);
  rc = WriteFile(path, out.data, out.len);
  fbx_wasm_buffer_free(&out);
  fbx_ir_free(ir);
  FreeTc(tc);
  return rc;
}

/* #599 — generic helper: lift a 2-instruction block (flag-writer + Jcc) and
 * dump to <dir>/<filename>.wasm.  This is the post-#599 emit surface. */
static int Dump599FlagJcc(const char *dir, const char *filename, u64 alu_mop,
                          u64 alu_rde, u64 jcc_mop) {
  /* Build 2-entry TC: ALU op + Jcc.  IPs are chosen so end of ALU op (oplen=3)
   * matches start of Jcc, satisfying the lifter's adjacency invariant. */
  struct FbxTcBlock *b = (struct FbxTcBlock *)calloc(1, sizeof *b);
  struct FbxIrBlock *ir;
  struct FbxWasmBuffer out;
  char path[1024];
  int rc;
  b->start_pc = 0x10000;
  b->end_pc = 0x10005;
  b->page = 0x10000;
  b->nentries = 2;
  b->entries = (struct FbxTcEntry *)calloc(2, sizeof(struct FbxTcEntry));
  b->entries[0].ip = 0x10000;
  b->entries[0].rde = (alu_mop << 050) | alu_rde;
  b->entries[0].oplen = 3;
  b->entries[0].kind = FBX_TC_KIND_NORMAL;
  b->entries[1].ip = 0x10003;
  b->entries[1].rde = (jcc_mop << 050);
  b->entries[1].disp = 0x10;
  b->entries[1].oplen = 2;
  b->entries[1].kind = FBX_TC_KIND_NORMAL;
  ir = fbx_ir_lift(b);
  if (!ir) {
    fprintf(stderr, "%s: lift failed\n", filename);
    free(b->entries);
    free(b);
    return 0;
  }
  fbx_wasm_buffer_init(&out);
  if (!fbx_ir_emit_wasm(ir, b, &out)) {
    fprintf(stderr, "%s: synthesis refused\n", filename);
    fbx_ir_free(ir);
    free(b->entries);
    free(b);
    return 0;
  }
  snprintf(path, sizeof path, "%s/%s.wasm", dir, filename);
  rc = WriteFile(path, out.data, out.len);
  fbx_wasm_buffer_free(&out);
  fbx_ir_free(ir);
  free(b->entries);
  free(b);
  return rc;
}

/* #602 — bare CALL_DIRECT module. */
static int Dump602Call(const char *dir) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  char path[1024];
  int rc;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_CALL_DIRECT;
  insts[0].src2_kind = FBX_IR_KIND_IMM;
  insts[0].src2 = 0xA005;
  insts[0].imm = 0xA100;
  ir.insts = insts;
  ir.ninsts = 1;
  ir.start_pc = 0xA000;
  ir.end_pc = 0xA005;
  tc = MakeTc(0x0E8, 0, 0, 0x100, 0xA000, 5);
  fbx_wasm_buffer_init(&out);
  if (!fbx_ir_emit_wasm(&ir, tc, &out)) {
    fprintf(stderr, "602_call: synthesis refused\n");
    FreeTc(tc);
    return 0;
  }
  snprintf(path, sizeof path, "%s/602_call.wasm", dir);
  rc = WriteFile(path, out.data, out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
  return rc;
}

/* #602 — bare RET module. */
static int Dump602Ret(const char *dir) {
  struct FbxIrBlock ir;
  struct FbxIrInst insts[1];
  struct FbxTcBlock *tc;
  struct FbxWasmBuffer out;
  char path[1024];
  int rc;
  memset(&ir, 0, sizeof ir);
  memset(insts, 0, sizeof insts);
  insts[0].opcode = FBX_IR_OP_RET;
  ir.insts = insts;
  ir.ninsts = 1;
  ir.start_pc = 0xB000;
  ir.end_pc = 0xB001;
  tc = MakeTc(0x0C3, 0, 0, 0, 0xB000, 1);
  fbx_wasm_buffer_init(&out);
  if (!fbx_ir_emit_wasm(&ir, tc, &out)) {
    fprintf(stderr, "602_ret: synthesis refused\n");
    FreeTc(tc);
    return 0;
  }
  snprintf(path, sizeof path, "%s/602_ret.wasm", dir);
  rc = WriteFile(path, out.data, out.len);
  fbx_wasm_buffer_free(&out);
  FreeTc(tc);
  return rc;
}

/* #602 — PUSH RBP + POP RBP end-to-end (lifted from real opcodes). */
static int Dump602PushPop(const char *dir) {
  struct FbxTcBlock *b = (struct FbxTcBlock *)calloc(1, sizeof *b);
  struct FbxIrBlock *ir;
  struct FbxWasmBuffer out;
  char path[1024];
  int rc;
  b->start_pc = 0xD000;
  b->end_pc = 0xD002;
  b->page = 0xD000;
  b->nentries = 2;
  b->entries = (struct FbxTcEntry *)calloc(2, sizeof(struct FbxTcEntry));
  b->entries[0].ip = 0xD000;
  b->entries[0].rde = (u64)0x055 << 050;   /* PUSH RBP */
  b->entries[0].oplen = 1;
  b->entries[0].kind = FBX_TC_KIND_NORMAL;
  b->entries[1].ip = 0xD001;
  b->entries[1].rde = (u64)0x05D << 050;   /* POP RBP */
  b->entries[1].oplen = 1;
  b->entries[1].kind = FBX_TC_KIND_NORMAL;
  ir = fbx_ir_lift(b);
  if (!ir) {
    fprintf(stderr, "602_push_pop: lift failed\n");
    free(b->entries);
    free(b);
    return 0;
  }
  fbx_wasm_buffer_init(&out);
  if (!fbx_ir_emit_wasm(ir, b, &out)) {
    fprintf(stderr, "602_push_pop: synthesis refused\n");
    fbx_ir_free(ir);
    free(b->entries);
    free(b);
    return 0;
  }
  snprintf(path, sizeof path, "%s/602_push_pop.wasm", dir);
  rc = WriteFile(path, out.data, out.len);
  fbx_wasm_buffer_free(&out);
  fbx_ir_free(ir);
  free(b->entries);
  free(b);
  return rc;
}

int main(int argc, char **argv) {
  const char *dir;
  if (argc < 2) {
    fprintf(stderr, "usage: %s <output_dir>\n", argv[0]);
    return 1;
  }
  dir = argv[1];
  if (!DumpMovRegReg(dir)) return 1;
  if (!DumpAluAdd(dir)) return 1;
  if (!DumpMovImm(dir)) return 1;
  if (!DumpLea(dir)) return 1;
  if (!DumpBranchTaken(dir)) return 1;
  if (!DumpBailout(dir)) return 1;
  /* §13.5 additions */
  if (!DumpMovzx(dir)) return 1;
  if (!DumpAluAddFromLift(dir)) return 1;
  /* #599 — lazy-flag synthesis + BRANCH_COND.  These exercise the
   * SET_FLAGS_RAW eager-update path for several (op_kind, jcc_cond)
   * combinations and the BRANCH_COND select-based dispatch. */
  if (!Dump599FlagJcc(dir, "599_add_je", 0x001, RDE_MOD3, 0x074)) return 1;
  if (!Dump599FlagJcc(dir, "599_sub_jne", 0x029, RDE_MOD3, 0x075)) return 1;
  if (!Dump599FlagJcc(dir, "599_and_jz", 0x021, RDE_MOD3, 0x074)) return 1;
  if (!Dump599FlagJcc(dir, "599_xor_jo", 0x031, RDE_MOD3, 0x070)) return 1;
  if (!Dump599FlagJcc(dir, "599_cmp_jb", 0x039, RDE_MOD3, 0x072)) return 1;
  if (!Dump599FlagJcc(dir, "599_test_jbe", 0x085, RDE_MOD3, 0x076)) return 1;
  if (!Dump599FlagJcc(dir, "599_add_jl", 0x001, RDE_MOD3, 0x07C)) return 1;
  if (!Dump599FlagJcc(dir, "599_cmp_jle", 0x039, RDE_MOD3, 0x07E)) return 1;
  /* #602 — CALL_DIRECT / RET / PUSH / POP. */
  if (!Dump602Call(dir)) return 1;
  if (!Dump602Ret(dir)) return 1;
  if (!Dump602PushPop(dir)) return 1;
  fprintf(stdout, "wrote 19 modules to %s\n", dir);
  return 0;
}
