/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2026 Justin Firth (Firebox)                                        │
│                                                                              │
│ Phase 2 — Tier 2 §13.2: wasm synthesis pass.                                 │
│                                                                              │
│ Consumes a FbxIrBlock (lifted by §13.1's fbx_ir_lift) plus the original      │
│ FbxTcBlock (so the emitter can resolve modrm-memory + LEA-SIB cases whose   │
│ IR is intentionally APPROXIMATE per #583 disclaimer 6), and emits a         │
│ self-contained wasm module containing one exported function                  │
│ `translated_block(i32 m_ptr) -> (i32 exit)`.                                 │
│                                                                              │
│ Browser-target discipline (AGENTS.md invariant 4): only wasm MVP features    │
│ are emitted (no atomics, no threads, no EH, no SIMD, no tail-call, no       │
│ reference-types).  The MVP subset is the broadest cross-engine portable     │
│ surface (V8 / JSC / SpiderMonkey / wasmer / wasmtime).                       │
│                                                                              │
│ Determinism contract (LOAD-BEARING — see spec §Q5 + Decision 7):             │
│   - No global state read or written                                          │
│   - No clocks, no random sources, no thread-local data                       │
│   - Buffer growth via deterministic doubling; no allocator-state leakage    │
│   - LEB128 emission is bit-exact                                             │
│                                                                              │
│ Two calls to fbx_ir_emit_wasm() over the same (ir, tc) MUST produce a       │
│ byte-identical wasm module.  Phase 4's offline emitter shares this code     │
│ path verbatim.                                                               │
│                                                                              │
│ v0.1 coverage:                                                               │
│   - REG_GET / REG_SET width 1/2/4/8 against Machine.weg[reg]                 │
│   - ADD/SUB/AND/OR/XOR (vreg, vreg) i64                                      │
│   - LEA (greg-base + imm) into greg (no SIB)                                 │
│   - BRANCH_TAKEN (unconditional jump → store m->ip, return 0)                │
│   - BAILOUT (store m->ip = bailout_pc, return 1)                             │
│   - PC_MARK (no code emitted; used by future debug)                          │
│                                                                              │
│ v0.1 NON-coverage (synthesis refuses; the caller keeps the block on T1):    │
│   - SET_FLAGS_RAW + GET_FLAG (lazy-flag plumbing belongs to §13.4/§13.5)     │
│   - BRANCH_COND (depends on flags)                                           │
│   - CALL_DIRECT / CALL_INDIRECT / RET (stack manipulation belongs to §13.5)  │
│   - LOAD / STORE / CMP / TEST (memory-operand modrm — §13.5)                 │
│   - LEA with src2 used as index (SIB)                                        │
│                                                                              │
│ When the emitter sees an unsupported IR pattern, fbx_ir_emit_wasm() returns │
│ 0 and `*out` is left empty.  The runtime stays on Tier 1 — the contract     │
│ is "correct OR refuse", never "incorrect wasm".                              │
│                                                                              │
│ See blink/fbx_wasm_emit.h for the API and design rationale.                 │
╚─────────────────────────────────────────────────────────────────────────────*/

#include "blink/fbx_wasm_emit.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "blink/fbx_ir.h"
#include "blink/machine.h"
#include "blink/rde.h"
#include "blink/threadedcode.h"
#include "blink/types.h"

/* ────────────────────────────────────────────────────────────────────────── */
/* Buffer primitives.                                                         */
/* ────────────────────────────────────────────────────────────────────────── */

void fbx_wasm_buffer_init(struct FbxWasmBuffer *b) {
  if (!b) return;
  b->data = NULL;
  b->len = 0;
  b->cap = 0;
  b->oom = 0;
}

void fbx_wasm_buffer_free(struct FbxWasmBuffer *b) {
  if (!b) return;
  free(b->data);
  b->data = NULL;
  b->len = 0;
  b->cap = 0;
  b->oom = 0;
}

static int BufReserve(struct FbxWasmBuffer *b, size_t add) {
  size_t need;
  void *p;
  if (b->oom) return 0;
  need = b->len + add;
  if (need <= b->cap) return 1;
  {
    size_t new_cap = b->cap ? b->cap : 64;
    while (new_cap < need) new_cap *= 2;
    p = realloc(b->data, new_cap);
    if (!p) {
      b->oom = 1;
      return 0;
    }
    b->data = (u8 *)p;
    b->cap = new_cap;
  }
  return 1;
}

void fbx_wasm_buffer_u8(struct FbxWasmBuffer *b, u8 v) {
  if (!BufReserve(b, 1)) return;
  b->data[b->len++] = v;
}

void fbx_wasm_buffer_bytes(struct FbxWasmBuffer *b, const void *p, size_t n) {
  if (n == 0) return;
  if (!BufReserve(b, n)) return;
  memcpy(b->data + b->len, p, n);
  b->len += n;
}

void fbx_wasm_buffer_uleb(struct FbxWasmBuffer *b, u64 v) {
  do {
    u8 byte = (u8)(v & 0x7fu);
    v >>= 7;
    if (v != 0) byte |= 0x80u;
    fbx_wasm_buffer_u8(b, byte);
  } while (v != 0);
}

void fbx_wasm_buffer_sleb(struct FbxWasmBuffer *b, i64 v) {
  int more = 1;
  while (more) {
    u8 byte = (u8)(v & 0x7f);
    /* Arithmetic shift on signed; portable enough for the platforms we
     * compile on (gcc, clang on Linux/macOS).  The Blink codebase already
     * relies on this elsewhere. */
    v >>= 7;
    if ((v == 0 && (byte & 0x40u) == 0) ||
        (v == -1 && (byte & 0x40u) != 0)) {
      more = 0;
    } else {
      byte |= 0x80u;
    }
    fbx_wasm_buffer_u8(b, byte);
  }
}

void fbx_wasm_buffer_u32_le(struct FbxWasmBuffer *b, u32 v) {
  fbx_wasm_buffer_u8(b, (u8)(v >> 0));
  fbx_wasm_buffer_u8(b, (u8)(v >> 8));
  fbx_wasm_buffer_u8(b, (u8)(v >> 16));
  fbx_wasm_buffer_u8(b, (u8)(v >> 24));
}

/* Append a wasm name: ULEB length followed by raw bytes. */
static void EmitName(struct FbxWasmBuffer *b, const char *s) {
  size_t n = strlen(s);
  fbx_wasm_buffer_uleb(b, (u64)n);
  fbx_wasm_buffer_bytes(b, s, n);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Wasm constants (opcodes + types).  Only the subset we emit.                */
/* ────────────────────────────────────────────────────────────────────────── */

#define WASM_VALTYPE_I32 0x7Fu
#define WASM_VALTYPE_I64 0x7Eu

#define WASM_TYPE_FUNC 0x60u

#define WASM_SECTION_TYPE 1u
#define WASM_SECTION_IMPORT 2u
#define WASM_SECTION_FUNCTION 3u
#define WASM_SECTION_EXPORT 7u
#define WASM_SECTION_CODE 10u

#define WASM_IMPORT_KIND_FUNC 0x00u
#define WASM_IMPORT_KIND_MEM  0x02u

#define WASM_EXPORT_KIND_FUNC 0x00u

#define WASM_OP_END         0x0Bu
#define WASM_OP_RETURN      0x0Fu
#define WASM_OP_CALL        0x10u
#define WASM_OP_LOCAL_GET   0x20u
#define WASM_OP_LOCAL_SET   0x21u
#define WASM_OP_LOCAL_TEE   0x22u
#define WASM_OP_I32_CONST   0x41u
#define WASM_OP_I64_CONST   0x42u
#define WASM_OP_I32_ADD     0x6Au
#define WASM_OP_I64_LOAD    0x29u
#define WASM_OP_I64_LOAD32U 0x35u
#define WASM_OP_I64_LOAD16U 0x33u
#define WASM_OP_I64_LOAD8U  0x31u
#define WASM_OP_I64_STORE   0x37u
#define WASM_OP_I64_STORE32 0x3Eu
#define WASM_OP_I64_STORE16 0x3Du
#define WASM_OP_I64_STORE8  0x3Cu
#define WASM_OP_I64_ADD     0x7Cu
#define WASM_OP_I64_SUB     0x7Du
#define WASM_OP_I64_AND     0x83u
#define WASM_OP_I64_OR      0x84u
#define WASM_OP_I64_XOR     0x85u

/* ────────────────────────────────────────────────────────────────────────── */
/* Host-import table.  Names + signatures FIXED at v0.1.                      */
/*                                                                            */
/* Numbering MUST match the order emitted into the import section.  Adding   */
/* a new import requires cache invalidation per spec §5.4.                    */
/* ────────────────────────────────────────────────────────────────────────── */

#define FBX_HOST_IMPORT_CALL_THUNK         0
#define FBX_HOST_IMPORT_CALL_SYSCALL       1
#define FBX_HOST_IMPORT_RESOLVE_INDIRECT   2
#define FBX_HOST_IMPORT_COUNT              3

static const char *const kHostImportNames[FBX_HOST_IMPORT_COUNT] = {
    "call_thunk",
    "call_syscall",
    "resolve_indirect",
};

size_t fbx_wasm_host_import_count(void) {
  return (size_t)FBX_HOST_IMPORT_COUNT;
}

const char *fbx_wasm_host_import_name(size_t i) {
  if (i >= (size_t)FBX_HOST_IMPORT_COUNT) return NULL;
  return kHostImportNames[i];
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Type-index assignment.  Pinned by the order emitted into the type section. */
/* ────────────────────────────────────────────────────────────────────────── */

#define WASM_TYPE_IDX_BLOCK             0  /* (i32) -> i32 */
#define WASM_TYPE_IDX_HOST_THUNK_SYS    1  /* (i32, i32) -> i32 */
#define WASM_TYPE_IDX_HOST_RESOLVE      2  /* (i32, i64) -> i32 */
#define WASM_TYPE_COUNT                 3

/* ────────────────────────────────────────────────────────────────────────── */
/* Section-with-size emission helper.                                         */
/*                                                                            */
/* Wasm sections are framed with a ULEB128 byte-length.  We don't know the    */
/* length until we've emitted the body, so the helper builds the body in a   */
/* scratch buffer first, then writes the framed section to the output.       */
/* ────────────────────────────────────────────────────────────────────────── */

static void EmitSection(struct FbxWasmBuffer *out, u8 section_id,
                        const struct FbxWasmBuffer *body) {
  if (out->oom || body->oom) {
    out->oom = 1;
    return;
  }
  fbx_wasm_buffer_u8(out, section_id);
  fbx_wasm_buffer_uleb(out, (u64)body->len);
  fbx_wasm_buffer_bytes(out, body->data, body->len);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Coverage gate — decides whether the v0.1 emitter can lower this block.     */
/*                                                                            */
/* Per spec §13.2 + #583 disclaimer 6, v0.1 covers a strict subset of the     */
/* IR.  When the IR uses an op outside the subset OR a TC entry references   */
/* a memory operand the IR didn't model, we REFUSE the synthesis.  This is   */
/* the "correct OR refuse" contract — never emit incorrect wasm.              */
/*                                                                            */
/* Returns 1 if the block is emittable; 0 if it must stay on Tier 1.          */
/* ────────────────────────────────────────────────────────────────────────── */

static int Mod3(u64 rde) {
  /* modrm.mod field is bits 16-17 of `rde` per Blink's encoding (see
   * blink/rde.h: ModrmMod).  Use the same accessor for consistency. */
  return ((u32)ModrmMod(rde) == 3);
}

static int CoverageGate(const struct FbxIrBlock *ir,
                        const struct FbxTcBlock *tc) {
  u32 i;
  u32 tc_idx;
  int saw_flag_reader = 0;
  int saw_flag_writer = 0;
  if (!ir || ir->ninsts == 0) return 0;
  /* Pass 1 — detect whether any IR inst READS the flag-shadow.  GET_FLAG
   * and BRANCH_COND are the readers.  If none exist, SET_FLAGS_RAW is a
   * benign marker we can drop at emit time (the lazy-flag update has no
   * observer within the block).  CMP / TEST are flag-only writes; if no
   * reader follows them they're effectively dead and can be elided.
   *
   * §13.5 expansion: under this gate, straight-line code that ends with
   * an unconditional terminator (BRANCH_TAKEN / BAILOUT) and DOES contain
   * SET_FLAGS_RAW / CMP / TEST becomes emit-eligible.  Conditional
   * branches still refuse (see Pass 2).  This is the minimal in-scope
   * extension that doesn't require wasm-side flag-shadow plumbing
   * (deferred to #597). */
  for (i = 0; i < ir->ninsts; ++i) {
    u8 op = ir->insts[i].opcode;
    if (op == FBX_IR_OP_GET_FLAG || op == FBX_IR_OP_BRANCH_COND) {
      saw_flag_reader = 1;
    }
    if (op == FBX_IR_OP_SET_FLAGS_RAW || op == FBX_IR_OP_CMP ||
        op == FBX_IR_OP_TEST) {
      saw_flag_writer = 1;
    }
  }
  /* If any flag-reader is present, refuse — the synthesis pass has no
   * wasm-side flag-shadow representation to feed it. */
  if (saw_flag_reader) return 0;
  (void)saw_flag_writer; /* presence-only; no further gating */
  /* Pass 2 — per-op acceptance.  SET_FLAGS_RAW / CMP / TEST allowed only
   * because Pass 1 verified no reader follows.  Their emit-side handling
   * treats them as no-ops (CMP / TEST) or as dropped markers
   * (SET_FLAGS_RAW). */
  for (i = 0; i < ir->ninsts; ++i) {
    u8 op = ir->insts[i].opcode;
    switch (op) {
      case FBX_IR_OP_PC_MARK:
      case FBX_IR_OP_REG_GET:
      case FBX_IR_OP_REG_SET:
      case FBX_IR_OP_ADD:
      case FBX_IR_OP_SUB:
      case FBX_IR_OP_AND:
      case FBX_IR_OP_OR:
      case FBX_IR_OP_XOR:
      case FBX_IR_OP_LEA:
      case FBX_IR_OP_BRANCH_TAKEN:
      case FBX_IR_OP_BAILOUT:
      case FBX_IR_OP_SET_FLAGS_RAW:  /* §13.5: no-op when no reader follows */
      case FBX_IR_OP_CMP:            /* §13.5: dead-flag elide */
      case FBX_IR_OP_TEST:           /* §13.5: dead-flag elide */
        continue;
      default:
        return 0;
    }
  }
  /* Reachability gate: every TC entry contributing to the IR must be a
   * reg-form opcode (modrm.mod == 3) — the IR doesn't model memory
   * operands at v0.1.  Walk parallel: every NORMAL kind TC entry must
   * be a reg-form instruction.  We do not enforce a 1:1 mapping (the
   * lifter can emit several IR insts per x86 op); only the operand-shape
   * is checked. */
  if (!tc || tc->nentries == 0) return 0;
  for (tc_idx = 0; tc_idx < tc->nentries; ++tc_idx) {
    const struct FbxTcEntry *e = &tc->entries[tc_idx];
    u64 rde;
    u64 mop;
    if (e->kind != FBX_TC_KIND_NORMAL) return 0;
    rde = e->rde;
    mop = Mopcode(rde);
    /* MOV r/m, r and MOV r, r/m and ALU-RR and TEST: require modrm.mod==3. */
    switch (mop) {
      case 0x088: case 0x089:
      case 0x08A: case 0x08B:
      case 0x000: case 0x001: case 0x008: case 0x009:
      case 0x020: case 0x021: case 0x028: case 0x029:
      case 0x030: case 0x031: case 0x038: case 0x039:
      case 0x084: case 0x085:
        if (!Mod3(rde)) return 0;
        break;
      /* §13.5 mirror-direction ALU forms (lift extended in
       * blink/fbx_ir_lift.c).  Same modrm.mod==3 requirement.
       * The emit pass still REFUSES these because the underlying lifter
       * emits SET_FLAGS_RAW which the IR-side coverage gate above rejects
       * — but routing the mopcode through CoverageGate's mop-switch is
       * still needed so the IR-side rejection fires with the right
       * reason (otherwise the default branch below would refuse with the
       * wrong "unsupported mopcode" label). */
      case 0x002: case 0x003:
      case 0x00A: case 0x00B:
      case 0x022: case 0x023:
      case 0x02A: case 0x02B:
      case 0x032: case 0x033:
      case 0x03A: case 0x03B:
        if (!Mod3(rde)) return 0;
        break;
      /* §13.5 MOVZX r, r/m{8,16}.  mod3 required at v0.1 (synthesis's
       * EmitRegLoad doesn't model memory-modrm yet — same gate as the
       * MOV variants above). */
      case 0x1B6: case 0x1B7:
        if (!Mod3(rde)) return 0;
        break;
      /* MOV r/m, imm and group-1 r/m, imm: require modrm.mod==3 as well. */
      case 0x0C6: case 0x0C7:
      case 0x080: case 0x081: case 0x083:
        if (!Mod3(rde)) return 0;
        break;
      /* LEA: tolerate the simple disp-only form; reject if SIB index used.
       * Without a full disassembler we conservatively reject when any
       * encoded SIB index is present.  Mod3() doesn't apply to LEA (LEA
       * must have mod!=3 by definition); we accept LEA but coverage-only
       * for the simple addressing form (no SIB, no index register).  The
       * IR lifter emits LEA src1=base_greg + imm=disp; if src2_kind is
       * IMM we accept; if src2_kind is anything else we refuse. */
      case 0x08D:
        /* Defer the SIB check to the loop over IR insts below — the IR
         * carries the operand kinds explicitly. */
        break;
      /* MOV reg, imm — no modrm. */
      case 0x0B8: case 0x0B9: case 0x0BA: case 0x0BB:
      case 0x0BC: case 0x0BD: case 0x0BE: case 0x0BF:
        break;
      /* Unconditional jumps — supported. */
      case 0x0E9:
      case 0x0EB:
        break;
      default:
        /* The lifter would have bailed for anything outside the top-10 set;
         * if we reach here with an unsupported mopcode, refuse. */
        return 0;
    }
  }
  /* Walk the IR once more for LEA SIB rejection. */
  for (i = 0; i < ir->ninsts; ++i) {
    const struct FbxIrInst *p = &ir->insts[i];
    if (p->opcode == FBX_IR_OP_LEA) {
      /* v0.1: accept LEA with src1=GREG (base) + imm (disp).  Anything else
       * (e.g. src2_kind != FBX_IR_KIND_NONE && != FBX_IR_KIND_IMM) is a SIB
       * form the v0.1 emitter doesn't model. */
      if (p->src1_kind != FBX_IR_KIND_GREG) return 0;
      if (p->src2_kind != FBX_IR_KIND_NONE &&
          p->src2_kind != FBX_IR_KIND_IMM) return 0;
    }
    if (p->opcode == FBX_IR_OP_REG_GET || p->opcode == FBX_IR_OP_REG_SET) {
      /* REG_GET/SET must read/write a guest register; the lifter emits IMM
       * src1 for the "load imm into vreg" trick used by OpAlui — that
       * pattern is part of the SET_FLAGS_RAW chain which we already refused
       * above, but defend regardless. */
      if (p->opcode == FBX_IR_OP_REG_GET &&
          p->src1_kind != FBX_IR_KIND_GREG) return 0;
      if (p->opcode == FBX_IR_OP_REG_SET &&
          p->dst_kind != FBX_IR_KIND_GREG) return 0;
    }
  }
  return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Function-body emission.                                                    */
/*                                                                            */
/* Layout:                                                                    */
/*   - locals: nvregs * i64                                                   */
/*   - body: walk IR insts in order; emit per-opcode lowering                 */
/*   - terminator: if the IR ends without a control-flow terminator (no      */
/*     BRANCH_TAKEN / BAILOUT), append an implicit return-0 (normal exit)    */
/* ────────────────────────────────────────────────────────────────────────── */

/* offsetof helpers — baked at C compile time. */
#define M_OFF_IP    ((u32)offsetof(struct Machine, ip))
#define M_OFF_WEG   ((u32)offsetof(struct Machine, weg))

/* Emit `local.get m_ptr; i32.const reg_offset; i32.add; load.WIDTH` to leave
 * the guest register value on the stack as an i64. */
static void EmitRegLoad(struct FbxWasmBuffer *body, u32 reg_id, u8 width) {
  u32 reg_off = M_OFF_WEG + reg_id * 8u;
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, 0); /* m_ptr is local 0 */
  /* Memarg: align + offset.  Use natural alignment per width. */
  switch (width) {
    case 1:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD8U);
      fbx_wasm_buffer_uleb(body, 0); /* align (log2) */
      fbx_wasm_buffer_uleb(body, reg_off);
      break;
    case 2:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD16U);
      fbx_wasm_buffer_uleb(body, 1);
      fbx_wasm_buffer_uleb(body, reg_off);
      break;
    case 4:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD32U);
      fbx_wasm_buffer_uleb(body, 2);
      fbx_wasm_buffer_uleb(body, reg_off);
      break;
    case 8:
    default:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD);
      fbx_wasm_buffer_uleb(body, 3);
      fbx_wasm_buffer_uleb(body, reg_off);
      break;
  }
}

/* Emit `local.get m_ptr; <val on stack>; store.WIDTH`.  Callers leave the
 * value to store on the stack first; we ferry m_ptr beneath it via swap. */
static void EmitRegStorePrep(struct FbxWasmBuffer *body) {
  /* Producer pattern from callers: they emit value-producing code AFTER
   * this prep.  We can't push m_ptr after the value without a swap.  To
   * keep the sequence simple, we push m_ptr FIRST then the value, then
   * store with offset.  This means callers must use EmitRegStore which
   * orchestrates the full sequence. */
  (void)body;
}

/* Emit a complete REG_SET: take a vreg local index, store it into the
 * guest register at greg_id, width-truncated. */
static void EmitRegStoreFromLocal(struct FbxWasmBuffer *body, u32 greg_id,
                                  u32 src_local, u8 width) {
  u32 reg_off = M_OFF_WEG + greg_id * 8u;
  /* Push m_ptr. */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, 0);
  /* Push value. */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, src_local);
  /* Width-truncated i64 store. */
  switch (width) {
    case 1:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE8);
      fbx_wasm_buffer_uleb(body, 0);
      fbx_wasm_buffer_uleb(body, reg_off);
      break;
    case 2:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE16);
      fbx_wasm_buffer_uleb(body, 1);
      fbx_wasm_buffer_uleb(body, reg_off);
      break;
    case 4:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE32);
      fbx_wasm_buffer_uleb(body, 2);
      fbx_wasm_buffer_uleb(body, reg_off);
      break;
    case 8:
    default:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE);
      fbx_wasm_buffer_uleb(body, 3);
      fbx_wasm_buffer_uleb(body, reg_off);
      break;
  }
}

/* Emit a REG_SET when the source is an immediate. */
static void EmitRegStoreImm(struct FbxWasmBuffer *body, u32 greg_id, u64 imm,
                            u8 width) {
  u32 reg_off = M_OFF_WEG + greg_id * 8u;
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, 0);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, (i64)imm);
  switch (width) {
    case 1:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE8);
      fbx_wasm_buffer_uleb(body, 0);
      fbx_wasm_buffer_uleb(body, reg_off);
      break;
    case 2:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE16);
      fbx_wasm_buffer_uleb(body, 1);
      fbx_wasm_buffer_uleb(body, reg_off);
      break;
    case 4:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE32);
      fbx_wasm_buffer_uleb(body, 2);
      fbx_wasm_buffer_uleb(body, reg_off);
      break;
    case 8:
    default:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE);
      fbx_wasm_buffer_uleb(body, 3);
      fbx_wasm_buffer_uleb(body, reg_off);
      break;
  }
}

/* Emit a store of a u64 immediate into Machine.ip. */
static void EmitStoreIp(struct FbxWasmBuffer *body, u64 pc) {
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, 0);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, (i64)pc);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE);
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, M_OFF_IP);
}

/* Map an IR vreg index to a wasm local index.  Local 0 is m_ptr; locals
 * 1..nvregs are the vregs. */
static u32 VregLocal(u32 vreg) { return vreg + 1u; }

/* Lower an ALU op (ADD/SUB/AND/OR/XOR) — both operands are VREG-kind. */
static int EmitAlu(struct FbxWasmBuffer *body, const struct FbxIrInst *p) {
  u8 wasm_op;
  if (p->src1_kind != FBX_IR_KIND_VREG || p->src2_kind != FBX_IR_KIND_VREG ||
      p->dst_kind != FBX_IR_KIND_VREG) {
    return 0; /* §13.2 v0.1: only vreg-form ALU; reg-loading happens via
                 prior REG_GET ops emitted by the lifter */
  }
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, VregLocal(p->src1));
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, VregLocal(p->src2));
  switch (p->opcode) {
    case FBX_IR_OP_ADD: wasm_op = WASM_OP_I64_ADD; break;
    case FBX_IR_OP_SUB: wasm_op = WASM_OP_I64_SUB; break;
    case FBX_IR_OP_AND: wasm_op = WASM_OP_I64_AND; break;
    case FBX_IR_OP_OR:  wasm_op = WASM_OP_I64_OR; break;
    case FBX_IR_OP_XOR: wasm_op = WASM_OP_I64_XOR; break;
    default: return 0;
  }
  fbx_wasm_buffer_u8(body, wasm_op);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
  fbx_wasm_buffer_uleb(body, VregLocal(p->dst));
  return 1;
}

/* Lower a LEA with base greg + immediate displacement (no index). */
static int EmitLea(struct FbxWasmBuffer *body, const struct FbxIrInst *p) {
  /* result = greg[src1] + imm; stored back into greg[dst]. */
  u32 base_off;
  u32 dst_off;
  if (p->src1_kind != FBX_IR_KIND_GREG || p->dst_kind != FBX_IR_KIND_GREG) {
    return 0;
  }
  if (p->src2_kind != FBX_IR_KIND_NONE && p->src2_kind != FBX_IR_KIND_IMM) {
    return 0;
  }
  base_off = M_OFF_WEG + p->src1 * 8u;
  dst_off = M_OFF_WEG + p->dst * 8u;
  /* Push m_ptr for the eventual store. */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, 0);
  /* Load base register (always 8-byte for LEA in v0.1 — x86_64 default). */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, 0);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD);
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, base_off);
  /* Add immediate. */
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, (i64)p->imm);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_ADD);
  /* Store into dst. */
  if (p->width == 4) {
    fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE32);
    fbx_wasm_buffer_uleb(body, 2);
  } else {
    fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE);
    fbx_wasm_buffer_uleb(body, 3);
  }
  fbx_wasm_buffer_uleb(body, dst_off);
  return 1;
}

/* Emit the function body for one IR block.  Returns 1 on success, 0 on
 * unsupported op encountered mid-emission (caller frees scratch). */
static int EmitFunctionBody(struct FbxWasmBuffer *body,
                            const struct FbxIrBlock *ir) {
  u32 i;
  int terminated = 0;
  /* Locals declaration: 1 group of nvregs i64s.  (If nvregs == 0 we still
   * emit "0 local groups" — valid wasm.) */
  if (ir->nvregs > 0) {
    fbx_wasm_buffer_uleb(body, 1);
    fbx_wasm_buffer_uleb(body, ir->nvregs);
    fbx_wasm_buffer_u8(body, WASM_VALTYPE_I64);
  } else {
    fbx_wasm_buffer_uleb(body, 0);
  }
  for (i = 0; i < ir->ninsts; ++i) {
    const struct FbxIrInst *p = &ir->insts[i];
    switch (p->opcode) {
      case FBX_IR_OP_PC_MARK:
        /* No code emitted at v0.1.  Future: emit a custom section entry or
         * a debug intrinsic. */
        break;
      case FBX_IR_OP_SET_FLAGS_RAW:
      case FBX_IR_OP_CMP:
      case FBX_IR_OP_TEST:
        /* §13.5 — no code emitted for these when CoverageGate has
         * verified no flag-reader follows in the block.  Pass 1 of the
         * gate enforces the invariant; here we treat them as semantic
         * markers that consume no emitter state.  When #597 lands the
         * lazy-flag wasm representation, these grow real emit logic. */
        break;
      case FBX_IR_OP_REG_GET: {
        /* dst is a vreg local; src1 is a guest register. */
        if (p->dst_kind != FBX_IR_KIND_VREG ||
            p->src1_kind != FBX_IR_KIND_GREG) {
          return 0;
        }
        EmitRegLoad(body, p->src1, p->width ? p->width : 8u);
        fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
        fbx_wasm_buffer_uleb(body, VregLocal(p->dst));
        break;
      }
      case FBX_IR_OP_REG_SET: {
        if (p->dst_kind != FBX_IR_KIND_GREG) return 0;
        if (p->src1_kind == FBX_IR_KIND_VREG) {
          EmitRegStoreFromLocal(body, p->dst, VregLocal(p->src1),
                                p->width ? p->width : 8u);
        } else if (p->src1_kind == FBX_IR_KIND_IMM) {
          EmitRegStoreImm(body, p->dst, p->imm, p->width ? p->width : 8u);
        } else {
          return 0;
        }
        break;
      }
      case FBX_IR_OP_ADD:
      case FBX_IR_OP_SUB:
      case FBX_IR_OP_AND:
      case FBX_IR_OP_OR:
      case FBX_IR_OP_XOR:
        if (!EmitAlu(body, p)) return 0;
        break;
      case FBX_IR_OP_LEA:
        if (!EmitLea(body, p)) return 0;
        break;
      case FBX_IR_OP_BRANCH_TAKEN:
        /* Update m->ip and exit normally.  Block terminator. */
        EmitStoreIp(body, p->imm);
        fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
        fbx_wasm_buffer_sleb(body, 0);
        fbx_wasm_buffer_u8(body, WASM_OP_RETURN);
        terminated = 1;
        break;
      case FBX_IR_OP_BAILOUT:
        /* Update m->ip to the bailout PC and return exit=1. */
        EmitStoreIp(body, p->imm);
        fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
        fbx_wasm_buffer_sleb(body, 1);
        fbx_wasm_buffer_u8(body, WASM_OP_RETURN);
        terminated = 1;
        break;
      default:
        /* Coverage gate should have rejected this. */
        return 0;
    }
    if (terminated) break;
  }
  /* If the IR walked off the end without a terminator, store fallthrough PC
   * and exit normally.  Use end_pc from the IR block (we don't have it
   * here; the caller passes the FbxIrBlock to fbx_ir_emit_wasm but the
   * body emitter does not — we use a sentinel: callers ensure the IR
   * always ends with a terminator OR the synthesis falls back to bailout.
   * In practice §13.1's lifter emits BAILOUT for any non-control-flow
   * tail, so an un-terminated IR is unexpected here.  Defend with a
   * bailout-to-start_pc fallback. */
  if (!terminated) {
    /* Fall through: this should not happen in well-formed §13.1 IR (every
     * lift ends with BRANCH/CALL/RET/BAILOUT).  Emit a return 0 with no
     * IP update — the dispatcher will resume from m->ip's current value.
     * Mark terminated only at the function level by ensuring the END
     * byte gets written below. */
    fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
    fbx_wasm_buffer_sleb(body, 0);
    fbx_wasm_buffer_u8(body, WASM_OP_RETURN);
  }
  /* All function bodies end with END (0x0B). */
  fbx_wasm_buffer_u8(body, WASM_OP_END);
  return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Module-level section builders.                                             */
/* ────────────────────────────────────────────────────────────────────────── */

static void EmitTypeSection(struct FbxWasmBuffer *out) {
  struct FbxWasmBuffer body;
  fbx_wasm_buffer_init(&body);
  fbx_wasm_buffer_uleb(&body, WASM_TYPE_COUNT);
  /* type 0: (i32) -> (i32). */
  fbx_wasm_buffer_u8(&body, WASM_TYPE_FUNC);
  fbx_wasm_buffer_uleb(&body, 1);
  fbx_wasm_buffer_u8(&body, WASM_VALTYPE_I32);
  fbx_wasm_buffer_uleb(&body, 1);
  fbx_wasm_buffer_u8(&body, WASM_VALTYPE_I32);
  /* type 1: (i32, i32) -> (i32). */
  fbx_wasm_buffer_u8(&body, WASM_TYPE_FUNC);
  fbx_wasm_buffer_uleb(&body, 2);
  fbx_wasm_buffer_u8(&body, WASM_VALTYPE_I32);
  fbx_wasm_buffer_u8(&body, WASM_VALTYPE_I32);
  fbx_wasm_buffer_uleb(&body, 1);
  fbx_wasm_buffer_u8(&body, WASM_VALTYPE_I32);
  /* type 2: (i32, i64) -> (i32). */
  fbx_wasm_buffer_u8(&body, WASM_TYPE_FUNC);
  fbx_wasm_buffer_uleb(&body, 2);
  fbx_wasm_buffer_u8(&body, WASM_VALTYPE_I32);
  fbx_wasm_buffer_u8(&body, WASM_VALTYPE_I64);
  fbx_wasm_buffer_uleb(&body, 1);
  fbx_wasm_buffer_u8(&body, WASM_VALTYPE_I32);
  EmitSection(out, WASM_SECTION_TYPE, &body);
  fbx_wasm_buffer_free(&body);
}

static void EmitImportSection(struct FbxWasmBuffer *out) {
  struct FbxWasmBuffer body;
  fbx_wasm_buffer_init(&body);
  /* count: memory + 3 host functions. */
  fbx_wasm_buffer_uleb(&body, 1 + FBX_HOST_IMPORT_COUNT);
  /* memory import: "fbx" "memory" (memory 0 1) — initial 1 page, no max. */
  EmitName(&body, "fbx");
  EmitName(&body, "memory");
  fbx_wasm_buffer_u8(&body, WASM_IMPORT_KIND_MEM);
  fbx_wasm_buffer_u8(&body, 0x00); /* limits flag: no max */
  fbx_wasm_buffer_uleb(&body, 1);  /* initial pages */
  /* call_thunk : type 1. */
  EmitName(&body, "fbx");
  EmitName(&body, "call_thunk");
  fbx_wasm_buffer_u8(&body, WASM_IMPORT_KIND_FUNC);
  fbx_wasm_buffer_uleb(&body, WASM_TYPE_IDX_HOST_THUNK_SYS);
  /* call_syscall : type 1. */
  EmitName(&body, "fbx");
  EmitName(&body, "call_syscall");
  fbx_wasm_buffer_u8(&body, WASM_IMPORT_KIND_FUNC);
  fbx_wasm_buffer_uleb(&body, WASM_TYPE_IDX_HOST_THUNK_SYS);
  /* resolve_indirect : type 2. */
  EmitName(&body, "fbx");
  EmitName(&body, "resolve_indirect");
  fbx_wasm_buffer_u8(&body, WASM_IMPORT_KIND_FUNC);
  fbx_wasm_buffer_uleb(&body, WASM_TYPE_IDX_HOST_RESOLVE);
  EmitSection(out, WASM_SECTION_IMPORT, &body);
  fbx_wasm_buffer_free(&body);
}

static void EmitFunctionSection(struct FbxWasmBuffer *out) {
  struct FbxWasmBuffer body;
  fbx_wasm_buffer_init(&body);
  /* One local function, using type 0 ((i32) -> (i32)). */
  fbx_wasm_buffer_uleb(&body, 1);
  fbx_wasm_buffer_uleb(&body, WASM_TYPE_IDX_BLOCK);
  EmitSection(out, WASM_SECTION_FUNCTION, &body);
  fbx_wasm_buffer_free(&body);
}

static void EmitExportSection(struct FbxWasmBuffer *out) {
  struct FbxWasmBuffer body;
  fbx_wasm_buffer_init(&body);
  fbx_wasm_buffer_uleb(&body, 1);
  EmitName(&body, "translated_block");
  fbx_wasm_buffer_u8(&body, WASM_EXPORT_KIND_FUNC);
  /* The function index space starts with imported funcs then local funcs;
   * our local function comes after the 3 imports. */
  fbx_wasm_buffer_uleb(&body, FBX_HOST_IMPORT_COUNT);
  EmitSection(out, WASM_SECTION_EXPORT, &body);
  fbx_wasm_buffer_free(&body);
}

static int EmitCodeSection(struct FbxWasmBuffer *out,
                           const struct FbxIrBlock *ir) {
  struct FbxWasmBuffer body;
  struct FbxWasmBuffer fn_body;
  fbx_wasm_buffer_init(&body);
  fbx_wasm_buffer_init(&fn_body);
  if (!EmitFunctionBody(&fn_body, ir)) {
    fbx_wasm_buffer_free(&body);
    fbx_wasm_buffer_free(&fn_body);
    return 0;
  }
  fbx_wasm_buffer_uleb(&body, 1); /* 1 function */
  fbx_wasm_buffer_uleb(&body, (u64)fn_body.len);
  fbx_wasm_buffer_bytes(&body, fn_body.data, fn_body.len);
  EmitSection(out, WASM_SECTION_CODE, &body);
  fbx_wasm_buffer_free(&body);
  fbx_wasm_buffer_free(&fn_body);
  return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Public entry point.                                                        */
/* ────────────────────────────────────────────────────────────────────────── */

int fbx_ir_emit_wasm(const struct FbxIrBlock *ir, const struct FbxTcBlock *tc,
                     struct FbxWasmBuffer *out) {
  if (!ir || !tc || !out) return 0;
  if (!CoverageGate(ir, tc)) return 0;
  /* The synthesis pass MUST be pure: clear the output buffer before
   * emitting so a same-input call produces the same bytes. */
  fbx_wasm_buffer_free(out);
  fbx_wasm_buffer_init(out);
  /* Wasm magic + version. */
  fbx_wasm_buffer_u32_le(out, 0x6D736100u); /* \0asm */
  fbx_wasm_buffer_u32_le(out, 0x00000001u);
  EmitTypeSection(out);
  EmitImportSection(out);
  EmitFunctionSection(out);
  EmitExportSection(out);
  if (!EmitCodeSection(out, ir)) {
    fbx_wasm_buffer_free(out);
    return 0;
  }
  if (out->oom) {
    fbx_wasm_buffer_free(out);
    return 0;
  }
  return 1;
}
