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

/* §13.5 follow-on (#599) — lazy-flag synthesis + BRANCH_COND.  Wasm MVP
 * opcodes only; portable across V8 Liftoff + wasmer Cranelift + JSC + SpiderMonkey.
 *
 * The flag-update path uses i32-typed flag-bit values (cf/zf/sf/of/af shifted
 * into position and OR'd into m->flags) and i64-typed operand-width arithmetic
 * to compute the bits.  i32.wrap_i64 is the bridge.
 *
 * The BRANCH_COND path reads m->flags as i32, masks the relevant bit, then
 * uses `select` to choose between taken_pc and fallthrough_pc as i64. */
#define WASM_OP_I32_LOAD    0x28u
#define WASM_OP_I32_STORE   0x36u
#define WASM_OP_I32_EQZ     0x45u
#define WASM_OP_I32_EQ      0x46u
#define WASM_OP_I32_NE      0x47u
#define WASM_OP_I32_LT_U    0x49u
#define WASM_OP_I32_AND     0x71u
#define WASM_OP_I32_OR      0x72u
#define WASM_OP_I32_XOR     0x73u
#define WASM_OP_I32_SHL     0x74u
#define WASM_OP_I32_SHR_U   0x76u
#define WASM_OP_I32_WRAP_I64 0xA7u
#define WASM_OP_I64_EQZ     0x50u
#define WASM_OP_I64_EQ      0x51u
#define WASM_OP_I64_NE      0x52u
#define WASM_OP_I64_LT_U    0x54u
#define WASM_OP_I64_LT_S    0x53u
#define WASM_OP_I64_SHL     0x86u
#define WASM_OP_I64_SHR_U   0x88u
#define WASM_OP_I64_SHR_S   0x87u
#define WASM_OP_I64_EXTEND_I32_U 0xADu
#define WASM_OP_SELECT      0x1Bu

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
  /* #599 (FBX_IR_VERSION 2): flag-reader-bearing blocks are now emit-eligible.
   * Two sub-gates apply:
   *
   *   - GET_FLAG: deferred to a follow-on (no consumer in v0.1; the lifter
   *     doesn't emit it as of #596).  Refuse for now.
   *   - BRANCH_COND: synthesizable for 14 of 16 Jcc predicates (Jcc PF/NP
   *     deferred — see EmitJccPredicate).  We can't tell predicate ID
   *     without scanning instructions; accept-then-refuse-at-emit is the
   *     pattern (mirror the LEA-SIB acceptance gate). */
  for (i = 0; i < ir->ninsts; ++i) {
    u8 op = ir->insts[i].opcode;
    if (op == FBX_IR_OP_GET_FLAG) {
      return 0; /* deferred — no consumer at v0.1 */
    }
  }
  /* SET_FLAGS_RAW: when a flag-reader follows, the eager-update path is
   * triggered.  This requires the v2 IR shape (src1/src2 = operand vregs).
   * Validate that every SET_FLAGS_RAW carries the expected encoding. */
  if (saw_flag_reader) {
    for (i = 0; i < ir->ninsts; ++i) {
      const struct FbxIrInst *p = &ir->insts[i];
      if (p->opcode == FBX_IR_OP_SET_FLAGS_RAW) {
        if (p->src1_kind != FBX_IR_KIND_VREG ||
            p->src2_kind != FBX_IR_KIND_VREG) {
          /* Stale v1 lift output without operand encoding — refuse. */
          return 0;
        }
        /* Width must be 1/2/4/8 (we use it as i64 shift amount). */
        if (p->width != 1 && p->width != 2 &&
            p->width != 4 && p->width != 8) {
          return 0;
        }
        /* op_kind in imm must be one of the supported flag-emitters. */
        switch ((u8)p->imm) {
          case FBX_IR_OP_ADD:
          case FBX_IR_OP_SUB:
          case FBX_IR_OP_AND:
          case FBX_IR_OP_OR:
          case FBX_IR_OP_XOR:
          case FBX_IR_OP_CMP:
          case FBX_IR_OP_TEST:
            break;
          default:
            return 0;
        }
      }
    }
  }
  (void)saw_flag_writer; /* presence-only; no further gating */
  /* Pass 2 — per-op acceptance. */
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
      case FBX_IR_OP_SET_FLAGS_RAW:
      case FBX_IR_OP_CMP:
      case FBX_IR_OP_TEST:
      case FBX_IR_OP_BRANCH_COND:  /* #599: synthesisable for Jcc 0-9, C-F */
      case FBX_IR_OP_CALL_DIRECT:  /* #602: stack-mutating control flow */
      case FBX_IR_OP_RET:          /* #602 */
      case FBX_IR_OP_PUSH:         /* #602 */
      case FBX_IR_OP_POP:          /* #602 */
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
      /* #602 — direct CALL rel32 (0x0E8) + RET (0x0C3).  Neither has a
       * modrm; no mod3 check applies. */
      case 0x0E8:
      case 0x0C3:
        break;
      /* #602 — PUSH reg (0x050-0x057) + POP reg (0x058-0x05F).  No modrm;
       * register encoded in low 3 bits of mopcode (extended via REX.B). */
      case 0x050: case 0x051: case 0x052: case 0x053:
      case 0x054: case 0x055: case 0x056: case 0x057:
      case 0x058: case 0x059: case 0x05A: case 0x05B:
      case 0x05C: case 0x05D: case 0x05E: case 0x05F:
        break;
      /* #599 — Conditional jumps.  The synthesis pass emits the lazy-flag
       * + select-based branch decision.  Predicate refusal (Jcc PF/NP)
       * happens at EmitJccPredicate-emit-time. */
      case 0x070: case 0x071: case 0x072: case 0x073:
      case 0x074: case 0x075: case 0x076: case 0x077:
      case 0x078: case 0x079: case 0x07A: case 0x07B:
      case 0x07C: case 0x07D: case 0x07E: case 0x07F:
      case 0x180: case 0x181: case 0x182: case 0x183:
      case 0x184: case 0x185: case 0x186: case 0x187:
      case 0x188: case 0x189: case 0x18A: case 0x18B:
      case 0x18C: case 0x18D: case 0x18E: case 0x18F:
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
/* #599 — Machine.flags is u32 (eflags register).  Position is compiler-
 * dependent but offsetof keeps the synthesis pass cross-compiler stable. */
#define M_OFF_FLAGS ((u32)offsetof(struct Machine, flags))

/* #602 — RSP register id in Machine.weg[].  Per the Machine struct union
 * (blink/machine.h:385-430), the ordered register file is ax(0), cx(1),
 * dx(2), bx(3), sp(4), bp(5), si(6), di(7), r8(8)..r15(15).  So RSP=4,
 * RBP=5.  Pinned constants: cross-build determinism (spec §Q5 axis 2). */
#define FBX_GREG_RSP 4u
#define FBX_GREG_RBP 5u
#define M_OFF_RSP   (M_OFF_WEG + FBX_GREG_RSP * 8u)

/* x86 EFLAGS bit positions — duplicated from blink/flags.h to keep this
 * file standalone (no further include dependency).  Phase 4's offline
 * emitter uses the same constants; cross-build determinism requires they
 * stay in sync with blink/flags.h.  See spec §Q5 axis 2. */
#define EMIT_FLAGS_CF 0u
#define EMIT_FLAGS_PF 2u
#define EMIT_FLAGS_AF 4u
#define EMIT_FLAGS_ZF 6u
#define EMIT_FLAGS_SF 7u
#define EMIT_FLAGS_OF 11u

/* Pre-baked AND-mask that clears CF|ZF|SF|OF|AF|0xFF000000 in one i32.const.
 * Matches blink/alu.c:AluFlags exactly:
 *   m->flags &= ~(CF | ZF | SF | OF | AF | 0xFF000000u);
 * Computed:
 *   ~((1<<0)|(1<<6)|(1<<7)|(1<<11)|(1<<4)|0xFF000000)
 *   = ~(0x000008D1u | 0xFF000000u)
 *   = ~0xFF0008D1u
 *   = 0x00FFF72Eu
 */
#define EMIT_FLAGS_CLEAR_MASK 0x00FFF72Eu

/* Scratch local indices (after m_ptr local 0 + nvregs i64 locals).
 *
 * The synthesis pass reserves 2 scratch locals at the end of the locals
 * array when the block needs them (any flag-writing ALU op present).  The
 * indices below are RELATIVE to `nvregs + 1`; the EmitFunctionBody adds
 * the offset at call sites.
 *
 * SCRATCH_Z holds the width-truncated i64 ALU result during a flag-update
 * sequence.  SCRATCH_FLAGS holds the i32 new-flags value being built up
 * before the final i32.store back into m->flags.
 *
 * These are PER-block scratch — re-used across multiple ALU ops in the
 * same lifted block.  Each ALU op's flag-update sequence writes them
 * before reading.  Wasm doesn't require local init beyond the default-0
 * fill the validator gives us.
 */
#define EMIT_LOCAL_OFF_SCRATCH_Z      0u  /* i64 */
#define EMIT_LOCAL_OFF_SCRATCH_FLAGS  1u  /* i32 */
#define EMIT_NUM_SCRATCH_LOCALS       2u

/* Width → sign-bit position (width*8 - 1). */
static u32 SignBitPos(u8 width) {
  switch (width) {
    case 1: return 7u;
    case 2: return 15u;
    case 4: return 31u;
    case 8:
    default: return 63u;
  }
}

/* Width → low-bits mask as i64.  For widths < 8, the wasm i64 result of
 * arithmetic naturally extends; we mask explicitly to keep the carry/zf
 * derivation matching x86 width semantics. */
static u64 WidthMask(u8 width) {
  switch (width) {
    case 1: return 0xFFull;
    case 2: return 0xFFFFull;
    case 4: return 0xFFFFFFFFull;
    case 8:
    default: return 0xFFFFFFFFFFFFFFFFull;
  }
}

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

/* ────────────────────────────────────────────────────────────────────────── */
/* #599 — Lazy-flag wasm synthesis.                                           */
/*                                                                            */
/* Each flag-emitting ALU op (ADD/SUB/AND/OR/XOR/CMP/TEST) generates wasm     */
/* that matches blink/alu.c's AluFlags() byte-for-byte.  The eager-update    */
/* representation:                                                            */
/*                                                                            */
/*   m->flags &= ~(CF|ZF|SF|OF|AF|0xFF000000u);                              */
/*   m->flags |= sf<<7 | cf<<0 | (z==0)<<6 | of<<11 | af<<4 | (z&0xFF)<<24;  */
/*                                                                            */
/* Per-op derivations (matching the int-width-specific helpers in alu.c):    */
/*                                                                            */
/*   ADD: z = (lhs + rhs) & mask                                              */
/*        cf = z <u rhs                                                       */
/*        af = (z & 15) <u (rhs & 15)                                         */
/*        of = ((z^lhs) & (z^rhs)) >> sign_bit_pos                           */
/*        sf = z >> sign_bit_pos                                              */
/*                                                                            */
/*   SUB / CMP: z = (lhs - rhs) & mask                                        */
/*        cf = lhs <u z                                                       */
/*        af = (lhs & 15) <u (z & 15)                                         */
/*        of = ((lhs^rhs) & (z^lhs)) >> sign_bit_pos                         */
/*        sf = z >> sign_bit_pos                                              */
/*                                                                            */
/*   AND / OR / XOR / TEST:                                                   */
/*        z = (lhs OP rhs) & mask    (AND for TEST)                          */
/*        cf = 0                                                              */
/*        af = 0                                                              */
/*        of = 0                                                              */
/*        sf = z >> sign_bit_pos                                              */
/*                                                                            */
/* The emitted wasm assigns the computed result to SCRATCH_Z (i64) first,    */
/* then builds the new-flags i32 value into SCRATCH_FLAGS, then writes it    */
/* back to m->flags.  ZF and PF derive from the final z value.                */
/*                                                                            */
/* Width handling: i64 arithmetic naturally width-extends; the explicit     */
/* mask after each op restores x86 semantics (lower 8/16/32/64 bits used).  */
/* The sign-bit extraction uses width*8-1 as the shift amount, so the same   */
/* helper handles all four widths.                                            */
/* ────────────────────────────────────────────────────────────────────────── */

/* Get the local index of the i64 scratch_z, given nvregs. */
static u32 ScratchZLocal(u16 nvregs) {
  return 1u + (u32)nvregs + EMIT_LOCAL_OFF_SCRATCH_Z;
}

/* Get the local index of the i32 scratch_flags, given nvregs. */
static u32 ScratchFlagsLocal(u16 nvregs) {
  return 1u + (u32)nvregs + EMIT_LOCAL_OFF_SCRATCH_FLAGS;
}

/* Emit code that pushes (vreg_local & width_mask) on the stack as i64.
 * width is 1/2/4/8; for width 8 no mask is needed but we emit one anyway
 * for uniformity (i64.const 0xFFFFFFFFFFFFFFFFu + i64.and is a no-op). */
static void EmitMaskedVreg(struct FbxWasmBuffer *body, u32 vreg_local,
                           u8 width) {
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, vreg_local);
  if (width != 8) {
    fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
    fbx_wasm_buffer_sleb(body, (i64)WidthMask(width));
    fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);
  }
}

/* Emit the AluFlags update wasm.  The caller has ensured the IR op_kind is
 * one of ADD/SUB/AND/OR/XOR/CMP/TEST.  `lhs_local` and `rhs_local` are the
 * vreg local indices carrying the i64 operands (already populated by prior
 * REG_GET emits).
 *
 * Postconditions: m->flags written, SCRATCH_Z + SCRATCH_FLAGS locals
 * clobbered.  No values left on the wasm stack.
 *
 * The op_kind selects the ALU op + cf/af/of formulae per the comment block
 * above.  CMP behaves identically to SUB for flag purposes (same formula);
 * TEST behaves identically to AND (cf=0/af=0/of=0; only sf/zf vary).
 */
static void EmitFlagsAfterAlu(struct FbxWasmBuffer *body, u16 nvregs,
                              u8 op_kind, u32 lhs_local, u32 rhs_local,
                              u8 width) {
  u32 scratch_z = ScratchZLocal(nvregs);
  u32 scratch_flags = ScratchFlagsLocal(nvregs);
  u32 sign_pos = SignBitPos(width);
  int is_sub_like = (op_kind == FBX_IR_OP_SUB || op_kind == FBX_IR_OP_CMP);
  int is_add = (op_kind == FBX_IR_OP_ADD);
  int is_logical = (op_kind == FBX_IR_OP_AND || op_kind == FBX_IR_OP_OR ||
                    op_kind == FBX_IR_OP_XOR || op_kind == FBX_IR_OP_TEST);

  /* Step 1: compute z = (lhs OP rhs) & width_mask, save in scratch_z. */
  EmitMaskedVreg(body, lhs_local, width);
  EmitMaskedVreg(body, rhs_local, width);
  switch (op_kind) {
    case FBX_IR_OP_ADD:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_ADD);
      break;
    case FBX_IR_OP_SUB:
    case FBX_IR_OP_CMP:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_SUB);
      break;
    case FBX_IR_OP_AND:
    case FBX_IR_OP_TEST:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);
      break;
    case FBX_IR_OP_OR:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_OR);
      break;
    case FBX_IR_OP_XOR:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_XOR);
      break;
    default:
      /* Caller violated precondition.  Emit a no-op (i64.drop equivalent
       * would leave stack dirty; emit a constant + drop instead).  In
       * practice the caller's switch must match this one's cases — keep
       * here as defensive zero. */
      fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);
      break;
  }
  if (width != 8) {
    fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
    fbx_wasm_buffer_sleb(body, (i64)WidthMask(width));
    fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);
  }
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
  fbx_wasm_buffer_uleb(body, scratch_z);

  /* Step 2: build the new flags value into scratch_flags.
   *   start with (old_flags & EMIT_FLAGS_CLEAR_MASK). */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, 0); /* m_ptr */
  fbx_wasm_buffer_u8(body, WASM_OP_I32_LOAD);
  fbx_wasm_buffer_uleb(body, 2); /* align log2 for i32 */
  fbx_wasm_buffer_uleb(body, M_OFF_FLAGS);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
  fbx_wasm_buffer_sleb(body, (i64)(i32)EMIT_FLAGS_CLEAR_MASK);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_AND);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
  fbx_wasm_buffer_uleb(body, scratch_flags);

  /* Step 3: ZF — (z == 0) << FLAGS_ZF.  i64.eqz returns i32 (0 or 1). */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, scratch_flags);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, scratch_z);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_EQZ);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
  fbx_wasm_buffer_sleb(body, (i64)EMIT_FLAGS_ZF);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_SHL);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_OR);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
  fbx_wasm_buffer_uleb(body, scratch_flags);

  /* Step 4: SF — (z >> sign_pos) << FLAGS_SF.  i64.shr_u → i32.wrap_i64. */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, scratch_flags);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, scratch_z);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, (i64)sign_pos);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_SHR_U);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_WRAP_I64);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
  fbx_wasm_buffer_sleb(body, (i64)EMIT_FLAGS_SF);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_SHL);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_OR);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
  fbx_wasm_buffer_uleb(body, scratch_flags);

  /* Step 5: CF.
   *   ADD-like: cf = z <u rhs_masked
   *   SUB-like: cf = lhs_masked <u z
   *   logical:  cf = 0  (skip emit)
   */
  if (is_add || is_sub_like) {
    fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
    fbx_wasm_buffer_uleb(body, scratch_flags);
    if (is_add) {
      fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
      fbx_wasm_buffer_uleb(body, scratch_z);
      EmitMaskedVreg(body, rhs_local, width);
    } else {
      EmitMaskedVreg(body, lhs_local, width);
      fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
      fbx_wasm_buffer_uleb(body, scratch_z);
    }
    fbx_wasm_buffer_u8(body, WASM_OP_I64_LT_U);
    /* CF is bit 0, no shift needed. */
    fbx_wasm_buffer_u8(body, WASM_OP_I32_OR);
    fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
    fbx_wasm_buffer_uleb(body, scratch_flags);
  }

  /* Step 6: OF.
   *   ADD: of = ((z^lhs) & (z^rhs)) >> sign_pos
   *   SUB: of = ((lhs^rhs) & (z^lhs)) >> sign_pos
   *   logical: of = 0 (skip)
   */
  if (is_add || is_sub_like) {
    fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
    fbx_wasm_buffer_uleb(body, scratch_flags);
    if (is_add) {
      /* (z ^ lhs_masked) */
      fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
      fbx_wasm_buffer_uleb(body, scratch_z);
      EmitMaskedVreg(body, lhs_local, width);
      fbx_wasm_buffer_u8(body, WASM_OP_I64_XOR);
      /* (z ^ rhs_masked) */
      fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
      fbx_wasm_buffer_uleb(body, scratch_z);
      EmitMaskedVreg(body, rhs_local, width);
      fbx_wasm_buffer_u8(body, WASM_OP_I64_XOR);
    } else {
      /* (lhs_masked ^ rhs_masked) */
      EmitMaskedVreg(body, lhs_local, width);
      EmitMaskedVreg(body, rhs_local, width);
      fbx_wasm_buffer_u8(body, WASM_OP_I64_XOR);
      /* (z ^ lhs_masked) */
      fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
      fbx_wasm_buffer_uleb(body, scratch_z);
      EmitMaskedVreg(body, lhs_local, width);
      fbx_wasm_buffer_u8(body, WASM_OP_I64_XOR);
    }
    fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);
    fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
    fbx_wasm_buffer_sleb(body, (i64)sign_pos);
    fbx_wasm_buffer_u8(body, WASM_OP_I64_SHR_U);
    fbx_wasm_buffer_u8(body, WASM_OP_I32_WRAP_I64);
    fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
    fbx_wasm_buffer_sleb(body, (i64)EMIT_FLAGS_OF);
    fbx_wasm_buffer_u8(body, WASM_OP_I32_SHL);
    fbx_wasm_buffer_u8(body, WASM_OP_I32_OR);
    fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
    fbx_wasm_buffer_uleb(body, scratch_flags);
  }

  /* Step 7: AF.
   *   ADD: af = (z & 15) <u (rhs_masked & 15)
   *   SUB: af = (lhs_masked & 15) <u (z & 15)
   *   logical: af = 0 (skip)
   */
  if (is_add || is_sub_like) {
    fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
    fbx_wasm_buffer_uleb(body, scratch_flags);
    if (is_add) {
      fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
      fbx_wasm_buffer_uleb(body, scratch_z);
      fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
      fbx_wasm_buffer_sleb(body, 15);
      fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);
      EmitMaskedVreg(body, rhs_local, width);
      fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
      fbx_wasm_buffer_sleb(body, 15);
      fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);
    } else {
      EmitMaskedVreg(body, lhs_local, width);
      fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
      fbx_wasm_buffer_sleb(body, 15);
      fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);
      fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
      fbx_wasm_buffer_uleb(body, scratch_z);
      fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
      fbx_wasm_buffer_sleb(body, 15);
      fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);
    }
    fbx_wasm_buffer_u8(body, WASM_OP_I64_LT_U);
    fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
    fbx_wasm_buffer_sleb(body, (i64)EMIT_FLAGS_AF);
    fbx_wasm_buffer_u8(body, WASM_OP_I32_SHL);
    fbx_wasm_buffer_u8(body, WASM_OP_I32_OR);
    fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
    fbx_wasm_buffer_uleb(body, scratch_flags);
  }
  (void)is_logical; /* logical ops skip CF/OF/AF emit; just SF + ZF + lazy-PF */

  /* Step 8: lazy-PF byte: (z & 0xFF) << 24. */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, scratch_flags);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, scratch_z);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, 0xFF);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_WRAP_I64);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
  fbx_wasm_buffer_sleb(body, 24);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_SHL);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_OR);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
  fbx_wasm_buffer_uleb(body, scratch_flags);

  /* Step 9: store scratch_flags → m->flags. */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, 0); /* m_ptr */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, scratch_flags);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_STORE);
  fbx_wasm_buffer_uleb(body, 2); /* align log2 for i32 */
  fbx_wasm_buffer_uleb(body, M_OFF_FLAGS);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* #599 — BRANCH_COND wasm synthesis.                                         */
/*                                                                            */
/* Reads m->flags as i32, evaluates the per-condition predicate, then         */
/* uses wasm `select` to choose between taken_pc and fallthrough_pc.  The     */
/* chosen i64 is stored into m->ip.  Block terminator (returns 0).            */
/*                                                                            */
/* Condition encoding from x86 Jcc opcode low-nibble (Intel SDM Vol 1 §B.1): */
/*   0 = O   (OF=1)             8 = S   (SF=1)                               */
/*   1 = NO  (OF=0)             9 = NS  (SF=0)                               */
/*   2 = B   (CF=1)             A = P   (PF=1)                               */
/*   3 = NB  (CF=0)             B = NP  (PF=0)                               */
/*   4 = Z   (ZF=1)             C = L   (SF!=OF)                             */
/*   5 = NZ  (ZF=0)             D = NL  (SF==OF)                             */
/*   6 = BE  (CF=1 or ZF=1)     E = LE  (ZF=1 or SF!=OF)                     */
/*   7 = A   (CF=0 and ZF=0)    F = G   (ZF=0 and SF==OF)                    */
/*                                                                            */
/* Each predicate evaluates to an i32 (0 or non-zero); `select` consumes 3   */
/* values [v1, v2, cond] and pushes v1 if cond≠0 else v2.  We pre-place the  */
/* (taken_pc, fallthrough_pc) operands then push the predicate.              */
/* ────────────────────────────────────────────────────────────────────────── */

/* Emit code that loads m->flags as i32 onto the stack. */
static void EmitLoadFlags(struct FbxWasmBuffer *body) {
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, 0); /* m_ptr */
  fbx_wasm_buffer_u8(body, WASM_OP_I32_LOAD);
  fbx_wasm_buffer_uleb(body, 2); /* align log2 */
  fbx_wasm_buffer_uleb(body, M_OFF_FLAGS);
}

/* Emit code that pushes (flags >> bit_pos) & 1 as i32 onto the stack. */
static void EmitFlagBit(struct FbxWasmBuffer *body, u32 bit_pos) {
  EmitLoadFlags(body);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
  fbx_wasm_buffer_sleb(body, (i64)bit_pos);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_SHR_U);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
  fbx_wasm_buffer_sleb(body, 1);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_AND);
}

/* Emit code that pushes the i32 (0 or 1) value of the Jcc predicate
 * for `cond_id` (0-15).  See the table above. */
static int EmitJccPredicate(struct FbxWasmBuffer *body, u32 cond_id) {
  /* cond_id bit 0 == 1 means "invert" — odd-numbered conditions are
   * the NOT of the prior even one.  Compute the base predicate, then
   * XOR with 1 if odd. */
  u32 base = cond_id & 0xFEu;
  switch (base) {
    case 0x0: /* O / NO  — OF */
      EmitFlagBit(body, EMIT_FLAGS_OF);
      break;
    case 0x2: /* B / NB  — CF */
      EmitFlagBit(body, EMIT_FLAGS_CF);
      break;
    case 0x4: /* Z / NZ  — ZF */
      EmitFlagBit(body, EMIT_FLAGS_ZF);
      break;
    case 0x6: /* BE / A  — CF or ZF */
      EmitFlagBit(body, EMIT_FLAGS_CF);
      EmitFlagBit(body, EMIT_FLAGS_ZF);
      fbx_wasm_buffer_u8(body, WASM_OP_I32_OR);
      break;
    case 0x8: /* S / NS  — SF */
      EmitFlagBit(body, EMIT_FLAGS_SF);
      break;
    case 0xA: /* P / NP  — PF (lazy parity from low byte) */
      /* PF semantics in Blink: GetParity((flags >> 24) & 0xFF).  Lazy.
       * The wasm-side computation is non-trivial (parity-of-byte).
       * v0.1 of #599 refuses Jcc PF/NP — extremely rare in compiler-
       * emitted code; defer to a follow-on.  Return 0 to signal refusal. */
      return 0;
    case 0xC: /* L / NL  — SF != OF */
      EmitFlagBit(body, EMIT_FLAGS_SF);
      EmitFlagBit(body, EMIT_FLAGS_OF);
      fbx_wasm_buffer_u8(body, WASM_OP_I32_XOR);
      break;
    case 0xE: /* LE / G  — ZF or (SF != OF) */
      EmitFlagBit(body, EMIT_FLAGS_ZF);
      EmitFlagBit(body, EMIT_FLAGS_SF);
      EmitFlagBit(body, EMIT_FLAGS_OF);
      fbx_wasm_buffer_u8(body, WASM_OP_I32_XOR);
      fbx_wasm_buffer_u8(body, WASM_OP_I32_OR);
      break;
    default:
      return 0;
  }
  /* Invert for odd cond_id. */
  if (cond_id & 1u) {
    fbx_wasm_buffer_u8(body, WASM_OP_I32_EQZ);
  }
  return 1;
}

/* Lower a BRANCH_COND IR inst.  Returns 1 on success, 0 if the predicate
 * isn't synthesisable (Jcc PF/NP at v0.1). */
static int EmitBranchCond(struct FbxWasmBuffer *body,
                          const struct FbxIrInst *p) {
  u64 taken_pc = p->imm;
  /* fallthrough_pc is encoded as the low 32 bits of src2 (per the lifter
   * in LiftJcc).  Sign-extension is irrelevant — we treat it as unsigned
   * absolute PC. */
  u64 fallthrough_pc = (u64)(u32)p->src2;
  u32 cond_id = p->src1; /* low nibble of mopcode */
  /* Push m_ptr for the eventual i64.store at m->ip. */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, 0);
  /* Push taken_pc (operand 1 of select). */
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, (i64)taken_pc);
  /* Push fallthrough_pc (operand 2 of select). */
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, (i64)fallthrough_pc);
  /* Push predicate (cond, i32; non-zero ⇒ taken_pc). */
  if (!EmitJccPredicate(body, cond_id)) {
    return 0;
  }
  /* select pops [v1, v2, cond] and pushes v1 if cond, else v2.  Result
   * is i64 (the operand type). */
  fbx_wasm_buffer_u8(body, WASM_OP_SELECT);
  /* Store the chosen i64 to m->ip. */
  fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE);
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, M_OFF_IP);
  /* Block terminator: return exit=0 (normal control transfer).  Note
   * that Tier 1 will re-resolve the PC; this matches BRANCH_TAKEN. */
  fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
  fbx_wasm_buffer_sleb(body, 0);
  fbx_wasm_buffer_u8(body, WASM_OP_RETURN);
  return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* #602 — Guest-memory + stack-mutating control-flow emit.                    */
/*                                                                            */
/* CALL_DIRECT / RET / PUSH / POP all mutate the guest stack at [RSP-8]      */
/* (push-side) or [RSP] (pop/ret-side).  The v0.1 emitter (§13.5c) accesses */
/* the guest stack through the imported wasm linear memory under the         */
/* assumption that guest virtual addresses alias host linear-memory          */
/* offsets — the same invariant the existing `m_ptr + offsetof(Machine, X)` */
/* loads/stores already rely on (see §13.4 + spec §3.1 preamble; ToHost(va) */
/* = va + kSkew where kSkew == 0 in linear-mapping mode).  Out-of-bounds    */
/* guest-stack access traps the wasm runtime; the bridge catches and bails */
/* to Tier 1 (T2Bridge::dispatch returns 1 on Err — see                      */
/* crates/firebox-wasix/src/t2_bridge.rs `dispatch_impl` + `fbx_t2_dispatch` */
/* shim).                                                                    */
/*                                                                            */
/* Stack-pointer width: RSP is read as i64, narrowed to i32 via              */
/* i32.wrap_i64 for use as a wasm linear-memory address.  Wasm MVP uses     */
/* i32 addresses (4 GiB max); the upper 32 bits of RSP must be zero for     */
/* the access to land in the linear-memory region.  Blink's loader pages    */
/* the stack into low VAs in linear-mapping mode (see blink/loader.c:806-   */
/* 814 — `stack` = ReserveVirtual result; under kSkew==0 + linear mapping  */
/* the stack lives in low 32 bits of guest VA space).                       */
/*                                                                            */
/* Width handling: PUSH/POP/CALL/RET only synthesize the 8-byte form in     */
/* v0.1 (matches x86-64 default operand size + §Q7 top-30 ranking which    */
/* counts 0x055 PUSH RBP at rank 16 and 0x05D POP RBP at rank 17 — both    */
/* 8-byte default).                                                          */
/* ────────────────────────────────────────────────────────────────────────── */

/* Emit code that stores the i64 currently on top of the wasm stack to the
 * guest memory location (m->weg[base_reg] + disp).  Leaves the stack
 * empty.  Width is the store width (1/2/4/8); only 8 used in v0.1.
 *
 * The emitted sequence:
 *   <produce i64 value on the wasm stack> (caller's responsibility)
 *   local.get 0                     ;; m_ptr (i32)
 *   i64.load offset=M_OFF_WEG+...   ;; guest base reg as i64
 *   i64.const disp
 *   i64.add
 *   i32.wrap_i64                    ;; convert to wasm i32 address
 *   <stash value via local>
 *   ...
 *
 * Wasm's store opcodes take [addr, value] from the stack.  Since the
 * caller already pushed the value FIRST, we need to interleave: we
 * pre-compute the address before the caller's value emit, OR we use a
 * different ordering helper.  This emitter inverts the call: the helper
 * itself emits the address+value sequence given a `src_local` that
 * carries the i64 value to store.  Callers stash the value in a vreg
 * local first (CALL/PUSH stash a constant or greg value into scratch_z;
 * RET/POP read the value back into a local before the helper runs). */
static void EmitGuestMemStoreFromLocal(struct FbxWasmBuffer *body,
                                       u32 base_reg, i64 disp,
                                       u32 src_local, u8 width) {
  u32 base_off = M_OFF_WEG + base_reg * 8u;
  /* Compute the wasm i32 address = i32.wrap_i64(m->weg[base_reg] + disp). */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, 0);              /* m_ptr (i32 base) */
  fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD); /* read base reg as i64 */
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, base_off);
  if (disp != 0) {
    fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
    fbx_wasm_buffer_sleb(body, disp);
    fbx_wasm_buffer_u8(body, WASM_OP_I64_ADD);
  }
  fbx_wasm_buffer_u8(body, WASM_OP_I32_WRAP_I64); /* → i32 address */
  /* Push value to store. */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, src_local);
  /* Width-tagged store; effective addr = i32_addr + offset=0. */
  switch (width) {
    case 1:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE8);
      fbx_wasm_buffer_uleb(body, 0);
      fbx_wasm_buffer_uleb(body, 0);
      break;
    case 2:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE16);
      fbx_wasm_buffer_uleb(body, 1);
      fbx_wasm_buffer_uleb(body, 0);
      break;
    case 4:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE32);
      fbx_wasm_buffer_uleb(body, 2);
      fbx_wasm_buffer_uleb(body, 0);
      break;
    case 8:
    default:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE);
      fbx_wasm_buffer_uleb(body, 3);
      fbx_wasm_buffer_uleb(body, 0);
      break;
  }
}

/* Emit code that stores the immediate `imm` (i64) to guest memory at
 * (m->weg[base_reg] + disp).  Helper for CALL_DIRECT (which stores
 * return_pc — a constant determined at lift time). */
static void EmitGuestMemStoreImm(struct FbxWasmBuffer *body, u32 base_reg,
                                 i64 disp, u64 imm, u8 width) {
  u32 base_off = M_OFF_WEG + base_reg * 8u;
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, 0);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD);
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, base_off);
  if (disp != 0) {
    fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
    fbx_wasm_buffer_sleb(body, disp);
    fbx_wasm_buffer_u8(body, WASM_OP_I64_ADD);
  }
  fbx_wasm_buffer_u8(body, WASM_OP_I32_WRAP_I64);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, (i64)imm);
  switch (width) {
    case 1:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE8);
      fbx_wasm_buffer_uleb(body, 0);
      fbx_wasm_buffer_uleb(body, 0);
      break;
    case 2:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE16);
      fbx_wasm_buffer_uleb(body, 1);
      fbx_wasm_buffer_uleb(body, 0);
      break;
    case 4:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE32);
      fbx_wasm_buffer_uleb(body, 2);
      fbx_wasm_buffer_uleb(body, 0);
      break;
    case 8:
    default:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE);
      fbx_wasm_buffer_uleb(body, 3);
      fbx_wasm_buffer_uleb(body, 0);
      break;
  }
}

/* Emit code that loads `width` bytes from guest memory at
 * (m->weg[base_reg] + disp) and leaves the value as i64 on the wasm
 * stack.  Width-tagged (zero-extending for narrow widths). */
static void EmitGuestMemLoad(struct FbxWasmBuffer *body, u32 base_reg,
                             i64 disp, u8 width) {
  u32 base_off = M_OFF_WEG + base_reg * 8u;
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, 0);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD);
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, base_off);
  if (disp != 0) {
    fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
    fbx_wasm_buffer_sleb(body, disp);
    fbx_wasm_buffer_u8(body, WASM_OP_I64_ADD);
  }
  fbx_wasm_buffer_u8(body, WASM_OP_I32_WRAP_I64);
  switch (width) {
    case 1:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD8U);
      fbx_wasm_buffer_uleb(body, 0);
      fbx_wasm_buffer_uleb(body, 0);
      break;
    case 2:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD16U);
      fbx_wasm_buffer_uleb(body, 1);
      fbx_wasm_buffer_uleb(body, 0);
      break;
    case 4:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD32U);
      fbx_wasm_buffer_uleb(body, 2);
      fbx_wasm_buffer_uleb(body, 0);
      break;
    case 8:
    default:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD);
      fbx_wasm_buffer_uleb(body, 3);
      fbx_wasm_buffer_uleb(body, 0);
      break;
  }
}

/* Emit `m->weg[base_reg] += delta` as an in-place i64 add.  Wraps natively
 * (x86-64 RSP is u64; modular arithmetic matches). */
static void EmitGregAddImm(struct FbxWasmBuffer *body, u32 base_reg,
                           i64 delta) {
  u32 base_off = M_OFF_WEG + base_reg * 8u;
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, 0); /* m_ptr (i32 base for the store) */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, 0); /* m_ptr (i32 base for the load) */
  fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD);
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, base_off);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, delta);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_ADD);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE);
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, base_off);
}

/* Lower a CALL_DIRECT IR inst.
 *
 *   m->weg[RSP] -= 8;
 *   *m->weg[RSP] = return_pc;
 *   m->ip = target_pc;
 *   return 0;
 *
 * Block-terminating.  Returns 1 on success. */
static int EmitCallDirect(struct FbxWasmBuffer *body,
                          const struct FbxIrInst *p) {
  u64 target_pc = p->imm;
  /* fallthrough/return PC is encoded as low 32 bits of src2 (matches
   * LiftCallJvds in fbx_ir_lift.c:706-715).  Treat as unsigned absolute
   * PC; sign-extension irrelevant since bench-corpus PCs are < 4 GiB. */
  u64 return_pc = (u64)(u32)p->src2;
  /* Step 1: RSP -= 8. */
  EmitGregAddImm(body, FBX_GREG_RSP, -8);
  /* Step 2: mem[RSP] = return_pc. */
  EmitGuestMemStoreImm(body, FBX_GREG_RSP, 0, return_pc, 8);
  /* Step 3: m->ip = target_pc and return exit=0. */
  EmitStoreIp(body, target_pc);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
  fbx_wasm_buffer_sleb(body, 0);
  fbx_wasm_buffer_u8(body, WASM_OP_RETURN);
  return 1;
}

/* Lower a RET IR inst.
 *
 *   m->ip = *m->weg[RSP];
 *   m->weg[RSP] += 8;
 *   return 0;
 *
 * Block-terminating.  Uses scratch_z (i64) to stash the popped PC across
 * the wasm-stack-emptying boundary required by the i64.store to m->ip. */
static int EmitRet(struct FbxWasmBuffer *body, u16 nvregs) {
  u32 scratch_z = ScratchZLocal(nvregs);
  /* Step 1: load *RSP into scratch_z. */
  EmitGuestMemLoad(body, FBX_GREG_RSP, 0, 8);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
  fbx_wasm_buffer_uleb(body, scratch_z);
  /* Step 2: m->ip = scratch_z. */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, 0); /* m_ptr */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, scratch_z);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE);
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, M_OFF_IP);
  /* Step 3: RSP += 8. */
  EmitGregAddImm(body, FBX_GREG_RSP, 8);
  /* Step 4: return exit=0. */
  fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
  fbx_wasm_buffer_sleb(body, 0);
  fbx_wasm_buffer_u8(body, WASM_OP_RETURN);
  return 1;
}

/* Lower a PUSH IR inst.
 *
 *   m->weg[RSP] -= width;
 *   *m->weg[RSP] = greg[reg_id];
 *
 * NOT block-terminating; the lifter emits a separate terminator (or
 * next opcode) after.  Uses scratch_z to stash the greg value.
 *
 * Width: v0.1 emits 8 only (matches lifter); the helper handles widths
 * 1/2/4/8 generically for forward-compatibility.  PUSH with 8-byte
 * width is the §Q7 top-30 ranking case. */
static int EmitPush(struct FbxWasmBuffer *body, const struct FbxIrInst *p,
                    u16 nvregs) {
  u32 scratch_z = ScratchZLocal(nvregs);
  u8 width = p->width ? p->width : 8u;
  if (p->src1_kind != FBX_IR_KIND_GREG) return 0;
  /* Step 1: stash greg[reg_id] into scratch_z. */
  EmitRegLoad(body, p->src1, width);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
  fbx_wasm_buffer_uleb(body, scratch_z);
  /* Step 2: RSP -= width. */
  EmitGregAddImm(body, FBX_GREG_RSP, -(i64)width);
  /* Step 3: *RSP = scratch_z. */
  EmitGuestMemStoreFromLocal(body, FBX_GREG_RSP, 0, scratch_z, width);
  return 1;
}

/* Lower a POP IR inst.
 *
 *   greg[reg_id] = *m->weg[RSP];
 *   m->weg[RSP] += width;
 *
 * NOT block-terminating.  Uses scratch_z to stash the popped value
 * across the load-then-RSP-bump-then-store sequence. */
static int EmitPop(struct FbxWasmBuffer *body, const struct FbxIrInst *p,
                   u16 nvregs) {
  u32 scratch_z = ScratchZLocal(nvregs);
  u8 width = p->width ? p->width : 8u;
  if (p->dst_kind != FBX_IR_KIND_GREG) return 0;
  /* Step 1: load *RSP into scratch_z. */
  EmitGuestMemLoad(body, FBX_GREG_RSP, 0, width);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
  fbx_wasm_buffer_uleb(body, scratch_z);
  /* Step 2: RSP += width. */
  EmitGregAddImm(body, FBX_GREG_RSP, (i64)width);
  /* Step 3: greg[dst] = scratch_z. */
  EmitRegStoreFromLocal(body, p->dst, scratch_z, width);
  return 1;
}

/* Decide whether this block needs the i64 + i32 scratch locals reserved
 * after the vreg block.  Any flag-reader (BRANCH_COND) is the trigger;
 * dead-flag-elision blocks (SET_FLAGS_RAW with no reader) still skip
 * scratch reservation because EmitFunctionBody drops SET_FLAGS_RAW in
 * that case (preserving the §13.5 fast-path).
 *
 * §599: blocks with a flag-reader present trigger the eager-update path;
 * every SET_FLAGS_RAW in such a block expands into the AluFlags emit
 * sequence and needs scratch_z + scratch_flags.
 */
static int BlockNeedsScratchLocals(const struct FbxIrBlock *ir) {
  u32 i;
  for (i = 0; i < ir->ninsts; ++i) {
    u8 op = ir->insts[i].opcode;
    if (op == FBX_IR_OP_BRANCH_COND || op == FBX_IR_OP_GET_FLAG) return 1;
    /* #602 — RET/PUSH/POP stash the popped or to-be-pushed value in
     * scratch_z to bridge the wasm-stack ordering between the load
     * and the subsequent RSP adjust + store-to-ip / store-to-greg.
     * CALL_DIRECT does NOT need scratch (it stores a constant return_pc
     * directly via EmitGuestMemStoreImm) but checking is cheap. */
    if (op == FBX_IR_OP_RET || op == FBX_IR_OP_PUSH ||
        op == FBX_IR_OP_POP) return 1;
  }
  return 0;
}

/* Emit the function body for one IR block.  Returns 1 on success, 0 on
 * unsupported op encountered mid-emission (caller frees scratch). */
static int EmitFunctionBody(struct FbxWasmBuffer *body,
                            const struct FbxIrBlock *ir) {
  u32 i;
  int terminated = 0;
  int needs_scratch = BlockNeedsScratchLocals(ir);
  /* Locals declaration.  When the block needs scratch locals (#599 path),
   * emit two groups: vregs as i64, then a pair (i64 scratch_z, i32
   * scratch_flags).  Wasm encodes per-group as (count, valtype); separate
   * groups are required because the types differ. */
  if (needs_scratch) {
    /* Group 1: nvregs i64s + 1 more i64 for scratch_z.  Combining them
     * into one group is byte-cheaper than two separate i64 groups. */
    fbx_wasm_buffer_uleb(body, 2); /* 2 groups */
    fbx_wasm_buffer_uleb(body, (u64)ir->nvregs + 1u);
    fbx_wasm_buffer_u8(body, WASM_VALTYPE_I64);
    /* Group 2: 1 i32 for scratch_flags. */
    fbx_wasm_buffer_uleb(body, 1);
    fbx_wasm_buffer_u8(body, WASM_VALTYPE_I32);
  } else if (ir->nvregs > 0) {
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
      case FBX_IR_OP_SET_FLAGS_RAW: {
        /* §13.5 (v1): dead-flag elision — when no flag-reader follows in
         * the block, the SET_FLAGS_RAW is a no-op and we drop it at emit
         * time.
         *
         * #599 (v2 / FBX_IR_VERSION 2): when a flag-reader is present in
         * the same block, the eager-update path emits the wasm that
         * matches blink/alu.c:AluFlags byte-for-byte.  The IR encoding
         * carries (op_kind in imm) + (lhs_vreg in src1) + (rhs_vreg in
         * src2) + (width in width) — see LiftAlui + LiftAluRR. */
        if (!needs_scratch) break; /* dead-flag elision path */
        if (p->src1_kind != FBX_IR_KIND_VREG ||
            p->src2_kind != FBX_IR_KIND_VREG) {
          return 0; /* malformed IR */
        }
        EmitFlagsAfterAlu(body, ir->nvregs, (u8)p->imm,
                          VregLocal(p->src1), VregLocal(p->src2),
                          p->width ? p->width : 8u);
        break;
      }
      case FBX_IR_OP_CMP:
      case FBX_IR_OP_TEST:
        /* CMP / TEST: no register write.  The flag update happens at the
         * subsequent SET_FLAGS_RAW.  No code emitted for the ALU op
         * itself — the flag computation re-derives the result from
         * lhs/rhs/op_kind in EmitFlagsAfterAlu. */
        break;
      case FBX_IR_OP_BRANCH_COND:
        if (!EmitBranchCond(body, p)) return 0;
        terminated = 1;
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
      case FBX_IR_OP_CALL_DIRECT:
        /* #602 — push return_pc onto guest stack; jump to target_pc. */
        if (!EmitCallDirect(body, p)) return 0;
        terminated = 1;
        break;
      case FBX_IR_OP_RET:
        /* #602 — pop guest stack into m->ip; bump RSP. */
        if (!EmitRet(body, ir->nvregs)) return 0;
        terminated = 1;
        break;
      case FBX_IR_OP_PUSH:
        /* #602 — stash greg, decrement RSP, store. */
        if (!EmitPush(body, p, ir->nvregs)) return 0;
        break;
      case FBX_IR_OP_POP:
        /* #602 — load from RSP, increment RSP, write greg. */
        if (!EmitPop(body, p, ir->nvregs)) return 0;
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
