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
#include "blink/fbx_t2_block_ctx.h"
#include "blink/machine.h"
#include "blink/rde.h"
#include "blink/threadedcode.h"
#include "blink/types.h"

/* ────────────────────────────────────────────────────────────────────────── */
/* #668 — Synth-failure plumbing.                                             */
/*                                                                            */
/* `FbxEmitFailCtx` is the threadable accumulator that the bailout sites       */
/* populate.  It is plumbed as a pointer through CoverageGate + EmitCodeSection*/
/* + EmitFunctionBody.  Purity-contract holds: no file-scope globals; the      */
/* pointer is a stack-allocated struct owned by the public entry point.        */
/*                                                                            */
/* SetFail() is sticky-first: only the FIRST rejection wins so the trace      */
/* line corresponds to the IR site that actually caused the bailout, not a    */
/* downstream cascade.  Pass NULL to skip — callers that don't care about the */
/* reason (the legacy fbx_ir_emit_wasm() wrapper) pay zero overhead.          */
/* ────────────────────────────────────────────────────────────────────────── */

struct FbxEmitFailCtx {
  enum FbxIrEmitFailReason reason;
  u8 opcode;
};

static void SetFail(struct FbxEmitFailCtx *f, enum FbxIrEmitFailReason r,
                    u8 opcode) {
  if (!f) return;
  if (f->reason != FBX_IR_EMIT_OK) return; /* sticky-first */
  f->reason = r;
  f->opcode = opcode;
}

const char *fbx_ir_fail_reason_name(enum FbxIrEmitFailReason r) {
  switch (r) {
    case FBX_IR_EMIT_OK: return "ok";
    case FBX_IR_EMIT_UNSUPPORTED_OPCODE: return "unsupported_opcode";
    case FBX_IR_EMIT_KIND_MISMATCH: return "kind_mismatch";
    case FBX_IR_EMIT_FLAG_READER_DEFERRED: return "flag_reader_deferred";
    case FBX_IR_EMIT_SET_FLAGS_RAW_BAD: return "set_flags_raw_bad";
    case FBX_IR_EMIT_LEA_SIB_FORM: return "lea_sib_form";
    case FBX_IR_EMIT_JCC_PREDICATE_DEFERRED: return "jcc_predicate_deferred";
    case FBX_IR_EMIT_TC_KIND_NON_NORMAL: return "tc_kind_non_normal";
    case FBX_IR_EMIT_MOD3_REQUIRED: return "mod3_required";
    case FBX_IR_EMIT_UNSUPPORTED_MOPCODE: return "unsupported_mopcode";
    case FBX_IR_EMIT_BUFFER_OOM: return "buffer_oom";
    case FBX_IR_EMIT_EMPTY_IR: return "empty_ir";
    case FBX_IR_EMIT_NONLINEAR_GUEST_MEM: return "nonlinear_guest_mem";
    case FBX_IR_EMIT_BAILOUT_AFTER_COMMIT: return "bailout_after_commit";
    case FBX_IR_EMIT_IMUL_FLAGS_LIVE: return "imul_flags_live";
  }
  return "unknown";
}

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
#define WASM_OP_IF          0x04u /* firebox#735 — MMU-miss bailout guard */
#define WASM_OP_CALL        0x10u
/* #695 — control-flow opcodes for the self-loop chain (§4 of the chaining
 * design).  `loop` + `br_if` keep a self-targeting hot block iterating
 * guest-side without crossing the FFI boundary every iteration.  All wasm
 * MVP (portable across V8 / wasmer-Cranelift / JSC / SpiderMonkey). */
#define WASM_OP_UNREACHABLE 0x00u
#define WASM_OP_LOOP        0x03u
#define WASM_OP_BR          0x0Cu
#define WASM_OP_BR_IF       0x0Du
/* Empty block-type for `loop` (no result on the value stack at the loop
 * header — the self-loop wrapper carries its value flow through locals +
 * an explicit `return`, so the loop body is balanced with zero stack
 * height at its top/bottom). */
#define WASM_BLOCKTYPE_EMPTY 0x40u
#define WASM_OP_LOCAL_GET   0x20u
#define WASM_OP_LOCAL_SET   0x21u
#define WASM_OP_LOCAL_TEE   0x22u
#define WASM_OP_I32_CONST   0x41u
#define WASM_OP_I64_CONST   0x42u
#define WASM_OP_I32_ADD     0x6Au
#define WASM_OP_I32_SUB     0x6Bu /* #695 — self-loop iter_budget decrement */
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
#define WASM_OP_I64_MUL     0x7Eu  /* #794 — IMUL truncating forms */
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
                        const struct FbxTcBlock *tc,
                        struct FbxEmitFailCtx *fail) {
  u32 i;
  u32 tc_idx;
  int saw_flag_reader = 0;
  int saw_flag_writer_local = 0;
  int saw_flag_writer = 0;
  if (!ir || ir->ninsts == 0) {
    SetFail(fail, FBX_IR_EMIT_EMPTY_IR, 0);
    return 0;
  }
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
#ifdef FBX735_DIAG_REFUSE_FLAGREADER
      /* firebox#738 Bug-A localization: refuse blocks with a flag reader
       * (BRANCH_COND/GET_FLAG) to test if the Jcc/CMOVcc lowering — NOT the
       * inline wide LOAD — is the corruptor. */
      SetFail(fail, FBX_IR_EMIT_FLAG_READER_DEFERRED, op);
      return 0;
#endif
    }
    if (op == FBX_IR_OP_SET_FLAGS_RAW || op == FBX_IR_OP_CMP ||
        op == FBX_IR_OP_TEST) {
      saw_flag_writer_local = 1;
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
      SetFail(fail, FBX_IR_EMIT_FLAG_READER_DEFERRED, op);
      return 0; /* deferred — no consumer at v0.1 */
    }
  }
  /* SET_FLAGS_RAW: when a flag-reader follows, the eager-update path is
   * triggered.  This requires the v2 IR shape (src1/src2 = operand vregs).
   * Validate that every SET_FLAGS_RAW carries the expected encoding. */
#ifdef FBX735_DIAG_REFUSE_XBLOCK_FLAGREADER
  /* firebox#738 Bug-A: refuse ONLY BRANCH_COND blocks that have NO in-block
   * flag writer — i.e. PURE cross-block flag readers (they read flags a PRIOR
   * block set). If this is CLEAN (wide LOAD still on), the corruptor is
   * cross-block lazy-flag staleness, not the in-block writer or the predicate. */
  if (saw_flag_reader && !saw_flag_writer_local) {
    SetFail(fail, FBX_IR_EMIT_FLAG_READER_DEFERRED, FBX_IR_OP_BRANCH_COND);
    return 0;
  }
#endif
  if (saw_flag_reader) {
    for (i = 0; i < ir->ninsts; ++i) {
      const struct FbxIrInst *p = &ir->insts[i];
      if (p->opcode == FBX_IR_OP_SET_FLAGS_RAW) {
        if (p->src1_kind != FBX_IR_KIND_VREG ||
            p->src2_kind != FBX_IR_KIND_VREG) {
          /* Stale v1 lift output without operand encoding — refuse. */
          SetFail(fail, FBX_IR_EMIT_SET_FLAGS_RAW_BAD, p->opcode);
          return 0;
        }
        /* Width must be 1/2/4/8 (we use it as i64 shift amount). */
        if (p->width != 1 && p->width != 2 &&
            p->width != 4 && p->width != 8) {
          SetFail(fail, FBX_IR_EMIT_SET_FLAGS_RAW_BAD, p->opcode);
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
            SetFail(fail, FBX_IR_EMIT_SET_FLAGS_RAW_BAD, p->opcode);
            return 0;
        }
      }
    }
  }
  (void)saw_flag_writer; /* presence-only; no further gating */

  /* #794 — IMUL flag-liveness refuse.  IMUL (FBX_IR_OP_IMUL) carries NO
   * synthesized flags at this increment: its CF/OF (full-product overflow)
   * and x86-undefined SF/ZF/AF/PF are deferred.  The eager-flag model emits a
   * SET_FLAGS_RAW after every OTHER flag-writing ALU op, so the flag shadow a
   * reader observes always comes from the LAST flag-definer before it.  If an
   * IMUL is that last definer (no intervening SET_FLAGS_RAW) at a BRANCH_COND
   * / GET_FLAG, the shadow is stale relative to Tier 1's eager imul-flag
   * computation → refuse the block so Tier 1 runs it (correct-or-refuse, the
   * #677/#735 pattern).  When a real flag-writer (add/sub/and/or/xor/cmp/test)
   * follows the imul before the reader — the hot-loop shape `imul…dec;jnz` —
   * imul's dead flags never matter and the block escalates.  (Flag-reader-free
   * blocks elide all SET_FLAGS_RAW anyway, so imul is no worse than the
   * existing dead-flag-elision model there; nothing to gate.) */
  {
    int imul_flags_pending = 0;
    for (i = 0; i < ir->ninsts; ++i) {
      u8 op = ir->insts[i].opcode;
      if (op == FBX_IR_OP_IMUL) {
        imul_flags_pending = 1;
      } else if (op == FBX_IR_OP_SET_FLAGS_RAW) {
        imul_flags_pending = 0; /* a real flag-def overwrites the shadow */
      } else if (op == FBX_IR_OP_BRANCH_COND || op == FBX_IR_OP_GET_FLAG) {
        if (imul_flags_pending) {
          SetFail(fail, FBX_IR_EMIT_IMUL_FLAGS_LIVE, op);
          return 0;
        }
      }
    }
  }

  /* firebox#735 — the #719 BAILOUT-after-committed-work refuse is REMOVED.
   *
   * #719 (root cause 2): a BAILOUT terminator emits
   * `EmitStoreIp(bailout_pc); i32.const 1; return` (m->ip = bailout PC,
   * exit=1).  The Tier-1 host USED to handle exit=1 by re-walking `entries[]`
   * FROM INDEX 0 instead of resuming at m->ip, so every op the T2 block
   * committed before the bailout (REG_SET / ALU / flags / stack) ran TWICE →
   * guest divergence → SIGSEGV.  #719 dodged this by refusing any block whose
   * BAILOUT followed a state-committing op.
   *
   * #735 fixes the HANDOFF at the source (threadedcode.c ExecuteBlock): on
   * exit=1, Tier 1 now RESUMES at the entry whose pc == m->ip — running only
   * the not-yet-committed suffix, never re-executing the committed prefix.
   * The lifter always emits BAILOUT at the FIRST un-liftable entry's `e->ip`
   * (fbx_ir_lift.c), so the committed prefix is exactly the entries at PCs
   * strictly below the bailout PC.  A BAILOUT-after-commit block is therefore
   * sound now and emittable — no gate needed.  (The mid-block software-MMU
   * miss bailout — EmitGuestVaToHostOffsetOrBailout — relies on the SAME
   * handoff and the same no-commit-before-translation discipline.) */
#ifdef FBX735_DIAG_REFUSE_BAILOUT_AFTER_COMMIT
  {
    int committed_before_bailout = 0;
    for (i = 0; i < ir->ninsts; ++i) {
      u8 op2 = ir->insts[i].opcode;
      switch (op2) {
        case FBX_IR_OP_REG_SET: case FBX_IR_OP_STORE: case FBX_IR_OP_ADD:
        case FBX_IR_OP_SUB: case FBX_IR_OP_AND: case FBX_IR_OP_OR:
        case FBX_IR_OP_XOR: case FBX_IR_OP_LEA: case FBX_IR_OP_PUSH:
        case FBX_IR_OP_POP: case FBX_IR_OP_CALL_DIRECT: case FBX_IR_OP_RET:
        case FBX_IR_OP_SET_FLAGS_RAW:
          committed_before_bailout = 1; break;
        case FBX_IR_OP_BAILOUT:
          if (committed_before_bailout) {
            SetFail(fail, FBX_IR_EMIT_BAILOUT_AFTER_COMMIT, op2);
            return 0;
          }
          break;
        default: break;
      }
    }
  }
#endif

  /* Pass 2 — per-op acceptance. */
  for (i = 0; i < ir->ninsts; ++i) {
    u8 op = ir->insts[i].opcode;
    /* firebox#735 — guest-VA memory-operand handling under a NON-linear
     * (software-MMU) build.
     *
     * Background (firebox#719): LOAD/STORE/PUSH/POP/CALL_DIRECT/RET compute
     * their effective address as a guest register VALUE (+ disp) — a guest
     * VIRTUAL address.  When HasLinearMapping() (the native 64-bit build,
     * CAN_64BIT && !FLAG_nolinear), `ToHost(va) == va` so the VA IS a linear
     * offset and the emit can `i32.wrap_i64` it directly.  On the wasm32
     * blink build CAN_64BIT==0 (builtin.h:239-245) → HasLinearMapping() is
     * compile-time FALSE → blink runs its SOFTWARE MMU: a guest VA (e.g. the
     * stack at 0x4fffff...) is NOT a linear offset and must be translated
     * through the page tables (blink/memory.c LookupAddress).  #719 REFUSED
     * all six ops here to dodge the OOB-trap-and-corrupt (dispatch-diag
     * witness: funcref 2/3 trapped at the `i64.load (reg+disp)` site).
     *
     * #735 LIFTS the refuse for LOAD/STORE: their emit now routes the VA
     * through EmitGuestVaToHostOffsetOrBailout, which inlines blink's TLB
     * fast path (reading the page tables out of the shared guest memory) and
     * BAILS OUT to Tier 1 on any miss/fault (correct-or-refuse at runtime,
     * spec §5.5; Tier 1 resumes at the bailout PC — firebox#735
     * threadedcode.c).  The stack-mutating ops (PUSH/POP/CALL_DIRECT/RET)
     * STAY refused this increment: they pre-decrement/advance RSP, so making
     * the translation no-commit-before requires reordering their emit — a
     * follow-on (work/tasks/737).  REG_GET/REG_SET/LEA were never refused
     * (they touch only `m_ptr + offsetof` into the Machine struct, a real
     * linear-memory C object). */
    if (!HasLinearMapping()) {
      /* firebox#738 — the wide (8-byte / i64) guest LOAD is RE-ENABLED here.
       * SESSION 4 proved the wide LOAD INNOCENT (the LOAD value is byte-faithful
       * to the host's own read at the same translated offset, three independent
       * ways).  The real corruptor was the #599 BRANCH_COND lazy-flag lowering
       * in blocks that ALSO carried a wide LOAD; the SESSION-3 interim refuse of
       * the wide LOAD was byte-clean only because it bailed those same blocks.
       * With the #599 fix below (EmitFlagsAfterAlu — see its block comment), the
       * wide LOAD emits inline again and the byte-identity gate is GREEN.
       *
       * FBX735_DIAG_REFUSE_LOAD_WIDE remains as a localization knob (refuse the
       * wide LOAD to A/B the fix); it is OFF by default. */
#ifdef FBX735_DIAG_REFUSE_LOAD_WIDE
      if (op == FBX_IR_OP_LOAD &&
          (!ir->insts[i].width || ir->insts[i].width == 8)) {
        SetFail(fail, FBX_IR_EMIT_NONLINEAR_GUEST_MEM, op);
        return 0;
      }
#endif
      switch (op) {
#ifdef FBX735_DIAG_REFUSE_LOADSTORE
        case FBX_IR_OP_LOAD:
        case FBX_IR_OP_STORE:
#endif
        case FBX_IR_OP_PUSH:
        case FBX_IR_OP_POP:
        case FBX_IR_OP_CALL_DIRECT:
        case FBX_IR_OP_RET:
          SetFail(fail, FBX_IR_EMIT_NONLINEAR_GUEST_MEM, op);
          return 0;
        default:
          break;
      }
    }
    switch (op) {
      case FBX_IR_OP_PC_MARK:
      case FBX_IR_OP_REG_GET:
      case FBX_IR_OP_REG_SET:
      case FBX_IR_OP_LOAD:  /* #677: MOV r/m memory-form load */
      case FBX_IR_OP_STORE: /* #677: MOV r/m memory-form store */
      case FBX_IR_OP_ADD:
      case FBX_IR_OP_SUB:
      case FBX_IR_OP_AND:
      case FBX_IR_OP_OR:
      case FBX_IR_OP_XOR:
      case FBX_IR_OP_IMUL:  /* #794 — truncating integer multiply */
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
        SetFail(fail, FBX_IR_EMIT_UNSUPPORTED_OPCODE, op);
        return 0;
    }
  }
  /* Reachability gate: every TC entry contributing to the IR must be a
   * reg-form opcode (modrm.mod == 3) — the IR doesn't model memory
   * operands at v0.1.  Walk parallel: every NORMAL kind TC entry must
   * be a reg-form instruction.  We do not enforce a 1:1 mapping (the
   * lifter can emit several IR insts per x86 op); only the operand-shape
   * is checked. */
  if (!tc || tc->nentries == 0) {
    SetFail(fail, FBX_IR_EMIT_EMPTY_IR, 0);
    return 0;
  }
  for (tc_idx = 0; tc_idx < tc->nentries; ++tc_idx) {
    const struct FbxTcEntry *e = &tc->entries[tc_idx];
    u64 rde;
    u64 mop;
    if (e->kind != FBX_TC_KIND_NORMAL) {
      SetFail(fail, FBX_IR_EMIT_TC_KIND_NON_NORMAL, 0);
      return 0;
    }
    rde = e->rde;
    mop = Mopcode(rde);
    /* MOV r/m, r and MOV r, r/m and ALU-RR and TEST. */
    switch (mop) {
      /* #677 — MOV r/m, r (0x88/0x89) and MOV r, r/m (0x8A/0x8B): the
       * lifter (blink/fbx_ir_lift.c) is now the authority on the operand
       * form.  For modrm.mod==3 it emits REG_GET/REG_SET (accepted by the
       * IR-side Pass-2 above); for the supported memory form `[base+disp]`
       * it emits LOAD/STORE (also accepted); for unsupported addressing
       * (SIB index, RIP-relative, no-base SIB) it emits BAILOUT.  So this
       * gate no longer rejects mem-form MOV — the IR shape it produced
       * already encodes the correct-or-bailout decision.  Removing the
       * mod3 refuse here is what flips these opcodes from
       * reason=mod3_required to synthesis. */
      case 0x088: case 0x089:
      case 0x08A: case 0x08B:
        break;
      /* ALU-RR (0x00/0x01/0x08/0x09/...) and TEST (0x84/0x85) still require
       * modrm.mod==3 — their lifter emits REG_GET/REG_SET unconditionally
       * and does NOT model the memory operand form yet (a later increment). */
      case 0x000: case 0x001: case 0x008: case 0x009:
      case 0x020: case 0x021: case 0x028: case 0x029:
      case 0x030: case 0x031: case 0x038: case 0x039:
      case 0x084: case 0x085:
        if (!Mod3(rde)) {
          SetFail(fail, FBX_IR_EMIT_MOD3_REQUIRED, (u8)(mop & 0xFF));
          return 0;
        }
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
        if (!Mod3(rde)) {
          SetFail(fail, FBX_IR_EMIT_MOD3_REQUIRED, (u8)(mop & 0xFF));
          return 0;
        }
        break;
      /* §13.5 MOVZX r, r/m{8,16}.  mod3 required at v0.1 (synthesis's
       * EmitRegLoad doesn't model memory-modrm yet — same gate as the
       * MOV variants above). */
      case 0x1B6: case 0x1B7:
        if (!Mod3(rde)) {
          SetFail(fail, FBX_IR_EMIT_MOD3_REQUIRED, (u8)(mop & 0xFF));
          return 0;
        }
        break;
      /* #794 — IMUL r,r/m,imm32 (0x69) / imm8 (0x6B) and IMUL r,r/m (0x1AF =
       * 0x0FAF).  Reg-form (modrm.mod==3) only at this increment — the lifter
       * (LiftImul) bails the memory-operand form.  i64.mul truncating product;
       * the imul-flag-liveness pass below refuses blocks where the (un-
       * synthesized) flags are observed. */
      case 0x069: case 0x06B: case 0x1AF:
        if (!Mod3(rde)) {
          SetFail(fail, FBX_IR_EMIT_MOD3_REQUIRED, (u8)(mop & 0xFF));
          return 0;
        }
        break;
      /* #794 — accumulator-immediate ALU forms (ADD/OR/AND/SUB/XOR/CMP
       * rAX,imm).  No modrm byte (dst/lhs is the implicit rAX) → no mod3
       * check, like MOV reg,imm below.  The lifter emits the same
       * REG_GET/ALU/REG_SET/SET_FLAGS_RAW shape as the group-1 imm forms. */
      case 0x004: case 0x005: case 0x00C: case 0x00D:
      case 0x024: case 0x025: case 0x02C: case 0x02D:
      case 0x034: case 0x035: case 0x03C: case 0x03D:
        break;
      /* MOV r/m, imm and group-1 r/m, imm: require modrm.mod==3 as well. */
      case 0x0C6: case 0x0C7:
      case 0x080: case 0x081: case 0x083:
        if (!Mod3(rde)) {
          SetFail(fail, FBX_IR_EMIT_MOD3_REQUIRED, (u8)(mop & 0xFF));
          return 0;
        }
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
        SetFail(fail, FBX_IR_EMIT_UNSUPPORTED_MOPCODE, (u8)(mop & 0xFF));
        return 0;
    }
  }
  /* Walk the IR once more for LEA SIB rejection. */
  for (i = 0; i < ir->ninsts; ++i) {
    const struct FbxIrInst *p = &ir->insts[i];
    if (p->opcode == FBX_IR_OP_LEA) {
      /* #778: accept BOTH the disp-only form (src1=GREG base, src2=NONE/IMM,
       * scale=0) AND the SIB-INDEX form (src1=GREG base, src2=GREG index,
       * scale ∈ {1,2,4,8}).  EmitLea lowers both — the indexed form adds
       * weg[index]*scale to the effective address (no MMU translation; LEA
       * never dereferences memory).  Anything else (a missing base, or a GREG
       * index with an invalid scale) is a form the emitter doesn't model. */
      if (p->src1_kind != FBX_IR_KIND_GREG) {
        SetFail(fail, FBX_IR_EMIT_LEA_SIB_FORM, p->opcode);
        return 0;
      }
      if (p->src2_kind == FBX_IR_KIND_GREG) {
        /* Indexed: scale must be a valid SIB scale. */
        if (p->scale != 1 && p->scale != 2 && p->scale != 4 &&
            p->scale != 8) {
          SetFail(fail, FBX_IR_EMIT_LEA_SIB_FORM, p->opcode);
          return 0;
        }
      } else if (p->src2_kind != FBX_IR_KIND_NONE &&
                 p->src2_kind != FBX_IR_KIND_IMM) {
        SetFail(fail, FBX_IR_EMIT_LEA_SIB_FORM, p->opcode);
        return 0;
      }
    }
    if (p->opcode == FBX_IR_OP_REG_GET || p->opcode == FBX_IR_OP_REG_SET) {
      /* REG_GET sources a guest register OR (#794) an immediate — the lifter
       * emits IMM src1 for the "load imm into a vreg" form used by every
       * immediate-operand ALU lift (LiftAluImmCommon / LiftImul).  The emit
       * (REG_GET dispatch) lowers GREG via EmitRegLoad and IMM via the
       * block-ctx IMM slot; both are accepted here.  REG_SET still requires a
       * GREG destination. */
      if (p->opcode == FBX_IR_OP_REG_GET &&
          p->src1_kind != FBX_IR_KIND_GREG &&
          p->src1_kind != FBX_IR_KIND_IMM) {
        SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
        return 0;
      }
      if (p->opcode == FBX_IR_OP_REG_SET &&
          p->dst_kind != FBX_IR_KIND_GREG) {
        SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
        return 0;
      }
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

/* firebox#735 — software-MMU inline-translation offsets + constants.
 *
 * On the wasm32 Blink build CAN_64BIT==0 so HasLinearMapping() is FALSE and
 * blink runs its software MMU (blink/memory.c): a guest VA is NOT a linear
 * offset — it is translated through the page tables to a host pointer (which,
 * on wasm32, IS a linear-memory offset).  The §13.5 memory-operand emit
 * (EmitGuestVaToHostOffsetOrBailout) inlines blink's TLB FAST PATH
 * (FindPageTableEntry's `m->tlb[(page>>12)&31]` hit branch + GetPageAddress's
 * PAGE_HOST branch + LookupAddress2's `host + (virt & 4095)`), reading the
 * page tables OUT OF the shared guest memory the T2 module imports.  Any case
 * the inline path does NOT cover (TLB miss, non-PAGE_HOST entry, protection
 * fault, metal/CPL!=3, page-crossing access) BAILS OUT to Tier 1, which runs
 * the authoritative full walk (filling the TLB) — correct-or-refuse, spec
 * §5.5.  These offsets/constants MUST stay in sync with blink/machine.h
 * (struct Machine / MachineTlb layout) + the PAGE_* bits there; offsetof keeps
 * them C-compile-time-derived (NOT hand-baked) per
 * `class_lesson_test_fixture_replicated_the_bug_it_was_meant_to_catch`. */
#define M_OFF_METAL ((u32)offsetof(struct Machine, metal))
#define M_OFF_TLB   ((u32)offsetof(struct Machine, tlb))
/* struct MachineTlb { i64 page; u64 entry; } — 16 bytes, page at +0,
 * entry at +8.  (Static-asserted at the top of EmitGuestVaToHostOffset*.) */
#define M_TLB_STRIDE      16u
#define M_TLB_OFF_PAGE     0u
#define M_TLB_OFF_ENTRY    8u
#define M_TLB_KEY_MASK    31u  /* ARRAYLEN(m->tlb)-1 == 32-1 (machine.h tlb[32]) */
/* PAGE_* bits — mirror blink/machine.h:84-92.  PAGE_TA is the 48-bit
 * host-address field; on wasm32 a PAGE_HOST entry's TA bits ARE the wasm
 * linear-memory offset of the page (AllocateAnonymousPage:
 * `real | PAGE_HOST | PAGE_U | PAGE_RW | PAGE_V`, with real a wasm mmap
 * offset that fits PAGE_TA — unassert(!(real & ~PAGE_TA))). */
#define M_PAGE_V    0x0000000000000001ull
#define M_PAGE_RW   0x0000000000000002ull
#define M_PAGE_U    0x0000000000000004ull
#define M_PAGE_HOST 0x0000000000000400ull
#define M_PAGE_TA   0x0000fffffffff000ull

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

/* ────────────────────────────────────────────────────────────────────────── */
/* #635 ABI redesign — local-index layout                                     */
/*                                                                            */
/* Local 0 = m_ptr (i32)            ← function param 0                        */
/* Local 1 = block_ctx_ptr (i32)    ← function param 1 (#635 new)             */
/* Locals 2..nvregs+1 = vreg locals (i64)                                     */
/* Locals nvregs+2..nvregs+3 = scratch_z (i64), scratch_flags (i32)           */
/* ────────────────────────────────────────────────────────────────────────── */

#define EMIT_LOCAL_M_PTR        0u
#define EMIT_LOCAL_BLOCK_CTX    1u
#define EMIT_LOCAL_VREG_BASE    2u

/* ────────────────────────────────────────────────────────────────────────── */
/* #635 — per-block constant index allocator.                                 */
/*                                                                            */
/* Three partitioned sub-ranges (PCs / immediates / greg-offsets) within the */
/* shared `consts[]` slot space defined by fbx_t2_block_ctx.h.  The allocator */
/* runs ONCE per block — first to build the `consts[]` table the bridge      */
/* fills the FbxT2BlockCtx with, and second (with identical input → identical */
/* allocation sequence) implicitly as the emit pass walks the IR and pulls   */
/* slot indices from the builder.                                            */
/*                                                                            */
/* PURITY (LOAD-BEARING): the allocator's decisions depend ONLY on the IR    */
/* shape (opcode, operand-kinds, encounter order).  They do NOT depend on    */
/* the constant VALUES.  Two IRs differing only in immediate values produce  */
/* the same slot indices → the same wasm bytes.                              │
* ────────────────────────────────────────────────────────────────────────── */

struct BlockCtxBuilder {
  u32 next_pc_idx;     /* relative within [0, FBX_T2_CTX_PC_COUNT) */
  u32 next_imm_idx;    /* relative within [0, FBX_T2_CTX_IMM_COUNT) */
  u32 next_greg_idx;   /* relative within [0, FBX_T2_CTX_GREG_COUNT) */
  /* greg dedupe: greg_slot_plus1[g] == 0 → not yet allocated; otherwise
   * slot_within_greg_range = greg_slot_plus1[g] - 1. */
  u8 greg_slot_plus1[16];
  int overflow;        /* sticky: set on any sub-range overflow */
  /* High-water mark used by EmitFunctionBody to populate the FbxT2BlockCtx
   * nconsts field analogue (the emit pass doesn't write this; the bridge   │
* does — but we still compute it to validate against the cap).             │
*/
  u32 high_water;  /* one-past-last absolute slot index used */
  /* When `consts` is non-NULL, allocator writes the const VALUE into the   │
* assigned slot.  When NULL (emit-time re-walk), it just allocates.        │
*/
  u64 *consts;
  u32 consts_cap;
  /* firebox#735 — the guest PC of the x86 instruction currently being
   * allocated for.  Updated by every PC_MARK in AllocSlotsForInst (which
   * BOTH walkers dispatch through in identical order), so a LOAD/STORE's
   * MMU-miss bailout-PC slot carries the failing op's PC in both the
   * consts-build walk and the emit re-walk. */
  u64 cur_pc;
};

static void BcInit(struct BlockCtxBuilder *b, u64 *consts, u32 consts_cap) {
  memset(b, 0, sizeof(*b));
  b->consts = consts;
  b->consts_cap = consts_cap;
}

/* Allocate the next PC slot.  Returns absolute slot index, or sets overflow  */
/* and returns 0xFFFFFFFFu.  Writes `value` into consts[slot] if consts is    */
/* non-NULL.                                                                  */
static u32 BcAllocPc(struct BlockCtxBuilder *b, u64 value) {
  u32 slot;
  if (b->overflow || b->next_pc_idx >= FBX_T2_CTX_PC_COUNT) {
    b->overflow = 1;
    return 0xFFFFFFFFu;
  }
  slot = FBX_T2_CTX_PC_BASE + b->next_pc_idx++;
  if (slot >= b->high_water) b->high_water = slot + 1u;
  if (b->consts && slot < b->consts_cap) b->consts[slot] = value;
  return slot;
}

/* Allocate the next IMM slot. */
static u32 BcAllocImm(struct BlockCtxBuilder *b, u64 value) {
  u32 slot;
  if (b->overflow || b->next_imm_idx >= FBX_T2_CTX_IMM_COUNT) {
    b->overflow = 1;
    return 0xFFFFFFFFu;
  }
  slot = FBX_T2_CTX_IMM_BASE + b->next_imm_idx++;
  if (slot >= b->high_water) b->high_water = slot + 1u;
  if (b->consts && slot < b->consts_cap) b->consts[slot] = value;
  return slot;
}

/* Allocate / lookup a greg-offset slot.  Same greg id → same slot within a  */
/* block.  Returns absolute slot index, or 0xFFFFFFFFu on overflow.  Writes  */
/* `M_OFF_WEG + greg_id*8` into consts[slot] on first allocation only.       */
static u32 BcAllocGreg(struct BlockCtxBuilder *b, u32 greg_id) {
  u32 slot;
  if (b->overflow || greg_id >= 16u) {
    b->overflow = 1;
    return 0xFFFFFFFFu;
  }
  if (b->greg_slot_plus1[greg_id] != 0) {
    return FBX_T2_CTX_GREG_BASE + (u32)(b->greg_slot_plus1[greg_id] - 1u);
  }
  if (b->next_greg_idx >= FBX_T2_CTX_GREG_COUNT) {
    b->overflow = 1;
    return 0xFFFFFFFFu;
  }
  slot = FBX_T2_CTX_GREG_BASE + b->next_greg_idx;
  b->greg_slot_plus1[greg_id] = (u8)(b->next_greg_idx + 1u);
  b->next_greg_idx++;
  if (slot >= b->high_water) b->high_water = slot + 1u;
  if (b->consts && slot < b->consts_cap) {
    b->consts[slot] = (u64)(M_OFF_WEG + greg_id * 8u);
  }
  return slot;
}

/* Forward decl: walks the IR allocating slots in encounter order.  Used by  */
/* both `fbx_ir_build_block_ctx_consts` (build mode: writes values) and the  */
/* emit pass's prep step (allocate mode: just establishes high-water mark).  */
/* The emit body then re-allocates as it walks, getting the same indices    */
/* by virtue of identical encounter order.                                   */
static void WalkAllocateAllSlots(const struct FbxIrBlock *ir,
                                 struct BlockCtxBuilder *b);

/* ────────────────────────────────────────────────────────────────────────── */
/* Per-instruction slot allocations.  Two walks (the up-front consts-table   */
/* builder and the emit-pass body walker) call these in identical sequence; */
/* the deterministic encounter-order policy from abi-redesign.md §6.1       */
/* yields the same slot indices in both.                                     */
/*                                                                            */
/* Each helper:                                                               */
/*   - Returns 1 on success, 0 on overflow (caller must abort escalation).  */
/*   - For build mode (`b->consts != NULL`), writes the constant VALUE into */
/*     consts[slot] at the same time it allocates the slot.                  */
/*                                                                            */
/* The helpers are called UNCONDITIONALLY for any inst that may carry per-  */
/* block constants in v0.1 coverage.  Their internal logic decides whether  */
/* a given operand actually consumes a slot (e.g. REG_SET only consumes an  */
/* IMM slot when src1_kind == IMM; LEA only consumes IMM if imm != 0; etc.) */
/* ────────────────────────────────────────────────────────────────────────── */

static u32 AllocRegGetSlot(struct BlockCtxBuilder *b,
                           const struct FbxIrInst *p) {
  /* REG_GET reads guest greg src1 — one greg-offset slot (deduped). */
  return BcAllocGreg(b, p->src1);
}

/* #794 — REG_GET with an IMM src ("load immediate into a vreg", used by the
 * immediate-operand ALU lifts).  Needs one IMM slot for p->imm — the
 * immediate cannot be baked as an inline i64.const (that would make the wasm
 * bytes value-dependent, breaking the #635 block-ctx byte-stability model);
 * it is hoisted into the per-block constants table exactly like REG_SET-IMM. */
static u32 AllocRegGetImmSlot(struct BlockCtxBuilder *b,
                              const struct FbxIrInst *p) {
  return BcAllocImm(b, p->imm);
}

static u32 AllocRegSetGregSlot(struct BlockCtxBuilder *b,
                               const struct FbxIrInst *p) {
  return BcAllocGreg(b, p->dst);
}

/* REG_SET with IMM src: needs one IMM slot for p->imm.  Returns the slot. */
static u32 AllocRegSetImmSlot(struct BlockCtxBuilder *b,
                              const struct FbxIrInst *p) {
  return BcAllocImm(b, p->imm);
}

/* LEA: greg base slot + greg dst slot + (when imm != 0) an IMM slot          */
/* + (#778, indexed form) a greg index slot.  The slot order is FIXED — the   */
/* up-front allocator (block-ctx builder walk) and the body walker must call  */
/* BcAlloc* in the SAME order so slot indices match; base, dst, [imm], [index]*/
/* is the canonical order here and in both call sites.                        */
struct LeaSlots {
  u32 base_slot;
  u32 dst_slot;
  u32 imm_slot;       /* 0xFFFFFFFFu when imm == 0 (no slot needed) */
  int has_imm;
  u32 index_slot;     /* 0xFFFFFFFFu when no index term (#778) */
  int has_index;
  u32 scale;          /* SIB scale ∈ {1,2,4,8}; 0 when no index */
};

static int AllocLeaSlots(struct BlockCtxBuilder *b,
                         const struct FbxIrInst *p, struct LeaSlots *out) {
  out->base_slot = BcAllocGreg(b, p->src1);
  out->dst_slot = BcAllocGreg(b, p->dst);
  if (p->imm != 0) {
    out->imm_slot = BcAllocImm(b, p->imm);
    out->has_imm = 1;
  } else {
    out->imm_slot = 0xFFFFFFFFu;
    out->has_imm = 0;
  }
  /* #778 — the indexed form carries a GREG index in src2 + a SIB scale. */
  if (p->src2_kind == FBX_IR_KIND_GREG) {
    out->index_slot = BcAllocGreg(b, p->src2);
    out->has_index = 1;
    out->scale = p->scale;
  } else {
    out->index_slot = 0xFFFFFFFFu;
    out->has_index = 0;
    out->scale = 0;
  }
  return !b->overflow;
}

/* BRANCH_TAKEN: one PC slot for imm (target). */
static u32 AllocBranchTakenSlot(struct BlockCtxBuilder *b,
                                const struct FbxIrInst *p) {
  return BcAllocPc(b, p->imm);
}

/* BAILOUT: one PC slot for imm. */
static u32 AllocBailoutSlot(struct BlockCtxBuilder *b,
                            const struct FbxIrInst *p) {
  return BcAllocPc(b, p->imm);
}

/* BRANCH_COND: two PC slots — taken_pc (imm), fallthrough_pc (src2). */
struct BranchCondSlots {
  u32 taken_pc_slot;
  u32 fallthrough_pc_slot;
};

static int AllocBranchCondSlots(struct BlockCtxBuilder *b,
                                const struct FbxIrInst *p,
                                struct BranchCondSlots *out) {
  out->taken_pc_slot = BcAllocPc(b, p->imm);
  out->fallthrough_pc_slot = BcAllocPc(b, (u64)(u32)p->src2);
  return !b->overflow;
}

/* CALL_DIRECT: PC slot for return_pc (the immediate pushed to guest stack), */
/* PC slot for target_pc (imm, the new IP).  RSP greg slot for stack ops.   */
struct CallDirectSlots {
  u32 rsp_slot;
  u32 return_pc_slot;
  u32 target_pc_slot;
};

static int AllocCallDirectSlots(struct BlockCtxBuilder *b,
                                const struct FbxIrInst *p,
                                struct CallDirectSlots *out) {
  out->rsp_slot = BcAllocGreg(b, FBX_GREG_RSP);
  out->return_pc_slot = BcAllocPc(b, (u64)(u32)p->src2);
  out->target_pc_slot = BcAllocPc(b, p->imm);
  return !b->overflow;
}

/* RET: only an RSP greg slot. */
static u32 AllocRetRspSlot(struct BlockCtxBuilder *b) {
  return BcAllocGreg(b, FBX_GREG_RSP);
}

/* PUSH: greg slot for the src greg + RSP slot. */
struct PushPopSlots {
  u32 rsp_slot;
  u32 reg_slot;
};

static int AllocPushSlots(struct BlockCtxBuilder *b,
                          const struct FbxIrInst *p,
                          struct PushPopSlots *out) {
  out->reg_slot = BcAllocGreg(b, p->src1);
  out->rsp_slot = BcAllocGreg(b, FBX_GREG_RSP);
  return !b->overflow;
}

static int AllocPopSlots(struct BlockCtxBuilder *b,
                         const struct FbxIrInst *p,
                         struct PushPopSlots *out) {
  out->rsp_slot = BcAllocGreg(b, FBX_GREG_RSP);
  out->reg_slot = BcAllocGreg(b, p->dst);
  return !b->overflow;
}

/* #677 — LOAD / STORE memory-form: one greg-offset slot for the base
 * address register, plus (only when disp != 0) one IMM slot for the
 * sign-extended displacement.  Same disp-zero-elision policy as LEA so
 * the scarce 4-entry IMM range isn't consumed by `[base+0]` forms. */
struct MemAddrSlots {
  u32 base_slot;
  u32 disp_slot;       /* 0xFFFFFFFFu when disp == 0 (no slot needed) */
  int has_disp;
  u32 bailout_pc_slot; /* firebox#735 — PC to bail to on a TLB miss/fault */
};

static int AllocLoadStoreSlots(struct BlockCtxBuilder *b,
                               const struct FbxIrInst *p,
                               struct MemAddrSlots *out) {
  /* src1 is always the base greg id for both LOAD and STORE. */
  out->base_slot = BcAllocGreg(b, p->src1);
  if (p->imm != 0) {
    out->disp_slot = BcAllocImm(b, p->imm);
    out->has_disp = 1;
  } else {
    out->disp_slot = 0xFFFFFFFFu;
    out->has_disp = 0;
  }
  /* firebox#735 — bailout PC for the software-MMU miss path.  Allocation
   * order (greg → imm → pc) is identical in both walkers; the PC VALUE is
   * b->cur_pc (the most recent PC_MARK, i.e. this op's x86 instruction PC). */
  out->bailout_pc_slot = BcAllocPc(b, b->cur_pc);
  return !b->overflow;
}

/* Per-inst dispatch: invoke whichever Alloc* helper(s) the inst opcode needs.
 * Used by both the consts-table-build pass (`fbx_ir_build_block_ctx_consts`) */
/* and the emit-pass body walker.  The two walkers therefore allocate slots */
/* in identical order. */
static void AllocSlotsForInst(struct BlockCtxBuilder *b,
                              const struct FbxIrInst *p) {
  if (b->overflow) return;
  /* firebox#735 — track the current x86 PC for the LOAD/STORE bailout slot.
   * PC_MARK precedes every lifted op (fbx_ir_lift.c), carrying the op's PC in
   * imm.  Both the consts-build walk and the emit re-walk update cur_pc here
   * in identical order → matching bailout-PC slot values/indices. */
  if (p->opcode == FBX_IR_OP_PC_MARK) {
    b->cur_pc = p->imm;
  }
  switch (p->opcode) {
    case FBX_IR_OP_PC_MARK:
    case FBX_IR_OP_CMP:
    case FBX_IR_OP_TEST:
    case FBX_IR_OP_ADD:
    case FBX_IR_OP_SUB:
    case FBX_IR_OP_AND:
    case FBX_IR_OP_OR:
    case FBX_IR_OP_XOR:
    case FBX_IR_OP_IMUL:  /* #794 — pure vreg-local, no block_ctx slots */
    case FBX_IR_OP_SET_FLAGS_RAW:
    case FBX_IR_OP_GET_FLAG:
      /* No per-block constants. */
      break;
    case FBX_IR_OP_REG_GET:
      if (p->src1_kind == FBX_IR_KIND_IMM) {
        (void)AllocRegGetImmSlot(b, p); /* #794 — load-imm-into-vreg */
      } else {
        (void)AllocRegGetSlot(b, p);
      }
      break;
    case FBX_IR_OP_REG_SET:
      if (p->src1_kind == FBX_IR_KIND_IMM) {
        (void)AllocRegSetImmSlot(b, p);
      }
      (void)AllocRegSetGregSlot(b, p);
      break;
    case FBX_IR_OP_LOAD:
    case FBX_IR_OP_STORE: {
      /* #677 — base greg slot (+ disp IMM slot when disp != 0).  Allocated
       * identically here and in the body walker so slot indices match. */
      struct MemAddrSlots s;
      (void)AllocLoadStoreSlots(b, p, &s);
      break;
    }
    case FBX_IR_OP_LEA: {
      struct LeaSlots s;
      (void)AllocLeaSlots(b, p, &s);
      break;
    }
    case FBX_IR_OP_BRANCH_TAKEN:
      (void)AllocBranchTakenSlot(b, p);
      break;
    case FBX_IR_OP_BAILOUT:
      (void)AllocBailoutSlot(b, p);
      break;
    case FBX_IR_OP_BRANCH_COND: {
      struct BranchCondSlots s;
      (void)AllocBranchCondSlots(b, p, &s);
      break;
    }
    case FBX_IR_OP_CALL_DIRECT: {
      struct CallDirectSlots s;
      (void)AllocCallDirectSlots(b, p, &s);
      break;
    }
    case FBX_IR_OP_RET:
      (void)AllocRetRspSlot(b);
      break;
    case FBX_IR_OP_PUSH: {
      struct PushPopSlots s;
      (void)AllocPushSlots(b, p, &s);
      break;
    }
    case FBX_IR_OP_POP: {
      struct PushPopSlots s;
      (void)AllocPopSlots(b, p, &s);
      break;
    }
    default:
      b->overflow = 1;
      break;
  }
}

static void WalkAllocateAllSlots(const struct FbxIrBlock *ir,
                                 struct BlockCtxBuilder *b) {
  u32 i;
  for (i = 0; i < ir->ninsts; ++i) {
    AllocSlotsForInst(b, &ir->insts[i]);
    if (b->overflow) break;
  }
}

int fbx_ir_build_block_ctx_consts(const struct FbxIrBlock *ir,
                                  u64 *out_consts, u32 out_consts_cap,
                                  u32 *out_nconsts) {
  struct BlockCtxBuilder b;
  u32 i;
  if (!ir || !out_consts || !out_nconsts) return 0;
  if (out_consts_cap < FBX_T2_BLOCK_CTX_MAX_CONSTS) return 0;
  for (i = 0; i < out_consts_cap; ++i) out_consts[i] = 0;
  BcInit(&b, out_consts, out_consts_cap);
  WalkAllocateAllSlots(ir, &b);
  if (b.overflow) return 0;
  *out_nconsts = b.high_water;
  return 1;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Block-ctx-aware load helpers.                                              */
/*                                                                            */
/* The bridge fills consts[slot] with a u64 value.  The emit pass reads it   */
/* with either i64.load (for full PCs / immediates) or i32.load (for          */
/* greg-offset values that fit in u32 — wasm linear memory is little-endian, */
/* so the low 32 bits live at the same byte offset).                          */
/* ────────────────────────────────────────────────────────────────────────── */

/* Emit `local.get $block_ctx; i64.load offset=<consts[slot]> align=3`.       */
/* Leaves an i64 on the wasm stack.                                           */
static void EmitLoadCtxI64(struct FbxWasmBuffer *body, u32 slot) {
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, EMIT_LOCAL_BLOCK_CTX);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD);
  fbx_wasm_buffer_uleb(body, 3); /* align log2 = 3 (8-byte) */
  fbx_wasm_buffer_uleb(body, FBX_T2_CTX_OFF_CONST(slot));
}

/* Emit `local.get $block_ctx; i32.load offset=<consts[slot]> align=2`.       */
/* Leaves the low 32 bits of consts[slot] on the wasm stack as i32.           */
static void EmitLoadCtxI32(struct FbxWasmBuffer *body, u32 slot) {
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, EMIT_LOCAL_BLOCK_CTX);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_LOAD);
  fbx_wasm_buffer_uleb(body, 2); /* align log2 = 2 (4-byte) */
  fbx_wasm_buffer_uleb(body, FBX_T2_CTX_OFF_CONST(slot));
}

/* Emit code that pushes the effective i32 host-memory address (m_ptr +     */
/* greg_offset) onto the stack, where greg_offset is read from the block_ctx */
/* slot.  Pattern:                                                            */
/*   local.get $m_ptr                                                          */
/*   local.get $block_ctx                                                      */
/*   i32.load offset=<ctx_off>                                                 */
/*   i32.add                                                                   */
/* Used as the base address for greg loads/stores in Class 1 sites. */
static void EmitPushGregBaseAddr(struct FbxWasmBuffer *body, u32 greg_slot) {
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, EMIT_LOCAL_M_PTR);
  EmitLoadCtxI32(body, greg_slot);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_ADD);
}

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

/* Emit `<push effective addr m_ptr + ctx[greg_slot]>; load.WIDTH offset=0`  */
/* to leave the guest register value on the stack as an i64.                  */
/*                                                                            */
/* #635 ABI redesign: the guest-register byte offset is read from the         */
/* per-block context slot at runtime, NOT baked into the wasm `offset=` ULEB. */
/* This is a Class 1 site (M1) per `abi-redesign.md` §2.2.                    */
static void EmitRegLoad(struct FbxWasmBuffer *body, u32 greg_slot, u8 width) {
  EmitPushGregBaseAddr(body, greg_slot);
  /* Memarg: align + offset=0 (effective address already computed). */
  switch (width) {
    case 1:
      fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD8U);
      fbx_wasm_buffer_uleb(body, 0); /* align (log2) */
      fbx_wasm_buffer_uleb(body, 0); /* offset */
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
 * guest register at greg_slot, width-truncated.
 *
 * #635: greg_slot is the per-block context slot holding the
 * (M_OFF_WEG + greg_id*8) byte offset.  Class 1 site M2. */
/* Emit a width-sized store of a value into a guest REGISTER slot.  Stack on
 * entry: [ addr(i32), value(i64) ]; both are consumed.
 *
 * x86-64 register-write width semantics (the authority here, since this writes
 * Machine.weg[]):
 *   - width 8: full i64.store.
 *   - width 4: a 32-bit register write ZERO-EXTENDS to 64 bits, so mask the
 *     value to 32 bits and store the FULL i64 — clearing the upper half of the
 *     8-byte greg slot.  (i64.store32 alone would leave the upper 32 bits
 *     STALE; #794 — latent until immediate-operand 32-bit ALU started
 *     escalating, then it corrupted any register that previously held a
 *     pointer, e.g. AX=0x00004fff_00000000 → OOB.)
 *   - width 2 / 1: 16-/8-bit writes do NOT zero-extend → i64.store16/8
 *     preserve the upper bits, which is correct.
 * REGISTER-only: guest MEMORY stores keep i64.store32 (a real 4-byte write). */
static void EmitRegSlotStore(struct FbxWasmBuffer *body, u8 width) {
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
      /* (value & 0xFFFFFFFF) then full i64.store — zero-extends to 64 bits. */
      fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
      fbx_wasm_buffer_sleb(body, (i64)0xFFFFFFFFll);
      fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);
      fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE);
      fbx_wasm_buffer_uleb(body, 3);
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

static void EmitRegStoreFromLocal(struct FbxWasmBuffer *body, u32 greg_slot,
                                  u32 src_local, u8 width) {
  /* Push effective address (m_ptr + ctx[greg_slot]). */
  EmitPushGregBaseAddr(body, greg_slot);
  /* Push value. */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, src_local);
  /* Width-sized store (width-4 zero-extends — see EmitRegSlotStore). */
  EmitRegSlotStore(body, width);
}

/* Emit a REG_SET when the source is an immediate.
 *
 * #635: BOTH the greg-base address AND the immediate value are now hoisted.
 * Class 1 site M3 (greg_slot) + Class 2 site #1 (imm_slot). */
static void EmitRegStoreImm(struct FbxWasmBuffer *body, u32 greg_slot,
                            u32 imm_slot, u8 width) {
  /* Push effective address (m_ptr + ctx[greg_slot]). */
  EmitPushGregBaseAddr(body, greg_slot);
  /* Push immediate value from block_ctx[imm_slot]. */
  EmitLoadCtxI64(body, imm_slot);
  /* Width-sized store (width-4 zero-extends — see EmitRegSlotStore). */
  EmitRegSlotStore(body, width);
}

/* Emit a store of `block_ctx[pc_slot]` (i64) into Machine.ip.
 *
 * #635: PC value is hoisted into the block_ctx slot.  Class 2 site #2.
 * The M_OFF_IP offset remains baked — it is an intra-block invariant. */
static void EmitStoreIp(struct FbxWasmBuffer *body, u32 pc_slot) {
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, EMIT_LOCAL_M_PTR);
  EmitLoadCtxI64(body, pc_slot);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE);
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, M_OFF_IP);
}

/* Map an IR vreg index to a wasm local index.  Local 0 is m_ptr,
 * local 1 is block_ctx_ptr (#635), locals 2..nvregs+1 are the vregs. */
static u32 VregLocal(u32 vreg) { return vreg + EMIT_LOCAL_VREG_BASE; }

/* Lower an ALU op (ADD/SUB/AND/OR/XOR/IMUL) — both operands are VREG-kind.
 * #794: IMUL emits i64.mul (low operand-width product), the same vreg→vreg→
 * vreg local shape as the bitwise/additive ALU ops; flags are NOT touched
 * here (deferred — see the coverage gate's imul-flag-liveness refuse). */
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
    case FBX_IR_OP_IMUL: wasm_op = WASM_OP_I64_MUL; break;
    default: return 0;
  }
  fbx_wasm_buffer_u8(body, wasm_op);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
  fbx_wasm_buffer_uleb(body, VregLocal(p->dst));
  return 1;
}

/* Lower a LEA: `dst = base + index*scale + disp` (the index term is #778).
 *
 * #635: base greg byte-offset (Class 1, M5 src1), dst greg byte-offset
 * (Class 1, M5 dst), and the immediate (Class 2, site #3) are all hoisted
 * into the block_ctx.  Width-stamp on the store is structural (LEAVE).
 * #778 adds the index greg byte-offset (Class 1) when the SIB-index form is
 * lifted; `slots->scale` carries the SIB scale.
 *
 * LEA computes an ADDRESS and never dereferences guest memory, so there is NO
 * software-MMU translation even on the wasm32 (non-linear) build — the value
 * is plain i64 register arithmetic written into the Machine struct.
 *
 * Width: the width-4 store uses i64.store32 (writes the low 32 bits, mirroring
 * REG_SET's EmitRegStoreFromLocal width-4 path).  This matches the existing
 * REG_SET convention the spec calls for ("mask to the dest width as REG_SET
 * does") — note it does NOT zero the upper 32 bits, exactly like REG_SET; the
 * hot SIB-index LEA (`lea r14,[r9+rax]`) is REX.W (width 8 → full store).
 *
 * The `slots` argument carries the slot indices the up-front allocator
 * assigned for THIS LEA inst; the emit-pass walker passes them through. */
static int EmitLea(struct FbxWasmBuffer *body, const struct FbxIrInst *p,
                   const struct LeaSlots *slots) {
  if (p->src1_kind != FBX_IR_KIND_GREG || p->dst_kind != FBX_IR_KIND_GREG) {
    return 0;
  }
  if (p->src2_kind == FBX_IR_KIND_GREG) {
    /* Indexed form (#778): src2 is the index greg; scale must be valid. */
    if (!slots->has_index || (slots->scale != 1 && slots->scale != 2 &&
                              slots->scale != 4 && slots->scale != 8)) {
      return 0;
    }
  } else if (p->src2_kind != FBX_IR_KIND_NONE &&
             p->src2_kind != FBX_IR_KIND_IMM) {
    return 0;
  }
  /* Push effective dst address for the eventual store. */
  EmitPushGregBaseAddr(body, slots->dst_slot);
  /* Load base register value (always 8-byte for LEA in v0.1). */
  EmitPushGregBaseAddr(body, slots->base_slot);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD);
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, 0);
  /* #778 — add index*scale when the SIB-index form was lifted.  Effective
   * address so far is `base`; add `weg[index] * scale` as i64.  Scale is a
   * power of two (1/2/4/8); use a shift for 2/4/8, no-op for 1. */
  if (slots->has_index) {
    EmitPushGregBaseAddr(body, slots->index_slot);
    fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD);
    fbx_wasm_buffer_uleb(body, 3);
    fbx_wasm_buffer_uleb(body, 0);
    if (slots->scale != 1) {
      u32 shift = (slots->scale == 2) ? 1u : (slots->scale == 4) ? 2u : 3u;
      fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
      fbx_wasm_buffer_sleb(body, (i64)shift);
      fbx_wasm_buffer_u8(body, WASM_OP_I64_SHL);
    }
    fbx_wasm_buffer_u8(body, WASM_OP_I64_ADD);
  }
  /* Add immediate (only if non-zero — zero is invariant and the allocator
   * skipped reserving a slot for it). */
  if (slots->has_imm) {
    EmitLoadCtxI64(body, slots->imm_slot);
    fbx_wasm_buffer_u8(body, WASM_OP_I64_ADD);
  }
  /* Store the effective address into dst (already on stack).  width-4 LEA
   * (`lea r32,[...]`) zero-extends the 32-bit EA into the 64-bit register —
   * the same register-write semantics as REG_SET (see EmitRegSlotStore); the
   * prior i64.store32 left the upper half stale. */
  EmitRegSlotStore(body, (u8)p->width);
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

/* Get the local index of the i64 scratch_z, given nvregs.
 * #635: locals 0 (m_ptr) and 1 (block_ctx_ptr) precede the vregs; scratch
 * locals sit at index `EMIT_LOCAL_VREG_BASE + nvregs + OFFSET`. */
static u32 ScratchZLocal(u16 nvregs) {
  return EMIT_LOCAL_VREG_BASE + (u32)nvregs + EMIT_LOCAL_OFF_SCRATCH_Z;
}

/* Get the local index of the i32 scratch_flags, given nvregs. */
static u32 ScratchFlagsLocal(u16 nvregs) {
  return EMIT_LOCAL_VREG_BASE + (u32)nvregs + EMIT_LOCAL_OFF_SCRATCH_FLAGS;
}

/* #695 — self-loop locals.  Declared AFTER the two scratch locals (a self-loop
 * candidate always terminates in BRANCH_COND, which forces needs_scratch=1,
 * so scratch_z + scratch_flags are always present when these are).
 *   entry_ip   (i64): m->ip captured at function entry (== block start_pc).
 *   next_ip    (i64): the IP the BRANCH_COND selected this pass.
 *   iter_budget(i32): remaining guest-side iterations before yielding. */
#define EMIT_LOCAL_OFF_ENTRY_IP    2u /* i64 */
#define EMIT_LOCAL_OFF_NEXT_IP     3u /* i64 */
#define EMIT_LOCAL_OFF_ITER_BUDGET 4u /* i32 */

static u32 EntryIpLocal(u16 nvregs) {
  return EMIT_LOCAL_VREG_BASE + (u32)nvregs + EMIT_LOCAL_OFF_ENTRY_IP;
}
static u32 NextIpLocal(u16 nvregs) {
  return EMIT_LOCAL_VREG_BASE + (u32)nvregs + EMIT_LOCAL_OFF_NEXT_IP;
}
static u32 IterBudgetLocal(u16 nvregs) {
  return EMIT_LOCAL_VREG_BASE + (u32)nvregs + EMIT_LOCAL_OFF_ITER_BUDGET;
}

/* firebox#735 — software-MMU scratch locals (i64 mmu_va, i64 mmu_entry).
 * Declared as the LAST locals group when the block contains a memory operand
 * (BlockNeedsMmuLocals).  Their base index follows ALL conditional prior
 * locals — vregs, the scratch pair (needs_scratch), and the self-loop trio
 * (self_loop) — so EmitGuestVaToHostOffsetOrBailout can address them.  Both
 * the locals-declaration site and the use sites compute the base from the
 * same (needs_scratch, self_loop) flags (deterministic over the IR), keeping
 * the index identical at declare and use. */
#define EMIT_MMU_LOCAL_COUNT 2u
static u32 MmuLocalsBase(const struct FbxIrBlock *ir, int needs_scratch,
                         int self_loop) {
  u32 base = EMIT_LOCAL_VREG_BASE + (u32)ir->nvregs;
  if (needs_scratch) base += EMIT_NUM_SCRATCH_LOCALS; /* scratch_z + scratch_flags */
  if (self_loop) base += 3u;                          /* entry_ip,next_ip,iter_budget */
  return base;
}
static u32 MmuVaLocal(const struct FbxIrBlock *ir, int needs_scratch,
                      int self_loop) {
  return MmuLocalsBase(ir, needs_scratch, self_loop) + 0u;
}
static u32 MmuEntryLocal(const struct FbxIrBlock *ir, int needs_scratch,
                         int self_loop) {
  return MmuLocalsBase(ir, needs_scratch, self_loop) + 1u;
}

/* True iff the block contains a memory-operand op needing inline MMU
 * translation (LOAD/STORE in this increment; stack ops PUSH/POP/CALL/RET are
 * a follow-on once LOAD/STORE proves out). */
static int BlockNeedsMmuLocals(const struct FbxIrBlock *ir) {
  u32 i;
  for (i = 0; i < ir->ninsts; ++i) {
    u8 op = ir->insts[i].opcode;
    if (op == FBX_IR_OP_LOAD || op == FBX_IR_OP_STORE) return 1;
  }
  return 0;
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
 * isn't synthesisable (Jcc PF/NP at v0.1).
 *
 * #635: taken_pc + fallthrough_pc are hoisted to the block_ctx (Class 2
 * sites #13 and #14).  cond_id is intra-block invariant (encoded in IR
 * inst shape, not a constant the bridge needs to supply). */
static int EmitBranchCond(struct FbxWasmBuffer *body,
                          const struct FbxIrInst *p,
                          const struct BranchCondSlots *slots) {
  u32 cond_id = p->src1; /* low nibble of mopcode */
  /* Push m_ptr for the eventual i64.store at m->ip. */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, EMIT_LOCAL_M_PTR);
  /* Push taken_pc (operand 1 of select). */
  EmitLoadCtxI64(body, slots->taken_pc_slot);
  /* Push fallthrough_pc (operand 2 of select). */
  EmitLoadCtxI64(body, slots->fallthrough_pc_slot);
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

/* #695 — self-loop variant of EmitBranchCond.  Emitted INSIDE the `loop $top`
 * wrapper (see EmitFunctionBody).  Computes the next IP exactly as
 * EmitBranchCond does, stashes it in `next_ip`, ALWAYS stores it to m->ip
 * (so the non-loop exit path is identical to the base emit), then branches
 * back to the loop header iff the self-loop condition holds at runtime:
 *
 *     next_ip == entry_ip   AND   (--iter_budget) != 0
 *
 * Both operands of the equality are runtime values (next_ip selected this
 * pass; entry_ip captured at function entry == the dispatched block's
 * start_pc).  For a cache-shared module reused on a non-self-loop block this
 * comparison is false → br_if not taken → falls through to the same
 * `i32.const 0; return` as the base path.  CACHE-SAFE: no PC relationship is
 * baked into the bytes.
 *
 * `loop_depth` is the relative label index of the enclosing `loop` from this
 * emit point (0 = innermost).  Returns 1, or 0 on predicate refusal. */
static int EmitBranchCondSelfLoop(struct FbxWasmBuffer *body,
                                  const struct FbxIrInst *p,
                                  const struct BranchCondSlots *slots,
                                  u16 nvregs, u32 loop_depth) {
  u32 cond_id = p->src1;
  u32 entry_ip = EntryIpLocal(nvregs);
  u32 next_ip = NextIpLocal(nvregs);
  u32 iter_budget = IterBudgetLocal(nvregs);
  /* next_ip = select(taken_pc, fallthrough_pc, predicate). */
  EmitLoadCtxI64(body, slots->taken_pc_slot);
  EmitLoadCtxI64(body, slots->fallthrough_pc_slot);
  if (!EmitJccPredicate(body, cond_id)) {
    return 0;
  }
  fbx_wasm_buffer_u8(body, WASM_OP_SELECT);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
  fbx_wasm_buffer_uleb(body, next_ip);
  /* m->ip = next_ip  (always — correct for both the loop-back and exit). */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, EMIT_LOCAL_M_PTR);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, next_ip);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE);
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, M_OFF_IP);
  /* loop-back predicate part 1: (next_ip == entry_ip) as i32 (0/1). */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, next_ip);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, entry_ip);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_EQ);
  /* loop-back predicate part 2: iter_budget -= 1; push (iter_budget != 0). */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, iter_budget);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
  fbx_wasm_buffer_sleb(body, 1);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_SUB);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_TEE); /* write back, keep on stack */
  fbx_wasm_buffer_uleb(body, iter_budget);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
  fbx_wasm_buffer_sleb(body, 0);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_NE); /* (iter_budget != 0) */
  /* combine: same-target AND budget-remaining. */
  fbx_wasm_buffer_u8(body, WASM_OP_I32_AND);
  /* br_if to the enclosing loop header. */
  fbx_wasm_buffer_u8(body, WASM_OP_BR_IF);
  fbx_wasm_buffer_uleb(body, loop_depth);
  /* Fell through: not looping this pass — normal exit=0 to the host.  m->ip
   * is already set to next_ip above. */
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

/* firebox#735 — inline software-MMU translation: guest VA → host linear
 * offset, with a TLB-miss / fault BAILOUT.
 *
 * CONSUMES the guest VA (i64) currently on top of the wasm stack; LEAVES an
 * i32 wasm-linear-memory offset on the stack (the host pointer the access
 * should use).  When the inline fast path can't safely resolve the VA it
 * stores `m->ip = bailout_pc` (from ctx slot `bailout_pc_slot`) and `return
 * 1` (exit=1 → Tier 1 resumes at that PC, firebox#735 threadedcode.c).
 *
 * This mirrors blink's own resolution path for the CPL-3 user case
 * (LookupAddress → LookupAddress2 → FindPageTableEntry's TLB-hit branch →
 * GetPageAddress's PAGE_HOST branch).  It DERIVES from blink's code, it does
 * not re-implement the fault path: anything the fast path doesn't cover bails
 * to Tier 1's authoritative walk.  Conditions that BAIL OUT:
 *   - TLB miss (m->tlb[(va>>12)&31].page != (va & ~4095))
 *   - entry lacks PAGE_V|PAGE_U (+ PAGE_RW for writes)  → not present /
 *     protection fault / kernel-only page
 *   - entry is not PAGE_HOST (file-backed `s->real`-region page — rarer; the
 *     host-offset there is `s->real + (entry & PAGE_TA)`, deferred)
 *   - m->metal != 0 (CPL may be != 3; the baked `need` assumes CPL-3 user)
 *   - page-crossing access ((va & 4095) + width > 4096)
 * The Tier-1 walk on bailout FILLS the TLB (FindPageTableEntry), so the very
 * next dispatch of this funcref hits the inline fast path.
 *
 * SOUNDNESS / NO-COMMIT-BEFORE: callers MUST emit this translation BEFORE any
 * state mutation of the memory op (RSP adjust, value store), so a bailout
 * re-runs the whole x86 instruction on Tier 1 with nothing double-committed.
 *
 * `need_rw`: 1 → require PAGE_RW (a write); 0 → read-only.
 * `width`:   access width in bytes (page-cross check).
 * `mmu_va` / `mmu_entry`: i64 scratch locals (see MmuVaLocal/MmuEntryLocal).
 * `bailout_pc_slot`: ctx PC slot holding the failing op's guest PC.
 */
static void EmitGuestVaToHostOffsetOrBailout(struct FbxWasmBuffer *body,
                                             int need_rw, u8 width,
                                             u32 mmu_va, u32 mmu_entry,
                                             u32 bailout_pc_slot) {
  u64 need = M_PAGE_V | M_PAGE_U | (need_rw ? M_PAGE_RW : 0ull);

  /* Stash the VA (consume the value the caller left on the stack). */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
  fbx_wasm_buffer_uleb(body, mmu_va);

  /* mmu_entry = m->tlb[(va>>12)&31].entry.  Compute the i32 slot byte
   * address = m_ptr + M_OFF_TLB + ((va>>12)&31)*16, then i64.load +8. */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, EMIT_LOCAL_M_PTR);          /* i32 m_ptr */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, mmu_va);                    /* i64 va */
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, 12);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_SHR_U);           /* va>>12 (i64) */
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, (i64)M_TLB_KEY_MASK);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);             /* &31 (i64) */
  fbx_wasm_buffer_u8(body, WASM_OP_I32_WRAP_I64);        /* → i32 key */
  fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
  fbx_wasm_buffer_sleb(body, (i64)M_TLB_STRIDE);
  /* i32.mul (0x6C). */
  fbx_wasm_buffer_u8(body, 0x6Cu);                       /* key*16 */
  fbx_wasm_buffer_u8(body, WASM_OP_I32_ADD);             /* m_ptr + key*16 */
  fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD);            /* entry @ +ENTRY */
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, M_OFF_TLB + M_TLB_OFF_ENTRY);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
  fbx_wasm_buffer_uleb(body, mmu_entry);

  /* Build the i32 FAIL predicate (1 = bail).  Accumulate via i32.or. */
  /* (a) tlb.page != (va & ~4095). */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, EMIT_LOCAL_M_PTR);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, mmu_va);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, 12);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_SHR_U);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, (i64)M_TLB_KEY_MASK);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_WRAP_I64);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
  fbx_wasm_buffer_sleb(body, (i64)M_TLB_STRIDE);
  fbx_wasm_buffer_u8(body, 0x6Cu);                       /* key*16 */
  fbx_wasm_buffer_u8(body, WASM_OP_I32_ADD);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD);            /* tlb.page @ +PAGE */
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, M_OFF_TLB + M_TLB_OFF_PAGE);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, mmu_va);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, (i64)~(u64)4095);           /* ~4095 (sign-ext) */
  fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);             /* va & ~4095 */
  fbx_wasm_buffer_u8(body, WASM_OP_I64_NE);              /* → i32 */

  /* (b) (entry & need) != need. */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, mmu_entry);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, (i64)need);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, (i64)need);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_NE);              /* → i32 */
  fbx_wasm_buffer_u8(body, WASM_OP_I32_OR);

  /* (c) (entry & PAGE_HOST) == 0. */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, mmu_entry);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, (i64)M_PAGE_HOST);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_EQZ);             /* → i32 (1 if no HOST) */
  fbx_wasm_buffer_u8(body, WASM_OP_I32_OR);

  /* (d) m->metal != 0  (bool, 1 byte). */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, EMIT_LOCAL_M_PTR);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD8U);
  fbx_wasm_buffer_uleb(body, 0);
  fbx_wasm_buffer_uleb(body, M_OFF_METAL);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, 0);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_NE);              /* → i32 */
  fbx_wasm_buffer_u8(body, WASM_OP_I32_OR);

  /* (e) page-crossing: (va & 4095) + width > 4096. */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, mmu_va);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, 4095);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, (i64)width);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_ADD);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, 4096);
  /* i64.gt_u (0x56). */
  fbx_wasm_buffer_u8(body, 0x56u);                       /* → i32 */
  fbx_wasm_buffer_u8(body, WASM_OP_I32_OR);

  /* if (fail) { m->ip = bailout_pc; return 1 } */
  fbx_wasm_buffer_u8(body, WASM_OP_IF);
  fbx_wasm_buffer_u8(body, WASM_BLOCKTYPE_EMPTY);
  EmitStoreIp(body, bailout_pc_slot);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
  fbx_wasm_buffer_sleb(body, 1);
  fbx_wasm_buffer_u8(body, WASM_OP_RETURN);
  fbx_wasm_buffer_u8(body, WASM_OP_END);

  /* host_offset = (i32)(entry & PAGE_TA) + (i32)(va & 4095). */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, mmu_entry);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, (i64)M_PAGE_TA);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_WRAP_I64);        /* page base (i32) */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, mmu_va);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, 4095);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_AND);
  fbx_wasm_buffer_u8(body, WASM_OP_I32_WRAP_I64);        /* page offset (i32) */
  fbx_wasm_buffer_u8(body, WASM_OP_I32_ADD);             /* → i32 host offset */
}

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
/* #635: base_slot is the per-block ctx slot holding (M_OFF_WEG + greg*8).
 * disp is invariant 0 in v0.1 (all callers pass 0); when v0.2 enables
 * variable disp, the caller must reserve an IMM slot via BcAllocImm and
 * thread the slot through here.  Site #15 / M10 of abi-redesign.md §2. */
static void EmitGuestMemStoreFromLocal(struct FbxWasmBuffer *body,
                                       u32 base_slot, i64 disp,
                                       u32 src_local, u8 width) {
  /* Compute the wasm i32 address = i32.wrap_i64(m->weg[base_reg] + disp). */
  EmitPushGregBaseAddr(body, base_slot);       /* effective addr to base reg */
  fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD);  /* read base reg as i64 */
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, 0);
  if (disp != 0) {
    /* v0.1 unreachable; placeholder for v0.2 disp lighting up. */
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

/* Emit code that stores ctx[imm_slot] (i64) to guest memory at
 * (m->weg[base_reg] + disp).  Helper for CALL_DIRECT (which stores
 * return_pc).
 *
 * #635: base_slot (Class 1, M11) + imm_slot (Class 2, site #17) are
 * hoisted.  disp invariant zero in v0.1 (only caller is EmitCallDirect
 * with disp=0). */
static void EmitGuestMemStoreImm(struct FbxWasmBuffer *body, u32 base_slot,
                                 i64 disp, u32 imm_slot, u8 width) {
  EmitPushGregBaseAddr(body, base_slot);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD);
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, 0);
  if (disp != 0) {
    fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
    fbx_wasm_buffer_sleb(body, disp);
    fbx_wasm_buffer_u8(body, WASM_OP_I64_ADD);
  }
  fbx_wasm_buffer_u8(body, WASM_OP_I32_WRAP_I64);
  EmitLoadCtxI64(body, imm_slot);
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
 * stack.  Width-tagged (zero-extending for narrow widths).
 *
 * #635: base_slot (Class 1, M12) hoisted.  disp invariant zero in v0.1. */
static void EmitGuestMemLoad(struct FbxWasmBuffer *body, u32 base_slot,
                             i64 disp, u8 width) {
  EmitPushGregBaseAddr(body, base_slot);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD);
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, 0);
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
 * (x86-64 RSP is u64; modular arithmetic matches).
 *
 * #635: base_slot (Class 1, M13) hoisted.  delta is invariant literal ±8
 * (PUSH/POP/CALL/RET) in v0.1 → stays baked as i64.const per design §2.1
 * site #19.  If PUSH/POP grows variable-width support in v0.2, delta
 * needs an IMM slot. */
static void EmitGregAddImm(struct FbxWasmBuffer *body, u32 base_slot,
                           i64 delta) {
  /* Push effective dst address for the store. */
  EmitPushGregBaseAddr(body, base_slot);
  /* Re-push effective addr for the load (same slot; same value). */
  EmitPushGregBaseAddr(body, base_slot);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD);
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, 0);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_CONST);
  fbx_wasm_buffer_sleb(body, delta);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_ADD);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE);
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, 0);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* #677 — MOV r/m memory-form (FBX_IR_OP_LOAD / FBX_IR_OP_STORE).             */
/*                                                                            */
/* Lowers the `[base + disp]` addressing shape against the imported wasm      */
/* linear memory, reusing the same guest-VA-aliases-linear-memory invariant   */
/* the PUSH/POP/CALL/RET stack ops rely on (see the §602 block above:         */
/* ToHost(va) = va + kSkew, kSkew == 0 in linear-mapping mode).               */
/*                                                                            */
/* The effective guest virtual address is computed as an i64:                 */
/*     guest_reg[base]  (i64.load of the register value)                      */
/*   + (i64)disp        (only when disp != 0; supplied from a hoisted ctx     */
/*                       IMM slot — NOT baked, so two same-shape blocks with  */
/*                       different disps share one compiled module per #635)  │
* then narrowed via i32.wrap_i64 to a wasm linear-memory address.  An OOB    */
/* access traps the runtime; the bridge catches it and bails to Tier 1.       */
/*                                                                            */
/* `base_slot` is the per-block ctx slot holding (M_OFF_WEG + base_greg*8).   */
/* `disp_slot` is the ctx IMM slot holding the i64 displacement; valid only   */
/* when `has_disp` is set. */

/* Push the i32 wasm address for [base + disp] onto the wasm stack. */
/* firebox#735 — push the GUEST VIRTUAL ADDRESS (m->weg[base] + disp) as an
 * i64 on the wasm stack.  Under the software MMU this is NOT a linear offset;
 * the caller routes it through EmitGuestVaToHostOffsetOrBailout to translate.
 * (Pre-#735 this appended an `i32.wrap_i64`, treating the VA as a linear
 * offset — the #709/#719 OOB-trap root cause.) */
static void EmitMemEffectiveVa(struct FbxWasmBuffer *body, u32 base_slot,
                               int has_disp, u32 disp_slot) {
  /* Load the guest base register's value as i64 (the guest VA base). */
  EmitPushGregBaseAddr(body, base_slot);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD);
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, 0);
  if (has_disp) {
    /* + (i64)disp, from the hoisted ctx slot. */
    EmitLoadCtxI64(body, disp_slot);
    fbx_wasm_buffer_u8(body, WASM_OP_I64_ADD);
  }
}

/* Lower FBX_IR_OP_LOAD: vreg = (zero-extended) mem[base + disp].  Width-tagged
 * unsigned loads (matches x86 MOV/MOVZX load semantics).  Leaves nothing on
 * the wasm stack — the loaded value is stored into the destination vreg
 * local.
 *
 * firebox#735: the guest VA is translated through the software-MMU TLB fast
 * path (EmitGuestVaToHostOffsetOrBailout) which leaves the host linear offset
 * on the stack, OR bails to Tier 1 on a TLB miss/fault.  A LOAD commits
 * nothing before the translation (read-only into dst_local), so a bailout is
 * safe to re-run on Tier 1. */
static void EmitLoadMem(struct FbxWasmBuffer *body, u32 dst_local,
                        u32 base_slot, int has_disp, u32 disp_slot, u8 width,
                        u32 mmu_va, u32 mmu_entry, u32 bailout_pc_slot) {
  EmitMemEffectiveVa(body, base_slot, has_disp, disp_slot);
  EmitGuestVaToHostOffsetOrBailout(body, /*need_rw=*/0, width, mmu_va,
                                   mmu_entry, bailout_pc_slot);
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
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
  fbx_wasm_buffer_uleb(body, dst_local);
}

/* Lower FBX_IR_OP_STORE: mem[base + disp] = vreg (width-truncated).
 * The wasm store opcode consumes [addr, value]; we push the address first,
 * then the value local, then the width-tagged store.
 *
 * firebox#735: the guest VA is translated to a host offset (or bails) BEFORE
 * the value is pushed, so a translation-miss bailout has committed nothing for
 * this op and Tier 1 can re-run the whole instruction soundly. */
static void EmitStoreMem(struct FbxWasmBuffer *body, u32 val_local,
                         u32 base_slot, int has_disp, u32 disp_slot, u8 width,
                         u32 mmu_va, u32 mmu_entry, u32 bailout_pc_slot) {
  EmitMemEffectiveVa(body, base_slot, has_disp, disp_slot);
  EmitGuestVaToHostOffsetOrBailout(body, /*need_rw=*/1, width, mmu_va,
                                   mmu_entry, bailout_pc_slot);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, val_local);
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

/* Lower a CALL_DIRECT IR inst.
 *
 *   m->weg[RSP] -= 8;
 *   *m->weg[RSP] = return_pc;
 *   m->ip = target_pc;
 *   return 0;
 *
 * Block-terminating.  Returns 1 on success. */
static int EmitCallDirect(struct FbxWasmBuffer *body,
                          const struct FbxIrInst *p,
                          const struct CallDirectSlots *slots) {
  (void)p;
  /* Step 1: RSP -= 8. */
  EmitGregAddImm(body, slots->rsp_slot, -8);
  /* Step 2: mem[RSP] = return_pc (from block_ctx). */
  EmitGuestMemStoreImm(body, slots->rsp_slot, 0, slots->return_pc_slot, 8);
  /* Step 3: m->ip = target_pc and return exit=0. */
  EmitStoreIp(body, slots->target_pc_slot);
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
static int EmitRet(struct FbxWasmBuffer *body, u16 nvregs, u32 rsp_slot) {
  u32 scratch_z = ScratchZLocal(nvregs);
  /* Step 1: load *RSP into scratch_z. */
  EmitGuestMemLoad(body, rsp_slot, 0, 8);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
  fbx_wasm_buffer_uleb(body, scratch_z);
  /* Step 2: m->ip = scratch_z. */
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, EMIT_LOCAL_M_PTR);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
  fbx_wasm_buffer_uleb(body, scratch_z);
  fbx_wasm_buffer_u8(body, WASM_OP_I64_STORE);
  fbx_wasm_buffer_uleb(body, 3);
  fbx_wasm_buffer_uleb(body, M_OFF_IP);
  /* Step 3: RSP += 8. */
  EmitGregAddImm(body, rsp_slot, 8);
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
                    u16 nvregs, const struct PushPopSlots *slots) {
  u32 scratch_z = ScratchZLocal(nvregs);
  u8 width = p->width ? p->width : 8u;
  if (p->src1_kind != FBX_IR_KIND_GREG) return 0;
  /* Step 1: stash greg[reg_id] into scratch_z. */
  EmitRegLoad(body, slots->reg_slot, width);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
  fbx_wasm_buffer_uleb(body, scratch_z);
  /* Step 2: RSP -= width. */
  EmitGregAddImm(body, slots->rsp_slot, -(i64)width);
  /* Step 3: *RSP = scratch_z. */
  EmitGuestMemStoreFromLocal(body, slots->rsp_slot, 0, scratch_z, width);
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
                   u16 nvregs, const struct PushPopSlots *slots) {
  u32 scratch_z = ScratchZLocal(nvregs);
  u8 width = p->width ? p->width : 8u;
  if (p->dst_kind != FBX_IR_KIND_GREG) return 0;
  /* Step 1: load *RSP into scratch_z. */
  EmitGuestMemLoad(body, slots->rsp_slot, 0, width);
  fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
  fbx_wasm_buffer_uleb(body, scratch_z);
  /* Step 2: RSP += width. */
  EmitGregAddImm(body, slots->rsp_slot, (i64)width);
  /* Step 3: greg[dst] = scratch_z. */
  EmitRegStoreFromLocal(body, slots->reg_slot, scratch_z, width);
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
 *
 * firebox#735 (CORRECTNESS — load-bearing): a block that contains a BAILOUT
 * ALSO triggers flag emission, because the §735 Tier-1 handoff RESUMES at
 * m->ip (the bailout PC) instead of re-walking entries[] from index 0.  The
 * old re-walk-from-0 RE-RAN the CMP/TEST whose flags were dead-flag-elided in
 * the T2 block (their flag result is consumed CROSS-block by a later Jcc, so
 * within THIS block there is no reader → elision drops them).  With
 * resume-at-m->ip, Tier 1 SKIPS those elided ops → the flags would be LOST →
 * the next block's conditional branch reads stale flags → wrong control flow
 * → silent output corruption (witnessed: grep matched 82/7693 lines).  Forcing
 * flag emission here makes the T2 block COMMIT the flags before its BAILOUT,
 * so what Tier 1 skips on resume is exactly what T2 already applied — the
 * resume contract's invariant ("T2 commits exactly the entries Tier 1 skips").
 * (REG_GET into a vreg is block-local — vregs die at block end — so no other
 * elided op carries cross-block state; the dead-flag SET_FLAGS_RAW was the
 * only one.) */
static int BlockNeedsScratchLocals(const struct FbxIrBlock *ir) {
  u32 i;
  for (i = 0; i < ir->ninsts; ++i) {
    u8 op = ir->insts[i].opcode;
    if (op == FBX_IR_OP_BRANCH_COND || op == FBX_IR_OP_GET_FLAG) return 1;
    /* firebox#735 — a BAILOUT forces flag emission (see the block comment):
     * resume-at-m->ip skips any flag-elided CMP/TEST, so T2 must commit the
     * flags instead of eliding them. */
#ifndef FBX735_DIAG_NO_BAILOUT_FLAG_FORCE
    if (op == FBX_IR_OP_BAILOUT) return 1;
#endif
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

/* ────────────────────────────────────────────────────────────────────────── */
/* #695 — self-loop chain.                                                     */
/*                                                                            */
/* A block that terminates in BRANCH_COND is the do/while hot-loop shape.      */
/* When at runtime its taken-PC equals the block's own entry PC, it is a       */
/* self-loop: instead of returning to the host every iteration (one ~270ns     */
/* FFI crossing per pass, #693), we keep iterating guest-side inside a wasm     */
/* `loop` until either the predicate stops selecting the self target OR a       */
/* bounded iteration budget is exhausted (so the `Actor` loop can re-check      */
/* `m->attention` for signals — see chaining-design.md §3/§4.3).                */
/*                                                                            */
/* CACHE SAFETY (load-bearing, [[#633→#635]]): the self-loop decision is a      */
/* RUNTIME comparison of runtime-supplied values (next_ip == entry_ip), NOT a   */
/* compile-time bake.  taken_pc is hoisted into block_ctx (#635), so a cache-   */
/* shared module reused for a non-self-loop block computes next_ip != entry_ip  */
/* → the loop-back br_if is not taken → identical store-ip + return-0 behavior  */
/* as before.  The wrapper is structurally present for EVERY BRANCH_COND block  */
/* (decision keyed on IR shape, deterministic over the IR); it is a runtime     */
/* no-op for non-self-loops.                                                   */
/* ────────────────────────────────────────────────────────────────────────── */

/* Iterations run guest-side per FFI crossing before yielding to the host so
 * the Actor loop can service async signals / futex wakes (machine.c:2274).
 * Bounds signal-delivery latency to one budget's worth of guest work — the
 * same granularity Tier-1 imposes implicitly per block, coarsened to N. */
#define FBX_T2_SELF_LOOP_BUDGET 4096u

/* True iff the block's terminating control-flow op is BRANCH_COND (the only
 * shape the self-loop wrapper applies to in this increment).  PC_MARK insts
 * are ignored — the LAST emit-relevant op decides.  A block with any OTHER
 * terminator (BRANCH_TAKEN / CALL_DIRECT / RET / BAILOUT) is emitted exactly
 * as before — byte-identical, preserving the #635 cache + all emit tests. */
static int BlockIsSelfLoopCandidate(const struct FbxIrBlock *ir) {
  u32 i;
  u8 last = FBX_IR_OP_PC_MARK;
  for (i = 0; i < ir->ninsts; ++i) {
    u8 op = ir->insts[i].opcode;
    if (op == FBX_IR_OP_PC_MARK) continue;
    last = op;
  }
  return last == FBX_IR_OP_BRANCH_COND;
}

/* Emit the function body for one IR block.  Returns 1 on success, 0 on
 * unsupported op encountered mid-emission (caller frees scratch).
 *
 * #635: the emit-pass walks the IR allocating per-block-ctx slots in the
 * same order the up-front consts-table builder uses.  Because both walkers
 * dispatch through `AllocSlotsForInst` with identical inputs, they assign
 * the same absolute slot indices — guaranteeing the bridge-side consts[]
 * table aligns with the wasm offsets the body emits. */
static int EmitFunctionBody(struct FbxWasmBuffer *body,
                            const struct FbxIrBlock *ir,
                            struct FbxEmitFailCtx *fail) {
  u32 i;
  int terminated = 0;
  int needs_scratch = BlockNeedsScratchLocals(ir);
  /* #695 — self-loop wrapper.  A BRANCH_COND-terminated block (do/while shape)
   * always has needs_scratch==1, so the self-loop locals can be declared after
   * the scratch pair as a third locals group. */
#ifdef FBX735_DIAG_NO_SELF_LOOP
  /* firebox#738 Bug-A localization: emit BRANCH_COND blocks WITHOUT the #695
   * self-loop wrapper (each iteration returns to the host).  Isolates whether
   * the corruptor lives in the self-loop control flow (entry_ip/resume) or in
   * the per-iteration flag/branch emit. */
  int self_loop = 0;
#else
  int self_loop = BlockIsSelfLoopCandidate(ir);
#endif
  /* firebox#735 — does the block need the 2×i64 software-MMU scratch group?
   * Declared LAST (after vregs/scratch/self-loop) so existing local indices
   * are unchanged; MmuVaLocal/MmuEntryLocal compute the matching base. */
  int needs_mmu = BlockNeedsMmuLocals(ir);
  struct BlockCtxBuilder ctx_b;
  BcInit(&ctx_b, NULL, 0); /* emit-time re-walk: just allocate, don't store */
  /* Locals declaration.  When the block needs scratch locals (#599 path),
   * emit two groups: vregs as i64, then a pair (i64 scratch_z, i32
   * scratch_flags).  Wasm encodes per-group as (count, valtype); separate
   * groups are required because the types differ.
   *
   * #695: a self-loop block adds two more groups after the scratch pair:
   *   (2 × i64) entry_ip + next_ip, then (1 × i32) iter_budget.
   * #735: a memory-operand block adds one final group: (2 × i64) mmu_va +
   *   mmu_entry. */
  if (needs_scratch) {
    /* Group 1: nvregs i64s + 1 more i64 for scratch_z.  Combining them
     * into one group is byte-cheaper than two separate i64 groups. */
    fbx_wasm_buffer_uleb(body,
                         (u64)(2u + (self_loop ? 2u : 0u) + (needs_mmu ? 1u : 0u)));
    fbx_wasm_buffer_uleb(body, (u64)ir->nvregs + 1u);
    fbx_wasm_buffer_u8(body, WASM_VALTYPE_I64);
    /* Group 2: 1 i32 for scratch_flags. */
    fbx_wasm_buffer_uleb(body, 1);
    fbx_wasm_buffer_u8(body, WASM_VALTYPE_I32);
    if (self_loop) {
      /* Group 3: 2 i64 (entry_ip, next_ip). */
      fbx_wasm_buffer_uleb(body, 2);
      fbx_wasm_buffer_u8(body, WASM_VALTYPE_I64);
      /* Group 4: 1 i32 (iter_budget). */
      fbx_wasm_buffer_uleb(body, 1);
      fbx_wasm_buffer_u8(body, WASM_VALTYPE_I32);
    }
    if (needs_mmu) {
      /* Final group: 2 i64 (mmu_va, mmu_entry). */
      fbx_wasm_buffer_uleb(body, EMIT_MMU_LOCAL_COUNT);
      fbx_wasm_buffer_u8(body, WASM_VALTYPE_I64);
    }
  } else if (ir->nvregs > 0 || needs_mmu) {
    /* Group count = (vreg group if nvregs>0) + (MMU group if needs_mmu).
     * Both are i64; when nvregs>0 AND needs_mmu we keep them as two SEPARATE
     * groups (not folded) so the MMU group's local base is exactly
     * EMIT_LOCAL_VREG_BASE + nvregs (MmuLocalsBase), matching the use sites. */
    fbx_wasm_buffer_uleb(body,
                         (u64)((ir->nvregs > 0 ? 1u : 0u) + (needs_mmu ? 1u : 0u)));
    if (ir->nvregs > 0) {
      fbx_wasm_buffer_uleb(body, ir->nvregs);
      fbx_wasm_buffer_u8(body, WASM_VALTYPE_I64);
    }
    if (needs_mmu) {
      fbx_wasm_buffer_uleb(body, EMIT_MMU_LOCAL_COUNT);
      fbx_wasm_buffer_u8(body, WASM_VALTYPE_I64);
    }
  } else {
    fbx_wasm_buffer_uleb(body, 0);
  }
  /* #695 — self-loop entry prelude + loop opener.  Capture m->ip (== this
   * block's start_pc, by the dispatcher's lookup contract) into entry_ip,
   * seed the iteration budget, then open a `loop` whose body is the block.
   * The BRANCH_COND terminator re-enters via `br_if 0` while it keeps
   * selecting the self target and the budget holds (EmitBranchCondSelfLoop). */
  if (self_loop) {
    u32 entry_ip = EntryIpLocal(ir->nvregs);
    u32 iter_budget = IterBudgetLocal(ir->nvregs);
    /* entry_ip = m->ip. */
    fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_GET);
    fbx_wasm_buffer_uleb(body, EMIT_LOCAL_M_PTR);
    fbx_wasm_buffer_u8(body, WASM_OP_I64_LOAD);
    fbx_wasm_buffer_uleb(body, 3);
    fbx_wasm_buffer_uleb(body, M_OFF_IP);
    fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
    fbx_wasm_buffer_uleb(body, entry_ip);
    /* iter_budget = FBX_T2_SELF_LOOP_BUDGET. */
    fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
    fbx_wasm_buffer_sleb(body, (i64)(u32)FBX_T2_SELF_LOOP_BUDGET);
    fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
    fbx_wasm_buffer_uleb(body, iter_budget);
    /* loop $top (empty block-type — the body carries values via locals). */
    fbx_wasm_buffer_u8(body, WASM_OP_LOOP);
    fbx_wasm_buffer_u8(body, WASM_BLOCKTYPE_EMPTY);
  }
  for (i = 0; i < ir->ninsts; ++i) {
    const struct FbxIrInst *p = &ir->insts[i];
    switch (p->opcode) {
      case FBX_IR_OP_PC_MARK:
        /* No code emitted at v0.1.  Future: emit a custom section entry or
         * a debug intrinsic.  firebox#735: track the current x86 PC so the
         * LOAD/STORE bailout-PC slot (allocated below via AllocLoadStoreSlots)
         * carries this op's PC — matching the consts-build walk, which
         * updates cur_pc on PC_MARK in AllocSlotsForInst in the same order. */
        ctx_b.cur_pc = p->imm;
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
          SetFail(fail, FBX_IR_EMIT_SET_FLAGS_RAW_BAD, p->opcode);
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
      case FBX_IR_OP_BRANCH_COND: {
        struct BranchCondSlots s;
        int ok;
        if (!AllocBranchCondSlots(&ctx_b, p, &s)) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        /* #695: inside the self-loop wrapper the terminator re-enters the
         * `loop` (depth 0) while it keeps selecting the self target; otherwise
         * it exits to the host exactly as the base EmitBranchCond does. */
        if (self_loop) {
          ok = EmitBranchCondSelfLoop(body, p, &s, ir->nvregs, /*loop_depth=*/0);
        } else {
          ok = EmitBranchCond(body, p, &s);
        }
        if (!ok) {
          /* Sole failure path is EmitJccPredicate refusal (Jcc PF/NP and
           * unrecognised predicates).  Other shape errors are caught by the
           * coverage gate above. */
          SetFail(fail, FBX_IR_EMIT_JCC_PREDICATE_DEFERRED, p->opcode);
          return 0;
        }
        terminated = 1;
        break;
      }
      case FBX_IR_OP_REG_GET: {
        u32 greg_slot;
        /* dst is a vreg local; src1 is a guest register OR (#794) an
         * immediate to materialize into the vreg. */
        if (p->dst_kind != FBX_IR_KIND_VREG) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        if (p->src1_kind == FBX_IR_KIND_IMM) {
          /* #794 — load the per-block IMM-slot value into the vreg.  The
           * immediate is hoisted into block_ctx (allocator order matches
           * AllocSlotsForInst's REG_GET-IMM branch) and read via
           * EmitLoadCtxI64; baking an inline i64.const would make the wasm
           * value-dependent (breaks #635 byte-stability). */
          u32 imm_slot = AllocRegGetImmSlot(&ctx_b, p);
          if (ctx_b.overflow) {
            SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
            return 0;
          }
          EmitLoadCtxI64(body, imm_slot);
          fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
          fbx_wasm_buffer_uleb(body, VregLocal(p->dst));
          break;
        }
        if (p->src1_kind != FBX_IR_KIND_GREG) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        greg_slot = AllocRegGetSlot(&ctx_b, p);
        if (ctx_b.overflow) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        EmitRegLoad(body, greg_slot, p->width ? p->width : 8u);
        fbx_wasm_buffer_u8(body, WASM_OP_LOCAL_SET);
        fbx_wasm_buffer_uleb(body, VregLocal(p->dst));
        break;
      }
      case FBX_IR_OP_REG_SET: {
        if (p->dst_kind != FBX_IR_KIND_GREG) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        if (p->src1_kind == FBX_IR_KIND_VREG) {
          /* allocator order MUST match AllocSlotsForInst: imm slot first   */
          /* (skipped here — no imm), then greg slot. */
          u32 greg_slot = AllocRegSetGregSlot(&ctx_b, p);
          if (ctx_b.overflow) {
            SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
            return 0;
          }
          EmitRegStoreFromLocal(body, greg_slot, VregLocal(p->src1),
                                p->width ? p->width : 8u);
        } else if (p->src1_kind == FBX_IR_KIND_IMM) {
          u32 imm_slot = AllocRegSetImmSlot(&ctx_b, p);
          u32 greg_slot = AllocRegSetGregSlot(&ctx_b, p);
          if (ctx_b.overflow) {
            SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
            return 0;
          }
          EmitRegStoreImm(body, greg_slot, imm_slot,
                          p->width ? p->width : 8u);
        } else {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        break;
      }
      case FBX_IR_OP_LOAD: {
        /* #677 — vreg = mem[base+disp].  dst is a vreg local; src1 is the
         * base guest register; imm is the (sign-extended) displacement. */
        struct MemAddrSlots s;
        if (p->dst_kind != FBX_IR_KIND_VREG ||
            p->src1_kind != FBX_IR_KIND_GREG) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        if (!AllocLoadStoreSlots(&ctx_b, p, &s)) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        EmitLoadMem(body, VregLocal(p->dst), s.base_slot, s.has_disp,
                    s.disp_slot, p->width ? p->width : 8u,
                    MmuVaLocal(ir, needs_scratch, self_loop),
                    MmuEntryLocal(ir, needs_scratch, self_loop),
                    s.bailout_pc_slot);
        break;
      }
      case FBX_IR_OP_STORE: {
        /* #677 — mem[base+disp] = vreg.  src1 is the base guest register;
         * src2 is the value vreg; imm is the (sign-extended) displacement. */
        struct MemAddrSlots s;
        if (p->src1_kind != FBX_IR_KIND_GREG ||
            p->src2_kind != FBX_IR_KIND_VREG) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        if (!AllocLoadStoreSlots(&ctx_b, p, &s)) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        EmitStoreMem(body, VregLocal(p->src2), s.base_slot, s.has_disp,
                     s.disp_slot, p->width ? p->width : 8u,
                     MmuVaLocal(ir, needs_scratch, self_loop),
                     MmuEntryLocal(ir, needs_scratch, self_loop),
                     s.bailout_pc_slot);
        break;
      }
      case FBX_IR_OP_ADD:
      case FBX_IR_OP_SUB:
      case FBX_IR_OP_AND:
      case FBX_IR_OP_OR:
      case FBX_IR_OP_XOR:
      case FBX_IR_OP_IMUL:  /* #794 — i64.mul, no flags (see EmitAlu) */
        if (!EmitAlu(body, p)) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        break;
      case FBX_IR_OP_LEA: {
        struct LeaSlots s;
        if (!AllocLeaSlots(&ctx_b, p, &s)) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        if (!EmitLea(body, p, &s)) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        break;
      }
      case FBX_IR_OP_BRANCH_TAKEN: {
        /* Update m->ip (from ctx PC slot) and exit normally.  Terminator. */
        u32 pc_slot = AllocBranchTakenSlot(&ctx_b, p);
        if (ctx_b.overflow) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        EmitStoreIp(body, pc_slot);
        fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
        fbx_wasm_buffer_sleb(body, 0);
        fbx_wasm_buffer_u8(body, WASM_OP_RETURN);
        terminated = 1;
        break;
      }
      case FBX_IR_OP_BAILOUT: {
        /* Update m->ip to the bailout PC (ctx slot) and return exit=1. */
        u32 pc_slot = AllocBailoutSlot(&ctx_b, p);
        if (ctx_b.overflow) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        EmitStoreIp(body, pc_slot);
        fbx_wasm_buffer_u8(body, WASM_OP_I32_CONST);
        fbx_wasm_buffer_sleb(body, 1);
        fbx_wasm_buffer_u8(body, WASM_OP_RETURN);
        terminated = 1;
        break;
      }
      case FBX_IR_OP_CALL_DIRECT: {
        struct CallDirectSlots s;
        if (!AllocCallDirectSlots(&ctx_b, p, &s)) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        /* #602 — push return_pc onto guest stack; jump to target_pc. */
        if (!EmitCallDirect(body, p, &s)) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        terminated = 1;
        break;
      }
      case FBX_IR_OP_RET: {
        u32 rsp_slot = AllocRetRspSlot(&ctx_b);
        if (ctx_b.overflow) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        /* #602 — pop guest stack into m->ip; bump RSP. */
        if (!EmitRet(body, ir->nvregs, rsp_slot)) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        terminated = 1;
        break;
      }
      case FBX_IR_OP_PUSH: {
        struct PushPopSlots s;
        if (!AllocPushSlots(&ctx_b, p, &s)) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        /* #602 — stash greg, decrement RSP, store. */
        if (!EmitPush(body, p, ir->nvregs, &s)) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        break;
      }
      case FBX_IR_OP_POP: {
        struct PushPopSlots s;
        if (!AllocPopSlots(&ctx_b, p, &s)) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        /* #602 — load from RSP, increment RSP, write greg. */
        if (!EmitPop(body, p, ir->nvregs, &s)) {
          SetFail(fail, FBX_IR_EMIT_KIND_MISMATCH, p->opcode);
          return 0;
        }
        break;
      }
      default:
        /* Coverage gate should have rejected this. */
        SetFail(fail, FBX_IR_EMIT_UNSUPPORTED_OPCODE, p->opcode);
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
  /* #695 — close the self-loop `loop` (its EmitBranchCondSelfLoop terminator
   * always `return`s on both the loop-back and exit paths, so control never
   * falls off the bottom of the loop dynamically).  But the `loop` has an
   * empty block-type, so after its `end` the value stack is [] while the
   * function result type is [i32].  Emit `unreachable` to satisfy the
   * stack-type validator for this statically-unreachable tail. */
  if (self_loop) {
    fbx_wasm_buffer_u8(body, WASM_OP_END);         /* end loop */
    fbx_wasm_buffer_u8(body, WASM_OP_UNREACHABLE); /* statically unreachable */
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
  /* type 0: (i32, i32) -> (i32).  #635 ABI: param 0 = m_ptr, param 1 =
   * block_ctx_ptr.  Coincides with type 1 by signature shape; emit as a
   * distinct entry to keep WASM_TYPE_IDX_BLOCK == 0 stable. */
  fbx_wasm_buffer_u8(&body, WASM_TYPE_FUNC);
  fbx_wasm_buffer_uleb(&body, 2);
  fbx_wasm_buffer_u8(&body, WASM_VALTYPE_I32);
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
                           const struct FbxIrBlock *ir,
                           struct FbxEmitFailCtx *fail) {
  struct FbxWasmBuffer body;
  struct FbxWasmBuffer fn_body;
  fbx_wasm_buffer_init(&body);
  fbx_wasm_buffer_init(&fn_body);
  if (!EmitFunctionBody(&fn_body, ir, fail)) {
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
/* Public entry points.                                                       */
/* ────────────────────────────────────────────────────────────────────────── */

int fbx_ir_emit_wasm_with_reason(const struct FbxIrBlock *ir,
                                 const struct FbxTcBlock *tc,
                                 struct FbxWasmBuffer *out,
                                 enum FbxIrEmitFailReason *out_reason,
                                 u8 *out_opcode) {
  struct FbxEmitFailCtx fail;
  fail.reason = FBX_IR_EMIT_OK;
  fail.opcode = 0;
  if (!ir || !tc || !out) {
    fail.reason = FBX_IR_EMIT_EMPTY_IR;
    goto emit_fail;
  }
  if (!CoverageGate(ir, tc, &fail)) goto emit_fail;
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
  if (!EmitCodeSection(out, ir, &fail)) {
    fbx_wasm_buffer_free(out);
    goto emit_fail;
  }
  if (out->oom) {
    fbx_wasm_buffer_free(out);
    fail.reason = FBX_IR_EMIT_BUFFER_OOM;
    goto emit_fail;
  }
  if (out_reason) *out_reason = FBX_IR_EMIT_OK;
  if (out_opcode) *out_opcode = 0;
  return 1;
emit_fail:
  if (out_reason) *out_reason = fail.reason;
  if (out_opcode) *out_opcode = fail.opcode;
  return 0;
}

int fbx_ir_emit_wasm(const struct FbxIrBlock *ir, const struct FbxTcBlock *tc,
                     struct FbxWasmBuffer *out) {
  return fbx_ir_emit_wasm_with_reason(ir, tc, out, NULL, NULL);
}
