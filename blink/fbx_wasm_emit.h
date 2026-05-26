#ifndef BLINK_FBX_WASM_EMIT_H_
#define BLINK_FBX_WASM_EMIT_H_

/*
 * Firebox ELF-perf Phase 2 — Tier 2 §13.2: wasm synthesis support.
 *
 * This file defines:
 *
 *   1. FbxWasmBuffer — a tiny growable byte buffer that the synthesis
 *      pass writes wasm bytes into.  Owned by the caller; freed via
 *      fbx_wasm_buffer_free().
 *
 *   2. The public synthesis entry point `fbx_ir_emit_wasm()` —
 *      consumes a (FbxIrBlock, FbxTcBlock) pair and writes a
 *      self-contained wasm module to the buffer.
 *
 * The synthesis pass is PURE (no globals, no clocks, no TLS) so that
 * Phase 4's offline emitter calls the same function path and produces
 * byte-identical output for the same input.  Spec §5.5 + Decision 7
 * (binding axiom).
 *
 * Why a separate header from fbx_ir.h: this is consumer-only surface;
 * the IR header stands alone so a Phase 4 offline walker can include
 * `fbx_ir.h` without dragging in the synthesis pass.  Cross-reference
 * spec §3, §8.
 *
 * Browser-target discipline (AGENTS.md invariant 4): the emitted wasm
 * uses only the MVP feature set (i32/i64 numeric ops, locals, control
 * flow, imports, exports, function table-less direct calls).  No
 * wasmer-specific intrinsics.  Native + browser MUST both compile it.
 */

#include <stddef.h>

#include "blink/types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct FbxIrBlock;
struct FbxTcBlock;

/* ────────────────────────────────────────────────────────────────────────── */
/* FbxWasmBuffer — append-only byte buffer.                                   */
/*                                                                            */
/* Grows geometrically.  `data` is owned by the buffer; caller frees via      */
/* fbx_wasm_buffer_free().  After fbx_ir_emit_wasm() succeeds, `data` points  */
/* at `len` bytes of a complete wasm module ready to feed to wasmer or the    */
/* JS WebAssembly.compile() API.                                              */
/* ────────────────────────────────────────────────────────────────────────── */

struct FbxWasmBuffer {
  u8 *data;
  size_t len;
  size_t cap;
  int oom; /* sticky; once set, all further appends are no-ops */
};

/* Initialise an empty buffer.  Zero-init via struct literal also works. */
void fbx_wasm_buffer_init(struct FbxWasmBuffer *b);

/* Release the buffer's storage.  Safe to call on a zero-initialised buffer. */
void fbx_wasm_buffer_free(struct FbxWasmBuffer *b);

/* Append a single byte. */
void fbx_wasm_buffer_u8(struct FbxWasmBuffer *b, u8 v);

/* Append raw bytes. */
void fbx_wasm_buffer_bytes(struct FbxWasmBuffer *b, const void *p, size_t n);

/* Append a wasm u32 in unsigned LEB128 (1-5 bytes). */
void fbx_wasm_buffer_uleb(struct FbxWasmBuffer *b, u64 v);

/* Append a wasm i32/i64 in signed LEB128 (1-5/1-10 bytes). */
void fbx_wasm_buffer_sleb(struct FbxWasmBuffer *b, i64 v);

/* Append a little-endian u32 (used for the wasm magic + version). */
void fbx_wasm_buffer_u32_le(struct FbxWasmBuffer *b, u32 v);

/* ────────────────────────────────────────────────────────────────────────── */
/* Synthesis entry point.                                                     */
/*                                                                            */
/* Lowers `ir` into a complete wasm module, written to `*out`.  The module    */
/* contains exactly one exported function `translated_block(i32) -> i32`      */
/* which executes the block.                                                  */
/*                                                                            */
/* Calling convention:                                                        */
/*   param 0 (i32): m_ptr — pointer to struct Machine in linear memory.      */
/*   return  (i32): exit reason                                              */
/*       0 = normal completion (block ran to fallthrough)                    */
/*       1 = bailout (m->ip already updated to the bailout PC)               */
/*       2 = host-call escape (m->ip updated; outer dispatcher resumes)      */
/*                                                                            */
/* `tc` MUST be the FbxTcBlock that `ir` was lifted from — the synthesis      */
/* pass reads tc->entries[] in parallel to resolve memory-modrm + LEA-SIB    */
/* cases whose IR is intentionally approximate at §13.1 (see disclaimer 6    */
/* on the #583 README).  When the parallel TC entry shows an unsupported     */
/* operand shape for v0.1 coverage, the synthesis pass emits a bailout       */
/* sequence at that PC instead of an incorrect translation.                  */
/*                                                                            */
/* Imports declared by the emitted module (spec §5.4 + §11.2):                */
/*   "fbx" "memory"            — linear memory holding the Machine struct     */
/*   "fbx" "call_thunk"        — (i32 m_ptr, i32 thunk_idx) -> i32           */
/*   "fbx" "call_syscall"      — (i32 m_ptr, i32 syscall_no) -> i32          */
/*   "fbx" "resolve_indirect"  — (i32 m_ptr, i64 target_pc) -> i32           */
/*                                                                            */
/* The exact import set (count + signatures) is FIXED at v0.1 — adding an    */
/* import requires invalidating any cached translations because import       */
/* indices are baked into the wasm bytes (spec §5.4).                        */
/*                                                                            */
/* Returns 1 on success, 0 on bailout (the IR contains an unsupported        */
/* pattern that the v0.1 emitter refuses; the caller must keep this block    */
/* on Tier 1 dispatch).  On bailout, *out is left untouched (zero bytes).    */
/*                                                                            */
/* Purity contract (LOAD-BEARING): no globals, no clocks, no TLS, no         */
/* random sources.  Two calls with the same (ir, tc) MUST produce            */
/* byte-identical wasm bytes — required by spec §Q5 (cache-key stability)    */
/* and Phase 4 reachability (Decision 7 + reachability constraint doc).      */
/* ────────────────────────────────────────────────────────────────────────── */

int fbx_ir_emit_wasm(const struct FbxIrBlock *ir, const struct FbxTcBlock *tc,
                     struct FbxWasmBuffer *out);

/* Diagnostic: which host imports does a synthesised module declare?  Used by
 * tests + future cache-invariant checks.  Returns the count (currently 4);
 * each name is a stable string identifier matching the import declaration. */
size_t fbx_wasm_host_import_count(void);
const char *fbx_wasm_host_import_name(size_t i);

#ifdef __cplusplus
}
#endif

#endif /* BLINK_FBX_WASM_EMIT_H_ */
