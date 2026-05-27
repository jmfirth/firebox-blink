/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2026 Justin Firth (Firebox)                                        │
│                                                                              │
│ Phase 2 — Tier 2 §13.2 ABI redesign (#635): per-block runtime constants.    │
│                                                                              │
│ Why this header exists                                                       │
│ ──────────────────────                                                       │
│ Before #635, the wasm synthesis pass baked per-block constants (guest PCs,  │
│ x86 instruction immediates, guest-register byte offsets) directly into the  │
│ emitted module as `i64.const <pc>` and `memarg offset=<u32>` ULEB128         │
│ literals.  Two structurally-identical-but-constant-different blocks         │
│ produced byte-different wasm modules → byte-different `ir_sha` keys → the  │
│ bridge-side compiled-Module cache missed (#624 measured 5.9-10.9% hit       │
│ ratio when ≥50%/≥90% was needed).                                          │
│                                                                              │
│ #635's Phase 1 design (`tests/tier-2-perf-baseline/abi-redesign.md`)       │
│ hoists 18 emit sites OUT of the wasm bytes and supplies them at dispatch   │
│ time via this per-block context.  The wasm bytes are now constant-free at   │
│ the hoisted sites; two same-shape blocks emit byte-identical wasm even      │
│ when their PCs / register selectors / immediates differ.                    │
│                                                                              │
│ Wire shape                                                                   │
│ ──────────                                                                   │
│ The block_ctx lives in guest linear memory.  Blink's `Fbxt2TryEscalate`     │
│ builds a `u64 consts[]` table by walking the IR with the same encounter-    │
│ order policy the emit pass uses (see fbx_ir_emit_wasm.c                      │
│ `AllocateBlockCtx`).  The bridge (`crates/firebox-wasix/src/t2_bridge.rs`)  │
│ copies that table into a per-block ctx slot at instantiation time and       │
│ passes the slot's guest-memory offset as the second argument of             │
│ `translated_block(m_ptr: i32, block_ctx: i32) -> i32`.                      │
│                                                                              │
│ The emit pass references each per-block constant via                        │
│ `local.get $block_ctx; <load> offset=<CTX_OFF>`, where the offset matches   │
│ the byte position of that slot within `struct FbxT2BlockCtx`.  Slot         │
│ indices are partitioned by const class (PC / immediate / greg-offset) so    │
│ blocks needing few PCs but many gregs (or vice versa) don't collide.        │
│                                                                              │
│ ABI versioning                                                               │
│ ──────────────                                                               │
│ `FBX_T2_BLOCK_CTX_VERSION` is hashed by the bridge as a prefix byte to      │
│ `ir_sha` (spec §6.5).  Old-ABI cache entries (no prefix or older version)   │
│ produce a different key → are silently invalidated on next compile.  No     │
│ migration code needed.                                                       │
│                                                                              │
│ Cross-target discipline (AGENTS.md invariant 4)                              │
│ ───────────────────────────────────────────────                              │
│ This header is identical on native + wasm builds.  The struct layout is     │
│ deterministic across host platforms because all fields are fixed-width      │
│ integers and we don't rely on host-endian behaviour at the wasm boundary    │
│ (wasm memory is little-endian by definition; the host writes consts[] as    │
│ explicit u64 little-endian via the wasm linear-memory ABI).                 │
│                                                                              │
│ See also:                                                                    │
│   - `tests/tier-2-perf-baseline/abi-redesign.md` (structural authority)    │
│   - `blink/fbx_ir_emit_wasm.c` (emit-pass implementation; `AllocateBlockCtx`,│
│     hoisted Class 1 / Class 2 emit sites)                                   │
│   - `crates/firebox-wasix/src/t2_bridge.rs` (bridge consumer; cache-key    │
│     prefix bump, block_ctx arena allocator, dual-arg call site)             │
╚─────────────────────────────────────────────────────────────────────────────*/

#ifndef BLINK_FBX_T2_BLOCK_CTX_H_
#define BLINK_FBX_T2_BLOCK_CTX_H_

#include <stddef.h>

#include "blink/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ────────────────────────────────────────────────────────────────────────── */
/* Version byte hashed as a prefix to wasm_bytes when computing `ir_sha` on  */
/* the bridge side (spec §6.5).  Bump when the layout, the partitioning     */
/* policy, or the const-index allocator's encounter-order changes.  Any     */
/* bump invalidates all cached compiled modules from the prior ABI by       */
/* construction; no migration code needed.                                    */
/* ────────────────────────────────────────────────────────────────────────── */

#define FBX_T2_BLOCK_CTX_VERSION 1u

/* ────────────────────────────────────────────────────────────────────────── */
/* Per-block constant capacity.                                                */
/*                                                                            */
/* The design doc §6.2 layout partitions 16 slots:                            */
/*   indices 0..5  → per-block PCs   (taken / fallthrough / return / branch) */
/*   indices 6..9  → per-block instruction immediates (REG_SET imm, LEA disp,*/
/*                  GuestMem disp)                                            */
/*   indices 10..15→ per-block guest-register byte-offsets                    */
/*                  (M_OFF_WEG + greg_id*8u), one per UNIQUE greg the block  │
*                   touches.                                                  │
*                                                                            │
* If any of the three sub-ranges overflows, the emit pass returns 0 and the   │
* block stays on Tier 1 — preserving the "correct OR refuse" contract.       │
* H3 of the design doc (#635 §8) is the falsifier: real workloads exceeding │
* 16 slots trigger a version bump + cache invalidation.                      │
*/

#define FBX_T2_BLOCK_CTX_MAX_CONSTS 16u

#define FBX_T2_CTX_PC_BASE          0u
#define FBX_T2_CTX_PC_COUNT         6u
#define FBX_T2_CTX_IMM_BASE         6u
#define FBX_T2_CTX_IMM_COUNT        4u
#define FBX_T2_CTX_GREG_BASE        10u
#define FBX_T2_CTX_GREG_COUNT       6u

/* ────────────────────────────────────────────────────────────────────────── */
/* The block-context struct.                                                  */
/*                                                                            */
/* Laid out as a header (version + nconsts) plus a fixed-size u64 array.  All */
/* fields are fixed-width integers; the bridge allocates it inside guest      */
/* linear memory and passes its byte-offset to `translated_block`.            */
/*                                                                            */
/* The emit pass loads:                                                        */
/*   - PCs (Class 2) via `i64.load offset=<BLOCK_CTX_CONST_OFF(idx)>`         */
/*   - immediates (Class 2 cont.) likewise (i64 load)                          │
*   - guest-register offsets (Class 1) via                                    │
*     `i32.load offset=<BLOCK_CTX_CONST_OFF(idx)>` — the slot is u64 but      │
*     the low 32 bits hold the M_OFF_WEG+greg*8 offset (which fits in u32).   │
*     Wasm linear memory is little-endian so the i32.load reads the lower     │
*     half of the u64 slot.                                                    │
*/

struct FbxT2BlockCtx {
  u32 version;  /* = FBX_T2_BLOCK_CTX_VERSION at allocate time */
  u32 nconsts;  /* number of valid entries in consts[] (≤ FBX_T2_BLOCK_CTX_MAX_CONSTS) */
  u64 consts[FBX_T2_BLOCK_CTX_MAX_CONSTS];
};

/* Byte offsets within FbxT2BlockCtx used by the emit pass and the bridge.    */
/* Computed at C compile time via offsetof so the layout stays                */
/* compiler-stable.  Wasm bytes reference these offsets via `memarg offset`.  */
/* The struct is plain-old-data with no implicit padding (4-byte version +    */
/* 4-byte nconsts + 16 × 8-byte consts) but we use offsetof regardless to     */
/* avoid hidden-padding bugs if the layout ever gains a field.                */

#define FBX_T2_CTX_OFF_VERSION  ((u32)offsetof(struct FbxT2BlockCtx, version))
#define FBX_T2_CTX_OFF_NCONSTS  ((u32)offsetof(struct FbxT2BlockCtx, nconsts))
#define FBX_T2_CTX_OFF_CONSTS   ((u32)offsetof(struct FbxT2BlockCtx, consts))

/* Byte offset of consts[idx] within the FbxT2BlockCtx.  Used as the          */
/* `memarg offset=` immediate in i32/i64 loads inside the emit pass.          */
#define FBX_T2_CTX_OFF_CONST(idx) \
  ((u32)(FBX_T2_CTX_OFF_CONSTS + (u32)(idx) * 8u))

#ifdef __cplusplus
}
#endif

#endif /* BLINK_FBX_T2_BLOCK_CTX_H_ */
