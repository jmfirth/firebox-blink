#ifndef BLINK_THUNKS_H_
#define BLINK_THUNKS_H_

/*
 * Firebox ELF-perf Phase 1 + Phase 2 batch-2 — libc thunk routing.
 *
 * At ELF load time, scan the binary's symbol table; when a known-hot libc
 * primitive is found we record its entry PC.  Then, on every interpreted
 * instruction, ExecuteInstruction front-checks the current %rip against
 * those PCs; if it matches, we run a wasm-native trampoline that performs
 * the operation in C against the host page and emulates `ret` (pop guest
 * return address, branch to it).  This replaces "interpret hundreds of
 * x86 instructions per call" with "1 host C call" for these primitives.
 *
 * Phase 2 Tier 1 update: thunks now compose into the threaded-code
 * dispatcher at COMPILE TIME — a matched PC produces a single
 * FBX_TC_KIND_THUNK block entry instead of a per-instruction check; the
 * dispatch tax is fully amortised against the block lookup that already
 * happens.  See blink/threadedcode.c::LookupThunkAt + CompileBlock.
 *
 * Currently routed primitives:
 *   Phase 1:         memcpy memset strlen memcmp strcmp
 *   Phase 2 batch 2: memchr strchr strncmp strcpy strncpy
 *   Phase 2 batch 3: strnlen strcasecmp strncasecmp strstr memmove
 *
 * Coverage caveat: the symbol-table path requires .symtab or .dynsym to
 * be present.  Stripped statically-linked binaries (Alpine's musl-static
 * busybox / jq, the dominant bench-corpus shape) fall through to the
 * Phase 1.4 + Phase 1.5 + Phase 2 batch-3 fingerprint path.  Fingerprint
 * coverage:
 *   Phase 1.4 (firebox-elf-v2-phase1.4-fingerprinting):
 *     memcpy / memset / strlen
 *   Phase 1.5 (firebox-elf-v2-phase1.5-fingerprint-expansion-batch-2):
 *     memchr / strchr / strncmp / strcpy / strncpy
 *   Phase 2 batch-3 (firebox-elf-v2-phase2-batch-3-thunks):
 *     strnlen / strcasecmp / strncasecmp / strstr / memmove
 * memcmp/strcmp remain symbol-only (per-build register-allocator
 * variation defeats a single-fingerprint match — see notes in thunks.c
 * below the Phase 1.4 block).  See
 * work/tasks/555-elf-perf-phase-2-batch-3-thunks-strnlen-strcasecmp-
 * strncasecmp-strstr-memmove/phase-1-report.md for the batch-3
 * empirical match table.
 *
 * The scan is one-shot, performed once at ELF load.  The dispatch path is
 * a min/max range bound followed by a tiny linear scan over <= MAX_THUNKS
 * entries — designed to keep the regression canary (pure-ALU workloads
 * that never call any of these primitives) within +-5% of unthunked.
 *
 * See work/tracks/elf-performance/phase-1-thunking/spec.md.
 */

#include "blink/elf.h"
#include "blink/machine.h"  /* defines struct FbxThunks (embedded in System) */
#include "blink/types.h"

/* Walk the ELF symbol table of the freshly-loaded image and register
 * thunks for any of the 5 hot primitives we find.  `aslr` is the elf
 * load-time skew (per LoadElf).  Idempotent within a single System (a
 * second call from exec replaces the prior set; this is intentional).
 * `image` / `imagesize` are the mmap'd host pointer + length of the
 * ELF file (NOT the in-guest mapping). */
void FbxThunksRegisterFromElf(struct System *sys,
                              Elf64_Ehdr_ *ehdr,
                              size_t esize,
                              i64 aslr);

/* Reset the table (called from LoadProgram before LoadElf to clear an
 * inherited table on exec). */
void FbxThunksClear(struct System *sys);

/* Test entry: register a thunk by name directly without parsing an ELF.
 * Returns true on success, false if `name` is unknown or table is full. */
bool FbxThunksRegisterByName(struct System *sys, const char *name, u64 pc);

/* Phase 1.4 fingerprint-based fallback for stripped binaries.
 *
 * When `FbxThunksRegisterFromElf` registers zero entries (e.g. statically-
 * linked binaries shipped stripped, the dominant shape in our bench corpus
 * per work/tracks/elf-performance/phase-1-thunking/phase-1-report.md), we
 * fall back to scanning each PT_LOAD/PF_X segment for byte-pattern matches
 * against a small registry of known libc-primitive prologues.
 *
 * `image` / `esize` are the mmap'd host pointer + length of the ELF file
 * (NOT the in-guest mapping).  `aslr` is the load-time vaddr skew applied
 * by LoadElf.  Returns the number of fingerprint-derived thunks that were
 * registered (0 if none matched).  Idempotent in the same sense as the
 * symbol-table path — overwrites any prior entries when re-invoked.
 *
 * Coverage today (see work/tracks/elf-performance/phase-1-thunking/
 * phase-1.4-report.md for the Phase 1.4 empirical match table,
 * work/tasks/549-elf-perf-phase-1.5-fingerprint-expansion-batch-2/
 * phase-1-report.md for Phase 1.5, and
 * work/tasks/555-elf-perf-phase-2-batch-3-thunks-strnlen-strcasecmp-
 * strncasecmp-strstr-memmove/phase-1-report.md for Phase 2 batch-3):
 *   Phase 1.4 (extremely stable — hand-asm or canonical HASZERO codegen):
 *     - memcpy_musl_x86_64
 *     - memset_musl_x86_64
 *     - strlen_musl_x86_64
 *   Phase 1.5 (musl-C compiled by gcc -Os; per-version short-jump
 *   displacements masked):
 *     - memchr_musl_x86_64   (alignment-prologue, 16-byte pattern)
 *     - strchr_musl_x86_64   (wrapper around __strchrnul; tail-distinctive
 *                             cmovne + immediate-0 pattern)
 *     - strncmp_musl_x86_64  (xor/test/dual-movzbl/setne sequence; 28 bytes)
 *     - strcpy_musl_x86_64   (wrapper around __stpcpy; disambiguated from
 *                             strncpy by following the embedded call
 *                             displacement)
 *     - strncpy_musl_x86_64  (wrapper around __stpncpy; same shape as
 *                             strcpy_musl_x86_64, distinct callee)
 *   Phase 2 batch-3:
 *     - strnlen_musl_x86_64       (calls memchr; 31-byte prologue captures
 *                                   the full body through the post-call
 *                                   cmovne; call displacement masked)
 *     - strcasecmp_musl_x86_64    (2-arg loop; 20-byte prologue; two short-
 *                                   jump displacements masked)
 *     - strncasecmp_musl_x86_64   (3-arg variant; discriminated from
 *                                   strcasecmp by an early `test rdx, rdx;
 *                                   je end_zero` block)
 *     - memmove_musl_x86_64       (musl hand-asm; 32-byte prologue ending in
 *                                   std; rep movsb; cld — extremely
 *                                   distinctive; jae displacement masked)
 *     - strstr_musl_x86_64        (17-byte prologue with `movsx (%rsi),
 *                                   %esi` early-return-on-empty-needle
 *                                   idiom — uncommon outside musl)
 *
 * Best-effort additions (may not match all builds — flagged via the
 * trace output when missing):
 *   - memcmp/strcmp deferred: gcc emits per-build register-allocator
 *     variations that defeat a single-fingerprint match; the call rate
 *     against grep/wc/sort doesn't justify a multi-fingerprint table yet.
 */
int FbxThunksScanFromText(struct System *sys,
                          Elf64_Ehdr_ *ehdr,
                          size_t esize,
                          i64 aslr);

/* Hot path — called from ExecuteInstruction.  Returns true if the
 * current m->ip was a registered thunk PC; in that case the trampoline
 * ran and m->ip was updated to the guest return address.  Inlined into
 * machine.c; see definition below. */
static inline bool FbxThunksMaybeDispatch(struct Machine *m) {
  const struct FbxThunks *t = &m->system->thunks;
  u64 ip;
  int i;
  if (!t->count) return false;
  ip = m->ip;
  if (ip < t->min_pc || ip > t->max_pc) return false;
  for (i = 0; i < t->count; ++i) {
    if (t->entries[i].pc == ip) {
      t->entries[i].trampoline(m);
      return true;
    }
  }
  return false;
}

#endif /* BLINK_THUNKS_H_ */
