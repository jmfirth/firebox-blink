#ifndef BLINK_THUNKS_H_
#define BLINK_THUNKS_H_

/*
 * Firebox ELF-perf Phase 1 — libc thunk routing.
 *
 * At ELF load time, scan the binary's symbol table; when a known-hot libc
 * primitive (memcpy / memset / strlen / memcmp / strcmp) is found we record
 * its entry PC.  Then, on every interpreted instruction, ExecuteInstruction
 * front-checks the current %rip against those PCs; if it matches, we run a
 * wasm-native trampoline that performs the operation in C against the host
 * page and emulates `ret` (pop guest return address, branch to it).  This
 * replaces "interpret hundreds of x86 instructions per call" with "1 host C
 * call" for these primitives.
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
 * phase-1.4-report.md for the empirical match table):
 *   - memcpy_musl_x86_64  (musl x86_64 hand-asm; extremely stable)
 *   - memset_musl_x86_64  (musl x86_64 hand-asm; extremely stable)
 *   - strlen_musl_x86_64  (musl C compiled by gcc with the published HASZERO
 *                          word-loop; matches the gcc 11+/14 / Alpine musl
 *                          combination used by upstream Alpine releases)
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
