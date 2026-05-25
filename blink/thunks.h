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
