/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2022 Justine Alexandra Roberts Tunney                              │
│                                                                              │
│ Permission to use, copy, modify, and/or distribute this software for         │
│ any purpose with or without fee is hereby granted, provided that the         │
│ above copyright notice and this permission notice appear in all copies.      │
│                                                                              │
│ THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL                │
│ WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED                │
│ WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE             │
│ AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL         │
│ DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR        │
│ PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER               │
│ TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR             │
│ PERFORMANCE OF THIS SOFTWARE.                                                │
╚─────────────────────────────────────────────────────────────────────────────*/
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "blink/assert.h"
#include "blink/endian.h"
#include "blink/flag.h"
#include "blink/linux.h"
#include "blink/log.h"
#include "blink/machine.h"
#include "blink/map.h"
#include "blink/syscall.h"
#include "blink/util.h"

#define STACKALIGN 16

#define PUSH_AUXV(k, v) \
  --naux;               \
  *--p = v;             \
  *--p = k

static size_t GetArgListLen(char **p) {
  size_t n;
  for (n = 0; *p; ++p) ++n;
  return n;
}

static i64 PushBuffer(struct Machine *m, void *s, size_t n) {
  i64 sp = Get64(m->sp) - n;
  Put64(m->sp, sp);
  unassert(!CopyToUser(m, sp, s, n));
  return sp;
}

static i64 PushString(struct Machine *m, char *s) {
  if (s) {
    return PushBuffer(m, s, strlen(s) + 1);
  } else {
    return 0;
  }
}

static long GetGuestPageSize(struct Machine *m) {
  if (HasLinearMapping()) {
    return FLAG_pagesize;
  } else {
    return 4096;
  }
}

void LoadArgv(struct Machine *m, char *execfn, char *prog, char **args,
              char **vars, u8 rng[16]) {
  u8 *bytes;
  struct Elf *elf;
  i64 sp, dx, *p, *bloc;
  size_t i, narg, nenv, naux, nall;
  elf = &m->system->elf;
  naux = 10;
  if (elf->at_entry) {
    naux += 4;
    if (elf->at_base != -1) {
      naux += 1;
    }
  }
  nenv = GetArgListLen(vars);
  narg = GetArgListLen(args);
  nall = 1 + narg + 1 + nenv + 1 + naux * 2;
  bloc = (i64 *)malloc(sizeof(i64) * nall);
  p = bloc + nall;
  dx = PushString(m, prog);
  PUSH_AUXV(0, 0);
  PUSH_AUXV(AT_UID_LINUX, getuid());
  PUSH_AUXV(AT_EUID_LINUX, geteuid());
  PUSH_AUXV(AT_GID_LINUX, getgid());
  PUSH_AUXV(AT_EGID_LINUX, getegid());
  PUSH_AUXV(AT_SECURE_LINUX, IsProcessTainted());
  PUSH_AUXV(AT_PAGESZ_LINUX, GetGuestPageSize(m));
  PUSH_AUXV(AT_CLKTCK_LINUX, sysconf(_SC_CLK_TCK));
  PUSH_AUXV(AT_RANDOM_LINUX, PushBuffer(m, rng, 16));
  PUSH_AUXV(AT_EXECFN_LINUX, PushString(m, execfn));
  if (elf->at_entry) {
    PUSH_AUXV(AT_PHDR_LINUX, elf->at_phdr);
    PUSH_AUXV(AT_PHENT_LINUX, elf->at_phent);
    PUSH_AUXV(AT_PHNUM_LINUX, elf->at_phnum);
    PUSH_AUXV(AT_ENTRY_LINUX, elf->at_entry);
    if (elf->at_base != -1) {
      PUSH_AUXV(AT_BASE_LINUX, elf->at_base);
    }
  }
  unassert(!naux);
  for (*--p = 0, i = nenv; i--;) *--p = PushString(m, vars[i]);
  for (*--p = 0, i = narg; i--;) *--p = PushString(m, args[i]);
  *--p = narg;
  sp = Read64(m->sp);
  while ((sp - nall * sizeof(i64)) & (STACKALIGN - 1)) --sp;
  sp -= nall * sizeof(i64);
  Write64(m->sp, sp);
  /* FIREBOX-535: Linux entry ABI requires %rdx=0 for static binaries
   * (kernel zeros it; ld.so sets it to rtld_fini only for dynamic).
   * Cosmopolitan libc instead reads __program_executable_name from %rdx
   * (Justine's `f96cdbe` 2024-01: "Pass __program_executable_name to
   * cosmo libc"). Blink unconditionally storing dx=&prog_string broke
   * EVERY glibc-static binary's atexit chain: glibc's _start saves rdx
   * to r9, __libc_start_main treats r9 as rtld_fini, and the non-NULL
   * pointer is registered via __cxa_atexit -> on exit, the mangled
   * function pointer demangles to the argv[0] string address, and
   * `call *%rax` faults on the `das` byte of the path. Repro:
   * `echo 'int main(){return 0;}' > t.c && gcc -static -o t t.c &&
   *  firebox run rust -- bash -c '/t; echo $?'` -> EXIT=139.
   * Gate on iscosmo so the Cosmo extension keeps working for APE
   * binaries (MZqFpD/jartsr) while plain ELFs (glibc/musl static) get
   * the Linux-correct ABI. See work/tasks/535-*. */
  if (m->system->iscosmo) {
    Write64(m->dx, dx);
  } else {
    Write64(m->dx, 0);
  }
  Write64(m->di, 0); /* or ape detects freebsd */
  bytes = (u8 *)malloc(nall * 8);
  for (i = 0; i < nall; ++i) {
    Write64(bytes + i * 8, bloc[i]);
  }
  unassert(!CopyToUser(m, sp, bytes, nall * 8));
  free(bytes);
  free(bloc);
}
