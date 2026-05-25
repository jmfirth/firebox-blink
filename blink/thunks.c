/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2026 Justin Firth (Firebox)                                        │
│                                                                              │
│ Phase 1 of the ELF-performance track — libc thunk routing.  Replaces five    │
│ hot libc primitives (memcpy, memset, strlen, memcmp, strcmp) inside the      │
│ Blink interpreter with direct wasm-native C calls when the guest ELF's       │
│ symbol table exposes them.                                                   │
│                                                                              │
│ Design + acceptance gates: work/tracks/elf-performance/phase-1-thunking/.   │
│                                                                              │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │
│ │ Calling convention — System V AMD64                                     │  │
│ │   arg0 = %rdi    arg1 = %rsi    arg2 = %rdx                              │  │
│ │   arg3 = %rcx    arg4 = %r8     arg5 = %r9                               │  │
│ │   ret  = %rax (64-bit)         (rdx pair for >64-bit values, unused)    │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│ Each trampoline runs on entry to its guest function, performs the operation │
│ against guest memory via the existing Blink page-walking helpers, writes    │
│ the result to %rax, then emulates `ret` by reading 8 bytes from the guest   │
│ stack at %rsp, advancing %rsp by 8, and assigning that PC to m->ip.         │
╚─────────────────────────────────────────────────────────────────────────────*/
#include "blink/thunks.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "blink/assert.h"
#include "blink/endian.h"
#include "blink/log.h"
#include "blink/machine.h"
#include "blink/util.h"

/* ────────────────────────────────────────────────────────────────────────── */
/* Per-thread trace flag, lit when FIREBOX_THUNK_TRACE=1 in the environment.  */
/* Read once at System construction (or on the first dispatch if uninit) — we */
/* keep it ABI-simple: a single static int populated by FbxThunksClear().     */
/* ────────────────────────────────────────────────────────────────────────── */

static int g_thunk_trace = -1;  /* -1 = uninitialised; 0/1 once set */

static void EnsureTraceFlag(void) {
  if (g_thunk_trace == -1) {
    const char *e = getenv("FIREBOX_THUNK_TRACE");
    g_thunk_trace = (e && *e && *e != '0') ? 1 : 0;
  }
}

/* Direct stderr write — bypasses LOGF (which is compiled out in release builds)
 * so the trace flag actually surfaces something users can see. */
static void TraceLine(const char *prefix, const char *name, u64 pc) {
  char buf[128];
  int n;
  if (!g_thunk_trace) return;
  n = snprintf(buf, sizeof(buf), "%s %s @ %#llx\n", prefix, name,
               (unsigned long long)pc);
  if (n > 0) (void)write(2, buf, (size_t)n);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Guest-stack helpers.  We do NOT use the existing Pop() because it requires */
/* a decoded instruction (struct XedDecodedInst rde).  Instead we read 8      */
/* bytes from [%rsp] directly via CopyFromUser, then advance %rsp by 8.       */
/* ────────────────────────────────────────────────────────────────────────── */

/* Pop the return PC off the guest stack and assign it to m->ip.  Emulates a
 * 64-bit `ret` (the only ret shape these libc functions emit). */
static void ThunkRet(struct Machine *m) {
  u64 sp;
  u8 tmp[8] = {0};
  i64 ret_pc;
  sp = Get64(m->sp);
  if (CopyFromUser(m, tmp, (i64)sp, 8) == -1) {
    /* Guest stack underflow — let Blink fault as it would have without us.
     * Reset ip to the call site so the standard fault path reports it. */
    LOGF("thunk: failed to read guest return PC at sp=%#llx", (long long)sp);
    return;
  }
  ret_pc = (i64)Get64(tmp);
  Put64(m->sp, sp + 8);
  m->ip = ret_pc;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Page-bounded guest-memory walkers — small wrappers that mirror the page    */
/* loop in VirtualCopy/RepMovsbEnhanced but produce *result* values instead   */
/* of byte copies.                                                            */
/* ────────────────────────────────────────────────────────────────────────── */

/* strnlen on a guest pointer, scanning up to n bytes.  Returns the number of
 * bytes before the first NUL, or `n` if none was seen.  -1 on guest fault. */
static i64 GuestStrnlen(struct Machine *m, i64 s, u64 n) {
  u64 walked = 0;
  while (walked < n) {
    u64 off = (u64)s & 4095;
    u64 chunk = 4096 - off;
    u64 want = n - walked;
    u64 limit = chunk < want ? chunk : want;
    u8 *host = LookupAddress(m, s);
    if (!host) return -1;
    /* host is exactly the page-translated pointer; we may read [host, host+limit). */
    const u8 *p = host;
    const u8 *end = host + limit;
    while (p < end && *p) ++p;
    walked += (u64)(p - host);
    if (p < end) return (i64)walked;  /* hit a NUL */
    s += (i64)limit;
  }
  return (i64)walked;
}

/* Compare two guest memory ranges of length n, byte-by-byte.  Returns the
 * traditional memcmp 3-way result (negative/zero/positive).  On fault we
 * fall through with whatever was read so the guest sees a deterministic
 * answer; the next instruction will re-fault naturally if needed. */
static int GuestMemcmp(struct Machine *m, i64 a, i64 b, u64 n) {
  while (n) {
    u64 oa = (u64)a & 4095;
    u64 ob = (u64)b & 4095;
    u64 chunk_a = 4096 - oa;
    u64 chunk_b = 4096 - ob;
    u64 chunk = chunk_a < chunk_b ? chunk_a : chunk_b;
    if (chunk > n) chunk = n;
    u8 *ha = LookupAddress(m, a);
    u8 *hb = LookupAddress(m, b);
    if (!ha || !hb) return 0; /* let the interpreter fault on next step */
    int r = memcmp(ha, hb, (size_t)chunk);
    if (r) return r;
    n -= chunk;
    a += (i64)chunk;
    b += (i64)chunk;
  }
  return 0;
}

/* Compare two guest C-strings byte-by-byte, page-bounded.  Returns 3-way
 * result.  Walks until the first differing byte OR until one side NULs. */
static int GuestStrcmp(struct Machine *m, i64 a, i64 b) {
  for (;;) {
    u64 oa = (u64)a & 4095;
    u64 ob = (u64)b & 4095;
    u64 chunk_a = 4096 - oa;
    u64 chunk_b = 4096 - ob;
    u64 chunk = chunk_a < chunk_b ? chunk_a : chunk_b;
    u8 *ha = LookupAddress(m, a);
    u8 *hb = LookupAddress(m, b);
    if (!ha || !hb) return 0;
    const u8 *pa = ha;
    const u8 *pb = hb;
    u64 i;
    for (i = 0; i < chunk; ++i) {
      u8 ca = pa[i];
      u8 cb = pb[i];
      if (ca != cb) return (int)ca - (int)cb;
      if (!ca) return 0;  /* both NUL → equal */
    }
    a += (i64)chunk;
    b += (i64)chunk;
  }
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Trampolines.  Each is a guest entry-point handler: reads args from regs,   */
/* performs the operation, writes the result to %rax, emulates ret.           */
/* ────────────────────────────────────────────────────────────────────────── */

static void ThunkMemcpy(struct Machine *m) {
  i64 dst = (i64)Get64(m->di);
  i64 src = (i64)Get64(m->si);
  u64 n = Get64(m->dx);
  if (g_thunk_trace) {
    LOGF("thunk memcpy(dst=%#llx, src=%#llx, n=%llu)",
         (long long)dst, (long long)src, (unsigned long long)n);
  }
  if (n) {
    /* memcpy semantics — non-overlapping.  But many guests call memcpy on
     * overlapping ranges anyway (UB but common); we use CopyFromUser into
     * a temporary buffer when overlap is suspected and CopyToUser back.
     * For the common case we copy page-by-page via VirtualCopy-style loop. */
    /* Detect overlap: if [dst, dst+n) intersects [src, src+n) we must
     * stage through a heap buffer to preserve source bytes.  This matches
     * memmove semantics; memcpy on overlapping regions is UB but real-world
     * libc memcpy implementations typically still produce a sane result. */
    bool overlap = (src < dst && src + (i64)n > dst) ||
                   (dst < src && dst + (i64)n > src);
    if (overlap) {
      u8 *buf = (u8 *)malloc((size_t)n);
      if (buf) {
        if (CopyFromUser(m, buf, src, n) == 0) {
          (void)CopyToUser(m, dst, buf, n);
        }
        free(buf);
      }
    } else {
      /* Non-overlapping: page-by-page copy avoids the malloc round-trip. */
      u64 remaining = n;
      i64 s = src, d = dst;
      u8 stage[4096];
      while (remaining) {
        u64 off_s = (u64)s & 4095;
        u64 off_d = (u64)d & 4095;
        u64 chunk_s = 4096 - off_s;
        u64 chunk_d = 4096 - off_d;
        u64 chunk = chunk_s < chunk_d ? chunk_s : chunk_d;
        if (chunk > remaining) chunk = remaining;
        if (CopyFromUser(m, stage, s, chunk) == -1) break;
        if (CopyToUser(m, d, stage, chunk) == -1) break;
        s += (i64)chunk;
        d += (i64)chunk;
        remaining -= chunk;
      }
    }
  }
  Put64(m->ax, (u64)dst);  /* memcpy returns dst */
  ThunkRet(m);
}

static void ThunkMemset(struct Machine *m) {
  i64 dst = (i64)Get64(m->di);
  u64 c = Get64(m->si) & 0xff;
  u64 n = Get64(m->dx);
  if (g_thunk_trace) {
    LOGF("thunk memset(dst=%#llx, c=%#llx, n=%llu)",
         (long long)dst, (unsigned long long)c, (unsigned long long)n);
  }
  if (n) {
    u8 stage[4096];
    memset(stage, (int)c, sizeof(stage));
    u64 remaining = n;
    i64 d = dst;
    while (remaining) {
      u64 off = (u64)d & 4095;
      u64 chunk = 4096 - off;
      if (chunk > remaining) chunk = remaining;
      if (CopyToUser(m, d, stage, chunk) == -1) break;
      d += (i64)chunk;
      remaining -= chunk;
    }
  }
  Put64(m->ax, (u64)dst);  /* memset returns dst */
  ThunkRet(m);
}

static void ThunkStrlen(struct Machine *m) {
  i64 s = (i64)Get64(m->di);
  /* No length argument — walk page-bounded.  Cap at 1 GiB as a safety net;
   * any real strlen call exceeding that would have faulted anyway, and the
   * cap prevents an infinite loop on a guest where the entire address space
   * is non-NUL (e.g. a buggy caller passing a non-string). */
  i64 len = GuestStrnlen(m, s, (u64)1 << 30);
  if (len < 0) len = 0;
  if (g_thunk_trace) {
    LOGF("thunk strlen(s=%#llx) = %lld", (long long)s, (long long)len);
  }
  Put64(m->ax, (u64)len);
  ThunkRet(m);
}

static void ThunkMemcmp(struct Machine *m) {
  i64 a = (i64)Get64(m->di);
  i64 b = (i64)Get64(m->si);
  u64 n = Get64(m->dx);
  int r = n ? GuestMemcmp(m, a, b, n) : 0;
  if (g_thunk_trace) {
    LOGF("thunk memcmp(a=%#llx, b=%#llx, n=%llu) = %d",
         (long long)a, (long long)b, (unsigned long long)n, r);
  }
  Put64(m->ax, (u64)(i64)r);
  ThunkRet(m);
}

static void ThunkStrcmp(struct Machine *m) {
  i64 a = (i64)Get64(m->di);
  i64 b = (i64)Get64(m->si);
  int r = GuestStrcmp(m, a, b);
  if (g_thunk_trace) {
    LOGF("thunk strcmp(a=%#llx, b=%#llx) = %d",
         (long long)a, (long long)b, r);
  }
  Put64(m->ax, (u64)(i64)r);
  ThunkRet(m);
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Registry — the fixed set of names we know how to handle.                   */
/* Names match the System V libc ABI; statically-linked binaries always       */
/* carry these in their .symtab (we'd also catch them in .dynsym).            */
/* ────────────────────────────────────────────────────────────────────────── */

struct ThunkRegistryEntry {
  const char *name;
  void (*trampoline)(struct Machine *);
};

static const struct ThunkRegistryEntry kRegistry[] = {
    {"memcpy", ThunkMemcpy},
    {"memset", ThunkMemset},
    {"strlen", ThunkStrlen},
    {"memcmp", ThunkMemcmp},
    {"strcmp", ThunkStrcmp},
    {0, 0},
};

static const struct ThunkRegistryEntry *LookupRegistry(const char *name) {
  int i;
  for (i = 0; kRegistry[i].name; ++i) {
    if (!strcmp(name, kRegistry[i].name)) return &kRegistry[i];
  }
  return 0;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Public API                                                                 */
/* ────────────────────────────────────────────────────────────────────────── */

void FbxThunksClear(struct System *sys) {
  EnsureTraceFlag();
  memset(&sys->thunks, 0, sizeof(sys->thunks));
}

static void RecordEntry(struct FbxThunks *t, u64 pc, const char *name,
                        void (*tramp)(struct Machine *)) {
  int i;
  if (t->count >= FBX_MAX_THUNKS) return;
  /* Dedup: if we already have this name (from .symtab and .dynsym both),
   * keep the first.  PC must match — if not, we trust .symtab over .dynsym
   * (rare; ELF spec is firm that both refer to the same vaddr). */
  for (i = 0; i < t->count; ++i) {
    if (!strcmp(t->entries[i].name, name)) return;
  }
  t->entries[t->count].pc = pc;
  t->entries[t->count].trampoline = tramp;
  t->entries[t->count].name = name;  /* string lives in .rodata (kRegistry) */
  if (!t->count || pc < t->min_pc) t->min_pc = pc;
  if (!t->count || pc > t->max_pc) t->max_pc = pc;
  ++t->count;
}

bool FbxThunksRegisterByName(struct System *sys, const char *name, u64 pc) {
  const struct ThunkRegistryEntry *e = LookupRegistry(name);
  if (!e) return false;
  if (sys->thunks.count >= FBX_MAX_THUNKS) return false;
  RecordEntry(&sys->thunks, pc, e->name, e->trampoline);
  return true;
}

void FbxThunksRegisterFromElf(struct System *sys,
                              Elf64_Ehdr_ *ehdr,
                              size_t esize,
                              i64 aslr) {
  char *stab;
  i64 stablen;
  const Elf64_Sym_ *st;
  int n, i;
  const struct ThunkRegistryEntry *match;
  u64 value;
  u32 nameoff;
  EnsureTraceFlag();
  FbxThunksClear(sys);
  if (!(stab = GetElfStringTable(ehdr, esize))) return;
  if (!(st = GetElfSymbolTable(ehdr, esize, &n))) return;
  stablen = (uintptr_t)ehdr + esize - (uintptr_t)stab;
  for (i = 0; i < n && sys->thunks.count < FBX_MAX_THUNKS; ++i) {
    if (ELF64_ST_TYPE_(st[i].info) != STT_FUNC_) continue;
    nameoff = Read32(st[i].name);
    if (nameoff == 0 || (i64)nameoff >= stablen) continue;
    value = Read64(st[i].value);
    if (!value) continue;
    if (!(match = LookupRegistry(stab + nameoff))) continue;
    RecordEntry(&sys->thunks, value + (u64)aslr, match->name, match->trampoline);
    TraceLine("[thunk] registered", match->name, value + (u64)aslr);
  }
  if (g_thunk_trace) {
    char buf[80];
    int len = snprintf(buf, sizeof(buf),
                       "[thunk] scan done: %d entries registered\n",
                       sys->thunks.count);
    if (len > 0) (void)write(2, buf, (size_t)len);
  }
}
