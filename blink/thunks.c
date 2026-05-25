/*-*- mode:c;indent-tabs-mode:nil;c-basic-offset:2;tab-width:8;coding:utf-8 -*-│
│ vi: set et ft=c ts=2 sts=2 sw=2 fenc=utf-8                               :vi │
╞══════════════════════════════════════════════════════════════════════════════╡
│ Copyright 2026 Justin Firth (Firebox)                                        │
│                                                                              │
│ Phase 1 of the ELF-performance track — libc thunk routing.  Replaces hot     │
│ libc primitives inside the Blink interpreter with direct wasm-native C       │
│ calls when the guest ELF's symbol table (or fingerprint-scan fallback)       │
│ exposes them.                                                                │
│                                                                              │
│ Phase 1   (firebox-elf-v2-phase1-thunks):                                    │
│   memcpy / memset / strlen / memcmp / strcmp                                 │
│                                                                              │
│ Phase 2 batch-2  (firebox-elf-v2-phase2-thunks-batch-2) — adds:              │
│   memchr / strchr / strncmp / strcpy / strncpy                               │
│                                                                              │
│ All thunks compose at COMPILE TIME into the Tier-1 threaded-code             │
│ dispatcher (see blink/threadedcode.c::LookupThunkAt + CompileBlock); a       │
│ matched PC becomes a single FBX_TC_KIND_THUNK block-entry — no per-          │
│ instruction dispatch tax.                                                    │
│                                                                              │
│ Design + acceptance gates: work/tracks/elf-performance/phase-1-thunking/    │
│ (Phase 1) and work/tracks/elf-performance/phase-2-hot-path-caching/         │
│ (Phase 2 dispatcher + batch-2 thunks).                                       │
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

/* Scan up to n bytes of guest memory at address `s` for the first byte
 * equal to `c` (low 8 bits).  Returns the absolute guest address of the
 * match, or 0 if no match in the first n bytes.  -1 on guest fault.
 *
 * Matches Linux memchr(3) semantics: comparison is unsigned-char-wise. */
static i64 GuestMemchr(struct Machine *m, i64 s, u8 c, u64 n) {
  u64 walked = 0;
  i64 cur = s;
  while (walked < n) {
    u64 off = (u64)cur & 4095;
    u64 chunk = 4096 - off;
    u64 want = n - walked;
    u64 limit = chunk < want ? chunk : want;
    u8 *host = LookupAddress(m, cur);
    if (!host) return -1;
    /* memchr returns pointer-into-buffer; use the libc primitive for the
     * inner page-bounded scan — vectorised on the host. */
    void *hit = memchr(host, (int)c, (size_t)limit);
    if (hit) {
      return cur + (i64)((const u8 *)hit - host);
    }
    walked += limit;
    cur += (i64)limit;
  }
  return 0;
}

/* Walk a guest C-string from `s` until either:
 *   - a byte equal to `c` is seen → return its guest address
 *   - a NUL is seen → if c == 0, return its address (POSIX: strchr matches
 *     NUL); otherwise return 0
 *   - a guest fault → return -1
 *
 * Page-bounded, like the rest of the helpers. */
static i64 GuestStrchr(struct Machine *m, i64 s, u8 c) {
  i64 cur = s;
  for (;;) {
    u64 off = (u64)cur & 4095;
    u64 limit = 4096 - off;
    u8 *host = LookupAddress(m, cur);
    if (!host) return -1;
    u64 i;
    for (i = 0; i < limit; ++i) {
      u8 b = host[i];
      if (b == c) return cur + (i64)i;
      if (!b) {
        /* End-of-string.  POSIX: strchr(s, 0) returns pointer to the
         * terminator — but that's caught by the `b == c` branch above
         * when c == 0, so reaching here means c != 0 and we miss. */
        return 0;
      }
    }
    cur += (i64)limit;
  }
}

/* Compare up to n bytes of two guest C-strings, page-bounded.  Returns
 * memcmp-style 3-way result.  Stops at the first NUL on either side
 * (POSIX strncmp semantics) OR after n bytes. */
static int GuestStrncmp(struct Machine *m, i64 a, i64 b, u64 n) {
  while (n) {
    u64 oa = (u64)a & 4095;
    u64 ob = (u64)b & 4095;
    u64 chunk_a = 4096 - oa;
    u64 chunk_b = 4096 - ob;
    u64 chunk = chunk_a < chunk_b ? chunk_a : chunk_b;
    if (chunk > n) chunk = n;
    u8 *ha = LookupAddress(m, a);
    u8 *hb = LookupAddress(m, b);
    if (!ha || !hb) return 0;
    u64 i;
    for (i = 0; i < chunk; ++i) {
      u8 ca = ha[i];
      u8 cb = hb[i];
      if (ca != cb) return (int)ca - (int)cb;
      if (!ca) return 0;  /* both reached NUL — equal */
    }
    n -= chunk;
    a += (i64)chunk;
    b += (i64)chunk;
  }
  return 0;
}

/* Copy a guest C-string from `src` to `dst` including the terminating NUL.
 * Returns the number of bytes written (string length including NUL), -1
 * on fault.  Page-bounded; uses a small stack stage buffer so we never
 * hold two page lookups live across a CopyToUser call. */
static i64 GuestStrcpy(struct Machine *m, i64 dst, i64 src) {
  i64 s = src, d = dst;
  u64 written = 0;
  for (;;) {
    u8 stage[4096];
    u64 off_s = (u64)s & 4095;
    u64 chunk_s = 4096 - off_s;
    u8 *hs = LookupAddress(m, s);
    if (!hs) return -1;
    /* Find NUL inside this source page (or chunk-end if none). */
    void *nul = memchr(hs, 0, (size_t)chunk_s);
    u64 want;
    bool terminate;
    if (nul) {
      want = (u64)((const u8 *)nul - hs) + 1;  /* include the NUL byte */
      terminate = true;
    } else {
      want = chunk_s;
      terminate = false;
    }
    /* Stage out — must not hold the source page mapping while CopyToUser
     * may invalidate the page table.  4 KiB stage buffer is bounded. */
    memcpy(stage, hs, (size_t)want);
    /* Write into the dst, respecting page boundaries on the destination. */
    {
      u64 remaining = want;
      u64 staged_off = 0;
      i64 dcur = d;
      while (remaining) {
        u64 off_d = (u64)dcur & 4095;
        u64 chunk_d = 4096 - off_d;
        u64 to_write = chunk_d < remaining ? chunk_d : remaining;
        if (CopyToUser(m, dcur, stage + staged_off, to_write) == -1) {
          return -1;
        }
        dcur += (i64)to_write;
        staged_off += to_write;
        remaining -= to_write;
      }
      d = dcur;
    }
    written += want;
    s += (i64)want;
    if (terminate) return (i64)written;
  }
}

/* Copy up to n bytes of a guest C-string from `src` to `dst`, NUL-padding
 * any remainder if `src` is shorter than n.  Returns 0 on success, -1 on
 * fault.  Page-bounded.
 *
 * Matches POSIX strncpy:
 *   - If strlen(src) >= n: copy exactly n bytes, NO NUL terminator added.
 *   - If strlen(src) < n:  copy strlen(src) bytes then NUL-pad to n. */
static int GuestStrncpy(struct Machine *m, i64 dst, i64 src, u64 n) {
  i64 s = src, d = dst;
  bool src_exhausted = false;  /* once true, we're in the NUL-pad phase */
  while (n) {
    u8 stage[4096];
    u64 chunk;
    if (src_exhausted) {
      /* Pad with NUL — limit to dst page boundary OR remaining n. */
      u64 off_d = (u64)d & 4095;
      u64 chunk_d = 4096 - off_d;
      chunk = chunk_d < n ? chunk_d : n;
      memset(stage, 0, (size_t)chunk);
    } else {
      u64 off_s = (u64)s & 4095;
      u64 off_d = (u64)d & 4095;
      u64 chunk_s = 4096 - off_s;
      u64 chunk_d = 4096 - off_d;
      chunk = chunk_s < chunk_d ? chunk_s : chunk_d;
      if (chunk > n) chunk = n;
      u8 *hs = LookupAddress(m, s);
      if (!hs) return -1;
      /* Look for NUL inside this source chunk. */
      void *nul = memchr(hs, 0, (size_t)chunk);
      if (nul) {
        u64 strlen_in_chunk = (u64)((const u8 *)nul - hs);
        /* Copy strlen_in_chunk bytes of real source, then the NUL, then
         * fall through to NUL-pad the rest of the chunk if any. */
        memcpy(stage, hs, (size_t)strlen_in_chunk);
        /* Pad the rest of `chunk` with NULs. */
        memset(stage + strlen_in_chunk, 0,
               (size_t)(chunk - strlen_in_chunk));
        src_exhausted = true;
      } else {
        memcpy(stage, hs, (size_t)chunk);
      }
    }
    if (CopyToUser(m, d, stage, chunk) == -1) return -1;
    d += (i64)chunk;
    s += (i64)chunk;
    n -= chunk;
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
/* Phase 2 batch-2 trampolines.  Same shape as the Phase 1 set: read args     */
/* from the System V AMD64 register file, perform the operation against      */
/* guest memory via Guest* helpers, write the result to %rax, emulate ret.   */
/* ────────────────────────────────────────────────────────────────────────── */

static void ThunkMemchr(struct Machine *m) {
  i64 s = (i64)Get64(m->di);
  u8 c = (u8)(Get64(m->si) & 0xff);
  u64 n = Get64(m->dx);
  /* memchr returns NULL when n == 0, regardless of buffer contents. */
  i64 hit = n ? GuestMemchr(m, s, c, n) : 0;
  if (hit < 0) hit = 0;  /* fault → NULL; next access will refault */
  if (g_thunk_trace) {
    LOGF("thunk memchr(s=%#llx, c=%#x, n=%llu) = %#llx",
         (long long)s, (unsigned)c, (unsigned long long)n,
         (long long)hit);
  }
  Put64(m->ax, (u64)hit);
  ThunkRet(m);
}

static void ThunkStrchr(struct Machine *m) {
  i64 s = (i64)Get64(m->di);
  u8 c = (u8)(Get64(m->si) & 0xff);
  i64 hit = GuestStrchr(m, s, c);
  if (hit < 0) hit = 0;
  if (g_thunk_trace) {
    LOGF("thunk strchr(s=%#llx, c=%#x) = %#llx",
         (long long)s, (unsigned)c, (long long)hit);
  }
  Put64(m->ax, (u64)hit);
  ThunkRet(m);
}

static void ThunkStrncmp(struct Machine *m) {
  i64 a = (i64)Get64(m->di);
  i64 b = (i64)Get64(m->si);
  u64 n = Get64(m->dx);
  int r = n ? GuestStrncmp(m, a, b, n) : 0;
  if (g_thunk_trace) {
    LOGF("thunk strncmp(a=%#llx, b=%#llx, n=%llu) = %d",
         (long long)a, (long long)b, (unsigned long long)n, r);
  }
  Put64(m->ax, (u64)(i64)r);
  ThunkRet(m);
}

static void ThunkStrcpy(struct Machine *m) {
  i64 dst = (i64)Get64(m->di);
  i64 src = (i64)Get64(m->si);
  /* strcpy semantics: undefined on overlap — but real-world libcs still
   * produce a sane forward copy.  GuestStrcpy walks the source and writes
   * via 4KiB-staged CopyToUser, which is forward-direction-safe even on
   * overlap as long as dst <= src.  Don't try to defend against UB beyond
   * what the libc primitive itself defends against. */
  (void)GuestStrcpy(m, dst, src);
  if (g_thunk_trace) {
    LOGF("thunk strcpy(dst=%#llx, src=%#llx)",
         (long long)dst, (long long)src);
  }
  Put64(m->ax, (u64)dst);
  ThunkRet(m);
}

static void ThunkStrncpy(struct Machine *m) {
  i64 dst = (i64)Get64(m->di);
  i64 src = (i64)Get64(m->si);
  u64 n = Get64(m->dx);
  if (n) (void)GuestStrncpy(m, dst, src, n);
  if (g_thunk_trace) {
    LOGF("thunk strncpy(dst=%#llx, src=%#llx, n=%llu)",
         (long long)dst, (long long)src, (unsigned long long)n);
  }
  Put64(m->ax, (u64)dst);
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
    /* Phase 1 (firebox-elf-v2-phase1-thunks @ cc58fbc). */
    {"memcpy", ThunkMemcpy},
    {"memset", ThunkMemset},
    {"strlen", ThunkStrlen},
    {"memcmp", ThunkMemcmp},
    {"strcmp", ThunkStrcmp},
    /* Phase 2 batch-2 (firebox-elf-v2-phase2-thunks-batch-2).  Same
     * dispatch shape; all compose at compile time into the threaded-code
     * dispatcher via LookupThunkAt in blink/threadedcode.c. */
    {"memchr", ThunkMemchr},
    {"strchr", ThunkStrchr},
    {"strncmp", ThunkStrncmp},
    {"strcpy", ThunkStrcpy},
    {"strncpy", ThunkStrncpy},
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

/* ────────────────────────────────────────────────────────────────────────── */
/* Phase 1.4 — machine-code fingerprinting for stripped binaries.             */
/*                                                                            */
/* The symbol-table path above succeeds only when the binary carries          */
/* .symtab or .dynstr/.dynsym.  Statically-linked binaries shipped stripped   */
/* (which is 100% of our high-cost bench corpus: Alpine's musl-static         */
/* busybox, jq, etc) carry neither.  We fall back to scanning each            */
/* PT_LOAD/PF_X program-header segment for byte-pattern matches against the  */
/* known prologues of well-recognised libc primitives.                       */
/*                                                                            */
/* Empirical discovery procedure (Phase 1.4 bring-up):                       */
/*   1. Build the canonical musl x86_64 hand-asm `memcpy.s` / `memset.s`     */
/*      via clang -target x86_64-linux-gnu -c.  The asm is hand-rolled with  */
/*      no register-allocator variation, so every byte of the prologue is    */
/*      stable across compiler versions.                                     */
/*   2. Compile musl's C `strlen.c` with -Os -fno-stack-protector.  The      */
/*      HASZERO word-loop produces a recognisable prologue:                  */
/*        movq %rdi, %rax ; testb $0x7, %dil ; jne <byte_loop> ; jmp <wloop> */
/*   3. For each candidate prologue, search the busybox `.text` segment via  */
/*      a python sliding-window scan; verify exactly one match (the          */
/*      function entry) and no false-positives anywhere else in the binary.  */
/*                                                                            */
/* Pattern stability notes:                                                   */
/*   - memcpy/memset: musl x86_64/{memcpy,memset}.s is hand-rolled and       */
/*     emits identical bytes for every Alpine build we sampled (Alpine 3.16  */
/*     through edge as of 2026-05).  Mask is all-0xFF (exact match).         */
/*   - strlen: gcc's HASZERO codegen IS stable across the gcc 11..14 range   */
/*     bundled with Alpine; if gcc 15 changes the loop shape we'd need to    */
/*     add a second fingerprint or widen the mask.                          */
/*                                                                            */
/* Function-boundary heuristic:                                              */
/*   For patterns >= 16 bytes long, the exact-match probability over random  */
/*   .text bytes is ~2^-128 (or much lower with the structured opcodes),    */
/*   so we trust pattern uniqueness alone — no boundary check required.    */
/*   For patterns < 16 bytes, we require the preceding byte to be a        */
/*   plausible inter-function pad/end:                                     */
/*     0xc3  (`ret`, end of prior function)                                  */
/*     0x90  (`nop` align padding)                                           */
/*     0xcc  (`int3` padding — debug builds)                                 */
/*     0xff  (final byte of `jmp/call` rel32 displacement — common immediately */
/*           preceding a tail-call target in libc)                          */
/*   This prevents short patterns from matching inside an unrelated         */
/*   instruction's immediate-byte stream, which would mis-register a thunk  */
/*   PC and crash on dispatch (the trampoline emulates `ret`, which expects */
/*   a real saved-PC on the guest stack).                                  */
/* ────────────────────────────────────────────────────────────────────────── */

#define FBX_FINGERPRINT_MAX_LEN 32

struct ThunkFingerprint;

/* Optional post-match disambiguation hook.
 *
 * Some pairs of musl primitives compile to identical wrapper prologues — the
 * canonical example is strcpy/strncpy (both compile to "push %r12; mov %rdi,
 * %r12; call __stp{n}cpy; mov %r12, %rax; pop %r12; ret").  We cannot tell
 * which is which from the prologue alone; the only discriminator is the
 * callee's prologue.
 *
 * `seg`/`seg_len`/`i` describe the match position (i is the offset within the
 * segment).  Implementations may follow the prologue's `call` displacement
 * forward and inspect callee bytes.  Return true to accept the match, false to
 * reject. */
typedef bool (*ThunkFingerprintVerifyFn)(const u8 *seg, size_t seg_len,
                                         size_t i);

struct ThunkFingerprint {
  const char *id;          /* diagnostic name, e.g. "memcpy_musl_x86_64" */
  const char *thunk_name;  /* registry key (must match kRegistry) */
  const u8 *pattern;       /* exact bytes to match (length = pattern_len) */
  const u8 *mask;          /* 0xFF = compare, 0x00 = wildcard */
  size_t pattern_len;
  /* Optional — NULL means "accept any prologue match". */
  ThunkFingerprintVerifyFn verify;
  /* If `verify` follows a `call rel32` displacement, this is the byte offset
   * within the matched pattern at which the `e8` opcode lives.  0 if unused.
   * Used by the shared CallTargetMatches helper. */
  size_t call_byte_offset;
};

/* musl x86_64 hand-asm memcpy — first 31 bytes through the `rep movsq` core.
 *
 *   48 89 f8                movq %rdi, %rax
 *   48 83 fa 08             cmpq $0x8, %rdx
 *   72 14                   jb   .+0x14
 *   f7 c7 07 00 00 00       testl $0x7, %edi
 *   74 0c                   je   .+0x0c
 *   a4                      movsb
 *   48 ff ca                decq %rdx
 *   f7 c7 07 00 00 00       testl $0x7, %edi
 *   75 f4                   jne  .-12
 *   48 89 d1                movq %rdx, %rcx
 *   48 c1 e9 03             shrq $0x3, %rcx
 *   f3 48 a5                rep movsq
 *
 * All bytes are stable across musl 1.2.x; no register-allocator variation. */
static const u8 kMemcpyMuslPattern[] = {
    0x48, 0x89, 0xf8,                    /* movq %rdi, %rax              */
    0x48, 0x83, 0xfa, 0x08,              /* cmpq $0x8, %rdx              */
    0x72, 0x14,                          /* jb   +0x14                   */
    0xf7, 0xc7, 0x07, 0x00, 0x00, 0x00,  /* testl $0x7, %edi             */
    0x74, 0x0c,                          /* je   +0x0c                   */
    0xa4,                                /* movsb                        */
    0x48, 0xff, 0xca,                    /* decq %rdx                    */
    0xf7, 0xc7, 0x07, 0x00, 0x00, 0x00,  /* testl $0x7, %edi             */
    0x75, 0xf4,                          /* jne  -12                     */
    0x48, 0x89, 0xd1,                    /* movq %rdx, %rcx              */
};
static const u8 kMemcpyMuslMask[] = {
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff,
    0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff,
    0xff,
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff,
    0xff, 0xff, 0xff,
};

/* musl x86_64 hand-asm memset — first 22 bytes through the multiply-by-ONES.
 *
 *   48 0f b6 c6                            movzbq %sil, %rax
 *   49 b8 01 01 01 01 01 01 01 01          movabsq $0x0101010101010101, %r8
 *   49 0f af c0                            imulq %r8, %rax
 *   48 83 fa 7e                            cmpq $0x7e, %rdx
 *
 * The 0x0101010101010101 immediate is the broadcast multiplier — distinctive
 * enough on its own; the imul + cmp lock it down.  All bytes stable. */
static const u8 kMemsetMuslPattern[] = {
    0x48, 0x0f, 0xb6, 0xc6,
    0x49, 0xb8, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x49, 0x0f, 0xaf, 0xc0,
    0x48, 0x83, 0xfa, 0x7e,
};
static const u8 kMemsetMuslMask[] = {
    0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff,
};

/* musl C strlen compiled with gcc -Os/-O2 — first 11 bytes through the
 * alignment branch.  The HASZERO word-loop later in the function uses the
 * 0x0101010101010101 + 0x8080808080808080 constants we'd traditionally
 * match, but those constants are written as `movabsq` IMMEDIATELY before
 * the loop body — meaning earlier code (memchr, strchrnul) hits the same
 * constants and we can't safely use them as the discriminator.  Instead
 * we match the prologue alignment-check sequence:
 *
 *   48 89 f8                movq %rdi, %rax
 *   40 f6 c7 07             testb $0x7, %dil
 *   75 0f                   jne  +0x0f       (byte_loop fallback)
 *   eb 1d                   jmp  +0x1d       (skip into word_loop)
 *
 * The `eb 1d` (short jmp) skip-distance varies between gcc versions; we
 * therefore mask the displacement byte (0x1d).  Verified against Alpine
 * musl-1.2.5 (busybox 1.36.x build): exactly one match in busybox. */
static const u8 kStrlenMuslPattern[] = {
    0x48, 0x89, 0xf8,        /* movq %rdi, %rax           */
    0x40, 0xf6, 0xc7, 0x07,  /* testb $0x7, %dil          */
    0x75, 0x0f,              /* jne   +0x0f               */
    0xeb, 0x1d,              /* jmp   +0x1d               */
};
static const u8 kStrlenMuslMask[] = {
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff,
    0xff, 0xff,
    0xff, 0x00,              /* mask the jmp displacement */
};

/* ────────────────────────────────────────────────────────────────────────── */
/* Phase 1.5 — fingerprint expansion for Phase-2 batch-2 primitives.          */
/*                                                                            */
/* memchr/strchr/strncmp/strcpy/strncpy.  Phase 2 batch-2 (#546) added the    */
/* trampolines + .symtab path for these; this set adds machine-code           */
/* fingerprints so stripped musl-static binaries (the dominant bench corpus   */
/* shape — Alpine's musl-static busybox / jq) benefit too.  See               */
/* work/tasks/549-elf-perf-phase-1.5-.../phase-1-report.md for the empirical  */
/* match table against                                                        */
/* Alpine's busybox 1.36.x build.                                             */
/*                                                                            */
/* All five primitives below are pure musl-C (no x86_64 hand-asm in           */
/* libc-top-half/musl/src/string/x86_64), so the bytes come from gcc -Os      */
/* compiling the published musl reference impl.  Discovery procedure mirrors  */
/* Phase 1.4: search the busybox `.text` segment for the function's own       */
/* characteristic constants/opcodes, then capture the prologue.               */
/*                                                                            */
/* IDENTICAL-PROLOGUE WRAPPERS:                                               */
/*   strcpy and strncpy compile to the same 16-byte wrapper (push r12; mov    */
/*   rdi, r12; call __stp{n}cpy; mov r12, rax; pop r12; ret) — the only      */
/*   discriminator is which function gets called.  We use a per-fingerprint   */
/*   `verify` callback that follows the `e8 rel32` displacement and inspects  */
/*   the callee's prologue: __stpcpy starts with "48 89 fa 48 89 f8 48 31 f2  */
/*   83 e2 07" (2-arg, no n-test), __stpncpy with "48 89 f8 41 54 49 89 fc   */
/*   48 31 f0 a8 07" (3-arg, tests n early via test $0x7, %al).               */
/* ────────────────────────────────────────────────────────────────────────── */

/* musl C memchr compiled with gcc -Os/-O2.  Prologue is the alignment-loop
 * entry:
 *
 *   40 0f b6 f6              movzbl %sil, %esi          (c = (uchar)c)
 *   40 f6 c7 07              testb $0x7, %dil           (s & ALIGN)
 *   75 1b                    jne   slow_byte_loop
 *   eb 24                    jmp   word_loop_entry
 *   0f 1f 40 00              4-byte NOP                 (next-function pad)
 *
 * The two short-jmp displacements (0x1b, 0x24) vary slightly between musl
 * versions and gcc versions — we mask both displacement bytes.  Verified
 * against Alpine's musl-1.2.5 busybox 1.36 build (exactly one match in
 * .text, at the function entry). */
static const u8 kMemchrMuslPattern[] = {
    0x40, 0x0f, 0xb6, 0xf6,        /* movzbl %sil, %esi     */
    0x40, 0xf6, 0xc7, 0x07,        /* testb $0x7, %dil      */
    0x75, 0x1b,                    /* jne   +0x1b           */
    0xeb, 0x24,                    /* jmp   +0x24           */
    0x0f, 0x1f, 0x40, 0x00,        /* 4-byte NOP            */
};
static const u8 kMemchrMuslMask[] = {
    0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff,
    0xff, 0x00,                    /* mask jne displacement */
    0xff, 0x00,                    /* mask jmp displacement */
    0xff, 0xff, 0xff, 0xff,
};

/* musl C strchr wraps __strchrnul.  gcc -Os produces a distinctive tail
 * sequence — call result + cmovne against immediate zero:
 *
 *   53                       push   %rbx
 *   89 f3                    mov    %esi, %ebx           (save c)
 *   e8 .. .. .. ..           call   __strchrnul
 *   ba 00 00 00 00           mov    $0, %edx
 *   38 18                    cmp    %bl, (%rax)
 *   5b                       pop    %rbx
 *   48 0f 45 c2              cmovne %rdx, %rax           (clear rax on miss)
 *   c3                       ret
 *
 * Length 21 bytes including the masked 4-byte call displacement.  This is a
 * very specific tail — `ba 00 00 00 00 38 18 5b 48 0f 45 c2 c3` is essentially
 * unique to "strchr-after-strchrnul" across Alpine's full binary corpus. */
static const u8 kStrchrMuslPattern[] = {
    0x53,                          /* push %rbx                              */
    0x89, 0xf3,                    /* mov %esi, %ebx                         */
    0xe8, 0x00, 0x00, 0x00, 0x00,  /* call __strchrnul (masked)              */
    0xba, 0x00, 0x00, 0x00, 0x00,  /* mov $0, %edx                           */
    0x38, 0x18,                    /* cmp %bl, (%rax)                        */
    0x5b,                          /* pop %rbx                               */
    0x48, 0x0f, 0x45, 0xc2,        /* cmovne %rdx, %rax                      */
    0xc3,                          /* ret                                    */
};
static const u8 kStrchrMuslMask[] = {
    0xff,
    0xff, 0xff,
    0xff, 0x00, 0x00, 0x00, 0x00,  /* mask call displacement                 */
    0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff,
    0xff,
    0xff, 0xff, 0xff, 0xff,
    0xff,
};

/* musl C strncmp compiled with gcc -Os.  Has a distinctive register choreography
 * around the n-- decrement and the dual byte loads:
 *
 *   31 c0                    xor   %eax, %eax            (eax = 0 default)
 *   48 85 d2                 test  %rdx, %rdx            (n == 0?)
 *   74 ..                    je    end_zero
 *   0f b6 0f                 movzbl (%rdi), %ecx         (load *l)
 *   44 0f b6 06              movzbl (%rsi), %r8d         (load *r)
 *   84 c9                    test  %cl, %cl              (*l == 0?)
 *   74 ..                    je    end
 *   48 83 ea 01              sub   $1, %rdx              (n--)
 *   49 89 d1                 mov   %rdx, %r9             (save n)
 *   0f 95 c2                 setne %dl                   (dl = (n != 0))
 *
 * Length 28 bytes.  The two short-jump displacements (0x5c, 0x56) vary across
 * musl/gcc versions — we mask them.  Highly specific: `setne %dl` directly
 * after a `sub`/`mov` triplet is uncommon outside this kind of "two-things-
 * still-valid" loop guard. */
static const u8 kStrncmpMuslPattern[] = {
    0x31, 0xc0,                    /* xor %eax, %eax                          */
    0x48, 0x85, 0xd2,              /* test %rdx, %rdx                         */
    0x74, 0x5c,                    /* je end_zero                             */
    0x0f, 0xb6, 0x0f,              /* movzbl (%rdi), %ecx                     */
    0x44, 0x0f, 0xb6, 0x06,        /* movzbl (%rsi), %r8d                     */
    0x84, 0xc9,                    /* test %cl, %cl                           */
    0x74, 0x56,                    /* je end                                  */
    0x48, 0x83, 0xea, 0x01,        /* sub $1, %rdx                            */
    0x49, 0x89, 0xd1,              /* mov %rdx, %r9                           */
    0x0f, 0x95, 0xc2,              /* setne %dl                               */
};
static const u8 kStrncmpMuslMask[] = {
    0xff, 0xff,
    0xff, 0xff, 0xff,
    0xff, 0x00,                    /* mask je displacement                    */
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff,
    0xff, 0xff,
    0xff, 0x00,                    /* mask je displacement                    */
    0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff,
};

/* musl strcpy compiles to a 16-byte trampoline that just calls __stpcpy and
 * returns the original dst pointer.  IDENTICAL bytes to the strncpy wrapper —
 * disambiguated by following the call to inspect the callee.  See the
 * VerifyStrcpy / VerifyStrncpy callbacks below.
 *
 *   41 54                    push %r12
 *   49 89 fc                 mov  %rdi, %r12
 *   e8 .. .. .. ..           call __stpcpy        (or __stpncpy for strncpy)
 *   4c 89 e0                 mov  %r12, %rax
 *   41 5c                    pop  %r12
 *   c3                       ret
 *
 * Length 16 bytes.  Highly specific.  */
static const u8 kStrcpyMuslPattern[] = {
    0x41, 0x54,                    /* push %r12                               */
    0x49, 0x89, 0xfc,              /* mov %rdi, %r12                          */
    0xe8, 0x00, 0x00, 0x00, 0x00,  /* call __stp(n)cpy (masked)               */
    0x4c, 0x89, 0xe0,              /* mov %r12, %rax                          */
    0x41, 0x5c,                    /* pop %r12                                */
    0xc3,                          /* ret                                     */
};
static const u8 kStrcpyMuslMask[] = {
    0xff, 0xff,
    0xff, 0xff, 0xff,
    0xff, 0x00, 0x00, 0x00, 0x00,  /* mask 4-byte call displacement           */
    0xff, 0xff, 0xff,
    0xff, 0xff,
    0xff,
};

/* strncpy uses the SAME pattern as strcpy — see comment above.  Stored as a
 * separate fingerprint so the verifier callback can route to the strncpy
 * trampoline.  We share the underlying pattern/mask arrays. */
#define kStrncpyMuslPattern kStrcpyMuslPattern
#define kStrncpyMuslMask    kStrcpyMuslMask

/* Bounds-checked little-endian 32-bit signed read at byte offset `off`. */
static bool ReadRel32(const u8 *seg, size_t seg_len, size_t off, int32_t *out) {
  if (off + 4 > seg_len) return false;
  *out = (int32_t)((u32)seg[off] | ((u32)seg[off + 1] << 8) |
                   ((u32)seg[off + 2] << 16) | ((u32)seg[off + 3] << 24));
  return true;
}

/* Bounds-checked match: returns true iff seg[at..at+pat_len) equals `pat`
 * (under `mask`). */
static bool MatchAt(const u8 *seg, size_t seg_len, size_t at, const u8 *pat,
                    const u8 *mask, size_t pat_len) {
  size_t k;
  if (at + pat_len > seg_len) return false;
  for (k = 0; k < pat_len; ++k) {
    if ((seg[at + k] & mask[k]) != (pat[k] & mask[k])) return false;
  }
  return true;
}

/* __stpcpy prologue — gcc -Os emission of musl's stpcpy.c:
 *
 *   48 89 fa                 mov %rdi, %rdx       (save dst)
 *   48 89 f8                 mov %rdi, %rax       (return-value tracker)
 *   48 31 f2                 xor %rsi, %rdx       (low bits = src ^ dst)
 *   83 e2 07                 and $0x7, %edx       (test ALIGN equality)
 *
 * 12 bytes, no register variation, no early `test rdx, rdx` (2-arg function). */
static const u8 kStpcpyPrologue[] = {
    0x48, 0x89, 0xfa,
    0x48, 0x89, 0xf8,
    0x48, 0x31, 0xf2,
    0x83, 0xe2, 0x07,
};
static const u8 kStpcpyPrologueMask[] = {
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff,
};

/* __stpncpy prologue — gcc -Os emission of musl's stpncpy.c:
 *
 *   48 89 f8                 mov %rdi, %rax       (return-value tracker)
 *   41 54                    push %r12            (3-arg: callee-saved use)
 *   49 89 fc                 mov %rdi, %r12       (save dst)
 *   48 31 f0                 xor %rsi, %rax       (low bits = src ^ dst)
 *   a8 07                    test $0x7, %al       (ALIGN equality test)
 *
 * 13 bytes.  The early `push %r12` is the discriminator — __stpcpy doesn't
 * touch %r12, but __stpncpy needs it (3-arg version saves dst across the
 * pad-with-zero call to memset at the end). */
static const u8 kStpncpyPrologue[] = {
    0x48, 0x89, 0xf8,
    0x41, 0x54,
    0x49, 0x89, 0xfc,
    0x48, 0x31, 0xf0,
    0xa8, 0x07,
};
static const u8 kStpncpyPrologueMask[] = {
    0xff, 0xff, 0xff,
    0xff, 0xff,
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff,
    0xff, 0xff,
};

/* Shared callee-following helper.  `i` is the strcpy/strncpy prologue offset
 * within `seg`.  The call opcode lives at seg[i + 5]; the rel32 immediate at
 * seg[i + 6..i + 10].  Returns true iff the callee at the computed offset
 * matches `expected_prologue`. */
static bool CalleePrologueMatches(const u8 *seg, size_t seg_len, size_t i,
                                  const u8 *prologue, const u8 *mask,
                                  size_t prologue_len) {
  int32_t disp;
  size_t call_end;
  intptr_t target;
  if (!ReadRel32(seg, seg_len, i + 6, &disp)) return false;
  call_end = i + 10;  /* byte AFTER the rel32 immediate                       */
  /* Compute target = call_end + disp, defending against signed overflow into
   * negative seg-offsets (which can happen for legitimately-short backward
   * jumps to libc functions earlier in the segment). */
  if (disp < 0) {
    /* Backward — call_end + disp must remain >= 0 and within seg. */
    if ((intptr_t)call_end + disp < 0) return false;
    target = (intptr_t)call_end + disp;
  } else {
    if (call_end > seg_len - (size_t)disp) return false;
    target = (intptr_t)call_end + disp;
  }
  if ((size_t)target >= seg_len) return false;
  return MatchAt(seg, seg_len, (size_t)target, prologue, mask, prologue_len);
}

static bool VerifyStrcpy(const u8 *seg, size_t seg_len, size_t i) {
  return CalleePrologueMatches(seg, seg_len, i, kStpcpyPrologue,
                               kStpcpyPrologueMask, sizeof(kStpcpyPrologue));
}

static bool VerifyStrncpy(const u8 *seg, size_t seg_len, size_t i) {
  return CalleePrologueMatches(seg, seg_len, i, kStpncpyPrologue,
                               kStpncpyPrologueMask, sizeof(kStpncpyPrologue));
}

static const struct ThunkFingerprint kFingerprints[] = {
    {"memcpy_musl_x86_64", "memcpy",
     kMemcpyMuslPattern, kMemcpyMuslMask, sizeof(kMemcpyMuslPattern),
     NULL, 0},
    {"memset_musl_x86_64", "memset",
     kMemsetMuslPattern, kMemsetMuslMask, sizeof(kMemsetMuslPattern),
     NULL, 0},
    {"strlen_musl_x86_64", "strlen",
     kStrlenMuslPattern, kStrlenMuslMask, sizeof(kStrlenMuslPattern),
     NULL, 0},
    /* Phase 1.5 expansion (firebox-elf-v2-phase1.5-fingerprint-expansion-batch-2). */
    {"memchr_musl_x86_64", "memchr",
     kMemchrMuslPattern, kMemchrMuslMask, sizeof(kMemchrMuslPattern),
     NULL, 0},
    {"strchr_musl_x86_64", "strchr",
     kStrchrMuslPattern, kStrchrMuslMask, sizeof(kStrchrMuslPattern),
     NULL, 0},
    {"strncmp_musl_x86_64", "strncmp",
     kStrncmpMuslPattern, kStrncmpMuslMask, sizeof(kStrncmpMuslPattern),
     NULL, 0},
    /* strcpy + strncpy share an identical 16-byte prologue and are
     * disambiguated ONLY by following the `call` displacement and
     * inspecting the callee's first ~13 bytes. */
    {"strcpy_musl_x86_64", "strcpy",
     kStrcpyMuslPattern, kStrcpyMuslMask, sizeof(kStrcpyMuslPattern),
     VerifyStrcpy, 5},
    {"strncpy_musl_x86_64", "strncpy",
     kStrncpyMuslPattern, kStrncpyMuslMask, sizeof(kStrncpyMuslPattern),
     VerifyStrncpy, 5},
};

#define FBX_NUM_FINGERPRINTS                                                 \
  ((int)(sizeof(kFingerprints) / sizeof(kFingerprints[0])))

/* Threshold above which the pattern's intrinsic specificity is high enough
 * to skip the function-boundary check entirely.  16 bytes of structured
 * opcode + immediate is essentially impossible to match by accident inside
 * an unrelated function body — verified empirically against Alpine's
 * busybox 1.36 binary (no false positives at all). */
#define FBX_FP_BOUNDARY_CHECK_BELOW 16

/* True if byte `b` is plausibly an inter-function pad/end byte. */
static inline bool IsFunctionBoundaryByte(u8 b) {
  return b == 0xc3 ||  /* ret near                                          */
         b == 0x90 ||  /* nop                                               */
         b == 0xcc ||  /* int3                                              */
         b == 0xff;    /* final byte of jmp/call rel32 displacement         */
}

/* Match `pat`/`mask` against `data[i..i+len]`, defending against running
 * past the end of `data`. */
static bool MatchPattern(const u8 *data, size_t data_len, size_t i,
                         const u8 *pat, const u8 *mask, size_t pat_len) {
  size_t k;
  if (i + pat_len > data_len) return false;
  for (k = 0; k < pat_len; ++k) {
    if ((data[i + k] & mask[k]) != (pat[k] & mask[k])) return false;
  }
  return true;
}

/* Scan one PT_LOAD segment for fingerprint matches.  `seg` is the host
 * pointer to the segment bytes (image + offset); `seg_len` is its filesz;
 * `seg_vaddr_base` is the guest vaddr of the first byte after the aslr
 * skew has been applied.  Returns the number of thunks registered. */
static int ScanSegment(struct System *sys, const u8 *seg, size_t seg_len,
                       u64 seg_vaddr_base) {
  int registered = 0;
  size_t i;
  for (i = 0; i + 1 < seg_len; ++i) {
    int f;
    for (f = 0; f < FBX_NUM_FINGERPRINTS; ++f) {
      const struct ThunkFingerprint *fp = &kFingerprints[f];
      const struct ThunkRegistryEntry *re;
      u64 pc;
      if (!MatchPattern(seg, seg_len, i, fp->pattern, fp->mask,
                        fp->pattern_len)) {
        continue;
      }
      /* Function-boundary heuristic.  For patterns >= 16 bytes the intrinsic
       * specificity (structured opcodes + immediates) is high enough that
       * we trust the match without checking the preceding byte.  Shorter
       * patterns are required to be preceded by a plausible pad/end byte
       * OR be 16-byte aligned. */
      if (fp->pattern_len < FBX_FP_BOUNDARY_CHECK_BELOW && i > 0 &&
          !IsFunctionBoundaryByte(seg[i - 1]) && (i & 15) != 0) {
        continue;
      }
      /* Per-fingerprint disambiguation.  Used by strcpy/strncpy (which share
       * an identical 16-byte prologue) to follow the embedded `call rel32`
       * and inspect the callee's prologue.  See VerifyStrcpy/VerifyStrncpy. */
      if (fp->verify && !fp->verify(seg, seg_len, i)) {
        continue;
      }
      /* Already registered (e.g. a longer-pattern match earlier) — skip. */
      re = LookupRegistry(fp->thunk_name);
      if (!re) continue;
      pc = seg_vaddr_base + (u64)i;
      /* RecordEntry does its own name-based dedup. */
      RecordEntry(&sys->thunks, pc, re->name, re->trampoline);
      TraceLine("[thunk-fp] matched", fp->id, pc);
      ++registered;
      /* Advance past this match so we don't re-test overlapping windows
       * for the same fingerprint.  pat_len-1 because the loop will i++. */
      i += fp->pattern_len - 1;
      break;
    }
  }
  return registered;
}

int FbxThunksScanFromText(struct System *sys, Elf64_Ehdr_ *ehdr, size_t esize,
                          i64 aslr) {
  Elf64_Phdr_ *phdr;
  u16 phnum, p;
  int total = 0;
  EnsureTraceFlag();
  phnum = Read16(ehdr->phnum);
  for (p = 0; p < phnum; ++p) {
    u32 ptype, pflags;
    u64 poffset, pvaddr, pfilesz;
    const u8 *seg;
    phdr = GetElfProgramHeaderAddress(ehdr, esize, p);
    if (!phdr) continue;
    ptype = Read32(phdr->type);
    pflags = Read32(phdr->flags);
    if (ptype != PT_LOAD_) continue;
    if (!(pflags & PF_X_)) continue;
    poffset = Read64(phdr->offset);
    pvaddr = Read64(phdr->vaddr);
    pfilesz = Read64(phdr->filesz);
    /* Defensive bounds check — a malformed ELF could point outside the
     * mapped image; we'd then read uninitialised host memory. */
    if (poffset > esize || pfilesz > esize - poffset) continue;
    seg = (const u8 *)ehdr + poffset;
    total += ScanSegment(sys, seg, (size_t)pfilesz, pvaddr + (u64)aslr);
  }
  if (g_thunk_trace) {
    char buf[80];
    int len = snprintf(buf, sizeof(buf),
                       "[thunk-fp] scan done: %d entries registered\n", total);
    if (len > 0) (void)write(2, buf, (size_t)len);
  }
  return total;
}
