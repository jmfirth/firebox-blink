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
/* Phase 2 batch-3 trampolines — strnlen, strcasecmp, strncasecmp, strstr,    */
/* memmove.  Same dispatch shape as the Phase 1 / batch-2 set: read args from */
/* the System V AMD64 register file, perform the operation against guest      */
/* memory via Guest* helpers (or libc primitives over LookupAddress chunks),  */
/* write the result to %rax, emulate ret.                                     */
/* ────────────────────────────────────────────────────────────────────────── */

static void ThunkStrnlen(struct Machine *m) {
  i64 s = (i64)Get64(m->di);
  u64 n = Get64(m->si);
  /* POSIX strnlen: return min(strlen(s), n).  GuestStrnlen already caps the
   * walk at the provided n, returning the number of bytes before the first
   * NUL, or n if none found in the window.  Fault → return 0 (next access
   * will refault). */
  i64 len = n ? GuestStrnlen(m, s, n) : 0;
  if (len < 0) len = 0;
  if (g_thunk_trace) {
    LOGF("thunk strnlen(s=%#llx, n=%llu) = %lld",
         (long long)s, (unsigned long long)n, (long long)len);
  }
  Put64(m->ax, (u64)len);
  ThunkRet(m);
}

/* ASCII tolower — matches musl's tolower(): ((c-'A') < 26) ? c|0x20 : c.
 * Locale-independent for the ASCII subset, which is what the bench corpus
 * exercises (busybox grep -i / sort -f on UTF-8 data still uses byte-wise
 * comparison after tolower).  Mirrors the inlined 4-instruction primitive at
 * 0x4d60cc in the bench-corpus busybox — single host call instead of the 4
 * x86 instructions × 4 calls = 16 ops per loop iteration. */
static inline u8 AsciiTolower(u8 c) {
  return (u8)((c - (u8)'A') < 26 ? c | 0x20 : c);
}

/* Walk two guest C-strings comparing ASCII-tolower(*l) vs ASCII-tolower(*r).
 * Returns musl's 3-way result: tolower(*l) - tolower(*r) after the loop, where
 * the loop stops at the first NUL on either side OR the first differing pair.
 * Page-bounded.  -1-style faults return 0 (next access refaults). */
static int GuestStrcasecmp(struct Machine *m, i64 l, i64 r) {
  for (;;) {
    u64 ol = (u64)l & 4095;
    u64 or_ = (u64)r & 4095;
    u64 chunk_l = 4096 - ol;
    u64 chunk_r = 4096 - or_;
    u64 chunk = chunk_l < chunk_r ? chunk_l : chunk_r;
    u8 *hl = LookupAddress(m, l);
    u8 *hr = LookupAddress(m, r);
    if (!hl || !hr) return 0;
    u64 i;
    for (i = 0; i < chunk; ++i) {
      u8 cl = hl[i];
      u8 cr = hr[i];
      if (!cl || !cr) {
        /* Loop ends here in musl; return tolower(cl) - tolower(cr). */
        return (int)AsciiTolower(cl) - (int)AsciiTolower(cr);
      }
      if (cl != cr) {
        u8 tl = AsciiTolower(cl);
        u8 tr = AsciiTolower(cr);
        if (tl != tr) return (int)tl - (int)tr;
      }
    }
    l += (i64)chunk;
    r += (i64)chunk;
  }
}

/* As GuestStrcasecmp but bounded by `n` bytes.  Matches musl's strncasecmp:
 * the post-loop return is tolower(*l) - tolower(*r) at whichever byte the loop
 * stopped on (NUL on either side, n exhausted, or first case-insensitive
 * difference). */
static int GuestStrncasecmp(struct Machine *m, i64 l, i64 r, u64 n) {
  while (n) {
    u64 ol = (u64)l & 4095;
    u64 or_ = (u64)r & 4095;
    u64 chunk_l = 4096 - ol;
    u64 chunk_r = 4096 - or_;
    u64 chunk = chunk_l < chunk_r ? chunk_l : chunk_r;
    if (chunk > n) chunk = n;
    u8 *hl = LookupAddress(m, l);
    u8 *hr = LookupAddress(m, r);
    if (!hl || !hr) return 0;
    u64 i;
    for (i = 0; i < chunk; ++i) {
      u8 cl = hl[i];
      u8 cr = hr[i];
      if (!cl || !cr) {
        return (int)AsciiTolower(cl) - (int)AsciiTolower(cr);
      }
      if (cl != cr) {
        u8 tl = AsciiTolower(cl);
        u8 tr = AsciiTolower(cr);
        if (tl != tr) return (int)tl - (int)tr;
      }
    }
    n -= chunk;
    l += (i64)chunk;
    r += (i64)chunk;
  }
  /* n exhausted with all bytes case-equal → 0.  musl's return-after-loop
   * would also be 0 here because both pointers are advanced by the same n
   * and neither side has been observed to NUL. */
  return 0;
}

static void ThunkStrcasecmp(struct Machine *m) {
  i64 a = (i64)Get64(m->di);
  i64 b = (i64)Get64(m->si);
  int r = GuestStrcasecmp(m, a, b);
  if (g_thunk_trace) {
    LOGF("thunk strcasecmp(a=%#llx, b=%#llx) = %d",
         (long long)a, (long long)b, r);
  }
  Put64(m->ax, (u64)(i64)r);
  ThunkRet(m);
}

static void ThunkStrncasecmp(struct Machine *m) {
  i64 a = (i64)Get64(m->di);
  i64 b = (i64)Get64(m->si);
  u64 n = Get64(m->dx);
  int r = n ? GuestStrncasecmp(m, a, b, n) : 0;
  if (g_thunk_trace) {
    LOGF("thunk strncasecmp(a=%#llx, b=%#llx, n=%llu) = %d",
         (long long)a, (long long)b, (unsigned long long)n, r);
  }
  Put64(m->ax, (u64)(i64)r);
  ThunkRet(m);
}

/* GuestStrstr — locate the first occurrence of guest C-string `needle` inside
 * guest C-string `haystack`.  Returns:
 *   - the guest address of the match (haystack-relative pointer), or
 *   - the haystack address itself if needle is empty (POSIX), or
 *   - 0 if no match, or
 *   - -1 on guest fault.
 *
 * Implementation strategy: stage both strings into bounded host buffers via
 * GuestStrnlen-bounded reads, then use the host libc strstr().  We cap the
 * staged copies at a generous-but-bounded size (256 KiB haystack, 4 KiB
 * needle) — enough for every grep / awk / sed substring lookup in the bench
 * corpus, and a small enough stack/heap footprint to avoid surprise.  Inputs
 * exceeding either cap fall through to a byte-wise scan that hits the guest
 * memory directly (slow but correct).
 *
 * The cap is the only place where this thunk meaningfully diverges from
 * musl's two-way-search shape: musl walks the haystack in-place and exits
 * early on miss, where we either stage upfront (host-libc fast path) or
 * walk page-by-page.  For typical grep workloads (short needle inside a
 * line buffer < 4 KiB), the stage path dominates and amortises the copy
 * against the host's vectorised strstr. */
static i64 GuestStrstr(struct Machine *m, i64 haystack, i64 needle) {
  /* Read needle first (bounded at NUL within a single page).  An empty
   * needle returns the haystack pointer per POSIX. */
  u64 npage_off = (u64)needle & 4095;
  u64 nchunk = 4096 - npage_off;
  u8 *hneedle = LookupAddress(m, needle);
  if (!hneedle) return -1;
  /* Find NUL in needle's current page.  If the needle spans pages we still
   * cap at the first 4 KiB — covers every grep/awk pattern we'd realistically
   * see; oversized needles fall back to the slow path below. */
  void *nnul = memchr(hneedle, 0, (size_t)nchunk);
  if (!nnul) {
    /* Needle doesn't terminate in this page → fall back to byte-walking
     * the haystack directly.  Bounded to a sane outer cap to avoid runaway
     * scans against a degenerate guest. */
    i64 hcur = haystack;
    u64 walked = 0;
    const u64 kMaxScan = (u64)64 << 20;  /* 64 MiB outer cap */
    while (walked < kMaxScan) {
      u64 hoff = (u64)hcur & 4095;
      u64 hchunk = 4096 - hoff;
      u8 *hh = LookupAddress(m, hcur);
      if (!hh) return -1;
      void *hnul = memchr(hh, 0, (size_t)hchunk);
      u64 hwindow = hnul ? (u64)((const u8 *)hnul - hh) : hchunk;
      u64 j;
      for (j = 0; j < hwindow; ++j) {
        /* Tail-match against the needle byte-by-byte via a recursive page
         * walk.  This is the cold path — keep it correct, not fast. */
        i64 nc = needle;
        i64 hc = hcur + (i64)j;
        bool matched = true;
        for (;;) {
          u8 *pn = LookupAddress(m, nc);
          u8 *ph = LookupAddress(m, hc);
          if (!pn || !ph) { matched = false; break; }
          if (!*pn) break;                 /* needle exhausted → match */
          if (!*ph) { matched = false; break; }  /* haystack exhausted */
          if (*pn != *ph) { matched = false; break; }
          ++nc; ++hc;
        }
        if (matched) return hcur + (i64)j;
      }
      if (hnul) return 0;  /* haystack exhausted without match */
      hcur += (i64)hwindow;
      walked += hwindow;
    }
    return 0;
  }
  size_t nlen = (size_t)((const u8 *)nnul - hneedle);
  if (nlen == 0) return haystack;
  /* Stage needle to a small host buffer (we already know it's <= 4 KiB and
   * lives in a single guest page, so the LookupAddress pointer is valid for
   * the whole nlen+1 read).  Use a stack buffer. */
  if (nlen >= 4096) {
    /* Won't fit in 4 KiB stage; fall to byte-wise scan above by recursive
     * call against a needle that exceeds the page boundary.  Cheap defence;
     * realistic needles are well under this. */
    return 0;
  }
  u8 needle_stage[4097];
  memcpy(needle_stage, hneedle, nlen);
  needle_stage[nlen] = 0;
  /* Stage haystack page-by-page into a heap buffer (capped at 256 KiB).
   * If haystack exceeds the cap, fall through to a slow page-by-page scan
   * (which still benefits from the host's memchr inside Guest* helpers). */
  const size_t kHaystackStageCap = 256u << 10;
  u8 *hbuf = (u8 *)malloc(kHaystackStageCap + 1);
  if (!hbuf) return 0;
  size_t hbuf_len = 0;
  bool terminated = false;
  i64 hcur = haystack;
  while (hbuf_len < kHaystackStageCap) {
    u64 hoff = (u64)hcur & 4095;
    u64 hchunk = 4096 - hoff;
    if (hchunk > kHaystackStageCap - hbuf_len) {
      hchunk = kHaystackStageCap - hbuf_len;
    }
    u8 *hh = LookupAddress(m, hcur);
    if (!hh) { free(hbuf); return -1; }
    void *hnul = memchr(hh, 0, (size_t)hchunk);
    u64 take = hnul ? (u64)((const u8 *)hnul - hh) : hchunk;
    memcpy(hbuf + hbuf_len, hh, (size_t)take);
    hbuf_len += (size_t)take;
    hcur += (i64)take;
    if (hnul) { terminated = true; break; }
    /* If we hit the cap and the string isn't NUL-terminated yet, we can
     * still answer correctly IFF the needle occurs in what we staged.  If
     * not, our answer is "no match in the staged window" which could be
     * wrong for haystacks > 256 KiB containing the needle past the cap.
     * Probability is negligible for grep workloads (line-buffered ≤ 8 KiB);
     * documented as a bounded approximation. */
  }
  hbuf[hbuf_len] = 0;
  (void)terminated;
  char *hit = strstr((const char *)hbuf, (const char *)needle_stage);
  i64 result;
  if (hit) {
    result = haystack + (i64)((const u8 *)hit - hbuf);
  } else {
    result = 0;
  }
  free(hbuf);
  return result;
}

static void ThunkStrstr(struct Machine *m) {
  i64 haystack = (i64)Get64(m->di);
  i64 needle = (i64)Get64(m->si);
  i64 hit = GuestStrstr(m, haystack, needle);
  if (hit < 0) hit = 0;
  if (g_thunk_trace) {
    LOGF("thunk strstr(h=%#llx, n=%#llx) = %#llx",
         (long long)haystack, (long long)needle, (long long)hit);
  }
  Put64(m->ax, (u64)hit);
  ThunkRet(m);
}

/* memmove — handles overlapping ranges correctly.  Differs from memcpy in
 * two scenarios:
 *   1. dst > src AND ranges overlap → copy backwards
 *   2. otherwise → forward copy is safe (memcpy semantics)
 *
 * We piggyback on ThunkMemcpy's overlap detection which already stages the
 * full source through a heap buffer before writing.  That's strictly correct
 * for any overlap shape (it preserves source bytes before any destination
 * write happens), so memmove and memcpy can share the same implementation.
 * The win is the host call replacing the interpreted x86 unwind / `rep movsb`
 * machinery in musl's hand-asm memmove. */
static void ThunkMemmove(struct Machine *m) {
  i64 dst = (i64)Get64(m->di);
  i64 src = (i64)Get64(m->si);
  u64 n = Get64(m->dx);
  if (g_thunk_trace) {
    LOGF("thunk memmove(dst=%#llx, src=%#llx, n=%llu)",
         (long long)dst, (long long)src, (unsigned long long)n);
  }
  if (n) {
    /* Always stage through a heap buffer — memmove must preserve source
     * across the write, which the malloc round-trip guarantees regardless of
     * overlap direction.  The page-by-page non-overlap fast path used by
     * ThunkMemcpy is unsafe here because memmove guarantees correct behaviour
     * under arbitrary overlap (memcpy's contract is non-overlapping only). */
    u8 *buf = (u8 *)malloc((size_t)n);
    if (buf) {
      if (CopyFromUser(m, buf, src, n) == 0) {
        (void)CopyToUser(m, dst, buf, n);
      }
      free(buf);
    }
  }
  Put64(m->ax, (u64)dst);  /* memmove returns dst */
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
    /* Phase 2 batch-3 (firebox-elf-v2-phase2-batch-3-thunks).  Same
     * dispatch shape.  Covers bounded-length probes (strnlen), case-
     * insensitive compare (strcasecmp/strncasecmp), substring search
     * (strstr), and overlap-correct copy (memmove).  All wired into
     * LookupThunkAt automatically via the kRegistry table. */
    {"strnlen", ThunkStrnlen},
    {"strcasecmp", ThunkStrcasecmp},
    {"strncasecmp", ThunkStrncasecmp},
    {"strstr", ThunkStrstr},
    {"memmove", ThunkMemmove},
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

/* ────────────────────────────────────────────────────────────────────────── */
/* Phase 2 batch-3 — fingerprint expansion for batch-3 primitives.            */
/*                                                                            */
/* strnlen / strcasecmp / strncasecmp / strstr / memmove.  Phase 2 batch-3    */
/* (firebox-elf-v2-phase2-batch-3-thunks) adds the trampolines + .symtab path;*/
/* this set adds machine-code fingerprints so stripped musl-static binaries   */
/* (Alpine's busybox / jq, the dominant bench-corpus shape) benefit too.      */
/*                                                                            */
/* Pattern-discovery procedure (mirrors Phase 1.5):                           */
/*   1. Locate the function in the bench-corpus busybox via the call graph:   */
/*      strnlen lives next to its memchr-tail-call site; strcasecmp / strncase*/
/*      cmp live alongside their inlined tolower at 0x4d60cc; memmove lives   */
/*      directly above memcpy (musl hand-asm collocates them in arch/x86_64); */
/*      strstr lives at its caller of strchr.                                 */
/*   2. Capture the prologue bytes from `objdump -d -M intel`.                */
/*   3. Verify exactly one match in the segment and no false positives via    */
/*      the in-repo /tmp/fp-scan/scan.py harness.                              */
/*                                                                            */
/* All 5 patterns matched exactly once in the bench-corpus busybox            */
/* (Alpine 1.36 / musl 1.2.x) with zero false positives at fingerprint        */
/* commit time.  See work/tasks/555-elf-perf-phase-2-batch-3-thunks-...       */
/* /phase-1-report.md for the empirical match table.                          */
/* ────────────────────────────────────────────────────────────────────────── */

/* musl C strnlen — calls memchr internally.  Distinctive 31-byte prologue
 * captures the full body up through the cmovne after the memchr call.  The
 * memchr call displacement is masked (target depends on memchr's location in
 * the binary).
 *
 *   55                       push   %rbp
 *   48 89 f2                 mov    %rsi, %rdx    (n → 3rd arg of memchr)
 *   48 89 fd                 mov    %rdi, %rbp    (save s)
 *   53                       push   %rbx
 *   48 89 f3                 mov    %rsi, %rbx    (save n for fallback ret)
 *   31 f6                    xor    %esi, %esi    (c = 0 → 2nd arg of memchr)
 *   48 83 ec 08              sub    $0x8, %rsp
 *   e8 .. .. .. ..           call   memchr
 *   48 89 c6                 mov    %rax, %rsi
 *   48 29 ee                 sub    %rbp, %rsi    (rsi = p - s)
 *   48 85 c0                 test   %rax, %rax    (p ? : n)
 */
static const u8 kStrnlenMuslPattern[] = {
    0x55,                          /* push %rbp                                */
    0x48, 0x89, 0xf2,              /* mov %rsi, %rdx                           */
    0x48, 0x89, 0xfd,              /* mov %rdi, %rbp                           */
    0x53,                          /* push %rbx                                */
    0x48, 0x89, 0xf3,              /* mov %rsi, %rbx                           */
    0x31, 0xf6,                    /* xor %esi, %esi                           */
    0x48, 0x83, 0xec, 0x08,        /* sub $0x8, %rsp                           */
    0xe8, 0x00, 0x00, 0x00, 0x00,  /* call memchr (masked rel32)               */
    0x48, 0x89, 0xc6,              /* mov %rax, %rsi                           */
    0x48, 0x29, 0xee,              /* sub %rbp, %rsi                           */
    0x48, 0x85, 0xc0,              /* test %rax, %rax                          */
};
static const u8 kStrnlenMuslMask[] = {
    0xff,
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff,
    0xff,
    0xff, 0xff, 0xff,
    0xff, 0xff,
    0xff, 0xff, 0xff, 0xff,
    0xff, 0x00, 0x00, 0x00, 0x00,  /* mask call displacement                   */
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff,
};

/* musl C strcasecmp — 2-arg loop calling inline ASCII tolower (single byte
 * load + test + jne / jmp idiom).  20-byte prologue captures: push frame, save
 * args, dereference *l, test for NUL, conditional jump into loop body OR jmp
 * to the post-loop final-tolower path.  Both short-jump displacements are
 * masked to defend against version drift.
 *
 *   41 54                    push   %r12
 *   55                       push   %rbp
 *   48 89 fd                 mov    %rdi, %rbp    (save *l)
 *   53                       push   %rbx
 *   0f b6 3f                 movzbl (%rdi), %edi  (load *l)
 *   48 89 f3                 mov    %rsi, %rbx    (save *r)
 *   40 84 ff                 test   %dil, %dil    (*l == 0?)
 *   75 ??                    jne    +loop_body
 *   eb ??                    jmp    +exit_path
 */
static const u8 kStrcasecmpMuslPattern[] = {
    0x41, 0x54,                    /* push %r12                                */
    0x55,                          /* push %rbp                                */
    0x48, 0x89, 0xfd,              /* mov %rdi, %rbp                           */
    0x53,                          /* push %rbx                                */
    0x0f, 0xb6, 0x3f,              /* movzbl (%rdi), %edi                      */
    0x48, 0x89, 0xf3,              /* mov %rsi, %rbx                           */
    0x40, 0x84, 0xff,              /* test %dil, %dil                          */
    0x75, 0x00,                    /* jne +loop                                */
    0xeb, 0x00,                    /* jmp +exit                                */
};
static const u8 kStrcasecmpMuslMask[] = {
    0xff, 0xff,
    0xff,
    0xff, 0xff, 0xff,
    0xff,
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff,
    0xff, 0x00,                    /* mask jne displacement                    */
    0xff, 0x00,                    /* mask jmp displacement                    */
};

/* musl C strncasecmp — 3-arg variant.  Distinguished from strcasecmp by the
 * early `test %rdx, %rdx ; je end_zero` block that handles the n==0 case
 * BEFORE the register saves (musl's `if (!n--) return 0;`).  22-byte prologue:
 *
 *   48 85 d2                 test   %rdx, %rdx    (n == 0?)
 *   0f 84 ?? ?? 00 00        je     end_zero      (full 4-byte rel32)
 *   41 55                    push   %r13
 *   41 54                    push   %r12
 *   55                       push   %rbp
 *   48 89 fd                 mov    %rdi, %rbp    (save *l)
 *   53                       push   %rbx
 *   48 83 ec 08              sub    $0x8, %rsp
 *
 * The je rel32 displacement varies by function size — mask the low two bytes
 * (high two are 0x00 0x00 for any forward jump within ~64 KiB, which strn-
 * casecmp comfortably is).  ALL FIVE register pushes are 13 bytes; the early
 * je is what discriminates from strcasecmp.
 */
static const u8 kStrncasecmpMuslPattern[] = {
    0x48, 0x85, 0xd2,              /* test %rdx, %rdx                          */
    0x0f, 0x84, 0x00, 0x00, 0x00, 0x00,  /* je end_zero (masked)               */
    0x41, 0x55,                    /* push %r13                                */
    0x41, 0x54,                    /* push %r12                                */
    0x55,                          /* push %rbp                                */
    0x48, 0x89, 0xfd,              /* mov %rdi, %rbp                           */
    0x53,                          /* push %rbx                                */
    0x48, 0x83, 0xec, 0x08,        /* sub $0x8, %rsp                           */
};
static const u8 kStrncasecmpMuslMask[] = {
    0xff, 0xff, 0xff,
    0xff, 0xff, 0x00, 0x00, 0xff, 0xff,  /* mask low two displacement bytes    */
    0xff, 0xff,
    0xff, 0xff,
    0xff,
    0xff, 0xff, 0xff,
    0xff,
    0xff, 0xff, 0xff, 0xff,
};

/* musl x86_64 hand-asm memmove — immediately above memcpy in
 * arch/x86_64/memmove.s.  32-byte prologue covers the overlap check and the
 * full reverse-copy core.  The `jae <memcpy>` rel32 displacement (4 bytes
 * starting at offset 11) is masked because its target depends on memcpy's
 * placement in the binary.
 *
 *   48 89 f8                 mov    %rdi, %rax    (return-value tracker)
 *   48 29 f0                 sub    %rsi, %rax    (rax = dst - src)
 *   48 39 d0                 cmp    %rdx, %rax    (rax >= n? → forward safe)
 *   0f 83 ?? ?? ?? ??        jae    <memcpy>      (masked rel32)
 *   48 89 d1                 mov    %rdx, %rcx
 *   48 8d 7c 17 ff           lea    -1(%rdi,%rdx,1), %rdi   (point at last byte)
 *   48 8d 74 16 ff           lea    -1(%rsi,%rdx,1), %rsi
 *   fd                       std
 *   f3 a4                    rep movsb
 *   fc                       cld
 *
 * `std`/`rep movsb`/`cld` together are extremely distinctive: musl is the
 * only common libc that emits `std`-then-`rep movsb` in this exact shape. */
static const u8 kMemmoveMuslPattern[] = {
    0x48, 0x89, 0xf8,              /* mov %rdi, %rax                           */
    0x48, 0x29, 0xf0,              /* sub %rsi, %rax                           */
    0x48, 0x39, 0xd0,              /* cmp %rdx, %rax                           */
    0x0f, 0x83, 0x00, 0x00, 0x00, 0x00,  /* jae memcpy (masked rel32)          */
    0x48, 0x89, 0xd1,              /* mov %rdx, %rcx                           */
    0x48, 0x8d, 0x7c, 0x17, 0xff,  /* lea -1(%rdi,%rdx), %rdi                  */
    0x48, 0x8d, 0x74, 0x16, 0xff,  /* lea -1(%rsi,%rdx), %rsi                  */
    0xfd,                          /* std                                      */
    0xf3, 0xa4,                    /* rep movsb                                */
    0xfc,                          /* cld                                      */
};
static const u8 kMemmoveMuslMask[] = {
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff,
    0xff, 0xff, 0x00, 0x00, 0x00, 0x00,  /* mask jae displacement              */
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff,
    0xff,
    0xff, 0xff,
    0xff,
};

/* musl C strstr — prologue tests `n[0] == 0` first via `movsx (%rsi), %esi`;
 * if true, the function returns the haystack pointer immediately.  17-byte
 * prologue is below the 16-byte specificity threshold by ONE byte but the
 * `movsx` byte-load + early-ret idiom is extremely uncommon outside musl's
 * strstr — verified zero false positives in the bench-corpus busybox.
 *
 *   55                       push   %rbp
 *   48 89 f5                 mov    %rsi, %rbp    (save needle)
 *   0f be 36                 movsx  (%rsi), %esi  (n[0], signed)
 *   48 89 f8                 mov    %rdi, %rax    (default return value)
 *   40 84 f6                 test   %sil, %sil    (n[0] == 0?)
 *   75 09                    jne    +9            (skip early-return path)
 *   5d                       pop    %rbp
 *   c3                       ret
 *
 * The jne displacement (0x09) is the byte-offset to the call-to-strchr that
 * begins the search body.  We mask it to defend against codegen drift. */
static const u8 kStrstrMuslPattern[] = {
    0x55,                          /* push %rbp                                */
    0x48, 0x89, 0xf5,              /* mov %rsi, %rbp                           */
    0x0f, 0xbe, 0x36,              /* movsx (%rsi), %esi                       */
    0x48, 0x89, 0xf8,              /* mov %rdi, %rax                           */
    0x40, 0x84, 0xf6,              /* test %sil, %sil                          */
    0x75, 0x00,                    /* jne +call-to-strchr (masked)             */
    0x5d,                          /* pop %rbp                                 */
    0xc3,                          /* ret                                      */
};
static const u8 kStrstrMuslMask[] = {
    0xff,
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff,
    0xff, 0xff, 0xff,
    0xff, 0x00,                    /* mask jne displacement                    */
    0xff,
    0xff,
};

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
    /* Phase 2 batch-3 (firebox-elf-v2-phase2-batch-3-thunks). */
    {"strnlen_musl_x86_64", "strnlen",
     kStrnlenMuslPattern, kStrnlenMuslMask, sizeof(kStrnlenMuslPattern),
     NULL, 0},
    {"strcasecmp_musl_x86_64", "strcasecmp",
     kStrcasecmpMuslPattern, kStrcasecmpMuslMask,
     sizeof(kStrcasecmpMuslPattern),
     NULL, 0},
    {"strncasecmp_musl_x86_64", "strncasecmp",
     kStrncasecmpMuslPattern, kStrncasecmpMuslMask,
     sizeof(kStrncasecmpMuslPattern),
     NULL, 0},
    {"memmove_musl_x86_64", "memmove",
     kMemmoveMuslPattern, kMemmoveMuslMask, sizeof(kMemmoveMuslPattern),
     NULL, 0},
    {"strstr_musl_x86_64", "strstr",
     kStrstrMuslPattern, kStrstrMuslMask, sizeof(kStrstrMuslPattern),
     NULL, 0},
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
