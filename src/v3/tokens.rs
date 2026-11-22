//! User ID to token: one concurrent open-addressing table per key kind, built by hand. A token is
//! the slot index (SHORT slots first, then LONG slots, then FOREIGN interned IDs), so no counter
//! is shared and the key can always be read back from the slot.
//!
//! SHORT slot: one u64 = value + 1 (0 = empty); inserted by compare-and-swap from 0.
//! LONG slot: state word (0 empty, 1 being written, 2 ready) and the two key words; the writer
//! claims the slot by compare-and-swap, stores the words, then publishes with a release store;
//! a reader that sees "being written" waits for "ready" before comparing.
//! FOREIGN: a locked map from bytes to index (rare: dirt only).
//!
//! Probing is linear from a home slot that starts a 64-byte line (8 SHORT slots, 2 LONG slots), so
//! a key is found in one line unless its bucket filled before it; a SHORT hit is taken from one
//! compare of the whole line, everything else (inserts, misses, overflow) from the atomic loop.
//!
//! Probes for up to 8 IDs are batched: all hashes first, prefetch, then probe.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::{Acquire, Relaxed, Release}};

use super::ids::{Key, long_bytes, short_bytes};

pub struct Tokens<'a> {
    short: Table,
    short_mask: usize,
    /// Home slots start a 64-byte line: SHORT at a multiple of 8, LONG at an even slot.
    short_home: usize,
    long_home: usize,
    /// LONG slots, 4 words each: state, a, b, unused (two slots a cache line).
    long: Table,
    long_cap: usize,
    long_mask: usize,
    foreign: Mutex<(HashMap<&'a [u8], u32>, Vec<&'a [u8]>)>,
    /// A table passed its load limit: the run must fall back to v1.
    pub full: AtomicBool,
    short_used: AtomicU64,
    long_used: AtomicU64,
    short_step: u32,
    long_step: u32,
    /// Every token whose person is not itself carries its stamp (set by persons; see `stamp`).
    stamped: AtomicBool,
    /// FOREIGN IDs interned so far (written under the lock, read without it by `occupied`).
    foreign_n: std::sync::atomic::AtomicU32,
    /// AVX-512F and DQ present: `homes8` hashes a whole batch in zmm registers.
    avx512: bool,
}

/// Tables sized from the bound on distinct users (`Tokens::with_bound`) only while they stay this
/// small: small inputs then never fall back for an unusual mix of ID kinds.
const BOUND_TABLES_MAX: usize = 64 << 20;

pub const NO_TOKEN: u32 = u32::MAX;

/// A SHORT slot: bits 0-40 the key (value + 1; value < 2^40), bits 41-63 its person stamp F
/// (`stamp`): 0 = the person is the token itself, 1 to 2^23 - 2 = person + 1, all ones = the person
/// is in the persons map. A LONG slot's fourth word: 0 or person + 1. Every key compare masks.
const KEY: u64 = (1 << 41) - 1;
const STAMP_SHIFT: u32 = 41;
const STAMP_FAR: u64 = (1 << 23) - 1;

/// Inserts are counted per thread and published to the shared counters in steps: one shared
/// atomic increment per insert bounced its cache line between all workers (links and clicks took
/// 90 ms with it, 47 ms without). A table's step is 1/256 of its slots, at most 1024, so the 3/4
/// check lags by at most threads/256 of the table (3% at 8 threads); a probe that wraps the whole
/// table still reports it full. `short_used` and `long_used` are therefore rounded down.
const STEP_MAX: u64 = 1024;

thread_local! {
    static SHORT_NEW: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static LONG_NEW: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// A started lookup (see `Tokens::start`).
#[derive(Clone, Copy, Default, Debug)]
pub struct Pending {
    pub long: bool,
    pub a: u64,
    pub b: u64,
    pub slot: usize,
}

unsafe extern "C" {
    fn madvise(addr: *mut core::ffi::c_void, len: usize, advice: i32) -> i32;
    fn mmap(addr: *mut core::ffi::c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut core::ffi::c_void;
    fn munmap(addr: *mut core::ffi::c_void, len: usize) -> i32;
}
const MADV_HUGEPAGE: i32 = 14;
const MADV_POPULATE_WRITE: i32 = 23;
const HUGE: usize = 2 << 20;

/// A zeroed table of atomics in a mapping of its own, 2 MB aligned, advised for huge pages when
/// asked, and populated before any use (see `prefault`).
///
/// Why its own mapping: every lookup lands on a random line of the table, so on 4 KB pages each
/// one also misses the TLB (measured in the events phase at p20v5 on box 1: 19M page walks for
/// 19.6M lookups, 0.8G cycles of walking, all 4 KB). The global allocator takes huge pages only
/// when the machine has free 2 MB blocks for 0.7 x the input, which a busy machine does not; the
/// table alone needs 96 MB at p20v5 and 192 MB at f60v5, so it asks for itself.
pub struct Table {
    map: *mut u8,
    map_len: usize,
    ptr: *const AtomicU64,
    len: usize,
}

// SAFETY: the mapping is owned by the table; shared access is through atomics only.
unsafe impl Send for Table {}
// SAFETY: as above.
unsafe impl Sync for Table {}

impl Table {
    pub fn new(n: usize, huge: bool) -> Table {
        let bytes = (n * 8).max(8);
        let map_len = bytes + HUGE;
        // SAFETY: a fresh private anonymous mapping (PROT_READ | PROT_WRITE; MAP_PRIVATE |
        // MAP_ANONYMOUS | MAP_NORESERVE); no existing memory is touched.
        let p = unsafe { mmap(std::ptr::null_mut(), map_len, 3, 0x02 | 0x20 | 0x4000, -1, 0) };
        assert!(p as usize != usize::MAX && !p.is_null(), "token table: mmap of {map_len} bytes failed");
        let a = (p as usize).next_multiple_of(HUGE);
        if huge {
            // SAFETY: [a, a + bytes rounded to pages) lies inside the mapping; a hint only.
            unsafe { madvise(a as *mut core::ffi::c_void, bytes.next_multiple_of(4096), MADV_HUGEPAGE) };
        }
        prefault(a as *mut u8, bytes);
        rt::exit::register(a, bytes.next_multiple_of(4096));
        Table { map: p as *mut u8, map_len, ptr: a as *const AtomicU64, len: n }
    }
}

impl std::ops::Deref for Table {
    type Target = [AtomicU64];
    #[inline(always)]
    fn deref(&self) -> &[AtomicU64] {
        // SAFETY: `len` zeroed words at `ptr`, inside the mapping, live as long as the table.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        // SAFETY: the table's own mapping; no reference to it outlives the table.
        unsafe { munmap(self.map as *mut core::ffi::c_void, self.map_len) };
    }
}

/// Huge pages for the tables: rt's verdict for their size (THP mode, and free 2 MB blocks for
/// them, so no fault waits in compaction); RT_HUGE=on|off overrides as for the allocator.
fn tables_huge(bytes: usize) -> bool {
    if cfg!(feature = "measure") {
        match std::env::var("ENGINE_TOK_HUGE").as_deref() {
            Ok("1") => return true,
            Ok("0") => return false,
            _ => {}
        }
    }
    match std::env::var("RT_HUGE").as_deref() {
        Ok("on") => true,
        Ok("off") => false,
        _ => rt::arena::huge_verdict(bytes).use_huge,
    }
}

struct TimeIt(std::time::Instant, usize);
impl Drop for TimeIt {
    fn drop(&mut self) {
        if cfg!(feature = "measure") && std::env::var("ENGINE_TOK_STAT").is_ok() {
            eprintln!("tokens.prefault {} MB in {:.2} ms", self.1 >> 20, self.0.elapsed().as_secs_f64() * 1e3);
        }
    }
}

/// Faults the pages of a fresh zeroed allocation by writing, in 2 MB pieces taken by the engine's
/// pool (8 workers; MADV_POPULATE_WRITE, one write per 4 KB page where the kernel lacks it). Without this the
/// table's first touches were loads: on a huge-page mapping a load maps the shared huge zero page,
/// and the CAS that follows breaks it with a TLB shootdown to every thread (profile: 1.5% of all
/// cycles in smp_call_function inside the links sink), besides faulting at random during inserts.
pub fn prefault(p: *mut u8, len: usize) {
    if len < (1 << 20) {
        return;
    }
    // Measurement switches (feature `measure` only): ENGINE_TOK_PREFAULT=0 skips this,
    // ENGINE_TOK_STAT prints its time, ENGINE_TOK_HUGE=0|1 overrides the huge-page verdict.
    let t0 = std::time::Instant::now();
    if cfg!(feature = "measure") && std::env::var("ENGINE_TOK_PREFAULT").as_deref() == Ok("0") {
        return;
    }
    let _guard = TimeIt(t0, len);
    let base = p as usize;
    let end = base + len;
    // Pieces on 2 MB boundaries (one huge page each), the first and last cut to the allocation.
    let first = (base + 1).next_multiple_of(HUGE);
    let pieces = 1 + end.saturating_sub(first).div_ceil(HUGE);
    crate::par::run(8, pieces, |_| (), |_, k| {
        let lo = if k == 0 { base } else { first + (k - 1) * HUGE };
        let hi = if k == 0 { first.min(end) } else { (lo + HUGE).min(end) };
        if lo >= hi {
            return;
        }
        // Page-align inward for madvise; the bytes outside are written by hand.
        let a = lo.next_multiple_of(4096);
        let ok = a < hi && unsafe { madvise(a as *mut core::ffi::c_void, (hi - a) & !4095, MADV_POPULATE_WRITE) } == 0;
        let mut q = if ok { (hi & !4095).max(a) } else { lo };
        // SAFETY: [lo, hi) lies in the allocation; writing zeros keeps it zeroed.
        unsafe {
            if ok && lo < a {
                std::ptr::write_volatile(lo as *mut u8, 0);
            }
            while q < hi {
                std::ptr::write_volatile(q as *mut u8, 0);
                q += 4096;
            }
        }
    });
}

fn avx512_dq() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512dq")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline(always)]
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetch never faults, whatever the address.
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

/// Prefetch for writing (prefetchw): where a lookup is likely to insert, a line fetched shared
/// would need a second trip for ownership at the CAS. Not for hit-heavy lookups: it takes the line
/// away from every other core that reads it.
#[inline(always)]
fn prefetch_w<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: as above.
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_ET0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

impl<'a> Tokens<'a> {
    /// Tables for about `users` distinct user IDs (load at most 0.75 before falling back).
    /// SHORT gets next_pow2(users) slots (load 0.31 at p20v5, 0.46 at f60v5; a lookup reads one
    /// 64-byte line, its home line, unless all 8 slots before the key are taken) and LONG
    /// next_pow2(users / 4) (0.18, 0.27). Half the SHORT slots, measured interleaved: at p20v5 a
    /// wash (6 ms less prefault, 1 to 3 ms more probing); at f60v5 25 ms slower in all (load 0.46:
    /// links and clicks +11 ms, events +25 ms, prefault -12 ms). Kept at twice.
    pub fn new(users: usize) -> Self {
        Self::with_bound(users, 0)
    }

    /// As `new`, and when `bound` (an upper bound on distinct user IDs, say from line-count bounds
    /// of the three inputs; 0 for none) gives tables of at most 64 MB, both tables are sized from it
    /// instead: SHORT for twice the bound, LONG for 4/3 of it, so even all-LONG IDs stay below the
    /// 3/4 limit. The rate estimate assumes one ID in eight is LONG; the fuzz "volume" shape (20K
    /// one-event LONG users among 61K, 5,000 events) filled LONG at 12K of 16K slots and fell back.
    pub fn with_bound(users: usize, bound: usize) -> Self {
        let mut short_cap = users.next_power_of_two().max(1 << 12);
        let mut long_cap = (users / 4).next_power_of_two().max(1 << 10);
        if bound > 0 {
            let (s, l) = ((bound * 2).next_power_of_two(), (bound * 4 / 3 + 1).next_power_of_two());
            if (s + 4 * l) * 8 <= BOUND_TABLES_MAX {
                short_cap = short_cap.max(s);
                long_cap = long_cap.max(l);
            }
        }
        let tbytes = (short_cap + long_cap * 4) * 8;
        let huge = tables_huge(tbytes) && rt::arena::take_huge(tbytes);
        Tokens {
            short: Table::new(short_cap, huge),
            short_mask: short_cap - 1,
            short_home: (short_cap - 1) & !7,
            long: Table::new(long_cap * 4, huge),
            long_cap,
            long_mask: long_cap - 1,
            long_home: (long_cap - 1) & !1,
            foreign: Mutex::new((HashMap::new(), Vec::new())),
            full: AtomicBool::new(false),
            stamped: AtomicBool::new(false),
            short_used: AtomicU64::new(0),
            long_used: AtomicU64::new(0),
            avx512: avx512_dq(),
            short_step: ((short_cap as u64) >> 8).clamp(1, STEP_MAX) as u32,
            long_step: ((long_cap as u64) >> 8).clamp(1, STEP_MAX) as u32,
            foreign_n: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Whether token `t` names an ID: a filled SHORT slot, a ready LONG slot, or an interned
    /// FOREIGN index. No lock; exact once the parse phases are over (for a sweep over
    /// 0..capacity()).
    #[inline]
    pub fn occupied(&self, t: u32) -> bool {
        let t = t as usize;
        if t < self.short.len() {
            self.short[t].load(Relaxed) != 0
        } else if t < self.short.len() + self.long_cap {
            self.long[4 * (t - self.short.len())].load(Acquire) == 2
        } else {
            t - self.short.len() - self.long_cap < self.foreign_n.load(Acquire) as usize
        }
    }

    pub fn short_used(&self) -> usize {
        self.short_used.load(Relaxed) as usize
    }

    pub fn long_used(&self) -> usize {
        self.long_used.load(Relaxed) as usize
    }

    pub fn short_cap(&self) -> usize {
        self.short.len()
    }

    pub fn long_cap(&self) -> usize {
        self.long_cap
    }

    /// Upper bound of tokens handed out so far (FOREIGN ones are after both tables).
    pub fn capacity(&self) -> usize {
        self.short.len() + self.long_cap + self.foreign.lock().unwrap().1.len()
    }

    /// Token range reserved for FOREIGN IDs starts here.
    pub fn foreign_base(&self) -> u32 {
        (self.short.len() + self.long_cap) as u32
    }

    /// The slot of the home line (8 slots from `home`, a multiple of 8) that holds `want`, if one
    /// does: the line compared at once. Only a hit is taken from it (keys are unique); everything
    /// else, inserts included, goes through the atomic loop.
    #[inline(always)]
    fn line_hit(&self, want: u64, home: usize) -> Option<usize> {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        // SAFETY: home is a multiple of 8 below the table's length (a multiple of 8, the base 2 MB
        // aligned), so both 32-byte loads are aligned and inside the table.
        unsafe {
            use core::arch::x86_64::*;
            let p = self.short.as_ptr().add(home) as *const __m256i;
            let w = _mm256_set1_epi64x(want as i64);
            let k = _mm256_set1_epi64x(KEY as i64);
            let lo = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(_mm256_and_si256(_mm256_load_si256(p), k), w)));
            let hi = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(_mm256_and_si256(_mm256_load_si256(p.add(1)), k), w)));
            let m = (lo | (hi << 4)) as u32;
            if m != 0 { Some(home + m.trailing_zeros() as usize) } else { None }
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            (home..home + 8).find(|&i| self.short[i].load(Relaxed) & KEY == want)
        }
    }

    #[inline(always)]
    fn short_token(&self, v: u64, mut i: usize) -> u32 {
        let want = v + 1;
        if let Some(t) = self.line_hit(want, i) {
            return t as u32;
        }
        let start = i;
        loop {
            let cur = self.short[i].load(Relaxed);
            if cur & KEY == want {
                return i as u32;
            }
            if cur == 0 {
                match self.short[i].compare_exchange(0, want, Relaxed, Relaxed) {
                    Ok(_) => {
                        let step = self.short_step;
                        let n = SHORT_NEW.with(|c| {
                            let n = c.get() + 1;
                            c.set(if n >= step { 0 } else { n });
                            n
                        });
                        if n >= step {
                            let used = self.short_used.fetch_add(u64::from(n), Relaxed) + u64::from(n);
                            if used as usize > self.short.len() / 4 * 3 {
                                self.full.store(true, Relaxed);
                            }
                        }
                        return i as u32;
                    }
                    Err(now) if now & KEY == want => return i as u32,
                    Err(_) => {}
                }
            }
            i = (i + 1) & self.short_mask;
            if i == start {
                // Every slot holds another key: the table is full.
                self.full.store(true, Relaxed);
                return NO_TOKEN;
            }
        }
    }

    #[inline(always)]
    fn long_token(&self, a: u64, b: u64, mut i: usize) -> u32 {
        let start = i;
        loop {
            let mut st = self.long[4 * i].load(Acquire);
            if st == 0 {
                match self.long[4 * i].compare_exchange(0, 1, Acquire, Acquire) {
                    Ok(_) => {
                        self.long[4 * i + 1].store(a, Relaxed);
                        self.long[4 * i + 2].store(b, Relaxed);
                        self.long[4 * i].store(2, Release);
                        let step = self.long_step;
                        let n = LONG_NEW.with(|c| {
                            let n = c.get() + 1;
                            c.set(if n >= step { 0 } else { n });
                            n
                        });
                        if n >= step {
                            let used = self.long_used.fetch_add(u64::from(n), Relaxed) + u64::from(n);
                            if used as usize > self.long_cap / 4 * 3 {
                                self.full.store(true, Relaxed);
                            }
                        }
                        return (self.short.len() + i) as u32;
                    }
                    Err(now) => st = now,
                }
            }
            while st == 1 {
                std::hint::spin_loop();
                st = self.long[4 * i].load(Acquire);
            }
            if self.long[4 * i + 1].load(Relaxed) == a && self.long[4 * i + 2].load(Relaxed) == b {
                return (self.short.len() + i) as u32;
            }
            i = (i + 1) & self.long_mask;
            if i == start {
                self.full.store(true, Relaxed);
                return NO_TOKEN;
            }
        }
    }

    /// Starts a lookup: computes the slot and prefetches its line (for writing when `W`: the
    /// caller expects inserts); `finish` completes it later.
    #[inline(always)]
    pub fn start_hint<const W: bool>(&self, long: bool, a: u64, b: u64) -> Pending {
        let slot;
        if !long {
            slot = (mix(a) as usize) & self.short_home;
            let at = &self.short[slot];
            if W { prefetch_w(at) } else { prefetch(at) }
        } else {
            slot = (mix(a ^ b.rotate_left(29)) as usize) & self.long_home;
            let at = &self.long[4 * slot];
            if W { prefetch_w(at) } else { prefetch(at) }
        }
        Pending { long, a, b, slot }
    }

    /// `home` for lanes 0..n of a batch at once (`long` bit k: lane k is LONG): slots into `out`
    /// (lanes n.. hold garbage), their lines prefetched (for writing with W, as `start_hint`).
    /// With AVX-512 the eight mixes run in zmm
    /// registers (about 30 instructions a batch against about 27 a lane for the scalar loop).
    #[inline(always)]
    pub fn homes8<const W: bool>(&self, long: u8, a: &[u64; 8], b: &[u64; 8], n: usize, out: &mut [u32; 8]) {
        #[cfg(target_arch = "x86_64")]
        if self.avx512 {
            // SAFETY: avx512 only when AVX-512F and DQ were detected (Tokens construction).
            unsafe { self.homes512::<W>(long, a, b, n, out) };
            return;
        }
        for k in 0..n {
            out[k] = self.start_hint::<W>(long >> k & 1 != 0, a[k], b[k]).slot as u32;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn homes512<const W: bool>(&self, long: u8, a: &[u64; 8], b: &[u64; 8], n: usize, out: &mut [u32; 8]) {
        use core::arch::x86_64::*;
        unsafe {
            let va = _mm512_loadu_si512(a.as_ptr() as *const _);
            let vb = _mm512_loadu_si512(b.as_ptr() as *const _);
            // as `home`: the key is a, or a ^ b.rotate_left(29) for LONG lanes; then `mix`
            let x = _mm512_mask_xor_epi64(va, long, va, _mm512_rol_epi64::<29>(vb));
            let mut z = _mm512_add_epi64(x, _mm512_set1_epi64(0x9e37_79b9_7f4a_7c15u64 as i64));
            z = _mm512_mullo_epi64(_mm512_xor_si512(z, _mm512_srli_epi64::<30>(z)), _mm512_set1_epi64(0xbf58_476d_1ce4_e5b9u64 as i64));
            z = _mm512_mullo_epi64(_mm512_xor_si512(z, _mm512_srli_epi64::<27>(z)), _mm512_set1_epi64(0x94d0_49bb_1331_11ebu64 as i64));
            z = _mm512_xor_si512(z, _mm512_srli_epi64::<31>(z));
            let hm = _mm512_mask_blend_epi64(long, _mm512_set1_epi64(self.short_home as i64), _mm512_set1_epi64(self.long_home as i64));
            let slot = _mm512_and_si512(z, hm);
            _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, _mm512_cvtepi64_epi32(slot));
            // line addresses: short + 8 * slot, long + 32 * slot (4 words a LONG slot)
            let off = _mm512_mask_slli_epi64::<5>(_mm512_slli_epi64::<3>(slot), long, slot);
            let bases = _mm512_mask_blend_epi64(long, _mm512_set1_epi64(self.short.as_ptr() as i64), _mm512_set1_epi64(self.long.as_ptr() as i64));
            let mut at = [0u64; 8];
            _mm512_storeu_si512(at.as_mut_ptr() as *mut _, _mm512_add_epi64(bases, off));
            for &p in &at[..n.min(8)] {
                if W {
                    _mm_prefetch(p as *const i8, _MM_HINT_ET0);
                } else {
                    _mm_prefetch(p as *const i8, _MM_HINT_T0);
                }
            }
        }
    }

    /// Home slot of a key (SHORT or LONG), its line prefetched (shared). Branchless.
    #[inline(always)]
    pub fn home(&self, long: bool, a: u64, b: u64) -> u32 {
        let h = mix(if long { a ^ b.rotate_left(29) } else { a }) as usize;
        let slot = h & if long { self.long_home } else { self.short_home };
        let at = if long { self.long.as_ptr().wrapping_add(4 * slot) } else { self.short.as_ptr().wrapping_add(slot) };
        prefetch(at);
        slot as u32
    }

    /// Token of a SHORT key whose home slot is `slot` (see `home`).
    #[inline(always)]
    pub fn short_at(&self, v: u64, slot: u32) -> u32 {
        self.short_token(v, slot as usize)
    }

    /// Token of a LONG key whose home slot is `slot` (see `home`).
    #[inline(always)]
    pub fn long_at(&self, a: u64, b: u64, slot: u32) -> u32 {
        self.long_token(a, b, slot as usize)
    }

    #[inline(always)]
    pub fn start(&self, long: bool, a: u64, b: u64) -> Pending {
        self.start_hint::<false>(long, a, b)
    }

    /// `start` for a Pending that carries only the key.
    #[inline(always)]
    pub fn start_pending(&self, p: Pending) -> Pending {
        self.start(p.long, p.a, p.b)
    }

    #[inline(always)]
    pub fn finish(&self, p: &Pending) -> u32 {
        if !p.long { self.short_token(p.a, p.slot) } else { self.long_token(p.a, p.b, p.slot) }
    }

    /// `finish` without inserting: the token of a started lookup, or NO_TOKEN if the ID is not in
    /// the table. For phases that run after every insert (no writer runs meanwhile).
    #[inline(always)]
    pub fn find(&self, p: &Pending) -> u32 {
        let mut i = p.slot;
        if !p.long {
            let want = p.a + 1;
            if let Some(t) = self.line_hit(want, i) {
                return t as u32;
            }
            loop {
                let cur = self.short[i].load(Relaxed);
                if cur & KEY == want {
                    return i as u32;
                }
                if cur == 0 {
                    return NO_TOKEN;
                }
                i = (i + 1) & self.short_mask;
                if i == p.slot {
                    return NO_TOKEN;
                }
            }
        }
        loop {
            if self.long[4 * i].load(Acquire) == 0 {
                return NO_TOKEN;
            }
            if self.long[4 * i + 1].load(Relaxed) == p.a && self.long[4 * i + 2].load(Relaxed) == p.b {
                return (self.short.len() + i) as u32;
            }
            i = (i + 1) & self.long_mask;
            if i == p.slot {
                return NO_TOKEN;
            }
        }
    }

    /// Token of an interned FOREIGN ID, or NO_TOKEN (no insert).
    pub fn find_foreign(&self, bytes: &[u8]) -> u32 {
        let g = self.foreign.lock().unwrap();
        g.0.get(bytes).map_or(NO_TOKEN, |&i| self.foreign_base() + i)
    }

    /// Tokens of up to 8 SHORT/LONG user IDs: lane i is LONG when bit i of `long` is set; lanes
    /// whose bit in `want` is clear are skipped (their output is NO_TOKEN).
    #[inline(always)]
    pub fn batch(&self, n: usize, want: u8, long: u8, a: &[u64; 8], b: &[u64; 8], out: &mut [u32; 8]) {
        let mut slot = [0usize; 8];
        for k in 0..n {
            if want >> k & 1 == 0 {
                continue;
            }
            if long >> k & 1 == 0 {
                slot[k] = (mix(a[k]) as usize) & self.short_home;
                prefetch(&self.short[slot[k]]);
            } else {
                slot[k] = (mix(a[k] ^ b[k].rotate_left(29)) as usize) & self.long_home;
                prefetch(&self.long[4 * slot[k]]);
            }
        }
        for k in 0..8 {
            out[k] = NO_TOKEN;
        }
        for k in 0..n {
            if want >> k & 1 == 0 {
                continue;
            }
            out[k] = if long >> k & 1 == 0 { self.short_token(a[k], slot[k]) } else { self.long_token(a[k], b[k], slot[k]) };
        }
    }

    /// Token of one ID of any kind (FOREIGN interned by its bytes).
    pub fn token(&self, key: Key, bytes: &'a [u8]) -> u32 {
        match key {
            Key::Short(v) => self.short_token(v, (mix(v) as usize) & self.short_home),
            Key::Long(a, b) => self.long_token(a, b, (mix(a ^ b.rotate_left(29)) as usize) & self.long_home),
            Key::Foreign => {
                let mut g = self.foreign.lock().unwrap();
                if let Some(&i) = g.0.get(bytes) {
                    return self.foreign_base() + i;
                }
                let i = g.1.len() as u32;
                g.1.push(bytes);
                g.0.insert(bytes, i);
                self.foreign_n.store(i + 1, Release);
                self.foreign_base() + i
            }
        }
    }

    /// Records the person of token `t` in its own slot (persons, once per linked token, after the
    /// person is final; no other writer changes an occupied slot). Then `set_stamped`.
    #[inline]
    pub fn stamp(&self, t: u32, person: u32) {
        let t = t as usize;
        if t < self.short.len() {
            let f = if u64::from(person) + 1 < STAMP_FAR { u64::from(person) + 1 } else { STAMP_FAR };
            let w = self.short[t].load(Relaxed) & KEY;
            self.short[t].store(w | (f << STAMP_SHIFT), Relaxed);
        } else if t < self.short.len() + self.long_cap {
            self.long[4 * (t - self.short.len()) + 3].store(u64::from(person) + 1, Relaxed);
        }
    }

    /// Every token whose person is not itself is stamped (persons calls this last; readers that
    /// see it may use `person_at`).
    pub fn set_stamped(&self) {
        self.stamped.store(true, Release);
    }

    pub fn stamped(&self) -> bool {
        self.stamped.load(Acquire)
    }

    /// The person of token `t` from its slot (the line a find just read), or None when it is in
    /// the persons map (FOREIGN tokens, a person too large for the stamp). Only after `stamped`.
    #[inline(always)]
    pub fn person_at(&self, t: u32) -> Option<u32> {
        let i = t as usize;
        if i < self.short.len() {
            let f = self.short[i].load(Relaxed) >> STAMP_SHIFT;
            if f == 0 {
                Some(t)
            } else if f == STAMP_FAR {
                None
            } else {
                Some((f - 1) as u32)
            }
        } else if i < self.short.len() + self.long_cap {
            let v = self.long[4 * (i - self.short.len()) + 3].load(Relaxed);
            Some(if v == 0 { t } else { (v - 1) as u32 })
        } else {
            None
        }
    }

    /// Starts loading the key of a token (prefetch).
    #[inline(always)]
    pub fn prefetch_key(&self, t: u32) {
        let t = t as usize;
        if t < self.short.len() {
            prefetch(&self.short[t]);
        } else if t < self.short.len() + self.long_cap {
            prefetch(&self.long[4 * (t - self.short.len())]);
        }
    }

    /// Kind and key of a token (after parsing; FOREIGN returns its bytes).
    pub fn key_of(&self, t: u32) -> (Key, Option<&'a [u8]>) {
        let t = t as usize;
        if t < self.short.len() {
            (Key::Short((self.short[t].load(Relaxed) & KEY) - 1), None)
        } else if t < self.short.len() + self.long_cap {
            let i = t - self.short.len();
            (Key::Long(self.long[4 * i + 1].load(Relaxed), self.long[4 * i + 2].load(Relaxed)), None)
        } else {
            let g = self.foreign.lock().unwrap_or_else(|e| e.into_inner());
            // An index past the interned IDs (only NO_TOKEN could be) reads as an empty ID; every
            // caller has rejected such tokens before (see `valid`).
            (Key::Foreign, Some(g.1.get(t - self.short.len() - self.long_cap).copied().unwrap_or(&[])))
        }
    }

    /// Appends the bytes of a user token.
    pub fn bytes(&self, t: u32, out: &mut Vec<u8>) {
        match self.key_of(t) {
            (Key::Short(v), _) => short_bytes(b'u', v, out),
            (Key::Long(a, b), _) => long_bytes(b'u', a, b, out),
            (Key::Foreign, Some(s)) => out.extend_from_slice(s),
            (Key::Foreign, None) => unreachable!(),
        }
    }

    /// Byte order of the user IDs behind two tokens.
    pub fn cmp(&self, x: u32, y: u32) -> std::cmp::Ordering {
        if x == y {
            return std::cmp::Ordering::Equal;
        }
        match (self.key_of(x), self.key_of(y)) {
            ((Key::Short(a), _), (Key::Short(b), _)) => super::ids::cmp_short(a, b),
            ((Key::Long(a1, b1), _), (Key::Long(a2, b2), _)) => (a1, b1).cmp(&(a2, b2)),
            _ => {
                let (mut p, mut q) = (Vec::new(), Vec::new());
                self.bytes(x, &mut p);
                self.bytes(y, &mut q);
                p.cmp(&q)
            }
        }
    }
}
