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
//! Probes for up to 8 IDs are batched: all hashes first, prefetch, then probe.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::{Acquire, Relaxed, Release}};

use super::ids::{Key, long_bytes, short_bytes};

pub struct Tokens<'a> {
    short: Table,
    short_mask: usize,
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
    /// FOREIGN IDs interned so far (written under the lock, read without it by `occupied`).
    foreign_n: std::sync::atomic::AtomicU32,
}

/// Tables sized from the bound on distinct users (`Tokens::with_bound`) only while they stay this
/// small: small inputs then never fall back for an unusual mix of ID kinds.
const BOUND_TABLES_MAX: usize = 64 << 20;

pub const NO_TOKEN: u32 = u32::MAX;

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
    /// SHORT gets next_pow2(2 x users) slots (load 0.155 at p20v5, 0.23 at f60v5) and LONG
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
        let mut short_cap = (users * 2).next_power_of_two().max(1 << 12);
        let mut long_cap = (users / 4).next_power_of_two().max(1 << 10);
        if bound > 0 {
            let (s, l) = ((bound * 2).next_power_of_two(), (bound * 4 / 3 + 1).next_power_of_two());
            if (s + 4 * l) * 8 <= BOUND_TABLES_MAX {
                short_cap = short_cap.max(s);
                long_cap = long_cap.max(l);
            }
        }
        let huge = tables_huge((short_cap + long_cap * 4) * 8);
        Tokens {
            short: Table::new(short_cap, huge),
            short_mask: short_cap - 1,
            long: Table::new(long_cap * 4, huge),
            long_cap,
            long_mask: long_cap - 1,
            foreign: Mutex::new((HashMap::new(), Vec::new())),
            full: AtomicBool::new(false),
            short_used: AtomicU64::new(0),
            long_used: AtomicU64::new(0),
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

    #[inline(always)]
    fn short_token(&self, v: u64, mut i: usize) -> u32 {
        let want = v + 1;
        let start = i;
        loop {
            let cur = self.short[i].load(Relaxed);
            if cur == want {
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
                    Err(now) if now == want => return i as u32,
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
            slot = (mix(a) as usize) & self.short_mask;
            let at = &self.short[slot];
            if W { prefetch_w(at) } else { prefetch(at) }
        } else {
            slot = (mix(a ^ b.rotate_left(29)) as usize) & self.long_mask;
            let at = &self.long[4 * slot];
            if W { prefetch_w(at) } else { prefetch(at) }
        }
        Pending { long, a, b, slot }
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
                slot[k] = (mix(a[k]) as usize) & self.short_mask;
                prefetch(&self.short[slot[k]]);
            } else {
                slot[k] = (mix(a[k] ^ b[k].rotate_left(29)) as usize) & self.long_mask;
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
            Key::Short(v) => self.short_token(v, (mix(v) as usize) & self.short_mask),
            Key::Long(a, b) => self.long_token(a, b, (mix(a ^ b.rotate_left(29)) as usize) & self.long_mask),
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
            (Key::Short(self.short[t].load(Relaxed) - 1), None)
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
