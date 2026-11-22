//! Fast-path memory: one region and four block classes (block-class decision, 2026-11-21).
//!
//! Every partitioned chunk is exactly one block:
//! - `B8K` (8 KB): SHORT event chunks (sink), move-2 survivor chunks (dedup), clicks-to-persons
//!   chunks (driver);
//! - `B2K` (2 KB): LONG event chunks (sink);
//! - `B256` (256 B): purchase chunks of both kinds (sink) and move-2 purchase chunks (dedup);
//! - `B2M` (2 MB): unpartitioned appends: parse-time clicks, edges, specials (sink).
//!
//! Contiguous takes (`REG.take` / `REG.give`) for the token table (whole run), the persons
//! arrays, session scratch and rows.
//!
//! Who frees what: dedup frees each move-1 chunk (B8K, B2K, B256) as soon as its partition is
//! done, onto its own worker's list, and move 2 allocates from that list first; the driver resets
//! B2M after the clicks stage. `alloc(w)` / `free(w, p)` take the pool participant index `w`
//! (the index `crate::par::run` passes to its `init`), unique among concurrent users.
//!
//! Until a stage takes memory from here this costs only address space: `init` reserves without
//! faulting, and `prefault` is called with what the users will certainly take.

use std::sync::atomic::{AtomicBool, Ordering};

use rt::arena::{Blocks, Huge, Region};

pub static REG: Region = Region::new();
pub static B8K: Blocks = Blocks::new();
pub static B2K: Blocks = Blocks::new();
pub static B256: Blocks = Blocks::new();
pub static B2M: Blocks = Blocks::new();

/// Address space reserved for the region (no memory until taken).
const RESERVE: usize = 64 << 30;

static READY: AtomicBool = AtomicBool::new(false);

/// Reserves the region and attaches the block classes; huge pages by rt's verdict for
/// `expected_peak` bytes. Once per process; false if the reservation failed (the run falls back).
pub fn init(expected_peak: usize) -> bool {
    if READY.load(Ordering::Acquire) {
        return true;
    }
    // One initialiser at a time (tests on parallel threads call this concurrently).
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if READY.load(Ordering::Acquire) {
        return true;
    }
    // Huge pages by rt's verdict for the whole peak. When the verdict is no only because the free
    // 2 MB blocks and the free memory are short of the WHOLE peak, the region still takes huge
    // pages for 80% of the blocks that are free now (less a quarter of the peak, left to the
    // allocator), under the stall guard, and 4 KB pages after that. It used to take none: at
    // f60v5 on box 1 (blocks 2.4 GB, MemFree 5.3 GB) that was 544K page faults against 98K and
    // 27.7G cycles against 25.0G. RT_TIGHT=<MB> forces this mode with that limit (measurements;
    // 0 = no huge pages).
    let v = rt::arena::huge_verdict(expected_peak);
    let limit = match std::env::var("RT_TIGHT").ok().and_then(|s| s.parse::<usize>().ok()) {
        Some(mb) => Some(mb << 20),
        None if !v.use_huge && v.enabled != b'n' && v.free_2m > 0 => Some(((v.free_2m as usize) * (2 << 20) / 5 * 4).saturating_sub(expected_peak / 4)),
        None => None,
    };
    let huge = match limit {
        Some(b) => if b > 0 { Huge::On } else { Huge::Off },
        None => if v.use_huge { Huge::On } else { Huge::Off },
    };
    let ok = REG.init(RESERVE, huge).is_ok()
        && B8K.attach(&REG, 8 << 10).is_ok()
        && B2K.attach(&REG, 2 << 10).is_ok()
        && B256.attach(&REG, 256).is_ok()
        && B2M.attach(&REG, 2 << 20).is_ok();
    if let Some(b) = limit.filter(|&b| b > 0) {
        REG.set_huge_limit(b);
    }
    READY.store(ok, Ordering::Release);
    ok
}

/// Persons: the token-indexed person map (read by move 2 and the clicks stage) and the persons
/// build's node arrays, on their own mapping so that the huge-page verdict is asked for their own
/// size (at f60v5 the 75 MB map on 4 KB pages cost about a page walk per lookup). The map is
/// this region's first take: fresh memory, so zero.
pub static PERS: Region = Region::new();

static PERS_READY: AtomicBool = AtomicBool::new(false);

/// Reserves the persons region for about `bytes` (huge pages by the verdict for that size). It is
/// not prefaulted: the worker that builds persons faults it during the event parse, where it is
/// off the critical path (prefaulting it in parallel at start-up cost 12 ms of wall time at
/// f60v5). False if the reservation failed (the run falls back).
pub fn init_persons(bytes: usize) -> bool {
    if PERS_READY.load(Ordering::Acquire) {
        return true;
    }
    // Reservation with room for an estimate that came out low (address space only).
    if PERS.init(bytes.saturating_mul(4).max(64 << 20), Huge::Auto(bytes)).is_err() {
        return false;
    }
    PERS_READY.store(true, Ordering::Release);
    true
}

/// `len` values of a plain integer type in one contiguous take from a region, given back on
/// drop. The content is whatever the memory held (zero when fresh); callers write before reading
/// unless they rely on a fresh take.
pub struct Buf<T: Copy + 'static> {
    p: *mut T,
    len: usize,
    reg: &'static Region,
}

// SAFETY: a Buf owns its memory exclusively, like a Vec.
unsafe impl<T: Copy + Send> Send for Buf<T> {}
unsafe impl<T: Copy + Sync> Sync for Buf<T> {}

impl<T: Copy + 'static> Buf<T> {
    pub fn take(reg: &'static Region, len: usize) -> Buf<T> {
        let bytes = (len * std::mem::size_of::<T>()).max(1);
        let p = reg.take(bytes) as *mut T;
        Buf { p, len, reg }
    }

}

impl<T: Copy + 'static> std::ops::Deref for Buf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: `p` holds `len` values (region memory is always readable and initialised).
        unsafe { std::slice::from_raw_parts(self.p, self.len) }
    }
}

impl<T: Copy + 'static> std::ops::DerefMut for Buf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as above, and the Buf is the only owner.
        unsafe { std::slice::from_raw_parts_mut(self.p, self.len) }
    }
}

impl<T: Copy + 'static> Drop for Buf<T> {
    fn drop(&mut self) {
        self.reg.give(self.p as *mut u8, (self.len * std::mem::size_of::<T>()).max(1));
    }
}
