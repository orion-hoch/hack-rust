//! Arenas: memory reserved once, faulted once, recycled between stages, never returned to the
//! kernel during the run.
//!
//! - `Region`: one MAP_NORESERVE reservation, handed out in 2 MB chunks. Freed chunks go to a
//!   sorted, coalescing free list and are reused (lowest address first) before fresh memory is
//!   cut from the top, so a stage that frees what an earlier stage used causes no new faults.
//!   The huge-page switch is per region (MADV_HUGEPAGE / MADV_NOHUGEPAGE / leave the default).
//! - `Bump`: per-worker bump allocation over region chunks; `reset` returns the chunks.
//! - `Blocks`: fixed-size blocks (power of two, 4 KB to 2 MB) with a per-worker free list and a
//!   shared overflow list; recycled blocks are preferred to fresh ones.
//! - `huge_verdict`: what the machine says about huge pages right now (THP mode, free 2 MB blocks
//!   from /proc/buddyinfo), read in about 20 us, and `Region::probe_huge` to confirm on the first
//!   chunk.
//!
//! Locks: the region free list and the blocks overflow list take a spinlock, once per 2 MB chunk
//! or per batch of blocks; nothing per allocation.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering::*};

use crate::pool::MAX_WORKERS;
use crate::sys::{self, Errno, HUGE, PAGE, Res};

pub const CHUNK: usize = HUGE;
const FREE_MAX: usize = 1024;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Huge {
    /// No madvise: the system default (THP "madvise" mode means 4 KB pages).
    Default,
    On,
    Off,
    /// `huge_verdict(need)` decides On or Off at `init` (need = expected peak bytes).
    Auto(usize),
}

/// A single 2 MB fault slower than this during `prefault` counts as a compaction stall
/// (a normal one measured 0.4 to 0.6 ms with 8 threads faulting at once on the x86 box).
pub const STALL_NS: u64 = 2_000_000;
/// Slow faults before the guard gives up huge pages: one slow fault happens on a loaded host
/// without compaction (seen once at 2 GB with compact_stall +0); compaction gives many in a row.
pub const STALL_STRIKES: u32 = 3;

struct SpinLock(AtomicU32);

impl SpinLock {
    const fn new() -> SpinLock {
        SpinLock(AtomicU32::new(0))
    }
    #[inline]
    fn lock(&self) {
        while self.0.compare_exchange_weak(0, 1, Acquire, Relaxed).is_err() {
            while self.0.load(Relaxed) != 0 {
                core::hint::spin_loop();
            }
        }
    }
    #[inline]
    fn unlock(&self) {
        self.0.store(0, Release);
    }
}

struct FreeList {
    n: usize,
    /// (first chunk, chunk count), sorted by first chunk, never adjacent.
    r: [(u32, u32); FREE_MAX],
}

pub struct Region {
    base: AtomicUsize,
    cap: AtomicUsize,
    top: AtomicUsize,
    lock: SpinLock,
    free: UnsafeCell<FreeList>,
    live: AtomicUsize,
    peak: AtomicUsize,
    huge: AtomicU32,
    stalled: AtomicU32,
    slow: AtomicU32,
    /// compact_stall at `init` (u64::MAX: unreadable).
    stall0: core::sync::atomic::AtomicU64,
}

unsafe impl Sync for Region {}
unsafe impl Send for Region {}

static REGIONS: [AtomicUsize; 16] = [const { AtomicUsize::new(0) }; 16];

/// Regions registered for `exit::leave`.
pub(crate) fn registered(i: usize) -> Option<&'static Region> {
    let p = REGIONS.get(i)?.load(Acquire);
    if p == 0 { None } else { Some(unsafe { &*(p as *const Region) }) }
}

impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}

impl Region {
    pub const fn new() -> Region {
        Region {
            base: AtomicUsize::new(0),
            cap: AtomicUsize::new(0),
            top: AtomicUsize::new(0),
            lock: SpinLock::new(),
            free: UnsafeCell::new(FreeList { n: 0, r: [(0, 0); FREE_MAX] }),
            live: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            huge: AtomicU32::new(0),
            stalled: AtomicU32::new(0),
            slow: AtomicU32::new(0),
            stall0: core::sync::atomic::AtomicU64::new(u64::MAX),
        }
    }

    /// Reserves `cap` bytes (rounded to 2 MB, 2 MB aligned) and registers the region for exit.
    pub fn init(&'static self, cap: usize, huge: Huge) -> Res<()> {
        let cap = sys::up(cap.max(CHUNK), CHUNK);
        let raw = sys::map_anon(cap + CHUNK)? as usize;
        let base = sys::up(raw, CHUNK);
        let huge = match huge {
            Huge::Auto(need) => {
                if huge_verdict(need).use_huge { Huge::On } else { Huge::Off }
            }
            h => h,
        };
        let (adv, code) = match huge {
            Huge::On => (Some(sys::MADV_HUGEPAGE), 1),
            Huge::Off => (Some(sys::MADV_NOHUGEPAGE), 2),
            _ => (None, 0),
        };
        if let Some(a) = adv {
            unsafe { sys::madvise(base, cap, a)? };
        }
        if code == 1 {
            self.stall0.store(compact_stalls().unwrap_or(u64::MAX), Relaxed);
        }
        self.base.store(base, Relaxed);
        self.cap.store(cap, Relaxed);
        self.huge.store(code, Relaxed);
        for r in REGIONS.iter() {
            if r.compare_exchange(0, self as *const Region as usize, AcqRel, Relaxed).is_ok() {
                break;
            }
        }
        Ok(())
    }

    #[inline]
    pub fn base(&self) -> usize {
        self.base.load(Relaxed)
    }
    pub fn cap(&self) -> usize {
        self.cap.load(Relaxed)
    }
    /// Bytes ever cut from the top: an upper bound on memory this region has faulted.
    pub fn high_water(&self) -> usize {
        self.top.load(Relaxed)
    }
    /// Bytes handed out now, and the most ever handed out at once.
    pub fn live(&self) -> (usize, usize) {
        (self.live.load(Relaxed), self.peak.load(Relaxed))
    }

    /// `bytes` rounded up to whole chunks, from the free list (first fit) or the top.
    pub fn take(&self, bytes: usize) -> *mut u8 {
        let n = (sys::up(bytes.max(1), CHUNK) / CHUNK) as u32;
        let base = self.base();
        let mut got = None;
        self.lock.lock();
        unsafe {
            let fl = &mut *self.free.get();
            for i in 0..fl.n {
                let (s, c) = fl.r[i];
                if c >= n {
                    got = Some(s);
                    if c == n {
                        fl.r.copy_within(i + 1..fl.n, i);
                        fl.n -= 1;
                    } else {
                        fl.r[i] = (s + n, c - n);
                    }
                    break;
                }
            }
        }
        self.lock.unlock();
        let bytes = n as usize * CHUNK;
        let p = match got {
            Some(s) => base + s as usize * CHUNK,
            None => {
                let off = self.top.fetch_add(bytes, AcqRel);
                if off + bytes > self.cap() {
                    crate::fail::fail(b"arena: region exhausted");
                }
                base + off
            }
        };
        let l = self.live.fetch_add(bytes, Relaxed) + bytes;
        self.peak.fetch_max(l, Relaxed);
        p as *mut u8
    }

    /// Returns chunks from `take` (same pointer and byte count).
    pub fn give(&self, p: *mut u8, bytes: usize) {
        let n = (sys::up(bytes.max(1), CHUNK) / CHUNK) as u32;
        let s = ((p as usize - self.base()) / CHUNK) as u32;
        self.lock.lock();
        unsafe {
            let fl = &mut *self.free.get();
            let mut i = 0;
            while i < fl.n && fl.r[i].0 < s {
                i += 1;
            }
            let joins_prev = i > 0 && fl.r[i - 1].0 + fl.r[i - 1].1 == s;
            let joins_next = i < fl.n && s + n == fl.r[i].0;
            match (joins_prev, joins_next) {
                (true, true) => {
                    fl.r[i - 1].1 += n + fl.r[i].1;
                    fl.r.copy_within(i + 1..fl.n, i);
                    fl.n -= 1;
                }
                (true, false) => fl.r[i - 1].1 += n,
                (false, true) => fl.r[i] = (s, n + fl.r[i].1),
                (false, false) => {
                    if fl.n == FREE_MAX {
                        self.lock.unlock();
                        crate::fail::fail(b"arena: free list full");
                    }
                    fl.r.copy_within(i..fl.n, i + 1);
                    fl.r[i] = (s, n);
                    fl.n += 1;
                }
            }
        }
        self.lock.unlock();
        self.live.fetch_sub(n as usize * CHUNK, Relaxed);
    }

    /// Faults `bytes` of fresh memory now (so a later hot stage does not), split over the pool's
    /// workers, by MADV_POPULATE_WRITE (`populate`, 41 ms/GB measured with 4 KB pages) or one
    /// write per page (62 to 66 ms/GB). The memory goes to the free list, first in line for
    /// `take`.
    ///
    /// Stall guard: on a huge-page region every 2 MB fault is timed; the first one slower than
    /// `STALL_NS` while the kernel's compact_stall counter has risen (direct compaction) switches
    /// the whole region to MADV_NOHUGEPAGE, so later
    /// faults take 4 KB pages at once. `stalled()` tells whether that happened.
    pub fn prefault(&self, bytes: usize, populate: bool) {
        let bytes = sys::up(bytes, CHUNK);
        if bytes == 0 {
            return;
        }
        let p = self.take_fresh(bytes) as usize;
        let nw = crate::pool::workers().max(1);
        let parts = (bytes / CHUNK).min(nw * 4).max(1);
        let step = sys::up(bytes.div_ceil(parts), CHUNK);
        let guard = self.huge.load(Relaxed) == 1;
        let touch = |_w: usize, i: u32| {
            let lo = i as usize * step;
            let hi = (lo + step).min(bytes);
            if lo >= hi {
                return;
            }
            if !guard {
                touch_range(p + lo, hi - lo, populate);
                return;
            }
            let mut a = p + lo;
            while a < p + hi {
                if self.stalled.load(Relaxed) != 0 {
                    touch_range(a, p + hi - a, populate);
                    break;
                }
                let t0 = sys::mono_ns();
                touch_range(a, CHUNK, populate);
                if sys::mono_ns() - t0 > STALL_NS && self.slow_fault_is_compaction() {
                    self.stop_huge();
                }
                a += CHUNK;
            }
        };
        if crate::pool::me() < MAX_WORKERS {
            crate::pool::run(0, parts as u32, &touch);
        } else {
            for i in 0..parts {
                touch(0, i as u32);
            }
        }
        self.give(p as *mut u8, bytes);
    }

    /// Switches the region to 4 KB pages for every fault from now on (huge pages already in
    /// place stay). Called by the stall guard; callable by anyone who sees stalls.
    pub fn stop_huge(&self) {
        if self.stalled.swap(1, AcqRel) == 0 {
            unsafe {
                let _ = sys::madvise(self.base(), self.cap(), sys::MADV_NOHUGEPAGE);
            }
        }
    }

    /// A slow 2 MB fault is compaction when the kernel's compact_stall counter rose since `init`
    /// (a VM's vCPU being preempted also makes a fault slow: seen on box 2 with compact_stall +0).
    /// Without /proc/vmstat: the third slow fault counts.
    fn slow_fault_is_compaction(&self) -> bool {
        let base = self.stall0.load(Relaxed);
        if base == u64::MAX {
            return self.slow.fetch_add(1, AcqRel) + 1 >= STALL_STRIKES;
        }
        compact_stalls().is_some_and(|now| now > base)
    }

    /// Whether the stall guard switched this region off huge pages.
    pub fn stalled(&self) -> bool {
        self.stalled.load(Relaxed) != 0
    }

    /// The mode in effect after `init` (Auto resolved).
    pub fn huge_mode(&self) -> Huge {
        match self.huge.load(Relaxed) {
            1 => Huge::On,
            2 => Huge::Off,
            _ => Huge::Default,
        }
    }

    fn take_fresh(&self, bytes: usize) -> *mut u8 {
        let off = self.top.fetch_add(bytes, AcqRel);
        if off + bytes > self.cap() {
            crate::fail::fail(b"arena: region exhausted");
        }
        let l = self.live.fetch_add(bytes, Relaxed) + bytes;
        self.peak.fetch_max(l, Relaxed);
        (self.base() + off) as *mut u8
    }

    /// Hands everything back to the free list (no allocation may be live).
    pub fn free_all(&self) {
        let top = self.top.load(Acquire);
        self.lock.lock();
        unsafe {
            let fl = &mut *self.free.get();
            fl.n = 0;
            if top > 0 {
                fl.r[0] = (0, (top / CHUNK) as u32);
                fl.n = 1;
            }
        }
        self.lock.unlock();
        self.live.store(0, Relaxed);
    }

    /// Confirms the huge-page choice on the first fresh chunk: faults it (one write), then checks
    /// that touching 1 MB further costs no fault. Returns (huge page obtained, fault time in ns).
    /// The chunk stays in the region (first in line for `take`).
    pub fn probe_huge(&self) -> (bool, u64) {
        let p = self.take_fresh(CHUNK) as usize;
        let t0 = sys::mono_ns();
        unsafe { core::ptr::write_volatile(p as *mut u8, 0) };
        let dt = sys::mono_ns() - t0;
        let r0 = sys::getrusage(sys::RUSAGE_THREAD);
        unsafe { core::ptr::write_volatile((p + (1 << 20)) as *mut u8, 0) };
        let r1 = sys::getrusage(sys::RUSAGE_THREAD);
        self.give(p as *mut u8, CHUNK);
        (r1.minflt == r0.minflt, dt)
    }

    /// Releases every page of the region (MADV_DONTNEED over [lo, hi) of the used part).
    pub(crate) fn used_range(&self) -> (usize, usize) {
        (self.base(), self.top.load(Relaxed))
    }
}

/// The system-wide count of direct-compaction stalls (/proc/vmstat compact_stall).
pub fn compact_stalls() -> Option<u64> {
    let mut buf = [0u8; 16384];
    let n = sys::read_small(b"/proc/vmstat", &mut buf).ok()?;
    let t = &buf[..n];
    let key = b"compact_stall ";
    let at = t.windows(key.len()).position(|w| w == key)? + key.len();
    let mut v = 0u64;
    for &c in &t[at..] {
        if !c.is_ascii_digit() {
            break;
        }
        v = v * 10 + (c - b'0') as u64;
    }
    Some(v)
}

/// Writes one byte per page of `[p, p + len)`, or MADV_POPULATE_WRITE.
pub fn touch_range(p: usize, len: usize, populate: bool) {
    if populate {
        unsafe {
            if sys::madvise(p, len, sys::MADV_POPULATE_WRITE).is_ok() {
                return;
            }
        }
    }
    let mut a = p;
    while a < p + len {
        unsafe { core::ptr::write_volatile(a as *mut u8, 0) };
        a += PAGE;
    }
}

// ------------------------------------------------------------------------------------------
// Bump.

#[repr(C, align(128))]
struct BumpW {
    cur: usize,
    end: usize,
    /// Intrusive list of taken ranges: [next, bytes] at each range start.
    head: usize,
    used: usize,
    peak: usize,
}

pub struct Bump {
    region: AtomicUsize,
    w: [UnsafeCell<BumpW>; MAX_WORKERS],
}

unsafe impl Sync for Bump {}

const HDR: usize = 64;

impl Default for Bump {
    fn default() -> Self {
        Self::new()
    }
}

impl Bump {
    pub const fn new() -> Bump {
        Bump { region: AtomicUsize::new(0), w: [const { UnsafeCell::new(BumpW { cur: 0, end: 0, head: 0, used: 0, peak: 0 }) }; MAX_WORKERS] }
    }
    pub fn attach(&self, r: &'static Region) {
        self.region.store(r as *const Region as usize, Release);
    }
    #[inline]
    fn region(&self) -> &Region {
        unsafe { &*(self.region.load(Relaxed) as *const Region) }
    }

    /// `size` bytes aligned to `align` (a power of two) for worker `w`.
    ///
    /// # Safety
    /// Only worker `w` allocates with index `w` at a time.
    #[inline]
    pub unsafe fn alloc(&self, w: usize, size: usize, align: usize) -> *mut u8 {
        let s = &mut *self.w[w].get();
        let p = sys::up(s.cur, align);
        if p + size <= s.end && s.end != 0 {
            s.cur = p + size;
            s.used += size;
            return p as *mut u8;
        }
        self.slow(w, size, align)
    }

    #[cold]
    unsafe fn slow(&self, w: usize, size: usize, align: usize) -> *mut u8 {
        let s = &mut *self.w[w].get();
        let need = HDR + size + align;
        let bytes = sys::up(need, CHUNK);
        let c = self.region().take(bytes) as usize;
        *(c as *mut usize) = s.head;
        *((c + 8) as *mut usize) = bytes;
        s.head = c;
        let p = sys::up(c + HDR, align);
        s.used += size;
        s.peak = s.peak.max(s.used);
        if bytes > CHUNK {
            // A large allocation gets its own range; keep bumping the current chunk.
            return p as *mut u8;
        }
        s.cur = p + size;
        s.end = c + bytes;
        p as *mut u8
    }

    /// Returns worker `w`'s chunks to the region.
    ///
    /// # Safety
    /// Nothing worker `w` allocated is used afterwards.
    pub unsafe fn reset(&self, w: usize) {
        let s = &mut *self.w[w].get();
        let mut c = s.head;
        while c != 0 {
            let next = *(c as *const usize);
            let bytes = *((c + 8) as *const usize);
            self.region().give(c as *mut u8, bytes);
            c = next;
        }
        s.head = 0;
        s.cur = 0;
        s.end = 0;
        s.used = 0;
    }

    /// # Safety
    /// As `reset`, for every worker; no allocation runs concurrently.
    pub unsafe fn reset_all(&self) {
        for w in 0..MAX_WORKERS {
            self.reset(w);
        }
    }

    /// Bytes allocated by worker `w` since its last reset, and its peak.
    pub fn used(&self, w: usize) -> (usize, usize) {
        unsafe {
            let s = &*self.w[w].get();
            (s.used, s.peak)
        }
    }
}

// ------------------------------------------------------------------------------------------
// Fixed-size blocks.

#[repr(C, align(128))]
struct BlocksW {
    free: usize,
    nfree: usize,
    cur: usize,
    end: usize,
    allocs: u64,
    frees: u64,
    carved: u64,
}

pub struct Blocks {
    region: AtomicUsize,
    bsize: AtomicUsize,
    w: [UnsafeCell<BlocksW>; MAX_WORKERS],
    glock: SpinLock,
    ghead: UnsafeCell<usize>,
    gcount: AtomicUsize,
    /// Free blocks a worker keeps before moving half to the shared list.
    local_max: AtomicUsize,
    /// Shared-list lock acquisitions: spills (free side) and refills (alloc side).
    spills: AtomicUsize,
    refills: AtomicUsize,
    /// Chunks taken, for `reset`.
    log: AtomicUsize,
    log_n: AtomicUsize,
    log_cap: AtomicUsize,
}

unsafe impl Sync for Blocks {}

/// A worker keeps up to this many bytes of free blocks before moving half to the shared list
/// (at least 64 blocks; `set_local_max` overrides). Measured on box 2 with the dedup pattern at
/// final size (8 workers): a 256-block cap cost 22 ms for 2 KB blocks and 110 ms for 256 B blocks
/// in shared-list spills and refills; 16 MB per worker cost 3.8 and 8.5 ms (8 KB: under 1 ms).
const LOCAL_BYTES: usize = 16 << 20;
const LOCAL_MAX: usize = 256;
const BATCH: usize = 64;

impl Default for Blocks {
    fn default() -> Self {
        Self::new()
    }
}

impl Blocks {
    pub const fn new() -> Blocks {
        Blocks {
            region: AtomicUsize::new(0),
            bsize: AtomicUsize::new(0),
            w: [const { UnsafeCell::new(BlocksW { free: 0, nfree: 0, cur: 0, end: 0, allocs: 0, frees: 0, carved: 0 }) }; MAX_WORKERS],
            glock: SpinLock::new(),
            ghead: UnsafeCell::new(0),
            gcount: AtomicUsize::new(0),
            local_max: AtomicUsize::new(LOCAL_MAX),
            spills: AtomicUsize::new(0),
            refills: AtomicUsize::new(0),
            log: AtomicUsize::new(0),
            log_n: AtomicUsize::new(0),
            log_cap: AtomicUsize::new(0),
        }
    }

    /// Blocks of `bsize` bytes (power of two, 64 B to 2 MB) from region `r`.
    pub fn attach(&self, r: &'static Region, bsize: usize) -> Res<()> {
        if !bsize.is_power_of_two() || !(64..=CHUNK).contains(&bsize) {
            return Err(Errno::EINVAL);
        }
        let cap = r.cap() / CHUNK + 1;
        let log = sys::map_anon(sys::up(cap * 8, PAGE))? as usize;
        self.log.store(log, Relaxed);
        self.log_cap.store(cap, Relaxed);
        self.bsize.store(bsize, Relaxed);
        self.local_max.store((LOCAL_BYTES / bsize).clamp(64, 65536), Relaxed);
        self.region.store(r as *const Region as usize, Release);
        Ok(())
    }
    #[inline]
    pub fn block_size(&self) -> usize {
        self.bsize.load(Relaxed)
    }
    #[inline]
    fn region(&self) -> &Region {
        unsafe { &*(self.region.load(Relaxed) as *const Region) }
    }

    /// One block for worker `w`: its own free list, then the shared list, then fresh memory.
    ///
    /// # Safety
    /// Only worker `w` uses index `w` at a time.
    #[inline]
    pub unsafe fn alloc(&self, w: usize) -> *mut u8 {
        let s = &mut *self.w[w].get();
        s.allocs += 1;
        if s.free != 0 {
            let p = s.free;
            s.free = *(p as *const usize);
            s.nfree -= 1;
            return p as *mut u8;
        }
        self.refill(w)
    }

    #[cold]
    unsafe fn refill(&self, w: usize) -> *mut u8 {
        let s = &mut *self.w[w].get();
        let bs = self.block_size();
        if self.gcount.load(Relaxed) != 0 {
            self.glock.lock();
            self.refills.fetch_add(1, Relaxed);
            let mut k = 0;
            let head = &mut *self.ghead.get();
            while k < BATCH && *head != 0 {
                let p = *head;
                *head = *(p as *const usize);
                *(p as *mut usize) = s.free;
                s.free = p;
                k += 1;
            }
            self.gcount.fetch_sub(k, Relaxed);
            self.glock.unlock();
            if s.free != 0 {
                s.nfree += k;
                let p = s.free;
                s.free = *(p as *const usize);
                s.nfree -= 1;
                return p as *mut u8;
            }
        }
        if s.cur + bs > s.end || s.end == 0 {
            let c = self.region().take(CHUNK) as usize;
            let i = self.log_n.fetch_add(1, Relaxed);
            if i >= self.log_cap.load(Relaxed) {
                crate::fail::fail(b"blocks: chunk log full");
            }
            *((self.log.load(Relaxed) + i * 8) as *mut usize) = c;
            s.cur = c;
            s.end = c + CHUNK;
        }
        let p = s.cur;
        s.cur += bs;
        s.carved += 1;
        p as *mut u8
    }

    /// Returns a block (to worker `w`'s list; half the list moves to the shared one when long).
    ///
    /// # Safety
    /// `p` came from `alloc` of this arena and is not used afterwards.
    #[inline]
    pub unsafe fn free(&self, w: usize, p: *mut u8) {
        let s = &mut *self.w[w].get();
        s.frees += 1;
        *(p as *mut usize) = s.free;
        s.free = p as usize;
        s.nfree += 1;
        if s.nfree > self.local_max.load(Relaxed) {
            self.spill(w);
        }
    }

    #[cold]
    unsafe fn spill(&self, w: usize) {
        let s = &mut *self.w[w].get();
        let k = s.nfree / 2;
        let first = s.free;
        let mut last = first;
        for _ in 1..k {
            last = *(last as *const usize);
        }
        s.free = *(last as *const usize);
        s.nfree -= k;
        self.glock.lock();
        self.spills.fetch_add(1, Relaxed);
        let head = &mut *self.ghead.get();
        *(last as *mut usize) = *head;
        *head = first;
        self.gcount.fetch_add(k, Relaxed);
        self.glock.unlock();
    }

    /// Returns every chunk to the region (no block may be live or used afterwards).
    ///
    /// # Safety
    /// No concurrent use; every block is dead.
    pub unsafe fn reset(&self) {
        let n = self.log_n.swap(0, AcqRel);
        let log = self.log.load(Relaxed);
        for i in 0..n {
            self.region().give(*((log + i * 8) as *const usize) as *mut u8, CHUNK);
        }
        for w in 0..MAX_WORKERS {
            let s = &mut *self.w[w].get();
            s.free = 0;
            s.nfree = 0;
            s.cur = 0;
            s.end = 0;
        }
        *self.ghead.get() = 0;
        self.gcount.store(0, Relaxed);
    }

    /// Free blocks a worker may hold before spilling half to the shared list.
    pub fn set_local_max(&self, blocks: usize) {
        self.local_max.store(blocks.max(2), Relaxed);
    }

    /// Shared-list lock acquisitions so far: (spills, refills).
    pub fn lock_ops(&self) -> (usize, usize) {
        (self.spills.load(Relaxed), self.refills.load(Relaxed))
    }

    /// (allocations, frees, blocks cut from fresh memory) summed over workers.
    pub fn stats(&self) -> (u64, u64, u64) {
        let mut t = (0, 0, 0);
        for w in 0..MAX_WORKERS {
            let s = unsafe { &*self.w[w].get() };
            t.0 += s.allocs;
            t.1 += s.frees;
            t.2 += s.carved;
        }
        t
    }
}

// ------------------------------------------------------------------------------------------
// Huge-page verdict.

#[derive(Clone, Copy, Debug)]
pub struct HugeInfo {
    /// /sys/kernel/mm/transparent_hugepage/enabled: b'a' always, b'm' madvise, b'n' never, 0 unread.
    pub enabled: u8,
    /// .../defrag: b'a' always, b'd' defer, b'D' defer+madvise, b'm' madvise, b'n' never, 0 unread.
    pub defrag: u8,
    /// Free 2 MB blocks (orders 9 and up) in the Normal and Movable zones, -1 unread.
    pub free_2m: i64,
    /// The same over every zone (DMA32 blocks are held back by the low-memory reserve).
    pub free_2m_all: i64,
    /// MemFree and MemAvailable from /proc/meminfo in bytes, -1 unread.
    pub mem_free: i64,
    pub mem_available: i64,
    /// Whether a MADV_HUGEPAGE fault can stall in direct compaction (defrag always, madvise or
    /// defer+madvise; also when defrag is unreadable).
    pub may_stall: bool,
    /// Use huge pages for `need` bytes (see `huge_verdict`).
    pub use_huge: bool,
    /// Why: b'b' enough free 2 MB blocks, b'f' plenty of free memory (compaction cheap), b'd' the
    /// defrag mode cannot stall, b'u' nothing readable (no evidence of trouble), b't' THP never,
    /// b'n' blocks short and free memory tight (compaction would reclaim: costly).
    pub reason: u8,
}

fn bracketed(t: &[u8]) -> Option<&[u8]> {
    let a = t.iter().position(|&c| c == b'[')?;
    let b = t[a..].iter().position(|&c| c == b']')? + a;
    Some(&t[a + 1..b])
}

/// "Key:   123 kB" from /proc/meminfo, in bytes.
fn meminfo(t: &[u8], key: &[u8]) -> Option<i64> {
    let at = t.windows(key.len()).position(|w| w == key)? + key.len();
    let mut v: i64 = 0;
    let mut seen = false;
    for &c in &t[at..] {
        if c.is_ascii_digit() {
            v = v * 10 + (c - b'0') as i64;
            seen = true;
        } else if seen {
            break;
        }
    }
    if seen { Some(v << 10) } else { None }
}

/// Whether to use huge pages for `need` bytes, from the THP settings, /proc/buddyinfo and
/// /proc/meminfo (all visible inside Docker under the scorer flags; about 30 us; nothing needs to
/// be initialised). Yes when THP is usable (always, madvise, or unreadable) and any of:
/// - the Normal and Movable zones hold `need` + 25% in free 2 MB blocks (no compaction needed);
/// - MemFree is at least twice `need`: blocks run short but free memory is plentiful, so
///   compaction only moves a few pages per block (measured on box 2 with 66 GB fragmented to one
///   page per 2 MB block: huge pages 47 to 54 ms/GB including compaction, 4 KB pages 78 to 83);
/// - the defrag mode cannot stall (defer, never);
/// - neither file can be read (no evidence of trouble).
/// No when blocks are short and free memory is tight: compaction must then reclaim page cache
/// (measured on box 1 with MemFree 2 to 3 GB: f60v5 with huge pages 3,000 to 3,850 compaction
/// stalls and 3 to 7 s more system time per run). Then a budget of the free blocks is the rule.
pub fn huge_verdict(need: usize) -> HugeInfo {
    let mut info = HugeInfo { enabled: 0, defrag: 0, free_2m: -1, free_2m_all: -1, mem_free: -1, mem_available: -1, may_stall: true, use_huge: false, reason: b'n' };
    let mut buf = [0u8; 4096];
    if let Ok(n) = sys::read_small(b"/sys/kernel/mm/transparent_hugepage/enabled", &mut buf) {
        info.enabled = match bracketed(&buf[..n]) {
            Some(b"always") => b'a',
            Some(b"madvise") => b'm',
            Some(b"never") => b'n',
            _ => 0,
        };
    }
    if let Ok(n) = sys::read_small(b"/sys/kernel/mm/transparent_hugepage/defrag", &mut buf) {
        info.defrag = match bracketed(&buf[..n]) {
            Some(b"always") => b'a',
            Some(b"defer") => b'd',
            Some(b"defer+madvise") => b'D',
            Some(b"madvise") => b'm',
            Some(b"never") => b'n',
            _ => 0,
        };
    }
    info.may_stall = !matches!(info.defrag, b'd' | b'n');
    if let Ok(n) = sys::read_small(b"/proc/buddyinfo", &mut buf) {
        // "Node 0, zone   Normal   a0 a1 ... a10" per line.
        let (mut normal, mut all) = (0i64, 0i64);
        for line in buf[..n].split(|&c| c == b'\n') {
            let Some(z) = line.windows(4).position(|w| w == b"zone") else { continue };
            let mut fields = line[z + 4..].split(|&c| c == b' ').filter(|f| !f.is_empty());
            let zone = fields.next().unwrap_or(b"");
            let counted = zone == b"Normal" || zone == b"Movable";
            for (order, f) in fields.enumerate() {
                let mut v: i64 = 0;
                for &c in f {
                    if c.is_ascii_digit() {
                        v = v * 10 + (c - b'0') as i64;
                    }
                }
                if order >= 9 {
                    all += v << (order - 9);
                    if counted {
                        normal += v << (order - 9);
                    }
                }
            }
        }
        info.free_2m = normal;
        info.free_2m_all = all;
    }
    if let Ok(n) = sys::read_small(b"/proc/meminfo", &mut buf) {
        info.mem_free = meminfo(&buf[..n], b"MemFree:").unwrap_or(-1);
        info.mem_available = meminfo(&buf[..n], b"MemAvailable:").unwrap_or(-1);
    }
    let need = need as i128;
    let blocks = info.free_2m >= 0 && (info.free_2m as i128) * (HUGE as i128) >= need * 5 / 4;
    let plenty = info.mem_free >= 0 && (info.mem_free as i128) >= need * 2;
    let blind = info.free_2m < 0 && info.mem_free < 0;
    info.reason = if info.enabled == b'n' {
        b't'
    } else if blocks {
        b'b'
    } else if plenty {
        b'f'
    } else if !info.may_stall {
        b'd'
    } else if blind {
        b'u'
    } else {
        b'n'
    };
    info.use_huge = !matches!(info.reason, b't' | b'n');
    info
}

/// Counts of the region's chunks on the free list (for tests).
pub fn free_ranges(r: &Region) -> usize {
    r.lock.lock();
    let n = unsafe { (*r.free.get()).n };
    r.lock.unlock();
    n
}
