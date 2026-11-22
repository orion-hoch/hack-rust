//! The engine's global allocator, built here (musl's own malloc halves v1's speed: it serialises
//! threads that allocate at once). Three tiers by size; blocks carry no header, because Rust passes
//! the layout to `dealloc` and the size picks the tier and class.
//!
//! * Small, up to 8 KiB: 32 size classes (16-byte steps to 128 B, then four per power of two).
//!   Each thread owns a heap: one free list per class, linked through the freed blocks, and a bump
//!   region carved from 64 MiB mappings; no lock. A freed block goes on the freeing thread's list.
//!   The block at the top of the bump region grows and shrinks in place.
//! * Medium, over 8 KiB to 256 KiB: runs of whole pages ("spans") from 64 MiB chunks aligned to
//!   their size, under one lock. Free spans are listed by length and merge with free neighbours
//!   (bitmaps of span starts and ends in each chunk's first page), so memory freed at one size
//!   serves any other size later. Measured on v1 at 20M events: per-size free lists for this range
//!   mapped 2.3 times the live peak, because each stage peaks at different sizes; spans brought the
//!   peak RSS down to mimalloc's. A growing span takes a free right neighbour in place.
//! * Large, over 256 KiB: each its own mapping of a power of two of pages (only touched pages cost
//!   memory). Freed ones wait on a global list per size and are reused; `realloc` stays in place up
//!   to that size and moves by `mremap` beyond it (pages move, nothing is copied).
//! * Mappings of 2 MiB and more start on a 2 MiB boundary and are advised MADV_HUGEPAGE: one fault
//!   per 2 MiB where the kernel allows it (measured: 10 times fewer faults at 20M events).
//! * Nothing is returned to the kernel during the run: no `munmap`, no `madvise(DONTNEED)`.
//! * Heaps live in a fixed table. A thread claims one at its first allocation and a pthread key
//!   destructor hands it back when the thread exits, so the next thread inherits its free lists.
//!   A thread without a heap (table full, or already exiting) uses one shared heap under a lock.
//!   Locks spin briefly, then yield, so a holder preempted under SCHED_FIFO gets the CPU back.
//!
//! The system calls are plain `extern "C"` declarations against the libc the static binary already
//! links (`mmap`, `mremap`, `madvise`, `pthread_key_create`, `pthread_setspecific`, `sched_yield`).
//! `STATS` and `LARGE_RELEASE` are measurement switches, off in the shipped build.

use core::alloc::{GlobalAlloc, Layout};
use core::cell::{Cell, UnsafeCell};
use core::ffi::c_void;
use core::ptr::{self, null_mut};
use core::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicUsize, Ordering};

unsafe extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
    fn mremap(old: *mut c_void, old_len: usize, new_len: usize, flags: i32, ...) -> *mut c_void;
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    fn pthread_key_create(key: *mut u32, dtor: Option<unsafe extern "C" fn(*mut c_void)>) -> i32;
    fn pthread_setspecific(key: u32, value: *const c_void) -> i32;
    fn sched_yield() -> i32;
    fn atexit(f: extern "C" fn()) -> i32;
    fn write(fd: i32, buf: *const u8, n: usize) -> isize;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}

/// Measurement builds only: counters printed to stderr at exit.
const STATS: bool = false;
/// Measurement switch: unmap freed large blocks instead of keeping them for reuse.
const LARGE_RELEASE: bool = false;

mod stats {
    use core::sync::atomic::{AtomicIsize, AtomicUsize, Ordering::Relaxed};
    pub static SMALL_LIVE: AtomicIsize = AtomicIsize::new(0);
    pub static LARGE_LIVE: AtomicIsize = AtomicIsize::new(0);
    pub static LARGE_REQ_LIVE: AtomicIsize = AtomicIsize::new(0);
    pub static LARGE_FREE: AtomicIsize = AtomicIsize::new(0);
    pub static SPAN_LIVE: AtomicIsize = AtomicIsize::new(0);
    pub static PEAK: AtomicIsize = AtomicIsize::new(0);
    pub static AT_PEAK: [AtomicIsize; 4] = [const { AtomicIsize::new(0) }; 4];
    pub static N: [AtomicUsize; 6] = [const { AtomicUsize::new(0) }; 6]; // small, large, hit, miss, mremap, chunks
    pub static CARVED: [AtomicIsize; super::NCLASS] = [const { AtomicIsize::new(0) }; super::NCLASS];
    pub static CLIVE: [AtomicIsize; super::NCLASS] = [const { AtomicIsize::new(0) }; super::NCLASS];
    pub static CPEAK: [AtomicIsize; super::NCLASS] = [const { AtomicIsize::new(0) }; super::NCLASS];
    pub fn note() {
        let s = SMALL_LIVE.load(Relaxed) + SPAN_LIVE.load(Relaxed);
        let l = LARGE_LIVE.load(Relaxed);
        let t = s + l;
        if t > PEAK.load(Relaxed) {
            PEAK.store(t, Relaxed);
            AT_PEAK[0].store(s, Relaxed);
            AT_PEAK[1].store(l, Relaxed);
            AT_PEAK[2].store(LARGE_REQ_LIVE.load(Relaxed), Relaxed);
            AT_PEAK[3].store(LARGE_FREE.load(Relaxed), Relaxed);
        }
    }
}

extern "C" fn print_stats() {
    use core::fmt::Write as _;
    struct Buf([u8; 4096], usize);
    impl core::fmt::Write for Buf {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let b = s.as_bytes();
            let n = b.len().min(self.0.len() - self.1);
            self.0[self.1..self.1 + n].copy_from_slice(&b[..n]);
            self.1 += n;
            Ok(())
        }
    }
    let r = |a: &core::sync::atomic::AtomicIsize| a.load(Ordering::Relaxed) >> 20;
    let c = |i: usize| stats::N[i].load(Ordering::Relaxed);
    let mut b = Buf([0; 4096], 0);
    let _ = write!(b, "class carved/peak-live MiB:");
    for c in 0..NCLASS {
        let carved = stats::CARVED[c].load(Ordering::Relaxed) >> 20;
        if carved > 8 {
            let _ = write!(b, " {}:{}/{}", class_size(c), carved, stats::CPEAK[c].load(Ordering::Relaxed) >> 20);
        }
    }
    let _ = writeln!(b);
    let _ = writeln!(
        b,
        "alloc stats MiB: peak live {} (small+spans {} large {} large requested {} large free-listed {}); \
         at exit small {} large {} free-listed {}; mapped {}; counts small {} large {} reuse {} fresh {} mremap {} chunks {}",
        r(&stats::PEAK), r(&stats::AT_PEAK[0]), r(&stats::AT_PEAK[1]), r(&stats::AT_PEAK[2]), r(&stats::AT_PEAK[3]),
        r(&stats::SMALL_LIVE), r(&stats::LARGE_LIVE), r(&stats::LARGE_FREE), MAPPED.load(Ordering::Relaxed) >> 20,
        c(0), c(1), c(2), c(3), c(4), c(5)
    );
    // SAFETY: plain write of our own buffer.
    unsafe { write(2, b.0.as_ptr(), b.1) };
}

const PROT_READ_WRITE: i32 = 1 | 2;
const MAP_PRIVATE_ANON_NORESERVE: i32 = 0x02 | 0x20 | 0x4000;
const MREMAP_MAYMOVE: i32 = 1;
const MAP_FAILED: *mut c_void = !0usize as *mut c_void;
const MADV_HUGEPAGE: i32 = 14;
/// Transparent huge pages for the chunks and for large blocks of at least HUGE bytes: one fault per
/// 2 MiB instead of per 4 KiB where the kernel allows it (THP "madvise" or "always").
/// Bytes of new mappings that may still be advised MADV_HUGEPAGE. Set once at start-up from rt's
/// verdict: everything when the machine has the free 2 MB blocks (or its THP defrag mode cannot
/// stall), about 80% of the free blocks when it has fewer, none when THP is unusable. Past the
/// budget a mapping gets 4 KB pages at once instead of a direct-compaction stall (118 to 200
/// ms/GB measured on a fragmented box, against 62 to 66 for 4 KB pages and 38 to 43 for free
/// huge pages).
static THP_BUDGET: AtomicUsize = AtomicUsize::new(usize::MAX);

pub fn set_thp_budget(bytes: usize) {
    THP_BUDGET.store(bytes, Ordering::Relaxed);
    rt::arena::set_huge_budget(bytes);
}

/// Takes `len` bytes of the budget; false when it does not fit (the budget stays as it is).
#[inline]
fn take_thp(len: usize) -> bool {
    // The process-wide budget, shared with the token tables and rt's regions.
    let _ = &THP_BUDGET;
    rt::arena::take_huge(len)
}
const HUGE: usize = 2 << 20;

const PAGE: usize = 4096;
/// Largest size served by the size classes (the classes are defined up to SMALL_MAX, used to here).
const CLASS_MAX: usize = 8 << 10;
/// Largest size served from spans (runs of whole pages); above it, blocks are their own mappings.
const SMALL_MAX: usize = 1 << 18;
/// Largest size served from spans. Blocks from SMALL_MAX to here were their own mappings on 4 KB
/// pages (under HUGE no huge-page advice): at 20M events 139 mmaps, 89 mremaps and about 18K page
/// faults, none of it reused across sizes. In the span chunks they sit on huge pages and a freed
/// span serves any size.
const SPAN_MAX: usize = 2 << 20;
/// Span chunks: 64 MiB, aligned to their size; page 0 holds the chunk's free-span bitmaps.
const SPAN_CHUNK: usize = 64 << 20;
const CPAGES: usize = SPAN_CHUNK / PAGE;
/// Free spans are listed by exact length up to this many pages; list 0 holds longer ones.
const SPAN_LISTS: usize = SMALL_MAX / PAGE;
const NCLASS: usize = 52;
/// Bytes of address space a heap takes at a time for its bump region.
const CHUNK: usize = 64 << 20;
const MAX_HEAPS: usize = 256;
/// Large-block size buckets: 2^b pages, b up to 47.
const NBUCKET: usize = 48;

/// Size class of a size of at most SMALL_MAX (used up to CLASS_MAX).
#[inline(always)]
fn class_of(size: usize) -> usize {
    if size <= 128 {
        return (size.max(1) + 15) / 16 - 1;
    }
    let s = size - 1;
    let k = (usize::BITS - 1 - s.leading_zeros()) as usize; // 7..=17
    8 + (k - 7) * 4 + ((s >> (k - 2)) - 4)
}

#[inline(always)]
fn class_size(c: usize) -> usize {
    if c < 8 { (c + 1) * 16 } else { (5 + (c - 8) % 4) << ((c - 8) / 4 + 5) }
}

/// Large bucket of a size over SMALL_MAX: the power of two of pages that holds it.
#[inline(always)]
fn bucket(size: usize) -> usize {
    let pages = size.div_ceil(PAGE);
    (usize::BITS - (pages - 1).leading_zeros()) as usize
}

#[inline(always)]
fn bucket_bytes(b: usize) -> usize {
    PAGE << b
}

/// Bytes of address space mapped so far (for measurements; never decreases).
static MAPPED: AtomicUsize = AtomicUsize::new(0);

#[allow(dead_code)]
pub fn mapped_bytes() -> usize {
    MAPPED.load(Ordering::Relaxed)
}

/// A fresh zeroed mapping of `len` bytes. From HUGE bytes up it starts on a 2 MiB boundary (the
/// mapping is HUGE longer; the unused head and tail are never touched) and is advised for huge pages.
fn map(len: usize) -> *mut u8 {
    let huge = len >= HUGE && take_thp(len);
    let total = if huge { len + HUGE } else { len };
    // SAFETY: a fresh private anonymous mapping; no existing memory is touched.
    let p = unsafe { mmap(null_mut(), total, PROT_READ_WRITE, MAP_PRIVATE_ANON_NORESERVE, -1, 0) };
    if p == MAP_FAILED {
        return null_mut();
    }
    MAPPED.fetch_add(total, Ordering::Relaxed);
    if !huge {
        return p as *mut u8;
    }
    let a = (p as usize + HUGE - 1) & !(HUGE - 1);
    // SAFETY: [a, a + len) lies inside the new mapping. The advice is a hint; failure is harmless.
    unsafe { madvise(a as *mut c_void, len, MADV_HUGEPAGE) };
    a as *mut u8
}

/// A fresh zeroed mapping of `len` bytes starting on an `align` boundary (power of two); the
/// mapping is `align` longer and its unused head and tail are never touched.
fn map_aligned(len: usize, align: usize) -> *mut u8 {
    // SAFETY: a fresh private anonymous mapping; no existing memory is touched.
    let p = unsafe { mmap(null_mut(), len + align, PROT_READ_WRITE, MAP_PRIVATE_ANON_NORESERVE, -1, 0) };
    if p == MAP_FAILED {
        return null_mut();
    }
    MAPPED.fetch_add(len + align, Ordering::Relaxed);
    let a = (p as usize + align - 1) & !(align - 1);
    if take_thp(len) {
        // SAFETY: inside the new mapping; a hint only.
        unsafe { madvise(a as *mut c_void, len, MADV_HUGEPAGE) };
    }
    a as *mut u8
}

/// A test-and-test-and-set lock that yields after a while, so a holder that was preempted (or
/// sleeps in mmap) gets the CPU back even under SCHED_FIFO.
struct SpinLock(AtomicBool);

impl SpinLock {
    const fn new() -> Self {
        SpinLock(AtomicBool::new(false))
    }
    fn lock(&self) {
        let mut spins = 0u32;
        while self.0.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            while self.0.load(Ordering::Relaxed) {
                spins += 1;
                if spins < 128 {
                    core::hint::spin_loop();
                } else {
                    // SAFETY: plain system call.
                    unsafe { sched_yield() };
                }
            }
        }
    }
    fn unlock(&self) {
        self.0.store(false, Ordering::Release);
    }
}

struct Heap {
    free: [*mut u8; NCLASS],
    bump: usize,
    end: usize,
}

impl Heap {
    const EMPTY: Heap = Heap { free: [null_mut(); NCLASS], bump: 0, end: 0 };

    #[inline(always)]
    unsafe fn alloc(&mut self, c: usize) -> *mut u8 {
        let p = self.free[c];
        if !p.is_null() {
            // SAFETY: a block on the list holds the next pointer in its first word.
            self.free[c] = unsafe { *(p as *mut *mut u8) };
            return p;
        }
        let size = class_size(c);
        if self.end - self.bump >= size {
            let p = self.bump;
            self.bump += size;
            if STATS {
                stats::CARVED[c].fetch_add(size as isize, Ordering::Relaxed);
            }
            return p as *mut u8;
        }
        self.refill(size)
    }

    #[cold]
    fn refill(&mut self, size: usize) -> *mut u8 {
        if STATS {
            stats::N[5].fetch_add(1, Ordering::Relaxed);
        }
        let p = map(CHUNK);
        if p.is_null() {
            return p;
        }
        // The old region's tail (under 256 KiB of 64 MiB) is left unused.
        self.bump = p as usize + size;
        self.end = p as usize + CHUNK;
        p
    }

    #[inline(always)]
    unsafe fn free(&mut self, p: *mut u8, c: usize) {
        let size = class_size(c);
        if p as usize + size == self.bump {
            self.bump = p as usize; // the top block goes back to the bump region
            if STATS {
                stats::CARVED[c].fetch_sub(size as isize, Ordering::Relaxed);
            }
            return;
        }
        // SAFETY: the block is at least 16 bytes and no longer in use.
        unsafe { *(p as *mut *mut u8) = self.free[c] };
        self.free[c] = p;
    }

    /// Resizes the block at the top of the bump region in place, if `p` is that block.
    #[inline(always)]
    fn resize_top(&mut self, p: *mut u8, old: usize, new: usize) -> bool {
        let p = p as usize;
        if p + old == self.bump && p + new <= self.end {
            self.bump = p + new;
            return true;
        }
        false
    }
}

struct Slot {
    busy: AtomicBool,
    heap: UnsafeCell<Heap>,
}

// SAFETY: a slot's heap is used only by the thread that set `busy` (or under SHARED_LOCK).
unsafe impl Sync for Slot {}

static SLOTS: [Slot; MAX_HEAPS] =
    [const { Slot { busy: AtomicBool::new(false), heap: UnsafeCell::new(Heap::EMPTY) } }; MAX_HEAPS];
static SHARED: Slot = Slot { busy: AtomicBool::new(true), heap: UnsafeCell::new(Heap::EMPTY) };
static SHARED_LOCK: SpinLock = SpinLock::new();

const NO_HEAP: usize = 0;
const USE_SHARED: usize = 1;

thread_local! {
    /// NO_HEAP before the first allocation, USE_SHARED, or slot index + 2.
    static MINE: Cell<usize> = const { Cell::new(NO_HEAP) };
}

static KEY: AtomicU32 = AtomicU32::new(0);
/// 0 not created, 1 being created, 2 ready, 3 failed.
static KEY_STATE: AtomicU8 = AtomicU8::new(0);

fn key() -> Option<u32> {
    loop {
        match KEY_STATE.load(Ordering::Acquire) {
            2 => return Some(KEY.load(Ordering::Relaxed)),
            3 => return None,
            0 => {
                if KEY_STATE.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed).is_ok() {
                    let mut k = 0u32;
                    // SAFETY: `k` is a valid out pointer; musl and glibc do not allocate here.
                    let ok = unsafe { pthread_key_create(&mut k, Some(release)) } == 0;
                    KEY.store(k, Ordering::Relaxed);
                    KEY_STATE.store(if ok { 2 } else { 3 }, Ordering::Release);
                }
            }
            _ => core::hint::spin_loop(),
        }
    }
}

/// pthread key destructor: the exiting thread hands its heap back.
unsafe extern "C" fn release(v: *mut c_void) {
    let slot = v as usize - 1;
    MINE.with(|m| m.set(USE_SHARED));
    SLOTS[slot].busy.store(false, Ordering::Release);
}

#[cold]
fn claim() -> usize {
    for (i, s) in SLOTS.iter().enumerate() {
        if !s.busy.load(Ordering::Relaxed)
            && s.busy.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok()
        {
            if let Some(k) = key() {
                // SAFETY: plain call; the value is never dereferenced.
                unsafe { pthread_setspecific(k, (i + 1) as *const c_void) };
            }
            return i + 2;
        }
    }
    USE_SHARED
}

#[inline(always)]
fn with_heap<R>(f: impl FnOnce(&mut Heap) -> R) -> R {
    let m = MINE.with(|m| m.get());
    if m >= 2 {
        // SAFETY: this thread owns slot m - 2.
        return f(unsafe { &mut *SLOTS[m - 2].heap.get() });
    }
    with_heap_slow(f)
}

#[cold]
fn with_heap_slow<R>(f: impl FnOnce(&mut Heap) -> R) -> R {
    let mut m = MINE.with(|m| m.get());
    if m == NO_HEAP {
        m = claim();
        MINE.with(|c| c.set(m));
    }
    if m >= 2 {
        // SAFETY: this thread now owns slot m - 2.
        return f(unsafe { &mut *SLOTS[m - 2].heap.get() });
    }
    SHARED_LOCK.lock();
    // SAFETY: SHARED is only used under SHARED_LOCK.
    let r = f(unsafe { &mut *SHARED.heap.get() });
    SHARED_LOCK.unlock();
    r
}

struct LargeLists(UnsafeCell<[*mut u8; NBUCKET]>);
// SAFETY: only touched under LARGE_LOCK.
unsafe impl Sync for LargeLists {}
static LARGE: LargeLists = LargeLists(UnsafeCell::new([null_mut(); NBUCKET]));
static LARGE_LOCK: SpinLock = SpinLock::new();

/// Once the engine's region is reserved, large blocks are contiguous takes from it (whole 2 MiB
/// chunks) and go back to its free list: a freed block then serves the block classes' next
/// chunks and any later large block of any size, instead of waiting for a request of its own
/// power of two while fresh memory is faulted beside it (measured at f60v5: 200 MiB of freed link
/// arrays idle from the middle of the parse on, and 17 fresh mappings in the person stage).
/// Blocks mapped before that stay as they are; `in_region` tells the two apart by address.
#[inline]
fn in_region(p: *mut u8) -> bool {
    let r = &crate::v3::mem::REG;
    let b = r.base();
    b != 0 && (p as usize).wrapping_sub(b) < r.cap()
}

/// A large block for `size` bytes, and whether it is fresh (all zero).
fn large_alloc(size: usize) -> (*mut u8, bool) {
    if crate::v3::mem::ready() {
        return crate::v3::mem::REG.take_tell(size);
    }
    let b = bucket(size);
    if STATS {
        use core::sync::atomic::Ordering::Relaxed;
        stats::LARGE_LIVE.fetch_add(bucket_bytes(b) as isize, Relaxed);
        stats::LARGE_REQ_LIVE.fetch_add(size as isize, Relaxed);
        stats::N[1].fetch_add(1, Relaxed);
        stats::note();
    }
    LARGE_LOCK.lock();
    // SAFETY: under LARGE_LOCK; a listed block holds the next pointer in its first word.
    let p = unsafe {
        let lists = &mut *LARGE.0.get();
        let p = lists[b];
        if !p.is_null() {
            lists[b] = *(p as *mut *mut u8);
        }
        p
    };
    LARGE_LOCK.unlock();
    if !p.is_null() {
        if STATS {
            stats::LARGE_FREE.fetch_sub(bucket_bytes(b) as isize, Ordering::Relaxed);
            stats::N[2].fetch_add(1, Ordering::Relaxed);
        }
        return (p, false);
    }
    if STATS {
        stats::N[3].fetch_add(1, Ordering::Relaxed);
    }
    (map(bucket_bytes(b)), true)
}

fn large_free(p: *mut u8, size: usize) {
    if in_region(p) {
        // The region's free list holds 1,024 ranges; past 900 the block is kept out of it (lost).
        if rt::arena::free_ranges(&crate::v3::mem::REG) < 900 {
            crate::v3::mem::REG.give(p, size);
        }
        return;
    }
    let b = bucket(size);
    if STATS {
        use core::sync::atomic::Ordering::Relaxed;
        stats::LARGE_LIVE.fetch_sub(bucket_bytes(b) as isize, Relaxed);
        stats::LARGE_REQ_LIVE.fetch_sub(size as isize, Relaxed);
    }
    if LARGE_RELEASE {
        // SAFETY: the block is a whole range we mapped and no longer in use.
        unsafe { munmap(p as *mut c_void, bucket_bytes(b)) };
        return;
    }
    if STATS {
        stats::LARGE_FREE.fetch_add(bucket_bytes(b) as isize, Ordering::Relaxed);
    }
    LARGE_LOCK.lock();
    // SAFETY: under LARGE_LOCK; the block is ours and no longer in use.
    unsafe {
        let lists = &mut *LARGE.0.get();
        *(p as *mut *mut u8) = lists[b];
        lists[b] = p;
    }
    LARGE_LOCK.unlock();
}

/// Grows or shrinks a large block: in place within its bucket, else by mremap (pages move, no
/// copy). Null if the kernel refuses.
fn large_realloc(p: *mut u8, old: usize, new: usize) -> *mut u8 {
    if in_region(p) {
        // In place within its 2 MiB chunks, else the caller allocates, copies and frees.
        return if old.div_ceil(HUGE) == new.div_ceil(HUGE) { p } else { null_mut() };
    }
    let (bo, bn) = (bucket(old), bucket(new));
    if bo == bn {
        return p;
    }
    // SAFETY: p is a whole mapping of bucket_bytes(bo) bytes made by `map` (or a previous mremap).
    let q = unsafe { mremap(p as *mut c_void, bucket_bytes(bo), bucket_bytes(bn), MREMAP_MAYMOVE) };
    if q == MAP_FAILED {
        return null_mut();
    }
    if bn > bo {
        MAPPED.fetch_add(bucket_bytes(bn) - bucket_bytes(bo), Ordering::Relaxed);
    }
    if STATS {
        use core::sync::atomic::Ordering::Relaxed;
        stats::LARGE_LIVE.fetch_add(bucket_bytes(bn) as isize - bucket_bytes(bo) as isize, Relaxed);
        stats::LARGE_REQ_LIVE.fetch_add(new as isize - old as isize, Relaxed);
        stats::N[4].fetch_add(1, Relaxed);
        stats::note();
    }
    q as *mut u8
}

/// Bitmaps in page 0 of a span chunk: bit i of `start` / `end` says page i begins / ends a free span.
#[repr(C)]
struct SpanMeta {
    start: [u64; CPAGES / 64],
    end: [u64; CPAGES / 64],
}

/// Header of a free span, in its first page. Its last page's first word repeats `pages` (for a
/// one-page span that word is this field), so a right neighbour finds the start.
#[repr(C)]
struct FreeSpan {
    pages: usize,
    next: *mut FreeSpan,
    prev: *mut FreeSpan,
}

/// Medium blocks: runs of whole pages from span chunks, under one lock. Free spans merge with
/// free neighbours, so memory freed at one size serves any other size later (per-size free lists
/// cannot: in v1 each stage peaks at different sizes).
struct Spans {
    lists: [*mut FreeSpan; SPAN_LISTS + 1],
    /// Bit l - 1 set when list l (1..=SPAN_LISTS) is non-empty.
    nonempty: u64,
}

struct SpansCell(UnsafeCell<Spans>);
// SAFETY: only touched under SPAN_LOCK.
unsafe impl Sync for SpansCell {}
static SPANS: SpansCell = SpansCell(UnsafeCell::new(Spans { lists: [null_mut(); SPAN_LISTS + 1], nonempty: 0 }));
static SPAN_LOCK: SpinLock = SpinLock::new();

#[inline(always)]
fn bit(b: &[u64], i: usize) -> bool {
    b[i / 64] >> (i % 64) & 1 != 0
}

#[inline(always)]
fn set_bit(b: &mut [u64], i: usize, on: bool) {
    if on {
        b[i / 64] |= 1 << (i % 64);
    } else {
        b[i / 64] &= !(1 << (i % 64));
    }
}

/// The chunk metadata and page index of an address inside a span chunk.
#[inline(always)]
fn locate(p: usize) -> (&'static mut SpanMeta, usize, usize) {
    let base = p & !(SPAN_CHUNK - 1);
    // SAFETY: span chunks are SPAN_CHUNK-aligned and page 0 is their SpanMeta.
    (unsafe { &mut *(base as *mut SpanMeta) }, (p - base) / PAGE, base)
}

impl Spans {
    #[inline(always)]
    fn list_of(pages: usize) -> usize {
        if pages <= SPAN_LISTS { pages } else { 0 }
    }

    unsafe fn insert(&mut self, p: usize, pages: usize) {
        let (m, i, _) = locate(p);
        set_bit(&mut m.start, i, true);
        set_bit(&mut m.end, i + pages - 1, true);
        let l = Self::list_of(pages);
        let s = p as *mut FreeSpan;
        // SAFETY: the span's pages are ours and free.
        unsafe {
            (*s).pages = pages;
            (*s).prev = null_mut();
            (*s).next = self.lists[l];
            if !self.lists[l].is_null() {
                (*self.lists[l]).prev = s;
            }
            if pages > 1 {
                *((p + (pages - 1) * PAGE) as *mut usize) = pages;
            }
        }
        self.lists[l] = s;
        if l > 0 {
            self.nonempty |= 1 << (l - 1);
        }
    }

    unsafe fn remove(&mut self, s: *mut FreeSpan) {
        // SAFETY: s is a listed free span.
        unsafe {
            let pages = (*s).pages;
            let (m, i, _) = locate(s as usize);
            set_bit(&mut m.start, i, false);
            set_bit(&mut m.end, i + pages - 1, false);
            let l = Self::list_of(pages);
            if (*s).prev.is_null() {
                self.lists[l] = (*s).next;
            } else {
                (*(*s).prev).next = (*s).next;
            }
            if !(*s).next.is_null() {
                (*(*s).next).prev = (*s).prev;
            }
            if l > 0 && self.lists[l].is_null() {
                self.nonempty &= !(1 << (l - 1));
            }
        }
    }

    unsafe fn alloc(&mut self, n: usize) -> *mut u8 {
        loop {
            // smallest non-empty exact list of at least n pages, else first fit among longer spans
            let mut found: *mut FreeSpan = null_mut();
            let mask = if n <= SPAN_LISTS { self.nonempty >> (n - 1) } else { 0 };
            if mask != 0 {
                found = self.lists[n + mask.trailing_zeros() as usize];
            } else {
                let mut s = self.lists[0];
                while !s.is_null() {
                    // SAFETY: listed spans are valid.
                    if unsafe { (*s).pages } >= n {
                        found = s;
                        break;
                    }
                    s = unsafe { (*s).next };
                }
            }
            if !found.is_null() {
                // SAFETY: found is a listed free span of at least n pages.
                unsafe {
                    let m = (*found).pages;
                    self.remove(found);
                    if m > n {
                        self.insert(found as usize + n * PAGE, m - n);
                    }
                }
                return found as *mut u8;
            }
            let base = map_aligned(SPAN_CHUNK, SPAN_CHUNK);
            if base.is_null() {
                return base;
            }
            // SAFETY: a fresh chunk; pages 1.. are one free span.
            unsafe { self.insert(base as usize + PAGE, CPAGES - 1) };
        }
    }

    unsafe fn free(&mut self, p: usize, pages: usize) {
        let (m, i, base) = locate(p);
        let (mut start, mut n) = (p, pages);
        // SAFETY: neighbours flagged in the bitmaps are listed free spans of the same chunk.
        unsafe {
            if i + pages < CPAGES && bit(&m.start, i + pages) {
                let r = (base + (i + pages) * PAGE) as *mut FreeSpan;
                n += (*r).pages;
                self.remove(r);
            }
            if i > 1 && bit(&m.end, i - 1) {
                let len = *((base + (i - 1) * PAGE) as *const usize);
                let l = (base + (i - len) * PAGE) as *mut FreeSpan;
                self.remove(l);
                start = l as usize;
                n += len;
            }
            self.insert(start, n);
        }
    }

    /// Resizes the span at p in place (shrinking frees the tail; growing takes a free right
    /// neighbour). False if it cannot grow in place.
    unsafe fn resize(&mut self, p: usize, old: usize, new: usize) -> bool {
        if new == old {
            return true;
        }
        // SAFETY: p is an allocated span of `old` pages.
        unsafe {
            if new < old {
                self.free(p + new * PAGE, old - new);
                return true;
            }
            let (m, i, base) = locate(p);
            if i + old < CPAGES && bit(&m.start, i + old) {
                let r = (base + (i + old) * PAGE) as *mut FreeSpan;
                let have = (*r).pages;
                if have >= new - old {
                    self.remove(r);
                    if have > new - old {
                        self.insert(base + (i + new) * PAGE, have - (new - old));
                    }
                    return true;
                }
            }
        }
        false
    }
}

fn span_alloc(size: usize) -> *mut u8 {
    if STATS {
        stats::SPAN_LIVE.fetch_add((size.div_ceil(PAGE) * PAGE) as isize, Ordering::Relaxed);
        stats::note();
    }
    SPAN_LOCK.lock();
    // SAFETY: under SPAN_LOCK.
    let p = unsafe { (*SPANS.0.get()).alloc(size.div_ceil(PAGE)) };
    SPAN_LOCK.unlock();
    p
}

fn span_free(p: *mut u8, size: usize) {
    if STATS {
        stats::SPAN_LIVE.fetch_sub((size.div_ceil(PAGE) * PAGE) as isize, Ordering::Relaxed);
    }
    SPAN_LOCK.lock();
    // SAFETY: under SPAN_LOCK; p is an allocated span of that many pages.
    unsafe { (*SPANS.0.get()).free(p as usize, size.div_ceil(PAGE)) };
    SPAN_LOCK.unlock();
}

fn span_resize(p: *mut u8, old: usize, new: usize) -> bool {
    SPAN_LOCK.lock();
    // SAFETY: under SPAN_LOCK; p is an allocated span of old's pages.
    let ok = unsafe { (*SPANS.0.get()).resize(p as usize, old.div_ceil(PAGE), new.div_ceil(PAGE)) };
    SPAN_LOCK.unlock();
    if ok && STATS {
        let d = (new.div_ceil(PAGE) as isize - old.div_ceil(PAGE) as isize) * PAGE as isize;
        stats::SPAN_LIVE.fetch_add(d, Ordering::Relaxed);
        stats::note();
    }
    ok
}

/// Blocks aligned beyond 16 bytes: an ordinary block of size + align + 16, the pointer moved up
/// to the alignment, and the distance stored in the word before it.
#[inline(always)]
fn padded(l: Layout) -> usize {
    l.size() + l.align() + 16
}

pub struct Alloc;

impl Alloc {
    #[inline(always)]
    unsafe fn raw_alloc(size: usize) -> (*mut u8, bool) {
        if STATS {
            use core::sync::atomic::Ordering::Relaxed;
            static ONCE: AtomicBool = AtomicBool::new(false);
            if !ONCE.swap(true, Relaxed) {
                // SAFETY: registering a plain function.
                unsafe { atexit(print_stats) };
            }
            if size <= CLASS_MAX {
                let c = class_of(size);
                let v = stats::CLIVE[c].fetch_add(class_size(c) as isize, Relaxed) + class_size(c) as isize;
                if v > stats::CPEAK[c].load(Relaxed) {
                    stats::CPEAK[c].store(v, Relaxed);
                }
                stats::SMALL_LIVE.fetch_add(class_size(class_of(size)) as isize, Relaxed);
                stats::N[0].fetch_add(1, Relaxed);
                stats::note();
            }
        }
        if size <= CLASS_MAX {
            // SAFETY: class_of(size) < NCLASS.
            (with_heap(|h| unsafe { h.alloc(class_of(size)) }), false)
        } else if size <= SPAN_MAX {
            (span_alloc(size), false)
        } else {
            large_alloc(size)
        }
    }

    #[inline(always)]
    unsafe fn raw_free(p: *mut u8, size: usize) {
        if STATS && size <= CLASS_MAX {
            stats::SMALL_LIVE.fetch_sub(class_size(class_of(size)) as isize, Ordering::Relaxed);
            stats::CLIVE[class_of(size)].fetch_sub(class_size(class_of(size)) as isize, Ordering::Relaxed);
        }
        if size <= CLASS_MAX {
            // SAFETY: p came from raw_alloc(size).
            with_heap(|h| unsafe { h.free(p, class_of(size)) })
        } else if size <= SPAN_MAX {
            span_free(p, size)
        } else {
            large_free(p, size)
        }
    }

    /// Over-aligned requests for the large tier go there unpadded: large blocks are whole
    /// mappings, page-aligned (2 MiB-aligned from HUGE up). Padding them pushed a power-of-two
    /// size just past its bucket: twice the mapping, and under HUGE no huge-page advice (the
    /// dedup builder measured 27K faults in one stage). alloc and free decide alike from the layout.
    #[inline(always)]
    fn direct(l: Layout) -> bool {
        l.align() <= PAGE && l.size() > SPAN_MAX
    }

    unsafe fn aligned_alloc(l: Layout, zero: bool) -> *mut u8 {
        if Self::direct(l) {
            // SAFETY: forwarded contract; the large tier returns page-aligned blocks.
            let (p, fresh) = unsafe { Self::raw_alloc(l.size()) };
            if !p.is_null() && zero && !fresh {
                // SAFETY: the block holds at least l.size() bytes.
                unsafe { ptr::write_bytes(p, 0, l.size()) };
            }
            return p;
        }
        // SAFETY: forwarded contract.
        let (raw, fresh) = unsafe { Self::raw_alloc(padded(l)) };
        if raw.is_null() {
            return raw;
        }
        let a = (raw as usize + 16 + l.align() - 1) & !(l.align() - 1);
        // SAFETY: a - 8 >= raw + 8 and a + size <= raw + padded(l), both inside the block.
        unsafe {
            *((a - 8) as *mut usize) = a - raw as usize;
            if zero && !fresh {
                ptr::write_bytes(a as *mut u8, 0, l.size());
            }
        }
        a as *mut u8
    }

    unsafe fn aligned_free(p: *mut u8, l: Layout) {
        if Self::direct(l) {
            // SAFETY: aligned_alloc took this block from raw_alloc(l.size()) unpadded.
            return unsafe { Self::raw_free(p, l.size()) };
        }
        // SAFETY: aligned_alloc stored the distance to the block start before p.
        unsafe {
            let raw = p as usize - *((p as usize - 8) as *const usize);
            Self::raw_free(raw as *mut u8, padded(l));
        }
    }
}

unsafe impl GlobalAlloc for Alloc {
    #[inline]
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if l.align() <= 16 {
            // SAFETY: forwarded contract.
            unsafe { Self::raw_alloc(l.size()).0 }
        } else {
            // SAFETY: forwarded contract.
            unsafe { Self::aligned_alloc(l, false) }
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        if l.align() > 16 {
            // SAFETY: forwarded contract.
            return unsafe { Self::aligned_alloc(l, true) };
        }
        // SAFETY: forwarded contract.
        let (p, fresh) = unsafe { Self::raw_alloc(l.size()) };
        if !p.is_null() && !fresh {
            // SAFETY: the block holds at least l.size() bytes.
            unsafe { ptr::write_bytes(p, 0, l.size()) };
        }
        p
    }

    #[inline]
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if l.align() <= 16 {
            // SAFETY: forwarded contract.
            unsafe { Self::raw_free(p, l.size()) }
        } else {
            // SAFETY: forwarded contract.
            unsafe { Self::aligned_free(p, l) }
        }
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        let old = l.size();
        if l.align() <= 16 {
            if old <= CLASS_MAX && new_size <= CLASS_MAX {
                let (co, cn) = (class_of(old), class_of(new_size));
                if co == cn || with_heap(|h| h.resize_top(p, class_size(co), class_size(cn))) {
                    return p;
                }
            } else if old > CLASS_MAX && old <= SPAN_MAX && new_size > CLASS_MAX && new_size <= SPAN_MAX {
                if span_resize(p, old, new_size) {
                    return p;
                }
            } else if old > SPAN_MAX && new_size > SPAN_MAX {
                let q = large_realloc(p, old, new_size);
                if !q.is_null() {
                    return q;
                }
            }
        }
        // SAFETY: the new layout keeps the alignment and has a non-zero size.
        let nl = unsafe { Layout::from_size_align_unchecked(new_size, l.align()) };
        // SAFETY: forwarded contract.
        let q = unsafe { self.alloc(nl) };
        if !q.is_null() {
            // SAFETY: both blocks hold at least min(old, new_size) bytes and do not overlap.
            unsafe {
                ptr::copy_nonoverlapping(p, q, old.min(new_size));
                self.dealloc(p, l);
            }
        }
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes() {
        assert_eq!(class_size(NCLASS - 1), SMALL_MAX);
        for c in 0..NCLASS {
            let s = class_size(c);
            assert_eq!(s % 16, 0);
            assert_eq!(class_of(s), c, "size {s}");
            if c + 1 < NCLASS {
                assert_eq!(class_of(s + 1), c + 1, "size {}", s + 1);
                assert!(class_size(c + 1) as f64 <= s as f64 * 1.26 || c < 8);
            }
        }
        assert_eq!(class_of(1), 0);
        assert_eq!(bucket(SMALL_MAX + 1), 7);
        assert_eq!(bucket_bytes(bucket(SMALL_MAX + 1)), 512 << 10);
        assert_eq!(bucket(1 << 20), 8);
        assert_eq!(bucket((1 << 20) + 1), 9);
    }

    fn fill(p: *mut u8, n: usize, seed: u8) {
        for i in 0..n {
            unsafe { *p.add(i) = seed.wrapping_add((i % 251) as u8) };
        }
    }

    fn check(p: *const u8, n: usize, seed: u8) {
        for i in 0..n {
            assert_eq!(unsafe { *p.add(i) }, seed.wrapping_add((i % 251) as u8), "byte {i} of {n}");
        }
    }

    /// Many threads allocate, grow, shrink and free random sizes and alignments, freeing some of
    /// each other's blocks; contents are checked before every free. Threads exit and are replaced,
    /// so heaps change hands.
    #[test]
    fn stress_threads() {
        use std::sync::Mutex;
        let shared: Mutex<Vec<(usize, Layout, u8)>> = Mutex::new(Vec::new());
        for wave in 0..3 {
            std::thread::scope(|s| {
                for t in 0..8u64 {
                    let shared = &shared;
                    s.spawn(move || {
                        let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ (t * 7919 + wave * 104729);
                        let mut next = move || {
                            x ^= x << 13;
                            x ^= x >> 7;
                            x ^= x << 17;
                            x
                        };
                        let mut live: Vec<(usize, Layout, u8)> = Vec::new();
                        for step in 0..20_000 {
                            let r = next();
                            let size = match r % 50 {
                                0..=29 => 1 + (next() % 200) as usize,
                                30..=39 => 1 + (next() % 9_000) as usize,
                                40..=47 => 1 + (next() % 270_000) as usize,
                                48 => 1 + (next() % 1_000_000) as usize,
                                _ => 1 + (next() % 4_000_000) as usize,
                            };
                            let align = [8usize, 16, 32, 64, 4096][(next() % 5) as usize];
                            let l = Layout::from_size_align(size, align).unwrap();
                            let seed = (next() % 256) as u8;
                            // bounded live set: at most 600 blocks a thread
                            let op = if live.len() >= 600 { 3 } else { next() % 6 };
                            match (op, live.is_empty()) {
                                (0 | 1 | 2, _) | (_, true) => {
                                    let zero = next() % 4 == 0;
                                    let p = unsafe { if zero { Alloc.alloc_zeroed(l) } else { Alloc.alloc(l) } };
                                    assert!(!p.is_null());
                                    assert_eq!(p as usize % align, 0);
                                    if zero {
                                        assert!((0..size).all(|i| unsafe { *p.add(i) } == 0));
                                    }
                                    fill(p, size, seed);
                                    live.push((p as usize, l, seed));
                                }
                                (3, false) => {
                                    let i = (next() as usize) % live.len();
                                    let (p, l, sd) = live.swap_remove(i);
                                    check(p as *const u8, l.size(), sd);
                                    unsafe { Alloc.dealloc(p as *mut u8, l) };
                                }
                                (4, false) => {
                                    let i = (next() as usize) % live.len();
                                    let (p, l, sd) = live[i];
                                    check(p as *const u8, l.size(), sd);
                                    let q = unsafe { Alloc.realloc(p as *mut u8, l, size) };
                                    assert!(!q.is_null());
                                    assert_eq!(q as usize % l.align(), 0);
                                    let nl = Layout::from_size_align(size, l.align()).unwrap();
                                    check(q, l.size().min(size), sd);
                                    fill(q, size, sd);
                                    live[i] = (q as usize, nl, sd);
                                }
                                _ => {
                                    // hand a block to another thread to free
                                    let i = (next() as usize) % live.len();
                                    shared.lock().unwrap().push(live.swap_remove(i));
                                    if step % 7 == 0 {
                                        if let Some((p, l, sd)) = shared.lock().unwrap().pop() {
                                            check(p as *const u8, l.size(), sd);
                                            unsafe { Alloc.dealloc(p as *mut u8, l) };
                                        }
                                    }
                                }
                            }
                        }
                        for (p, l, sd) in live {
                            check(p as *const u8, l.size(), sd);
                            unsafe { Alloc.dealloc(p as *mut u8, l) };
                        }
                    });
                }
            });
        }
        for (p, l, sd) in shared.into_inner().unwrap() {
            check(p as *const u8, l.size(), sd);
            unsafe { Alloc.dealloc(p as *mut u8, l) };
        }
        // Leak check: 8 threads x 600 blocks (about 2% of 4 MB average, 2% of 1 MB) stay well under
        // 2 GiB live, so three waves must reuse memory rather than map ever more.
        assert!(mapped_bytes() < 16 << 30, "mapped {} MiB", mapped_bytes() >> 20);
        eprintln!("stress: mapped {} MiB", mapped_bytes() >> 20);
    }
}
