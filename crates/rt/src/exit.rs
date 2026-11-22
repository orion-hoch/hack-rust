//! Leaving: release what is worth releasing in parallel on all workers, then exit_group. No
//! destructors run.
//!
//! Why release at all: exit_group tears the address space down on one thread. Measured on the
//! x86 box: input still mapped costs 70 ms at p20v5 in exit; anonymous memory 68 to 104 ms per
//! GB with 4 KB pages. MADV_DONTNEED runs under the mmap lock taken for reading, so workers
//! release disjoint ranges concurrently (munmap takes it for writing and serialises: never).

use core::sync::atomic::{AtomicUsize, Ordering::*};

use crate::sys::{self, PAGE};

#[derive(Clone, Copy, Debug)]
pub struct Release {
    /// The registered input reservation: every file's pages.
    pub inputs: bool,
    /// Every registered arena region up to its high-water mark.
    pub arenas: bool,
    /// Every other private anonymous read-write mapping of at least `ANON_MIN` bytes, from
    /// /proc/self/maps (another allocator's memory): never the executable image, this crate's
    /// thread stacks, or the calling thread's stack. Only for the very end: nothing may use that
    /// memory afterwards (`leave` calls exit_group next, without destructors).
    pub anon: bool,
    /// Bytes per release task (smaller: better balance, more calls).
    pub task_bytes: usize,
}

impl Release {
    pub const ALL: Release = Release { inputs: true, arenas: true, anon: true, task_bytes: 64 << 20 };
    pub const NONE: Release = Release { inputs: false, arenas: false, anon: false, task_bytes: 64 << 20 };
}

/// Smallest foreign anonymous mapping `anon` releases (thread stacks are 2 MiB and stay).
pub const ANON_MIN: usize = 4 << 20;

/// Calls `add(start, len)` for each anonymous mapping `Release::anon` covers.
pub fn anon_ranges(add: &mut dyn FnMut(usize, usize)) {
    const CAP: usize = 2 << 20;
    let Ok(buf) = sys::map_anon(CAP) else { return };
    let buf = unsafe { core::slice::from_raw_parts_mut(buf, CAP) };
    let Ok(n) = sys::read_small(b"/proc/self/maps", buf) else { return };
    let (img_lo, img_hi) = crate::tls::image_range();
    let (thr_lo, thr_hi) = crate::pool::thread_memory();
    let (in_lo, in_hi) = crate::input::registered().map(|i| {
        let (b, s) = i.reservation();
        (b, b + s)
    }).unwrap_or((0, 0));
    let sp = &n as *const usize as usize;
    let overlaps = |a: usize, b: usize, lo: usize, hi: usize| a < hi && lo < b;
    // The previous mapping's end and whether it was a PROT_NONE guard.
    let (mut prev_end, mut prev_guard) = (0usize, false);
    for line in buf[..n].split(|&c| c == b'\n') {
        // "start-end perms offset dev inode [path]"
        let mut f = line.split(|&c| c == b' ').filter(|x| !x.is_empty());
        let (Some(range), Some(perms), Some(_off), Some(_dev), Some(inode)) = (f.next(), f.next(), f.next(), f.next(), f.next()) else { continue };
        let Some(dash) = range.iter().position(|&c| c == b'-') else { continue };
        let (a, b) = (hex(&range[..dash]), hex(&range[dash + 1..]));
        let after_guard = prev_guard && prev_end == a;
        prev_end = b;
        prev_guard = perms.starts_with(b"---");
        if f.next().is_some() || perms != b"rw-p" || inode != b"0" {
            continue;
        }
        // Directly above a guard page: a thread stack or alternate stack (musl, glibc and Rust
        // put the guard right below), possibly merged with a neighbouring mapping. Never.
        if after_guard || b <= a || b - a < ANON_MIN {
            continue;
        }
        if overlaps(a, b, img_lo, img_hi) || overlaps(a, b, thr_lo, thr_hi) || overlaps(a, b, in_lo, in_hi) || (a <= sp && sp < b) {
            continue;
        }
        let mut regional = false;
        for i in 0..16 {
            if let Some(r) = crate::arena::registered(i) {
                if overlaps(a, b, r.base(), r.base() + r.cap()) {
                    regional = true;
                }
            }
        }
        if !regional {
            add(a, b - a);
        }
    }
}

fn hex(t: &[u8]) -> usize {
    let mut v = 0usize;
    for &c in t {
        let d = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => return v,
        };
        v = v * 16 + d as usize;
    }
    v
}

static DEBUG: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Prints every released range and the excluded thread memory to stderr (diagnostics).
pub fn set_debug(on: bool) {
    DEBUG.store(on, Relaxed);
}

const MAX_RANGES: usize = 4096 + 16;
static RANGES: [(AtomicUsize, AtomicUsize); MAX_RANGES] = [const { (AtomicUsize::new(0), AtomicUsize::new(0)) }; MAX_RANGES];

/// Releases (MADV_DONTNEED) the selected memory on the pool and returns the bytes covered.
pub fn release(rel: &Release) -> usize {
    let mut n = 0;
    let mut total = 0;
    let debug = DEBUG.load(Relaxed);
    if debug {
        let (lo, hi) = crate::pool::thread_memory();
        let mut b = crate::fmt::Buf::<128>::new();
        b.s(b"rt exit: thread memory ").hex(lo as u64).c(b'-').hex(hi as u64).c(b'\n');
        b.flush(2);
    }
    let mut add = |a: usize, l: usize| {
        if debug {
            let mut b = crate::fmt::Buf::<128>::new();
            b.s(b"rt exit: release ").hex(a as u64).c(b'-').hex((a + l) as u64).c(b'\n');
            b.flush(2);
        }
        if l > 0 && n < MAX_RANGES {
            RANGES[n].0.store(a, Relaxed);
            RANGES[n].1.store(l, Relaxed);
            n += 1;
            total += l;
        }
    };
    if rel.inputs {
        if let Some(inp) = crate::input::registered() {
            for f in inp.files() {
                add(f.ptr as usize, sys::up(f.len, PAGE));
            }
        }
    }
    if rel.anon {
        anon_ranges(&mut add);
    }
    if rel.arenas {
        for i in 0..16 {
            if let Some(r) = crate::arena::registered(i) {
                let (b, used) = r.used_range();
                add(b, used);
            }
        }
    }
    if total == 0 {
        return 0;
    }
    let tb = rel.task_bytes.max(1 << 20);
    let tasks = total.div_ceil(tb);
    let step = sys::up(total.div_ceil(tasks), PAGE);
    let nranges = n;
    let one = |_w: usize, t: u32| {
        // Task t covers bytes [t * step, (t + 1) * step) of the concatenated ranges.
        let lo = t as usize * step;
        let hi = lo + step;
        let mut pos = 0;
        for r in RANGES.iter().take(nranges) {
            let (a, l) = (r.0.load(Relaxed), r.1.load(Relaxed));
            let (s, e) = (pos.max(lo), (pos + l).min(hi));
            if s < e {
                unsafe {
                    let _ = sys::madvise(a + (s - pos), e - s, sys::MADV_DONTNEED);
                }
            }
            pos += l;
            if pos >= hi {
                break;
            }
        }
    };
    if crate::pool::me() < crate::pool::MAX_WORKERS {
        crate::pool::run(0, tasks as u32, &one);
    } else {
        for t in 0..tasks {
            one(0, t as u32);
        }
    }
    total
}

/// Releases per `rel` in parallel (starting this crate's pool if it is not running: about
/// 0.35 ms), writes the report when enabled, and ends the process with `code`.
pub fn leave(code: i32, rel: &Release) -> ! {
    if !crate::pool::started() {
        let mut cfg = crate::pool::Config::DEFAULT;
        cfg.spin_us = 50;
        cfg.fault_handlers = false;
        crate::pool::start(&cfg);
    }
    release(rel);
    crate::clock::mark("released");
    crate::clock::report();
    sys::exit_group(code)
}
