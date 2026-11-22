//! A cheap stage clock and a compact end-of-run report.
//!
//! `ticks()` is the time-stamp counter (x86-64 `rdtsc`, aarch64 `cntvct_el0`): about 20 cycles,
//! no system call. On x86-64 the tick rate is calibrated without spending time: the clock keeps
//! the (ticks, CLOCK_MONOTONIC) pair taken at `init()` and divides by a second pair taken when a
//! conversion is first needed (at the report, after the run), so the calibration spans the run.

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering::*};

use crate::fmt::Buf;
use crate::sys;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn ticks() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn ticks() -> u64 {
    let v: u64;
    unsafe { core::arch::asm!("mrs {}, cntvct_el0", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

static T0: AtomicU64 = AtomicU64::new(0);
static N0: AtomicU64 = AtomicU64::new(0);
/// Ticks per second times 2^0 (0 = not yet known).
static HZ: AtomicU64 = AtomicU64::new(0);

/// Takes the reference pair. Called by `pool::start`; harmless to call again (keeps the first).
pub fn init() {
    if T0.load(Relaxed) == 0 {
        let n = sys::mono_ns();
        let t = ticks();
        N0.store(n, Relaxed);
        T0.store(t, Relaxed);
    }
    #[cfg(target_arch = "aarch64")]
    {
        let f: u64;
        unsafe { core::arch::asm!("mrs {}, cntfrq_el0", out(reg) f, options(nomem, nostack, preserves_flags)) };
        HZ.store(f, Relaxed);
    }
}

/// The tick rate in Hz. On x86-64 the first call after at least 2 ms since `init` fixes it;
/// before that a short calibration (about 1 ms) is done.
pub fn hz() -> u64 {
    let h = HZ.load(Relaxed);
    if h != 0 {
        return h;
    }
    if T0.load(Relaxed) == 0 {
        init();
    }
    let (t0, n0) = (T0.load(Relaxed), N0.load(Relaxed));
    let mut n1 = sys::mono_ns();
    let mut t1 = ticks();
    while n1 < n0 + 2_000_000 {
        core::hint::spin_loop();
        n1 = sys::mono_ns();
        t1 = ticks();
    }
    let h = ((t1 - t0) as u128 * 1_000_000_000 / (n1 - n0) as u128) as u64;
    HZ.store(h, Relaxed);
    h
}

/// A rough rate for spin budgets only, never for reports: from a calibration over `us`
/// microseconds (x86-64), or the architectural rate (aarch64).
pub fn rough_hz(us: u64) -> u64 {
    let h = HZ.load(Relaxed);
    if h != 0 {
        return h;
    }
    let n0 = sys::mono_ns();
    let t0 = ticks();
    let mut n1 = n0;
    let mut t1 = t0;
    while n1 < n0 + us * 1000 {
        core::hint::spin_loop();
        n1 = sys::mono_ns();
        t1 = ticks();
    }
    ((t1 - t0) as u128 * 1_000_000_000 / (n1 - n0).max(1) as u128) as u64
}

#[inline]
pub fn ticks_to_ns(dt: u64) -> u64 {
    (dt as u128 * 1_000_000_000 / hz().max(1) as u128) as u64
}

/// CLOCK_MONOTONIC (ns) of a tick value, through the reference pair.
pub fn ticks_to_mono(t: u64) -> u64 {
    let t0 = T0.load(Relaxed);
    let n0 = N0.load(Relaxed);
    if t >= t0 { n0 + ticks_to_ns(t - t0) } else { n0 - ticks_to_ns(t0 - t) }
}

// ------------------------------------------------------------------------------------------
// Stage marks.

const MARKS: usize = 128;
static MARK_N: AtomicUsize = AtomicUsize::new(0);
static MARK_T: [AtomicU64; MARKS] = [const { AtomicU64::new(0) }; MARKS];
static MARK_NAME: [AtomicUsize; MARKS] = [const { AtomicUsize::new(0) }; MARKS];
static MARK_LEN: [AtomicUsize; MARKS] = [const { AtomicUsize::new(0) }; MARKS];
/// Set by `enable_report`; marks are recorded either way (they cost one rdtsc and three stores).
static REPORT: AtomicU64 = AtomicU64::new(0);
/// An external time origin in CLOCK_MONOTONIC ns (for example the launcher's spawn instant).
static ORIGIN: AtomicU64 = AtomicU64::new(0);

/// Records "stage `name` ended now". Any thread.
#[inline]
pub fn mark(name: &'static str) {
    let t = ticks();
    let i = MARK_N.fetch_add(1, Relaxed);
    if i < MARKS {
        MARK_NAME[i].store(name.as_ptr() as usize, Relaxed);
        MARK_LEN[i].store(name.len(), Relaxed);
        MARK_T[i].store(t, Release);
    }
}

pub fn enable_report(on: bool) {
    REPORT.store(on as u64, Relaxed);
}
pub fn report_enabled() -> bool {
    REPORT.load(Relaxed) != 0
}
/// Times in the report are measured from this CLOCK_MONOTONIC instant when set, else from `init`.
pub fn set_origin(mono_ns: u64) {
    ORIGIN.store(mono_ns, Relaxed);
}

/// Writes the marks, resource use and pool facts to stderr (only when enabled).
pub fn report() {
    if !report_enabled() {
        return;
    }
    let mut b = Buf::<8192>::new();
    let origin = match ORIGIN.load(Relaxed) {
        0 => N0.load(Relaxed),
        o => o,
    };
    b.s(b"rt report (ms from ").s(if ORIGIN.load(Relaxed) != 0 { b"origin" } else { b"rt init" }).s(b")\n");
    let n = MARK_N.load(Acquire).min(MARKS);
    let mut prev = origin;
    for i in 0..n {
        let t = ticks_to_mono(MARK_T[i].load(Acquire));
        let name = unsafe { core::slice::from_raw_parts(MARK_NAME[i].load(Relaxed) as *const u8, MARK_LEN[i].load(Relaxed)) };
        b.s(b"  ").pad(9, |x| {
            x.fix(t.saturating_sub(origin) / 1000, 3);
        });
        b.s(b"  +").pad(8, |x| {
            x.fix(t.saturating_sub(prev) / 1000, 3);
        });
        b.s(b"  ").s(name).c(b'\n');
        prev = t;
    }
    if crate::pool::started() {
        let l = crate::pool::layout();
        b.s(b"  cpus ").u(l.allowed_cpus as u64).s(b" allowed on ").u(l.allowed_cores as u64).s(b" cores (siblings ");
        b.s(if l.known { b"read" } else { b"unknown" }).s(b"); ").u(l.workers as u64).s(b" workers, ");
        if l.pinned {
            b.s(b"pinned to ").u(l.worker_cores as u64).s(b" cores\n");
        } else {
            b.s(b"not pinned\n");
        }
    }
    let r = sys::getrusage(sys::RUSAGE_SELF);
    b.s(b"  user ").fix(r.user_us() as u64, 6).s(b" s, sys ").fix(r.sys_us() as u64, 6);
    b.s(b" s, minflt ").i(r.minflt).s(b", majflt ").i(r.majflt);
    b.s(b", vcsw ").i(r.nvcsw).s(b", ivcsw ").i(r.nivcsw).s(b", maxrss ").i(r.maxrss / 1024).s(b" MB\n");
    b.flush(2);
}
