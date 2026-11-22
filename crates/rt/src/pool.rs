//! A persistent pinned worker pool with several live task queues and no barriers.
//!
//! # Model
//! - `start` creates one worker per usable CPU (allowed set, capped by the cgroup CPU quota). The
//!   calling thread is worker 0; workers 1.. are raw `clone` threads with mmap'd stacks (guard page
//!   below, alternate signal stack and static TLS above), pinned one per CPU in core order.
//! - A queue is `n` numbered tasks `f(w, i)`, `i` in `0..n`, taken by an atomic counter. Up to 64
//!   queues are live at once in 8 priority levels (0 = most urgent); a free worker always takes
//!   from the most urgent non-empty queue, so queues interleave by priority without any barrier.
//! - Any thread may `push` at any time, including from inside a task, while other queues drain.
//! - `push_then(.., done)` runs `done(w)` exactly once, on the thread that completes the last
//!   task, before the queue counts as finished. That is how stages chain without barriers: the
//!   continuation pushes the next stage (or counts down a join of several predecessors).
//! - `wait(q)` / `wait_event(e)` return when the queue is finished / the event is set; a pool
//!   thread helps meanwhile by running tasks of any queue (highest priority first).
//! - Idle workers spin (`pause`) for `spin_us`, then sleep on one futex word that every push,
//!   completion and event bumps.
//!
//! # Why this is enough
//! Every stage of the fast path is a set of independent numbered tasks (chunks, buckets,
//! partitions). Dependencies between stages are "B may start when A (or A1..Ak) is complete":
//! a continuation expresses that with no thread blocked. Preferences ("run the union-find job
//! before more event chunks") are priorities. Fork-join inside a task is `push` + `wait`.
//! Per-worker scratch is indexed by the `w` every task receives (`PerWorker`).
//!
//! # The rule for code that runs on workers
//! Raw threads have correct Rust TLS (checked at start), but musl does not know they exist: it
//! skips its internal locks until the first `pthread_create`. Task code must not use the C
//! library's malloc unless the process has a live std thread (musl then locks), or the global
//! allocator is our own. `libc_threaded_hint` in rtcheck shows the parked-thread remedy.

use core::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::*};

use crate::clock::ticks;
use crate::cpu::{self, CpuSet};
use crate::sys::{self, PAGE};
use crate::tls;

pub const MAX_WORKERS: usize = 256;
pub const PRIOS: usize = 8;
const SLOTS: usize = 64;
const PER_PRIO: usize = SLOTS / PRIOS;

pub type TaskFn = unsafe fn(ctx: usize, w: usize, i: u32);
pub type DoneFn = unsafe fn(ctx: usize, w: usize);

#[repr(C, align(128))]
struct Line<T>(T);

#[repr(C, align(128))]
struct Desc {
    seq: AtomicU32,
    f: AtomicUsize,
    fctx: AtomicUsize,
    d: AtomicUsize,
    dctx: AtomicUsize,
}

struct State {
    /// (n << 32) | next: one fetch_add takes a task and reads the count consistently.
    take: [Line<AtomicU64>; SLOTS],
    done: [Line<AtomicU32>; SLOTS],
    desc: [Desc; SLOTS],
    active: Line<AtomicU64>,
    free: Line<AtomicU64>,
    epoch: Line<AtomicU32>,
    sleepers: Line<AtomicU32>,
}

static S: State = State {
    take: [const { Line(AtomicU64::new(0)) }; SLOTS],
    done: [const { Line(AtomicU32::new(0)) }; SLOTS],
    desc: [const {
        Desc { seq: AtomicU32::new(0), f: AtomicUsize::new(0), fctx: AtomicUsize::new(0), d: AtomicUsize::new(0), dctx: AtomicUsize::new(0) }
    }; SLOTS],
    active: Line(AtomicU64::new(0)),
    free: Line(AtomicU64::new(u64::MAX)),
    epoch: Line(AtomicU32::new(0)),
    sleepers: Line(AtomicU32::new(0)),
};

// Facts about the started pool.
static STARTED: AtomicU32 = AtomicU32::new(0);
static NW: AtomicUsize = AtomicUsize::new(1);
static RUNNING: AtomicUsize = AtomicUsize::new(0);
static PINNED: AtomicU32 = AtomicU32::new(0);
static KNOWN_TOPO: AtomicU32 = AtomicU32::new(0);
static ALLOWED: AtomicUsize = AtomicUsize::new(0);
static QUOTA: AtomicUsize = AtomicUsize::new(0);
static SPIN_TICKS: AtomicU64 = AtomicU64::new(0);
static CPU_OF: [AtomicU32; MAX_WORKERS] = [const { AtomicU32::new(u32::MAX) }; MAX_WORKERS];
static CORE_OF: [AtomicU32; MAX_WORKERS] = [const { AtomicU32::new(0) }; MAX_WORKERS];
static THREADS: AtomicUsize = AtomicUsize::new(0);
static SLOT_BYTES: AtomicUsize = AtomicUsize::new(0);
static STACK_BYTES: AtomicUsize = AtomicUsize::new(0);
static T_START: AtomicU64 = AtomicU64::new(0);
static T_LAST_RUNNING: AtomicU64 = AtomicU64::new(0);
pub(crate) static ORIG_MASK: [AtomicU64; cpu::MAX_CPUS / 64] = [const { AtomicU64::new(0) }; cpu::MAX_CPUS / 64];
pub(crate) static ORIG_SAVED: AtomicU32 = AtomicU32::new(0);

#[repr(C, align(128))]
struct WStat {
    tasks: AtomicU64,
    sleeps: AtomicU64,
}
static WSTAT: [WStat; MAX_WORKERS] = [const { WStat { tasks: AtomicU64::new(0), sleeps: AtomicU64::new(0) } }; MAX_WORKERS];

pub const ALT_STACK: usize = 64 << 10;
const GUARD: usize = 64 << 10;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pin {
    /// Pin one worker per CPU when the worker count equals the allowed CPU count; otherwise
    /// (a quota smaller than the allowed set) leave threads free to move inside the set.
    Auto,
    Yes,
    No,
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// 0: one per allowed CPU, capped by the cgroup CPU quota. Otherwise exactly this many.
    pub workers: usize,
    pub pin: Pin,
    /// Idle spin before sleeping on the futex.
    pub spin_us: u32,
    pub stack_bytes: usize,
    /// Install the SIGSEGV/SIGBUS/SIGILL/SIGFPE/SIGABRT/SIGTRAP handlers that call `fail`.
    pub fault_handlers: bool,
}

impl Config {
    pub const DEFAULT: Config = Config { workers: 0, pin: Pin::Auto, spin_us: 300, stack_bytes: 16 << 20, fault_handlers: true };
}

/// A queue handle. `Q::DONE` is a queue that is already finished.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Q {
    slot: u32,
    tag: u32,
}

impl Q {
    pub const DONE: Q = Q { slot: u32::MAX, tag: 0 };
}

/// The worker index of the calling thread, or `usize::MAX` on a thread the pool does not know.
#[inline(always)]
pub fn me() -> usize {
    tls::widx().wrapping_sub(1)
}

#[inline]
pub fn workers() -> usize {
    NW.load(Relaxed)
}

/// CPU that worker `w` is pinned to (u32::MAX when not pinned).
pub fn cpu_of(w: usize) -> u32 {
    if PINNED.load(Relaxed) != 0 { CPU_OF[w].load(Relaxed) } else { u32::MAX }
}
/// Core number of worker `w` (siblings share one; equals `w` when the topology is unknown).
pub fn core_of(w: usize) -> u32 {
    CORE_OF[w].load(Relaxed)
}
pub fn topology_known() -> bool {
    KNOWN_TOPO.load(Relaxed) != 0
}
pub fn pinned() -> bool {
    PINNED.load(Relaxed) != 0
}
/// Allowed CPUs at start and the cgroup quota (0 = none).
pub fn allowed_cpus() -> (usize, usize) {
    (ALLOWED.load(Relaxed), QUOTA.load(Relaxed))
}
/// Workers that have entered their loop (excluding worker 0).
pub fn running() -> usize {
    RUNNING.load(Acquire)
}
/// Ticks from the start of `start` to the last worker entering its loop (0 until all have).
pub fn start_to_all_running_ticks() -> u64 {
    T_LAST_RUNNING.load(Acquire).saturating_sub(T_START.load(Relaxed))
}
/// Tasks run and futex sleeps of worker `w`.
pub fn stats(w: usize) -> (u64, u64) {
    (WSTAT[w].tasks.load(Relaxed), WSTAT[w].sleeps.load(Relaxed))
}

// ------------------------------------------------------------------------------------------
// Start.

/// Starts the pool once (later calls return the worker count). Returns the worker count.
pub fn start(cfg: &Config) -> usize {
    if STARTED.swap(1, AcqRel) != 0 {
        return workers();
    }
    crate::clock::init();
    let t_a = ticks();
    let n_a = sys::mono_ns();
    T_START.store(t_a, Relaxed);

    let set = cpu::affinity();
    for (i, w) in set.bits.iter().enumerate() {
        ORIG_MASK[i].store(*w, Relaxed);
    }
    ORIG_SAVED.store(1, Release);
    let topo = cpu::topology(&set, b"/sys/devices/system/cpu");
    let quota = cpu::cgroup_quota();
    ALLOWED.store(topo.n, Relaxed);
    ALLOWED_CORES.store(topo.cores, Relaxed);
    QUOTA.store(quota.unwrap_or(0), Relaxed);
    KNOWN_TOPO.store(topo.known as u32, Relaxed);
    let mut n = if cfg.workers > 0 { cfg.workers } else { topo.n.min(quota.unwrap_or(usize::MAX)) };
    n = n.clamp(1, MAX_WORKERS);
    let pin = match cfg.pin {
        Pin::Yes => topo.n > 0,
        Pin::No => false,
        Pin::Auto => n == topo.n,
    };
    PINNED.store(pin as u32, Relaxed);
    for w in 0..n {
        let k = if topo.n > 0 { w % topo.n } else { 0 };
        CPU_OF[w].store(topo.cpu[k] as u32, Relaxed);
        CORE_OF[w].store(if topo.known { topo.core[k] as u32 } else { w as u32 }, Relaxed);
    }
    // Rough tick rate for spin budgets until the real calibration; refined below.
    SPIN_TICKS.store(cfg.spin_us as u64 * 3000, Relaxed);

    if cfg.fault_handlers {
        crate::fail::install();
    }
    tls::capture_template();
    tls::set_widx(1);

    let seg = tls::segment();
    let tls_bytes = tls::area_size(&seg);
    let stack = sys::up(cfg.stack_bytes.max(64 << 10), PAGE);
    let slot = sys::up(GUARD + stack + ALT_STACK + tls_bytes, PAGE);
    SLOT_BYTES.store(slot, Relaxed);
    STACK_BYTES.store(stack, Relaxed);
    // Slot 0 holds only worker 0's alternate signal stack; slots 1.. are the workers'.
    let region = unsafe {
        sys::mmap(0, slot * n, sys::PROT_READ | sys::PROT_WRITE, sys::MAP_PRIVATE | sys::MAP_ANONYMOUS | sys::MAP_NORESERVE | sys::MAP_STACK, -1, 0)
    };
    let region = match region {
        Ok(p) => p as usize,
        Err(_) => {
            NW.store(1, Release);
            return 1;
        }
    };
    THREADS.store(region, Relaxed);
    if cfg.fault_handlers {
        crate::fail::thread_init(region + GUARD + stack);
    }
    let mut started = 1;
    for w in 1..n {
        let base = region + w * slot;
        unsafe {
            let _ = sys::mprotect(base, GUARD, sys::PROT_NONE);
            let top = base + GUARD + stack;
            let tp = tls::init_area(&seg, top + ALT_STACK);
            let arg = w | if cfg.fault_handlers { 1 << 32 } else { 0 };
            if clone_thread(top, tp, tls::tid_field(tp), worker_entry, arg) < 0 {
                break;
            }
        }
        started += 1;
    }
    NW.store(started, Release);
    if pin {
        let _ = cpu::set_affinity(0, &CpuSet::one(CPU_OF[0].load(Relaxed) as usize));
    }
    // Spin budget from the ticks that passed during start.
    let t_b = ticks();
    let n_b = sys::mono_ns();
    if n_b > n_a + 5_000 {
        let hz = (t_b - t_a) as u128 * 1_000_000_000 / (n_b - n_a) as u128;
        SPIN_TICKS.store((cfg.spin_us as u128 * hz / 1_000_000) as u64, Relaxed);
    }
    if started == 1 {
        T_LAST_RUNNING.store(ticks(), Release);
    }
    started
}

/// The machine's shape as the pool sees it.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    /// CPUs the process may run on, and the distinct cores they cover.
    pub allowed_cpus: usize,
    pub allowed_cores: usize,
    /// Whether the sibling lists were read (false: every CPU counted as its own core).
    pub known: bool,
    /// Workers, and whether each is pinned to one CPU (then `core_of(w)` is exact; unpinned
    /// workers float over the allowed set and share cores at the scheduler's choice).
    pub workers: usize,
    pub pinned: bool,
    /// Distinct cores the workers are pinned to (= workers when each has its own core).
    pub worker_cores: usize,
}

static ALLOWED_CORES: AtomicUsize = AtomicUsize::new(0);

/// One call for layout-dependent choices: allowed CPUs and cores, and the workers' cores.
/// Workers sharing a core have equal `core_of(w)`.
pub fn layout() -> Layout {
    let n = workers();
    let mut seen = [0u64; MAX_WORKERS / 64 * 4];
    let mut wc = 0;
    for w in 0..n {
        let c = core_of(w) as usize % (seen.len() * 64);
        if seen[c / 64] >> (c % 64) & 1 == 0 {
            seen[c / 64] |= 1 << (c % 64);
            wc += 1;
        }
    }
    Layout {
        allowed_cpus: ALLOWED.load(Relaxed),
        allowed_cores: ALLOWED_CORES.load(Relaxed),
        known: topology_known(),
        workers: n,
        pinned: pinned(),
        worker_cores: if pinned() { wc } else { 0 },
    }
}

/// Whether `start` has run.
pub fn started() -> bool {
    STARTED.load(Acquire) != 0
}

/// The address range of the workers' stacks, alternate stacks and TLS (0, 0 before `start`).
pub fn thread_memory() -> (usize, usize) {
    let b = THREADS.load(Relaxed);
    (b, b + SLOT_BYTES.load(Relaxed) * NW.load(Relaxed))
}

/// Spins until every worker has entered its loop (for measurements; nothing needs it).
pub fn wait_running() {
    while RUNNING.load(Acquire) + 1 < workers() {
        core::hint::spin_loop();
    }
}

extern "C" fn worker_entry(arg: usize) -> ! {
    let w = arg & 0xffff_ffff;
    tls::set_widx(w + 1);
    if tls::magic() != tls::MAGIC {
        crate::fail::fail(b"pool: worker TLS layout check failed");
    }
    let slot = SLOT_BYTES.load(Relaxed);
    let base = THREADS.load(Relaxed) + w * slot;
    if arg >> 32 != 0 {
        crate::fail::thread_init(base + GUARD + STACK_BYTES.load(Relaxed));
    }
    if PINNED.load(Relaxed) != 0 {
        let _ = cpu::set_affinity(0, &CpuSet::one(CPU_OF[w].load(Relaxed) as usize));
    }
    let r = RUNNING.fetch_add(1, AcqRel) + 1;
    if r + 1 == NW.load(Acquire) {
        T_LAST_RUNNING.store(ticks(), Release);
    }
    idle(w)
}

#[cfg(target_arch = "x86_64")]
unsafe fn clone_thread(stack_top: usize, tp: usize, ptid: usize, entry: extern "C" fn(usize) -> !, arg: usize) -> isize {
    use core::arch::asm;
    let sp = (stack_top - 16) & !15;
    *(sp as *mut usize) = arg;
    *((sp + 8) as *mut usize) = entry as usize;
    let mut flags = sys::CLONE_VM | sys::CLONE_FS | sys::CLONE_FILES | sys::CLONE_SIGHAND | sys::CLONE_THREAD | sys::CLONE_SYSVSEM | sys::CLONE_SETTLS;
    if ptid != 0 {
        flags |= sys::CLONE_PARENT_SETTID;
    }
    let ret: isize;
    asm!(
        "syscall",
        "test rax, rax",
        "jnz 2f",
        "xor ebp, ebp",
        "pop rdi",
        "pop rax",
        "call rax",
        "ud2",
        "2:",
        inlateout("rax") sys::nr::CLONE as isize => ret,
        in("rdi") flags, in("rsi") sp, in("rdx") ptid, in("r10") 0usize, in("r8") tp,
        lateout("rcx") _, lateout("r11") _,
        options(nostack),
    );
    ret
}

#[cfg(target_arch = "aarch64")]
unsafe fn clone_thread(stack_top: usize, tp: usize, ptid: usize, entry: extern "C" fn(usize) -> !, arg: usize) -> isize {
    use core::arch::asm;
    let sp = (stack_top - 32) & !15;
    *(sp as *mut usize) = arg;
    *((sp + 8) as *mut usize) = entry as usize;
    *((sp + 16) as *mut usize) = tp;
    // TLS is set by the child itself (msr), so the clone argument order of this architecture
    // for the tls/ctid pair does not matter.
    // The parent-tid pointer is the third argument in every clone argument order.
    let mut flags = sys::CLONE_VM | sys::CLONE_FS | sys::CLONE_FILES | sys::CLONE_SIGHAND | sys::CLONE_THREAD | sys::CLONE_SYSVSEM;
    if ptid != 0 {
        flags |= sys::CLONE_PARENT_SETTID;
    }
    let ret: isize;
    asm!(
        "svc 0",
        "cbnz x0, 2f",
        "ldp x0, x1, [sp]",
        "ldr x2, [sp, #16]",
        "msr tpidr_el0, x2",
        "add sp, sp, #32",
        "mov x29, xzr",
        "mov x30, xzr",
        "blr x1",
        "brk #1",
        "2:",
        in("x8") sys::nr::CLONE,
        inlateout("x0") flags as isize => ret,
        in("x1") sp, in("x2") ptid, in("x3") 0usize, in("x4") 0usize,
        options(nostack),
    );
    ret
}

// ------------------------------------------------------------------------------------------
// Taking and running tasks.

#[inline]
unsafe fn run_one(s: usize, i: u32, n: u32, w: usize) {
    let d = &S.desc[s];
    let f: TaskFn = core::mem::transmute::<usize, TaskFn>(d.f.load(Relaxed));
    f(d.fctx.load(Relaxed), w, i);
    if w < MAX_WORKERS {
        WSTAT[w].tasks.fetch_add(1, Relaxed);
    }
    if S.done[s].0.fetch_add(1, AcqRel) + 1 == n {
        finish(s, w);
    }
}

#[cold]
unsafe fn finish(s: usize, w: usize) {
    let d = &S.desc[s];
    let df = d.d.load(Relaxed);
    if df != 0 {
        let df: DoneFn = core::mem::transmute::<usize, DoneFn>(df);
        df(d.dctx.load(Relaxed), w);
    }
    let g = d.seq.load(Relaxed);
    d.seq.store(g.wrapping_add(1), Release);
    S.free.0.fetch_or(1 << s, AcqRel);
    bump();
}

/// Wakes every sleeper (idle workers and waiters).
#[inline]
fn bump() {
    S.epoch.0.fetch_add(1, SeqCst);
    if S.sleepers.0.load(SeqCst) != 0 {
        unsafe {
            let _ = sys::futex_wake(S.epoch.0.as_ptr(), i32::MAX);
        }
    }
}

/// Takes and runs one task of the most urgent queue. False when nothing was available.
#[inline]
fn try_run(w: usize) -> bool {
    let mut act = S.active.0.load(Acquire);
    while act != 0 {
        let s = act.trailing_zeros() as usize;
        let v = S.take[s].0.fetch_add(1, AcqRel);
        let i = v as u32;
        let n = (v >> 32) as u32;
        if i < n {
            unsafe { run_one(s, i, n, w) };
            return true;
        }
        // Exhausted: clear the bit, then restore it if the slot was refilled meanwhile.
        S.active.0.fetch_and(!(1u64 << s), AcqRel);
        let v2 = S.take[s].0.load(Acquire);
        if (v2 as u32) < (v2 >> 32) as u32 {
            S.active.0.fetch_or(1u64 << s, AcqRel);
        }
        act = S.active.0.load(Acquire) & !(1u64 << s);
    }
    false
}

fn idle(w: usize) -> ! {
    loop {
        if try_run(w) {
            continue;
        }
        let t0 = ticks();
        let budget = SPIN_TICKS.load(Relaxed);
        loop {
            for _ in 0..16 {
                core::hint::spin_loop();
            }
            if S.active.0.load(Relaxed) != 0 {
                break;
            }
            if ticks().wrapping_sub(t0) > budget {
                sleep(w, &|| S.active.0.load(SeqCst) != 0);
                break;
            }
        }
    }
}

/// Sleeps on the epoch word unless `ready()` holds after announcing the sleep.
fn sleep(w: usize, ready: &dyn Fn() -> bool) {
    S.sleepers.0.fetch_add(1, SeqCst);
    let e = S.epoch.0.load(SeqCst);
    if !ready() {
        unsafe {
            let _ = sys::futex_wait(S.epoch.0.as_ptr(), e);
        }
        if w < MAX_WORKERS {
            WSTAT[w].sleeps.fetch_add(1, Relaxed);
        }
    }
    S.sleepers.0.fetch_sub(1, SeqCst);
}

// ------------------------------------------------------------------------------------------
// Queues.

fn alloc_slot(prio: usize) -> usize {
    let mask = ((1u64 << PER_PRIO) - 1) << (prio * PER_PRIO);
    loop {
        let f = S.free.0.load(Acquire);
        let avail = f & mask;
        if avail != 0 {
            let s = avail.trailing_zeros() as usize;
            if S.free.0.compare_exchange_weak(f, f & !(1u64 << s), AcqRel, Relaxed).is_ok() {
                return s;
            }
            continue;
        }
        // Every slot of this priority is live: help until one frees.
        let w = me();
        if !(w < MAX_WORKERS && try_run(w)) {
            core::hint::spin_loop();
        }
    }
}

/// Publishes a queue of `n` tasks: `f(fctx, w, i)` for each `i` in `0..n`, then `d(dctx, w)` once.
///
/// # Safety
/// `fctx`/`dctx` must stay valid until the queue is finished; `f` and `d` must be safe to call
/// concurrently from any pool thread.
pub unsafe fn push_raw(prio: usize, n: u32, f: TaskFn, fctx: usize, d: Option<DoneFn>, dctx: usize) -> Q {
    let prio = prio.min(PRIOS - 1);
    assert!(n < 1 << 31);
    if n == 0 {
        if let Some(d) = d {
            d(dctx, me());
        }
        return Q::DONE;
    }
    let s = alloc_slot(prio);
    let desc = &S.desc[s];
    desc.f.store(f as usize, Relaxed);
    desc.fctx.store(fctx, Relaxed);
    desc.d.store(d.map_or(0, |d| d as usize), Relaxed);
    desc.dctx.store(dctx, Relaxed);
    let g = desc.seq.load(Relaxed).wrapping_add(1);
    desc.seq.store(g, Relaxed);
    S.done[s].0.store(0, Relaxed);
    S.take[s].0.store((n as u64) << 32, Release);
    S.active.0.fetch_or(1u64 << s, SeqCst);
    bump();
    Q { slot: s as u32, tag: g }
}

unsafe fn tramp<F: Fn(usize, u32) + Sync>(ctx: usize, w: usize, i: u32) {
    (*(ctx as *const F))(w, i)
}
unsafe fn dtramp<D: Fn(usize) + Sync>(ctx: usize, w: usize) {
    (*(ctx as *const D))(w)
}

/// Publishes `n` tasks `f(w, i)` at priority `prio` (0 = most urgent).
///
/// # Safety
/// `f` must outlive the queue: wait for it (or for an event its continuation sets) before
/// dropping `f`.
pub unsafe fn push<F: Fn(usize, u32) + Sync>(prio: usize, n: u32, f: &F) -> Q {
    push_raw(prio, n, tramp::<F>, f as *const F as usize, None, 0)
}

/// Like `push`, and `done(w)` runs once on the thread that completes the last task, before the
/// queue counts as finished (immediately on the caller when `n == 0`).
///
/// # Safety
/// As `push`, for both closures.
pub unsafe fn push_then<F: Fn(usize, u32) + Sync, D: Fn(usize) + Sync>(prio: usize, n: u32, f: &F, done: &D) -> Q {
    push_raw(prio, n, tramp::<F>, f as *const F as usize, Some(dtramp::<D>), done as *const D as usize)
}

/// Runs `n` tasks `f(w, i)` and returns when all are done (the caller helps).
pub fn run<F: Fn(usize, u32) + Sync>(prio: usize, n: u32, f: &F) {
    let q = unsafe { push(prio, n, f) };
    wait(q);
}

#[inline]
pub fn is_done(q: Q) -> bool {
    q.slot == u32::MAX || S.desc[q.slot as usize].seq.load(Acquire) != q.tag
}

/// Returns when `q` is finished (its continuation included). A pool thread runs tasks meanwhile.
pub fn wait(q: Q) {
    wait_until(&|| is_done(q));
}

/// A one-shot event: `set` wakes every waiter.
pub struct Event(AtomicU32);

impl Default for Event {
    fn default() -> Self {
        Self::new()
    }
}

impl Event {
    pub const fn new() -> Event {
        Event(AtomicU32::new(0))
    }
    pub fn set(&self) {
        self.0.store(1, Release);
        bump();
    }
    pub fn is_set(&self) -> bool {
        self.0.load(Acquire) != 0
    }
    pub fn reset(&self) {
        self.0.store(0, Release);
    }
}

pub fn wait_event(e: &Event) {
    wait_until(&|| e.is_set());
}

/// A countdown: `arrive` returns true for the caller that brings it to zero (that caller then
/// starts what depends on all of the arrivals). For joins of several queues' continuations.
pub struct Join(AtomicU32);

impl Join {
    pub const fn new(n: u32) -> Join {
        Join(AtomicU32::new(n))
    }
    #[inline]
    pub fn arrive(&self) -> bool {
        self.0.fetch_sub(1, AcqRel) == 1
    }
}

/// Helps (on pool threads) until `cond()` holds; sleeps when there is nothing to run.
pub fn wait_until(cond: &dyn Fn() -> bool) {
    let w = me();
    let helper = w < MAX_WORKERS;
    loop {
        if cond() {
            return;
        }
        if helper && try_run(w) {
            continue;
        }
        let t0 = ticks();
        let budget = SPIN_TICKS.load(Relaxed);
        loop {
            for _ in 0..16 {
                core::hint::spin_loop();
            }
            if cond() {
                return;
            }
            if helper && S.active.0.load(Relaxed) != 0 {
                break;
            }
            if ticks().wrapping_sub(t0) > budget {
                sleep(w, &|| cond() || (helper && S.active.0.load(SeqCst) != 0));
                break;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------
// Per-worker state.

#[repr(C, align(128))]
pub struct Pad<T>(pub T);

/// One `T` per worker, each on its own pair of cache lines, in memory mapped once.
pub struct PerWorker<T> {
    ptr: *mut Pad<T>,
    n: usize,
}

unsafe impl<T: Send> Sync for PerWorker<T> {}
unsafe impl<T: Send> Send for PerWorker<T> {}

impl<T> PerWorker<T> {
    /// `init(w)` for every worker of the started pool (call after `start`).
    pub fn new(init: impl Fn(usize) -> T) -> PerWorker<T> {
        Self::with_len(workers(), init)
    }
    pub fn with_len(n: usize, init: impl Fn(usize) -> T) -> PerWorker<T> {
        let bytes = sys::up(n.max(1) * core::mem::size_of::<Pad<T>>(), PAGE);
        let ptr = sys::map_anon(bytes).unwrap_or_else(|_| crate::fail::fail(b"PerWorker: mmap failed")) as *mut Pad<T>;
        for w in 0..n {
            unsafe { ptr.add(w).write(Pad(init(w))) };
        }
        PerWorker { ptr, n }
    }
    /// # Safety
    /// Only worker `w` (or one thread at a time outside the pool) uses entry `w`.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get(&self, w: usize) -> &mut T {
        debug_assert!(w < self.n);
        &mut (*self.ptr.add(w)).0
    }
    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}
