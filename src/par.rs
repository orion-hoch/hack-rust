//! A minimal work pool: `threads` workers pull task indexes from one atomic counter.
//!
//! With `init_pool` called, the workers are persistent threads woken per call (about 10 us)
//! instead of threads spawned and joined per call (0.2 ms under real-time priority on the
//! loaded x86 box, 4.6 ms without); without it, `run` spawns scoped threads as before.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

type Job = dyn Fn(usize) + Sync;

struct State {
    generation: u64,
    job: Option<&'static Job>,
    active: usize,
    pending: usize,
    /// A worker's job panicked during this generation (re-raised on the caller).
    panicked: bool,
}

struct Pool {
    state: Mutex<State>,
    wake: Condvar,
    done: Condvar,
    threads: usize,
}

static POOL: OnceLock<&'static Pool> = OnceLock::new();

thread_local! {
    static IN_POOL: Cell<bool> = const { Cell::new(false) };
}

/// Starts `threads - 1` persistent workers (the caller is the last participant). Idempotent.
pub fn init_pool(threads: usize) {
    if threads < 2 {
        return;
    }
    POOL.get_or_init(|| {
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            state: Mutex::new(State { generation: 0, job: None, active: 0, pending: 0, panicked: false }),
            wake: Condvar::new(),
            done: Condvar::new(),
            threads,
        }));
        for w in 1..threads {
            std::thread::spawn(move || worker(pool, w));
        }
        pool
    });
}

fn worker(pool: &'static Pool, w: usize) {
    IN_POOL.with(|c| c.set(true));
    let mut seen = 0u64;
    loop {
        let job = {
            let mut s = pool.state.lock().unwrap_or_else(|e| e.into_inner());
            while s.generation == seen {
                s = pool.wake.wait(s).unwrap_or_else(|e| e.into_inner());
            }
            seen = s.generation;
            if w >= s.active {
                continue;
            }
            s.job.expect("an active generation has a job")
        };
        let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(w))).is_ok();
        let mut s = pool.state.lock().unwrap_or_else(|e| e.into_inner());
        if !ok {
            s.panicked = true;
        }
        s.pending -= 1;
        if s.pending == 0 {
            pool.done.notify_all();
        }
    }
}

/// Runs `f(t)` for t in 0..n on the pool (t = 0 on the caller) and returns when all are done.
fn broadcast<'a>(pool: &Pool, n: usize, f: &'a (dyn Fn(usize) + Sync + 'a)) {
    // SAFETY: the reference is only used by workers between publishing it here and `pending`
    // reaching 0 below, and this function does not return before that; `job` is cleared after.
    let f: &'static Job = unsafe { std::mem::transmute::<&'a (dyn Fn(usize) + Sync + 'a), &'static Job>(f) };
    {
        let mut s = pool.state.lock().unwrap_or_else(|e| e.into_inner());
        s.job = Some(f);
        s.active = n;
        s.pending = n - 1;
        s.panicked = false;
        s.generation += 1;
    }
    pool.wake.notify_all();
    // The caller's share runs caught too, so the workers are always waited for.
    let mine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(0)));
    let mut s = pool.state.lock().unwrap_or_else(|e| e.into_inner());
    while s.pending > 0 {
        s = pool.done.wait(s).unwrap_or_else(|e| e.into_inner());
    }
    s.job = None;
    let worker_panicked = s.panicked;
    drop(s);
    if let Err(p) = mine {
        std::panic::resume_unwind(p);
    }
    if worker_panicked {
        panic!("a pool worker panicked");
    }
}

/// Runs `work(state, task)` for every task in `0..tasks` on up to `threads` threads, each with
/// its own `init(thread)` state, and returns the states. Tasks are handed out in index order,
/// so callers that want the biggest first sort their task list.
pub fn run<S, I, F>(threads: usize, tasks: usize, init: I, work: F) -> Vec<S>
where
    S: Send,
    I: Fn(usize) -> S + Sync,
    F: Fn(&mut S, usize) + Sync,
{
    let threads = threads.clamp(1, tasks.max(1));
    if threads == 1 {
        let mut state = init(0);
        for task in 0..tasks {
            work(&mut state, task);
        }
        return vec![state];
    }
    let next = AtomicUsize::new(0);
    let body = |t: usize| -> S {
        let mut state = init(t);
        loop {
            let task = next.fetch_add(1, Ordering::Relaxed);
            if task >= tasks {
                break;
            }
            work(&mut state, task);
        }
        state
    };
    let pooled = POOL.get().filter(|p| p.threads >= threads && !IN_POOL.with(|c| c.get()));
    if let Some(pool) = pooled {
        let results: Vec<Mutex<Option<S>>> = (0..threads).map(|_| Mutex::new(None)).collect();
        broadcast(pool, threads, &|t| {
            let s = body(t);
            *results[t].lock().unwrap() = Some(s);
        });
        return results.into_iter().map(|m| m.into_inner().unwrap().expect("every participant ran")).collect();
    }
    let body = &body;
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads).map(|t| scope.spawn(move || body(t))).collect();
        handles.into_iter().map(|h| h.join().expect("worker panicked")).collect()
    })
}
