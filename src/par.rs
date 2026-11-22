//! A minimal work pool: `threads` scoped workers pull task indexes from one atomic counter.

use std::sync::atomic::{AtomicUsize, Ordering};

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
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (next, init, work) = (&next, &init, &work);
                scope.spawn(move || {
                    let mut state = init(t);
                    loop {
                        let task = next.fetch_add(1, Ordering::Relaxed);
                        if task >= tasks {
                            break;
                        }
                        work(&mut state, task);
                    }
                    state
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("worker panicked")).collect()
    })
}
