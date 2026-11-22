//! Splitting files into byte ranges and running a parser over them on several threads.

use crate::dataset::{Dataset, Stream};

#[derive(Clone, Copy, Debug)]
pub struct Chunk {
    pub stream: Stream,
    pub file: u32,
    pub from: usize,
    pub to: usize,
}

/// Byte ranges of about `target` bytes, largest files first, each file cut at arbitrary bytes (the
/// parser resyncs on `\n`).
pub fn plan(ds: &Dataset, streams: &[Stream], target: usize) -> Vec<Chunk> {
    let mut out = Vec::new();
    for &s in streams {
        for (file, m) in ds.stream(s).iter().enumerate() {
            let len = m.len();
            if len == 0 {
                continue;
            }
            let pieces = len.div_ceil(target.max(1)).max(1);
            for p in 0..pieces {
                let from = len * p / pieces;
                let to = len * (p + 1) / pieces;
                out.push(Chunk { stream: s, file: file as u32, from, to });
            }
        }
    }
    out.sort_by_key(|c| std::cmp::Reverse(c.to - c.from));
    out
}

/// Runs `f` over every chunk on `threads` threads, one sink per thread, and returns the sinks.
pub fn run<S, F>(ds: &Dataset, chunks: &[Chunk], threads: usize, f: F) -> Vec<S>
where
    S: Default + Send,
    F: Fn(&Chunk, &[u8], &mut S) + Sync,
{
    use std::sync::atomic::{AtomicUsize, Ordering};
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads.max(1))
            .map(|_| {
                scope.spawn(|| {
                    let mut sink = S::default();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(c) = chunks.get(i) else { break };
                        let data = ds.stream(c.stream)[c.file as usize].bytes();
                        f(c, data, &mut sink);
                    }
                    sink
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}
