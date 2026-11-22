//! parselab: bytes to records for Shopping Sessions (RECORDS.md version 3).
//!
//! - `records`: the batch shapes and the `Sink` trait (the contract with the engine).
//! - `reference`: the plain scalar parser, straight from rules.md; the judge for the fast paths.
//! - `fast` (per-line v1), `fast2` (block structure + 8-line lane batches), `fast3` (AVX-512).
//! - `dataset`, `tables`, `par`: opening a dataset, the lookup tables and context, threading.

pub mod check;
pub mod csv;
pub mod dataset;
pub mod equiv;
pub mod fast;
pub mod fast2;
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;
pub mod fast3;
pub mod ids;
pub mod mutate;
pub mod par;
pub mod records;
pub mod reference;
pub mod tables;
pub mod ts;

pub use dataset::{Dataset, MapMode, Stream};
pub use records::*;
pub use tables::{Ctx, Fallback};

/// The fast path's entry: the context (window, products, campaigns). Err means the run goes to
/// the general engine. No key recovery: IDs are opaque keys (RECORDS.md v3).
pub fn prepare(ds: &Dataset) -> Result<Ctx, Fallback> {
    ds.ctx()
}

/// Runs the fast parser over every file of the dataset on `threads` threads, cutting files into
/// ranges of about `chunk_bytes` at arbitrary bytes (0: shrinking chunks, `par::plan_shrinking`).
pub fn fast_all<S: Sink + Default + Send>(
    ds: &Dataset,
    ctx: &Ctx,
    threads: usize,
    isa: fast::Isa,
    chunk_bytes: usize,
) -> Vec<S> {
    fast_streams(ds, ctx, threads, isa, chunk_bytes, &[Stream::Clicks, Stream::Links, Stream::Events])
}

/// As `fast_all`, for the given streams only.
pub fn fast_streams<S: Sink + Default + Send>(
    ds: &Dataset,
    ctx: &Ctx,
    threads: usize,
    isa: fast::Isa,
    chunk_bytes: usize,
    streams: &[Stream],
) -> Vec<S> {
    let f = fast::Fast::new(ctx);
    let chunks = if chunk_bytes == 0 { par::plan_shrinking(ds, streams, threads) } else { par::plan(ds, streams, chunk_bytes) };
    par::run(ds, &chunks, threads, |c, data, sink: &mut S| {
        let kind = match c.stream {
            Stream::Events => fast::Kind::Events,
            Stream::Clicks => fast::Kind::Clicks,
            Stream::Links => fast::Kind::Links,
        };
        fast::parse(&f, isa, kind, c.file, fast::Buf::plain(data), c.from, c.to, sink)
    })
}

/// Runs the reference parser over every file of the dataset on `threads` threads.
pub fn reference_all<S: Sink + Default + Send>(ds: &Dataset, ctx: &Ctx, threads: usize) -> Vec<S> {
    let chunks = par::plan(ds, &[Stream::Events, Stream::Clicks, Stream::Links], 8 << 20);
    par::run(ds, &chunks, threads, |c, data, sink: &mut S| match c.stream {
        Stream::Events => reference::events(ctx, c.file, data, c.from, c.to, sink),
        Stream::Clicks => reference::clicks(ctx, c.file, data, c.from, c.to, sink),
        Stream::Links => reference::links(ctx, c.file, data, c.from, c.to, sink),
    })
}
