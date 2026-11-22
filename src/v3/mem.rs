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
    let ok = REG.init(RESERVE, Huge::Auto(expected_peak)).is_ok()
        && B8K.attach(&REG, 8 << 10).is_ok()
        && B2K.attach(&REG, 2 << 10).is_ok()
        && B256.attach(&REG, 256).is_ok()
        && B2M.attach(&REG, 2 << 20).is_ok();
    READY.store(ok, Ordering::Release);
    ok
}

/// Faults `bytes` of fresh region memory now, split over `threads` participants of the engine
/// pool; it goes to the region's free list, first in line for takes and blocks.
pub fn prefault(bytes: usize, threads: usize) {
    if bytes == 0 || !READY.load(Ordering::Acquire) {
        return;
    }
    let threads = threads.max(1);
    let part = bytes.div_ceil(threads).next_multiple_of(rt::arena::CHUNK);
    crate::par::run(threads, threads, |_| (), |_, _| REG.prefault(part, true));
}
