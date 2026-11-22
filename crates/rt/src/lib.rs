//! `rt`: the runtime layer of the fast path, between the operating system and the data code.
//!
//! No dependencies and no standard library: every operating-system service is a direct system
//! call (`sys`). Pieces, each usable on its own:
//!
//! - `pool`: persistent pinned workers, prioritised task queues, continuations, waits.
//! - `input`: directory listing, one read-only reservation for all inputs, release-behind of
//!   consumed ranges, fault-around probe, pread reader.
//! - `arena`: regions recycled between stages, bump and fixed-block allocators, huge-page verdict.
//! - `out`: output folders and files by direct calls.
//! - `exit`: parallel release then exit_group.
//! - `fail`: re-execute as the general engine on any failure or fault signal.
//! - `clock`: stage marks and the end-of-run report.
//! - `start`: an entry point for binaries without the standard library (feature-free; see docs).

#![no_std]
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc, clippy::new_without_default)]

pub mod arena;
pub mod clock;
pub mod cpu;
pub mod exit;
pub mod fail;
pub mod fmt;
pub mod input;
pub mod out;
pub mod pool;
pub mod sys;
pub mod tls;

/// First call of the process, while it is still single-threaded: starts the clock and grows the
/// descriptor table for `fds` descriptors (8192 if 0; 1024 if the limit refuses that). Returns
/// whether the table was grown.
pub fn init(fds: usize) -> bool {
    clock::init();
    let n = if fds == 0 { 8192 } else { fds };
    sys::reserve_fds(n) || sys::reserve_fds(1024)
}
