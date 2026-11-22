//! Pieces shared by the engines after parsing: bins, the campaign aggregate, order keys, rows and
//! the pqw writer. (The first fast path, which decoded IDs to generator numbers, was removed when
//! the organizers made IDs opaque; see FASTPATH.md.)

pub mod agg;
pub mod names;
pub mod scan;
pub mod sink;
pub mod write;

pub enum Outcome {
    Done,
    Fallback(String),
}
