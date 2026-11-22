//! Stage 3: events dedup (step 2), then the kept copy's validity and lateness (steps 3 and 4,
//! already evaluated per copy at parse time), then hand-off to the person partitions.
//!
//! Input: every copy whose event_id falls in one radix partition (all copies of an event_id
//! are in the same partition, whichever thread parsed them). Output: one `Kept` per surviving
//! event, partitioned by person.

use crate::hash::FastMap;
use crate::links::Persons;
use crate::parse::{EventRec, Scatter};

/// A surviving event, assigned to its person.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Kept {
    pub person: u64,
    pub ts: i64,
    /// price_cents for a purchase, 0 otherwise.
    pub price: u32,
    pub cat: u32,
}

/// Dedups one partition. `parts` are the partition's copies from each parser thread;
/// `table` is scratch reused across calls.
pub fn partition(
    parts: &[Vec<EventRec>],
    persons: &Persons,
    table: &mut FastMap<u64, (u32, u32)>,
    out: &mut Scatter<Kept>,
) {
    table.clear();
    let total: usize = parts.iter().map(|p| p.len()).sum();
    table.reserve(total);
    for (t, part) in parts.iter().enumerate() {
        for (i, rec) in part.iter().enumerate() {
            let here = (t as u32, i as u32);
            table
                .entry(rec.eid)
                .and_modify(|best| {
                    let kept = &parts[best.0 as usize][best.1 as usize];
                    // Equal ingest_ts copies are identical (rules.md): keep the first seen.
                    if rec.ingest < kept.ingest {
                        *best = here;
                    }
                })
                .or_insert(here);
        }
    }
    for &(t, i) in table.values() {
        let rec = &parts[t as usize][i as usize];
        if rec.user == 0 {
            continue;
        }
        let person = persons.person(rec.user);
        out.push(person, Kept { person, ts: rec.ts, price: rec.price, cat: rec.cat });
    }
}
