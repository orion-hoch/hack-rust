//! End-to-end checks of the records against the expected tables: dedup (rule 2) over the record
//! streams, re-reading foreign lines, must give the surviving-event count and revenue that the
//! expected user_daily table sums to.

use std::collections::HashMap;
use std::path::Path;

use crate::dataset::{Dataset, Stream};
use crate::records::*;
use crate::reference::{self, EventLine};
use crate::tables::Ctx;

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Survivors {
    /// Distinct event_ids (after dedup).
    pub events: u64,
    /// Kept copies that survive rules 3 and 4.
    pub surviving: u64,
    pub purchases: u64,
    pub revenue: u64,
}

/// Why kept copies were dropped, and why lines produced nothing.
#[derive(Default, Debug, Clone)]
pub struct Drops {
    pub lines: u64,
    pub malformed_or_empty_id: u64,
    pub copies: u64,
    pub kept_invalid_user: u64,
    pub kept_invalid_ts: u64,
    pub kept_invalid_type: u64,
    pub kept_invalid_product: u64,
    pub kept_invalid_price: u64,
    pub kept_late: u64,
}

fn reasons(e: &EventLine) -> [bool; 6] {
    let late = e.valid && !e.survives;
    [e.user_id.is_empty(), e.ts.is_none(), !e.kind_ok, e.product_id.is_empty(), e.purchase && e.price.is_none(), late]
}

/// Dedup over the line meanings (the plain way): id string -> copy with the smallest ingest.
pub fn survivors_from_lines(ds: &Dataset, ctx: &Ctx) -> (Survivors, Drops) {
    let mut kept: HashMap<Vec<u8>, (i64, bool, u64, bool, [bool; 6])> = HashMap::new();
    let mut drops = Drops::default();
    let mut fields = Vec::new();
    for m in ds.stream(Stream::Events) {
        let data = m.bytes();
        crate::csv::for_each_line(data, 0, data.len(), |_, line| {
            drops.lines += 1;
            let Some(e) = reference::event_line(ctx, line, &mut fields) else {
                drops.malformed_or_empty_id += 1;
                return;
            };
            drops.copies += 1;
            let ingest = e.ingest.unwrap_or(i64::MAX);
            let v = (ingest, e.survives, e.price.unwrap_or(0) as u64, e.purchase, reasons(&e));
            kept.entry(e.event_id.to_vec())
                .and_modify(|old| {
                    if ingest < old.0 {
                        *old = v
                    }
                })
                .or_insert(v);
        });
    }
    let mut s = Survivors::default();
    for (_, (_, survives, price, purchase, why)) in kept.iter() {
        s.events += 1;
        if *survives {
            s.surviving += 1;
            if *purchase {
                s.purchases += 1;
                s.revenue += price;
            }
        } else {
            let [u, t, k, p, pr, late] = *why;
            drops.kept_invalid_user += u as u64;
            drops.kept_invalid_ts += t as u64;
            drops.kept_invalid_type += k as u64;
            drops.kept_invalid_product += p as u64;
            drops.kept_invalid_price += pr as u64;
            drops.kept_late += late as u64;
        }
    }
    (s, drops)
}

/// Dedup over the record streams, the way the engine will see them: batched copies by event key,
/// foreign lines by their event_id bytes (re-read with the reference). The two key spaces cannot
/// collide: a batched event_id is SHORT or LONG, a foreign one is FOREIGN or a copy whose
/// user_id or ingest made it foreign, so its event_id may also be SHORT/LONG: such copies are
/// merged into the keyed group by their key.
pub fn survivors_from_records(ds: &Dataset, ctx: &Ctx, sink: &VecSink) -> Survivors {
    #[derive(Clone, Copy)]
    struct Copy {
        ingest: i64,
        survives: bool,
        purchase: bool,
        price: u64,
    }
    let mut by_key: HashMap<(u8, Vec<u8>), Copy> = HashMap::new();
    let mut put = |k: (u8, Vec<u8>), c: Copy| {
        by_key
            .entry(k)
            .and_modify(|old| {
                if c.ingest < old.ingest {
                    *old = c
                }
            })
            .or_insert(c);
    };
    for r in &sink.events {
        let id = crate::ids::key_bytes(b'e', r.ev);
        put((0, id), Copy { ingest: r.ingest as i64, survives: r.surv, purchase: r.purch, price: r.price as u64 });
    }
    let mut fields = Vec::new();
    for &at in &sink.foreign_events {
        let e = reference::event_line(ctx, ds.line(Stream::Events, at), &mut fields).unwrap();
        let ingest = e.ingest.map_or(i64::MAX, |ms| ms.div_euclid(1000) - ctx.ingest_base_s);
        put(
            (0, e.event_id.to_vec()),
            Copy { ingest, survives: e.survives, purchase: e.purchase, price: e.price.unwrap_or(0) as u64 },
        );
    }
    let mut s = Survivors::default();
    for c in by_key.values() {
        s.events += 1;
        if c.survives {
            s.surviving += 1;
            if c.purchase {
                s.purchases += 1;
                s.revenue += c.price;
            }
        }
    }
    s
}

/// Sums of `events` and `revenue_cents` in an expected user_daily.csv.
pub fn expected_sums(path: &Path) -> Option<(u64, u64)> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next()?.split(',').collect();
    let ei = header.iter().position(|h| *h == "events")?;
    let ri = header.iter().position(|h| *h == "revenue_cents")?;
    let (mut e, mut r) = (0u64, 0u64);
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        e += f[ei].parse::<u64>().ok()?;
        r += f[ri].parse::<u64>().ok()?;
    }
    Some((e, r))
}
