//! Stage 2: bytes to typed records, one byte range at a time.
//!
//! Each parser takes one line-aligned slice and pushes records into a `Scatter`, a set of
//! radix partitions owned by the calling thread. Event records are partitioned by event_id
//! (for dedup); click records, which need no dedup, go straight to their person's partition.

use crate::csv::{for_each_line, split};
use crate::hash::bucket;
use crate::id::{Id, Interner, NULL};
use crate::links::Persons;
use crate::tables::{Campaigns, Products};
use crate::ts::{self, DAY};

/// Radix partitions of records, owned by one thread.
pub struct Scatter<T> {
    pub bits: u32,
    pub parts: Vec<Vec<T>>,
}

impl<T> Scatter<T> {
    pub fn new(bits: u32) -> Self {
        Scatter { bits, parts: (0..1usize << bits).map(|_| Vec::new()).collect() }
    }

    #[inline]
    pub fn push(&mut self, key: u64, rec: T) {
        let b = bucket(key, self.bits);
        // SAFETY: bucket() < 1 << bits == parts.len().
        unsafe { self.parts.get_unchecked_mut(b) }.push(rec);
    }

    pub fn len(&self) -> usize {
        self.parts.iter().map(|p| p.len()).sum()
    }
}

/// One parsed copy of an event. `user == NULL` marks a copy that fails validation (step 3) or
/// is late (step 4); such a copy still takes part in dedup, so it can shadow later copies.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct EventRec {
    pub eid: Id,
    /// ingest_ts in ms; `i64::MAX` if it does not parse (rules.md says it always does).
    pub ingest: i64,
    pub user: Id,
    pub ts: i64,
    /// price_cents for a valid purchase, 0 for a view or a cart.
    pub price: u32,
    /// Category index of a purchase's product.
    pub cat: u32,
}

/// One valid click, already assigned to its person.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ClickRec {
    pub person: Id,
    pub ts: i64,
    pub cid: Id,
    pub campaign: u32,
}

pub struct EventCtx<'a> {
    pub ids: &'a Interner,
    pub products: &'a Products,
    pub window_start: i64,
    pub window_end: i64,
}

/// Events steps 1 (parse) and the per-copy parts of 3 (validate) and 4 (late), recorded as
/// one flag per copy. Dedup (step 2) happens later, over all copies.
pub fn events(data: &[u8], ctx: &EventCtx, out: &mut Scatter<EventRec>) {
    for_each_line(data, |line| {
        let Some([eid, user, ts_text, ingest_text, kind, product, price_text]) = split::<7>(line) else {
            return;
        };
        if eid.is_empty() {
            return;
        }
        let eid = ctx.ids.id(eid);
        let ingest = ts::parse(ingest_text);
        let mut rec = EventRec { eid, ingest: ingest.unwrap_or(i64::MAX), user: NULL, ts: 0, price: 0, cat: 0 };
        'valid: {
            if user.is_empty() || product.is_empty() {
                break 'valid;
            }
            let Some(t) = ts::parse(ts_text) else { break 'valid };
            if t < ctx.window_start || t >= ctx.window_end {
                break 'valid;
            }
            let Some(ingest) = ingest else { break 'valid };
            if ingest.saturating_sub(t) > DAY {
                break 'valid;
            }
            match kind {
                b"view" | b"cart" => {}
                b"purchase" => {
                    let Some(p) = ts::price(price_text) else { break 'valid };
                    rec.price = p;
                    rec.cat = ctx.products.category(product);
                }
                _ => break 'valid,
            }
            rec.ts = t;
            rec.user = ctx.ids.id(user);
        }
        out.push(eid, rec);
    });
}

pub struct ClickCtx<'a> {
    pub ids: &'a Interner,
    pub campaigns: &'a Campaigns,
    pub persons: &'a Persons,
    pub window_start: i64,
    pub window_end: i64,
}

/// Clicks steps 1 (parse) and 3 (validate). Step 2 (dedup) is not needed: copies of a click
/// have identical values (rules.md), so a copy can neither change which click credits a session
/// nor add a campaign to its assists. See STAGES.md.
pub fn clicks(data: &[u8], ctx: &ClickCtx, out: &mut Scatter<ClickRec>) {
    let earliest = ctx.window_start.saturating_sub(7 * DAY);
    for_each_line(data, |line| {
        let Some([cid, user, campaign, ts_text]) = split::<4>(line) else {
            return;
        };
        if cid.is_empty() || user.is_empty() {
            return;
        }
        let Some(t) = ts::parse(ts_text) else { return };
        if t < earliest || t >= ctx.window_end {
            return;
        }
        let Some(campaign) = ctx.campaigns.get(campaign) else { return };
        let person = ctx.persons.person(ctx.ids.id(user));
        out.push(person, ClickRec { person, ts: t, cid: ctx.ids.id(cid), campaign });
    });
}

/// Links step 1: lines with two non-empty IDs.
pub fn links(data: &[u8], ids: &Interner, out: &mut Vec<(Id, Id)>) {
    for_each_line(data, |line| {
        let Some([a, b]) = split::<2>(line) else { return };
        if a.is_empty() || b.is_empty() {
            return;
        }
        out.push((ids.id(a), ids.id(b)));
    });
}
