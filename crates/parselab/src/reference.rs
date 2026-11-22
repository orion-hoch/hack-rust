//! The scalar reference parser: records straight from rules.md and RECORDS.md, one line at a time,
//! no cleverness. The fast parser is tested against this.
//!
//! Each function takes the whole file and a byte range; it handles every line whose first byte is in
//! `from..to` (the header line at offset 0 is never a data line). Whole file: `from = 0, to = len`.

use crate::csv;
use crate::ids::key_of;
use crate::records::*;
use crate::tables::Ctx;
use crate::ts::{parse_f1, parse_price, parse_ts};

/// The exact field values of one line (quotes removed), or None when the line is malformed:
/// what the engine uses to read a locator (for example to intern a FOREIGN ID).
pub fn line_fields<'a>(line: &'a [u8], out: &mut Vec<&'a [u8]>) -> bool {
    csv::split(line, out)
}

/// What one well-formed event line means (used by the reference and by checks).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventLine<'a> {
    pub event_id: &'a [u8],
    pub user_id: &'a [u8],
    /// ts in ms if valid and inside the event window.
    pub ts: Option<i64>,
    /// ingest_ts in ms if valid F1.
    pub ingest: Option<i64>,
    pub purchase: bool,
    /// event_type is exactly view, cart or purchase.
    pub kind_ok: bool,
    /// Passes rule 3.
    pub valid: bool,
    /// Passes rule 3 and rule 4 (needs a valid ingest).
    pub survives: bool,
    pub price: Option<u32>,
    pub product_id: &'a [u8],
}

/// Field values of a well-formed event line with a non-empty event_id, or None (nothing emitted).
pub fn event_line<'a>(ctx: &Ctx, line: &'a [u8], fields: &mut Vec<&'a [u8]>) -> Option<EventLine<'a>> {
    if !csv::split(line, fields) || fields.len() != 7 || fields[0].is_empty() {
        return None;
    }
    let f = &fields[..];
    let ts = parse_ts(f[2]).filter(|&t| ctx.window_start_ms <= t && t < ctx.window_end_ms);
    let ingest = parse_f1(f[3]);
    let kind = f[4];
    let purchase = kind == b"purchase";
    let kind_ok = purchase || kind == b"view" || kind == b"cart";
    let price = if purchase { parse_price(f[6]) } else { None };
    let valid = !f[1].is_empty()
        && ts.is_some()
        && kind_ok
        && !f[5].is_empty()
        && (!purchase || price.is_some());
    let late = match (ts, ingest) {
        (Some(t), Some(i)) => i - t > DAY_MS,
        _ => true,
    };
    Some(EventLine {
        event_id: f[0],
        user_id: f[1],
        ts,
        ingest,
        purchase,
        kind_ok,
        valid,
        survives: valid && !late,
        price,
        product_id: f[5],
    })
}

/// Ingest as whole seconds since window_start - 30 days, if it fits below 2^23 - 1.
pub fn ingest_field(ctx: &Ctx, ingest_ms: Option<i64>) -> Option<u32> {
    let s = ingest_ms?.div_euclid(1000) - ctx.ingest_base_s;
    (0..INGEST_LIMIT as i64).contains(&s).then_some(s as u32)
}

/// One event line (without its `\n`) that starts at `start` in file `file`.
#[inline(never)]
pub fn event_one<'a, S: Sink>(
    ctx: &Ctx,
    file: u32,
    start: usize,
    line: &'a [u8],
    fields: &mut Vec<&'a [u8]>,
    sink: &mut S,
) {
    let Some(e) = event_line(ctx, line, fields) else { return };
    let at = Locator::new(file, start as u64);
    let Some(ev) = key_of(e.event_id, b'e') else { return sink.foreign_event(at) };
    let Some(ingest) = ingest_field(ctx, e.ingest) else { return sink.foreign_event(at) };
    let mut b = EvBatch::default();
    if !e.survives {
        b.push(ev, Key::default(), false, false, 0, ingest, 0, 0);
        return sink.events(&b);
    }
    let Some(user) = key_of(e.user_id, b'u') else { return sink.foreign_event(at) };
    let ts_rel = (e.ts.unwrap() - ctx.window_start_ms) as u32;
    let (cat, price) = if e.purchase { (ctx.products.code(e.product_id), e.price.unwrap()) } else { (0, 0) };
    b.push(ev, user, true, e.purchase, ts_rel, ingest, cat, price);
    sink.events(&b);
}

pub fn events<S: Sink>(ctx: &Ctx, file: u32, data: &[u8], from: usize, to: usize, sink: &mut S) {
    let mut fields = Vec::with_capacity(8);
    csv::for_each_line(data, from, to, |start, line| event_one(ctx, file, start, line, &mut fields, sink));
}

/// A click's meaning if the line is well-formed, has a click_id and passes rule 3.
pub struct ClickLine<'a> {
    pub click_id: &'a [u8],
    pub user_id: &'a [u8],
    pub campaign: u16,
    pub ts: i64,
}

pub fn click_line<'a>(ctx: &Ctx, line: &'a [u8], fields: &mut Vec<&'a [u8]>) -> Option<ClickLine<'a>> {
    if !csv::split(line, fields) || fields.len() != 4 || fields[0].is_empty() {
        return None;
    }
    let f = &fields[..];
    if f[1].is_empty() {
        return None;
    }
    let ts = parse_ts(f[3]).filter(|&t| ctx.click_base_ms <= t && t < ctx.window_end_ms)?;
    let campaign = ctx.campaigns.number(f[2])?;
    Some(ClickLine { click_id: f[0], user_id: f[1], campaign, ts })
}

#[inline(never)]
pub fn click_one<'a, S: Sink>(
    ctx: &Ctx,
    file: u32,
    start: usize,
    line: &'a [u8],
    fields: &mut Vec<&'a [u8]>,
    sink: &mut S,
) {
    let Some(c) = click_line(ctx, line, fields) else { return };
    let at = Locator::new(file, start as u64);
    match key_of(c.user_id, b'u') {
        Some(user) => {
            let mut b = ClickBatch::default();
            b.push(user, (c.ts - ctx.click_base_ms) as u32, c.campaign, at);
            sink.clicks(&b);
        }
        None => sink.foreign_click(at),
    }
}

pub fn clicks<S: Sink>(ctx: &Ctx, file: u32, data: &[u8], from: usize, to: usize, sink: &mut S) {
    let mut fields = Vec::with_capacity(8);
    csv::for_each_line(data, from, to, |start, line| click_one(ctx, file, start, line, &mut fields, sink));
}

#[inline(never)]
pub fn link_one<'a, S: Sink>(
    _ctx: &Ctx,
    file: u32,
    start: usize,
    line: &'a [u8],
    fields: &mut Vec<&'a [u8]>,
    sink: &mut S,
) {
    if !csv::split(line, fields) || fields.len() != 2 {
        return;
    }
    if fields[0].is_empty() || fields[1].is_empty() {
        return;
    }
    match (key_of(fields[0], b'u'), key_of(fields[1], b'u')) {
        (Some(a), Some(b)) => {
            let mut lb = LinkBatch::default();
            lb.push(a, b);
            sink.links(&lb);
        }
        _ => sink.foreign_link(Locator::new(file, start as u64)),
    }
}

pub fn links<S: Sink>(ctx: &Ctx, file: u32, data: &[u8], from: usize, to: usize, sink: &mut S) {
    let mut fields = Vec::with_capacity(4);
    csv::for_each_line(data, from, to, |start, line| link_one(ctx, file, start, line, &mut fields, sink));
}

/// The line that starts at `offset` (without its `\n`): how stage 2 re-reads a locator.
pub fn line_at(data: &[u8], offset: u64) -> &[u8] {
    let start = offset as usize;
    let end = data[start..].iter().position(|&c| c == b'\n').map_or(data.len(), |p| start + p);
    &data[start..end]
}
