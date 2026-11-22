//! The v3 sink, one per parser worker: user tokens for every batch (probes batched), event
//! records into move-1 partitions by a mix of the event key, purchases into side streams, clicks
//! and link edges appended flat, and every rare line (locator) read exactly and turned into the
//! same records or into a "special" record that carries its ingest in milliseconds.
//!
//! Record shapes (all u64 words):
//! - SHORT event [w0, w1]: w0 = key | ingest << 40 | surv << 63; w1 = token | ts << 32 | purch << 63
//! - LONG event [a, b, w2, w3]: w2 = ingest | surv << 63; w3 as w1 above
//! - SHORT purchase [w0, buy]: w0 = key | ingest << 40; buy = category | price << 8 (price below
//!   2^24: a larger one reaching the sink sets `bad`, a fallback, never a wrap)
//! - LONG purchase [a, b, ingest, buy]
//! - click [w0, w1]: w0 = token | ts << 32 (ts since window_start - 7 days); w1 = locator (48 bits:
//!   file 12, offset 36) | campaign << 48
//! - edge: token_a | token_b << 32

use parselab::reference::{click_line, event_line};
use parselab::{ClickBatch, Ctx, Dataset, EvBatch, LinkBatch, Locator, Sink, Stream};

use super::ids::{Key, classify};
use super::tokens::{NO_TOKEN, Pending, Tokens};
use crate::fast::sink::Bins;

/// A copy that cannot be a plain record: its ingest does not fit 23 bits, or its event_id is
/// FOREIGN (then `fid` holds the bytes). Compared by milliseconds.
#[derive(Clone, Copy, Debug)]
pub struct Special<'a> {
    pub key: Key,
    pub fid: &'a [u8],
    pub ingest_ms: i64,
    pub surv: bool,
    pub token: u32,
    pub ts: u32,
    pub buy: u32,
}

#[inline(always)]
pub fn mix(x: u64) -> u64 {
    let mut z = x ^ (x >> 33);
    z = z.wrapping_mul(0xff51_afd7_ed55_8ccd);
    z ^= z >> 33;
    z = z.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    z ^ (z >> 33)
}

/// Move-1 partition of an event key.
#[inline(always)]
pub fn part_of(a: u64, b: u64, bits: u32) -> usize {
    (mix(a ^ b.rotate_left(17).wrapping_mul(0x9e37_79b9_7f4a_7c15)) >> (64 - bits)) as usize
}

/// Lookups are resolved this many records after they start, so the prefetch has time to land.
const RING: usize = 64;

/// Event lookups prefetch shared: they are mostly hits, and a line taken for writing is taken
/// away from every other core that reads it. Links and clicks bring most new IDs: for writing.
const EVENT_PREFETCH_W: bool = false;

/// A started lookup and the word its token goes into (the record is already in its bin: written
/// at once, its line is still cached when the token is OR-ed in; measured better than holding
/// whole records here and writing them complete through staged non-temporal lines).
#[derive(Clone, Copy)]
struct Pend {
    user: Pending,
    at: *mut u64,
}

impl Default for Pend {
    fn default() -> Self {
        Pend { user: Pending::default(), at: std::ptr::null_mut() }
    }
}

pub struct V3Sink<'a, 'b> {
    pub tok: &'b Tokens<'a>,
    pub ds: &'a Dataset,
    pub ctx: &'b Ctx,
    pub bits: u32,
    pub short: Bins<[u64; 2]>,
    pub long: Bins<[u64; 4]>,
    pub pshort: Bins<[u64; 2]>,
    pub plong: Bins<[u64; 4]>,
    pub clicks: Bins<[u64; 2]>,
    pub edges: Bins<u64>,
    pub special: Vec<Special<'a>>,
    pub bad: bool,
    /// Measurement switch ENGINE_SINK (feature `measure` only; always 0 otherwise): 1 drops event
    /// batches, 2 skips the token lookups of events, 3 skips their bin writes.
    mode: u8,
    mbits: u32,
    /// Heap word for mode 3's tokens (stays put when the sink moves).
    scratch: Box<u64>,
    ring: Box<[Pend; RING]>,
    ring_n: usize,
    ring_at: usize,
}

// SAFETY: the ring's pointers point into this sink's own bins (moved with it).
unsafe impl Send for V3Sink<'_, '_> {}
// SAFETY: shared references are only used after `finish` (the ring is drained and never read),
// to read finished chunks.
unsafe impl Sync for V3Sink<'_, '_> {}

impl<'a, 'b> V3Sink<'a, 'b> {
    pub fn new(tok: &'b Tokens<'a>, ds: &'a Dataset, ctx: &'b Ctx, bits: u32, events_hint: usize) -> Self {
        let parts = 1usize << bits;
        V3Sink {
            tok,
            ds,
            ctx,
            bits,
            // Chunks about what one sink writes to a partition: events / parts / threads.
            short: Bins::with_chunk(parts, (events_hint / 8).clamp(1 << 14, 1 << 21), (events_hint / parts / 2).clamp(32, 512)),
            long: Bins::with_chunk(parts, (events_hint / 32).clamp(1 << 12, 1 << 19), (events_hint / parts / 16).clamp(16, 256)),
            pshort: Bins::with_chunk(parts, 1 << 13, (events_hint / parts / 64).clamp(8, 64)),
            plong: Bins::with_chunk(parts, 1 << 12, (events_hint / parts / 256).clamp(4, 32)),
            clicks: Bins::with_chunk(1, (events_hint / 4).clamp(1 << 14, 1 << 20), 1 << 14),
            edges: Bins::with_chunk(1, (events_hint / 20).clamp(1 << 12, 1 << 19), 1 << 13),
            special: Vec::new(),
            bad: false,
            mode: if cfg!(feature = "measure") { std::env::var("ENGINE_SINK").ok().and_then(|v| v.parse().ok()).unwrap_or(0) } else { 0 },
            scratch: Box::new(0),
            mbits: std::env::var("ENGINE_SINK_BITS").ok().and_then(|v| v.parse().ok()).unwrap_or(bits).min(bits),
            ring: Box::new([Pend::default(); RING]),
            ring_n: 0,
            ring_at: 0,
        }
    }

    /// Starts a lookup whose token is to be OR-ed into `*at` (bits 0-31) later (prefetched for
    /// writing when `W`).
    #[inline(always)]
    fn defer<const W: bool>(&mut self, user: Pending, at: *mut u64) {
        let p = Pend { user: self.tok.start_hint::<W>(user.long, user.a, user.b), at };
        if self.ring_n == RING {
            self.resolve(self.ring_at);
            self.ring[self.ring_at] = p;
            self.ring_at = (self.ring_at + 1) % RING;
        } else {
            self.ring[(self.ring_at + self.ring_n) % RING] = p;
            self.ring_n += 1;
        }
    }

    #[inline(always)]
    fn resolve(&mut self, i: usize) {
        let p = self.ring[i];
        let t = self.tok.finish(&p.user);
        if t == NO_TOKEN {
            self.bad = true;
        }
        // SAFETY: `at` points at a word of a record in this sink's bins (or at `scratch`); chunks
        // never move.
        unsafe { *p.at |= u64::from(t) };
    }

    /// Resolves every pending lookup. Call after the last delivery of a phase.
    pub fn drain(&mut self) {
        for k in 0..self.ring_n {
            self.resolve((self.ring_at + k) % RING);
        }
        self.ring_n = 0;
        self.ring_at = 0;
    }

    pub fn finish(&mut self) {
        self.drain();
        self.short.finish();
        self.long.finish();
        self.pshort.finish();
        self.plong.finish();
        self.clicks.finish();
        self.edges.finish();
    }

    /// ENGINE_SINK=1: nothing; 2: records without token lookups; 3: lookups without records
    /// (tokens OR-ed into a scratch word); 4: only the per-lane field work. ENGINE_SINK_BITS lowers the partition bits of mode 2.
    #[cold]
    #[inline(never)]
    fn events_measure(&mut self, b: &EvBatch) {
        let n = b.n as usize;
        if self.mode == 1 {
            std::hint::black_box(b);
            return;
        }
        let bits = self.mbits;
        let scratch: *mut u64 = &mut *self.scratch;
        for k in 0..n {
            let surv = b.surv >> k & 1 != 0;
            let ing = u64::from(b.ingest[k]);
            let w1 = if surv { u64::from(b.ts[k]) << 32 } else { 0 };
            if self.mode == 4 {
                // The per-lane work both of the others share: fields, partition, lookup slot.
                let a = b.ev_a[k];
                let bb = if b.ev_long >> k & 1 == 0 { 0 } else { b.ev_b[k] };
                std::hint::black_box((part_of(a, bb, bits), ing, w1, b.user_a[k] ^ b.user_b[k].rotate_left(29)));
                continue;
            }
            if self.mode == 3 {
                if surv {
                    let user = Pending { long: b.user_long >> k & 1 != 0, a: b.user_a[k], b: b.user_b[k], slot: 0 };
                    self.defer::<EVENT_PREFETCH_W>(user, scratch);
                }
                continue;
            }
            if b.ev_long >> k & 1 == 0 {
                let a = b.ev_a[k];
                self.short.push(part_of(a, 0, bits), [a | (ing << 40) | (u64::from(surv) << 63), w1]);
            } else {
                let (a, bb) = (b.ev_a[k], b.ev_b[k]);
                self.long.push(part_of(a, bb, bits), [a, bb, ing | (u64::from(surv) << 63), w1]);
            }
        }
    }

    fn user_token(&self, id: &'a [u8]) -> u32 {
        self.tok.token(classify(id, b'u'), id)
    }
}

impl<'a> Sink for V3Sink<'a, '_> {
    #[inline]
    fn events(&mut self, b: &EvBatch) {
        let n = b.n as usize;
        if cfg!(feature = "measure") && self.mode != 0 {
            return self.events_measure(b);
        }
        for k in 0..n {
            let surv = b.surv >> k & 1 != 0;
            let purch = surv && b.purch >> k & 1 != 0;
            let ing = u64::from(b.ingest[k]);
            let ts = if surv { u64::from(b.ts[k]) } else { 0 };
            let w1 = (ts << 32) | (u64::from(purch) << 63);
            let buy = if purch {
                // The parser accepts 1 to 10,000,000; anything at 2^24 or above would not fit.
                self.bad |= b.price[k] >> 24 != 0;
                u64::from(b.cat[k]) | (u64::from(b.price[k]) << 8)
            } else {
                0
            };
            let at: *mut u64 = if b.ev_long >> k & 1 == 0 {
                let a = b.ev_a[k];
                let p = part_of(a, 0, self.bits);
                if purch {
                    self.pshort.push(p, [a | (ing << 40), buy]);
                }
                let r = self.short.push_at(p, [a | (ing << 40) | (u64::from(surv) << 63), w1]);
                // SAFETY: r points at the record just written; word 1 holds the token.
                unsafe { (r as *mut u64).add(1) }
            } else {
                let (a, bb) = (b.ev_a[k], b.ev_b[k]);
                let p = part_of(a, bb, self.bits);
                if purch {
                    self.plong.push(p, [a, bb, ing, buy]);
                }
                let r = self.long.push_at(p, [a, bb, ing | (u64::from(surv) << 63), w1]);
                // SAFETY: as above; word 3 holds the token.
                unsafe { (r as *mut u64).add(3) }
            };
            if surv {
                // A dead copy only shadows: its key, ingest and dead bit; no token needed.
                let user = Pending { long: b.user_long >> k & 1 != 0, a: b.user_a[k], b: b.user_b[k], slot: 0 };
                self.defer::<EVENT_PREFETCH_W>(user, at);
            }
        }
    }

    #[inline]
    fn clicks(&mut self, b: &ClickBatch) {
        let n = b.n as usize;
        for k in 0..n {
            if b.loc[k] >> 48 != 0 {
                self.bad = true;
            }
            let r = self.clicks.push_at(0, [u64::from(b.ts[k]) << 32, b.loc[k] | (u64::from(b.campaign[k]) << 48)]);
            self.defer::<true>(Pending { long: b.user_long >> k & 1 != 0, a: b.user_a[k], b: b.user_b[k], slot: 0 }, r as *mut u64);
        }
    }

    #[inline]
    fn links(&mut self, b: &LinkBatch) {
        let n = b.n as usize;
        for k in 0..n {
            let r = self.edges.push_at(0, 0);
            self.defer::<true>(Pending { long: b.a_long >> k & 1 != 0, a: b.a_a[k], b: b.a_b[k], slot: 0 }, r);
            // The second side's token goes into the high half: resolve it through a shifted word.
            let r2 = self.edges.push_at(0, 0);
            self.defer::<true>(Pending { long: b.b_long >> k & 1 != 0, a: b.b_a[k], b: b.b_b[k], slot: 0 }, r2);
        }
    }

    fn foreign_event(&mut self, at: Locator) {
        let line = self.ds.line(Stream::Events, at);
        let mut fields: Vec<&'a [u8]> = Vec::with_capacity(8);
        let Some(e) = event_line(self.ctx, line, &mut fields) else {
            self.bad = true;
            return;
        };
        let key = classify(e.event_id, b'e');
        let ws = self.ctx.window_start_ms;
        let token = if e.survives { self.user_token(e.user_id) } else { NO_TOKEN };
        let buy = if e.survives && e.purchase {
            let price = e.price.unwrap_or(0);
            self.bad |= price >> 24 != 0;
            u32::from(self.ctx.products.code(e.product_id)) | (price << 8)
        } else {
            0
        };
        self.special.push(Special {
            key,
            fid: e.event_id,
            ingest_ms: e.ingest.unwrap_or(i64::MAX),
            surv: e.survives,
            token,
            ts: if e.survives { (e.ts.unwrap() - ws) as u32 } else { 0 },
            buy,
        });
    }

    fn foreign_click(&mut self, at: Locator) {
        let line = self.ds.line(Stream::Clicks, at);
        let mut fields = Vec::with_capacity(8);
        let Some(c) = click_line(self.ctx, line, &mut fields) else {
            self.bad = true;
            return;
        };
        let token = self.user_token(c.user_id);
        let ts = (c.ts - self.ctx.click_base_ms) as u64;
        if at.0 >> 48 != 0 {
            self.bad = true;
        }
        self.clicks.push(0, [u64::from(token) | (ts << 32), at.0 | (u64::from(c.campaign) << 48)]);
    }

    fn foreign_link(&mut self, at: Locator) {
        let line = self.ds.line(Stream::Links, at);
        let mut fields = Vec::with_capacity(4);
        if !parselab::csv::split(line, &mut fields) || fields.len() != 2 || fields[0].is_empty() || fields[1].is_empty() {
            self.bad = true;
            return;
        }
        let (x, y) = (self.user_token(fields[0]), self.user_token(fields[1]));
        self.edges.push(0, u64::from(x));
        self.edges.push(0, u64::from(y));
    }
}
