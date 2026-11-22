//! The shapes of RECORDS.md version 3: opaque ID keys, structure-of-arrays batches of up to 8
//! lines, and locators for everything the parser does not key.

pub const DAY_MS: i64 = 86_400_000;
/// ingest is whole seconds since window_start - 30 days.
pub const INGEST_BASE_BEFORE_WINDOW_MS: i64 = 30 * DAY_MS;
/// An ingest field fits when it is below this (2^23 - 1).
pub const INGEST_LIMIT: u64 = (1 << 23) - 1;
/// click ts is ms since window_start - 7 days.
pub const CLICK_BASE_BEFORE_WINDOW_MS: i64 = 7 * DAY_MS;

/// Where a line starts: file number (index in the name-sorted list of `*.csv` files of its folder)
/// and byte offset of the line's first byte in that file.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(transparent)]
pub struct Locator(pub u64);

impl Locator {
    pub const FILE_BITS: u32 = 12;
    pub const OFFSET_BITS: u32 = 40;
    pub const MAX_FILES: usize = 1 << Self::FILE_BITS;
    pub const MAX_OFFSET: u64 = 1 << Self::OFFSET_BITS;

    #[inline(always)]
    pub fn new(file: u32, offset: u64) -> Self {
        debug_assert!((file as usize) < Self::MAX_FILES && offset < Self::MAX_OFFSET);
        Locator(file as u64 | (offset << Self::FILE_BITS))
    }
    #[inline(always)]
    pub fn file(self) -> u32 {
        (self.0 & ((1 << Self::FILE_BITS) - 1)) as u32
    }
    #[inline(always)]
    pub fn offset(self) -> u64 {
        (self.0 >> Self::FILE_BITS) & (Self::MAX_OFFSET - 1)
    }
}

/// The key of a SHORT or LONG ID (RECORDS.md "IDs are opaque keys"): SHORT has `a` = base-36
/// value and `b` = 0; LONG has `a`, `b` = the first and last 16 hex digits.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Key {
    pub long: bool,
    pub a: u64,
    pub b: u64,
}

/// Up to 8 event lines, structure of arrays. Lanes 0..n are used.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct EvBatch {
    // Layout: the 64-byte arrays first, so each is exactly one cache line of the 64-aligned
    // struct. With the 8 header bytes in front, every one of them straddled two lines (a split
    // store in the decoder and a split load in the sink, per array per batch).
    pub ev_a: [u64; 8],
    pub ev_b: [u64; 8],
    pub user_a: [u64; 8],
    pub user_b: [u64; 8],
    /// ms since window_start, < 2^31 (meaningful when surv is set)
    pub ts: [u32; 8],
    /// whole seconds since window_start - 30 days, < 2^23 - 1
    pub ingest: [u32; 8],
    /// price_cents 1..=10,000,000 (only with purch)
    pub price: [u32; 8],
    /// category code (only with purch)
    pub cat: [u8; 8],
    pub n: u32,
    /// bit i: event_id of lane i is LONG (else SHORT)
    pub ev_long: u8,
    /// bit i: user_id of lane i is LONG (else SHORT); meaningful when the surv bit is set
    pub user_long: u8,
    /// bit i: this copy passes rule 3 (validate) and rule 4 (not late)
    pub surv: u8,
    /// bit i: event_type is purchase (only with surv)
    pub purch: u8,
}

impl Default for EvBatch {
    fn default() -> Self {
        EvBatch {
            n: 0,
            ev_long: 0,
            user_long: 0,
            surv: 0,
            purch: 0,
            ev_a: [0; 8],
            ev_b: [0; 8],
            user_a: [0; 8],
            user_b: [0; 8],
            ts: [0; 8],
            ingest: [0; 8],
            cat: [0; 8],
            price: [0; 8],
        }
    }
}

impl EvBatch {
    /// Appends one lane (n < 8). `user` and `ts` matter when `surv`; `cat`, `price` when `purch`.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn push(&mut self, ev: Key, user: Key, surv: bool, purch: bool, ts: u32, ingest: u32, cat: u8, price: u32) {
        let i = self.n as usize;
        debug_assert!(i < 8);
        let bit = 1u8 << i;
        self.ev_long |= (ev.long as u8) * bit;
        self.user_long |= (user.long as u8) * bit;
        self.surv |= (surv as u8) * bit;
        self.purch |= ((purch & surv) as u8) * bit;
        self.ev_a[i] = ev.a;
        self.ev_b[i] = ev.b;
        self.user_a[i] = user.a;
        self.user_b[i] = user.b;
        self.ts[i] = ts;
        self.ingest[i] = ingest;
        self.cat[i] = cat;
        self.price[i] = price;
        self.n += 1;
    }

    /// Lane i as a row with the fields that carry no meaning set to zero (for comparisons).
    pub fn row(&self, i: usize) -> EvRow {
        let bit = 1u8 << i;
        let surv = self.surv & bit != 0;
        let purch = surv && self.purch & bit != 0;
        EvRow {
            ev: Key { long: self.ev_long & bit != 0, a: self.ev_a[i], b: self.ev_b[i] },
            user: if surv {
                Key { long: self.user_long & bit != 0, a: self.user_a[i], b: self.user_b[i] }
            } else {
                Key::default()
            },
            surv,
            purch,
            ts: if surv { self.ts[i] } else { 0 },
            ingest: self.ingest[i],
            cat: if purch { self.cat[i] } else { 0 },
            price: if purch { self.price[i] } else { 0 },
        }
    }
}

/// Up to 8 valid clicks with SHORT or LONG user_id.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct ClickBatch {
    // The 64-byte arrays first: one cache line each (see `EvBatch`).
    pub user_a: [u64; 8],
    pub user_b: [u64; 8],
    /// locator of the line (its click_id bytes are read only to break a ts tie)
    pub loc: [u64; 8],
    /// ms since window_start - 7 days
    pub ts: [u32; 8],
    /// row number in campaigns.csv
    pub campaign: [u16; 8],
    pub n: u32,
    pub user_long: u8,
}

impl Default for ClickBatch {
    fn default() -> Self {
        ClickBatch { n: 0, user_long: 0, user_a: [0; 8], user_b: [0; 8], ts: [0; 8], campaign: [0; 8], loc: [0; 8] }
    }
}

impl ClickBatch {
    #[inline(always)]
    pub fn push(&mut self, user: Key, ts: u32, campaign: u16, loc: Locator) {
        let i = self.n as usize;
        debug_assert!(i < 8);
        self.user_long |= (user.long as u8) << i;
        self.user_a[i] = user.a;
        self.user_b[i] = user.b;
        self.ts[i] = ts;
        self.campaign[i] = campaign;
        self.loc[i] = loc.0;
        self.n += 1;
    }
    pub fn row(&self, i: usize) -> ClickRow {
        ClickRow {
            user: Key { long: self.user_long & (1 << i) != 0, a: self.user_a[i], b: self.user_b[i] },
            ts: self.ts[i],
            campaign: self.campaign[i],
            loc: Locator(self.loc[i]),
        }
    }
}

/// Up to 8 well-formed link lines, both sides SHORT or LONG.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct LinkBatch {
    pub n: u32,
    pub a_long: u8,
    pub b_long: u8,
    pub a_a: [u64; 8],
    pub a_b: [u64; 8],
    pub b_a: [u64; 8],
    pub b_b: [u64; 8],
}

impl Default for LinkBatch {
    fn default() -> Self {
        LinkBatch { n: 0, a_long: 0, b_long: 0, a_a: [0; 8], a_b: [0; 8], b_a: [0; 8], b_b: [0; 8] }
    }
}

impl LinkBatch {
    #[inline(always)]
    pub fn push(&mut self, a: Key, b: Key) {
        let i = self.n as usize;
        debug_assert!(i < 8);
        self.a_long |= (a.long as u8) << i;
        self.b_long |= (b.long as u8) << i;
        self.a_a[i] = a.a;
        self.a_b[i] = a.b;
        self.b_a[i] = b.a;
        self.b_b[i] = b.b;
        self.n += 1;
    }
    pub fn row(&self, i: usize) -> LinkRow {
        LinkRow {
            a: Key { long: self.a_long & (1 << i) != 0, a: self.a_a[i], b: self.a_b[i] },
            b: Key { long: self.b_long & (1 << i) != 0, a: self.b_a[i], b: self.b_b[i] },
        }
    }
}

/// One event lane with meaningless fields zeroed.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct EvRow {
    pub ev: Key,
    pub user: Key,
    pub surv: bool,
    pub purch: bool,
    pub ts: u32,
    pub ingest: u32,
    pub cat: u8,
    pub price: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct ClickRow {
    pub user: Key,
    pub ts: u32,
    pub campaign: u16,
    pub loc: Locator,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct LinkRow {
    pub a: Key,
    pub b: Key,
}

/// Where the parser delivers. One sink per worker; calls are inlined through generics. Every
/// well-formed line with a non-empty own ID is delivered exactly once, in a batch or by locator.
/// Batches may hold 1 to 8 lanes; delivery order is arbitrary.
pub trait Sink {
    fn events(&mut self, b: &EvBatch);
    /// Well-formed event line with a non-empty event_id that is not in a batch: FOREIGN
    /// event_id, a surviving copy with a FOREIGN user_id, or an ingest that does not fit.
    fn foreign_event(&mut self, at: Locator);
    fn clicks(&mut self, b: &ClickBatch);
    /// Valid click whose user_id is FOREIGN.
    fn foreign_click(&mut self, at: Locator);
    fn links(&mut self, b: &LinkBatch);
    /// Well-formed link line, both sides non-empty, at least one FOREIGN.
    fn foreign_link(&mut self, at: Locator);
}

/// Collects everything as rows (tests, the reference, small runs).
#[derive(Default, Debug, Clone)]
pub struct VecSink {
    pub events: Vec<EvRow>,
    pub clicks: Vec<ClickRow>,
    pub links: Vec<LinkRow>,
    pub foreign_events: Vec<Locator>,
    pub foreign_clicks: Vec<Locator>,
    pub foreign_links: Vec<Locator>,
    /// Batches received (how full they are is lanes / batches).
    pub batches: u64,
}

impl Sink for VecSink {
    fn events(&mut self, b: &EvBatch) {
        self.batches += 1;
        for i in 0..b.n as usize {
            self.events.push(b.row(i));
        }
    }
    fn foreign_event(&mut self, at: Locator) {
        self.foreign_events.push(at)
    }
    fn clicks(&mut self, b: &ClickBatch) {
        self.batches += 1;
        for i in 0..b.n as usize {
            self.clicks.push(b.row(i));
        }
    }
    fn foreign_click(&mut self, at: Locator) {
        self.foreign_clicks.push(at)
    }
    fn links(&mut self, b: &LinkBatch) {
        self.batches += 1;
        for i in 0..b.n as usize {
            self.links.push(b.row(i));
        }
    }
    fn foreign_link(&mut self, at: Locator) {
        self.foreign_links.push(at)
    }
}

impl VecSink {
    pub fn append(&mut self, other: VecSink) {
        self.events.extend(other.events);
        self.clicks.extend(other.clicks);
        self.links.extend(other.links);
        self.foreign_events.extend(other.foreign_events);
        self.foreign_clicks.extend(other.foreign_clicks);
        self.foreign_links.extend(other.foreign_links);
        self.batches += other.batches;
    }
    /// Sorts every stream, so two sinks can be compared as multisets.
    pub fn canonicalize(&mut self) {
        self.events.sort_unstable();
        self.clicks.sort_unstable();
        self.links.sort_unstable();
        self.foreign_events.sort_unstable();
        self.foreign_clicks.sort_unstable();
        self.foreign_links.sort_unstable();
    }
    pub fn same_records(&self, o: &VecSink) -> bool {
        self.events == o.events
            && self.clicks == o.clicks
            && self.links == o.links
            && self.foreign_events == o.foreign_events
            && self.foreign_clicks == o.foreign_clicks
            && self.foreign_links == o.foreign_links
    }
    pub fn total(&self) -> u64 {
        (self.events.len()
            + self.clicks.len()
            + self.links.len()
            + self.foreign_events.len()
            + self.foreign_clicks.len()
            + self.foreign_links.len()) as u64
    }
}

#[inline(always)]
pub fn mix64(mut x: u64) -> u64 {
    x ^= x >> 31;
    x = x.wrapping_mul(0x7FB5_D329_728E_A185);
    x ^= x >> 27;
    x = x.wrapping_mul(0x81DA_DEF4_BC2D_D44D);
    x ^ (x >> 33)
}

#[inline(always)]
fn key_hash(k: Key) -> u64 {
    mix64(k.a ^ mix64(k.b ^ k.long as u64))
}

/// Counts lanes and folds every meaningful field into an order-independent checksum. It reads
/// each batch the way a real sink must, but stores nothing.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct CountSink {
    pub events: u64,
    pub clicks: u64,
    pub links: u64,
    pub foreign_events: u64,
    pub foreign_clicks: u64,
    pub foreign_links: u64,
    pub survives: u64,
    pub purchases: u64,
    pub batches: u64,
    pub hash: u64,
}

impl Sink for CountSink {
    #[inline(always)]
    fn events(&mut self, b: &EvBatch) {
        self.batches += 1;
        for i in 0..b.n as usize {
            let r = b.row(i);
            self.events += 1;
            self.survives += r.surv as u64;
            self.purchases += r.purch as u64;
            let h = key_hash(r.ev)
                ^ key_hash(r.user).rotate_left(17)
                ^ mix64(r.ts as u64 | ((r.ingest as u64) << 32) | ((r.surv as u64) << 62) | ((r.purch as u64) << 63))
                ^ mix64(r.price as u64 | ((r.cat as u64) << 32) | (1 << 40));
            self.hash = self.hash.wrapping_add(mix64(h));
        }
    }
    #[inline(always)]
    fn foreign_event(&mut self, at: Locator) {
        self.foreign_events += 1;
        self.hash = self.hash.wrapping_add(mix64(at.0 ^ 0x4444));
    }
    #[inline(always)]
    fn clicks(&mut self, b: &ClickBatch) {
        self.batches += 1;
        for i in 0..b.n as usize {
            let r = b.row(i);
            self.clicks += 1;
            let h = key_hash(r.user) ^ mix64(r.ts as u64 | ((r.campaign as u64) << 32) | (2 << 48)) ^ mix64(r.loc.0 ^ 0x2222);
            self.hash = self.hash.wrapping_add(mix64(h));
        }
    }
    #[inline(always)]
    fn foreign_click(&mut self, at: Locator) {
        self.foreign_clicks += 1;
        self.hash = self.hash.wrapping_add(mix64(at.0 ^ 0x5555));
    }
    #[inline(always)]
    fn links(&mut self, b: &LinkBatch) {
        self.batches += 1;
        for i in 0..b.n as usize {
            let r = b.row(i);
            self.links += 1;
            let h = key_hash(r.a) ^ key_hash(r.b).rotate_left(29) ^ 0x3333;
            self.hash = self.hash.wrapping_add(mix64(h));
        }
    }
    #[inline(always)]
    fn foreign_link(&mut self, at: Locator) {
        self.foreign_links += 1;
        self.hash = self.hash.wrapping_add(mix64(at.0 ^ 0x6666));
    }
}

impl CountSink {
    pub fn merge(&mut self, o: &CountSink) {
        self.events += o.events;
        self.clicks += o.clicks;
        self.links += o.links;
        self.foreign_events += o.foreign_events;
        self.foreign_clicks += o.foreign_clicks;
        self.foreign_links += o.foreign_links;
        self.survives += o.survives;
        self.purchases += o.purchases;
        self.batches += o.batches;
        self.hash = self.hash.wrapping_add(o.hash);
    }
    /// Same records (counts and checksum); batch counts may differ between parsers.
    pub fn same_records(&self, o: &CountSink) -> bool {
        CountSink { batches: 0, ..*self } == CountSink { batches: 0, ..*o }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_roundtrip() {
        let at = Locator::new(4095, (1 << 40) - 1);
        assert_eq!((at.file(), at.offset()), (4095, (1 << 40) - 1));
        let mut b = EvBatch::default();
        let e = Key { long: true, a: 1, b: 2 };
        let u = Key { long: false, a: 3, b: 0 };
        b.push(e, u, true, true, 7, 9, 4, 500);
        b.push(e, u, false, true, 7, 9, 4, 500);
        let r0 = b.row(0);
        assert_eq!((r0.ev, r0.user, r0.surv, r0.purch, r0.ts, r0.ingest, r0.cat, r0.price), (e, u, true, true, 7, 9, 4, 500));
        let r1 = b.row(1);
        assert_eq!((r1.user, r1.surv, r1.purch, r1.ts, r1.cat, r1.price), (Key::default(), false, false, 0, 0, 0));
    }
}

/// The cheapest sink that still makes the parser produce every batch: it counts lanes and folds
/// one word per batch (to measure the parser without a sink's per-lane work).
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct TouchSink {
    pub lanes: u64,
    pub foreign: u64,
    pub fold: u64,
}

impl Sink for TouchSink {
    #[inline(always)]
    fn events(&mut self, b: &EvBatch) {
        self.lanes += b.n as u64;
        self.fold ^= b.ev_a[0] ^ b.ts[0] as u64 ^ b.surv as u64;
    }
    #[inline(always)]
    fn foreign_event(&mut self, at: Locator) {
        self.foreign += 1;
        self.fold ^= at.0;
    }
    #[inline(always)]
    fn clicks(&mut self, b: &ClickBatch) {
        self.lanes += b.n as u64;
        self.fold ^= b.user_a[0] ^ b.ts[0] as u64;
    }
    #[inline(always)]
    fn foreign_click(&mut self, at: Locator) {
        self.foreign += 1;
        self.fold ^= at.0;
    }
    #[inline(always)]
    fn links(&mut self, b: &LinkBatch) {
        self.lanes += b.n as u64;
        self.fold ^= b.a_a[0];
    }
    #[inline(always)]
    fn foreign_link(&mut self, at: Locator) {
        self.foreign += 1;
        self.fold ^= at.0;
    }
}
