//! The v3 sink, one per parser worker: user tokens for every batch (probes batched), event
//! records into move-1 partitions by a mix of the event key (purchases as side records in the same
//! bins), link edges appended flat (clicks go to `ClickSink`, after dedup), and every rare line (locator) read exactly and turned into the
//! same records or into a "special" record that carries its ingest in milliseconds.
//!
//! Record shapes (all u64 words):
//! - SHORT event [w0, w1]: w0 = key | ingest << 40 | surv << 63; w1 = token | ts << 32 | purch << 63
//! - LONG event [a, b, w2, w3]: w2 = ingest | surv << 63; w3 as w1 above
//! - SHORT purchase, in the SHORT bins [w0, buy | PURCHASE_SIDE]: w0 = key | ingest << 40 (surv
//!   clear); buy = category | price << 8 (price below 2^24: a larger one reaching the sink sets
//!   `bad`, a fallback, never a wrap)
//! - LONG purchase, in the LONG bins [a, b, ingest, buy | PURCHASE_SIDE]
//! - edge: two consecutive words, token of side a, token of side b

use parselab::reference::{click_line, event_line};
use parselab::{ClickBatch, Ctx, Dataset, EvBatch, LinkBatch, Locator, Sink, Stream};

use super::ids::{Key, classify};
use super::persons::People;
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

/// `part_of` for the 8 lanes of an event batch at once (`long` bit k: lane k's event key is LONG;
/// a SHORT lane's b is taken as 0, as the callers do), into `out` (lanes past b.n hold garbage).
/// With AVX-512 the mixes run in zmm registers: about 2 instructions a lane against about 14.
#[inline(always)]
pub fn parts8(a: &[u64; 8], b: &[u64; 8], long: u8, bits: u32, out: &mut [u32; 8]) {
    #[cfg(target_arch = "x86_64")]
    if bits >= 1 && std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx512dq") {
        // SAFETY: both features were just detected.
        unsafe { parts512(a, b, long, bits, out) };
        return;
    }
    for k in 0..8 {
        out[k] = part_of(a[k], if long >> k & 1 != 0 { b[k] } else { 0 }, bits) as u32;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn parts512(a: &[u64; 8], b: &[u64; 8], long: u8, bits: u32, out: &mut [u32; 8]) {
    use core::arch::x86_64::*;
    unsafe {
        let va = _mm512_loadu_si512(a.as_ptr() as *const _);
        let vb = _mm512_maskz_loadu_epi64(long, b.as_ptr() as *const i64);
        let x = _mm512_xor_si512(va, _mm512_mullo_epi64(_mm512_rol_epi64::<17>(vb), _mm512_set1_epi64(0x9e37_79b9_7f4a_7c15u64 as i64)));
        let mut z = _mm512_xor_si512(x, _mm512_srli_epi64::<33>(x));
        z = _mm512_mullo_epi64(z, _mm512_set1_epi64(0xff51_afd7_ed55_8ccdu64 as i64));
        z = _mm512_xor_si512(z, _mm512_srli_epi64::<33>(z));
        z = _mm512_mullo_epi64(z, _mm512_set1_epi64(0xc4ce_b9fe_1a85_ec53u64 as i64));
        z = _mm512_xor_si512(z, _mm512_srli_epi64::<33>(z));
        let q = _mm512_srlv_epi64(z, _mm512_set1_epi64(i64::from(64 - bits)));
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, _mm512_cvtepi64_epi32(q));
    }
}

/// A purchase's side record goes into its event's own bin (SHORT [key | ingest << 40, buy | this],
/// LONG [a, b, ingest, buy | this]): the surv bit is clear and the last word is not zero, which no
/// event record has (a dead copy's last word is 0). Two streams of write fronts fewer; at 8,192
/// partitions every purchase push missed on its own front.
pub const PURCHASE_SIDE: u64 = 1 << 62;

/// A LONG record is two 16-byte units in its partition's one stream, pushed back to back: unit 1
/// [LONG_MARK | w2, w3] (w2 = ingest | surv << 63; w3 = token | ts << 32 | purch << 63, or
/// buy | PURCHASE_SIDE), unit 2 [a, b]. A SHORT w0 holds ingest < 2^23 - 1 in bits 40-62, so
/// all ones there marks a LONG unit 1 (dedup reads both layouts since d1cd61a).
pub const LONG_MARK: u64 = 0x7f_ffff << 40;

/// Appends a LONG record's two units to bin `p`.
#[inline(always)]
fn push_long(bins: &mut Bins<[u64; 2]>, p: usize, a: u64, b: u64, w2: u64, w3: u64) {
    bins.push(p, [LONG_MARK | w2, w3]);
    bins.push(p, [a, b]);
}

/// Event lookups prefetch shared: they are mostly hits, and a line taken for writing is taken
/// away from every other core that reads it. Links and clicks bring most new IDs: for writing.
const EVENT_PREFETCH_W: bool = false;

/// Link batches wait here, LD deep: each copied whole on arrival, both sides' home slots computed
/// and prefetched for writing (links bring most new IDs). The oldest is resolved when the ring is
/// full, in three passes: every side found with loads only; the sides not found inserted in one
/// burst (each insert's locked compare-exchange waits for all older stores, so none are left
/// pending); the batch's edge words written once, complete.
const LD: usize = 4;

struct LinkRing {
    b: [LinkBatch; LD],
    /// Home slots: side a of lane k at 2 k, side b at 2 k + 1.
    slot: [[u32; 16]; LD],
    head: usize,
    n: usize,
}

/// Event batches wait here, D deep: each batch is copied whole on arrival and its users' home
/// slots computed and prefetched; when the ring is full the oldest batch is resolved in one loop
/// (probe, then its complete records into their bins). Against blocks of 32 staged 64-byte
/// records (box 1, perf, p20v5): 0.94G fewer instructions, cycles equal until Bins prefetches
/// each write front's next line (the phase is store-bound: see fast::sink::Bins::push_at).
const D: usize = 4;

/// Partition bits from which the grouped path is used. Box 1, interleaved, cycles up to the end of
/// the event parse: p20v5 (2^11) ring + next-line prefetch 4.38G, grouped 4.71G, grouped +
/// prefetch 4.66G; f60v5 (2^13) grouped 16.62G, grouped + prefetch 17.14G, ring + prefetch
/// 17.21G. With 4 x 2^13 write fronts (2 MB) beyond L2 the fill buffers bind, and the
/// prefetches only compete with the token probes for them.
const BIG_BITS: u32 = 13;

struct EvRing {
    b: [EvBatch; D],
    slot: [[u32; 8]; D],
    /// Next ring entry to fill (the oldest when the ring is full).
    head: usize,
    n: usize,
}

/// At 2^13 partitions and more (final size): surviving events wait in blocks of G with their lookups started; when a block fills, the other
/// block (started G records earlier, its lines landed) is resolved and its records are written
/// complete. Against writing each record at once and OR-ing its token in 64 records later
/// (interleaved, box 1): equal at p20v5 (2,048 partitions); at f60v5 (8,192 partitions, 2 MB of
/// write fronts, beyond L2, so the later OR missed) 400 to 700 ms less CPU in the phase.
const G: usize = 32;

/// A surviving event whose token is pending: user key and slot, partition, the record words
/// (SHORT [w0, w1]; LONG [a, b, w2, w3]; the token goes into the last word), kind flags.
#[derive(Clone, Copy, Default)]
struct EvPend {
    ua: u64,
    ub: u64,
    slot: usize,
    w: [u64; 4],
    part: u32,
    /// Bit 0: the user ID is LONG; bit 1: the event ID is LONG.
    flags: u32,
}

struct Groups {
    blk: [[EvPend; G]; 2],
    /// Block being filled; the other one waits for its tokens.
    cur: usize,
    n: [usize; 2],
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
    pub edges: Bins<u64>,
    pub special: Vec<Special<'a>>,
    pub bad: bool,
    /// Measurement switch ENGINE_SINK (feature `measure` only; always 0 otherwise): 1 drops event
    /// batches, 2 skips the token lookups of events, 3 skips their bin writes.
    mode: u8,
    mbits: u32,
    lnk: Box<LinkRing>,
    evr: Box<EvRing>,
    /// 2^13 partitions or more: the grouped path (blocks of G staged records), no next-line
    /// prefetch on the event bins; below, the ring of D batches with it (see `new`).
    big: bool,
    grp: Box<Groups>,
}

// SAFETY: the ring's pointers point into this sink's own bins (moved with it).
unsafe impl Send for V3Sink<'_, '_> {}
// SAFETY: shared references are only used after `finish` (the ring is drained and never read),
// to read finished chunks.
unsafe impl Sync for V3Sink<'_, '_> {}

impl<'a, 'b> V3Sink<'a, 'b> {
    /// `w`: the pool participant index this sink is made for (its blocks are taken under it).
    pub fn new(tok: &'b Tokens<'a>, ds: &'a Dataset, ctx: &'b Ctx, bits: u32, events_hint: usize, w: usize) -> Self {
        use super::mem::{B256, B2K, B2M, B8K};
        let parts = 1usize << bits;
        let _ = events_hint;
        V3Sink {
            tok,
            ds,
            ctx,
            bits,
            // One block per chunk (v3::mem's classes): SHORT 512 x 16 B, LONG 64 x 32 B, purchases
            // 16 x 16 B and 8 x 32 B, clicks and edges 2 MB.
            short: Bins::from_blocks(parts, &B8K, w).next_line_prefetch(bits < BIG_BITS),
            // Empty: LONG records go into the SHORT stream as two units (push_long).
            long: Bins::from_blocks(parts, &B2K, w),
            // Empty: purchases go into the event bins (PURCHASE_SIDE); dedup still takes these.
            pshort: Bins::from_blocks(parts, &B256, w),
            plong: Bins::from_blocks(parts, &B256, w),
            edges: Bins::from_blocks(1, &B2M, w),
            special: Vec::new(),
            bad: false,
            mode: if cfg!(feature = "measure") { std::env::var("ENGINE_SINK").ok().and_then(|v| v.parse().ok()).unwrap_or(0) } else { 0 },
            mbits: std::env::var("ENGINE_SINK_BITS").ok().and_then(|v| v.parse().ok()).unwrap_or(bits).min(bits),
            lnk: Box::new(LinkRing { b: [LinkBatch::default(); LD], slot: [[0; 16]; LD], head: 0, n: 0 }),
            big: bits >= BIG_BITS,
            grp: Box::new(Groups { blk: [[EvPend::default(); G]; 2], cur: 0, n: [0; 2] }),
            evr: Box::new(EvRing { b: [EvBatch::default(); D], slot: [[0; 8]; D], head: 0, n: 0 }),
        }
    }

    /// Queues a surviving event (its lookup started); a full block resolves the other one.
    #[inline(always)]
    fn group(&mut self, e: EvPend) {
        let g = &mut *self.grp;
        let c = g.cur;
        g.blk[c][g.n[c]] = e;
        g.n[c] += 1;
        if g.n[c] == G {
            self.resolve_block(c ^ 1);
            self.grp.cur = c ^ 1;
        }
    }

    /// Resolves block `k`: tokens, then the complete records into their bins.
    #[inline(always)]
    fn resolve_block(&mut self, k: usize) {
        let n = self.grp.n[k];
        let mut bad = false;
        for i in 0..n {
            let e = self.grp.blk[k][i];
            let t = self.tok.finish(&Pending { long: e.flags & 1 != 0, a: e.ua, b: e.ub, slot: e.slot });
            bad |= t == NO_TOKEN;
            let t = u64::from(t);
            if e.flags & 2 == 0 {
                self.short.push(e.part as usize, [e.w[0], e.w[1] | t]);
            } else {
                push_long(&mut self.short, e.part as usize, e.w[0], e.w[1], e.w[2], e.w[3] | t);
            }
        }
        self.bad |= bad;
        self.grp.n[k] = 0;
    }

    /// Resolves ring entry `j`: every lane's token (survivors), then its complete records.
    #[inline(always)]
    fn resolve_batch(&mut self, j: usize) {
        let V3Sink { tok, evr, short, bad, bits, mbits, mode, .. } = self;
        let b = &evr.b[j];
        let sl = &evr.slot[j];
        let n = b.n as usize;
        let pbits = if cfg!(feature = "measure") { *mbits } else { *bits };
        let notab = cfg!(feature = "measure") && *mode == 5;
        let mut nb = false;
        let mut parts = [0u32; 8];
        parts8(&b.ev_a, &b.ev_b, b.ev_long, pbits, &mut parts);
        for k in 0..n {
            let surv = b.surv >> k & 1 != 0;
            let purch = surv && b.purch >> k & 1 != 0;
            let ing = u64::from(b.ingest[k]);
            let ev_long = b.ev_long >> k & 1 != 0;
            let (a, bb) = (b.ev_a[k], if ev_long { b.ev_b[k] } else { 0 });
            let p = parts[k] as usize;
            let w1 = if surv {
                let t = if notab {
                    sl[k]
                } else if b.user_long >> k & 1 == 0 {
                    tok.short_at(b.user_a[k], sl[k])
                } else {
                    tok.long_at(b.user_a[k], b.user_b[k], sl[k])
                };
                nb |= t == NO_TOKEN;
                u64::from(t) | (u64::from(b.ts[k]) << 32) | (u64::from(purch) << 63)
            } else {
                0
            };
            if purch {
                // The parser accepts 1 to 10,000,000; anything at 2^24 or above would not fit.
                nb |= b.price[k] >> 24 != 0;
                let buy = u64::from(b.cat[k]) | (u64::from(b.price[k]) << 8);
                if ev_long {
                    push_long(short, p, a, bb, ing, buy | PURCHASE_SIDE);
                } else {
                    short.push(p, [a | (ing << 40), buy | PURCHASE_SIDE]);
                }
            }
            if ev_long {
                push_long(short, p, a, bb, ing | (u64::from(surv) << 63), w1);
            } else {
                short.push(p, [a | (ing << 40) | (u64::from(surv) << 63), w1]);
            }
        }
        *bad |= nb;
    }

    /// Resolves link ring entry `j`: finds (loads only), then the inserts in one burst, then the
    /// complete edge words (side a, side b), in lane order.
    #[inline(always)]
    fn resolve_links(&mut self, j: usize) {
        let V3Sink { tok, lnk, edges, bad, .. } = self;
        let b = &lnk.b[j];
        let sl = &lnk.slot[j];
        let n = b.n as usize;
        let mut t = [NO_TOKEN; 16];
        let side = |i: usize| -> Pending {
            let k = i >> 1;
            if i & 1 == 0 {
                Pending { long: b.a_long >> k & 1 != 0, a: b.a_a[k], b: b.a_b[k], slot: sl[i] as usize }
            } else {
                Pending { long: b.b_long >> k & 1 != 0, a: b.b_a[k], b: b.b_b[k], slot: sl[i] as usize }
            }
        };
        for i in 0..2 * n {
            t[i] = tok.find(&side(i));
        }
        let mut nb = false;
        for i in 0..2 * n {
            if t[i] == NO_TOKEN {
                t[i] = tok.finish(&side(i));
                nb |= t[i] == NO_TOKEN;
            }
        }
        for &x in &t[..2 * n] {
            edges.push(0, u64::from(x));
        }
        *bad |= nb;
    }

    /// Resolves every pending lookup. Call after the last delivery of a phase.
    pub fn drain(&mut self) {
        let first = (self.lnk.head + LD - self.lnk.n) % LD;
        for k in 0..self.lnk.n {
            self.resolve_links((first + k) % LD);
        }
        self.lnk.n = 0;
        self.lnk.head = 0;
        let r = &mut *self.evr;
        let first = (r.head + D - r.n) % D;
        for k in 0..r.n {
            self.resolve_batch((first + k) % D);
        }
        self.evr.n = 0;
        self.evr.head = 0;
        let c = self.grp.cur;
        self.resolve_block(c ^ 1);
        self.resolve_block(c);
    }

    pub fn finish(&mut self) {
        self.drain();
        self.short.finish();
        self.long.finish();
        self.pshort.finish();
        self.plong.finish();
        self.edges.finish();
    }

    /// ENGINE_SINK=1: nothing; 2: records without token lookups; 4: only the per-lane field work. ENGINE_SINK_BITS lowers the partition bits of mode 2.
    #[cold]
    #[inline(never)]
    fn events_measure(&mut self, b: &EvBatch) {
        let n = b.n as usize;
        if self.mode == 1 {
            std::hint::black_box(b);
            return;
        }
        let bits = self.mbits;
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
            if b.ev_long >> k & 1 == 0 {
                let a = b.ev_a[k];
                self.short.push(part_of(a, 0, bits), [a | (ing << 40) | (u64::from(surv) << 63), w1]);
            } else {
                let (a, bb) = (b.ev_a[k], b.ev_b[k]);
                push_long(&mut self.short, part_of(a, bb, bits), a, bb, ing | (u64::from(surv) << 63), w1);
            }
        }
    }

    /// The ring path (below 2^13 partitions), for a batch the parser filled on its own.
    #[inline(never)]
    fn events_ring(&mut self, b: &EvBatch) {
        let slot = self.ring_slot();
        // SAFETY: the ring's head entry, resolved and handed out just now.
        unsafe { *slot = *b };
        self.ring_commit();
    }

    /// The ring's head entry, cleared for filling (its old batch resolved first when full).
    #[inline(always)]
    fn ring_slot(&mut self) -> *mut EvBatch {
        if self.evr.n == D {
            self.resolve_batch(self.evr.head);
            self.evr.n = D - 1;
        }
        let b = &mut self.evr.b[self.evr.head];
        b.n = 0;
        b.ev_long = 0;
        b.user_long = 0;
        b.surv = 0;
        b.purch = 0;
        b
    }

    /// The head entry is filled: its users' home slots computed and prefetched, the ring advanced.
    #[inline(always)]
    fn ring_commit(&mut self) {
        let i = self.evr.head;
        let r = &mut *self.evr;
        let b = &r.b[i];
        let n = b.n as usize;
        let notab = cfg!(feature = "measure") && self.mode == 5;
        if notab {
            for k in 0..n {
                r.slot[i][k] = (b.user_a[k] ^ b.user_b[k]) as u32 & 0xff_ffff;
            }
        } else {
            self.tok.homes8::<false>(b.user_long, &b.user_a, &b.user_b, n, &mut r.slot[i]);
        }
        r.head = (i + 1) % D;
        r.n += 1;
    }

    /// The grouped path (2^13 partitions or more).
    #[inline(never)]
    fn events_grouped(&mut self, b: &EvBatch) {
        let n = b.n as usize;
        // all lanes' home slots at once (a dead copy's is unused: one spare prefetch)
        let mut home = [0u32; 8];
        self.tok.homes8::<EVENT_PREFETCH_W>(b.user_long, &b.user_a, &b.user_b, n, &mut home);
        let mut parts = [0u32; 8];
        parts8(&b.ev_a, &b.ev_b, b.ev_long, self.bits, &mut parts);
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
            let ev_long = b.ev_long >> k & 1 != 0;
            let (a, bb) = (b.ev_a[k], if ev_long { b.ev_b[k] } else { 0 });
            let p = parts[k] as usize;
            if purch {
                if ev_long {
                    push_long(&mut self.short, p, a, bb, ing, buy | PURCHASE_SIDE);
                } else {
                    self.short.push(p, [a | (ing << 40), buy | PURCHASE_SIDE]);
                }
            }
            let w: [u64; 4] = if ev_long { [a, bb, ing | (u64::from(surv) << 63), w1] } else { [a | (ing << 40) | (u64::from(surv) << 63), w1, 0, 0] };
            if surv {
                let ul = b.user_long >> k & 1 != 0;
                let slot = home[k] as usize;
                self.group(EvPend { ua: b.user_a[k], ub: b.user_b[k], slot, w, part: p as u32, flags: u32::from(ul) | (u32::from(ev_long) << 1) });
            } else if ev_long {
                // A dead copy only shadows: its key, ingest and dead bit; no token needed.
                push_long(&mut self.short, p, w[0], w[1], w[2], w[3]);
            } else {
                self.short.push(p, [w[0], w[1]]);
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
        if cfg!(feature = "measure") && self.mode != 0 && self.mode != 5 {
            return self.events_measure(b);
        }
        if self.big { self.events_grouped(b) } else { self.events_ring(b) }
    }

    /// Clicks are parsed after dedup by `ClickSink`.
    fn clicks(&mut self, _: &ClickBatch) {
        self.bad = true;
    }

    #[inline]
    fn links(&mut self, b: &LinkBatch) {
        let i = self.lnk.head;
        if self.lnk.n == LD {
            self.resolve_links(i);
        } else {
            self.lnk.n += 1;
        }
        // The appends' lines, owned before they are written.
        self.edges.prefetch_ahead(0, 512);
        let l = &mut *self.lnk;
        l.b[i] = *b;
        let n = b.n as usize;
        let (mut ha, mut hb) = ([0u32; 8], [0u32; 8]);
        self.tok.homes8::<true>(b.a_long, &b.a_a, &b.a_b, n, &mut ha);
        self.tok.homes8::<true>(b.b_long, &b.b_a, &b.b_b, n, &mut hb);
        for k in 0..n {
            l.slot[i][2 * k] = ha[k];
            l.slot[i][2 * k + 1] = hb[k];
        }
        l.head = (i + 1) % LD;
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

    fn foreign_click(&mut self, _: Locator) {
        self.bad = true;
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

/// Clicks, parsed after the events and persons: a click needs its user's person, and a user with
/// no link and no event has no token by then, so its click can reach no session and is dropped
/// here. The table is read-only (no compare-and-swap, shared prefetch). Two random reads per click
/// (token slot, then person), pipelined in blocks of CG: a full block resolves the block before it
/// to tokens and prefetches their persons, and pushes the block before that to person partitions
/// as [person << 32 | ts, locator | campaign << 48] (the clicks stage's record).
pub struct ClickSink<'a, 'b> {
    pub tok: &'b Tokens<'a>,
    pub people: &'b People,
    pub ds: &'a Dataset,
    pub ctx: &'b Ctx,
    pub bins: Bins<[u64; 2]>,
    pub bad: bool,
    pub dropped: u64,
    pbits: u32,
    /// Person partition = person token >> pshift (as Move2::part).
    pshift: u32,
    /// Lookups started: [block being filled, block waiting for its token lines].
    blk: Box<[[ClickPend; CG]; 2]>,
    cur: usize,
    n: [usize; 2],
    /// Tokens found, persons prefetched.
    ready: Box<[(u32, u64, u64); CG]>,
    nready: usize,
}

const CG: usize = 32;

#[derive(Clone, Copy, Default)]
struct ClickPend {
    user: Pending,
    ts: u64,
    w1: u64,
}

impl<'a, 'b> ClickSink<'a, 'b> {
    /// `w`: the pool participant index (the person partitions' chunks are B8K blocks under it:
    /// region memory, not 512 KB allocator blocks on 4 KB pages, which cost 20K faults).
    pub fn new(tok: &'b Tokens<'a>, people: &'b People, ds: &'a Dataset, ctx: &'b Ctx, pbits: u32, pshift: u32, w: usize) -> Self {
        ClickSink {
            tok,
            people,
            ds,
            ctx,
            bins: Bins::from_blocks(1 << pbits, &super::mem::B8K, w),
            bad: false,
            dropped: 0,
            pbits,
            pshift,
            blk: Box::new([[ClickPend::default(); CG]; 2]),
            cur: 0,
            n: [0; 2],
            ready: Box::new([(0, 0, 0); CG]),
            nready: 0,
        }
    }

    #[inline(always)]
    fn part(&self, person: u32) -> usize {
        // Masked to the bins (a token past the driver's capacity would set `bad` in push_person).
        ((person >> self.pshift) as usize) & ((1 << self.pbits) - 1)
    }

    #[inline(always)]
    fn push_person(&mut self, person: u32, ts: u64, w1: u64) {
        self.bad |= (person >> self.pshift) >> self.pbits != 0;
        let p = self.part(person);
        self.bins.push(p, [(u64::from(person) << 32) | ts, w1]);
    }

    /// Pushes the ready block, then resolves block `k`: with persons stamped in the user table, a
    /// find's line holds the person too and the click is written at once; else into the ready
    /// block, its person-map line prefetched.
    #[inline(always)]
    fn advance(&mut self, k: usize) {
        for i in 0..self.nready {
            let (t, ts, w1) = self.ready[i];
            let person = self.people.person_of(t);
            self.push_person(person, ts, w1);
        }
        self.nready = 0;
        if self.tok.stamped() {
            let mut lost = 0;
            for i in 0..self.n[k] {
                let c = self.blk[k][i];
                let t = self.tok.find(&c.user);
                if t == NO_TOKEN {
                    lost += 1;
                    continue;
                }
                let person = match self.tok.person_at(t) {
                    Some(p) => p,
                    None => self.people.person_of(t),
                };
                self.push_person(person, c.ts, c.w1);
            }
            self.dropped += lost;
            self.n[k] = 0;
            return;
        }
        let mut r = 0;
        for i in 0..self.n[k] {
            let c = self.blk[k][i];
            let t = self.tok.find(&c.user);
            if t != NO_TOKEN {
                self.people.prefetch(t);
                self.ready[r] = (t, c.ts, c.w1);
                r += 1;
            }
        }
        self.dropped += (self.n[k] - r) as u64;
        self.nready = r;
        self.n[k] = 0;
    }

    /// Writes every pending click. Call after the last delivery.
    pub fn drain(&mut self) {
        let c = self.cur;
        self.advance(c ^ 1);
        self.advance(c);
        self.advance(c ^ 1);
    }
}

impl<'a> Sink for ClickSink<'a, '_> {
    fn events(&mut self, _: &EvBatch) {
        self.bad = true;
    }
    fn foreign_event(&mut self, _: Locator) {
        self.bad = true;
    }
    fn links(&mut self, _: &LinkBatch) {
        self.bad = true;
    }
    fn foreign_link(&mut self, _: Locator) {
        self.bad = true;
    }

    #[inline]
    fn clicks(&mut self, b: &ClickBatch) {
        let n = b.n as usize;
        let mut h = [0u32; 8];
        self.tok.homes8::<false>(b.user_long, &b.user_a, &b.user_b, n, &mut h);
        for k in 0..n {
            if b.loc[k] >> 48 != 0 {
                self.bad = true;
            }
            let user = Pending { long: b.user_long >> k & 1 != 0, a: b.user_a[k], b: b.user_b[k], slot: h[k] as usize };
            let c = self.cur;
            self.blk[c][self.n[c]] = ClickPend { user, ts: u64::from(b.ts[k]), w1: b.loc[k] | (u64::from(b.campaign[k]) << 48) };
            self.n[c] += 1;
            if self.n[c] == CG {
                self.advance(c ^ 1);
                self.cur = c ^ 1;
            }
        }
    }

    fn foreign_click(&mut self, at: Locator) {
        let line = self.ds.line(Stream::Clicks, at);
        let mut fields = Vec::with_capacity(8);
        let Some(c) = click_line(self.ctx, line, &mut fields) else {
            self.bad = true;
            return;
        };
        if at.0 >> 48 != 0 {
            self.bad = true;
        }
        let t = match classify(c.user_id, b'u') {
            Key::Foreign => self.tok.find_foreign(c.user_id),
            Key::Short(v) => self.tok.find(&self.tok.start(false, v, 0)),
            Key::Long(a, bb) => self.tok.find(&self.tok.start(true, a, bb)),
        };
        if t == NO_TOKEN {
            self.dropped += 1;
            return;
        }
        let ts = (c.ts - self.ctx.click_base_ms) as u64;
        let person = self.people.person_of(t);
        self.push_person(person, ts, at.0 | (u64::from(c.campaign) << 48));
    }
}
