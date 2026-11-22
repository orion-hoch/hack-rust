//! v4 I4: persons from the links alone, with no table bigger than the cache. Events and clicks
//! never touch any of it.
//!
//! 1. Link tasks (one per chunk, in the pool run shared with the event and click chunks): every
//!    line's two ends become records of the task, grouped by user partition when the task ends
//!    (a counting sort while they are still cached). SHORT end: mu << 24 | e << 1 | side (e = the
//!    line's index in its task); wide end (LONG, FOREIGN): [a, b, kind << 62 | e << 1 | side].
//! 2. `finish`, partition by partition: a small table (ru -> node) gives every distinct user of
//!    the partition a node, nodes numbered in partition order; each end writes its node into the
//!    edge array (two nodes per line).
//! 3. Union-find over the edges, the root of a union being the member whose id sorts first, so
//!    every root is its component's name holder (its key names the person: pk).
//! 4. Layout per user partition, in node order (nodes are already grouped by partition).

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use parselab::{ClickBatch, Dataset, EvBatch, LinkBatch, Locator, Sink, Stream};

use crate::fast::names::POW36;
use crate::v3::ids::{Key, classify, long_bytes, short_bytes};

use super::keys::{intern, long_hash, mix40, unmix40, FOREIGN_MAX};

const NONE: u32 = u32::MAX;
const KIND_LONG: u64 = 1;
const KIND_FOREIGN: u64 = 2;
const WIDE: u64 = 1 << 63;
/// Lines per task are numbered in 23 bits (a 4 MB chunk holds at most about 700K lines).
const E_BITS: u32 = 23;
const E_MASK: u64 = (1 << E_BITS) - 1;
const AHEAD: usize = 16;

/// Byte order of id texts as one integer where it can be: SHORT = order_key (its base-36 digits
/// left-aligned to 8, then its length); LONG = its first 8 hex digits as base-36 digits (hex and
/// base-36 digit values agree, and so does their byte order), length 9. Exact except two LONGs
/// with equal first 8 digits (then by (a, b)) and FOREIGN (OKEY_FOREIGN: by bytes).
const OKEY_FOREIGN: u64 = u64::MAX;

#[inline(always)]
fn long_okey(a: u64) -> u64 {
    let mut x = 0u64;
    for i in 0..8 {
        x = x * 36 + ((a >> (60 - 4 * i)) & 15);
    }
    (x << 4) | 9
}

/// `names::order_key` for x < 2^40, counting the digits down from 8 (nearly every id has 8).
#[inline(always)]
fn okey40(x: u64) -> u64 {
    // Eight digits (93% of the values below 2^40): the value itself, no table, no multiply.
    if x >= POW36[7] {
        return (x << 4) | 8;
    }
    let mut len = 7usize;
    while len > 1 && x < POW36[len - 1] {
        len -= 1;
    }
    ((x * POW36[8 - len]) << 4) | len as u64
}

#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetch never faults.
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

/// One link task's ends, grouped by user partition (`group`).
#[derive(Default)]
pub struct Ends {
    short: Vec<u64>,
    wide: Vec<[u64; 3]>,
    soff: Vec<u32>,
    woff: Vec<u32>,
    /// Lines delivered (two ends each).
    lines: u64,
    bad: bool,
}

impl Ends {
    #[inline(always)]
    fn end(&mut self, long: bool, a: u64, b: u64, tag: u64) {
        if long {
            self.wide.push([a, b, (KIND_LONG << 62) | tag]);
        } else {
            self.short.push((mix40(a) << 24) | tag);
        }
    }

    /// One line's two ends (the tests' way in; `LinkSink::links` writes a batch's ends itself).
    #[cfg_attr(not(test), allow(dead_code))]
    #[inline(always)]
    fn line(&mut self, x: (bool, u64, u64), y: (bool, u64, u64)) {
        let e = self.lines << 1;
        self.end(x.0, x.1, x.2, e);
        self.end(y.0, y.1, y.2, e | 1);
        self.lines += 1;
    }

    /// One side of a line from its bytes (the careful path): SHORT, LONG, or FOREIGN by the
    /// run's interner.
    fn side(&mut self, id: &[u8], tag: u64) {
        match classify(id, b'u') {
            Key::Short(v) => self.end(false, v, 0, tag),
            Key::Long(a, b) => self.end(true, a, b, tag),
            Key::Foreign => {
                let f = intern(id);
                self.bad |= f >= FOREIGN_MAX;
                self.wide.push([u64::from(f) | (1 << 63), 0, (KIND_FOREIGN << 62) | tag]);
            }
        }
    }

    /// Groups the ends by user partition (counting sorts, while they are cached).
    fn group(&mut self, b2: u32) {
        let parts = 1usize << b2;
        self.bad |= self.lines > E_MASK;
        let mut soff = vec![0u32; parts + 1];
        for &r in &self.short {
            // SAFETY: r >> (64 - b2) < 2^b2 = parts, and soff has parts + 1 entries.
            unsafe { *soff.get_unchecked_mut((r >> (64 - b2)) as usize + 1) += 1 };
        }
        let mut woff = vec![0u32; parts + 1];
        for w in &self.wide {
            woff[(long_hash(w[0], w[1]) >> (64 - b2)) as usize + 1] += 1;
        }
        for q in 0..parts {
            soff[q + 1] += soff[q];
            woff[q + 1] += woff[q];
        }
        let mut s = vec![0u64; self.short.len()];
        let mut at = soff.clone();
        for &r in &self.short {
            let q = (r >> (64 - b2)) as usize;
            // SAFETY: q < parts; at[q] runs over partition q's stretch soff[q]..soff[q + 1] of s
            // (the counts above are of these same records), below short.len() = s.len().
            unsafe {
                let a = at.get_unchecked_mut(q);
                *s.get_unchecked_mut(*a as usize) = r;
                *a += 1;
            }
        }
        let mut w = vec![[0u64; 3]; self.wide.len()];
        let mut at = woff.clone();
        for x in &self.wide {
            let q = (long_hash(x[0], x[1]) >> (64 - b2)) as usize;
            w[at[q] as usize] = *x;
            at[q] += 1;
        }
        (self.short, self.wide, self.soff, self.woff) = (s, w, soff, woff);
    }
}

/// The parser's sink of one link task: every delivered line goes into `ends`.
pub struct LinkSink<'a> {
    pub ds: &'a Dataset,
    pub ends: Ends,
}

impl Sink for LinkSink<'_> {
    fn events(&mut self, _: &EvBatch) {
        self.ends.bad = true;
    }
    fn foreign_event(&mut self, _: Locator) {
        self.ends.bad = true;
    }
    fn clicks(&mut self, _: &ClickBatch) {
        self.ends.bad = true;
    }
    fn foreign_click(&mut self, _: Locator) {
        self.ends.bad = true;
    }

    #[inline]
    fn links(&mut self, b: &LinkBatch) {
        let n = (b.n as usize).min(8);
        let e = &mut self.ends;
        // Room for the batch's SHORT ends asked once; they are then written without a check.
        e.short.reserve(16);
        let sp = e.short.as_mut_ptr();
        let mut len = e.short.len();
        let mut t = e.lines << 1;
        for k in 0..n {
            if b.a_long >> k & 1 != 0 {
                e.wide.push([b.a_a[k], b.a_b[k], (KIND_LONG << 62) | t]);
            } else {
                // SAFETY: at most 2 * n <= 16 SHORT ends a batch, reserved above.
                unsafe { *sp.add(len) = (mix40(b.a_a[k]) << 24) | t };
                len += 1;
            }
            if b.b_long >> k & 1 != 0 {
                e.wide.push([b.b_a[k], b.b_b[k], (KIND_LONG << 62) | t | 1]);
            } else {
                // SAFETY: as above.
                unsafe { *sp.add(len) = (mix40(b.b_a[k]) << 24) | t | 1 };
                len += 1;
            }
            t += 2;
        }
        // SAFETY: the first `len` entries are written, within the reserved capacity.
        unsafe { e.short.set_len(len) };
        e.lines += n as u64;
    }

    fn foreign_link(&mut self, at: Locator) {
        let line = self.ds.line(Stream::Links, at);
        let mut fields = Vec::with_capacity(4);
        if !parselab::csv::split(line, &mut fields) || fields.len() != 2 || fields[0].is_empty() || fields[1].is_empty() {
            self.ends.bad = true;
            return;
        }
        let e = self.ends.lines << 1;
        self.ends.side(fields[0], e);
        self.ends.side(fields[1], e | 1);
        self.ends.lines += 1;
    }
}

/// A node's union-find state and byte-order key (16 bytes: a find's chain stays dense).
#[derive(Clone, Copy)]
struct Uf {
    parent: u32,
    /// A wide representative's wkeys index + 1 (0: none yet).
    wix: u32,
    okey: u64,
}

/// A node's place: SHORT = mu (bit 63 clear); wide = WIDE | kind << 61 | widx << 16 | q, with
/// widx its key's index in `Nodes::wkey`.
const WIDX_SHIFT: u32 = 16;

struct Nodes {
    uf: Vec<Uf>,
    place: Vec<u64>,
    /// Keys of the wide nodes, in node order: [a, b].
    wkey: Vec<[u64; 2]>,
}

impl Nodes {
    #[inline(always)]
    fn kind(place: u64) -> u64 {
        if place & WIDE == 0 { 0 } else { (place >> 61) & 3 }
    }

    fn id_bytes(&self, x: u32) -> Vec<u8> {
        let p = self.place[x as usize];
        let mut out = Vec::new();
        match Nodes::kind(p) {
            0 => short_bytes(b'u', unmix40(p), &mut out),
            KIND_LONG => {
                let k = self.wkey[((p & !(7 << 61)) >> WIDX_SHIFT) as usize];
                long_bytes(b'u', k[0], k[1], &mut out);
            }
            _ => {
                let k = self.wkey[((p & !(7 << 61)) >> WIDX_SHIFT) as usize];
                out.extend_from_slice(super::keys::bytes((k[0] & !(1 << 63)) as u32));
            }
        }
        out
    }

    /// Byte order of the ids of nodes x and y (x != y).
    #[inline(always)]
    fn less(&self, x: u32, y: u32) -> bool {
        debug_assert!((x as usize) < self.uf.len() && (y as usize) < self.uf.len());
        // SAFETY: x and y are nodes (roots returned by `find`), below uf.len().
        let (kx, ky) = unsafe { (self.uf.get_unchecked(x as usize).okey, self.uf.get_unchecked(y as usize).okey) };
        if kx != ky && kx != OKEY_FOREIGN && ky != OKEY_FOREIGN {
            return kx < ky;
        }
        self.less_slow(x, y)
    }

    #[cold]
    #[inline(never)]
    fn less_slow(&self, x: u32, y: u32) -> bool {
        let (px, py) = (self.place[x as usize], self.place[y as usize]);
        if Nodes::kind(px) == KIND_LONG && Nodes::kind(py) == KIND_LONG {
            let kx = self.wkey[((px & !(7 << 61)) >> WIDX_SHIFT) as usize];
            let ky = self.wkey[((py & !(7 << 61)) >> WIDX_SHIFT) as usize];
            return (kx[0], kx[1]) < (ky[0], ky[1]);
        }
        self.id_bytes(x) < self.id_bytes(y)
    }

    fn push(&mut self, okey: u64, place: u64) -> u32 {
        let id = self.uf.len();
        debug_assert!(id < self.uf.capacity() && id < self.place.capacity());
        // SAFETY: capacity is the total number of ends, and each end pushes at most one node.
        unsafe {
            self.uf.as_mut_ptr().add(id).write(Uf { parent: id as u32, wix: 0, okey });
            self.place.as_mut_ptr().add(id).write(place);
            self.uf.set_len(id + 1);
            self.place.set_len(id + 1);
        }
        id as u32
    }
}

/// A small open-addressing table of one partition's users: key (nonzero) -> node. It is never
/// cleared: nodes are numbered in partition order, so an entry whose node is below the current
/// partition's first node is stale and reads as an empty slot.
struct Small {
    slot: Vec<Slot>,
    mask: usize,
}

/// `n` is the node + 1 (0: never used).
#[derive(Clone, Copy)]
struct Slot {
    k: u64,
    n: u32,
}

impl Small {
    fn new(cap: usize) -> Small {
        let n = (2 * cap).next_power_of_two().max(64);
        Small { slot: vec![Slot { k: 0, n: 0 }; n], mask: n - 1 }
    }
}

/// Root of `x` with path halving. Every node index in `uf` is below its length (nodes are only
/// ever pushed, and parents are nodes).
#[inline(always)]
fn find(uf: &mut [Uf], mut x: u32) -> u32 {
    let p = uf.as_mut_ptr();
    loop {
        // SAFETY: x and every parent are < uf.len().
        unsafe {
            let px = (*p.add(x as usize)).parent;
            if px == x {
                return x;
            }
            let g = (*p.add(px as usize)).parent;
            // the parent is the root (the usual case below a root): no store, no third read
            if g == px {
                return px;
            }
            (*p.add(x as usize)).parent = g;
            x = g;
        }
    }
}

/// The link result per user partition q (RECORDS-V4.md section 6, as agreed with P1): every
/// linked user, representatives included (their pk is their own key). A person's representative
/// is its byte-wise smallest member, so pk also names it: SHORT pk = mu (unmix40 gives the id),
/// wide pk = 1 << 63 | kind << 61 | home << 32 | i (RECORDS-V4 section 7; home = the person's
/// partition, bits 32-44, also in wkeys[i][2]) with `wkeys[i]` its key.
#[derive(Default)]
pub struct Links {
    /// SHORT linked users of partition q at off[q]..off[q + 1]: their ru and their person's pk.
    pub ru: Vec<u32>,
    pub pk: Vec<u64>,
    pub off: Vec<u32>,
    /// LONG/FOREIGN linked users of q at wide_off[q]..wide_off[q + 1]: [a, b, pk, kind] (kind 1
    /// LONG, 2 FOREIGN; a FOREIGN key is (f | 1 << 63, 0)).
    pub wide: Vec<[u64; 4]>,
    pub wide_off: Vec<u32>,
    /// Wide representatives: [a, b, home partition] (index i of a wide pk; kind in pk bits 61-62).
    pub wkeys: Vec<[u64; 3]>,
}

impl Links {
    /// The id behind a wide person key (bit 63 set, bit 60 clear): (kind, a, b); kind 1 LONG
    /// (the id is 'u' + the 32 hex digits of a, b), kind 2 FOREIGN (the id is keys::bytes(f),
    /// f = a without bit 63). A SHORT pk needs no lookup: its id is 'u' + base36(unmix40(pk)).
    pub fn wide_key(&self, pk: u64) -> (u64, u64, u64) {
        let w = self.wkeys[(pk & 0xffff_ffff) as usize];
        ((pk >> 61) & 3, w[0], w[1])
    }
}

/// Timing of one resolution (ENGINE_TIMING).
#[derive(Default)]
struct Spent {
    nodes_ms: f64,
    union_ms: f64,
    layout_ms: f64,
    persons: usize,
    /// Inside nodes: allocation, pass 1 SHORT, pass 1 wide, pass 2 (ms).
    sub: [f64; 4],
}

#[cfg(test)]
fn stop() -> u32 {
    std::env::var("LNK_STOP").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

/// Pass 1 over one task's SHORT ends of one partition (`recs`): each end's node (found in the
/// partition's table, or opened as node `id`) is written to `en` at the end's tag. Returns the
/// next free node. Its own function, its state in plain values: the loop keeps them in registers.
///
/// # Safety
/// `slot` has `mask + 1` entries; `uf` and `place` have room for one more node per record from
/// `id`; every record's tag (`r & tag`) is inside `en`.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn short_pass(recs: &[u64], slot: *mut Slot, mask: usize, base: u32, rmask: u64, tag: u64, uf: *mut Uf, place: *mut u64, mut id: u32, en: *mut u32) -> u32 {
    for &r in recs {
        let mu = r >> 24;
        // ru + 1: nonzero, and the partition's users are told apart by ru alone.
        let k = (mu & rmask) + 1;
        let mut i = mu as usize & mask;
        // SAFETY: the function's contract (i is masked; id is below the arrays' room; the tag is
        // inside en).
        unsafe {
            let node = loop {
                let s = slot.add(i);
                // an entry of a node below the partition's first is stale: an empty slot
                if (*s).n <= base {
                    uf.add(id as usize).write(Uf { parent: id, wix: 0, okey: okey40(unmix40(mu)) });
                    place.add(id as usize).write(mu);
                    *s = Slot { k, n: id + 1 };
                    id += 1;
                    break id - 1;
                }
                if (*s).k == k {
                    break (*s).n - 1;
                }
                i = (i + 1) & mask;
            };
            *en.add((r & tag) as usize) = node;
        }
    }
    id
}

/// Steps 2 to 4 over every task's grouped ends (each end's record is overwritten with its node).
fn resolve(tasks: &mut [Ends], b2: u32, sp: &mut Spent) -> Links {
    let t0 = std::time::Instant::now();
    let parts = 1usize << b2;
    let rbits = 40 - b2;
    let rmask = (1u64 << rbits) - 1;
    let mut ebase = Vec::with_capacity(tasks.len());
    let mut lines = 0u64;
    for t in tasks.iter() {
        ebase.push(lines);
        lines += t.lines;
    }
    // Every line's two nodes; every slot is written below (each line pushed both its ends).
    let mut en = vec![NONE; 2 * lines as usize];
    let ends = tasks.iter().map(|t| t.short.len() + t.wide.len()).sum::<usize>();
    let mut nd = Nodes { uf: Vec::with_capacity(ends), place: Vec::with_capacity(ends), wkey: Vec::new() };
    let (mut sbase, mut wbase) = (vec![0u32; parts + 1], vec![0u32; parts + 1]);
    let most = (0..parts).map(|q| tasks.iter().map(|t| (t.soff[q + 1] - t.soff[q] + t.woff[q + 1] - t.woff[q]) as usize).sum::<usize>()).max().unwrap_or(0);
    let mut tab = Small::new(most);
    let ta = std::time::Instant::now();
    let tsc = || {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: rdtsc has no preconditions.
        unsafe {
            core::arch::x86_64::_rdtsc()
        }
        #[cfg(not(target_arch = "x86_64"))]
        0u64
    };
    let (mut t_s, mut t_w) = (0u64, 0u64);
    let c0 = tsc();
    // Pass 1, partition by partition: each end's node, written over its own record (records are
    // read in order, so the write is to a line just read); the line tag (e << 1 | side) stays.
    let tag = 2 * E_MASK + 1;
    for q in 0..parts {
        let tq = tsc();
        let base = nd.uf.len() as u32;
        sbase[q] = base;
        for (t, te) in tasks.iter().enumerate() {
            let eb = 2 * ebase[t];
            let recs = &te.short[te.soff[q] as usize..te.soff[q + 1] as usize];
            // SAFETY: uf and place have room for every end (each end adds at most one node) and
            // hold the same number of nodes; a record's tag is below 2 * its task's lines, inside
            // the task's stretch of en (from eb); the table's mask is its length - 1.
            unsafe {
                let id = short_pass(recs, tab.slot.as_mut_ptr(), tab.mask, base, rmask, tag, nd.uf.as_mut_ptr(), nd.place.as_mut_ptr(), nd.uf.len() as u32, en.as_mut_ptr().add(eb as usize));
                nd.uf.set_len(id as usize);
                nd.place.set_len(id as usize);
            }
        }
        let tw = tsc();
        t_s += tw - tq;
        wbase[q] = nd.uf.len() as u32;
        for (t, te) in tasks.iter().enumerate() {
            let eb = 2 * ebase[t];
            for w in &te.wide[te.woff[q] as usize..te.woff[q + 1] as usize] {
                let (a, b, kind) = (w[0], w[1], w[2] >> 62);
                // Wide users of one partition: hashed by their key, told apart by comparing it.
                let h = mix40(a ^ b.rotate_left(17)) | 1;
                let base = wbase[q];
                let mut i = h as usize & tab.mask;
                let id = loop {
                    let c = tab.slot[i];
                    if c.n <= base {
                        let okey = if kind == KIND_LONG { long_okey(a) } else { OKEY_FOREIGN };
                        let widx = nd.wkey.len() as u64;
                        nd.wkey.push([a, b]);
                        let id = nd.push(okey, WIDE | (kind << 61) | (widx << WIDX_SHIFT) | q as u64);
                        tab.slot[i] = Slot { k: h, n: id + 1 };
                        break id;
                    }
                    if c.k == h {
                        let n = c.n - 1;
                        let p = nd.place[n as usize];
                        if nd.wkey[((p & !(7 << 61)) >> WIDX_SHIFT) as usize] == [a, b] {
                            break n;
                        }
                    }
                    i = (i + 1) & tab.mask;
                };
                // SAFETY: as for SHORT ends.
                unsafe { *en.get_unchecked_mut((eb + (w[2] & tag)) as usize) = id };
            }
        }
        t_w += tsc() - tw;
    }
    #[cfg(test)]
    if stop() == 1 {
        return Links::default();
    }
    let tp2 = std::time::Instant::now();
    let c1 = tsc();
    let pass1_ms = (tp2 - ta).as_secs_f64() * 1e3;
    let cyc = (c1 - c0).max(1) as f64;
    #[cfg(test)]
    if stop() == 2 {
        return Links::default();
    }
    sbase[parts] = nd.uf.len() as u32;
    wbase[parts] = nd.uf.len() as u32;
    let t1 = std::time::Instant::now();
    sp.sub = [(ta - t0).as_secs_f64() * 1e3, pass1_ms * t_s as f64 / cyc, pass1_ms * t_w as f64 / cyc, (t1 - tp2).as_secs_f64() * 1e3];
    // Union-find, the smaller id's node the root.
    let n = nd.uf.len();
    let nl = lines as usize;
    let ufp = nd.uf.as_ptr();
    for j in 0..nl {
        if j + AHEAD < nl {
            // SAFETY: every en entry is a node < n; prefetch never faults.
            prefetch(unsafe { ufp.add(*en.get_unchecked(2 * (j + AHEAD)) as usize) });
            prefetch(unsafe { ufp.add(*en.get_unchecked(2 * (j + AHEAD) + 1) as usize) });
        }
        // SAFETY: en holds 2 * nl entries; roots are nodes < n = uf.len().
        let (ea, eb) = unsafe { (*en.get_unchecked(2 * j), *en.get_unchecked(2 * j + 1)) };
        let (ra, rb) = (find(&mut nd.uf, ea), find(&mut nd.uf, eb));
        if ra != rb {
            if nd.less(ra, rb) {
                unsafe { nd.uf.get_unchecked_mut(rb as usize).parent = ra };
            } else {
                unsafe { nd.uf.get_unchecked_mut(ra as usize).parent = rb };
            }
        }
    }
    drop(en);
    #[cfg(test)]
    if stop() == 3 {
        return Links::default();
    }
    let t2 = std::time::Instant::now();
    // Layout: nodes are grouped by partition (SHORT then wide in each), so entries go out in node
    // order.
    let mut l = Links { off: vec![0; parts + 1], wide_off: vec![0; parts + 1], ..Default::default() };
    let (mut ns, mut nw) = (0u32, 0u32);
    for q in 0..parts {
        l.off[q] = ns;
        l.wide_off[q] = nw;
        ns += wbase[q] - sbase[q];
        nw += sbase[q + 1] - wbase[q];
    }
    l.off[parts] = ns;
    l.wide_off[parts] = nw;
    l.ru = Vec::with_capacity(ns as usize);
    l.pk = Vec::with_capacity(ns as usize);
    l.wide = Vec::with_capacity(nw as usize);
    let qmask = (1u64 << WIDX_SHIFT) - 1;
    let wix_of = |nd: &mut Nodes, wkeys: &mut Vec<[u64; 3]>, r: usize| -> u64 {
        let p = nd.place[r];
        if nd.uf[r].wix == 0 {
            let k = nd.wkey[((p & !(7 << 61)) >> WIDX_SHIFT) as usize];
            wkeys.push([k[0], k[1], p & qmask]);
            nd.uf[r].wix = wkeys.len() as u32;
        }
        WIDE | (Nodes::kind(p) << 61) | ((p & qmask) << 32) | u64::from(nd.uf[r].wix - 1)
    };
    let mut persons = 0;
    debug_assert!(nd.place.len() == n);
    // SHORT entries written so far (ns in all, the capacity of ru and pk: their lengths are set
    // once, after the loop).
    let (ru_p, pk_p) = (l.ru.as_mut_ptr(), l.pk.as_mut_ptr());
    let mut si = 0usize;
    for x in 0..n {
        if x + AHEAD < n {
            // SAFETY: a parent is a node < n = uf.len() = place.len(); prefetch never faults.
            unsafe {
                let p = nd.uf.get_unchecked(x + AHEAD).parent as usize;
                prefetch(nd.uf.as_ptr().add(p));
                prefetch(nd.place.as_ptr().add(p));
            }
        }
        let r = find(&mut nd.uf, x as u32) as usize;
        persons += usize::from(r == x);
        // SAFETY: r (a root) and x are nodes < n = place.len().
        let pr = unsafe { *nd.place.get_unchecked(r) };
        let pk = if pr & WIDE == 0 { pr } else { wix_of(&mut nd, &mut l.wkeys, r) };
        let p = unsafe { *nd.place.get_unchecked(x) };
        if p & WIDE == 0 {
            debug_assert!(si < ns as usize);
            // SAFETY: ns SHORT nodes in all, the capacity of both.
            unsafe {
                ru_p.add(si).write((p & rmask) as u32);
                pk_p.add(si).write(pk);
            }
            si += 1;
        } else {
            let k = nd.wkey[((p & !(7 << 61)) >> WIDX_SHIFT) as usize];
            l.wide.push([k[0], k[1], pk, Nodes::kind(p)]);
        }
    }
    // SAFETY: the first si entries of both were written above, si <= ns = their capacity.
    unsafe {
        l.ru.set_len(si);
        l.pk.set_len(si);
    }
    sp.nodes_ms = (t1 - t0).as_secs_f64() * 1e3;
    sp.union_ms = (t2 - t1).as_secs_f64() * 1e3;
    sp.layout_ms = t2.elapsed().as_secs_f64() * 1e3;
    sp.persons = persons;
    l
}

/// I4 as tasks of a pool run shared with the event and click chunks: `task(cx, i, w)` parses link
/// chunk i into its grouped ends; `finish` (on a thread of its own) waits until every task is
/// done, then resolves the nodes, the persons and the layout.
pub struct LinkTasks<'a> {
    ds: &'a Dataset,
    chunks: Vec<parselab::par::Chunk>,
    b2: u32,
    slots: Vec<Mutex<Option<Ends>>>,
    done: Mutex<usize>,
    all_done: std::sync::Condvar,
    /// The finished links: the worker that completes the last link chunk resolves them itself
    /// and leaves the result here.
    out: Mutex<Option<super::Stage<Links>>>,
    defer: std::sync::atomic::AtomicBool,
    timing: bool,
    t0: std::time::Instant,
    busy_ns: AtomicU64,
}

impl<'a> LinkTasks<'a> {
    pub fn new(cx: &super::Cx<'a>) -> Self {
        let ds = cx.ds;
        let link_bytes = ds.bytes(Stream::Links) as usize;
        // At least four chunks a worker (4 MB chunks gave 5 chunks to 8 workers at p20v5).
        let chunk = (link_bytes / (4 * cx.plan.threads.max(1))).clamp(256 << 10, 4 << 20);
        let chunks = parselab::par::plan(ds, &[Stream::Links], chunk);
        LinkTasks {
            ds,
            slots: (0..chunks.len()).map(|_| Mutex::new(None)).collect(),
            chunks,
            b2: cx.plan.b2,
            done: Mutex::new(0),
            all_done: std::sync::Condvar::new(),
            out: Mutex::new(None),
            defer: std::sync::atomic::AtomicBool::new(false),
            timing: cx.timing,
            t0: std::time::Instant::now(),
            busy_ns: AtomicU64::new(0),
        }
    }

    /// Number of link tasks.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Parses link chunk i into its ends, grouped by user partition (any participant w).
    pub fn task(&self, cx: &super::Cx, i: usize, w: usize) {
        let _ = w;
        let tc = std::time::Instant::now();
        let c = &self.chunks[i];
        let file = &self.ds.stream(c.stream)[c.file as usize];
        let lines = (c.to - c.from) / 20 + 16;
        let mut s = LinkSink { ds: self.ds, ends: Ends { short: Vec::with_capacity(2 * lines), ..Default::default() } };
        super::linkkernel::chunk(cx.ctx, c.file, file.bytes(), c.from, c.to, &mut s);
        cx.inp.consumed(cx.inp.first(c.stream as usize) + c.file as usize, c.from, c.to);
        s.ends.group(self.b2);
        *self.slots[i].lock().unwrap() = Some(s.ends);
        self.busy_ns.fetch_add(tc.elapsed().as_nanos() as u64, Relaxed);
        let last = {
            let mut d = self.done.lock().unwrap();
            *d += 1;
            *d == self.chunks.len()
        };
        if last && !self.defer.load(Relaxed) {
            // The worker that completes the links resolves them itself, beside the other workers'
            // event and click chunks: no extra thread (a ninth thread starved under SCHED_FIFO).
            let r = self.finish(self.b2);
            *self.out.lock().unwrap() = Some(r);
            self.all_done.notify_all();
        }
    }

    /// The link chunks run in a pool run of their own (before the side tables are ready): the
    /// last one must not resolve inline; `finish_deferred` does it as a task of the next run.
    pub fn set_defer(&self) {
        self.defer.store(true, Relaxed);
    }

    /// Resolves the links now (every link task has finished) and keeps the result for `result`.
    pub fn finish_deferred(&self) {
        let r = self.finish(self.b2);
        *self.out.lock().unwrap() = Some(r);
    }

    /// The links, after the pool run that held the link tasks has returned.
    pub fn result(&self) -> super::Stage<Links> {
        if let Some(r) = self.out.lock().unwrap().take() {
            return r;
        }
        if self.chunks.is_empty() {
            return self.finish(self.b2);
        }
        self.out.lock().unwrap().take().unwrap_or_else(|| Err("links: the last link task left no result".into()))
    }

    /// Waits for every task, then resolves (on this thread).
    pub fn finish(&self, b2: u32) -> super::Stage<Links> {
        {
            let mut d = self.done.lock().unwrap();
            while *d < self.chunks.len() {
                d = self.all_done.wait(d).unwrap();
            }
        }
        let parse_ms = self.t0.elapsed().as_secs_f64() * 1e3;
        let mut tasks: Vec<Ends> = self.slots.iter().map(|m| m.lock().unwrap().take().expect("every link task filled its slot")).collect();
        if tasks.iter().any(|t| t.bad) || b2 != self.b2 {
            return Err("a bad link line or a link bound passed".into());
        }
        let mut sp = Spent::default();
        let l = resolve(&mut tasks, b2, &mut sp);
        if self.timing {
            eprintln!(
                "timing v4.links parse {:.2} ms (tasks {}, busy {:.1} CPU-ms), nodes {:.2} ms {:.1?}, persons {:.2} ms, layout {:.2} ms; lines {}, users {} ({} wide), persons {} ({} wide)",
                parse_ms,
                self.chunks.len(),
                self.busy_ns.load(Relaxed) as f64 / 1e6,
                sp.nodes_ms,
                sp.sub,
                sp.union_ms,
                sp.layout_ms,
                tasks.iter().map(|t| t.lines).sum::<u64>(),
                l.ru.len() + l.wide.len(),
                l.wide.len(),
                sp.persons,
                l.wkeys.len()
            );
        }
        Ok(l)
    }
}

/// The parsed links (all link tasks run; `finish` completes them).
pub struct LinkParse<'a> {
    tasks: LinkTasks<'a>,
}

/// I4, first half, as its own pool run: every link task.
pub fn parse<'a>(cx: &super::Cx<'a>) -> super::Stage<LinkParse<'a>> {
    let tasks = LinkTasks::new(cx);
    crate::par::run(cx.plan.threads, tasks.len(), |t| t, |w, i| tasks.task(cx, i, *w));
    Ok(LinkParse { tasks })
}

/// I4, second half: nodes, persons and layout, on the calling thread.
pub fn finish(p: LinkParse, b2: u32) -> super::Stage<Links> {
    p.tasks.finish(b2)
}

/// I4 in one call: `parse`, then `finish` on this thread.
pub fn run(cx: &super::Cx) -> super::Stage<Links> {
    let p = parse(cx)?;
    finish(p, cx.plan.b2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn bytes_of(k: &(bool, u64, u64)) -> Vec<u8> {
        let mut b = Vec::new();
        if k.0 { long_bytes(b'u', k.1, k.2, &mut b) } else { short_bytes(b'u', k.1, &mut b) }
        b
    }

    /// Ends of random SHORT/LONG graphs, split over several tasks, through the resolution: every
    /// linked user listed once, in its partition, with the key of its component's byte-smallest
    /// member as pk.
    #[test]
    fn persons_match_brute_force() {
        let mut x = 0x1234_5678_9abc_def1u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let b2 = 9;
        for round in 0..40 {
            let nkeys = 50 + (rnd() % 400) as usize;
            let keys: Vec<(bool, u64, u64)> = (0..nkeys)
                .map(|i| if rnd() % 8 == 0 { (true, rnd() >> (rnd() % 4), rnd()) } else { (false, (rnd() % 5000) * (1 + i as u64 % 3) % (1 << 40), 0) })
                .collect();
            let ntasks = 1 + (rnd() % 4) as usize;
            let mut tasks: Vec<Ends> = (0..ntasks).map(|_| Ends::default()).collect();
            let mut pairs = Vec::new();
            for _ in 0..(round * 7 + 30) {
                let (i, j) = ((rnd() as usize) % nkeys, (rnd() as usize) % nkeys);
                tasks[(rnd() as usize) % ntasks].line(keys[i], keys[j]);
                pairs.push((i, j));
            }
            for t in tasks.iter_mut() {
                t.group(b2);
            }
            let mut sp = Spent::default();
            let l = resolve(&mut tasks, b2, &mut sp);
            // Brute force over distinct keys.
            let mut idx: HashMap<(bool, u64, u64), usize> = HashMap::new();
            for k in &keys {
                let n = idx.len();
                idx.entry(*k).or_insert(n);
            }
            let mut comp: Vec<usize> = (0..idx.len()).collect();
            let used: std::collections::HashSet<usize> = pairs.iter().flat_map(|&(i, j)| [idx[&keys[i]], idx[&keys[j]]]).collect();
            loop {
                let mut changed = false;
                for &(i, j) in &pairs {
                    let (a, b) = (idx[&keys[i]], idx[&keys[j]]);
                    let m = comp[a].min(comp[b]);
                    for c in [a, b] {
                        if comp[c] != m {
                            comp[c] = m;
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
            let mut by_idx: Vec<(bool, u64, u64)> = vec![(false, 0, 0); idx.len()];
            for (k, &i) in &idx {
                by_idx[i] = *k;
            }
            let mut best: HashMap<usize, (Vec<u8>, usize)> = HashMap::new();
            for &u in &used {
                let b = bytes_of(&by_idx[u]);
                let e = best.entry(comp[u]).or_insert((b.clone(), u));
                if b < e.0 {
                    *e = (b, u);
                }
            }
            // The pk each user must carry.
            let pk_bytes = |pk: u64| -> Vec<u8> {
                if pk >> 63 == 0 {
                    let mut b = Vec::new();
                    short_bytes(b'u', unmix40(pk), &mut b);
                    b
                } else {
                    let (_, a, bb) = l.wide_key(pk);
                    let mut b = Vec::new();
                    long_bytes(b'u', a, bb, &mut b);
                    b
                }
            };
            assert_eq!(l.ru.len() + l.wide.len(), used.len(), "round {round}");
            for q in 0..(1usize << b2) {
                for i in l.off[q] as usize..l.off[q + 1] as usize {
                    let mu = ((q as u64) << (40 - b2)) | u64::from(l.ru[i]);
                    let u = idx[&(false, unmix40(mu), 0)];
                    assert_eq!(pk_bytes(l.pk[i]), best[&comp[u]].0, "round {round} short user");
                }
                for w in &l.wide[l.wide_off[q] as usize..l.wide_off[q + 1] as usize] {
                    let u = idx[&(true, w[0], w[1])];
                    assert_eq!((long_hash(w[0], w[1]) >> (64 - b2)) as usize, q);
                    assert_eq!(pk_bytes(w[2]), best[&comp[u]].0, "round {round} wide user");
                }
            }
            // A wide pk's home is its representative's partition.
            for (i, k) in l.wkeys.iter().enumerate() {
                assert_eq!(k[2], long_hash(k[0], k[1]) >> (64 - b2), "wkeys {i}");
            }
        }
    }

    /// Instruction-count bench (run under perf stat with LNK_STOP = 0..4): ~20M-event shape.
    #[test]
    #[ignore]
    fn bench_resolve() {
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let b2 = 9;
        let nu = 1_200_000usize;
        let keys: Vec<(bool, u64, u64)> = (0..nu).map(|_| if rnd() % 8 == 0 { (true, rnd(), rnd()) } else { (false, rnd() % (1 << 40), 0) }).collect();
        let mut tasks: Vec<Ends> = (0..34).map(|_| Ends::default()).collect();
        for i in 0..888_629usize {
            let a = if i < nu / 2 { i } else { (rnd() as usize) % nu };
            let b = (rnd() as usize) % nu;
            tasks[i % 34].line(keys[a], keys[b]);
        }
        for t in tasks.iter_mut() {
            t.group(b2);
        }
        if std::env::var("LNK_STOP").ok().as_deref() == Some("-1") {
            return;
        }
        let mut sp = Spent::default();
        let l = resolve(&mut tasks, b2, &mut sp);
        std::hint::black_box(l);
    }
}
