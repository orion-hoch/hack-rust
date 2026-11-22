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

use crate::fast::names::order_key;
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
            soff[(r >> (64 - b2)) as usize + 1] += 1;
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
            s[at[q] as usize] = r;
            at[q] += 1;
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
        for k in 0..b.n as usize {
            self.ends.line((b.a_long >> k & 1 != 0, b.a_a[k], b.a_b[k]), (b.b_long >> k & 1 != 0, b.b_a[k], b.b_b[k]));
        }
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
        let (kx, ky) = (self.uf[x as usize].okey, self.uf[y as usize].okey);
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
        let id = self.uf.len() as u32;
        self.uf.push(Uf { parent: id, wix: 0, okey });
        self.place.push(place);
        id
    }
}

/// A small open-addressing table of one partition's users: key (nonzero) -> node.
struct Small {
    key: Vec<u64>,
    node: Vec<u32>,
    mask: usize,
    /// Slots filled since the last clear (only those are reset).
    used: Vec<u32>,
}

impl Small {
    fn new(cap: usize) -> Small {
        let n = (2 * cap).next_power_of_two().max(64);
        Small { key: vec![0; n], node: vec![0; n], mask: n - 1, used: Vec::with_capacity(cap) }
    }

    fn clear(&mut self) {
        for &i in &self.used {
            self.key[i as usize] = 0;
        }
        self.used.clear();
    }

    /// The node of `k` (k != 0; `h` its hash), or `fresh()` stored for it.
    #[inline(always)]
    fn get(&mut self, k: u64, h: u64, fresh: impl FnOnce() -> u32) -> u32 {
        let mut i = h as usize & self.mask;
        loop {
            let c = self.key[i];
            if c == k {
                return self.node[i];
            }
            if c == 0 {
                let n = fresh();
                self.key[i] = k;
                self.node[i] = n;
                self.used.push(i as u32);
                return n;
            }
            i = (i + 1) & self.mask;
        }
    }
}

/// Root of `x` with path halving.
#[inline(always)]
fn find(uf: &mut [Uf], mut x: u32) -> u32 {
    loop {
        let p = uf[x as usize].parent;
        if p == x {
            return x;
        }
        let g = uf[p as usize].parent;
        uf[x as usize].parent = g;
        x = g;
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
        sbase[q] = nd.uf.len() as u32;
        tab.clear();
        for te in tasks.iter_mut() {
            for r in &mut te.short[te.soff[q] as usize..te.soff[q + 1] as usize] {
                let mu = *r >> 24;
                // ru + 1: nonzero, and the partition's users are told apart by ru alone.
                let id = tab.get((mu & rmask) + 1, mu, || nd.push(order_key(unmix40(mu)), mu));
                *r = (*r & tag) | (u64::from(id) << 24);
            }
        }
        let tw = tsc();
        t_s += tw - tq;
        wbase[q] = nd.uf.len() as u32;
        tab.clear();
        for te in tasks.iter_mut() {
            for w in &mut te.wide[te.woff[q] as usize..te.woff[q + 1] as usize] {
                let (a, b, kind) = (w[0], w[1], w[2] >> 62);
                // Wide users of one partition: hashed by their key, told apart by comparing it.
                let h = mix40(a ^ b.rotate_left(17)) | 1;
                let mut i = h as usize & tab.mask;
                let id = loop {
                    let c = tab.key[i];
                    if c == 0 {
                        let okey = if kind == KIND_LONG { long_okey(a) } else { OKEY_FOREIGN };
                        let widx = nd.wkey.len() as u64;
                        nd.wkey.push([a, b]);
                        let id = nd.push(okey, WIDE | (kind << 61) | (widx << WIDX_SHIFT) | q as u64);
                        tab.key[i] = h;
                        tab.node[i] = id;
                        tab.used.push(i as u32);
                        break id;
                    }
                    if c == h {
                        let n = tab.node[i];
                        let p = nd.place[n as usize];
                        if nd.wkey[((p & !(7 << 61)) >> WIDX_SHIFT) as usize] == [a, b] {
                            break n;
                        }
                    }
                    i = (i + 1) & tab.mask;
                };
                w[0] = u64::from(id);
            }
        }
        t_w += tsc() - tw;
    }
    let tp2 = std::time::Instant::now();
    let c1 = tsc();
    let pass1_ms = (tp2 - ta).as_secs_f64() * 1e3;
    let cyc = (c1 - c0).max(1) as f64;
    // Pass 2, task by task: the nodes into the task's own stretch of the edge array (a task's
    // lines are its own: the stretch stays in cache while it is filled).
    for (t, te) in tasks.iter().enumerate() {
        let eb = 2 * ebase[t];
        for &r in &te.short {
            en[(eb + (r & tag)) as usize] = (r >> 24) as u32;
        }
        for w in &te.wide {
            en[(eb + (w[2] & tag)) as usize] = w[0] as u32;
        }
    }
    sbase[parts] = nd.uf.len() as u32;
    wbase[parts] = nd.uf.len() as u32;
    let t1 = std::time::Instant::now();
    sp.sub = [(ta - t0).as_secs_f64() * 1e3, pass1_ms * t_s as f64 / cyc, pass1_ms * t_w as f64 / cyc, (t1 - tp2).as_secs_f64() * 1e3];
    // Union-find, the smaller id's node the root.
    let n = nd.uf.len();
    for j in 0..lines as usize {
        if j + AHEAD < lines as usize {
            prefetch(&nd.uf[en[2 * (j + AHEAD)] as usize]);
            prefetch(&nd.uf[en[2 * (j + AHEAD) + 1] as usize]);
        }
        let (ra, rb) = (find(&mut nd.uf, en[2 * j]), find(&mut nd.uf, en[2 * j + 1]));
        if ra != rb {
            if nd.less(ra, rb) {
                nd.uf[rb as usize].parent = ra;
            } else {
                nd.uf[ra as usize].parent = rb;
            }
        }
    }
    drop(en);
    let mut persons = 0;
    for x in 0..n {
        if x + AHEAD < n {
            let p = nd.uf[x + AHEAD].parent as usize;
            prefetch(&nd.uf[p]);
        }
        let r = find(&mut nd.uf, x as u32);
        persons += usize::from(r == x as u32);
        nd.uf[x].parent = r;
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
    for x in 0..n {
        if x + AHEAD < n {
            prefetch(&nd.place[nd.uf[x + AHEAD].parent as usize]);
        }
        let r = nd.uf[x].parent as usize;
        let pr = nd.place[r];
        let pk = if pr & WIDE == 0 { pr } else { wix_of(&mut nd, &mut l.wkeys, r) };
        let p = nd.place[x];
        if p & WIDE == 0 {
            l.ru.push((p & rmask) as u32);
            l.pk.push(pk);
        } else {
            let k = nd.wkey[((p & !(7 << 61)) >> WIDX_SHIFT) as usize];
            l.wide.push([k[0], k[1], pk, Nodes::kind(p)]);
        }
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
        if last {
            // The worker that completes the links resolves them itself, beside the other workers'
            // event and click chunks: no extra thread (a ninth thread starved under SCHED_FIFO).
            let r = self.finish(self.b2);
            *self.out.lock().unwrap() = Some(r);
            self.all_done.notify_all();
        }
    }

    /// The links, after the pool run that held the link tasks has returned.
    pub fn result(&self) -> super::Stage<Links> {
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
}
