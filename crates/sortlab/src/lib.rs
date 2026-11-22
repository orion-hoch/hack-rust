//! sortlab: sessions of one person partition, straight from the 8-byte survivor records.
//!
//! Input, per partition: survivor records `person << 32 | ts << 1 | flag` (ts in ms since the
//! window start, under 2^31; the flag is not needed here) and purchase records
//! `[person << 32 | ts, payload]`, both in any order.
//!
//! Output (`Out`): per person its sessions (start ts and event count, in time order) and, for
//! every purchase in input order, the index of the session that holds it. That is all the rest of
//! the stage (crediting, assists, days, rows) needs; it never looks at single events again.
//!
//! Kernels, all with the same output:
//! * `Kernel::Slots`: group by person (one counting pass), then per person fold the events into
//!   15-minute slots (min ts, max ts, count) in a direct array and cut sessions from the occupied
//!   slots alone: same or adjacent slot, same session; two apart, same session exactly when the
//!   later slot's min minus the earlier slot's max is at most GAP; three or more apart, a new
//!   session (lemma proved by the Lean track). No sort at all.
//! * `Kernel::Sort`: group by person, then sort each person's timestamps (insertion for small,
//!   LSD radix for large) and cut at gaps over GAP.
//! * `Kernel::Radix`: one LSD radix over (local person, ts) for the whole partition, then the cut.
//! * `Kernel::Hybrid(t)`: Sort for persons of at most t events, Slots above.
//!
//! Purchases: move 2 writes each purchase together with its own survivor, so a purchase's ts is
//! always one of its person's event timestamps. The kernels rely on that: a purchase goes to the
//! session holding its slot (Slots) or the last session starting at or before it (Sort, Radix).
//! A purchase of an absent person, in a slot without events, or before the first session is an
//! error (the engine falls back), but a purchase with a ts its person never had is not detected.
//!
//! The person field is either a global token (mapped to a partition-local index by a small
//! open-addressing table) or, with `dense = Some(n)`, already a local index below n.

#[cfg(target_arch = "x86_64")]
pub mod net;
mod measure;
pub use measure::*;
pub mod walk;
mod blocks;
pub use blocks::{sessions_blocks, PersonBlocks};

pub const GAP: u32 = 1_800_000;
pub const SLOT: u32 = 900_000;
/// Slots a ts below 2^31 can fall in.
pub const NSLOT: usize = ((1u64 << 31) / SLOT as u64) as usize + 1;
const NWORD: usize = NSLOT.div_ceil(64);
const EMPTY: u32 = u32::MAX;

#[inline(always)]
pub fn ts_of(r: u64) -> u32 {
    ((r >> 1) & 0x7fff_ffff) as u32
}

#[inline(always)]
pub fn person_of(r: u64) -> u32 {
    (r >> 32) as u32
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Slots,
    Sort,
    Radix,
    Hybrid(u32),
    /// The recommended kernel: `NetWith { width: Auto, max: 128 }`.
    Net,
    /// Persons of at most `max` events (capped at 128) by an AVX-512 sorting network, the rest by
    /// slots; slots for everyone where AVX-512 is missing. Width by `NetWidth`.
    NetWith { width: NetWidth, max: u32 },
}

/// Register width of the network.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetWidth {
    /// 256-bit on family 6 model 0x55 (Skylake-SP, Cascade Lake, Cooper Lake: 512-bit code lowers
    /// the clock there), 512-bit on any other CPU with AVX-512; the parser crate's rule. The cargo
    /// feature `net256` (or `net512`) forces the width.
    Auto,
    W512,
    W256,
}

/// The recommended kernel: the network for persons of at most 128 events, slots above.
pub const BEST: Kernel = Kernel::Net;

/// The network path this process would take: 0 none (slots only), 256 or 512.
pub fn net_bits(width: NetWidth) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        let f = std::arch::is_x86_feature_detected!("avx512f");
        let vl = std::arch::is_x86_feature_detected!("avx512vl");
        match width {
            NetWidth::W512 if f => 512,
            NetWidth::W256 if f && vl => 256,
            // cargo features force the width for whole-suite runs (as the parser's mul256 / mul512)
            NetWidth::Auto if f && vl && cfg!(feature = "net256") => 256,
            NetWidth::Auto if f && cfg!(feature = "net512") => 512,
            NetWidth::Auto if f => {
                let (family, model) = net::cpu_family_model();
                if family == 6 && model == 0x55 && vl { 256 } else { 512 }
            }
            _ => 0,
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = width;
        0
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// A purchase whose person has no survivor in this partition.
    BuyWithoutPerson(usize),
    /// A purchase whose ts is not inside any of its person's sessions.
    BuyOutsideSessions(usize),
    /// A person field equal to u32::MAX, or a dense index out of range.
    BadPerson(usize),
    /// `sessions_counted`: the given counts do not match the records.
    CountMismatch,
    /// `walk`: a click (index) whose campaign is out of range or organic.
    BadCampaign(usize),
    /// `walk`: a session (index into `Out::start`) whose day is outside the cells.
    DayOutOfRange(usize),
    /// `walk`: inputs that do not belong together (sizes, or `out` not from these purchases).
    WalkInput,
    /// `sessions_blocks`: a block shorter than its header says, or its purchases not matching
    /// its purch bits.
    BlockShape,
}

#[derive(Default, Debug, Clone)]
pub struct Out {
    /// Person field value of each output person (global token, or the dense index).
    pub persons: Vec<u32>,
    /// Per person, the exclusive end of its sessions in `start` / `events`.
    pub sess_end: Vec<u32>,
    pub start: Vec<u32>,
    pub events: Vec<u32>,
    /// Per purchase (input order), the index of its session.
    pub buy_session: Vec<u32>,
    /// `sessions_blocks`: per purchase (same order), its payload `cat | price << 8`.
    pub buy_pay: Vec<u32>,
}

impl Out {
    fn clear(&mut self) {
        self.persons.clear();
        self.sess_end.clear();
        self.start.clear();
        self.events.clear();
        self.buy_session.clear();
        self.buy_pay.clear();
    }
}

/// Reusable per-thread state; sized on first use, kept between partitions.
pub struct Scratch {
    keys: Vec<u32>,
    vals: Vec<u32>,
    pos: Vec<u32>,
    tbits: u32,
    idx: Vec<u32>,
    first: Vec<u32>,
    cur: Vec<u32>,
    grouped: Vec<u32>,
    bidx: Vec<u32>,
    bfirst: Vec<u32>,
    bcur: Vec<u32>,
    bgrouped: Vec<u32>,
    slots: Box<[Slot; NSLOT]>,
    sid: Box<[u32; NSLOT]>,
    sown: Box<[u32; NSLOT]>,
    sbits: [u64; NWORD],
    serial: u32,
    tmp: Vec<u32>,
    rkeys: Vec<u64>,
    rtmp: Vec<u64>,
    /// Last ts of each session, kept for persons with purchases (indexed like `out.start`).
    ends: Vec<u32>,
    /// Network widths available: (512-bit, 256-bit, Auto's choice), as `net_bits`.
    nets: [u32; 3],
    /// Direct mode (`sessions_chunks` with `Some((base, shift))`): person tokens of the partition
    /// are base + local, local < 2^shift; dshift = 0 when off. dcnt: per local its count, then its
    /// cursor (all zero between partitions); dmap: per local its output index (EMPTY when absent);
    /// dloc: the present locals in output order.
    dbase: u32,
    dshift: u32,
    dcnt: Vec<u32>,
    dmap: Vec<u32>,
    dloc: Vec<u32>,
}

impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

/// One 15-minute slot of the person being folded: min ts, max ts, events.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Slot {
    min: u32,
    max: u32,
    cnt: u32,
    _pad: u32,
}

const SLOT_EMPTY: Slot = Slot { min: u32::MAX, max: 0, cnt: 0, _pad: 0 };

fn boxed(v: u32) -> Box<[u32; NSLOT]> {
    vec![v; NSLOT].into_boxed_slice().try_into().unwrap()
}

impl Scratch {
    pub fn new() -> Self {
        let tbits = 11;
        Scratch {
            keys: vec![EMPTY; 1 << tbits],
            vals: vec![0; 1 << tbits],
            pos: Vec::new(),
            tbits,
            idx: Vec::new(),
            first: Vec::new(),
            cur: Vec::new(),
            grouped: Vec::new(),
            bidx: Vec::new(),
            bfirst: Vec::new(),
            bcur: Vec::new(),
            bgrouped: Vec::new(),
            slots: vec![SLOT_EMPTY; NSLOT].into_boxed_slice().try_into().unwrap_or_else(|_| unreachable!()),
            sid: boxed(0),
            sown: boxed(0),
            sbits: [0; NWORD],
            serial: 0,
            tmp: Vec::new(),
            rkeys: Vec::new(),
            rtmp: Vec::new(),
            ends: Vec::new(),
            nets: [net_bits(NetWidth::W512), net_bits(NetWidth::W256), net_bits(NetWidth::Auto)],
            dbase: 0,
            dshift: 0,
            dcnt: Vec::new(),
            dmap: Vec::new(),
            dloc: Vec::new(),
        }
    }

    /// A fresh owner stamp for the slot fold of one person. `sown` holds 0 or stamps already
    /// issued; on wrap it is cleared, so a stale entry never equals the live stamp.
    #[inline(always)]
    fn next_serial(&mut self) {
        self.serial = self.serial.wrapping_add(1);
        if self.serial == 0 {
            self.sown.fill(0);
            self.serial = 1;
        }
    }

    /// Room in `ends` for the sessions of a person of `n` events (sessions <= events), if it has
    /// purchases; null otherwise.
    #[inline(always)]
    fn ends_for(&mut self, base: usize, n: usize, has_buys: bool) -> *mut u32 {
        if !has_buys {
            return core::ptr::null_mut();
        }
        if self.ends.len() < base + n {
            self.ends.resize(base + n, 0);
        }
        self.ends.as_mut_ptr()
    }

    #[inline(always)]
    fn hash(&self, tok: u32) -> usize {
        (tok.wrapping_mul(0x9E37_79B1) >> (32 - self.tbits)) as usize
    }

    /// Local index of a token, inserting it (with a zero count) if new.
    #[inline(always)]
    fn local(&mut self, tok: u32) -> u32 {
        let mask = (1usize << self.tbits) - 1;
        let mut h = self.hash(tok);
        loop {
            let k = self.keys[h];
            if k == tok {
                return self.vals[h];
            }
            if k == EMPTY {
                return self.insert(tok, h);
            }
            h = (h + 1) & mask;
        }
    }

    #[cold]
    #[inline(never)]
    fn insert(&mut self, tok: u32, h: usize) -> u32 {
        // h is a probe position < keys.len() found empty by `local`
        let p = self.pos.len() as u32;
        self.keys[h] = tok;
        self.vals[h] = p;
        self.pos.push(h as u32);
        self.first.push(0);
        if self.pos.len() * 2 > self.keys.len() {
            self.grow();
        }
        p
    }

    fn grow(&mut self) {
        let toks: Vec<u32> = self.pos.iter().map(|&h| self.keys[h as usize]).collect();
        self.tbits += 1;
        self.keys = vec![EMPTY; 1 << self.tbits];
        self.vals = vec![0; 1 << self.tbits];
        let mask = (1usize << self.tbits) - 1;
        for (p, &tok) in toks.iter().enumerate() {
            let mut h = self.hash(tok);
            while self.keys[h] != EMPTY {
                h = (h + 1) & mask;
            }
            self.keys[h] = tok;
            self.vals[h] = p as u32;
            self.pos[p] = h as u32;
        }
    }

    fn lookup(&self, tok: u32) -> Option<u32> {
        if self.dshift != 0 {
            // direct mode: the local is the token's low bits inside this partition's range
            let l = tok.wrapping_sub(self.dbase);
            if l >> self.dshift != 0 {
                return None;
            }
            // SAFETY: l < 2^dshift = dmap.len() (checked just above).
            let k = unsafe { *self.dmap.get_unchecked(l as usize) };
            return if k == EMPTY { None } else { Some(k) };
        }
        let mask = (1usize << self.tbits) - 1;
        let mut h = self.hash(tok);
        loop {
            let k = self.keys[h];
            if k == tok {
                return Some(self.vals[h]);
            }
            if k == EMPTY {
                return None;
            }
            h = (h + 1) & mask;
        }
    }

    fn reset_table(&mut self) {
        for &h in &self.pos {
            self.keys[h as usize] = EMPTY;
        }
        self.pos.clear();
    }
}

/// Group: local index per record, counts, then ts grouped by person into `sc.grouped`, with
/// `sc.first[p]..sc.cur[p]` the range of person p. Purchases likewise into `sc.bgrouped`.
fn group(ev: &[u64], buys: &[[u64; 2]], dense: Option<usize>, move_ts: bool, sc: &mut Scratch, out: &mut Out) -> Result<usize, Error> {
    group_with(ev, buys, dense, None, move_ts, sc, out)
}

/// `group`, optionally with the per-person counts given (dense mode): then the counting pass is
/// skipped and every write is checked against its person's range instead.
fn group_with(
    ev: &[u64],
    buys: &[[u64; 2]],
    dense: Option<usize>,
    given: Option<&[u32]>,
    move_ts: bool,
    sc: &mut Scratch,
    out: &mut Out,
) -> Result<usize, Error> {
    let n = ev.len();
    if let Some(counts) = given {
        let np = counts.len();
        if counts.iter().map(|&c| c as u64).sum::<u64>() != n as u64 {
            return Err(Error::CountMismatch);
        }
        sc.first.clear();
        sc.first.extend_from_slice(counts);
        out.persons.extend(0..np as u32);
        let mut sum = 0u32;
        sc.cur.clear();
        sc.cur.resize(np, 0);
        sc.idx.clear();
        sc.idx.resize(np, 0); // range ends
        for p in 0..np {
            let c = sc.first[p];
            sc.first[p] = sum;
            sc.cur[p] = sum;
            sum += c;
            sc.idx[p] = sum;
        }
        sc.grouped.clear();
        sc.grouped.resize(n, 0);
        let (cur, lim, grouped) = (&mut sc.cur[..], &sc.idx[..], &mut sc.grouped[..]);
        for (i, &r) in ev.iter().enumerate() {
            let p = person_of(r) as usize;
            if p >= np {
                return Err(Error::BadPerson(i));
            }
            // SAFETY: p < np = cur.len() = lim.len() (checked above).
            let (c, l) = unsafe { (*cur.get_unchecked(p), *lim.get_unchecked(p)) };
            if c >= l {
                return Err(Error::CountMismatch);
            }
            // SAFETY: c < lim[p] <= n = grouped.len() (lim are prefix sums totalling n).
            unsafe {
                *grouped.get_unchecked_mut(c as usize) = ts_of(r);
                *cur.get_unchecked_mut(p) = c + 1;
            }
        }
        // every person filled exactly (the counts sum to n and none overflowed, so all are full)
        group_buys(buys, dense, np, sc)?;
        return Ok(np);
    }
    let np = match dense {
        Some(np) => {
            if np == 0 && n > 0 {
                // no person can hold these records (and np - 1 below would wrap)
                return Err(Error::BadPerson(0));
            }
            sc.first.clear();
            sc.first.resize(np, 0);
            let first = &mut sc.first[..];
            let mut bad = false;
            for &r in ev {
                let p = person_of(r) as usize;
                bad |= p >= np;
                // SAFETY: the loop runs only when n > 0, so np >= 1 (guard above) and
                // min(p, np - 1) < np = first.len(). An out-of-range p only miscounts person
                // np - 1, and is returned as an error below before any count is read.
                unsafe { *first.get_unchecked_mut(p.min(np - 1)) += 1 };
            }
            if bad {
                let i = ev.iter().position(|&r| person_of(r) as usize >= np).unwrap();
                return Err(Error::BadPerson(i));
            }
            out.persons.extend(0..np as u32);
            np
        }
        None => {
            sc.reset_table();
            sc.first.clear();
            sc.idx.clear();
            sc.idx.resize(n, 0);
            for i in 0..n {
                let tok = person_of(ev[i]);
                if tok == EMPTY {
                    return Err(Error::BadPerson(i));
                }
                let p = sc.local(tok);
                sc.idx[i] = p;
                // SAFETY: p < first.len(): `local` pushed a slot for every new index.
                unsafe { *sc.first.get_unchecked_mut(p as usize) += 1 };
            }
            out.persons.extend(sc.pos.iter().map(|&h| sc.keys[h as usize]));
            sc.pos.len()
        }
    };
    // counts to starts; `cur` runs to the ends
    let mut sum = 0u32;
    sc.cur.clear();
    sc.cur.resize(np, 0);
    for p in 0..np {
        let c = sc.first[p];
        sc.first[p] = sum;
        sc.cur[p] = sum;
        sum += c;
    }
    if move_ts {
        sc.grouped.clear();
        sc.grouped.resize(n, 0);
        let cur = &mut sc.cur[..];
        let grouped = &mut sc.grouped[..];
        if dense.is_some() {
            for &r in ev {
                let p = person_of(r) as usize;
                // SAFETY: p < np (checked in the counting pass); cur[p] runs from the person's
                // start up to its end, and the ends are prefix sums of counts totalling n.
                unsafe {
                    let c = cur.get_unchecked_mut(p);
                    *grouped.get_unchecked_mut(*c as usize) = ts_of(r);
                    *c += 1;
                }
            }
        } else {
            let idx = &sc.idx[..];
            for (i, &r) in ev.iter().enumerate() {
                // SAFETY: idx[i] < np; cur[p] < n since the counts sum to n.
                unsafe {
                    let c = cur.get_unchecked_mut(*idx.get_unchecked(i) as usize);
                    *grouped.get_unchecked_mut(*c as usize) = ts_of(r);
                    *c += 1;
                }
            }
        }
    }
    group_buys(buys, dense, np, sc)?;
    Ok(np)
}

/// `group` in token mode with timestamps moved, over the records in pieces (no gather copy),
/// written for instruction count: the table's pointers, mask and shift live in registers for the
/// whole pass (only the cold insert, which may grow the table, reloads them), no bounds checks.
/// Pass 1 per record: token, hash, probe, local index stored, count; pass 2: index, cursor, store.
fn group_chunks(ev: &[&[u64]], buys: &[[u64; 2]], sc: &mut Scratch, out: &mut Out) -> Result<usize, Error> {
    let n: usize = ev.iter().map(|c| c.len()).sum();
    sc.reset_table();
    sc.first.clear();
    sc.idx.clear();
    sc.idx.reserve(n);
    let mut i = 0usize;
    for c in ev {
        let mut j = 0usize;
        let m = c.len();
        let rec = c.as_ptr();
        'table: while j < m {
            let shift = 32 - sc.tbits;
            let mask = (1usize << sc.tbits) - 1;
            let keys = sc.keys.as_ptr();
            let vals = sc.vals.as_ptr();
            let first = sc.first.as_mut_ptr();
            let idx = sc.idx.as_mut_ptr();
            while j < m {
                // SAFETY: j < m = c.len(); h is masked below keys.len() = 1 << tbits = vals.len();
                // p < first.len() (every index handed out has its count slot); i < n <= capacity.
                unsafe {
                    let tok = (*rec.add(j) >> 32) as u32;
                    if tok == EMPTY {
                        return Err(Error::BadPerson(i));
                    }
                    let mut h = (tok.wrapping_mul(0x9E37_79B1) >> shift) as usize;
                    loop {
                        let k = *keys.add(h);
                        if k == tok {
                            break;
                        }
                        if k == EMPTY {
                            // new person (cold): may grow the table, so the registers reload
                            let p = sc.insert(tok, h);
                            *sc.idx.as_mut_ptr().add(i) = p;
                            *sc.first.as_mut_ptr().add(p as usize) += 1;
                            i += 1;
                            j += 1;
                            continue 'table;
                        }
                        h = (h + 1) & mask;
                    }
                    let p = *vals.add(h);
                    *idx.add(i) = p;
                    *first.add(p as usize) += 1;
                }
                i += 1;
                j += 1;
            }
        }
    }
    // SAFETY: idx[0..n] were all written above (one per record).
    unsafe { sc.idx.set_len(n) };
    out.persons.extend(sc.pos.iter().map(|&h| sc.keys[h as usize]));
    let np = sc.pos.len();
    let mut sum = 0u32;
    sc.cur.clear();
    sc.cur.resize(np, 0);
    for p in 0..np {
        let c = sc.first[p];
        sc.first[p] = sum;
        sc.cur[p] = sum;
        sum += c;
    }
    sc.grouped.clear();
    sc.grouped.reserve(n);
    {
        let cur = sc.cur.as_mut_ptr();
        let grouped = sc.grouped.as_mut_ptr();
        let idx = sc.idx.as_ptr();
        let mut i = 0usize;
        for c in ev {
            for &r in c.iter() {
                // SAFETY: i < n; idx[i] < np = cur.len(); cur[p] < n (counts sum to n).
                unsafe {
                    let at = cur.add(*idx.add(i) as usize);
                    *grouped.add(*at as usize) = ts_of(r);
                    *at += 1;
                }
                i += 1;
            }
        }
        // SAFETY: the cursors cover 0..n exactly once (prefix sums of the counts).
        unsafe { sc.grouped.set_len(n) };
    }
    group_buys(buys, None, np, sc)?;
    Ok(np)
}

/// `group` for a partition whose person tokens are `base + local`, local < 2^shift (person id =
/// root token, partition = token >> shift): the table is indexed by the local directly, no hash,
/// no probe, no index array. Pass 1 per record: token, one xor-or into the range check, one
/// count; a scan of the 2^shift counts hands out the output indices (present locals ascending);
/// pass 2: cursor, store. A token outside the range is an error (the run falls back).
fn group_direct(ev: &[&[u64]], buys: &[[u64; 2]], base: u32, shift: u32, sc: &mut Scratch, out: &mut Out) -> Result<usize, Error> {
    let size = 1usize << shift;
    if sc.dcnt.len() != size {
        sc.dcnt = vec![0; size];
        sc.dmap = vec![EMPTY; size];
        sc.dloc.clear();
    }
    for &l in &sc.dloc {
        sc.dmap[l as usize] = EMPTY;
    }
    sc.dloc.clear();
    sc.dbase = base;
    sc.dshift = shift;
    let mask = (size - 1) as u32;
    let n: usize = ev.iter().map(|c| c.len()).sum();
    let mut acc = 0u32;
    {
        let cnt = sc.dcnt.as_mut_ptr();
        for c in ev {
            for &r in c.iter() {
                let tok = (r >> 32) as u32;
                acc |= tok ^ base;
                // SAFETY: tok & mask < size = dcnt.len().
                unsafe { *cnt.add((tok & mask) as usize) += 1 };
            }
        }
    }
    if acc >> shift != 0 {
        sc.dcnt.fill(0);
        sc.dshift = 0;
        return Err(Error::BadPerson(0));
    }
    sc.first.clear();
    let mut sum = 0u32;
    for l in 0..size {
        let c = sc.dcnt[l];
        if c != 0 {
            sc.dmap[l] = sc.dloc.len() as u32;
            sc.dloc.push(l as u32);
            sc.first.push(sum);
            sc.dcnt[l] = sum;
            sum += c;
        }
    }
    let np = sc.dloc.len();
    out.persons.extend(sc.dloc.iter().map(|&l| base + l));
    sc.grouped.clear();
    sc.grouped.reserve(n);
    {
        let cur = sc.dcnt.as_mut_ptr();
        let grouped = sc.grouped.as_mut_ptr();
        for c in ev {
            for &r in c.iter() {
                // SAFETY: the local is masked below size; its cursor runs inside its person's
                // range of 0..n (prefix sums of the counts, which total n).
                unsafe {
                    let at = cur.add(((r >> 32) as u32 & mask) as usize);
                    *grouped.add(*at as usize) = ts_of(r);
                    *at += 1;
                }
            }
        }
        // SAFETY: the cursors cover 0..n exactly once.
        unsafe { sc.grouped.set_len(n) };
    }
    sc.cur.clear();
    for k in 0..np {
        let l = sc.dloc[k] as usize;
        sc.cur.push(sc.dcnt[l]);
        sc.dcnt[l] = 0;
    }
    group_buys(buys, None, np, sc)?;
    Ok(np)
}

/// `sessions` in token mode over the partition's records in pieces (no gather copy). With
/// `direct = Some((base, shift))` the partition's person tokens are base + local, local below
/// 2^shift, and the grouping indexes by the local (`group_direct`); `Scratch::local_of` /
/// lookups then answer from that map until the next call.
pub fn sessions_chunks(ev: &[&[u64]], buys: &[[u64; 2]], kernel: Kernel, direct: Option<(u32, u32)>, sc: &mut Scratch, out: &mut Out) -> Result<(), Error> {
    if let Kernel::Radix = kernel {
        sc.dshift = 0;
        let flat: Vec<u64> = ev.iter().flat_map(|c| c.iter().copied()).collect();
        return sessions(&flat, buys, kernel, None, sc, out);
    }
    out.clear();
    out.buy_session.resize(buys.len(), u32::MAX);
    let np = match direct {
        Some((base, shift)) if (1..=16).contains(&shift) => group_direct(ev, buys, base, shift, sc, out)?,
        _ => {
            sc.dshift = 0;
            group_chunks(ev, buys, sc, out)?
        }
    };
    finish(np, buys, kernel, sc, out)
}

/// Purchases grouped by local person into `sc.bgrouped` (`sc.bfirst[p]..sc.bcur[p]`).
fn group_buys(buys: &[[u64; 2]], dense: Option<usize>, np: usize, sc: &mut Scratch) -> Result<(), Error> {
    let m = buys.len();
    sc.bidx.clear();
    sc.bidx.resize(m, 0);
    sc.bfirst.clear();
    sc.bfirst.resize(np, 0);
    for (j, b) in buys.iter().enumerate() {
        let tok = person_of(b[0]);
        let p = match dense {
            Some(np) if (tok as usize) < np => tok,
            Some(_) => return Err(Error::BuyWithoutPerson(j)),
            None => sc.lookup(tok).ok_or(Error::BuyWithoutPerson(j))?,
        };
        sc.bidx[j] = p;
        sc.bfirst[p as usize] += 1;
    }
    let mut sum = 0u32;
    sc.bcur.clear();
    sc.bcur.resize(np, 0);
    for p in 0..np {
        let c = sc.bfirst[p];
        sc.bfirst[p] = sum;
        sc.bcur[p] = sum;
        sum += c;
    }
    sc.bgrouped.clear();
    sc.bgrouped.resize(m, 0);
    for j in 0..m {
        let p = sc.bidx[j] as usize;
        let at = sc.bcur[p] as usize;
        sc.bgrouped[at] = j as u32;
        sc.bcur[p] = at as u32 + 1;
    }
    Ok(())
}

/// Sessions of one person from its timestamps (any order) by 15-minute slots. Leaves the slot
/// array clean. With `ends` non-null (the person has purchases), `sid` / `sown` say which session
/// holds each occupied slot and `ends[i]` is the last ts of session i (caller: `ends_for`).
#[inline(always)]
fn person_slots(ts: &[u32], sc: &mut Scratch, out: &mut Out, ends: *mut u32) {
    let mark = !ends.is_null();
    let slots = sc.slots.as_mut_ptr();
    let bits = sc.sbits.as_mut_ptr();
    for &t in ts {
        // t < 2^31, so s < NSLOT and s >> 6 < NWORD
        let s = (t / SLOT) as usize;
        debug_assert!(s < NSLOT);
        // SAFETY: t < 2^31 (31-bit field), so s <= (2^31 - 1) / SLOT = NSLOT - 1 and
        // s >> 6 < NWORD; the slot array is NSLOT long and the bitmap NWORD.
        unsafe {
            let e = &mut *slots.add(s);
            e.min = e.min.min(t);
            e.max = e.max.max(t);
            e.cnt += 1;
            *bits.add(s >> 6) |= 1 << (s & 63);
        }
    }
    // Words with an occupied slot, from the bitmap (persons here have over 128 events, so a scan
    // of the NWORD words is cheaper than a summary bit per event).
    let mut summary = 0u64;
    for w in 0..NWORD {
        // SAFETY: w < NWORD.
        summary |= u64::from(unsafe { *bits.add(w) } != 0) << w;
    }
    let serial = sc.serial;
    let sid = sc.sid.as_mut_ptr();
    let sown = sc.sown.as_mut_ptr();
    // Sessions are written branch-free into reserved space: every occupied slot rewrites the
    // current session's start and count at index k, and a cut moves k on first.
    let base = out.start.len();
    out.start.reserve(ts.len());
    out.events.reserve(ts.len());
    let st = out.start.as_mut_ptr();
    let ev = out.events.as_mut_ptr();
    let (mut cur_start, mut cur_cnt, mut prev_s, mut prev_max) = (0u32, 0u32, usize::MAX - 2, 0u32);
    // k starts one below the first session so the first slot's "cut" lands on index 0
    let mut k = usize::MAX;
    while summary != 0 {
        let w = summary.trailing_zeros() as usize;
        summary &= summary - 1;
        // SAFETY: bits of `summary` are only set from s >> 6 < NWORD.
        let mut b = unsafe { core::mem::replace(&mut *bits.add(w), 0) };
        while b != 0 {
            let s = (w << 6) | b.trailing_zeros() as usize;
            b &= b - 1;
            // SAFETY: bit b of word w was set from a slot s < NSLOT, so (w << 6) | b < NSLOT.
            let e = unsafe { core::mem::replace(&mut *slots.add(s), SLOT_EMPTY) };
            // boundary: slots three or more apart, or two apart with a gap over GAP; the first
            // slot (prev_s far below) always cuts
            let d = s.wrapping_sub(prev_s);
            let cut = (d >= 3) | ((d == 2) & (e.min.wrapping_sub(prev_max) > GAP));
            k = k.wrapping_add(cut as usize);
            cur_start = if cut { e.min } else { cur_start };
            cur_cnt = if cut { e.cnt } else { cur_cnt + e.cnt };
            // SAFETY: after the first slot's cut k = 0; each later slot adds at most 1, so
            // k < occupied slots <= ts.len(), and base + ts.len() <= capacity (reserved above).
            unsafe {
                *st.add(base + k) = cur_start;
                *ev.add(base + k) = cur_cnt;
            }
            if mark {
                // SAFETY: s < NSLOT (sid, sown are NSLOT long); base + k < base + ts.len(), the
                // room `ends_for` made. Slots ascend, so the last write per session is its end.
                unsafe {
                    *sid.add(s) = (base + k) as u32;
                    *sown.add(s) = serial;
                    *ends.add(base + k) = e.max;
                }
            }
            prev_s = s;
            prev_max = e.max;
        }
    }
    // SAFETY: indices base..=base+k were written above (none if no slot was occupied).
    unsafe {
        let len = base + k.wrapping_add(1);
        out.start.set_len(len);
        out.events.set_len(len);
    }
}

/// Purchases of one person (indices into `buys`) to sessions, after `person_slots`.
#[inline(always)]
fn buys_by_slot(bj: &[u32], buys: &[[u64; 2]], sc: &Scratch, out: &mut Out) -> Result<(), Error> {
    for &j in bj {
        let t = (buys[j as usize][0] as u32) & 0x7fff_ffff;
        let s = (t / SLOT) as usize; // < NSLOT, as t < 2^31
        if sc.sown[s] != sc.serial {
            return Err(Error::BuyOutsideSessions(j as usize));
        }
        let si = sc.sid[s] as usize;
        // the reference's criterion: inside [start, last ts] of the session
        if t < out.start[si] || t > sc.ends[si] {
            return Err(Error::BuyOutsideSessions(j as usize));
        }
        out.buy_session[j as usize] = si as u32;
    }
    Ok(())
}

/// Sorts timestamps: insertion below 48, else LSD radix over 31 bits in 11-bit digits.
#[inline(always)]
fn sort_ts(ts: &mut [u32], tmp: &mut Vec<u32>) {
    let n = ts.len();
    if n <= 48 {
        for i in 1..n {
            let x = ts[i];
            let mut j = i;
            while j > 0 && ts[j - 1] > x {
                ts[j] = ts[j - 1];
                j -= 1;
            }
            ts[j] = x;
        }
        return;
    }
    tmp.clear();
    tmp.resize(n, 0);
    let mut src_is_ts = true;
    for shift in [0u32, 11, 22] {
        let mut count = [0u32; 2048];
        {
            let src: &[u32] = if src_is_ts { ts } else { tmp };
            for &x in src {
                count[((x >> shift) & 2047) as usize] += 1;
            }
            if count[((src[0] >> shift) & 2047) as usize] as usize == n {
                continue;
            }
        }
        let mut sum = 0;
        for c in count.iter_mut() {
            let x = *c;
            *c = sum;
            sum += x;
        }
        if src_is_ts {
            for &x in ts.iter() {
                let d = ((x >> shift) & 2047) as usize;
                tmp[count[d] as usize] = x;
                count[d] += 1;
            }
        } else {
            for &x in tmp.iter() {
                let d = ((x >> shift) & 2047) as usize;
                ts[count[d] as usize] = x;
                count[d] += 1;
            }
        }
        src_is_ts = !src_is_ts;
    }
    if !src_is_ts {
        ts.copy_from_slice(tmp);
    }
}

/// Sessions of one person from its sorted timestamps; with `ends` non-null also the last ts of
/// each session (caller: `ends_for`).
#[inline(always)]
fn walk_sorted(ts: &[u32], out: &mut Out, ends: *mut u32) {
    let mut start = ts[0];
    let mut cnt = 1u32;
    let mut prev = ts[0];
    for &t in &ts[1..] {
        if t - prev > GAP {
            if !ends.is_null() {
                // SAFETY: session index < base + ts.len(), the room `ends_for` made.
                unsafe { *ends.add(out.start.len()) = prev };
            }
            out.start.push(start);
            out.events.push(cnt);
            start = t;
            cnt = 0;
        }
        cnt += 1;
        prev = t;
    }
    if !ends.is_null() {
        // SAFETY: as above.
        unsafe { *ends.add(out.start.len()) = prev };
    }
    out.start.push(start);
    out.events.push(cnt);
}

/// Purchases of one person to sessions `s0..` by start ts (the last session starting at or
/// before the purchase).
#[inline(always)]
fn buys_by_start(bj: &[u32], buys: &[[u64; 2]], s0: usize, out: &mut Out, ends: &[u32]) -> Result<(), Error> {
    let starts = &out.start[s0..];
    for &j in bj {
        let t = (buys[j as usize][0] as u32) & 0x7fff_ffff;
        // number of sessions with start <= t
        let k = starts.partition_point(|&s| s <= t);
        // the reference's criterion: inside [start, last ts] of that session
        if k == 0 || t > ends[s0 + k - 1] {
            return Err(Error::BuyOutsideSessions(j as usize));
        }
        out.buy_session[j as usize] = (s0 + k - 1) as u32;
    }
    Ok(())
}

/// `sessions` in dense mode with each person's record count given (`counts[p]` records have
/// person field p): the counting pass is skipped. Wrong counts are an error, never an
/// out-of-bounds write. The radix kernel ignores the counts.
pub fn sessions_counted(ev: &[u64], buys: &[[u64; 2]], kernel: Kernel, counts: &[u32], sc: &mut Scratch, out: &mut Out) -> Result<(), Error> {
    if let Kernel::Radix = kernel {
        return sessions(ev, buys, kernel, Some(counts.len()), sc, out);
    }
    out.clear();
    out.buy_session.resize(buys.len(), u32::MAX);
    sc.dshift = 0;
    let np = group_with(ev, buys, Some(counts.len()), Some(counts), true, sc, out)?;
    finish(np, buys, kernel, sc, out)
}

/// Sessions of one partition. `out` is cleared first.
pub fn sessions(
    ev: &[u64],
    buys: &[[u64; 2]],
    kernel: Kernel,
    dense: Option<usize>,
    sc: &mut Scratch,
    out: &mut Out,
) -> Result<(), Error> {
    out.clear();
    out.buy_session.resize(buys.len(), u32::MAX);
    sc.dshift = 0;
    if let Kernel::Radix = kernel {
        return radix_partition(ev, buys, dense, sc, out);
    }
    let np = group(ev, buys, dense, true, sc, out)?;
    finish(np, buys, kernel, sc, out)
}

/// The per-person part after grouping.
fn finish(np: usize, buys: &[[u64; 2]], kernel: Kernel, sc: &mut Scratch, out: &mut Out) -> Result<(), Error> {
    #[cfg(target_arch = "x86_64")]
    let net = match kernel {
        Kernel::Net => Some((NetWidth::Auto, 128)),
        Kernel::NetWith { width, max } => Some((width, max)),
        _ => None,
    };
    #[cfg(target_arch = "x86_64")]
    if let Some((width, max)) = net {
        let bits = sc.nets[match width {
            NetWidth::W512 => 0,
            NetWidth::W256 => 1,
            NetWidth::Auto => 2,
        }];
        let max = (max as usize).min(net::MAX_N);
        if bits == 512 {
            // SAFETY: bits == 512 only when AVX-512F was detected (`net_bits`).
            return unsafe { net_partition512(np, max, buys, sc, out) };
        }
        if bits == 256 {
            // SAFETY: bits == 256 only when AVX-512F and AVX-512VL were detected (`net_bits`).
            return unsafe { net_partition256(np, max, buys, sc, out) };
        }
        // no AVX-512: slots for everyone (the generic loop below)
    }
    let mut grouped = std::mem::take(&mut sc.grouped);
    for p in 0..np {
        let (a, z) = (sc.first[p] as usize, sc.cur[p] as usize);
        if a < z {
            let ts = &mut grouped[a..z];
            let small = match kernel {
                Kernel::Sort => true,
                Kernel::Hybrid(t) => ts.len() <= t as usize,
                _ => false,
            };
            let net = false;
            let s0 = out.start.len();
            let (ba, bz) = (sc.bfirst[p] as usize, sc.bcur[p] as usize);
            let ends = sc.ends_for(s0, ts.len(), ba < bz);
            if net {
                unreachable!()
            } else if small {
                sort_ts(ts, &mut sc.tmp);
                walk_sorted(ts, out, ends);
                if ba < bz {
                    let bj = std::mem::take(&mut sc.bgrouped);
                    let r = buys_by_start(&bj[ba..bz], buys, s0, out, &sc.ends);
                    sc.bgrouped = bj;
                    r?;
                }
            } else {
                sc.next_serial();
                person_slots(ts, sc, out, ends);
                if ba < bz {
                    let bj = std::mem::take(&mut sc.bgrouped);
                    let r = buys_by_slot(&bj[ba..bz], buys, sc, out);
                    sc.bgrouped = bj;
                    r?;
                }
            }
        } else if sc.bfirst[p] < sc.bcur[p] {
            sc.grouped = grouped;
            return Err(Error::BuyWithoutPerson(sc.bgrouped[sc.bfirst[p] as usize] as usize));
        }
        out.sess_end.push(out.start.len() as u32);
    }
    sc.grouped = grouped;
    Ok(())
}

/// One LSD radix sort of (local person, ts) keys for the whole partition, then the cut.
fn radix_partition(ev: &[u64], buys: &[[u64; 2]], dense: Option<usize>, sc: &mut Scratch, out: &mut Out) -> Result<(), Error> {
    let np = group(ev, buys, dense, false, sc, out)?;
    let n = ev.len();
    if dense.is_some() {
        sc.idx.clear();
        sc.idx.extend(ev.iter().map(|&r| person_of(r)));
    }
    let pbits = (usize::BITS - np.max(1).leading_zeros()).max(1);
    let total = 31 + pbits;
    let passes = total.div_ceil(11);
    let digit = total.div_ceil(passes);
    sc.rkeys.clear();
    sc.rkeys.extend(ev.iter().zip(&sc.idx).map(|(&r, &p)| (u64::from(p) << 31) | u64::from(ts_of(r))));
    sc.rtmp.clear();
    sc.rtmp.resize(n, 0);
    let mask = (1u64 << digit) - 1;
    let mut count = vec![0u32; 1 << digit];
    let mut shift = 0;
    while shift < total {
        count.iter_mut().for_each(|c| *c = 0);
        for &k in &sc.rkeys {
            count[((k >> shift) & mask) as usize] += 1;
        }
        let mut sum = 0;
        for c in count.iter_mut() {
            let x = *c;
            *c = sum;
            sum += x;
        }
        for &k in &sc.rkeys {
            let d = ((k >> shift) & mask) as usize;
            sc.rtmp[count[d] as usize] = k;
            count[d] += 1;
        }
        std::mem::swap(&mut sc.rkeys, &mut sc.rtmp);
        shift += digit;
    }
    // cut per person
    let keys = std::mem::take(&mut sc.rkeys);
    let mut i = 0;
    let mut p = 0usize;
    while p < np {
        let s0 = out.start.len();
        let mut j = i;
        while j < n && (keys[j] >> 31) as usize == p {
            j += 1;
        }
        let (ba, bz) = (sc.bfirst[p] as usize, sc.bcur[p] as usize);
        if j > i {
            let ends = sc.ends_for(s0, j - i, ba < bz);
            let mut start = (keys[i] & 0x7fff_ffff) as u32;
            let (mut prev, mut cnt) = (start, 1u32);
            for &k in &keys[i + 1..j] {
                let t = (k & 0x7fff_ffff) as u32;
                if t - prev > GAP {
                    if !ends.is_null() {
                        // SAFETY: session index < s0 + (j - i), the room `ends_for` made.
                        unsafe { *ends.add(out.start.len()) = prev };
                    }
                    out.start.push(start);
                    out.events.push(cnt);
                    start = t;
                    cnt = 0;
                }
                cnt += 1;
                prev = t;
            }
            if !ends.is_null() {
                // SAFETY: as above.
                unsafe { *ends.add(out.start.len()) = prev };
            }
            out.start.push(start);
            out.events.push(cnt);
        }
        if ba < bz {
            if j == i {
                sc.rkeys = keys;
                return Err(Error::BuyWithoutPerson(sc.bgrouped[ba] as usize));
            }
            let bj = std::mem::take(&mut sc.bgrouped);
            let r = buys_by_start(&bj[ba..bz], buys, s0, out, &sc.ends);
            sc.bgrouped = bj;
            if r.is_err() {
                sc.rkeys = keys;
                return r;
            }
        }
        out.sess_end.push(out.start.len() as u32);
        i = j;
        p += 1;
    }
    sc.rkeys = keys;
    Ok(())
}


/// The per-person loop of `Kernel::Net` for register type L: the network for persons of at most
/// `max` (<= 128) events, slots above. Inlined into the two target-feature entry points below.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn net_partition<L: net::Lanes>(np: usize, max: usize, buys: &[[u64; 2]], sc: &mut Scratch, out: &mut Out) -> Result<(), Error> {
    let grouped = std::mem::take(&mut sc.grouped);
    let bj = std::mem::take(&mut sc.bgrouped);
    let n = grouped.len();
    // One reservation for the partition (sessions <= events; W lanes of slack for the widest
    // masked store), so the per-person paths never check capacity; `ends` sized once when
    // any person has purchases.
    out.start.reserve(n + 64);
    out.events.reserve(n + 64);
    out.sess_end.reserve(np);
    let any_buys = !bj.is_empty();
    if any_buys && sc.ends.len() < n + 64 {
        sc.ends.resize(n + 64, 0);
    }
    let ends_all = if any_buys { sc.ends.as_mut_ptr() } else { core::ptr::null_mut() };
    // The per-person bounds, as locals (nothing reloaded through `sc` in the loop).
    let (first, cur, bfirst, bcur) = (core::mem::take(&mut sc.first), core::mem::take(&mut sc.cur), core::mem::take(&mut sc.bfirst), core::mem::take(&mut sc.bcur));
    let mut res = Ok(());
    let se = out.sess_end.as_mut_ptr();
    let mut np_done = 0usize;
    for p in 0..np {
        let (a, z) = (*first.get_unchecked(p) as usize, *cur.get_unchecked(p) as usize);
        let (ba, bz) = (*bfirst.get_unchecked(p) as usize, *bcur.get_unchecked(p) as usize);
        if a < z {
            let ts = grouped.get_unchecked(a..z);
            let s0 = out.start.len();
            let ends = if ba < bz { ends_all } else { core::ptr::null_mut() };
            if ts.len() <= max {
                // SAFETY: 1 <= ts.len() <= max <= MAX_N; L's features (caller); ts < 2^31 (31-bit
                // field); room for s0 + ts.len() sessions in start/events/ends (reserved above).
                net::person_net_inl::<L>(ts, out, ends);
                if ba < bz {
                    if let Err(e) = buys_by_start(bj.get_unchecked(ba..bz), buys, s0, out, &sc.ends) {
                        res = Err(e);
                        break;
                    }
                }
            } else {
                sc.next_serial();
                person_slots(ts, sc, out, ends);
                if ba < bz {
                    if let Err(e) = buys_by_slot(bj.get_unchecked(ba..bz), buys, sc, out) {
                        res = Err(e);
                        break;
                    }
                }
            }
        } else if ba < bz {
            res = Err(Error::BuyWithoutPerson(bj[ba] as usize));
            break;
        }
        // SAFETY: np entries reserved above.
        *se.add(p) = out.start.len() as u32;
        np_done = p + 1;
    }
    out.sess_end.set_len(out.sess_end.len() + np_done);
    sc.first = first;
    sc.cur = cur;
    sc.bfirst = bfirst;
    sc.bcur = bcur;
    sc.grouped = grouped;
    sc.bgrouped = bj;
    res
}

/// # Safety
/// AVX-512F present.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn net_partition512(np: usize, max: usize, buys: &[[u64; 2]], sc: &mut Scratch, out: &mut Out) -> Result<(), Error> {
    net_partition::<net::Z>(np, max, buys, sc, out)
}

/// # Safety
/// AVX-512F and AVX-512VL present.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl")]
unsafe fn net_partition256(np: usize, max: usize, buys: &[[u64; 2]], sc: &mut Scratch, out: &mut Out) -> Result<(), Error> {
    net_partition::<net::Y>(np, max, buys, sc, out)
}
