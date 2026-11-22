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
    /// Persons of at most 32 events by an AVX-512 sorting network, the rest by slots (slots for
    /// everyone where AVX-512 is missing).
    Net,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// A purchase whose person has no survivor in this partition.
    BuyWithoutPerson(usize),
    /// A purchase whose ts is not inside any of its person's sessions.
    BuyOutsideSessions(usize),
    /// A person field equal to u32::MAX, or a dense index out of range.
    BadPerson(usize),
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
}

impl Out {
    fn clear(&mut self) {
        self.persons.clear();
        self.sess_end.clear();
        self.start.clear();
        self.events.clear();
        self.buy_session.clear();
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
    avx512: bool,
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

fn has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

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
            avx512: has_avx512(),
        }
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
    let n = ev.len();
    let np = match dense {
        Some(np) => {
            sc.first.clear();
            sc.first.resize(np, 0);
            let first = &mut sc.first[..];
            let mut bad = false;
            for &r in ev {
                let p = person_of(r) as usize;
                bad |= p >= np;
                // SAFETY: an out-of-range p is caught below before anything reads `first`;
                // index min(p, np - 1) keeps the write in bounds meanwhile.
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
                // SAFETY: p < np (checked above); cur[p] < n since the counts sum to n.
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
    // purchases
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
    Ok(np)
}

/// Sessions of one person from its timestamps (any order) by 15-minute slots. Leaves the slot
/// array clean. With `mark`, `sid` / `sown` say which session holds each occupied slot (for the
/// person's purchases).
#[inline(always)]
fn person_slots(ts: &[u32], sc: &mut Scratch, out: &mut Out, mark: bool) {
    let slots = sc.slots.as_mut_ptr();
    let bits = sc.sbits.as_mut_ptr();
    let mut summary = 0u64;
    for &t in ts {
        // t < 2^31, so s < NSLOT and s >> 6 < NWORD
        let s = (t / SLOT) as usize;
        debug_assert!(s < NSLOT);
        // SAFETY: s < NSLOT (above); the slot array and bitmap are NSLOT / NWORD long.
        unsafe {
            let e = &mut *slots.add(s);
            e.min = e.min.min(t);
            e.max = e.max.max(t);
            e.cnt += 1;
            *bits.add(s >> 6) |= 1 << (s & 63);
        }
        summary |= 1 << (s >> 6);
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
        // SAFETY: w < NWORD (a bit of `summary` set above from s >> 6).
        let mut b = unsafe { core::mem::replace(&mut *bits.add(w), 0) };
        while b != 0 {
            let s = (w << 6) | b.trailing_zeros() as usize;
            b &= b - 1;
            // SAFETY: s < NSLOT (set above).
            let e = unsafe { core::mem::replace(&mut *slots.add(s), SLOT_EMPTY) };
            // boundary: slots three or more apart, or two apart with a gap over GAP; the first
            // slot (prev_s far below) always cuts
            let d = s.wrapping_sub(prev_s);
            let cut = (d >= 3) | ((d == 2) & (e.min.wrapping_sub(prev_max) > GAP));
            k = k.wrapping_add(cut as usize);
            cur_start = if cut { e.min } else { cur_start };
            cur_cnt = if cut { e.cnt } else { cur_cnt + e.cnt };
            // SAFETY: k < number of occupied slots <= ts.len(), reserved above.
            unsafe {
                *st.add(base + k) = cur_start;
                *ev.add(base + k) = cur_cnt;
            }
            if mark {
                // SAFETY: s < NSLOT.
                unsafe {
                    *sid.add(s) = (base + k) as u32;
                    *sown.add(s) = serial;
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
        let s = (t / SLOT) as usize;
        if sc.sown[s] != sc.serial {
            return Err(Error::BuyOutsideSessions(j as usize));
        }
        out.buy_session[j as usize] = sc.sid[s];
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

/// Sessions of one person from its sorted timestamps.
#[inline(always)]
fn walk_sorted(ts: &[u32], out: &mut Out) {
    let mut start = ts[0];
    let mut cnt = 1u32;
    let mut prev = ts[0];
    for &t in &ts[1..] {
        if t - prev > GAP {
            out.start.push(start);
            out.events.push(cnt);
            start = t;
            cnt = 0;
        }
        cnt += 1;
        prev = t;
    }
    out.start.push(start);
    out.events.push(cnt);
}

/// Purchases of one person to sessions `s0..` by start ts (the last session starting at or
/// before the purchase).
#[inline(always)]
fn buys_by_start(bj: &[u32], buys: &[[u64; 2]], s0: usize, out: &mut Out) -> Result<(), Error> {
    let starts = &out.start[s0..];
    for &j in bj {
        let t = (buys[j as usize][0] as u32) & 0x7fff_ffff;
        // number of sessions with start <= t
        let k = starts.partition_point(|&s| s <= t);
        if k == 0 {
            return Err(Error::BuyOutsideSessions(j as usize));
        }
        out.buy_session[j as usize] = (s0 + k - 1) as u32;
    }
    Ok(())
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
    if let Kernel::Radix = kernel {
        return radix_partition(ev, buys, dense, sc, out);
    }
    let np = group(ev, buys, dense, true, sc, out)?;
    #[cfg(target_arch = "x86_64")]
    if kernel == Kernel::Net && sc.avx512 {
        // SAFETY: AVX-512F was detected.
        return unsafe { net_partition(np, buys, sc, out) };
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
            let net = kernel == Kernel::Net && sc.avx512 && ts.len() <= 32;
            let s0 = out.start.len();
            let (ba, bz) = (sc.bfirst[p] as usize, sc.bcur[p] as usize);
            if net {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: AVX-512F was detected; 1 <= ts.len() <= 32.
                unsafe {
                    net::person_net(ts, out)
                };
                if ba < bz {
                    let bj = std::mem::take(&mut sc.bgrouped);
                    let r = buys_by_start(&bj[ba..bz], buys, s0, out);
                    sc.bgrouped = bj;
                    r?;
                }
            } else if small {
                sort_ts(ts, &mut sc.tmp);
                walk_sorted(ts, out);
                if ba < bz {
                    let bj = std::mem::take(&mut sc.bgrouped);
                    let r = buys_by_start(&bj[ba..bz], buys, s0, out);
                    sc.bgrouped = bj;
                    r?;
                }
            } else {
                sc.serial = sc.serial.wrapping_add(1);
                person_slots(ts, sc, out, ba < bz);
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
        if j > i {
            let mut start = (keys[i] & 0x7fff_ffff) as u32;
            let (mut prev, mut cnt) = (start, 1u32);
            for &k in &keys[i + 1..j] {
                let t = (k & 0x7fff_ffff) as u32;
                if t - prev > GAP {
                    out.start.push(start);
                    out.events.push(cnt);
                    start = t;
                    cnt = 0;
                }
                cnt += 1;
                prev = t;
            }
            out.start.push(start);
            out.events.push(cnt);
        }
        let (ba, bz) = (sc.bfirst[p] as usize, sc.bcur[p] as usize);
        if ba < bz {
            if j == i {
                sc.rkeys = keys;
                return Err(Error::BuyWithoutPerson(sc.bgrouped[ba] as usize));
            }
            let bj = std::mem::take(&mut sc.bgrouped);
            let r = buys_by_start(&bj[ba..bz], buys, s0, out);
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


/// One person's sessions by slots, for measurement (sessions appended to `out.start/events`).
pub fn person_by_slots(ts: &[u32], sc: &mut Scratch, out: &mut Out) {
    sc.serial = sc.serial.wrapping_add(1);
    person_slots(ts, sc, out, false);
}

/// One person's sessions by sorting `ts` in place, for measurement.
pub fn person_by_sort(ts: &mut [u32], sc: &mut Scratch, out: &mut Out) {
    sort_ts(ts, &mut sc.tmp);
    walk_sorted(ts, out);
}

/// Clears only the session arrays of `out` (for measurement loops).
pub fn clear_sessions(out: &mut Out) {
    out.start.clear();
    out.events.clear();
}

/// Grouping alone (persons, counts, timestamps grouped, purchases grouped), for measurement.
/// Returns the number of persons.
pub fn group_only(ev: &[u64], buys: &[[u64; 2]], dense: Option<usize>, sc: &mut Scratch, out: &mut Out) -> Result<usize, Error> {
    out.clear();
    group(ev, buys, dense, true, sc, out)
}

/// One person's sessions by the AVX-512 network (1..=32 events), for measurement. False where
/// AVX-512 is missing or the person is larger.
pub fn person_by_net(ts: &[u32], sc: &Scratch, out: &mut Out) -> bool {
    if !sc.avx512 || ts.is_empty() || ts.len() > 32 {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: AVX-512F detected; 1 <= ts.len() <= 32.
    unsafe {
        net::person_net(ts, out)
    };
    true
}

/// The per-person loop of `Kernel::Net`, compiled with AVX-512 so the small-person network and
/// the slot fold inline.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn net_partition(np: usize, buys: &[[u64; 2]], sc: &mut Scratch, out: &mut Out) -> Result<(), Error> {
    let grouped = std::mem::take(&mut sc.grouped);
    let bj = std::mem::take(&mut sc.bgrouped);
    let mut res = Ok(());
    for p in 0..np {
        let (a, z) = (sc.first[p] as usize, sc.cur[p] as usize);
        let (ba, bz) = (sc.bfirst[p] as usize, sc.bcur[p] as usize);
        if a < z {
            let ts = &grouped[a..z];
            let s0 = out.start.len();
            if ts.len() <= 32 {
                net::person_net_inl::<true>(ts, out);
                if ba < bz {
                    if let Err(e) = buys_by_start(&bj[ba..bz], buys, s0, out) {
                        res = Err(e);
                        break;
                    }
                }
            } else {
                sc.serial = sc.serial.wrapping_add(1);
                person_slots(ts, sc, out, ba < bz);
                if ba < bz {
                    if let Err(e) = buys_by_slot(&bj[ba..bz], buys, sc, out) {
                        res = Err(e);
                        break;
                    }
                }
            }
        } else if ba < bz {
            res = Err(Error::BuyWithoutPerson(bj[ba] as usize));
            break;
        }
        out.sess_end.push(out.start.len() as u32);
    }
    sc.grouped = grouped;
    sc.bgrouped = bj;
    res
}
