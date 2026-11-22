//! P1: one user partition q, in cache. Reads the partition's survivors (I2), clicks (I3) and link
//! entries (I4) and writes one person block per user with events and one click block per user
//! with clicks (RECORDS-V4 section 8), each to its person's home partition.
//!
//! * Users: every user gets a dense id in insertion order and three 8-byte words under it (key
//!   and count, pk, click and purchase counts in pass 1; the three block cursors in pass 2), each
//!   in its own array so a pass touches one. SHORT users are found through a direct table of
//!   ids (4 bytes a slot) indexed by the top k bits of ru (ru is the low bits of mix40, already
//!   uniform: no hash), linear probing, id 0 = empty (its key is EMPTY; ru has at most 31 bits).
//!   LONG/FOREIGN users ("wide") through a small hash of (a, b, kind). Nothing per user is
//!   indexed by the sparse slot: a partition's working set is the table plus the dense words.
//! * Link entries are inserted first (they carry the person key pk; a representative's pk is its
//!   own key). An unlinked SHORT user's pk is its own mu = q << rbits | ru (home q); an unlinked
//!   wide user's pk is P1-local: 1 << 63 | kind << 61 | 1 << 60 | q << 32 | i, with [a, b] at
//!   `State::wide[i]` for P2's names (home q).
//! * Pass 1 per event: slot, count, the slot and the 4-byte event (ts << 1 | purch) stored in
//!   reading order; a purchase also stores its slot and payload. Clicks: slot by lookup only (a
//!   click of a user with no events here and no link entry can reach no session: dropped).
//! * Layout: per user its blocks are allocated in its home partition's output, headers written.
//! * Pass 2: events, payloads and clicks are scattered straight into their blocks. Inside a block
//!   events and payloads keep reading order, so payload k is the k-th purch event's (RECORDS-V4).
//!
//! Errors (the driver falls back to v3): a count past 2^24, 32-bit ru, a slot space too large.

use core::ptr::null_mut;

use super::dedup::Survivors;
use super::links::Links;
use super::sink::ClickBins;
use super::{Cx, Stage};

pub const EMPTY: u32 = u32::MAX;

/// Where P1's blocks go. `block(home, n)` hands out room for one block of `n` words in home
/// partition `home`; the memory must stay valid, unmoved and unaliased until the `run` call that
/// asked for it returns (P1 writes the bodies after all blocks of the partition are allocated).
pub trait Out {
    fn block(&mut self, home: u32, n: usize) -> *mut u64;
}

/// One user partition's inputs (RECORDS-V4 sections 4-6).
pub struct In<'a> {
    pub q: u32,
    /// Bits of ru: 40 - b2.
    pub rbits: u32,
    /// I2 SHORT survivors `ru << 32 | ts << 1 | purch`, the payload word after each purchase.
    pub surv: &'a [&'a [u64]],
    /// I2 wide survivors `[a, b, ts << 1 | purch | kind << 62]`, the payload word after a purchase.
    pub wsurv: &'a [&'a [u64]],
    /// I3 SHORT clicks `[ru << 32 | tc, campaign << 52 | loc]`.
    pub clicks: &'a [&'a [[u64; 2]]],
    /// I3 wide clicks `[a, b, tc | kind << 62, campaign << 52 | loc]`.
    pub wclicks: &'a [&'a [[u64; 4]]],
    /// I4: every linked SHORT user of q (representatives too) and its person key.
    pub link_ru: &'a [u32],
    pub link_pk: &'a [u64],
    /// I4: every linked wide user of q `[a, b, pk, kind]`.
    pub wlink: &'a [[u64; 4]],
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// ru of 32 bits (u32::MAX would be a real key).
    RuBits,
    /// More than 2^24 events, purchases or clicks for one user.
    Count,
    /// The slot space would pass 2^28.
    Size,
}

pub struct State {
    /// The direct table, 2^k entries: the user's id (home slot of ru = its top k bits); 0 = empty.
    tab: Vec<u32>,
    /// Per user id (dense, in insertion order; id 0 is the spare user of the clicks that hit no
    /// user, its key EMPTY), three 8-byte arrays, each the only one its pass touches at random:
    /// `ua`: key | cnt << 32 (a wide user's key is EMPTY), after the layout the event cursor;
    /// `ub`: the links' pk, after the layout the payload cursor;
    /// `uc`: nck | nbuy << 32, after the layout the click cursor.
    ua: Vec<u64>,
    ub: Vec<u64>,
    uc: Vec<u64>,
    /// The next free id.
    next: usize,
    nwide: u32,
    /// Wide users: hash table of [a, b, kind, id] (0 = empty), and per wide index
    /// [a, b, kind, linked].
    wt: Vec<[u64; 4]>,
    /// Entries of wtab in use (cleared by the next reset).
    wpos: Vec<u32>,
    wkey: Vec<[u64; 4]>,
    /// Per event (reading order): user id and the 4-byte event.
    idx: Vec<u32>,
    ev: Vec<u32>,
    /// Per purchase: user id and payload.
    bidx: Vec<u32>,
    bval: Vec<u64>,
    /// Per click (reading order, SHORT then wide): user id (0: no user here).
    cidx: Vec<u32>,
    /// Where pass 2 writes the clicks of no user here (they reach no session): the click cursor
    /// of user 0 points into it, so neither click loop tests for a miss.
    trash: Vec<[u64; 2]>,
    /// P1-local wide person keys of the last partition: [a, b] at index i of pk.
    pub wide: Vec<[u64; 2]>,
    /// Users too big for one chunk and their scratch (reused across partitions).
    bigs: Vec<Big>,
    big_ev: Vec<u32>,
    big_pay: Vec<u64>,
    big_ck: Vec<[u64; 2]>,
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
fn wide_hash(a: u64, b: u64, kind: u64, bits: u32) -> usize {
    // its own constant: not a partition hash (RECORDS-V4 trap (a))
    (((a ^ b.rotate_left(23) ^ kind).wrapping_mul(0x9e37_79b9_7f4a_7c15)) >> (64 - bits)) as usize
}

impl State {
    pub fn new() -> Self {
        State {
            tab: Vec::new(),
            ua: Vec::new(),
            ub: Vec::new(),
            uc: Vec::new(),
            next: 1,
            nwide: 0,
            wt: Vec::new(),
            wpos: Vec::new(),
            wkey: Vec::new(),
            idx: Vec::new(),
            ev: Vec::new(),
            bidx: Vec::new(),
            bval: Vec::new(),
            cidx: Vec::new(),
            trash: Vec::new(),
            wide: Vec::new(),
            bigs: Vec::new(),
            big_ev: Vec::new(),
            big_pay: Vec::new(),
            big_ck: Vec::new(),
        }
    }

    /// A clean direct table of 2^k slots, room for `cap` users, a wide table for `wide` wide users.
    fn reset(&mut self, k: u32, wide: usize, cap: usize) {
        for &h in &self.wpos {
            self.wt[h as usize] = [0; 4];
        }
        self.wpos.clear();
        self.nwide = 0;
        let ds = 1usize << k;
        // The whole direct table in one sequential fill (4 bytes a slot): no per-user clearing
        // of sparse arrays, and the table is in cache for the inserts that follow.
        self.tab.clear();
        self.tab.reserve(ds);
        // SAFETY: ds entries reserved; zero bytes are a valid u32.
        unsafe {
            core::ptr::write_bytes(self.tab.as_mut_ptr(), 0, ds);
            self.tab.set_len(ds);
        }
        if self.ua.len() < cap {
            self.ua.resize(cap, 0);
            self.ub.resize(cap, 0);
            self.uc.resize(cap, 0);
        }
        self.ua[0] = u64::from(EMPTY);
        self.ub[0] = 0;
        self.uc[0] = 0;
        self.next = 1;
        let wsize = (2 * wide).max(16).next_power_of_two();
        if self.wt.len() != wsize {
            self.wt.clear();
            self.wt.resize(wsize, [0; 4]);
        }
        self.wkey.clear();
        self.idx.clear();
        self.ev.clear();
        self.bidx.clear();
        self.bval.clear();
        self.cidx.clear();
        self.wide.clear();
    }

    /// Id of wide user (a, b, kind), inserted (pk 0: decided at layout) if new.
    #[inline(always)]
    fn wide_slot(&mut self, a: u64, b: u64, kind: u64) -> usize {
        let mask = self.wt.len() - 1;
        let bits = self.wt.len().trailing_zeros();
        let mut h = wide_hash(a, b, kind, bits);
        let wt = self.wt.as_mut_ptr();
        loop {
            // SAFETY: h is masked below wt.len().
            let e = unsafe { &mut *wt.add(h) };
            if e[3] == 0 {
                return self.wide_new(h, a, b, kind);
            }
            if e[0] == a && e[1] == b && e[2] == kind {
                return e[3] as usize;
            }
            h = (h + 1) & mask;
        }
    }

    #[cold]
    #[inline(never)]
    fn wide_new(&mut self, h: usize, a: u64, b: u64, kind: u64) -> usize {
        let id = self.next;
        self.next += 1;
        self.nwide += 1;
        self.wt[h] = [a, b, kind, id as u64];
        self.wpos.push(h as u32);
        self.wkey.push([a, b, kind, 0]);
        self.ua[id] = u64::from(EMPTY);
        self.ub[id] = 0;
        self.uc[id] = 0;
        id
    }
}

/// The hot loop's state: raw pointers into tables reserved for the whole pass (nothing
/// reallocates them during it). The first six fields are read by the asm at fixed offsets.
#[repr(C)]
struct Hot {
    /// The next free user id (advances).
    next: u64,      // 0
    /// A `next` past this after an insert: the direct table is over half full.
    lim: u64,       // 8
    /// uc: nck | nbuy << 32 per id.
    uc: *mut u64,   // 16
    mask: u64,      // 24
    /// The next purchase's payload and user id (advance).
    bval: *mut u64, // 32
    bidx: *mut u32, // 40
    tab: *mut u32,
    /// ua: key | cnt << 32 per id.
    ua: *mut u64,
    /// The next event's user id and 4-byte event (advance).
    idx: *mut u32,
    ev: *mut u32,
    /// 32 + shift: a record's home slot is its word >> sh (the top k bits of ru).
    sh: u64,
}

/// How `hot` stopped.
#[derive(PartialEq, Eq)]
enum Stop {
    /// The piece's end.
    End,
    /// The piece's last word was a purchase of this user (counted): its payload starts the next piece.
    Pend(u32),
    /// The direct table passed half full (the pass is retried larger).
    Full,
    /// ru = EMPTY (not a valid remainder).
    BadRu,
}

/// SHORT survivors of one piece from word i: per record its user (home slot = the top k bits of ru,
/// then linear probing; a new user gets the next id), its count, the id and the 4-byte event stored;
/// a purchase counts in nbuy and its payload (the next word) goes to bval/bidx. One asm loop: 13
/// instructions per record that hits its home slot (load, rorx, shrx, the id, cmp, the count, two
/// stores, the purchase test, the counter). An empty slot holds id 0, whose key (EMPTY) is no ru.
///
/// # Safety
/// The Hot pointers have room for every record of the pass (idx, ev, bval, bidx: one per word;
/// ua, uc: ids up to lim); tab covers every slot (ru < 2^rbits).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hot(c: &[u64], i: usize, h: &mut Hot) -> Stop {
    let n = (c.len() - i) as isize;
    if n == 0 {
        return Stop::End;
    }
    let (mut t, stat, s): (isize, u64, u64);
    // SAFETY: the caller's contract; the asm writes only through the Hot pointers.
    unsafe {
        let inp = c.as_ptr().add(c.len());
        let mut idx = h.idx.offset(n);
        let mut ev = h.ev.offset(n);
        t = -n;
        std::arch::asm!(
            "2:",
            "mov {r}, qword ptr [{inp} + 8*{t}]",
            "rorx {ru}, {r}, 32",
            "shrx {s}, {r}, {sh}",
            "mov {id:e}, dword ptr [{tab} + 4*{s}]",
            "cmp {ru:e}, dword ptr [{ua} + 8*{id}]",
            "jne 5f",
            "3:",
            "inc dword ptr [{ua} + 8*{id} + 4]",
            "mov dword ptr [{idx} + 4*{t}], {id:e}",
            "mov dword ptr [{ev} + 4*{t}], {r:e}",
            "test {r:l}, 1",
            "jnz 6f",
            "inc {t}",
            "jnz 2b",
            "xor {id:e}, {id:e}",
            "jmp 9f",
            // not at its home slot: probe on (moved by an earlier user), or insert at an empty slot
            "5:",
            "test {id:e}, {id:e}",
            "jz 7f",
            "4:",
            "inc {s}",
            "and {s}, qword ptr [{ctx} + 24]",
            "mov {id:e}, dword ptr [{tab} + 4*{s}]",
            "cmp {ru:e}, dword ptr [{ua} + 8*{id}]",
            "je 3b",
            "test {id:e}, {id:e}",
            "jnz 4b",
            // a new user: the next id into the slot, its key with count 0, no clicks or purchases
            "7:",
            "cmp {ru:e}, -1",
            "je 8f",
            "mov {id}, qword ptr [{ctx}]",
            "mov dword ptr [{tab} + 4*{s}], {id:e}",
            "mov {ru:e}, {ru:e}",
            "mov qword ptr [{ua} + 8*{id}], {ru}",
            "mov {s}, qword ptr [{ctx} + 16]",
            "mov qword ptr [{s} + 8*{id}], 0",
            "lea {s}, [{id} + 1]",
            "mov qword ptr [{ctx}], {s}",
            "cmp {s}, qword ptr [{ctx} + 8]",
            "jbe 3b",
            "mov {id:e}, 2",
            "jmp 9f",
            "8:",
            "mov {id:e}, 3",
            "jmp 9f",
            // a purchase of user id: count it, then its payload word (or it starts the next piece)
            "6:",
            "mov {s}, qword ptr [{ctx} + 16]",
            "inc dword ptr [{s} + 8*{id} + 4]",
            "inc {t}",
            "jz 22f",
            "mov {r}, qword ptr [{inp} + 8*{t}]",
            "mov {s}, qword ptr [{ctx} + 32]",
            "mov qword ptr [{s}], {r}",
            "add qword ptr [{ctx} + 32], 8",
            "mov {s}, qword ptr [{ctx} + 40]",
            "mov dword ptr [{s}], {id:e}",
            "add qword ptr [{ctx} + 40], 4",
            "sub {idx}, 4",
            "sub {ev}, 4",
            "inc {t}",
            "jnz 2b",
            "xor {id:e}, {id:e}",
            "jmp 9f",
            "22:",
            "mov {s}, {id}",
            "mov {id:e}, 1",
            "9:",
            inp = in(reg) inp,
            t = inout(reg) t,
            idx = inout(reg) idx,
            ev = inout(reg) ev,
            tab = in(reg) h.tab,
            ua = in(reg) h.ua,
            sh = in(reg) h.sh,
            ctx = in(reg) h as *mut Hot,
            r = out(reg) _,
            ru = out(reg) _,
            s = out(reg) s,
            id = out(reg) stat,
            options(nostack),
        );
        h.idx = idx.offset(t);
        h.ev = ev.offset(t);
    }
    match stat {
        0 => Stop::End,
        1 => Stop::Pend(s as u32),
        2 => Stop::Full,
        _ => Stop::BadRu,
    }
}

/// The same contract in plain Rust (other targets).
#[cfg(not(target_arch = "x86_64"))]
unsafe fn hot(c: &[u64], mut i: usize, h: &mut Hot) -> Stop {
    // SAFETY: the caller's contract.
    unsafe {
        while i < c.len() {
            let r = c[i];
            let ru = (r >> 32) as u32;
            let mut s = (r >> h.sh) as usize;
            let id = loop {
                let id = *h.tab.add(s) as usize;
                if *h.ua.add(id) as u32 == ru {
                    break id;
                }
                if id == 0 {
                    if ru == EMPTY {
                        return Stop::BadRu;
                    }
                    let id = h.next as usize;
                    *h.tab.add(s) = id as u32;
                    *h.ua.add(id) = u64::from(ru);
                    *h.uc.add(id) = 0;
                    h.next += 1;
                    if h.next > h.lim {
                        return Stop::Full;
                    }
                    break id;
                }
                s = (s + 1) & h.mask as usize;
            };
            *h.ua.add(id) += 1 << 32;
            *h.idx = id as u32;
            *h.ev = r as u32;
            h.idx = h.idx.add(1);
            h.ev = h.ev.add(1);
            i += 1;
            if r & 1 != 0 {
                *h.uc.add(id) += 1 << 32;
                if i == c.len() {
                    return Stop::Pend(id as u32);
                }
                *h.bval = c[i];
                *h.bidx = id as u32;
                h.bval = h.bval.add(1);
                h.bidx = h.bidx.add(1);
                i += 1;
            }
        }
    }
    Stop::End
}

/// The pass's wide hash table, hoisted: its entries, bits and mask, and the number of wide users
/// it may hold.
struct WideTab {
    wt: *mut [u64; 4],
    bits: u32,
    mask: usize,
    max: usize,
}

/// One wide record (user key a, b, kind; the 4-byte event e = ts << 1 | purch; the payload if
/// purch): its user (a new wide user is inserted), count, event and payload stored through h's
/// pointers, in the same reading order as the SHORT records. false: no wide user was left.
///
/// # Safety
/// h's pointers have room for this record; t is the current pass's table (st.wt, not resized).
#[inline(always)]
unsafe fn wrec(st: &mut State, h: &mut Hot, t: &WideTab, a: u64, b: u64, kind: u64, e: u32, pay: u64) -> bool {
    let mut x = wide_hash(a, b, kind, t.bits);
    // SAFETY: x is masked below wt.len(); h as the caller's contract.
    unsafe {
        let id = loop {
            let en = &*t.wt.add(x);
            if en[0] == a && en[1] == b && en[2] == kind {
                break en[3] as usize;
            }
            if en[3] == 0 {
                if st.nwide as usize >= t.max {
                    return false;
                }
                break st.wide_new(x, a, b, kind);
            }
            x = (x + 1) & t.mask;
        };
        *(h.ua.add(id) as *mut u32).add(1) += 1;
        *h.idx = id as u32;
        *h.ev = e;
        h.idx = h.idx.add(1);
        h.ev = h.ev.add(1);
        if e & 1 != 0 {
            *(h.uc.add(id) as *mut u32).add(1) += 1;
            *h.bval = pay;
            *h.bidx = id as u32;
            h.bval = h.bval.add(1);
            h.bidx = h.bidx.add(1);
        }
    }
    true
}

/// The wide stream of one piece from word *i on, whole records only (`*i` ends at the first one cut by
/// the piece end): `wrec` with the Hot pointers and the table held in locals for the whole piece.
/// false: stopped at a user not in the table yet (the record at *i, whole).
///
/// # Safety
/// As `wrec`; *i <= c.len().
#[inline(always)]
unsafe fn wfast(c: &[u64], i: &mut usize, h: &mut Hot, t: &WideTab) -> bool {
    let n = c.len();
    let cp = c.as_ptr();
    let (wt, bits, mask) = (t.wt, t.bits, t.mask);
    let (ua, uc) = (h.ua, h.uc);
    let (mut idxp, mut evp, mut bvp, mut bip) = (h.idx, h.ev, h.bval, h.bidx);
    let mut j = *i;
    let mut miss = false;
    // SAFETY: the caller's contract; j + 4 <= n for every word read below.
    unsafe {
        while j + 3 <= n {
            let w = *cp.add(j + 2);
            let p = (w & 1) as usize;
            if j + 3 + p > n {
                break;
            }
            let (a, b) = (*cp.add(j), *cp.add(j + 1));
            let kind = w >> 62;
            let mut x = wide_hash(a, b, kind, bits);
            // a miss leaves the loop (no call inside it: nothing spills); the caller inserts
            let id = loop {
                let en = &*wt.add(x);
                if en[0] == a && en[1] == b && en[2] == kind {
                    break en[3] as usize;
                }
                if en[3] == 0 {
                    miss = true;
                    break 0;
                }
                x = (x + 1) & mask;
            };
            if miss {
                break;
            }
            *(ua.add(id) as *mut u32).add(1) += 1;
            *idxp = id as u32;
            *evp = w as u32;
            idxp = idxp.add(1);
            evp = evp.add(1);
            if p != 0 {
                *(uc.add(id) as *mut u32).add(1) += 1;
                *bvp = *cp.add(j + 3);
                *bip = id as u32;
                bvp = bvp.add(1);
                bip = bip.add(1);
            }
            j += 3 + p;
        }
    }
    (h.idx, h.ev, h.bval, h.bidx) = (idxp, evp, bvp, bip);
    *i = j;
    !miss
}

/// P1 for one user partition: blocks into `out`.
pub fn part<O: Out>(inp: &In, st: &mut State, out: &mut O) -> Result<(), Error> {
    if inp.rbits >= 32 || inp.rbits < 8 {
        return Err(Error::RuBits);
    }
    let n_surv: usize = inp.surv.iter().map(|c| c.len()).sum();
    let n_wsurv: usize = inp.wsurv.iter().map(|c| c.len()).sum();
    // users: the linked ones plus about one per 8 survivor words (about 13 events a user at
    // p20v5); load at most 1/2, or the pass runs again with a table twice the size
    let est = inp.link_ru.len() + n_surv / 8 + 64;
    let mut k = (2 * est).next_power_of_two().trailing_zeros().clamp(8, 22).min(inp.rbits);
    // wide slots: one per wide USER (clicks only look up). Linked ones are known; the others are
    // estimated at 1 per 24 words of the wide stream (about 13 events a user; p20v5 has about 88
    // wide users a partition), and a pass that runs out runs again with twice as many.
    let mut wide_max = inp.wlink.len() + n_wsurv / 24 + 64;
    loop {
        if (1usize << 22) + wide_max > 1 << 28 {
            return Err(Error::Size);
        }
        match pass(inp, st, out, k, wide_max)? {
            Pass::Done => return Ok(()),
            Pass::Direct => k += 1, // the direct table passed load 1/2: again, larger
            Pass::Wide => wide_max *= 2,
        }
        if k > 24 || k > inp.rbits {
            return Err(Error::Size);
        }
    }
}

/// How a pass ended.
enum Pass {
    Done,
    /// The direct table passed half full.
    Direct,
    /// More wide users than wide slots.
    Wide,
}

/// One attempt with 2^k direct slots and wide_max wide users.
fn pass<O: Out>(inp: &In, st: &mut State, out: &mut O, k: u32, wide_max: usize) -> Result<Pass, Error> {
    let shift = inp.rbits - k;
    let limit = 1usize << (k - 1);
    let mask = (1usize << k) - 1;
    // more link entries than slots: the insert loop below must always find an empty slot
    if inp.link_ru.len() > mask {
        return Ok(Pass::Direct);
    }
    // ids: the spare one, the SHORT users (the links', or half the table and the one that
    // passes it), the wide users, one spare
    let cap = limit.max(inp.link_ru.len()) + wide_max + 3;
    if cap >= 1 << 27 {
        return Err(Error::Size);
    }
    st.reset(k, wide_max, cap);
    let q = inp.q;
    let rbits = inp.rbits;
    let own = u64::from(q) << rbits;
    let tab = st.tab.as_mut_ptr();
    let (ua, ub, uc) = (st.ua.as_mut_ptr(), st.ub.as_mut_ptr(), st.uc.as_mut_ptr());
    // links first: their person keys (a representative's is its own key)
    let mut next = 1usize;
    for (&ru, &pk) in inp.link_ru.iter().zip(inp.link_pk) {
        if ru == EMPTY {
            return Err(Error::RuBits);
        }
        let mut s = (ru >> shift) as usize & mask;
        // SAFETY: s is masked below 2^k = tab.len(); at most link_ru.len() ids are taken
        // (cap covers them) and the table has an empty slot (link_ru.len() <= mask).
        unsafe {
            loop {
                let id = *tab.add(s) as usize;
                if id == 0 {
                    *tab.add(s) = next as u32;
                    *ua.add(next) = u64::from(ru);
                    *ub.add(next) = pk;
                    *uc.add(next) = 0;
                    next += 1;
                    break;
                }
                if *ua.add(id) as u32 == ru {
                    *ub.add(id) = pk;
                    break;
                }
                s = (s + 1) & mask;
            }
        }
    }
    st.next = next;
    // ids [1, a_end): the links' SHORT users; [a_end, b_end): the links' wide users
    let a_end = next;
    if a_end - 1 > limit {
        return Ok(Pass::Direct);
    }
    for w in inp.wlink {
        let id = st.wide_slot(w[0], w[1], w[3]);
        st.ub[id] = w[2];
        st.wkey[id - a_end][3] = 1;
    }
    let b_end = st.next;
    // pass 1, SHORT survivors (the hot loop, `hot`): user, count, the id and the event stored
    let n_surv: usize = inp.surv.iter().map(|c| c.len()).sum();
    let n_wsurv: usize = inp.wsurv.iter().map(|c| c.len()).sum();
    // one entry per SHORT word, one per wide-stream record (3 words or more)
    let n_rec = n_surv + n_wsurv / 3 + 1;
    st.idx.reserve(n_rec);
    st.ev.reserve(n_rec);
    st.bval.reserve(n_rec);
    st.bidx.reserve(n_rec);
    let mut h = Hot {
        next: b_end as u64,
        // the SHORT users are the a_end - 1 of the links and those taken from b_end on: more
        // than `limit` of them is a table over half full
        lim: (b_end + limit + 1 - a_end) as u64,
        uc,
        mask: mask as u64,
        bval: st.bval.as_mut_ptr(),
        bidx: st.bidx.as_mut_ptr(),
        tab,
        ua,
        idx: st.idx.as_mut_ptr(),
        ev: st.ev.as_mut_ptr(),
        sh: u64::from(32 + shift),
    };
    let mut pend = EMPTY; // a purchase whose payload word starts the next piece
    let mut why = Stop::End;
    let wt = WideTab { wt: st.wt.as_mut_ptr(), bits: st.wt.len().trailing_zeros(), mask: st.wt.len() - 1, max: wide_max };
    let mut wide_full = false;
    for c in inp.surv {
        let mut i = 0usize;
        if pend != EMPTY && !c.is_empty() {
            // SAFETY: bval/bidx have room for one entry per survivor word.
            unsafe {
                *h.bval = c[0];
                *h.bidx = pend;
                h.bval = h.bval.add(1);
                h.bidx = h.bidx.add(1);
            }
            pend = EMPTY;
            i = 1;
        }
        // SAFETY: h points into tables sized by reset and reserved above.
        why = unsafe { hot(c, i, &mut h) };
        match why {
            Stop::End => {}
            Stop::Pend(s) => pend = s,
            _ => break,
        }
    }
    // ids [b_end, c_end): the SHORT users first seen in the survivors; [c_end, d_end): wide
    st.next = h.next as usize;
    let c_end = st.next;
    // wide survivors: [a, b, word] (+ the payload word after a purchase),
    // read in place; a record cut by a piece end is assembled in `carry`
    let mut carry = [0u64; 4];
    let mut nc = 0usize; // words in carry
    if matches!(why, Stop::End | Stop::Pend(_)) {
        'wpieces: for c in inp.wsurv {
            let mut i = 0usize;
            let n = c.len();
            // finish a record begun in the last piece
            while nc > 0 && i < n {
                carry[nc] = c[i];
                nc += 1;
                i += 1;
                if nc == 4 || (nc == 3 && carry[2] & 1 == 0) {
                    nc = 0;
                    // SAFETY: as above.
                    if !unsafe { wrec(st, &mut h, &wt, carry[0], carry[1], carry[2] >> 62, carry[2] as u32, carry[3]) } {
                        wide_full = true;
                        break 'wpieces;
                    }
                }
            }
            // SAFETY: as above; i <= n.
            while !unsafe { wfast(c, &mut i, &mut h, &wt) } {
                // a new wide user: this record through the inserting path
                let w = c[i + 2];
                let len = 3 + (w & 1) as usize;
                let pay = if len == 4 { c[i + 3] } else { 0 };
                // SAFETY: as above.
                if !unsafe { wrec(st, &mut h, &wt, c[i], c[i + 1], w >> 62, w as u32, pay) } {
                    wide_full = true;
                    break 'wpieces;
                }
                i += len;
            }
            while i < n {
                carry[nc] = c[i];
                nc += 1;
                i += 1;
            }
        }
    }
    let d_end = st.next;
    // SAFETY: the entries before each advanced pointer were written by `hot` or `wrec`.
    unsafe {
        st.idx.set_len(h.idx.offset_from(st.idx.as_ptr()) as usize);
        st.ev.set_len(h.ev.offset_from(st.ev.as_ptr()) as usize);
        st.bval.set_len(h.bval.offset_from(st.bval.as_ptr()) as usize);
        st.bidx.set_len(h.bidx.offset_from(st.bidx.as_ptr()) as usize);
    }
    if wide_full {
        return Ok(Pass::Wide);
    }
    match why {
        Stop::Full => return Ok(Pass::Direct),
        Stop::BadRu => return Err(Error::RuBits),
        _ => {}
    }
    // an event counted on the spare user: its ru was EMPTY (the spare key), not a valid remainder
    if st.ua[0] >> 32 != 0 {
        return Err(Error::RuBits);
    }
    // clicks: lookup only. A miss takes user 0 (counted there, never laid out), so the loops
    // have no miss branch on the store: pass 2 writes such a click to the trash.
    let n_ck: usize = inp.clicks.iter().map(|c| c.len()).sum::<usize>() + inp.wclicks.iter().map(|c| c.len()).sum::<usize>();
    st.cidx.reserve(n_ck);
    {
        let cidx = st.cidx.as_mut_ptr();
        let mut j = 0usize;
        for c in inp.clicks {
            let cp = c.as_ptr();
            for i in 0..c.len() {
                // SAFETY: i < c.len(); slots are masked below 2^k = tab.len(); j + i < n_ck =
                // cidx's capacity; an id from the table is a user's.
                unsafe {
                    let ru = ((*cp.add(i))[0] >> 32) as u32;
                    let mut s = (ru >> shift) as usize;
                    let mut id = *tab.add(s) as usize;
                    while *ua.add(id) as u32 != ru {
                        if id == 0 {
                            break;
                        }
                        s = (s + 1) & mask;
                        id = *tab.add(s) as usize;
                    }
                    *(uc.add(id) as *mut u32) += 1;
                    *cidx.add(j + i) = id as u32;
                }
            }
            j += c.len();
        }
        for c in inp.wclicks {
            for r in c.iter() {
                let (a, b, kind) = (r[0], r[1], r[2] >> 62);
                let mut x = wide_hash(a, b, kind, wt.bits);
                // SAFETY: x is masked below the table's length.
                let id = loop {
                    let en = unsafe { &*wt.wt.add(x) };
                    if en[0] == a && en[1] == b && en[2] == kind {
                        break en[3] as usize;
                    }
                    if en[3] == 0 {
                        break 0;
                    }
                    x = (x + 1) & wt.mask;
                };
                // SAFETY: as above.
                unsafe {
                    *(uc.add(id) as *mut u32) += 1;
                    *cidx.add(j) = id as u32;
                }
                j += 1;
            }
        }
        // SAFETY: entries 0..j written above.
        unsafe { st.cidx.set_len(j) };
    }
    // the clicks of no user here: counted on user 0
    let nmiss = st.uc[0] as u32 as usize;
    // layout: blocks in the homes, headers, cursors. The users are walked in id order (dense).
    // SHORT users, the common case: one allocation holds the person block and the click block
    // back to back. Anything unusual (a count past 2^24, a home out of range, blocks over one
    // chunk) and every wide user go to `layout_one`, the general path.
    let nb_homes = 40 - rbits;
    for (lo, hi, linked) in [(1, a_end, true), (b_end, c_end, false)] {
        for id in lo..hi {
            // SAFETY: id < hi <= the ids taken; a block has the words asked for.
            unsafe {
                let a = *ua.add(id);
                let c = *uc.add(id);
                let (ne, nc, nb) = ((a >> 32) as usize, c as u32 as usize, (c >> 32) as usize);
                if ne | nc != 0 {
                    // a link's pk, or the user's own key
                    let pk = if linked { *ub.add(id) } else { own | u64::from(a as u32) };
                    // SHORT: the top b2 bits of mu; wide (links'): bits 32-44 (RECORDS-V4 7)
                    let home = if pk >> 63 == 0 { pk >> rbits } else { (pk >> 32) & 0x1FFF } as u32;
                    let ew = ne.div_ceil(2);
                    let pw = if ne != 0 { 2 + ew + nb } else { 0 };
                    let cw = if nc != 0 { 2 + 2 * nc } else { 0 };
                    if pw + cw > CW - 2 || home >> nb_homes != 0 {
                        layout_one(st, out, id, pk, usize::MAX, q, rbits)?;
                    } else {
                        let p = out.block(home, pw + cw);
                        // every line of the two blocks is wanted for writing by pass 2: asked
                        // for here, it arrives while the layout goes on (a store that misses in
                        // pass 2 waits in the store buffer instead)
                        want(p, pw + cw);
                        let wide = pk >> 63;
                        if ne != 0 {
                            *p = ne as u64 | (nb as u64) << 24 | wide << 63;
                            *p.add(1) = pk;
                            // the odd event's pad (for an even count pass 2 overwrites both halves)
                            *p.add(1 + ew) = 0xFFFF_FFFF_0000_0000;
                            *ua.add(id) = p.add(2) as u64;
                            *ub.add(id) = p.add(2 + ew) as u64;
                        }
                        if nc != 0 {
                            let cb = p.add(pw);
                            *cb = nc as u64 | 1 << 62 | wide << 63;
                            *cb.add(1) = pk;
                            *uc.add(id) = cb.add(2) as u64;
                        }
                    }
                }
            }
        }
    }
    // the wide users, in the order of their index (wkey): the links', then the survivors'
    let nwl = b_end - a_end;
    for j in 0..nwl {
        let pk = st.ub[a_end + j];
        layout_one(st, out, a_end + j, pk, j, q, rbits)?;
    }
    for j in 0..d_end - c_end {
        layout_one(st, out, c_end + j, 0, nwl + j, q, rbits)?;
    }
    // the big users' cursors, now that their scratch has its size
    for b in &st.bigs {
        let id = b.su as usize;
        // SAFETY: the offsets lie inside the scratch vectors, sized above; they do not move
        // until `split` has read them.
        unsafe {
            if b.clicks {
                st.uc[id] = st.big_ck.as_mut_ptr().add(b.at as usize) as u64;
            } else {
                st.ua[id] = st.big_ev.as_mut_ptr().add(b.at as usize) as u64;
                st.ub[id] = st.big_pay.as_mut_ptr().add(b.pat as usize) as u64;
            }
        }
    }
    // the misses' clicks: the trash, through user 0's click cursor
    if st.trash.len() < nmiss {
        st.trash.resize(nmiss, [0, 0]);
    }
    st.uc[0] = st.trash.as_mut_ptr() as u64;
    // pass 2: scatter
    {
        let (ua, ub, uc) = (st.ua.as_mut_ptr(), st.ub.as_mut_ptr(), st.uc.as_mut_ptr());
        for (&s, &e) in st.idx.iter().zip(&st.ev) {
            // SAFETY: every user with events got a block with room for exactly its events.
            unsafe {
                let c = ua.add(s as usize) as *mut *mut u32;
                **c = e;
                *c = (*c).add(1);
            }
        }
        for (&s, &v) in st.bidx.iter().zip(&st.bval) {
            // SAFETY: as above, room for its purchases.
            unsafe {
                let c = ub.add(s as usize) as *mut *mut u64;
                **c = v;
                *c = (*c).add(1);
            }
        }
        let cidx = st.cidx.as_ptr();
        let mut j = 0usize;
        for c in inp.clicks {
            let cp = c.as_ptr();
            for i in 0..c.len() {
                // SAFETY: j + i < cidx.len() (one entry per click, both forms); every user's
                // cursor has room for its clicks (user 0's: the trash, nmiss pairs).
                unsafe {
                    let r = *cp.add(i);
                    let cs = uc.add(*cidx.add(j + i) as usize) as *mut *mut [u64; 2];
                    let p = *cs;
                    *p = [r[1], r[0] & 0xffff_ffff];
                    *cs = p.add(1);
                }
            }
            j += c.len();
        }
        for c in inp.wclicks {
            for r in c.iter() {
                // SAFETY: as above.
                unsafe {
                    let cs = uc.add(*cidx.add(j) as usize) as *mut *mut [u64; 2];
                    let p = *cs;
                    *p = [r[3], r[2] & 0xffff_ffff];
                    *cs = p.add(1);
                }
                j += 1;
            }
        }
    }
    split(st, out);
    Ok(Pass::Done)
}

/// Asks for the cache lines of the n words at p, for writing.
#[inline(always)]
fn want(p: *mut u64, n: usize) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch reads and writes nothing (prefetchw is a no-op where unsupported).
    unsafe {
        let end = p as usize + 8 * n;
        let mut l = p as usize & !63;
        while l < end {
            core::arch::asm!("prefetchw byte ptr [{0}]", in(reg) l, options(nostack, preserves_flags, readonly));
            l += 64;
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (p, n);
}

/// The general layout of user `id` with person key `pk` (the cold path; `widx` = its wide index,
/// usize::MAX for a SHORT user): a P1-local key for an unlinked wide user, the count bounds,
/// separate blocks, big users to the scratch for `split`.
#[cold]
#[inline(never)]
fn layout_one<O: Out>(st: &mut State, out: &mut O, id: usize, pk: u64, widx: usize, q: u32, rbits: u32) -> Result<(), Error> {
    let lim24 = 1u32 << 24;
    let (a, c) = (st.ua[id], st.uc[id]);
    let (ne, nb, nc) = ((a >> 32) as u32, (c >> 32) as u32, c as u32);
    if ne == 0 && nc == 0 {
        return Ok(());
    }
    if ne >= lim24 || nb >= lim24 || nc >= lim24 {
        return Err(Error::Count);
    }
    let mut pk = pk;
    if widx != usize::MAX && st.wkey[widx][3] == 0 {
        // an unlinked wide user: a P1-local person key, its [a, b] for P2's names
        let i = st.wide.len() as u64;
        let w = st.wkey[widx];
        st.wide.push([w[0], w[1]]);
        pk = 1 << 63 | w[2] << 61 | 1 << 60 | u64::from(q) << 32 | i;
    }
    // SHORT: the top b2 bits of mu; wide (links' or P1-local): bits 32-44 (RECORDS-V4 7)
    let home = if pk >> 63 == 0 { (pk >> rbits) as u32 } else { (pk >> 32) as u32 & 0x1FFF };
    if home >> (40 - rbits) != 0 {
        return Err(Error::Size);
    }
    let wide = pk >> 63;
    let ew = (ne as usize).div_ceil(2);
    if 2 + ew + nb as usize > CW - 2 {
        // too big for one chunk: its events and payloads into the worker's scratch, cut into
        // several blocks after pass 2 (`split`); no region memory per block
        st.bigs.push(Big { su: id as u32, home, pk, at: st.big_ev.len() as u32, n: ne, pat: st.big_pay.len() as u32, nb, clicks: false });
        st.big_ev.resize(st.big_ev.len() + ne as usize, 0);
        st.big_pay.resize(st.big_pay.len() + nb as usize, 0);
    } else if ne > 0 {
        let p = out.block(home, 2 + ew + nb as usize);
        // SAFETY: the block has 2 + ew + nb words (Out contract).
        unsafe {
            *p = u64::from(ne) | u64::from(nb) << 24 | wide << 63;
            *p.add(1) = pk;
            if ne & 1 != 0 {
                *p.add(1 + ew) = 0xFFFF_FFFF_0000_0000;
            }
            st.ua[id] = p.add(2) as u64;
            st.ub[id] = p.add(2 + ew) as u64;
        }
    }
    if 2 + 2 * nc as usize > CW - 2 {
        st.bigs.push(Big { su: id as u32, home, pk, at: st.big_ck.len() as u32, n: nc, pat: 0, nb: 0, clicks: true });
        st.big_ck.resize(st.big_ck.len() + nc as usize, [0, 0]);
    } else if nc > 0 {
        let p = out.block(home, 2 + 2 * nc as usize);
        // SAFETY: as above, 2 + 2 nc words.
        unsafe {
            *p = u64::from(nc) | 1 << 62 | wide << 63;
            *p.add(1) = pk;
            st.uc[id] = p.add(2) as u64;
        }
    }
    Ok(())
}

/// A user (id `su`) whose person block or click block would not fit a chunk: its events
/// (and payloads) or clicks in the worker's scratch from offset `at` (`pat`), `n` of them.
#[derive(Clone, Copy)]
struct Big {
    su: u32,
    home: u32,
    pk: u64,
    at: u32,
    n: u32,
    pat: u32,
    nb: u32,
    clicks: bool,
}

/// Events per cut person block (2 + 340 + 680 payloads at most = 1,022 words) and clicks per cut
/// click block (2 + 1,020 words).
const CUT_EV: usize = 680;
const CUT_CK: usize = 510;

/// Writes the big users' data as several blocks each (RECORDS-V4 8: a person may arrive as
/// several blocks); payload k of a cut block stays the k-th purch event's of that block.
fn split<O: Out>(st: &mut State, out: &mut O) {
    for b in &st.bigs {
        let wide = b.pk >> 63;
        if b.clicks {
            let ck = &st.big_ck[b.at as usize..(b.at + b.n) as usize];
            for g in ck.chunks(CUT_CK) {
                let p = out.block(b.home, 2 + 2 * g.len());
                // SAFETY: the block has 2 + 2 g.len() words.
                unsafe {
                    *p = g.len() as u64 | 1 << 62 | wide << 63;
                    *p.add(1) = b.pk;
                    core::ptr::copy_nonoverlapping(g.as_ptr(), p.add(2) as *mut [u64; 2], g.len());
                }
            }
        } else {
            let ev = &st.big_ev[b.at as usize..(b.at + b.n) as usize];
            let pay = &st.big_pay[b.pat as usize..(b.pat + b.nb) as usize];
            let mut pi = 0usize;
            for g in ev.chunks(CUT_EV) {
                let nbg = g.iter().filter(|&&e| e & 1 != 0).count();
                let ew = g.len().div_ceil(2);
                let p = out.block(b.home, 2 + ew + nbg);
                // SAFETY: the block has 2 + ew + nbg words.
                unsafe {
                    *p = g.len() as u64 | (nbg as u64) << 24 | wide << 63;
                    *p.add(1) = b.pk;
                    if g.len() & 1 != 0 {
                        *p.add(1 + ew) = 0xFFFF_FFFF_0000_0000;
                    }
                    core::ptr::copy_nonoverlapping(g.as_ptr(), p.add(2) as *mut u32, g.len());
                    core::ptr::copy_nonoverlapping(pay[pi..pi + nbg].as_ptr(), p.add(2 + ew), nbg);
                }
                pi += nbg;
            }
        }
    }
    st.bigs.clear();
    st.big_ev.clear();
    st.big_pay.clear();
    st.big_ck.clear();
}

/// One worker's blocks: per home partition a chain of `v3::mem::B8K` chunks (1,024 words), whole
/// blocks back to back; a chunk not filled to its end has a zero word after its last block (P2
/// stops a chunk at h0 = 0; no real header is 0). A block over 1,022 words takes an exact-size
/// region `Buf` of its own. Chunks never move, so a block's address is stable to the end of the
/// run (S's memory rules, 8a5e114). Drop returns every chunk and Buf.
pub struct Chains {
    w: usize,
    /// Per home: the write position and the end of its current chunk (null, null: none yet), so
    /// a take is a compare and an add; the chunks themselves in `homes`.
    heads: Vec<(*mut u64, *mut u64)>,
    /// Per home: its chunks (the last one current).
    homes: Vec<Vec<*mut u64>>,
    big: Vec<(u32, crate::v3::mem::Buf<u64>)>,
    /// P1-local wide person keys: (partition, its table).
    wide: Vec<(u32, Vec<[u64; 2]>)>,
}

const CW: usize = 1024;

// SAFETY: the chunks are owned by this value alone (taken from B8K, returned on drop).
unsafe impl Send for Chains {}
unsafe impl Sync for Chains {}

impl Chains {
    pub fn new(homes: usize, w: usize) -> Self {
        Chains {
            w,
            heads: vec![(null_mut(), null_mut()); homes],
            homes: (0..homes).map(|_| Vec::new()).collect(),
            big: Vec::new(),
            wide: Vec::new(),
        }
    }

    #[inline(always)]
    fn take(&mut self, home: u32, n: usize) -> *mut u64 {
        // SAFETY: home < homes (the layout checks it); cur and end bound the current chunk.
        unsafe {
            let h = &mut *self.heads.as_mut_ptr().add(home as usize);
            if (h.1 as usize).wrapping_sub(h.0 as usize) >= 8 * n && !h.0.is_null() {
                let p = h.0;
                h.0 = p.add(n);
                return p;
            }
        }
        self.take_cold(home, n)
    }

    /// The fill of home h's current chunk (CW when it has none).
    fn fill(&self, h: usize) -> usize {
        match self.homes[h].last() {
            Some(&c) => (self.heads[h].0 as usize - c as usize) / 8,
            None => CW,
        }
    }

    #[cold]
    #[inline(never)]
    fn take_cold(&mut self, home: u32, n: usize) -> *mut u64 {
        if n > CW - 2 {
            let mut b = crate::v3::mem::Buf::<u64>::take(&crate::v3::mem::REG, n);
            let p = b.as_mut_ptr();
            self.big.push((home, b));
            return p;
        }
        let h = home as usize;
        let (cur, end) = self.heads[h];
        if !cur.is_null() && cur < end {
            // SAFETY: cur < end, inside the current chunk: the end mark.
            unsafe { *cur = 0 };
        }
        // SAFETY: worker w is the only user of index w during the pool run (par::run's w).
        let c = unsafe { crate::v3::mem::B8K.alloc(self.w) } as *mut u64;
        self.homes[h].push(c);
        // SAFETY: the chunk has CW words.
        unsafe { self.heads[h] = (c.add(n), c.add(CW)) };
        c
    }
}

impl Drop for Chains {
    fn drop(&mut self) {
        for ch in &self.homes {
            for &c in ch {
                // SAFETY: every chunk came from B8K.alloc and nobody reads it after the drop.
                unsafe { crate::v3::mem::B8K.free(self.w, c as *mut u8) };
            }
        }
    }
}

/// `Out` over a worker's chains.
pub struct ChainOut<'c> {
    pub chains: &'c mut Chains,
}

impl Out for ChainOut<'_> {
    #[inline(always)]
    fn block(&mut self, home: u32, n: usize) -> *mut u64 {
        self.chains.take(home, n)
    }
}

/// Person and click blocks by home partition (RECORDS-V4 section 8).
#[derive(Default)]
pub struct Blocks {
    workers: Vec<Chains>,
    wide: Vec<Vec<[u64; 2]>>,
}

impl Blocks {
    /// The chunks of home partition `home`, each whole blocks back to back, ended by its last
    /// word or a zero header word (as P2's `group` reads them).
    pub fn chunks(&self, home: usize) -> Vec<&[u64]> {
        let mut v = Vec::new();
        for w in &self.workers {
            let ch = &w.homes[home];
            let fill = w.fill(home);
            for (i, &c) in ch.iter().enumerate() {
                let len = if i + 1 == ch.len() { fill } else { CW };
                // SAFETY: a chunk holds CW words, the first `len` written (blocks, end mark).
                v.push(unsafe { core::slice::from_raw_parts(c as *const u64, len) });
            }
            for (h, b) in &w.big {
                if *h as usize == home {
                    v.push(&b[..]);
                }
            }
        }
        v
    }

    /// P1-local wide person keys of partition q: [a, b] at index pk & (2^24 - 1).
    pub fn wide(&self, q: usize) -> &[[u64; 2]] {
        self.wide.get(q).map_or(&[], |v| &v[..])
    }
}

/// What P1 reads per user partition: the dedup output (I2), the clicks (I3) and the links (I4).
pub trait Inputs: Sync {
    fn surv(&self, q: usize) -> Vec<&[u64]>;
    fn wsurv(&self, q: usize) -> Vec<&[u64]>;
    fn clicks(&self, q: usize) -> Vec<&[[u64; 2]]>;
    fn wclicks(&self, q: usize) -> Vec<&[[u64; 4]]>;
    fn link_ru(&self, q: usize) -> &[u32];
    fn link_pk(&self, q: usize) -> &[u64];
    fn wlink(&self, q: usize) -> &[[u64; 4]];
    /// Partition q's input chunks are read for good: return them to worker w's block list, so
    /// the next partitions' blocks recycle them (S's rule 2).
    ///
    /// # Safety
    /// Called once per q, by the task that ran q, after its blocks are written.
    unsafe fn free_part(&self, _q: usize, _w: usize) {}
}

/// The click stage's output as P1 reads it (I3, per user partition, pieces in bin order).
pub trait ClickSrc: Sync {
    fn short(&self, q: usize) -> Vec<&[[u64; 2]]>;
    fn wide(&self, q: usize) -> Vec<&[[u64; 4]]>;
    /// As `Inputs::free_part`, for the click chunks of q.
    ///
    /// # Safety
    /// As `Inputs::free_part`.
    unsafe fn free_part(&self, _q: usize, _w: usize) {}
}

/// The links stage's output as P1 reads it (I4, per user partition; RECORDS-V4 section 6).
pub trait LinkSrc: Sync {
    fn ru(&self, q: usize) -> &[u32];
    fn pk(&self, q: usize) -> &[u64];
    /// [a, b, pk, kind] per linked wide user of q.
    fn wide(&self, q: usize) -> &[[u64; 4]];
    /// Home partition of a wide person key of the links stage (P1 reads it from the pk's bits
    /// 32-44 instead, RECORDS-V4 7).
    fn wide_home(&self, pk: u64) -> u32;
}

struct Src<'a, C: ClickSrc, L: LinkSrc> {
    sv: &'a Survivors,
    ck: &'a C,
    ln: &'a L,
}

impl<C: ClickSrc, L: LinkSrc> Inputs for Src<'_, C, L> {
    fn surv(&self, q: usize) -> Vec<&[u64]> {
        self.sv.short(q).collect()
    }
    fn wsurv(&self, q: usize) -> Vec<&[u64]> {
        self.sv.wide(q).collect()
    }
    fn clicks(&self, q: usize) -> Vec<&[[u64; 2]]> {
        self.ck.short(q)
    }
    fn wclicks(&self, q: usize) -> Vec<&[[u64; 4]]> {
        self.ck.wide(q)
    }
    fn link_ru(&self, q: usize) -> &[u32] {
        self.ln.ru(q)
    }
    fn link_pk(&self, q: usize) -> &[u64] {
        self.ln.pk(q)
    }
    fn wlink(&self, q: usize) -> &[[u64; 4]] {
        self.ln.wide(q)
    }
    unsafe fn free_part(&self, q: usize, w: usize) {
        // SAFETY: passed on (the caller's contract: q ran on this task, its inputs are done).
        unsafe {
            self.sv.free_part(q, w);
            self.ck.free_part(q, w);
        }
    }
}

/// P1 over every user partition (1 << b2), on the pool.
pub fn run_on<I: Inputs>(threads: usize, b2: u32, src: &I) -> Result<Blocks, String> {
    let parts = 1usize << b2;
    let rbits = 40 - b2;
    let res = crate::par::run(
        threads,
        parts,
        |w| (State::new(), Chains::new(parts, w), None::<Error>, w),
        |(st, ch, err, w), q| {
            if err.is_some() {
                return;
            }
            let (surv, wsurv, clicks, wclicks) = (src.surv(q), src.wsurv(q), src.clicks(q), src.wclicks(q));
            let inp = In {
                q: q as u32,
                rbits,
                surv: &surv,
                wsurv: &wsurv,
                clicks: &clicks,
                wclicks: &wclicks,
                link_ru: src.link_ru(q),
                link_pk: src.link_pk(q),
                wlink: src.wlink(q),
            };
            let mut out = ChainOut { chains: ch };
            match part(&inp, st, &mut out) {
                Ok(()) => {
                    if !st.wide.is_empty() {
                        ch.wide.push((q as u32, core::mem::take(&mut st.wide)));
                    }
                    // SAFETY: q ran here, its blocks are written, its inputs are not read again.
                    unsafe { src.free_part(q, *w) };
                }
                Err(e) => *err = Some(e),
            }
        },
    );
    let mut blocks = Blocks { workers: Vec::with_capacity(res.len()), wide: (0..parts).map(|_| Vec::new()).collect() };
    for (_, mut ch, err, _) in res {
        if let Some(e) = err {
            return Err(format!("P1: {e:?}"));
        }
        for (q, w) in ch.wide.drain(..) {
            blocks.wide[q as usize] = w;
        }
        blocks.workers.push(ch);
    }
    Ok(blocks)
}

/// The stage: person and click blocks of every user partition (RECORDS-V4 section 8).
pub fn run<C: ClickSrc, L: LinkSrc>(cx: &Cx, sv: &Survivors, ck: &C, links: &L) -> Stage<Blocks> {
    run_on(cx.plan.threads, cx.plan.b2, &Src { sv, ck, ln: links })
}

// Until the click and link stages land their accessors: empty inputs (never reached: their stub
// stages return Err first, so the run falls back before P1).
impl ClickSrc for ClickBins {
    /// [ru << 32 | tc, campaign << 52 | loc] (RECORDS-V4.md section 5), every sink's pieces.
    fn short(&self, q: usize) -> Vec<&[[u64; 2]]> {
        self.s.chunks(q).collect()
    }
    /// [a, b, tc | kind << 62, campaign << 52 | loc].
    fn wide(&self, q: usize) -> Vec<&[[u64; 4]]> {
        self.wide.chunks(q).collect()
    }
    unsafe fn free_part(&self, q: usize, w: usize) {
        // SAFETY: the caller's contract (q's clicks are read for good).
        unsafe { ClickBins::free_part(self, q, w) }
    }
}

impl LinkSrc for Links {
    fn ru(&self, q: usize) -> &[u32] {
        &self.ru[self.off[q] as usize..self.off[q + 1] as usize]
    }
    fn pk(&self, q: usize) -> &[u64] {
        &self.pk[self.off[q] as usize..self.off[q + 1] as usize]
    }
    fn wide(&self, q: usize) -> &[[u64; 4]] {
        &self.wide[self.wide_off[q] as usize..self.wide_off[q + 1] as usize]
    }
    fn wide_home(&self, pk: u64) -> u32 {
        self.wkeys[(pk & 0xffff_ffff) as usize][2] as u32
    }
}

// SAFETY: the raw cursors in State only point into blocks handed out during one `run` call and
// are never read after it; a State moves between threads only between calls.
unsafe impl Send for State {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct VecOut {
        homes: HashMap<u32, Vec<Box<[u64]>>>,
    }
    impl Out for VecOut {
        fn block(&mut self, home: u32, n: usize) -> *mut u64 {
            let v = self.homes.entry(home).or_default();
            v.push(vec![0x5a5a_5a5a_5a5a_5a5au64; n].into_boxed_slice());
            v.last_mut().unwrap().as_mut_ptr()
        }

    }

    fn rng(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// Blocks as comparable values: (home, words).
    /// Every block, by home, sorted. An allocation may hold several blocks back to back (the
    /// layout puts a user's person and click blocks in one): split by their headers.
    fn canon(homes: &HashMap<u32, Vec<Box<[u64]>>>) -> Vec<(u32, Vec<u64>)> {
        let mut v: Vec<(u32, Vec<u64>)> = Vec::new();
        for (&h, bs) in homes {
            for b in bs {
                let mut at = 0;
                while at < b.len() {
                    let h0 = b[at];
                    let len = if h0 >> 62 & 1 != 0 { 2 + 2 * (h0 & 0xFF_FFFF) as usize } else { 2 + ((h0 & 0xFF_FFFF) as usize).div_ceil(2) + (h0 >> 24 & 0xFF_FFFF) as usize };
                    assert!(at + len <= b.len(), "a block past its allocation");
                    v.push((h, b[at..at + len].to_vec()));
                    at += len;
                }
            }
        }
        v.sort();
        v
    }

    #[test]
    fn p1_matches_reference() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        for case in 0..300 {
            let rbits = [27u32, 29, 31][case % 3];
            let q = (rng(&mut seed) % 64) as u32;
            let own = u64::from(q) << rbits;
            // every tenth case one or two users only: their blocks are cut (`split`)
            let nu = if case % 10 == 0 { 1 + case % 2 } else { 1 + (rng(&mut seed) % 400) as usize };
            let users: Vec<u32> = (0..nu).map(|_| (rng(&mut seed) % (1u64 << rbits)) as u32).collect();
            let nw = (rng(&mut seed) % 20) as usize;
            let wusers: Vec<[u64; 3]> = (0..nw).map(|_| [rng(&mut seed), rng(&mut seed) % 4, 1 + rng(&mut seed) % 2]).collect();
            // links: some users linked to random person keys (some SHORT pks elsewhere, some wide)
            let mut whome = HashMap::new();
            let (mut link_ru, mut link_pk) = (Vec::new(), Vec::new());
            let mut pk_of: HashMap<u32, u64> = HashMap::new();
            for &u in &users {
                if pk_of.contains_key(&u) || rng(&mut seed) % 3 != 0 {
                    continue;
                }
                let pk = match rng(&mut seed) % 3 {
                    0 => own | u64::from(u), // a representative
                    1 => (rng(&mut seed) % 64) << rbits | (rng(&mut seed) % (1u64 << rbits)),
                    _ => {
                        let home = rng(&mut seed) % 64;
                        let pk = 1 << 63 | 1 << 61 | home << 32 | (rng(&mut seed) % 1000);
                        whome.insert(pk, home as u32);
                        pk
                    }
                };
                pk_of.insert(u, pk);
                link_ru.push(u);
                link_pk.push(pk);
            }
            let mut wlink = Vec::new();
            let mut wpk_of: HashMap<[u64; 3], u64> = HashMap::new();
            for w in &wusers {
                if wpk_of.contains_key(w) || rng(&mut seed) % 2 != 0 {
                    continue;
                }
                let pk = (rng(&mut seed) % 64) << rbits | (rng(&mut seed) % (1u64 << rbits));
                wpk_of.insert(*w, pk);
                wlink.push([w[0], w[1], pk, w[2]]);
            }
            // survivors (SHORT stream with payload words, wide stream), clicks
            let ne = (rng(&mut seed) % 3000) as usize;
            let (mut surv, mut wsurv) = (Vec::new(), Vec::new());
            let mut exp_ev: HashMap<(u8, u64), (Vec<u32>, Vec<u64>)> = HashMap::new();
            for _ in 0..ne {
                let ts = (rng(&mut seed) % (1u64 << 31)) as u32;
                let purch = rng(&mut seed) % 10 == 0;
                let e = ts << 1 | u32::from(purch);
                let pay = rng(&mut seed) & 0xffff_ffff;
                if nw > 0 && rng(&mut seed) % 8 == 0 {
                    let w = wusers[(rng(&mut seed) % nw as u64) as usize];
                    wsurv.extend_from_slice(&[w[0], w[1], u64::from(e) | w[2] << 62]);
                    if purch {
                        wsurv.push(pay);
                    }
                    let x = exp_ev.entry((1, w[0] ^ w[1] << 8 ^ w[2] << 60)).or_default();
                    x.0.push(e);
                    if purch {
                        x.1.push(pay);
                    }
                } else {
                    let u = users[(rng(&mut seed) % nu as u64) as usize];
                    surv.push(u64::from(u) << 32 | u64::from(e));
                    if purch {
                        surv.push(pay);
                    }
                }
            }
            let nc = if case % 10 == 0 { 1500 } else { (rng(&mut seed) % 600) as usize };
            let (mut clicks, mut wclicks) = (Vec::new(), Vec::new());
            for _ in 0..nc {
                let tc = rng(&mut seed) & 0xffff_ffff;
                let cl = rng(&mut seed);
                if nw > 0 && rng(&mut seed) % 8 == 0 {
                    let w = wusers[(rng(&mut seed) % nw as u64) as usize];
                    wclicks.push([w[0], w[1], tc | w[2] << 62, cl]);
                } else {
                    // some clicks of users never seen
                    let u = if rng(&mut seed) % 10 == 0 { (rng(&mut seed) % (1u64 << rbits)) as u32 } else { users[(rng(&mut seed) % nu as u64) as usize] };
                    clicks.push([u64::from(u) << 32 | tc, cl]);
                }
            }
            // reference: per user in reading order
            let mut r_ev: HashMap<u32, (Vec<u32>, Vec<u64>)> = HashMap::new();
            let mut order: Vec<u32> = Vec::new();
            let mut i = 0;
            while i < surv.len() {
                let r = surv[i];
                let u = (r >> 32) as u32;
                let x = r_ev.entry(u).or_insert_with(|| {
                    order.push(u);
                    (Vec::new(), Vec::new())
                });
                x.0.push(r as u32);
                i += 1;
                if r & 1 != 0 {
                    x.1.push(surv[i]);
                    i += 1;
                }
            }
            let mut w_ev: HashMap<[u64; 3], (Vec<u32>, Vec<u64>)> = HashMap::new();
            let mut i = 0;
            while i < wsurv.len() {
                let key = [wsurv[i], wsurv[i + 1], wsurv[i + 2] >> 62];
                let e = wsurv[i + 2] as u32;
                let x = w_ev.entry(key).or_default();
                x.0.push(e);
                i += 3;
                if e & 1 != 0 {
                    x.1.push(wsurv[i]);
                    i += 1;
                }
            }
            let mut r_ck: HashMap<u32, Vec<[u64; 2]>> = HashMap::new();
            for c in &clicks {
                let u = (c[0] >> 32) as u32;
                if r_ev.contains_key(&u) || pk_of.contains_key(&u) {
                    r_ck.entry(u).or_default().push([c[1], c[0] & 0xffff_ffff]);
                }
            }
            let mut w_ck: HashMap<[u64; 3], Vec<[u64; 2]>> = HashMap::new();
            for c in &wclicks {
                let key = [c[0], c[1], c[2] >> 62];
                if w_ev.contains_key(&key) || wpk_of.contains_key(&key) {
                    w_ck.entry(key).or_default().push([c[3], c[2] & 0xffff_ffff]);
                }
            }
            // run P1 with random piece cuts
            let cut = |v: &Vec<u64>, s: &mut u64| -> Vec<Vec<u64>> {
                let mut out = Vec::new();
                let mut a = 0;
                while a < v.len() {
                    let b = (a + 1 + (rng(s) % 700) as usize).min(v.len());
                    out.push(v[a..b].to_vec());
                    a = b;
                }
                out
            };
            let sp = cut(&surv, &mut seed);
            let wp = cut(&wsurv, &mut seed);
            let sps: Vec<&[u64]> = sp.iter().map(|v| &v[..]).collect();
            let wps: Vec<&[u64]> = wp.iter().map(|v| &v[..]).collect();
            let ck2: Vec<&[[u64; 2]]> = vec![&clicks[..]];
            let wck2: Vec<&[[u64; 4]]> = vec![&wclicks[..]];
            let inp = In { q, rbits, surv: &sps, wsurv: &wps, clicks: &ck2, wclicks: &wck2, link_ru: &link_ru, link_pk: &link_pk, wlink: &wlink };
            let mut st = State::new();
            let mut out = VecOut { homes: HashMap::new() };
            part(&inp, &mut st, &mut out).unwrap();
            // expected blocks; P1-local wide pks are numbered in P1's own order, so map them back
            let local_index: HashMap<[u64; 2], u64> = st.wide.iter().enumerate().map(|(i, w)| (*w, i as u64)).collect();
            let mut exp: HashMap<u32, Vec<Box<[u64]>>> = HashMap::new();
            let home_of = |pk: u64| -> u32 {
                if pk >> 63 == 0 { (pk >> rbits) as u32 } else if pk >> 60 & 1 != 0 { q } else { whome[&pk] }
            };
            let mut put = |pk: u64, evs: &[u32], pays: &[u64], cks: &[[u64; 2]]| {
                let wide = pk >> 63;
                let home = home_of(pk);
                // one block each, or cut like P1's `split` when it would not fit a chunk
                let one = 2 + evs.len().div_ceil(2) + pays.len() <= CW - 2;
                let mut pi = 0;
                for g in evs.chunks(if one { evs.len().max(1) } else { CUT_EV }) {
                    if g.is_empty() {
                        break;
                    }
                    let nbg = g.iter().filter(|&&e| e & 1 != 0).count();
                    let mut b = vec![g.len() as u64 | (nbg as u64) << 24 | wide << 63, pk];
                    for c in g.chunks(2) {
                        b.push(u64::from(c[0]) | u64::from(*c.get(1).unwrap_or(&0xFFFF_FFFF)) << 32);
                    }
                    b.extend_from_slice(&pays[pi..pi + nbg]);
                    pi += nbg;
                    exp.entry(home).or_default().push(b.into_boxed_slice());
                }
                let onec = 2 + 2 * cks.len() <= CW - 2;
                for g in cks.chunks(if onec { cks.len().max(1) } else { CUT_CK }) {
                    if g.is_empty() {
                        break;
                    }
                    let mut b = vec![g.len() as u64 | 1 << 62 | wide << 63, pk];
                    for c in g {
                        b.extend_from_slice(c);
                    }
                    exp.entry(home).or_default().push(b.into_boxed_slice());
                }
            };
            let mut all_users: Vec<u32> = r_ev.keys().chain(r_ck.keys()).copied().collect();
            all_users.sort();
            all_users.dedup();
            for u in all_users {
                let pk = pk_of.get(&u).copied().unwrap_or(own | u64::from(u));
                let (e, p) = r_ev.get(&u).cloned().unwrap_or_default();
                let c = r_ck.get(&u).cloned().unwrap_or_default();
                put(pk, &e, &p, &c);
            }
            let mut all_w: Vec<[u64; 3]> = w_ev.keys().chain(w_ck.keys()).copied().collect();
            all_w.sort();
            all_w.dedup();
            for w in all_w {
                let pk = wpk_of.get(&w).copied().unwrap_or_else(|| 1 << 63 | w[2] << 61 | 1 << 60 | u64::from(q) << 32 | local_index[&[w[0], w[1]]]);
                let (e, p) = w_ev.get(&w).cloned().unwrap_or_default();
                let c = w_ck.get(&w).cloned().unwrap_or_default();
                put(pk, &e, &p, &c);
            }
            let _ = &order;
            let _ = &exp_ev;
            assert_eq!(canon(&out.homes), canon(&exp), "case {case}");
            // the same partition through the real chains (B8K chunks, end marks): equal too
            assert!(crate::v3::mem::init(1 << 30));
            let homes = 1usize << (40 - rbits);
            // a worker index of its own: tests run on parallel threads and share the B8K class
            let mut ch = Chains::new(homes, 200);
            {
                let mut st2 = State::new();
                let mut co = ChainOut { chains: &mut ch };
                part(&inp, &mut st2, &mut co).unwrap();
            }
            let blocks = Blocks { workers: vec![ch], wide: Vec::new() };
            let mut got: HashMap<u32, Vec<Box<[u64]>>> = HashMap::new();
            for home in 0..homes {
                for c in blocks.chunks(home) {
                    let mut at = 0;
                    while at < c.len() && c[at] != 0 {
                        let h0 = c[at];
                        let len = if h0 >> 62 & 1 != 0 { 2 + 2 * (h0 & 0xFF_FFFF) as usize } else { 2 + ((h0 & 0xFF_FFFF) as usize).div_ceil(2) + (h0 >> 24 & 0xFF_FFFF) as usize };
                        assert!(at + len <= c.len(), "case {case}: a block past its chunk");
                        got.entry(home as u32).or_default().push(c[at..at + len].to_vec().into_boxed_slice());
                        at += len;
                    }
                }
            }
            assert!(canon(&got) == canon(&exp), "chains, case {case}");
        }
    }
}
