//! v4 dedup (RECORDS-V4 sections 3 and 4). Per event partition: a bitmap pre-filter over the
//! event keys, an exact table only for the records whose key bit collided, and every survivor
//! straight to its USER partition (a shift and a store: the user key travels in the record).
//!
//! Exact: copies of an event share its partition (the partition is a function of the key). Pass 1
//! sets a [seen, again] bit pair per key hash; a key with two or more records sets "again". A record
//! whose "again" bit is clear is the only copy of its key: the winner, a survivor when alive. The
//! others go through the table: smallest ingest wins, equal ingest means identical copies (any one
//! stays), a dead copy shadows like a live one. Both lanes of a partition are marked before either
//! is read for output, so copies of one key in different lanes meet in the table.
//!
//! Input per event partition, two lanes (sink.rs):
//! - plain units [w0, w1] (SHORT event, SHORT or dead user, no purchase): re = w0 bits 0-31,
//!   ingest bits 32-54, mu >> 33 in bits 55-61, surv bit 62 (bit 63 clear); ts = w1 bits 0-30, mu
//!   low 33 bits in w1 bits 31-63.
//! - rare records (2 or 3 units, `sink::RARE_BIG`): every purchase and every record with a LONG
//!   or FOREIGN id.
//! Every record is one fixed-size item: no decoding from the left, no record across chunks.
//!
//! Output per user partition pu: SHORT users `ru << 32 | ts << 1 | purch` then the payload word
//! when purch; wide users `a, b, ts << 1 | purch | kind << 62` then the payload word when purch.

use super::keys::long_hash;
use super::sink::{EventBins, RARE_BIG, RARE_EXT};
use super::{Cx, Stage};
use crate::v3::dedup::Out;

const M23: u64 = (1 << 23) - 1;
const M32: u64 = 0xffff_ffff;
const TS: u64 = 0x7fff_ffff;
/// The pre-filter's multiplier (unrelated to mix40's MIX_C, as RECORDS-V4 section 1 requires).
const FH: u64 = 0x9e37_79b9_7f4a_7c15;
/// The exact table's multiplier (another one again: the table holds keys whose FH bits collided).
const TH: u64 = 0xc2b2_ae3d_27d4_eb4f;

type Unit = [u64; 2];

/// The event key (a, b, kind) of the rare record at `u`: (re, 0, SHORT) for a SHORT event.
///
/// # Safety
/// `u` is a rare record (2 or 3 units), or `DEAD`.
#[inline(always)]
unsafe fn rkey(u: *const Unit) -> (u64, u64, u64) {
    // SAFETY: a record has at least 2 units.
    let (h, w1, n) = unsafe { ((*u)[0], (*u)[1], *u.add(1)) };
    let ek = (h >> 23) & 3;
    // Selects, not a branch: the event kind is SHORT in about half the rare records.
    let a = std::hint::select_unpredictable(ek != 0, n[0], w1 & M32);
    let b = std::hint::select_unpredictable(ek != 0, n[1], 0);
    (a, b, ek)
}

/// A key's hash: (re, 0, SHORT) hashes to re, as the plain lane's keys do.
#[inline(always)]
fn key_hash(a: u64, b: u64, kind: u64) -> u64 {
    a ^ b.rotate_left(29) ^ (kind << 62)
}

/// The user side of the rare record at `u`: (user kind, ua, ub, ts, purch, payload).
///
/// # Safety
/// As `rkey`.
#[inline(always)]
unsafe fn ruser(u: *const Unit) -> (u64, u64, u64, u64, u64, u64) {
    // SAFETY: a record has 2 units, 3 when big.
    unsafe {
        let (h, w1) = ((*u)[0], (*u)[1]);
        let (ek0, big) = ((h >> 23) & 3 == 0, h & RARE_BIG != 0);
        // The user unit: u2 when big, else u1 (a SHORT event's record is never big). Selects:
        // the shapes are mixed.
        let x = *std::hint::select_unpredictable(big, u.add(2), u.add(1));
        let user = std::hint::select_unpredictable(ek0 | big, x, [w1, 0]);
        // The payload (only read with a purchase, which a 2-unit LONG-event record never has).
        let pay = std::hint::select_unpredictable(ek0, w1 >> 32, w1);
        ((h >> 25) & 3, user[0], user[1], (h >> 29) & TS, (h >> 28) & 1, pay)
    }
}

/// The record after the rare record at `u` (2 units on, 3 for a big one: a branch, predicted
/// not taken, so the walk does not wait for each header).
///
/// # Safety
/// `u` is a rare record.
#[inline(always)]
unsafe fn rnext(u: *const Unit) -> *const Unit {
    // SAFETY: the record's own units.
    unsafe {
        let v = u.add(2);
        let h = (*u)[0];
        if h & (RARE_BIG | RARE_EXT) != 0 {
            std::hint::cold_path();
            return v.add(((h >> 60) & 1) as usize + ((h >> 61) & 1) as usize);
        }
        v
    }
}

/// A record whose key bit collided, for the exact table: the key (a, b, kind << 62 | ingest)
/// and where the record is (`DEAD` once another copy of its key wins). The ingest is
/// `ING_BIAS` + milliseconds from the ingest base: the field's seconds times 1000, or the
/// out-of-field value of a `RARE_EXT` record (as v3 compares its specials).
#[derive(Clone, Copy)]
struct Cd {
    a: u64,
    b: u64,
    ki: u64,
    at: usize,
}

/// See `Cd`: keeps every compared ingest positive below 2^62.
const ING_BIAS: u64 = 1 << 52;
const M62: u64 = (1 << 62) - 1;

/// A dead record, as a plain unit or a rare record (every field 0: not alive): where a table
/// entry that lost points, so the winners' pass reads it as dead without a test.
static DEAD: [Unit; 3] = [[0; 2]; 3];

/// One worker's reusable state.
#[derive(Default)]
pub struct Tables {
    fm: Vec<[u64; 2]>,
    /// Addresses of the colliding records, per lane (written for every record, kept when the
    /// bit collided).
    cp: Vec<usize>,
    cr: Vec<usize>,
    /// Addresses of the rare records that are the only copy and alive: SHORT users, wide users.
    rs: Vec<usize>,
    rw: Vec<usize>,
    cd: Vec<Cd>,
    slots: Vec<u32>,
}

impl Tables {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Survivors by user partition: SHORT users and wide users, one bin per pu.
pub struct Out2 {
    pub short: Out,
    pub wide: Out,
    b2: u32,
}

impl Out2 {
    /// 2^b2 user partitions; chunks are 8 KB blocks taken under pool participant `w`.
    pub fn new(b2: u32, w: usize) -> Self {
        Out2 { short: Out::new_in(1 << b2, w), wide: Out::new_in(1 << b2, w), b2 }
    }

    pub fn finish(&mut self) {
        self.short.finish();
        self.wide.finish();
    }

    /// A SHORT user's survivor: pu, ru from mu (40 bits: below 2^b2 and 2^(40 - b2)).
    #[inline(always)]
    fn emit_short(&mut self, mu: u64, ts: u64, purch: u64, pay: u64) {
        let q = (mu >> (40 - self.b2)) as usize;
        let ru = mu & ((1 << (40 - self.b2)) - 1);
        put(&mut self.short, q, &[(ru << 32) | (ts << 1) | purch, pay][..1 + purch as usize]);
    }

    /// A LONG or FOREIGN user's survivor.
    #[inline(always)]
    fn emit_wide(&mut self, ukind: u64, ua: u64, ub: u64, ts: u64, purch: u64, pay: u64) {
        let q = (long_hash(ua, ub) >> (64 - self.b2)) as usize;
        put(&mut self.wide, q, &[ua, ub, (ts << 1) | purch | (ukind << 62), pay][..3 + purch as usize]);
    }

    #[inline(always)]
    fn emit(&mut self, ukind: u64, ua: u64, ub: u64, ts: u64, purch: u64, pay: u64) {
        if ukind == 0 {
            self.emit_short(ua, ts, purch, pay);
        } else {
            self.emit_wide(ukind, ua, ub, ts, purch, pay);
        }
    }

    /// A live rare record.
    ///
    /// # Safety
    /// As `rkey`.
    #[inline(always)]
    unsafe fn emit_rare(&mut self, u: *const Unit) {
        // SAFETY: the caller's.
        let (uk, ua, ub, ts, purch, pay) = unsafe { ruser(u) };
        self.emit(uk, ua, ub, ts, purch, pay);
    }
}

/// Appends `words` to partition q of `o` (q below its partition count) with one load and one
/// store of the fill count.
#[inline(always)]
fn put(o: &mut Out, q: usize, words: &[u64]) {
    let (stage, fill, _) = o.raw_parts();
    // SAFETY: q is below the partition count; the fill count stays below 8 (see raw_parts).
    unsafe {
        let f = fill.add(q);
        let mut n = *f as usize;
        for &w in words {
            *stage.add(q * 8 + n) = w;
            n += 1;
            if n == 8 {
                o.flush_line(q);
                n = 0;
            }
        }
        *f = n as u32;
    }
}

/// Sets the [seen, again] pair of key hash `hk`.
///
/// # Safety
/// `fp` holds at least `2^(64 - fsh) / 64` word pairs.
#[inline(always)]
unsafe fn mark(fp: *mut [u64; 2], fsh: u32, hk: u64) {
    let k = (hk.wrapping_mul(FH) >> fsh) as usize;
    // SAFETY: k < 2^(64 - fsh), so k >> 6 is inside (caller).
    unsafe {
        let wd = fp.add(k >> 6);
        let bit = 1u64 << (k & 63);
        (*wd)[1] |= (*wd)[0] & bit;
        (*wd)[0] |= bit;
    }
}

/// The "again" bit (0 or 1) of key hash `hk`.
///
/// # Safety
/// As `mark`.
#[inline(always)]
unsafe fn again(fp: *const [u64; 2], fsh: u32, hk: u64) -> u64 {
    let k = (hk.wrapping_mul(FH) >> fsh) as usize;
    // SAFETY: as in mark.
    unsafe { ((*fp.add(k >> 6))[1] >> (k & 63)) & 1 }
}

/// Pass 1 over plain units: one load, a multiply, a bit pair.
///
/// # Safety
/// As `mark`.
#[inline(never)]
unsafe fn mark_plain(c: &[Unit], fp: *mut [u64; 2], fsh: u32) {
    for u in c {
        // SAFETY: the caller's bound.
        unsafe { mark(fp, fsh, u[0] & M32) };
    }
}

/// Pass 1 over rare records.
///
/// # Safety
/// As `mark`.
#[inline(never)]
unsafe fn mark_rare(c: &[Unit], fp: *mut [u64; 2], fsh: u32) {
    let mut u = c.as_ptr();
    // SAFETY: one past the end of `c`; records never cross a chunk, so the walk ends there.
    let end = unsafe { u.add(c.len()) };
    while u < end {
        // SAFETY: u is a record of this chunk; the caller's bound.
        unsafe {
            let (a, b, k) = rkey(u);
            mark(fp, fsh, key_hash(a, b, k));
            u = rnext(u);
        }
    }
}

/// x below 2^n (BMI2 bzhi where the target has it).
#[inline(always)]
fn low_bits(x: u64, n: u32) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    // SAFETY: the target has BMI2.
    return unsafe { core::arch::x86_64::_bzhi_u64(x, n) };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    return x & ((1u64 << n) - 1);
}

/// Stores plain unit [w, w1]'s survivor word in its user partition's staging line, kept (the
/// fill count + 1) when `keep` is 1; a word not kept is overwritten by the next one.
///
/// # Safety
/// `stage`, `fill` from `short.raw_parts()`; `rsh` = 40 - b2.
#[inline(always)]
unsafe fn stage_plain(stage: *mut u64, fill: *mut u32, short: &mut Out, rsh: u32, w: u64, w1: u64, keep: u64) {
    let mu = (((w >> 55) & 0x7f) << 33) | (w1 >> 31);
    let q = (mu >> rsh) as usize;
    let word = (low_bits(mu, rsh) << 32) | ((w1 & TS) << 1);
    // SAFETY: mu is below 2^40, so q is below 2^b2 = the partition count; the fill count is
    // below 8 (see `Out::raw_parts`), so the slot is inside q's line.
    unsafe {
        let f = fill.add(q);
        let n = *f as usize;
        *stage.add(q * 8 + n) = word;
        let n = n + keep as usize;
        if n == 8 {
            *f = 0;
            short.flush_line(q);
        } else {
            *f = n as u32;
        }
    }
}

/// Pass 2 over plain units, without a data-dependent branch but a full staging line: every
/// unit's address is stored at `cw` and kept (cw + 1) when its bit collided; every unit's output
/// word is staged and kept when it is the only copy and alive. The inner loop makes no call (a
/// full line leaves it, is flushed, and the walk resumes), so its state stays in registers.
/// Returns the next `cw`.
///
/// # Safety
/// As `mark`; `cw` has room for `c.len()` addresses.
#[inline(never)]
unsafe fn emit_plain(c: &[Unit], fp: *const [u64; 2], fsh: u32, out: &mut Out2, mut cw: *mut usize) -> *mut usize {
    let rsh = 40 - out.b2;
    let (stage, fill, _) = out.short.raw_parts();
    let mut u = c.as_ptr();
    // SAFETY: one past the end of `c`.
    let end = unsafe { u.add(c.len()) };
    loop {
        let mut full = usize::MAX;
        while u != end {
            // SAFETY: u < end; the caller's bounds for fp and cw; q is below 2^b2 (mu is below
            // 2^40) and the fill count below 8 (see `Out::raw_parts`).
            unsafe {
                let (w, w1) = ((*u)[0], (*u)[1]);
                let ag = again(fp, fsh, w & M32);
                *(cw as *mut *const Unit) = u;
                cw = cw.add(ag as usize);
                // The only copy, alive (surv is bit 62; bit 63 is clear in this lane).
                let keep = (ag ^ 1) & (w >> 62);
                let mu = (((w >> 55) & 0x7f) << 33) | (w1 >> 31);
                let q = mu >> rsh;
                let word = ((mu ^ (q << rsh)) << 32) | ((w1 & TS) << 1);
                let f = fill.add(q as usize);
                let n = *f as usize;
                *stage.add(q as usize * 8 + n) = word;
                let n = n + keep as usize;
                u = u.add(1);
                if n == 8 {
                    *f = 0;
                    full = q as usize;
                    break;
                }
                *f = n as u32;
            }
        }
        if full == usize::MAX {
            return cw;
        }
        // SAFETY: line `full` holds eight words.
        unsafe { out.short.flush_line(full) };
    }
}

/// Pass 2 over rare records, without a data-dependent branch but the record size: each record's
/// address goes to `cw` (kept when its bit collided), to `sw` (kept when it is the only copy,
/// alive, with a SHORT user) and to `ww` (the same with a wide user). Returns the three next
/// pointers.
///
/// # Safety
/// As `mark`; each pointer has room for `c.len() / 2` addresses.
#[inline(never)]
unsafe fn sort_rare(c: &[Unit], fp: *const [u64; 2], fsh: u32, mut cw: *mut usize, mut sw: *mut usize, mut ww: *mut usize) -> (*mut usize, *mut usize, *mut usize) {
    let mut u = c.as_ptr();
    // SAFETY: as in mark_rare.
    let end = unsafe { u.add(c.len()) };
    while u < end {
        // SAFETY: u is a record of this chunk; the caller's bounds.
        unsafe {
            let h = (*u)[0];
            let (a, b, k) = rkey(u);
            let ag = again(fp, fsh, key_hash(a, b, k));
            let at = u as usize;
            let keep = (ag ^ 1) & (h >> 27) & 1;
            let wide = u64::from((h >> 25) & 3 != 0);
            *cw = at;
            *sw = at;
            *ww = at;
            cw = cw.add(ag as usize);
            sw = sw.add((keep & (wide ^ 1)) as usize);
            ww = ww.add((keep & wide) as usize);
            u = rnext(u);
        }
    }
    (cw, sw, ww)
}

/// Sets `v`'s length to the addresses written up to `end`.
///
/// # Safety
/// `end` was derived from `v.as_mut_ptr()` within its capacity, every slot below it written.
#[inline(always)]
unsafe fn written(v: &mut Vec<usize>, end: *mut usize) {
    // SAFETY: the caller's contract.
    unsafe { v.set_len(end.offset_from(v.as_ptr()) as usize) };
}

/// Dedups one event partition (its plain and rare chunks) into `out` by user partition.
pub fn partition(plain: &[&[Unit]], rare: &[&[Unit]], t: &mut Tables, out: &mut Out2) {
    let np: usize = plain.iter().map(|c| c.len()).sum();
    let nr: usize = rare.iter().map(|c| c.len()).sum();
    let fbits = ((np + nr) * 16).next_power_of_two().clamp(1 << 12, 1 << 20).trailing_zeros();
    let fsh = 64 - fbits;
    t.fm.clear();
    t.fm.resize(1 << (fbits - 6), [0; 2]);
    let fp = t.fm.as_mut_ptr();
    // Pass 1: every key of both lanes. SAFETY (both passes): fm holds 2^fbits / 64 pairs.
    for c in plain {
        unsafe { mark_plain(c, fp, fsh) };
    }
    for c in rare {
        unsafe { mark_rare(c, fp, fsh) };
    }
    // Pass 2: the only copies straight out; the others' addresses for the table.
    t.cp.clear();
    t.cp.reserve(np);
    let mut cw = t.cp.as_mut_ptr();
    for c in plain {
        // SAFETY: cp has room for np addresses, and the chunks hold np units.
        cw = unsafe { emit_plain(c, fp, fsh, out, cw) };
    }
    // SAFETY: as written by emit_plain.
    unsafe { written(&mut t.cp, cw) };
    // A rare record is at least 2 units.
    for v in [&mut t.cr, &mut t.rs, &mut t.rw] {
        v.clear();
        v.reserve(nr / 2);
    }
    let (mut cw, mut sw, mut ww) = (t.cr.as_mut_ptr(), t.rs.as_mut_ptr(), t.rw.as_mut_ptr());
    for c in rare {
        // SAFETY: each list has room for nr / 2 addresses, and the chunks hold at most that many records.
        (cw, sw, ww) = unsafe { sort_rare(c, fp, fsh, cw, sw, ww) };
    }
    // SAFETY: as written by sort_rare.
    unsafe {
        written(&mut t.cr, cw);
        written(&mut t.rs, sw);
        written(&mut t.rw, ww);
    }
    for &at in &t.rs {
        // SAFETY: a rare record's address in this partition's chunks.
        let (_, mu, _, ts, purch, pay) = unsafe { ruser(at as *const Unit) };
        out.emit_short(mu, ts, purch, pay);
    }
    for &at in &t.rw {
        // SAFETY: as above.
        let (uk, ua, ub, ts, purch, pay) = unsafe { ruser(at as *const Unit) };
        out.emit_wide(uk, ua, ub, ts, purch, pay);
    }
    // The exact table on the colliding records: key, ingest and address of each, plain first.
    let cd = &mut t.cd;
    cd.clear();
    for &at in &t.cp {
        // SAFETY: a plain unit's address in this partition's chunks.
        let w = unsafe { (*(at as *const Unit))[0] };
        cd.push(Cd { a: w & M32, b: 0, ki: ((w >> 32) & M23) * 1000 + ING_BIAS, at });
    }
    let ncp = cd.len();
    for &at in &t.cr {
        // SAFETY: a rare record's address in this partition's chunks.
        let u = at as *const Unit;
        let (a, b, k) = unsafe { rkey(u) };
        let h = unsafe { (*u)[0] };
        let ing = if h & RARE_EXT == 0 {
            (h & M23) * 1000 + ING_BIAS
        } else {
            // SAFETY: the extra unit after the record's own (2 units, 3 when big).
            let x = unsafe { (*u.add(2 + ((h >> 60) & 1) as usize))[0] } as i64;
            (x + ING_BIAS as i64) as u64
        };
        cd.push(Cd { a, b, ki: k << 62 | ing, at });
    }
    let m = cd.len();
    let tbits = (2 * m).next_power_of_two().max(16).trailing_zeros();
    let slots = &mut t.slots;
    slots.clear();
    slots.resize(1 << tbits, 0);
    let mask = (1usize << tbits) - 1;
    let dead = DEAD.as_ptr() as usize;
    for i in 0..m {
        let e = cd[i];
        let hk = key_hash(e.a, e.b, e.ki >> 62);
        let mut s = (hk.wrapping_mul(TH) >> (64 - tbits)) as usize;
        loop {
            let c = slots[s] as usize;
            if c == 0 {
                slots[s] = (i + 1) as u32;
                break;
            }
            let o = cd[c - 1];
            if o.a == e.a && o.b == e.b && o.ki >> 62 == e.ki >> 62 {
                // The smaller ingest wins (equal: identical copies, the one in place stays);
                // the loser's address becomes DEAD.
                let wins = (e.ki & M62) < (o.ki & M62);
                slots[s] = if wins { (i + 1) as u32 } else { c as u32 };
                let lose = if wins { c - 1 } else { i };
                cd[lose].at = dead;
                break;
            }
            // Terminates: at most half the slots are used.
            s = (s + 1) & mask;
        }
    }
    // The winners, in record order (a loser reads as dead).
    let rsh = 40 - out.b2;
    let (stage, fill, _) = out.short.raw_parts();
    for e in &cd[..ncp] {
        // SAFETY: a plain unit's address, or DEAD; stage and fill from raw_parts.
        unsafe {
            let u = *(e.at as *const Unit);
            stage_plain(stage, fill, &mut out.short, rsh, u[0], u[1], (u[0] >> 62) & 1);
        }
    }
    for e in &cd[ncp..] {
        // SAFETY: a rare record's address, or DEAD.
        unsafe {
            if ((*(e.at as *const Unit))[0] >> 27) & 1 != 0 {
                out.emit_rare(e.at as *const Unit);
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a store fence: the non-temporal lines are visible before the task is reported done.
    unsafe {
        core::arch::x86_64::_mm_sfence()
    };
}

/// Survivors by user partition (I2): one Out2 per worker; partition q's words are every worker's
/// `short.slices(q)` and `wide.slices(q)`, each bin read in chunk order (a record may straddle).
#[derive(Default)]
pub struct Survivors {
    pub outs: Vec<Out2>,
}

impl Survivors {
    pub fn short(&self, q: usize) -> impl Iterator<Item = &[u64]> + '_ {
        self.outs.iter().flat_map(move |o| o.short.slices(q))
    }

    pub fn wide(&self, q: usize) -> impl Iterator<Item = &[u64]> + '_ {
        self.outs.iter().flat_map(move |o| o.wide.slices(q))
    }

    /// Hands user partition q's chunks (every worker's, SHORT and wide) back to B8K under
    /// participant `w`, so the next writer takes them before fresh memory.
    /// SAFETY: called by the one task that owns q, once it no longer reads q; `w` is that task's
    /// participant index; q's chunks are never read again (dropping Survivors skips them).
    pub unsafe fn free_part(&self, q: usize, w: usize) {
        for o in &self.outs {
            // SAFETY: the caller's contract.
            unsafe {
                o.short.free_bin(q, w);
                o.wide.free_bin(q, w);
            }
        }
    }
}

/// Dedups `parts` event partitions on the pool; `chunks(p)`: partition p's plain and rare chunks;
/// `release(p, w)` returns partition p's chunks once it is read (worker index `w`).
pub fn run_parts<'a, F, R>(cx: &Cx, parts: usize, chunks: F, release: R) -> Stage<Survivors>
where
    F: Fn(usize) -> (Vec<&'a [Unit]>, Vec<&'a [Unit]>) + Sync,
    R: Fn(usize, usize) + Sync,
{
    let b2 = cx.plan.b2;
    let dd = crate::par::run(
        cx.plan.threads,
        parts,
        |w| (Tables::new(), Out2::new(b2, w), w),
        |(t, o, w), p| {
            let (pc, rc) = chunks(p);
            partition(&pc, &rc, t, o);
            // The owner of the chunks returns them (once per partition; its drop frees the rest).
            release(p, *w);
        },
    );
    let mut sv = Survivors::default();
    for (_, mut o, _) in dd {
        o.finish();
        sv.outs.push(o);
    }
    Ok(sv)
}

/// The driver's entry: partition p's chunks from the event bins (wired by the sink's owner).
pub fn run(cx: &Cx, ev: &EventBins) -> Stage<Survivors> {
    run_parts(
        cx,
        1usize << ev.b1,
        |p| (ev.chunks(p).collect(), ev.rare(p).collect()),
        // SAFETY: `w` is this pool participant's index, and partition p's slices are not read
        // after this call (run_parts calls it once the partition is done).
        |p, w| unsafe { ev.free_part(p, w) },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct R(u64);
    impl R {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// One copy: (event key, ingest, surv, purch, ts, ukind, ua, ub, pay).
    type C = (Key, u64, u64, u64, u64, u64, u64, u64, u64);

    /// An event key in the test: SHORT (a = re, b = 0, kind 0) or LONG/FOREIGN.
    #[derive(Clone, Copy, PartialEq, Eq)]
    struct Key {
        a: u64,
        b: u64,
        kind: u64,
    }

    /// One copy into its lane: plain units one per record; rare records as groups of units.
    /// The ingest of a test copy: milliseconds from the ingest base (two's complement); a whole
    /// second below 50 s is an in-field value, anything else an out-of-field one (RARE_EXT).
    fn field(ing: u64) -> Option<u64> {
        (ing % 1000 == 0 && ing / 1000 < 50).then_some(ing / 1000)
    }

    fn encode(c: &C, plain: &mut Vec<Unit>, rare: &mut Vec<Vec<Unit>>) {
        let &(k, ing_ms, surv, purch, ts, uk, ua, ub, pay) = c;
        if let Some(ing) = field(ing_ms) {
            encode_in(c, ing, plain, rare);
            return;
        }
        let uk = if surv != 0 { uk } else { 0 };
        let user = if surv == 0 { [0, 0] } else if uk == 0 { [ua, 0] } else { [ua, ub] };
        let h = RARE_EXT | k.kind << 23 | uk << 25 | surv << 27 | purch << 28 | (ts * surv) << 29;
        let x = [ing_ms, 0];
        rare.push(if k.kind == 0 {
            vec![[h, k.a | pay << 32], user, x]
        } else if uk == 0 && purch == 0 {
            vec![[h, user[0]], [k.a, k.b], x]
        } else {
            vec![[h | RARE_BIG, pay], [k.a, k.b], user, x]
        });
    }

    fn encode_in(c: &C, ing: u64, plain: &mut Vec<Unit>, rare: &mut Vec<Vec<Unit>>) {
        let &(k, _, surv, purch, ts, uk, ua, ub, pay) = c;
        if k.kind == 0 && (surv == 0 || uk == 0) && purch == 0 {
            let w0 = k.a | (ing << 32) | if surv != 0 { ((ua >> 33) & 0x7f) << 55 } else { 0 } | (surv << 62);
            let w1 = if surv != 0 { ts | ((ua & ((1 << 33) - 1)) << 31) } else { 0 };
            plain.push([w0, w1]);
            return;
        }
        let uk = if surv != 0 { uk } else { 0 };
        let user = if surv == 0 { [0, 0] } else if uk == 0 { [ua, 0] } else { [ua, ub] };
        let h = ing | k.kind << 23 | uk << 25 | surv << 27 | purch << 28 | (ts * surv) << 29;
        rare.push(if k.kind == 0 {
            vec![[h, k.a | pay << 32], user]
        } else if uk == 0 && purch == 0 {
            vec![[h, user[0]], [k.a, k.b]]
        } else {
            vec![[h | RARE_BIG, pay], [k.a, k.b], user]
        });
    }

    fn cut<'a, T>(v: &'a [T], r: &mut R) -> Vec<&'a [T]> {
        let mut cuts = vec![0];
        while *cuts.last().unwrap() < v.len() {
            let step = 1 + (r.next() % 40) as usize;
            cuts.push((cuts.last().unwrap() + step).min(v.len()));
        }
        cuts.windows(2).map(|w| &v[w[0]..w[1]]).collect()
    }

    /// Rare records into chunks of whole records, cut at random record boundaries.
    fn cut_groups(g: &[Vec<Unit>], r: &mut R) -> Vec<Vec<Unit>> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < g.len() {
            let n = (1 + (r.next() % 12) as usize).min(g.len() - i);
            out.push(g[i..i + n].iter().flatten().copied().collect());
            i += n;
        }
        out
    }

    fn words(o: &Out2, b2: u32) -> Vec<Vec<u64>> {
        let mut v = Vec::new();
        for q in 0..1usize << b2 {
            // A survivor's words may straddle chunks (one bin's chunks are in order): flatten.
            let c: Vec<u64> = o.short.slices(q).flat_map(|c| c.iter().copied()).collect();
            let mut i = 0;
            while i < c.len() {
                let w = c[i];
                let p = if w & 1 != 0 { i += 1; c[i] } else { 0 };
                v.push(vec![0, q as u64, w, p]);
                i += 1;
            }
            let c: Vec<u64> = o.wide.slices(q).flat_map(|c| c.iter().copied()).collect();
            let mut i = 0;
            while i < c.len() {
                let w = c[i + 2];
                let p = if w & 1 != 0 { c[i + 3] } else { 0 };
                v.push(vec![1, q as u64, c[i], c[i + 1], w, p]);
                i += if w & 1 != 0 { 4 } else { 3 };
            }
        }
        v.sort();
        v
    }

    #[test]
    fn matches_brute_force() {
        let mut r = R(0x1234_5678_9abc_def1);
        for round in 0..40 {
            let b2 = 8 + (round % 4) as u32;
            let nkeys = 1 + r.next() % 3000;
            let ncopies = r.next() % 6000;
            let keys: Vec<Key> = (0..nkeys)
                .map(|_| match r.next() % 10 {
                    0 => Key { a: r.next(), b: r.next(), kind: 1 },
                    1 => Key { a: (r.next() & 0xffff) | 1 << 63, b: 0, kind: 2 },
                    _ => Key { a: r.next() & M32, b: 0, kind: 0 },
                })
                .collect();
            let mut copies: Vec<C> = Vec::new();
            for _ in 0..ncopies {
                let k = keys[(r.next() % nkeys) as usize];
                // Mostly whole seconds in the field; one in eight out of it (before the base, in
                // 1970, 2^23 - 1 s and more after it), compared in milliseconds.
                let ing = if r.next() % 8 != 0 {
                    (r.next() % 50) * 1000
                } else {
                    [(-31i64 * 86_400_000) as u64, (-1000i64) as u64, (-1_700_000_000_000i64) as u64, ((1 << 23) - 1) * 1000, ((1 << 23) - 1) * 1000 + 1, 4_000_000_000_000][(r.next() % 6) as usize]
                };
                let surv = u64::from(r.next() % 4 != 0);
                let purch = surv & u64::from(r.next() % 5 == 0);
                let ts = r.next() & TS;
                let uk = if r.next() % 8 == 0 { 1 + r.next() % 2 } else { 0 };
                let (ua, ub) = if uk == 0 { (r.next() & ((1 << 40) - 1), 0) } else { (r.next(), r.next()) };
                let pay = 1 + r.next() % 1000;
                // Copies with equal ingest are identical (the rules): reuse the first such copy.
                if let Some(c) = copies.iter().find(|c| c.0 == k && c.1 == ing).copied() {
                    copies.push(c);
                } else if surv == 0 {
                    copies.push((k, ing, 0, 0, 0, 0, 0, 0, 0));
                } else {
                    copies.push((k, ing, surv, purch, ts, uk, ua, ub, if purch != 0 { pay } else { 0 }));
                }
            }
            // Brute force: per key the smallest ingest; it survives when alive.
            let mut best: Vec<C> = Vec::new();
            for c in &copies {
                match best.iter_mut().find(|b| b.0 == c.0) {
                    Some(b) => {
                        if (c.1 as i64) < (b.1 as i64) {
                            *b = *c;
                        }
                    }
                    None => best.push(*c),
                }
            }
            let mut want = Out2 { short: Out::new_in(1 << b2, 0), wide: Out::new_in(1 << b2, 0), b2 };
            for b in &best {
                if b.2 != 0 {
                    want.emit(b.5, b.6, b.7, b.4, b.3, b.8);
                }
            }
            want.finish();
            // Encode into the two lanes and cut each into chunks at random points.
            let (mut plain, mut rare) = (Vec::new(), Vec::new());
            for c in &copies {
                encode(c, &mut plain, &mut rare);
            }
            let rchunks = cut_groups(&rare, &mut r);
            let (pc, rc) = (cut(&plain, &mut r), rchunks.iter().map(|c| c.as_slice()).collect::<Vec<_>>());
            let mut t = Tables::new();
            let mut got = Out2 { short: Out::new_in(1 << b2, 0), wide: Out::new_in(1 << b2, 0), b2 };
            partition(&pc, &rc, &mut t, &mut got);
            got.finish();
            assert!(words(&got, b2) == words(&want, b2), "round {round}");
        }
    }
}
