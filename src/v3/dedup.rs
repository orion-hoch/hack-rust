//! Move-1 partitions, one task each: exact dedup by event key, then move 2.
//!
//! Tables, reused by a worker across partitions; every slot is zero between partitions (the
//! survivor scan zeroes what it read), so nothing is cleared or generation-tagged:
//! - SHORT slot `[w0 ^ X0, w1]` (16 bytes, four to a 64-byte bucket, probed bucket by bucket with
//!   one vector compare) and LONG slot `[a, b, w2 ^ ING, w3]` (32 bytes, linear probing) hold
//!   the winning copy itself, with its ingest field complemented, so an empty slot is zero. A
//!   record never stores 0 there: its ingest is below 2^23 - 1 (RECORDS v3), checked per record.
//! - A special (ingest that does not fit, compared in milliseconds) is stored with the
//!   complemented field 0 and bit 63 set: `[key | 1 << 63, index]`, `[a, b, 1 << 63, index]`.
//! - Smallest ingest wins; equal ingest means identical copies, the first stays. A dead copy
//!   shadows like a live one (L: dedup by strict-minimum ingest is the same for every reading
//!   order and every partition function of the key).
//! - A purchase side record joins its winner by key and ingest, first match only (a bit per
//!   slot), and every surviving purchase must find one (else the run falls back).
//! - Survivors: one branch-free scan of the slots collects the winners' survivor words densely;
//!   then person lookups (prefetched `DP` survivors ahead) and the move-2 append. Move-2 event
//!   records fill a 64-byte staging line per person partition, written out with non-temporal
//!   stores (no read for ownership of the target line).
//! - Table size: 1.5 records (SHORT) or 3 records (LONG, linear probing wants a lower load)
//!   rounded up to a power of two, at most the worker's table. An
//!   outlier partition (many copies of few events) starts at the worker's size and grows only
//!   with its distinct keys, up to a cap; past the cap, or past a global budget, the run falls
//!   back. So copies of one event take one slot however many there are.
//!
//! Move-2 records: 8 bytes `person << 32 | ts << 1 | purchase`; purchases also
//! `[person << 32 | ts, category | price << 8]`.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use super::persons::People;
use super::sink::{Special, mix};
use super::super::v3::Abort;
use crate::fast::sink::Bins;

const KEY40: u64 = (1 << 40) - 1;
const ING: u64 = (1 << 23) - 1;
/// SHORT word 0 is stored XOR this: its ingest field complemented.
const X0: u64 = ING << 40;
const TOP: u64 = 1 << 63;
const FIB: u64 = 0x9e37_79b9_7f4a_7c15;
/// Records ahead whose slot is prefetched.
const DT: usize = 16;
/// Survivors ahead whose person entry is prefetched (box 1: 16 costs 40% more than 32, 64 is 2-6%
/// better than 32).
const DP: usize = 64;
/// Largest tables a worker may grow to (slots); past them the run falls back. Test builds (feature
/// `plant`) cap them low so that the growth, the cap and the fallback can be exercised by data.
const MAX_SHORT: usize = if cfg!(feature = "plant") { 1 << 16 } else { 1 << 24 };
const MAX_LONG: usize = if cfg!(feature = "plant") { 1 << 15 } else { 1 << 23 };
/// Bytes of tables grown past their first size, all workers together; past it the run falls back.
static GROWN: AtomicUsize = AtomicUsize::new(0);
const GROWN_LIMIT: usize = 3 << 30;

#[inline(always)]
fn slot_of(h: u64, shift: u32) -> usize {
    (h.wrapping_mul(FIB) >> shift) as usize
}

#[inline(always)]
fn long_hash(a: u64, b: u64) -> u64 {
    a ^ b.rotate_left(31)
}

#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch never faults, whatever the address.
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

/// log2 of the slots for `n` records: `n * num / 2` rounded up to a power of two, 16 to `cap`.
fn bits_for(n: usize, num: usize, cap: usize) -> u32 {
    (n * num / 2).next_power_of_two().clamp(16, cap).trailing_zeros()
}

/// Lane `l` of a bucket compare holds slot `lane_slot(l)` (the vector unpack takes slots 0, 2, 1, 3).
#[inline(always)]
fn lane_slot(l: u32) -> usize {
    (((l & 1) << 1) | (l >> 1)) as usize
}

/// Bucket `b` of a SHORT table (four slots, one 64-byte line) against `key`: bit `l` of the first
/// mask = the slot in lane `l` holds the key; of the second = that slot is empty.
/// SAFETY: `4 * b + 3 < slots` of the table at `tab`, which is 64-byte aligned.
#[inline(always)]
unsafe fn probe4(tab: *const [u64; 2], b: usize, key: u64) -> (u32, u32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // SAFETY: the bucket is one aligned line inside the table (caller); AVX2 is enabled.
    unsafe {
        use core::arch::x86_64::*;
        let p = tab.add(4 * b) as *const __m256i;
        let w = _mm256_unpacklo_epi64(_mm256_load_si256(p), _mm256_load_si256(p.add(1)));
        let e = _mm256_cmpeq_epi64(w, _mm256_setzero_si256());
        let m = _mm256_cmpeq_epi64(_mm256_and_si256(w, _mm256_set1_epi64x(KEY40 as i64)), _mm256_set1_epi64x(key as i64));
        let em = _mm256_movemask_pd(_mm256_castsi256_pd(e)) as u32;
        let mm = _mm256_movemask_pd(_mm256_castsi256_pd(m)) as u32;
        // A zero word has key bits 0: an empty slot "matches" key 0, so empties are removed.
        (mm & !em, em)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    // SAFETY: the four slots of the bucket are inside the table (caller).
    unsafe {
        let (mut mm, mut em) = (0u32, 0u32);
        for l in 0..4u32 {
            let w = (*tab.add(4 * b + lane_slot(l)))[0];
            em |= u32::from(w == 0) << l;
            mm |= u32::from(w != 0 && w & KEY40 == key) << l;
        }
        (mm, em)
    }
}

/// One worker's reusable tables.
pub struct Tables {
    /// SHORT slots, four to a line (the line type gives the 64-byte alignment of the buckets).
    s: Vec<Line>,
    l: Vec<[u64; 4]>,
    /// Purchase-matched bits, one per slot; set bits are listed in `marked` and cleared after.
    mark_s: Vec<u64>,
    mark_l: Vec<u64>,
    marked: Vec<u32>,
    /// The partition's winners' survivor words (dense); never shorter than s.len() + l.len().
    surv: Vec<u64>,
    /// The partition's winning purchases: (survivor word, category | price << 8).
    buys: Vec<(u64, u32)>,
    /// Special winners that survive: (token, ts, buy).
    spec: Vec<(u32, u32, u32)>,
    /// A failed check left the tables in an unknown state: every later call refuses.
    poisoned: bool,
}

impl Tables {
    pub fn new(short_slots: usize, long_slots: usize) -> Self {
        let s = short_slots.next_power_of_two().clamp(16, MAX_SHORT);
        let l = long_slots.next_power_of_two().clamp(16, MAX_LONG);
        Tables {
            s: vec![Line([0; 8]); s / 4],
            l: vec![[0; 4]; l],
            mark_s: vec![0; s.div_ceil(64)],
            mark_l: vec![0; l.div_ceil(64)],
            marked: Vec::new(),
            surv: vec![0; s + l],
            buys: Vec::new(),
            spec: Vec::new(),
            poisoned: false,
        }
    }

    /// SHORT slots of the table.
    #[inline(always)]
    fn s_len(&self) -> usize {
        self.s.len() * 4
    }

    #[inline(always)]
    fn s_ptr(&mut self) -> *mut [u64; 2] {
        self.s.as_mut_ptr() as *mut [u64; 2]
    }

    /// Doubles the SHORT table (slots in use `1 << bits`): every live slot is inserted again.
    /// Err past the cap or the budget.
    #[cold]
    #[inline(never)]
    fn grow_short(&mut self, bits: &mut u32) -> Result<(), Abort> {
        let new = 2usize << *bits;
        if new > MAX_SHORT || GROWN.fetch_add(new * 16, Relaxed) + new * 16 > GROWN_LIMIT {
            return Err(Abort("a dedup table would pass its size limit".into()));
        }
        let mut t: Vec<Line> = vec![Line([0; 8]); new.max(self.s_len()) / 4];
        let tp = t.as_mut_ptr() as *mut [u64; 2];
        let shift = 64 - (*bits + 1 - 2);
        let bmask = new / 4 - 1;
        let old = self.s_ptr();
        for i in 0..1usize << *bits {
            // SAFETY: i < 2^bits <= old slots.
            let s = unsafe { *old.add(i) };
            if s[0] != 0 {
                let key = s[0] & KEY40;
                let mut b = slot_of(key, shift);
                loop {
                    // SAFETY: b <= bmask, so 4 b + 3 < new <= slots of t.
                    let (_, em) = unsafe { probe4(tp, b, key) };
                    if em != 0 {
                        // SAFETY: as above.
                        unsafe { *tp.add(4 * b + lane_slot(em.trailing_zeros())) = s };
                        break;
                    }
                    // Terminates: the new table holds at most 3/8 of its slots.
                    b = (b + 1) & bmask;
                }
            }
        }
        self.s = t;
        self.mark_s = vec![0; self.s_len().div_ceil(64)];
        self.surv = vec![0; self.s_len() + self.l.len()];
        *bits += 1;
        Ok(())
    }

    #[cold]
    #[inline(never)]
    fn grow_long(&mut self, bits: &mut u32) -> Result<(), Abort> {
        let new = 2usize << *bits;
        if new > MAX_LONG || GROWN.fetch_add(new * 32, Relaxed) + new * 32 > GROWN_LIMIT {
            return Err(Abort("a dedup table would pass its size limit".into()));
        }
        let mut t: Vec<[u64; 4]> = vec![[0; 4]; new.max(self.l.len())];
        let shift = 64 - (*bits + 1);
        let mask = new - 1;
        for s in &self.l[..1 << *bits] {
            if s[2] != 0 {
                let mut i = slot_of(long_hash(s[0], s[1]), shift);
                while t[i][2] != 0 {
                    i = (i + 1) & mask;
                }
                t[i] = *s;
            }
        }
        self.l = t;
        self.mark_l = vec![0; self.l.len().div_ceil(64)];
        self.surv = vec![0; self.s_len() + self.l.len()];
        *bits += 1;
        Ok(())
    }
}

/// Move-2 event records: per person partition a 64-byte staging line; a full line goes to the
/// partition's current chunk with non-temporal stores. Chunks are runs of lines bumped from
/// large blocks owned here; nothing moves or is copied again.
pub struct Out {
    stage: Vec<Line>,
    fill: Vec<u32>,
    cur: Vec<*mut Line>,
    end: Vec<*mut Line>,
    /// Finished chunks per partition: (first record, records).
    full: Vec<Vec<(*const u64, u32)>>,
    blocks: Vec<Vec<u64>>,
    bump: *mut Line,
    bump_end: *mut Line,
    block_lines: usize,
    chunk_lines: usize,
    /// Partitions - 1 (a power of two minus one): every per-partition array has mask + 1 entries.
    mask: usize,
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Line([u64; 8]);

// SAFETY: the raw pointers point into `blocks`, owned by the struct; moving it moves that
// ownership (the blocks' heap memory does not move).
unsafe impl Send for Out {}
// SAFETY: shared references only read finished chunks (`slices`, after `finish`).
unsafe impl Sync for Out {}

/// Copies one line with non-temporal stores (plain copy where there are none).
#[inline(always)]
unsafe fn stream_line(dst: *mut Line, src: *const Line) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    // SAFETY: both pointers are 64-byte aligned lines (Line's alignment); AVX is enabled.
    unsafe {
        use core::arch::x86_64::{__m256i, _mm256_load_si256, _mm256_stream_si256};
        let (s, d) = (src as *const __m256i, dst as *mut __m256i);
        _mm256_stream_si256(d, _mm256_load_si256(s));
        _mm256_stream_si256(d.add(1), _mm256_load_si256(s.add(1)));
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx")))]
    // SAFETY: both pointers are 64-byte aligned lines.
    unsafe {
        use core::arch::x86_64::{__m128i, _mm_load_si128, _mm_stream_si128};
        let (s, d) = (src as *const __m128i, dst as *mut __m128i);
        for k in 0..4 {
            _mm_stream_si128(d.add(k), _mm_load_si128(s.add(k)));
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    // SAFETY: both pointers are valid lines.
    unsafe {
        *dst = *src
    };
}

impl Out {
    /// `n` partitions, blocks of about `block` records, chunks of about `chunk` records.
    pub fn new(n: usize, block: usize, chunk: usize) -> Self {
        let chunk_lines = chunk.div_ceil(8).max(1);
        let n = n.next_power_of_two();
        Out {
            mask: n - 1,
            stage: vec![Line([0; 8]); n],
            fill: vec![0; n],
            cur: vec![std::ptr::null_mut(); n],
            end: vec![std::ptr::null_mut(); n],
            full: (0..n).map(|_| Vec::new()).collect(),
            blocks: Vec::new(),
            bump: std::ptr::null_mut(),
            bump_end: std::ptr::null_mut(),
            block_lines: block.next_power_of_two().div_ceil(8).max(2 * chunk_lines),
            chunk_lines,
        }
    }

    /// Appends `x` to partition `q & mask` (callers pass q below the partition count).
    #[inline(always)]
    pub fn push(&mut self, q: usize, x: u64) {
        let q = q & self.mask;
        // SAFETY: q <= mask < fill.len() = stage.len() = mask + 1; the slot c & 7 < 8 (fill never
        // holds more than 7: it is reset when it would reach 8).
        unsafe {
            let f = self.fill.as_mut_ptr().add(q);
            let c = (*f & 7) as usize;
            *(*self.stage.as_mut_ptr().add(q)).0.as_mut_ptr().add(c) = x;
            if c == 7 {
                *f = 0;
                self.flush(q);
            } else {
                *f = c as u32 + 1;
            }
        }
    }

    /// SAFETY: q <= mask.
    #[inline(always)]
    unsafe fn flush(&mut self, q: usize) {
        // SAFETY: q <= mask < cur.len() = end.len() = stage.len().
        unsafe {
            if *self.cur.get_unchecked(q) == *self.end.get_unchecked(q) {
                self.refill(q);
            }
            // cur[q] lies in [chunk start, chunk end) of a chunk in a live block (refill
            // guarantees a free line); stage[q] is a line of this struct.
            let c = self.cur.get_unchecked_mut(q);
            stream_line(*c, self.stage.as_ptr().add(q));
            *c = c.add(1);
        }
    }

    #[cold]
    #[inline(never)]
    fn refill(&mut self, q: usize) {
        if !self.cur[q].is_null() {
            // SAFETY: end[q] is chunk_lines lines past the start of the same chunk.
            let start = unsafe { self.end[q].sub(self.chunk_lines) };
            self.full[q].push((start as *const u64, (self.chunk_lines * 8) as u32));
        }
        if self.bump == self.bump_end {
            // Blocks are plain u64 vectors of a power-of-two size, like every other big buffer
            // of the run, so the allocator can hand back a block freed earlier (no fresh pages);
            // the lines start at the first 64-byte boundary (large blocks are page-aligned, so
            // nothing is lost in practice; at most one line otherwise).
            let mut v: Vec<u64> = Vec::with_capacity(self.block_lines * 8);
            let p = v.as_mut_ptr();
            let a = (p as usize).next_multiple_of(64);
            // Whole chunks only: block_lines >= 2 chunk_lines and the alignment costs at most one
            // line, so lines >= chunk_lines > 0.
            let lines = (self.block_lines * 8 - (a - p as usize) / 8) / 8 / self.chunk_lines * self.chunk_lines;
            self.bump = a as *mut Line;
            // SAFETY: [a, a + lines * 64) lies inside the block's capacity; lines is a multiple
            // of chunk_lines.
            self.bump_end = unsafe { self.bump.add(lines) };
            self.blocks.push(v);
        }
        self.cur[q] = self.bump;
        // SAFETY: at least chunk_lines lines remain in the block (blocks are whole chunks).
        unsafe {
            self.end[q] = self.bump.add(self.chunk_lines);
            self.bump = self.bump.add(self.chunk_lines);
        }
    }

    /// Writes out the partly filled lines and closes every chunk; call once, after the last push.
    pub fn finish(&mut self) {
        for q in 0..self.cur.len() {
            let c = self.fill[q] as usize;
            if c > 0 {
                if self.cur[q] == self.end[q] {
                    self.refill(q);
                }
                // SAFETY: as in flush; a plain store of the partial line.
                unsafe {
                    *self.cur[q] = self.stage[q];
                    self.cur[q] = self.cur[q].add(1);
                }
            }
            if !self.cur[q].is_null() {
                // SAFETY: end[q] is chunk_lines lines past the start of the chunk; cur[q] is
                // within it, and the last written line holds c records (8 when c is 0).
                let start = unsafe { self.end[q].sub(self.chunk_lines) };
                let lines = (self.cur[q] as usize - start as usize) / 64;
                let n = if c > 0 { (lines - 1) * 8 + c } else { lines * 8 };
                if n > 0 {
                    self.full[q].push((start as *const u64, n as u32));
                }
                self.cur[q] = std::ptr::null_mut();
                self.end[q] = std::ptr::null_mut();
            }
            self.fill[q] = 0;
        }
        #[cfg(target_arch = "x86_64")]
        // SAFETY: a store fence: the non-temporal stores are visible before anything later.
        unsafe {
            core::arch::x86_64::_mm_sfence()
        };
    }

    /// The finished chunks of partition `q` (after `finish`).
    pub fn slices(&self, q: usize) -> impl Iterator<Item = &[u64]> + '_ {
        // SAFETY: each (ptr, len) is a written prefix of a chunk inside a block owned by self.
        self.full[q].iter().map(|&(p, n)| unsafe { std::slice::from_raw_parts(p, n as usize) })
    }
}

pub struct Move2 {
    pub ev: Out,
    pub buys: Bins<[u64; 2]>,
    pub bits: u32,
}

impl Move2 {
    pub fn new(bits: u32, hint: usize) -> Self {
        let parts = 1usize << bits;
        Move2 {
            ev: Out::new(parts, hint.clamp(1 << 14, 1 << 21), (hint / parts / 2).clamp(64, 512)),
            buys: Bins::with_chunk(parts, 1 << 12, (hint / parts / 64).clamp(4, 64)),
            bits,
        }
    }

    #[inline(always)]
    pub fn part(&self, person: u32) -> usize {
        (mix(u64::from(person) | 0xabcd_0000_0000_0000) >> (64 - self.bits)) as usize
    }

    /// One survivor: `w` is its survivor word (token | ts << 32 | purchase << 63).
    #[inline(always)]
    fn push_word(&mut self, person: u32, w: u64) {
        let q = self.part(person);
        // ts << 1 | purchase: the high half of w rotated left by one.
        self.ev.push(q, (u64::from(person) << 32) | u64::from(((w >> 32) as u32).rotate_left(1)));
    }

    #[inline(always)]
    fn push(&mut self, person: u32, ts: u32, buy: u32) {
        let q = self.part(person);
        self.ev.push(q, (u64::from(person) << 32) | (u64::from(ts) << 1) | u64::from(buy != 0));
        if buy != 0 {
            self.buys.push(q, [(u64::from(person) << 32) | u64::from(ts), u64::from(buy)]);
        }
    }
}

pub struct Ctx3<'p> {
    pub people: &'p People,
    pub ingest_base_ms: i64,
}

/// For a 4-bit lane mask: 32-bit permute indexes that move the 64-bit lanes whose bit is set to
/// the front, in order.
const fn compress_lut() -> [[u32; 8]; 16] {
    let mut t = [[0u32; 8]; 16];
    let mut m = 0;
    while m < 16 {
        let (mut k, mut l) = (0, 0);
        while l < 4 {
            if (m >> l) & 1 == 1 {
                t[m][2 * k] = 2 * l as u32;
                t[m][2 * k + 1] = 2 * l as u32 + 1;
                k += 1;
            }
            l += 1;
        }
        m += 1;
    }
    t
}
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
static COMPRESS: [[u32; 8]; 16] = compress_lut();

/// Survivor scan of the first `lines` lines of a SHORT table without specials: appends each live
/// winner's survivor word at `sv` (dense), zeroes the lines, returns (survivors, purchases).
/// SAFETY: `lines` lines at `tab`; `sv` has room for 4 * lines words.
#[inline(always)]
unsafe fn scan_short(tab: *mut Line, lines: usize, sv: *mut u64) -> (usize, usize) {
    let (mut n, mut purchases) = (0usize, 0usize);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // SAFETY: each line is an aligned 64-byte line of the table; the store at sv + n writes four
    // words below 4 * (lines so far + 1) <= 4 * lines (n counts live slots of earlier lines).
    unsafe {
        use core::arch::x86_64::*;
        let z = _mm256_setzero_si256();
        for i in 0..lines {
            let p = tab.add(i) as *mut __m256i;
            let (v0, v1) = (_mm256_load_si256(p), _mm256_load_si256(p.add(1)));
            // Lanes: slots 0, 2, 1, 3 (order does not matter here).
            let w0 = _mm256_unpacklo_epi64(v0, v1);
            let w1 = _mm256_unpackhi_epi64(v0, v1);
            // Alive: bit 63 of word 0 (0 for an empty slot or a dead winner).
            let live = _mm256_movemask_pd(_mm256_castsi256_pd(w0)) as usize;
            let buy = _mm256_movemask_pd(_mm256_castsi256_pd(w1)) as usize & live;
            let idx = _mm256_loadu_si256(COMPRESS.as_ptr().add(live) as *const __m256i);
            _mm256_storeu_si256(sv.add(n) as *mut __m256i, _mm256_permutevar8x32_epi32(w1, idx));
            _mm256_store_si256(p, z);
            _mm256_store_si256(p.add(1), z);
            n += live.count_ones() as usize;
            purchases += buy.count_ones() as usize;
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    // SAFETY: slot i < 4 * lines lies in the table; n <= i < 4 * lines.
    unsafe {
        let tab = tab as *mut [u64; 2];
        for i in 0..4 * lines {
            let s = *tab.add(i);
            *tab.add(i) = [0; 2];
            *sv.add(n) = s[1];
            let live = (s[0] >> 63) as usize;
            n += live;
            purchases += live & (s[1] >> 63) as usize;
        }
    }
    (n, purchases)
}

/// One SHORT record into the table (bucket probing). Returns true when its key is new.
/// SAFETY: `tab` is a 64-byte aligned SHORT table of at least 4 (bmask + 1) slots, bmask + 1 =
/// 2^(64 - bshift) buckets, with at most 3/4 of those slots used (so an empty slot exists).
#[inline(always)]
unsafe fn put_short(tab: *mut [u64; 2], bshift: u32, bmask: usize, r: [u64; 2]) -> bool {
    let key = r[0] & KEY40;
    let mut b = slot_of(key, bshift);
    loop {
        // SAFETY: b <= bmask: a product shifted right by bshift, then only ever masked.
        let (mm, em) = unsafe { probe4(tab, b, key) };
        if mm != 0 {
            // The key's slot: the smaller ingest wins (the complement touches bits 40-62 only).
            // SAFETY: 4 b + lane < 4 (bmask + 1) = slots.
            let s = unsafe { &mut *tab.add(4 * b + lane_slot(mm.trailing_zeros())) };
            if (r[0] >> 40) & ING < ((s[0] >> 40) & ING) ^ ING {
                *s = [r[0] ^ X0, r[1]];
            }
            return false;
        }
        if em != 0 {
            // No earlier bucket holds the key (it would have been put in the first bucket with
            // room, and slots never empty during a partition): the key is new.
            // SAFETY: as above.
            unsafe { *tab.add(4 * b + lane_slot(em.trailing_zeros())) = [r[0] ^ X0, r[1]] };
            return true;
        }
        // A full bucket: the next one. Terminates: an empty slot exists (at most 3/4 used).
        b = (b + 1) & bmask;
    }
}

/// Inserts SHORT records; returns (the least complemented ingest field seen, so 0 means a record
/// had the impossible field 2^23 - 1; new keys). `GROW`: the partition may hold more distinct keys
/// than 3/4 of the slots, so the table doubles before passing that load. Without `GROW` the caller
/// has shown that it cannot.
#[inline(always)]
fn insert_short<const GROW: bool>(t: &mut Tables, bits: &mut u32, chunks: &[&[[u64; 2]]]) -> Result<(u64, usize), Abort> {
    let mut low = X0;
    let mut distinct = 0usize;
    for c in chunks {
        let n = c.len();
        let p = c.as_ptr();
        if !GROW {
            let bshift = 64 - (*bits - 2);
            let bmask = (1usize << (*bits - 2)) - 1;
            let tab = t.s_ptr();
            let pre = n.saturating_sub(DT);
            for j in 0..pre {
                // SAFETY: j + DT < n = c.len().
                let ka = unsafe { (*p.add(j + DT))[0] } & KEY40;
                // Address only (wrapping): a prefetch never faults.
                prefetch(tab.wrapping_add(4 * slot_of(ka, bshift)));
                // SAFETY: j < n.
                let r = unsafe { *p.add(j) };
                low = low.min(!r[0] & X0);
                // SAFETY: bits <= log2(t.s_len()), and the caller's bound keeps 3/4 free.
                distinct += usize::from(unsafe { put_short(tab, bshift, bmask, r) });
            }
            for j in pre..n {
                // SAFETY: j < n.
                let r = unsafe { *p.add(j) };
                low = low.min(!r[0] & X0);
                // SAFETY: as above.
                distinct += usize::from(unsafe { put_short(tab, bshift, bmask, r) });
            }
            continue;
        }
        for j in 0..n {
            let bshift = 64 - (*bits - 2);
            let bmask = (1usize << (*bits - 2)) - 1;
            let tab = t.s_ptr();
            if j + DT < n {
                // SAFETY: j + DT < n.
                let ka = unsafe { (*p.add(j + DT))[0] } & KEY40;
                prefetch(tab.wrapping_add(4 * slot_of(ka, bshift)));
            }
            // SAFETY: j < n.
            let r = unsafe { *p.add(j) };
            low = low.min(!r[0] & X0);
            // SAFETY: bits <= log2(t.s_len()); growth below keeps at most 3/4 used.
            if unsafe { put_short(tab, bshift, bmask, r) } {
                distinct += 1;
                if distinct * 4 > 3 << *bits {
                    t.grow_short(bits)?;
                }
            }
        }
    }
    Ok((low, distinct))
}

#[inline(always)]
fn insert_long<const GROW: bool>(t: &mut Tables, bits: &mut u32, chunks: &[&[[u64; 4]]]) -> Result<(u64, usize), Abort> {
    let mut bad = 0u64;
    let mut distinct = 0usize;
    for c in chunks {
        let n = c.len();
        let p = c.as_ptr();
        let mut j = 0;
        while j < n {
            let shift = 64 - *bits;
            let mask = (1usize << *bits) - 1;
            let tab = t.l.as_mut_ptr();
            let mut grow = false;
            while j < n && !grow {
                if j + DT < n {
                    // SAFETY: j + DT < n.
                    let ra = unsafe { &*p.add(j + DT) };
                    prefetch(tab.wrapping_add(slot_of(long_hash(ra[0], ra[1]), shift)));
                }
                // SAFETY: j < n.
                let r = unsafe { *p.add(j) };
                j += 1;
                bad |= u64::from(r[2] & ING == ING);
                let mut i = slot_of(long_hash(r[0], r[1]), shift);
                loop {
                    // SAFETY: i <= mask < t.l.len(), as in insert_short.
                    let s = unsafe { &mut *tab.add(i) };
                    if s[2] == 0 {
                        *s = [r[0], r[1], r[2] ^ ING, r[3]];
                        distinct += 1;
                        grow = GROW && distinct * 4 > 3 << *bits;
                        break;
                    }
                    if s[0] == r[0] && s[1] == r[1] {
                        if r[2] & ING < (s[2] & ING) ^ ING {
                            *s = [r[0], r[1], r[2] ^ ING, r[3]];
                        }
                        break;
                    }
                    // Terminates as in insert_short.
                    i = (i + 1) & mask;
                }
            }
            if grow {
                t.grow_long(bits)?;
            }
        }
    }
    Ok((bad, distinct))
}

/// Specials, after every plain record: compared in milliseconds against the slot's copy.
/// `sd`, `ld`: keys in each table so far; a table doubles before passing 3/4.
fn insert_specials(t: &mut Tables, sb: &mut u32, lb: &mut u32, sd: &mut usize, ld: &mut usize, specials: &[Special], base: i64) -> Result<(), Abort> {
    for (si, sp) in specials.iter().enumerate() {
        match sp.key {
            super::ids::Key::Short(key) => {
                if (*sd + 1) * 4 > 3 << *sb {
                    t.grow_short(sb)?;
                }
                let bmask = (1usize << (*sb - 2)) - 1;
                let mut b = slot_of(key, 64 - (*sb - 2));
                let mine = [key | TOP, si as u64];
                let tab = t.s_ptr();
                loop {
                    // SAFETY: b <= bmask: 4 b + 3 < 2^sb <= t.s_len().
                    let (mm, em) = unsafe { probe4(tab, b, key) };
                    if mm != 0 {
                        // SAFETY: as above.
                        let s = unsafe { &mut *tab.add(4 * b + lane_slot(mm.trailing_zeros())) };
                        let cur = if (s[0] >> 40) & ING == 0 {
                            specials[s[1] as usize].ingest_ms
                        } else {
                            base + ((((s[0] >> 40) & ING) ^ ING) as i64) * 1000
                        };
                        if sp.ingest_ms < cur {
                            *s = mine;
                        }
                        break;
                    }
                    if em != 0 {
                        // SAFETY: as above.
                        unsafe { *tab.add(4 * b + lane_slot(em.trailing_zeros())) = mine };
                        *sd += 1;
                        break;
                    }
                    // Terminates: at most 3/4 used (checked above).
                    b = (b + 1) & bmask;
                }
            }
            super::ids::Key::Long(a, b) => {
                if (*ld + 1) * 4 > 3 << *lb {
                    t.grow_long(lb)?;
                }
                let mask = (1usize << *lb) - 1;
                let mut i = slot_of(long_hash(a, b), 64 - *lb);
                let mine = [a, b, TOP, si as u64];
                loop {
                    let s = t.l[i];
                    if s[2] == 0 {
                        t.l[i] = mine;
                        *ld += 1;
                        break;
                    }
                    if s[0] == a && s[1] == b {
                        let cur = if s[2] & ING == 0 {
                            specials[s[3] as usize].ingest_ms
                        } else {
                            base + (((s[2] & ING) ^ ING) as i64) * 1000
                        };
                        if sp.ingest_ms < cur {
                            t.l[i] = mine;
                        }
                        break;
                    }
                    i = (i + 1) & mask;
                }
            }
            super::ids::Key::Foreign => return Err(Abort("a FOREIGN event ID among the keyed specials".into())),
        }
    }
    Ok(())
}

/// Dedups one partition and appends its survivors to `m2`. Returns the number of survivors.
#[allow(clippy::too_many_arguments)]
pub fn partition(
    short: &[&[[u64; 2]]],
    long: &[&[[u64; 4]]],
    pshort: &[&[[u64; 2]]],
    plong: &[&[[u64; 4]]],
    specials: &[Special],
    cx: &Ctx3,
    t: &mut Tables,
    m2: &mut Move2,
) -> Result<usize, Abort> {
    if t.poisoned {
        return Err(Abort("dedup tables left in an unknown state by an earlier failure".into()));
    }
    t.poisoned = true;
    let ns: usize = short.iter().map(|c| c.len()).sum::<usize>() + specials.len();
    let nl: usize = long.iter().map(|c| c.len()).sum::<usize>() + specials.len();
    let mut sb = bits_for(ns, 3, t.s_len());
    let mut lb = bits_for(nl, 6, t.l.len());
    // Every record (specials included) a new key still leaves a quarter of the slots empty:
    // no growth check needed.
    let (low_s, mut sd) = if ns * 4 <= 3 << sb { insert_short::<false>(t, &mut sb, short)? } else { insert_short::<true>(t, &mut sb, short)? };
    let (bad_l, mut ld) = if nl * 4 <= 3 << lb { insert_long::<false>(t, &mut lb, long)? } else { insert_long::<true>(t, &mut lb, long)? };
    if low_s == 0 || bad_l != 0 {
        return Err(Abort("an event record with an ingest field of 2^23 - 1".into()));
    }
    insert_specials(t, &mut sb, &mut lb, &mut sd, &mut ld, specials, cx.ingest_base_ms)?;
    let (s_slots, l_slots) = (1usize << sb, 1usize << lb);
    let (b_shift, l_shift) = (64 - (sb - 2), 64 - lb);
    let (b_mask, l_mask) = (s_slots / 4 - 1, l_slots - 1);

    // Purchases of the winning copies: same key and same ingest, the first match only.
    t.buys.clear();
    // A purchase payload must fit 32 bits (category | price << 8, price below 2^24): anything
    // wider would be cut by the 32-bit payload, so the run falls back instead.
    let mut wide = 0u64;
    let tab = t.s_ptr();
    for c in pshort {
        for r in c.iter() {
            let key = r[0] & KEY40;
            let mut b = slot_of(key, b_shift);
            loop {
                // SAFETY: b <= b_mask: 4 b + 3 < s_slots <= t.s_len().
                let (mm, em) = unsafe { probe4(tab, b, key) };
                if mm != 0 {
                    let i = 4 * b + lane_slot(mm.trailing_zeros());
                    // SAFETY: i < s_slots.
                    let s = unsafe { *tab.add(i) };
                    // A plain winner (field not 0), alive, flagged purchase, same ingest.
                    if (s[0] >> 40) & ING != 0
                        && s[0] & TOP != 0
                        && s[1] & TOP != 0
                        && (s[0] ^ X0) & !TOP == r[0]
                        && t.mark_s[i / 64] >> (i % 64) & 1 == 0
                    {
                        t.mark_s[i / 64] |= 1 << (i % 64);
                        t.marked.push(i as u32);
                        wide |= r[1] >> 32;
                        t.buys.push((s[1], r[1] as u32));
                    }
                    break;
                }
                if em != 0 {
                    break;
                }
                // Terminates: at most 3/4 of the slots are used.
                b = (b + 1) & b_mask;
            }
        }
    }
    for c in plong {
        for r in c.iter() {
            let mut i = slot_of(long_hash(r[0], r[1]), l_shift);
            loop {
                let s = t.l[i];
                if s[2] == 0 {
                    break;
                }
                if s[0] == r[0] && s[1] == r[1] {
                    if s[2] & ING != 0
                        && s[2] & TOP != 0
                        && s[3] & TOP != 0
                        && (s[2] ^ ING) & ING == r[2]
                        && t.mark_l[i / 64] >> (i % 64) & 1 == 0
                    {
                        t.mark_l[i / 64] |= 1 << (i % 64);
                        t.marked.push(i as u32 | 1 << 31);
                        wide |= r[3] >> 32;
                        t.buys.push((s[3], r[3] as u32));
                    }
                    break;
                }
                i = (i + 1) & l_mask;
            }
        }
    }
    for &m in &t.marked {
        let i = (m & !(1 << 31)) as usize;
        if m >> 31 == 0 {
            t.mark_s[i / 64] = 0;
        } else {
            t.mark_l[i / 64] = 0;
        }
    }
    t.marked.clear();

    // Survivor scan: the winners' survivor words, densely; the slots are zeroed after.
    let mut n = 0usize;
    let mut purchases = 0usize;
    t.spec.clear();
    let sv = t.surv.as_mut_ptr();
    if specials.is_empty() {
        // SAFETY: s_slots <= t.s_len(); the scan writes sv[n..n + 4) with n <= 4 * (lines before
        // this one), so below s_slots <= t.surv.len() - l_slots; it zeroes what it read.
        let (k, p) = unsafe { scan_short(t.s.as_mut_ptr(), s_slots / 4, sv) };
        n += k;
        purchases += p;
        let tab = t.l.as_mut_ptr();
        for i in 0..l_slots {
            // SAFETY: i < l_slots <= t.l.len(); n <= s_slots + i < t.surv.len().
            unsafe {
                let s = *tab.add(i);
                *tab.add(i) = [0; 4];
                *sv.add(n) = s[3];
                let live = (s[2] >> 63) as usize;
                n += live;
                purchases += live & (s[3] >> 63) as usize;
            }
        }
    } else {
        let tab = t.s.as_ptr() as *const [u64; 2];
        for i in 0..s_slots {
            // SAFETY: i < s_slots <= t.s_len().
            let s = unsafe { *tab.add(i) };
            if s[0] & TOP == 0 {
                continue;
            }
            if (s[0] >> 40) & ING == 0 {
                let sp = &specials[s[1] as usize];
                if sp.surv {
                    t.spec.push((sp.token, sp.ts, sp.buy));
                }
                continue;
            }
            t.surv[n] = s[1];
            n += 1;
            purchases += (s[1] >> 63) as usize;
        }
        for i in 0..l_slots {
            let s = t.l[i];
            if s[2] & TOP == 0 {
                continue;
            }
            if s[2] & ING == 0 {
                let sp = &specials[s[3] as usize];
                if sp.surv {
                    t.spec.push((sp.token, sp.ts, sp.buy));
                }
                continue;
            }
            t.surv[n] = s[3];
            n += 1;
            purchases += (s[3] >> 63) as usize;
        }
    }
    if !specials.is_empty() {
        t.s[..s_slots / 4].fill(Line([0; 8]));
        t.l[..l_slots].fill([0; 4]);
    }
    t.poisoned = false;
    if purchases != t.buys.len() {
        return Err(Abort("a surviving purchase has no purchase record".into()));
    }
    if wide != 0 {
        return Err(Abort("a purchase payload wider than 32 bits (price of 2^24 or more)".into()));
    }

    // Move 2: person lookups prefetched DP survivors ahead.
    let people = cx.people;
    let sv = &t.surv[..n];
    for k in 0..n {
        if k + DP < n {
            people.prefetch(sv[k + DP] as u32);
        }
        let w = sv[k];
        m2.push_word(people.person_of(w as u32), w);
    }
    for &(w, buy) in &t.buys {
        let person = people.person_of(w as u32);
        let q = m2.part(person);
        m2.buys.push(q, [(u64::from(person) << 32) | ((w >> 32) & 0x7fff_ffff), u64::from(buy)]);
    }
    for &(token, ts, buy) in &t.spec {
        m2.push(people.person_of(token), ts, buy);
    }
    // The move-2 lines were written with non-temporal stores, which are weakly ordered: this
    // thread fences them before it reports the task done (other workers read them later).
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a store fence has no memory-safety preconditions.
    unsafe {
        core::arch::x86_64::_mm_sfence()
    };
    Ok(n + t.spec.len())
}

/// FOREIGN event IDs: copies only of each other; smallest ingest by bytes, then to move 2.
pub fn foreign(specials: &[&Special], cx: &Ctx3, m2: &mut Move2) -> usize {
    let mut best: std::collections::HashMap<&[u8], usize> = std::collections::HashMap::new();
    for (i, sp) in specials.iter().enumerate() {
        best.entry(sp.fid).and_modify(|b| if sp.ingest_ms < specials[*b].ingest_ms { *b = i }).or_insert(i);
    }
    let mut out = 0;
    for (_, i) in best {
        let sp = specials[i];
        if sp.surv {
            m2.push(cx.people.person_of(sp.token), sp.ts, sp.buy);
            out += 1;
        }
    }
    out
}
