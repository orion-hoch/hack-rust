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

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

use super::persons::People;
use super::sink::Special;
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
    /// Purchase side records met in the main streams (layout with purchases inline: surv 0 and a
    /// nonzero second word, `buy | PURCH_MARK`); joined with the side streams' records.
    pin_s: Vec<[u64; 2]>,
    /// Pre-filter bitmaps (a key's bit set once in `fa`; again: also in `fb`), the unique
    /// survivors' words, and the colliding records for the exact table.
    fm: Vec<[u64; 2]>,
    uniq: Vec<u64>,
    ex_s: Vec<[u64; 2]>,
    ex_l: Vec<[u64; 4]>,
    pin_l: Vec<[u64; 4]>,
    /// LONG records found in the SHORT stream (one-stream layout), for the LONG table.
    long_in: Vec<[u64; 4]>,
}

/// Marks an inline purchase side record's last word (a dead copy's is 0, a live one has surv).
const PURCH_MARK: u64 = 1 << 62;

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
            pin_s: Vec::new(),
            fm: Vec::new(),
            uniq: Vec::new(),
            ex_s: Vec::new(),
            ex_l: Vec::new(),
            pin_l: Vec::new(),
            long_in: Vec::new(),
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
    /// Chunks come from the fast path's 8 KB block class for this pool participant (blocks the
    /// worker freed first: no fresh pages), or from `blocks` when NO_ARENA.
    arena: usize,
    /// Bins whose chunks were handed back early (`free_bin`): skipped on drop.
    freed: Vec<AtomicBool>,
    /// Block-class mode: each bin's FIRST chunk is a 2 KB block (B2K), the rest 8 KB. Every
    /// worker opens a chunk in every bin while its first partition runs, before the stage has
    /// freed anything, so those opens are fresh memory (v4 I2 at 60M: 8 x 2048 x 2 opens = 229 MB
    /// of fresh 8 KB blocks); later chunks recycle the input's freed blocks.
    small: bool,
}

const NO_ARENA: usize = usize::MAX;

/// Lines of a bin's first chunk in block-class mode (one 2 KB block).
const SMALL_LINES: usize = 2048 / 64;

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Line([u64; 8]);

/// Bytes of a move-2 chunk in arena mode (the 8 KB block class).
const BLOCK_BYTES: usize = 8 << 10;

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
            arena: NO_ARENA,
            freed: (0..n).map(|_| AtomicBool::new(false)).collect(),
            small: false,
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

    /// The staging lines (8 words per partition), the fill counts and the partition mask, for a
    /// caller that writes `push` out in its own loop with these held in registers: the next free
    /// word of partition q is `stage[8 * q + (fill[q] & 7)]`; after storing the eighth word the
    /// caller zeroes `fill[q]` and calls `flush_line(q)`, otherwise it stores the count plus one.
    /// The pointers stay valid for the life of `self` (the two vectors never grow).
    #[inline(always)]
    pub(crate) fn raw_parts(&mut self) -> (*mut u64, *mut u32, usize) {
        (self.stage.as_mut_ptr().cast::<u64>(), self.fill.as_mut_ptr(), self.mask)
    }

    /// Flushes partition q's full staging line (see `raw_parts`).
    ///
    /// # Safety
    /// `q <= mask`, and the line holds eight words.
    #[inline(never)]
    pub(crate) unsafe fn flush_line(&mut self, q: usize) {
        // SAFETY: the caller's contract.
        unsafe { self.flush(q) }
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

    /// `n` partitions with chunks of one 8 KB block each (1,024 records) from the fast path's block
    /// class, for pool participant `w` (unique among the stage's concurrent users).
    pub fn new_in(n: usize, w: usize) -> Self {
        let b = super::mem::B8K.block_size();
        let mut o = Out::new(n, 0, b / 8);
        // Only when the class is attached with 8 KB blocks: a block is then exactly one chunk.
        if b == BLOCK_BYTES && w < rt::pool::MAX_WORKERS {
            o.arena = w;
            o.small = super::mem::B2K.block_size() == SMALL_LINES * 64;
        }
        o
    }

    /// Lines of bin q's current chunk (the first one is small in block-class mode).
    #[inline(always)]
    fn lines_of(&self, q: usize) -> usize {
        if self.small && self.full[q].is_empty() { SMALL_LINES } else { self.chunk_lines }
    }

    #[cold]
    #[inline(never)]
    fn refill(&mut self, q: usize) {
        if !self.cur[q].is_null() {
            let lines = self.lines_of(q);
            // SAFETY: end[q] is `lines` lines past the start of the same chunk.
            let start = unsafe { self.end[q].sub(lines) };
            self.full[q].push((start as *const u64, (lines * 8) as u32));
        }
        if self.arena != NO_ARENA {
            // SAFETY: only this worker uses index `arena` while the stage runs (new_in's caller);
            // a block is lines_of(q) lines (2 KB for a bin's first chunk, else BLOCK_BYTES),
            // aligned to its size (so to 64 bytes).
            let lines = self.lines_of(q);
            let p = unsafe { if lines == SMALL_LINES && self.small { super::mem::B2K.alloc(self.arena) } else { super::mem::B8K.alloc(self.arena) } } as *mut Line;
            self.cur[q] = p;
            // SAFETY: the block holds `lines` lines.
            self.end[q] = unsafe { p.add(lines) };
            return;
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
                // SAFETY: end[q] is lines_of(q) lines past the start of the chunk; cur[q] is
                // within it, and the last written line holds c records (8 when c is 0).
                let start = unsafe { self.end[q].sub(self.lines_of(q)) };
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

    /// Hands bin q's chunks back to the block class under participant `w` (block-class mode).
    /// SAFETY: after `finish`; bin q is never read again; `w` is the caller's own participant
    /// index; one call per bin (later ones do nothing).
    pub unsafe fn free_bin(&self, q: usize, w: usize) {
        if self.arena == NO_ARENA || self.freed[q].swap(true, Relaxed) {
            return;
        }
        for (i, &(p, _)) in self.full[q].iter().enumerate() {
            // SAFETY: each listed chunk is one whole block, B2K for a bin's first one when small,
            // else B8K (caller's contract for the rest).
            unsafe { self.free_chunk(w, i, p as *mut u8) };
        }
    }

    /// Chunk i of a bin back to its class under participant `w`.
    /// SAFETY: `p` is chunk i of a bin of this Out, dead from here on.
    #[inline(always)]
    unsafe fn free_chunk(&self, w: usize, i: usize, p: *mut u8) {
        unsafe {
            if i == 0 && self.small {
                super::mem::B2K.free(w, p);
            } else {
                super::mem::B8K.free(w, p);
            }
        }
    }

    /// The finished chunks of partition `q` (after `finish`).
    pub fn slices(&self, q: usize) -> impl Iterator<Item = &[u64]> + '_ {
        // SAFETY: each (ptr, len) is a written prefix of a chunk inside a block owned by self.
        self.full[q].iter().map(|&(p, n)| unsafe { std::slice::from_raw_parts(p, n as usize) })
    }
}

/// Block-class chunks go back to the class on drop, on every path (a decline to v3 included), so
/// whatever runs next takes them before fresh memory. Dropped where no pool run is active (the
/// driver, between stages), so participant `arena`'s list is not in use elsewhere.
impl Drop for Out {
    fn drop(&mut self) {
        if self.arena == NO_ARENA {
            return;
        }
        let w = self.arena;
        // SAFETY: in block-class mode every chunk is one whole B8K block from alloc(w), listed
        // once (closed chunks in `full`, the open one by cur/end), dead once its Out is dropped.
        unsafe {
            for q in 0..self.cur.len() {
                if self.freed[q].load(Relaxed) {
                    continue;
                }
                if !self.cur[q].is_null() {
                    let i = self.full[q].len();
                    self.free_chunk(w, i, self.end[q].sub(self.lines_of(q)) as *mut u8);
                }
                for (i, &(p, _)) in self.full[q].iter().enumerate() {
                    self.free_chunk(w, i, p as *mut u8);
                }
            }
        }
    }
}

pub struct Move2 {
    pub ev: Out,
    pub buys: Bins<[u64; 2]>,
    pub bits: u32,
    /// Block-class mode (`new_in`): the pool participant index, else NO_ARENA.
    w: usize,
    /// Person partitions are token ranges: partition = person token >> shift (tokens are hash
    /// slots, so names spread evenly; no mixing needed).
    shift: u32,
}

impl Move2 {
    /// Block-class mode for pool participant `w` (unique among the stage's concurrent users):
    /// survivor chunks are 8 KB blocks (`mem::B8K`), purchase chunks 256-byte blocks (`mem::B256`),
    /// taken from the blocks this worker freed first. AND `partition` frees every move-1 chunk it
    /// is handed once the partition no longer reads it: the caller promises that those chunks
    /// are whole blocks of `mem::B8K` (SHORT), `mem::B2K` (LONG) and `mem::B256` (purchases), each
    /// in exactly one partition's lists and read by nobody after its partition (the sink's
    /// `Bins::from_blocks` chunks, as the driver hands them over).
    pub fn new_in(bits: u32, hint: usize, w: usize, shift: u32) -> Self {
        let mut m = Move2::new(bits, hint);
        m.shift = shift;
        let parts = 1usize << bits;
        m.ev = Out::new_in(parts, w);
        if m.ev.arena == w && super::mem::B256.block_size() == 256 {
            m.buys = Bins::from_blocks(parts, &super::mem::B256, w);
            m.w = w;
        }
        m
    }

    pub fn new(bits: u32, hint: usize) -> Self {
        let parts = 1usize << bits;
        Move2 {
            ev: Out::new(parts, hint.clamp(1 << 14, 1 << 21), (hint / parts / 2).clamp(64, 512)),
            buys: Bins::with_chunk(parts, 1 << 12, (hint / parts / 64).clamp(4, 64)),
            bits,
            w: NO_ARENA,
            shift: 0,
        }
    }

    #[inline(always)]
    pub fn part(&self, person: u32) -> usize {
        // Below 2^bits: the driver picks shift so that every token's range has its partition.
        (person >> self.shift) as usize
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

/// Inserts SHORT records; returns the new keys. `GROW`: the partition may hold more distinct
/// keys than 3/4 of the slots, so the table doubles before passing that load. Without `GROW` the
/// caller has shown that it cannot.
///
/// The stream may also carry LONG records as two units (one-stream layout): unit 1 has the
/// ingest field 2^23 - 1, which no SHORT record has, and holds [ingest | surv << 63 in the low
/// bits, w3]; unit 2 is [a, b]. They go to `t.long_in` as [a, b, w2, w3] for the LONG table.
/// A LONG may straddle two chunks of one sink (they come consecutive); a lone unit 1 at the end
/// is a failed check.
#[inline(always)]
fn insert_short<const GROW: bool>(t: &mut Tables, bits: &mut u32, chunks: &[&[[u64; 2]]]) -> Result<usize, Abort> {
    let mut distinct = 0usize;
    let mut pend: Option<[u64; 2]> = None;
    for c in chunks {
        let n = c.len();
        let p = c.as_ptr();
        let mut j = 0usize;
        if let Some(u1) = pend.take() {
            if n == 0 {
                pend = Some(u1);
                continue;
            }
            long_unit(t, u1, c[0]);
            j = 1;
        }
        let bshift = 64 - (*bits - 2);
        let bmask = (1usize << (*bits - 2)) - 1;
        let mut tab = t.s_ptr();
        while j < n {
            if j + DT < n {
                // SAFETY: j + DT < n = c.len().
                let ka = unsafe { (*p.add(j + DT))[0] } & KEY40;
                // Address only (wrapping): a prefetch never faults.
                prefetch(tab.wrapping_add(4 * slot_of(ka, if GROW { 64 - (*bits - 2) } else { bshift })));
            }
            // SAFETY: j < n.
            let r = unsafe { *p.add(j) };
            j += 1;
            if !r[0] & X0 == 0 {
                // LONG unit 1 (cold): with its unit 2 here or at the next chunk's start.
                if j < n {
                    long_unit(t, r, c[j]);
                    j += 1;
                } else {
                    pend = Some(r);
                }
                continue;
            }
            if r[0] & TOP == 0 && r[1] != 0 {
                // An inline purchase side record: joined after the inserts.
                t.pin_s.push(r);
                continue;
            }
            if !GROW {
                // SAFETY: bits <= log2(t.s_len()), and the caller's bound keeps 3/4 free.
                distinct += usize::from(unsafe { put_short(tab, bshift, bmask, r) });
                continue;
            }
            // SAFETY: bits <= log2(t.s_len()); growth below keeps at most 3/4 used.
            if unsafe { put_short(tab, 64 - (*bits - 2), (1usize << (*bits - 2)) - 1, r) } {
                distinct += 1;
                if distinct * 4 > 3 << *bits {
                    t.grow_short(bits)?;
                    tab = t.s_ptr();
                }
            }
        }
    }
    if pend.is_some() {
        return Err(Abort("a LONG record's first unit without its second".into()));
    }
    Ok(distinct)
}

/// A LONG record from its two units in the SHORT stream (one-stream layout).
#[cold]
#[inline(never)]
fn long_unit(t: &mut Tables, u1: [u64; 2], u2: [u64; 2]) {
    t.long_in.push([u2[0], u2[1], u1[0] & (ING | TOP), u1[1]]);
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
                if r[2] & TOP == 0 && r[3] != 0 {
                    t.pin_l.push(r);
                    continue;
                }
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
    // Pre-filter: a bit per key (another hash than the tables'). Copies of a key are all in this
    // partition, so a key met twice sets its "again" bit; a record whose "again" bit is clear is
    // the only copy of its key in the input: the winner, a survivor when alive. The rest (and
    // every purchase side record and special) go through the exact table below.
    // Bitmap words come in pairs [seen, again] so one line serves both.
    let n_s: usize = short.iter().map(|c| c.len()).sum();
    let n_l: usize = long.iter().map(|c| c.len()).sum();
    let n_all = n_s + n_l + pshort.iter().map(|c| c.len()).sum::<usize>() + plong.iter().map(|c| c.len()).sum::<usize>() + specials.len();
    let fbits = (n_all * 16).next_power_of_two().clamp(1 << 12, 1 << 20).trailing_zeros();
    let fsh = 64 - fbits;
    let words = 1usize << (fbits - 6);
    let mut fm = std::mem::take(&mut t.fm);
    fm.clear();
    fm.resize(words, [0; 2]);
    let fp = fm.as_mut_ptr();
    let hs = |k: u64| ((k & KEY40).wrapping_mul(FH) >> fsh) as usize;
    let hl = |a: u64, b: u64| (long_hash(a, b).wrapping_mul(FH) >> fsh) as usize;
    // SAFETY (mark, again): h < 2^fbits, so h >> 6 < words = fm.len().
    let mark = |h: usize| unsafe {
        let w = &mut *fp.add(h >> 6);
        let bit = 1u64 << (h & 63);
        w[1] |= w[0] & bit;
        w[0] |= bit;
    };
    let again = |h: usize| unsafe { ((*fp.add(h >> 6))[1] >> (h & 63)) & 1 };
    let pf = |h: usize| prefetch(fp.wrapping_add(h >> 6));
    // Pass 1 over the SHORT stream (LONG pairs inside it marked by their second unit's key).
    let mut pend = false;
    for c in short {
        let (n, p) = (c.len(), c.as_ptr());
        let mut j = 0;
        if pend && n > 0 {
            // SAFETY: n > 0.
            let u = unsafe { *p };
            mark(hl(u[0], u[1]));
            pend = false;
            j = 1;
        }
        while j < n {
            if j + DT < n {
                // SAFETY: j + DT < n.
                pf(hs(unsafe { (*p.add(j + DT))[0] }));
            }
            // SAFETY: j < n.
            let r = unsafe { *p.add(j) };
            j += 1;
            if !r[0] & X0 == 0 {
                if j < n {
                    // SAFETY: j < n.
                    let u = unsafe { *p.add(j) };
                    mark(hl(u[0], u[1]));
                    j += 1;
                } else {
                    pend = true;
                }
                continue;
            }
            mark(hs(r[0]));
        }
    }
    for c in long {
        for r in c.iter() {
            mark(hl(r[0], r[1]));
        }
    }
    for c in pshort {
        for r in c.iter() {
            mark(hs(r[0]));
        }
    }
    for c in plong {
        for r in c.iter() {
            mark(hl(r[0], r[1]));
        }
    }
    for sp in specials {
        match sp.key {
            super::ids::Key::Short(k) => mark(hs(k)),
            super::ids::Key::Long(a, b) => mark(hl(a, b)),
            super::ids::Key::Foreign => {}
        }
    }
    // Pass 2, without branches on the common path: every SHORT record is written to both the
    // unique list (its survivor word) and the exact list, and each list's end advances only for
    // the records it keeps.
    let mut uniq = std::mem::take(&mut t.uniq);
    let mut ex_s = std::mem::take(&mut t.ex_s);
    let mut ex_l = std::mem::take(&mut t.ex_l);
    uniq.clear();
    uniq.reserve(n_s + n_l + 1);
    ex_s.clear();
    ex_s.reserve(n_s + 1);
    ex_l.clear();
    ex_l.reserve(n_s / 2 + n_l + 1);
    let (up, sp_, lp) = (uniq.as_mut_ptr(), ex_s.as_mut_ptr(), ex_l.as_mut_ptr());
    let (mut nu, mut ns_, mut nl_) = (0usize, 0usize, 0usize);
    // A LONG record [a, b, w2, w3]: exact if its key met another, or a purchase side record.
    let mut long_rec = |r: [u64; 4], nu: &mut usize, nl_: &mut usize| {
        let ex = again(hl(r[0], r[1])) | u64::from(r[2] & TOP == 0 && r[3] != 0);
        // SAFETY: at most one write per LONG record: nu < n_s + n_l + 1 and nl_ < n_s / 2 + n_l + 1
        // (a LONG in the SHORT stream takes two of its units).
        unsafe {
            *lp.add(*nl_) = r;
            *up.add(*nu) = r[3];
        }
        *nl_ += ex as usize;
        *nu += ((ex ^ 1) & (r[2] >> 63)) as usize;
    };
    let mut pend: Option<[u64; 2]> = None;
    for c in short {
        let (n, p) = (c.len(), c.as_ptr());
        let mut j = 0;
        if let Some(u1) = pend.take() {
            if n == 0 {
                pend = Some(u1);
                continue;
            }
            // SAFETY: n > 0.
            let u = unsafe { *p };
            long_rec([u[0], u[1], u1[0] & (ING | TOP), u1[1]], &mut nu, &mut nl_);
            j = 1;
        }
        while j < n {
            if j + DT < n {
                // SAFETY: j + DT < n.
                pf(hs(unsafe { (*p.add(j + DT))[0] }));
            }
            // SAFETY: j < n.
            let r = unsafe { *p.add(j) };
            j += 1;
            if !r[0] & X0 == 0 {
                if j < n {
                    // SAFETY: j < n.
                    let u = unsafe { *p.add(j) };
                    long_rec([u[0], u[1], r[0] & (ING | TOP), r[1]], &mut nu, &mut nl_);
                    j += 1;
                } else {
                    pend = Some(r);
                }
                continue;
            }
            let ex = again(hs(r[0])) | u64::from(r[0] & TOP == 0 && r[1] != 0);
            // SAFETY: one write each per SHORT record: nu, ns_ < n_s + 1.
            unsafe {
                *sp_.add(ns_) = r;
                *up.add(nu) = r[1];
            }
            ns_ += ex as usize;
            nu += ((ex ^ 1) & (r[0] >> 63)) as usize;
        }
    }
    for c in long {
        for &r in c.iter() {
            long_rec(r, &mut nu, &mut nl_);
        }
    }
    // SAFETY: the first nu / ns_ / nl_ entries were written above.
    unsafe {
        uniq.set_len(nu);
        ex_s.set_len(ns_);
        ex_l.set_len(nl_);
    }
    t.fm = fm;
    if pend.is_some() {
        t.uniq = uniq;
        t.ex_s = ex_s;
        t.ex_l = ex_l;
        return Err(Abort("a LONG record's first unit without its second".into()));
    }
    let r = exact(&[&ex_s[..]], &[&ex_l[..]], pshort, plong, specials, cx, t, m2, &uniq);
    t.uniq = uniq;
    t.ex_s = ex_s;
    t.ex_l = ex_l;
    if r.is_ok() && m2.w != NO_ARENA {
        // SAFETY: new_in's contract: every chunk is a whole block of its class, in this
        // partition only, and nothing reads it after this partition (the lists are copies).
        unsafe { release(m2.w, short, long, pshort, plong) };
    }
    r
}

/// Second multiplier for the pre-filter's bit (independent of the tables' FIB index).
const FH: u64 = 0xc2b2_ae3d_27d4_eb4f;

/// The exact table on the colliding records, then move 2 of the unique survivors `uniq` and the
/// table's survivors.
#[allow(clippy::too_many_arguments)]
fn exact(
    short: &[&[[u64; 2]]],
    long: &[&[[u64; 4]]],
    pshort: &[&[[u64; 2]]],
    plong: &[&[[u64; 4]]],
    specials: &[Special],
    cx: &Ctx3,
    t: &mut Tables,
    m2: &mut Move2,
    uniq: &[u64],
) -> Result<usize, Abort> {
    if t.poisoned {
        return Err(Abort("dedup tables left in an unknown state by an earlier failure".into()));
    }
    t.poisoned = true;
    t.pin_s.clear();
    t.pin_l.clear();
    let ns: usize = short.iter().map(|c| c.len()).sum::<usize>() + specials.len();
    let nl: usize = long.iter().map(|c| c.len()).sum::<usize>() + specials.len();
    let mut sb = bits_for(ns, 3, t.s_len());
    // Every record (specials included) a new key still leaves a quarter of the slots empty:
    // no growth check needed.
    t.long_in.clear();
    let mut sd = if ns * 4 <= 3 << sb { insert_short::<false>(t, &mut sb, short)? } else { insert_short::<true>(t, &mut sb, short)? };
    // LONG records met in the SHORT stream join the LONG stream's (sized for both).
    let long_in = std::mem::take(&mut t.long_in);
    let nl = nl + long_in.len();
    let mut lb = bits_for(nl, 6, t.l.len());
    let mut lc: Vec<&[[u64; 4]]> = long.to_vec();
    lc.push(&long_in);
    let r = if nl * 4 <= 3 << lb { insert_long::<false>(t, &mut lb, &lc) } else { insert_long::<true>(t, &mut lb, &lc) };
    drop(lc);
    t.long_in = long_in;
    let (bad_l, mut ld) = r?;
    if bad_l != 0 {
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
    let pin_s = std::mem::take(&mut t.pin_s);
    let pin_l = std::mem::take(&mut t.pin_l);
    {
        let side = pshort.iter().flat_map(|c| c.iter().copied());
        for r in side.chain(pin_s.iter().map(|r| [r[0], r[1] & !PURCH_MARK])) {
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
    {
        let side = plong.iter().flat_map(|c| c.iter().copied());
        for r in side.chain(pin_l.iter().map(|r| [r[0], r[1], r[2], r[3] & !PURCH_MARK])) {
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
    t.pin_s = pin_s;
    t.pin_l = pin_l;
    for &m in &t.marked {
        let i = (m & !(1 << 31)) as usize;
        if m >> 31 == 0 {
            t.mark_s[i / 64] = 0;
        } else {
            t.mark_l[i / 64] = 0;
        }
    }
    t.marked.clear();
    // The move-1 chunks are read for the last time above: in block-class mode they go back to
    // this worker's lists now, so the move-2 output below reuses them (no fresh pages).

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
    emit(people, uniq, m2);
    emit(people, &t.surv[..n], m2);
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
    Ok(uniq.len() + n + t.spec.len())
}

/// Survivor words to move 2, person lookups prefetched DP ahead.
#[inline(always)]
fn emit(people: &People, sv: &[u64], m2: &mut Move2) {
    let n = sv.len();
    for k in 0..n {
        if k + DP < n {
            people.prefetch(sv[k + DP] as u32);
        }
        let w = sv[k];
        m2.push_word(people.person_of(w as u32), w);
    }
}

/// Returns a partition's move-1 chunks to their block classes for pool participant `w`.
/// SAFETY: as Move2::new_in's contract; `w` is the caller's participant index. The blocks are
/// addressed by number, as the classes themselves keep them (their free lists are addresses), so
/// the pointer takes the region's exposed provenance, not the read-only one of the slices; the
/// slices are not read again.
unsafe fn release(w: usize, short: &[&[[u64; 2]]], long: &[&[[u64; 4]]], pshort: &[&[[u64; 2]]], plong: &[&[[u64; 4]]]) {
    use super::mem::{B256, B2K, B8K};
    let blk = |c: usize| std::ptr::with_exposed_provenance_mut::<u8>(c);
    // SAFETY (each free): the block came from that class's alloc (the sink's from_blocks) and is
    // dead from here on (caller).
    unsafe {
        for c in short {
            B8K.free(w, blk(c.as_ptr() as usize));
        }
        for c in long {
            B2K.free(w, blk(c.as_ptr() as usize));
        }
        for c in pshort {
            B256.free(w, blk(c.as_ptr() as usize));
        }
        for c in plong {
            B256.free(w, blk(c.as_ptr() as usize));
        }
    }
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
