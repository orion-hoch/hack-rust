//! Event lines, v2: line structure per 64-byte block, decoding 8 lines at a time.
//!
//! Phase A walks a window of the buffer in 64-byte blocks: comma, quote and newline masks; every
//! separator flattened into `sep` as a window-relative position with two flags (newline; the next
//! byte is a quote); quote legality as mask identities with
//! carries across blocks, any violation position into `viol`.
//!
//! Phase B takes 8 consecutive lines whose separators sit at stride 7 (6 commas and the newline)
//! and no violation: field bounds come from `sep` and its flags, and the decoders run lane-wise on
//! [u64; 8] arrays (the same arithmetic as v1, so it vectorizes with the target features of the
//! entry function). Any other line goes through the v1 per-line path, which falls back to the
//! reference itself, so every line ends up exact.

use std::ptr::read_unaligned as rd;

use crate::fast::{self, Buf, Fast, Kind, PRE, dec8, hms, offset};
use crate::records::*;
use crate::reference;
use crate::ts::DAY_MS;

const LANES: usize = 8;

/// Software prefetch distance in phase A (bytes ahead; 0 = off). A prefetch never faults, so
/// reading past the end of the mapping is harmless.
pub static PREFETCH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(4096);

/// Measurement only: stop each event batch after this phase (255 = full parse). 1 phase A only,
/// 2 + field bounds, 3 + loads, 4 + IDs, 5 + ts, 6 + ingest, 7 + unscramble and validity.
pub static PHASE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(255);

/// The cutoffs are compiled in only with `--features phases` (they perturb the code they measure).
#[inline(always)]
fn phase() -> u8 {
    if cfg!(feature = "phases") { PHASE.load(std::sync::atomic::Ordering::Relaxed) } else { 255 }
}
/// Bytes of input per phase A window.
const W: usize = 32 * 1024;
const NONE40: u64 = u64::MAX;

thread_local! {
    static SCRATCH: std::cell::Cell<Option<Scratch>> = const { std::cell::Cell::new(None) };
}

/// Puts the scratch back for the thread's next chunk on every exit path.
struct ScratchGuard(Option<Scratch>);
impl Drop for ScratchGuard {
    fn drop(&mut self) {
        if let Some(s) = self.0.take() {
            SCRATCH.with(|c| c.set(Some(s)));
        }
    }
}

/// A separator entry: window-relative position (bits 0..30), bit 31 set when the separator is a
/// newline, bit 30 set when the byte after it is a quote (the next field is quoted).
pub(crate) const NL_FLAG: u32 = 1 << 31;
pub(crate) const Q_FLAG: u32 = 1 << 30;
pub(crate) const POS_MASK: u32 = (1 << 30) - 1;

pub(crate) struct Scratch {
    sep: Vec<u32>,
    /// Index in `sep` of the last newline of the window (valid when `has_nl`).
    last_nl: usize,
    has_nl: bool,
    /// The window's first byte is a quote (field 0 of its first line is quoted).
    pub(crate) wq0: bool,
    viol: Vec<u32>,
    nsep: usize,
    nviol: usize,
}

impl Scratch {
    #[inline(always)]
    pub(crate) fn sep_ptr(&self) -> *const u32 {
        self.sep.as_ptr()
    }
    /// Window-relative position of separator k.
    #[inline(always)]
    fn pos(&self, k: usize) -> usize {
        (self.sep[k] & POS_MASK) as usize
    }
    /// Index of the first newline separator at or after k (k <= last_nl).
    #[inline(always)]
    fn next_nl(&self, k: usize) -> usize {
        let mut k = k;
        while self.sep[k] & NL_FLAG == 0 {
            k += 1;
        }
        k
    }
    /// Window-relative start of the line whose first separator is k.
    #[inline(always)]
    fn line_start(&self, k: usize) -> usize {
        if k == 0 { 0 } else { self.pos(k - 1) + 1 }
    }
    /// Whether separator k - 1 (the one before field 0 of the line starting at k) is followed by a
    /// quote.
    #[inline(always)]
    fn q_before(&self, k: usize) -> bool {
        if k == 0 { self.wq0 } else { self.sep[k - 1] & Q_FLAG != 0 }
    }
    fn new() -> Scratch {
        let n = W + 256;
        Scratch { sep: vec![0; n + 128], last_nl: 0, has_nl: false, wq0: false, viol: vec![0; n + 64], nsep: 0, nviol: 0 }
    }
}

/// Whether the 8 * nf entries from k are 8 well-formed lines: a newline exactly at every nf-th.
#[inline(always)]
unsafe fn nl_pattern<const ISA: u8>(sc: &Scratch, k: usize, nf: usize) -> bool {
    let n = 8 * nf;
    let p = unsafe { sc.sep.as_ptr().add(k) };
    let mut bits = 0u64;
    #[cfg(target_arch = "x86_64")]
    {
        if ISA != fast::ISA_SCALAR {
            for c in 0..n.div_ceil(8) {
                bits |= (unsafe { signs8(p.add(8 * c)) } as u64) << (8 * c);
            }
        } else {
            for i in 0..n {
                bits |= ((unsafe { *p.add(i) } >> 31) as u64) << i;
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    for i in 0..n {
        bits |= ((unsafe { *p.add(i) } >> 31) as u64) << i;
    }
    let span = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
    let pattern: u64 = if nf == 7 { 0x0081_0204_0810_2040 } else { 0x8888_8888 };
    bits & span == pattern
}

/// The sign bits (newline flags) of 8 entries.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn signs8(p: *const u32) -> u32 {
    use std::arch::x86_64::*;
    unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_loadu_si256(p as *const __m256i))) as u32 }
}

/// Days table indexed by (year - y0) * 384 + month * 32 + day, for two years.
pub(crate) struct Dates {
    y0: i64,
    day: Vec<i32>,
}

pub(crate) const BAD_DAY: i32 = i32::MIN;
/// Slots of the direct date table: (year 0..2) * 416 + month (1..=12) * 32 + day (1..=31) is at
/// most 831; the table has 1024 slots and every index is masked, so no input reads past it.
pub(crate) const DATE_SLOTS: usize = 1024;

#[inline(always)]
pub(crate) fn date_index(yy: i64, m: i64, d: i64) -> usize {
    ((yy * 416 + m * 32 + d) as usize) & (DATE_SLOTS - 1)
}

impl Dates {
    #[inline(always)]
    pub(crate) fn y0_(&self) -> i64 {
        self.y0
    }
    #[inline(always)]
    pub(crate) fn day_ptr(&self) -> *const i32 {
        self.day.as_ptr()
    }
    pub(crate) fn new(ws_ms: i64, we_ms: i64) -> Dates {
        let ws_day = ws_ms.div_euclid(DAY_MS);
        let we_day = (we_ms - 1).div_euclid(DAY_MS);
        let (y0, _, _) = crate::ts::civil_from_days(ws_day - 9);
        let mut day = vec![BAD_DAY; DATE_SLOTS];
        for d in (ws_day - 9)..=(we_day + 7) {
            let (y, m, dd) = crate::ts::civil_from_days(d);
            if y - y0 < 2 && (0..=9999).contains(&y) {
                day[date_index(y - y0, m, dd)] = (d - ws_day) as i32;
            }
        }
        Dates { y0, day }
    }
}

// ------------------------------------------------------------------------------------------

#[inline(always)]
unsafe fn masks64<const ISA: u8>(p: *const u8) -> (u64, u64, u64) {
    #[cfg(target_arch = "x86_64")]
    {
        if ISA == fast::ISA_AVX512 {
            return unsafe { masks64_avx512(p) };
        }
        if ISA == fast::ISA_AVX2 {
            return unsafe { masks64_avx2(p) };
        }
    }
    let m = unsafe { fast::masks::<{ fast::ISA_SCALAR }>(p) };
    (m.comma as u64, m.quote as u64, m.nl as u64)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn masks64_avx512(p: *const u8) -> (u64, u64, u64) {
    use std::arch::x86_64::*;
    unsafe {
        let v = _mm512_loadu_si512(p as *const __m512i);
        (
            _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b',' as i8)),
            _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'"' as i8)),
            _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\n' as i8)),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn masks64_avx2(p: *const u8) -> (u64, u64, u64) {
    use std::arch::x86_64::*;
    unsafe {
        let a = _mm256_loadu_si256(p as *const __m256i);
        let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
        let m = |x: u8| {
            let xx = _mm256_set1_epi8(x as i8);
            (_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, xx)) as u32 as u64)
                | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, xx)) as u32 as u64) << 32)
        };
        (m(b','), m(b'"'), m(b'\n'))
    }
}

#[inline(always)]
fn pext<const ISA: u8>(x: u64, mask: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if ISA != fast::ISA_SCALAR {
            return unsafe { pext_bmi2(x, mask) };
        }
    }
    let (mut m, mut out, mut bit) = (mask, 0u64, 0u32);
    while m != 0 {
        let low = m & m.wrapping_neg();
        if x & low != 0 {
            out |= 1 << bit;
        }
        bit += 1;
        m ^= low;
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
#[inline]
unsafe fn pext_bmi2(x: u64, mask: u64) -> u64 {
    std::arch::x86_64::_pext_u64(x, mask)
}

/// Writes the entries of the set bits of `m` (position base + bit, flags from `nl` and `fq`) at
/// out[n..]; returns the new n.
#[inline(always)]
unsafe fn flatten(mut m: u64, nl: u64, fq: u64, base: u32, out: *mut u32, n: usize) -> usize {
    let cnt = m.count_ones() as usize;
    let p = unsafe { out.add(n) };
    let entry = |t: u32| (base + t) | (((nl >> t) as u32 & 1) << 31) | (((fq >> t) as u32 & 1) << 30);
    for i in 0..8 {
        let t = m.trailing_zeros() & 63;
        unsafe { *p.add(i) = entry(t) };
        m &= m.wrapping_sub(1);
    }
    if cnt > 8 {
        let mut i = 8;
        while m != 0 {
            unsafe { *p.add(i) = entry(m.trailing_zeros()) };
            m &= m.wrapping_sub(1);
            i += 1;
        }
    }
    n + cnt
}

/// Plain position list (the violation list).
#[inline(always)]
unsafe fn flatten_pos(mut m: u64, base: u32, out: *mut u32, n: usize) -> usize {
    let mut i = n;
    while m != 0 {
        unsafe { *out.add(i) = base + m.trailing_zeros() };
        m &= m.wrapping_sub(1);
        i += 1;
    }
    i
}

/// Separator entries of one block by compress (AVX-512F vpcompressd in 16-bit pieces), flags
/// set by masked ors: writes out[n..n + popcnt(m)] (and up to 16 more entries of scratch).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn flatten_compress(m: u64, nl: u64, fq: u64, base: u32, out: *mut u32, n: usize) -> usize {
    use std::arch::x86_64::*;
    let iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let b = _mm512_add_epi32(_mm512_set1_epi32(base as i32), iota);
    let nlf = _mm512_set1_epi32(NL_FLAG as i32);
    let qf = _mm512_set1_epi32(Q_FLAG as i32);
    let mut n = n;
    let pieces = [0i32, 16, 32, 48];
    for (piece, &off) in pieces.iter().enumerate() {
        let sh = 16 * piece as u32;
        let k = (m >> sh) as u16;
        let v = _mm512_add_epi32(b, _mm512_set1_epi32(off));
        let v = _mm512_mask_or_epi32(v, (nl >> sh) as u16, v, nlf);
        let v = _mm512_mask_or_epi32(v, (fq >> sh) as u16, v, qf);
        let v = _mm512_maskz_compress_epi32(k, v);
        unsafe { _mm512_storeu_si512(out.add(n) as *mut __m512i, v) };
        n += k.count_ones() as usize;
    }
    n
}

/// The AVX-512 phase A: one function with few live values (block pointer, the three masks, two
/// carry bits, the write pointer), so nothing spills. Returns the number of entries written.
/// Violations go through an out-of-line path into `viol` (rare). Blocks read [p, p + 128).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,bmi1,bmi2,popcnt,lzcnt")]
#[inline(never)]
unsafe fn phase_a_avx512(p0: *const u8, nblocks: usize, pf: usize, sep: *mut u32, viol: &mut ViolList) -> usize {
    use std::arch::x86_64::*;
    unsafe {
        let cc = _mm512_set1_epi8(b',' as i8);
        let qq = _mm512_set1_epi8(b'"' as i8);
        let nn = _mm512_set1_epi8(b'\n' as i8);
        let iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let nlf = _mm512_set1_epi32(NL_FLAG as i32);
        let qf = _mm512_set1_epi32(Q_FLAG as i32);
        let v = _mm512_loadu_si512(p0 as *const __m512i);
        let mut q = _mm512_cmpeq_epi8_mask(v, qq);
        let mut n = _mm512_cmpeq_epi8_mask(v, nn);
        let mut d = _mm512_cmpeq_epi8_mask(v, cc) | n;
        let mut dprev = 1u64;
        let mut eprev = 0u64;
        let mut carry = 0u64;
        let mut out = sep;
        let mut rel = 0u32;
        let mut p = p0;
        for _ in 0..nblocks {
            if pf != 0 {
                _mm_prefetch::<_MM_HINT_T0>(p.wrapping_add(pf) as *const i8);
            }
            let v2 = _mm512_loadu_si512(p.add(64) as *const __m512i);
            let q2 = _mm512_cmpeq_epi8_mask(v2, qq);
            let n2 = _mm512_cmpeq_epi8_mask(v2, nn);
            let d2 = _mm512_cmpeq_epi8_mask(v2, cc) | n2;
            let fs = (d << 1) | dprev;
            let fe = (d >> 1) | (d2 << 63);
            let s = q & fs;
            let e = q & fe;
            let (t1, o1) = (!d).overflowing_add(s);
            let (t2, o2) = t1.overflowing_add(carry);
            carry = (o1 | o2) as u64;
            let bad = (q & !(fs | fe)) | (s & e) | ((t2 & d) ^ ((e << 1) | eprev));
            if bad != 0 {
                viol.push_mask(bad, rel);
            }
            let fq = ((q >> 1) | (q2 << 63)) & d;
            // Entries: position, newline flag, quote-after flag; compressed by d in 16-bit pieces.
            let b = _mm512_add_epi32(_mm512_set1_epi32(rel as i32), iota);
            for piece in 0..4u32 {
                let sh = 16 * piece;
                let k = (d >> sh) as u16;
                let x = _mm512_add_epi32(b, _mm512_set1_epi32(sh as i32));
                let x = _mm512_mask_or_epi32(x, (n >> sh) as u16, x, nlf);
                let x = _mm512_mask_or_epi32(x, (fq >> sh) as u16, x, qf);
                _mm512_storeu_si512(out as *mut __m512i, _mm512_maskz_compress_epi32(k, x));
                out = out.add(k.count_ones() as usize);
            }
            dprev = d >> 63;
            eprev = e >> 63;
            q = q2;
            n = n2;
            d = d2;
            p = p.add(64);
            rel += 64;
        }
        out.offset_from(sep) as usize
    }
}

/// Violation positions (absolute), filled on the rare path.
pub(crate) struct ViolList {
    ptr: *mut u32,
    n: usize,
    base: u32,
}

impl ViolList {
    #[inline(never)]
    #[cold]
    fn push_mask(&mut self, mut m: u64, rel: u32) {
        while m != 0 {
            unsafe { *self.ptr.add(self.n) = self.base + rel + m.trailing_zeros() };
            self.n += 1;
            m &= m - 1;
        }
    }
}

/// Phase A over blocks starting at wpos (a line start) up to wend: separator entries (window-
/// relative, with flags), the violation list, and the index of the last newline.
#[inline(always)]
unsafe fn phase_a<const ISA: u8>(base: *const u8, wpos: usize, wend: usize, sc: &mut Scratch) -> usize {
    #[cfg(target_arch = "x86_64")]
    if ISA == fast::ISA_AVX512 {
        sc.wq0 = unsafe { *base.add(wpos) } == b'"';
        let nblocks = (wend - wpos).div_ceil(64);
        let pf = PREFETCH.load(std::sync::atomic::Ordering::Relaxed);
        let mut vl = ViolList { ptr: sc.viol.as_mut_ptr(), n: 0, base: wpos as u32 };
        let nsep = unsafe { phase_a_avx512(base.add(wpos), nblocks, pf, sc.sep.as_mut_ptr(), &mut vl) };
        unsafe { *sc.viol.as_mut_ptr().add(vl.n) = u32::MAX };
        sc.nviol = vl.n;
        sc.nsep = nsep;
        // The last newline: scan back over the flags (the window ends inside a line).
        let mut k = nsep;
        while k > 0 && sc.sep[k - 1] & NL_FLAG == 0 {
            k -= 1;
        }
        sc.has_nl = k > 0;
        sc.last_nl = k.saturating_sub(1);
        return wpos + 64 * nblocks;
    }
    unsafe { phase_a_portable::<ISA>(base, wpos, wend, sc) }
}

#[inline(always)]
unsafe fn phase_a_portable<const ISA: u8>(base: *const u8, wpos: usize, wend: usize, sc: &mut Scratch) -> usize {
    let (mut nsep, mut nviol) = (0usize, 0usize);
    let mut last_nl = 0usize;
    let mut has_nl = false;
    let sep = sc.sep.as_mut_ptr();
    let viol = sc.viol.as_mut_ptr();
    sc.wq0 = unsafe { *base.add(wpos) } == b'"';
    let mut b = wpos;
    let (mut c, mut q, mut n) = unsafe { masks64::<ISA>(base.add(b)) };
    let mut dprev = 1u64; // the window starts at a line start
    let mut eprev = 0u64;
    let mut carry = 0u64;
    let pf = PREFETCH.load(std::sync::atomic::Ordering::Relaxed);
    while b < wend {
        #[cfg(target_arch = "x86_64")]
        if pf != 0 {
            unsafe { std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(base.wrapping_add(b + pf) as *const i8) };
        }
        let (c2, q2, n2) = unsafe { masks64::<ISA>(base.add(b + 64)) };
        let d = c | n;
        let d2 = c2 | n2;
        let fs = (d << 1) | dprev;
        let fe = (d >> 1) | ((d2 & 1) << 63);
        let s = q & fs;
        let e = q & fe;
        let (t1, o1) = (!d).overflowing_add(s);
        let (t2, o2) = t1.overflowing_add(carry);
        carry = (o1 | o2) as u64;
        let closes = t2 & d;
        let bad = (q & !(fs | fe)) | (s & e) | (closes ^ ((e << 1) | eprev));
        if bad != 0 {
            nviol = unsafe { flatten_pos(bad, b as u32, viol, nviol) };
        }
        let fq = ((q >> 1) | ((q2 & 1) << 63)) & d;
        // Index of the block's last newline separator.
        let top = 63 - (n | 1).leading_zeros();
        let below = u64::MAX >> (63 - top);
        let idx = nsep + (d & below).count_ones() as usize - 1;
        if n != 0 {
            last_nl = idx;
            has_nl = true;
        }
        let rel = (b - wpos) as u32;
        #[cfg(target_arch = "x86_64")]
        {
            nsep = if ISA == fast::ISA_AVX512 {
                unsafe { flatten_compress(d, n, fq, rel, sep, nsep) }
            } else {
                unsafe { flatten(d, n, fq, rel, sep, nsep) }
            };
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            nsep = unsafe { flatten(d, n, fq, rel, sep, nsep) };
        }
        dprev = d >> 63;
        eprev = e >> 63;
        c = c2;
        q = q2;
        n = n2;
        b += 64;
    }
    let _ = c;
    unsafe { *viol.add(nviol) = u32::MAX };
    sc.nsep = nsep;
    sc.nviol = nviol;
    sc.last_nl = last_nl;
    sc.has_nl = has_nl;
    b
}


/// A conforming ID's base-36 value from the 8 bytes ending at its end, its digit count and whether
/// the prefix byte matched, or NONE40.
#[inline(always)]
fn id_from(w: u64, k: u64, prefix_ok: bool) -> u64 {
    let kc = k.clamp(1, 8);
    let keep = !0u64 << (8 * (8 - kc));
    let w = (w & keep) | (0x3030_3030_3030_3030 & !keep);
    let (v, ok) = fast::b36_digits(w);
    let first = (v >> (8 * (8 - kc))) & 0xFF;
    let value = fast::b36_value(v);
    let good = ok & prefix_ok & (k >= 1) & (k <= 8) & ((k == 1) | (first != 0)) & (value <= crate::ids::MASK40);
    if good { value } else { NONE40 }
}

/// Date of `YYYY-MM-DD` spread over HI (bytes 0..8) and MID (bytes 8..10): (day, ok).
#[inline(always)]
fn date_of(dates: &Dates, hi: u64, mid: u64) -> (i64, bool) {
    let x = hi ^ 0x3030_3030_3030_3030;
    let xd = x & 0x00FF_FF00_FFFF_FFFF;
    let digits = (xd & 0xF0F0_F0F0_F0F0_F0F0 == 0)
        & ((xd.wrapping_add(0x0606_0606_0606_0606) & 0x1010_1010_1010_1010) == 0);
    let seps = (hi & 0xFF00_00FF_0000_0000) == 0x2D00_002D_0000_0000;
    let y = ((x & 0xFF) * 1000 + ((x >> 8) & 0xFF) * 100 + ((x >> 16) & 0xFF) * 10 + ((x >> 24) & 0xFF)) as i64;
    let m = (((x >> 40) & 0xFF) * 10 + ((x >> 48) & 0xFF)) as i64;
    let xm = (mid ^ 0x3030) & 0xFFFF;
    let mdig = (xm & 0xF0F0 == 0) & ((xm.wrapping_add(0x0606) & 0x1010) == 0);
    let d = ((xm & 0xFF) * 10 + (xm >> 8)) as i64;
    let yy = y - dates.y0;
    let in_range = digits & seps & mdig & (yy >= 0) & (yy < 2) & (m >= 1) & (m <= 12) & (d >= 1) & (d <= 31);
    let idx = if in_range { date_index(yy, m, d) } else { 0 };
    // In bounds: date_index masks to DATE_SLOTS - 1 and dates.day.len() == DATE_SLOTS. Slot 0
    // (month 0, day 0) is never a date, so out-of-range lanes read BAD_DAY.
    let day = unsafe { *dates.day.get_unchecked(idx) };
    (day as i64, in_range & (day != BAD_DAY))
}

struct Lanes {
    ls: [u32; LANES],
    end: [u32; LANES],
    vs: [[u32; LANES]; 7],
    ve: [[u32; LANES]; 7],
}

/// Decodes 8 lines (i0..i0+8 of the window, separators from k0) and emits their records.
#[inline(always)]
unsafe fn batch<'b, S: Sink, const ISA: u8>(
    f: &Fast,
    dates: &Dates,
    file: u32,
    buf: Buf<'b>,
    sc: &Scratch,
    wpos: usize,
    i0: usize,
    k0: usize,
    fields: &mut Vec<&'b [u8]>,
    sink: &mut S,
) {
    let base = buf.data.as_ptr();
    let mut ln = Lanes { ls: [0; LANES], end: [0; LANES], vs: [[0; LANES]; 7], ve: [[0; LANES]; 7] };
    let mut pre = [0u8; LANES];
    let mut pre_u = [0u8; LANES];
    // Field bounds.
    for l in 0..LANES {
        let kl = k0 + 7 * l;
        let _ = i0;
        let w = wpos as u32;
        let ls = w + sc.line_start(kl) as u32;
        ln.ls[l] = ls;
        ln.end[l] = w + sc.pos(kl + 6) as u32;
        let q0 = sc.q_before(kl);
        let b01 = unsafe { rd(base.add(ls as usize) as *const u16) };
        pre[l] = if q0 { (b01 >> 8) as u8 } else { b01 as u8 };
        ln.vs[0][l] = ls + q0 as u32;
        ln.ve[0][l] = w + sc.pos(kl) as u32 - q0 as u32;
        for j in 1..7 {
            let q = (sc.sep[kl + j - 1] & Q_FLAG != 0) as u32;
            ln.vs[j][l] = w + sc.pos(kl + j - 1) as u32 + 1 + q;
            ln.ve[j][l] = w + sc.pos(kl + j) as u32 - q;
        }
    }
    if phase() <= 2 {
        std::hint::black_box(&ln);
        std::hint::black_box(&pre);
        return;
    }
    // Loads.
    let mut w_e = [0u64; LANES];
    let mut w_u = [0u64; LANES];
    // The 24 bytes before the last 8 of each ID (a LONG ID's first 24 hex digits).
    let mut le_w = [[0u64; LANES]; 3];
    let mut lu_w = [[0u64; LANES]; 3];
    let mut t_hi = [0u64; LANES];
    let mut t_mid = [0u64; LANES];
    let mut t_16 = [0u64; LANES];
    let mut t_17 = [0u64; LANES];
    let mut i_hi = [0u64; LANES];
    let mut i_mid = [0u64; LANES];
    let mut i_16 = [0u64; LANES];
    let mut tw = [0u64; LANES];
    unsafe {
        for l in 0..LANES {
            // ve >= line start >= wpos >= PRE - pre, so ve - 8 >= -pre: inside the readable bytes.
            w_e[l] = rd(base.offset(ln.ve[0][l] as isize - 8) as *const u64);
            w_u[l] = rd(base.offset(ln.ve[1][l] as isize - 8) as *const u64);
            for t in 0..3 {
                le_w[t][l] = rd(base.offset(ln.ve[0][l] as isize - 32 + 8 * t as isize) as *const u64);
                lu_w[t][l] = rd(base.offset(ln.ve[1][l] as isize - 32 + 8 * t as isize) as *const u64);
            }
            pre_u[l] = *base.add(ln.vs[1][l] as usize);
            let t = base.add(ln.vs[2][l] as usize);
            t_hi[l] = rd(t as *const u64);
            t_mid[l] = rd(t.add(8) as *const u64);
            t_16[l] = rd(t.add(16) as *const u64);
            t_17[l] = rd(t.add(17) as *const u64);
            let g = base.add(ln.vs[3][l] as usize);
            i_hi[l] = rd(g as *const u64);
            i_mid[l] = rd(g.add(8) as *const u64);
            i_16[l] = rd(g.add(16) as *const u64);
            tw[l] = rd(base.add(ln.vs[4][l] as usize) as *const u64);
        }
    }
    if phase() <= 3 {
        std::hint::black_box((&w_e, &w_u, &t_hi, &t_mid, &t_16, &t_17, &i_hi, &i_mid, &i_16, &tw, &pre_u));
        return;
    }
    // Decode, lane-wise.
    let mut x_e = [0u64; LANES];
    let mut x_u = [0u64; LANES];
    let mut ts_rel = [0i64; LANES];
    let mut ts_fast = [false; LANES];
    let mut ingest = [0i64; LANES];
    let mut ing_fast = [false; LANES];
    for l in 0..LANES {
        let le = (ln.ve[0][l] - ln.vs[0][l]) as u64;
        x_e[l] = id_from(w_e[l], le.wrapping_sub(1), pre[l] == b'e');
        let lu = (ln.ve[1][l] - ln.vs[1][l]) as u64;
        x_u[l] = id_from(w_u[l], lu.wrapping_sub(1), pre_u[l] == b'u');
    }
    if phase() <= 4 {
        std::hint::black_box((&x_e, &x_u, &t_hi, &t_mid, &t_16, &t_17, &i_hi, &i_mid, &i_16, &tw));
        return;
    }
    for l in 0..LANES {
        let len = ln.ve[2][l] - ln.vs[2][l];
        let (day, date_ok) = date_of(dates, t_hi[l], t_mid[l]);
        let t_ok = ((t_mid[l] >> 16) & 0xFF) == b'T' as u64;
        let (secs, hms_ok) = hms((t_mid[l] >> 24) | (t_16[l] << 40));
        let z = ((t_16[l] >> 24) & 0xFF) == b'Z' as u64;
        let (off, off_ok) = offset(t_17[l]);
        let dt_ok = date_ok & t_ok & hms_ok;
        let f1 = (len == 20) & z & dt_ok;
        let f2 = (len == 25) & off_ok & dt_ok;
        let (hi, hi_ok) = dec8(t_hi[l]);
        let (lo, lo_ok) = dec8((t_mid[l] << 24) | 0x0030_3030);
        let f3 = (len == 13) & hi_ok & lo_ok;
        let local = day * DAY_MS + secs * 1000 - f.ws_frac;
        let f12 = local - if f2 { off } else { 0 };
        let v3 = (hi * 100_000 + lo) as i64 - f.ws_ms;
        ts_rel[l] = if f3 { v3 } else { f12 };
        ts_fast[l] = f1 | f2 | f3;
    }
    if phase() <= 5 {
        std::hint::black_box((&x_e, &x_u, &ts_rel, &ts_fast, &i_hi, &i_mid, &i_16, &tw));
        return;
    }
    for l in 0..LANES {
        let len = ln.ve[3][l] - ln.vs[3][l];
        let (day, date_ok) = date_of(dates, i_hi[l], i_mid[l]);
        let t_ok = ((i_mid[l] >> 16) & 0xFF) == b'T' as u64;
        let (secs, hms_ok) = hms((i_mid[l] >> 24) | (i_16[l] << 40));
        let z = ((i_16[l] >> 24) & 0xFF) == b'Z' as u64;
        ingest[l] = day * 86_400 + secs + f.ingest_add;
        ing_fast[l] = (len == 20) & date_ok & t_ok & hms_ok & z;
    }
    if phase() <= 6 {
        std::hint::black_box((&x_e, &x_u, &ts_rel, &ts_fast, &ingest, &ing_fast, &tw));
        return;
    }
    // LONG keys (used where the ID is not SHORT): length 33, prefix, 32 hex digits.
    let mut e_long = [false; LANES];
    let mut e_la = [0u64; LANES];
    let mut e_lb = [0u64; LANES];
    let mut u_long = [false; LANES];
    let mut u_la = [0u64; LANES];
    let mut u_lb = [0u64; LANES];
    for l in 0..LANES {
        let (a0, k0) = fast::hex8(le_w[0][l]);
        let (a1, k1) = fast::hex8(le_w[1][l]);
        let (b0, k2) = fast::hex8(le_w[2][l]);
        let (b1, k3) = fast::hex8(w_e[l]);
        e_long[l] = (ln.ve[0][l] - ln.vs[0][l] == 33) & (pre[l] == b'e') & k0 & k1 & k2 & k3;
        e_la[l] = (a0 << 32) | a1;
        e_lb[l] = (b0 << 32) | b1;
        let (a0, k0) = fast::hex8(lu_w[0][l]);
        let (a1, k1) = fast::hex8(lu_w[1][l]);
        let (b0, k2) = fast::hex8(lu_w[2][l]);
        let (b1, k3) = fast::hex8(w_u[l]);
        u_long[l] = (ln.ve[1][l] - ln.vs[1][l] == 33) & (pre_u[l] == b'u') & k0 & k1 & k2 & k3;
        u_la[l] = (a0 << 32) | a1;
        u_lb[l] = (b0 << 32) | b1;
    }
    let mut base_ok = [false; LANES];
    let mut purchase = [false; LANES];
    for l in 0..LANES {
        let tlen = ln.ve[4][l] - ln.vs[4][l];
        let p = (tlen == 8) & (tw[l] == u64::from_le_bytes(*b"purchase"));
        let v = (tlen == 4) & (tw[l] as u32 == u32::from_le_bytes(*b"view"));
        let c = (tlen == 4) & (tw[l] as u32 == u32::from_le_bytes(*b"cart"));
        let ts_ok = (ts_rel[l] >= 0) & (ts_rel[l] < f.win_ms);
        let ingest_rel_ms = (ingest[l] - f.ingest_add) * 1000 - f.ws_frac;
        let late = ingest_rel_ms - ts_rel[l] > DAY_MS;
        base_ok[l] = (ln.ve[1][l] != ln.vs[1][l]) & ts_ok & (p | v | c) & (ln.ve[5][l] != ln.vs[5][l]) & !late;
        purchase[l] = p;
    }
    if phase() <= 7 {
        std::hint::black_box((&e_la, &u_la, &base_ok, &purchase, &ts_rel, &ts_fast, &ingest, &ing_fast));
        return;
    }
    // Emit: batchable lanes into one batch, the rest one by one.
    let mut out = EvBatch::default();
    for l in 0..LANES {
        let start = ln.ls[l] as usize;
        if !(ts_fast[l] & ing_fast[l]) {
            fast::count_slow();
            let line = &buf.data[start..ln.end[l] as usize];
            reference::event_one(f.ctx, file, start, line, fields, sink);
            continue;
        }
        if ln.ve[0][l] == ln.vs[0][l] {
            continue;
        }
        let at = Locator::new(file, start as u64);
        let ev = if x_e[l] != NONE40 {
            Key { long: false, a: x_e[l], b: 0 }
        } else if e_long[l] {
            Key { long: true, a: e_la[l], b: e_lb[l] }
        } else {
            sink.foreign_event(at);
            continue;
        };
        if !(0..INGEST_LIMIT as i64).contains(&ingest[l]) {
            sink.foreign_event(at);
            continue;
        }
        let ing = ingest[l] as u32;
        let mut price = 0u32;
        let survives = if purchase[l] & base_ok[l] {
            let (ps, pe) = (ln.vs[6][l] as usize, ln.ve[6][l] as usize);
            if pe - ps > 8 {
                match crate::ts::parse_price(&buf.data[ps..pe]) {
                    Some(v) => {
                        price = v;
                        true
                    }
                    None => false,
                }
            } else {
                let (v, ok) = unsafe { fast::dec_field(base, ps, pe) };
                price = v as u32;
                ok & (v >= 1) & (v <= 10_000_000)
            }
        } else {
            base_ok[l]
        };
        if !survives {
            out.push(ev, Key::default(), false, false, 0, ing, 0, 0);
            continue;
        }
        let user = if x_u[l] != NONE40 {
            Key { long: false, a: x_u[l], b: 0 }
        } else if u_long[l] {
            Key { long: true, a: u_la[l], b: u_lb[l] }
        } else {
            sink.foreign_event(at);
            continue;
        };
        let cat = if purchase[l] {
            fast::category(f, &buf.data[ln.vs[5][l] as usize..ln.ve[5][l] as usize])
        } else {
            0
        };
        out.push(ev, user, true, purchase[l], ts_rel[l] as u32, ing, cat, price);
    }
    if out.n > 0 {
        sink.events(&out);
    }
}

const EVENTS: u8 = 0;
const CLICKS: u8 = 1;

/// Every event line whose start is in from..to.
#[inline(always)]
pub(crate) unsafe fn events<S: Sink, const ISA: u8>(f: &Fast, file: u32, buf: Buf, from: usize, to: usize, sink: &mut S) {
    unsafe { lines::<S, ISA, EVENTS>(f, file, buf, from, to, sink) }
}

/// Every click line whose start is in from..to.
#[inline(always)]
pub(crate) unsafe fn clicks<S: Sink, const ISA: u8>(f: &Fast, file: u32, buf: Buf, from: usize, to: usize, sink: &mut S) {
    unsafe { lines::<S, ISA, CLICKS>(f, file, buf, from, to, sink) }
}

#[inline(always)]
unsafe fn lines<S: Sink, const ISA: u8, const KIND: u8>(f: &Fast, file: u32, buf: Buf, from: usize, to: usize, sink: &mut S) {
    let (kind, nf) = if KIND == EVENTS { (Kind::Events, 7usize) } else { (Kind::Clicks, 4usize) };
    let data = buf.data;
    let len = data.len();
    let base = data.as_ptr();
    let mut fields: Vec<&[u8]> = Vec::with_capacity(8);
    // One scratch per thread, reused across chunks (no allocation per chunk).
    let mut guard = ScratchGuard(Some(SCRATCH.with(|c| c.take()).unwrap_or_else(Scratch::new)));
    let sc = guard.0.as_mut().unwrap();
    let dates = &f.dates2;
    #[cfg(target_arch = "x86_64")]
    let mut ev_out = crate::fast3::Out { b: EvBatch::default(), slow: 0, foreign: 0, ls: [0; 8], le: [0; 8] };
    #[cfg(target_arch = "x86_64")]
    let mut cl_out = crate::fast3::Out { b: ClickBatch::default(), slow: 0, foreign: 0, ls: [0; 8], le: [0; 8] };
    // Phase A blocks read [b, b + 128); phase B reads up to 32 bytes past a line end.
    let a_limit = (len + buf.post).saturating_sub(256);
    let lo_ok = PRE.saturating_sub(buf.pre);
    let mut start = fast::first_start(data, from);
    while start < to {
        if start < lo_ok || start + 512 > a_limit || len >= (u32::MAX as usize) - 1024 {
            let l = unsafe { fast::one_line::<S, ISA>(f, kind, file, buf, start, &mut fields, sink) };
            start += l + 1;
            continue;
        }
        let wpos = start;
        let wend = (wpos + W).min(a_limit - 64);
        unsafe { phase_a::<ISA>(base, wpos, wend, sc) };
        if !sc.has_nl {
            let l = unsafe { fast::one_line::<S, ISA>(f, kind, file, buf, start, &mut fields, sink) };
            start += l + 1;
            continue;
        }
        let last = sc.last_nl;
        let next_window = wpos + sc.pos(last) + 1;
        if KIND == EVENTS && phase() <= 1 {
            std::hint::black_box(&sc.sep[..sc.nsep]);
            if next_window >= to {
                return;
            }
            start = next_window;
            continue;
        }
        let (mut k, mut vi) = (0usize, 0usize);
        let mut finished = false;
        while k <= last {
            let ls = wpos + sc.line_start(k);
            if ls >= to {
                finished = true;
                break;
            }
            if k + nf * LANES - 1 <= last
                && (wpos + sc.pos(k + nf * (LANES - 1) - 1) + 1) < to
                && sc.viol[vi] as usize > wpos + sc.pos(k + nf * LANES - 1)
                && unsafe { nl_pattern::<ISA>(sc, k, nf) }
            {
                unsafe {
                    if KIND == EVENTS {
                        #[cfg(target_arch = "x86_64")]
                        if ISA == fast::ISA_AVX512 && crate::fast3::VEC512.load(std::sync::atomic::Ordering::Relaxed) && crate::fast3::READY {
                            if crate::fast3::MUL256.load(std::sync::atomic::Ordering::Relaxed) {
                                crate::fast3::ev_batch::<true>(f, dates, buf.data.as_ptr(), sc, wpos, k, &mut ev_out);
                            } else {
                                crate::fast3::ev_batch::<false>(f, dates, buf.data.as_ptr(), sc, wpos, k, &mut ev_out);
                            }
                            crate::fast3::emit_events(f, file, buf, wpos, &ev_out, &mut fields, sink);
                            k += nf * LANES;
                            continue;
                        }
                        batch::<S, ISA>(f, dates, file, buf, sc, wpos, 0, k, &mut fields, sink)
                    } else {
                        #[cfg(target_arch = "x86_64")]
                        if ISA == fast::ISA_AVX512
                            && !f.camp32.is_empty()
                            && crate::fast3::VEC512.load(std::sync::atomic::Ordering::Relaxed)
                            && crate::fast3::READY
                        {
                            if crate::fast3::MUL256.load(std::sync::atomic::Ordering::Relaxed) {
                                crate::fast3::click_batch::<true>(f, dates, buf.data.as_ptr(), file, sc, wpos, k, &mut cl_out);
                            } else {
                                crate::fast3::click_batch::<false>(f, dates, buf.data.as_ptr(), file, sc, wpos, k, &mut cl_out);
                            }
                            crate::fast3::emit_clicks(f, file, buf, wpos, &cl_out, &mut fields, sink);
                            k += nf * LANES;
                            continue;
                        }
                        batch_clicks::<S, ISA>(f, dates, file, buf, sc, wpos, 0, k, &mut fields, sink)
                    }
                };
                k += nf * LANES;
                continue;
            }
            let e = sc.next_nl(k);
            let l = unsafe { fast::one_line::<S, ISA>(f, kind, file, buf, ls, &mut fields, sink) };
            debug_assert_eq!(ls + l, wpos + sc.pos(e));
            let end = (wpos + sc.pos(e)) as u32;
            while sc.viol[vi] <= end {
                vi += 1;
            }
            k = e + 1;
        }
        if finished {
            return;
        }
        start = next_window;
    }
}

/// The ts of a lane from its four loads (HI = [vs, vs+8), MID = +8, T16 = +16, T17 = +17) and
/// length: (ms relative to window_start, whether a fast shape decided it).
#[inline(always)]
fn ts_lane(f: &Fast, dates: &Dates, len: u32, hi8: u64, mid: u64, t16: u64, t17: u64) -> (i64, bool) {
    let (day, date_ok) = date_of(dates, hi8, mid);
    let t_ok = ((mid >> 16) & 0xFF) == b'T' as u64;
    let (secs, hms_ok) = hms((mid >> 24) | (t16 << 40));
    let z = ((t16 >> 24) & 0xFF) == b'Z' as u64;
    let (off, off_ok) = offset(t17);
    let dt_ok = date_ok & t_ok & hms_ok;
    let f1 = (len == 20) & z & dt_ok;
    let f2 = (len == 25) & off_ok & dt_ok;
    let (hi, hi_ok) = dec8(hi8);
    let (lo, lo_ok) = dec8((mid << 24) | 0x0030_3030);
    let f3 = (len == 13) & hi_ok & lo_ok;
    let local = day * DAY_MS + secs * 1000 - f.ws_frac;
    let f12 = local - if f2 { off } else { 0 };
    let v3 = (hi * 100_000 + lo) as i64 - f.ws_ms;
    (if f3 { v3 } else { f12 }, f1 | f2 | f3)
}

/// Decodes 8 click lines (i0..i0+8 of the window, separators from k0, stride 4) and emits them.
#[inline(always)]
unsafe fn batch_clicks<'b, S: Sink, const ISA: u8>(
    f: &Fast,
    dates: &Dates,
    file: u32,
    buf: Buf<'b>,
    sc: &Scratch,
    wpos: usize,
    i0: usize,
    k0: usize,
    fields: &mut Vec<&'b [u8]>,
    sink: &mut S,
) {
    let base = buf.data.as_ptr();
    let mut ls = [0u32; LANES];
    let mut end = [0u32; LANES];
    let mut vs = [[0u32; LANES]; 4];
    let mut ve = [[0u32; LANES]; 4];
    let mut pre_k = [0u8; LANES];
    for l in 0..LANES {
        let kl = k0 + 4 * l;
        let _ = i0;
        let w = wpos as u32;
        let s = w + sc.line_start(kl) as u32;
        ls[l] = s;
        end[l] = w + sc.pos(kl + 3) as u32;
        let q0 = sc.q_before(kl);
        let b01 = unsafe { rd(base.add(s as usize) as *const u16) };
        pre_k[l] = if q0 { (b01 >> 8) as u8 } else { b01 as u8 };
        vs[0][l] = s + q0 as u32;
        ve[0][l] = w + sc.pos(kl) as u32 - q0 as u32;
        for j in 1..4 {
            let q = (sc.sep[kl + j - 1] & Q_FLAG != 0) as u32;
            vs[j][l] = w + sc.pos(kl + j - 1) as u32 + 1 + q;
            ve[j][l] = w + sc.pos(kl + j) as u32 - q;
        }
    }
    let mut w_k = [0u64; LANES];
    let mut w_u = [0u64; LANES];
    let mut lu_w = [[0u64; LANES]; 3];
    let mut pre_u = [0u8; LANES];
    let mut c_pre = [0u16; LANES];
    let mut c_w = [0u64; LANES];
    let mut t_hi = [0u64; LANES];
    let mut t_mid = [0u64; LANES];
    let mut t_16 = [0u64; LANES];
    let mut t_17 = [0u64; LANES];
    unsafe {
        for l in 0..LANES {
            // ve >= line start >= wpos >= PRE - pre, so ve - 8 >= -pre.
            w_k[l] = rd(base.offset(ve[0][l] as isize - 8) as *const u64);
            w_u[l] = rd(base.offset(ve[1][l] as isize - 8) as *const u64);
            for t in 0..3 {
                lu_w[t][l] = rd(base.offset(ve[1][l] as isize - 32 + 8 * t as isize) as *const u64);
            }
            pre_u[l] = *base.add(vs[1][l] as usize);
            c_pre[l] = rd(base.add(vs[2][l] as usize) as *const u16);
            c_w[l] = rd(base.offset(ve[2][l] as isize - 8) as *const u64);
            let t = base.add(vs[3][l] as usize);
            t_hi[l] = rd(t as *const u64);
            t_mid[l] = rd(t.add(8) as *const u64);
            t_16[l] = rd(t.add(16) as *const u64);
            t_17[l] = rd(t.add(17) as *const u64);
        }
    }
    let mut x_u = [0u64; LANES];
    let mut u_long = [false; LANES];
    let mut u_la = [0u64; LANES];
    let mut u_lb = [0u64; LANES];
    let mut ts_rel = [0i64; LANES];
    let mut ts_fast = [false; LANES];
    let mut camp_n = [0u64; LANES];
    let mut camp_ok = [false; LANES];
    for l in 0..LANES {
        let lu = (ve[1][l] - vs[1][l]) as u64;
        x_u[l] = id_from(w_u[l], lu.wrapping_sub(1), pre_u[l] == b'u');
        let (a0, k0) = fast::hex8(lu_w[0][l]);
        let (a1, k1) = fast::hex8(lu_w[1][l]);
        let (b0, k2) = fast::hex8(lu_w[2][l]);
        let (b1, k3) = fast::hex8(w_u[l]);
        u_long[l] = (lu == 33) & (pre_u[l] == b'u') & k0 & k1 & k2 & k3;
        u_la[l] = (a0 << 32) | a1;
        u_lb[l] = (b0 << 32) | b1;
        let (r, fs) = ts_lane(f, dates, ve[3][l] - vs[3][l], t_hi[l], t_mid[l], t_16[l], t_17[l]);
        ts_rel[l] = r;
        ts_fast[l] = fs;
        // Campaign `c_` + 1..8 canonical digits (the numbered table's only possible keys).
        let kc = ((ve[2][l] - vs[2][l]) as u64).wrapping_sub(2);
        let kcl = kc.clamp(1, 8);
        let keep = !0u64 << (8 * (8 - kcl));
        let w = (c_w[l] & keep) | (0x3030_3030_3030_3030 & !keep);
        let (v, ok) = dec8(w);
        let first = ((w >> (8 * (8 - kcl))) & 0xFF) as u8;
        camp_n[l] = v;
        camp_ok[l] = ok & (kc >= 1) & (kc <= 8) & ((kc == 1) | (first != b'0')) & (c_pre[l] == u16::from_le_bytes(*b"c_"));
    }
    let lo = -CLICK_BASE_BEFORE_WINDOW_MS;
    let mut out = ClickBatch::default();
    for l in 0..LANES {
        let start = ls[l] as usize;
        if !ts_fast[l] {
            fast::count_slow();
            let line = &buf.data[start..end[l] as usize];
            reference::click_one(f.ctx, file, start, line, fields, sink);
            continue;
        }
        if ve[0][l] == vs[0][l] || ve[1][l] == vs[1][l] {
            continue;
        }
        if !((ts_rel[l] >= lo) & (ts_rel[l] < f.win_ms)) {
            continue;
        }
        let camp = match f.campaigns {
            Some(t) => {
                if !camp_ok[l] || camp_n[l] as usize >= t.by_number.len() {
                    continue;
                }
                let v = t.by_number[camp_n[l] as usize];
                if v == crate::tables::NumberedIds::NONE {
                    continue;
                }
                v
            }
            None => match f.ctx.campaigns.number(&buf.data[vs[2][l] as usize..ve[2][l] as usize]) {
                Some(v) => v,
                None => continue,
            },
        };
        let at = Locator::new(file, start as u64);
        let user = if x_u[l] != NONE40 {
            Key { long: false, a: x_u[l], b: 0 }
        } else if u_long[l] {
            Key { long: true, a: u_la[l], b: u_lb[l] }
        } else {
            sink.foreign_click(at);
            continue;
        };
        out.push(user, (ts_rel[l] - lo) as u32, camp, at);
    }
    if out.n > 0 {
        sink.clicks(&out);
    }
    let _ = (&w_k, &pre_k);
}

