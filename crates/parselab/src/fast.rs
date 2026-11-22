//! The fast parser: the same records as `reference`, generic over the sink.
//!
//! Per line: two 64-byte loads give 128-bit masks of commas, quotes and newlines (AVX-512BW, AVX2
//! or SWAR, chosen at run time; the masks are the only part that differs, so all paths decode
//! identically). Quote legality is three mask identities; field bounds come from the comma chain;
//! IDs, timestamps and numbers are decoded with branch-free SWAR. A timestamp's F1/F2 and F3
//! readings are both computed and one is selected by its length. Calendar dates are looked up in a
//! perfect-hash table of every valid date that can matter (a date not in it cannot be in any window).
//!
//! Anything outside the fast shapes goes to the reference's per-line function, so it is exact by
//! construction: a line longer than 127 bytes, a line too close to the start or end of the buffer
//! for the wide loads, a wrong field count, illegal quoting, a timestamp of another length or
//! pattern, an ingest outside the date table, a price longer than 8 digits.

use std::ptr::read_unaligned as rd;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    Scalar,
    Avx2,
    Avx512,
}

impl Isa {
    /// The widest path this CPU runs (AVX-512 needs F, BW, DQ, VL and BMI2).
    pub fn detect() -> Isa {
        #[cfg(target_arch = "x86_64")]
        {
            if (std::env::var_os("ENGINE_NO512").is_none() && std::is_x86_feature_detected!("avx512f"))
                && std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512dq")
                && std::is_x86_feature_detected!("avx512vl")
                && std::is_x86_feature_detected!("bmi2")
            {
                return Isa::Avx512;
            }
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("bmi2") {
                return Isa::Avx2;
            }
        }
        Isa::Scalar
    }
}
use crate::ids::MASK40;
use crate::records::*;
use crate::reference;
use crate::tables::{Ctx, NumberedIds};
use crate::ts::{DAY_MS, days_from_civil};

/// Lines are read through a 128-byte window.
pub const WINDOW: usize = 128;
/// Bytes that must be readable before a line start: the lane batches read the 32 bytes that end
/// at an ID's end (a LONG ID's hex digits), and the ID field can start at the line start.
pub const PRE: usize = 40;

#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub(crate) struct DateEnt {
    /// Bytes 2..10 of `YYYY-MM-DD`, little-endian.
    key: u64,
    /// Bytes 0..2.
    cent: u16,
    /// True only for slots that hold a date. An empty slot never matches, whatever the input
    /// bytes are (all-NUL input included): no key value is reserved for "empty".
    valid: bool,
    /// Days since the day of window_start.
    day: i32,
}

/// Precomputed constants and tables for one dataset.
pub struct Fast<'a> {
    pub ctx: &'a Ctx,
    pub(crate) dates: Vec<DateEnt>,
    pub(crate) date_magic: u64,
    pub(crate) date_shift: u32,
    /// window_start - its day's midnight, ms.
    pub(crate) ws_frac: i64,
    pub(crate) ws_ms: i64,
    /// window_end - window_start, ms.
    pub(crate) win_ms: i64,
    /// ingest field = (day * 86400 + secs) + ingest_add, with day relative to window_start's day.
    pub(crate) ingest_add: i64,
    pub(crate) products: Option<&'a NumberedIds>,
    pub(crate) campaigns: Option<&'a NumberedIds>,
    pub(crate) unknown: u8,
    /// Direct date table of the v2 event path.
    pub(crate) dates2: crate::fast2::Dates,
    /// Campaign number by `c_` number as u32 (NONE = 0xFFFF), padded for 32-bit gathers; empty
    /// when campaign IDs are not all numbered.
    pub(crate) camp32: Vec<u32>,
    /// Category code by `p` number as u32 (0xFFFF = not in the file), padded; empty when
    /// product IDs are not all numbered.
    pub(crate) prod32: Vec<u32>,
}

impl<'a> Fast<'a> {
    pub fn new(ctx: &'a Ctx) -> Fast<'a> {
        #[cfg(target_arch = "x86_64")]
        crate::fast3::choose_paths();
        let ws = ctx.window_start_ms;
        let we = ctx.window_end_ms;
        let ws_day = ws.div_euclid(DAY_MS);
        let we_day = (we - 1).div_euclid(DAY_MS);
        // Every date a ts in either window (with an offset up to 23:59) or an ingest within about
        // 5 days of one can have. Dates outside: ts is out of window, ingest goes slow.
        let mut wanted = Vec::new();
        for day in (ws_day - 9)..=(we_day + 7) {
            let (y, m, d) = crate::ts::civil_from_days(day);
            if !(0..=9999).contains(&y) {
                continue;
            }
            let text = format!("{y:04}-{m:02}-{d:02}");
            let b = text.as_bytes();
            let key = u64::from_le_bytes(b[2..10].try_into().unwrap());
            let cent = u16::from_le_bytes(b[0..2].try_into().unwrap());
            debug_assert_eq!(days_from_civil(y, m, d), day);
            wanted.push(DateEnt { key, cent, valid: true, day: (day - ws_day) as i32 });
        }
        let (dates, date_magic, date_shift) = perfect_hash(&wanted);
        let ingest_base_s = ctx.ingest_base_s;
        Fast {
            ctx,
            dates,
            date_magic,
            date_shift,
            ws_frac: ws - ws_day * DAY_MS,
            ws_ms: ws,
            win_ms: we - ws,
            ingest_add: ws_day * 86_400 - ingest_base_s,
            products: ctx.products.numbered.as_ref(),
            campaigns: ctx.campaigns.numbered.as_ref(),
            unknown: ctx.products.unknown,
            dates2: crate::fast2::Dates::new(ws, we),
            camp32: match ctx.campaigns.numbered.as_ref() {
                Some(t) => {
                    let mut v: Vec<u32> = t.by_number.iter().map(|&x| x as u32).collect();
                    v.resize(v.len() + 16, NumberedIds::NONE as u32);
                    v
                }
                None => Vec::new(),
            },
            prod32: match ctx.products.numbered.as_ref() {
                Some(t) => {
                    let mut v: Vec<u32> = t.by_number.iter().map(|&x| x as u32).collect();
                    v.resize(v.len() + 16, NumberedIds::NONE as u32);
                    v
                }
                None => Vec::new(),
            },
        }
    }

    /// Day (relative to window_start's day) of the `YYYY-MM-DD` at p, if it is in the table.
    #[inline(always)]
    pub(crate) unsafe fn date(&self, p: *const u8) -> (i64, bool) {
        let key = unsafe { rd(p.add(2) as *const u64) };
        let cent = unsafe { rd(p as *const u16) };
        // In bounds: dates.len() == 1 << (64 - date_shift) (perfect_hash returns them together),
        // and the mask makes the bound hold even if that pairing were broken.
        let h = (key.wrapping_mul(self.date_magic) >> self.date_shift) as usize & (self.dates.len() - 1);
        let e = unsafe { *self.dates.get_unchecked(h) };
        (e.day as i64, e.valid & (e.key == key) & (e.cent == cent))
    }
}

/// A collision-free multiplicative hash of the date keys into 64..1024 slots. Empty slots hold
/// key 0, which no date string has (its bytes include `-`). If none is found, every slot stays
/// empty and every timestamp takes the reference path (correct, slow).
fn perfect_hash(wanted: &[DateEnt]) -> (Vec<DateEnt>, u64, u32) {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for bits in 6..=10u32 {
        let size = 1usize << bits;
        for _ in 0..20_000 {
            state = mix64(state.wrapping_add(0x9E37_79B9_7F4A_7C15));
            let magic = state | 1;
            let shift = 64 - bits;
            let mut table = vec![DateEnt::default(); size];
            let mut ok = true;
            for e in wanted {
                let h = (e.key.wrapping_mul(magic) >> shift) as usize;
                if table[h].valid {
                    ok = false;
                    break;
                }
                table[h] = *e;
            }
            if ok {
                return (table, magic, shift);
            }
        }
    }
    (vec![DateEnt::default(); 64], 1, 58)
}

// ------------------------------------------------------------------------------------------
// Masks of a 128-byte window.

#[derive(Clone, Copy)]
pub(crate) struct Masks {
    pub(crate) comma: u128,
    pub(crate) quote: u128,
    pub(crate) nl: u128,
}

#[inline(always)]
fn swar_eq(w: u64, b: u8) -> u64 {
    // 0x80 in every byte of w equal to b (exact, no false positives).
    let t = w ^ (0x0101_0101_0101_0101u64.wrapping_mul(b as u64));
    let f = ((t & 0x7F7F_7F7F_7F7F_7F7F).wrapping_add(0x7F7F_7F7F_7F7F_7F7F)) | t;
    !f & 0x8080_8080_8080_8080
}

#[inline(always)]
fn gather8(flags: u64) -> u64 {
    ((flags >> 7).wrapping_mul(0x0102_0408_1020_4080)) >> 56
}

#[inline(always)]
unsafe fn masks_scalar(p: *const u8) -> Masks {
    let (mut c, mut q, mut n) = (0u128, 0u128, 0u128);
    for i in 0..16 {
        let w = unsafe { rd(p.add(8 * i) as *const u64) };
        c |= (gather8(swar_eq(w, b',')) as u128) << (8 * i);
        q |= (gather8(swar_eq(w, b'"')) as u128) << (8 * i);
        n |= (gather8(swar_eq(w, b'\n')) as u128) << (8 * i);
    }
    Masks { comma: c, quote: q, nl: n }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn masks_avx2(p: *const u8) -> Masks {
    use std::arch::x86_64::*;
    unsafe {
        let cc = _mm256_set1_epi8(b',' as i8);
        let qq = _mm256_set1_epi8(b'"' as i8);
        let nn = _mm256_set1_epi8(b'\n' as i8);
        let (mut c, mut q, mut n) = (0u128, 0u128, 0u128);
        for i in 0..4 {
            let v = _mm256_loadu_si256(p.add(32 * i) as *const __m256i);
            c |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, cc)) as u32 as u128) << (32 * i);
            q |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, qq)) as u32 as u128) << (32 * i);
            n |= (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, nn)) as u32 as u128) << (32 * i);
        }
        Masks { comma: c, quote: q, nl: n }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn masks_avx512(p: *const u8) -> Masks {
    use std::arch::x86_64::*;
    unsafe {
        let v0 = _mm512_loadu_si512(p as *const __m512i);
        let v1 = _mm512_loadu_si512(p.add(64) as *const __m512i);
        let cc = _mm512_set1_epi8(b',' as i8);
        let qq = _mm512_set1_epi8(b'"' as i8);
        let nn = _mm512_set1_epi8(b'\n' as i8);
        let m = |v, x| _mm512_cmpeq_epi8_mask(v, x) as u128;
        Masks {
            comma: m(v0, cc) | (m(v1, cc) << 64),
            quote: m(v0, qq) | (m(v1, qq) << 64),
            nl: m(v0, nn) | (m(v1, nn) << 64),
        }
    }
}

pub(crate) const ISA_SCALAR: u8 = 0;
pub(crate) const ISA_AVX2: u8 = 1;
pub(crate) const ISA_AVX512: u8 = 2;

#[inline(always)]
pub(crate) unsafe fn masks<const ISA: u8>(p: *const u8) -> Masks {
    #[cfg(target_arch = "x86_64")]
    {
        if ISA == ISA_AVX512 {
            return unsafe { masks_avx512(p) };
        }
        if ISA == ISA_AVX2 {
            return unsafe { masks_avx2(p) };
        }
    }
    unsafe { masks_scalar(p) }
}

// ------------------------------------------------------------------------------------------
// Line structure.

/// Field value bounds (quotes removed) of a line with exactly `N - 1` commas and legal quoting.
#[inline(always)]
unsafe fn fields<const N: usize>(p: *const u8, len: usize, m: &Masks) -> Option<[(usize, usize); N]> {
    let lmask: u128 = (1u128 << len) - 1;
    let comma = m.comma & lmask;
    if comma.count_ones() as usize != N - 1 {
        return None;
    }
    let quote = m.quote & lmask;
    // Separators: the commas and everything from the line end on.
    let d = comma | !lmask;
    let n = !d;
    let fs = (comma << 1) | 1;
    let fe = (comma >> 1) | (1u128 << (len - 1));
    let s = quote & fs;
    let e = quote & fe;
    let closes = (n.wrapping_add(s) & d) >> 1;
    let legal = (quote == (s | e)) & (s & e == 0) & (e == closes);
    if !legal {
        return None;
    }
    let mut out = [(0usize, 0usize); N];
    let mut c = comma;
    let mut begin = 0usize;
    for f in out.iter_mut().take(N - 1) {
        let at = c.trailing_zeros() as usize;
        c &= c - 1;
        *f = (begin, at);
        begin = at + 1;
    }
    out[N - 1] = (begin, len);
    // Quote adjustment: a legal quote at a field's first byte has its partner at the last byte.
    for f in out.iter_mut() {
        let q = (f.0 < f.1 && unsafe { *p.add(f.0) } == b'"') as usize;
        f.0 += q;
        f.1 -= q;
    }
    Some(out)
}

// ------------------------------------------------------------------------------------------
// Field decoders. All take the line pointer and value bounds; all loads stay inside
// [line - PRE, line + WINDOW + 32).

const NONE40: u64 = u64::MAX;

/// Base-36 digit values of the 8 bytes in `w` (byte 0 most significant), and whether every byte
/// is `0-9a-z`.
#[inline(always)]
pub(crate) fn b36_digits(w: u64) -> (u64, bool) {
    let ascii = w & 0x8080_8080_8080_8080 == 0;
    // In [lo, hi] for bytes < 0x80: (x + (0x80 - lo)) has bit 7, (x + (0x7F - hi)) does not.
    let ge0 = w.wrapping_add(0x5050_5050_5050_5050); // x >= 0x30
    let gt9 = w.wrapping_add(0x4646_4646_4646_4646); // x >= 0x3A
    let gea = w.wrapping_add(0x1F1F_1F1F_1F1F_1F1F); // x >= 0x61
    let gtz = w.wrapping_add(0x0505_0505_0505_0505); // x >= 0x7B
    let digit = ge0 & !gt9 & 0x8080_8080_8080_8080;
    let lower = gea & !gtz & 0x8080_8080_8080_8080;
    let ok = ascii & ((digit | lower) == 0x8080_8080_8080_8080);
    let v = w.wrapping_sub(0x3030_3030_3030_3030).wrapping_sub((lower >> 7).wrapping_mul(39));
    (v, ok)
}

/// Combines 8 base-36 digit bytes (byte 0 most significant) into their value.
#[inline(always)]
pub(crate) fn b36_value(v: u64) -> u64 {
    let v = (v & 0x00FF_00FF_00FF_00FF).wrapping_mul(36) + ((v >> 8) & 0x00FF_00FF_00FF_00FF);
    let v = (v & 0x0000_FFFF_0000_FFFF).wrapping_mul(1296) + ((v >> 16) & 0x0000_FFFF_0000_FFFF);
    (v & 0xFFFF_FFFF).wrapping_mul(1_679_616) + (v >> 32)
}

/// The base-36 value of a conforming ID (`prefix` + 1..8 canonical digits, value < 2^40) at
/// [vs, ve), or NONE40.
#[inline(always)]
unsafe fn id_value(p: *const u8, vs: usize, ve: usize, prefix: u8) -> u64 {
    let k = ve.wrapping_sub(vs).wrapping_sub(1); // digit count
    let kc = k.clamp(1, 8);
    let w = unsafe { rd(p.add(ve).sub(8) as *const u64) };
    let keep = !0u64 << (8 * (8 - kc));
    let w = (w & keep) | (0x3030_3030_3030_3030 & !keep);
    let (v, ok) = b36_digits(w);
    let first = (v >> (8 * (8 - kc))) & 0xFF;
    let value = b36_value(v);
    let pre = unsafe { *p.add(vs) } == prefix;
    let good = ok & pre & (k >= 1) & (k <= 8) & ((k == 1) | (first != 0)) & (value <= MASK40);
    if good { value } else { NONE40 }
}


/// 8 ASCII digits (byte 0 most significant) to their value, and whether all are digits.
#[inline(always)]
pub(crate) fn dec8(w: u64) -> (u64, bool) {
    let x = w ^ 0x3030_3030_3030_3030;
    let ok = (x & 0xF0F0_F0F0_F0F0_F0F0 == 0)
        & ((x.wrapping_add(0x0606_0606_0606_0606) & 0x1010_1010_1010_1010) == 0);
    let v = x.wrapping_mul(10).wrapping_add(x >> 8);
    let v = ((v & 0x0000_00FF_0000_00FF).wrapping_mul(100 + (1_000_000 << 32))
        .wrapping_add(((v >> 16) & 0x0000_00FF_0000_00FF).wrapping_mul(1 + (10_000 << 32))))
        >> 32;
    (v, ok)
}

/// 1..8 decimal digits at [vs, ve), any leading zeros: (value, ok).
#[inline(always)]
pub(crate) unsafe fn dec_field(p: *const u8, vs: usize, ve: usize) -> (u64, bool) {
    let k = ve.wrapping_sub(vs);
    let kc = k.clamp(1, 8);
    let w = unsafe { rd(p.add(ve).sub(8) as *const u64) };
    let keep = !0u64 << (8 * (8 - kc));
    let w = (w & keep) | (0x3030_3030_3030_3030 & !keep);
    let (v, ok) = dec8(w);
    (v, ok & (k >= 1) & (k <= 8))
}

/// `HH:MM:SS` in the 8 bytes `t`: (seconds of day, ok).
#[inline(always)]
pub(crate) fn hms(t: u64) -> (i64, bool) {
    let colons = (t & 0x0000_FF00_00FF_0000) == 0x0000_3A00_003A_0000;
    let x = (t ^ 0x3030_3030_3030_3030) & 0xFFFF_00FF_FF00_FFFF;
    let digits = (x & 0xF0F0_F0F0_F0F0_F0F0 == 0)
        & ((x.wrapping_add(0x0606_0606_0606_0606) & 0x1010_1010_1010_1010) == 0);
    let v = x.wrapping_mul(10).wrapping_add(x >> 8);
    let h = (v & 0xFF) as i64;
    let m = ((v >> 24) & 0xFF) as i64;
    let s = ((v >> 48) & 0xFF) as i64;
    (h * 3600 + m * 60 + s, colons & digits & (h <= 23) & (m <= 59) & (s <= 59))
}

/// Offset `SS±HH:MM` in the 8 bytes at value + 17 of an F2: (signed offset in ms, ok).
#[inline(always)]
pub(crate) fn offset(o: u64) -> (i64, bool) {
    let sign = ((o >> 16) & 0xFF) as u8;
    let colon = ((o >> 40) & 0xFF) as u8 == b':';
    let x = (o ^ 0x3030_3030_3030_3030) & 0xFFFF_00FF_FF00_0000;
    let digits = (x & 0xF0F0_F0F0_F0F0_F0F0 == 0)
        & ((x.wrapping_add(0x0606_0606_0606_0606) & 0x1010_1010_1010_1010) == 0);
    let v = x.wrapping_mul(10).wrapping_add(x >> 8);
    let h = ((v >> 24) & 0xFF) as i64;
    let m = ((v >> 48) & 0xFF) as i64;
    let mag = (h * 60 + m) * 60_000;
    let plus = sign == b'+';
    let minus = sign == b'-';
    (if plus { mag } else { -mag }, (plus | minus) & colon & digits & (h <= 23) & (m <= 59))
}

/// A `ts` value at [vs, ve) relative to window_start, ms. `Some(ok, rel)` when the fast shapes
/// decide it (F1, F2 with the date in the table, F3 of 13 digits); None sends it to the reference.
#[inline(always)]
unsafe fn ts_fast(f: &Fast, p: *const u8, vs: usize, ve: usize) -> Option<i64> {
    let tp = unsafe { p.add(vs) };
    let len = ve - vs;
    let (day, date_ok) = unsafe { f.date(tp) };
    let t_ok = unsafe { *tp.add(10) } == b'T';
    let (secs, hms_ok) = hms(unsafe { rd(tp.add(11) as *const u64) });
    let (off, off_ok) = offset(unsafe { rd(tp.add(17) as *const u64) });
    let z = unsafe { *tp.add(19) } == b'Z';
    let dt_ok = date_ok & t_ok & hms_ok;
    let f1 = (len == 20) & z & dt_ok;
    let f2 = (len == 25) & off_ok & dt_ok;
    let (hi, hi_ok) = dec8(unsafe { rd(tp as *const u64) });
    let lo_raw = unsafe { rd(tp.add(5) as *const u64) };
    let (lo, lo_ok) = dec8((lo_raw & 0xFFFF_FFFF_FF00_0000) | 0x0030_3030);
    let (_, lo_all) = dec8(lo_raw);
    let f3 = (len == 13) & hi_ok & lo_ok & lo_all;
    let local = day * DAY_MS + secs * 1000 - f.ws_frac;
    let f12 = local - if f2 { off } else { 0 };
    let v3 = (hi * 100_000 + lo) as i64 - f.ws_ms;
    let rel = if f3 { v3 } else { f12 };
    if f1 | f2 | f3 { Some(rel) } else { None }
}

/// An F1 ingest at [vs, ve): the ingest field if it is F1 with its date in the table.
#[inline(always)]
unsafe fn ingest_fast(f: &Fast, p: *const u8, vs: usize, ve: usize) -> Option<i64> {
    let tp = unsafe { p.add(vs) };
    let (day, date_ok) = unsafe { f.date(tp) };
    let t_ok = unsafe { *tp.add(10) } == b'T';
    let (secs, hms_ok) = hms(unsafe { rd(tp.add(11) as *const u64) });
    let z = unsafe { *tp.add(19) } == b'Z';
    let ok = (ve - vs == 20) & date_ok & t_ok & hms_ok & z;
    if ok { Some(day * 86_400 + secs + f.ingest_add) } else { None }
}

#[inline(always)]
pub(crate) fn slice<'b>(p: *const u8, (s, e): (usize, usize)) -> &'b [u8] {
    unsafe { std::slice::from_raw_parts(p.add(s), e - s) }
}

// ------------------------------------------------------------------------------------------
// Lines.

/// 8 hex characters (byte 0 most significant) to their 32-bit value, and whether all are
/// `0-9a-f`.
#[inline(always)]
pub(crate) fn hex8(w: u64) -> (u64, bool) {
    let ascii = w & 0x8080_8080_8080_8080 == 0;
    let ge0 = w.wrapping_add(0x5050_5050_5050_5050); // x >= 0x30
    let gt9 = w.wrapping_add(0x4646_4646_4646_4646); // x >= 0x3A
    let gea = w.wrapping_add(0x1F1F_1F1F_1F1F_1F1F); // x >= 0x61
    let gtf = w.wrapping_add(0x1919_1919_1919_1919); // x >= 0x67
    let digit = ge0 & !gt9 & 0x8080_8080_8080_8080;
    let lower = gea & !gtf & 0x8080_8080_8080_8080;
    let ok = ascii & ((digit | lower) == 0x8080_8080_8080_8080);
    let v = w.wrapping_sub(0x3030_3030_3030_3030).wrapping_sub((lower >> 7).wrapping_mul(39));
    let t = ((v & 0x00FF_00FF_00FF_00FF) << 4) | ((v >> 8) & 0x00FF_00FF_00FF_00FF);
    let u = ((t & 0x0000_FFFF_0000_FFFF) << 8) | ((t >> 16) & 0x0000_FFFF_0000_FFFF);
    (((u & 0xFFFF_FFFF) << 16) | (u >> 32), ok)
}

/// The key of the ID at [vs, ve) of the line at p: SHORT, LONG, or None (FOREIGN). Reads
/// [vs, ve) and the 8 bytes before ve.
#[inline(always)]
pub(crate) unsafe fn key_at(p: *const u8, vs: usize, ve: usize, prefix: u8) -> Option<Key> {
    let x = unsafe { id_value(p, vs, ve, prefix) };
    if x != NONE40 {
        return Some(Key { long: false, a: x, b: 0 });
    }
    if ve - vs == 33 && unsafe { *p.add(vs) } == prefix {
        let h = |o: usize| hex8(unsafe { rd(p.add(vs + 1 + o) as *const u64) });
        let (a0, k0) = h(0);
        let (a1, k1) = h(8);
        let (b0, k2) = h(16);
        let (b1, k3) = h(24);
        if k0 & k1 & k2 & k3 {
            return Some(Key { long: true, a: (a0 << 32) | a1, b: (b0 << 32) | b1 });
        }
    }
    None
}

/// One event line of length `len` at p (file offset `start`). False: give it to the reference.
#[inline(always)]
unsafe fn event_line<S: Sink>(f: &Fast, file: u32, start: usize, p: *const u8, len: usize, m: &Masks, sink: &mut S) -> bool {
    let Some(fl) = (unsafe { fields::<7>(p, len, m) }) else { return false };
    let [f_id, f_user, f_ts, f_ing, f_type, f_prod, f_price] = fl;
    if f_id.0 == f_id.1 {
        return true; // empty event_id: nothing
    }
    // Timestamps first: an unusual one sends the whole line to the reference.
    let ts = unsafe { ts_fast(f, p, f_ts.0, f_ts.1) };
    let ing = unsafe { ingest_fast(f, p, f_ing.0, f_ing.1) };
    let (Some(ts_rel), Some(ingest)) = (ts, ing) else { return false };
    let at = Locator::new(file, start as u64);
    let Some(ev) = (unsafe { key_at(p, f_id.0, f_id.1, b'e') }) else {
        sink.foreign_event(at);
        return true;
    };
    if !(0..INGEST_LIMIT as i64).contains(&ingest) {
        sink.foreign_event(at);
        return true;
    }
    let ingest = ingest as u32;
    // Validation.
    let tlen = f_type.1 - f_type.0;
    let tw = unsafe { rd(p.add(f_type.0) as *const u64) };
    let purchase = (tlen == 8) & (tw == u64::from_le_bytes(*b"purchase"));
    let view = (tlen == 4) & (tw as u32 == u32::from_le_bytes(*b"view"));
    let cart = (tlen == 4) & (tw as u32 == u32::from_le_bytes(*b"cart"));
    let ts_ok = (ts_rel >= 0) & (ts_rel < f.win_ms);
    let ingest_rel_ms = (ingest as i64 - f.ingest_add) * 1000 - f.ws_frac;
    let late = ingest_rel_ms - ts_rel > DAY_MS;
    let base_ok = (f_user.0 != f_user.1) & ts_ok & (purchase | view | cart) & (f_prod.0 != f_prod.1) & !late;
    let mut price = 0u32;
    let survives = if purchase & base_ok {
        if f_price.1 - f_price.0 > 8 {
            // Long prices (leading zeros) are read by the reference rule.
            match crate::ts::parse_price(slice(p, f_price)) {
                Some(v) => {
                    price = v;
                    true
                }
                None => false,
            }
        } else {
            let (v, ok) = unsafe { dec_field(p, f_price.0, f_price.1) };
            price = v as u32;
            ok & (v >= 1) & (v <= 10_000_000)
        }
    } else {
        base_ok
    };
    let mut b = EvBatch::default();
    if !survives {
        b.push(ev, Key::default(), false, false, 0, ingest, 0, 0);
        sink.events(&b);
        return true;
    }
    let Some(user) = (unsafe { key_at(p, f_user.0, f_user.1, b'u') }) else {
        sink.foreign_event(at);
        return true;
    };
    let cat = if purchase { category(f, slice(p, f_prod)) } else { 0 };
    b.push(ev, user, true, purchase, ts_rel as u32, ingest, cat, price);
    sink.events(&b);
    true
}

#[inline(never)]
pub(crate) fn category(f: &Fast, product: &[u8]) -> u8 {
    match f.products {
        Some(t) => {
            let v = t.get(product);
            if v == NumberedIds::NONE { f.unknown } else { v as u8 }
        }
        None => f.ctx.products.code(product),
    }
}

#[inline(always)]
fn campaign(f: &Fast, id: &[u8]) -> Option<u16> {
    match f.campaigns {
        Some(t) => {
            let v = t.get(id);
            (v != NumberedIds::NONE).then_some(v)
        }
        None => f.ctx.campaigns.number(id),
    }
}

#[inline(always)]
unsafe fn click_line<S: Sink>(f: &Fast, file: u32, start: usize, p: *const u8, len: usize, m: &Masks, sink: &mut S) -> bool {
    let Some(fl) = (unsafe { fields::<4>(p, len, m) }) else { return false };
    let [f_id, f_user, f_camp, f_ts] = fl;
    if f_id.0 == f_id.1 {
        return true;
    }
    let Some(ts_rel) = (unsafe { ts_fast(f, p, f_ts.0, f_ts.1) }) else { return false };
    let lo = -(CLICK_BASE_BEFORE_WINDOW_MS);
    let ts_ok = (ts_rel >= lo) & (ts_rel < f.win_ms);
    if !(ts_ok & (f_user.0 != f_user.1)) {
        return true;
    }
    let Some(camp) = campaign(f, slice(p, f_camp)) else { return true };
    let at = Locator::new(file, start as u64);
    match unsafe { key_at(p, f_user.0, f_user.1, b'u') } {
        Some(user) => {
            let mut b = ClickBatch::default();
            b.push(user, (ts_rel - lo) as u32, camp, at);
            sink.clicks(&b);
        }
        None => sink.foreign_click(at),
    }
    true
}

#[inline(always)]
unsafe fn link_line<S: Sink>(_f: &Fast, file: u32, start: usize, p: *const u8, len: usize, m: &Masks, sink: &mut S) -> bool {
    let Some([a, b]) = (unsafe { fields::<2>(p, len, m) }) else { return false };
    if (a.0 == a.1) | (b.0 == b.1) {
        return true;
    }
    match unsafe { (key_at(p, a.0, a.1, b'u'), key_at(p, b.0, b.1, b'u')) } {
        (Some(ka), Some(kb)) => {
            let mut lb = LinkBatch::default();
            lb.push(ka, kb);
            sink.links(&lb);
        }
        _ => sink.foreign_link(Locator::new(file, start as u64)),
    }
    true
}

/// `link_line` for the windowed loop: the lane goes into `lb`, handed to the sink at 8.
#[inline(always)]
unsafe fn link_lane<S: Sink>(file: u32, start: usize, p: *const u8, len: usize, m: &Masks, lb: &mut LinkBatch, sink: &mut S) -> bool {
    let Some([a, b]) = (unsafe { fields::<2>(p, len, m) }) else { return false };
    if (a.0 == a.1) | (b.0 == b.1) {
        return true;
    }
    match unsafe { (key_at(p, a.0, a.1, b'u'), key_at(p, b.0, b.1, b'u')) } {
        (Some(ka), Some(kb)) => {
            lb.push(ka, kb);
            if lb.n == 8 {
                sink.links(lb);
                *lb = LinkBatch::default();
            }
        }
        _ => sink.foreign_link(Locator::new(file, start as u64)),
    }
    true
}

/// Links: every line that ends inside one 128-byte window from that window's masks (a link line
/// is about 18 bytes; one mask computation per line was half the links phase). Lanes reach the
/// sink 8 at a time. Bounds as in `one_line`: the window starts inside [fast_lo, fast_hi], and
/// every read of a line in it stays below window start + WINDOW + 32.
#[inline(always)]
unsafe fn links<S: Sink, const ISA: u8>(f: &Fast, file: u32, buf: Buf, from: usize, to: usize, sink: &mut S) {
    let data = buf.data;
    let len = data.len();
    let base = data.as_ptr();
    let fast_lo = PRE.saturating_sub(buf.pre);
    let fast_hi = (len + buf.post).saturating_sub(WINDOW + 32);
    let mut fields: Vec<&[u8]> = Vec::with_capacity(8);
    let mut lb = LinkBatch::default();
    let mut start = first_start(data, from);
    while start < to {
        if start < fast_lo || start > fast_hi {
            let l = unsafe { one_line::<S, ISA>(f, Kind::Links, file, buf, start, &mut fields, sink) };
            start += l + 1;
            continue;
        }
        let p = unsafe { base.add(start) };
        let mut m = unsafe { masks::<ISA>(p) };
        if len - start < WINDOW {
            m.nl |= 1u128 << (len - start);
        }
        let mut off = 0usize;
        while off < WINDOW && start + off < to {
            let nl = m.nl >> off;
            if nl == 0 {
                break;
            }
            let l = nl.trailing_zeros() as usize;
            let at = start + off;
            let ms = Masks { comma: m.comma >> off, quote: m.quote >> off, nl };
            if !unsafe { link_lane(file, at, p.add(off), l, &ms, &mut lb, sink) } {
                count_slow();
                reference::link_one(f.ctx, file, at, &data[at..at + l], &mut fields, sink);
            }
            off += l + 1;
        }
        if off == 0 {
            let l = unsafe { one_line::<S, ISA>(f, Kind::Links, file, buf, start, &mut fields, sink) };
            start += l + 1;
        } else {
            start += off;
        }
    }
    if lb.n > 0 {
        sink.links(&lb);
    }
}

// ------------------------------------------------------------------------------------------
// Chunks.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Events,
    Clicks,
    Links,
}

/// A file's bytes and how far reads may stray outside them: `pre` readable bytes before `ptr`,
/// `post` after `ptr + len`. A plain mapping has (0, 0); a padded copy (PRE, WINDOW).
#[derive(Clone, Copy)]
pub struct Buf<'b> {
    pub data: &'b [u8],
    pub pre: usize,
    pub post: usize,
}

impl<'b> Buf<'b> {
    pub fn plain(data: &'b [u8]) -> Self {
        Buf { data, pre: 0, post: 0 }
    }
}

/// A copy of `data` with PRE bytes before and WINDOW bytes after (tests and the tail of files).
pub struct Padded {
    store: Vec<u8>,
    len: usize,
}

impl Padded {
    pub fn new(data: &[u8]) -> Padded {
        let mut store = vec![0u8; PRE + data.len() + WINDOW + 64];
        store[PRE..PRE + data.len()].copy_from_slice(data);
        Padded { store, len: data.len() }
    }
    pub fn buf(&self) -> Buf<'_> {
        Buf { data: &self.store[PRE..PRE + self.len], pre: PRE, post: WINDOW + 64 }
    }
}

/// Lines handed to the reference (diagnostics, only with `--features phases`: a shared counter
/// bounces a cache line between threads).
pub static SLOW_LINES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[inline(always)]
pub(crate) fn count_slow() {
    if cfg!(feature = "phases") {
        SLOW_LINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

pub(crate) fn next_nl(data: &[u8], from: usize) -> usize {
    data[from..].iter().position(|&c| c == b'\n').map_or(data.len(), |p| from + p)
}

/// The one line starting at `start` (v1 path, reference fallback inside). Returns its length.
#[inline(always)]
pub(crate) unsafe fn one_line<'b, S: Sink, const ISA: u8>(
    f: &Fast,
    kind: Kind,
    file: u32,
    buf: Buf<'b>,
    start: usize,
    fields: &mut Vec<&'b [u8]>,
    sink: &mut S,
) -> usize {
    let data = buf.data;
    let len = data.len();
    let base = data.as_ptr();
    // Lines that may use the wide loads: start - PRE readable, start + WINDOW + 32 readable.
    let fast_lo = PRE.saturating_sub(buf.pre);
    let fast_hi = (len + buf.post).saturating_sub(WINDOW + 32);
    let mut done = false;
    let line_len: usize;
    if start >= fast_lo && start <= fast_hi {
        let p = unsafe { base.add(start) };
        let mut m = unsafe { masks::<ISA>(p) };
        if len - start < WINDOW {
            // The buffer ends inside the window: a final line without `\n` ends there.
            m.nl |= 1u128 << (len - start);
        }
        if m.nl != 0 {
            line_len = m.nl.trailing_zeros() as usize;
            done = unsafe {
                match kind {
                    Kind::Events => event_line(f, file, start, p, line_len, &m, sink),
                    Kind::Clicks => click_line(f, file, start, p, line_len, &m, sink),
                    Kind::Links => link_line(f, file, start, p, line_len, &m, sink),
                }
            };
        } else {
            line_len = next_nl(data, start) - start;
        }
    } else {
        line_len = next_nl(data, start) - start;
    }
    if !done {
        count_slow();
        let line = &data[start..start + line_len];
        match kind {
            Kind::Events => reference::event_one(f.ctx, file, start, line, fields, sink),
            Kind::Clicks => reference::click_one(f.ctx, file, start, line, fields, sink),
            Kind::Links => reference::link_one(f.ctx, file, start, line, fields, sink),
        }
    }
    line_len
}

/// The first line start at or after `from` (the header at offset 0 is skipped).
pub(crate) fn first_start(data: &[u8], from: usize) -> usize {
    if from == 0 { next_nl(data, 0) + 1 } else { next_nl(data, from - 1) + 1 }
}

/// Every line whose start is in from..to (the header at offset 0 excluded), like the reference.
#[inline(always)]
unsafe fn chunk<S: Sink, const ISA: u8>(f: &Fast, kind: Kind, file: u32, buf: Buf, from: usize, to: usize, sink: &mut S) {
    let data = buf.data;
    let to = to.min(data.len());
    if from >= to {
        return;
    }
    if kind == Kind::Events && V2.load(std::sync::atomic::Ordering::Relaxed) {
        return unsafe { crate::fast2::events::<S, ISA>(f, file, buf, from, to, sink) };
    }
    if kind == Kind::Clicks && V2.load(std::sync::atomic::Ordering::Relaxed) {
        return unsafe { crate::fast2::clicks::<S, ISA>(f, file, buf, from, to, sink) };
    }
    if kind == Kind::Links && V2.load(std::sync::atomic::Ordering::Relaxed) {
        return unsafe { links::<S, ISA>(f, file, buf, from, to, sink) };
    }
    let mut start = first_start(data, from);
    let mut fields: Vec<&[u8]> = Vec::with_capacity(8);
    while start < to {
        let line_len = unsafe { one_line::<S, ISA>(f, kind, file, buf, start, &mut fields, sink) };
        start += line_len + 1;
    }
}

/// Whether event lines take the v2 (block structure, 8-line batches) path.
pub static V2: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512cd,avx2,fma,bmi1,bmi2,popcnt,lzcnt")]
unsafe fn chunk_avx512<S: Sink>(f: &Fast, kind: Kind, file: u32, buf: Buf, from: usize, to: usize, sink: &mut S) {
    unsafe { chunk::<S, ISA_AVX512>(f, kind, file, buf, from, to, sink) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,bmi1,bmi2,popcnt,lzcnt")]
unsafe fn chunk_avx2<S: Sink>(f: &Fast, kind: Kind, file: u32, buf: Buf, from: usize, to: usize, sink: &mut S) {
    unsafe { chunk::<S, ISA_AVX2>(f, kind, file, buf, from, to, sink) }
}

fn chunk_scalar<S: Sink>(f: &Fast, kind: Kind, file: u32, buf: Buf, from: usize, to: usize, sink: &mut S) {
    unsafe { chunk::<S, ISA_SCALAR>(f, kind, file, buf, from, to, sink) }
}

/// Parses the lines of `buf` whose start is in from..to into `sink`, with the given ISA path.
///
/// Release contract (for releasing consumed input behind the parse): when this returns, no later
/// read by this call touches the bytes of [from + 4096, to - 4096). Neighbouring ranges of the
/// same file read only within about 512 bytes of their own boundaries (the line resync, the last
/// line, the 128-byte look-ahead block, the 40-byte look-behind), so the whole pages strictly
/// inside [from + 4096, to - 4096) may be released at once, and a boundary page once both ranges
/// that share it have returned. Software prefetches beyond `to` never fault.
pub fn parse<S: Sink>(f: &Fast, isa: Isa, kind: Kind, file: u32, buf: Buf, from: usize, to: usize, sink: &mut S) {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => unsafe { chunk_avx512(f, kind, file, buf, from, to, sink) },
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => unsafe { chunk_avx2(f, kind, file, buf, from, to, sink) },
        _ => chunk_scalar(f, kind, file, buf, from, to, sink),
    }
}
