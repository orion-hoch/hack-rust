//! AVX2 lane decoders for the portable batch (machines without AVX-512: Zen 2 measured 123 cycles
//! per event line in the scalar ts and ingest decoders, every format computed for every lane,
//! since the formats come mixed). Four lanes per 256-bit vector, two vectors per 8-line batch;
//! lane-wise copies of `fast::hms`, `fast::dec8`, `fast::offset` and `fast2::date_of`.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use crate::fast::Fast;
use crate::fast2::{BAD_DAY, DATE_SLOTS, Dates};

const DAY_MS: i64 = 86_400_000;

#[inline(always)]
unsafe fn c(x: u64) -> __m256i {
    _mm256_set1_epi64x(x as i64)
}

/// Lanes whose bytes selected by `mask` of x (= bytes ^ 0x30) are all digits.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn digits_ok(x: __m256i, mask: u64) -> __m256i {
    let z = _mm256_setzero_si256();
    let xd = _mm256_and_si256(x, unsafe { c(mask) });
    let hi = _mm256_cmpeq_epi64(_mm256_and_si256(xd, unsafe { c(0xF0F0_F0F0_F0F0_F0F0) }), z);
    let ov = _mm256_cmpeq_epi64(_mm256_and_si256(_mm256_add_epi8(xd, _mm256_set1_epi8(6)), unsafe { c(0x1010_1010_1010_1010) }), z);
    _mm256_and_si256(hi, ov)
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn le(a: __m256i, b: i64) -> __m256i {
    // a <= b, signed lanes
    _mm256_xor_si256(_mm256_cmpgt_epi64(a, _mm256_set1_epi64x(b)), _mm256_set1_epi64x(-1))
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn ge(a: __m256i, b: i64) -> __m256i {
    _mm256_cmpgt_epi64(a, _mm256_set1_epi64x(b - 1))
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn eq(a: __m256i, b: u64) -> __m256i {
    _mm256_cmpeq_epi64(a, unsafe { c(b) })
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn date(dates: &Dates, hi: __m256i, mid: __m256i) -> (__m256i, __m256i) {
    let c30 = unsafe { c(0x3030_3030_3030_3030) };
    let x = _mm256_xor_si256(hi, c30);
    let dig = unsafe { digits_ok(x, 0x00FF_FF00_FFFF_FFFF) };
    let seps = unsafe { eq(_mm256_and_si256(hi, c(0xFF00_00FF_0000_0000)), 0x2D00_002D_0000_0000) };
    let xd = _mm256_and_si256(x, unsafe { c(0x00FF_FF00_FFFF_FFFF) });
    let t = _mm256_maddubs_epi16(xd, unsafe { c(0x0001_0A00_010A_010A) });
    let t = _mm256_madd_epi16(t, unsafe { c(0x0001_0001_0001_0064) });
    let year = _mm256_and_si256(t, unsafe { c(0xFFFF_FFFF) });
    let month = _mm256_srli_epi64::<32>(t);
    let xm = _mm256_xor_si256(mid, c30);
    let mdig = unsafe { digits_ok(xm, 0xFFFF) };
    let ff = unsafe { c(0xFF) };
    let d0 = _mm256_and_si256(xm, ff);
    let d = _mm256_add_epi64(_mm256_add_epi64(_mm256_slli_epi64::<3>(d0), _mm256_slli_epi64::<1>(d0)), _mm256_and_si256(_mm256_srli_epi64::<8>(xm), ff));
    let yy = _mm256_sub_epi64(year, _mm256_set1_epi64x(dates.y0_()));
    let in_range = _mm256_and_si256(
        _mm256_and_si256(_mm256_and_si256(dig, seps), _mm256_and_si256(mdig, _mm256_and_si256(unsafe { ge(yy, 0) }, unsafe { le(yy, 1) }))),
        _mm256_and_si256(
            _mm256_and_si256(unsafe { ge(month, 1) }, unsafe { le(month, 12) }),
            _mm256_and_si256(unsafe { ge(d, 1) }, unsafe { le(d, 31) }),
        ),
    );
    let idx = _mm256_add_epi64(_mm256_add_epi64(_mm256_mul_epu32(yy, _mm256_set1_epi64x(416)), _mm256_slli_epi64::<5>(month)), d);
    let idx = _mm256_and_si256(_mm256_and_si256(idx, in_range), _mm256_set1_epi64x((DATE_SLOTS - 1) as i64));
    let mut ix = [0u64; 4];
    unsafe { _mm256_storeu_si256(ix.as_mut_ptr() as *mut __m256i, idx) };
    let p = dates.day_ptr();
    // SAFETY: every index is masked to DATE_SLOTS - 1 and the table has DATE_SLOTS slots.
    let day = unsafe {
        _mm256_set_epi64x(
            *p.add(ix[3] as usize) as i64,
            *p.add(ix[2] as usize) as i64,
            *p.add(ix[1] as usize) as i64,
            *p.add(ix[0] as usize) as i64,
        )
    };
    let bad = _mm256_cmpeq_epi64(day, _mm256_set1_epi64x(BAD_DAY as i64));
    (day, _mm256_andnot_si256(bad, in_range))
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hms(t: __m256i) -> (__m256i, __m256i) {
    let colons = unsafe { eq(_mm256_and_si256(t, c(0x0000_FF00_00FF_0000)), 0x0000_3A00_003A_0000) };
    let x = _mm256_and_si256(_mm256_xor_si256(t, unsafe { c(0x3030_3030_3030_3030) }), unsafe { c(0xFFFF_00FF_FF00_FFFF) });
    let dig = unsafe { digits_ok(x, u64::MAX) };
    let v = _mm256_add_epi64(_mm256_add_epi64(_mm256_slli_epi64::<3>(x), _mm256_slli_epi64::<1>(x)), _mm256_srli_epi64::<8>(x));
    let ff = unsafe { c(0xFF) };
    let h = _mm256_and_si256(v, ff);
    let m = _mm256_and_si256(_mm256_srli_epi64::<24>(v), ff);
    let s = _mm256_and_si256(_mm256_srli_epi64::<48>(v), ff);
    let ok = _mm256_and_si256(_mm256_and_si256(colons, dig), _mm256_and_si256(unsafe { le(h, 23) }, _mm256_and_si256(unsafe { le(m, 59) }, unsafe { le(s, 59) })));
    let secs = _mm256_add_epi64(_mm256_add_epi64(_mm256_mul_epu32(h, _mm256_set1_epi64x(3600)), _mm256_mul_epu32(m, _mm256_set1_epi64x(60))), s);
    (secs, ok)
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dec8(w: __m256i) -> (__m256i, __m256i) {
    let x = _mm256_xor_si256(w, unsafe { c(0x3030_3030_3030_3030) });
    let ok = unsafe { digits_ok(x, u64::MAX) };
    let t = _mm256_maddubs_epi16(x, _mm256_set1_epi16(0x010A));
    let t = _mm256_madd_epi16(t, _mm256_set1_epi32(0x0001_0064));
    let v = _mm256_add_epi64(_mm256_mul_epu32(t, _mm256_set1_epi64x(10_000)), _mm256_srli_epi64::<32>(t));
    (v, ok)
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn offset(o: __m256i) -> (__m256i, __m256i) {
    let ff = unsafe { c(0xFF) };
    let sign = _mm256_and_si256(_mm256_srli_epi64::<16>(o), ff);
    let colon = unsafe { eq(_mm256_and_si256(_mm256_srli_epi64::<40>(o), ff), b':' as u64) };
    let x = _mm256_and_si256(_mm256_xor_si256(o, unsafe { c(0x3030_3030_3030_3030) }), unsafe { c(0xFFFF_00FF_FF00_0000) });
    let dig = unsafe { digits_ok(x, u64::MAX) };
    let v = _mm256_add_epi64(_mm256_add_epi64(_mm256_slli_epi64::<3>(x), _mm256_slli_epi64::<1>(x)), _mm256_srli_epi64::<8>(x));
    let h = _mm256_and_si256(_mm256_srli_epi64::<24>(v), ff);
    let m = _mm256_and_si256(_mm256_srli_epi64::<48>(v), ff);
    let mag = _mm256_mul_epu32(_mm256_add_epi64(_mm256_mul_epu32(h, _mm256_set1_epi64x(60)), m), _mm256_set1_epi64x(60_000));
    let plus = unsafe { eq(sign, b'+' as u64) };
    let minus = unsafe { eq(sign, b'-' as u64) };
    let neg = _mm256_sub_epi64(_mm256_setzero_si256(), mag);
    let off = _mm256_blendv_epi8(neg, mag, plus);
    let ok = _mm256_and_si256(_mm256_and_si256(_mm256_or_si256(plus, minus), colon), _mm256_and_si256(dig, _mm256_and_si256(unsafe { le(h, 23) }, unsafe { le(m, 59) })));
    (off, ok)
}

#[inline(always)]
unsafe fn ld(a: &[u64; 8], h: usize) -> __m256i {
    unsafe { _mm256_loadu_si256(a.as_ptr().add(4 * h) as *const __m256i) }
}

#[inline(always)]
unsafe fn ld_len(a: &[u32; 8], h: usize) -> __m256i {
    unsafe { _mm256_cvtepu32_epi64(_mm_loadu_si128(a.as_ptr().add(4 * h) as *const __m128i)) }
}

/// The batch's ts and ingest decoders for 8 lanes (as the scalar loops in `fast2::batch`).
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn ts_ingest(
    f: &Fast,
    dates: &Dates,
    tlen: &[u32; 8],
    t_hi: &[u64; 8],
    t_mid: &[u64; 8],
    t_16: &[u64; 8],
    t_17: &[u64; 8],
    ilen: &[u32; 8],
    i_hi: &[u64; 8],
    i_mid: &[u64; 8],
    i_16: &[u64; 8],
    ts_rel: &mut [i64; 8],
    ts_fast: &mut [bool; 8],
    ingest: &mut [i64; 8],
    ing_fast: &mut [bool; 8],
) {
    let ff = unsafe { c(0xFF) };
    for h in 0..2 {
        unsafe {
            // ts
            let (hi, mid, t16, t17) = (ld(t_hi, h), ld(t_mid, h), ld(t_16, h), ld(t_17, h));
            let len = ld_len(tlen, h);
            let (day, date_ok) = date(dates, hi, mid);
            let t_ok = eq(_mm256_and_si256(_mm256_srli_epi64::<16>(mid), ff), b'T' as u64);
            let (secs, hms_ok) = hms(_mm256_or_si256(_mm256_srli_epi64::<24>(mid), _mm256_slli_epi64::<40>(t16)));
            let z = eq(_mm256_and_si256(_mm256_srli_epi64::<24>(t16), ff), b'Z' as u64);
            let (off, off_ok) = offset(t17);
            let dt_ok = _mm256_and_si256(date_ok, _mm256_and_si256(t_ok, hms_ok));
            let f1 = _mm256_and_si256(eq(len, 20), _mm256_and_si256(z, dt_ok));
            let f2 = _mm256_and_si256(eq(len, 25), _mm256_and_si256(off_ok, dt_ok));
            let (dhi, hi_ok) = dec8(hi);
            let (dlo, lo_ok) = dec8(_mm256_or_si256(_mm256_slli_epi64::<24>(mid), c(0x0030_3030)));
            let f3 = _mm256_and_si256(eq(len, 13), _mm256_and_si256(hi_ok, lo_ok));
            let local = _mm256_sub_epi64(
                _mm256_add_epi64(_mm256_mul_epi32(day, _mm256_set1_epi64x(DAY_MS)), _mm256_mul_epu32(secs, _mm256_set1_epi64x(1000))),
                _mm256_set1_epi64x(f.ws_frac),
            );
            let f12 = _mm256_sub_epi64(local, _mm256_and_si256(off, f2));
            let v3 = _mm256_sub_epi64(_mm256_add_epi64(_mm256_mul_epu32(dhi, _mm256_set1_epi64x(100_000)), dlo), _mm256_set1_epi64x(f.ws_ms));
            let ts = _mm256_blendv_epi8(f12, v3, f3);
            _mm256_storeu_si256(ts_rel.as_mut_ptr().add(4 * h) as *mut __m256i, ts);
            let fast = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_or_si256(f1, _mm256_or_si256(f2, f3))));
            for l in 0..4 {
                ts_fast[4 * h + l] = fast >> l & 1 != 0;
            }
            // ingest
            let (hi, mid, t16) = (ld(i_hi, h), ld(i_mid, h), ld(i_16, h));
            let len = ld_len(ilen, h);
            let (day, date_ok) = date(dates, hi, mid);
            let t_ok = eq(_mm256_and_si256(_mm256_srli_epi64::<16>(mid), ff), b'T' as u64);
            let (secs, hms_ok) = hms(_mm256_or_si256(_mm256_srli_epi64::<24>(mid), _mm256_slli_epi64::<40>(t16)));
            let z = eq(_mm256_and_si256(_mm256_srli_epi64::<24>(t16), ff), b'Z' as u64);
            let v = _mm256_add_epi64(_mm256_add_epi64(_mm256_mul_epi32(day, _mm256_set1_epi64x(86_400)), secs), _mm256_set1_epi64x(f.ingest_add));
            _mm256_storeu_si256(ingest.as_mut_ptr().add(4 * h) as *mut __m256i, v);
            let okv = _mm256_and_si256(_mm256_and_si256(eq(len, 20), date_ok), _mm256_and_si256(_mm256_and_si256(t_ok, hms_ok), z));
            let ok = _mm256_movemask_pd(_mm256_castsi256_pd(okv));
            for l in 0..4 {
                ing_fast[4 * h + l] = ok >> l & 1 != 0;
            }
        }
    }
}

/// Bytes of x in [lo, hi] (signed byte compares: bytes >= 0x80 never are).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn in_range8(x: __m256i, lo: u8, hi: u8) -> __m256i {
    _mm256_and_si256(_mm256_cmpgt_epi8(x, _mm256_set1_epi8(lo as i8 - 1)), _mm256_cmpgt_epi8(_mm256_set1_epi8(hi as i8 + 1), x))
}

/// Base-36 IDs of 4 lanes (value or u64::MAX), as `fast2::id_from`: w = the 8 bytes ending at the
/// ID's end, k = digit count (may be <= 0), pre = whether the prefix byte matched (all-ones lanes).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn ids4(w: __m256i, k: __m256i, pre: __m256i) -> __m256i {
    let one = _mm256_set1_epi64x(1);
    let eight = _mm256_set1_epi64x(8);
    // clamp(k, 1, 8)
    let kc = _mm256_blendv_epi8(k, one, _mm256_cmpgt_epi64(one, k));
    let kc = _mm256_blendv_epi8(kc, eight, _mm256_cmpgt_epi64(kc, eight));
    let sh = _mm256_slli_epi64::<3>(_mm256_sub_epi64(eight, kc));
    let keep = _mm256_sllv_epi64(_mm256_set1_epi64x(-1), sh);
    let c30 = unsafe { c(0x3030_3030_3030_3030) };
    let w = _mm256_or_si256(_mm256_and_si256(keep, w), _mm256_andnot_si256(keep, c30));
    let digit = unsafe { in_range8(w, b'0', b'9') };
    let lower = unsafe { in_range8(w, b'a', b'z') };
    let all = _mm256_cmpeq_epi64(_mm256_or_si256(digit, lower), _mm256_set1_epi64x(-1));
    let v = _mm256_sub_epi8(w, c30);
    let v = _mm256_sub_epi8(v, _mm256_and_si256(lower, _mm256_set1_epi8(39)));
    let t = _mm256_maddubs_epi16(v, _mm256_set1_epi16(0x0124));
    let t = _mm256_madd_epi16(t, _mm256_set1_epi32(0x0001_0510));
    let value = _mm256_add_epi64(_mm256_mul_epu32(t, _mm256_set1_epi64x(1_679_616)), _mm256_srli_epi64::<32>(t));
    let first = _mm256_and_si256(_mm256_srlv_epi64(v, sh), _mm256_set1_epi64x(0xFF));
    let z = _mm256_setzero_si256();
    let first_ok = _mm256_or_si256(_mm256_xor_si256(_mm256_cmpeq_epi64(first, z), _mm256_set1_epi64x(-1)), _mm256_cmpeq_epi64(k, one));
    let k_ok = _mm256_and_si256(unsafe { ge(k, 1) }, unsafe { le(k, 8) });
    let small = unsafe { le(value, crate::ids::MASK40 as i64) };
    let good = _mm256_and_si256(_mm256_and_si256(_mm256_and_si256(all, pre), first_ok), _mm256_and_si256(k_ok, small));
    _mm256_blendv_epi8(_mm256_set1_epi64x(-1), value, good)
}

/// 8 hex characters per lane (byte 0 most significant) to their 32-bit value, and whether all are
/// `0-9a-f` (as `fast::hex8`).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hex8x4(w: __m256i) -> (__m256i, __m256i) {
    let digit = unsafe { in_range8(w, b'0', b'9') };
    let letter = unsafe { in_range8(w, b'a', b'f') };
    let ok = _mm256_cmpeq_epi64(_mm256_or_si256(digit, letter), _mm256_set1_epi64x(-1));
    let v = _mm256_sub_epi8(w, _mm256_set1_epi8(b'0' as i8));
    let v = _mm256_sub_epi8(v, _mm256_and_si256(letter, _mm256_set1_epi8(39)));
    let t = _mm256_maddubs_epi16(v, _mm256_set1_epi16(0x0110));
    let t = _mm256_madd_epi16(t, _mm256_set1_epi32(0x0001_0100));
    let val = _mm256_or_si256(_mm256_slli_epi64::<16>(_mm256_and_si256(t, unsafe { c(0xFFFF_FFFF) })), _mm256_srli_epi64::<32>(t));
    (val, ok)
}

/// LONG key of 4 lanes from its four 8-byte words (w0..w2 = the 24 bytes before the last 8, w3 =
/// the last 8): (a, b, all hex).
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn long4(w0: __m256i, w1: __m256i, w2: __m256i, w3: __m256i) -> (__m256i, __m256i, __m256i) {
    let (a0, k0) = unsafe { hex8x4(w0) };
    let (a1, k1) = unsafe { hex8x4(w1) };
    let (b0, k2) = unsafe { hex8x4(w2) };
    let (b1, k3) = unsafe { hex8x4(w3) };
    let a = _mm256_or_si256(_mm256_slli_epi64::<32>(a0), a1);
    let b = _mm256_or_si256(_mm256_slli_epi64::<32>(b0), b1);
    (a, b, _mm256_and_si256(_mm256_and_si256(k0, k1), _mm256_and_si256(k2, k3)))
}

/// The batch's ID decoders for 8 lanes: SHORT values (or u64::MAX) of the event and user IDs, and
/// their LONG keys with the LONG test (length 33, prefix, 32 hex digits).
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn keys(
    lens: &[[u32; 8]; 2],
    pres: &[[u8; 8]; 2],
    prefix: [u8; 2],
    last8: [&[u64; 8]; 2],
    words: [&[[u64; 8]; 3]; 2],
    short: &mut [[u64; 8]; 2],
    long_ok: &mut [[bool; 8]; 2],
    long_a: &mut [[u64; 8]; 2],
    long_b: &mut [[u64; 8]; 2],
) {
    for id in 0..2 {
        for h in 0..2 {
            unsafe {
                let len = ld_len(&lens[id], h);
                let pre8 = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(i32::from_le_bytes(pres[id][4 * h..4 * h + 4].try_into().unwrap())));
                let pre_ok = _mm256_cmpeq_epi64(pre8, _mm256_set1_epi64x(prefix[id] as i64));
                let w = ld(last8[id], h);
                let k = _mm256_sub_epi64(len, _mm256_set1_epi64x(1));
                let v = ids4(w, k, pre_ok);
                _mm256_storeu_si256(short[id].as_mut_ptr().add(4 * h) as *mut __m256i, v);
                let (a, b, hx) = long4(ld(&words[id][0], h), ld(&words[id][1], h), ld(&words[id][2], h), w);
                _mm256_storeu_si256(long_a[id].as_mut_ptr().add(4 * h) as *mut __m256i, a);
                _mm256_storeu_si256(long_b[id].as_mut_ptr().add(4 * h) as *mut __m256i, b);
                let okv = _mm256_and_si256(_mm256_and_si256(eq(len, 33), pre_ok), hx);
                let m = _mm256_movemask_pd(_mm256_castsi256_pd(okv));
                for l in 0..4 {
                    long_ok[id][4 * h + l] = m >> l & 1 != 0;
                }
            }
        }
    }
}

/// Field value bounds of the 8 event lines of a batch, as the scalar loop in `fast2::batch`:
/// `sep[0]` is the newline before the first line and `sep[1..57]` the 56 separators; bit
/// 7 * lane + field of `q` says that field is quoted; `w` is the window's file offset. One 8x8
/// transpose of the separator positions gives every field's column for the 8 lanes.
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn bounds7(sep: *const u32, w: u32, q: u64, ls: &mut [u32; 8], end: &mut [u32; 8], vs: &mut [[u32; 8]; 7], ve: &mut [[u32; 8]; 7]) {
    unsafe {
        let r = |l: usize| _mm256_loadu_si256(sep.add(7 * l) as *const __m256i);
        let (r0, r1, r2, r3, r4, r5, r6, r7) = (r(0), r(1), r(2), r(3), r(4), r(5), r(6), r(7));
        let t0 = _mm256_unpacklo_epi32(r0, r1);
        let t1 = _mm256_unpackhi_epi32(r0, r1);
        let t2 = _mm256_unpacklo_epi32(r2, r3);
        let t3 = _mm256_unpackhi_epi32(r2, r3);
        let t4 = _mm256_unpacklo_epi32(r4, r5);
        let t5 = _mm256_unpackhi_epi32(r4, r5);
        let t6 = _mm256_unpacklo_epi32(r6, r7);
        let t7 = _mm256_unpackhi_epi32(r6, r7);
        let u0 = _mm256_unpacklo_epi64(t0, t2);
        let u1 = _mm256_unpackhi_epi64(t0, t2);
        let u2 = _mm256_unpacklo_epi64(t1, t3);
        let u3 = _mm256_unpackhi_epi64(t1, t3);
        let u4 = _mm256_unpacklo_epi64(t4, t6);
        let u5 = _mm256_unpackhi_epi64(t4, t6);
        let u6 = _mm256_unpacklo_epi64(t5, t7);
        let u7 = _mm256_unpackhi_epi64(t5, t7);
        // col[j] lane l = sep[7 * l + j]: the separator before field j of line l.
        let col = [
            _mm256_permute2x128_si256::<0x20>(u0, u4),
            _mm256_permute2x128_si256::<0x20>(u1, u5),
            _mm256_permute2x128_si256::<0x20>(u2, u6),
            _mm256_permute2x128_si256::<0x20>(u3, u7),
            _mm256_permute2x128_si256::<0x31>(u0, u4),
            _mm256_permute2x128_si256::<0x31>(u1, u5),
            _mm256_permute2x128_si256::<0x31>(u2, u6),
            _mm256_permute2x128_si256::<0x31>(u3, u7),
        ];
        // Lanes 0..4 test bit 7 * l + j of q, lanes 4..8 bit 7 * (l - 4) + j of q >> 28.
        let (lo, hi) = (q as u32 as i32, (q >> 28) as u32 as i32);
        let x = _mm256_setr_epi32(lo, lo, lo, lo, hi, hi, hi, hi);
        let bits0 = _mm256_setr_epi32(1, 1 << 7, 1 << 14, 1 << 21, 1, 1 << 7, 1 << 14, 1 << 21);
        let w1 = _mm256_set1_epi32(w.wrapping_add(1) as i32);
        let wv = _mm256_set1_epi32(w as i32);
        _mm256_storeu_si256(ls.as_mut_ptr() as *mut __m256i, _mm256_add_epi32(col[0], w1));
        _mm256_storeu_si256(end.as_mut_ptr() as *mut __m256i, _mm256_add_epi32(col[7], wv));
        for j in 0..7 {
            let bits = _mm256_sll_epi32(bits0, _mm_cvtsi32_si128(j as i32));
            let m = _mm256_cmpeq_epi32(_mm256_and_si256(x, bits), bits); // -1 where quoted
            let s = _mm256_sub_epi32(_mm256_add_epi32(col[j], w1), m);
            let e = _mm256_add_epi32(_mm256_add_epi32(col[j + 1], wv), m);
            _mm256_storeu_si256(vs[j].as_mut_ptr() as *mut __m256i, s);
            _mm256_storeu_si256(ve[j].as_mut_ptr() as *mut __m256i, e);
        }
    }
}
