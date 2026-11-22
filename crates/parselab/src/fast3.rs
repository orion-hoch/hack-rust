//! The AVX-512 batches (RECORDS v3): the same decisions as `fast2::batch`, written with
//! intrinsics on 8 lanes (one line per 64-bit lane). `fast2::batch` stays the reference for it in
//! the equivalence runs (and is the AVX2 and scalar path).
//!
//! Bounds: the 56 separator positions of the batch are loaded as four 512-bit words and
//! transposed with permutes (no gathers); the 7 quote flags of each line are one 56-bit window of
//! the quote-bit stream, split into per-field lane masks with pext. Field bytes come from
//! gathers relative to the window start. Every lane that needs anything unusual is handed to the
//! scalar emission, which falls back to the reference for it.

#![cfg(target_arch = "x86_64")]
#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::x86_64::*;

use crate::fast::{self, Buf, Fast};
use crate::fast2::{BAD_DAY, DATE_SLOTS, Dates, POS_MASK, Q_FLAG, Scratch};
use crate::ids::MASK40;
use crate::records::*;
use crate::reference;
use crate::ts::DAY_MS;

const NONE40: u64 = u64::MAX;

/// Run-time knob: use these batches (true) or the portable lane batches (false).
pub static VEC512: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
/// This build has the AVX-512 batches.
pub const READY: bool = true;
/// Run-time knob: do the batches' multiplies as 256-bit halves (no heavy 512-bit instruction).
/// Set once by `choose_paths` (from the CPU model, or a cargo feature), read per batch.
pub static MUL256: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

static CHOSEN: std::sync::Once = std::sync::Once::new();

/// CPUID family and model (display values: extended fields folded in).
pub fn cpu_family_model() -> (u32, u32) {
    let r = unsafe { __cpuid(1) };
    let base_family = (r.eax >> 8) & 0xF;
    let family = if base_family == 0xF { base_family + ((r.eax >> 20) & 0xFF) } else { base_family };
    let base_model = (r.eax >> 4) & 0xF;
    let model = if base_family == 6 || base_family == 0xF { base_model | (((r.eax >> 16) & 0xF) << 4) } else { base_model };
    (family, model)
}

/// Chooses the multiply width once per process: the cargo feature `mul256` or `mul512` forces it;
/// otherwise family 6 model 0x55 (Skylake-SP, Cascade Lake, Cooper Lake: heavy 512-bit
/// multiplies lower the clock there) takes the 256-bit multiplies and everything else the full
/// 512-bit path. Later calls do nothing; `MUL256.store` after it still overrides (tests).
pub fn choose_paths() {
    CHOSEN.call_once(|| {
        let (family, model) = cpu_family_model();
        let m256 = if cfg!(feature = "mul256") {
            true
        } else if cfg!(feature = "mul512") {
            false
        } else {
            family == 6 && model == 0x55
        };
        MUL256.store(m256, std::sync::atomic::Ordering::Relaxed);
    });
}

/// A line for the timing report: ISA path, multiply width and why.
pub fn path_report() -> String {
    let (family, model) = cpu_family_model();
    let why = if cfg!(feature = "mul256") {
        "forced by feature mul256"
    } else if cfg!(feature = "mul512") {
        "forced by feature mul512"
    } else {
        "by CPU model"
    };
    format!(
        "parser isa={:?} vec512={} mul={} ({why}; cpu family {family} model {model:#x})",
        crate::fast::Isa::detect(),
        VEC512.load(std::sync::atomic::Ordering::Relaxed),
        if MUL256.load(std::sync::atomic::Ordering::Relaxed) { "256" } else { "512" }
    )
}

#[inline(always)]
unsafe fn c64(x: u64) -> __m512i {
    _mm512_set1_epi64(x as i64)
}

/// Lane l of the result: element 7l + j of the 64 u32 in z0..z3.
#[inline(always)]
unsafe fn column(z: &[__m512i; 4], j: u32) -> __m256i {
    // No std::array::from_fn in kernels: with two instantiations LLVM once left it as a call.
    let mut idx = [0i32; 16];
    for (l, x) in idx.iter_mut().enumerate().take(8) {
        *x = (7 * l as i32 + j as i32) & 31;
    }
    let iv = unsafe { _mm512_loadu_si512(idx.as_ptr() as *const __m512i) };
    let lo = _mm512_permutex2var_epi32(z[0], iv, z[1]);
    let hi = _mm512_permutex2var_epi32(z[2], iv, z[3]);
    // Lanes whose element is >= 32 come from z2/z3.
    let mut m: u16 = 0;
    for l in 0..8 {
        if 7 * l + j >= 32 {
            m |= 1 << l;
        }
    }
    _mm512_castsi512_si256(_mm512_mask_blend_epi32(m, lo, hi))
}


/// The 512-bit multiplies, optionally done as two 256-bit halves (`S`), so the zmm path uses no
/// heavy 512-bit instruction: on Skylake-SP/Cascade Lake those set the lowest clock license, while
/// light 512-bit work (loads, compares, shuffles, adds) runs at the 256-bit-heavy license.
#[inline(always)]
unsafe fn split2<const S: bool>(a: __m512i, b: __m512i, f512: impl Fn(__m512i, __m512i) -> __m512i, f256: impl Fn(__m256i, __m256i) -> __m256i) -> __m512i {
    if !S {
        return f512(a, b);
    }
    let lo = f256(_mm512_castsi512_si256(a), _mm512_castsi512_si256(b));
    let hi = f256(_mm512_extracti64x4_epi64::<1>(a), _mm512_extracti64x4_epi64::<1>(b));
    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
}

// The 256-bit halves are asm blocks: as plain intrinsics LLVM fuses the two halves straight back
// into one 512-bit instruction.
macro_rules! asm256 {
    ($name:ident, $ins:literal, $feat:literal) => {
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn $name(a: __m256i, b: __m256i) -> __m256i {
            let r: __m256i;
            unsafe {
                std::arch::asm!(concat!($ins, " {r}, {a}, {b}"), r = lateout(ymm_reg) r, a = in(ymm_reg) a, b = in(ymm_reg) b, options(pure, nomem, nostack));
            }
            r
        }
    };
}
/// For the no-heavy-512 build: hides a value from LLVM so a shift-add like (x << 3) + (x << 1)
/// is not rewritten into a 512-bit vpmullq.
#[inline(always)]
unsafe fn opaque<const S: bool>(v: __m512i) -> __m512i {
    // Both paths: on the 512-bit path too LLVM folded (x << 3) + (x << 1) into vpmullq (3 uops,
    // latency 15).
    unsafe { opaque512(v) }
}

/// A chain of multiplies on one value, done whole on 512 bits or (`S`) once per 256-bit half:
/// one split and one join for the chain instead of one per multiply.
#[inline(always)]
unsafe fn chain<const S: bool>(x: __m512i, f512: impl Fn(__m512i) -> __m512i, f256: impl Fn(__m256i) -> __m256i) -> __m512i {
    if !S {
        return f512(x);
    }
    let lo = f256(_mm512_castsi512_si256(x));
    let hi = f256(_mm512_extracti64x4_epi64::<1>(x));
    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
}

/// x * 60 as (x << 6) - (x << 2) (light 512-bit work).
#[inline(always)]
unsafe fn times60(x: __m512i) -> __m512i {
    _mm512_sub_epi64(unsafe { opaque512(_mm512_slli_epi64::<6>(x)) }, _mm512_slli_epi64::<2>(x))
}

/// x * 10 by shifts (light 512-bit work, both paths).
#[inline(always)]
unsafe fn times10(x: __m512i) -> __m512i {
    _mm512_add_epi64(unsafe { opaque512(_mm512_slli_epi64::<3>(x)) }, _mm512_slli_epi64::<1>(x))
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn opaque512(v: __m512i) -> __m512i {
    let mut v = v;
    unsafe { std::arch::asm!("/* {v} */", v = inout(zmm_reg) v, options(pure, nomem, nostack, preserves_flags)) };
    v
}

asm256!(mul_epu32_256, "vpmuludq", "avx2");

/// vpmuludq on 512 bits as written (LLVM otherwise rewrites some into 3-uop vpmullq).
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn mul_epu32_512(a: __m512i, b: __m512i) -> __m512i {
    let r: __m512i;
    unsafe { std::arch::asm!("vpmuludq {r}, {a}, {b}", r = lateout(zmm_reg) r, a = in(zmm_reg) a, b = in(zmm_reg) b, options(pure, nomem, nostack, preserves_flags)) };
    r
}
asm256!(mul_epi32_256, "vpmuldq", "avx2");
asm256!(maddubs_256, "vpmaddubsw", "avx2");
asm256!(madd_256, "vpmaddwd", "avx2");
asm256!(mullo64_256, "vpmullq", "avx512dq,avx512vl");

#[inline(always)]
unsafe fn mul_epu32<const S: bool>(a: __m512i, b: __m512i) -> __m512i {
    split2::<S>(a, b, |x, y| _mm512_mul_epu32(x, y), |x, y| mul_epu32_256(x, y))
}
#[inline(always)]
unsafe fn mul_epi32<const S: bool>(a: __m512i, b: __m512i) -> __m512i {
    split2::<S>(a, b, |x, y| _mm512_mul_epi32(x, y), |x, y| mul_epi32_256(x, y))
}
#[inline(always)]
unsafe fn maddubs<const S: bool>(a: __m512i, b: __m512i) -> __m512i {
    split2::<S>(a, b, |x, y| _mm512_maddubs_epi16(x, y), |x, y| maddubs_256(x, y))
}
#[inline(always)]
unsafe fn madd<const S: bool>(a: __m512i, b: __m512i) -> __m512i {
    split2::<S>(a, b, |x, y| _mm512_madd_epi16(x, y), |x, y| madd_256(x, y))
}
#[inline(always)]
unsafe fn mullo64<const S: bool>(a: __m512i, b: __m512i) -> __m512i {
    split2::<S>(a, b, |x, y| _mm512_mullo_epi64(x, y), |x, y| mullo64_256(x, y))
}

/// 8 base-36 IDs: (value or NONE40), from the 8 bytes ending at each ID's end, its digit count
/// k (as i64, may be <= 0) and whether its prefix byte matched.
#[inline(always)]
unsafe fn ids<const S: bool>(w: __m512i, k: __m512i, prefix_ok: __mmask8) -> __m512i {
    let one = _mm512_set1_epi64(1);
    let eight = _mm512_set1_epi64(8);
    let kc = _mm512_max_epi64(one, _mm512_min_epi64(k, eight));
    let sh = _mm512_slli_epi64::<3>(_mm512_sub_epi64(eight, kc));
    let keep = _mm512_sllv_epi64(_mm512_set1_epi64(-1), sh);
    let c30 = unsafe { c64(0x3030_3030_3030_3030) };
    let w = _mm512_ternarylogic_epi64::<0xCA>(keep, w, c30);
    let digit = _mm512_cmpge_epu8_mask(w, _mm512_set1_epi8(b'0' as i8)) & _mm512_cmple_epu8_mask(w, _mm512_set1_epi8(b'9' as i8));
    let lower = _mm512_cmpge_epu8_mask(w, _mm512_set1_epi8(b'a' as i8)) & _mm512_cmple_epu8_mask(w, _mm512_set1_epi8(b'z' as i8));
    let all = _mm512_cmpeq_epi64_mask(_mm512_movm_epi8(digit | lower), _mm512_set1_epi64(-1));
    let v = _mm512_sub_epi8(w, c30);
    let v = _mm512_mask_sub_epi8(v, lower, v, _mm512_set1_epi8(39));
    // bytes [36, 1] -> i16 [1296, 1] -> u32 pairs -> hi * 36^4 + lo
    let value = unsafe {
        chain::<S>(
            v,
            |v| {
                let t = _mm512_madd_epi16(_mm512_maddubs_epi16(v, _mm512_set1_epi16(0x0124)), _mm512_set1_epi32(0x0001_0510));
                _mm512_add_epi64(mul_epu32_512(t, _mm512_set1_epi64(1_679_616)), _mm512_srli_epi64::<32>(t))
            },
            |v| {
                let t = madd_256(maddubs_256(v, _mm256_set1_epi16(0x0124)), _mm256_set1_epi32(0x0001_0510));
                _mm256_add_epi64(mul_epu32_256(t, _mm256_set1_epi64x(1_679_616)), _mm256_srli_epi64::<32>(t))
            },
        )
    };
    let first = _mm512_and_si512(_mm512_srlv_epi64(v, sh), _mm512_set1_epi64(0xFF));
    let first_ok = _mm512_cmpneq_epi64_mask(first, _mm512_setzero_si512()) | _mm512_cmpeq_epi64_mask(k, one);
    let k_ok = _mm512_cmpge_epi64_mask(k, one) & _mm512_cmple_epi64_mask(k, eight);
    let small = _mm512_cmple_epu64_mask(value, _mm512_set1_epi64(MASK40 as i64));
    let good = all & prefix_ok & first_ok & k_ok & small;
    _mm512_mask_blend_epi64(good, _mm512_set1_epi64(-1), value)
}

/// Per-byte digit test of the bytes selected by `mask` of x = bytes ^ 0x30.
#[inline(always)]
unsafe fn digits_ok(x: __m512i, mask: u64) -> __mmask8 {
    let xd = _mm512_and_si512(x, unsafe { c64(mask) });
    _mm512_testn_epi64_mask(xd, unsafe { c64(0xF0F0_F0F0_F0F0_F0F0) })
        & _mm512_testn_epi64_mask(_mm512_add_epi8(xd, _mm512_set1_epi8(6)), unsafe { c64(0x1010_1010_1010_1010) })
}

#[inline(always)]
unsafe fn byte_at(v: __m512i, i: u32) -> __m512i {
    _mm512_and_si512(_mm512_srl_epi64(v, _mm_cvtsi32_si128((8 * i) as i32)), _mm512_set1_epi64(0xFF))
}

/// `YYYY-MM-DD` over hi (bytes 0..8) and mid (bytes 0..2): (day relative to window_start's day,
/// valid and in the table).
#[inline(always)]
unsafe fn date<const S: bool>(dates: &Dates, hi: __m512i, mid: __m512i) -> (__m512i, __mmask8) {
    let c30 = unsafe { c64(0x3030_3030_3030_3030) };
    let x = _mm512_xor_si512(hi, c30);
    let dig = unsafe { digits_ok(x, 0x00FF_FF00_FFFF_FFFF) };
    let seps = _mm512_cmpeq_epi64_mask(
        _mm512_and_si512(hi, unsafe { c64(0xFF00_00FF_0000_0000) }),
        unsafe { c64(0x2D00_002D_0000_0000) },
    );
    let xd = _mm512_and_si512(x, unsafe { c64(0x00FF_FF00_FFFF_FFFF) });
    // bytes [10,1,10,1,0,10,1,0] -> i16 [Y12, Y34, 10*M1, M2] -> i32 [year, month]
    let w_ym = unsafe { c64(0x0001_0A00_010A_010A) };
    let t = unsafe {
        chain::<S>(
            xd,
            |x| _mm512_madd_epi16(_mm512_maddubs_epi16(x, w_ym), c64(0x0001_0001_0001_0064)),
            |x| madd_256(maddubs_256(x, _mm512_castsi512_si256(w_ym)), _mm256_set1_epi64x(0x0001_0001_0001_0064)),
        )
    };
    let year = _mm512_and_si512(t, unsafe { c64(0xFFFF_FFFF) });
    let month = _mm512_srli_epi64::<32>(t);
    let xm = _mm512_xor_si512(mid, c30);
    let mdig = unsafe { digits_ok(xm, 0xFFFF) };
    let d = _mm512_add_epi64(unsafe { times10(byte_at(xm, 0)) }, unsafe { byte_at(xm, 1) });
    let yy = _mm512_sub_epi64(year, _mm512_set1_epi64(dates.y0_()));
    let one = _mm512_set1_epi64(1);
    let in_range = dig
        & seps
        & mdig
        & _mm512_cmplt_epu64_mask(yy, _mm512_set1_epi64(2))
        & _mm512_cmpge_epi64_mask(month, one)
        & _mm512_cmple_epi64_mask(month, _mm512_set1_epi64(12))
        & _mm512_cmpge_epi64_mask(d, one)
        & _mm512_cmple_epi64_mask(d, _mm512_set1_epi64(31));
    // yy * 416 as (0 - yy) & 416: exact for yy in {0, 1}, and other lanes are masked out below.
    let idx = _mm512_add_epi64(
        _mm512_add_epi64(_mm512_and_si512(_mm512_sub_epi64(_mm512_setzero_si512(), yy), _mm512_set1_epi64(416)), opaque::<S>(_mm512_slli_epi64::<5>(month))),
        d,
    );
    // In bounds: masked to DATE_SLOTS - 1, and lanes out of range read slot 0 (never a date).
    let idx = _mm512_maskz_and_epi64(in_range, idx, _mm512_set1_epi64((DATE_SLOTS - 1) as i64));
    let day32 = unsafe { _mm512_i64gather_epi32::<4>(idx, dates.day_ptr() as *const i32) };
    let day = _mm512_cvtepi32_epi64(day32);
    let ok = in_range & _mm512_cmpneq_epi64_mask(day, _mm512_set1_epi64(BAD_DAY as i64));
    (day, ok)
}

/// `HH:MM:SS` in t (8 bytes): (seconds of day, ok). Lane-wise copy of `fast::hms`.
#[inline(always)]
unsafe fn hms<const S: bool>(t: __m512i) -> (__m512i, __mmask8) {
    let colons = _mm512_cmpeq_epi64_mask(
        _mm512_and_si512(t, unsafe { c64(0x0000_FF00_00FF_0000) }),
        unsafe { c64(0x0000_3A00_003A_0000) },
    );
    let x = _mm512_and_si512(_mm512_xor_si512(t, unsafe { c64(0x3030_3030_3030_3030) }), unsafe { c64(0xFFFF_00FF_FF00_FFFF) });
    let dig = unsafe { digits_ok(x, u64::MAX) };
    let v = _mm512_add_epi64(_mm512_add_epi64(opaque::<S>(_mm512_slli_epi64::<3>(x)), _mm512_slli_epi64::<1>(x)), _mm512_srli_epi64::<8>(x));
    let ff = _mm512_set1_epi64(0xFF);
    let h = _mm512_and_si512(v, ff);
    let m = _mm512_and_si512(_mm512_srli_epi64::<24>(v), ff);
    let s = _mm512_and_si512(_mm512_srli_epi64::<48>(v), ff);
    let ok = colons
        & dig
        & _mm512_cmple_epu64_mask(h, _mm512_set1_epi64(23))
        & _mm512_cmple_epu64_mask(m, _mm512_set1_epi64(59))
        & _mm512_cmple_epu64_mask(s, _mm512_set1_epi64(59));
    let secs = if S {
        // (h * 60 + m) * 60 + s, each * 60 as (x << 6) - (x << 2): light work, no split.
        let hm = _mm512_add_epi64(times60(h), m);
        _mm512_add_epi64(times60(hm), s)
    } else {
        _mm512_add_epi64(_mm512_add_epi64(_mm512_mul_epu32(h, _mm512_set1_epi64(3600)), _mm512_mul_epu32(m, _mm512_set1_epi64(60))), s)
    };
    (secs, ok)
}

/// 8 ASCII digits per lane to their value, and whether all are digits.
#[inline(always)]
unsafe fn dec8<const S: bool>(w: __m512i) -> (__m512i, __mmask8) {
    let x = _mm512_xor_si512(w, unsafe { c64(0x3030_3030_3030_3030) });
    let ok = unsafe { digits_ok(x, u64::MAX) };
    // bytes [10, 1] -> i16 [100, 1] -> u32 pairs -> hi * 10^4 + lo
    let v = unsafe {
        chain::<S>(
            x,
            |x| {
                let t = _mm512_madd_epi16(_mm512_maddubs_epi16(x, _mm512_set1_epi16(0x010A)), _mm512_set1_epi32(0x0001_0064));
                _mm512_add_epi64(mul_epu32_512(t, _mm512_set1_epi64(10_000)), _mm512_srli_epi64::<32>(t))
            },
            |x| {
                let t = madd_256(maddubs_256(x, _mm256_set1_epi16(0x010A)), _mm256_set1_epi32(0x0001_0064));
                _mm256_add_epi64(mul_epu32_256(t, _mm256_set1_epi64x(10_000)), _mm256_srli_epi64::<32>(t))
            },
        )
    };
    (v, ok)
}

/// Offset `SS±HH:MM` (bytes 0..8 of t17): (signed offset ms, ok).
#[inline(always)]
unsafe fn offset<const S: bool>(o: __m512i) -> (__m512i, __mmask8) {
    let sign = unsafe { byte_at(o, 2) };
    let colon = _mm512_cmpeq_epi64_mask(unsafe { byte_at(o, 5) }, _mm512_set1_epi64(b':' as i64));
    let x = _mm512_and_si512(_mm512_xor_si512(o, unsafe { c64(0x3030_3030_3030_3030) }), unsafe { c64(0xFFFF_00FF_FF00_0000) });
    let dig = unsafe { digits_ok(x, u64::MAX) };
    let v = _mm512_add_epi64(_mm512_add_epi64(opaque::<S>(_mm512_slli_epi64::<3>(x)), _mm512_slli_epi64::<1>(x)), _mm512_srli_epi64::<8>(x));
    let ff = _mm512_set1_epi64(0xFF);
    let h = _mm512_and_si512(_mm512_srli_epi64::<24>(v), ff);
    let m = _mm512_and_si512(_mm512_srli_epi64::<48>(v), ff);
    let hm = if S { _mm512_add_epi64(unsafe { times60(h) }, m) } else { _mm512_add_epi64(_mm512_mul_epu32(h, _mm512_set1_epi64(60)), m) };
    let mag = mul_epu32::<S>(hm, _mm512_set1_epi64(60_000));
    let plus = _mm512_cmpeq_epi64_mask(sign, _mm512_set1_epi64(b'+' as i64));
    let minus = _mm512_cmpeq_epi64_mask(sign, _mm512_set1_epi64(b'-' as i64));
    let off = _mm512_mask_sub_epi64(mag, !plus, _mm512_setzero_si512(), mag);
    let ok = (plus | minus)
        & colon
        & dig
        & _mm512_cmple_epu64_mask(h, _mm512_set1_epi64(23))
        & _mm512_cmple_epu64_mask(m, _mm512_set1_epi64(59));
    (off, ok)
}

/// 8 hex characters per lane (byte 0 most significant) to their 32-bit value, and whether all
/// are `0-9a-f`.
#[inline(always)]
unsafe fn hex8<const S: bool>(w: __m512i) -> (__m512i, __mmask8) {
    let digit = _mm512_cmpge_epu8_mask(w, _mm512_set1_epi8(b'0' as i8)) & _mm512_cmple_epu8_mask(w, _mm512_set1_epi8(b'9' as i8));
    let letter = _mm512_cmpge_epu8_mask(w, _mm512_set1_epi8(b'a' as i8)) & _mm512_cmple_epu8_mask(w, _mm512_set1_epi8(b'f' as i8));
    let ok = _mm512_cmpeq_epi64_mask(_mm512_movm_epi8(digit | letter), _mm512_set1_epi64(-1));
    let v = _mm512_sub_epi8(w, _mm512_set1_epi8(b'0' as i8));
    let v = _mm512_mask_sub_epi8(v, letter, v, _mm512_set1_epi8(39));
    // bytes [16, 1] -> i16 [256, 1], split once on the 256-bit path
    let t = unsafe {
        chain::<S>(
            v,
            |v| _mm512_madd_epi16(_mm512_maddubs_epi16(v, _mm512_set1_epi16(0x0110)), _mm512_set1_epi32(0x0001_0100)),
            |v| madd_256(maddubs_256(v, _mm256_set1_epi16(0x0110)), _mm256_set1_epi32(0x0001_0100)),
        )
    };
    let val = _mm512_or_si512(_mm512_slli_epi64::<16>(_mm512_and_si512(t, _mm512_set1_epi64(0xFFFF_FFFF))), _mm512_srli_epi64::<32>(t));
    (val, ok)
}

/// LONG keys of 8 IDs ending at `ve` (window-relative): (a, b, ok) where ok also needs the length
/// to be 33 and the prefix to match (passed in `pre_ok`).
#[inline(always)]
unsafe fn long_keys<const S: bool>(wbase: *const u8, ve: __m256i, last8: __m512i, len: __m512i, pre_ok: __mmask8) -> (__m512i, __m512i, __mmask8) {
    unsafe {
        let q0 = gather64(wbase, _mm256_sub_epi32(ve, _mm256_set1_epi32(32)));
        let q1 = gather64(wbase, _mm256_sub_epi32(ve, _mm256_set1_epi32(24)));
        let q2 = gather64(wbase, _mm256_sub_epi32(ve, _mm256_set1_epi32(16)));
        let (h0, k0) = hex8::<S>(q0);
        let (h1, k1) = hex8::<S>(q1);
        let (h2, k2) = hex8::<S>(q2);
        let (h3, k3) = hex8::<S>(last8);
        let a = _mm512_or_si512(_mm512_slli_epi64::<32>(h0), h1);
        let b = _mm512_or_si512(_mm512_slli_epi64::<32>(h2), h3);
        let ok = k0 & k1 & k2 & k3 & pre_ok & _mm512_cmpeq_epi64_mask(len, _mm512_set1_epi64(33));
        (a, b, ok)
    }
}

/// `prefix` + 1..8 canonical decimal digits ending at ve, looked up in a padded u32 table
/// (0xFFFF = absent): (value from the table, found).
#[inline(always)]
unsafe fn numbered<const S: bool>(wbase: *const u8, ve: __m256i, len: __m512i, pre_ok: __mmask8, table: &[u32]) -> (__m512i, __mmask8) {
    unsafe {
        let one = _mm512_set1_epi64(1);
        let w = gather64(wbase, _mm256_sub_epi32(ve, _mm256_set1_epi32(8)));
        let k = _mm512_sub_epi64(len, one);
        let kcl = _mm512_max_epi64(one, _mm512_min_epi64(k, _mm512_set1_epi64(8)));
        let sh = _mm512_slli_epi64::<3>(_mm512_sub_epi64(_mm512_set1_epi64(8), kcl));
        let keep = _mm512_sllv_epi64(_mm512_set1_epi64(-1), sh);
        let w = _mm512_ternarylogic_epi64::<0xCA>(keep, w, c64(0x3030_3030_3030_3030));
        let (n, dig) = dec8::<S>(w);
        let first = _mm512_and_si512(_mm512_srlv_epi64(w, sh), _mm512_set1_epi64(0xFF));
        let first_ok = _mm512_cmpneq_epi64_mask(first, _mm512_set1_epi64(b'0' as i64)) | _mm512_cmpeq_epi64_mask(k, one);
        let ok = dig
            & pre_ok
            & first_ok
            & _mm512_cmpge_epi64_mask(k, one)
            & _mm512_cmple_epi64_mask(k, _mm512_set1_epi64(8))
            & _mm512_cmplt_epu64_mask(n, _mm512_set1_epi64((table.len() - 16) as i64));
        let v = _mm512_cvtepu32_epi64(_mm512_mask_i64gather_epi32::<4>(
            _mm256_set1_epi32(0xFFFF),
            ok,
            _mm512_maskz_mov_epi64(ok, n),
            table.as_ptr() as *const i32,
        ));
        (v, ok & _mm512_cmpneq_epi64_mask(v, _mm512_set1_epi64(0xFFFF)))
    }
}

#[inline(always)]
unsafe fn store64(v: __m512i, out: &mut [u64; 8]) {
    unsafe { _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, v) };
}

#[inline(always)]
unsafe fn store32(v: __m512i, out: &mut [u32; 8]) {
    unsafe { _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, _mm512_cvtepi64_epi32(v)) };
}

#[inline(always)]
unsafe fn gather64(base: *const u8, idx: __m256i) -> __m512i {
    unsafe { _mm512_i32gather_epi64::<1>(idx, base as *const i64) }
}

#[inline(always)]
unsafe fn dword_bytes(base: *const u8, idx: __m256i) -> __m256i {
    unsafe { _mm256_i32gather_epi32::<1>(base as *const i32, idx) }
}

/// Separator positions of 8 lines at stride `nf` (from k0), window-relative, and the line
/// starts and quote-adjusted value bounds of every field. Entries carry the flags (newline,
/// next byte is a quote) in bits 31 and 30.
#[inline(always)]
unsafe fn bounds<const NF: usize>(sc: &Scratch, k0: usize) -> ([__m256i; NF], [__m256i; NF], __m256i, [__m256i; NF]) {
    unsafe {
        let sp = sc.sep_ptr().add(k0);
        let z = [
            _mm512_loadu_si512(sp as *const __m512i),
            _mm512_loadu_si512(sp.add(16) as *const __m512i),
            _mm512_loadu_si512(sp.add(32) as *const __m512i),
            _mm512_loadu_si512(sp.add(48) as *const __m512i),
        ];
        let mut raw = [_mm256_setzero_si256(); NF];
        for (j, r) in raw.iter_mut().enumerate() {
            *r = if NF == 7 {
                column(&z, j as u32)
            } else {
                let mut idx = [0i32; 16];
                for (l, x) in idx.iter_mut().enumerate().take(8) {
                    *x = (NF * l + j) as i32;
                }
                let iv = _mm512_loadu_si512(idx.as_ptr() as *const __m512i);
                _mm512_castsi512_si256(_mm512_permutex2var_epi32(z[0], iv, z[1]))
            };
        }
        let posm = _mm256_set1_epi32(POS_MASK as i32);
        let qf = _mm256_set1_epi32(Q_FLAG as i32);
        let mut sep = [_mm256_setzero_si256(); NF];
        for j in 0..NF {
            sep[j] = _mm256_and_si256(raw[j], posm);
        }
        // The separator before each line's field 0: the previous line's newline (lane l - 1), or
        // for lane 0 the entry before k0 (the window start when k0 == 0).
        let (prev_pos, prev_q) = if k0 == 0 { (-1i32, sc.wq0) } else {
            let e = *sc.sep_ptr().add(k0 - 1);
            ((e & POS_MASK) as i32, e & Q_FLAG != 0)
        };
        let rot = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
        let prev_raw = _mm256_permutexvar_epi32(rot, raw[NF - 1]);
        let q0 = (_mm256_test_epi32_mask(prev_raw, qf) & !1) | prev_q as u8;
        let shifted = _mm256_permutexvar_epi32(rot, sep[NF - 1]);
        let ls = _mm256_add_epi32(_mm256_mask_set1_epi32(shifted, 1, prev_pos), _mm256_set1_epi32(1));
        let one = _mm256_set1_epi32(1);
        let mut vs = [_mm256_setzero_si256(); NF];
        let mut ve = [_mm256_setzero_si256(); NF];
        for j in 0..NF {
            let q = if j == 0 { q0 } else { _mm256_test_epi32_mask(raw[j - 1], qf) };
            let s = if j == 0 { ls } else { _mm256_add_epi32(sep[j - 1], one) };
            vs[j] = _mm256_mask_add_epi32(s, q, s, one);
            ve[j] = _mm256_mask_sub_epi32(sep[j], q, sep[j], one);
        }
        (vs, ve, ls, sep)
    }
}

/// What a decoded batch of 8 lines needs from the generic caller: the batched lanes (compacted),
/// and the lanes to send to the reference (`slow`) or out by locator (`foreign`), with their
/// line starts and ends.
pub(crate) struct Out<B> {
    pub(crate) b: B,
    pub(crate) slow: u8,
    pub(crate) foreign: u8,
    pub(crate) ls: [u32; 8],
    pub(crate) le: [u32; 8],
}

/// Decodes 8 event lines (separators from k0). Not generic over the sink, so its code is the same
/// whatever the caller's sink is (two sink types once made LLVM compile both copies 40% worse).
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512cd,avx2,fma,bmi1,bmi2,popcnt,lzcnt")]
#[inline(never)]
pub(crate) unsafe fn ev_batch<const S: bool>(f: &Fast, dates: &Dates, base: *const u8, sc: &Scratch, wpos: usize, k0: usize, out: &mut Out<EvBatch>) {
    unsafe {
        let wbase = base.add(wpos);
        let (vs, ve, ls, sep) = bounds::<7>(sc, k0);
        let mut len = [_mm512_setzero_si512(); 7];
        for j in 0..7 {
            len[j] = _mm512_cvtepi32_epi64(_mm256_sub_epi32(ve[j], vs[j]));
        }
        let eight = _mm256_set1_epi32(8);
        let zero = _mm512_setzero_si512();
        let one64 = _mm512_set1_epi64(1);
        let ff32 = _mm256_set1_epi32(0xFF);
        // IDs: SHORT from the 8 bytes ending at the end, LONG from the 32 ending there.
        let w_e = gather64(wbase, _mm256_sub_epi32(ve[0], eight));
        let w_u = gather64(wbase, _mm256_sub_epi32(ve[1], eight));
        let pe = _mm256_cmpeq_epi32_mask(_mm256_and_si256(dword_bytes(wbase, vs[0]), ff32), _mm256_set1_epi32(b'e' as i32));
        let pu = _mm256_cmpeq_epi32_mask(_mm256_and_si256(dword_bytes(wbase, vs[1]), ff32), _mm256_set1_epi32(b'u' as i32));
        let x_e = ids::<S>(w_e, _mm512_sub_epi64(len[0], one64), pe);
        let x_u = ids::<S>(w_u, _mm512_sub_epi64(len[1], one64), pu);
        let e_short = _mm512_cmpneq_epi64_mask(x_e, _mm512_set1_epi64(-1));
        let u_short = _mm512_cmpneq_epi64_mask(x_u, _mm512_set1_epi64(-1));
        let n33 = _mm512_set1_epi64(33);
        let (mut e_a, mut e_b, mut e_long) = (x_e, zero, 0u8);
        if _mm512_cmpeq_epi64_mask(len[0], n33) != 0 {
            let (a, b, ok) = long_keys::<S>(wbase, ve[0], w_e, len[0], pe);
            e_long = ok & !e_short;
            e_a = _mm512_mask_blend_epi64(e_long, x_e, a);
            e_b = _mm512_maskz_mov_epi64(e_long, b);
        }
        let (mut u_a, mut u_b, mut u_long) = (x_u, zero, 0u8);
        if _mm512_cmpeq_epi64_mask(len[1], n33) != 0 {
            let (a, b, ok) = long_keys::<S>(wbase, ve[1], w_u, len[1], pu);
            u_long = ok & !u_short;
            u_a = _mm512_mask_blend_epi64(u_long, x_u, a);
            u_b = _mm512_maskz_mov_epi64(u_long, b);
        }
        let ev_ok = e_short | e_long;
        let user_ok = u_short | u_long;
        // ts.
        let t_hi = gather64(wbase, vs[2]);
        let t_mid = gather64(wbase, _mm256_add_epi32(vs[2], eight));
        let t_16 = gather64(wbase, _mm256_add_epi32(vs[2], _mm256_set1_epi32(16)));
        let t_17 = gather64(wbase, _mm256_add_epi32(vs[2], _mm256_set1_epi32(17)));
        let (day, date_ok) = date::<S>(dates, t_hi, t_mid);
        let t_ok = _mm512_cmpeq_epi64_mask(byte_at(t_mid, 2), _mm512_set1_epi64(b'T' as i64));
        let (secs, hms_ok) = hms::<S>(_mm512_or_si512(_mm512_srli_epi64::<24>(t_mid), _mm512_slli_epi64::<40>(t_16)));
        let z = _mm512_cmpeq_epi64_mask(byte_at(t_16, 3), _mm512_set1_epi64(b'Z' as i64));
        let (off, off_ok) = offset::<S>(t_17);
        let dt_ok = date_ok & t_ok & hms_ok;
        let f1 = _mm512_cmpeq_epi64_mask(len[2], _mm512_set1_epi64(20)) & z & dt_ok;
        let f2 = _mm512_cmpeq_epi64_mask(len[2], _mm512_set1_epi64(25)) & off_ok & dt_ok;
        let (hi, hi_ok) = dec8::<S>(t_hi);
        let (lo, lo_ok) = dec8::<S>(_mm512_or_si512(_mm512_slli_epi64::<24>(t_mid), _mm512_set1_epi64(0x0030_3030)));
        let f3 = _mm512_cmpeq_epi64_mask(len[2], _mm512_set1_epi64(13)) & hi_ok & lo_ok;
        let local = _mm512_sub_epi64(
            _mm512_add_epi64(mul_epi32::<S>(day, _mm512_set1_epi64(DAY_MS)), mul_epu32::<S>(secs, _mm512_set1_epi64(1000))),
            _mm512_set1_epi64(f.ws_frac),
        );
        let f12 = _mm512_mask_sub_epi64(local, f2, local, off);
        let v3 = _mm512_sub_epi64(_mm512_add_epi64(mul_epu32::<S>(hi, _mm512_set1_epi64(100_000)), lo), _mm512_set1_epi64(f.ws_ms));
        let ts_rel = _mm512_mask_blend_epi64(f3, f12, v3);
        let ts_fast = f1 | f2 | f3;
        // ingest (always F1).
        let i_hi = gather64(wbase, vs[3]);
        let i_mid = gather64(wbase, _mm256_add_epi32(vs[3], eight));
        let i_16 = gather64(wbase, _mm256_add_epi32(vs[3], _mm256_set1_epi32(16)));
        let (iday, idate_ok) = date::<S>(dates, i_hi, i_mid);
        let it_ok = _mm512_cmpeq_epi64_mask(byte_at(i_mid, 2), _mm512_set1_epi64(b'T' as i64));
        let (isecs, ihms_ok) = hms::<S>(_mm512_or_si512(_mm512_srli_epi64::<24>(i_mid), _mm512_slli_epi64::<40>(i_16)));
        let iz = _mm512_cmpeq_epi64_mask(byte_at(i_16, 3), _mm512_set1_epi64(b'Z' as i64));
        let ing_fast = _mm512_cmpeq_epi64_mask(len[3], _mm512_set1_epi64(20)) & idate_ok & it_ok & ihms_ok & iz;
        let ing_s = _mm512_add_epi64(mul_epi32::<S>(iday, _mm512_set1_epi64(86_400)), isecs);
        let ingest = _mm512_add_epi64(ing_s, _mm512_set1_epi64(f.ingest_add));
        // Type, validity, lateness.
        let tw = gather64(wbase, vs[4]);
        let purchase = _mm512_cmpeq_epi64_mask(len[4], _mm512_set1_epi64(8))
            & _mm512_cmpeq_epi64_mask(tw, _mm512_set1_epi64(i64::from_le_bytes(*b"purchase")));
        let t32 = _mm512_and_si512(tw, _mm512_set1_epi64(0xFFFF_FFFF));
        let four = _mm512_cmpeq_epi64_mask(len[4], _mm512_set1_epi64(4));
        let view = four & _mm512_cmpeq_epi64_mask(t32, _mm512_set1_epi64(u32::from_le_bytes(*b"view") as i64));
        let cart = four & _mm512_cmpeq_epi64_mask(t32, _mm512_set1_epi64(u32::from_le_bytes(*b"cart") as i64));
        let ts_ok = _mm512_cmplt_epu64_mask(ts_rel, _mm512_set1_epi64(f.win_ms));
        let ing_ms = _mm512_sub_epi64(mul_epi32::<S>(ing_s, _mm512_set1_epi64(1000)), _mm512_set1_epi64(f.ws_frac));
        let late = _mm512_cmpgt_epi64_mask(_mm512_sub_epi64(ing_ms, ts_rel), _mm512_set1_epi64(DAY_MS));
        let base_ok = _mm512_cmpneq_epi64_mask(len[1], zero)
            & ts_ok
            & (purchase | view | cart)
            & _mm512_cmpneq_epi64_mask(len[5], zero)
            & !late;
        // Purchases: price (up to 8 digits here; longer goes to the reference) and category.
        let mut price = zero;
        let mut cat = zero;
        let mut buy_ok: __mmask8 = 0xFF; // lanes whose purchase fields the vector code decided
        let mut price_ok: __mmask8 = 0;
        let pb = purchase & base_ok;
        if pb != 0 {
            let pw = gather64(wbase, _mm256_sub_epi32(ve[6], eight));
            let k = len[6];
            let kc = _mm512_max_epi64(one64, _mm512_min_epi64(k, _mm512_set1_epi64(8)));
            let sh = _mm512_slli_epi64::<3>(_mm512_sub_epi64(_mm512_set1_epi64(8), kc));
            let keep = _mm512_sllv_epi64(_mm512_set1_epi64(-1), sh);
            let (v, dig) = dec8::<S>(_mm512_ternarylogic_epi64::<0xCA>(keep, pw, c64(0x3030_3030_3030_3030)));
            price = v;
            price_ok = dig
                & _mm512_cmpge_epi64_mask(k, one64)
                & _mm512_cmpge_epu64_mask(v, one64)
                & _mm512_cmple_epu64_mask(v, _mm512_set1_epi64(10_000_000));
            let long_price = _mm512_cmpgt_epi64_mask(k, _mm512_set1_epi64(8));
            buy_ok &= !(pb & long_price);
            if f.prod32.is_empty() {
                buy_ok &= !pb;
            } else {
                let pp = _mm256_cmpeq_epi32_mask(_mm256_and_si256(dword_bytes(wbase, vs[5]), ff32), _mm256_set1_epi32(b'p' as i32));
                let (code, found) = numbered::<S>(wbase, ve[5], len[5], pp, &f.prod32);
                cat = _mm512_mask_blend_epi64(found, _mm512_set1_epi64(f.unknown as i64), code);
            }
        }
        let surv = base_ok & (!purchase | price_ok);
        let purch = surv & purchase;
        let fast_ok = ts_fast & ing_fast & buy_ok;
        let nonempty = _mm512_cmpneq_epi64_mask(len[0], zero);
        let ing_fits = _mm512_cmplt_epu64_mask(ingest, _mm512_set1_epi64(INGEST_LIMIT as i64));
        // A lane goes into the batch when it is decided, non-empty, keyed, its ingest fits, and it
        // does not survive or has a keyed user_id.
        let batchable = fast_ok & nonempty & ev_ok & ing_fits & (!surv | user_ok);
        // Compact the batchable lanes (vpcompressq), branch-free.
        let m = batchable;
        let b = &mut out.b;
        b.n = m.count_ones();
        store64(_mm512_maskz_compress_epi64(m, e_a), &mut b.ev_a);
        store64(_mm512_maskz_compress_epi64(m, e_b), &mut b.ev_b);
        store64(_mm512_maskz_compress_epi64(m, u_a), &mut b.user_a);
        store64(_mm512_maskz_compress_epi64(m, u_b), &mut b.user_b);
        store32(_mm512_maskz_compress_epi64(m, ts_rel), &mut b.ts);
        store32(_mm512_maskz_compress_epi64(m, ingest), &mut b.ingest);
        store32(_mm512_maskz_compress_epi64(m, price), &mut b.price);
        _mm_storel_epi64(b.cat.as_mut_ptr() as *mut __m128i, _mm512_cvtepi64_epi8(_mm512_maskz_compress_epi64(m, cat)));
        b.ev_long = _pext_u32(e_long as u32, m as u32) as u8;
        b.user_long = _pext_u32(u_long as u32, m as u32) as u8;
        b.surv = _pext_u32(surv as u32, m as u32) as u8;
        b.purch = _pext_u32(purch as u32, m as u32) as u8;
        out.slow = !fast_ok;
        out.foreign = fast_ok & nonempty & !batchable;
        _mm256_storeu_si256(out.ls.as_mut_ptr() as *mut __m256i, ls);
        _mm256_storeu_si256(out.le.as_mut_ptr() as *mut __m256i, sep[6]);
    }
}

/// Decodes 8 click lines (separators from k0, stride 4). Needs numbered campaigns. Not generic
/// over the sink (see `ev_batch`).
#[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl,avx512cd,avx2,fma,bmi1,bmi2,popcnt,lzcnt")]
#[inline(never)]
pub(crate) unsafe fn click_batch<const S: bool>(f: &Fast, dates: &Dates, base: *const u8, file: u32, sc: &Scratch, wpos: usize, k0: usize, out: &mut Out<ClickBatch>) {
    unsafe {
        let wbase = base.add(wpos);
        let (vs, ve, ls, sep) = bounds::<4>(sc, k0);
        let mut len = [_mm512_setzero_si512(); 4];
        for j in 0..4 {
            len[j] = _mm512_cvtepi32_epi64(_mm256_sub_epi32(ve[j], vs[j]));
        }
        let eight = _mm256_set1_epi32(8);
        let zero = _mm512_setzero_si512();
        let one64 = _mm512_set1_epi64(1);
        let ff32 = _mm256_set1_epi32(0xFF);
        let w_u = gather64(wbase, _mm256_sub_epi32(ve[1], eight));
        let pu = _mm256_cmpeq_epi32_mask(_mm256_and_si256(dword_bytes(wbase, vs[1]), ff32), _mm256_set1_epi32(b'u' as i32));
        let x_u = ids::<S>(w_u, _mm512_sub_epi64(len[1], one64), pu);
        let u_short = _mm512_cmpneq_epi64_mask(x_u, _mm512_set1_epi64(-1));
        let (mut u_a, mut u_b, mut u_long) = (x_u, zero, 0u8);
        if _mm512_cmpeq_epi64_mask(len[1], _mm512_set1_epi64(33)) != 0 {
            let (a, b, ok) = long_keys::<S>(wbase, ve[1], w_u, len[1], pu);
            u_long = ok & !u_short;
            u_a = _mm512_mask_blend_epi64(u_long, x_u, a);
            u_b = _mm512_maskz_mov_epi64(u_long, b);
        }
        let user_ok = u_short | u_long;
        // Campaign `c_` + 1..8 canonical digits.
        let pc = _mm256_cmpeq_epi32_mask(
            _mm256_and_si256(dword_bytes(wbase, vs[2]), _mm256_set1_epi32(0xFFFF)),
            _mm256_set1_epi32(u16::from_le_bytes(*b"c_") as i32),
        );
        let (camp, camp_ok) = numbered::<S>(wbase, ve[2], _mm512_sub_epi64(len[2], one64), pc, &f.camp32);
        // ts.
        let t_hi = gather64(wbase, vs[3]);
        let t_mid = gather64(wbase, _mm256_add_epi32(vs[3], eight));
        let t_16 = gather64(wbase, _mm256_add_epi32(vs[3], _mm256_set1_epi32(16)));
        let t_17 = gather64(wbase, _mm256_add_epi32(vs[3], _mm256_set1_epi32(17)));
        let (day, date_ok) = date::<S>(dates, t_hi, t_mid);
        let t_ok = _mm512_cmpeq_epi64_mask(byte_at(t_mid, 2), _mm512_set1_epi64(b'T' as i64));
        let (secs, hms_ok) = hms::<S>(_mm512_or_si512(_mm512_srli_epi64::<24>(t_mid), _mm512_slli_epi64::<40>(t_16)));
        let z = _mm512_cmpeq_epi64_mask(byte_at(t_16, 3), _mm512_set1_epi64(b'Z' as i64));
        let (off, off_ok) = offset::<S>(t_17);
        let dt_ok = date_ok & t_ok & hms_ok;
        let f1 = _mm512_cmpeq_epi64_mask(len[3], _mm512_set1_epi64(20)) & z & dt_ok;
        let f2 = _mm512_cmpeq_epi64_mask(len[3], _mm512_set1_epi64(25)) & off_ok & dt_ok;
        let (hi, hi_ok) = dec8::<S>(t_hi);
        let (lo, lo_ok) = dec8::<S>(_mm512_or_si512(_mm512_slli_epi64::<24>(t_mid), _mm512_set1_epi64(0x0030_3030)));
        let f3 = _mm512_cmpeq_epi64_mask(len[3], _mm512_set1_epi64(13)) & hi_ok & lo_ok;
        let local = _mm512_sub_epi64(
            _mm512_add_epi64(mul_epi32::<S>(day, _mm512_set1_epi64(DAY_MS)), mul_epu32::<S>(secs, _mm512_set1_epi64(1000))),
            _mm512_set1_epi64(f.ws_frac),
        );
        let f12 = _mm512_mask_sub_epi64(local, f2, local, off);
        let v3 = _mm512_sub_epi64(_mm512_add_epi64(mul_epu32::<S>(hi, _mm512_set1_epi64(100_000)), lo), _mm512_set1_epi64(f.ws_ms));
        // ts relative to window_start - 7 days.
        let ts_rel = _mm512_add_epi64(_mm512_mask_blend_epi64(f3, f12, v3), _mm512_set1_epi64(CLICK_BASE_BEFORE_WINDOW_MS));
        let ts_fast = f1 | f2 | f3;
        let ts_ok = _mm512_cmplt_epu64_mask(ts_rel, _mm512_set1_epi64(f.win_ms + CLICK_BASE_BEFORE_WINDOW_MS));
        let valid = _mm512_cmpneq_epi64_mask(len[0], zero) & _mm512_cmpneq_epi64_mask(len[1], zero) & ts_ok & camp_ok;
        let batchable = ts_fast & valid & user_ok;
        let loc = _mm512_or_si512(
            _mm512_slli_epi64::<{ Locator::FILE_BITS }>(_mm512_add_epi64(_mm512_cvtepu32_epi64(ls), _mm512_set1_epi64(wpos as i64))),
            _mm512_set1_epi64(file as i64),
        );
        let m = batchable;
        let b = &mut out.b;
        b.n = m.count_ones();
        store64(_mm512_maskz_compress_epi64(m, u_a), &mut b.user_a);
        store64(_mm512_maskz_compress_epi64(m, u_b), &mut b.user_b);
        store32(_mm512_maskz_compress_epi64(m, ts_rel), &mut b.ts);
        _mm_storeu_si128(b.campaign.as_mut_ptr() as *mut __m128i, _mm512_cvtepi64_epi16(_mm512_maskz_compress_epi64(m, camp)));
        store64(_mm512_maskz_compress_epi64(m, loc), &mut b.loc);
        b.user_long = _pext_u32(u_long as u32, m as u32) as u8;
        out.slow = !ts_fast;
        out.foreign = ts_fast & valid & !batchable;
        _mm256_storeu_si256(out.ls.as_mut_ptr() as *mut __m256i, ls);
        _mm256_storeu_si256(out.le.as_mut_ptr() as *mut __m256i, sep[3]);
    }
}

/// Delivers a decoded event batch: the batch, then the lanes for the reference and the foreign
/// lanes. The only part generic over the sink.
#[inline(always)]
pub(crate) fn emit_events<'b, S: Sink>(f: &Fast, file: u32, buf: Buf<'b>, wpos: usize, out: &Out<EvBatch>, fields: &mut Vec<&'b [u8]>, sink: &mut S) {
    if out.b.n > 0 {
        sink.events(&out.b);
    }
    let mut rest = out.slow | out.foreign;
    while rest != 0 {
        let l = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        let start = wpos + out.ls[l] as usize;
        if out.slow & (1 << l) != 0 {
            fast::count_slow();
            let line = &buf.data[start..wpos + out.le[l] as usize];
            reference::event_one(f.ctx, file, start, line, fields, sink);
        } else {
            sink.foreign_event(Locator::new(file, start as u64));
        }
    }
}

/// As `emit_events`, for clicks.
#[inline(always)]
pub(crate) fn emit_clicks<'b, S: Sink>(f: &Fast, file: u32, buf: Buf<'b>, wpos: usize, out: &Out<ClickBatch>, fields: &mut Vec<&'b [u8]>, sink: &mut S) {
    if out.b.n > 0 {
        sink.clicks(&out.b);
    }
    let mut rest = out.slow | out.foreign;
    while rest != 0 {
        let l = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        let start = wpos + out.ls[l] as usize;
        if out.slow & (1 << l) != 0 {
            fast::count_slow();
            let line = &buf.data[start..wpos + out.le[l] as usize];
            reference::click_one(f.ctx, file, start, line, fields, sink);
        } else {
            sink.foreign_click(Locator::new(file, start as u64));
        }
    }
}
