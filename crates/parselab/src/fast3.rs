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
use crate::fast2::{BAD_DAY, Dates, Scratch};
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

// The 256-bit multiplies by a constant, with the constant as a memory operand of the instruction.
// Given as a value, the constant is loaded into a register once per block and shared by the two
// halves (and the other chains of the block): one more instruction per constant per chain.
#[repr(C, align(32))]
struct K32<T>(T);
static K_36_1: K32<[u16; 16]> = K32([0x0124; 16]);
static K_1296_1: K32<[u32; 8]> = K32([0x0001_0510; 8]);
static K_36P4: K32<[u64; 4]> = K32([1_679_616; 4]);
static K_10_1: K32<[u16; 16]> = K32([0x010A; 16]);
static K_100_1: K32<[u32; 8]> = K32([0x0001_0064; 8]);
static K_10P4: K32<[u64; 4]> = K32([10_000; 4]);
static K_100000: K32<[u64; 4]> = K32([100_000; 4]);
static K_60000: K32<[u64; 4]> = K32([60_000; 4]);
static K_1000: K32<[u64; 4]> = K32([1000; 4]);
macro_rules! asm256k {
    ($name:ident, $ins:literal, $feat:literal, $k:ident) => {
        #[target_feature(enable = $feat)]
        #[inline]
        unsafe fn $name(a: __m256i) -> __m256i {
            let r: __m256i;
            unsafe {
                std::arch::asm!(concat!($ins, " {r}, {a}, ymmword ptr [rip + {k}]"), r = lateout(ymm_reg) r, a = in(ymm_reg) a, k = sym $k, options(pure, readonly, nostack, preserves_flags));
            }
            r
        }
    };
}
asm256k!(maddubs_36, "vpmaddubsw", "avx2", K_36_1);
asm256k!(madd_1296, "vpmaddwd", "avx2", K_1296_1);
asm256k!(mulu_36p4, "vpmuludq", "avx2", K_36P4);
asm256k!(maddubs_10, "vpmaddubsw", "avx2", K_10_1);
asm256k!(madd_100, "vpmaddwd", "avx2", K_100_1);
asm256k!(mulu_10p4, "vpmuludq", "avx2", K_10P4);
asm256k!(mulu_100000, "vpmuludq", "avx2", K_100000);
asm256k!(mulu_60000, "vpmuludq", "avx2", K_60000);
asm256k!(muls_1000, "vpmuldq", "avx2", K_1000);

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

/// 8 base-36 IDs: (value or NONE40, which lanes have a value), from the 8 bytes ending at each ID's end, the field's
/// length (prefix + k digits, as i64, any value) and whether its prefix byte matched.
#[inline(always)]
unsafe fn ids<const S: bool>(w: __m512i, len: __m512i, prefix_ok: __mmask8) -> (__m512i, __mmask8) {
    unsafe {
        // Keep the last k = len - 1 bytes: shift 8 * (8 - k). No clamp: for k outside 1..=8 the
        // shift count is out of range (the lane becomes all '0') and the count check rejects it.
        let sh = _mm512_slli_epi64::<3>(_mm512_sub_epi64(_mm512_set1_epi64(9), len));
        let keep = _mm512_sllv_epi64(_mm512_set1_epi64(-1), sh);
        let w = _mm512_ternarylogic_epi64::<0xCA>(keep, w, c64(0x3030_3030_3030_3030));
        // t = c - '0' is 0..=9 only for a digit; a = c - 'a' is 0..=25 only for a letter (the
        // letter test must be on a: c - 87 <= 35 would also take the ten characters before 'a').
        // bad is nonzero in a byte that is neither. The value is the smaller of t and a + 10 (for a
        // digit a + 10 is 217 and up, for a letter t is 49 and up). All byte arithmetic: no
        // compare into a mask (those are port-5 only).
        let t = _mm512_sub_epi8(w, _mm512_set1_epi8(b'0' as i8));
        let a = _mm512_sub_epi8(w, _mm512_set1_epi8(b'a' as i8));
        let bad = _mm512_min_epu8(_mm512_subs_epu8(t, _mm512_set1_epi8(9)), _mm512_subs_epu8(a, _mm512_set1_epi8(25)));
        let v = _mm512_min_epu8(t, _mm512_add_epi8(a, _mm512_set1_epi8(10)));
        // bytes [36, 1] -> i16 [1296, 1] -> u32 pairs -> hi * 36^4 + lo
        let value = chain::<S>(
            v,
            |v| {
                let t = _mm512_madd_epi16(_mm512_maddubs_epi16(v, _mm512_set1_epi16(0x0124)), _mm512_set1_epi32(0x0001_0510));
                _mm512_add_epi64(mul_epu32_512(t, _mm512_set1_epi64(1_679_616)), _mm512_srli_epi64::<32>(t))
            },
            |v| {
                let t = madd_1296(maddubs_36(v));
                _mm256_add_epi64(mulu_36p4(t), _mm256_srli_epi64::<32>(t))
            },
        );
        // Canonical (first digit nonzero unless k == 1) and at most 40 bits: 36^(k-1) <= value (0 for
        // k == 1) <= MASK40, the lower bound looked up by len & 7 (len 2..=9 for k 1..=8).
        let low = _mm512_permutexvar_epi64(len, _mm512_setr_epi64(2_176_782_336, 78_364_164_096, 0, 36, 1296, 46_656, 1_679_616, 60_466_176));
        // Everything that can be wrong, OR-ed into one word and tested once: a bad character, bits
        // above 40, value below the bound (the sign of value - low), a count outside 1..=8 (sh not
        // one of 0, 8, .., 56).
        let r = _mm512_ternarylogic_epi64::<0xF8>(bad, value, _mm512_set1_epi64(!(MASK40 as i64)));
        let r = _mm512_ternarylogic_epi64::<0xF8>(r, _mm512_sub_epi64(value, low), _mm512_set1_epi64(i64::MIN));
        let r = _mm512_ternarylogic_epi64::<0xF8>(r, sh, _mm512_set1_epi64(!0x38));
        let good = _mm512_testn_epi64_mask(r, r) & prefix_ok;
        (_mm512_mask_blend_epi64(good, _mm512_set1_epi64(-1), value), good)
    }
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

/// `YYYY-MM-DD` over hi (bytes 0..8) and mid (bytes 0..2): (day * 86400 relative to
/// window_start's day, valid and in the table, mid ^ `3x3xT`: its byte 2 is 0 when the `T` is
/// there, for `hms` to fold into its one test). No digit decode: the low bits of the characters
/// pick a slot of `Dates::ent` (byte sums, no multiply) and the text is compared with the one date
/// that lives there, so it is valid exactly when it is the text of a table date. Other lanes get
/// BAD_DAY * 86400, as the multiply of the old day table's BAD_DAY gave.
#[inline(always)]
unsafe fn date(dates: &Dates, hi: __m512i, mid: __m512i) -> (__m512i, __mmask8, __m512i) {
    unsafe {
        // Y4 bit -> 0x40 of byte 3, M1 bit -> 0x80 of byte 5, M2 nibble -> 0x3C of byte 6; the
        // byte sum is slot bits 2..8, and D1 (bits 0..2) and D2 (bits 8..12) come from mid as is.
        let x = _mm512_and_si512(hi, c64(0x000F_0100_0100_0000));
        let h = _mm512_sad_epu8(_mm512_sllv_epi16(x, c64(0x0002_0007_0006_0000)), _mm512_setzero_si512());
        let idx = _mm512_ternarylogic_epi64::<0xF8>(h, mid, c64(0x0F03));
        let e = _mm512_i64gather_epi64::<8>(idx, dates.ent_ptr() as *const i64);
        // Expected bytes 0..8: the slot's year, then `-0x-3x-` up to the slot bits.
        let want = _mm512_ternarylogic_epi64::<0xEA>(e, c64(0xFFFF_FFFF), c64(0x2D30_302D_0000_0000));
        // mid against `3x3xT`: the day digits' high bits are part of the date; the `T` after them
        // is not part of what selects the day, and is left in xm for the caller.
        let xm = _mm512_xor_si512(mid, c64(0x0054_3030));
        // One test for the date: the differences of hi (under its mask) and of mid's day digits.
        let r = _mm512_ternarylogic_epi64::<0x28>(hi, want, c64(0xFFF0_FEFF_FFFF_FFFF));
        let r = _mm512_ternarylogic_epi64::<0xF8>(r, xm, c64(0xF0FC));
        let ok = _mm512_testn_epi64_mask(r, r);
        let day_s = _mm512_mask_srai_epi64::<32>(_mm512_set1_epi64(BAD_DAY as i64 * 86_400), ok, e);
        (day_s, ok, xm)
    }
}

/// x * 1000 as (x << 10) - (x << 4) - (x << 3): exact on 64 bits, light work on both paths.
#[inline(always)]
unsafe fn times1000(x: __m512i) -> __m512i {
    unsafe {
        let a = opaque512(_mm512_sub_epi64(opaque512(_mm512_slli_epi64::<10>(x)), _mm512_slli_epi64::<4>(x)));
        _mm512_sub_epi64(a, _mm512_slli_epi64::<3>(x))
    }
}

/// `HH:MM:SS` in t (8 bytes), and a word tx whose bytes `txm` must be zero (from `date`: the `T`
/// before the time): (seconds of day, all ok).
#[inline(always)]
unsafe fn hms<const S: bool>(t: __m512i, tx: __m512i, txm: u64) -> (__m512i, __mmask8) {
    unsafe {
        // y: 0 at the colons and the digit values elsewhere, when the text is well formed; z adds
        // 15 - limit to each digit (H 2 9, M 5 9, S 5 9), so a digit over its limit reaches bit 4.
        // Everything that can be wrong is OR-ed into one word r and tested once (mask compares and
        // byte shuffles are port-5 only on the scorer's CPU; logic, shifts and adds are not):
        // high nibbles of y and z, the colon bytes, and the hour over 23.
        let y = _mm512_xor_si512(t, c64(0x3030_3A30_303A_3030));
        let z = _mm512_add_epi8(y, c64(0x060A_0006_0A00_060D));
        let r = _mm512_ternarylogic_epi64::<0xA8>(y, z, c64(0xF0F0_FFF0_F0FF_F0F0));
        let x = _mm512_and_si512(y, c64(0xFFFF_00FF_FF00_FFFF));
        // Bytes 0, 3, 6 of v: hour, minute, second. Byte 1 is 10 * (a digit): even.
        let v = _mm512_add_epi64(_mm512_add_epi64(opaque::<S>(_mm512_slli_epi64::<3>(x)), _mm512_slli_epi64::<1>(x)), _mm512_srli_epi64::<8>(x));
        // hour + 232 carries out of byte 0 exactly when the hour is over 23: bit 8 (clear before).
        let over = _mm512_add_epi64(v, _mm512_set1_epi64(232));
        let r = _mm512_ternarylogic_epi64::<0xF8>(r, over, _mm512_set1_epi64(0x100));
        // and what the caller wants tested with it: the bytes `txm` of tx must be zero (the `T`
        // before the time: byte 2 of the date's mid word, see `date`; for ingest also the `Z`).
        let r = _mm512_ternarylogic_epi64::<0xF8>(r, tx, c64(txm));
        let ok = _mm512_testn_epi64_mask(r, r);
        // (h * 60 + m) * 60 + s on the three fields in place (24 bits apart; each * 60 as
        // (x << 6) - (x << 2) stays inside its field), no byte extraction.
        let w = _mm512_and_si512(v, c64(0x00FF_0000_FF00_00FF));
        let hm = _mm512_add_epi64(times60(w), _mm512_srli_epi64::<24>(w));
        let secs = _mm512_and_si512(_mm512_add_epi64(times60(hm), _mm512_srli_epi64::<48>(w)), _mm512_set1_epi64(0xFF_FFFF));
        (secs, ok)
    }
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
                let t = madd_100(maddubs_10(x));
                _mm256_add_epi64(mulu_10p4(t), _mm256_srli_epi64::<32>(t))
            },
        )
    };
    (v, ok)
}

/// As `dec8`, with the digit check left as a word that is zero when all 8 are digits (a digit's
/// high nibble is zero, and so is the high nibble of digit + 6): the caller ORs several and tests
/// once.
#[inline(always)]
unsafe fn dec8r<const S: bool>(w: __m512i) -> (__m512i, __m512i) {
    unsafe {
        let x = _mm512_xor_si512(w, c64(0x3030_3030_3030_3030));
        let r = _mm512_ternarylogic_epi64::<0xA8>(x, _mm512_add_epi8(x, _mm512_set1_epi8(6)), c64(0xF0F0_F0F0_F0F0_F0F0));
        let v = chain::<S>(
            x,
            |x| {
                let t = _mm512_madd_epi16(_mm512_maddubs_epi16(x, _mm512_set1_epi16(0x010A)), _mm512_set1_epi32(0x0001_0064));
                _mm512_add_epi64(mul_epu32_512(t, _mm512_set1_epi64(10_000)), _mm512_srli_epi64::<32>(t))
            },
            |x| {
                let t = madd_100(maddubs_10(x));
                _mm256_add_epi64(mulu_10p4(t), _mm256_srli_epi64::<32>(t))
            },
        );
        (v, r)
    }
}

/// Offset `SS±HH:MM` (bytes 0..8 of t17): (signed offset ms, ok).
#[inline(always)]
unsafe fn offset<const S: bool>(o: __m512i) -> (__m512i, __mmask8) {
    unsafe {
        let sign = byte_at(o, 2);
        // As in `hms`: the colon, the digit limits (H 2 9, M 5 9) and the hour over 23 in one word,
        // tested once.
        let y = _mm512_xor_si512(o, c64(0x3030_3A30_3000_0000));
        let z = _mm512_add_epi8(y, c64(0x060A_0006_0D00_0000));
        let r = _mm512_ternarylogic_epi64::<0xA8>(y, z, c64(0xF0F0_FFF0_F000_0000));
        let x = _mm512_and_si512(y, c64(0xFFFF_00FF_FF00_0000));
        // Bytes 3 and 6 of v: hour and minute. Byte 4 is 10 * (a digit): even.
        let v = _mm512_add_epi64(_mm512_add_epi64(opaque::<S>(_mm512_slli_epi64::<3>(x)), _mm512_slli_epi64::<1>(x)), _mm512_srli_epi64::<8>(x));
        let over = _mm512_add_epi64(v, _mm512_set1_epi64(232 << 24));
        let r = _mm512_ternarylogic_epi64::<0xF8>(r, over, _mm512_set1_epi64(1 << 32));
        let shape = _mm512_testn_epi64_mask(r, r);
        // h * 60 + m on the two fields in place, then down to bit 0.
        let w = _mm512_and_si512(v, c64(0x00FF_0000_FF00_0000));
        let hm = _mm512_and_si512(_mm512_srli_epi64::<24>(_mm512_add_epi64(times60(w), _mm512_srli_epi64::<24>(w))), _mm512_set1_epi64(0xFF_FFFF));
        let mag = chain::<S>(hm, |x| _mm512_mul_epu32(x, _mm512_set1_epi64(60_000)), |x| mulu_60000(x));
        let plus = _mm512_cmpeq_epi64_mask(sign, _mm512_set1_epi64(b'+' as i64));
        let minus = _mm512_cmpeq_epi64_mask(sign, _mm512_set1_epi64(b'-' as i64));
        let off = _mm512_mask_sub_epi64(mag, !plus, _mm512_setzero_si512(), mag);
        let ok = (plus | minus) & shape;
        (off, ok)
    }
}

/// The LONG lanes of one ID field merged into its key vectors: (a, b, long lanes). `x` is the SHORT
/// decode (a lane of length 33 is never SHORT: `ids` takes at most 8 characters). One lane at a
/// time: the 32 hex characters ending at the lane's `ve` (window-relative) as one 256-bit load, to
/// a (first 16 characters) and b (last 16), each put into its lane of the key vectors by a masked
/// broadcast when all 32 are `0-9a-f`. One lane in eight is LONG: a batch has one such lane on
/// average, and the old 8-lane decode (three more gathers, four hex8) ran for every batch that
/// had any.
///
/// The keys go register to register. Writing them to a stack array and loading the array as one
/// vector cost more than the whole decode: a 64-byte load cannot be forwarded from 8-byte stores
/// (it waits for them to retire), and that one load was a quarter of ev_batch's cycles.
#[inline(always)]
unsafe fn with_long(wbase: *const u8, ve: __m256i, len: __m512i, pre_ok: __mmask8, x: __m512i) -> (__m512i, __m512i, __mmask8) {
    unsafe {
        let lanes = _mm512_cmpeq_epi64_mask(len, _mm512_set1_epi64(33)) & pre_ok;
        let (mut a, mut b) = (x, _mm512_setzero_si512());
        if lanes == 0 {
            return (a, b, 0);
        }
        // One 32-byte store, then 4-byte loads inside it: those forward.
        let mut ends = [0u32; 8];
        _mm256_storeu_si256(ends.as_mut_ptr() as *mut __m256i, ve);
        let swap = _mm_set_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
        let mut rest = lanes as u32;
        let mut long = 0u32;
        // Written out eight times, not as a loop: a loop gets its seven constants loaded into
        // registers before it (once per batch that has a LONG lane, for one or two lanes); here
        // each copy uses them as memory operands.
        macro_rules! lane {
            ($done:lifetime) => {
                if rest == 0 {
                    break $done;
                }
                let l = rest.trailing_zeros() as usize;
                rest &= rest - 1;
                let v = _mm256_loadu_si256(wbase.add(*ends.get_unchecked(l) as usize).sub(32) as *const __m256i);
                let d = _mm256_sub_epi8(v, _mm256_set1_epi8(b'0' as i8));
                let t = _mm256_sub_epi8(v, _mm256_set1_epi8(b'a' as i8));
                let digit = _mm256_cmplt_epu8_mask(d, _mm256_set1_epi8(10));
                let letter = _mm256_cmplt_epu8_mask(t, _mm256_set1_epi8(6));
                let val = _mm256_mask_add_epi8(d, letter, t, _mm256_set1_epi8(10));
                // nibble pairs [16, 1] -> 16 bytes, then each half byte-reversed: character 0 most significant
                let bytes = _mm_shuffle_epi8(_mm256_cvtepi16_epi8(_mm256_maddubs_epi16(val, _mm256_set1_epi16(0x0110))), swap);
                let k = (((digit | letter) == u32::MAX) as u32) << l;
                a = _mm512_mask_broadcastq_epi64(a, k as u8, bytes);
                b = _mm512_mask_broadcastq_epi64(b, k as u8, _mm_unpackhi_epi64(bytes, bytes));
                long |= k;
            };
        }
        'lanes: {
            lane!('lanes);
            lane!('lanes);
            lane!('lanes);
            lane!('lanes);
            lane!('lanes);
            lane!('lanes);
            lane!('lanes);
            lane!('lanes);
        }
        (a, b, long as u8)
    }
}

/// `prefix` + 1..8 canonical decimal digits ending at ve, looked up in a padded u32 table
/// (0xFFFF = absent): (value from the table, found).
#[inline(always)]
unsafe fn numbered<const S: bool>(wbase: *const u8, ve: __m256i, len: __m512i, pre_ok: __mmask8, table: &[u32]) -> (__m512i, __mmask8) {
    unsafe {
        let one = _mm512_set1_epi64(1);
        let w = gather64o(wbase, ve, -8);
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

/// `gather64` at `idx + off`: the constant goes into the gather's address (its displacement), not
/// into a vector add of the indices.
#[inline(always)]
unsafe fn gather64o(base: *const u8, idx: __m256i, off: isize) -> __m512i {
    unsafe { _mm512_i32gather_epi64::<1>(idx, base.wrapping_offset(off) as *const i64) }
}

#[inline(always)]
unsafe fn dword_bytes(base: *const u8, idx: __m256i) -> __m256i {
    unsafe { _mm256_i32gather_epi32::<1>(base as *const i32, idx) }
}

// v - m (or v + m) in the lanes of k, else v, as one masked instruction: as intrinsics LLVM turns
// the masked add of a constant into a mask-to-vector conversion and an add.
macro_rules! asm_mask32 {
    ($name:ident, $ins:literal) => {
        #[target_feature(enable = "avx512f,avx512vl")]
        #[inline]
        unsafe fn $name(v: __m256i, k: __mmask8, m: __m256i) -> __m256i {
            let mut r = v;
            unsafe {
                std::arch::asm!(concat!($ins, " {r}{{{k}}}, {r}, {m}"), r = inout(ymm_reg) r, k = in(kreg) k, m = in(ymm_reg) m, options(pure, nomem, nostack, preserves_flags));
            }
            r
        }
    };
}
asm_mask32!(mask_sub32, "vpsubd");
asm_mask32!(mask_add32, "vpaddd");

/// Separator positions of 8 lines at stride `nf` (from k0), window-relative, and the line
/// starts and quote-adjusted value bounds of every field. The entries are pure positions; the
/// quote flag before field j of lane l is bit NF * l + j of the quote window from k0 (one load
/// and one pext per field).
#[inline(always)]
unsafe fn bounds<const NF: usize>(sc: &Scratch, k0: usize) -> ([__m256i; NF], [__m256i; NF], __m256i, [__m256i; NF]) {
    unsafe {
        let sp = sc.sep_ptr().add(k0);
        let mut sep = [_mm256_setzero_si256(); NF];
        if NF == 7 {
            // Four loads of two whole lines each (14 entries; 2 spare). One permute per pair of
            // loads puts lines (0,1,4,5) and (2,3,6,7) of four columns side by side as 64-bit
            // halves, and a 64-bit unpack joins them into two whole columns (low and high 256 bits).
            let r = [
                _mm512_loadu_si512(sp as *const __m512i),
                _mm512_loadu_si512(sp.add(14) as *const __m512i),
                _mm512_loadu_si512(sp.add(28) as *const __m512i),
                _mm512_loadu_si512(sp.add(42) as *const __m512i),
            ];
            let ia = _mm512_setr_epi32(0, 7, 1, 8, 16, 23, 17, 24, 2, 9, 3, 10, 18, 25, 19, 26);
            let ib = _mm512_setr_epi32(4, 11, 5, 12, 20, 27, 21, 28, 6, 13, 6, 13, 22, 29, 22, 29);
            let (pa, qa) = (_mm512_permutex2var_epi32(r[0], ia, r[2]), _mm512_permutex2var_epi32(r[1], ia, r[3]));
            let (pb, qb) = (_mm512_permutex2var_epi32(r[0], ib, r[2]), _mm512_permutex2var_epi32(r[1], ib, r[3]));
            let (c02, c13) = (_mm512_unpacklo_epi64(pa, qa), _mm512_unpackhi_epi64(pa, qa));
            let (c46, c5) = (_mm512_unpacklo_epi64(pb, qb), _mm512_unpackhi_epi64(pb, qb));
            let cols = [
                _mm512_castsi512_si256(c02),
                _mm512_castsi512_si256(c13),
                _mm512_extracti64x4_epi64::<1>(c02),
                _mm512_extracti64x4_epi64::<1>(c13),
                _mm512_castsi512_si256(c46),
                _mm512_castsi512_si256(c5),
                _mm512_extracti64x4_epi64::<1>(c46),
            ];
            for (j, c) in sep.iter_mut().enumerate() {
                *c = cols[j];
            }
        } else {
            let z0 = _mm512_loadu_si512(sp as *const __m512i);
            let z1 = _mm512_loadu_si512(sp.add(16) as *const __m512i);
            for (j, c) in sep.iter_mut().enumerate() {
                let mut idx = [0i32; 16];
                for (l, x) in idx.iter_mut().enumerate().take(8) {
                    *x = (NF * l + j) as i32;
                }
                let iv = _mm512_loadu_si512(idx.as_ptr() as *const __m512i);
                *c = _mm512_castsi512_si256(_mm512_permutex2var_epi32(z0, iv, z1));
            }
        }
        // The separator before each line's field 0: the previous line's newline (lane l - 1), or
        // for lane 0 the entry before k0 (the window start when k0 == 0).
        let prev_pos = if k0 == 0 { -1i32 } else { *sp.sub(1) as i32 };
        // Lane l's NF quote flags in the low bits of 64-bit lane l: one test per field, and the
        // value bounds are one masked add each.
        let n = NF as i64;
        let qv = _mm512_srlv_epi64(_mm512_set1_epi64(sc.q_window(k0) as i64), _mm512_setr_epi64(0, n, 2 * n, 3 * n, 4 * n, 5 * n, 6 * n, 7 * n));
        let m1 = _mm256_set1_epi32(-1);
        // [prev_pos, last separators of lanes 0..7] + 1: one element shift across the pair.
        let ls = _mm256_sub_epi32(_mm256_alignr_epi32::<7>(sep[NF - 1], _mm256_set1_epi32(prev_pos)), m1);
        let mut vs = [_mm256_setzero_si256(); NF];
        let mut ve = [_mm256_setzero_si256(); NF];
        for j in 0..NF {
            let q = _mm512_test_epi64_mask(qv, _mm512_set1_epi64(1 << j));
            let s = if j == 0 { ls } else { _mm256_sub_epi32(sep[j - 1], m1) };
            vs[j] = mask_sub32(s, q, m1);
            ve[j] = mask_add32(sep[j], q, m1);
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
    /// Event lanes whose ts alone is undecided (every other field decided, keyed, ingest fitting): if the
    /// ts is not valid inside the window the copy is dead, and its record is pushed from these fields
    /// without the reference. The arrays are set only when `dead` is nonzero.
    pub(crate) dead: u8,
    pub(crate) d_long: u8,
    pub(crate) d_a: [u64; 8],
    pub(crate) d_b: [u64; 8],
    pub(crate) d_ing: [u32; 8],
    /// The ts value bounds (window-relative, quotes excluded).
    pub(crate) t_vs: [u32; 8],
    pub(crate) t_ve: [u32; 8],
}

impl<B> Out<B> {
    pub(crate) fn new(b: B) -> Out<B> {
        Out { b, slow: 0, foreign: 0, ls: [0; 8], le: [0; 8], dead: 0, d_long: 0, d_a: [0; 8], d_b: [0; 8], d_ing: [0; 8], t_vs: [0; 8], t_ve: [0; 8] }
    }
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
        let zero = _mm512_setzero_si512();
        let one64 = _mm512_set1_epi64(1);
        let ff32 = _mm256_set1_epi32(0xFF);
        // IDs: SHORT from the 8 bytes ending at the end, LONG from the 32 ending there.
        let w_e = gather64o(wbase, ve[0], -8);
        let w_u = gather64o(wbase, ve[1], -8);
        // Both prefix bytes in one 16-lane gather.
        let pg = _mm512_i32gather_epi32::<1>(_mm512_inserti64x4::<1>(_mm512_castsi256_si512(vs[0]), vs[1]), wbase as *const i32);
        let pk = _mm512_cmpeq_epi32_mask(
            _mm512_and_si512(pg, _mm512_set1_epi32(0xFF)),
            _mm512_inserti64x4::<1>(_mm512_set1_epi32(b'e' as i32), _mm256_set1_epi32(b'u' as i32)),
        );
        let (pe, pu) = (pk as u8, (pk >> 8) as u8);
        // Order matters for the instruction count: the LONG branch between the two decodes ends
        // the block, so each decode's constants are single-use memory operands instead of one
        // register load each shared by both.
        let (x_e, e_short) = ids::<S>(w_e, len[0], pe);
        let (e_a, e_b, e_long) = with_long(wbase, ve[0], len[0], pe, x_e);
        let (x_u, u_short) = ids::<S>(w_u, len[1], pu);
        let (u_a, u_b, u_long) = with_long(wbase, ve[1], len[1], pu, x_u);
        let ev_ok = e_short | e_long;
        let user_ok = u_short | u_long;
        // ts.
        let t_hi = gather64(wbase, vs[2]);
        let t_mid = gather64o(wbase, vs[2], 8);
        // HH:MM:SS (bytes 11..19) as its own gather: no shifts to join it from two others.
        let t_11 = gather64o(wbase, vs[2], 11);
        let t_17 = gather64o(wbase, vs[2], 17);
        let (day_s, date_ok, t_x) = date(dates, t_hi, t_mid);
        let (secs, hms_ok) = hms::<S>(t_11, t_x, 0x00FF_0000);
        let z = _mm512_cmpeq_epi64_mask(byte_at(t_17, 2), _mm512_set1_epi64(b'Z' as i64));
        let (off, off_ok) = offset::<S>(t_17);
        let dt_ok = date_ok & hms_ok;
        let f1 = _mm512_cmpeq_epi64_mask(len[2], _mm512_set1_epi64(20)) & z & dt_ok;
        let f2 = _mm512_cmpeq_epi64_mask(len[2], _mm512_set1_epi64(25)) & off_ok & dt_ok;
        // day * DAY_MS + secs * 1000 as (day * 86400 + secs) * 1000, by shifts.
        let local = _mm512_sub_epi64(times1000(_mm512_add_epi64(day_s, secs)), _mm512_set1_epi64(f.ws_frac));
        let f12 = _mm512_mask_sub_epi64(local, f2, local, off);
        // Epoch ms: only when a lane has 13 characters (nearly every batch: one line in three).
        // The branch is predictable, and it ends the block: the ISO decode above and the ingest
        // decode below then each fold their constants into memory operands instead of sharing
        // one register load per constant.
        let m13 = _mm512_cmpeq_epi64_mask(len[2], _mm512_set1_epi64(13));
        let (mut f3, mut v3) = (0u8, zero);
        if m13 != 0 {
            let (hi, hi_r) = dec8r::<S>(t_hi);
            let (lo, lo_r) = dec8r::<S>(_mm512_or_si512(_mm512_slli_epi64::<24>(t_mid), _mm512_set1_epi64(0x0030_3030)));
            let r = _mm512_or_si512(hi_r, lo_r);
            f3 = m13 & _mm512_testn_epi64_mask(r, r);
            v3 = _mm512_sub_epi64(_mm512_add_epi64(chain::<S>(hi, |x| _mm512_mul_epu32(x, _mm512_set1_epi64(100_000)), |x| mulu_100000(x)), lo), _mm512_set1_epi64(f.ws_ms));
        }
        let ts_rel = _mm512_mask_blend_epi64(f3, f12, v3);
        let ts_fast = f1 | f2 | f3;
        // ingest (always F1).
        let i_hi = gather64(wbase, vs[3]);
        let i_mid = gather64o(wbase, vs[3], 8);
        let i_16 = gather64o(wbase, vs[3], 16);
        let (iday_s, idate_ok, i_x) = date(dates, i_hi, i_mid);
        // The `Z` (byte 3 of i_16) joins the `T` (byte 2 of i_x) in the word hms tests: bytes 2 and 3.
        let i_zx = _mm512_ternarylogic_epi64::<0xD8>(i_x, _mm512_xor_si512(i_16, _mm512_set1_epi64((b'Z' as i64) << 24)), _mm512_set1_epi64(0xFF00_0000));
        let (isecs, ihms_ok) = hms::<S>(_mm512_or_si512(_mm512_srli_epi64::<24>(i_mid), _mm512_slli_epi64::<40>(i_16)), i_zx, 0xFFFF_0000);
        let ing_fast = _mm512_cmpeq_epi64_mask(len[3], _mm512_set1_epi64(20)) & idate_ok & ihms_ok;
        let ing_s = _mm512_add_epi64(iday_s, isecs);
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
        let ing_ms = _mm512_sub_epi64(chain::<S>(ing_s, |x| _mm512_mul_epi32(x, _mm512_set1_epi64(1000)), |x| muls_1000(x)), _mm512_set1_epi64(f.ws_frac));
        let late = _mm512_cmpgt_epi64_mask(_mm512_sub_epi64(ing_ms, ts_rel), _mm512_set1_epi64(DAY_MS));
        // user_id and product_id both non-empty: the smaller length is not zero (one compare).
        let both = _mm256_min_epu32(_mm256_sub_epi32(ve[1], vs[1]), _mm256_sub_epi32(ve[5], vs[5]));
        let base_ok = _mm256_cmpneq_epi32_mask(both, _mm256_setzero_si256()) & ts_ok & (purchase | view | cart) & !late;
        // Purchases: price (up to 8 digits here; longer goes to the reference) and category.
        let mut price = zero;
        let mut cat = zero;
        let mut buy_ok: __mmask8 = 0xFF; // lanes whose purchase fields the vector code decided
        let mut price_ok: __mmask8 = 0;
        let pb = purchase & base_ok;
        if pb != 0 {
            let pw = gather64o(wbase, ve[6], -8);
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
        let m = batchable;
        let b = &mut out.b;
        b.n = m.count_ones();
        if m == 0xFF {
            // Every lane goes into the batch (nearly every batch): compaction is the identity, so
            // the eight compresses (two port-5 uops each) and the four bit extracts are skipped.
            store64(e_a, &mut b.ev_a);
            store64(e_b, &mut b.ev_b);
            store64(u_a, &mut b.user_a);
            store64(u_b, &mut b.user_b);
            store32(ts_rel, &mut b.ts);
            store32(ingest, &mut b.ingest);
            store32(price, &mut b.price);
            _mm_storel_epi64(b.cat.as_mut_ptr() as *mut __m128i, _mm512_cvtepi64_epi8(cat));
            b.ev_long = e_long;
            b.user_long = u_long;
            b.surv = surv;
            b.purch = purch;
        } else {
            // Compact the batchable lanes (vpcompressq).
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
        }
        out.slow = !fast_ok;
        out.foreign = fast_ok & nonempty & !batchable;
        _mm256_storeu_si256(out.ls.as_mut_ptr() as *mut __m256i, ls);
        _mm256_storeu_si256(out.le.as_mut_ptr() as *mut __m256i, sep[6]);
        // Lanes undecided only by their ts (dirty ts text is most of the slow lanes).
        let dead = !ts_fast & ing_fast & ev_ok & nonempty & ing_fits;
        out.dead = dead;
        if dead != 0 {
            out.d_long = e_long;
            store64(e_a, &mut out.d_a);
            store64(e_b, &mut out.d_b);
            store32(ingest, &mut out.d_ing);
            _mm256_storeu_si256(out.t_vs.as_mut_ptr() as *mut __m256i, vs[2]);
            _mm256_storeu_si256(out.t_ve.as_mut_ptr() as *mut __m256i, ve[2]);
        }
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
        let zero = _mm512_setzero_si512();
        let one64 = _mm512_set1_epi64(1);
        let ff32 = _mm256_set1_epi32(0xFF);
        let w_u = gather64o(wbase, ve[1], -8);
        let pu = _mm256_cmpeq_epi32_mask(_mm256_and_si256(dword_bytes(wbase, vs[1]), ff32), _mm256_set1_epi32(b'u' as i32));
        let (x_u, u_short) = ids::<S>(w_u, len[1], pu);
        let (u_a, u_b, u_long) = with_long(wbase, ve[1], len[1], pu, x_u);
        let user_ok = u_short | u_long;
        // Campaign `c_` + 1..8 canonical digits.
        let pc = _mm256_cmpeq_epi32_mask(
            _mm256_and_si256(dword_bytes(wbase, vs[2]), _mm256_set1_epi32(0xFFFF)),
            _mm256_set1_epi32(u16::from_le_bytes(*b"c_") as i32),
        );
        let (camp, camp_ok) = numbered::<S>(wbase, ve[2], _mm512_sub_epi64(len[2], one64), pc, &f.camp32);
        // ts.
        let t_hi = gather64(wbase, vs[3]);
        let t_mid = gather64o(wbase, vs[3], 8);
        // HH:MM:SS (bytes 11..19) as its own gather: no shifts to join it from two others.
        let t_11 = gather64o(wbase, vs[3], 11);
        let t_17 = gather64o(wbase, vs[3], 17);
        let (day_s, date_ok, t_x) = date(dates, t_hi, t_mid);
        let (secs, hms_ok) = hms::<S>(t_11, t_x, 0x00FF_0000);
        let z = _mm512_cmpeq_epi64_mask(byte_at(t_17, 2), _mm512_set1_epi64(b'Z' as i64));
        let (off, off_ok) = offset::<S>(t_17);
        let dt_ok = date_ok & hms_ok;
        let f1 = _mm512_cmpeq_epi64_mask(len[3], _mm512_set1_epi64(20)) & z & dt_ok;
        let f2 = _mm512_cmpeq_epi64_mask(len[3], _mm512_set1_epi64(25)) & off_ok & dt_ok;
        let (hi, hi_r) = dec8r::<S>(t_hi);
        let (lo, lo_r) = dec8r::<S>(_mm512_or_si512(_mm512_slli_epi64::<24>(t_mid), _mm512_set1_epi64(0x0030_3030)));
        let r = _mm512_or_si512(hi_r, lo_r);
        let f3 = _mm512_cmpeq_epi64_mask(len[3], _mm512_set1_epi64(13)) & _mm512_testn_epi64_mask(r, r);
        // day * DAY_MS + secs * 1000 as (day * 86400 + secs) * 1000, by shifts.
        let local = _mm512_sub_epi64(times1000(_mm512_add_epi64(day_s, secs)), _mm512_set1_epi64(f.ws_frac));
        let f12 = _mm512_mask_sub_epi64(local, f2, local, off);
        let v3 = _mm512_sub_epi64(_mm512_add_epi64(chain::<S>(hi, |x| _mm512_mul_epu32(x, _mm512_set1_epi64(100_000)), |x| mulu_100000(x)), lo), _mm512_set1_epi64(f.ws_ms));
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
        if m == 0xFF {
            // Every lane goes into the batch: compaction is the identity (see `ev_batch`).
            store64(u_a, &mut b.user_a);
            store64(u_b, &mut b.user_b);
            store32(ts_rel, &mut b.ts);
            _mm_storeu_si128(b.campaign.as_mut_ptr() as *mut __m128i, _mm512_cvtepi64_epi16(camp));
            store64(loc, &mut b.loc);
            b.user_long = u_long;
        } else {
            store64(_mm512_maskz_compress_epi64(m, u_a), &mut b.user_a);
            store64(_mm512_maskz_compress_epi64(m, u_b), &mut b.user_b);
            store32(_mm512_maskz_compress_epi64(m, ts_rel), &mut b.ts);
            _mm_storeu_si128(b.campaign.as_mut_ptr() as *mut __m128i, _mm512_cvtepi64_epi16(_mm512_maskz_compress_epi64(m, camp)));
            store64(_mm512_maskz_compress_epi64(m, loc), &mut b.loc);
            b.user_long = _pext_u32(u_long as u32, m as u32) as u8;
        }
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
    let mut dead = EvBatch::default();
    while rest != 0 {
        let l = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        let start = wpos + out.ls[l] as usize;
        if out.dead & (1 << l) != 0 {
            // The reference makes a dead copy of a line whose ts is not valid inside the window (every
            // other field of this lane is decided, keyed and fits): the same record, without it.
            let tv = &buf.data[wpos + out.t_vs[l] as usize..wpos + out.t_ve[l] as usize];
            let ok = crate::ts::parse_ts(tv).is_some_and(|t| f.ctx.window_start_ms <= t && t < f.ctx.window_end_ms);
            if !ok {
                let ev = Key { long: out.d_long & (1 << l) != 0, a: out.d_a[l], b: out.d_b[l] };
                dead.push(ev, Key::default(), false, false, 0, out.d_ing[l], 0, 0);
                continue;
            }
        }
        if out.slow & (1 << l) != 0 {
            fast::count_slow();
            let line = &buf.data[start..wpos + out.le[l] as usize];
            reference::event_one(f.ctx, file, start, line, fields, sink);
        } else {
            sink.foreign_event(Locator::new(file, start as u64));
        }
    }
    if dead.n > 0 {
        sink.events(&dead);
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
