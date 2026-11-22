//! Persons of at most 128 events: sort the timestamps in AVX-512 registers with a bitonic network
//! (branch-free), then cut sessions with vector compares and compress.
//!
//! Generic over the register: `Z` = 16 u32 lanes in a zmm (AVX-512F), `Y` = 8 u32 lanes in a ymm
//! (AVX-512F + VL: same instructions at 256 bits, for CPUs where 512-bit code lowers the clock).
//!
//! Network (comparators i < j take min to i and max to j; all ascending, "with flip"):
//! * in one register of W lanes, phases k = 1..=log2 W: the mirror comparator i ^ (2^k - 1), then
//!   i ^ 2^(k-2), ..., i ^ 1. For W = 16 that is m = 1; 3, 1; 7, 2, 1; 15, 4, 2, 1. A person of at
//!   most W/2 events stops after phase log2 W - 1 (the upper half is padding u32::MAX, sorted);
//! * R registers (R a power of two): sort each register, then for blk = 1, 2, ..., R/2 merge
//!   pairs of sorted blocks of blk registers: element e of the 2*blk-register block against
//!   element 2*blk*W - 1 - e (register 2*blk - 1 - i, lanes reversed); then half-cleaners between
//!   registers i and i + d for d = blk/2, ..., 1 (lanes aligned); then in-register cleaners
//!   i ^ W/2, ..., i ^ 1.
//! Padding lanes hold u32::MAX, above every ts (< 2^31), so they sort last.
//!
//! Cut: after sorting, lane e (e >= 1, e < n) starts a session when t[e] - t[e-1] > GAP; lane 0
//! always starts one. Lane e ends a session when lane e + 1 starts one or e = n - 1. Starts, their
//! positions and end positions are compressed in lane order; events = end - start + 1.

#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

use crate::{Out, GAP};

/// One register of u32 lanes, with the operations the network and the cut need. All methods are
/// `unsafe`: they need the target features of the implementing type.
pub trait Lanes: Copy {
    /// Lanes per register (16 or 8).
    const W: usize;
    /// log2 W.
    const LOG: u32;
    unsafe fn splat(x: u32) -> Self;
    unsafe fn iota() -> Self;
    /// Lanes with bit l of `valid` set load p[l]; the others hold u32::MAX. Only lanes in `valid`
    /// are read (masked load: no fault, no access outside them).
    unsafe fn load_pad(p: *const u32, valid: u32) -> Self;
    unsafe fn load(p: *const u32) -> Self;
    unsafe fn minu(a: Self, b: Self) -> Self;
    unsafe fn maxu(a: Self, b: Self) -> Self;
    /// Lane l of the result is v[idx[l] mod W].
    unsafe fn perm(idx: Self, v: Self) -> Self;
    /// Lane l: mask bit l set takes b, else a.
    unsafe fn blend(m: u32, a: Self, b: Self) -> Self;
    unsafe fn xor(a: Self, b: Self) -> Self;
    unsafe fn sub(a: Self, b: Self) -> Self;
    unsafe fn add(a: Self, b: Self) -> Self;
    /// Bit l: a[l] > b[l] unsigned.
    unsafe fn gt(a: Self, b: Self) -> u32;
    /// Lanes of v selected by m, packed to the low lanes; the rest zero.
    unsafe fn compress(m: u32, v: Self) -> Self;
    /// Writes the lanes of v selected by m to p[0..popcount(m)].
    unsafe fn compress_store(p: *mut u32, m: u32, v: Self);
    /// Writes lane l to p[l] for bits l of m.
    unsafe fn store(p: *mut u32, m: u32, v: Self);
    /// [q[W-1], v[0], ..., v[W-2]].
    unsafe fn prev_lane(v: Self, q: Self) -> Self;
    unsafe fn reverse(v: Self) -> Self;
}

/// 16 lanes, AVX-512F.
#[derive(Clone, Copy)]
pub struct Z(__m512i);

impl Lanes for Z {
    const W: usize = 16;
    const LOG: u32 = 4;
    #[inline(always)]
    unsafe fn splat(x: u32) -> Self {
        Z(_mm512_set1_epi32(x as i32))
    }
    #[inline(always)]
    unsafe fn iota() -> Self {
        Z(_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
    }
    #[inline(always)]
    unsafe fn load_pad(p: *const u32, valid: u32) -> Self {
        Z(_mm512_mask_loadu_epi32(_mm512_set1_epi32(-1), valid as u16, p as *const i32))
    }
    #[inline(always)]
    unsafe fn load(p: *const u32) -> Self {
        Z(_mm512_loadu_si512(p as *const _))
    }
    #[inline(always)]
    unsafe fn minu(a: Self, b: Self) -> Self {
        Z(_mm512_min_epu32(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn maxu(a: Self, b: Self) -> Self {
        Z(_mm512_max_epu32(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn perm(idx: Self, v: Self) -> Self {
        Z(_mm512_permutexvar_epi32(idx.0, v.0))
    }
    #[inline(always)]
    unsafe fn blend(m: u32, a: Self, b: Self) -> Self {
        Z(_mm512_mask_mov_epi32(a.0, m as u16, b.0))
    }
    #[inline(always)]
    unsafe fn xor(a: Self, b: Self) -> Self {
        Z(_mm512_xor_si512(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn sub(a: Self, b: Self) -> Self {
        Z(_mm512_sub_epi32(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn add(a: Self, b: Self) -> Self {
        Z(_mm512_add_epi32(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn gt(a: Self, b: Self) -> u32 {
        u32::from(_mm512_cmpgt_epu32_mask(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn compress(m: u32, v: Self) -> Self {
        Z(_mm512_maskz_compress_epi32(m as u16, v.0))
    }
    #[inline(always)]
    unsafe fn compress_store(p: *mut u32, m: u32, v: Self) {
        _mm512_mask_compressstoreu_epi32(p as *mut _, m as u16, v.0)
    }
    #[inline(always)]
    unsafe fn store(p: *mut u32, m: u32, v: Self) {
        _mm512_mask_storeu_epi32(p as *mut _, m as u16, v.0)
    }
    #[inline(always)]
    unsafe fn prev_lane(v: Self, q: Self) -> Self {
        // index 31 selects the second operand's lane 15
        Z(_mm512_permutex2var_epi32(v.0, _mm512_set_epi32(14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 31), q.0))
    }
    #[inline(always)]
    unsafe fn reverse(v: Self) -> Self {
        Z(_mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), v.0))
    }
}

/// 8 lanes, AVX-512F + AVX-512VL (256-bit registers).
#[derive(Clone, Copy)]
pub struct Y(__m256i);

impl Lanes for Y {
    const W: usize = 8;
    const LOG: u32 = 3;
    #[inline(always)]
    unsafe fn splat(x: u32) -> Self {
        Y(_mm256_set1_epi32(x as i32))
    }
    #[inline(always)]
    unsafe fn iota() -> Self {
        Y(_mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0))
    }
    #[inline(always)]
    unsafe fn load_pad(p: *const u32, valid: u32) -> Self {
        Y(_mm256_mask_loadu_epi32(_mm256_set1_epi32(-1), valid as u8, p as *const i32))
    }
    #[inline(always)]
    unsafe fn load(p: *const u32) -> Self {
        Y(_mm256_loadu_si256(p as *const _))
    }
    #[inline(always)]
    unsafe fn minu(a: Self, b: Self) -> Self {
        Y(_mm256_min_epu32(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn maxu(a: Self, b: Self) -> Self {
        Y(_mm256_max_epu32(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn perm(idx: Self, v: Self) -> Self {
        Y(_mm256_permutevar8x32_epi32(v.0, idx.0))
    }
    #[inline(always)]
    unsafe fn blend(m: u32, a: Self, b: Self) -> Self {
        Y(_mm256_mask_mov_epi32(a.0, m as u8, b.0))
    }
    #[inline(always)]
    unsafe fn xor(a: Self, b: Self) -> Self {
        Y(_mm256_xor_si256(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn sub(a: Self, b: Self) -> Self {
        Y(_mm256_sub_epi32(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn add(a: Self, b: Self) -> Self {
        Y(_mm256_add_epi32(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn gt(a: Self, b: Self) -> u32 {
        u32::from(_mm256_cmpgt_epu32_mask(a.0, b.0))
    }
    #[inline(always)]
    unsafe fn compress(m: u32, v: Self) -> Self {
        Y(_mm256_maskz_compress_epi32(m as u8, v.0))
    }
    #[inline(always)]
    unsafe fn compress_store(p: *mut u32, m: u32, v: Self) {
        _mm256_mask_compressstoreu_epi32(p as *mut _, m as u8, v.0)
    }
    #[inline(always)]
    unsafe fn store(p: *mut u32, m: u32, v: Self) {
        _mm256_mask_storeu_epi32(p as *mut _, m as u8, v.0)
    }
    #[inline(always)]
    unsafe fn prev_lane(v: Self, q: Self) -> Self {
        // index 15 selects the second operand's lane 7
        Y(_mm256_permutex2var_epi32(v.0, _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 15), q.0))
    }
    #[inline(always)]
    unsafe fn reverse(v: Self) -> Self {
        Y(_mm256_permutevar8x32_epi32(v.0, _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7)))
    }
}

/// Comparators (i, i ^ m) for m whose highest set bit is `hb`: the lane with bit hb set takes the
/// max.
#[inline(always)]
unsafe fn step<L: Lanes>(v: L, m: u32, hb: u32) -> L {
    let w = L::perm(L::xor(L::iota(), L::splat(m)), v);
    let lo = L::minu(v, w);
    let hi = L::maxu(v, w);
    let mut upper = 0u32;
    let mut i = 0;
    while i < L::W {
        if i & (1 << hb) != 0 {
            upper |= 1 << i;
        }
        i += 1;
    }
    L::blend(upper, lo, hi)
}

/// Phases 1..=p of the in-register sort (p = LOG sorts the register; p = LOG - 1 sorts each
/// half). p is a constant at every call site, so the loops unroll.
#[inline(always)]
unsafe fn sort_reg<L: Lanes>(mut v: L, p: u32) -> L {
    debug_assert!(p <= L::LOG);
    let mut k = 1u32;
    while k <= p {
        v = step(v, (1 << k) - 1, k - 1);
        let mut j = k as i32 - 2;
        while j >= 0 {
            v = step(v, 1 << j, j as u32);
            j -= 1;
        }
        k += 1;
    }
    v
}

/// In-register half-cleaners i ^ W/2, ..., i ^ 1.
#[inline(always)]
unsafe fn clean_reg<L: Lanes>(mut v: L) -> L {
    let mut j = L::LOG as i32 - 1;
    while j >= 0 {
        v = step(v, 1 << j, j as u32);
        j -= 1;
    }
    v
}

/// Sorts R registers (R a power of two) as one sequence of R * W values.
#[inline(always)]
unsafe fn sort_regs<L: Lanes, const R: usize>(r: &mut [L; R]) {
    for x in r.iter_mut() {
        *x = sort_reg(*x, L::LOG);
    }
    let mut blk = 1;
    while blk < R {
        let mut b0 = 0;
        while b0 < R {
            // mirror comparators across the 2 * blk registers
            for i in 0..blk {
                let x = r[b0 + i];
                let y = L::reverse(r[b0 + 2 * blk - 1 - i]);
                r[b0 + i] = L::minu(x, y);
                r[b0 + 2 * blk - 1 - i] = L::reverse(L::maxu(x, y));
            }
            // half-cleaners between registers
            let mut d = blk / 2;
            while d >= 1 {
                for i in 0..2 * blk {
                    if i & d == 0 {
                        let (a, b) = (r[b0 + i], r[b0 + i + d]);
                        r[b0 + i] = L::minu(a, b);
                        r[b0 + i + d] = L::maxu(a, b);
                    }
                }
                d /= 2;
            }
            for i in 0..2 * blk {
                r[b0 + i] = clean_reg(r[b0 + i]);
            }
            b0 += 2 * blk;
        }
        blk *= 2;
    }
}

/// Sessions of a person of W < n <= R * W events (any order, each below 2^31).
#[inline(always)]
unsafe fn person_regs<L: Lanes, const R: usize>(ts: &[u32], out: &mut Out, ends: *mut u32) {
    let n = ts.len();
    let w = L::W;
    let mut r = [L::splat(u32::MAX); R];
    for (i, x) in r.iter_mut().enumerate() {
        if i * w < n {
            let left = n - i * w;
            let valid = if left >= w { (1u32 << w) - 1 } else { (1u32 << left) - 1 };
            // SAFETY: i * w < n, so the pointer is inside ts; the mask selects lanes
            // i * w + l < n only (masked load, no access outside them).
            *x = L::load_pad(ts.as_ptr().add(i * w), valid);
        }
    }
    sort_regs(&mut r);
    let gap = L::splat(GAP);
    let mut cut_all = 0u128;
    for i in 0..R {
        let prev = L::prev_lane(r[i], if i == 0 { r[0] } else { r[i - 1] });
        // sorted, so the unsigned difference is the gap (lane 0 of register 0 is forced below)
        cut_all |= u128::from(L::gt(L::sub(r[i], prev), gap)) << (i * w);
    }
    let valid: u128 = if n == 128 { u128::MAX } else { (1u128 << n) - 1 }; // n <= R * W <= 128
    cut_all = (cut_all | 1) & valid;
    let end_all = ((cut_all >> 1) | (1u128 << (n - 1))) & valid;
    let k = cut_all.count_ones() as usize;
    let base = out.start.len();
    // sessions <= n
    out.start.reserve(n);
    out.events.reserve(n);
    // SAFETY: base + n <= capacity (reserved above).
    let st = out.start.as_mut_ptr().add(base);
    let ev = out.events.as_mut_ptr().add(base);
    let mut spos = [0u32; 128];
    let mut epos = [0u32; 128];
    let (mut ks, mut ke) = (0usize, 0usize);
    let wm: u128 = (1u128 << w) - 1;
    for i in 0..R {
        let cm = ((cut_all >> (i * w)) & wm) as u32;
        let em = ((end_all >> (i * w)) & wm) as u32;
        let idx = L::add(L::iota(), L::splat((i * w) as u32));
        // SAFETY: ks + popcount(cm) <= k <= n for st and <= 128 for spos; likewise ke for epos
        // and for ends (caller: room for base + n entries).
        L::compress_store(st.add(ks), cm, r[i]);
        L::compress_store(spos.as_mut_ptr().add(ks), cm, idx);
        L::compress_store(epos.as_mut_ptr().add(ke), em, idx);
        if !ends.is_null() {
            L::compress_store(ends.add(base + ke), em, r[i]);
        }
        ks += cm.count_ones() as usize;
        ke += em.count_ones() as usize;
    }
    let one = L::splat(1);
    let mut j = 0;
    while j < k {
        let m = if k - j >= w { (1u32 << w) - 1 } else { (1u32 << (k - j)) - 1 };
        // SAFETY: j is a multiple of w below k <= 128, so j + w <= 128 (both arrays are 128 long);
        // the store writes k - j <= n - j lanes inside the reserved room.
        let c = L::add(L::sub(L::load(epos.as_ptr().add(j)), L::load(spos.as_ptr().add(j))), one);
        L::store(ev.add(j), m, c);
        j += w;
    }
    // SAFETY: entries base..base + k were written above.
    out.start.set_len(base + k);
    out.events.set_len(base + k);
}

/// Sessions of a person of 1 <= n <= W events: one register, register-form compress.
#[inline(always)]
unsafe fn person_reg<L: Lanes>(ts: &[u32], out: &mut Out, ends: *mut u32, phases: u32) {
    let n = ts.len();
    let valid = (1u32 << n) - 1; // n <= W <= 16
    // SAFETY: masked load of lanes l < n, all inside ts.
    let a = sort_reg(L::load_pad(ts.as_ptr(), valid), phases);
    let prev = L::prev_lane(a, a);
    let cut = (L::gt(L::sub(a, prev), L::splat(GAP)) | 1) & valid;
    let end = ((cut >> 1) | (1 << (n - 1))) & valid;
    let k = cut.count_ones();
    let keep = (1u32 << k) - 1;
    let starts = L::compress(cut, a);
    let sp = L::compress(cut, L::iota());
    let ep = L::compress(end, L::iota());
    let cnt = L::add(L::sub(ep, sp), L::splat(1));
    let base = out.start.len();
    out.start.reserve(L::W);
    out.events.reserve(L::W);
    // SAFETY: k <= n <= W lanes written after base, inside the reserved room; `ends` has room for
    // base + n entries (caller).
    L::store(out.start.as_mut_ptr().add(base), keep, starts);
    L::store(out.events.as_mut_ptr().add(base), keep, cnt);
    if !ends.is_null() {
        L::store(ends.add(base), keep, L::compress(end, a));
    }
    out.start.set_len(base + k as usize);
    out.events.set_len(base + k as usize);
}

/// Largest person the networks take.
pub const MAX_N: usize = 128;

/// Sessions of one person of 1..=MAX_N events with register type L.
///
/// # Safety
/// L's target features present; 1 <= ts.len() <= MAX_N; each ts < 2^31; `ends` null or writable
/// for base + ts.len() entries (base = out.start.len()).
#[inline(always)]
pub unsafe fn person_net_inl<L: Lanes>(ts: &[u32], out: &mut Out, ends: *mut u32) {
    let n = ts.len();
    let w = L::W;
    debug_assert!(n >= 1 && n <= MAX_N);
    if n <= w / 2 {
        // the upper half is padding: sorting each half sorts the register
        person_reg::<L>(ts, out, ends, L::LOG - 1)
    } else if n <= w {
        person_reg::<L>(ts, out, ends, L::LOG)
    } else if n <= 2 * w {
        person_regs::<L, 2>(ts, out, ends)
    } else if n <= 4 * w {
        person_regs::<L, 4>(ts, out, ends)
    } else if n <= 8 * w {
        person_regs::<L, 8>(ts, out, ends)
    } else {
        // only W = 8 gets here (8 * 16 = MAX_N)
        person_regs::<L, 16>(ts, out, ends)
    }
}

/// 512-bit network for one person (see `person_net_inl`).
///
/// # Safety
/// AVX-512F present, and the conditions of `person_net_inl`.
#[target_feature(enable = "avx512f")]
pub unsafe fn person_net512(ts: &[u32], out: &mut Out, ends: *mut u32) {
    person_net_inl::<Z>(ts, out, ends)
}

/// 256-bit network for one person (see `person_net_inl`).
///
/// # Safety
/// AVX-512F and AVX-512VL present, and the conditions of `person_net_inl`.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn person_net256(ts: &[u32], out: &mut Out, ends: *mut u32) {
    person_net_inl::<Y>(ts, out, ends)
}

/// CPU family and model from CPUID leaf 1 (display values).
pub fn cpu_family_model() -> (u32, u32) {
    // SAFETY: CPUID leaf 1 exists on every x86-64 CPU.
    let r = __cpuid(1);
    let base_family = (r.eax >> 8) & 0xF;
    let family = if base_family == 0xF { base_family + ((r.eax >> 20) & 0xFF) } else { base_family };
    let base_model = (r.eax >> 4) & 0xF;
    let model = if base_family == 6 || base_family == 0xF { base_model | (((r.eax >> 16) & 0xF) << 4) } else { base_model };
    (family, model)
}
