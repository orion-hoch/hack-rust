//! Small persons (at most 32 events): sort the timestamps in AVX-512 registers with a bitonic
//! network (branch-free), then cut sessions with one vector compare and compress stores.
//!
//! Network: comparators i < j = i ^ m take min to i and max to j. Sorting 16 lanes uses the
//! steps m = 1; 3, 1; 7, 2, 1; 15, 4, 2, 1 (each merge phase starts with the mirror comparator,
//! so every comparator is ascending). 32 lanes: sort both halves, compare lane i of the first with
//! lane 15 - i of the second (the mirror across 32), then half-cleaners m = 8, 4, 2, 1 in each.

#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

use crate::{Out, GAP};

/// Lane indices i ^ m.
#[inline(always)]
unsafe fn partner(m: i32) -> __m512i {
    let iota = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    _mm512_xor_si512(iota, _mm512_set1_epi32(m))
}

/// One comparator step: lanes whose bit `hb` is set (the upper of each pair) take the max.
#[inline(always)]
unsafe fn step(v: __m512i, m: i32, hb: u32) -> __m512i {
    let w = _mm512_permutexvar_epi32(partner(m), v);
    let lo = _mm512_min_epu32(v, w);
    let hi = _mm512_max_epu32(v, w);
    let mut upper = 0u16;
    for i in 0..16 {
        if i & (1 << hb) != 0 {
            upper |= 1 << i;
        }
    }
    _mm512_mask_mov_epi32(lo, upper, hi)
}

#[inline(always)]
unsafe fn sort16(mut v: __m512i) -> __m512i {
    v = step(v, 1, 0);
    v = step(v, 3, 1);
    v = step(v, 1, 0);
    v = step(v, 7, 2);
    v = step(v, 2, 1);
    v = step(v, 1, 0);
    v = step(v, 15, 3);
    v = step(v, 4, 2);
    v = step(v, 2, 1);
    v = step(v, 1, 0);
    v
}

/// Sorts lanes 0..8 (lanes 8..16 must already be sorted, e.g. all padding).
#[inline(always)]
unsafe fn sort8(mut v: __m512i) -> __m512i {
    v = step(v, 1, 0);
    v = step(v, 3, 1);
    v = step(v, 1, 0);
    v = step(v, 7, 2);
    v = step(v, 2, 1);
    v = step(v, 1, 0);
    v
}

#[inline(always)]
unsafe fn clean16(mut v: __m512i) -> __m512i {
    v = step(v, 8, 3);
    v = step(v, 4, 2);
    v = step(v, 2, 1);
    v = step(v, 1, 0);
    v
}

#[inline(always)]
unsafe fn reverse(v: __m512i) -> __m512i {
    _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), v)
}

/// Sessions of one person with 1..=32 events (timestamps in any order, below 2^31). Appends to
/// `out.start` / `out.events`. `SHALLOW`: persons of at most 8 events use the 6-step network.
#[target_feature(enable = "avx512f")]
pub unsafe fn person_net(ts: &[u32], out: &mut Out) {
    person_net_inl::<true>(ts, out)
}

#[inline(always)]
pub unsafe fn person_net_inl<const SHALLOW: bool>(ts: &[u32], out: &mut Out) {
    let n = ts.len();
    debug_assert!((1..=32).contains(&n));
    let pad = _mm512_set1_epi32(-1); // u32::MAX sorts last
    let iota = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    let gap = _mm512_set1_epi32(GAP as i32);
    // previous lane (lane 0 takes lane 0; fixed up by the forced cut)
    let down = _mm512_set_epi32(14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0);
    let base = out.start.len();
    out.start.reserve(32);
    out.events.reserve(32);
    let st = out.start.as_mut_ptr().add(base);
    let ev = out.events.as_mut_ptr().add(base);
    let mut spos = [0u32; 32];
    let mut epos = [0u32; 32];
    let k;
    if n <= 16 {
        let valid = ((1u32 << n) - 1) as u16;
        let raw = _mm512_mask_loadu_epi32(pad, valid, ts.as_ptr() as *const i32);
        let a = if SHALLOW && n <= 8 { sort8(raw) } else { sort16(raw) };
        let prev = _mm512_permutexvar_epi32(down, a);
        let big = _mm512_cmpgt_epu32_mask(_mm512_sub_epi32(a, prev), gap);
        let cut = (big | 1) & valid;
        let end = ((cut >> 1) | (1 << (n - 1))) & valid;
        let kk = cut.count_ones();
        let keep = ((1u32 << kk) - 1) as u16;
        // register-form compress, then one masked store each
        let starts = _mm512_maskz_compress_epi32(cut, a);
        let sp = _mm512_maskz_compress_epi32(cut, iota);
        let ep = _mm512_maskz_compress_epi32(end, iota);
        let cnt = _mm512_add_epi32(_mm512_sub_epi32(ep, sp), _mm512_set1_epi32(1));
        _mm512_mask_storeu_epi32(st as *mut _, keep, starts);
        _mm512_mask_storeu_epi32(ev as *mut _, keep, cnt);
        out.start.set_len(base + kk as usize);
        out.events.set_len(base + kk as usize);
        return;
    } else {
        let vb = ((1u32 << (n - 16)) - 1) as u16;
        let a = sort16(_mm512_loadu_si512(ts.as_ptr() as *const _));
        let b = sort16(_mm512_mask_loadu_epi32(pad, vb, ts.as_ptr().add(16) as *const i32));
        // merge: mirror comparator across 32, then half-cleaners
        let rb = reverse(b);
        let lo = _mm512_min_epu32(a, rb);
        let hi = _mm512_max_epu32(a, rb);
        let a = clean16(lo);
        let b = clean16(reverse(hi));
        // previous lanes: b's lane 0 takes a's lane 15
        let pa = _mm512_permutexvar_epi32(down, a);
        let pb = _mm512_permutex2var_epi32(b, _mm512_set_epi32(14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 31), a);
        let ca = _mm512_cmpgt_epu32_mask(_mm512_sub_epi32(a, pa), gap) | 1;
        let cb = _mm512_cmpgt_epu32_mask(_mm512_sub_epi32(b, pb), gap) & vb;
        let cut = u32::from(ca) | (u32::from(cb) << 16);
        let valid = ((1u64 << n) - 1) as u32;
        let end = ((cut >> 1) | (1 << (n - 1))) & valid;
        let ka = ca.count_ones() as usize;
        _mm512_mask_compressstoreu_epi32(st as *mut _, ca, a);
        _mm512_mask_compressstoreu_epi32(st.add(ka) as *mut _, cb, b);
        _mm512_mask_compressstoreu_epi32(spos.as_mut_ptr() as *mut _, ca, iota);
        let iota16 = _mm512_add_epi32(iota, _mm512_set1_epi32(16));
        _mm512_mask_compressstoreu_epi32(spos.as_mut_ptr().add(ka) as *mut _, cb, iota16);
        let ea = end as u16;
        let eb = (end >> 16) as u16;
        let kea = ea.count_ones() as usize;
        _mm512_mask_compressstoreu_epi32(epos.as_mut_ptr() as *mut _, ea, iota);
        _mm512_mask_compressstoreu_epi32(epos.as_mut_ptr().add(kea) as *mut _, eb, iota16);
        k = cut.count_ones() as usize;
    }
    // events per session: end position - start position + 1
    let one = _mm512_set1_epi32(1);
    let c0 = _mm512_add_epi32(_mm512_sub_epi32(_mm512_loadu_si512(epos.as_ptr() as *const _), _mm512_loadu_si512(spos.as_ptr() as *const _)), one);
    _mm512_mask_storeu_epi32(ev as *mut _, ((1u32 << k.min(16)) - 1) as u16, c0);
    if k > 16 {
        let c1 = _mm512_add_epi32(
            _mm512_sub_epi32(_mm512_loadu_si512(epos.as_ptr().add(16) as *const _), _mm512_loadu_si512(spos.as_ptr().add(16) as *const _)),
            one,
        );
        _mm512_mask_storeu_epi32(ev.add(16) as *mut _, ((1u64 << (k - 16)) - 1) as u16, c1);
    }
    out.start.set_len(base + k);
    out.events.set_len(base + k);
}
