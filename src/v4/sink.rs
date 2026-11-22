//! I1 and I3 (RECORDS-V4.md sections 3 and 5; owner: D): the event sink writes I1, one stream
//! per event partition; the click sink writes I3, one stream per user partition. Both are pure
//! scatters: a lane's partition is the top bits of its mixed key (one multiply), its record is
//! packed from the batch and appended with an unchecked push that prefetches the stream's next
//! line. No token table, no probe, no ring. LONG and FOREIGN ids take the wide forms, out of line.
//!
//! Stage API (the driver): `events(cx) -> EventBins` parses every event line; `clicks(cx) ->
//! ClickBins` every click line. I1 chunks are whole B8K blocks taken under the pool participant
//! index (`EventBins::w[t]` was filled by participant t): dedup frees them under the same index.


use parselab::reference::{click_line, event_line, ingest_field};
use parselab::{ClickBatch, Ctx, Dataset, EvBatch, LinkBatch, Locator, Sink, Stream};

use super::keys::{long_hash, mix40};
use super::{Cx, Stage};
use crate::fast::sink::Bins;
use crate::v3::ids::{Key, classify};

/// The 23-bit ingest field (the tests decode it).
#[cfg(test)]
const M23: u64 = (1 << 23) - 1;
const M33: u64 = (1 << 33) - 1;
const SHORT: u64 = 0;
const LONG: u64 = 1;
const FOREIGN: u64 = 2;

/// FOREIGN ids interned once per run to indices f, shared by every stream (one id text, one f).
/// FOREIGN ids: the run's one interner is keys.rs's (shared with links, so a FOREIGN user has
/// one index in every stream); these delegate to it.
pub const FOREIGN_CAP: usize = super::keys::FOREIGN_MAX as usize;

pub fn intern(bytes: &[u8]) -> u32 {
    super::keys::intern(bytes)
}

/// The id text of interned index f.
pub fn foreign_bytes(f: u32) -> Vec<u8> {
    super::keys::bytes(f).to_vec()
}

pub fn foreign_count() -> usize {
    super::keys::count()
}

/// A parsed key as (kind, a, b): SHORT (0, value, 0), LONG (1, a, b), FOREIGN (2, f | 1 << 63, 0).
fn key_of(id: &[u8], prefix: u8) -> (u64, u64, u64) {
    match classify(id, prefix) {
        Key::Short(v) => (SHORT, v, 0),
        Key::Long(a, b) => (LONG, a, b),
        Key::Foreign => (FOREIGN, u64::from(intern(id)) | 1 << 63, 0),
    }
}

/// I1's rare lane: a purchase, or any record with a LONG or FOREIGN id, as 2 or 3 16-byte units
/// (never across chunks). Unit 0 word 0 is the header:
///
///   ingest (bits 0-22) | event kind << 23 | user kind << 25 | surv << 27 | purch << 28 |
///   ts << 29 (31 bits; 0 when dead) | big << 60 (3 units)
///
/// - SHORT event (2 units): u0.w1 = re | payload << 32; u1 = the user key: (mu, 0) SHORT,
///   (a, b) LONG/FOREIGN, (0, 0) dead.
/// - LONG/FOREIGN event, SHORT or dead user, no purchase (2 units): u0.w1 = mu (0 when dead);
///   u1 = the event key (a, b).
/// - LONG/FOREIGN event with a wide user or a purchase (3 units, big): u0.w1 = payload; u1 =
///   the event key; u2 = the user key.
///
/// payload = cat | price << 8 (0 without a purchase).
///
/// An ingest outside the 23-bit field (before window_start - 30 days, or 2^23 - 1 seconds or
/// more after it) sets `RARE_EXT` and leaves the field 0: one more unit after the record's own,
/// [ingest_ms - ingest_base_ms (two's complement), 0], so dedup compares it in milliseconds.
pub const RARE_BIG: u64 = 1 << 60;
/// See `RARE_BIG`.
pub const RARE_EXT: u64 = 1 << 61;

/// I1: the event sink of one parser worker.
pub struct EvSink<'a, 'b> {
    pub ds: &'a Dataset,
    pub ctx: &'b Ctx,
    pub b1: u32,
    /// The plain lane per event partition: one 16-byte unit per record whose ids are SHORT and
    /// which is no purchase (about three records in four).
    pub s: Bins<[u64; 2]>,
    /// The rare lane per event partition: every other record, 32 or 48 bytes (B2K chunks).
    pub o: Bins<[u64; 2]>,
    /// A record v4 cannot hold (the run falls back to v3).
    pub bad: bool,
    /// AVX-512 (F, DQ, BW, VL) detected: the batch is packed 8 lanes at a time.
    v512: bool,
    /// AVX2 (and BMI2) detected: the batch is packed 4 lanes at a time.
    v256: bool,
}

impl<'a, 'b> EvSink<'a, 'b> {
    /// `w`: the pool participant index (the streams' chunks are B8K blocks under it).
    pub fn new(ds: &'a Dataset, ctx: &'b Ctx, b1: u32, w: usize) -> Self {
        #[cfg(target_arch = "x86_64")]
        let v512 = (std::env::var_os("ENGINE_NO512").is_none() && std::arch::is_x86_feature_detected!("avx512f"))
            && std::arch::is_x86_feature_detected!("avx512dq")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl");
        #[cfg(not(target_arch = "x86_64"))]
        let v512 = false;
        #[cfg(target_arch = "x86_64")]
        let v256 = std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("bmi2");
        #[cfg(not(target_arch = "x86_64"))]
        let v256 = false;
        EvSink { ds, ctx, b1, s: Bins::from_blocks(1 << b1, &crate::v3::mem::B8K, w), o: Bins::from_blocks(1 << b1, &crate::v3::mem::B2K, w), bad: false, v512, v256 }
    }

    #[inline(always)]
    fn push(&mut self, p: usize, u: [u64; 2]) {
        // SAFETY: p < 2^b1, the number of streams (a top-bits partition).
        unsafe { self.s.push_fast(p, u) };
    }

    /// A rare record (see `RARE_BIG`) to its event partition.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn rare(&mut self, ek: u64, ea: u64, eb: u64, uk: u64, ua: u64, ub: u64, ing: u64, ts: u64, surv: u64, purch: u64, buy: u64) {
        let uk = if surv != 0 { uk } else { 0 };
        let h = ing | ek << 23 | uk << 25 | surv << 27 | purch << 28 | (ts * surv) << 29;
        let user = if surv == 0 {
            [0, 0]
        } else if uk == SHORT {
            [mix40(ua), 0]
        } else {
            [ua, ub]
        };
        let o = &mut self.o;
        // SAFETY (each push): p < 2^b1, the number of streams; a B2K chunk holds 128 units.
        if ek == SHORT {
            let me = mix40(ea);
            let p = (me >> (40 - self.b1)) as usize;
            unsafe { o.push_all(p, [[h, (me & ((1 << (40 - self.b1)) - 1)) | buy << 32], user]) };
            return;
        }
        let p = (long_hash(ea, eb) >> (64 - self.b1)) as usize;
        if uk == SHORT && purch == 0 {
            unsafe { o.push_all(p, [[h, user[0]], [ea, eb]]) };
        } else {
            unsafe { o.push_all(p, [[h | RARE_BIG, buy], [ea, eb], user]) };
        }
    }
}

impl<'a> Sink for EvSink<'a, '_> {
    #[inline]
    fn events(&mut self, b: &EvBatch) {
        #[cfg(target_arch = "x86_64")]
        if self.v512 {
            // SAFETY: the features were detected at construction.
            return unsafe { self.scatter512(b) };
        }
        #[cfg(target_arch = "x86_64")]
        if self.v256 {
            // SAFETY: as above.
            return unsafe { self.scatter256(b) };
        }
        self.scatter(b);
    }

    fn foreign_event(&mut self, at: Locator) {
        self.foreign(at);
    }

    fn clicks(&mut self, _: &ClickBatch) {
        self.bad = true;
    }
    fn foreign_click(&mut self, _: Locator) {
        self.bad = true;
    }
    fn links(&mut self, _: &LinkBatch) {
        self.bad = true;
    }
    fn foreign_link(&mut self, _: Locator) {
        self.bad = true;
    }
}

/// a * b per 64-bit lane as two 256-bit multiplies. A 512-bit vpmullq is a HEAVY 512-bit
/// instruction: on Skylake-X and Cascade Lake it asks the core for its lowest clock licence
/// (the scorer measured a build whose parser used 512-bit multiplies 0.037 s slower at equal
/// output), and the parser's own batches avoid it for that reason. Written with asm so the
/// compiler cannot merge the halves back into one 512-bit multiply.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512vl")]
#[inline]
unsafe fn mullo64_halves(a: std::arch::x86_64::__m512i, b: std::arch::x86_64::__m512i) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::*;
    unsafe {
        let (al, ah) = (_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64::<1>(a));
        let (bl, bh) = (_mm512_castsi512_si256(b), _mm512_extracti64x4_epi64::<1>(b));
        let (rl, rh): (__m256i, __m256i);
        std::arch::asm!("vpmullq {r}, {x}, {y}", r = lateout(ymm_reg) rl, x = in(ymm_reg) al, y = in(ymm_reg) bl, options(pure, nomem, nostack, preserves_flags));
        std::arch::asm!("vpmullq {r}, {x}, {y}", r = lateout(ymm_reg) rh, x = in(ymm_reg) ah, y = in(ymm_reg) bh, options(pure, nomem, nostack, preserves_flags));
        _mm512_inserti64x4::<1>(_mm512_castsi256_si512(rl), rh)
    }
}

/// Bit 2k set for each bit k of the index: `store_lanes`' mask for the AVX2 sink (the AVX-512
/// sinks take it from one pdep, which is microcode on AMD before Zen 3).
#[cfg(target_arch = "x86_64")]
const SPREAD: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let mut k = 0;
        while k < 8 {
            if i >> k & 1 != 0 {
                t[i] |= 1 << (2 * k);
            }
            k += 1;
        }
        i += 1;
    }
    t
};

/// Appends lane k's 16-byte unit to the bin whose front is at byte offset `part[k]` of the
/// fronts, for every lane k in `lanes`: the front loaded, one 16-byte store, the front advanced
/// and its next line asked for. In assembly: 12 instructions a lane (the compiler's loop
/// spends 14). The lanes are taken from a mask with bit 2k for lane k, so the count of
/// trailing zeros scales to both arrays (times 4 in `part`, times 8 in `units`). A lane whose
/// chunk is at its end leaves the loop, the next chunk is opened, and the loop takes it again.
///
/// # Safety
/// `part[k] / 16` is below the number of bins for every lane in `lanes`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,bmi1,bmi2")]
#[inline]
unsafe fn store_lanes(bins: &mut Bins<[u64; 2]>, mut m: u32, part: &[u64; 8], units: &[[u64; 2]; 8]) {
    let fs = bins.fronts();
    while m != 0 {
        // SAFETY: the caller's bound for each front; a front's pointer below its chunk's end has
        // room for one unit.
        unsafe {
            core::arch::asm!(
                "2:",
                "tzcnt {k:e}, {m:e}",
                "mov {o}, qword ptr [{part} + {k}*4]",
                "vmovdqu xmm0, xmmword ptr [{units} + {k}*8]",
                "mov {p}, qword ptr [{fs} + {o}]",
                "cmp {p}, qword ptr [{fs} + {o} + 8]",
                "je 3f",
                "vmovdqu xmmword ptr [{p}], xmm0",
                "add {p}, 16",
                "mov qword ptr [{fs} + {o}], {p}",
                "prefetcht0 byte ptr [{p} + 48]",
                "blsr {m:e}, {m:e}",
                "jne 2b",
                "3:",
                m = inout(reg) m,
                part = in(reg) part.as_ptr(),
                units = in(reg) units.as_ptr(),
                fs = in(reg) fs,
                k = out(reg) _,
                o = out(reg) _,
                p = out(reg) _,
                out("xmm0") _,
                options(nostack),
            );
        }
        if m != 0 {
            // The lowest lane met its chunk's end.
            let k = (m.trailing_zeros() / 2) as usize;
            bins.next_chunk(part[k] as usize >> 4);
        }
    }
}

/// As `store_lanes` for records of two units: lane k's `r0[k]` then `r1[k]` go to the bin at
/// `part[k]`, both inside one chunk (a chunk without room for 32 bytes is closed short first).
/// 13 instructions a lane (the compiler's loop spends 17).
///
/// # Safety
/// As `store_lanes`; a chunk of `bins` holds at least two units.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,bmi1,bmi2")]
#[inline]
unsafe fn store_lanes2(bins: &mut Bins<[u64; 2]>, lanes: u8, part: &[u64; 8], r0: &[[u64; 2]; 8], r1: &[[u64; 2]; 8]) {
    let fs = bins.fronts();
    let mut m = core::arch::x86_64::_pdep_u32(u32::from(lanes), 0x5555);
    while m != 0 {
        // SAFETY: the caller's bound for each front; the two stores are below the chunk's end
        // (the pointer + 32 is not above it).
        unsafe {
            core::arch::asm!(
                "2:",
                "tzcnt {k:e}, {m:e}",
                "mov {o}, qword ptr [{part} + {k}*4]",
                "vmovdqu xmm0, xmmword ptr [{r0} + {k}*8]",
                "vmovdqu xmm1, xmmword ptr [{r1} + {k}*8]",
                "mov {p}, qword ptr [{fs} + {o}]",
                "lea {t}, [{p} + 32]",
                "cmp {t}, qword ptr [{fs} + {o} + 8]",
                "ja 3f",
                "vmovdqu xmmword ptr [{p}], xmm0",
                "vmovdqu xmmword ptr [{p} + 16], xmm1",
                "mov qword ptr [{fs} + {o}], {t}",
                "prefetcht0 byte ptr [{t} + 64]",
                "blsr {m:e}, {m:e}",
                "jne 2b",
                "3:",
                m = inout(reg) m,
                part = in(reg) part.as_ptr(),
                r0 = in(reg) r0.as_ptr(),
                r1 = in(reg) r1.as_ptr(),
                fs = in(reg) fs,
                k = out(reg) _,
                o = out(reg) _,
                p = out(reg) _,
                t = out(reg) _,
                out("xmm0") _,
                out("xmm1") _,
                options(nostack),
            );
        }
        if m != 0 {
            // The lowest lane's chunk has no room for the record.
            let k = (m.trailing_zeros() / 2) as usize;
            bins.next_chunk_short(part[k] as usize >> 4);
        }
    }
}

impl<'a> EvSink<'a, '_> {
    /// One batch with AVX-512. Every lane's plain unit, and every lane's rare record (header,
    /// second word, event key, user key, partition, form), is built in zmm registers for the 8
    /// lanes at once; then each plain lane is a load of its front, a 16-byte store and the front
    /// advanced, and each rare lane the same with its 2 or 3 units. Lanes are taken by mask (the
    /// order of records inside a stream is free).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,bmi1,bmi2")]
    #[inline(never)]
    pub unsafe fn scatter512(&mut self, b: &EvBatch) {
        use core::arch::x86_64::*;
        let n = (b.n as usize).min(8);
        let all = ((1u32 << n) - 1) as u8;
        let live: __mmask8 = b.surv;
        let purch: __mmask8 = b.purch & live;
        let uwide: __mmask8 = b.user_long & live;
        let elong: __mmask8 = b.ev_long;
        let rare = (elong | uwide | purch) & all;
        let common = all & !rare;
        let sh = 40 - self.b1;
        // Lane arrays, written by the vector stores below before any lane reads them (left
        // uninitialised: zeroing them would cost more instructions than a lane's whole store).
        // Plain units and their fronts' byte offsets (partition * 16); rare records' units 0, 1,
        // 2, their fronts' byte offsets, and the lanes with 3 units.
        use std::mem::MaybeUninit;
        let mut units = MaybeUninit::<[[u64; 2]; 8]>::uninit();
        let mut part = MaybeUninit::<[u64; 8]>::uninit();
        let mut r0 = MaybeUninit::<[[u64; 2]; 8]>::uninit();
        let mut r1 = MaybeUninit::<[[u64; 2]; 8]>::uninit();
        let mut r2 = MaybeUninit::<[[u64; 2]; 8]>::uninit();
        let mut rpart = MaybeUninit::<[u64; 8]>::uninit();
        let mut big: __mmask8 = 0;
        unsafe {
            let c = _mm512_set1_epi64(super::keys::MIX_C as i64);
            let m40 = _mm512_set1_epi64(super::keys::M40 as i64);
            let x = _mm512_and_si512(mullo64_halves(_mm512_loadu_si512(b.ev_a.as_ptr() as *const _), c), m40);
            let me = _mm512_xor_si512(x, _mm512_srli_epi64::<20>(x));
            let ua = _mm512_loadu_si512(b.user_a.as_ptr() as *const _);
            let y = _mm512_and_si512(mullo64_halves(ua, c), m40);
            let mu = _mm512_xor_si512(y, _mm512_srli_epi64::<20>(y));
            let ing = _mm512_cvtepu32_epi64(_mm256_loadu_si256(b.ingest.as_ptr() as *const __m256i));
            let ts = _mm512_maskz_mov_epi64(live, _mm512_cvtepu32_epi64(_mm256_loadu_si256(b.ts.as_ptr() as *const __m256i)));
            let cnt = _mm_cvtsi32_si128(sh as i32);
            let ps = _mm512_srl_epi64(me, cnt);
            _mm512_storeu_si512(part.as_mut_ptr() as *mut _, _mm512_slli_epi64::<4>(ps));
            let re = _mm512_and_si512(me, _mm512_set1_epi64(((1u64 << sh) - 1) as i64));
            // survivor bits: high user bits, surv
            let hi = _mm512_or_si512(_mm512_slli_epi64::<55>(_mm512_srli_epi64::<33>(mu)), _mm512_set1_epi64(1 << 62));
            let w0 = _mm512_or_si512(re, _mm512_slli_epi64::<32>(ing));
            let w0 = _mm512_mask_or_epi64(w0, live, w0, hi);
            let w1 = _mm512_or_si512(ts, _mm512_maskz_slli_epi64::<31>(live, _mm512_and_si512(mu, _mm512_set1_epi64(M33 as i64))));
            // [w0, w1] pairs in lane order
            let i0 = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
            let i1 = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
            _mm512_storeu_si512(units.as_mut_ptr() as *mut _, _mm512_permutex2var_epi64(w0, i0, w1));
            _mm512_storeu_si512((units.as_mut_ptr() as *mut [u64; 2]).add(4) as *mut _, _mm512_permutex2var_epi64(w0, i1, w1));
            if rare != 0 {
                // RARE_BIG's three forms: SHORT event (A); LONG event with a SHORT or dead user and
                // no purchase (B); LONG event with a wide user or a purchase (C, 3 units).
                big = elong & (uwide | purch);
                let fa = !elong;
                let price = _mm512_cvtepu32_epi64(_mm256_loadu_si256(b.price.as_ptr() as *const __m256i));
                // The parser accepts 1 to 10,000,000; anything at 2^24 or above would not fit.
                self.bad |= _mm512_mask_cmpge_epu64_mask(purch & rare, price, _mm512_set1_epi64(1 << 24)) != 0;
                let cat = _mm512_cvtepu8_epi64(_mm_loadl_epi64(b.cat.as_ptr() as *const __m128i));
                let buy = _mm512_maskz_or_epi64(purch, cat, _mm512_slli_epi64::<8>(price));
                let one = |m: __mmask8, bit: u32| _mm512_maskz_mov_epi64(m, _mm512_set1_epi64(1i64 << bit));
                let h = _mm512_or_si512(ing, _mm512_slli_epi64::<29>(ts));
                let h = _mm512_or_si512(h, _mm512_or_si512(one(elong, 23), one(uwide, 25)));
                let h = _mm512_or_si512(h, _mm512_or_si512(one(live, 27), one(purch, 28)));
                let h = _mm512_or_si512(h, one(big, 60));
                // The user key: (mu, 0) SHORT, (a, b) wide, (0, 0) dead.
                let uxa = _mm512_maskz_mov_epi64(live, _mm512_mask_mov_epi64(mu, uwide, ua));
                let uxb = _mm512_maskz_loadu_epi64(uwide, b.user_b.as_ptr() as *const _);
                // Word 1: A re | payload << 32; B mu (0 dead); C payload.
                let rw1 = _mm512_mask_mov_epi64(_mm512_mask_mov_epi64(uxa, big, buy), fa, _mm512_or_si512(re, _mm512_slli_epi64::<32>(buy)));
                // Unit 1: A the user key; B and C the event key.
                let ea = _mm512_loadu_si512(b.ev_a.as_ptr() as *const _);
                let eb = _mm512_loadu_si512(b.ev_b.as_ptr() as *const _);
                let s1a = _mm512_mask_mov_epi64(ea, fa, uxa);
                let s1b = _mm512_mask_mov_epi64(eb, fa, uxb);
                // Partition: SHORT event the top bits of mix40, LONG the top bits of long_hash.
                let lh = mullo64_halves(_mm512_xor_si512(ea, _mm512_rol_epi64::<29>(eb)), _mm512_set1_epi64(super::keys::LONG_C as i64));
                let pl = _mm512_srl_epi64(lh, _mm_cvtsi32_si128(64 - self.b1 as i32));
                _mm512_storeu_si512(rpart.as_mut_ptr() as *mut _, _mm512_slli_epi64::<4>(_mm512_mask_mov_epi64(ps, elong, pl)));
                _mm512_storeu_si512(r0.as_mut_ptr() as *mut _, _mm512_permutex2var_epi64(h, i0, rw1));
                _mm512_storeu_si512((r0.as_mut_ptr() as *mut [u64; 2]).add(4) as *mut _, _mm512_permutex2var_epi64(h, i1, rw1));
                _mm512_storeu_si512(r1.as_mut_ptr() as *mut _, _mm512_permutex2var_epi64(s1a, i0, s1b));
                _mm512_storeu_si512((r1.as_mut_ptr() as *mut [u64; 2]).add(4) as *mut _, _mm512_permutex2var_epi64(s1a, i1, s1b));
                _mm512_storeu_si512(r2.as_mut_ptr() as *mut _, _mm512_permutex2var_epi64(uxa, i0, uxb));
                _mm512_storeu_si512((r2.as_mut_ptr() as *mut [u64; 2]).add(4) as *mut _, _mm512_permutex2var_epi64(uxa, i1, uxb));
            }
        }
        // SAFETY: every lane of the arrays read below was written above (units and part always;
        // the rare arrays when a lane is rare, and only rare lanes read them).
        let (units, part) = unsafe { (units.assume_init_ref(), part.assume_init_ref()) };
        // The plain lanes: front, store, advance (the chunk-end test is the only branch).
        // SAFETY: part[k] / 16 < 2^b1, the number of bins, for every lane.
        unsafe { store_lanes(&mut self.s, _pdep_u32(u32::from(common), 0x5555), part, units) };
        if rare == 0 {
            return;
        }
        // SAFETY: as above.
        let (r0, r1, r2, rpart) = unsafe { (r0.assume_init_ref(), r1.assume_init_ref(), r2.assume_init_ref(), rpart.assume_init_ref()) };
        // The rare lanes: 2 units (forms A and B), then 3 (form C), inside one chunk each.
        // SAFETY: as above, for the rare bins (a B2K chunk holds 128 units).
        unsafe { store_lanes2(&mut self.o, rare & !big, rpart, r0, r1) };
        let fo = self.o.fronts();
        let mut m = u32::from(big);
        while m != 0 {
            let k = m.trailing_zeros() as usize;
            // SAFETY: as above.
            unsafe {
                let f = (fo as *mut u8).add(rpart[k] as usize) as *mut [*mut [u64; 2]; 2];
                let mut p = (*f)[0];
                if ((*f)[1] as usize).wrapping_sub(p as usize) < 48 {
                    self.o.next_chunk_short(rpart[k] as usize >> 4);
                    p = (*f)[0];
                }
                p.write(r0[k]);
                p.add(1).write(r1[k]);
                p.add(2).write(r2[k]);
                (*f)[0] = p.add(3);
            }
            m &= m - 1;
        }
    }

    /// One batch with AVX2: as `scatter512`, 4 lanes a vector (the 40-bit multiply from three
    /// 32-bit ones: AVX2 has no 64-bit multiply).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2")]
    #[inline(never)]
    pub unsafe fn scatter256(&mut self, b: &EvBatch) {
        use core::arch::x86_64::*;
        let n = (b.n as usize).min(8);
        let all = ((1u32 << n) - 1) as u8;
        let purch = b.purch & b.surv;
        let wide = b.ev_long | (b.user_long & b.surv);
        let sh = 40 - self.b1;
        let mut units = [[0u64; 2]; 8];
        let mut part = [0u64; 8];
        unsafe {
            let clo = _mm256_set1_epi64x((super::keys::MIX_C & 0xFFFF_FFFF) as i64);
            let chi = _mm256_set1_epi64x((super::keys::MIX_C >> 32) as i64);
            let m40 = _mm256_set1_epi64x(super::keys::M40 as i64);
            let mix = |x: __m256i| {
                // x * MIX_C mod 2^64, then mod 2^40 and the xor-shift
                let lo = _mm256_mul_epu32(x, clo);
                let mid = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64::<32>(x), clo), _mm256_mul_epu32(x, chi));
                let y = _mm256_and_si256(_mm256_add_epi64(lo, _mm256_slli_epi64::<32>(mid)), m40);
                _mm256_xor_si256(y, _mm256_srli_epi64::<20>(y))
            };
            let bits = _mm256_setr_epi64x(1, 2, 4, 8);
            let lanes = |m: u8| _mm256_cmpeq_epi64(_mm256_and_si256(_mm256_set1_epi64x(i64::from(m)), bits), bits);
            let rmask = _mm256_set1_epi64x(((1u64 << sh) - 1) as i64);
            let cnt = _mm_cvtsi32_si128(sh as i32);
            let m33 = _mm256_set1_epi64x(M33 as i64);
            let s62 = _mm256_set1_epi64x(1 << 62);
            let s63 = _mm256_set1_epi64x(i64::MIN);
            for h in 0..2 {
                let me = mix(_mm256_loadu_si256(b.ev_a.as_ptr().add(4 * h) as *const __m256i));
                let mu = mix(_mm256_loadu_si256(b.user_a.as_ptr().add(4 * h) as *const __m256i));
                let ing = _mm256_cvtepu32_epi64(_mm_loadu_si128(b.ingest.as_ptr().add(4 * h) as *const __m128i));
                let ts = _mm256_cvtepu32_epi64(_mm_loadu_si128(b.ts.as_ptr().add(4 * h) as *const __m128i));
                // The bin's byte offset in the fronts (partition * 16), as `store_lanes` takes it.
                _mm256_storeu_si256(part.as_mut_ptr().add(4 * h) as *mut __m256i, _mm256_slli_epi64::<4>(_mm256_srl_epi64(me, cnt)));
                let live = lanes(b.surv >> (4 * h));
                let pu = lanes(purch >> (4 * h));
                let hi = _mm256_or_si256(_mm256_or_si256(_mm256_slli_epi64::<55>(_mm256_srli_epi64::<33>(mu)), s62), _mm256_and_si256(pu, s63));
                let w0 = _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(me, rmask), _mm256_slli_epi64::<32>(ing)), _mm256_and_si256(live, hi));
                let w1 = _mm256_and_si256(live, _mm256_or_si256(ts, _mm256_slli_epi64::<31>(_mm256_and_si256(mu, m33))));
                // [w0, w1] pairs in lane order
                let a = _mm256_unpacklo_epi64(w0, w1); // lanes 0, 2
                let c = _mm256_unpackhi_epi64(w0, w1); // lanes 1, 3
                let u = units.as_mut_ptr().add(4 * h) as *mut __m256i;
                _mm256_storeu_si256(u, _mm256_permute2x128_si256::<0x20>(a, c));
                _mm256_storeu_si256(u.add(1), _mm256_permute2x128_si256::<0x31>(a, c));
            }
        }
        // The plain lanes by mask (no branch per lane on which lanes are rare), then the rare
        // lanes one by one (the order of records inside a stream is free).
        let rare = (wide | purch) & all;
        // SAFETY: part[k] / 16 < 2^b1, the number of bins, for every lane.
        unsafe { store_lanes(&mut self.s, u32::from(SPREAD[usize::from(all & !rare)]), &part, &units) };
        let mut m = u32::from(rare);
        while m != 0 {
            self.rare_lane(b, m.trailing_zeros() as usize);
            m &= m - 1;
        }
    }

    /// A purchase lane or a wide lane (about one lane in four: a purchase, a LONG event or a
    /// surviving LONG user) as one rare record.
    #[inline(always)]
    fn rare_lane(&mut self, b: &EvBatch, k: usize) {
        let surv = u64::from(b.surv >> k & 1);
        let purch = u64::from(b.purch >> k & 1) & surv;
        // The parser accepts 1 to 10,000,000; anything at 2^24 or above would not fit.
        self.bad |= purch != 0 && b.price[k] >> 24 != 0;
        let buy = (u64::from(b.cat[k]) | u64::from(b.price[k]) << 8) * purch;
        let ek = u64::from(b.ev_long >> k & 1);
        let uk = u64::from(b.user_long >> k & 1);
        self.rare(ek, b.ev_a[k], b.ev_b[k], uk, b.user_a[k], b.user_b[k], u64::from(b.ingest[k]), u64::from(b.ts[k]), surv, purch, buy);
    }

    /// One batch: every lane's unit appended to its partition (out of line: one call a batch).
    #[inline(never)]
    pub fn scatter(&mut self, b: &EvBatch) {
        let n = (b.n as usize).min(8);
        let sh = 40 - self.b1;
        let rmask = (1u64 << sh) - 1;
        // rare lanes: a purchase, a LONG event key, or a surviving LONG user
        let rare = b.ev_long | ((b.user_long | b.purch) & b.surv);
        for k in 0..n {
            if rare >> k & 1 != 0 {
                self.rare_lane(b, k);
                continue;
            }
            let surv = u64::from(b.surv >> k & 1);
            let ing = u64::from(b.ingest[k]);
            let me = mix40(b.ev_a[k]);
            let mu = mix40(b.user_a[k]);
            let live = surv.wrapping_neg(); // all ones for a survivor
            let w0 = (me & rmask) | ing << 32 | (live & ((mu >> 33) << 55 | 1 << 62));
            let w1 = live & (u64::from(b.ts[k]) | (mu & M33) << 31);
            self.push((me >> sh) as usize, [w0, w1]);
        }
    }

    #[cold]
    #[inline(never)]
    fn foreign(&mut self, at: Locator) {
        let line = self.ds.line(Stream::Events, at);
        let mut fields: Vec<&'a [u8]> = Vec::with_capacity(8);
        let Some(e) = event_line(self.ctx, line, &mut fields) else {
            self.bad = true;
            return;
        };
        // No valid ingest: the copy never wins (smallest ingest wins) and never survives.
        let Some(ing_ms) = e.ingest else { return };
        // An ingest outside the 23-bit field: a rare record with the ingest in milliseconds.
        let ext = match ingest_field(self.ctx, e.ingest) {
            Some(_) => None,
            None => Some(ing_ms - self.ctx.ingest_base_s * 1000),
        };
        let ing = ingest_field(self.ctx, e.ingest).unwrap_or(0);
        let (ek, ea, eb) = key_of(e.event_id, b'e');
        let surv = u64::from(e.survives);
        let purch = u64::from(e.survives && e.purchase);
        let (uk, ua, ub) = if e.survives { key_of(e.user_id, b'u') } else { (SHORT, 0, 0) };
        let ts = if e.survives { (e.ts.unwrap() - self.ctx.window_start_ms) as u64 } else { 0 };
        let buy = if purch != 0 {
            let price = e.price.unwrap_or(0);
            self.bad |= price >> 24 != 0;
            u64::from(self.ctx.products.code(e.product_id)) | u64::from(price) << 8
        } else {
            0
        };
        let ing = u64::from(ing);
        if let Some(x) = ext {
            return self.rare_ext(ek, ea, eb, uk, ua, ub, x, ts, surv, purch, buy);
        }
        if ek == SHORT && uk == SHORT && purch == 0 {
            let me = mix40(ea);
            let p = (me >> (40 - self.b1)) as usize;
            let re = me & ((1 << (40 - self.b1)) - 1);
            if surv != 0 {
                let mu = mix40(ua);
                self.push(p, [re | ing << 32 | (mu >> 33) << 55 | 1 << 62, ts | (mu & M33) << 31]);
            } else {
                self.push(p, [re | ing << 32, 0]);
            }
        } else {
            self.rare(ek, ea, eb, uk, ua, ub, ing, ts, surv, purch, buy);
        }
    }

    /// A rare record with an ingest outside the field: `x` = ingest_ms - ingest_base_ms, in the
    /// unit after the record's own (`RARE_EXT`).
    #[cold]
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn rare_ext(&mut self, ek: u64, ea: u64, eb: u64, uk: u64, ua: u64, ub: u64, x: i64, ts: u64, surv: u64, purch: u64, buy: u64) {
        let uk = if surv != 0 { uk } else { 0 };
        let h = RARE_EXT | ek << 23 | uk << 25 | surv << 27 | purch << 28 | (ts * surv) << 29;
        let user = if surv == 0 {
            [0, 0]
        } else if uk == SHORT {
            [mix40(ua), 0]
        } else {
            [ua, ub]
        };
        let xu = [x as u64, 0];
        let o = &mut self.o;
        // SAFETY (each push): as in `rare`.
        if ek == SHORT {
            let me = mix40(ea);
            let p = (me >> (40 - self.b1)) as usize;
            unsafe { o.push_all(p, [[h, (me & ((1 << (40 - self.b1)) - 1)) | buy << 32], user, xu]) };
            return;
        }
        let p = (long_hash(ea, eb) >> (64 - self.b1)) as usize;
        if uk == SHORT && purch == 0 {
            unsafe { o.push_all(p, [[h, user[0]], [ea, eb], xu]) };
        } else {
            unsafe { o.push_all(p, [[h | RARE_BIG, buy], [ea, eb], user, xu]) };
        }
    }
}

/// I3: the click sink of one parser worker.
pub struct ClickSink<'a, 'b> {
    pub ds: &'a Dataset,
    pub ctx: &'b Ctx,
    pub b2: u32,
    /// SHORT users: one stream of 16-byte records per user partition.
    pub s: Bins<[u64; 2]>,
    /// LONG/FOREIGN users: one stream of 32-byte records per user partition.
    pub wide: Bins<[u64; 4]>,
    pub bad: bool,
    /// AVX-512 (F, DQ, BW, VL) detected: the batch is packed 8 lanes at a time.
    v512: bool,
    /// AVX2 (and BMI1, BMI2) detected: the batch is packed 4 lanes at a time.
    v256: bool,
}

impl<'a, 'b> ClickSink<'a, 'b> {
    pub fn new(ds: &'a Dataset, ctx: &'b Ctx, b2: u32, w: usize) -> Self {
        // 2 KB chunks: every worker holds an open chunk per user partition and form, mostly part
        // filled (8 KB chunks held 267 MB for 202 MB of short clicks and 135 MB for 62 MB of wide
        // ones at 60M); P1's person blocks are fed by I2's B8K frees, not by these.
        use crate::v3::mem::B2K;
        #[cfg(target_arch = "x86_64")]
        let v512 = (std::env::var_os("ENGINE_NO512").is_none() && std::arch::is_x86_feature_detected!("avx512f"))
            && std::arch::is_x86_feature_detected!("avx512dq")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vl");
        #[cfg(not(target_arch = "x86_64"))]
        let v512 = false;
        #[cfg(target_arch = "x86_64")]
        let v256 = std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("bmi1") && std::arch::is_x86_feature_detected!("bmi2");
        #[cfg(not(target_arch = "x86_64"))]
        let v256 = false;
        ClickSink { ds, ctx, b2, s: Bins::from_blocks(1 << b2, &B2K, w), wide: Bins::from_blocks(1 << b2, &B2K, w), bad: false, v512, v256 }
    }

    /// One batch with AVX2, as `clicks512`, 4 lanes a vector (the 40-bit multiply from three
    /// 32-bit ones, as `EvSink::scatter256`): the path of a host without AVX-512.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2")]
    #[inline(never)]
    unsafe fn clicks256(&mut self, b: &ClickBatch) {
        use core::arch::x86_64::*;
        let n = (b.n as usize).min(8);
        let all = ((1u32 << n) - 1) as u8;
        let wide = b.user_long & all;
        let short = all & !wide;
        let sh = 40 - self.b2;
        let mut units = [[0u64; 2]; 8];
        let mut part = [0u64; 8];
        // Bit k: lane k's campaign or locator is past its field.
        let mut over = 0u32;
        unsafe {
            let clo = _mm256_set1_epi64x((super::keys::MIX_C & 0xFFFF_FFFF) as i64);
            let chi = _mm256_set1_epi64x((super::keys::MIX_C >> 32) as i64);
            let m40 = _mm256_set1_epi64x(super::keys::M40 as i64);
            let mix = |x: __m256i| {
                // x * MIX_C mod 2^64, then mod 2^40 and the xor-shift
                let lo = _mm256_mul_epu32(x, clo);
                let mid = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64::<32>(x), clo), _mm256_mul_epu32(x, chi));
                let y = _mm256_and_si256(_mm256_add_epi64(lo, _mm256_slli_epi64::<32>(mid)), m40);
                _mm256_xor_si256(y, _mm256_srli_epi64::<20>(y))
            };
            let rmask = _mm256_set1_epi64x(((1u64 << sh) - 1) as i64);
            let cnt = _mm_cvtsi32_si128(sh as i32);
            let zero = _mm256_setzero_si256();
            for h in 0..2 {
                let mu = mix(_mm256_loadu_si256(b.user_a.as_ptr().add(4 * h) as *const __m256i));
                // The bin's byte offset in the fronts (partition * 16).
                _mm256_storeu_si256(part.as_mut_ptr().add(4 * h) as *mut __m256i, _mm256_slli_epi64::<4>(_mm256_srl_epi64(mu, cnt)));
                let ts = _mm256_cvtepu32_epi64(_mm_loadu_si128(b.ts.as_ptr().add(4 * h) as *const __m128i));
                let camp = _mm256_cvtepu16_epi64(_mm_loadl_epi64(b.campaign.as_ptr().add(4 * h) as *const __m128i));
                let loc = _mm256_loadu_si256(b.loc.as_ptr().add(4 * h) as *const __m256i);
                let ov = _mm256_or_si256(_mm256_srli_epi64::<12>(camp), _mm256_srli_epi64::<52>(loc));
                let is0 = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(ov, zero))) as u32;
                over |= (!is0 & 0xF) << (4 * h);
                let w0 = _mm256_or_si256(_mm256_slli_epi64::<32>(_mm256_and_si256(mu, rmask)), ts);
                let w1 = _mm256_or_si256(_mm256_slli_epi64::<52>(camp), loc);
                // [w0, w1] pairs in lane order
                let a = _mm256_unpacklo_epi64(w0, w1); // lanes 0, 2
                let c = _mm256_unpackhi_epi64(w0, w1); // lanes 1, 3
                let u = units.as_mut_ptr().add(4 * h) as *mut __m256i;
                _mm256_storeu_si256(u, _mm256_permute2x128_si256::<0x20>(a, c));
                _mm256_storeu_si256(u.add(1), _mm256_permute2x128_si256::<0x31>(a, c));
            }
        }
        // 4,096 campaigns or more, or a locator past 52 bits (lanes 0..n): v4 cannot hold it.
        self.bad |= over & u32::from(all) != 0;
        // SAFETY: part[k] / 16 < 2^b2, the number of bins, for every lane.
        unsafe { store_lanes(&mut self.s, u32::from(SPREAD[usize::from(short)]), &part, &units) };
        let mut m = u32::from(wide);
        while m != 0 {
            let k = m.trailing_zeros() as usize;
            self.wide(LONG, b.user_a[k], b.user_b[k], u64::from(b.ts[k]), units[k][1]);
            m &= m - 1;
        }
    }

    /// One batch with AVX-512, as `EvSink::scatter512`: every lane's 16-byte record (the user
    /// key mixed, its partition, ru << 32 | tc, campaign << 52 | locator) is built in zmm
    /// registers for the 8 lanes at once; then each SHORT lane is its front loaded, one
    /// 16-byte store and the front advanced. LONG lanes take `wide`, one by one.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,bmi1,bmi2")]
    #[inline(never)]
    unsafe fn clicks512(&mut self, b: &ClickBatch) {
        use core::arch::x86_64::*;
        use std::mem::MaybeUninit;
        let n = (b.n as usize).min(8);
        let all = ((1u32 << n) - 1) as u8;
        let wide: __mmask8 = b.user_long & all;
        let short = all & !wide;
        let sh = 40 - self.b2;
        // Written by the vector stores below before any lane reads them.
        let mut units = MaybeUninit::<[[u64; 2]; 8]>::uninit();
        let mut part = MaybeUninit::<[u64; 8]>::uninit();
        unsafe {
            let c = _mm512_set1_epi64(super::keys::MIX_C as i64);
            let m40 = _mm512_set1_epi64(super::keys::M40 as i64);
            // Two 256-bit multiplies, never a 512-bit one (see `mullo64_halves`).
            let x = _mm512_and_si512(mullo64_halves(_mm512_loadu_si512(b.user_a.as_ptr() as *const _), c), m40);
            let mu = _mm512_xor_si512(x, _mm512_srli_epi64::<20>(x));
            let ps = _mm512_srl_epi64(mu, _mm_cvtsi32_si128(sh as i32));
            _mm512_storeu_si512(part.as_mut_ptr() as *mut _, _mm512_slli_epi64::<4>(ps));
            let ru = _mm512_and_si512(mu, _mm512_set1_epi64(((1u64 << sh) - 1) as i64));
            let ts = _mm512_cvtepu32_epi64(_mm256_loadu_si256(b.ts.as_ptr() as *const __m256i));
            let camp = _mm512_cvtepu16_epi64(_mm_loadu_si128(b.campaign.as_ptr() as *const __m128i));
            let loc = _mm512_loadu_si512(b.loc.as_ptr() as *const _);
            // 4,096 campaigns or more, or a locator past 52 bits (lanes 0..n): v4 cannot hold it.
            let over = _mm512_or_si512(_mm512_srli_epi64::<12>(camp), _mm512_srli_epi64::<52>(loc));
            self.bad |= _mm512_mask_test_epi64_mask(all, over, over) != 0;
            let w0 = _mm512_or_si512(_mm512_slli_epi64::<32>(ru), ts);
            let w1 = _mm512_or_si512(_mm512_slli_epi64::<52>(camp), loc);
            // [w0, w1] pairs in lane order
            let i0 = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
            let i1 = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
            _mm512_storeu_si512(units.as_mut_ptr() as *mut _, _mm512_permutex2var_epi64(w0, i0, w1));
            _mm512_storeu_si512((units.as_mut_ptr() as *mut [u64; 2]).add(4) as *mut _, _mm512_permutex2var_epi64(w0, i1, w1));
        }
        // SAFETY: every lane of both arrays was written above.
        let (units, part) = unsafe { (units.assume_init_ref(), part.assume_init_ref()) };
        // SAFETY: part[k] / 16 < 2^b2, the number of bins, for every lane.
        unsafe { store_lanes(&mut self.s, _pdep_u32(u32::from(short), 0x5555), part, units) };
        let mut m = u32::from(wide);
        while m != 0 {
            let k = m.trailing_zeros() as usize;
            self.wide(LONG, b.user_a[k], b.user_b[k], u64::from(b.ts[k]), units[k][1]);
            m &= m - 1;
        }
    }

    #[inline]
    fn wide(&mut self, kind: u64, a: u64, b: u64, tc: u64, cl: u64) {
        let p = (long_hash(a, b) >> (64 - self.b2)) as usize;
        // SAFETY: p < 2^b2.
        unsafe { self.wide.push_fast(p, [a, b, tc | kind << 62, cl]) };
    }

    #[inline(always)]
    fn short(&mut self, v: u64, tc: u64, cl: u64) {
        let mu = mix40(v);
        let sh = 40 - self.b2;
        // SAFETY: the top b2 bits of a 40-bit value: below 2^b2.
        unsafe { self.s.push_fast((mu >> sh) as usize, [(mu & ((1 << sh) - 1)) << 32 | tc, cl]) };
    }
}

impl<'a> Sink for ClickSink<'a, '_> {
    #[inline]
    fn clicks(&mut self, b: &ClickBatch) {
        #[cfg(target_arch = "x86_64")]
        if self.v512 {
            // SAFETY: the features were detected at construction.
            return unsafe { self.clicks512(b) };
        }
        #[cfg(target_arch = "x86_64")]
        if self.v256 {
            // SAFETY: as above.
            return unsafe { self.clicks256(b) };
        }
        let n = (b.n as usize).min(8);
        let mut over = 0u64;
        for k in 0..n {
            let (c, loc) = (u64::from(b.campaign[k]), b.loc[k]);
            over |= c >> 12 | loc >> 52;
            let cl = c << 52 | loc;
            let tc = u64::from(b.ts[k]);
            if b.user_long >> k & 1 == 0 {
                self.short(b.user_a[k], tc, cl);
            } else {
                self.wide(LONG, b.user_a[k], b.user_b[k], tc, cl);
            }
        }
        // 4,096 campaigns or more, or a locator past 52 bits: v4 cannot hold it.
        self.bad |= over != 0;
    }

    fn foreign_click(&mut self, at: Locator) {
        let line = self.ds.line(Stream::Clicks, at);
        let mut fields = Vec::with_capacity(8);
        let Some(c) = click_line(self.ctx, line, &mut fields) else {
            self.bad = true;
            return;
        };
        let tc = c.ts - self.ctx.click_base_ms;
        self.bad |= !(0..1i64 << 32).contains(&tc) || c.campaign >= 4096 || at.0 >> 52 != 0;
        let cl = u64::from(c.campaign) << 52 | at.0;
        match key_of(c.user_id, b'u') {
            (SHORT, v, _) => self.short(v, tc as u64, cl),
            (kind, a, b) => self.wide(kind, a, b, tc as u64, cl),
        }
    }

    fn events(&mut self, _: &EvBatch) {
        self.bad = true;
    }
    fn foreign_event(&mut self, _: Locator) {
        self.bad = true;
    }
    fn links(&mut self, _: &LinkBatch) {
        self.bad = true;
    }
    fn foreign_link(&mut self, _: Locator) {
        self.bad = true;
    }
}

/// Partition streams with their chunk lifetime (S's rules): each chunk is one block of `class`; a
/// consumer frees a partition's chunks once (`free_partition`, under its own worker index) before
/// writing that partition's output, so the output reuses them; whatever is still held on drop
/// (also on a decline to v3) goes back to the class's lists.
pub struct Streams<T: Copy> {
    /// `w[t]`: participant t's bins.
    pub w: Vec<Bins<T>>,
    freed: Vec<std::sync::atomic::AtomicBool>,
    /// The block class the chunks came from (B8K for I1, whose frees feed I2; B2K for the click
    /// streams: 2 KB chunks halve to a quarter the open chunk each (worker, user partition) holds).
    class: &'static rt::arena::Blocks,
}

impl<T: Copy> Streams<T> {
    /// (chunks, records): every participant's chunk count and records written.
    pub fn usage(&self) -> (usize, usize) {
        let (mut c, mut r) = (0, 0);
        for b in &self.w {
            for p in 0..self.freed.len() {
                for x in b.slices(p) {
                    c += 1;
                    r += x.len();
                }
            }
        }
        (c, r)
    }

    /// Bytes of one chunk (its block class).
    pub fn chunk_bytes(&self) -> usize {
        self.class.block_size()
    }

    fn new(w: Vec<Bins<T>>, parts: usize, class: &'static rt::arena::Blocks) -> Self {
        Streams { w, freed: (0..parts).map(|_| std::sync::atomic::AtomicBool::new(false)).collect(), class }
    }

    /// Partition p's chunks, every participant's (not after `free_partition(p)`).
    pub fn chunks(&self, p: usize) -> impl Iterator<Item = &[T]> + '_ {
        self.w.iter().flat_map(move |b| b.slices(p))
    }

    /// Returns partition p's chunks to their class's lists under worker index `w` (once; later calls
    /// do nothing).
    ///
    /// # Safety
    /// `w` is the caller's pool participant index (no other thread allocates or frees under it
    /// meanwhile), and no slice from `chunks(p)` is used afterwards.
    pub unsafe fn free_partition(&self, p: usize, w: usize) {
        if self.freed[p].swap(true, std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        for b in &self.w {
            for c in b.slices(p) {
                // SAFETY: each chunk is a whole block of `class` (Bins::from_blocks), freed once.
                unsafe { self.class.free(w, c.as_ptr() as *mut u8) };
            }
        }
    }
}

impl<T: Copy> Drop for Streams<T> {
    fn drop(&mut self) {
        // The pool is idle here: each participant's index is free to use for its own chunks.
        for p in 0..self.freed.len() {
            if self.freed[p].load(std::sync::atomic::Ordering::Relaxed) {
                continue;
            }
            for (t, b) in self.w.iter().enumerate() {
                for c in b.slices(p) {
                    // SAFETY: as in free_partition; t is not in use by any thread now.
                    unsafe { self.class.free(t, c.as_ptr() as *mut u8) };
                }
            }
        }
    }
}

/// I1: every event partition's plain units (16 bytes) and rare records (48 bytes), per participant.
pub struct EventBins {
    pub b1: u32,
    pub s: Streams<[u64; 2]>,
    pub o: Streams<[u64; 2]>,
}

/// I3: SHORT-user click records (16 bytes) and LONG/FOREIGN ones (32 bytes) per user partition.
pub struct ClickBins {
    pub b2: u32,
    pub s: Streams<[u64; 2]>,
    pub wide: Streams<[u64; 4]>,
}

impl ClickBins {
    /// Returns user partition q's click chunks (both forms) to the B2K lists, once.
    ///
    /// # Safety
    /// As `Streams::free_partition`.
    pub unsafe fn free_part(&self, q: usize, w: usize) {
        unsafe {
            self.s.free_partition(q, w);
            self.wide.free_partition(q, w);
        }
    }
}

impl EventBins {
    /// Event partition p's plain chunks, every sink's (one unit = one record).
    pub fn chunks(&self, p: usize) -> impl Iterator<Item = &[[u64; 2]]> + '_ {
        self.s.chunks(p)
    }

    /// Event partition p's rare chunks, every sink's.
    pub fn rare(&self, p: usize) -> impl Iterator<Item = &[[u64; 2]]> + '_ {
        self.o.chunks(p)
    }

    /// Returns event partition p's chunks (both lanes) to their block lists (B8K, B2K), once.
    ///
    /// # Safety
    /// As `Streams::free_partition`.
    pub unsafe fn free_part(&self, p: usize, w: usize) {
        unsafe {
            self.s.free_partition(p, w);
            self.o.free_partition(p, w);
        }
    }
}

/// The link tasks, every event chunk and every click chunk in ONE pool run (links first: their
/// union-find waits for them on its own thread, beside the rest), so no stage stands in front of
/// the event parse. Every link task runs exactly once (`LinkTasks::finish` waits for them all).
pub fn parse_all(cx: &Cx, lt: &super::links::LinkTasks) -> (Stage<EventBins>, Stage<ClickBins>) {
    let ev = parselab::par::plan(cx.ds, &[Stream::Events], 4 << 20);
    let ck = parselab::par::plan(cx.ds, &[Stream::Clicks], 4 << 20);
    let (nl, ne) = (lt.len(), ev.len());
    let t0 = std::time::Instant::now();
    let st: Vec<(EvSink, ClickSink, usize)> = crate::par::run(
        cx.plan.threads,
        nl + ne + ck.len(),
        |t| (EvSink::new(cx.ds, cx.ctx, cx.plan.b1, t), ClickSink::new(cx.ds, cx.ctx, cx.plan.b2, t), t),
        |(es, cs, t), i| {
            if i == 0 {
                super::lap("first link byte");
            } else if i == nl {
                super::lap("first event byte");
            }
            if i < nl {
                lt.task(cx, i, *t);
            } else if i < nl + ne {
                let c = &ev[i - nl];
                let bytes = cx.ds.stream(c.stream)[c.file as usize].bytes();
                parselab::fast::parse(&cx.fp, cx.isa, parselab::fast::Kind::Events, c.file, parselab::fast::Buf::plain(bytes), c.from, c.to, es);
                cx.inp.consumed(cx.inp.first(c.stream as usize) + c.file as usize, c.from, c.to);
            } else {
                let c = &ck[i - nl - ne];
                let bytes = cx.ds.stream(c.stream)[c.file as usize].bytes();
                parselab::fast::parse(&cx.fp, cx.isa, parselab::fast::Kind::Clicks, c.file, parselab::fast::Buf::plain(bytes), c.from, c.to, cs);
            }
        },
    );
    let t1 = std::time::Instant::now();
    cx.inp.flush();
    let t2 = std::time::Instant::now();
    let (mut evs, mut cks) = (Vec::with_capacity(st.len()), Vec::with_capacity(st.len()));
    for (e, c, _) in st {
        evs.push(e);
        cks.push(c);
    }
    let r = (close_events(cx, evs), close_clicks(cx, cks));
    if cx.timing {
        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        eprintln!(
            "timing v4.  parse_all: pool run {:.2} ms ({} link + {} event + {} click tasks), input flush {:.2} ms, closing the bins {:.2} ms (one thread)",
            ms(t0, t1),
            nl,
            ne,
            ck.len(),
            ms(t1, t2),
            ms(t2, std::time::Instant::now())
        );
    }
    r
}

/// Parses every event line into I1.
pub fn events(cx: &Cx) -> Stage<EventBins> {
    let chunks = parselab::par::plan(cx.ds, &[Stream::Events], 4 << 20);
    let sinks: Vec<EvSink> = crate::par::run(
        cx.plan.threads,
        chunks.len(),
        |t| EvSink::new(cx.ds, cx.ctx, cx.plan.b1, t),
        |s, i| {
            let c = &chunks[i];
            let bytes = cx.ds.stream(c.stream)[c.file as usize].bytes();
            parselab::fast::parse(&cx.fp, cx.isa, parselab::fast::Kind::Events, c.file, parselab::fast::Buf::plain(bytes), c.from, c.to, s);
            cx.inp.consumed(cx.inp.first(c.stream as usize) + c.file as usize, c.from, c.to);
        },
    );
    cx.inp.flush();
    close_events(cx, sinks)
}

/// The event sinks' streams as EventBins, then the declines.
fn close_events(cx: &Cx, mut sinks: Vec<EvSink>) -> Stage<EventBins> {
    let bad = sinks.iter().any(|s| s.bad);
    for s in sinks.iter_mut() {
        s.s.finish();
        s.o.finish();
    }
    // Built before any decline, so a decline returns every chunk (Streams' drop).
    let (mut sv, mut ov) = (Vec::new(), Vec::new());
    for s in sinks {
        sv.push(s.s);
        ov.push(s.o);
    }
    let parts = 1 << cx.plan.b1;
    // The plain lane on B8K (its frees feed I2's chunks); the rare lane on B2K (42 records a
    // chunk: every worker holds an open chunk per partition, a quarter of the memory of B8K's).
    let out = EventBins { b1: cx.plan.b1, s: Streams::new(sv, parts, &crate::v3::mem::B8K), o: Streams::new(ov, parts, &crate::v3::mem::B2K) };
    if cx.timing {
        let (c, r) = out.s.usage();
        let (co, ro) = out.o.usage();
        eprintln!("timing v4.  I1: plain {c} chunks ({} MB), {r} units ({} MB); rare {co} chunks ({} MB), {ro} units ({} MB)", c * out.s.chunk_bytes() >> 20, r * 16 >> 20, co * out.o.chunk_bytes() >> 20, ro * 16 >> 20);
    }
    if bad {
        return Err("an event record v4 cannot hold".into());
    }
    if foreign_count() >= FOREIGN_CAP {
        return Err("FOREIGN ids past the cap".into());
    }
    Ok(out)
}

/// Parses every click line into I3.
pub fn clicks(cx: &Cx) -> Stage<ClickBins> {
    let chunks = parselab::par::plan(cx.ds, &[Stream::Clicks], 4 << 20);
    let sinks: Vec<ClickSink> = crate::par::run(
        cx.plan.threads,
        chunks.len(),
        |t| ClickSink::new(cx.ds, cx.ctx, cx.plan.b2, t),
        |s, i| {
            let c = &chunks[i];
            let bytes = cx.ds.stream(c.stream)[c.file as usize].bytes();
            parselab::fast::parse(&cx.fp, cx.isa, parselab::fast::Kind::Clicks, c.file, parselab::fast::Buf::plain(bytes), c.from, c.to, s);
        },
    );
    close_clicks(cx, sinks)
}

/// The click sinks' streams as ClickBins, then the declines.
fn close_clicks(cx: &Cx, mut sinks: Vec<ClickSink>) -> Stage<ClickBins> {
    let bad = sinks.iter().any(|s| s.bad);
    for s in sinks.iter_mut() {
        s.s.finish();
        s.wide.finish();
    }
    let (mut sv, mut wv) = (Vec::new(), Vec::new());
    for s in sinks {
        sv.push(s.s);
        wv.push(s.wide);
    }
    let parts = 1 << cx.plan.b2;
    let out = ClickBins { b2: cx.plan.b2, s: Streams::new(sv, parts, &crate::v3::mem::B2K), wide: Streams::new(wv, parts, &crate::v3::mem::B2K) };
    if cx.timing {
        let (c, r) = out.s.usage();
        let (cw, rw) = out.wide.usage();
        let cb = out.s.chunk_bytes();
        eprintln!("timing v4.  I3: {c} chunks ({} MB), {r} records ({} MB); wide {cw} chunks ({} MB), {rw} records ({} MB)", c * cb >> 20, r * 16 >> 20, cw * cb >> 20, rw * 32 >> 20);
    }
    if bad {
        return Err("a click record v4 cannot hold".into());
    }
    if foreign_count() >= FOREIGN_CAP {
        return Err("FOREIGN ids past the cap".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    //! I1 decoded back against the parser's own records (dev5 or $V4_DATA), every lane.
    use super::*;
    use crate::v4::keys::unmix40;
    use parselab::{EvRow, Key as PKey, VecSink};

    /// Worker indices of these tests: at the top of the range, so no other module's test (which
    /// take 0, 1, ...) uses them at the same time.
    const TEST_W: usize = rt::pool::MAX_WORKERS - 5;

    struct Both<'a, 'b> {
        v4: EvSink<'a, 'b>,
        v: VecSink,
    }
    impl<'a> Sink for Both<'a, '_> {
        fn events(&mut self, b: &EvBatch) {
            self.v4.events(b);
            self.v.events(b);
        }
        fn foreign_event(&mut self, _: Locator) {}
        fn clicks(&mut self, _: &ClickBatch) {}
        fn foreign_click(&mut self, _: Locator) {}
        fn links(&mut self, _: &LinkBatch) {}
        fn foreign_link(&mut self, _: Locator) {}
    }

    fn norm(mut r: EvRow) -> EvRow {
        if !r.surv {
            r.user = PKey::default();
            r.ts = 0;
            r.purch = false;
        }
        if !r.purch {
            r.cat = 0;
            r.price = 0;
        }
        r
    }

    #[test]
    fn i1_decodes_to_the_parser_records() {
        i1_check(Some(true));
    }

    #[test]
    fn i1_avx2_path_decodes_to_the_parser_records() {
        i1_check(Some(false));
    }

    #[test]
    fn i1_scalar_path_decodes_to_the_parser_records() {
        i1_check(None);
    }

    struct ClickBoth<'a, 'b> {
        v4: ClickSink<'a, 'b>,
        v: VecSink,
    }
    impl<'a> Sink for ClickBoth<'a, '_> {
        fn events(&mut self, _: &EvBatch) {}
        fn foreign_event(&mut self, _: Locator) {}
        fn clicks(&mut self, b: &ClickBatch) {
            self.v4.clicks(b);
            self.v.clicks(b);
        }
        fn foreign_click(&mut self, _: Locator) {}
        fn links(&mut self, _: &LinkBatch) {}
        fn foreign_link(&mut self, _: Locator) {}
    }

    #[test]
    fn i3_decodes_to_the_parser_records() {
        let dir = std::env::var("V4_DATA").unwrap_or_else(|_| format!("{}/shop/data/dev5", std::env::var("HOME").unwrap()));
        let ds = Dataset::open(std::path::Path::new(&dir), parselab::MapMode::Lazy).unwrap();
        let ctx = parselab::prepare(&ds).unwrap();
        assert!(crate::v3::mem::init(1 << 30));
        let b2 = 11;
        let mut both = ClickBoth { v4: ClickSink::new(&ds, &ctx, b2, TEST_W + 1), v: VecSink::default() };
        let f = parselab::fast::Fast::new(&ctx);
        let isa = parselab::fast::Isa::detect();
        for (i, m) in ds.stream(Stream::Clicks).iter().enumerate() {
            let d = m.bytes();
            parselab::fast::parse(&f, isa, parselab::fast::Kind::Clicks, i as u32, parselab::fast::Buf::plain(d), 0, d.len(), &mut both);
        }
        both.v4.s.finish();
        both.v4.wide.finish();
        assert!(!both.v4.bad);
        let sh = 40 - b2;
        let mut got: Vec<parselab::ClickRow> = Vec::new();
        for q in 0..1usize << b2 {
            for c in both.v4.s.slices(q) {
                for &[w0, w1] in c {
                    let user = PKey { long: false, a: unmix40((q as u64) << sh | w0 >> 32), b: 0 };
                    got.push(parselab::ClickRow { user, ts: w0 as u32, campaign: (w1 >> 52) as u16, loc: Locator(w1 & ((1 << 52) - 1)) });
                }
            }
            for c in both.v4.wide.slices(q) {
                for &[a, b, t, cl] in c {
                    got.push(parselab::ClickRow { user: PKey { long: true, a, b }, ts: t as u32, campaign: (cl >> 52) as u16, loc: Locator(cl & ((1 << 52) - 1)) });
                }
            }
        }
        let mut want = both.v.clicks.clone();
        got.sort();
        want.sort();
        assert_eq!(got.len(), want.len());
        assert!(got == want, "I3 differs from the parser's records");
    }

    /// `v512`: Some(true) AVX-512, Some(false) AVX2, None scalar (each where detected).
    fn i1_check(v512: Option<bool>) {
        let dir = std::env::var("V4_DATA").unwrap_or_else(|_| format!("{}/shop/data/dev5", std::env::var("HOME").unwrap()));
        let ds = Dataset::open(std::path::Path::new(&dir), parselab::MapMode::Lazy).unwrap();
        let ctx = parselab::prepare(&ds).unwrap();
        assert!(crate::v3::mem::init(1 << 30));
        let b1 = 10;
        // distinct worker indices: the tests run on parallel threads and share the B8K class
        let w = match v512 { Some(true) => TEST_W, Some(false) => TEST_W + 2, None => TEST_W + 3 };
        let mut both = Both { v4: EvSink::new(&ds, &ctx, b1, w), v: VecSink::default() };
        both.v4.v512 &= v512 == Some(true);
        both.v4.v256 &= v512 != None;
        let f = parselab::fast::Fast::new(&ctx);
        let isa = parselab::fast::Isa::detect();
        for (i, m) in ds.stream(Stream::Events).iter().enumerate() {
            let d = m.bytes();
            parselab::fast::parse(&f, isa, parselab::fast::Kind::Events, i as u32, parselab::fast::Buf::plain(d), 0, d.len(), &mut both);
        }
        both.v4.s.finish();
        both.v4.o.finish();
        let mut got: Vec<EvRow> = Vec::new();
        let sh = 40 - b1;
        let key = |k: u64, a: u64, b: u64| if k == SHORT { PKey { long: false, a: unmix40(a), b: 0 } } else { PKey { long: true, a, b } };
        for p in 0..1usize << b1 {
            for c in both.v4.s.slices(p) {
                for &[w0, w1] in c {
                    let mut r = EvRow::default();
                    r.surv = w0 >> 62 & 1 != 0;
                    assert!(w0 >> 63 == 0, "a purchase in the plain lane");
                    r.ingest = (w0 >> 32 & M23) as u32;
                    r.ev = key(SHORT, (p as u64) << sh | (w0 & 0xFFFF_FFFF), 0);
                    if r.surv {
                        r.user = key(SHORT, (w0 >> 55 & 0x7F) << 33 | w1 >> 31, 0);
                        r.ts = (w1 & ((1 << 31) - 1)) as u32;
                    }
                    got.push(norm(r));
                }
            }
            for c in both.v4.o.slices(p) {
                let mut i = 0;
                while i < c.len() {
                    let [h, w1] = c[i];
                    let big = h & RARE_BIG != 0;
                    let ext = h & RARE_EXT != 0;
                    let mut r = EvRow::default();
                    let (ek, uk) = (h >> 23 & 3, h >> 25 & 3);
                    r.ingest = (h & M23) as u32;
                    r.surv = h >> 27 & 1 != 0;
                    r.purch = h >> 28 & 1 != 0;
                    r.ts = (h >> 29 & ((1 << 31) - 1)) as u32;
                    let (ev, user, pay) = if ek == SHORT {
                        (key(SHORT, (p as u64) << sh | (w1 & 0xFFFF_FFFF), 0), c[i + 1], w1 >> 32)
                    } else if big {
                        (key(LONG, c[i + 1][0], c[i + 1][1]), c[i + 2], w1)
                    } else {
                        (key(LONG, c[i + 1][0], c[i + 1][1]), [w1, 0], 0)
                    };
                    r.ev = ev;
                    if r.surv {
                        r.user = key(uk, user[0], user[1]);
                    }
                    if r.purch {
                        r.cat = pay as u8;
                        r.price = (pay >> 8) as u32;
                    }
                    got.push(norm(r));
                    i += 2 + usize::from(big) + usize::from(ext);
                }
            }
        }
        let mut want: Vec<EvRow> = both.v.events.iter().map(|&r| norm(r)).collect();
        got.sort();
        want.sort();
        assert!(!both.v4.bad);
        assert_eq!(got.len(), want.len());
        assert!(got == want, "I1 differs from the parser's records");
        eprintln!("I1: {} records decoded equal", got.len());
    }
}
