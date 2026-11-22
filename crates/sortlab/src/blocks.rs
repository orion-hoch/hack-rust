//! v4 entry: sessions straight from P1's person blocks (RECORDS-V4 section 8), with no grouping
//! pass: P2 hands each person's blocks by reference, the kernel cuts each person on its own.
//!
//! Person block (8-byte words): h0 = n_ev | n_purch << 24 | wide << 63, h1 = pk, then n_ev u32
//! events `ts << 1 | purch` two per word (low half first; an odd count pads the high half), then
//! n_purch payload words `cat | price << 8`, payload k for the k-th event with the purch bit.
//! Click block: h0 = n_ck | 1 << 62 | wide << 63, h1 = pk, then pairs [campaign << 52 | loc, tc].

use super::*;

#[cfg(not(target_endian = "little"))]
compile_error!("person blocks pack two u32 events per word, low half first: little-endian only");

/// One partition's persons as block references (P2 groups the blocks by person key).
pub struct PersonBlocks<'a> {
    /// Person blocks, whole as P1 wrote them, grouped by person.
    pub ev: &'a [&'a [u64]],
    /// Person k (0..np) owns `ev[ev_at[k]..ev_at[k + 1]]`; np + 1 entries.
    pub ev_at: &'a [u32],
    /// Click blocks, grouped by the same person index.
    pub ck: &'a [&'a [u64]],
    pub ck_at: &'a [u32],
}

impl PersonBlocks<'_> {
    pub fn persons(&self) -> usize {
        self.ev_at.len().saturating_sub(1)
    }
}

/// The events (u32 `ts << 1 | purch`) and payload words of one person block.
#[inline(always)]
fn parts(b: &[u64]) -> Result<(&[u32], &[u64]), Error> {
    let h0 = *b.first().ok_or(Error::BlockShape)?;
    let (n, m) = ((h0 & 0xFF_FFFF) as usize, ((h0 >> 24) & 0xFF_FFFF) as usize);
    let w = n.div_ceil(2);
    if h0 >> 62 & 1 != 0 || b.len() < 2 + w + m {
        return Err(Error::BlockShape);
    }
    // SAFETY: words 2..2 + w are in the block (checked), and hold n u32 (little-endian: the low
    // half of a word is the first u32 in memory); u32 alignment divides u64 alignment.
    let ev = unsafe { core::slice::from_raw_parts(b.as_ptr().add(2) as *const u32, n) };
    Ok((ev, &b[2 + w..2 + w + m]))
}

/// Without the network, persons up to this many events are sorted (insertion) and walked.
const SMALL_SORT: usize = 24;

/// How a person is cut: the network up to `MAX` events (0 = never), slots above.
trait Cut {
    const MAX: usize;
    /// The block loops below may use 256-bit AVX2 loads and stores (the caller has AVX-512F,
    /// which implies AVX2).
    const VEC: bool;
    /// # Safety
    /// 1 <= ts.len() <= MAX, ts < 2^31, room in `out.start` / `out.events` for ts.len() + 64 more.
    unsafe fn net(ts: &[u32], out: &mut Out);
}

struct SlotsOnly;
impl Cut for SlotsOnly {
    const MAX: usize = 0;
    const VEC: bool = false;
    #[inline(always)]
    unsafe fn net(_: &[u32], _: &mut Out) {}
}

#[cfg(target_arch = "x86_64")]
struct NetCut<L>(core::marker::PhantomData<L>);
#[cfg(target_arch = "x86_64")]
impl<L: net::Lanes> Cut for NetCut<L> {
    const MAX: usize = net::MAX_N;
    const VEC: bool = true;
    #[inline(always)]
    unsafe fn net(ts: &[u32], out: &mut Out) {
        // SAFETY: the caller's contract, and L's features (the target-feature entry points)
        unsafe { net::person_net_inl::<L>(ts, out, core::ptr::null_mut()) }
    }
}

/// Appends `e >> 1` of every event of one block to `ts`; true when some event has the purch bit.
///
/// # Safety
/// With `C::VEC`, AVX2 present.
#[inline(always)]
unsafe fn shift_into<C: Cut>(ev: &[u32], ts: &mut Vec<u32>) -> bool {
    let n = ev.len();
    ts.reserve(n);
    let len = ts.len();
    let src = ev.as_ptr();
    // SAFETY: room for n more was reserved; every write below is to dst[i], i < n, and every
    // read is ev[i], i < n (the overlapping last vector covers [n - 8, n) with n >= 8).
    unsafe {
        let dst = ts.as_mut_ptr().add(len);
        let any;
        #[cfg(target_arch = "x86_64")]
        if C::VEC && n >= 8 {
            use core::arch::x86_64::*;
            let mut acc = _mm256_setzero_si256();
            let mut i = 0usize;
            while i + 8 <= n {
                let v = _mm256_loadu_si256(src.add(i) as *const __m256i);
                acc = _mm256_or_si256(acc, v);
                _mm256_storeu_si256(dst.add(i) as *mut __m256i, _mm256_srli_epi32::<1>(v));
                i += 8;
            }
            if i < n {
                // the last 8 once more: the same values are written again
                let v = _mm256_loadu_si256(src.add(n - 8) as *const __m256i);
                acc = _mm256_or_si256(acc, v);
                _mm256_storeu_si256(dst.add(n - 8) as *mut __m256i, _mm256_srli_epi32::<1>(v));
            }
            any = _mm256_testz_si256(acc, _mm256_set1_epi32(1)) == 0;
            ts.set_len(len + n);
            return any;
        }
        let mut bits = 0u32;
        for i in 0..n {
            let e = *src.add(i);
            bits |= e;
            *dst.add(i) = e >> 1;
        }
        any = bits & 1 != 0;
        ts.set_len(len + n);
        any
    }
}

/// The purchases of one block, in event order: each one's session (the last session of `starts`
/// starting at or before it, as an index from `s0`) and payload. False when the block's purch
/// bits and payload words do not pair up one to one (nothing is appended then).
///
/// # Safety
/// With `C::VEC`, AVX2 present.
#[inline(always)]
unsafe fn buys<C: Cut>(ev: &[u32], pay: &[u64], s0: usize, starts: &[u32], bs: &mut Vec<u32>, bp: &mut Vec<u32>) -> bool {
    let (n, m) = (ev.len(), pay.len());
    bs.reserve(m);
    bp.reserve(m);
    let src = ev.as_ptr();
    let mut k = 0usize;
    // SAFETY: reads are ev[j], j < n, and pay[k], k < m (checked before each use); writes go to
    // the m reserved entries past each vector's length, index k < m.
    unsafe {
        let (dbs, dbp) = (bs.as_mut_ptr().add(bs.len()), bp.as_mut_ptr().add(bp.len()));
        let mut one = |j: usize, k: &mut usize| -> bool {
            if *k == m {
                return false;
            }
            let t = *src.add(j) >> 1;
            *dbs.add(*k) = (s0 + starts.partition_point(|&s| s <= t) - 1) as u32;
            *dbp.add(*k) = *pay.get_unchecked(*k) as u32;
            *k += 1;
            true
        };
        let mut i = 0usize;
        #[cfg(target_arch = "x86_64")]
        if C::VEC {
            use core::arch::x86_64::*;
            while i + 8 <= n {
                let v = _mm256_loadu_si256(src.add(i) as *const __m256i);
                // bit 0 of each event into its lane's sign bit, then one bit per lane
                let mut mk = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32::<31>(v))) as u32;
                while mk != 0 {
                    let j = i + mk.trailing_zeros() as usize;
                    mk &= mk - 1;
                    if !one(j, &mut k) {
                        return false;
                    }
                }
                i += 8;
            }
        }
        while i < n {
            if *src.add(i) & 1 != 0 && !one(i, &mut k) {
                return false;
            }
            i += 1;
        }
        if k != m {
            return false;
        }
        bs.set_len(bs.len() + m);
        bp.set_len(bp.len() + m);
    }
    true
}

/// Sessions of one partition from person blocks: `out.persons` = 0..np, and per purchase (person,
/// then block, then event order) its session in `out.buy_session` and its payload in
/// `out.buy_pay`. Kernel: `Net` / `NetWith` take the network where AVX-512 exists, anything else
/// the slots for every person.
pub fn sessions_blocks(pb: &PersonBlocks, kernel: Kernel, sc: &mut Scratch, out: &mut Out) -> Result<(), Error> {
    out.clear();
    #[cfg(target_arch = "x86_64")]
    {
        let net = match kernel {
            Kernel::Net => Some(NetWidth::Auto),
            Kernel::NetWith { width, .. } => Some(width),
            _ => None,
        };
        if let Some(width) = net {
            let bits = sc.nets[match width {
                NetWidth::W512 => 0,
                NetWidth::W256 => 1,
                NetWidth::Auto => 2,
            }];
            if bits == 512 {
                // SAFETY: bits == 512 only when AVX-512F was detected (`net_bits`).
                return unsafe { blocks512(pb, sc, out) };
            }
            if bits == 256 {
                // SAFETY: bits == 256 only when AVX-512F and AVX-512VL were detected.
                return unsafe { blocks256(pb, sc, out) };
            }
            if bits == 2 {
                // SAFETY: bits == 2 only when AVX2 was detected and AVX-512 is absent or switched off.
                return unsafe { blocks_avx2(pb, sc, out) };
            }
        }
    }
    let _ = kernel;
    // SAFETY: SlotsOnly never calls the network.
    unsafe { blocks_partition::<SlotsOnly>(pb, sc, out) }
}

/// # Safety
/// AVX-512F present.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn blocks512(pb: &PersonBlocks, sc: &mut Scratch, out: &mut Out) -> Result<(), Error> {
    unsafe { blocks_partition::<NetCut<net::Z>>(pb, sc, out) }
}

/// The 256-bit network behind a call, so that it is the ONLY code compiled with AVX-512 enabled.
/// Why: inside a function that has AVX-512F enabled the compiler vectorises ordinary loops with
/// 512-bit registers (the reservation sum, the person index fill and constants did: 166 zmm
/// instructions in blocks256, run once per partition, about every millisecond). Any 512-bit
/// instruction makes a Skylake-X / Cascade Lake core lower its clock for about 2 ms, so the whole
/// session stage ran at the reduced clock for the sake of loops that gain nothing from it.
#[cfg(target_arch = "x86_64")]
struct NetCut256;
#[cfg(target_arch = "x86_64")]
impl Cut for NetCut256 {
    const MAX: usize = net::MAX_N;
    const VEC: bool = true;
    #[inline(always)]
    unsafe fn net(ts: &[u32], out: &mut Out) {
        // SAFETY: the caller's contract; blocks256 is only entered when AVX-512F and VL exist.
        unsafe { net256_call(ts, out) }
    }
}

/// # Safety
/// AVX-512F and AVX-512VL present, and the conditions of `net::person_net_inl`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl")]
#[inline(never)]
unsafe fn net256_call(ts: &[u32], out: &mut Out) {
    unsafe { net::person_net_inl::<net::Y>(ts, out, core::ptr::null_mut()) }
}

/// The same network on AVX2 lanes, for CPUs without AVX-512.
#[cfg(target_arch = "x86_64")]
struct NetCutA;
#[cfg(target_arch = "x86_64")]
impl Cut for NetCutA {
    const MAX: usize = net::MAX_N;
    const VEC: bool = true;
    #[inline(always)]
    unsafe fn net(ts: &[u32], out: &mut Out) {
        // SAFETY: the caller's contract; blocks_avx2 is only entered when AVX2 was detected.
        unsafe { neta_call(ts, out) }
    }
}

/// # Safety
/// AVX2 present, and the conditions of `net::person_net_inl`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline(never)]
unsafe fn neta_call(ts: &[u32], out: &mut Out) {
    unsafe { net::person_net_inl::<net::A>(ts, out, core::ptr::null_mut()) }
}

/// # Safety
/// AVX2 present.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn blocks_avx2(pb: &PersonBlocks, sc: &mut Scratch, out: &mut Out) -> Result<(), Error> {
    unsafe { blocks_partition::<NetCutA>(pb, sc, out) }
}

/// # Safety
/// AVX-512F and AVX-512VL present (for the network it calls); this function itself is compiled
/// for AVX2 only, so nothing in it can use a 512-bit register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn blocks256(pb: &PersonBlocks, sc: &mut Scratch, out: &mut Out) -> Result<(), Error> {
    unsafe { blocks_partition::<NetCut256>(pb, sc, out) }
}

/// # Safety
/// C's features present.
#[inline(always)]
unsafe fn blocks_partition<C: Cut>(pb: &PersonBlocks, sc: &mut Scratch, out: &mut Out) -> Result<(), Error> {
    let np = pb.persons();
    if pb.ck_at.len() != np + 1 {
        return Err(Error::BlockShape);
    }
    // room for every session at once (sessions <= events) plus the network's store slack
    // (an upper bound from the block lengths alone: at most two events per word after the two
    // header words; the headers themselves are read, and checked, once per person below)
    let mut total = 0usize;
    for b in pb.ev {
        total += 2 * b.len();
    }
    out.persons.extend(0..np as u32);
    out.sess_end.reserve(np);
    out.start.reserve(total + 64);
    out.events.reserve(total + 64);
    let mut ts = core::mem::take(&mut sc.grouped);
    // the person's blocks that hold payloads, parsed once (most persons have none)
    let mut paid: Vec<(&[u32], &[u64])> = Vec::new();
    let mut res = Ok(());
    'person: for k in 0..np {
        // A person's blocks sit wherever P1's streams put them: ask for the first two lines of
        // the block of the person three ahead while this one is cut (each block read was a miss).
        #[cfg(target_arch = "x86_64")]
        if k + 3 < np {
            if let Some(nb) = pb.ev.get(pb.ev_at[k + 3] as usize) {
                let q = nb.as_ptr() as *const i8;
                // SAFETY: a prefetch never faults; the addresses are in or just past a live block.
                unsafe {
                    core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(q);
                    core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(q.wrapping_add(64));
                }
            }
        }
        let blocks = &pb.ev[pb.ev_at[k] as usize..pb.ev_at[k + 1] as usize];
        ts.clear();
        paid.clear();
        for b in blocks {
            match parts(b) {
                Ok((ev, pay)) => {
                    // SAFETY: C's features (AVX-512F implies AVX2 where C::VEC).
                    let purch = unsafe { shift_into::<C>(ev, &mut ts) };
                    if !pay.is_empty() {
                        paid.push((ev, pay));
                    } else if purch {
                        // a purch bit in a block without payloads
                        res = Err(Error::BlockShape);
                        break 'person;
                    }
                }
                Err(e) => {
                    res = Err(e);
                    break 'person;
                }
            }
        }
        let s0 = out.start.len();
        if !ts.is_empty() {
            if ts.len() <= C::MAX {
                // SAFETY: 1 <= len <= MAX; ts < 2^31 (31-bit field); room reserved above.
                unsafe { C::net(&ts, out) };
            } else if C::MAX == 0 && ts.len() <= SMALL_SORT {
                // No network on this CPU: a small person by insertion sort and one walk, instead
                // of the slot method's fixed scan of every bitmap word (the v3 kernel's own small
                // path; same sessions, proved equal on sorted input).
                sort_ts(&mut ts, &mut sc.tmp);
                walk_sorted(&ts, out, core::ptr::null_mut());
            } else {
                sc.next_serial();
                person_slots(&ts, sc, out, core::ptr::null_mut());
            }
        }
        // purchases: each is one of the person's events, so it lies in the last session starting
        // at or before it (blocks in the person's order, events in block order, as before)
        for &(ev, pay) in &paid {
            // SAFETY: C's features (AVX-512F implies AVX2 where C::VEC).
            if !unsafe { buys::<C>(ev, pay, s0, &out.start[s0..], &mut out.buy_session, &mut out.buy_pay) } {
                res = Err(Error::BlockShape);
                break 'person;
            }
        }
        out.sess_end.push(out.start.len() as u32);
    }
    sc.grouped = ts;
    res
}
