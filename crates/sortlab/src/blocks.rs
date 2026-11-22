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

/// How a person is cut: the network up to `MAX` events (0 = never), slots above.
trait Cut {
    const MAX: usize;
    /// # Safety
    /// 1 <= ts.len() <= MAX, ts < 2^31, room in `out.start` / `out.events` for ts.len() + 64 more.
    unsafe fn net(ts: &[u32], out: &mut Out);
}

struct SlotsOnly;
impl Cut for SlotsOnly {
    const MAX: usize = 0;
    #[inline(always)]
    unsafe fn net(_: &[u32], _: &mut Out) {}
}

#[cfg(target_arch = "x86_64")]
struct NetCut<L>(core::marker::PhantomData<L>);
#[cfg(target_arch = "x86_64")]
impl<L: net::Lanes> Cut for NetCut<L> {
    const MAX: usize = net::MAX_N;
    #[inline(always)]
    unsafe fn net(ts: &[u32], out: &mut Out) {
        // SAFETY: the caller's contract, and L's features (the target-feature entry points)
        unsafe { net::person_net_inl::<L>(ts, out, core::ptr::null_mut()) }
    }
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

/// # Safety
/// AVX-512F and AVX-512VL present.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl")]
unsafe fn blocks256(pb: &PersonBlocks, sc: &mut Scratch, out: &mut Out) -> Result<(), Error> {
    unsafe { blocks_partition::<NetCut<net::Y>>(pb, sc, out) }
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
    let mut total = 0usize;
    for b in pb.ev {
        total += (*b.first().ok_or(Error::BlockShape)? & 0xFF_FFFF) as usize;
    }
    out.persons.extend(0..np as u32);
    out.sess_end.reserve(np);
    out.start.reserve(total + 64);
    out.events.reserve(total + 64);
    let mut ts = core::mem::take(&mut sc.grouped);
    let mut res = Ok(());
    'person: for k in 0..np {
        let blocks = &pb.ev[pb.ev_at[k] as usize..pb.ev_at[k + 1] as usize];
        ts.clear();
        for b in blocks {
            match parts(b) {
                Ok((ev, pay)) => {
                    let mut bits = 0u32;
                    ts.extend(ev.iter().map(|&e| {
                        bits |= e;
                        e >> 1
                    }));
                    // a purch bit in a block without payloads
                    if pay.is_empty() && bits & 1 != 0 {
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
            } else {
                sc.next_serial();
                person_slots(&ts, sc, out, core::ptr::null_mut());
            }
        }
        // purchases: each is one of the person's events, so it lies in the last session starting
        // at or before it
        let starts = &out.start[s0..];
        for b in blocks {
            let (ev, pay) = parts(b).unwrap_or((&[], &[]));
            if pay.is_empty() {
                // no purchase in this block (a purch bit without payloads was refused above)
                continue;
            }
            let mut k = 0usize;
            for &e in ev {
                if e & 1 != 0 {
                    let Some(&p) = pay.get(k) else {
                        res = Err(Error::BlockShape);
                        break 'person;
                    };
                    let t = e >> 1;
                    out.buy_session.push((s0 + starts.partition_point(|&s| s <= t) - 1) as u32);
                    out.buy_pay.push(p as u32);
                    k += 1;
                }
            }
            if k != pay.len() {
                res = Err(Error::BlockShape);
                break 'person;
            }
        }
        out.sess_end.push(out.start.len() as u32);
    }
    sc.grouped = ts;
    res
}
