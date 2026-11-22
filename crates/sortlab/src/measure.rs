//! Measurement-only entry points for the bench (not used by the kernels). Kept out of net.rs and
//! the kernel code so their hashes only move when a kernel changes.

use super::*;

/// One person's sessions by slots, for measurement (sessions appended to `out.start/events`).
pub fn person_by_slots(ts: &[u32], sc: &mut Scratch, out: &mut Out) {
    sc.next_serial();
    person_slots(ts, sc, out, core::ptr::null_mut());
}

/// One person's sessions by sorting `ts` in place, for measurement.
pub fn person_by_sort(ts: &mut [u32], sc: &mut Scratch, out: &mut Out) {
    sort_ts(ts, &mut sc.tmp);
    walk_sorted(ts, out, core::ptr::null_mut());
}

/// Clears only the session arrays of `out` (for measurement loops).
pub fn clear_sessions(out: &mut Out) {
    out.start.clear();
    out.events.clear();
}

/// Grouping alone (persons, counts, timestamps grouped, purchases grouped), for measurement.
/// Returns the number of persons.
pub fn group_only(ev: &[u64], buys: &[[u64; 2]], dense: Option<usize>, sc: &mut Scratch, out: &mut Out) -> Result<usize, Error> {
    out.clear();
    group(ev, buys, dense, true, sc, out)
}

/// One person's sessions by the network of `bits` (512 or 256; 1..=128 events), for
/// measurement. False where that width is missing or the person is out of range.
pub fn person_by_net(ts: &[u32], bits: u32, out: &mut Out) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if ts.is_empty() || ts.len() > net::MAX_N {
            return false;
        }
        if bits == 512 && net_bits(NetWidth::W512) == 512 {
            // SAFETY: AVX-512F detected; 1 <= ts.len() <= 128; no ends wanted.
            unsafe { net::person_net512(ts, out, core::ptr::null_mut()) };
            return true;
        }
        if bits == 256 && net_bits(NetWidth::W256) == 256 {
            // SAFETY: AVX-512F and VL detected; 1 <= ts.len() <= 128; no ends wanted.
            unsafe { net::person_net256(ts, out, core::ptr::null_mut()) };
            return true;
        }
        false
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (ts, bits, out);
        false
    }
}

/// Measurement only: the session walk over one person's already sorted timestamps.
pub fn person_walk_sorted(ts: &[u32], out: &mut Out) {
    walk_sorted(ts, out, core::ptr::null_mut());
}

/// Measurement only: the slot fold of one person without the session walk (the slots are
/// visited and cleared through the bitmap, nothing else). Returns the occupied slot count.
pub fn person_fold_only(ts: &[u32], sc: &mut Scratch) -> u32 {
    let slots = sc.slots.as_mut_ptr();
    let bits = sc.sbits.as_mut_ptr();
    let mut summary = 0u64;
    for &t in ts {
        let s = (t / SLOT) as usize;
        // SAFETY: t < 2^31, so s < NSLOT and s >> 6 < NWORD (as in person_slots).
        unsafe {
            let e = &mut *slots.add(s);
            e.min = e.min.min(t);
            e.max = e.max.max(t);
            e.cnt += 1;
            *bits.add(s >> 6) |= 1 << (s & 63);
        }
        summary |= 1 << (s >> 6);
    }
    let mut occ = 0;
    while summary != 0 {
        let w = summary.trailing_zeros() as usize;
        summary &= summary - 1;
        // SAFETY: bits of `summary` come from s >> 6 < NWORD; set bits from slots s < NSLOT.
        let mut b = unsafe { core::mem::replace(&mut *bits.add(w), 0) };
        while b != 0 {
            let s = (w << 6) | b.trailing_zeros() as usize;
            b &= b - 1;
            unsafe { *slots.add(s) = SLOT_EMPTY };
            occ += 1;
        }
    }
    occ
}
