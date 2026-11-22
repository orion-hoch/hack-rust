//! v4 keys (RECORDS-V4.md section 1): the one copy of the SHORT-key mix and of the LONG hash, and the
//! partition bits (section 2). Every other copy (vector code) is tested against these.

pub const M40: u64 = (1 << 40) - 1;
/// Odd, below 2^40, unrelated to the golden-ratio constant dedup hashes with inside a partition.
pub const MIX_C: u64 = 0xC2_B2AE_3D27;
/// MIX_C * MIX_CINV = 1 mod 2^40.
pub const MIX_CINV: u64 = 0x14_BE47_6297;
const LONG_C: u64 = 0xD6E8_FEB8_6659_FD93;

/// A bijection on [0, 2^40): odd multiply mod 2^40, then a reversible xor-shift (shift 20 of 40 bits).
#[inline(always)]
pub fn mix40(v: u64) -> u64 {
    let x = v.wrapping_mul(MIX_C) & M40;
    x ^ (x >> 20)
}

/// The inverse of mix40 (for names): x ^ (x >> 20) undoes itself on 40 bits, then the inverse multiply.
#[inline(always)]
pub fn unmix40(m: u64) -> u64 {
    let x = m ^ (m >> 20);
    x.wrapping_mul(MIX_CINV) & M40
}

/// The 64-bit hash of a LONG (or interned FOREIGN) key; its top bits are the partition.
#[inline(always)]
pub fn long_hash(a: u64, b: u64) -> u64 {
    (a ^ b.rotate_left(29)).wrapping_mul(LONG_C)
}

/// (b1, b2): event and user partition bits from the events' bytes (RECORDS-V4.md section 2).
pub fn part_bits(events_bytes: u64) -> (u32, u32) {
    let n = events_bytes / 70 + 1;
    let lg = |x: u64| if x <= 1 { 0 } else { 64 - (x - 1).leading_zeros() };
    // User partitions: measured on box 2 (8 workers), 2^9 at 20M and 2^11 at 60M are best. More
    // of them cost memory, not time saved: every worker keeps an open 8 KB chunk per partition
    // and stream (at 60M with 2^13: 6.6 GB peak and 49.3G cycles against 5.3 GB and 47.3G).
    let (b1, b2) = (lg(n / 32768).clamp(8, 11), lg(n / 65536).clamp(9, 12));
    // Measurement only: ENGINE_B1 / ENGINE_B2 override the partition bits.
    let ov = |k: &str, d: u32| std::env::var(k).ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(d);
    (ov("ENGINE_B1", b1), ov("ENGINE_B2", b2))
}

/// FOREIGN ids interned at or past this index send the run to v3 (RECORDS-V4.md section 9).
pub const FOREIGN_MAX: u32 = 1 << 20;

type Interned = (std::collections::HashMap<&'static [u8], u32>, Vec<&'static [u8]>);
static FOREIGN: std::sync::Mutex<Option<Interned>> = std::sync::Mutex::new(None);

/// The run-wide index f of a FOREIGN id (RECORDS-V4.md section 1: its key is (f | 1 << 63, 0)).
/// One locked map for every stage (FOREIGN ids are dirt: rare). Returns FOREIGN_MAX, and stores
/// nothing, once FOREIGN_MAX ids are interned: the caller falls back.
pub fn intern(id: &[u8]) -> u32 {
    let mut g = FOREIGN.lock().unwrap_or_else(|e| e.into_inner());
    let (map, list) = g.get_or_insert_with(Default::default);
    if let Some(&f) = map.get(id) {
        return f;
    }
    if list.len() as u32 >= FOREIGN_MAX {
        return FOREIGN_MAX;
    }
    let f = list.len() as u32;
    // Kept for the run (the process exits without destructors).
    let b: &'static [u8] = Box::leak(id.to_vec().into_boxed_slice());
    list.push(b);
    map.insert(b, f);
    f
}

/// FOREIGN ids interned so far.
pub fn count() -> usize {
    let g = FOREIGN.lock().unwrap_or_else(|e| e.into_inner());
    g.as_ref().map_or(0, |(_, list)| list.len())
}

/// The bytes of FOREIGN id f (from `intern`).
pub fn bytes(f: u32) -> &'static [u8] {
    let g = FOREIGN.lock().unwrap_or_else(|e| e.into_inner());
    g.as_ref().and_then(|(_, list)| list.get(f as usize).copied()).unwrap_or(&[])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix_is_a_bijection_with_its_inverse() {
        assert_eq!(MIX_C.wrapping_mul(MIX_CINV) & M40, 1);
        assert_eq!(mix40(1), 0xC2_B2A2_160D);
        assert_eq!(mix40(2), 0x85_6554_2C1B);
        assert_eq!(mix40(M40), 0x3D_4D52_160C);
        assert_eq!(mix40(0x12_3456_789A), 0x93_A09C_2B7F);
        let mut v = 0x1234_5678u64;
        for _ in 0..200_000 {
            v = v.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let k = v >> 24;
            assert!(mix40(k) <= M40);
            assert_eq!(unmix40(mix40(k)), k);
        }
        for k in [0, 1, 2, M40] {
            assert_eq!(unmix40(mix40(k)), k);
        }
    }

    #[test]
    fn partition_bits() {
        assert_eq!(part_bits(1_762_906_114), (10, 9)); // p20v5 events
        assert_eq!(part_bits(5_400_000_000), (11, 11)); // about f60v5 (events bytes)
        assert_eq!(part_bits(1000), (8, 9));
    }
}
