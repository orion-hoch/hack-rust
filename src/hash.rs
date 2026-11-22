//! Hashing for integer ID codes and short byte strings.
//!
//! Two different mixers on purpose: `bucket` picks a radix partition from the high bits, and
//! `FastHasher` feeds hash maps that live inside one partition. If both used the same function,
//! every key in a partition would share the top bits the map uses for its control bytes.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// MurmurHash3's 64-bit finalizer.
#[inline]
pub fn fmix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

/// SplitMix64's finalizer.
#[inline]
fn splitmix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The radix partition of `key` among `1 << bits` partitions.
#[inline]
pub fn bucket(key: u64, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        (fmix(key) >> (64 - bits)) as usize
    }
}

/// FNV-1a over bytes, finished with SplitMix64.
#[inline]
pub fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    splitmix(h)
}

#[derive(Default, Clone, Copy)]
pub struct FastHasher(u64);

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.0 = splitmix(self.0 ^ hash_bytes(bytes));
    }
    #[inline]
    fn write_u64(&mut self, x: u64) {
        self.0 = splitmix(self.0 ^ x);
    }
    #[inline]
    fn write_u32(&mut self, x: u32) {
        self.write_u64(u64::from(x));
    }
    #[inline]
    fn write_usize(&mut self, x: usize) {
        self.write_u64(x as u64);
    }
}

pub type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

pub fn map_with_capacity<K, V>(n: usize) -> FastMap<K, V> {
    FastMap::with_capacity_and_hasher(n, BuildHasherDefault::default())
}
