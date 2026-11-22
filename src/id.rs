//! ID codes: every non-empty ID string becomes one `u64`, equal exactly when the strings are equal.
//!
//! * Packed form (bit 63 clear): a string of 1 to 10 bytes drawn from `0-9 A-Z _ a-z` is stored
//!   as 6-bit symbols, first byte in the highest bits, 0 as the terminator. The symbols follow
//!   ASCII order and a shorter string has 0 where a longer one has a symbol, so comparing two
//!   packed codes as integers is exactly the byte-wise comparison of their strings. Every ID the
//!   generator writes fits (`u` plus up to 8 base-36 characters, `e...`, `k...`, `p123`, `c_123`,
//!   upper-cased variants too).
//! * Interned form (bit 63 set): anything else (spaces, `\r`, tabs, non-ASCII, longer than 10
//!   bytes) is stored once in a sharded global table and named by its shard and slot.
//! * 0 is the empty string, which the rules treat as null.

use std::cmp::Ordering;
use std::sync::Mutex;

use crate::hash::{FastMap, hash_bytes};

pub type Id = u64;
pub const NULL: Id = 0;
pub const INTERNED: u64 = 1 << 63;
const MAX_PACKED: usize = 10;
const SHARD_BITS: u32 = 6;

const fn symbol_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut next = 1u8;
    let mut byte = 0usize;
    while byte < 256 {
        let c = byte as u8;
        if c.is_ascii_digit() || c.is_ascii_uppercase() || c == b'_' || c.is_ascii_lowercase() {
            table[byte] = next;
            next += 1;
        }
        byte += 1;
    }
    table
}

const fn byte_table() -> [u8; 64] {
    let symbols = symbol_table();
    let mut table = [0u8; 64];
    let mut byte = 0usize;
    while byte < 256 {
        if symbols[byte] != 0 {
            table[symbols[byte] as usize] = byte as u8;
        }
        byte += 1;
    }
    table
}

static SYMBOL: [u8; 256] = symbol_table();
static BYTE: [u8; 64] = byte_table();

/// The packed code of `s`, if it has one. `s` must not be empty.
#[inline]
pub fn pack(s: &[u8]) -> Option<Id> {
    if s.is_empty() || s.len() > MAX_PACKED {
        return None;
    }
    let mut code = 0u64;
    let mut shift = 54u32;
    for &b in s {
        let symbol = SYMBOL[b as usize];
        if symbol == 0 {
            return None;
        }
        code |= u64::from(symbol) << shift;
        shift = shift.wrapping_sub(6);
    }
    Some(code)
}

/// Appends the bytes of a packed code.
#[inline]
pub fn unpack(code: Id, out: &mut Vec<u8>) {
    let mut shift = 54i32;
    while shift >= 0 {
        let symbol = ((code >> shift) & 63) as usize;
        if symbol == 0 {
            break;
        }
        out.push(BYTE[symbol]);
        shift -= 6;
    }
}

#[derive(Default)]
struct Shard {
    slots: FastMap<Box<[u8]>, u32>,
    strings: Vec<Box<[u8]>>,
}

/// The global table of IDs that do not pack. Shared by all parser threads; rare in real data.
pub struct Interner {
    shards: Vec<Mutex<Shard>>,
}

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}

impl Interner {
    pub fn new() -> Self {
        Interner { shards: (0..1usize << SHARD_BITS).map(|_| Mutex::new(Shard::default())).collect() }
    }

    /// The code of an ID string: NULL for empty, packed if possible, else interned.
    #[inline]
    pub fn id(&self, s: &[u8]) -> Id {
        if s.is_empty() {
            return NULL;
        }
        match pack(s) {
            Some(code) => code,
            None => self.intern(s),
        }
    }

    #[cold]
    fn intern(&self, s: &[u8]) -> Id {
        let shard = (hash_bytes(s) >> (64 - SHARD_BITS)) as usize;
        let mut guard = self.shards[shard].lock().unwrap();
        let slot = match guard.slots.get(s) {
            Some(&slot) => slot,
            None => {
                let slot = guard.strings.len() as u32;
                guard.strings.push(s.into());
                guard.slots.insert(s.into(), slot);
                slot
            }
        };
        INTERNED | ((shard as u64) << 32) | u64::from(slot)
    }

    /// Appends the original bytes of a non-null code.
    pub fn append(&self, code: Id, out: &mut Vec<u8>) {
        if code & INTERNED == 0 {
            unpack(code, out);
        } else {
            let shard = ((code >> 32) & ((1 << SHARD_BITS) - 1)) as usize;
            let slot = (code & 0xffff_ffff) as usize;
            let guard = self.shards[shard].lock().unwrap();
            out.extend_from_slice(&guard.strings[slot]);
        }
    }

    pub fn bytes(&self, code: Id) -> Vec<u8> {
        let mut out = Vec::new();
        self.append(code, &mut out);
        out
    }

    /// Byte-wise comparison of the strings behind two codes.
    #[inline]
    pub fn cmp(&self, a: Id, b: Id) -> Ordering {
        if (a | b) & INTERNED == 0 {
            a.cmp(&b)
        } else {
            self.cmp_slow(a, b)
        }
    }

    #[cold]
    fn cmp_slow(&self, a: Id, b: Id) -> Ordering {
        if a == b {
            return Ordering::Equal;
        }
        self.bytes(a).cmp(&self.bytes(b))
    }

    pub fn interned_count(&self) -> usize {
        self.shards.iter().map(|s| s.lock().unwrap().strings.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order_matches(a: &[u8], b: &[u8]) {
        let ids = Interner::new();
        let (x, y) = (ids.id(a), ids.id(b));
        assert_eq!(ids.cmp(x, y), a.cmp(b), "{:?} vs {:?}", a, b);
        assert_eq!(x == y, a == b);
        assert_eq!(ids.bytes(x), a);
    }

    #[test]
    fn packed_order_is_byte_order() {
        let cases: &[&[u8]] = &[
            b"u1", b"u10", b"u9", b"U1", b"u", b"uzzzzzzzz", b"u00000000", b"u0", b"c_1", b"c_10",
            b"c_2", b"A", b"Z", b"_", b"a", b"0", b"9", b"user_id_a", b"user_id_b", b"zzzzzzzzzz",
            b" u1", b"u1 ", b"e3\r", b"A\tB", b"u1234567890", b"\xd9\xa2",
        ];
        for a in cases {
            for b in cases {
                order_matches(a, b);
            }
        }
    }

    #[test]
    fn interned_codes_are_stable() {
        let ids = Interner::new();
        let a = ids.id(b" u1");
        let b = ids.id(b" u1");
        let c = ids.id(b"u1 ");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a & INTERNED != 0);
        assert_eq!(ids.id(b""), NULL);
    }
}
