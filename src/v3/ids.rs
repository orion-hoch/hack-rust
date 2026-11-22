//! IDs as opaque keys (RECORDS.md v3): SHORT (prefix + 1..8 canonical base-36 characters, value
//! below 2^40), LONG (prefix + exactly 32 lowercase hex characters), FOREIGN (anything else).
//! Original bytes can be rebuilt from kind and key; byte order is compared on the rebuilt bytes,
//! with two shortcuts that are exact: SHORT against SHORT by the left-aligned base-36 order key,
//! LONG against LONG by the 128-bit value (fixed width, 0-9 sorts before a-f).

use std::cmp::Ordering;

use crate::fast::names::order_key;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Key {
    Short(u64),
    Long(u64, u64),
    Foreign,
}

#[inline(always)]
fn b36(c: u8) -> Option<u64> {
    match c {
        b'0'..=b'9' => Some(u64::from(c - b'0')),
        b'a'..=b'z' => Some(u64::from(c - b'a') + 10),
        _ => None,
    }
}

#[inline(always)]
fn hex(c: u8) -> Option<u64> {
    match c {
        b'0'..=b'9' => Some(u64::from(c - b'0')),
        b'a'..=b'f' => Some(u64::from(c - b'a') + 10),
        _ => None,
    }
}

/// Classifies a non-empty ID with the given prefix letter.
pub fn classify(id: &[u8], prefix: u8) -> Key {
    let Some((&first, digits)) = id.split_first() else { return Key::Foreign };
    if first != prefix {
        return Key::Foreign;
    }
    if digits.len() == 32 {
        let mut a = 0u64;
        let mut b = 0u64;
        for (i, &c) in digits.iter().enumerate() {
            let Some(d) = hex(c) else { return Key::Foreign };
            if i < 16 {
                a = (a << 4) | d;
            } else {
                b = (b << 4) | d;
            }
        }
        return Key::Long(a, b);
    }
    if digits.is_empty() || digits.len() > 8 || (digits.len() > 1 && digits[0] == b'0') {
        return Key::Foreign;
    }
    let mut v = 0u64;
    for &c in digits {
        let Some(d) = b36(c) else { return Key::Foreign };
        v = v * 36 + d;
    }
    if v < (1u64 << 40) { Key::Short(v) } else { Key::Foreign }
}

const DIG36: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
const HEX: &[u8; 16] = b"0123456789abcdef";

/// Appends the bytes of a SHORT ID.
pub fn short_bytes(prefix: u8, v: u64, out: &mut Vec<u8>) {
    out.push(prefix);
    let mut buf = [0u8; 13];
    let mut at = buf.len();
    let mut x = v;
    loop {
        at -= 1;
        buf[at] = DIG36[(x % 36) as usize];
        x /= 36;
        if x == 0 {
            break;
        }
    }
    out.extend_from_slice(&buf[at..]);
}

/// Appends the bytes of a LONG ID.
pub fn long_bytes(prefix: u8, a: u64, b: u64, out: &mut Vec<u8>) {
    out.push(prefix);
    for w in [a, b] {
        for i in (0..16).rev() {
            out.push(HEX[((w >> (4 * i)) & 15) as usize]);
        }
    }
}

/// Byte order of two SHORT IDs of one prefix.
#[inline(always)]
pub fn cmp_short(x: u64, y: u64) -> Ordering {
    order_key(x).cmp(&order_key(y))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(k: Key) -> Vec<u8> {
        let mut v = Vec::new();
        match k {
            Key::Short(x) => short_bytes(b'u', x, &mut v),
            Key::Long(a, b) => long_bytes(b'u', a, b, &mut v),
            Key::Foreign => {}
        }
        v
    }

    #[test]
    fn classify_and_rebuild() {
        for s in [&b"u0"[..], b"uzzzzzzz", b"u1", b"u10", b"uabc", b"u0123456789abcdef0123456789abcdef", b"uffffffffffffffffffffffffffffffff"] {
            let k = classify(s, b'u');
            assert_ne!(k, Key::Foreign, "{:?}", String::from_utf8_lossy(s));
            assert_eq!(bytes(k), s);
        }
        for s in [&b"U1"[..], b"u", b"u01", b"u1234567890", b"e1", b"u0123456789abcdef0123456789abcdeF", b"u0123456789abcdef0123456789abcdeg", b"uzzzzzzzz"] {
            assert_eq!(classify(s, b'u'), Key::Foreign, "{:?}", String::from_utf8_lossy(s));
        }
        // 2^40 - 1 is SHORT, 2^40 is not.
        let mut s = Vec::new();
        short_bytes(b'u', (1 << 40) - 1, &mut s);
        assert_eq!(classify(&s, b'u'), Key::Short((1 << 40) - 1));
        let mut s = Vec::new();
        short_bytes(b'u', 1 << 40, &mut s);
        assert_eq!(classify(&s, b'u'), Key::Foreign);
    }

    #[test]
    fn shortcuts_are_byte_order() {
        let ids: Vec<&[u8]> = vec![b"u0", b"u1", b"u10", b"u9", b"uz", b"uzz", b"u00000000000000000000000000000000",
            b"u0123456789abcdef0123456789abcdef", b"u9", b"ua", b"ua0000000000000000000000000000000",
            b"uffffffffffffffffffffffffffffffff", b"u10000000000000000000000000000000"];
        for x in &ids {
            for y in &ids {
                let (kx, ky) = (classify(x, b'u'), classify(y, b'u'));
                match (kx, ky) {
                    (Key::Short(a), Key::Short(b)) => assert_eq!(cmp_short(a, b), x.cmp(y)),
                    (Key::Long(a1, b1), Key::Long(a2, b2)) => assert_eq!((a1, b1).cmp(&(a2, b2)), x.cmp(y)),
                    _ => {}
                }
            }
        }
    }
}
