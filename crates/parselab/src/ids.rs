//! IDs as opaque keys (RECORDS.md v3): SHORT (prefix + 1..8 canonical base-36 characters, value
//! below 2^40), LONG (prefix + exactly 32 lowercase hex characters), FOREIGN (anything else).

pub const MASK40: u64 = (1 << 40) - 1;

pub const BASE36: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";

/// Lowercase canonical base 36 of `value` appended to `out`.
pub fn base36(out: &mut Vec<u8>, value: u64) {
    let mut buffer = [0u8; 13];
    let mut at = buffer.len();
    let mut rest = value;
    loop {
        at -= 1;
        buffer[at] = BASE36[(rest % 36) as usize];
        rest /= 36;
        if rest == 0 {
            break;
        }
    }
    out.extend_from_slice(&buffer[at..]);
}

#[inline(always)]
fn digit36(c: u8) -> Option<u64> {
    match c {
        b'0'..=b'9' => Some((c - b'0') as u64),
        b'a'..=b'z' => Some((c - b'a' + 10) as u64),
        _ => None,
    }
}

/// RECORDS.md "Conforming IDs": `prefix` then 1 to 8 characters of `0-9a-z`, canonical (no leading
/// `0` unless the whole number is `0`), value below 2^40. Returns the base-36 value (still scrambled).
pub fn conforming(id: &[u8], prefix: u8) -> Option<u64> {
    let (&first, digits) = id.split_first()?;
    if first != prefix || digits.is_empty() || digits.len() > 8 {
        return None;
    }
    if digits.len() > 1 && digits[0] == b'0' {
        return None;
    }
    let mut value = 0u64;
    for &c in digits {
        value = value * 36 + digit36(c)?;
    }
    (value <= MASK40).then_some(value)
}

/// RECORDS.md v2 click order key: a conforming click_id's digits left-aligned in 8 base-36 places,
/// shifted left 4 bits, plus the length. Integer order of ckeys is byte order of the click_ids.
pub fn ckey(click_id: &[u8]) -> Option<u64> {
    let value = conforming(click_id, b'k')?;
    let len = click_id.len() as u32 - 1;
    Some((value * 36u64.pow(8 - len)) << 4 | len as u64)
}

fn hex16(s: &[u8]) -> Option<u64> {
    let mut v = 0u64;
    for &c in s {
        let d = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => return None,
        };
        v = (v << 4) | d as u64;
    }
    Some(v)
}

/// The key of a non-empty ID: SHORT, LONG, or None for FOREIGN.
pub fn key_of(id: &[u8], prefix: u8) -> Option<crate::records::Key> {
    if let Some(v) = conforming(id, prefix) {
        return Some(crate::records::Key { long: false, a: v, b: 0 });
    }
    if id.len() == 33 && id[0] == prefix {
        let a = hex16(&id[1..17])?;
        let b = hex16(&id[17..33])?;
        return Some(crate::records::Key { long: true, a, b });
    }
    None
}

/// The ID bytes of a key (the inverse of `key_of`).
pub fn key_bytes(prefix: u8, k: crate::records::Key) -> Vec<u8> {
    let mut out = vec![prefix];
    if k.long {
        out.extend_from_slice(format!("{:016x}{:016x}", k.a, k.b).as_bytes());
    } else {
        base36(&mut out, k.a);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ckey_order_is_byte_order() {
        let mut ids: Vec<Vec<u8>> = Vec::new();
        let mut state = 7u64;
        for len in 1..=8u32 {
            for _ in 0..300 {
                state = crate::records::mix64(state + 1);
                let mut id = vec![b'k'];
                for i in 0..len {
                    let d = (state >> (5 * i)) % 36;
                    id.push(if i == 0 && len > 1 && d == 0 { b'1' } else { BASE36[d as usize] });
                }
                if conforming(&id, b'k').is_some() {
                    ids.push(id);
                }
            }
        }
        ids.extend([b"k0".to_vec(), b"k1".to_vec(), b"k10".to_vec(), b"k100".to_vec(), b"kz".to_vec(), b"k9".to_vec()]);
        for a in &ids {
            for b in ids.iter().take(400) {
                assert_eq!(ckey(a).unwrap().cmp(&ckey(b).unwrap()), a.cmp(b), "{a:?} {b:?}");
            }
            assert!(ckey(a).unwrap() < 1 << 46);
        }
    }

    #[test]
    fn keys() {
        use crate::records::Key;
        assert_eq!(key_of(b"e0", b'e'), Some(Key { long: false, a: 0, b: 0 }));
        assert_eq!(key_of(b"e00", b'e'), None);
        let long = b"e514442f69928d7b6568bd6397145fb0b";
        let k = key_of(long, b'e').unwrap();
        assert!(k.long && k.a == 0x514442f69928d7b6 && k.b == 0x568bd6397145fb0b);
        assert_eq!(key_bytes(b'e', k), long.to_vec());
        assert_eq!(key_of(b"e514442f69928d7b6568bd6397145fb0", b'e'), None); // 31 hex
        assert_eq!(key_of(b"e514442f69928d7b6568bd6397145fb0bb", b'e'), None); // 33 hex
        assert_eq!(key_of(b"e514442F69928d7b6568bd6397145fb0b", b'e'), None); // upper case
        assert_eq!(key_of(b"e514442g69928d7b6568bd6397145fb0b", b'e'), None);
        assert_eq!(key_of(b"u514442f69928d7b6568bd6397145fb0b", b'e'), None);
        for id in [&b"e1"[..], b"ezzzzzzz", b"e10"] {
            assert_eq!(key_bytes(b'e', key_of(id, b'e').unwrap()), id.to_vec());
        }
    }

    #[test]
    fn conformance() {
        assert_eq!(conforming(b"e0", b'e'), Some(0));
        assert_eq!(conforming(b"e00", b'e'), None);
        assert_eq!(conforming(b"e01", b'e'), None);
        assert_eq!(conforming(b"e", b'e'), None);
        assert_eq!(conforming(b"E1", b'e'), None);
        assert_eq!(conforming(b"e1A", b'e'), None);
        assert_eq!(conforming(b"u1", b'e'), None);
        assert_eq!(conforming(b"ezzzzzzzz", b'e'), None); // 36^8 - 1 > 2^40
        assert_eq!(conforming(b"ezzzzzzz", b'e'), Some(36u64.pow(7) - 1));
        let max = {
            let mut v = Vec::new();
            base36(&mut v, MASK40);
            v
        };
        let mut id = vec![b'e'];
        id.extend_from_slice(&max);
        assert_eq!(conforming(&id, b'e'), Some(MASK40));
        let mut over = Vec::new();
        base36(&mut over, MASK40 + 1);
        let mut id = vec![b'e'];
        id.extend_from_slice(&over);
        assert_eq!(conforming(&id, b'e'), None);
        assert_eq!(conforming(b"e123456789", b'e'), None);
    }
}
