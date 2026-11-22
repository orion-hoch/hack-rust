//! Person names as order keys. A conforming ID is a prefix letter and 1 to 8 base-36 digits of a
//! value x < 2^40. Its order key puts the digits left-aligned in 8 places, shifts left 4 bits and
//! adds the length (the RECORDS.md ckey formula). For one prefix, integer order of order keys is
//! byte order of the strings. Numbers themselves are never compared in place of strings.

pub const POW36: [u64; 9] = [
    1,
    36,
    1_296,
    46_656,
    1_679_616,
    60_466_176,
    2_176_782_336,
    78_364_164_096,
    2_821_109_907_456,
];

const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";

/// Order key of the canonical base-36 string of `x` (x < 2^40, so at most 8 digits).
#[inline(always)]
pub fn order_key(x: u64) -> u64 {
    let mut len = 1usize;
    while len < 8 && x >= POW36[len] {
        len += 1;
    }
    ((x * POW36[8 - len]) << 4) | len as u64
}

/// Appends `prefix` and the digits of an order key.
#[inline]
pub fn append_bytes(prefix: u8, key: u64, out: &mut Vec<u8>) {
    let (buf, n) = key_bytes(prefix, key);
    out.extend_from_slice(&buf[..n]);
}

/// The bytes of an order key with its prefix, on the stack: (buffer, length). One division by a
/// table power, then base-36 digits with constant divisors.
#[inline]
pub fn key_bytes(prefix: u8, key: u64) -> ([u8; 9], usize) {
    let mut buf = [0u8; 9];
    buf[0] = prefix;
    let len = (key & 15) as usize;
    let mut x = (key >> 4) / POW36[8 - len];
    let mut at = len;
    while at > 0 {
        buf[at] = DIGITS[(x % 36) as usize];
        x /= 36;
        at -= 1;
    }
    (buf, 1 + len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b36(mut x: u64) -> Vec<u8> {
        let d = b"0123456789abcdefghijklmnopqrstuvwxyz";
        let mut v = Vec::new();
        loop {
            v.push(d[(x % 36) as usize]);
            x /= 36;
            if x == 0 {
                break;
            }
        }
        v.reverse();
        v
    }

    #[test]
    fn order_key_is_byte_order_and_round_trips() {
        let mut values = vec![0u64, 1, 9, 10, 35, 36, 37, 1295, 1296, 46655, (1 << 40) - 1, 123_456_789];
        let mut s = 0x1234_5678_9abc_def0u64;
        for _ in 0..2000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            values.push((s >> 20) & ((1 << 40) - 1));
            values.push((s >> 40) & 0xffff);
        }
        let strings: Vec<Vec<u8>> = values.iter().map(|&x| b36(x)).collect();
        for (i, &x) in values.iter().enumerate() {
            let mut back = Vec::new();
            append_bytes(b'u', order_key(x), &mut back);
            assert_eq!(&back[1..], strings[i].as_slice());
            for (j, &y) in values.iter().enumerate().take(300) {
                assert_eq!(order_key(x).cmp(&order_key(y)), strings[i].cmp(&strings[j]));
            }
        }
    }
}
