//! v4 link kernel: a link line is two IDs and one comma, so it needs none of the event framing.
//! One 64-byte window per line gives the line end, the comma and the quotes as bit masks; the two
//! fields are checked by mask arithmetic and decoded by table; the line goes into a LinkBatch for
//! the sink. Anything else (no newline in the window, not exactly one comma, quoting other than a
//! whole field, an empty or FOREIGN side, a line in a file's last 64 bytes) goes through the
//! reference `link_one`, so the semantics are the reference's by construction: lines split on every
//! comma, a field with a quote must be "..." with none inside, exactly two non-empty fields.

use parselab::{Ctx, Key, LinkBatch, Sink};

use super::links::LinkSink;

const BAD: u8 = 0xFF;

/// Base-36 digit values of `0-9a-z` (BAD elsewhere).
static DIG36: [u8; 256] = {
    let mut t = [BAD; 256];
    let mut c = 0;
    while c < 10 {
        t[b'0' as usize + c] = c as u8;
        c += 1;
    }
    let mut c = 0;
    while c < 26 {
        t[b'a' as usize + c] = 10 + c as u8;
        c += 1;
    }
    t
};

/// Hex digit values of `0-9a-f` (BAD elsewhere).
static HEX: [u8; 256] = {
    let mut t = [BAD; 256];
    let mut c = 0;
    while c < 16 {
        t[b"0123456789abcdef"[c] as usize] = c as u8;
        c += 1;
    }
    t
};

const MASK40: u64 = (1 << 40) - 1;

/// A conforming SHORT user id (`u`, 1 to 8 base-36 digits, canonical, value < 2^40) or a LONG
/// one (`u`, 32 hex digits); None for anything else (FOREIGN; the caller takes the reference path).
#[inline(always)]
fn key(f: &[u8]) -> Option<Key> {
    let n = f.len();
    if n < 2 || f[0] != b'u' {
        return None;
    }
    if n <= 9 {
        if n > 2 && f[1] == b'0' {
            return None;
        }
        let mut v = 0u64;
        let mut bad = 0u8;
        for &c in &f[1..] {
            let d = DIG36[c as usize];
            bad |= d;
            v = v * 36 + u64::from(d);
        }
        // A BAD digit has its high bit set; a valid one is below 36.
        return (bad & 0x80 == 0 && v <= MASK40).then_some(Key { long: false, a: v, b: 0 });
    }
    if n == 33 {
        let (mut a, mut b, mut bad) = (0u64, 0u64, 0u8);
        for i in 0..16 {
            let (x, y) = (HEX[f[1 + i] as usize], HEX[f[17 + i] as usize]);
            bad |= x | y;
            a = (a << 4) | u64::from(x & 15);
            b = (b << 4) | u64::from(y & 15);
        }
        return (bad & 0x80 == 0).then_some(Key { long: true, a, b });
    }
    None
}

/// Newline, comma and quote masks of the 64 bytes at `p`.
#[inline(always)]
unsafe fn masks(p: *const u8) -> (u64, u64, u64) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // SAFETY: the caller guarantees 64 readable bytes at p.
    unsafe {
        use core::arch::x86_64::*;
        let (v0, v1) = (_mm256_loadu_si256(p as *const __m256i), _mm256_loadu_si256(p.add(32) as *const __m256i));
        let m = |c: u8| {
            let k = _mm256_set1_epi8(c as i8);
            (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, k)) as u32 as u64) | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, k)) as u32 as u64) << 32)
        };
        (m(b'\n'), m(b','), m(b'"'))
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    // SAFETY: as above.
    unsafe {
        let (mut nl, mut cm, mut qm) = (0u64, 0u64, 0u64);
        for i in 0..64 {
            let c = *p.add(i);
            nl |= u64::from(c == b'\n') << i;
            cm |= u64::from(c == b',') << i;
            qm |= u64::from(c == b'"') << i;
        }
        (nl, cm, qm)
    }
}

/// The value of a field [lo, hi) of the line given its quote bits (relative to the line): the
/// whole field quoted, or no quote at all; None for any other quoting (or an empty value).
#[inline(always)]
fn field(lo: usize, hi: usize, q: u64) -> Option<(usize, usize)> {
    let span = if hi > lo { (u64::MAX >> (64 - (hi - lo))) << lo } else { 0 };
    let fq = q & span;
    if fq == 0 {
        return (hi > lo).then_some((lo, hi));
    }
    // Quoted: exactly the first and the last byte, at least one byte inside.
    (hi >= lo + 3 && fq == ((1u64 << lo) | (1u64 << (hi - 1)))).then_some((lo + 1, hi - 1))
}

/// Every link line whose first byte is in from..to (the header at offset 0 excluded).
pub fn chunk(ctx: &Ctx, file: u32, data: &[u8], from: usize, to: usize, sink: &mut LinkSink) {
    let len = data.len();
    let to = to.min(len);
    if from >= to {
        return;
    }
    let nl_after = |i: usize| data[i..].iter().position(|&c| c == b'\n').map_or(len, |p| i + p);
    let mut start = if from == 0 { nl_after(0) + 1 } else { nl_after(from - 1) + 1 };
    let mut lb = LinkBatch::default();
    let mut fields = Vec::with_capacity(4);
    let base = data.as_ptr();
    while start < to {
        let mut next = None;
        if start + 64 <= len {
            // SAFETY: 64 bytes at start are inside data.
            let (nl, cm, qm) = unsafe { masks(base.add(start)) };
            if nl != 0 {
                let n = nl.trailing_zeros() as usize;
                let inl = (1u64 << n) - 1;
                let (cm, qm) = (cm & inl, qm & inl);
                if cm.count_ones() == 1 {
                    let c = cm.trailing_zeros() as usize;
                    if let (Some((a0, a1)), Some((b0, b1))) = (field(0, c, qm), field(c + 1, n, qm)) {
                        let line = &data[start..start + n];
                        if let (Some(a), Some(b)) = (key(&line[a0..a1]), key(&line[b0..b1])) {
                            lb.push(a, b);
                            if lb.n == 8 {
                                sink.links(&lb);
                                lb = LinkBatch::default();
                            }
                            next = Some(start + n + 1);
                        }
                    }
                }
            }
        }
        start = match next {
            Some(s) => s,
            None => {
                // The reference, for this one line (drops it, delivers it, or sends it by locator).
                let end = nl_after(start);
                parselab::reference::link_one(ctx, file, start, &data[start..end], &mut fields, sink);
                end + 1
            }
        };
    }
    if lb.n > 0 {
        sink.links(&lb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fast fields and keys agree with the reference splitter and classifier on every
    /// shape a line can take, valid or not.
    #[test]
    fn fields_and_keys_match_the_reference() {
        let ids: [&[u8]; 14] = [
            b"u0", b"u1", b"u00", b"uzzzzzzzz", b"u1x2y3z4w", b"uabcdefgh", b"uABC", b"u",
            b"u36bd2c2d49c8c2510c9bf35ff4e8a398", b"u36bd2c2d49c8c2510c9bf35ff4e8a39g", b"", b"x123", b"u12345678", b"u7oso278q",
        ];
        for a in ids {
            for b in ids {
                for qa in 0..3 {
                    for qb in 0..3 {
                        let wrap = |f: &[u8], q: usize| -> Vec<u8> {
                            match q {
                                0 => f.to_vec(),
                                1 => [b"\"", f, b"\""].concat(),
                                _ => [f, b"\""].concat(),
                            }
                        };
                        let line = [wrap(a, qa), b",".to_vec(), wrap(b, qb)].concat();
                        // Reference: split, unquote, two non-empty fields, both keyed.
                        let mut fv = Vec::new();
                        let want = if parselab::csv::split(&line, &mut fv) && fv.len() == 2 && !fv[0].is_empty() && !fv[1].is_empty() {
                            match (parselab::ids::key_of(fv[0], b'u'), parselab::ids::key_of(fv[1], b'u')) {
                                (Some(x), Some(y)) => Some((x, y)),
                                _ => None,
                            }
                        } else {
                            None
                        };
                        // Fast: masks over the line, then fields and keys. A line of 64 bytes or more
                        // never takes the fast path (its newline is not in the window).
                        if line.len() >= 64 {
                            continue;
                        }
                        let (mut cm, mut qm) = (0u64, 0u64);
                        for (i, &ch) in line.iter().enumerate() {
                            cm |= u64::from(ch == b',') << i;
                            qm |= u64::from(ch == b'"') << i;
                        }
                        let got = if cm.count_ones() == 1 {
                            let c = cm.trailing_zeros() as usize;
                            match (field(0, c, qm), field(c + 1, line.len(), qm)) {
                                (Some((a0, a1)), Some((b0, b1))) => match (key(&line[a0..a1]), key(&line[b0..b1])) {
                                    (Some(x), Some(y)) => Some((x, y)),
                                    _ => None,
                                },
                                _ => None,
                            }
                        } else {
                            None
                        };
                        // The fast path may only decline (the reference then decides), never differ.
                        if let Some(g) = got {
                            assert_eq!(Some(g), want, "line {:?}", String::from_utf8_lossy(&line));
                        }
                        if want.is_some() {
                            assert_eq!(got, want, "line {:?}", String::from_utf8_lossy(&line));
                        }
                    }
                }
            }
        }
    }
}
