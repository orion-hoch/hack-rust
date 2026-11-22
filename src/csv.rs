//! The CSV dialect of rules.md, to the byte.
//!
//! * Only `\n` ends a line; `\r` is data. The last line of a file may lack its `\n`.
//! * Every comma separates fields. A field that holds a `"` anywhere must be exactly `"` + text
//!   without quotes + `"`, and its value is the text between; any other quote makes the line
//!   malformed. `""` and an empty field are both empty, which means null.
//! * A line with the wrong number of fields is malformed, and so is an empty line.
//! * Nothing is trimmed.

/// Index of the first `needle` in `hay`. Hand-vectorised: 32 bytes a step with AVX2 when the
/// build enables it, 16 with SSE2 otherwise on x86-64, 16 with NEON on aarch64, 8 by SWAR elsewhere.
#[inline]
pub fn find_byte(needle: u8, hay: &[u8]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is part of x86-64; the AVX2 path is compiled only when the target has it.
        unsafe { find_x86(needle, hay) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of aarch64.
        unsafe { find_neon(needle, hay) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        find_swar(needle, hay, 0)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn find_x86(needle: u8, hay: &[u8]) -> Option<usize> {
    use core::arch::x86_64::*;
    let (p, n) = (hay.as_ptr(), hay.len());
    let mut i = 0usize;
    #[cfg(target_feature = "avx2")]
    unsafe {
        let v = _mm256_set1_epi8(needle as i8);
        while i + 64 <= n {
            let a = _mm256_cmpeq_epi8(_mm256_loadu_si256(p.add(i) as *const __m256i), v);
            let b = _mm256_cmpeq_epi8(_mm256_loadu_si256(p.add(i + 32) as *const __m256i), v);
            if _mm256_movemask_epi8(_mm256_or_si256(a, b)) != 0 {
                let ma = _mm256_movemask_epi8(a) as u32 as u64;
                let mb = _mm256_movemask_epi8(b) as u32 as u64;
                return Some(i + (ma | mb << 32).trailing_zeros() as usize);
            }
            i += 64;
        }
        if i + 32 <= n {
            let m = _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(p.add(i) as *const __m256i), v)) as u32;
            if m != 0 {
                return Some(i + m.trailing_zeros() as usize);
            }
            i += 32;
        }
    }
    unsafe {
        let v = _mm_set1_epi8(needle as i8);
        while i + 16 <= n {
            let m = _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(p.add(i) as *const __m128i), v)) as u32;
            if m != 0 {
                return Some(i + m.trailing_zeros() as usize);
            }
            i += 16;
        }
    }
    find_swar(needle, hay, i)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn find_neon(needle: u8, hay: &[u8]) -> Option<usize> {
    use core::arch::aarch64::*;
    let (p, n) = (hay.as_ptr(), hay.len());
    let mut i = 0usize;
    unsafe {
        let v = vdupq_n_u8(needle);
        while i + 16 <= n {
            let eq = vceqq_u8(vld1q_u8(p.add(i)), v);
            if vmaxvq_u8(eq) != 0 {
                // 4 bits per byte: narrow the 0xFF/0x00 lanes to a 64-bit mask
                let m = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(eq), 4)), 0);
                return Some(i + (m.trailing_zeros() / 4) as usize);
            }
            i += 16;
        }
    }
    find_swar(needle, hay, i)
}

/// 8 bytes a step from `from`, then byte by byte.
#[inline(always)]
fn find_swar(needle: u8, hay: &[u8], from: usize) -> Option<usize> {
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    let rep = LO.wrapping_mul(needle as u64);
    let mut i = from;
    while i + 8 <= hay.len() {
        let x = u64::from_le_bytes(hay[i..i + 8].try_into().unwrap()) ^ rep;
        // exact zero-byte test (no false positives above a true zero, since we take the lowest)
        let t = x.wrapping_sub(LO) & !x & HI;
        if t != 0 {
            return Some(i + (t.trailing_zeros() / 8) as usize);
        }
        i += 8;
    }
    hay[i..].iter().position(|&b| b == needle).map(|k| i + k)
}

/// Index of the first occurrence of `needle` (non-empty) in `hay`.
pub fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    let (&first, rest) = needle.split_first()?;
    let mut from = 0;
    while from + needle.len() <= hay.len() {
        let i = from + find_byte(first, &hay[from..hay.len() - rest.len()])?;
        if &hay[i + 1..i + needle.len()] == rest {
            return Some(i);
        }
        from = i + 1;
    }
    None
}

/// Calls `f` on every line of `data` (without its `\n`). `data` must start at a line start.
#[inline]
pub fn for_each_line<'a>(data: &'a [u8], mut f: impl FnMut(&'a [u8])) {
    let mut pos = 0;
    let end = data.len();
    while pos < end {
        match find_byte(b'\n', &data[pos..]) {
            Some(i) => {
                f(&data[pos..pos + i]);
                pos += i + 1;
            }
            None => {
                f(&data[pos..]);
                break;
            }
        }
    }
}

/// Splits a line into exactly `N` field values, or `None` if the line is malformed.
#[inline]
pub fn split<const N: usize>(line: &[u8]) -> Option<[&[u8]; N]> {
    if line.is_empty() {
        return None;
    }
    let mut out: [&[u8]; N] = [&[]; N];
    let mut k = 0usize;
    let mut start = 0usize;
    let mut quoted = false;
    for (i, &b) in line.iter().enumerate() {
        if b == b',' {
            if k + 1 >= N {
                return None;
            }
            out[k] = value(&line[start..i], quoted)?;
            k += 1;
            start = i + 1;
            quoted = false;
        } else if b == b'"' {
            quoted = true;
        }
    }
    if k + 1 != N {
        return None;
    }
    out[k] = value(&line[start..], quoted)?;
    Some(out)
}

#[inline]
fn value(field: &[u8], quoted: bool) -> Option<&[u8]> {
    if !quoted {
        return Some(field);
    }
    let n = field.len();
    if n >= 2 && field[0] == b'"' && field[n - 1] == b'"' && !field[1..n - 1].contains(&b'"') {
        Some(&field[1..n - 1])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialect() {
        assert_eq!(split::<2>(b"a,b"), Some([&b"a"[..], b"b"]));
        assert_eq!(split::<2>(b"\"a\",\"\""), Some([&b"a"[..], b""]));
        assert_eq!(split::<2>(b"u1,u5\""), None);
        assert_eq!(split::<2>(b"u1,u2,u3"), None);
        assert_eq!(split::<2>(b"u1"), None);
        assert_eq!(split::<2>(b""), None);
        assert_eq!(split::<2>(b","), Some([&b""[..], b""]));
        assert_eq!(split::<2>(b"\"\"\",b"), None);
        assert_eq!(split::<2>(b"\",b"), None);
        assert_eq!(split::<2>(b"a\"b\",c"), None);
        assert_eq!(split::<2>(b" a ,b\r"), Some([&b" a "[..], b"b\r"]));
        assert_eq!(split::<2>(b"\" u1\",x"), Some([&b" u1"[..], b"x"]));
        assert_eq!(split::<2>(b"\"a\"b,c"), None);
    }

    #[test]
    fn find_byte_matches_naive() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut buf = vec![0u8; 400];
        for round in 0..4000 {
            let len = (next() % 300) as usize;
            let off = (next() % 64) as usize;
            let density = [0u64, 1, 8, 64, 256][round % 5];
            for b in &mut buf[..off + len] {
                *b = if density > 0 && next() % density == 0 { b'\n' } else { (next() % 255) as u8 + 1 };
                if *b == b'\n' && density == 0 {
                    *b = b'x';
                }
            }
            let hay = &buf[off..off + len];
            for needle in [b'\n', b'x', 0, 255] {
                assert_eq!(find_byte(needle, hay), hay.iter().position(|&b| b == needle), "len {len} off {off}");
            }
        }
        assert_eq!(find_bytes(b"ab\"window_end\": 1", b"\"window_end\""), Some(2));
        assert_eq!(find_bytes(b"aaab", b"ab"), Some(2));
        assert_eq!(find_bytes(b"ab", b"abc"), None);
        assert_eq!(find_bytes(b"", b"a"), None);
    }

    #[test]
    fn lines() {
        let mut seen = Vec::new();
        for_each_line(b"a\n\nb\r\nc", |l| seen.push(l.to_vec()));
        assert_eq!(seen, vec![b"a".to_vec(), b"".to_vec(), b"b\r".to_vec(), b"c".to_vec()]);
        let mut seen = Vec::new();
        for_each_line(b"a\n", |l| seen.push(l.to_vec()));
        assert_eq!(seen, vec![b"a".to_vec()]);
    }
}
