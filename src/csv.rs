//! The CSV dialect of rules.md, to the byte.
//!
//! * Only `\n` ends a line; `\r` is data. The last line of a file may lack its `\n`.
//! * Every comma separates fields. A field that holds a `"` anywhere must be exactly `"` + text
//!   without quotes + `"`, and its value is the text between; any other quote makes the line
//!   malformed. `""` and an empty field are both empty, which means null.
//! * A line with the wrong number of fields is malformed, and so is an empty line.
//! * Nothing is trimmed.

/// Calls `f` on every line of `data` (without its `\n`). `data` must start at a line start.
#[inline]
pub fn for_each_line<'a>(data: &'a [u8], mut f: impl FnMut(&'a [u8])) {
    let mut pos = 0;
    let end = data.len();
    while pos < end {
        match memchr::memchr(b'\n', &data[pos..]) {
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
    fn lines() {
        let mut seen = Vec::new();
        for_each_line(b"a\n\nb\r\nc", |l| seen.push(l.to_vec()));
        assert_eq!(seen, vec![b"a".to_vec(), b"".to_vec(), b"b\r".to_vec(), b"c".to_vec()]);
        let mut seen = Vec::new();
        for_each_line(b"a\n", |l| seen.push(l.to_vec()));
        assert_eq!(seen, vec![b"a".to_vec()]);
    }
}
