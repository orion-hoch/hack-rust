//! rules.md "CSV dialect", plainly: only `\n` ends a line, every comma separates fields, a field is
//! either quote-free or wrapped whole in one pair of quotes with none inside.

/// Splits `line` (without its `\n`) into field values with quotes removed. Returns false when a
/// field has a quote anywhere except as a wrapping pair. `out` is cleared first.
pub fn split<'a>(line: &'a [u8], out: &mut Vec<&'a [u8]>) -> bool {
    out.clear();
    for field in line.split(|&c| c == b',') {
        match unquote(field) {
            Some(value) => out.push(value),
            None => return false,
        }
    }
    true
}

/// The value of one raw field, or None if its quoting is illegal.
pub fn unquote(field: &[u8]) -> Option<&[u8]> {
    if !field.contains(&b'"') {
        return Some(field);
    }
    if field.len() >= 2 && field[0] == b'"' && field[field.len() - 1] == b'"' {
        let inner = &field[1..field.len() - 1];
        if !inner.contains(&b'"') {
            return Some(inner);
        }
    }
    None
}

/// Calls `f(start_offset, line)` for every line whose first byte is at an offset in `from..to`,
/// skipping the header (the line at offset 0). A final line without `\n` counts; the empty piece
/// after a final `\n` is not a line.
pub fn for_each_line<'a>(data: &'a [u8], from: usize, to: usize, mut f: impl FnMut(usize, &'a [u8])) {
    let to = to.min(data.len());
    if from >= to {
        return;
    }
    let mut start = if from == 0 {
        // Skip the header.
        match data.iter().position(|&c| c == b'\n') {
            Some(p) => p + 1,
            None => return,
        }
    } else {
        match data[from - 1..].iter().position(|&c| c == b'\n') {
            Some(p) => from - 1 + p + 1,
            None => return,
        }
    };
    while start < to {
        let end = match data[start..].iter().position(|&c| c == b'\n') {
            Some(p) => start + p,
            None => data.len(),
        };
        f(start, &data[start..end]);
        start = end + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(line: &str) -> Option<Vec<String>> {
        let mut out = Vec::new();
        split(line.as_bytes(), &mut out)
            .then(|| out.iter().map(|f| String::from_utf8_lossy(f).into_owned()).collect())
    }

    #[test]
    fn dialect() {
        assert_eq!(fields("a,\"b\",,\"\""), Some(vec!["a".into(), "b".into(), "".into(), "".into()]));
        assert_eq!(fields(""), Some(vec!["".into()]));
        assert_eq!(fields(" a ,\" b\""), Some(vec![" a ".into(), " b".into()]));
        for bad in ["\"", "\"a", "a\"", "\"a\"b", "\"\"\"", "a\"b", "\"a,b\"", "x,\"", "u1,u5\""] {
            assert_eq!(fields(bad), None, "{bad}");
        }
    }

    #[test]
    fn lines() {
        let data = b"h\na\n\nb\nc";
        let mut seen = Vec::new();
        for_each_line(data, 0, data.len(), |s, l| seen.push((s, l.to_vec())));
        assert_eq!(seen, vec![(2, b"a".to_vec()), (4, b"".to_vec()), (5, b"b".to_vec()), (7, b"c".to_vec())]);
        // Every split point gives each line to exactly one range.
        for cut in 0..=data.len() {
            let mut parts = Vec::new();
            for_each_line(data, 0, cut, |s, l| parts.push((s, l.to_vec())));
            for_each_line(data, cut, data.len(), |s, l| parts.push((s, l.to_vec())));
            assert_eq!(parts, seen, "cut {cut}");
        }
        let mut n = 0;
        for_each_line(b"header only", 0, 11, |_, _| n += 1);
        for_each_line(b"", 0, 0, |_, _| n += 1);
        for_each_line(b"h\n", 0, 2, |_, _| n += 1);
        assert_eq!(n, 0);
    }
}
