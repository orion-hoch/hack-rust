//! Stage 1: find the input files, map them, and cut them into line-aligned byte ranges.

use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::ts;

/// One mapped input file. Empty files are not mapped.
pub struct Source {
    map: Option<Mmap>,
}

impl Source {
    pub fn open(path: PathBuf) -> io::Result<Source> {
        let with_path = |e: io::Error| io::Error::new(e.kind(), format!("{}: {e}", path.display()));
        let file = File::open(&path).map_err(with_path)?;
        let len = file.metadata().map_err(with_path)?.len();
        let map = if len == 0 {
            None
        } else {
            // SAFETY: the inputs are read-only for the whole run; nobody truncates them.
            Some(unsafe { Mmap::map(&file) }.map_err(with_path)?)
        };
        Ok(Source { map })
    }

    #[inline]
    pub fn bytes(&self) -> &[u8] {
        match &self.map {
            Some(map) => map,
            None => &[],
        }
    }

    /// The offset of the first data line: just past the header's `\n`, or the end.
    pub fn data_start(&self) -> usize {
        let b = self.bytes();
        match memchr::memchr(b'\n', b) {
            Some(i) => i + 1,
            None => b.len(),
        }
    }
}

/// A byte range of one source that starts at a line start and ends just past a `\n` or at EOF.
#[derive(Clone, Copy, Debug)]
pub struct Chunk {
    pub source: u32,
    pub start: usize,
    pub end: usize,
}

impl Chunk {
    pub fn len(&self) -> usize {
        self.end - self.start
    }
}

/// Every `*.csv` file directly in `dir` (following symlinks), sorted by name.
/// A missing directory has no files.
pub fn list_csv(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.ends_with(".csv") {
            continue;
        }
        let path = entry.path();
        if std::fs::metadata(&path).map(|m| m.is_file()).unwrap_or(false) {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

pub fn open_all(paths: Vec<PathBuf>) -> io::Result<Vec<Source>> {
    paths.into_iter().map(Source::open).collect()
}

/// Cuts the data part (after the header line) of each source into chunks of about `target`
/// bytes, each boundary moved forward to just past the next `\n`.
pub fn chunks(sources: &[Source], target: usize, out: &mut Vec<Chunk>) {
    let target = target.max(1 << 16);
    for (index, source) in sources.iter().enumerate() {
        let bytes = source.bytes();
        let mut start = source.data_start();
        let end = bytes.len();
        while start < end {
            let mut cut = start.saturating_add(target);
            if cut >= end {
                cut = end;
            } else {
                cut = match memchr::memchr(b'\n', &bytes[cut..]) {
                    Some(i) => cut + i + 1,
                    None => end,
                };
            }
            out.push(Chunk { source: index as u32, start, end: cut });
            start = cut;
        }
    }
}

/// The two instants of manifest.json, in milliseconds: `[window_start, window_end)`.
pub fn manifest_window(text: &[u8]) -> io::Result<(i64, i64)> {
    let start = manifest_value(text, b"window_start");
    let end = manifest_value(text, b"window_end");
    match (start.and_then(ts::parse), end.and_then(ts::parse)) {
        (Some(s), Some(e)) => Ok((s, e)),
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, "manifest.json: no window_start/window_end")),
    }
}

/// The string value of a top-level key in a flat JSON object.
fn manifest_value<'a>(text: &'a [u8], key: &[u8]) -> Option<&'a [u8]> {
    let mut quoted = Vec::with_capacity(key.len() + 2);
    quoted.push(b'"');
    quoted.extend_from_slice(key);
    quoted.push(b'"');
    let at = memchr::memmem::find(text, &quoted)? + quoted.len();
    let mut i = at;
    while i < text.len() && text[i].is_ascii_whitespace() {
        i += 1;
    }
    if text.get(i) != Some(&b':') {
        return None;
    }
    i += 1;
    while i < text.len() && text[i].is_ascii_whitespace() {
        i += 1;
    }
    if text.get(i) != Some(&b'"') {
        return None;
    }
    let from = i + 1;
    let len = memchr::memchr(b'"', &text[from..])?;
    Some(&text[from..from + len])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest() {
        let text = br#"{
  "window_start": "2026-10-01T00:00:00Z",
  "window_end": "2026-10-15T00:00:00Z"
}"#;
        let (s, e) = manifest_window(text).unwrap();
        assert_eq!(e - s, 14 * ts::DAY);
        assert_eq!(s, ts::parse(b"2026-10-01T00:00:00Z").unwrap());
    }
}
