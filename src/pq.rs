//! Stage 7 implementation: both tables through pqw, our own Parquet writer (crates/pqw).
//! Callers only use the two functions below and the column structs in `output.rs`.
//!
//! What the scorer's loader needs, kept here: `day` is a DATE, `top_category` is a nullable
//! string, and every file of a table has the same column types (all counts INT64). The callers
//! always write at least one file per table, and pqw writes a valid file for zero rows.
//! Names are written as the bytes they are (IDs may hold any byte the CSV dialect allows).

use std::io;
use std::path::{Path, PathBuf};

use pqw::{Codes, Col, Data, Options, Strings};

use crate::output::{CampaignDaily, NO_CATEGORY, StrColumn, UserDaily};

/// pqw takes u32 string offsets, so one file's bytes of any string column stay below this; a
/// bigger table is written as more files (row order and file count do not matter).
const MAX_FILE_BYTES: usize = 1 << 30;
/// Most distinct dictionary codes in one file with u16 codes (0xFFFF is null).
const MAX_CODES: usize = 0xFFFF;

/// Row ranges of at most MAX_FILE_BYTES per string column and, if `remap` is given as
/// (codes, category count), at most MAX_CODES distinct non-null codes. At least one range.
fn groups(n: usize, strs: &[&StrColumn], remap: Option<(&[u32], usize)>) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut stamp = vec![u32::MAX; remap.map_or(0, |(_, k)| k)];
    let (mut a, mut used, mut g) = (0usize, 0usize, 0u32);
    for i in 0..n {
        let too_big = strs.iter().any(|s| s.offsets[i + 1] - s.offsets[a] > MAX_FILE_BYTES);
        let code = remap.map(|(c, _)| c[i]).filter(|&c| c != NO_CATEGORY);
        let new_code = code.is_some_and(|c| stamp[c as usize] != g);
        if i > a && (too_big || (new_code && used == MAX_CODES)) {
            out.push((a, i));
            a = i;
            used = 0;
            g += 1;
        }
        if let Some(c) = code {
            if stamp[c as usize] != g {
                stamp[c as usize] = g;
                used += 1;
            }
        }
    }
    out.push((a, n));
    out
}

/// Rows [a, b) of a string column, rebased so the offsets start at 0.
fn strings(col: &StrColumn, a: usize, b: usize) -> (&[u8], Vec<u32>) {
    let base = col.offsets[a];
    let offsets = col.offsets[a..=b].iter().map(|&o| (o - base) as u32).collect();
    (&col.data[base..col.offsets[b]], offsets)
}

/// The k-th file of a table: `path` itself, then `<stem>-<k>.parquet` beside it.
fn nth_path(path: &Path, k: usize) -> PathBuf {
    if k == 0 {
        return path.to_path_buf();
    }
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    path.with_file_name(format!("{stem}-{k}.parquet"))
}

/// top_category of rows [a, b) as pqw codes plus the dictionary they index.
enum Top<'a> {
    U8(Vec<u8>, Vec<&'a [u8]>),
    U16(Vec<u16>, Vec<&'a [u8]>),
}

fn top_codes<'a>(top: &[u32], categories: &'a [Vec<u8>]) -> Top<'a> {
    let k = categories.len();
    if k <= pqw::NULL8 as usize {
        let codes = top.iter().map(|&c| if c == NO_CATEGORY { pqw::NULL8 } else { c as u8 }).collect();
        return Top::U8(codes, categories.iter().map(|c| c.as_slice()).collect());
    }
    if k <= pqw::NULL16 as usize {
        let codes = top.iter().map(|&c| if c == NO_CATEGORY { pqw::NULL16 } else { c as u16 }).collect();
        return Top::U16(codes, categories.iter().map(|c| c.as_slice()).collect());
    }
    // More categories than u16 codes: a dictionary of the ones these rows use (at most
    // MAX_CODES, see `groups`), in the same byte-wise order.
    let mut used: Vec<u32> = top.iter().copied().filter(|&c| c != NO_CATEGORY).collect();
    used.sort_unstable();
    used.dedup();
    let codes = top
        .iter()
        .map(|&c| if c == NO_CATEGORY { pqw::NULL16 } else { used.binary_search(&c).unwrap() as u16 })
        .collect();
    Top::U16(codes, used.iter().map(|&c| categories[c as usize].as_slice()).collect())
}

pub fn write_user_daily(path: &Path, t: &UserDaily, categories: &[Vec<u8>]) -> io::Result<()> {
    let remap = (categories.len() > pqw::NULL16 as usize).then_some((&t.top_category[..], categories.len()));
    for (k, (a, b)) in groups(t.len(), &[&t.user_id], remap).into_iter().enumerate() {
        let (bytes, offsets) = strings(&t.user_id, a, b);
        let top = top_codes(&t.top_category[a..b], categories);
        let top_col = match &top {
            Top::U8(c, d) => Data::Dict { codes: Codes::U8(c), dict: d, nullable: true },
            Top::U16(c, d) => Data::Dict { codes: Codes::U16(c), dict: d, nullable: true },
        };
        let cols = [
            Col::new("user_id", Data::Str(Strings::new(bytes, &offsets))),
            Col::new("day", Data::Date(&t.day[a..b])),
            Col::new("sessions", Data::I64(&t.sessions[a..b])),
            Col::new("events", Data::I64(&t.events[a..b])),
            Col::new("revenue_cents", Data::I64(&t.revenue_cents[a..b])),
            Col::new("top_category", top_col),
        ];
        pqw::write_file(nth_path(path, k), &cols, &Options::default())?;
    }
    Ok(())
}

pub fn write_campaign_daily(path: &Path, t: &CampaignDaily) -> io::Result<()> {
    for (k, (a, b)) in groups(t.len(), &[&t.campaign_id, &t.channel], None).into_iter().enumerate() {
        let (ids, id_offsets) = strings(&t.campaign_id, a, b);
        let (channels, channel_offsets) = strings(&t.channel, a, b);
        let cols = [
            Col::new("campaign_id", Data::Str(Strings::new(ids, &id_offsets))),
            Col::new("channel", Data::Str(Strings::new(channels, &channel_offsets))),
            Col::new("day", Data::Date(&t.day[a..b])),
            Col::new("sessions", Data::I64(&t.sessions[a..b])),
            Col::new("purchasing_sessions", Data::I64(&t.purchasing_sessions[a..b])),
            Col::new("revenue_cents", Data::I64(&t.revenue_cents[a..b])),
            Col::new("assisted_sessions", Data::I64(&t.assisted_sessions[a..b])),
        ];
        pqw::write_file(nth_path(path, k), &cols, &Options::default())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(rows: &[&[u8]]) -> StrColumn {
        let mut c = StrColumn::new();
        for r in rows {
            c.push(r);
        }
        c
    }

    #[test]
    fn groups_split_on_distinct_codes() {
        let s = col(&[&b"a"[..]; 6]);
        let top = [0, 1, NO_CATEGORY, 1, 2, 0];
        assert_eq!(groups(6, &[&s], None), vec![(0, 6)]);
        assert_eq!(groups(0, &[&col(&[])], None), vec![(0, 0)]);
        // with room for 65535 codes nothing splits here
        assert_eq!(groups(6, &[&s], Some((&top, 3))), vec![(0, 6)]);
    }

    #[test]
    fn groups_split_past_u16_codes() {
        let n = 70_000;
        let s = col(&vec![&b"a"[..]; n]);
        let top: Vec<u32> = (0..n as u32).collect();
        assert_eq!(groups(n, &[&s], Some((&top, n))), vec![(0, 65_535), (65_535, n)]);
        // end to end: two files, the second named after the first
        let cats: Vec<Vec<u8>> = (0..n as u32).map(|i| format!("c{i:06}").into_bytes()).collect();
        let mut t = UserDaily::new();
        for i in 0..n {
            t.user_id.push(b"u1");
            t.day.push(1);
            t.sessions.push(1);
            t.events.push(1);
            t.revenue_cents.push(1);
            t.top_category.push(i as u32);
        }
        let dir = std::env::temp_dir().join(format!("pq-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_user_daily(&dir.join("part-00000.parquet"), &t, &cats).unwrap();
        assert!(dir.join("part-00000.parquet").is_file() && dir.join("part-00000-1.parquet").is_file());
        if std::env::var_os("PQ_TEST_KEEP").is_none() {
            std::fs::remove_dir_all(&dir).unwrap();
        } else {
            eprintln!("kept {}", dir.display());
        }
    }

    #[test]
    fn remapped_codes_keep_order() {
        let cats: Vec<Vec<u8>> = (0..70_000u32).map(|i| format!("c{i:06}").into_bytes()).collect();
        let top = [69_999, NO_CATEGORY, 5, 69_999];
        match top_codes(&top, &cats) {
            Top::U16(c, d) => {
                assert_eq!(c, vec![1, pqw::NULL16, 0, 1]);
                assert_eq!(d, vec![&b"c000005"[..], b"c069999"]);
            }
            Top::U8(..) => panic!("expected u16 codes"),
        }
    }
}
