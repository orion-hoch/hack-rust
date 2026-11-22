//! Equivalence of the fast parser with the reference: per file, the record streams must be equal
//! as multisets for every ISA path, every buffer mode (plain mapping, padded copy) and every set
//! of chunk cut points.

use crate::dataset::{Dataset, Stream};
use crate::fast::{self, Buf, Fast, Isa, Kind, Padded};
use crate::records::*;
use crate::reference;
use crate::tables::Ctx;

#[derive(Default, Debug, Clone)]
pub struct Report {
    pub files: u64,
    pub configs: u64,
    /// Records (all streams, foreign included) compared, summed over configs.
    pub records_compared: u64,
    /// Records the reference produced (once per file).
    pub reference_records: u64,
    pub mismatches: Vec<String>,
}

fn total(s: &VecSink) -> u64 {
    s.total()
}

fn reference_file(ctx: &Ctx, kind: Kind, file: u32, data: &[u8], from: usize, to: usize, sink: &mut VecSink) {
    match kind {
        Kind::Events => reference::events(ctx, file, data, from, to, sink),
        Kind::Clicks => reference::clicks(ctx, file, data, from, to, sink),
        Kind::Links => reference::links(ctx, file, data, from, to, sink),
    }
}

/// Cut sets for a file: whole, random cuts, and cuts at line starts and one byte either side.
fn cut_sets(data: &[u8], rng: &mut u64, random_sets: usize) -> Vec<Vec<usize>> {
    let len = data.len();
    let mut next = || {
        *rng = mix64(rng.wrapping_add(0x9E37_79B9_7F4A_7C15));
        *rng
    };
    let mut sets = vec![vec![0, len]];
    if len < 2 {
        return sets;
    }
    for _ in 0..random_sets {
        let k = 1 + (next() % 9) as usize;
        let mut cuts: Vec<usize> = (0..k).map(|_| (next() % len as u64) as usize).collect();
        cuts.push(0);
        cuts.push(len);
        cuts.sort_unstable();
        sets.push(cuts);
    }
    // Adversarial: around a few newlines.
    let nls: Vec<usize> = data.iter().enumerate().filter(|(_, c)| **c == b'\n').map(|(i, _)| i).take(50_000).collect();
    if !nls.is_empty() {
        let mut cuts = vec![0, len];
        for _ in 0..8 {
            let n = nls[(next() % nls.len() as u64) as usize];
            for d in [0usize, 1, 2] {
                cuts.push((n + d).min(len));
            }
            cuts.push(n.saturating_sub(1));
        }
        cuts.sort_unstable();
        sets.push(cuts);
    }
    sets
}

/// First line whose records differ between reference and fast (for the mismatch message).
fn first_bad_line(f: &Fast, isa: Isa, kind: Kind, file: u32, buf: Buf, ctx: &Ctx) -> String {
    let data = buf.data;
    let mut start = data.iter().position(|&c| c == b'\n').map_or(data.len(), |p| p + 1);
    while start < data.len() {
        let end = data[start..].iter().position(|&c| c == b'\n').map_or(data.len(), |p| start + p);
        let mut a = VecSink::default();
        let mut b = VecSink::default();
        reference_file(ctx, kind, file, data, start, start + 1, &mut a);
        fast::parse(f, isa, kind, file, buf, start, start + 1, &mut b);
        a.canonicalize();
        b.canonicalize();
        if !a.same_records(&b) {
            return format!(
                "line at {start}: {:?}\n  reference {:?} {:?} {:?} {:?}\n  fast      {:?} {:?} {:?} {:?}",
                String::from_utf8_lossy(&data[start..end]),
                a.events, a.clicks, a.links, (a.foreign_events.len(), a.foreign_clicks.len(), a.foreign_links.len()),
                b.events, b.clicks, b.links, (b.foreign_events.len(), b.foreign_clicks.len(), b.foreign_links.len())
            );
        }
        start = end + 1;
    }
    "no single line differs (cut-dependent)".into()
}

pub fn check_dataset(ds: &Dataset, ctx: &Ctx, isas: &[Isa], random_sets: usize, seed: u64) -> Report {
    let f = Fast::new(ctx);
    let mut rep = Report::default();
    let mut rng = seed ^ 0xD1B5_4A32_D192_ED03;
    for (stream, kind) in [(Stream::Events, Kind::Events), (Stream::Clicks, Kind::Clicks), (Stream::Links, Kind::Links)] {
        for (file, m) in ds.stream(stream).iter().enumerate() {
            let file = file as u32;
            let data = m.bytes();
            rep.files += 1;
            let mut want = VecSink::default();
            reference_file(ctx, kind, file, data, 0, data.len(), &mut want);
            want.canonicalize();
            rep.reference_records += total(&want);
            let padded = Padded::new(data);
            let sets = cut_sets(data, &mut rng, random_sets);
            for &isa in isas {
                for (mode, buf) in [("plain", Buf::plain(data)), ("padded", padded.buf())] {
                    for cuts in &sets {
                        let mut got = VecSink::default();
                        for w in cuts.windows(2) {
                            fast::parse(&f, isa, kind, file, buf, w[0], w[1], &mut got);
                        }
                        got.canonicalize();
                        rep.configs += 1;
                        rep.records_compared += total(&got);
                        if !got.same_records(&want) {
                            let detail = first_bad_line(&f, isa, kind, file, buf, ctx);
                            rep.mismatches.push(format!(
                                "{} {:?} file {file} ({}) isa {isa:?} {mode} cuts {}: ref {} fast {}: {detail}",
                                ds.dir.display(),
                                kind,
                                m.path.display(),
                                cuts.len() - 1,
                                total(&want),
                                total(&got)
                            ));
                            if rep.mismatches.len() > 20 {
                                return rep;
                            }
                        }
                    }
                }
            }
        }
    }
    rep
}
