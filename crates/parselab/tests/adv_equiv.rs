//! Adversarial equivalence of the AVX-512 batches with the reference, on both multiply paths:
//! every byte value at every position of the first 96 bytes of a set of diverse base lines (40
//! event lines, 24 click lines, one per signature of field lengths and quoting), plus a mutated
//! copy of the source dataset. It caught the id letter-range bug of 93fb23e that the datasets'
//! MATCH and the unit tests did not (16 mismatches there, 0 after the fix).
//!
//! Does nothing unless ADV_SRC is set. On box 1, about 25 s with the build:
//!   cd crates/parselab && ADV_SRC=$HOME/shop/data/dev5 ADV_DST=<scratch dir> \
//!     cargo test --release --test adv_equiv -- --nocapture
use parselab::fast::Isa;
use parselab::*;
use std::fs;
use std::path::Path;
use std::sync::atomic::Ordering::Relaxed;

fn first_file(dir: &Path) -> Vec<u8> {
    let mut v: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    v.sort();
    fs::read(&v[0]).unwrap()
}

fn variants(data: &[u8], take: usize, nbase: usize, span: usize) -> Vec<u8> {
    let mut it = data.split(|c| *c == b'\n');
    let header = it.next().unwrap().to_vec();
    let lines: Vec<&[u8]> = it.filter(|l| !l.is_empty()).take(take).collect();
    // Diverse bases: one per (length of each of the first four fields, quoted or not) signature.
    let mut seen = std::collections::BTreeSet::new();
    let mut bases = Vec::new();
    for l in &lines {
        let sig: Vec<(usize, bool)> = l.split(|c| *c == b',').take(4).map(|f| (f.len(), f.first() == Some(&b'"'))).collect();
        if seen.insert(sig) {
            bases.push(*l);
        }
    }
    bases.truncate(nbase);
    let mut out = header.clone();
    out.push(b'\n');
    for l in &lines {
        out.extend_from_slice(l);
        out.push(b'\n');
    }
    for base in &bases {
        for b in 0..=255u8 {
            if b == b'\n' {
                continue;
            }
            for pos in 0..base.len().min(span) {
                let mut l = base.to_vec();
                l[pos] = b;
                out.extend_from_slice(&l);
                out.push(b'\n');
            }
        }
    }
    out
}

fn check(dir: &Path, what: &str) {
    let ds = Dataset::open(dir, MapMode::Lazy).unwrap();
    let ctx = ds.ctx().map_err(|e| format!("{e:?}")).unwrap();
    parselab::fast3::choose_paths();
    for m256 in [false, true] {
        parselab::fast3::MUL256.store(m256, Relaxed);
        let rep = equiv::check_dataset(&ds, &ctx, &[Isa::Avx512], 2, 7);
        println!("{what} mul256={m256}: files {} configs {} records compared {} reference {} mismatches {}", rep.files, rep.configs, rep.records_compared, rep.reference_records, rep.mismatches.len());
        for m in rep.mismatches.iter().take(4) {
            println!("  MISMATCH {m}");
        }
        assert!(rep.mismatches.is_empty(), "{what} mul256={m256}");
    }
}

#[test]
fn adversarial() {
    let Ok(src) = std::env::var("ADV_SRC") else { return };
    let src = Path::new(&src);
    let dst = std::env::var("ADV_DST").unwrap();
    let dst = Path::new(&dst);
    let _ = fs::remove_dir_all(dst);
    let a = dst.join("bytes");
    for sub in ["events", "clicks", "links"] {
        fs::create_dir_all(a.join(sub)).unwrap();
    }
    for f in ["products.csv", "campaigns.csv", "manifest.json"] {
        fs::copy(src.join(f), a.join(f)).unwrap();
    }
    fs::write(a.join("events/part-00000.csv"), variants(&first_file(&src.join("events")), 4000, 40, 96)).unwrap();
    fs::write(a.join("clicks/part-00000.csv"), variants(&first_file(&src.join("clicks")), 4000, 24, 80)).unwrap();
    let links = first_file(&src.join("links"));
    fs::write(a.join("links/part-00000.csv"), &links[..links.len().min(200_000)]).unwrap();
    check(&a, "bytes");
    let m = dst.join("mutated");
    let n = parselab::mutate::mutate(src, &m, 11, 0.25).unwrap();
    println!("mutated lines {n}");
    check(&m, "mutated");
}
