//! Direct-index grouping (`sessions_chunks` with `Some((base, shift))`) against token mode on the
//! same records: same sessions per person, same session for every purchase, same lookups.

use sortlab::*;
use std::collections::BTreeMap;

fn rng(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn sessions_by_person(out: &Out, buys: &[[u64; 2]]) -> BTreeMap<u32, (Vec<(u32, u32)>, Vec<(usize, u32)>)> {
    let mut m = BTreeMap::new();
    let mut s0 = 0usize;
    for (k, &p) in out.persons.iter().enumerate() {
        let s1 = out.sess_end[k] as usize;
        let v: Vec<(u32, u32)> = (s0..s1).map(|s| (out.start[s], out.events[s])).collect();
        let b: Vec<(usize, u32)> = buys
            .iter()
            .enumerate()
            .filter(|(_, b)| (b[0] >> 32) as u32 == p)
            .map(|(j, _)| (j, out.start[out.buy_session[j] as usize]))
            .collect();
        m.insert(p, (v, b));
        s0 = s1;
    }
    m
}

#[test]
fn direct_matches_token_mode() {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for case in 0..200 {
        let shift = [4u32, 8, 13][case % 3];
        let q = (rng(&mut seed) % 50) as u32;
        let base = q << shift;
        let persons = 1 + (rng(&mut seed) % (1u64 << shift).min(300)) as u32;
        let locals: Vec<u32> = (0..persons).map(|_| (rng(&mut seed) % (1u64 << shift)) as u32).collect();
        let n = (rng(&mut seed) % 3000) as usize + 1;
        let mut ev: Vec<u64> = Vec::with_capacity(n);
        let mut buys: Vec<[u64; 2]> = Vec::new();
        for _ in 0..n {
            let tok = base + locals[(rng(&mut seed) % persons as u64) as usize];
            // clustered times so that sessions form, some ties, gaps around 30 minutes
            let ts = ((rng(&mut seed) % 40) * 3_600_000 + (rng(&mut seed) % 2_400_000)) as u32 & 0x7fff_ffff;
            let buy = rng(&mut seed) % 20 == 0;
            ev.push((u64::from(tok) << 32) | (u64::from(ts) << 1) | u64::from(buy));
            if buy {
                buys.push([(u64::from(tok) << 32) | u64::from(ts), rng(&mut seed) & 0xffff_ffff]);
            }
        }
        // pieces of uneven sizes
        let cut = n / 3;
        let pieces: Vec<&[u64]> = vec![&ev[..cut], &ev[cut..], &[]];
        for kernel in [Kernel::Net, Kernel::Slots, Kernel::Sort] {
            let (mut sa, mut oa) = (Scratch::new(), Out::default());
            let (mut sb, mut ob) = (Scratch::new(), Out::default());
            sessions_chunks(&pieces, &buys, kernel, None, &mut sa, &mut oa).unwrap();
            sessions_chunks(&pieces, &buys, kernel, Some((base, shift)), &mut sb, &mut ob).unwrap();
            assert_eq!(sessions_by_person(&oa, &buys), sessions_by_person(&ob, &buys), "case {case} {kernel:?}");
            // reuse of the direct scratch for a second partition, then token mode again
            sessions_chunks(&pieces, &buys, kernel, Some((base, shift)), &mut sb, &mut ob).unwrap();
            assert_eq!(sessions_by_person(&oa, &buys), sessions_by_person(&ob, &buys), "case {case} {kernel:?} again");
        }
        // a token outside the range is refused
        let mut bad = ev.clone();
        bad.push((u64::from(base + (1 << shift)) << 32) | 2);
        let (mut sc, mut o) = (Scratch::new(), Out::default());
        assert!(sessions_chunks(&[&bad[..]], &buys, Kernel::Net, Some((base, shift)), &mut sc, &mut o).is_err());
    }
}
