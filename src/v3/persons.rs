//! Persons: union-find over the token edges of every link line, one worker, while the others
//! parse events. Tokens are hash slots (no locality), so the linked tokens are first relabelled to
//! dense node numbers by a radix sort of (token, position); union-find then runs on arrays that fit
//! the cache. Each person is named by the byte-wise smallest member (rebuilt bytes, with the exact
//! SHORT and LONG shortcuts). The result maps a token to its person's name token; a token in no
//! link is its own person.

use super::mem::{Buf, PERS};
use super::tokens::Tokens;

pub struct People {
    /// 0 = the token is its own person; else the name token + 1.
    person: Buf<u32>,
    pub linked: usize,
    pub persons: usize,
}

/// Persons-region bytes for `endpoints` link endpoints over a token space of `capacity`: the
/// map, then at most 24 bytes per endpoint at once (sort keys and scratch, positions and node
/// tokens; later parents, names and keys of at most one node per endpoint).
pub fn region_bytes(capacity: usize, endpoints: usize) -> usize {
    4 * capacity + 24 * endpoints + (8 << 20)
}

/// LSD radix sort of u64 keys over bits [lo, hi), 11 bits a pass, skipping passes whose digit is
/// the same for every key, ping-ponging between `a` and `b` (same length). True if the result is
/// in `b`.
fn radix_sort(a: &mut [u64], b: &mut [u64], lo: u32, hi: u32) -> bool {
    let n = a.len();
    if n < 64 {
        a.sort_unstable();
        return false;
    }
    let mut in_b = false;
    let mut shift = lo;
    let mut count = [0usize; 2048];
    while shift < hi {
        let bits = (hi - shift).min(11);
        let mask = (1u64 << bits) - 1;
        let (src, dst): (&mut [u64], &mut [u64]) = if in_b { (&mut *b, &mut *a) } else { (&mut *a, &mut *b) };
        count[..(1 << bits)].iter_mut().for_each(|c| *c = 0);
        for &k in src.iter() {
            count[((k >> shift) & mask) as usize] += 1;
        }
        if count[((src[0] >> shift) & mask) as usize] == n {
            shift += bits;
            continue;
        }
        let mut sum = 0;
        for c in count[..(1 << bits)].iter_mut() {
            let x = *c;
            *c = sum;
            sum += x;
        }
        for &k in src.iter() {
            let d = ((k >> shift) & mask) as usize;
            dst[count[d]] = k;
            count[d] += 1;
        }
        in_b = !in_b;
        shift += bits;
    }
    in_b
}

#[inline]
fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let p = parent[x as usize];
        let pp = parent[p as usize];
        parent[x as usize] = pp;
        x = pp;
    }
    x
}

impl People {
    #[inline(always)]
    pub fn prefetch(&self, t: u32) {
        #[cfg(target_arch = "x86_64")]
        if let Some(p) = self.person.get(t as usize) {
            // SAFETY: prefetch never faults.
            unsafe { core::arch::x86_64::_mm_prefetch(p as *const u32 as *const i8, core::arch::x86_64::_MM_HINT_T0) };
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = t;
    }

    #[inline(always)]
    pub fn person_of(&self, t: u32) -> u32 {
        match self.person.get(t as usize) {
            Some(&p) if p != 0 => p - 1,
            _ => t,
        }
    }

    /// `edges`: token pairs as consecutive words (side a, side b), in chunks of even length.
    /// None if any token is not a valid token (the run falls back).
    pub fn build(tok: &Tokens, edges: &[&[u64]], timing: bool) -> Option<People> {
        let mut t0 = std::time::Instant::now();
        let mut lap = |what: &str| {
            if timing {
                let now = std::time::Instant::now();
                eprintln!("timing v3.    persons.{what:<10} {:8.2} ms", (now - t0).as_secs_f64() * 1e3);
                t0 = now;
            }
        };
        // Dense node numbers: sort (token << 32 | endpoint position), number distinct tokens.
        let m: usize = edges.iter().map(|c| c.len()).sum();
        // Every token must name a slot or an interned ID (one read of the bound, not one per
        // token: the bound takes a lock).
        let cap = tok.capacity() as u64;
        // The map first: the region's first take is fresh memory, so zero.
        let mut person: Buf<u32> = Buf::take(&PERS, cap as usize);
        let mut keys: Buf<u64> = Buf::take(&PERS, m);
        let mut at = 0usize;
        for c in edges {
            if c.len() % 2 != 0 {
                return None;
            }
            for &t in c.iter() {
                if t >= cap {
                    return None;
                }
                keys[at] = (t << 32) | at as u64;
                at += 1;
            }
        }
        lap("collect");
        let mut tmp: Buf<u64> = Buf::take(&PERS, m);
        let sorted: &[u64] = if radix_sort(&mut keys, &mut tmp, 32, 64) { &tmp } else { &keys };
        lap("sort");
        let mut node_of_pos: Buf<u32> = Buf::take(&PERS, m);
        let mut node_token: Buf<u32> = Buf::take(&PERS, m);
        let mut n = 0usize;
        let mut last = u64::MAX;
        for &k in sorted {
            let t = k >> 32;
            if t != last {
                node_token[n] = t as u32;
                n += 1;
                last = t;
            }
            node_of_pos[(k & 0xffff_ffff) as usize] = (n - 1) as u32;
        }
        drop(tmp);
        drop(keys);
        let node_token = &node_token[..n];
        lap("number");
        // Union-find on nodes (union by index: a root points to a smaller index).
        let mut parent: Buf<u32> = Buf::take(&PERS, n);
        for (i, p) in parent.iter_mut().enumerate() {
            *p = i as u32;
        }
        for e in node_of_pos.chunks_exact(2) {
            let (ra, rb) = (find(&mut parent, e[0]), find(&mut parent, e[1]));
            if ra != rb {
                let (hi, lo) = if ra > rb { (ra, rb) } else { (rb, ra) };
                parent[hi as usize] = lo;
            }
        }
        drop(node_of_pos);
        // Full compression in one forward pass: a parent is a smaller index, already final.
        for i in 0..n {
            let p = parent[i] as usize;
            parent[i] = parent[p];
        }
        lap("union");
        // Name keys, read once per node in token order (tokens are slot indexes, so this walks
        // the token table forward): a SHORT ID's left-aligned base-36 order key; anything else is
        // marked and kept aside with its key for a byte comparison.
        const OTHER: u64 = 1 << 63;
        let mut nk: Buf<u64> = Buf::take(&PERS, n);
        // Bytes of every LONG or FOREIGN member, rebuilt once in this streaming pass.
        let mut other_bytes: Vec<Vec<u8>> = Vec::new();
        for (i, &t) in node_token.iter().enumerate() {
            match tok.key_of(t) {
                (super::ids::Key::Short(v), _) => nk[i] = crate::fast::names::order_key(v),
                _ => {
                    nk[i] = OTHER | other_bytes.len() as u64;
                    let mut b = Vec::with_capacity(33);
                    tok.bytes(t, &mut b);
                    other_bytes.push(b);
                }
            }
        }
        // Bytes of a node from its key: SHORT from the order key, others from the side list.
        let node_bytes = |k: u64, out: &mut Vec<u8>| {
            out.clear();
            if k & OTHER == 0 {
                crate::fast::names::append_bytes(b'u', k, out);
            } else {
                out.extend_from_slice(&other_bytes[(k & !OTHER) as usize]);
            }
        };
        lap("keys");
        // Names: the smallest member by bytes, per root.
        let mut best: Buf<u32> = Buf::take(&PERS, n);
        best.iter_mut().for_each(|b| *b = u32::MAX);
        let mut roots = 0usize;
        let (mut bx, mut by) = (Vec::new(), Vec::new());
        for i in 0..n as u32 {
            let r = parent[i as usize] as usize;
            let cur = best[r];
            if cur == u32::MAX {
                best[r] = i;
                roots += 1;
                continue;
            }
            let (ki, kc) = (nk[i as usize], nk[cur as usize]);
            let smaller = if (ki | kc) & OTHER == 0 {
                ki < kc
            } else {
                // A LONG or FOREIGN member on either side: compare the rebuilt bytes.
                node_bytes(ki, &mut bx);
                node_bytes(kc, &mut by);
                bx < by
            };
            if smaller {
                best[r] = i;
            }
        }
        drop(nk);
        lap("names");
        // Person map, written once per node in token order.
        for i in 0..n {
            let r = parent[i] as usize;
            person[node_token[i] as usize] = node_token[best[r] as usize] + 1;
        }
        lap("map");
        Some(People { person, linked: n, persons: roots })
    }
}
