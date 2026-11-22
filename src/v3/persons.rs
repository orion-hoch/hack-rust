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
    4 * capacity + capacity / 4 + 24 * endpoints + (8 << 20)
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
    ///
    /// Nodes are numbered in token order without a sort: a bitmap over the token space marks
    /// every endpoint, prefix popcounts give each token its rank. Union-find runs on the ranks
    /// straight from the edges (union by index, so a root is its component's smallest index and
    /// one forward pass compresses fully). Names: one order key per node (`name_key`: integer
    /// order is byte order but for two LONG IDs sharing their first 8 digits, and FOREIGN IDs),
    /// the smallest per root; the random reads of the names and map passes are prefetched.
    pub fn build(tok: &Tokens, edges: &[&[u64]], timing: bool) -> Option<People> {
        let mut t0 = std::time::Instant::now();
        let mut lap = |what: &str| {
            if timing {
                let now = std::time::Instant::now();
                eprintln!("timing v3.    persons.{what:<10} {:8.2} ms", (now - t0).as_secs_f64() * 1e3);
                t0 = now;
            }
        };
        let cap = tok.capacity();
        // The map first: the region's first take is fresh memory, so zero.
        let mut person: Buf<u32> = Buf::take(&PERS, cap);
        let words = cap.div_ceil(64);
        // Per 64 tokens: [marks, rank of the first] (one line holds both for a lookup).
        let mut bits: Buf<[u64; 2]> = Buf::take(&PERS, words);
        bits.iter_mut().for_each(|w| *w = [0; 2]);
        for c in edges {
            if c.len() % 2 != 0 {
                return None;
            }
            for &t in c.iter() {
                if t >= cap as u64 {
                    return None;
                }
                bits[(t >> 6) as usize][0] |= 1 << (t & 63);
            }
        }
        lap("mark");
        let mut n = 0u64;
        for w in bits.iter_mut() {
            w[1] = n;
            n += u64::from(w[0].count_ones());
        }
        let n = n as usize;
        let mut node_token: Buf<u32> = Buf::take(&PERS, n);
        let mut nk: Buf<u64> = Buf::take(&PERS, n);
        let mut k = 0usize;
        for (wi, w) in bits.iter().enumerate() {
            let mut b = w[0];
            while b != 0 {
                let t = (wi * 64 + b.trailing_zeros() as usize) as u32;
                node_token[k] = t;
                nk[k] = name_key(tok, t);
                k += 1;
                b &= b - 1;
            }
        }
        lap("keys");
        // Union-find on nodes, straight from the edges; node lookups and the first parent read
        // of edge j + EDGE_AHEAD are started early.
        let mut parent: Buf<u32> = Buf::take(&PERS, n);
        for (i, p) in parent.iter_mut().enumerate() {
            *p = i as u32;
        }
        let node = |t: u64| -> u32 {
            let w = &bits[(t >> 6) as usize];
            (w[1] + u64::from((w[0] & ((1u64 << (t & 63)) - 1)).count_ones())) as u32
        };
        // Three stages: at j + 2D the mark lines of both ends are prefetched; at j + D their
        // nodes are computed (lines now near), kept in a ring, and their parent lines
        // prefetched; at j the edge is united.
        const D: usize = 16;
        let mut ring = [[0u32; 2]; D];
        for c in edges {
            let pairs = c.len() / 2;
            for j in 0..pairs.min(D) {
                ring[j] = [node(c[2 * j]), node(c[2 * j + 1])];
            }
            for j in 0..pairs {
                if j + 2 * D < pairs {
                    prefetch(&bits[(c[2 * (j + 2 * D)] >> 6) as usize]);
                    prefetch(&bits[(c[2 * (j + 2 * D) + 1] >> 6) as usize]);
                }
                let [a, b] = ring[j % D];
                if j + D < pairs {
                    let nx = [node(c[2 * (j + D)]), node(c[2 * (j + D) + 1])];
                    prefetch(&parent[nx[0] as usize]);
                    prefetch(&parent[nx[1] as usize]);
                    ring[j % D] = nx;
                }
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                if ra != rb {
                    let (hi, lo) = if ra > rb { (ra, rb) } else { (rb, ra) };
                    parent[hi as usize] = lo;
                }
            }
        }
        drop(bits);
        // Full compression in one forward pass: a parent is a smaller index, already final.
        for i in 0..n {
            let p = parent[i] as usize;
            parent[i] = parent[p];
        }
        lap("union");
        // Names: the smallest member per root. The root is its component's smallest index, so it
        // is met first and starts the component's best.
        const AHEAD: usize = 64;
        let mut best: Buf<u32> = Buf::take(&PERS, n);
        let mut roots = 0usize;
        for i in 0..n {
            if i + 2 * AHEAD < n {
                prefetch(&best[parent[i + 2 * AHEAD] as usize]);
            }
            if i + AHEAD < n {
                let r = parent[i + AHEAD] as usize;
                if r < i {
                    prefetch(&nk[best[r] as usize]);
                }
            }
            let r = parent[i] as usize;
            if r == i {
                best[i] = i as u32;
                roots += 1;
                continue;
            }
            let cur = best[r] as usize;
            if name_less(tok, &nk, &node_token, i, cur) {
                best[r] = i as u32;
            }
        }
        drop(nk);
        lap("names");
        // Person map, written once per node in token order.
        for i in 0..n {
            if i + 2 * AHEAD < n {
                prefetch(&best[parent[i + 2 * AHEAD] as usize]);
            }
            if i + AHEAD < n {
                prefetch(&node_token[best[parent[i + AHEAD] as usize] as usize]);
            }
            let r = parent[i] as usize;
            person[node_token[i] as usize] = node_token[best[r] as usize] + 1;
        }
        lap("map");
        Some(People { person, linked: n, persons: roots })
    }
}

#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: a prefetch never faults.
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

/// Marks a key that needs a byte comparison (FOREIGN IDs).
const OTHER: u64 = 1 << 63;

/// Order key of a user ID by token. SHORT: the left-aligned base-36 order key (digits in 8
/// places, length 1 to 8 in the low 4 bits). LONG: its first 8 hex digits as base-36 digits in the
/// same 8 places and length 9 (it is longer than any SHORT). Hex digits are base-36 digits with
/// the same values and the same byte order, so integer order of keys is byte order of the IDs,
/// except that two LONG IDs with equal first 8 digits tie (then their 128-bit values decide).
/// FOREIGN: OTHER (bytes decide).
fn name_key(tok: &Tokens, t: u32) -> u64 {
    match tok.key_of(t).0 {
        super::ids::Key::Short(v) => {
            // crate::fast::names::order_key without its loop: the length is 1 plus the number of
            // powers 36^1..36^7 at or below v.
            let p = &crate::fast::names::POW36;
            let len = 1 + (1..8).map(|k| usize::from(v >= p[k])).sum::<usize>();
            ((v * p[8 - len]) << 4) | len as u64
        }
        super::ids::Key::Long(a, _) => {
            let mut x = 0u64;
            for i in 0..8 {
                x = x * 36 + ((a >> (60 - 4 * i)) & 15);
            }
            (x << 4) | 9
        }
        super::ids::Key::Foreign => OTHER,
    }
}

/// Byte order of nodes i and j's IDs (i != j, distinct IDs): the keys, then the exact rules.
#[inline(always)]
fn name_less(tok: &Tokens, nk: &[u64], node_token: &[u32], i: usize, j: usize) -> bool {
    let (ki, kj) = (nk[i], nk[j]);
    if (ki | kj) & OTHER == 0 && ki != kj {
        return ki < kj;
    }
    name_less_slow(tok, node_token[i], node_token[j])
}

#[cold]
#[inline(never)]
fn name_less_slow(tok: &Tokens, ti: u32, tj: u32) -> bool {
    tok.cmp(ti, tj) == std::cmp::Ordering::Less
}

