//! Persons: union-find over the token edges of every link line, one worker, while the others
//! parse events. Tokens are hash slots (no locality), so the linked tokens are first relabelled to
//! dense node numbers by a radix sort of (token, position); union-find then runs on arrays that fit
//! the cache. Each person is named by the byte-wise smallest member (rebuilt bytes, with the exact
//! SHORT and LONG shortcuts). The result maps a token to its person's name token; a token in no
//! link is its own person.

use super::people::radix_sort;
use super::tokens::Tokens;

pub struct People {
    /// 0 = the token is its own person; else the name token + 1.
    person: Vec<u32>,
    pub linked: usize,
    pub persons: usize,
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
    pub fn build(tok: &Tokens, edges: &[&[u64]]) -> Option<People> {
        // Dense node numbers: sort (token << 32 | endpoint position), number distinct tokens.
        let m: usize = edges.iter().map(|c| c.len()).sum();
        let mut keys: Vec<u64> = Vec::with_capacity(m);
        for c in edges {
            if c.len() % 2 != 0 {
                return None;
            }
            for &t in c.iter() {
                if t > u64::from(u32::MAX) || !tok.valid(t as u32) {
                    return None;
                }
                keys.push(((t & 0xffff_ffff) << 32) | keys.len() as u64);
            }
        }
        let mut tmp = Vec::new();
        radix_sort(&mut keys, &mut tmp, 32, 64);
        drop(tmp);
        let mut node_of_pos = vec![0u32; m];
        let mut node_token: Vec<u32> = Vec::new();
        let mut last = u64::MAX;
        for &k in &keys {
            let t = k >> 32;
            if t != last {
                node_token.push(t as u32);
                last = t;
            }
            node_of_pos[(k & 0xffff_ffff) as usize] = (node_token.len() - 1) as u32;
        }
        drop(keys);
        let n = node_token.len();
        // Union-find on nodes (union by index: a root points to a smaller index).
        let mut parent: Vec<u32> = (0..n as u32).collect();
        for e in node_of_pos.chunks_exact(2) {
            let (ra, rb) = (find(&mut parent, e[0]), find(&mut parent, e[1]));
            if ra != rb {
                let (hi, lo) = if ra > rb { (ra, rb) } else { (rb, ra) };
                parent[hi as usize] = lo;
            }
        }
        drop(node_of_pos);
        // Names: the smallest member by bytes, per root. Nodes are in token order, so the key
        // loads walk the token table forward.
        let mut best: Vec<u32> = vec![u32::MAX; n];
        let mut roots = 0usize;
        for i in 0..n as u32 {
            let r = find(&mut parent, i) as usize;
            let cur = best[r];
            if cur == u32::MAX {
                best[r] = i;
                roots += 1;
            } else if tok.cmp(node_token[i as usize], node_token[cur as usize]).is_lt() {
                best[r] = i;
            }
        }
        // Person map, written once per node in token order.
        let mut person: Vec<u32> = vec![0; tok.capacity()];
        for i in 0..n {
            let r = find(&mut parent, i as u32) as usize;
            person[node_token[i] as usize] = node_token[best[r] as usize] + 1;
        }
        Some(People { person, linked: n, persons: roots })
    }
}
