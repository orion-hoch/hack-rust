//! Stage 4: persons. Union-find over every ID that appears in a kept link line; each person is
//! named by the byte-wise smallest of its IDs. An ID in no kept line is its own person.

use crate::hash::{FastMap, map_with_capacity};
use crate::id::{Id, Interner};

/// Maps a user ID code to its person's name code. Only IDs whose name is another ID are stored.
pub struct Persons {
    names: FastMap<Id, Id>,
    pub linked_ids: usize,
    pub components: usize,
}

impl Persons {
    #[inline]
    pub fn person(&self, id: Id) -> Id {
        if self.names.is_empty() {
            return id;
        }
        match self.names.get(&id) {
            Some(&name) => name,
            None => id,
        }
    }

    pub fn build(pairs: &[Vec<(Id, Id)>], ids: &Interner) -> Persons {
        let total: usize = pairs.iter().map(|p| p.len()).sum();
        let mut index: FastMap<Id, u32> = map_with_capacity(total + total / 2);
        let mut members: Vec<Id> = Vec::with_capacity(total);
        let mut parent: Vec<u32> = Vec::with_capacity(total);
        let mut size: Vec<u32> = Vec::with_capacity(total);
        let mut slot = |id: Id, members: &mut Vec<Id>, parent: &mut Vec<u32>, size: &mut Vec<u32>| -> u32 {
            *index.entry(id).or_insert_with(|| {
                let i = members.len() as u32;
                members.push(id);
                parent.push(i);
                size.push(1);
                i
            })
        };
        for part in pairs {
            for &(a, b) in part {
                let x = slot(a, &mut members, &mut parent, &mut size);
                let y = slot(b, &mut members, &mut parent, &mut size);
                let (rx, ry) = (find(&mut parent, x), find(&mut parent, y));
                if rx != ry {
                    let (big, small) = if size[rx as usize] >= size[ry as usize] { (rx, ry) } else { (ry, rx) };
                    parent[small as usize] = big;
                    size[big as usize] += size[small as usize];
                }
            }
        }
        let n = members.len();
        // The smallest member of each root, by bytes.
        let mut best: Vec<u32> = vec![u32::MAX; n];
        let mut components = 0;
        for i in 0..n as u32 {
            let r = find(&mut parent, i) as usize;
            let current = best[r];
            if current == u32::MAX {
                components += 1;
                best[r] = i;
            } else if ids.cmp(members[i as usize], members[current as usize]).is_lt() {
                best[r] = i;
            }
        }
        let mut names: FastMap<Id, Id> = map_with_capacity(n.saturating_sub(components));
        for i in 0..n as u32 {
            let r = find(&mut parent, i) as usize;
            let name = members[best[r] as usize];
            if name != members[i as usize] {
                names.insert(members[i as usize], name);
            }
        }
        Persons { names, linked_ids: n, components }
    }
}

#[inline]
fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let up = parent[parent[x as usize] as usize];
        parent[x as usize] = up;
        x = up;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chains_and_names() {
        let ids = Interner::new();
        let id = |s: &str| ids.id(s.as_bytes());
        let pairs = vec![
            vec![(id("u3"), id("u9")), (id("u9"), id("u2"))],
            vec![(id("U7"), id("u9")), (id("u10"), id("u10"))],
        ];
        let p = Persons::build(&pairs, &ids);
        assert_eq!(p.person(id("u3")), id("U7"));
        assert_eq!(p.person(id("u2")), id("U7"));
        assert_eq!(p.person(id("U7")), id("U7"));
        assert_eq!(p.person(id("u10")), id("u10"));
        assert_eq!(p.person(id("u4")), id("u4"));
    }

    #[test]
    fn interned_names_compare_by_bytes() {
        let ids = Interner::new();
        let id = |s: &[u8]| ids.id(s);
        let pairs = vec![vec![
            (id(b"user_id_a"), id(b"user_id_b")),
            (id(b"u4"), id(b"user_id_b")),
            (id(b"A\tB"), id(b"user_id_a")),
            (id(b"u4"), id(b"u1")),
        ]];
        let p = Persons::build(&pairs, &ids);
        assert_eq!(p.person(id(b"u1")), id(b"A\tB"));
        assert_eq!(p.person(id(b"user_id_b")), id(b"A\tB"));
    }
}
