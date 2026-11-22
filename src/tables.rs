//! The two clean lookup files: products.csv (product -> category) and campaigns.csv
//! (campaign -> channel). Lookups go by exact bytes; nothing from the event or click side is
//! interned just to look it up.

use crate::csv;
use crate::hash::{FastMap, map_with_capacity};
use crate::id::pack;

/// Exact-bytes lookup from an ID string to a dense index.
pub struct Lookup {
    packed: FastMap<u64, u32>,
    other: FastMap<Box<[u8]>, u32>,
}

impl Lookup {
    fn new(n: usize) -> Self {
        Lookup { packed: map_with_capacity(n), other: map_with_capacity(0) }
    }

    fn insert(&mut self, key: &[u8], value: u32) {
        match pack(key) {
            Some(code) => {
                self.packed.insert(code, value);
            }
            None => {
                self.other.insert(key.into(), value);
            }
        }
    }

    #[inline]
    pub fn get(&self, key: &[u8]) -> Option<u32> {
        match pack(key) {
            Some(code) => self.packed.get(&code).copied(),
            None => {
                if self.other.is_empty() {
                    None
                } else {
                    self.other.get(key).copied()
                }
            }
        }
    }
}

/// Data lines (after the header) of a two-column file, dialect-checked, both fields non-empty.
fn rows<'a>(bytes: &'a [u8], mut f: impl FnMut(&'a [u8], &'a [u8])) {
    let start = match memchr::memchr(b'\n', bytes) {
        Some(i) => i + 1,
        None => bytes.len(),
    };
    csv::for_each_line(&bytes[start..], |line| {
        if let Some([a, b]) = csv::split::<2>(line) {
            if !a.is_empty() {
                f(a, b);
            }
        }
    });
}

pub const UNKNOWN_CATEGORY: &[u8] = b"unknown";

pub struct Products {
    lookup: Lookup,
    /// Category names, sorted byte-wise, so a smaller index is a byte-wise smaller name.
    pub categories: Vec<Vec<u8>>,
    pub unknown: u32,
}

impl Products {
    pub fn parse(bytes: &[u8]) -> Products {
        let mut pairs: Vec<(&[u8], &[u8])> = Vec::new();
        rows(bytes, |product, category| pairs.push((product, category)));
        let mut categories: Vec<Vec<u8>> = pairs.iter().map(|(_, c)| c.to_vec()).collect();
        categories.push(UNKNOWN_CATEGORY.to_vec());
        categories.sort();
        categories.dedup();
        let index: FastMap<&[u8], u32> =
            categories.iter().enumerate().map(|(i, c)| (c.as_slice(), i as u32)).collect();
        let mut lookup = Lookup::new(pairs.len());
        for (product, category) in &pairs {
            lookup.insert(product, index[category]);
        }
        let unknown = index[UNKNOWN_CATEGORY];
        Products { lookup, categories, unknown }
    }

    /// The category index of a product ID (the `unknown` category if it is not in the file).
    #[inline]
    pub fn category(&self, product: &[u8]) -> u32 {
        self.lookup.get(product).unwrap_or(self.unknown)
    }
}

pub const ORGANIC: &[u8] = b"organic";

pub struct Campaigns {
    lookup: Lookup,
    /// Campaign IDs and channels by index; index `ids.len()` is `organic`.
    pub ids: Vec<Vec<u8>>,
    pub channels: Vec<Vec<u8>>,
}

impl Campaigns {
    pub fn parse(bytes: &[u8]) -> Campaigns {
        let mut lookup = Lookup::new(4096);
        let mut ids = Vec::new();
        let mut channels = Vec::new();
        let mut seen: FastMap<Vec<u8>, u32> = map_with_capacity(4096);
        rows(bytes, |campaign, channel| {
            if let Some(&i) = seen.get(campaign) {
                // campaign_id is unique in a clean file; if not, the last line wins.
                channels[i as usize] = channel.to_vec();
                return;
            }
            let i = ids.len() as u32;
            seen.insert(campaign.to_vec(), i);
            lookup.insert(campaign, i);
            ids.push(campaign.to_vec());
            channels.push(channel.to_vec());
        });
        Campaigns { lookup, ids, channels }
    }

    #[inline]
    pub fn get(&self, campaign: &[u8]) -> Option<u32> {
        self.lookup.get(campaign)
    }

    /// The index used for `organic`.
    pub fn organic(&self) -> u32 {
        self.ids.len() as u32
    }

    pub fn count_with_organic(&self) -> usize {
        self.ids.len() + 1
    }

    pub fn id_bytes(&self, i: u32) -> &[u8] {
        self.ids.get(i as usize).map(|v| v.as_slice()).unwrap_or(ORGANIC)
    }

    pub fn channel_bytes(&self, i: u32) -> &[u8] {
        self.channels.get(i as usize).map(|v| v.as_slice()).unwrap_or(ORGANIC)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn products() {
        let p = Products::parse(b"product_id,category\np1,shoes\np2,books\n\" p3\",zoo\n");
        assert_eq!(p.categories, vec![b"books".to_vec(), b"shoes".to_vec(), b"unknown".to_vec(), b"zoo".to_vec()]);
        assert_eq!(p.category(b"p1"), 1);
        assert_eq!(p.category(b"p2"), 0);
        assert_eq!(p.category(b" p3"), 3);
        assert_eq!(p.category(b"p9"), p.unknown);
        assert_eq!(p.category(b" p1"), p.unknown);
        assert_eq!(p.category(b"P1"), p.unknown);
    }

    #[test]
    fn campaigns() {
        let c = Campaigns::parse(b"campaign_id,channel\nc_1,search\nc_2,social");
        assert_eq!(c.get(b"c_1"), Some(0));
        assert_eq!(c.get(b"c_2"), Some(1));
        assert_eq!(c.get(b" c_1"), None);
        assert_eq!(c.get(b"C_1"), None);
        assert_eq!(c.id_bytes(c.organic()), b"organic");
        assert_eq!(c.channel_bytes(1), b"social");
    }
}
