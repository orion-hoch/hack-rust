//! products.csv, campaigns.csv and manifest.json, and the parse context built from them.

use std::collections::HashMap;

use crate::csv;
use crate::records::{CLICK_BASE_BEFORE_WINDOW_MS, INGEST_BASE_BEFORE_WINDOW_MS};

/// A direct table for IDs of the form `prefix` + canonical decimal (no leading zero), when every
/// ID in the file has that form. Exact: a value not of that form cannot be in the file.
#[derive(Clone, Debug, Default)]
pub struct NumberedIds {
    pub prefix: Vec<u8>,
    /// by number; `NONE` where no ID has that number.
    pub by_number: Vec<u16>,
}

impl NumberedIds {
    pub const NONE: u16 = u16::MAX;
    pub(crate) const MAX_NUMBER: u64 = 1 << 24;

    pub(crate) fn number(prefix: &[u8], id: &[u8]) -> Option<u64> {
        let digits = id.strip_prefix(prefix)?;
        if digits.is_empty() || digits.len() > 9 || (digits.len() > 1 && digits[0] == b'0') {
            return None;
        }
        if !digits.iter().all(u8::is_ascii_digit) {
            return None;
        }
        Some(digits.iter().fold(0u64, |v, &c| v * 10 + (c - b'0') as u64))
    }

    /// Builds the table if every id fits; `values[i]` is stored for `ids[i]`.
    fn build(prefix: &[u8], ids: &[Vec<u8>], values: &[u16]) -> Option<Self> {
        let mut by_number = Vec::new();
        for (id, &v) in ids.iter().zip(values) {
            let n = Self::number(prefix, id)?;
            if n >= Self::MAX_NUMBER || v == Self::NONE {
                return None;
            }
            let n = n as usize;
            if by_number.len() <= n {
                by_number.resize(n + 1, Self::NONE);
            }
            by_number[n] = v;
        }
        Some(NumberedIds { prefix: prefix.to_vec(), by_number })
    }

    #[inline]
    pub fn get(&self, id: &[u8]) -> u16 {
        match Self::number(&self.prefix, id) {
            Some(n) if (n as usize) < self.by_number.len() => self.by_number[n as usize],
            _ => Self::NONE,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Products {
    /// Category names by code: every name in products.csv plus `unknown`, sorted byte-wise, so
    /// code order is name order (RECORDS.md v2). At most 255 names.
    pub categories: Vec<Vec<u8>>,
    /// The code of `unknown` (a product not in the file).
    pub unknown: u8,
    pub map: HashMap<Vec<u8>, u8>,
    /// Present when every product_id is `p` + canonical decimal.
    pub numbered: Option<NumberedIds>,
}

impl Products {
    /// Category code of a product_id by exact byte match; the code of `unknown` if absent.
    pub fn code(&self, product_id: &[u8]) -> u8 {
        if let Some(t) = &self.numbered {
            let v = t.get(product_id);
            return if v == NumberedIds::NONE { self.unknown } else { v as u8 };
        }
        self.map.get(product_id).copied().unwrap_or(self.unknown)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Campaigns {
    /// campaign_id by row number (row order of campaigns.csv).
    pub ids: Vec<Vec<u8>>,
    pub channels: Vec<Vec<u8>>,
    pub map: HashMap<Vec<u8>, u16>,
    /// Present when every campaign_id is `c_` + canonical decimal.
    pub numbered: Option<NumberedIds>,
}

impl Campaigns {
    pub fn number(&self, campaign_id: &[u8]) -> Option<u16> {
        if let Some(t) = &self.numbered {
            let v = t.get(campaign_id);
            return (v != NumberedIds::NONE).then_some(v);
        }
        self.map.get(campaign_id).copied()
    }
}

/// Why the fast path cannot take a dataset (the run goes to the general engine).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fallback {
    Manifest(String),
    WindowTooLong,
    TooManyCategories(usize),
    TooManyCampaigns(usize),
    DuplicateId(&'static str),
    TooManyFiles(&'static str, usize),
    FileTooLarge(String),
}

/// Calls `f(fields)` for every well-formed 2-field data line, without allocating per line.
fn pairs<'a>(data: &'a [u8], mut f: impl FnMut(&'a [u8], &'a [u8])) {
    let mut fields: Vec<&'a [u8]> = Vec::with_capacity(4);
    csv::for_each_line(data, 0, data.len(), |_, line| {
        if csv::split(line, &mut fields) && fields.len() == 2 {
            f(fields[0], fields[1]);
        }
    });
}

/// products.csv: `product_id,category`. Malformed lines are dropped as everywhere (the file is
/// promised clean). A repeated product_id is outside the contract and refuses the fast path.
/// When every product_id is `p` + canonical decimal, the direct table is the only lookup.
pub fn products(data: &[u8]) -> Result<Products, Fallback> {
    if let Some((names, mut by_number)) = products_numbered(data) {
        let (categories, code_of_name, unknown) = category_codes(&names)?;
        for v in by_number.iter_mut() {
            if *v != NumberedIds::NONE {
                *v = code_of_name[*v as usize] as u16;
            }
        }
        let numbered = Some(NumberedIds { prefix: b"p".to_vec(), by_number });
        return Ok(Products { categories, unknown, map: HashMap::new(), numbered });
    }
    products_general(data)
}

/// Sorted category names (plus `unknown`), each first-seen name's code, and `unknown`'s code.
fn category_codes(names: &[&[u8]]) -> Result<(Vec<Vec<u8>>, Vec<u8>, u8), Fallback> {
    let mut categories: Vec<Vec<u8>> = names.iter().map(|n| n.to_vec()).collect();
    categories.push(b"unknown".to_vec());
    categories.sort_unstable();
    categories.dedup();
    if categories.len() > 255 {
        return Err(Fallback::TooManyCategories(categories.len()));
    }
    let code_of_name: Vec<u8> =
        names.iter().map(|n| categories.binary_search_by(|c| c.as_slice().cmp(n)).unwrap() as u8).collect();
    let unknown = categories.binary_search_by(|c| c.as_slice().cmp(b"unknown")).unwrap() as u8;
    Ok((categories, code_of_name, unknown))
}

/// One pass over the common products.csv: after the header, every line `p<canonical decimal
/// below 2^24>,<category>` with exactly one comma and no quote, the numbers distinct, at most
/// 1,000 names. Gives the first-seen names and, by number, each product's name index (NONE
/// where absent): what `products_general` computes on such a file. None on anything else (the
/// general path then decides, errors included). 64-byte blocks: newline, comma and quote masks.
fn products_numbered(data: &[u8]) -> Option<(Vec<&[u8]>, Vec<u16>)> {
    let hdr = data.iter().position(|&c| c == b'\n')? + 1;
    let mut names: Vec<&[u8]> = Vec::new();
    // By a hash of the name, 2,048 slots for at most 1,000 names: (first 8 bytes, last 8 bytes,
    // length << 16 | index + 1); 0 in the last = empty. Up to 16 bytes the three are the name.
    let mut slots = vec![(0u64, 0u64, 0u32); 2048];
    // By number, grown by doubling (NONE-filled); cut to the largest number + 1 at the end.
    let mut by: Vec<u16> = vec![NumberedIds::NONE; data.len() / 12 + 16];
    let mut top = 0usize;
    let mut take = |ls: usize, comma: usize, end: usize| -> Option<()> {
        let id = &data[ls..comma];
        let digits = id.strip_prefix(b"p")?;
        if digits.is_empty() || digits.len() > 9 || (digits.len() > 1 && digits[0] == b'0') {
            return None;
        }
        let mut n = 0usize;
        for &c in digits {
            let d = c.wrapping_sub(b'0');
            if d > 9 {
                return None;
            }
            n = n * 10 + d as usize;
        }
        if n as u64 >= NumberedIds::MAX_NUMBER {
            return None;
        }
        let cat = &data[comma + 1..end];
        let (a, z) = ends8(data, comma + 1, end);
        let tag = (cat.len() as u32) << 16;
        let mut h = ((a ^ z.rotate_left(29) ^ cat.len() as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93) >> 53) as usize;
        let at = loop {
            let (sa, sz, st) = slots[h];
            if st == 0 {
                if names.len() >= 1000 || cat.len() >= 1 << 15 {
                    return None;
                }
                names.push(cat);
                slots[h] = (a, z, tag | names.len() as u32);
                break names.len() as u16 - 1;
            }
            if st & !0xFFFF == tag && sa == a && sz == z && (cat.len() <= 16 || names[(st & 0xFFFF) as usize - 1] == cat) {
                break (st & 0xFFFF) as u16 - 1;
            }
            h = (h + 1) & 2047;
        };
        if n >= by.len() {
            by.resize((n + 1).max(2 * by.len()), NumberedIds::NONE);
        }
        if by[n] != NumberedIds::NONE {
            return None;
        }
        by[n] = at;
        top = top.max(n + 1);
        Some(())
    };
    let (mut ls, mut comma) = (hdr, usize::MAX);
    let mut b = hdr;
    while b < data.len() {
        let (nl, cm, qt) = masks64(&data[b..]);
        if qt != 0 {
            return None;
        }
        let mut m = nl | cm;
        while m != 0 {
            let pos = b + m.trailing_zeros() as usize;
            m &= m - 1;
            if data[pos] == b',' {
                if comma != usize::MAX {
                    return None;
                }
                comma = pos;
            } else {
                if comma == usize::MAX {
                    return None;
                }
                take(ls, comma, pos)?;
                ls = pos + 1;
                comma = usize::MAX;
            }
        }
        b += 64;
    }
    if ls < data.len() {
        // A final line without a newline.
        if comma == usize::MAX {
            return None;
        }
        take(ls, comma, data.len())?;
    }
    by.truncate(top);
    Some((names, by))
}

/// The first and last eight bytes of `data[from..to]`, zero-padded when shorter (then both are
/// the whole name). Plain 8-byte loads while they stay inside `data`.
#[inline]
fn ends8(data: &[u8], from: usize, to: usize) -> (u64, u64) {
    let len = to - from;
    let at = |i: usize| u64::from_le_bytes(data[i..i + 8].try_into().unwrap());
    if len >= 8 {
        return (at(from), at(to - 8));
    }
    let a = if from + 8 <= data.len() {
        at(from) & ((1u64 << (8 * len)) - 1)
    } else {
        let mut b = [0u8; 8];
        b[..len].copy_from_slice(&data[from..to]);
        u64::from_le_bytes(b)
    };
    (a, a)
}

/// Newline, comma and quote masks of up to 64 bytes from the start of `d` (bit i: byte i).
#[inline]
fn masks64(d: &[u8]) -> (u64, u64, u64) {
    #[cfg(target_arch = "x86_64")]
    if d.len() >= 64 {
        use core::arch::x86_64::*;
        // SAFETY: SSE2 is in every x86-64; the four loads are inside `d` (64 bytes or more).
        unsafe {
            let (nl, cm, qt) = (_mm_set1_epi8(b'\n' as i8), _mm_set1_epi8(b',' as i8), _mm_set1_epi8(b'"' as i8));
            let (mut a, mut c, mut q) = (0u64, 0u64, 0u64);
            for k in 0..4 {
                let v = _mm_loadu_si128(d.as_ptr().add(16 * k) as *const __m128i);
                a |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, nl)) as u16 as u64) << (16 * k);
                c |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, cm)) as u16 as u64) << (16 * k);
                q |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, qt)) as u16 as u64) << (16 * k);
            }
            return (a, c, q);
        }
    }
    let (mut a, mut c, mut q) = (0u64, 0u64, 0u64);
    for (i, &x) in d.iter().take(64).enumerate() {
        a |= ((x == b'\n') as u64) << i;
        c |= ((x == b',') as u64) << i;
        q |= ((x == b'"') as u64) << i;
    }
    (a, c, q)
}

/// products.csv by the general rules (any id, quoting, dropped lines, every refusal).
fn products_general(data: &[u8]) -> Result<Products, Fallback> {
    // Distinct category names in first-seen order (at most a few hundred), and per row its index.
    let mut names: Vec<&[u8]> = Vec::new();
    let mut index: HashMap<&[u8], u16> = HashMap::new();
    let mut rows: Vec<(&[u8], u16)> = Vec::with_capacity(data.len() / 12);
    let mut last: (&[u8], u16) = (&[], u16::MAX);
    pairs(data, |id, category| {
        let at = if category == last.0 && last.1 != u16::MAX {
            last.1
        } else {
            let at = *index.entry(category).or_insert_with(|| {
                names.push(category);
                (names.len() - 1) as u16
            });
            last = (category, at);
            at
        };
        rows.push((id, at));
    });
    if names.len() > 1000 {
        return Err(Fallback::TooManyCategories(names.len()));
    }
    let (categories, code_of_name, unknown) = category_codes(&names)?;
    // Direct table when every id is numbered.
    let mut by_number: Vec<u16> = Vec::new();
    let mut numbered_ok = true;
    for &(id, name) in &rows {
        match NumberedIds::number(b"p", id) {
            Some(n) if n < NumberedIds::MAX_NUMBER => {
                let n = n as usize;
                if by_number.len() <= n {
                    by_number.resize(n + 1, NumberedIds::NONE);
                }
                if by_number[n] != NumberedIds::NONE {
                    return Err(Fallback::DuplicateId("products.csv"));
                }
                by_number[n] = code_of_name[name as usize] as u16;
            }
            _ => {
                numbered_ok = false;
                break;
            }
        }
    }
    if numbered_ok {
        let numbered = Some(NumberedIds { prefix: b"p".to_vec(), by_number });
        return Ok(Products { categories, unknown, map: HashMap::new(), numbered });
    }
    let mut map = HashMap::with_capacity(rows.len());
    for &(id, name) in &rows {
        if map.insert(id.to_vec(), code_of_name[name as usize]).is_some() {
            return Err(Fallback::DuplicateId("products.csv"));
        }
    }
    Ok(Products { categories, unknown, map, numbered: None })
}

/// campaigns.csv: `campaign_id,channel`; the campaign number is the row's position among the
/// well-formed data rows.
pub fn campaigns(data: &[u8]) -> Result<Campaigns, Fallback> {
    let mut ids = Vec::new();
    let mut channels = Vec::new();
    pairs(data, |id, channel| {
        ids.push(id.to_vec());
        channels.push(channel.to_vec());
    });
    // RECORDS.md v2: 65,536 or more campaigns refuse the fast path (the number has 16 bits).
    if ids.len() >= 1 << 16 {
        return Err(Fallback::TooManyCampaigns(ids.len()));
    }
    let values: Vec<u16> = (0..ids.len() as u16).collect();
    if let Some(numbered) = NumberedIds::build(b"c_", &ids, &values) {
        // build() refuses repeated numbers only by overwriting; check them here.
        let distinct = numbered.by_number.iter().filter(|&&v| v != NumberedIds::NONE).count();
        if distinct != ids.len() {
            return Err(Fallback::DuplicateId("campaigns.csv"));
        }
        return Ok(Campaigns { ids, channels, map: HashMap::new(), numbered: Some(numbered) });
    }
    let mut map = HashMap::with_capacity(ids.len());
    for (i, id) in ids.iter().enumerate() {
        if map.insert(id.clone(), i as u16).is_some() {
            return Err(Fallback::DuplicateId("campaigns.csv"));
        }
    }
    Ok(Campaigns { ids, channels, map, numbered: None })
}

/// The two window bounds from manifest.json, as ms since the epoch.
pub fn manifest(data: &[u8]) -> Result<(i64, i64), Fallback> {
    let text = std::str::from_utf8(data).map_err(|_| Fallback::Manifest("not UTF-8".into()))?;
    let value = |name: &str| -> Result<i64, Fallback> {
        let key = format!("\"{name}\"");
        let at = text.find(&key).ok_or_else(|| Fallback::Manifest(format!("no {name}")))?;
        let rest = text[at + key.len()..].trim_start();
        let rest = rest.strip_prefix(':').ok_or_else(|| Fallback::Manifest("no colon".into()))?;
        let rest = rest.trim_start();
        let rest = rest.strip_prefix('"').ok_or_else(|| Fallback::Manifest("no string".into()))?;
        let end = rest.find('"').ok_or_else(|| Fallback::Manifest("open string".into()))?;
        crate::ts::parse_ts(rest[..end].as_bytes())
            .filter(|&v| v != i64::MAX)
            .ok_or_else(|| Fallback::Manifest(format!("bad {name}")))
    };
    Ok((value("window_start")?, value("window_end")?))
}

/// Everything a parser needs besides the bytes.
#[derive(Clone, Debug)]
pub struct Ctx {
    pub window_start_ms: i64,
    pub window_end_ms: i64,
    /// floor((window_start - 30 days) / 1 s): ingest is stored as whole seconds since this.
    pub ingest_base_s: i64,
    /// window_start - 7 days: click ts is stored as ms since this.
    pub click_base_ms: i64,
    pub products: Products,
    pub campaigns: Campaigns,
}

impl Ctx {
    pub fn new(window: (i64, i64), products: Products, campaigns: Campaigns) -> Result<Ctx, Fallback> {
        let (ws, we) = window;
        // RECORDS.md: window_end - window_start >= 2^31 ms refuses the fast path. Also refuse an
        // empty or reversed window and a click range that does not fit 32 bits.
        if we <= ws || we - ws >= 1 << 31 || we - (ws - CLICK_BASE_BEFORE_WINDOW_MS) >= 1 << 32 {
            return Err(Fallback::WindowTooLong);
        }
        Ok(Ctx {
            window_start_ms: ws,
            window_end_ms: we,
            ingest_base_s: (ws - INGEST_BASE_BEFORE_WINDOW_MS).div_euclid(1000),
            click_base_ms: ws - CLICK_BASE_BEFORE_WINDOW_MS,
            products,
            campaigns,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn same(data: &[u8]) {
        let a = products(data);
        let b = products_general(data);
        match (a, b) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a.categories, b.categories);
                assert_eq!(a.unknown, b.unknown);
                assert_eq!(a.map, b.map);
                let (na, nb) = (a.numbered.map(|n| (n.prefix, n.by_number)), b.numbered.map(|n| (n.prefix, n.by_number)));
                assert_eq!(na, nb, "{:?}", String::from_utf8_lossy(&data[..data.len().min(200)]));
            }
            (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}")),
            (a, b) => panic!("one path refused: {:?} vs {:?}", a.err(), b.err()),
        }
    }

    #[test]
    fn products_one_pass_is_the_general_rule() {
        for d in [
            &b"product_id,category\np1,pets\np2,baby\np3,pets\n"[..],
            b"product_id,category\np1,pets\np2,baby\np3,pets",
            b"product_id,category\n",
            b"product_id,category",
            b"",
            b"product_id,category\np1,pets\n\np2,baby\n",
            b"product_id,category\np1,\"pets\"\np2,baby\n",
            b"\"product\"_id,category\np1,pets\np2,baby\n",
            b"product_id,category\np1,pets\np1,baby\n",
            b"product_id,category\np01,pets\n",
            b"product_id,category\np0,pets\np1,a,b\n",
            b"product_id,category\nq1,pets\np2,baby\n",
            b"product_id,category\np1,pets\r\np2,baby\r\n",
            b"product_id,category\np16777216,pets\n",
            b"product_id,category\np16777215,pets\np5,\n",
            b"product_id,category\np1234567890,pets\n",
            b"product_id,category\np,pets\n",
        ] {
            same(d);
        }
        // Many names, long names, every block boundary.
        let mut s = b"product_id,category\n".to_vec();
        for i in 0..5000u32 {
            let name = format!("category-number-{}-{}", i % 300, "x".repeat((i % 23) as usize));
            s.extend_from_slice(format!("p{},{}\n", (i * 7919) % 100_003, name).as_bytes());
        }
        same(&s);
        let mut t = b"product_id,category\n".to_vec();
        for i in 0..1100u32 {
            t.extend_from_slice(format!("p{i},c{i}\n").as_bytes());
        }
        same(&t);
        // Random small mutations of a clean file.
        let base = s[..4000].to_vec();
        let mut x = 0x1234_5678_9abc_def0u64;
        for _ in 0..3000 {
            let mut d = base.clone();
            for _ in 0..3 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let at = (x as usize >> 8) % d.len();
                d[at] = b",\n\"p0x9"[(x % 7) as usize];
            }
            same(&d);
        }
        if let Ok(dir) = std::env::var("V4_DATA") {
            let real = std::fs::read(std::path::Path::new(&dir).join("products.csv")).unwrap();
            assert!(products_numbered(&real).is_some(), "the real file takes the one-pass path");
            same(&real);
        }
    }
}
