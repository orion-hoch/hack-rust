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
