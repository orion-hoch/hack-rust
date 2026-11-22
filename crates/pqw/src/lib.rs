//! pqw: a zero-dependency writer for flat, uncompressed Parquet tables.
//!
//! One call writes one complete file with one row group (no row group when there are no rows; the
//! schema is still there). Columns are borrowed slices; nothing global is touched, so any number of
//! threads may write their own files at once.
//!
//! Default encodings (chosen by measurement, see the crate's bench): strings DELTA_BYTE_ARRAY,
//! integers and dates DELTA_BINARY_PACKED, coded strings a dictionary page with RLE_DICTIONARY indices.
//! No compression, statistics or page indexes.
//!
//! ```
//! use pqw::{Ids, Strings, UserDaily, NULL8};
//! // two users: "u1" on two days, "u22" on one; user ids given once each, as runs
//! let ids = Strings::new(b"u1u22", &[0, 2, 5]);
//! let runs = [2u32, 1];
//! let cats: [&[u8]; 2] = [b"baby", b"books"];
//! let t = UserDaily {
//!     user_id: Ids::Runs { values: ids, runs: &runs },
//!     day: &[21162, 21163, 21162], // days since 1970-01-01
//!     sessions: &[1, 2, 1],
//!     events: &[9, 14, 1],
//!     revenue_cents: &[0, 5_000_000_000, 0],
//!     top_category: &[NULL8, 1, NULL8], // codes into cats
//!     categories: &cats,
//! };
//! let dir = std::env::temp_dir().join(format!("pqw-doc-{}", std::process::id()));
//! std::fs::create_dir_all(&dir).unwrap();
//! let bytes = t.write_file(dir.join("part-0.parquet")).unwrap();
//! assert!(bytes > 0);
//! ```
//!
//! The general form is `write_table` / `write_file` / `write_fd` over a list of `Col`.

use std::io::{self, Write};

pub mod enc;
mod pack_gen;
mod thrift;

use thrift::{Thrift, BINARY, I32 as T_I32, STRUCT};

/// Null code for `Codes::U8`.
pub const NULL8: u8 = u8::MAX;
/// Null code for `Codes::U16`.
pub const NULL16: u16 = u16::MAX;

/// Strings as one byte buffer plus `n + 1` offsets: value i is `bytes[offsets[i]..offsets[i + 1]]`.
#[derive(Clone, Copy, Debug)]
pub struct Strings<'a> {
    pub bytes: &'a [u8],
    pub offsets: &'a [u32],
}

impl<'a> Strings<'a> {
    pub fn new(bytes: &'a [u8], offsets: &'a [u32]) -> Self {
        Strings { bytes, offsets }
    }
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    pub fn get(&self, i: usize) -> &'a [u8] {
        &self.bytes[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }
}

/// Per-row codes into a dictionary. `NULL8` / `NULL16` mark nulls.
#[derive(Clone, Copy, Debug)]
pub enum Codes<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
}

/// One column's values.
#[derive(Clone, Copy, Debug)]
pub enum Data<'a> {
    /// Required UTF-8 strings.
    Str(Strings<'a>),
    /// Required UTF-8 strings given as runs: `values[i]` repeated `runs[i]` times, in order. Same file as
    /// `Str` with every row spelled out, but the writer never compares repeated rows.
    StrRuns { values: Strings<'a>, runs: &'a [u32] },
    /// Strings given as codes into `dict`; nullable if `nullable` (null code = NULL8 / NULL16).
    Dict { codes: Codes<'a>, dict: &'a [&'a [u8]], nullable: bool },
    /// Required DATE, days since 1970-01-01.
    Date(&'a [i32]),
    /// Required INT32.
    I32(&'a [i32]),
    /// Required integer: written as INT32 if every value fits, else INT64.
    U32(&'a [u32]),
    /// Required INT64.
    I64(&'a [i64]),
}

impl Data<'_> {
    pub fn len(&self) -> usize {
        match self {
            Data::Str(s) => s.len(),
            Data::StrRuns { runs, .. } => runs.iter().map(|&r| r as usize).sum(),
            Data::Dict { codes: Codes::U8(c), .. } => c.len(),
            Data::Dict { codes: Codes::U16(c), .. } => c.len(),
            Data::Date(v) | Data::I32(v) => v.len(),
            Data::U32(v) => v.len(),
            Data::I64(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Encoding request for a column. `Auto` picks the measured default for the data kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Enc {
    Auto,
    /// PLAIN values.
    Plain,
    /// Integers: DELTA_BINARY_PACKED.
    Delta,
    /// Integers: dictionary page holding every value from min to max, RLE_DICTIONARY indices
    /// (falls back to Delta if max - min >= 65536). Dict data: its dictionary.
    Dict,
    /// Strings: DELTA_LENGTH_BYTE_ARRAY.
    DeltaLength,
    /// Strings: DELTA_BYTE_ARRAY (prefix shared with the previous row).
    DeltaByteArray,
}

pub struct Col<'a> {
    pub name: &'a str,
    pub data: Data<'a>,
    pub enc: Enc,
}

impl<'a> Col<'a> {
    pub fn new(name: &'a str, data: Data<'a>) -> Self {
        Col { name, data, enc: Enc::Auto }
    }
    pub fn enc(mut self, enc: Enc) -> Self {
        self.enc = enc;
        self
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    /// Rows per data page.
    pub page_rows: usize,
    /// For a table with zero rows: write one row group with empty column chunks (as pyarrow does)
    /// instead of no row group at all.
    pub empty_row_group: bool,
    /// Bytes gathered before each write call; pieces at least this large are written directly.
    pub buffer_bytes: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options { page_rows: 1 << 16, empty_row_group: false, buffer_bytes: 1 << 19 }
    }
}

// Parquet enums.
const PT_INT32: i32 = 1;
const PT_INT64: i32 = 2;
const PT_BYTE_ARRAY: i32 = 6;
const E_PLAIN: i32 = 0;
const E_RLE: i32 = 3;
const E_DELTA_BINARY_PACKED: i32 = 5;
const E_DELTA_LENGTH_BYTE_ARRAY: i32 = 6;
const E_DELTA_BYTE_ARRAY: i32 = 7;
const E_RLE_DICTIONARY: i32 = 8;
const CT_UTF8: i32 = 0;
const CT_DATE: i32 = 6;
const CREATED_BY: &[u8] = b"pqw version 0.1.0";

/// Buffered output that tracks the file offset. Pieces of at least `cap` bytes are written directly.
struct Sink<W: Write> {
    w: W,
    buf: Vec<u8>,
    cap: usize,
    off: u64,
}

impl<W: Write> Sink<W> {
    fn put(&mut self, b: &[u8]) -> io::Result<()> {
        if b.len() >= self.cap {
            if !self.buf.is_empty() {
                self.w.write_all(&self.buf)?;
                self.buf.clear();
            }
            self.w.write_all(b)?;
        } else {
            if self.buf.len() + b.len() > self.cap {
                self.w.write_all(&self.buf)?;
                self.buf.clear();
            }
            self.buf.extend_from_slice(b);
        }
        self.off += b.len() as u64;
        Ok(())
    }
    fn finish(&mut self) -> io::Result<()> {
        self.w.write_all(&self.buf)?;
        self.buf.clear();
        self.w.flush()
    }
}

/// Reusable per-call buffers.
#[derive(Default)]
struct Scratch {
    page: Vec<u8>,
    hdr: Vec<u8>,
    u32s: Vec<u32>,
    u8s: Vec<u8>,
    bits: Vec<u8>,
    i32a: Vec<i32>,
    i32b: Vec<i32>,
}

struct Chunk {
    phys: i32,
    encodings: Vec<i32>,
    num_values: i64,
    start: i64,
    size: i64,
    data_off: i64,
    dict_off: Option<i64>,
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

/// Write one Parquet file with these columns to `w`. Returns the number of bytes written.
pub fn write_table<W: Write>(w: W, cols: &[Col], opt: &Options) -> io::Result<u64> {
    if cols.is_empty() {
        return Err(invalid("no columns".into()));
    }
    let n = cols[0].data.len();
    for c in cols {
        if c.data.len() != n {
            return Err(invalid(format!("column {} has {} rows, expected {}", c.name, c.data.len(), n)));
        }
        if c.name.is_empty() {
            return Err(invalid("empty column name".into()));
        }
        validate(c)?;
    }
    if n > i32::MAX as usize / 2 {
        return Err(invalid("too many rows for one row group".into()));
    }
    let page_rows = opt.page_rows.clamp(1, 1 << 24);
    let cap = opt.buffer_bytes.max(4096);
    let mut sink = Sink { w, buf: Vec::with_capacity(cap), cap, off: 0 };
    let mut sc = Scratch::default();
    sink.put(b"PAR1")?;
    let mut chunks = Vec::with_capacity(cols.len());
    let write_rg = n > 0 || opt.empty_row_group;
    if write_rg {
        for c in cols {
            chunks.push(write_column(&mut sink, c, n, page_rows, &mut sc)?);
        }
    }
    let mut footer = Vec::with_capacity(256 + 128 * cols.len());
    file_metadata(&mut footer, cols, &chunks, n, write_rg);
    let flen = (footer.len() as u32).to_le_bytes();
    sink.put(&footer)?;
    sink.put(&flen)?;
    sink.put(b"PAR1")?;
    sink.finish()?;
    Ok(sink.off)
}

/// Create (or truncate) `path` and write the table to it.
pub fn write_file<P: AsRef<std::path::Path>>(path: P, cols: &[Col], opt: &Options) -> io::Result<u64> {
    let f = std::fs::File::create(path)?;
    write_table(f, cols, opt)
}

/// Write the table to an open file descriptor (not closed afterwards).
#[cfg(unix)]
pub fn write_fd(fd: std::os::fd::BorrowedFd<'_>, cols: &[Col], opt: &Options) -> io::Result<u64> {
    let f = std::fs::File::from(fd.try_clone_to_owned()?);
    write_table(f, cols, opt)
}

fn validate(c: &Col) -> io::Result<()> {
    match c.data {
        Data::StrRuns { values: s, runs } if runs.len() != s.len() => {
            return Err(invalid(format!("{}: {} runs for {} values", c.name, runs.len(), s.len())));
        }
        Data::Str(s) | Data::StrRuns { values: s, .. } => {
            if s.offsets.is_empty() {
                return Err(invalid(format!("{}: offsets must have n + 1 entries", c.name)));
            }
            if !s.offsets.windows(2).all(|w| w[0] <= w[1]) || *s.offsets.last().unwrap() as usize > s.bytes.len() {
                return Err(invalid(format!("{}: offsets must be non-decreasing and within bytes", c.name)));
            }
        }
        Data::Dict { codes, dict, nullable } => {
            // (largest non-null code + 1, null code, any null)
            fn scan<C: Code>(v: &[C], null: C) -> (usize, bool) {
                // branch-free so it vectorises: nulls count as code -1
                let mut end = 0u32;
                let mut nulls = 0u32;
                for &x in v {
                    let isnull = (x == null) as u32;
                    nulls |= isnull;
                    end = end.max((x.u() + 1) * (1 - isnull));
                }
                (end as usize, nulls != 0)
            }
            let ((end, has_null), null) = match codes {
                Codes::U8(v) => (scan(v, NULL8), NULL8 as usize),
                Codes::U16(v) => (scan(v, NULL16), NULL16 as usize),
            };
            if dict.len() > null {
                return Err(invalid(format!("{}: dictionary too large for the code width", c.name)));
            }
            if has_null && !nullable {
                return Err(invalid(format!("{}: null code in a non-nullable column", c.name)));
            }
            if end > dict.len() {
                return Err(invalid(format!("{}: code {} outside dictionary of {}", c.name, end - 1, dict.len())));
            }
        }
        _ => {}
    }
    if matches!(c.data, Data::Str(_)) && matches!(c.enc, Enc::Delta | Enc::Dict) {
        return Err(invalid(format!("{}: encoding {:?} not available for strings", c.name, c.enc)));
    }
    if matches!(c.data, Data::StrRuns { .. }) && !matches!(c.enc, Enc::Auto | Enc::DeltaByteArray | Enc::DeltaLength) {
        return Err(invalid(format!("{}: encoding {:?} not available for strings", c.name, c.enc)));
    }
    if matches!(c.data, Data::Dict { .. }) && !matches!(c.enc, Enc::Auto | Enc::Dict | Enc::Plain) {
        return Err(invalid(format!("{}: encoding {:?} not available for dictionary data", c.name, c.enc)));
    }
    if matches!(c.data, Data::Date(_) | Data::I32(_) | Data::U32(_) | Data::I64(_))
        && matches!(c.enc, Enc::DeltaLength | Enc::DeltaByteArray)
    {
        return Err(invalid(format!("{}: encoding {:?} not available for integers", c.name, c.enc)));
    }
    Ok(())
}

fn data_page_header(out: &mut Vec<u8>, len: usize, nvals: usize, enc: i32) {
    out.clear();
    let mut t = Thrift::new(out);
    t.i32(1, 0); // DATA_PAGE
    t.i32(2, len as i32);
    t.i32(3, len as i32);
    t.begin(5);
    t.i32(1, nvals as i32);
    t.i32(2, enc);
    t.i32(3, E_RLE);
    t.i32(4, E_RLE);
    t.end();
    t.stop();
}

fn dict_page_header(out: &mut Vec<u8>, len: usize, nvals: usize) {
    out.clear();
    let mut t = Thrift::new(out);
    t.i32(1, 2); // DICTIONARY_PAGE
    t.i32(2, len as i32);
    t.i32(3, len as i32);
    t.begin(7);
    t.i32(1, nvals as i32);
    t.i32(2, E_PLAIN);
    t.bool(3, false);
    t.end();
    t.stop();
}

mod col;
use col::{write_column, Code};

fn file_metadata(out: &mut Vec<u8>, cols: &[Col], chunks: &[Chunk], n: usize, write_rg: bool) {
    let mut t = Thrift::new(out);
    t.i32(1, 2); // version
    t.list(2, STRUCT, cols.len() + 1);
    t.elem_begin();
    t.i32(3, 0); // REQUIRED
    t.binary(4, b"schema");
    t.i32(5, cols.len() as i32);
    t.end();
    for (i, c) in cols.iter().enumerate() {
        let (phys, nullable, ct, lt) = match c.data {
            Data::Str(_) | Data::StrRuns { .. } => (PT_BYTE_ARRAY, false, Some(CT_UTF8), Some(1)),
            Data::Dict { nullable, .. } => (PT_BYTE_ARRAY, nullable, Some(CT_UTF8), Some(1)),
            Data::Date(_) => (PT_INT32, false, Some(CT_DATE), Some(6)),
            Data::I32(_) => (PT_INT32, false, None, None),
            Data::I64(_) => (PT_INT64, false, None, None),
            Data::U32(v) => {
                let p = match chunks.get(i) {
                    Some(ch) => ch.phys,
                    None if v.iter().all(|&x| x <= i32::MAX as u32) => PT_INT32,
                    None => PT_INT64,
                };
                (p, false, None, None)
            }
        };
        t.elem_begin();
        t.i32(1, phys);
        t.i32(3, nullable as i32);
        t.binary(4, c.name.as_bytes());
        if let Some(ct) = ct {
            t.i32(6, ct);
        }
        if let Some(lt) = lt {
            t.begin(10); // LogicalType union
            t.begin(lt); // StringType or DateType: empty struct
            t.end();
            t.end();
        }
        t.end();
    }
    t.i64(3, n as i64);
    if write_rg {
        t.list(4, STRUCT, 1);
        t.elem_begin();
        t.list(1, STRUCT, chunks.len());
        for (c, ch) in cols.iter().zip(chunks) {
            t.elem_begin();
            t.i64(2, 0); // file_offset: deprecated, 0 as parquet-cpp writes
            t.begin(3);
            t.i32(1, ch.phys);
            t.list(2, T_I32, ch.encodings.len());
            for &e in &ch.encodings {
                t.elem_i32(e);
            }
            t.list(3, BINARY, 1);
            t.elem_binary(c.name.as_bytes());
            t.i32(4, 0); // UNCOMPRESSED
            t.i64(5, ch.num_values);
            t.i64(6, ch.size);
            t.i64(7, ch.size);
            t.i64(9, ch.data_off);
            if let Some(d) = ch.dict_off {
                t.i64(11, d);
            }
            t.end();
            t.end();
        }
        let total: i64 = chunks.iter().map(|c| c.size).sum();
        t.i64(2, total);
        t.i64(3, n as i64);
        t.i64(5, chunks.first().map(|c| c.start).unwrap_or(4));
        t.i64(6, total);
        t.end();
    } else {
        t.list(4, STRUCT, 0);
    }
    t.binary(6, CREATED_BY);
    t.list(7, STRUCT, cols.len());
    for _ in cols {
        t.elem_begin();
        t.begin(1); // TypeDefinedOrder
        t.end();
        t.end();
    }
    t.stop();
}

/// A required string column for the convenience tables: one value per row, or runs of equal values.
#[derive(Clone, Copy, Debug)]
pub enum Ids<'a> {
    Rows(Strings<'a>),
    Runs { values: Strings<'a>, runs: &'a [u32] },
}

impl<'a> Ids<'a> {
    fn data(self) -> Data<'a> {
        match self {
            Ids::Rows(s) => Data::Str(s),
            Ids::Runs { values, runs } => Data::StrRuns { values, runs },
        }
    }
}

/// The `user_daily` table. `top_category` holds codes into `categories`, NULL8 for null.
pub struct UserDaily<'a> {
    pub user_id: Ids<'a>,
    pub day: &'a [i32],
    pub sessions: &'a [u32],
    pub events: &'a [u32],
    pub revenue_cents: &'a [i64],
    pub top_category: &'a [u8],
    pub categories: &'a [&'a [u8]],
}

impl UserDaily<'_> {
    pub fn columns(&self) -> Vec<Col<'_>> {
        vec![
            Col::new("user_id", self.user_id.data()),
            Col::new("day", Data::Date(self.day)),
            Col::new("sessions", Data::U32(self.sessions)),
            Col::new("events", Data::U32(self.events)),
            Col::new("revenue_cents", Data::I64(self.revenue_cents)),
            Col::new(
                "top_category",
                Data::Dict { codes: Codes::U8(self.top_category), dict: self.categories, nullable: true },
            ),
        ]
    }
    pub fn write_file<P: AsRef<std::path::Path>>(&self, path: P) -> io::Result<u64> {
        write_file(path, &self.columns(), &Options::default())
    }
}

/// The `campaign_daily` table. `channel` holds codes into `channels` (no nulls).
pub struct CampaignDaily<'a> {
    pub campaign_id: Ids<'a>,
    pub channel: &'a [u8],
    pub channels: &'a [&'a [u8]],
    pub day: &'a [i32],
    pub sessions: &'a [u32],
    pub purchasing_sessions: &'a [u32],
    pub revenue_cents: &'a [i64],
    pub assisted_sessions: &'a [u32],
}

impl CampaignDaily<'_> {
    pub fn columns(&self) -> Vec<Col<'_>> {
        vec![
            Col::new("campaign_id", self.campaign_id.data()),
            Col::new("channel", Data::Dict { codes: Codes::U8(self.channel), dict: self.channels, nullable: false }),
            Col::new("day", Data::Date(self.day)),
            Col::new("sessions", Data::U32(self.sessions)),
            Col::new("purchasing_sessions", Data::U32(self.purchasing_sessions)),
            Col::new("revenue_cents", Data::I64(self.revenue_cents)),
            Col::new("assisted_sessions", Data::U32(self.assisted_sessions)),
        ]
    }
    pub fn write_file<P: AsRef<std::path::Path>>(&self, path: P) -> io::Result<u64> {
        write_file(path, &self.columns(), &Options::default())
    }
}
