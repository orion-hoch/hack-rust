//! Person partitions (move 2), one task each: sortlab cuts the sessions of every person from the
//! 8-byte survivor records (no sort of events) and places each purchase in its session; then
//! sortlab's walk groups the clicks by the same local person index, sorts them per person, credits
//! by two pointers (a ts tie is resolved through the click_id bytes only when it is the credited
//! group), counts assists by one sweep (the interval union) and makes the user-day rows; here
//! only the names are added. Nothing is quadratic in a person's size.

use parselab::{Dataset, Stream};

use crate::fast::agg::FastAgg;
use crate::fast::scan::{CLICK_REACH, Rows, SESSION_GAP};
const _: () = assert!(CLICK_REACH == sortlab::walk::REACH && DAY == sortlab::walk::DAY as i64);

// sortlab cuts sessions at its own constant: it must be the rules' 30 minutes.
const _: () = assert!(SESSION_GAP == sortlab::GAP as i64);
use crate::ts::DAY;

use super::tokens::Tokens;

pub struct PeopleCtx<'a, 'b> {
    pub ds: &'a Dataset,
    pub tok: &'b Tokens<'a>,
    /// The sortlab session kernel.
    pub kernel: sortlab::Kernel,
    /// Click base, day shift, cells shape for sortlab's walk.
    pub walk: sortlab::walk::Params,
    /// Absolute day number of relative day 0.
    pub day0: i32,
    /// Person partition q holds the person tokens q << shift .. (q + 1) << shift.
    pub shift: u32,
}

pub struct Worker {
    /// The user_daily file this worker wrote at the end of the session stage, or its error.
    pub wrote: bool,
    pub write_err: Option<std::io::Error>,
    /// user_daily files this worker has written (its next file's sequence number), and the
    /// pool participant it runs on (in the file names).
    pub files: u32,
    pub index: usize,
    /// A check failed (a purchase outside every session, a purchase of an absent person, a bad
    /// campaign or day): the run falls back.
    pub bad: bool,
    pub rows: Rows,
    pub agg: FastAgg,
    pub sessions: u64,
    pub events: u64,
    buys: Vec<[u64; 2]>,
    cl: Vec<[u64; 2]>,
    sc: sortlab::Scratch,
    out: sortlab::Out,
    ws: sortlab::walk::WalkScratch,
    wrows: sortlab::walk::Rows,
}

/// The click_id of the click line at `loc`: its first field, unquoted (the line was accepted by
/// the parser, so its quoting is legal), read without scanning the rest of the line.
fn click_id(ds: &Dataset, loc: parselab::Locator) -> &[u8] {
    let data = ds.stream(Stream::Clicks)[loc.file() as usize].bytes();
    let rest = &data[loc.offset() as usize..];
    let end = rest.iter().position(|&c| c == b',' || c == b'\n').unwrap_or(rest.len());
    let f = &rest[..end];
    parselab::csv::unquote(f).unwrap_or(f)
}

impl Worker {
    /// `est_rows`: a guess of the user-day rows this worker will write. The row columns are
    /// reserved up front, each at least 2 MB, so they are huge-page mappings that never move
    /// (the allocator advises 2 MB+ mappings MADV_HUGEPAGE): growing them on 4 KB pages was most
    /// of the stage's 20K page faults.
    pub fn new(agg: FastAgg, est_rows: usize) -> Self {
        const MB2: usize = 2 << 20;
        let rows_n = est_rows.max(MB2 / 4);
        let persons_n = (est_rows / 2).max(MB2 / 4);
        let mut rows = Rows::new();
        rows.names.reserve((persons_n * 16).max(MB2));
        rows.name_offsets.reserve(persons_n);
        rows.runs.reserve(persons_n);
        rows.day.reserve(rows_n);
        rows.sessions.reserve(rows_n);
        rows.events.reserve(rows_n);
        rows.revenue.reserve(rows_n.max(MB2 / 8));
        rows.top.reserve(rows_n.max(MB2));
        Worker {
            wrote: false,
            write_err: None,
            files: 0,
            index: 0,
            bad: false,
            rows,
            agg,
            sessions: 0,
            events: 0,
            buys: Vec::new(),
            cl: Vec::new(),
            sc: sortlab::Scratch::new(),
            out: sortlab::Out::default(),
            ws: sortlab::walk::WalkScratch::new(),
            wrows: sortlab::walk::Rows::default(),
        }
    }

    /// One person partition: survivor records, purchase records and click records of it.
    pub fn partition(&mut self, ev: &[&[u64]], buys: &[&[[u64; 2]]], clicks: &[&[[u64; 2]]], cx: &PeopleCtx, q: usize) {
        let nev: usize = ev.iter().map(|c| c.len()).sum();
        if nev == 0 {
            if buys.iter().any(|c| !c.is_empty()) {
                self.bad = true;
            }
            return;
        }
        self.buys.clear();
        for c in buys {
            self.buys.extend_from_slice(c);
        }
        // Sessions per person (start, event count, in time order) and each purchase's session.
        // Person ids are root tokens and partition q holds the tokens q << shift .. (q + 1) << shift:
        // grouping indexes them directly (a token outside the range is an error: the run falls back).
        if sortlab::sessions_chunks(ev, &self.buys, cx.kernel, Some(((q as u32) << cx.shift, cx.shift)), &mut self.sc, &mut self.out).is_err() {
            self.bad = true;
            return;
        }
        self.cl.clear();
        for c in clicks {
            self.cl.extend_from_slice(c);
        }
        // Clicks into the sessions: credit and assist cells, user-day rows.
        let ds = cx.ds;
        let mut tie = |a: u64, b: u64| click_id(ds, parselab::Locator(a)).cmp(click_id(ds, parselab::Locator(b)));
        let (hot, buy) = self.agg.cells_mut();
        if sortlab::walk::walk(&self.cl, &self.buys, None, &cx.walk, &mut tie, &self.sc, &mut self.ws, &self.out, hot, buy, &mut self.wrows).is_err() {
            self.bad = true;
            return;
        }
        // Rows with the person names.
        let (out, wr, r) = (&self.out, &self.wrows, &mut self.rows);
        let np = out.persons.len();
        const AHEAD: usize = 8;
        for k in 0..np.min(AHEAD) {
            cx.tok.prefetch_key(out.persons[k]);
        }
        // Room for every name of the partition at once (an ID is at most 33 bytes unless
        // FOREIGN; those grow the vector themselves), so the loop does not check capacity.
        r.names.reserve(np * 33);
        r.name_offsets.reserve(np);
        r.runs.reserve(np);
        let mut r0 = 0u32;
        for (k, (&person, &r1)) in out.persons.iter().zip(&wr.row_end).enumerate() {
            if let Some(&next) = out.persons.get(k + AHEAD) {
                cx.tok.prefetch_key(next);
            }
            if r1 > r0 {
                cx.tok.bytes(person, &mut r.names);
                r.name_offsets.push(r.names.len() as u32);
                r.runs.push(r1 - r0);
            }
            r0 = r1;
        }
        let day0 = cx.day0;
        r.day.extend(wr.day.iter().map(|&d| day0 + d as i32));
        r.sessions.extend_from_slice(&wr.sessions);
        r.events.extend_from_slice(&wr.events);
        r.revenue.extend(wr.revenue.iter().map(|&x| x as i64));
        r.top.extend_from_slice(&wr.top);
        self.sessions += out.start.len() as u64;
        self.events += nev as u64;
    }
}
