//! Clicks into sessions, for one person partition, right after `sessions` on the same `Scratch`
//! and `Out`: crediting, assists, per-session revenue, and the user-day rows.
//!
//! Input: click records `[person << 32 | ts_click, campaign << 48 | locator]` (ts_click in ms since
//! the click base, any order; person a token, or with `dense` the local index; u32::MAX = no
//! person), and the purchases given to `sessions` (payload: category | price << 8).
//!
//! Rules (as the engine and the Lean track state them):
//! * A click's ts in the event base is ts_click + base (base = click base - window start).
//! * Credit: per person, clicks in order (ts ascending, click_id descending); a session is
//!   credited to the last click at or before its start if that click is at most 7 days back
//!   (inclusive), else to organic. Equivalently: among the clicks at the latest ts <= start,
//!   the one with the smallest click_id. So tie order matters only for a credited tie group, and
//!   only there is the caller's `tie(loc_a, loc_b)` (click_id bytes of a vs b) called.
//! * Assists: each distinct campaign with a click in [start - 7 days, start], one per session, on
//!   the session's day; computed as the interval union (per campaign, the union of
//!   [t, t + 7 days] over its clicks; the sessions starting inside), by one sweep (`assists`).
//! * A session belongs to the day its start is on; so do its purchases. Revenue of a session is
//!   the sum of its prices; it counts as purchasing when the revenue is above 0 (the engine's
//!   rule). Top category of a user-day: the largest category sum, ties to the smaller index,
//!   every category with a purchase is a candidate (even at price 0); `NO_TOP` when none.
//! * Clicks of persons without events here are dropped.
//!
//! Output: `Cells` (the engine's FastAgg layout, cell = day * campaigns + campaign:
//! hot = [credited sessions, assisted sessions], buy = [purchasing sessions, revenue], the latter
//! touched only by sessions with revenue), accumulated across calls; `Rows`, cleared per call: per output person (as `Out::persons`) the end of its rows,
//! and per row the relative day, sessions, events, revenue and top category.

use super::*;
use crate::PersonBlocks;
use core::cmp::Ordering;

pub const DAY: u32 = 86_400_000;
pub const REACH: i64 = 7 * DAY as i64;
pub const NO_TOP: u8 = u8::MAX;
const LOC: u64 = (1 << 48) - 1;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// click base minus window start, ms (negative: the click base is 7 days earlier).
    pub base: i64,
    /// window start modulo one day, ms (rel day of a session = (start + day_shift) / DAY).
    pub day_shift: u32,
    pub days: usize,
    /// Campaign cells per day, organic included.
    pub campaigns: usize,
    pub organic: u32,
}

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Cells {
    pub hot: Vec<[u32; 2]>,
    pub buy: Vec<[u64; 2]>,
}

impl Cells {
    pub fn new(p: &Params) -> Self {
        let n = p.days * p.campaigns;
        Cells { hot: vec![[0, 0]; n], buy: vec![[0, 0]; n] }
    }
}

/// The cells as the caller's slices (e.g. the engine's FastAgg), same layout as `Cells`.
struct Cells2<'a> {
    hot: &'a mut [[u32; 2]],
    buy: &'a mut [[u64; 2]],
}

#[derive(Default, Debug, Clone)]
pub struct Rows {
    /// Per output person, the exclusive end of its rows.
    pub row_end: Vec<u32>,
    pub day: Vec<u32>,
    pub sessions: Vec<u32>,
    pub events: Vec<u32>,
    pub revenue: Vec<u64>,
    pub top: Vec<u8>,
}

impl Rows {
    fn clear(&mut self) {
        self.row_end.clear();
        self.day.clear();
        self.sessions.clear();
        self.events.clear();
        self.revenue.clear();
        self.top.clear();
    }
}

/// Counters, accumulated across calls.
#[derive(Default, Debug, Clone, Copy)]
pub struct Stats {
    pub clicks_kept: u64,
    pub clicks_dropped: u64,
    /// Credited tie groups with more than one campaign (resolved once each).
    pub tie_groups: u64,
    pub tie_calls: u64,
    /// Assist additions (one per click and day of the sessions it newly covers).
    pub assist_ranges: u64,
}

/// Reusable per-thread state of `walk`.
pub struct WalkScratch {
    cidx: Vec<u32>,
    cfirst: Vec<u32>,
    ccur: Vec<u32>,
    /// Grouped clicks: ts_click << 32 | index into the input clicks.
    ckey: Vec<u64>,
    tmp: Vec<u64>,
    sb_at: Vec<u32>,
    sbuy: Vec<u32>,
    stamp: Vec<u32>,
    upto: Vec<u32>,
    serial: u32,
    cat_sum: [u64; 256],
    cat_seen: [u64; 4],
    /// Per session of the person: start in the click base, rel day, and the index of the first
    /// later session on another day.
    s_at: Vec<i64>,
    s_day: Vec<u32>,
    day_end: Vec<u32>,
    /// Per session of the partition: its revenue, and the index of its user-day row.
    sess_rev: Vec<u64>,
    row_of: Vec<u32>,
    /// Single pass: packed click keys and the radix buffer; the two histograms over click
    /// positions (sessions by credit pointer, sessions by window-start pointer); per row
    /// [end session, day].
    ka: Vec<u64>,
    kb: Vec<u64>,
    last_at: Vec<u32>,
    row_info: Vec<[u32; 2]>,
    /// The click keys of this call carry the campaign (see `walk`).
    packed: bool,
    /// Single pass: the campaign of each sorted click.
    camp_s: Vec<u32>,
    /// Person blocks: the click words (campaign << 52 | locator), by key index.
    cw: Vec<u64>,
    pub stats: Stats,
}

impl Default for WalkScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl WalkScratch {
    pub fn new() -> Self {
        WalkScratch {
            cidx: Vec::new(),
            cfirst: Vec::new(),
            ccur: Vec::new(),
            ckey: Vec::new(),
            tmp: Vec::new(),
            sb_at: Vec::new(),
            sbuy: Vec::new(),
            stamp: Vec::new(),
            upto: Vec::new(),
            serial: 0,
            cat_sum: [0; 256],
            cat_seen: [0; 4],
            s_at: Vec::new(),
            s_day: Vec::new(),
            day_end: Vec::new(),
            sess_rev: Vec::new(),
            row_of: Vec::new(),
            ka: Vec::new(),
            kb: Vec::new(),
            last_at: Vec::new(),
            row_info: Vec::new(),
            packed: false,
            camp_s: Vec::new(),
            cw: Vec::new(),
            stats: Stats::default(),
        }
    }
}

/// Sorts keys `ts << 32 | input index` into (ts, input index) order: insertion up to 32 (stable
/// on the grouped input order), the standard unstable sort of the whole key up to 4096, else
/// LSD radix over the high 32 bits in 11-bit digits (stable).
fn sort_hi(keys: &mut [u64], tmp: &mut Vec<u64>) {
    let n = keys.len();
    if n > 32 && n <= 4096 {
        keys.sort_unstable();
        return;
    }
    if n <= 32 {
        for i in 1..n {
            let x = keys[i];
            let mut j = i;
            while j > 0 && keys[j - 1] >> 32 > x >> 32 {
                keys[j] = keys[j - 1];
                j -= 1;
            }
            keys[j] = x;
        }
        return;
    }
    tmp.clear();
    tmp.resize(n, 0);
    let mut in_keys = true;
    for shift in [32u32, 43, 54] {
        let mut count = [0u32; 2048];
        {
            let src: &[u64] = if in_keys { keys } else { tmp };
            for &x in src {
                count[((x >> shift) & 2047) as usize] += 1;
            }
            if count[((src[0] >> shift) & 2047) as usize] as usize == n {
                continue;
            }
        }
        let mut sum = 0;
        for c in count.iter_mut() {
            let x = *c;
            *c = sum;
            sum += x;
        }
        let (src, dst): (&[u64], &mut [u64]) = if in_keys { (&*keys, &mut tmp[..]) } else { (&tmp[..], &mut *keys) };
        for &x in src {
            let d = ((x >> shift) & 2047) as usize;
            dst[count[d] as usize] = x;
            count[d] += 1;
        }
        in_keys = !in_keys;
    }
    if !in_keys {
        keys.copy_from_slice(tmp);
    }
}

/// Clicks into the sessions `sessions` just produced in `out` (same `sc`, `buys`, `dense`).
#[allow(clippy::too_many_arguments)]
pub fn walk<F: FnMut(u64, u64) -> Ordering>(
    clicks: &[[u64; 2]],
    buys: &[[u64; 2]],
    dense: Option<usize>,
    prm: &Params,
    tie: &mut F,
    sc: &Scratch,
    ws: &mut WalkScratch,
    out: &Out,
    hot: &mut [[u32; 2]],
    buy: &mut [[u64; 2]],
    rows: &mut Rows,
) -> Result<(), Error> {
    rows.clear();
    let mut cells = Cells2 { hot, buy };
    let cells = &mut cells;
    let np = out.persons.len();
    let cn = prm.campaigns;
    if out.sess_end.len() != np || out.buy_session.len() != buys.len() || cells.hot.len() != prm.days * cn || cells.buy.len() != prm.days * cn || prm.organic as usize >= cn {
        return Err(Error::WalkInput);
    }
    if let Some(d) = dense {
        if d != np {
            return Err(Error::WalkInput);
        }
    }
    if prm.base <= 0 && prm.base > -(1i64 << 31) {
        purchases_by_session(buys.len(), |j| buys[j][1] as u32, out, ws)?;
        if single_pass(clicks, dense, prm, tie, sc, ws, out, cells, rows)? {
            top_categories(|j| buys[j][1] as u32, ws, out, rows);
            return Ok(());
        }
        rows.clear();
    }
    // 1. clicks grouped by local person (dropping those of persons without events here)
    let n = clicks.len();
    ws.cidx.clear();
    ws.cidx.resize(n, EMPTY);
    ws.cfirst.clear();
    ws.cfirst.resize(np, 0);
    let mut kept = 0u64;
    for (i, c) in clicks.iter().enumerate() {
        let tok = (c[0] >> 32) as u32;
        if tok == EMPTY {
            continue;
        }
        let p = match dense {
            Some(_) if (tok as usize) < np => tok,
            Some(_) => return Err(Error::BadPerson(i)),
            None => match sc.lookup(tok) {
                Some(p) => p,
                None => continue,
            },
        };
        let camp = (c[1] >> 48) as u32;
        if camp as usize >= cn || camp == prm.organic {
            return Err(Error::BadCampaign(i));
        }
        ws.cidx[i] = p;
        ws.cfirst[p as usize] += 1;
        kept += 1;
    }
    ws.stats.clicks_kept += kept;
    ws.stats.clicks_dropped += n as u64 - kept;
    let mut sum = 0u32;
    ws.ccur.clear();
    ws.ccur.resize(np, 0);
    for p in 0..np {
        let c = ws.cfirst[p];
        ws.cfirst[p] = sum;
        ws.ccur[p] = sum;
        sum += c;
    }
    let packed = n <= 1 << 16;
    ws.packed = packed;
    ws.ckey.clear();
    ws.ckey.resize(kept as usize, 0);
    for (i, c) in clicks.iter().enumerate() {
        let p = ws.cidx[i];
        if p != EMPTY {
            let at = &mut ws.ccur[p as usize];
            // ts_click << 32 | campaign << 16 | input index (partitions of at most 2^16 clicks):
            // the credit and the sweep read the campaign from the key, not the click record.
            ws.ckey[*at as usize] = if packed { (c[0] << 32) | ((c[1] >> 48) << 16) | i as u64 } else { (c[0] << 32) | i as u64 };
            *at += 1;
        }
    }
    purchases_by_session(buys.len(), |j| buys[j][1] as u32, out, ws)?;
    if ws.stamp.len() < cn {
        ws.stamp.resize(cn, 0);
        ws.upto.resize(cn, 0);
    }
    // 3. per person
    let base = prm.base;
    let mut ckey = core::mem::take(&mut ws.ckey);
    let mut res = Ok(());
    let mut s0 = 0usize;
    for p in 0..np {
        let s1 = out.sess_end[p] as usize;
        let ck = &mut ckey[ws.cfirst[p] as usize..ws.ccur[p] as usize];
        if s1 > s0 {
            sort_hi(ck, &mut ws.tmp);
            if let Err(e) = person(ck, s0, s1, clicks, buys, prm, base, tie, ws, out, cells, rows) {
                res = Err(e);
                break;
            }
        }
        rows.row_end.push(rows.day.len() as u32);
        s0 = s1;
    }
    ws.ckey = ckey;
    if res.is_ok() {
        top_categories(|j| buys[j][1] as u32, ws, out, rows);
    }
    res
}

/// Purchases grouped by session (`sbuy`, by `sb_at`), each session's revenue (`sess_rev`), and
/// `row_of` sized for the sessions.
fn purchases_by_session<P: Fn(usize) -> u32>(npay: usize, pay: P, out: &Out, ws: &mut WalkScratch) -> Result<(), Error> {
    let ns = out.start.len();
    ws.sb_at.clear();
    ws.sb_at.resize(ns + 2, 0);
    for &s in &out.buy_session {
        if s as usize >= ns {
            return Err(Error::WalkInput);
        }
        ws.sb_at[s as usize + 2] += 1;
    }
    for s in 2..ns + 2 {
        ws.sb_at[s] += ws.sb_at[s - 1];
    }
    ws.sbuy.clear();
    ws.sbuy.resize(npay, 0);
    ws.sess_rev.clear();
    ws.sess_rev.resize(ns, 0);
    for (j, &s) in out.buy_session.iter().enumerate() {
        let at = &mut ws.sb_at[s as usize + 1];
        ws.sbuy[*at as usize] = j as u32;
        *at += 1;
        ws.sess_rev[s as usize] += u64::from(pay(j) >> 8);
    }
    ws.row_of.clear();
    ws.row_of.resize(ns, 0);
    Ok(())
}

/// Top category of every user-day row with purchases, from the purchases in session order
/// (sessions, hence rows, ascending): the largest category sum, ties to the smaller index, any
/// category with a purchase a candidate (even at price 0). Other rows keep `NO_TOP`.
fn top_categories<P: Fn(usize) -> u32>(pay: P, ws: &mut WalkScratch, out: &Out, rows: &mut Rows) {
    let mut cur = u32::MAX;
    for i in 0..ws.sbuy.len() {
        let j = ws.sbuy[i] as usize;
        let row = ws.row_of[out.buy_session[j] as usize];
        if row != cur {
            if cur != u32::MAX {
                rows.top[cur as usize] = take_top(ws);
            }
            cur = row;
        }
        let pay = pay(j);
        let cat = (pay & 255) as usize;
        ws.cat_sum[cat] += u64::from(pay >> 8);
        ws.cat_seen[cat >> 6] |= 1 << (cat & 63);
    }
    if cur != u32::MAX {
        rows.top[cur as usize] = take_top(ws);
    }
}

/// The top category of the sums gathered, which are then cleared.
#[inline(always)]
fn take_top(ws: &mut WalkScratch) -> u8 {
    let mut top = NO_TOP;
    let mut top_sum = 0u64;
    let mut any = false;
    for w in 0..4 {
        let mut bits = ws.cat_seen[w];
        while bits != 0 {
            let cat = w * 64 + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let s = ws.cat_sum[cat];
            // ascending categories: the first is taken, later ones only when strictly larger
            if !any || s > top_sum {
                any = true;
                top = cat as u8;
                top_sum = s;
            }
            ws.cat_sum[cat] = 0;
        }
        ws.cat_seen[w] = 0;
    }
    top
}

#[inline(always)]
fn ts_c(k: u64) -> i64 {
    (k >> 32) as i64
}

#[inline(always)]
fn camp_of(clicks: &[[u64; 2]], k: u64, packed: bool) -> u32 {
    if packed { ((k >> 16) & 0xffff) as u32 } else { (clicks[(k & 0xffff_ffff) as usize][1] >> 48) as u32 }
}

#[inline(always)]
fn idx_of(k: u64, packed: bool) -> usize {
    (if packed { k & 0xffff } else { k & 0xffff_ffff }) as usize
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn person<F: FnMut(u64, u64) -> Ordering>(
    ck: &[u64],
    s0: usize,
    s1: usize,
    clicks: &[[u64; 2]],
    _buys: &[[u64; 2]],
    prm: &Params,
    base: i64,
    tie: &mut F,
    ws: &mut WalkScratch,
    out: &Out,
    cells: &mut Cells2,
    rows: &mut Rows,
) -> Result<(), Error> {
    let packed = ws.packed;
    let n = ck.len();
    let cn = prm.campaigns;
    ws.s_at.clear();
    ws.s_day.clear();
    let mut k = 0usize;
    let (mut memo_g, mut memo_c) = (usize::MAX, 0u32);
    // Rows are written in place, one slot per session at most: every session rewrites the
    // current row with its running sums, and a new day moves to the next slot first (no branch
    // on the day change; the top category comes later, from the purchases).
    let r0 = rows.day.len();
    let m = s1 - s0;
    rows.day.reserve(m);
    rows.sessions.reserve(m);
    rows.events.reserve(m);
    rows.revenue.reserve(m);
    rows.top.reserve(m);
    let (pd, ps, pe, pr, pt) = (
        rows.day.as_mut_ptr(),
        rows.sessions.as_mut_ptr(),
        rows.events.as_mut_ptr(),
        rows.revenue.as_mut_ptr(),
        rows.top.as_mut_ptr(),
    );
    let mut r = r0.wrapping_sub(1);
    let mut day_now = u32::MAX;
    let (mut d_sessions, mut d_events, mut d_revenue) = (0u32, 0u32, 0u64);
    for s in s0..s1 {
        let start = out.start[s];
        // start < 2^31 and day_shift < DAY, so the sum fits in u32
        let day = (start + prm.day_shift) / DAY;
        if day as usize >= prm.days {
            return Err(Error::DayOutOfRange(s));
        }
        let new_day = day != day_now;
        day_now = day;
        r = r.wrapping_add(usize::from(new_day));
        let revenue = ws.sess_rev[s];
        let ev = out.events[s];
        d_sessions = if new_day { 1 } else { d_sessions + 1 };
        d_events = if new_day { ev } else { d_events + ev };
        d_revenue = if new_day { revenue } else { d_revenue + revenue };
        // SAFETY: r - r0 < m (one slot per session at most), within the reserved room.
        unsafe {
            *pd.add(r) = day;
            *ps.add(r) = d_sessions;
            *pe.add(r) = d_events;
            *pr.add(r) = d_revenue;
            *pt.add(r) = NO_TOP;
        }
        ws.row_of[s] = r as u32;
        let mut camp = prm.organic;
        if n > 0 {
            let at = i64::from(start) - base;
            ws.s_at.push(at);
            ws.s_day.push(day);
            while k < n && ts_c(ck[k]) <= at {
                k += 1;
            }
            if k > 0 {
                let t = ts_c(ck[k - 1]);
                if at - t <= REACH {
                    camp = camp_of(clicks, ck[k - 1], packed);
                    if k >= 2 && ts_c(ck[k - 2]) == t {
                        let mut g = k - 1;
                        while g > 0 && ts_c(ck[g - 1]) == t {
                            g -= 1;
                        }
                        if g == memo_g {
                            camp = memo_c;
                        } else {
                            let mixed = ck[g..k].iter().any(|&x| camp_of(clicks, x, packed) != camp);
                            if mixed {
                                ws.stats.tie_groups += 1;
                                // smallest click_id; among equal ids the later click (as a stable
                                // sort by click_id descending would leave last)
                                let mut best = g;
                                for i in g + 1..k {
                                    ws.stats.tie_calls += 1;
                                    let (lb, li) = (clicks[idx_of(ck[best], packed)][1] & LOC, clicks[idx_of(ck[i], packed)][1] & LOC);
                                    if tie(lb, li) != Ordering::Less {
                                        best = i;
                                    }
                                }
                                camp = camp_of(clicks, ck[best], packed);
                            }
                            memo_g = g;
                            memo_c = camp;
                        }
                    }
                }
            }
        }
        let cell = day as usize * cn + camp as usize;
        cells.hot[cell][0] += 1;
        if revenue > 0 {
            let b = &mut cells.buy[cell];
            b[0] += 1;
            b[1] += revenue;
        }
    }
    // SAFETY: slots r0..=r were all written above.
    unsafe {
        let len = r.wrapping_add(1);
        rows.day.set_len(len);
        rows.sessions.set_len(len);
        rows.events.set_len(len);
        rows.revenue.set_len(len);
        rows.top.set_len(len);
    }
    if n > 0 {
        assists(ck, clicks, prm, ws, cells);
    }
    Ok(())
}

/// Assists of one person as the interval union, by one sweep over its clicks in ts order: a
/// session s gets campaign c iff its start lies in the union over c's clicks t of [t, t + REACH].
/// For click t the sessions with start in [t, t + REACH] are the index range [lo, hi); both ends
/// only grow with t, so per campaign the union is counted by keeping the end `hi` already
/// counted and adding [max(lo, counted end), hi) (intervals sorted by start with nondecreasing
/// ends). Ranges are split into days by `day_end`.
#[inline(always)]
fn assists(ck: &[u64], clicks: &[[u64; 2]], prm: &Params, ws: &mut WalkScratch, cells: &mut Cells2) {
    let packed = ws.packed;
    let m = ws.s_at.len();
    ws.day_end.clear();
    ws.day_end.resize(m, m as u32);
    for i in (0..m.saturating_sub(1)).rev() {
        ws.day_end[i] = if ws.s_day[i + 1] != ws.s_day[i] { i as u32 + 1 } else { ws.day_end[i + 1] };
    }
    ws.serial = ws.serial.wrapping_add(1);
    if ws.serial == 0 {
        ws.stamp.fill(0);
        ws.serial = 1;
    }
    let serial = ws.serial;
    let cn = prm.campaigns;
    // locals so that nothing is reloaded through `ws` inside the loop
    let (s_at, s_day, day_end) = (&ws.s_at[..m], &ws.s_day[..m], &ws.day_end[..m]);
    let (stamp, upto) = (&mut ws.stamp[..cn], &mut ws.upto[..cn]);
    let hot = &mut *cells.hot;
    let mut ranges = 0u64;
    let (mut lo, mut hi) = (0usize, 0usize);
    for &x in ck {
        let t = ts_c(x);
        while lo < m && s_at[lo] < t {
            lo += 1;
        }
        let te = t + REACH;
        while hi < m && s_at[hi] <= te {
            hi += 1;
        }
        if lo == hi {
            continue;
        }
        let c = camp_of(clicks, x, packed) as usize;
        // stamp[c]: serial of the person that last counted c; upto[c]: the end it counted to
        let from = if stamp[c] == serial { lo.max(upto[c] as usize) } else { lo };
        stamp[c] = serial;
        upto[c] = hi as u32;
        let mut i = from;
        while i < hi {
            let end = (day_end[i] as usize).min(hi);
            ranges += 1;
            hot[s_day[i] as usize * cn + c][1] += (end - i) as u32;
            i = end;
        }
    }
    ws.stats.assist_ranges += ranges;
}

/// Packed click key of the single pass: local person << (32 + kt) | ts_click << kt | index into
/// the clicks, with kt = the index bits of the partition (`key_bits`); person bits + kt <= 32.
/// Comparable click time: key >> kt = person << 32 | ts_click (the single pass takes only clicks
/// whose ts_click + REACH stays below 2^32, so adding REACH never reaches the person bits).
#[inline(always)]
fn ckey(x: u64, kt: u32) -> u64 {
    x >> kt
}

/// Index bits for `n` clicks, if `np` persons fit beside them in the key.
fn key_bits(n: usize, np: usize) -> Option<u32> {
    let kt = (usize::BITS - n.leading_zeros()).max(1);
    let pb = usize::BITS - np.leading_zeros();
    (pb + kt <= 32).then_some(kt)
}

/// Largest ts_click the single pass takes.
const TS_MAX: u64 = (1 << 32) - REACH as u64 - 2;

/// Stable LSD radix sort of packed keys by (person, ts) (bits kt..64): insertion up to 24 keys,
/// else 8-bit digits (256 buckets: the counts and the 256 write streams stay in L1), all
/// histograms from one read, passes whose digit is the same for every key skipped.
fn radix_keys(a: &mut Vec<u64>, b: &mut Vec<u64>, kt: u32) {
    let n = a.len();
    if n <= 24 {
        for i in 1..n {
            let x = a[i];
            let mut j = i;
            while j > 0 && a[j - 1] >> kt > x >> kt {
                a[j] = a[j - 1];
                j -= 1;
            }
            a[j] = x;
        }
        return;
    }
    b.clear();
    b.resize(n, 0);
    let nd = (64 - kt as usize).div_ceil(8);
    let mut count = [[0u32; 256]; 8];
    for &x in a.iter() {
        let k = x >> kt;
        for (d, c) in count[..nd].iter_mut().enumerate() {
            c[((k >> (8 * d)) & 255) as usize] += 1;
        }
    }
    for (d, c) in count[..nd].iter_mut().enumerate() {
        let shift = kt + 8 * d as u32;
        if c[((a[0] >> shift) & 255) as usize] as usize == n {
            continue;
        }
        let mut sum = 0u32;
        for v in c.iter_mut() {
            let x = *v;
            *v = sum;
            sum += x;
        }
        for &x in a.iter() {
            let dg = ((x >> shift) & 255) as usize;
            // SAFETY: dg < 256; c[dg] < n = b.len() (prefix sums of the n keys)
            unsafe {
                let at = c.get_unchecked_mut(dg);
                *b.get_unchecked_mut(*at as usize) = x;
                *at += 1;
            }
        }
        core::mem::swap(a, b);
    }
}

/// The whole partition after one sort of the clicks by (person, ts), in two straight passes.
/// Sessions, once: key, row (running sums), and two click pointers that only grow: cp = clicks
/// at or before the start (the last of them is the credited candidate), ap = clicks more than
/// REACH before it; each lands in a histogram over click positions. Clicks, once: running sums
/// of those histograms are the click's session range, lo = sessions starting after it,
/// hi = sessions starting at most REACH after it; assists are the per (person, campaign) union
/// of the ranges (a person's ranges only grow), split by rows.
#[allow(clippy::too_many_arguments)]
fn single_pass<F: FnMut(u64, u64) -> Ordering>(
    clicks: &[[u64; 2]],
    dense: Option<usize>,
    prm: &Params,
    tie: &mut F,
    sc: &Scratch,
    ws: &mut WalkScratch,
    out: &Out,
    cells: &mut Cells2,
    rows: &mut Rows,
) -> Result<bool, Error> {
    let np = out.persons.len();
    let cn = prm.campaigns;
    let mut tmax = 0u64;
    let Some(kt) = key_bits(clicks.len(), np) else {
        return Ok(false);
    };
    // clicks: packed keys
    let mut ka = core::mem::take(&mut ws.ka);
    ka.clear();
    for (i, c) in clicks.iter().enumerate() {
        let tok = (c[0] >> 32) as u32;
        let p = match dense {
            _ if tok == EMPTY => continue,
            Some(_) if (tok as usize) < np => tok,
            Some(_) => {
                ws.ka = ka;
                return Err(Error::BadPerson(i));
            }
            None => match sc.lookup(tok) {
                Some(p) => p,
                None => continue,
            },
        };
        let camp = (c[1] >> 48) as u32;
        if camp as usize >= cn || camp == prm.organic {
            ws.ka = ka;
            return Err(Error::BadCampaign(i));
        }
        let t = c[0] & 0xffff_ffff;
        tmax = tmax.max(t);
        ka.push((u64::from(p) << (32 + kt)) | (t << kt) | i as u64);
    }
    if tmax > TS_MAX {
        ws.ka = ka;
        return Ok(false);
    }
    ws.stats.clicks_dropped += (clicks.len() - ka.len()) as u64;
    ws.ka = ka;
    // SAFETY (closure): i is the index of a kept input click
    single_core(|i| unsafe { clicks.get_unchecked(i)[1] }, 48, LOC, kt, false, prm, tie, ws, out, cells, rows)?;
    Ok(true)
}

/// The single pass after the click keys are built (`ws.ka`, unsorted): `wd(i)` is the word of
/// click i holding its campaign at bit `cs` and its locator under `lm`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn single_core<F: FnMut(u64, u64) -> Ordering, W: Fn(usize) -> u64>(
    wd: W,
    cs: u32,
    lm: u64,
    kt: u32,
    presorted: bool,
    prm: &Params,
    tie: &mut F,
    ws: &mut WalkScratch,
    out: &Out,
    cells: &mut Cells2,
    rows: &mut Rows,
) -> Result<(), Error> {
    let np = out.persons.len();
    let ns = out.start.len();
    let cn = prm.campaigns;
    let reach = REACH as u64;
    let mut ka = core::mem::take(&mut ws.ka);
    let nc = ka.len();
    ws.stats.clicks_kept += nc as u64;
    if !presorted {
        radix_keys(&mut ka, &mut ws.kb, kt);
    }
    let (kp, im) = (32 + kt, (1u64 << kt) - 1);
    ka.push(u64::MAX);
    // sessions: rows (each session rewrites its row with the running sums; a new day moves to
    // the next slot first) and the day check
    ws.row_info.clear();
    ws.row_info.resize(ns, [0, 0]);
    for v in [&mut rows.day, &mut rows.sessions, &mut rows.events] {
        v.clear();
        v.resize(ns, 0);
    }
    rows.revenue.clear();
    rows.revenue.resize(ns, 0);
    rows.top.clear();
    rows.top.resize(ns, NO_TOP);
    let mut r = usize::MAX;
    {
        let (starts, evs, revs) = (&out.start[..ns], &out.events[..ns], &ws.sess_rev[..ns]);
        let (rd, rs, re, rr) = (&mut rows.day[..ns], &mut rows.sessions[..ns], &mut rows.events[..ns], &mut rows.revenue[..ns]);
        let (row_info, row_of) = (&mut ws.row_info[..ns], &mut ws.row_of[..ns]);
        let mut s0 = 0usize;
        for p in 0..np {
            let s1 = out.sess_end[p] as usize;
            let mut prev = u32::MAX;
            let (mut ds, mut de, mut dr) = (0u32, 0u32, 0u64);
            for s in s0..s1.min(ns) {
                // SAFETY: s < ns, the length of every per-session slice here; r <= s (at most one
                // row per session, and r starts at -1)
                unsafe {
                    let day = (*starts.get_unchecked(s) + prm.day_shift) / DAY;
                    if day as usize >= prm.days {
                        ws.ka = ka;
                        return Err(Error::DayOutOfRange(s));
                    }
                    let new = day != prev;
                    prev = day;
                    r = r.wrapping_add(new as usize);
                    let (ev, rev) = (*evs.get_unchecked(s), *revs.get_unchecked(s));
                    ds = if new { 1 } else { ds + 1 };
                    de = if new { ev } else { de + ev };
                    dr = if new { rev } else { dr + rev };
                    *rd.get_unchecked_mut(r) = day;
                    *rs.get_unchecked_mut(r) = ds;
                    *re.get_unchecked_mut(r) = de;
                    *rr.get_unchecked_mut(r) = dr;
                    *row_info.get_unchecked_mut(r) = [s as u32 + 1, day];
                    *row_of.get_unchecked_mut(s) = r as u32;
                }
            }
            rows.row_end.push(r.wrapping_add(1) as u32);
            s0 = s1;
        }
    }
    let nr = r.wrapping_add(1);
    rows.day.truncate(nr);
    rows.sessions.truncate(nr);
    rows.events.truncate(nr);
    rows.revenue.truncate(nr);
    rows.top.truncate(nr);
    // sessions: credit, and the two click pointers into their histograms
    ws.last_at.clear();
    ws.last_at.resize(2 * (nc + 1), 0); // [credit-pointer histogram | window-start histogram]
    // the campaigns in sorted order, read from the click records once (they are still near
    // from the key pass); the loops below never touch the records again
    ws.camp_s.clear();
    ws.camp_s.extend(ka[..nc].iter().map(|&x| (wd((x & im) as usize) >> cs) as u32));
    let camp_s = &ws.camp_s[..];
    {
        let (starts, revs) = (&out.start[..ns], &ws.sess_rev[..ns]);
        let (hl, hh) = ws.last_at.split_at_mut(nc + 1);
        let (hot, buy) = (&mut *cells.hot, &mut *cells.buy);
        let kq = ka.as_ptr();
        let (mut memo_g, mut memo_c) = (usize::MAX, 0usize);
        let (mut cp, mut ap) = (0usize, 0usize);
        let mut s0 = 0usize;
        for p in 0..np {
            let s1 = (out.sess_end[p] as usize).min(ns);
            for s in s0..s1 {
                // SAFETY: s < ns; cp, ap <= nc, as ka[nc] is the sentinel above every key and key
                // + REACH that stops both; cp - 1 and cp - 2 are checked >= 0; day < days and
                // camp < campaigns (checked when the keys were made), so the cell is in range
                unsafe {
                    let start = *starts.get_unchecked(s);
                    let day = (start + prm.day_shift) / DAY;
                    // start - base < 2^32 (base > -2^31)
                    let sk = ((p as u64) << 32) | (i64::from(start) - prm.base) as u64;
                    while ckey(*kq.add(cp), kt) <= sk {
                        cp += 1;
                    }
                    while ckey(*kq.add(ap), kt) + reach < sk {
                        ap += 1;
                    }
                    *hl.get_unchecked_mut(cp) += 1;
                    *hh.get_unchecked_mut(ap) += 1;
                    // credit: the last click at or before the start, same person, at most REACH back
                    let mut camp = prm.organic as usize;
                    if cp > 0 {
                        let x = *kq.add(cp - 1);
                        let xk = ckey(x, kt);
                        if xk >> 32 == p as u64 && sk - xk <= reach {
                            camp = *camp_s.get_unchecked(cp - 1) as usize;
                            if cp >= 2 && *kq.add(cp - 2) >> kt == x >> kt {
                                // a ts tie: the smallest click_id of the group (equal ids: the later)
                                let mut g = cp - 1;
                                while g > 0 && *kq.add(g - 1) >> kt == x >> kt {
                                    g -= 1;
                                }
                                if g == memo_g {
                                    camp = memo_c;
                                } else {
                                    if camp_s[g..cp].iter().any(|&y| y as usize != camp) {
                                        ws.stats.tie_groups += 1;
                                        let mut best = g;
                                        for i in g + 1..cp {
                                            ws.stats.tie_calls += 1;
                                            if tie(wd((ka[best] & im) as usize) & lm, wd((ka[i] & im) as usize) & lm) != Ordering::Less {
                                                best = i;
                                            }
                                        }
                                        camp = camp_s[best] as usize;
                                    }
                                    memo_g = g;
                                    memo_c = camp;
                                }
                            }
                        }
                    }
                    let cell = day as usize * cn + camp;
                    hot.get_unchecked_mut(cell)[0] += 1;
                    let rev = *revs.get_unchecked(s);
                    if rev > 0 {
                        let b = buy.get_unchecked_mut(cell);
                        b[0] += 1;
                        b[1] += rev;
                    }
                }
            }
            s0 = s1;
        }
    }
    // clicks, once: assists
    if ws.stamp.len() < cn {
        ws.stamp.resize(cn, 0);
        ws.upto.resize(cn, 0);
    }
    if ws.serial as u64 + np as u64 + 1 >= u64::from(u32::MAX) {
        ws.stamp.fill(0);
        ws.serial = 0;
    }
    let sbase = ws.serial + 1;
    ws.serial += np as u32 + 1;
    {
        let (hl, hh) = ws.last_at.split_at(nc + 1);
        let (row_of, row_info) = (&ws.row_of[..ns], &ws.row_info[..]);
        let (stamp, upto) = (&mut ws.stamp[..cn], &mut ws.upto[..cn]);
        let hot = &mut *cells.hot;
        let (mut lo, mut hi) = (0usize, 0usize);
        for j in 0..nc {
            // SAFETY: j < nc < len of ka, hl and hh; lo <= hi <= ns (running sums of session
            // counts); camp < campaigns (checked when the keys were made); from < hi <= ns
            unsafe {
                lo += *hl.get_unchecked(j) as usize;
                hi += *hh.get_unchecked(j) as usize;
                let x = *ka.get_unchecked(j);
                let camp = *camp_s.get_unchecked(j) as usize;
                let st = sbase + (x >> kp) as u32;
                let sc_ = stamp.get_unchecked_mut(camp);
                let up = upto.get_unchecked_mut(camp);
                let from = if *sc_ == st { lo.max(*up as usize) } else { lo };
                *sc_ = st;
                // an empty range adds nothing; upto = hi is still right (later clicks have lo >= hi)
                *up = hi as u32;
                if from < hi {
                    let mut a = from;
                    let (ra, rb) = (*row_of.get_unchecked(from) as usize, *row_of.get_unchecked(hi - 1) as usize);
                    for ri in row_info.get_unchecked(ra..=rb) {
                        let e = (ri[0] as usize).min(hi);
                        hot.get_unchecked_mut(ri[1] as usize * cn + camp)[1] += (e - a) as u32;
                        a = e;
                    }
                }
            }
        }
    }
    ka.pop();
    ws.ka = ka;
    Ok(())
}

/// `walk` for v4 person blocks (`PersonBlocks`, after `sessions_blocks` on the same `out`):
/// clicks straight from the click blocks (person k = the block's person index, no lookup),
/// purchases from `out.buy_pay`. Err: the caller falls back (v3).
#[allow(clippy::too_many_arguments)]
pub fn walk_blocks<F: FnMut(u64, u64) -> Ordering>(
    pb: &PersonBlocks,
    prm: &Params,
    tie: &mut F,
    ws: &mut WalkScratch,
    out: &Out,
    hot: &mut [[u32; 2]],
    buy: &mut [[u64; 2]],
    rows: &mut Rows,
) -> Result<(), Error> {
    rows.clear();
    let mut cells = Cells2 { hot, buy };
    let cells = &mut cells;
    let np = out.persons.len();
    let cn = prm.campaigns;
    if pb.ev_at.len() != np + 1 || pb.ck_at.len() != np + 1 || out.sess_end.len() != np || out.buy_pay.len() != out.buy_session.len() || cells.hot.len() != prm.days * cn || cells.buy.len() != prm.days * cn || prm.organic as usize >= cn {
        return Err(Error::WalkInput);
    }
    if prm.base > 0 || prm.base <= -(1i64 << 31) {
        return Err(Error::WalkInput);
    }
    let mut n_in = 0usize;
    for blk in pb.ck {
        n_in += (blk.first().copied().unwrap_or(0) & 0xFF_FFFF) as usize;
    }
    let Some(kt) = key_bits(n_in, np) else {
        return Err(Error::WalkInput);
    };
    // keys and the click words, from the blocks
    let mut ka = core::mem::take(&mut ws.ka);
    ka.clear();
    ws.cw.clear();
    let mut dropped = 0u64;
    for p in 0..np {
        let k0 = ka.len();
        for blk in &pb.ck[pb.ck_at[p] as usize..pb.ck_at[p + 1] as usize] {
            let n = (blk[0] & 0xFF_FFFF) as usize;
            if blk.len() < 2 + 2 * n {
                ws.ka = ka;
                return Err(Error::WalkInput);
            }
            for pair in blk[2..2 + 2 * n].chunks_exact(2) {
                let (w0, tc) = (pair[0], pair[1]);
                let camp = (w0 >> 52) as u32;
                if camp as usize >= cn || camp == prm.organic {
                    ws.ka = ka;
                    return Err(Error::BadCampaign(ws.cw.len()));
                }
                if tc > TS_MAX {
                    // after every session start (starts are below 2^31 - base): credits and
                    // assists nothing
                    dropped += 1;
                    continue;
                }
                ka.push(((p as u64) << (32 + kt)) | (tc << kt) | ws.cw.len() as u64);
                ws.cw.push(w0);
            }
        }
        // the person's clicks by (ts, index): the keys are already grouped by person
        let g = &mut ka[k0..];
        if g.len() <= 24 {
            for i in 1..g.len() {
                let x = g[i];
                let mut j = i;
                while j > 0 && g[j - 1] > x {
                    g[j] = g[j - 1];
                    j -= 1;
                }
                g[j] = x;
            }
        } else {
            g.sort_unstable();
        }
    }
    ws.stats.clicks_dropped += dropped;
    ws.ka = ka;
    let pay = &out.buy_pay;
    purchases_by_session(pay.len(), |j| pay[j], out, ws)?;
    let cw = core::mem::take(&mut ws.cw);
    // SAFETY (closure): i < cw.len() (the index stored in a key)
    let r = single_core(|i| unsafe { *cw.get_unchecked(i) }, 52, (1u64 << 52) - 1, kt, true, prm, tie, ws, out, cells, rows);
    ws.cw = cw;
    r?;
    top_categories(|j| pay[j], ws, out, rows);
    Ok(())
}
