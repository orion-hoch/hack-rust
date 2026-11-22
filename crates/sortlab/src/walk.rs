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

/// The per-person click buffers of `walk_blocks`.
#[derive(Default)]
struct KeyBufs {
    /// Keys `tc << 32 | index << 12 | campaign` of the person at hand, and the sentinel.
    kb: Vec<u64>,
    /// The click words (campaign << 52 | locator), by key index: read only by a credited tie
    /// group with several campaigns (a re-read of the blocks instead cost 1.5G instructions at
    /// 20M events: the ties are in the persons with tens of thousands of clicks).
    cw: Vec<u64>,
    /// Per key position: the session that moved the credit pointer past it.
    lo: Vec<u32>,
    /// Distribution sort: its output and its counts.
    tmp: Vec<u64>,
    cnt: Vec<u32>,
    dropped: u64,
}

/// Room for `need` keys and the sentinel.
#[cold]
#[inline(never)]
fn grow_person(kv: &mut KeyBufs, need: usize) {
    kv.kb.clear();
    kv.kb.reserve(need + 1);
    kv.cw.clear();
    kv.cw.reserve(need + 1);
    kv.lo.clear();
    kv.lo.reserve(need + 1);
}

/// The click keys of one person, sorted, with the sentinel after them: (keys, count). A count of
/// usize::MAX: a block is malformed, a campaign is not below `lim`, or there are over 2^20 clicks.
/// On its own (not inlined) so that the loops keep their few values in registers.
#[inline(never)]
fn person_keys(blocks: &[&[u64]], kv: &mut KeyBufs, lim: u64) -> (*mut u64, usize) {
    const BAD: (*mut u64, usize) = (core::ptr::null_mut(), usize::MAX);
    let mut need = 0usize;
    for blk in blocks {
        let c = (blk.first().copied().unwrap_or(0) & 0xFF_FFFF) as usize;
        if blk.len() < 2 + 2 * c {
            return BAD;
        }
        need += c;
    }
    if need > 1 << 20 {
        return BAD;
    }
    if kv.kb.capacity() <= need || kv.cw.capacity() <= need || kv.lo.capacity() <= need {
        grow_person(kv, need);
    }
    let (kw, cww) = (kv.kb.as_mut_ptr(), kv.cw.as_mut_ptr());
    let mut n = 0usize;
    let mut dropped = 0u64;
    for blk in blocks {
        let c = (blk[0] & 0xFF_FFFF) as usize;
        // SAFETY: 2 + 2 * c <= blk.len() (checked above), so the words of the c clicks are in
        // the block; n < need < the room of kb and cw.
        unsafe {
            let mut q = blk.as_ptr().add(2);
            for _ in 0..c {
                let (w0, tc) = (*q, *q.add(1));
                q = q.add(2);
                let camp = w0 >> 52;
                if camp >= lim {
                    return BAD;
                }
                if tc > TS_MAX {
                    // after every session start (starts are below 2^31 - base): credits and
                    // assists nothing
                    dropped += 1;
                    continue;
                }
                *kw.add(n) = (tc << 32) | ((n as u64) << 12) | camp;
                *cww.add(n) = w0;
                n += 1;
            }
        }
    }
    kv.dropped += dropped;
    let mut ks = kw;
    // SAFETY: entries 0..n were written above, and entry n is within the room (of kb, and of tmp
    // when the distribution sort moved the keys there).
    unsafe {
        if n <= 24 {
            for i in 1..n {
                let x = *kw.add(i);
                let mut j = i;
                while j > 0 && *kw.add(j - 1) > x {
                    *kw.add(j) = *kw.add(j - 1);
                    j -= 1;
                }
                *kw.add(j) = x;
            }
        } else if sort_keys_big(kw, n, &mut kv.tmp, &mut kv.cnt) {
            ks = kv.tmp.as_mut_ptr();
        }
        // above every bound the pointers are compared with
        *ks.add(n) = u64::MAX;
    }
    (ks, n)
}

/// What the per-person loops of `walk_blocks` share: raw pointers into `out`, the rows, the
/// scratch and the cells, and the parameters.
struct Bx {
    starts: *const u32,
    evs: *const u32,
    revs: *const u64,
    rd: *mut u32,
    rs: *mut u32,
    re: *mut u32,
    rr: *mut u64,
    row_of: *mut u32,
    /// Per row of a person with clicks: [end session, day * campaigns].
    rinfo: *mut [u32; 2],
    upto: *mut u32,
    hot: *mut [u32; 2],
    buy: *mut [u64; 2],
    shift: u32,
    days: usize,
    cn: usize,
    org: usize,
    base: i64,
}

/// Rows and organic cells of the sessions s0..s1 of a person without clicks, after row `r`
/// (rows are summed in place: the sessions, events and revenue slots start at 0). Returns the
/// person's last row, and false when a day is out of range.
///
/// # Safety
/// s0 < s1 <= the length of start, events, sess_rev and row_of; the row slots r + 1.. have room
/// for one row per session; the cells hold days * campaigns entries and organic < campaigns.
#[inline(never)]
unsafe fn person_organic(x: &Bx, s0: usize, s1: usize, mut r: usize) -> (usize, bool) {
    let mut prev = u32::MAX;
    for s in s0..s1 {
        // SAFETY: the contract (r grows by one per session at most; day < days is checked).
        unsafe {
            // start < 2^31 and day_shift < DAY, so the sum fits in u32
            let day = (*x.starts.add(s) + x.shift) / DAY;
            if day as usize >= x.days {
                return (r, false);
            }
            r = r.wrapping_add((day != prev) as usize);
            prev = day;
            let rev = *x.revs.add(s);
            *x.rd.add(r) = day;
            *x.rs.add(r) += 1;
            *x.re.add(r) += *x.evs.add(s);
            *x.rr.add(r) += rev;
            *x.row_of.add(s) = r as u32;
            let cell = day as usize * x.cn + x.org;
            (*x.hot.add(cell))[0] += 1;
            if rev > 0 {
                let b = &mut *x.buy.add(cell);
                b[0] += 1;
                b[1] += rev;
            }
        }
    }
    (r, true)
}

/// A person with clicks: its rows (as `person_organic`, plus each row's end session and day
/// cell base), then one pass over its sessions with two pointers into the sorted keys `ks[..n]`
/// that only grow: cp = clicks at or before the start, ap = clicks more than REACH before it. The
/// credit is click cp - 1 when cp > ap. A click assists the sessions from the one that moved cp
/// past it (`low`) to the one that moves ap past it (or the person's last): the range is added
/// when ap passes, in click order, per campaign from `upto` on (session indices grow across the
/// partition's persons, so `upto` needs no stamp), split by rows.
///
/// # Safety
/// As `person_organic`; `ks[..n]` sorted keys of campaigns below `x.cn`, `ks[n]` = u64::MAX;
/// `low` has room for n and `cww[..n]` holds the click words by key index; `upto` holds `x.cn` entries, none above s0; `rinfo` has room like the
/// rows.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
unsafe fn person_clicks<F: FnMut(u64, u64) -> Ordering>(
    x: &Bx,
    ks: *const u64,
    low: *mut u32,
    cww: *const u64,
    s0: usize,
    s1: usize,
    mut r: usize,
    tie: &mut F,
    ties: &mut [u64; 2],
) -> (usize, bool) {
    const LM: u64 = (1 << 52) - 1;
    const CM: u64 = 0xfff;
    let reach = REACH as u64;
    let mut prev = u32::MAX;
    for s in s0..s1 {
        // SAFETY: the contract (r grows by one per session at most; day < days is checked).
        unsafe {
            let day = (*x.starts.add(s) + x.shift) / DAY;
            if day as usize >= x.days {
                return (r, false);
            }
            r = r.wrapping_add((day != prev) as usize);
            prev = day;
            *x.rd.add(r) = day;
            *x.rs.add(r) += 1;
            *x.re.add(r) += *x.evs.add(s);
            *x.rr.add(r) += *x.revs.add(s);
            *x.row_of.add(s) = r as u32;
            *x.rinfo.add(r) = [s as u32 + 1, (day as usize * x.cn) as u32];
        }
    }
    let (starts, revs, row_of, rinfo, upto, hot, rs) = (x.starts, x.revs, x.row_of, x.rinfo, x.upto, x.hot, x.rs as *const u32);
    let base = x.base;
    // The assists of click $j, whose sessions end before $hi: [max(low, upto), hi) by rows.
    // SAFETY (both uses): j < cp <= n; low[j] was written when cp passed j; campaign < campaigns;
    // from < hi <= s1, sessions of this person, whose rows (row_of, rinfo, sessions) are written;
    // the row holding session hi - 1 ends at hi or later, which ends the loop inside the person's
    // rows, and a row after the first has a row before it.
    macro_rules! range {
        ($j:expr, $hi:expr) => {{
            let (j, hi): (usize, usize) = ($j, $hi);
            let c = (*ks.add(j) & CM) as usize;
            let u = upto.add(c);
            let from = (*low.add(j)).max(*u) as usize;
            *u = hi as u32;
            if from < hi {
                // the campaign's assist counter of day cell base dc: hc[2 * dc]
                let hc = (hot as *mut u32).add(2 * c + 1);
                let mut ri = rinfo.add(*row_of.add(from) as usize);
                // first row: from `from` to the row's end, or to hi
                let [e, dc] = *ri;
                let e = (e as usize).min(hi);
                *hc.add(2 * dc as usize) += (e - from) as u32;
                if e < hi {
                    // whole rows (their session counts are final: the rows were written above),
                    // then the row that holds session hi - 1, from its first session
                    ri = ri.add(1);
                    let mut rsp = rs.add(ri.offset_from(rinfo) as usize);
                    while ((*ri)[0] as usize) < hi {
                        *hc.add(2 * (*ri)[1] as usize) += *rsp;
                        ri = ri.add(1);
                        rsp = rsp.add(1);
                    }
                    *hc.add(2 * (*ri)[1] as usize) += hi as u32 - (*ri.sub(1))[0];
                }
            }
        }};
    }
    let (mut cp, mut ap) = (0usize, 0usize);
    let (mut memo_g, mut memo_c) = (usize::MAX, 0usize);
    for s in s0..s1 {
        // SAFETY: see `range`; cp, ap <= n, as ks[n] is the sentinel above every bound; ap <= cp
        // (a click more than REACH before the start is before it); cp - 1 and cp - 2 are read
        // only when cp - ap >= 1 and 2; day < days and campaign < campaigns: the cell is in range.
        unsafe {
            // start - base < 2^32 (start < 2^31, base > -2^31)
            let at = (i64::from(*starts.add(s)) - base) as u64;
            let b1 = (at + 1) << 32;
            while *ks.add(cp) < b1 {
                *low.add(cp) = s as u32;
                cp += 1;
            }
            let b2 = at.saturating_sub(reach) << 32;
            while *ks.add(ap) < b2 {
                range!(ap, s);
                ap += 1;
            }
            // credit: the last click at or before the start, if it is still in reach
            let mut camp = x.org;
            if cp > ap {
                let k = *ks.add(cp - 1);
                camp = (k & CM) as usize;
                if cp - ap >= 2 && (*ks.add(cp - 2) ^ k) >> 32 == 0 {
                    // a ts tie: the smallest click_id of the group (equal ids: the later)
                    let mut g = cp - 1;
                    while g > 0 && (*ks.add(g - 1) ^ k) >> 32 == 0 {
                        g -= 1;
                    }
                    if g == memo_g {
                        camp = memo_c;
                    } else {
                        let mut mixed = false;
                        for i in g..cp {
                            mixed |= (*ks.add(i) & CM) as usize != camp;
                        }
                        if mixed {
                            ties[0] += 1;
                            let mut best = g;
                            for i in g + 1..cp {
                                ties[1] += 1;
                                let (lb, li) = (((*ks.add(best) >> 12) & 0xF_FFFF) as usize, ((*ks.add(i) >> 12) & 0xF_FFFF) as usize);
                                if tie(*cww.add(lb) & LM, *cww.add(li) & LM) != Ordering::Less {
                                    best = i;
                                }
                            }
                            camp = (*ks.add(best) & CM) as usize;
                        }
                        memo_g = g;
                        memo_c = camp;
                    }
                }
            }
            let cell = (*rinfo.add(*row_of.add(s) as usize))[1] as usize + camp;
            (*hot.add(cell))[0] += 1;
            let rev = *revs.add(s);
            if rev > 0 {
                let b = &mut *x.buy.add(cell);
                b[0] += 1;
                b[1] += rev;
            }
        }
    }
    // the clicks still in reach of the last session assist up to it
    for j in ap..cp {
        // SAFETY: see `range`.
        unsafe {
            range!(j, s1);
        }
    }
    (r, true)
}

/// Sorts the `n` keys at `kw` (n > 24, distinct, `tc << 32 | ..` with tc below 2^32) ascending:
/// one distribution by the top bits of the time into about n / 2 buckets (clicks are spread over
/// the click window, so a bucket holds a few keys), then one insertion pass over the nearly
/// sorted result. Returns true when the sorted keys are in `tmp[..n]` (room for n + 1), false when
/// they were sorted in place (a bucket above 24 keys: the comparison sort, so no input is slow).
///
/// # Safety
/// `kw[..n]` is initialised and writable; n > 24.
#[inline(never)]
unsafe fn sort_keys_big(kw: *mut u64, n: usize, tmp: &mut Vec<u64>, cnt: &mut Vec<u32>) -> bool {
    let b = (usize::BITS - (n - 1).leading_zeros() - 1).min(14);
    let nb = 1usize << b;
    let sh = 64 - b;
    cnt.clear();
    cnt.resize(nb, 0);
    let c = cnt.as_mut_ptr();
    // SAFETY: key >> sh < nb; the prefix sums place each key below n; tmp has room for n + 1.
    unsafe {
        for i in 0..n {
            *c.add((*kw.add(i) >> sh) as usize) += 1;
        }
        let (mut sum, mut mx) = (0u32, 0u32);
        for i in 0..nb {
            let x = *c.add(i);
            *c.add(i) = sum;
            sum += x;
            mx = mx.max(x);
        }
        if mx > 24 {
            core::slice::from_raw_parts_mut(kw, n).sort_unstable();
            return false;
        }
        if tmp.capacity() < n + 1 {
            tmp.clear();
            tmp.reserve(n + 1);
        }
        let t = tmp.as_mut_ptr();
        for i in 0..n {
            let x = *kw.add(i);
            let at = c.add((x >> sh) as usize);
            *t.add(*at as usize) = x;
            *at += 1;
        }
        for i in 1..n {
            let x = *t.add(i);
            let mut j = i;
            while j > 0 && *t.add(j - 1) > x {
                *t.add(j) = *t.add(j - 1);
                j -= 1;
            }
            *t.add(j) = x;
        }
    }
    true
}

/// `walk` for v4 person blocks (`PersonBlocks`, after `sessions_blocks` on the same `out`):
/// clicks straight from the click blocks (person k = the block's person index, no lookup),
/// purchases from `out.buy_pay`. Err: the caller falls back (v3).
///
/// One person at a time, in three small loops that each keep their values in registers
/// (`person_keys`, `person_organic`, `person_clicks`): a person without sessions skips its
/// clicks, a person without clicks takes its rows and the organic cell, a person with both sorts
/// its keys in a buffer that stays in L1 and walks its sessions once.
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
    let np = out.persons.len();
    let ns = out.start.len();
    let cn = prm.campaigns;
    // organic is the last campaign cell (as P2 lays them out): one compare validates a click
    if pb.ev_at.len() != np + 1
        || pb.ck_at.len() != np + 1
        || out.sess_end.len() != np
        || out.events.len() != ns
        || out.buy_pay.len() != out.buy_session.len()
        || hot.len() != prm.days * cn
        || buy.len() != prm.days * cn
        || prm.organic as usize + 1 != cn
        || prm.days * cn > u32::MAX as usize
        || ns >= u32::MAX as usize
    {
        return Err(Error::WalkInput);
    }
    if prm.base > 0 || prm.base <= -(1i64 << 31) {
        return Err(Error::WalkInput);
    }
    let pay = &out.buy_pay;
    purchases_by_session(pay.len(), |j| pay[j], out, ws)?;
    // one row per session at most; sessions, events and revenue are summed in place from 0
    rows.day.reserve(ns);
    rows.sessions.resize(ns, 0);
    rows.events.resize(ns, 0);
    rows.revenue.resize(ns, 0);
    rows.row_end.reserve(np);
    ws.row_info.clear();
    ws.row_info.reserve(ns);
    ws.upto.clear();
    ws.upto.resize(cn, 0);
    let mut kv = KeyBufs { kb: core::mem::take(&mut ws.ka), cw: core::mem::take(&mut ws.cw), lo: core::mem::take(&mut ws.last_at), tmp: core::mem::take(&mut ws.kb), cnt: core::mem::take(&mut ws.cidx), dropped: 0 };
    let x = Bx {
        starts: out.start.as_ptr(),
        evs: out.events.as_ptr(),
        revs: ws.sess_rev.as_ptr(),
        rd: rows.day.as_mut_ptr(),
        rs: rows.sessions.as_mut_ptr(),
        re: rows.events.as_mut_ptr(),
        rr: rows.revenue.as_mut_ptr(),
        row_of: ws.row_of.as_mut_ptr(),
        rinfo: ws.row_info.as_mut_ptr(),
        upto: ws.upto.as_mut_ptr(),
        hot: hot.as_mut_ptr(),
        buy: buy.as_mut_ptr(),
        shift: prm.day_shift,
        days: prm.days,
        cn,
        org: prm.organic as usize,
        base: prm.base,
    };
    let mut ties = [0u64; 2];
    let mut kept = 0u64;
    let mut res = Ok(());
    // current row
    let mut r = usize::MAX;
    let mut s0 = 0usize;
    for p in 0..np {
        // Click blocks sit wherever P1's streams put them: ask for the first two lines of the
        // click block of the person three ahead (each block read was a miss).
        #[cfg(target_arch = "x86_64")]
        if p + 3 < np {
            if let Some(nb) = pb.ck.get(pb.ck_at[p + 3] as usize) {
                let q = nb.as_ptr() as *const i8;
                // SAFETY: a prefetch never faults; the addresses are in or just past a live block.
                unsafe {
                    core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(q);
                    core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(q.wrapping_add(64));
                }
            }
        }
        let s1 = out.sess_end[p] as usize;
        if s1 < s0 || s1 > ns {
            res = Err(Error::WalkInput);
            break;
        }
        if s1 > s0 {
            let (c0, c1) = (pb.ck_at[p] as usize, pb.ck_at[p + 1] as usize);
            let mut n = 0usize;
            let mut ks = core::ptr::null_mut();
            if c1 > c0 {
                let Some(blocks) = pb.ck.get(c0..c1) else {
                    res = Err(Error::WalkInput);
                    break;
                };
                (ks, n) = person_keys(blocks, &mut kv, prm.organic as u64);
                if n == usize::MAX {
                    res = Err(Error::WalkInput);
                    break;
                }
                kept += n as u64;
            }
            // SAFETY: s0 < s1 <= ns, the length of start, events, sess_rev and row_of; the rows
            // (and row_info) have room for ns, one per session; the cells hold days * campaigns
            // entries; person_keys gave sorted keys of campaigns below organic with the sentinel,
            // room in `lo` for them and their words in `cw`; upto holds campaigns entries, none above s0 (cleared
            // above, then set to session indices of earlier persons only).
            let (r1, ok) = unsafe {
                if n == 0 {
                    person_organic(&x, s0, s1, r)
                } else {
                    person_clicks(&x, ks, kv.lo.as_mut_ptr(), kv.cw.as_ptr(), s0, s1, r, tie, &mut ties)
                }
            };
            if !ok {
                res = Err(Error::DayOutOfRange(s0));
                break;
            }
            r = r1;
        }
        rows.row_end.push(r.wrapping_add(1) as u32);
        s0 = s1;
    }
    ws.stats.clicks_kept += kept;
    ws.stats.clicks_dropped += kv.dropped;
    ws.stats.tie_groups += ties[0];
    ws.stats.tie_calls += ties[1];
    ws.ka = kv.kb;
    ws.cw = kv.cw;
    ws.last_at = kv.lo;
    ws.kb = kv.tmp;
    ws.cidx = kv.cnt;
    res?;
    let nr = r.wrapping_add(1);
    // SAFETY: the loops wrote the day of rows 0..nr (nr <= ns, the reserved room).
    unsafe {
        rows.day.set_len(nr);
    }
    rows.sessions.truncate(nr);
    rows.events.truncate(nr);
    rows.revenue.truncate(nr);
    rows.top.resize(nr, NO_TOP);
    top_categories(|j| pay[j], ws, out, rows);
    Ok(())
}
