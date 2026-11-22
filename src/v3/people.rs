//! Person partitions (move 2), one task each: sortlab cuts the sessions of every person from the
//! 8-byte survivor records (no sort of events) and places each purchase in its session; clicks are
//! grouped by person and sorted per person (ts ties broken by click_id bytes, descending, read
//! through the locators); then per person a walk over its sessions does the crediting, assists
//! and the user_daily rows. Every person is handled the same way; nothing is quadratic in a
//! person's size.

use parselab::{Ctx, Dataset, Stream};

use crate::fast::agg::FastAgg;
use crate::fast::scan::{CLICK_REACH, Rows, SESSION_GAP};

// sortlab cuts sessions at its own constant: it must be the rules' 30 minutes.
const _: () = assert!(SESSION_GAP == sortlab::GAP as i64);
use crate::ts::DAY;

use super::tokens::Tokens;

#[derive(Clone, Copy)]
struct Ck {
    ts: i64,
    campaign: u32,
}

/// One person's sessions: starts (ms since the window start) and event counts in time order;
/// the purchases of session s are `sbuy[sb_at[s]..sb_at[s + 1]]`, indexes into `buys`.
struct Sess<'x> {
    start: &'x [u32],
    events: &'x [u32],
    sb_at: &'x [u32],
    sbuy: &'x [u32],
    buys: &'x [[u64; 2]],
}

pub struct PeopleCtx<'a, 'b> {
    pub ds: &'a Dataset,
    pub ctx: &'b Ctx,
    pub tok: &'b Tokens<'a>,
    pub organic: u32,
    /// The sortlab session kernel.
    pub kernel: sortlab::Kernel,
}

pub struct Worker {
    /// A check failed (a purchase outside every session, a purchase of an absent person): the
    /// run falls back.
    pub bad: bool,
    pub rows: Rows,
    pub agg: FastAgg,
    pub sessions: u64,
    pub events: u64,
    ev: Vec<u64>,
    buys: Vec<[u64; 2]>,
    sc: sortlab::Scratch,
    out: sortlab::Out,
    /// Purchases by session: `sbuy[sb_at[s]..sb_at[s + 1]]` are the purchases of session s.
    sb_at: Vec<u32>,
    sbuy: Vec<u32>,
    /// Person token to output person index (open addressing, EMPTY keys), for the clicks.
    pkeys: Vec<u32>,
    pvals: Vec<u32>,
    /// Clicks grouped by output person: `cgrp[c_at[k]..c_at[k + 1]]`.
    c_at: Vec<u32>,
    cidx: Vec<u32>,
    cgrp: Vec<[u64; 2]>,
    ck: Vec<Ck>,
    starts: Vec<(i64, i32)>,
    /// Per entry of `starts`, the index of the first later entry on another day.
    day_end: Vec<u32>,
    /// Interval assists: per campaign stamp and local index, the person's campaigns, group
    /// ends and timestamps grouped by campaign.
    pno: u32,
    cstamp: Vec<u32>,
    cidx_of: Vec<u32>,
    clist: Vec<u32>,
    cat: Vec<u32>,
    cts: Vec<i64>,
    cat_sum: [u64; 256],
    touched: Vec<u8>,
    stamp: Vec<u32>,
    session_no: u32,
    name_buf: Vec<u8>,
    /// Nanoseconds by phase (gather, sessions kernel, purchases, clicks, walk) when timing.
    pub phase_ns: [u64; 5],
    pub timing: bool,
    /// Counts when timing: clicks, tie groups re-read, persons, rows, direct assist steps,
    /// persons on the interval route.
    pub counts: [u64; 6],
    /// Interval route counts when timing: clicks, sessions, campaigns, intervals, day steps.
    pub icount: [u64; 5],
}


const DIRECT_ASSISTS: usize = 64;
const EMPTY: u32 = u32::MAX;

/// The click_id of the click line at `loc`: its first field, unquoted (the line was accepted by
/// the parser, so its quoting is legal), read without scanning the rest of the line.
fn click_id(ds: &Dataset, loc: parselab::Locator) -> &[u8] {
    let data = ds.stream(Stream::Clicks)[loc.file() as usize].bytes();
    let rest = &data[loc.offset() as usize..];
    let end = rest.iter().position(|&c| c == b',' || c == b'\n').unwrap_or(rest.len());
    let f = &rest[..end];
    parselab::csv::unquote(f).unwrap_or(f)
}

/// The first index at or after `from` whose start fails `before` (which holds on a prefix of
/// `st[from..]`), by doubling steps from `from` and then a binary search.
#[inline]
fn gallop(st: &[(i64, i32)], from: usize, before: impl Fn(i64) -> bool) -> usize {
    let n = st.len();
    if from >= n || !before(st[from].0) {
        return from;
    }
    let mut lo = from;
    let mut step = 1;
    loop {
        let hi = lo + step;
        if hi >= n || !before(st[hi].0) {
            let hi = hi.min(n);
            return lo + 1 + st[lo + 1..hi].partition_point(|s| before(s.0));
        }
        lo = hi;
        step *= 2;
    }
}

#[inline(always)]
fn phash(tok: u32, bits: u32) -> usize {
    (tok.wrapping_mul(0x9E37_79B1) >> (32 - bits)) as usize
}

impl Worker {
    pub fn new(agg: FastAgg) -> Self {
        Worker {
            bad: false,
            rows: Rows::new(),
            agg,
            sessions: 0,
            events: 0,
            ev: Vec::new(),
            buys: Vec::new(),
            sc: sortlab::Scratch::new(),
            out: sortlab::Out::default(),
            sb_at: Vec::new(),
            sbuy: Vec::new(),
            pkeys: Vec::new(),
            pvals: Vec::new(),
            c_at: Vec::new(),
            cidx: Vec::new(),
            cgrp: Vec::new(),
            ck: Vec::new(),
            starts: Vec::new(),
            day_end: Vec::new(),
            pno: 0,
            cstamp: Vec::new(),
            cidx_of: Vec::new(),
            clist: Vec::new(),
            cat: Vec::new(),
            cts: Vec::new(),
            cat_sum: [0; 256],
            touched: Vec::new(),
            stamp: Vec::new(),
            session_no: 0,
            name_buf: Vec::new(),
            phase_ns: [0; 5],
            timing: false,
            counts: [0; 6],
            icount: [0; 5],
        }
    }

    /// One person partition: survivor records, purchase records and click records of it.
    pub fn partition(&mut self, ev: &[&[u64]], buys: &[&[[u64; 2]]], clicks: &[&[[u64; 2]]], cx: &PeopleCtx) {
        let timing = self.timing;
        let mut t_last = if timing { Some(std::time::Instant::now()) } else { None };
        let mut lap = |w: &mut [u64; 5], i: usize| {
            if let Some(t) = t_last.as_mut() {
                let now = std::time::Instant::now();
                w[i] += (now - *t).as_nanos() as u64;
                *t = now;
            }
        };
        self.ev.clear();
        for c in ev {
            self.ev.extend_from_slice(c);
        }
        if self.ev.is_empty() {
            if buys.iter().any(|c| !c.is_empty()) {
                self.bad = true;
            }
            return;
        }
        self.buys.clear();
        for c in buys {
            self.buys.extend_from_slice(c);
        }
        lap(&mut self.phase_ns, 0);
        // Sessions per person (start, event count, in time order) and each purchase's session.
        if sortlab::sessions(&self.ev, &self.buys, cx.kernel, None, &mut self.sc, &mut self.out).is_err() {
            self.bad = true;
            return;
        }
        lap(&mut self.phase_ns, 1);
        // Purchases grouped by session (input order within a session).
        let ns = self.out.start.len();
        self.sb_at.clear();
        self.sb_at.resize(ns + 2, 0);
        for &s in &self.out.buy_session {
            if s as usize >= ns {
                self.bad = true;
                return;
            }
            self.sb_at[s as usize + 2] += 1;
        }
        for s in 2..ns + 2 {
            self.sb_at[s] += self.sb_at[s - 1];
        }
        self.sbuy.clear();
        self.sbuy.resize(self.out.buy_session.len(), 0);
        for (j, &s) in self.out.buy_session.iter().enumerate() {
            let at = &mut self.sb_at[s as usize + 1];
            self.sbuy[*at as usize] = j as u32;
            *at += 1;
        }
        lap(&mut self.phase_ns, 2);
        // Clicks grouped by output person; clicks of persons without events here are dropped.
        let np = self.out.persons.len();
        let bits = ((2 * np).max(64).next_power_of_two().trailing_zeros()).min(31);
        let mask = (1usize << bits) - 1;
        self.pkeys.clear();
        self.pkeys.resize(1 << bits, EMPTY);
        self.pvals.clear();
        self.pvals.resize(1 << bits, 0);
        for (k, &p) in self.out.persons.iter().enumerate() {
            let mut h = phash(p, bits);
            while self.pkeys[h] != EMPTY {
                h = (h + 1) & mask;
            }
            self.pkeys[h] = p;
            self.pvals[h] = k as u32;
        }
        self.c_at.clear();
        self.c_at.resize(np + 2, 0);
        self.cidx.clear();
        for c in clicks {
            for r in c.iter() {
                let p = (r[0] >> 32) as u32;
                let mut k = EMPTY;
                if p != EMPTY {
                    let mut h = phash(p, bits);
                    loop {
                        let key = self.pkeys[h];
                        if key == p {
                            k = self.pvals[h];
                            break;
                        }
                        if key == EMPTY {
                            break;
                        }
                        h = (h + 1) & mask;
                    }
                }
                self.cidx.push(k);
                if k != EMPTY {
                    self.c_at[k as usize + 2] += 1;
                }
            }
        }
        for k in 2..np + 2 {
            self.c_at[k] += self.c_at[k - 1];
        }
        let kept = self.c_at[np + 1] as usize;
        self.cgrp.clear();
        self.cgrp.resize(kept, [0, 0]);
        let mut i = 0;
        for c in clicks {
            for r in c.iter() {
                let k = self.cidx[i];
                i += 1;
                if k != EMPTY {
                    let at = &mut self.c_at[k as usize + 1];
                    self.cgrp[*at as usize] = *r;
                    *at += 1;
                }
            }
        }
        self.counts[0] += kept as u64;
        // Per person: ts ascending, then click_id descending on ties.
        let ds = cx.ds;
        let mut keyed: Vec<(&[u8], [u64; 2])> = Vec::new();
        let mut cgrp = std::mem::take(&mut self.cgrp);
        for k in 0..np {
            let g = &mut cgrp[self.c_at[k] as usize..self.c_at[k + 1] as usize];
            if g.len() < 2 {
                continue;
            }
            if g.len() <= 24 {
                for a in 1..g.len() {
                    let x = g[a];
                    let mut b = a;
                    while b > 0 && g[b - 1][0] > x[0] {
                        g[b] = g[b - 1];
                        b -= 1;
                    }
                    g[b] = x;
                }
            } else {
                g.sort_unstable_by_key(|r| r[0]);
            }
            let mut a = 0;
            while a < g.len() {
                let mut b = a + 1;
                while b < g.len() && g[b][0] == g[a][0] {
                    b += 1;
                }
                // A ts tie matters only when the tied clicks name different campaigns (copies of
                // a click are identical): then order them by click_id descending.
                if b - a > 1 && g[a + 1..b].iter().any(|c| c[1] >> 48 != g[a][1] >> 48) {
                    self.counts[1] += 1;
                    keyed.clear();
                    for c in &g[a..b] {
                        let loc = parselab::Locator(c[1] & ((1 << 48) - 1));
                        keyed.push((click_id(ds, loc), *c));
                    }
                    keyed.sort_by(|x, y| y.0.cmp(x.0));
                    for (o, (_, c)) in keyed.iter().enumerate() {
                        g[a + o] = *c;
                    }
                }
                a = b;
            }
        }
        lap(&mut self.phase_ns, 3);
        // Walk persons over their sessions.
        let out = std::mem::take(&mut self.out);
        let (sb_at, sbuy, buys) = (std::mem::take(&mut self.sb_at), std::mem::take(&mut self.sbuy), std::mem::take(&mut self.buys));
        let base = cx.ctx.click_base_ms - cx.ctx.window_start_ms;
        let mut s0 = 0usize;
        for k in 0..np {
            let person = out.persons[k];
            let s1 = out.sess_end[k] as usize;
            if k + 1 < np {
                // The next person's name is read when its first row is written: start the load.
                cx.tok.prefetch_key(out.persons[k + 1]);
            }
            self.ck.clear();
            self.ck.extend(
                cgrp[self.c_at[k] as usize..self.c_at[k + 1] as usize]
                    .iter()
                    .map(|r| Ck { ts: base + (r[0] & 0xffff_ffff) as i64, campaign: (r[1] >> 48) as u32 }),
            );
            self.counts[2] += 1;
            let ck = std::mem::take(&mut self.ck);
            let ss = Sess { start: &out.start[s0..s1], events: &out.events[s0..s1], sb_at: &sb_at[s0..s1 + 1], sbuy: &sbuy, buys: &buys };
            self.person(person, &ss, &ck, cx);
            self.ck = ck;
            s0 = s1;
        }
        self.events += self.ev.len() as u64;
        self.out = out;
        self.sb_at = sb_at;
        self.sbuy = sbuy;
        self.buys = buys;
        self.cgrp = cgrp;
        lap(&mut self.phase_ns, 4);
    }

    /// One person: its sessions in time order, its clicks sorted by (ts, click_id descending), ts
    /// in the event base.
    fn person(&mut self, person: u32, ss: &Sess, clicks: &[Ck], cx: &PeopleCtx) {
        let ws = cx.ctx.window_start_ms;
        self.starts.clear();
        let mut k = 0usize;
        let mut day_now = i32::MIN;
        // Start (since the window start) of the day after `day_now`: sessions come in time order,
        // so the day changes only when a start reaches it.
        let mut next_day_at = i64::MIN;
        let (mut d_sessions, mut d_events, mut d_revenue) = (0u32, 0u32, 0i64);
        let mut first_row = true;
        let direct = clicks.len() <= DIRECT_ASSISTS;
        let want_starts = !clicks.is_empty() && !direct;
        let mut lo = 0usize;
        for s in 0..ss.start.len() {
            let start = i64::from(ss.start[s]);
            if start >= next_day_at {
                let day = (ws + start).div_euclid(DAY) as i32;
                if day_now != i32::MIN {
                    self.emit_day(person, day_now, d_sessions, d_events, d_revenue, &mut first_row, cx);
                }
                day_now = day;
                next_day_at = (i64::from(day) + 1) * DAY - ws;
                d_sessions = 0;
                d_events = 0;
                d_revenue = 0;
            }
            let day = day_now;
            let mut revenue = 0u64;
            for &j in &ss.sbuy[ss.sb_at[s] as usize..ss.sb_at[s + 1] as usize] {
                let buy = ss.buys[j as usize][1] as u32;
                let price = u64::from(buy >> 8);
                let cat = (buy & 255) as usize;
                revenue += price;
                if self.cat_sum[cat] == 0 {
                    self.touched.push(cat as u8);
                }
                self.cat_sum[cat] += price;
            }
            d_sessions += 1;
            d_events += ss.events[s];
            d_revenue += revenue as i64;
            while k < clicks.len() && clicks[k].ts <= start {
                k += 1;
            }
            let campaign = if k > 0 && start - clicks[k - 1].ts <= CLICK_REACH { clicks[k - 1].campaign } else { cx.organic };
            let off = self.agg.offset(day);
            self.agg.credit(campaign, off, revenue > 0, revenue);
            if direct && k > 0 {
                while lo < k && start - clicks[lo].ts > CLICK_REACH {
                    lo += 1;
                }
                if lo < k {
                    self.session_no = self.session_no.wrapping_add(1);
                    if self.session_no == 0 {
                        self.stamp.iter_mut().for_each(|x| *x = 0);
                        self.session_no = 1;
                    }
                    self.counts[4] += (k - lo) as u64;
                    for c in &clicks[lo..k] {
                        let ci = c.campaign as usize;
                        if ci >= self.stamp.len() {
                            self.stamp.resize(ci + 1, 0);
                        }
                        if self.stamp[ci] != self.session_no {
                            self.stamp[ci] = self.session_no;
                            self.agg.assist(c.campaign, off, 1);
                        }
                    }
                }
            }
            if want_starts {
                self.starts.push((start, day));
            }
            self.sessions += 1;
        }
        if day_now != i32::MIN {
            self.emit_day(person, day_now, d_sessions, d_events, d_revenue, &mut first_row, cx);
        }
        if want_starts {
            self.counts[5] += 1;
            self.assists(clicks);
        }
    }

    fn assists(&mut self, clicks: &[Ck]) {
        let n = self.starts.len();
        self.day_end.clear();
        self.day_end.resize(n, n as u32);
        for i in (0..n.saturating_sub(1)).rev() {
            self.day_end[i] = if self.starts[i + 1].1 != self.starts[i].1 { i as u32 + 1 } else { self.day_end[i + 1] };
        }
        // Group the clicks by campaign, keeping time order inside a campaign (counting sort over
        // the person's distinct campaigns; no comparison sort).
        self.pno = self.pno.wrapping_add(1);
        if self.pno == 0 {
            self.cstamp.iter_mut().for_each(|x| *x = 0);
            self.pno = 1;
        }
        self.clist.clear();
        self.cat.clear();
        for c in clicks {
            let ci = c.campaign as usize;
            if ci >= self.cstamp.len() {
                self.cstamp.resize(ci + 1, 0);
                self.cidx_of.resize(ci + 1, 0);
            }
            if self.cstamp[ci] != self.pno {
                self.cstamp[ci] = self.pno;
                self.cidx_of[ci] = self.clist.len() as u32;
                self.clist.push(c.campaign);
                self.cat.push(0);
            }
            self.cat[self.cidx_of[ci] as usize] += 1;
        }
        let mut sum = 0u32;
        for x in self.cat.iter_mut() {
            let c = *x;
            *x = sum;
            sum += c;
        }
        self.cts.clear();
        self.cts.resize(clicks.len(), 0);
        for c in clicks {
            let g = self.cidx_of[c.campaign as usize] as usize;
            let at = self.cat[g] as usize;
            self.cts[at] = c.ts;
            self.cat[g] += 1;
        }
        // Per campaign: merge [t, t + 7 days] into disjoint intervals in time order, and count the
        // sessions starting inside each, by day; the search position only moves forward.
        let (clist, cts, cat) = (std::mem::take(&mut self.clist), std::mem::take(&mut self.cts), std::mem::take(&mut self.cat));
        if self.timing {
            self.icount[0] += clicks.len() as u64;
            self.icount[1] += n as u64;
            self.icount[2] += clist.len() as u64;
        }
        let mut g0 = 0usize;
        for (g, &campaign) in clist.iter().enumerate() {
            let ts = &cts[g0..cat[g] as usize];
            g0 = cat[g] as usize;
            let mut at = 0usize;
            let mut a = ts[0];
            let mut b = a + CLICK_REACH;
            for &t in &ts[1..] {
                if t <= b {
                    b = b.max(t + CLICK_REACH);
                } else {
                    at = self.count_assists(campaign, a, b, at);
                    a = t;
                    b = t + CLICK_REACH;
                }
            }
            self.count_assists(campaign, a, b, at);
        }
        self.clist = clist;
        self.cts = cts;
        self.cat = cat;
    }

    /// Sessions starting in [a, b], by day, as assists of `campaign`; the search starts at
    /// `from` (no session before it starts at or after a). Returns the end of the range.
    fn count_assists(&mut self, campaign: u32, a: i64, b: i64, from: usize) -> usize {
        let lo = gallop(&self.starts, from, |s| s < a);
        let hi = gallop(&self.starts, lo, |s| s <= b);
        let mut i = lo;
        if self.timing {
            self.icount[3] += 1;
        }
        while i < hi {
            if self.timing {
                self.icount[4] += 1;
            }
            let day = self.starts[i].1;
            let end = (self.day_end[i] as usize).min(hi);
            let off = self.agg.offset(day);
            self.agg.assist(campaign, off, (end - i) as u32);
            i = end;
        }
        hi
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_day(&mut self, person: u32, day: i32, sessions: u32, events: u32, revenue: i64, first: &mut bool, cx: &PeopleCtx) {
        let mut top = pqw::NULL8;
        let mut top_sum = 0u64;
        for &cat in &self.touched {
            let sum = self.cat_sum[cat as usize];
            if sum > top_sum || (sum == top_sum && cat < top) {
                top = cat;
                top_sum = sum;
            }
            self.cat_sum[cat as usize] = 0;
        }
        self.touched.clear();
        let r = &mut self.rows;
        if *first {
            *first = false;
            self.name_buf.clear();
            cx.tok.bytes(person, &mut self.name_buf);
            r.names.extend_from_slice(&self.name_buf);
            r.name_offsets.push(r.names.len() as u32);
            r.runs.push(0);
        }
        *r.runs.last_mut().unwrap() += 1;
        self.counts[3] += 1;
        r.day.push(day);
        r.sessions.push(sessions);
        r.events.push(events);
        r.revenue.push(revenue);
        r.top.push(top);
    }
}
