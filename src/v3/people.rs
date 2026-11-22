//! Person partitions (move 2), one task each: radix sort of the 8-byte survivor records by
//! (person, ts); purchases and clicks of the partition sorted the same way (click ts ties broken
//! by click_id bytes, descending, read through the locators); then per person the session scan,
//! crediting, assists and the user_daily rows. Every person is handled the same way; nothing is
//! quadratic in a person's size.

use parselab::{Ctx, Dataset, Stream};

use crate::fast::agg::FastAgg;
use crate::fast::scan::{CLICK_REACH, Rows, SESSION_GAP};
use crate::ts::DAY;

use super::tokens::Tokens;

/// LSD radix sort of u64 keys over bits [lo, hi), 11 bits a pass, skipping passes whose digit
/// is the same for every key. `tmp` is scratch of at least keys.len().
pub fn radix_sort(keys: &mut Vec<u64>, tmp: &mut Vec<u64>, lo: u32, hi: u32) {
    let n = keys.len();
    if n < 64 {
        keys.sort_unstable();
        return;
    }
    tmp.clear();
    tmp.resize(n, 0);
    let mut shift = lo;
    let mut count = [0usize; 2048];
    while shift < hi {
        let bits = (hi - shift).min(11);
        let mask = (1u64 << bits) - 1;
        count[..(1 << bits)].iter_mut().for_each(|c| *c = 0);
        for &k in keys.iter() {
            count[((k >> shift) & mask) as usize] += 1;
        }
        if count[((keys[0] >> shift) & mask) as usize] == n {
            shift += bits;
            continue;
        }
        let mut sum = 0;
        for c in count[..(1 << bits)].iter_mut() {
            let x = *c;
            *c = sum;
            sum += x;
        }
        for &k in keys.iter() {
            let d = ((k >> shift) & mask) as usize;
            tmp[count[d]] = k;
            count[d] += 1;
        }
        std::mem::swap(keys, tmp);
        shift += bits;
    }
}

#[derive(Clone, Copy)]
struct Ck {
    ts: i64,
    campaign: u32,
}

pub struct PeopleCtx<'a, 'b> {
    pub ds: &'a Dataset,
    pub ctx: &'b Ctx,
    pub tok: &'b Tokens<'a>,
    pub organic: u32,
    /// Highest bit of a person token plus one (sort bits).
    pub person_bits: u32,
}

pub struct Worker {
    /// A check failed (a purchase flag without its purchase record): the run falls back.
    pub bad: bool,
    pub rows: Rows,
    pub agg: FastAgg,
    pub sessions: u64,
    pub events: u64,
    ev: Vec<u64>,
    tmp: Vec<u64>,
    buys: Vec<[u64; 2]>,
    clicks: Vec<[u64; 2]>,
    ck: Vec<Ck>,
    starts: Vec<(i64, i32)>,
    pairs: Vec<(u32, i64)>,
    cat_sum: [u64; 256],
    touched: Vec<u8>,
    stamp: Vec<u32>,
    session_no: u32,
    name_buf: Vec<u8>,
}

const DIRECT_ASSISTS: usize = 64;

impl Worker {
    pub fn new(agg: FastAgg) -> Self {
        Worker {
            bad: false,
            rows: Rows::new(),
            agg,
            sessions: 0,
            events: 0,
            ev: Vec::new(),
            tmp: Vec::new(),
            buys: Vec::new(),
            clicks: Vec::new(),
            ck: Vec::new(),
            starts: Vec::new(),
            pairs: Vec::new(),
            cat_sum: [0; 256],
            touched: Vec::new(),
            stamp: Vec::new(),
            session_no: 0,
            name_buf: Vec::new(),
        }
    }

    /// One person partition: survivor records, purchase records and click records of it.
    pub fn partition(&mut self, ev: &[&[u64]], buys: &[&[[u64; 2]]], clicks: &[&[[u64; 2]]], cx: &PeopleCtx) {
        self.ev.clear();
        for c in ev {
            self.ev.extend_from_slice(c);
        }
        if self.ev.is_empty() {
            return;
        }
        let mut ev = std::mem::take(&mut self.ev);
        let mut tmp = std::mem::take(&mut self.tmp);
        // The purchase flag (bit 0) needs no order: sort by (person, ts) only.
        radix_sort(&mut ev, &mut tmp, 1, 32 + cx.person_bits);
        self.tmp = tmp;
        self.buys.clear();
        for c in buys {
            self.buys.extend_from_slice(c);
        }
        self.buys.sort_unstable_by_key(|b| b[0]);
        self.clicks.clear();
        for c in clicks {
            self.clicks.extend_from_slice(c);
        }
        let mut cl = std::mem::take(&mut self.clicks);
        // (person, ts) ascending, then click_id descending on ties.
        cl.sort_unstable_by(|x, y| x[0].cmp(&y[0]));
        let mut i = 0;
        while i < cl.len() {
            let mut j = i + 1;
            while j < cl.len() && cl[j][0] == cl[i][0] {
                j += 1;
            }
            // A ts tie matters only when the tied clicks name different campaigns (copies of a
            // click are identical): then order them by click_id descending, bytes read once each.
            if j - i > 1 && cl[i + 1..j].iter().any(|c| c[1] >> 48 != cl[i][1] >> 48) {
                let ds = cx.ds;
                let mut keyed: Vec<(Vec<u8>, [u64; 2])> = cl[i..j]
                    .iter()
                    .map(|c| {
                        let loc = parselab::Locator(c[1] & ((1 << 48) - 1));
                        let line = ds.line(Stream::Clicks, loc);
                        let mut f = Vec::new();
                        parselab::csv::split(line, &mut f);
                        (f.first().map(|s| s.to_vec()).unwrap_or_default(), *c)
                    })
                    .collect();
                keyed.sort_by(|x, y| y.0.cmp(&x.0));
                for (k, (_, c)) in keyed.into_iter().enumerate() {
                    cl[i + k] = c;
                }
            }
            i = j;
        }
        // Walk persons.
        let (mut a, mut b, mut c) = (0usize, 0usize, 0usize);
        while a < ev.len() {
            let person = (ev[a] >> 32) as u32;
            let mut z = a + 1;
            while z < ev.len() && (ev[z] >> 32) as u32 == person {
                z += 1;
            }
            while b < self.buys.len() && ((self.buys[b][0] >> 32) as u32) < person {
                b += 1;
            }
            let mut bz = b;
            while bz < self.buys.len() && (self.buys[bz][0] >> 32) as u32 == person {
                bz += 1;
            }
            while c < cl.len() && ((cl[c][0] >> 32) as u32) < person {
                c += 1;
            }
            let mut cz = c;
            while cz < cl.len() && (cl[cz][0] >> 32) as u32 == person {
                cz += 1;
            }
            let base = cx.ctx.click_base_ms - cx.ctx.window_start_ms;
            self.ck.clear();
            self.ck.extend(cl[c..cz].iter().map(|r| Ck { ts: base + (r[0] >> 32 << 32 ^ r[0]) as i64, campaign: (r[1] >> 48) as u32 }));
            if z < ev.len() {
                // The next person's name is read when its first row is written: start the load.
                cx.tok.prefetch_key((ev[z] >> 32) as u32);
            }
            let buys = std::mem::take(&mut self.buys);
            let ck = std::mem::take(&mut self.ck);
            self.person(person, &ev[a..z], &buys[b..bz], &ck, cx);
            self.buys = buys;
            self.ck = ck;
            a = z;
            b = bz;
            c = cz;
        }
        self.events += ev.len() as u64;
        self.ev = ev;
        self.clicks = cl;
    }

    /// `events`: this person's records sorted by ts; `buys`: its purchases sorted by ts;
    /// `clicks`: sorted by (ts, click_id descending), ts in the event base.
    fn person(&mut self, person: u32, events: &[u64], buys: &[[u64; 2]], clicks: &[Ck], cx: &PeopleCtx) {
        let ws = cx.ctx.window_start_ms;
        self.starts.clear();
        let mut k = 0usize;
        let mut bi = 0usize;
        let mut day_now = i32::MIN;
        let (mut d_sessions, mut d_events, mut d_revenue) = (0u32, 0u32, 0i64);
        let mut first_row = true;
        let n = events.len();
        let direct = clicks.len() <= DIRECT_ASSISTS;
        let want_starts = !clicks.is_empty() && !direct;
        let mut lo = 0usize;
        let mut i = 0;
        while i < n {
            let start = ((events[i] >> 1) & 0x7fff_ffff) as i64;
            let day = (ws + start).div_euclid(DAY) as i32;
            if day != day_now {
                if day_now != i32::MIN {
                    self.emit_day(person, day_now, d_sessions, d_events, d_revenue, &mut first_row, cx);
                }
                day_now = day;
                d_sessions = 0;
                d_events = 0;
                d_revenue = 0;
            }
            let mut j = i;
            let mut revenue = 0u64;
            loop {
                let e = events[j];
                if e & 1 != 0 {
                    // The next purchase record of this person (same ts order).
                    let Some(rec) = buys.get(bi) else {
                        self.bad = true;
                        return;
                    };
                    let buy = rec[1] as u32;
                    bi += 1;
                    let price = u64::from(buy >> 8);
                    let cat = (buy & 255) as usize;
                    revenue += price;
                    if self.cat_sum[cat] == 0 {
                        self.touched.push(cat as u8);
                    }
                    self.cat_sum[cat] += price;
                }
                j += 1;
                if j >= n || ((events[j] >> 1) & 0x7fff_ffff) as i64 - ((e >> 1) & 0x7fff_ffff) as i64 > SESSION_GAP {
                    break;
                }
            }
            d_sessions += 1;
            d_events += (j - i) as u32;
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
            i = j;
        }
        if day_now != i32::MIN {
            self.emit_day(person, day_now, d_sessions, d_events, d_revenue, &mut first_row, cx);
        }
        if want_starts {
            self.assists(clicks);
        }
    }

    fn assists(&mut self, clicks: &[Ck]) {
        self.pairs.clear();
        self.pairs.extend(clicks.iter().map(|c| (c.campaign, c.ts)));
        self.pairs.sort_unstable();
        let pairs = std::mem::take(&mut self.pairs);
        let mut g = 0;
        while g < pairs.len() {
            let campaign = pairs[g].0;
            let mut a = pairs[g].1;
            let mut b = a + CLICK_REACH;
            let mut h = g + 1;
            while h < pairs.len() && pairs[h].0 == campaign {
                let t = pairs[h].1;
                if t <= b {
                    b = b.max(t + CLICK_REACH);
                } else {
                    self.count_assists(campaign, a, b);
                    a = t;
                    b = t + CLICK_REACH;
                }
                h += 1;
            }
            self.count_assists(campaign, a, b);
            g = h;
        }
        self.pairs = pairs;
    }

    fn count_assists(&mut self, campaign: u32, a: i64, b: i64) {
        let lo = self.starts.partition_point(|s| s.0 < a);
        let hi = self.starts.partition_point(|s| s.0 <= b);
        let mut i = lo;
        while i < hi {
            let day = self.starts[i].1;
            let end = i + self.starts[i..hi].partition_point(|s| s.1 <= day);
            let off = self.agg.offset(day);
            self.agg.assist(campaign, off, (end - i) as u32);
            i = end;
        }
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
        r.day.push(day);
        r.sessions.push(sessions);
        r.events.push(events);
        r.revenue.push(revenue);
        r.top.push(top);
    }
}
