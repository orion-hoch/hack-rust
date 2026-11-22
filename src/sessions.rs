//! Stages 5 and 6 and the aggregation: per person, sessions from the sorted surviving events,
//! the crediting click and the assisting campaigns of each session, then the user_daily rows
//! and the campaign_daily counters.
//!
//! Per person the work is O((E + K) log K + M * (log S + D)): E events, K clicks, S sessions,
//! M merged click intervals, D days an interval spans. Nothing is quadratic in a whale.

use crate::dedup::Kept;
use crate::hash::{FastMap, map_with_capacity};
use crate::id::{Id, Interner};
use crate::output::{NO_CATEGORY, UserDaily};
use crate::parse::ClickRec;
use crate::ts::{DAY, MINUTE};

pub const SESSION_GAP: i64 = 30 * MINUTE;
pub const CLICK_REACH: i64 = 7 * DAY;

/// Counters per (campaign, day): credited sessions, purchasing sessions, revenue, assisted.
pub type Counters = [u64; 4];

/// The campaign_daily counters of one thread. Dense over (campaign, day) when that is small.
pub struct CampaignAgg {
    pub day0: i64,
    pub span: usize,
    dense: Vec<Counters>,
    sparse: FastMap<(u32, i32), Counters>,
}

const DENSE_LIMIT: usize = 1 << 20;

impl CampaignAgg {
    pub fn new(campaigns: usize, day0: i64, span: usize) -> Self {
        let cells = campaigns.saturating_mul(span);
        let dense = if cells <= DENSE_LIMIT { vec![[0; 4]; cells] } else { Vec::new() };
        CampaignAgg { day0, span, dense, sparse: map_with_capacity(0) }
    }

    #[inline]
    fn slot(&mut self, campaign: u32, day: i32) -> &mut Counters {
        if !self.dense.is_empty() {
            let offset = (i64::from(day) - self.day0) as usize;
            debug_assert!(offset < self.span);
            &mut self.dense[campaign as usize * self.span + offset]
        } else {
            self.sparse.entry((campaign, day)).or_insert([0; 4])
        }
    }

    /// Adds another thread's counters into this one.
    pub fn merge(&mut self, other: CampaignAgg) {
        if !self.dense.is_empty() {
            for (a, b) in self.dense.iter_mut().zip(other.dense) {
                for k in 0..4 {
                    a[k] += b[k];
                }
            }
        } else {
            for (key, b) in other.sparse {
                let a = self.sparse.entry(key).or_insert([0; 4]);
                for k in 0..4 {
                    a[k] += b[k];
                }
            }
        }
    }

    /// Every (campaign, day) with a credited or assisted session.
    pub fn rows(&self) -> Vec<(u32, i32, Counters)> {
        let mut out = Vec::new();
        if !self.dense.is_empty() {
            for (cell, c) in self.dense.iter().enumerate() {
                if c[0] > 0 || c[3] > 0 {
                    let campaign = (cell / self.span) as u32;
                    let day = (self.day0 + (cell % self.span) as i64) as i32;
                    out.push((campaign, day, *c));
                }
            }
        } else {
            for (&(campaign, day), c) in &self.sparse {
                if c[0] > 0 || c[3] > 0 {
                    out.push((campaign, day, *c));
                }
            }
        }
        out
    }
}

pub struct SessionCtx<'a> {
    pub ids: &'a Interner,
    pub organic: u32,
}

#[derive(Clone, Copy)]
struct Session {
    start: i64,
    day: i32,
}

/// One thread's sessionizer: its output rows and counters, plus scratch reused per person.
pub struct Sessionizer {
    pub user: UserDaily,
    pub camp: CampaignAgg,
    pub sessions_total: u64,
    cat_sum: Vec<u64>,
    touched: Vec<u32>,
    sessions: Vec<Session>,
    pairs: Vec<(u32, i64)>,
}

impl Sessionizer {
    pub fn new(categories: usize, camp: CampaignAgg) -> Self {
        Sessionizer {
            user: UserDaily::new(),
            camp,
            sessions_total: 0,
            cat_sum: vec![0; categories],
            touched: Vec::new(),
            sessions: Vec::new(),
            pairs: Vec::new(),
        }
    }

    /// Processes one person partition: sorts it and walks it person by person.
    pub fn partition(&mut self, events: &mut [Kept], clicks: &mut [ClickRec], ctx: &SessionCtx) {
        events.sort_unstable_by_key(|e| (e.person, e.ts));
        let ids = ctx.ids;
        // By person, then ts; equal ts puts the byte-wise smallest click_id last, so the last
        // click at or before a session start is the one rules.md credits.
        clicks.sort_unstable_by(|a, b| {
            a.person.cmp(&b.person).then(a.ts.cmp(&b.ts)).then_with(|| ids.cmp(b.cid, a.cid))
        });
        let mut c = 0;
        let mut i = 0;
        while i < events.len() {
            let person = events[i].person;
            let mut j = i + 1;
            while j < events.len() && events[j].person == person {
                j += 1;
            }
            while c < clicks.len() && clicks[c].person < person {
                c += 1;
            }
            let mut d = c;
            while d < clicks.len() && clicks[d].person == person {
                d += 1;
            }
            self.person(person, &events[i..j], &clicks[c..d], ctx);
            i = j;
            c = d;
        }
    }

    /// `events` sorted by ts, `clicks` sorted by (ts, click_id descending).
    fn person(&mut self, person: Id, events: &[Kept], clicks: &[ClickRec], ctx: &SessionCtx) {
        self.sessions.clear();
        let mut k = 0usize;
        let mut day_now = i32::MIN;
        let (mut day_sessions, mut day_events, mut day_revenue) = (0i64, 0i64, 0i64);
        let mut i = 0;
        while i < events.len() {
            let start = events[i].ts;
            let mut j = i + 1;
            let mut previous = start;
            while j < events.len() && events[j].ts - previous <= SESSION_GAP {
                previous = events[j].ts;
                j += 1;
            }
            let day = start.div_euclid(DAY) as i32;
            if day != day_now {
                if day_now != i32::MIN {
                    self.emit_day(person, day_now, day_sessions, day_events, day_revenue, ctx);
                }
                day_now = day;
                day_sessions = 0;
                day_events = 0;
                day_revenue = 0;
            }
            let mut revenue = 0u64;
            let mut purchase = false;
            for e in &events[i..j] {
                if e.price > 0 {
                    purchase = true;
                    revenue += u64::from(e.price);
                    let slot = &mut self.cat_sum[e.cat as usize];
                    if *slot == 0 {
                        self.touched.push(e.cat);
                    }
                    *slot += u64::from(e.price);
                }
            }
            day_sessions += 1;
            day_events += (j - i) as i64;
            day_revenue += revenue as i64;

            // Credit: the latest click at or before the start, within 7 days.
            while k < clicks.len() && clicks[k].ts <= start {
                k += 1;
            }
            let campaign = if k > 0 && start - clicks[k - 1].ts <= CLICK_REACH {
                clicks[k - 1].campaign
            } else {
                ctx.organic
            };
            let counters = self.camp.slot(campaign, day);
            counters[0] += 1;
            counters[1] += u64::from(purchase);
            counters[2] += revenue;

            self.sessions.push(Session { start, day });
            i = j;
        }
        if day_now != i32::MIN {
            self.emit_day(person, day_now, day_sessions, day_events, day_revenue, ctx);
        }
        self.sessions_total += self.sessions.len() as u64;
        if !clicks.is_empty() {
            self.assists(clicks);
        }
    }

    /// Each campaign assists the sessions whose start lies in the union of its clicks'
    /// `[ts, ts + 7 days]` intervals, once per session.
    fn assists(&mut self, clicks: &[ClickRec]) {
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

    #[inline]
    fn count_assists(&mut self, campaign: u32, a: i64, b: i64) {
        let lo = self.sessions.partition_point(|s| s.start < a);
        let hi = self.sessions.partition_point(|s| s.start <= b);
        let mut i = lo;
        while i < hi {
            let day = self.sessions[i].day;
            let end = i + self.sessions[i..hi].partition_point(|s| s.day <= day);
            self.camp.slot(campaign, day)[3] += (end - i) as u64;
            i = end;
        }
    }

    fn emit_day(&mut self, person: Id, day: i32, sessions: i64, events: i64, revenue: i64, ctx: &SessionCtx) {
        let mut top = NO_CATEGORY;
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
        let u = &mut self.user;
        ctx.ids.append(person, &mut u.user_id.data);
        u.user_id.finish_row();
        u.day.push(day);
        u.sessions.push(sessions);
        u.events.push(events);
        u.revenue_cents.push(revenue);
        u.top_category.push(top);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kept(person: u64, ts: i64, price: u32, cat: u32) -> Kept {
        Kept { person, ts, price, cat }
    }

    #[test]
    fn assists_match_a_brute_force_count() {
        let ids = Interner::new();
        let ctx = SessionCtx { ids: &ids, organic: 3 };
        let p = ids.id(b"u1");
        let day0 = 20_000i64;
        let base = day0 * DAY;
        let mut s = Sessionizer::new(1, CampaignAgg::new(4, day0, 30));
        // 100 sessions two hours apart from day 8; clicks of three campaigns spread over 10 days.
        let mut events = Vec::new();
        let mut clicks = Vec::new();
        for n in 0..100i64 {
            events.push(kept(p, base + 8 * DAY + n * 2 * crate::ts::HOUR, 0, 0));
        }
        for m in 0..300i64 {
            let ts = base + m * 47 * MINUTE + (m % 7);
            let cid = ids.id(format!("k{m}").as_bytes());
            clicks.push(ClickRec { person: p, ts, cid, campaign: (m % 3) as u32 });
        }
        let starts: Vec<i64> = events.iter().map(|e| e.ts).collect();
        let mut expected = [0u64; 3];
        for &start in &starts {
            for c in 0..3u32 {
                let any = clicks.iter().any(|k| k.campaign == c && k.ts <= start && start - k.ts <= CLICK_REACH);
                expected[c as usize] += u64::from(any);
            }
        }
        s.partition(&mut events, &mut clicks, &ctx);
        let rows = s.camp.rows();
        for c in 0..3u32 {
            let got: u64 = rows.iter().filter(|r| r.0 == c).map(|r| r.2[3]).sum();
            assert_eq!(got, expected[c as usize], "campaign {c}");
        }
        assert_eq!(rows.iter().map(|r| r.2[0]).sum::<u64>(), 100);
    }
}
