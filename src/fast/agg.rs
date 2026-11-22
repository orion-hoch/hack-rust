//! campaign_daily counters of one fast worker, laid out for the cache.
//!
//! A cell is (day of the window, campaign), day-major. Every session touches only the 8-byte hot
//! pair of its cells (credited sessions, assisted sessions as two u32): 2,001 campaigns x 15 days
//! is 240 KB, which stays in a 512 KB L2 beside the rest of the stage (the old 16-byte credit pair
//! plus a separate assist array was 600 KB). The purchasing count and revenue (16 bytes) are
//! touched only by sessions with a purchase. Counts per thread are below 2^32 (at most the number
//! of sessions).

use crate::sessions::CampaignAgg;

pub struct FastAgg {
    campaigns: usize,
    span: usize,
    day0: i64,
    /// [credited sessions, assisted sessions] per cell.
    hot: Vec<[u32; 2]>,
    /// [purchasing sessions, revenue] per cell.
    buy: Vec<[u64; 2]>,
}

impl FastAgg {
    pub fn new(campaigns: usize, day0: i64, span: usize) -> Self {
        let cells = campaigns * span;
        FastAgg { campaigns, span, day0, hot: vec![[0, 0]; cells], buy: vec![[0, 0]; cells] }
    }

    /// The cells, for sortlab's walk (same layout: day offset * campaigns + campaign).
    pub fn cells_mut(&mut self) -> (&mut [[u32; 2]], &mut [[u64; 2]]) {
        (&mut self.hot, &mut self.buy)
    }

    /// Day offset of an absolute day number.
    #[inline(always)]
    pub fn offset(&self, day: i32) -> usize {
        (i64::from(day) - self.day0) as usize
    }

    /// One credited session.
    #[inline(always)]
    pub fn credit(&mut self, campaign: u32, off: usize, purchase: bool, revenue: u64) {
        let c = off * self.campaigns + campaign as usize;
        self.hot[c][0] += 1;
        if purchase {
            let b = &mut self.buy[c];
            b[0] += 1;
            b[1] += revenue;
        }
    }

    /// `n` sessions assisted.
    #[inline(always)]
    pub fn assist(&mut self, campaign: u32, off: usize, n: u32) {
        self.hot[off * self.campaigns + campaign as usize][1] += n;
    }

    pub fn merge(&mut self, other: &FastAgg) {
        for (a, b) in self.hot.iter_mut().zip(&other.hot) {
            a[0] += b[0];
            a[1] += b[1];
        }
        for (a, b) in self.buy.iter_mut().zip(&other.buy) {
            a[0] += b[0];
            a[1] += b[1];
        }
    }

    /// Adds these counters into v1's aggregate (the pool's), which writes the table.
    pub fn add_into(&self, v1: &mut CampaignAgg) {
        for off in 0..self.span {
            let day = (self.day0 + off as i64) as i32;
            for campaign in 0..self.campaigns {
                let c = off * self.campaigns + campaign;
                let ([cr, a], [pu, rv]) = (self.hot[c], self.buy[c]);
                if cr == 0 && a == 0 {
                    continue;
                }
                let slot = v1.slot(campaign as u32, day);
                slot[0] += u64::from(cr);
                slot[1] += pu;
                slot[2] += rv;
                slot[3] += u64::from(a);
            }
        }
    }
}
