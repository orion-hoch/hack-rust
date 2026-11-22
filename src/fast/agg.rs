//! campaign_daily counters of one fast worker, laid out for the cache.
//!
//! A cell is (day of the window, campaign), day-major. A credit touches one 16-byte pair (credited
//! and purchasing sessions as two u32 halves of one u64, then revenue), so one cache line per
//! credited session; assists are a separate u32 array. 2,001 campaigns x 15 days: 480 KB of pairs
//! and 120 KB of assists. Counts per thread are below 2^32 (at most the number of sessions).

use crate::sessions::CampaignAgg;

pub struct FastAgg {
    campaigns: usize,
    span: usize,
    day0: i64,
    /// [sessions | purchasing << 32, revenue] per cell.
    cred: Vec<[u64; 2]>,
    asst: Vec<u32>,
}

impl FastAgg {
    pub fn new(campaigns: usize, day0: i64, span: usize) -> Self {
        let cells = campaigns * span;
        FastAgg { campaigns, span, day0, cred: vec![[0, 0]; cells], asst: vec![0; cells] }
    }

    /// The cells, for sortlab's walk (same layout: day offset * campaigns + campaign).
    pub fn cells_mut(&mut self) -> (&mut [[u64; 2]], &mut [u32]) {
        (&mut self.cred, &mut self.asst)
    }

    /// Day offset of an absolute day number.
    #[inline(always)]
    pub fn offset(&self, day: i32) -> usize {
        (i64::from(day) - self.day0) as usize
    }

    /// One credited session.
    #[inline(always)]
    pub fn credit(&mut self, campaign: u32, off: usize, purchase: bool, revenue: u64) {
        let c = &mut self.cred[off * self.campaigns + campaign as usize];
        c[0] += 1 | (u64::from(purchase) << 32);
        c[1] += revenue;
    }

    /// `n` sessions assisted.
    #[inline(always)]
    pub fn assist(&mut self, campaign: u32, off: usize, n: u32) {
        self.asst[off * self.campaigns + campaign as usize] += n;
    }

    pub fn merge(&mut self, other: &FastAgg) {
        for (a, b) in self.cred.iter_mut().zip(&other.cred) {
            a[0] += b[0];
            a[1] += b[1];
        }
        for (a, b) in self.asst.iter_mut().zip(&other.asst) {
            *a += b;
        }
    }

    /// Adds these counters into v1's aggregate (the pool's), which writes the table.
    pub fn add_into(&self, v1: &mut CampaignAgg) {
        for off in 0..self.span {
            let day = (self.day0 + off as i64) as i32;
            for campaign in 0..self.campaigns {
                let c = off * self.campaigns + campaign;
                let ([cr, rv], a) = (self.cred[c], self.asst[c]);
                if cr == 0 && a == 0 {
                    continue;
                }
                let slot = v1.slot(campaign as u32, day);
                slot[0] += cr & 0xffff_ffff;
                slot[1] += cr >> 32;
                slot[2] += rv;
                slot[3] += u64::from(a);
            }
        }
    }
}
