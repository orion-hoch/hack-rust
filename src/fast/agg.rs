//! campaign_daily counters of one fast worker, laid out for the cache.
//!
//! A cell is (day of the window, campaign), day-major. Credited sessions and purchasing sessions
//! share one u64 (two u32 halves), revenue is a u64 array, assists a u32 array: 8 + 8 + 4 bytes a
//! cell in three arrays (2,001 campaigns x 15 days: about 120 to 240 KB each) instead of one
//! 32-byte cell, so the per-session updates stay in L2. Counts per thread are below 2^32 (at most
//! the number of sessions).

use crate::sessions::CampaignAgg;

pub struct FastAgg {
    campaigns: usize,
    span: usize,
    day0: i64,
    cred: Vec<u64>,
    rev: Vec<u64>,
    asst: Vec<u32>,
}

impl FastAgg {
    pub fn new(campaigns: usize, day0: i64, span: usize) -> Self {
        let cells = campaigns * span;
        FastAgg { campaigns, span, day0, cred: vec![0; cells], rev: vec![0; cells], asst: vec![0; cells] }
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
        self.cred[c] += 1 | (u64::from(purchase) << 32);
        self.rev[c] += revenue;
    }

    /// `n` sessions assisted.
    #[inline(always)]
    pub fn assist(&mut self, campaign: u32, off: usize, n: u32) {
        self.asst[off * self.campaigns + campaign as usize] += n;
    }

    pub fn merge(&mut self, other: &FastAgg) {
        for (a, b) in self.cred.iter_mut().zip(&other.cred) {
            *a += b;
        }
        for (a, b) in self.rev.iter_mut().zip(&other.rev) {
            *a += b;
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
                let (cr, rv, a) = (self.cred[c], self.rev[c], self.asst[c]);
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
