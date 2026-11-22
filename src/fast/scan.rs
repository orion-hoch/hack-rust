//! Shared by the engines after parsing: session constants and the user_daily rows in pqw's shape.

use crate::ts::{DAY, MINUTE};

pub const SESSION_GAP: i64 = 30 * MINUTE;
pub const CLICK_REACH: i64 = 7 * DAY;

/// user_daily rows for pqw: one name per person, with its row count.
#[derive(Default)]
pub struct Rows {
    pub names: Vec<u8>,
    pub name_offsets: Vec<u32>,
    pub runs: Vec<u32>,
    pub day: Vec<i32>,
    pub sessions: Vec<u32>,
    pub events: Vec<u32>,
    pub revenue: Vec<i64>,
    pub top: Vec<u8>,
}

impl Rows {
    pub fn new() -> Self {
        Rows { name_offsets: vec![0], ..Default::default() }
    }
    pub fn len(&self) -> usize {
        self.day.len()
    }
}
