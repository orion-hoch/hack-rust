//! Stage 7 interface: the two output tables as plain columns. A Parquet writer only sees these
//! structs, so any writer (the `parquet` crate today, a hand-rolled one later) can be swapped in
//! behind `write_user_daily` / `write_campaign_daily` in `pq.rs`.

/// A string column: row `i` is `data[offsets[i]..offsets[i + 1]]`.
#[derive(Default)]
pub struct StrColumn {
    pub data: Vec<u8>,
    pub offsets: Vec<usize>,
}

impl StrColumn {
    pub fn new() -> Self {
        StrColumn { data: Vec::new(), offsets: vec![0] }
    }

    pub fn push(&mut self, s: &[u8]) {
        self.data.extend_from_slice(s);
        self.offsets.push(self.data.len());
    }

    /// Call after appending a row's bytes to `data` directly.
    pub fn finish_row(&mut self) {
        self.offsets.push(self.data.len());
    }

    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }
}

/// `top_category` value meaning null.
pub const NO_CATEGORY: u32 = u32::MAX;

/// user_daily rows. `top_category[i]` indexes `categories` (NO_CATEGORY = null).
pub struct UserDaily {
    pub user_id: StrColumn,
    /// Days since 1970-01-01 (Parquet DATE).
    pub day: Vec<i32>,
    pub sessions: Vec<i64>,
    pub events: Vec<i64>,
    pub revenue_cents: Vec<i64>,
    pub top_category: Vec<u32>,
}

impl Default for UserDaily {
    fn default() -> Self {
        Self::new()
    }
}

impl UserDaily {
    pub fn new() -> Self {
        UserDaily {
            user_id: StrColumn::new(),
            day: Vec::new(),
            sessions: Vec::new(),
            events: Vec::new(),
            revenue_cents: Vec::new(),
            top_category: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.day.len()
    }
}

/// campaign_daily rows.
pub struct CampaignDaily {
    pub campaign_id: StrColumn,
    pub channel: StrColumn,
    pub day: Vec<i32>,
    pub sessions: Vec<i64>,
    pub purchasing_sessions: Vec<i64>,
    pub revenue_cents: Vec<i64>,
    pub assisted_sessions: Vec<i64>,
}

impl Default for CampaignDaily {
    fn default() -> Self {
        Self::new()
    }
}

impl CampaignDaily {
    pub fn new() -> Self {
        CampaignDaily {
            campaign_id: StrColumn::new(),
            channel: StrColumn::new(),
            day: Vec::new(),
            sessions: Vec::new(),
            purchasing_sessions: Vec::new(),
            revenue_cents: Vec::new(),
            assisted_sessions: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.day.len()
    }
}
