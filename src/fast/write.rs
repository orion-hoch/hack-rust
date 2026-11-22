//! Output through pqw: fast rows (one name per person, as runs), pool rows (v1 columns) and the
//! campaign table.

use std::io;
use std::path::Path;

use parselab::Ctx;
use pqw::{Col, Data, Ids, Options, Strings};

use super::scan::Rows;
use crate::output::{NO_CATEGORY, UserDaily as V1UserDaily};
use crate::sessions::CampaignAgg;

pub fn user_rows(path: &Path, rows: &Rows, categories: &[&[u8]]) -> io::Result<()> {
    pqw::UserDaily {
        user_id: Ids::Runs { values: Strings::new(&rows.names, &rows.name_offsets), runs: &rows.runs },
        day: &rows.day,
        sessions: &rows.sessions,
        events: &rows.events,
        revenue_cents: &rows.revenue,
        top_category: &rows.top,
        categories,
    }
    .write_file(path)?;
    Ok(())
}

/// Rows from v1's sessionizer (the pool), converted to pqw's shapes.
pub fn v1_user_rows(path: &Path, u: &V1UserDaily, categories: &[&[u8]]) -> io::Result<()> {
    let offsets: Vec<u32> = u.user_id.offsets.iter().map(|&o| o as u32).collect();
    let sessions: Vec<u32> = u.sessions.iter().map(|&v| v as u32).collect();
    let events: Vec<u32> = u.events.iter().map(|&v| v as u32).collect();
    let top: Vec<u8> = u.top_category.iter().map(|&c| if c == NO_CATEGORY { pqw::NULL8 } else { c as u8 }).collect();
    pqw::UserDaily {
        user_id: Ids::Rows(Strings::new(&u.user_id.data, &offsets)),
        day: &u.day,
        sessions: &sessions,
        events: &events,
        revenue_cents: &u.revenue_cents,
        top_category: &top,
        categories,
    }
    .write_file(path)?;
    Ok(())
}

pub fn campaigns(path: &Path, agg: &CampaignAgg, ctx: &Ctx) -> io::Result<()> {
    let organic: &[u8] = b"organic";
    let (mut ids, mut id_off) = (Vec::new(), vec![0u32]);
    let (mut ch, mut ch_off) = (Vec::new(), vec![0u32]);
    let (mut day, mut s, mut ps, mut rev, mut asst) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (campaign, d, c) in agg.rows() {
        let i = campaign as usize;
        ids.extend_from_slice(ctx.campaigns.ids.get(i).map(|v| v.as_slice()).unwrap_or(organic));
        id_off.push(ids.len() as u32);
        ch.extend_from_slice(ctx.campaigns.channels.get(i).map(|v| v.as_slice()).unwrap_or(organic));
        ch_off.push(ch.len() as u32);
        day.push(d);
        s.push(c[0] as i64);
        ps.push(c[1] as i64);
        rev.push(c[2] as i64);
        asst.push(c[3] as i64);
    }
    let cols = [
        Col::new("campaign_id", Data::Str(Strings::new(&ids, &id_off))),
        Col::new("channel", Data::Str(Strings::new(&ch, &ch_off))),
        Col::new("day", Data::Date(&day)),
        Col::new("sessions", Data::I64(&s)),
        Col::new("purchasing_sessions", Data::I64(&ps)),
        Col::new("revenue_cents", Data::I64(&rev)),
        Col::new("assisted_sessions", Data::I64(&asst)),
    ];
    pqw::write_file(path, &cols, &Options::default())?;
    Ok(())
}
