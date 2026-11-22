//! P2 (owners: B for the adapter, C for the walk on blocks). Per home partition: group the person
//! blocks by person key, concatenate a person's events, then today's session kernel, walk and
//! rows (unchanged in what they compute); user_daily written by the workers as they go, then
//! campaign_daily. Names: unlinked = unmix of its own key; linked = the name key from I4.

use std::path::Path;

use parselab::{Dataset, Stream};

use super::links::Links;
use super::p1::Blocks;
use super::{Cx, Stage};
use crate::fast::agg::FastAgg;
use crate::fast::scan::Rows;
use crate::sessions::CampaignAgg;
use crate::ts::DAY;

/// Rows per user_daily file written during P2 (about 1 MB of output).
const ROWS_PER_FILE: usize = 1 << 16;


struct Worker<'a> {
    index: usize,
    files: u32,
    wrote: bool,
    write_err: Option<std::io::Error>,
    bad: bool,
    rows: Rows,
    agg: FastAgg,
    gs: GroupScratch,
    g: Grouped<'a>,
    sc: sortlab::Scratch,
    out: sortlab::Out,
    ws: sortlab::walk::WalkScratch,
    wrows: sortlab::walk::Rows,
}

fn flush(w: &mut Worker, dir: &Path, cats: &[&[u8]]) {
    let path = dir.join(format!("part-{:03}-{:05}.parquet", w.index, w.files));
    w.write_err = crate::fast::write::user_rows(&path, &w.rows, cats).err();
    w.files += 1;
    w.wrote = true;
    w.rows.clear();
}

/// P2 over every home partition; user_daily written by the workers as they go, then
/// campaign_daily from the merged counters.
pub fn run(cx: &Cx, blocks: &Blocks, links: &Links, out: &Path) -> Stage<()> {
    let ctx = cx.ctx;
    let day0 = ctx.window_start_ms.div_euclid(DAY);
    let span = ((ctx.window_end_ms - 1).div_euclid(DAY) - day0 + 1).max(1) as usize;
    let campaigns = ctx.campaigns.ids.len() + 1;
    let organic = ctx.campaigns.ids.len() as u32;
    let prm = sortlab::walk::Params {
        base: ctx.click_base_ms - ctx.window_start_ms,
        day_shift: ctx.window_start_ms.rem_euclid(DAY) as u32,
        days: span,
        campaigns,
        organic,
    };
    let user_dir = out.join("user_daily");
    let campaign_dir = out.join("campaign_daily");
    std::fs::create_dir_all(&user_dir).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&campaign_dir).map_err(|e| e.to_string())?;
    let cats: Vec<&[u8]> = ctx.products.categories.iter().map(|c| c.as_slice()).collect();
    let ds = cx.ds;
    let threads = cx.plan.threads;
    // Rows are flushed every ROWS_PER_FILE, so a worker holds at most that plus one partition's
    // rows (a partition has a few thousand); about one person per 3 rows, names of about 10 bytes.
    let keep = ROWS_PER_FILE + (cx.plan.est_events >> cx.plan.b2) / 4 + 4096;
    // campaign_daily inside the pool's finish step: each worker adds its cells as it finishes and
    // the last one writes the file (it was a serial step after the pool: 3 ms at 60M, the other
    // workers idle). run_fin has min(threads, tasks) participants, each finishing once.
    let parts = 1usize << cx.plan.b2;
    let participants = threads.clamp(1, parts.max(1));
    let camp = std::sync::Mutex::new(CampaignAgg::new(campaigns, day0, span));
    let finished = std::sync::atomic::AtomicUsize::new(0);
    let camp_err: std::sync::Mutex<Option<std::io::Error>> = std::sync::Mutex::new(None);
    let mut workers = crate::par::run_fin(
        threads,
        1usize << cx.plan.b2,
        |t| {
            let mut rows = Rows::new();
            rows.reserve(keep, keep / 2, keep / 2 * 12);
            Worker {
                index: t,
                files: 0,
                wrote: false,
                write_err: None,
                bad: false,
                rows,
                agg: FastAgg::new(campaigns, day0, span),
                gs: GroupScratch::default(),
                g: Grouped::default(),
                sc: sortlab::Scratch::new(),
                out: sortlab::Out::default(),
                ws: sortlab::walk::WalkScratch::new(),
                wrows: sortlab::walk::Rows::default(),
            }
        },
        |w, q| {
            if w.bad {
                return;
            }
            let chunks = blocks.chunks(q);
            if group(&chunks, &mut w.gs, &mut w.g).is_none() {
                w.bad = true;
                return;
            }
            let pb = sortlab::PersonBlocks { ev: &w.g.ev, ev_at: &w.g.ev_at, ck: &w.g.ck, ck_at: &w.g.ck_at };
            if sortlab::sessions_blocks(&pb, sortlab::Kernel::Net, &mut w.sc, &mut w.out).is_err() {
                w.bad = true;
                return;
            }
            let mut tie = |a: u64, b: u64| click_id(ds, parselab::Locator(a)).cmp(click_id(ds, parselab::Locator(b)));
            let (hot, buy) = w.agg.cells_mut();
            if sortlab::walk::walk_blocks(&pb, &prm, &mut tie, &mut w.ws, &w.out, hot, buy, &mut w.wrows).is_err() {
                w.bad = true;
                return;
            }
            // Rows with the person names (k -> pk -> name).
            let (wr, r) = (&w.wrows, &mut w.rows);
            let mut r0 = 0u32;
            for (k, &r1) in wr.row_end.iter().enumerate() {
                if r1 > r0 {
                    name_of(q, w.g.pks[k], links, blocks, &mut r.names);
                    r.name_offsets.push(r.names.len() as u32);
                    r.runs.push(r1 - r0);
                }
                r0 = r1;
            }
            let d0 = day0 as i32;
            r.day.extend(wr.day.iter().map(|&d| d0 + d as i32));
            r.sessions.extend_from_slice(&wr.sessions);
            r.events.extend_from_slice(&wr.events);
            r.revenue.extend(wr.revenue.iter().map(|&x| x as i64));
            r.top.extend_from_slice(&wr.top);
            if w.write_err.is_none() && w.rows.len() >= ROWS_PER_FILE {
                flush(w, &user_dir, &cats);
            }
        },
        |w, _| {
            if !w.bad && w.write_err.is_none() && w.rows.len() > 0 {
                flush(w, &user_dir, &cats);
            }
            {
                let mut c = camp.lock().unwrap_or_else(|e| e.into_inner());
                if !w.bad {
                    w.agg.add_into(&mut c);
                }
                if finished.fetch_add(1, std::sync::atomic::Ordering::AcqRel) + 1 == participants {
                    // the last to finish: every worker's cells are in (a bad run declines anyway)
                    if let Err(e) = crate::fast::write::campaigns(&campaign_dir.join("part-00000.parquet"), &c, ctx) {
                        *camp_err.lock().unwrap_or_else(|e| e.into_inner()) = Some(e);
                    }
                }
            }
        },
    );
    if workers.iter().any(|w| w.bad) {
        return Err("a person partition did not fit the session kernel".into());
    }
    let mut wrote = false;
    for w in workers.iter_mut() {
        if let Some(e) = w.write_err.take() {
            return Err(e.to_string());
        }
        wrote |= w.wrote;
    }
    if let Some(e) = camp_err.into_inner().unwrap_or_else(|e| e.into_inner()) {
        return Err(e.to_string());
    }
    if finished.into_inner() != participants {
        return Err("a P2 worker did not finish".into());
    }
    if !wrote {
        crate::fast::write::user_rows(&user_dir.join("part-00000.parquet"), &Rows::new(), &cats).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// The click_id of the click line at `loc` (its first field, unquoted).
fn click_id(ds: &Dataset, loc: parselab::Locator) -> &[u8] {
    let data = ds.stream(Stream::Clicks)[loc.file() as usize].bytes();
    let rest = &data[loc.offset() as usize..];
    let end = rest.iter().position(|&c| c == b',' || c == b'\n').unwrap_or(rest.len());
    let f = &rest[..end];
    parselab::csv::unquote(f).unwrap_or(f)
}

/// The id text of a SHORT key mu (a mixed user key): 'u' and the base-36 digits of its value.
pub fn short_name(mu: u64, out: &mut Vec<u8>) {
    crate::v3::ids::short_bytes(b'u', super::keys::unmix40(mu), out);
}

/// The user_daily name of person key pk of home partition q (RECORDS-V4 sections 6-7): SHORT pk
/// = 'u' + base36(unmix40(pk)); a links-wide pk (bit 63, bit 60 clear) through links.wide_key;
/// a P1-local wide pk (bit 60) through blocks.wide(q). LONG = 'u' + 32 hex digits; FOREIGN =
/// the interned bytes.
pub fn name_of(q: usize, pk: u64, links: &Links, blocks: &Blocks, out: &mut Vec<u8>) {
    if pk >> 63 == 0 {
        return short_name(pk, out);
    }
    let kind = (pk >> 61) & 3;
    let (a, b) = if pk >> 60 & 1 != 0 {
        let w = blocks.wide(q)[(pk & 0xFFFF_FFFF) as usize];
        (w[0], w[1])
    } else {
        let (_, a, b) = links.wide_key(pk);
        (a, b)
    };
    if kind == 2 {
        out.extend_from_slice(super::keys::bytes((a & !(1 << 63)) as u32));
    } else {
        crate::v3::ids::long_bytes(b'u', a, b, out);
    }
}


/// A person's blocks of one partition, grouped by person key (RECORDS-V4.md section 8), as
/// references into the chunks P1 wrote: nothing is copied.
#[derive(Default)]
pub struct Grouped<'a> {
    /// Person key of each local person k (0..np).
    pub pks: Vec<u64>,
    /// Person blocks of person k: ev[ev_at[k]..ev_at[k + 1]].
    pub ev: Vec<&'a [u64]>,
    pub ev_at: Vec<u32>,
    /// Click blocks of person k: ck[ck_at[k]..ck_at[k + 1]].
    pub ck: Vec<&'a [u64]>,
    pub ck_at: Vec<u32>,
}

const CLICK_MARK: u64 = 1 << 62;
const N24: u64 = (1 << 24) - 1;

/// Length in words of the block at the start of `w`, whether it is a click block, and its person
/// key; None when the header does not fit in `w` (a malformed stream: the run falls back).
#[inline(always)]
fn block_at(w: &[u64]) -> Option<(usize, bool, u64)> {
    let (&h0, &pk) = (w.first()?, w.get(1)?);
    let click = h0 & CLICK_MARK != 0;
    let len = if click {
        2 + 2 * (h0 & N24) as usize
    } else {
        let (n_ev, n_purch) = ((h0 & N24) as usize, ((h0 >> 24) & N24) as usize);
        2 + n_ev.div_ceil(2) + n_purch
    };
    (len <= w.len()).then_some((len, click, pk))
}

/// Reusable state of `group` (kept between partitions; nothing is allocated once warm).
#[derive(Default)]
pub struct GroupScratch {
    keys: Vec<u64>,
    vals: Vec<u32>,
    /// Slot of person k in this partition (only those are reset).
    used: Vec<u32>,
    /// Person of each person block and of each click block, in input order.
    kev: Vec<u32>,
    kck: Vec<u32>,
    /// The block lists in input order, for the placement by person.
    ev_in: Vec<*const [u64]>,
    ck_in: Vec<*const [u64]>,
    /// Per person k: [person blocks, click blocks] counted by the pass, then their next places.
    cur: Vec<[u32; 2]>,
}

// SAFETY: the raw block pointers in `ev_in` / `ck_in` are only used inside `group`, while the
// chunks they point into are borrowed.
unsafe impl Send for GroupScratch {}

const NO_KEY: u64 = u64::MAX;

/// Doubles the person map and re-inserts `pks` (person k at its key); returns the new mask.
#[cold]
#[inline(never)]
fn grow(gs: &mut GroupScratch, pks: &[u64]) -> usize {
    let n = gs.keys.len() * 2;
    gs.keys.clear();
    gs.keys.resize(n, NO_KEY);
    gs.vals.clear();
    gs.vals.resize(n, 0);
    let mask = n - 1;
    for (k, &pk) in pks.iter().enumerate() {
        let mut h = (pk ^ (pk >> 29)) as usize & mask;
        while gs.keys[h] != NO_KEY {
            h = (h + 1) & mask;
        }
        gs.keys[h] = pk;
        gs.vals[h] = k as u32;
        // SAFETY: `used` has room for every person of the partition (reserved by `group`).
        unsafe { *gs.used.as_mut_ptr().add(k) = h as u32 };
    }
    mask
}

/// Groups the blocks of `chunks` (each holding whole blocks, ended by its last word or a zero
/// header word) by person key. None when a block does not fit its chunk.
///
/// One pass finds each block's person (opened at its first block), counts the person's blocks
/// and notes the block reference with its person in input order; a prefix sum over the persons
/// and one placement pass finish the counting sort. (On generator data a third of the blocks
/// belong to a person opened earlier, in every partition: the blocks never arrive sorted.)
#[inline(never)]
pub fn group<'a>(chunks: &[&'a [u64]], gs: &mut GroupScratch, g: &mut Grouped<'a>) -> Option<()> {
    // Sized for about one block per 8 words (generator data: one per 13), so the map stays in L1
    // or L2; it doubles if it passes half full.
    let words: usize = chunks.iter().map(|c| c.len()).sum();
    let want = (words / 4 + 64).next_power_of_two();
    if gs.keys.len() != want {
        gs.keys.clear();
        gs.keys.resize(want, NO_KEY);
        gs.vals.clear();
        gs.vals.resize(want, 0);
    }
    // A block has at least 2 words: at most words / 2 blocks and persons. Reserved once (the
    // capacity stays), so the pass writes through raw pointers with no checks.
    let cap = words / 2 + 2;
    g.pks.clear();
    g.pks.reserve(cap);
    gs.used.clear();
    gs.used.reserve(cap);
    gs.cur.clear();
    gs.cur.reserve(cap);
    gs.kev.clear();
    gs.kev.reserve(cap);
    gs.kck.clear();
    gs.kck.reserve(cap);
    gs.ev_in.clear();
    gs.ev_in.reserve(cap);
    gs.ck_in.clear();
    gs.ck_in.reserve(cap);
    let mut mask = gs.keys.len() - 1;
    // The number of persons at which the map is first past half full (it then doubles).
    let mut full = (mask + 1) / 2 + 1;
    let (mut keys, mut vals) = (gs.keys.as_mut_ptr(), gs.vals.as_mut_ptr());
    let used = gs.used.as_mut_ptr();
    let cur = gs.cur.as_mut_ptr();
    let (kev, kck) = (gs.kev.as_mut_ptr(), gs.kck.as_mut_ptr());
    let (ev_in, ck_in) = (gs.ev_in.as_mut_ptr(), gs.ck_in.as_mut_ptr());
    let pks = g.pks.as_mut_ptr();
    let (mut np, mut nev, mut nck) = (0usize, 0usize, 0usize);
    let mut ok = true;
    // A click block usually directly follows its person block: the same key as the last block
    // needs no lookup.
    let (mut last_pk, mut last_k) = (NO_KEY, 0usize);
    // SAFETY: every read is below the chunk's length (`rem >= 2` for the two header words, the
    // block's length checked against the rest before it is referenced); every write is below the
    // capacity reserved above (at most words / 2 blocks, so at most that many persons); map slots
    // are masked.
    unsafe {
        'all: for c in chunks {
            // The block at `p`, with `rem` words left in the chunk from it.
            let (mut p, mut rem) = (c.as_ptr(), c.len());
            while rem >= 2 {
                let h0 = *p;
                if h0 == 0 {
                    break;
                }
                let pk = *p.add(1);
                let k = if pk == last_pk {
                    last_k
                } else {
                    // pk is a mixed key (uniform low bits) or a wide index: its low bits hash it.
                    let mut h = (pk ^ (pk >> 29)) as usize & mask;
                    loop {
                        let key = *keys.add(h);
                        if key == NO_KEY {
                            if pk == NO_KEY {
                                ok = false;
                                break 'all;
                            }
                            let k = np;
                            *keys.add(h) = pk;
                            *vals.add(h) = k as u32;
                            *used.add(k) = h as u32;
                            *pks.add(k) = pk;
                            *cur.add(k) = [0, 0];
                            np += 1;
                            if np == full {
                                mask = grow(gs, std::slice::from_raw_parts(pks, np));
                                full = (mask + 1) / 2 + 1;
                                keys = gs.keys.as_mut_ptr();
                                vals = gs.vals.as_mut_ptr();
                            }
                            break k;
                        }
                        if key == pk {
                            break *vals.add(h) as usize;
                        }
                        h = (h + 1) & mask;
                    }
                };
                last_pk = pk;
                last_k = k;
                let lo = (h0 & N24) as usize;
                let len;
                if h0 & CLICK_MARK != 0 {
                    len = 2 + 2 * lo;
                    if len > rem {
                        ok = false;
                        break 'all;
                    }
                    *ck_in.add(nck) = std::ptr::slice_from_raw_parts(p, len);
                    *kck.add(nck) = k as u32;
                    (*cur.add(k))[1] += 1;
                    nck += 1;
                } else {
                    len = 2 + ((lo + 1) >> 1) + ((h0 >> 24) & N24) as usize;
                    if len > rem {
                        ok = false;
                        break 'all;
                    }
                    *ev_in.add(nev) = std::ptr::slice_from_raw_parts(p, len);
                    *kev.add(nev) = k as u32;
                    (*cur.add(k))[0] += 1;
                    nev += 1;
                }
                p = p.add(len);
                rem -= len;
            }
            // a header word with no room for its key
            if rem == 1 && *p != 0 {
                ok = false;
                break;
            }
        }
        for i in 0..np {
            *keys.add(*used.add(i) as usize) = NO_KEY;
        }
        if !ok {
            return None;
        }
        // Counts to offsets and next places.
        g.ev_at.clear();
        g.ev_at.reserve(np + 1);
        g.ck_at.clear();
        g.ck_at.reserve(np + 1);
        let (ev_at, ck_at) = (g.ev_at.as_mut_ptr(), g.ck_at.as_mut_ptr());
        let (mut a, mut b) = (0u32, 0u32);
        for k in 0..np {
            let [ca, cb] = *cur.add(k);
            *ev_at.add(k) = a;
            *ck_at.add(k) = b;
            *cur.add(k) = [a, b];
            a += ca;
            b += cb;
        }
        *ev_at.add(np) = a;
        *ck_at.add(np) = b;
        // A key of NO_KEY is refused where a person is opened; blocks carrying it before the first
        // person passed as "the same as the last key" and were counted for nobody.
        if a as usize != nev || b as usize != nck {
            return None;
        }
        g.pks.set_len(np);
        g.ev_at.set_len(np + 1);
        g.ck_at.set_len(np + 1);
        g.ev.clear();
        g.ev.reserve(nev);
        g.ck.clear();
        g.ck.reserve(nck);
        let (ev, ck) = (g.ev.as_mut_ptr(), g.ck.as_mut_ptr());
        for i in 0..nev {
            let c = &mut (*cur.add(*kev.add(i) as usize))[0];
            *ev.add(*c as usize) = &*(*ev_in.add(i));
            *c += 1;
        }
        for i in 0..nck {
            let c = &mut (*cur.add(*kck.add(i) as usize))[1];
            *ck.add(*c as usize) = &*(*ck_in.add(i));
            *c += 1;
        }
        g.ev.set_len(nev);
        g.ck.set_len(nck);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn person(pk: u64, ts: &[u32], pays: &[u64]) -> Vec<u64> {
        let mut v = vec![ts.len() as u64 | (pays.len() as u64) << 24, pk];
        for p in ts.chunks(2) {
            v.push(u64::from(p[0]) | u64::from(*p.get(1).unwrap_or(&u32::MAX)) << 32);
        }
        v.extend_from_slice(pays);
        v
    }

    fn clicks(pk: u64, n: usize) -> Vec<u64> {
        let mut v = vec![n as u64 | CLICK_MARK, pk];
        for i in 0..n {
            v.push(7 << 52 | i as u64);
            v.push(1000 + i as u64);
        }
        v
    }

    #[test]
    fn groups_blocks_by_person_without_copies() {
        let mut c0 = person(5, &[2, 4, 6], &[9]);
        c0.extend(clicks(7, 2));
        c0.extend(person(7, &[8], &[]));
        let mut c1 = person(5, &[10, 12], &[]);
        c1.extend(clicks(5, 1));
        let chunks: Vec<&[u64]> = vec![&c0, &c1];
        let (mut gs, mut g) = (GroupScratch::default(), Grouped::default());
        group(&chunks, &mut gs, &mut g).unwrap();
        assert_eq!(g.pks, vec![5, 7]);
        assert_eq!(g.ev_at, vec![0, 2, 3]);
        assert_eq!(g.ck_at, vec![0, 1, 2]);
        assert_eq!(g.ev[0][1], 5);
        assert_eq!(g.ev[1].len(), 2 + 1);
        assert_eq!(g.ev[2][1], 7);
        assert_eq!(g.ck[0][1], 5);
        assert_eq!(g.ck[1][0] & N24, 2);
        // a zero header word ends a chunk
        let mut c2 = person(9, &[1], &[]);
        c2.extend([0u64, 77, 77]);
        group(&[&c2], &mut gs, &mut g).unwrap();
        assert_eq!(g.pks, vec![9]);
        // a block longer than its chunk is refused
        let bad = vec![10u64, 3];
        assert!(group(&[&bad], &mut gs, &mut g).is_none());
    }
}

