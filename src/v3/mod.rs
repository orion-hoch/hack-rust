//! The v3 engine (RECORDS.md version 3): opaque ID keys to user tokens, exact dedup per move-1
//! partition, persons by union-find on tokens, person partitions sorted by (person, ts) and scanned
//! into sessions. Every person is handled the same way. Any failed internal check aborts to v1.

pub mod dedup;
pub mod ids;
pub mod people;
pub mod persons;
pub mod sink;
pub mod tokens;

use std::io;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use parselab::{Dataset, MapMode, Stream};

use crate::fast::Outcome;
use crate::fast::agg::FastAgg;
use crate::fast::scan::Rows;
use crate::fast::sink::Bins;
use crate::sessions::CampaignAgg;
use crate::ts::DAY;

use self::dedup::{Ctx3, Move2, Tables};
use self::people::{PeopleCtx, Worker};
use self::persons::People;
use self::sink::{Special, V3Sink, part_of};
use self::tokens::Tokens;

/// A failed internal check: the run falls back to v1.
#[derive(Debug)]
pub struct Abort(pub String);

#[repr(C)]
#[derive(Default)]
struct Rusage {
    utime: [i64; 2],
    stime: [i64; 2],
    rest: [i64; 14],
}

unsafe extern "C" {
    fn getrusage(who: i32, usage: *mut Rusage) -> i32;
}

/// (system CPU ms, minor faults, major faults) of the whole process so far.
fn usage() -> (f64, i64, i64) {
    let mut r = Rusage::default();
    // SAFETY: `r` is a writable struct with the layout of Linux `struct rusage`.
    unsafe { getrusage(0, &mut r) };
    (r.stime[0] as f64 * 1e3 + r.stime[1] as f64 / 1e3, r.rest[4], r.rest[5])
}

struct Clock {
    on: bool,
    last: Instant,
    last_use: (f64, i64, i64),
}

impl Clock {
    fn mark(&mut self, what: &str) {
        let now = Instant::now();
        if self.on {
            let u = usage();
            eprintln!(
                "timing v3.{what:<12} {:9.2} ms   sys {:7.1} ms  minflt {:7}  majflt {}",
                (now - self.last).as_secs_f64() * 1e3,
                u.0 - self.last_use.0,
                u.1 - self.last_use.1,
                u.2 - self.last_use.2
            );
            self.last_use = u;
        }
        self.last = now;
    }
}

fn hidden_csv(dir: &Path) -> bool {
    ["events", "clicks", "links"].iter().any(|sub| {
        std::fs::read_dir(dir.join(sub))
            .map(|rd| rd.flatten().any(|e| {
                let n = e.file_name();
                let n = n.to_string_lossy();
                n.starts_with('.') && n.ends_with(".csv")
            }))
            .unwrap_or(false)
    })
}

pub fn try_run(data: &Path, out: &Path, threads: usize, timing: bool) -> io::Result<Outcome> {
    let t0 = Instant::now();
    let mut clock = Clock { on: timing, last: t0, last_use: if timing { usage() } else { (0.0, 0, 0) } };
    crate::par::init_pool(threads);
    if hidden_csv(data) {
        return Ok(Outcome::Fallback("a *.csv name starts with '.'".into()));
    }
    let ds = Dataset::open(data, MapMode::Lazy)?;
    let ctx = match parselab::prepare(&ds) {
        Ok(c) => c,
        Err(f) => return Ok(Outcome::Fallback(format!("{f:?}"))),
    };
    let ev_bytes = ds.bytes(Stream::Events) as usize;
    let est_events = ev_bytes / 70 + 1;
    // User IDs come from all three inputs: every link line can add two, a click or an event
    // line one. Generator data has about one distinct user per 11 event lines, and every linked
    // user is in a link line; this estimate is at least that from each input, and the tables
    // are sized to twice it, so generator data sits near a quarter full (fallback at 3/4).
    let link_lines = ds.bytes(Stream::Links) as usize / 18 + 1;
    let click_lines = ds.bytes(Stream::Clicks) as usize / 40 + 1;
    let est_users = 2 * link_lines + click_lines / 8 + est_events / 12 + 4096;
    let tok = Tokens::new(est_users);
    // Move-1 partitions of about 16K records: the dedup table (32K slots) stays in L2.
    let p1_bits = ((est_events / 16_000).max(1).next_power_of_two().trailing_zeros()).clamp(4, 14);
    clock.mark("open");

    // Parse A: links, then clicks.
    let mut chunks = parselab::par::plan(&ds, &[Stream::Links], 4 << 20);
    chunks.extend(parselab::par::plan(&ds, &[Stream::Clicks], 4 << 20));
    let per_thread = est_events / threads.max(1);
    let reference = std::env::var("ENGINE_PARSER").is_ok_and(|v| v == "ref");
    let fp = parselab::fast::Fast::new(&ctx);
    let isa = parselab::fast::Isa::detect();
    let parse = |c: &parselab::par::Chunk, s: &mut V3Sink| {
        let file = &ds.stream(c.stream)[c.file as usize];
        let bytes = file.bytes();
        if reference {
            match c.stream {
                Stream::Events => parselab::reference::events(&ctx, c.file, bytes, c.from, c.to, s),
                Stream::Clicks => parselab::reference::clicks(&ctx, c.file, bytes, c.from, c.to, s),
                Stream::Links => parselab::reference::links(&ctx, c.file, bytes, c.from, c.to, s),
            }
        } else {
            let kind = match c.stream {
                Stream::Events => parselab::fast::Kind::Events,
                Stream::Clicks => parselab::fast::Kind::Clicks,
                Stream::Links => parselab::fast::Kind::Links,
            };
            parselab::fast::parse(&fp, isa, kind, c.file, parselab::fast::Buf::plain(bytes), c.from, c.to, s);
        }
        // Release the parsed range: pages stay in the page cache, so a later re-read by locator
        // (rare) faults them back; keeping them mapped costs 70 ms at exit at practice size.
        file.advise(c.from, c.to, 4);
    };
    let sinks = crate::par::run(
        threads,
        chunks.len(),
        |_| V3Sink::new(&tok, &ds, &ctx, p1_bits, per_thread),
        |s, i| parse(&chunks[i], s),
    );
    clock.mark("parse.links+clicks");

    // Parse B: events; one worker builds persons meanwhile.
    let mut sinks = sinks;
    for s in sinks.iter_mut() {
        s.drain();
        s.edges.finish();
    }
    // Every NO_TOKEN is caught here, before persons is built from the edges.
    if tok.full.load(std::sync::atomic::Ordering::Relaxed) || sinks.iter().any(|s| s.bad) {
        return Ok(Outcome::Fallback("token table full or a link/click record did not fit".into()));
    }
    let edges: Vec<Bins<u64>> = sinks.iter_mut().map(|s| std::mem::replace(&mut s.edges, Bins::with_chunk(1, 2, 2))).collect();
    let slots: Vec<Mutex<Option<V3Sink>>> = sinks.into_iter().map(|s| Mutex::new(Some(s))).collect();
    let ev_chunks = parselab::par::plan(&ds, &[Stream::Events], 4 << 20);
    let people_slot: Mutex<Option<People>> = Mutex::new(None);
    let tasks = ev_chunks.len() + 1;
    let workers_n = slots.len();
    let mut sinks: Vec<V3Sink> = crate::par::run(
        threads,
        tasks,
        |t| {
            slots
                .get(t)
                .and_then(|m| m.lock().unwrap().take())
                .unwrap_or_else(|| V3Sink::new(&tok, &ds, &ctx, p1_bits, per_thread))
        },
        |s, i| {
            if i == 0 {
                let t = Instant::now();
                let refs: Vec<&[u64]> = edges.iter().flat_map(|b| b.all()).collect();
                *people_slot.lock().unwrap() = People::build(&tok, &refs);
                if timing {
                    eprintln!("timing v3.  persons(task) {:9.2} ms", t.elapsed().as_secs_f64() * 1e3);
                }
                return;
            }
            parse(&ev_chunks[i - 1], s);
        },
    );
    // A pool with fewer participants than sinks leaves some sinks untaken: keep them too.
    for m in slots.iter().take(workers_n) {
        if let Some(s) = m.lock().unwrap().take() {
            sinks.push(s);
        }
    }
    let Some(people) = people_slot.into_inner().unwrap_or(None) else {
        return Ok(Outcome::Fallback("persons could not be built (bad token)".into()));
    };
    clock.mark("parse.events");
    if tok.full.load(std::sync::atomic::Ordering::Relaxed) || sinks.iter().any(|s| s.bad) {
        return Ok(Outcome::Fallback("token table full or a record did not fit".into()));
    }
    for s in sinks.iter_mut() {
        s.finish();
    }

    // Specials by partition; FOREIGN event IDs apart.
    let parts = 1usize << p1_bits;
    let mut spec_by_part: Vec<Vec<Special>> = (0..parts).map(|_| Vec::new()).collect();
    let mut foreign_specials: Vec<Special> = Vec::new();
    for s in sinks.iter_mut() {
        for sp in s.special.drain(..) {
            match sp.key {
                ids::Key::Short(a) => spec_by_part[part_of(a, 0, p1_bits)].push(sp),
                ids::Key::Long(a, b) => spec_by_part[part_of(a, b, p1_bits)].push(sp),
                ids::Key::Foreign => foreign_specials.push(sp),
            }
        }
    }
    // Dedup and move 2.
    let max_short = (0..parts)
        .map(|p| sinks.iter().map(|s| s.short.slices(p).map(|c| c.len()).sum::<usize>()).sum::<usize>() + spec_by_part[p].len())
        .max()
        .unwrap_or(0);
    let max_long = (0..parts)
        .map(|p| sinks.iter().map(|s| s.long.slices(p).map(|c| c.len()).sum::<usize>()).sum::<usize>() + spec_by_part[p].len())
        .max()
        .unwrap_or(0);
    let est_surv = est_events;
    let p2_bits = ((est_surv / 24_000).max(1).next_power_of_two().trailing_zeros()).clamp(4, 13);
    let cx3 = Ctx3 { people: &people, ingest_base_ms: ctx.ingest_base_s * 1000 };
    let sinks_ref = &sinks;
    let spec_ref = &spec_by_part;
    let dd = crate::par::run(
        threads,
        parts,
        |_| (Tables::new((max_short + max_short / 2).next_power_of_two().max(16), (max_long + max_long / 2).next_power_of_two().max(16)), Move2::new(p2_bits, est_surv / threads.max(1)), 0usize, None::<Abort>),
        |(t, m2, n, err), p| {
            let short: Vec<&[[u64; 2]]> = sinks_ref.iter().flat_map(|s| s.short.slices(p)).collect();
            let long: Vec<&[[u64; 4]]> = sinks_ref.iter().flat_map(|s| s.long.slices(p)).collect();
            let ps: Vec<&[[u64; 2]]> = sinks_ref.iter().flat_map(|s| s.pshort.slices(p)).collect();
            let pl: Vec<&[[u64; 4]]> = sinks_ref.iter().flat_map(|s| s.plong.slices(p)).collect();
            match dedup::partition(&short, &long, &ps, &pl, &spec_ref[p], &cx3, t, m2) {
                Ok(k) => *n += k,
                Err(e) => *err = Some(e),
            }
        },
    );
    let mut m2s: Vec<Move2> = Vec::with_capacity(dd.len());
    let mut survivors = 0usize;
    for (_, m2, n, err) in dd {
        if let Some(e) = err {
            return Ok(Outcome::Fallback(format!("abort: {}", e.0)));
        }
        survivors += n;
        m2s.push(m2);
    }
    if !foreign_specials.is_empty() {
        let refs: Vec<&Special> = foreign_specials.iter().collect();
        survivors += dedup::foreign(&refs, &cx3, &mut m2s[0]);
    }
    for m in m2s.iter_mut() {
        m.ev.finish();
        m.buys.finish();
    }
    clock.mark("dedup");

    // Clicks to person partitions.
    let probe = Move2::new(p2_bits, 1);
    let ck_parts = crate::par::run(
        threads,
        sinks_ref.len(),
        |_| Bins::<[u64; 2]>::new(1 << p2_bits, 1 << 15),
        |bins, i| {
            for c in sinks_ref[i].clicks.all().flat_map(|c| c.iter()) {
                let person = people.person_of(c[0] as u32);
                bins.push(probe.part(person), [(u64::from(person) << 32) | (c[0] >> 32), c[1]]);
            }
        },
    );
    let mut ck_parts = ck_parts;
    for b in ck_parts.iter_mut() {
        b.finish();
    }
    clock.mark("clicks");

    // Person partitions.
    let day0 = ctx.window_start_ms.div_euclid(DAY);
    let span = ((ctx.window_end_ms - 1).div_euclid(DAY) - day0 + 1).max(1) as usize;
    let campaigns_with_organic = ctx.campaigns.ids.len() + 1;
    let organic = ctx.campaigns.ids.len() as u32;
    let person_bits = 32 - (tok.capacity() as u32).leading_zeros();
    let pcx = PeopleCtx { ds: &ds, ctx: &ctx, tok: &tok, organic, person_bits };
    let (m2s_ref, ck_ref) = (&m2s, &ck_parts);
    let workers = crate::par::run(
        threads,
        1 << p2_bits,
        |_| Worker::new(FastAgg::new(campaigns_with_organic, day0, span)),
        |w, q| {
            let ev: Vec<&[u64]> = m2s_ref.iter().flat_map(|m| m.ev.slices(q)).collect();
            let buys: Vec<&[[u64; 2]]> = m2s_ref.iter().flat_map(|m| m.buys.slices(q)).collect();
            let cl: Vec<&[[u64; 2]]> = ck_ref.iter().flat_map(|b| b.slices(q)).collect();
            w.partition(&ev, &buys, &cl, &pcx);
        },
    );
    clock.mark("sessions");
    if workers.iter().any(|w| w.bad) {
        return Ok(Outcome::Fallback("a purchase flag without its purchase record".into()));
    }

    // Write.
    let user_dir = out.join("user_daily");
    let campaign_dir = out.join("campaign_daily");
    std::fs::create_dir_all(&user_dir)?;
    std::fs::create_dir_all(&campaign_dir)?;
    let cats: Vec<&[u8]> = ctx.products.categories.iter().map(|c| c.as_slice()).collect();
    let mut camp = CampaignAgg::new(campaigns_with_organic, day0, span);
    let mut rows: Vec<&Rows> = Vec::new();
    let (mut sessions, mut events) = (0u64, 0u64);
    for w in &workers {
        w.agg.add_into(&mut camp);
        sessions += w.sessions;
        events += w.events;
        if w.rows.len() > 0 {
            rows.push(&w.rows);
        }
    }
    let empty = Rows::new();
    if rows.is_empty() {
        rows.push(&empty);
    }
    let errors: Mutex<Vec<io::Error>> = Mutex::new(Vec::new());
    crate::par::run(threads, rows.len() + 1, |_| (), |_, i| {
        let r = if i == rows.len() {
            crate::fast::write::campaigns(&campaign_dir.join("part-00000.parquet"), &camp, &ctx)
        } else {
            crate::fast::write::user_rows(&user_dir.join(format!("part-{i:05}.parquet")), rows[i], &cats)
        };
        if let Err(e) = r {
            errors.lock().unwrap().push(e);
        }
    });
    if let Some(e) = errors.into_inner().unwrap().into_iter().next() {
        return Err(e);
    }
    clock.mark("write");
    if timing {
        eprintln!(
            "v3: tokens {} (short {} of {}, load {:.3}; long {} of {}, load {:.3}), linked {}, persons {}, survivors {survivors}, events {events}, sessions {sessions}; total {:.2} ms",
            tok.capacity(),
            tok.short_used(),
            tok.short_cap(),
            tok.short_used() as f64 / tok.short_cap() as f64,
            tok.long_used(),
            tok.long_cap(),
            tok.long_used() as f64 / tok.long_cap() as f64,
            people.linked,
            people.persons,
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok(Outcome::Done)
}
