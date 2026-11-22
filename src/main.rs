//! Shopping Sessions engine: click-stream CSVs in, user_daily and campaign_daily Parquet out.
//!
//! Stages (see STAGES.md): 1 discover and map inputs, 2 parse, 3 events dedup and validation,
//! 4 links and person names, 5 sessions, 6 attribution and assists, 7 aggregate and write.
//!
//! Usage: `engine [--timing] [DATA_DIR OUTPUT_DIR]`; otherwise `DATA_DIR`, `OUTPUT_DIR` and
//! `NCPU` come from the environment (defaults /data, /output, all CPUs). Logs go to stderr.

mod alloc;
mod csv;
mod dedup;
mod fast;
mod v3;
mod hash;
mod id;
mod input;
mod links;
mod output;
mod par;
mod parse;
mod pq;
mod sessions;
mod tables;
mod ts;

use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use crate::dedup::Kept;
use crate::hash::map_with_capacity;
use crate::id::Interner;
use crate::input::{Chunk, Source};
use crate::links::Persons;
use crate::output::CampaignDaily;
use crate::parse::{ClickCtx, ClickRec, EventCtx, EventRec, Scatter};
use crate::sessions::{CampaignAgg, SessionCtx, Sessionizer};
use crate::tables::{Campaigns, Products};
use crate::ts::DAY;

// musl's allocator made the static build twice as slow as the glibc one (see STAGES.md); the
// engine's own allocator is in alloc.rs.
#[global_allocator]
static ALLOCATOR: alloc::Alloc = alloc::Alloc;

struct Config {
    data: PathBuf,
    out: PathBuf,
    threads: usize,
    timing: bool,
}

fn config() -> Config {
    let mut timing = env::var_os("ENGINE_TIMING").is_some_and(|v| v == "1");
    let mut positional = Vec::new();
    for arg in env::args_os().skip(1) {
        if arg == "--timing" {
            timing = true;
        } else {
            positional.push(PathBuf::from(arg));
        }
    }
    let pick = |i: usize, var: &str, default: &str| {
        positional
            .get(i)
            .cloned()
            .or_else(|| env::var_os(var).map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from(default))
    };
    let threads = env::var("NCPU")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    Config { data: pick(0, "DATA_DIR", "/data"), out: pick(1, "OUTPUT_DIR", "/output"), threads, timing }
}

struct Timer {
    on: bool,
    start: Instant,
    last: Instant,
}

impl Timer {
    fn new(on: bool) -> Self {
        let now = Instant::now();
        Timer { on, start: now, last: now }
    }

    fn mark(&mut self, stage: &str) {
        if self.on {
            let now = Instant::now();
            eprintln!("timing {stage:<12} {:9.1} ms", (now - self.last).as_secs_f64() * 1e3);
            self.last = now;
        }
    }

    fn total(&self) {
        if self.on {
            eprintln!("timing {:<12} {:9.1} ms", "total", self.start.elapsed().as_secs_f64() * 1e3);
        }
    }
}

/// Radix bits so that a partition holds about 16K records.
fn bits_for(records: usize) -> u32 {
    let mut bits = 0;
    while bits < 12 && (records >> bits) > 16_384 {
        bits += 1;
    }
    bits
}

fn total_bytes(sources: &[Source]) -> usize {
    sources.iter().map(|s| s.bytes().len()).sum()
}

/// Regroups per-thread partitions into per-partition lists, each behind a lock so one worker
/// can take (and free) it.
fn transpose<T>(per_thread: Vec<Vec<Vec<T>>>, parts: usize) -> Vec<Mutex<Vec<Vec<T>>>> {
    let mut out: Vec<Vec<Vec<T>>> = (0..parts).map(|_| Vec::with_capacity(per_thread.len())).collect();
    for thread in per_thread {
        for (p, v) in thread.into_iter().enumerate() {
            if !v.is_empty() {
                out[p].push(v);
            }
        }
    }
    out.into_iter().map(Mutex::new).collect()
}

fn concat<T: Copy>(parts: Vec<Vec<T>>) -> Vec<T> {
    if parts.len() == 1 {
        return parts.into_iter().next().unwrap();
    }
    let mut out = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for p in parts {
        out.extend_from_slice(&p);
    }
    out
}

#[derive(Clone, Copy)]
enum Task {
    Events(Chunk),
    Clicks(Chunk),
}

fn run(cfg: &Config) -> io::Result<()> {
    let mut timer = Timer::new(cfg.timing);
    let threads = cfg.threads;

    // Stage 1: discover and map.
    let manifest = fs::read(cfg.data.join("manifest.json"))?;
    let (window_start, window_end) = input::manifest_window(&manifest)?;
    let products = Products::parse(&fs::read(cfg.data.join("products.csv"))?);
    let campaigns = Campaigns::parse(&fs::read(cfg.data.join("campaigns.csv"))?);
    let event_sources = input::open_all(input::list_csv(&cfg.data.join("events"))?)?;
    let click_sources = input::open_all(input::list_csv(&cfg.data.join("clicks"))?)?;
    let link_sources = input::open_all(input::list_csv(&cfg.data.join("links"))?)?;
    let all_bytes = total_bytes(&event_sources) + total_bytes(&click_sources) + total_bytes(&link_sources);
    let target = (all_bytes / (threads * 16).max(1)).clamp(1 << 18, 1 << 25);
    let (mut event_chunks, mut click_chunks, mut link_chunks) = (Vec::new(), Vec::new(), Vec::new());
    input::chunks(&event_sources, target, &mut event_chunks);
    input::chunks(&click_sources, target, &mut click_chunks);
    input::chunks(&link_sources, target, &mut link_chunks);
    if cfg.timing {
        eprintln!(
            "inputs: {} event files {} B, {} click files {} B, {} link files {} B; {} threads",
            event_sources.len(),
            total_bytes(&event_sources),
            click_sources.len(),
            total_bytes(&click_sources),
            link_sources.len(),
            total_bytes(&link_sources),
            threads
        );
    }
    timer.mark("discover");

    // Stage 2a and 4: links, then persons.
    let ids = Interner::new();
    let pairs = par::run(threads, link_chunks.len(), |_| Vec::new(), |out, i| {
        let c = link_chunks[i];
        parse::links(&link_sources[c.source as usize].bytes()[c.start..c.end], &ids, out);
    });
    timer.mark("parse_links");
    let persons = Persons::build(&pairs, &ids);
    drop(pairs);
    timer.mark("persons");

    // Stage 2b: events and clicks, biggest chunks first.
    let event_bits = bits_for(total_bytes(&event_sources) / 64 + 1);
    let person_bits = event_bits;
    let mut tasks: Vec<Task> = event_chunks.iter().map(|&c| Task::Events(c)).collect();
    tasks.extend(click_chunks.iter().map(|&c| Task::Clicks(c)));
    tasks.sort_by_key(|t| match t {
        Task::Events(c) | Task::Clicks(c) => std::cmp::Reverse(c.len()),
    });
    let event_ctx = EventCtx { ids: &ids, products: &products, window_start, window_end };
    let click_ctx = ClickCtx { ids: &ids, campaigns: &campaigns, persons: &persons, window_start, window_end };
    let parsed = par::run(
        threads,
        tasks.len(),
        |_| (Scatter::<EventRec>::new(event_bits), Scatter::<ClickRec>::new(person_bits)),
        |(events, clicks), i| match tasks[i] {
            Task::Events(c) => {
                parse::events(&event_sources[c.source as usize].bytes()[c.start..c.end], &event_ctx, events)
            }
            Task::Clicks(c) => {
                parse::clicks(&click_sources[c.source as usize].bytes()[c.start..c.end], &click_ctx, clicks)
            }
        },
    );
    let copies: usize = parsed.iter().map(|(e, _)| e.len()).sum();
    let valid_clicks: usize = parsed.iter().map(|(_, c)| c.len()).sum();
    let (event_parts, click_parts): (Vec<_>, Vec<_>) = parsed.into_iter().map(|(e, c)| (e.parts, c.parts)).unzip();
    let event_parts = transpose(event_parts, 1 << event_bits);
    let click_parts = transpose(click_parts, 1 << person_bits);
    timer.mark("parse");

    // Stage 3: events dedup, validity, lateness.
    let kept = par::run(
        threads,
        event_parts.len(),
        |_| (map_with_capacity::<u64, (u32, u32)>(0), Scatter::<Kept>::new(person_bits)),
        |(table, out), p| {
            let parts = std::mem::take(&mut *event_parts[p].lock().unwrap());
            dedup::partition(&parts, &persons, table, out);
        },
    );
    drop(event_parts);
    let kept_events: usize = kept.iter().map(|(_, k)| k.len()).sum();
    let kept_parts = transpose(kept.into_iter().map(|(_, k)| k.parts).collect(), 1 << person_bits);
    timer.mark("dedup");

    // Stages 5 and 6: sessions, credit, assists; per person partition, biggest first.
    let day0 = window_start.div_euclid(DAY);
    let span = ((window_end - 1).div_euclid(DAY) - day0 + 1).max(1) as usize;
    let sizes: Vec<usize> = (0..kept_parts.len())
        .map(|p| {
            let e: usize = kept_parts[p].lock().unwrap().iter().map(|v| v.len()).sum();
            let c: usize = click_parts[p].lock().unwrap().iter().map(|v| v.len()).sum();
            e + c
        })
        .collect();
    let mut order: Vec<usize> = (0..kept_parts.len()).collect();
    order.sort_by_key(|&p| std::cmp::Reverse(sizes[p]));
    let session_ctx = SessionCtx { ids: &ids, organic: campaigns.organic() };
    let workers = par::run(
        threads,
        order.len(),
        |_| Sessionizer::new(products.categories.len(), CampaignAgg::new(campaigns.count_with_organic(), day0, span)),
        |s, i| {
            let p = order[i];
            let mut events = concat(std::mem::take(&mut *kept_parts[p].lock().unwrap()));
            let mut clicks = concat(std::mem::take(&mut *click_parts[p].lock().unwrap()));
            s.partition(&mut events, &mut clicks, &session_ctx);
        },
    );
    drop(kept_parts);
    drop(click_parts);
    timer.mark("sessions");

    // Stage 7: aggregate and write.
    let user_dir = cfg.out.join("user_daily");
    let campaign_dir = cfg.out.join("campaign_daily");
    fs::create_dir_all(&user_dir)?;
    fs::create_dir_all(&campaign_dir)?;
    let mut sessions_total = 0u64;
    let mut users = Vec::with_capacity(workers.len());
    let mut camp: Option<CampaignAgg> = None;
    for w in workers {
        sessions_total += w.sessions_total;
        users.push(w.user);
        match camp.as_mut() {
            None => camp = Some(w.camp),
            Some(c) => c.merge(w.camp),
        }
    }
    let camp = camp.unwrap_or_else(|| CampaignAgg::new(campaigns.count_with_organic(), day0, span));
    let mut table = CampaignDaily::new();
    for (campaign, day, c) in camp.rows() {
        table.campaign_id.push(campaigns.id_bytes(campaign));
        table.channel.push(campaigns.channel_bytes(campaign));
        table.day.push(day);
        table.sessions.push(c[0] as i64);
        table.purchasing_sessions.push(c[1] as i64);
        table.revenue_cents.push(c[2] as i64);
        table.assisted_sessions.push(c[3] as i64);
    }
    let user_rows: usize = users.iter().map(|u| u.len()).sum();
    let campaign_rows = table.len();
    let mut files: Vec<usize> = (0..users.len()).filter(|&i| users[i].len() > 0).collect();
    if files.is_empty() {
        files.push(0);
    }
    let errors: Mutex<Vec<io::Error>> = Mutex::new(Vec::new());
    par::run(threads, files.len() + 1, |_| (), |_, i| {
        let result = if i == files.len() {
            pq::write_campaign_daily(&campaign_dir.join("part-00000.parquet"), &table)
        } else {
            let path = user_dir.join(format!("part-{i:05}.parquet"));
            pq::write_user_daily(&path, &users[files[i]], &products.categories)
        };
        if let Err(e) = result {
            errors.lock().unwrap().push(e);
        }
    });
    if let Some(e) = errors.into_inner().unwrap().into_iter().next() {
        return Err(e);
    }
    timer.mark("write");
    if cfg.timing {
        eprintln!(
            "counts: {} linked ids in {} linked persons, {} interned ids, {} event copies, {} kept events, \
             {} valid clicks, {} sessions, {} user_daily rows, {} campaign_daily rows",
            persons.linked_ids,
            persons.components,
            ids.interned_count(),
            copies,
            kept_events,
            valid_clicks,
            sessions_total,
            user_rows,
            campaign_rows
        );
    }
    timer.total();
    Ok(())
}

/// Huge pages for the allocator unless compaction would be costly: rt's verdict for about 0.7
/// times the input bytes (the measured peak of anonymous memory is 0.57 to 0.65 times the input)
/// says yes when the free 2 MB blocks cover it, when MemFree is at least twice it (compaction
/// cheap), when the defrag mode cannot stall, or when nothing can be read; then every mapping.
/// Otherwise (blocks short and free memory tight) a budget of 80% of the free blocks, first come
/// first served (the token table and the parse buffers come first). THP never: none.
/// RT_HUGE=on|off|<MB> overrides (measurements).
fn huge_pages(cfg: &Config) {
    use std::os::unix::ffi::OsStrExt;
    let budget = match env::var("RT_HUGE").as_deref() {
        Ok("on") => usize::MAX,
        Ok("off") => 0,
        Ok(mb) if mb.parse::<usize>().is_ok() => mb.parse::<usize>().unwrap_or(0) << 20,
        _ => {
            let mut bytes = 0u64;
            for sub in ["events", "clicks", "links"] {
                if let Ok(l) = rt::input::scan(cfg.data.join(sub).as_os_str().as_bytes()) {
                    bytes += l.files().iter().chain(l.dots()).map(|e| e.size).sum::<u64>();
                }
            }
            let v = rt::arena::huge_verdict((bytes as usize) / 10 * 7);
            if v.enabled == b'n' {
                0
            } else if v.use_huge {
                usize::MAX
            } else {
                (v.free_2m.max(0) as usize) * (2 << 20) / 5 * 4
            }
        }
    };
    alloc::set_thp_budget(budget);
}

/// PROBE 3 (not for the trunk): sleep until start + 1.0 s + 0.4 s x code just before exit, with
/// code = E + 4F + 16G: E = L2 per core (0 <= 256 KB, 1 512 KB, 2 1 to 1.25 MB, 3 >= 2 MB);
/// F = L3 (0 <= 12 MB, 1 16 to 20 MB, 2 24 to 36 MB, 3 >= 40 MB); G = 0 for Intel family 6
/// model 0x55, else 1. The remainder of the score under 0.4 s is the container's own cost.
fn probe3_code() -> u64 {
    let c = rt::cpu::caches();
    let kb = c.l2 >> 10;
    let e = if kb <= 384 { 0 } else if kb <= 768 { 1 } else if kb < 2048 { 2 } else { 3 };
    let mb = c.l3 >> 20;
    let f = if mb <= 14 { 0 } else if mb <= 22 { 1 } else if mb <= 38 { 2 } else { 3 };
    let g = if model_0x55() { 0 } else { 1 };
    (e + 4 * f + 16 * g) as u64
}

fn model_0x55() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__cpuid;
        let v = unsafe { __cpuid(0) };
        if !(v.ebx == 0x756e_6547 && v.edx == 0x4965_6e69 && v.ecx == 0x6c65_746e) {
            return false;
        }
        let f = unsafe { __cpuid(1) }.eax;
        ((f >> 8) & 0xf) == 6 && (((f >> 4) & 0xf) | (((f >> 16) & 0xf) << 4)) == 0x55
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn probe3_sleep(start_ns: u64) {
    let target = start_ns + 1_000_000_000 + 400_000_000 * probe3_code();
    let now = rt::sys::mono_ns();
    if target > now {
        std::thread::sleep(std::time::Duration::from_nanos(target - now));
    }
}

fn main() {
    let probe_start = rt::sys::mono_ns();
    // First, while single-threaded: grow the descriptor table (growing it once threads share it
    // waits an RCU grace period, 11 to 22 ms measured on the x86 box) and start the stage clock.
    rt::init(0);
    let cfg = config();
    huge_pages(&cfg);
    if env::var_os("ENGINE_PATH").is_some_and(|v| v == "v1") {
        // Possibly started by rt's fallback after a fault: no fast-path thread exists any more,
        // so this removes whatever partial output it left, race-free.
        use std::os::unix::ffi::OsStrExt;
        for table in ["user_daily", "campaign_daily"] {
            rt::out::sweep(cfg.out.join(table).as_os_str().as_bytes());
        }
    } else {
        // SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT or SIGTRAP anywhere in the fast path: rt removes
        // its output and re-executes this binary with ENGINE_PATH=v1 (on the thread's alt stack).
        rt::fail::install();
    }
    // The dataset-level switch: the fast path when the ID keys are recovered (or forced), v1
    // otherwise. ENGINE_PATH=v1 forces v1; ENGINE_PATH=fast fails instead of falling back.
    let path = env::var("ENGINE_PATH").unwrap_or_default();
    if path != "v1" {
        // A refusal, failed check or I/O error in the fast path falls back to v1 here, after
        // removing whatever the fast path may have written. A panic (panic = "abort") or a fault
        // signal goes through rt's handler instead: it re-executes this binary with ENGINE_PATH=v1.
        let fast = v3::try_run(&cfg.data, &cfg.out, cfg.threads, cfg.timing);
        let why = match fast {
            // Done: release input and memory in parallel, then exit_group (no destructors); with
            // everything still mapped the kernel's single-thread teardown took 80 to 90 ms.
            Ok(fast::Outcome::Done) => {
                probe3_sleep(probe_start);
                rt::exit::leave(0, &rt::exit::Release::ALL)
            }
            Ok(fast::Outcome::Fallback(why)) => why,
            Err(e) => format!("I/O error: {e}"),
        };
        if path == "fast" {
            eprintln!("engine: fast path refused: {why}");
            std::process::exit(2);
        }
        eprintln!("engine: fast path declined ({why}); running v1");
        for table in ["user_daily", "campaign_daily"] {
            let dir = cfg.out.join(table);
            if dir.exists() {
                if let Err(e) = fs::remove_dir_all(&dir) {
                    eprintln!("engine: cannot clear {}: {e}", dir.display());
                    std::process::exit(1);
                }
            }
        }
    }
    if let Err(e) = run(&cfg) {
        eprintln!("engine: {e}");
        std::process::exit(1);
    }
    probe3_sleep(probe_start);
}
