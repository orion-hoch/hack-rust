//! v4: the user-partitioned pipeline (CHECKPOINT.md "V4"; bit-exact layouts in RECORDS-V4.md).
//! No global user table: survivors are partitioned by a reversible mix of the user key and users
//! are resolved inside the cache per partition (P1); linked users' blocks move once to their
//! representative's partition; P2 runs the session kernel, walk and rows on person blocks.
//!
//! The default path (ENGINE_V4=0 turns it off and runs v3). Every stage returns `Err(why)` for anything it cannot hold (a bound
//! overflowed, FOREIGN volume, a failed check, or a stage not built yet), and the run then
//! falls back to v3 (which falls back to v1).
//!
//! Stages and their interfaces (owners in CHECKPOINT.md):
//! - `links::run`  -> `Links`       (I4: union-find over link users, laid out per user partition)
//! - `sink::events`-> `EventBins`   (I1: 16-byte records per event partition, a pure scatter)
//! - `sink::clicks`-> `ClickBins`   (I3: click records per user partition)
//! - `dedup::run`  -> `Survivors`   (I2: 8-byte survivors per user partition)
//! - `p1::run`     -> `Blocks`      (person blocks per partition, in cache)
//! - `p2::run`     -> output files  (session kernel, walk, rows; user_daily and campaign_daily)

pub mod dedup;
pub mod keys;
pub mod linkkernel;
pub mod links;
pub mod p1;
pub mod p2;
pub mod sink;

use std::io;
use std::path::Path;

use parselab::{Ctx, Dataset, Stream};

use crate::fast::Outcome;

/// A stage's result: `Err(why)` sends the run to v3.
pub type Stage<T> = Result<T, String>;

/// Sizes fixed before the first parse byte.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub threads: usize,
    /// Event lines, estimated from the bytes (about 70 per line).
    pub est_events: usize,
    /// Event partitions: 1 << b1 (top bits of mix40(event key)); 8..=11 (RECORDS-V4.md 2).
    pub b1: u32,
    /// User partitions: 1 << b2 (top bits of mix40(user key)); 8..=13.
    pub b2: u32,
}

/// Everything a stage may read: inputs, tables, parser, plan.
pub struct Cx<'a> {
    pub ds: &'a Dataset,
    pub ctx: &'a Ctx,
    pub inp: &'static rt::input::Inputs,
    pub fp: parselab::fast::Fast<'a>,
    pub isa: parselab::fast::Isa,
    pub plan: Plan,
    pub timing: bool,
}

macro_rules! stage {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(why) => return Ok(Outcome::Fallback(format!("v4: {why}"))),
        }
    };
}

/// Test only (feature "plant"), ENGINE_PLANT = v4-decline | v4-segv | v4-panic: partial output
/// in both tables, then a decline, a fault or a panic. Each must end with v3's output alone.
/// Test builds (`plant` feature): ENGINE_PLANT=v4-segv|v4-panic|v4-decline writes junk into both
/// tables, then faults, panics or declines. True: decline (the caller returns a Fallback, so the
/// path "decline after partial output" is exercised).
fn plant(out: &Path) -> bool {
    let k = std::env::var("ENGINE_PLANT").unwrap_or_default();
    if !k.starts_with("v4-") {
        return false;
    }
    for t in ["user_daily", "campaign_daily"] {
        let _ = std::fs::create_dir_all(out.join(t));
        let _ = std::fs::write(out.join(t).join("part-v4-plant.parquet"), b"partial");
    }
    match k.as_str() {
        "v4-segv" => {
            unsafe { std::ptr::write_volatile(16 as *mut u64, 1) };
            false
        }
        "v4-panic" => panic!("planted panic in v4"),
        _ => true,
    }
}

/// ENGINE_T0 = CLOCK_MONOTONIC ns at the exec (set by a launcher): start-up laps on stderr,
/// "lap <what> <ms since exec>". Unset: nothing.
pub(crate) fn lap(what: &str) {
    static T0: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let t0 = *T0.get_or_init(|| std::env::var("ENGINE_T0").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
    if t0 != 0 {
        eprintln!("lap {what:<14} {:7.3} ms", rt::sys::mono_ns().saturating_sub(t0) as f64 / 1e6);
    }
}

/// A whole file mapped read-only and populated (no copy, no fresh pages; kept for the run), as
/// `std::fs::read(..).unwrap_or_default()` would read it: empty when missing or unreadable.
fn map_file(path: &Path) -> &'static [u8] {
    use std::os::fd::AsRawFd;
    let Ok(f) = std::fs::File::open(path) else { return &[] };
    let len = f.metadata().map(|m| m.len() as usize).unwrap_or(0);
    if len == 0 {
        return std::fs::read(path).map(|v| &*v.leak()).unwrap_or(&[]);
    }
    // SAFETY: a fresh private read-only mapping of an open file; never unmapped.
    match unsafe { rt::sys::mmap(0, len, rt::sys::PROT_READ, rt::sys::MAP_PRIVATE | rt::sys::MAP_POPULATE, f.as_raw_fd(), 0) } {
        Ok(p) => unsafe { std::slice::from_raw_parts(p as *const u8, len) },
        Err(_) => std::fs::read(path).map(|v| &*v.leak()).unwrap_or(&[]),
    }
}

pub fn try_run(data: &Path, out: &Path, threads: usize, timing: bool) -> io::Result<Outcome> {
    let mut t = std::time::Instant::now();
    let mut cpu = crate::v3::usage();
    let mut mark = |what: &str| {
        if timing {
            let now = std::time::Instant::now();
            // CPU time of the stage against its wall time on `threads` workers: how busy they were.
            let c = crate::v3::usage();
            let (ms, used) = ((now - t).as_secs_f64() * 1e3, (c.0 - cpu.0) + (c.1 - cpu.1));
            eprintln!(
                "timing v4.{what:<14} cpu: user {:7.1} ms  sys {:6.1} ms  busy {:3.0}% of {} workers",
                c.0 - cpu.0,
                c.1 - cpu.1,
                100.0 * used / (ms * threads as f64).max(1e-9),
                threads
            );
            cpu = c;
            // Resident memory at the end of the stage (anonymous = ours; file = mapped input).
            let st = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
            let kb = |k: &str| st.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
            eprintln!(
                "timing v4.{what:<14} {:8.2} ms   anon {:5} MB  file {:5} MB  peak {:5} MB   8K blocks: taken {} freed {} fresh {} ({} MB fresh)",
                (now - t).as_secs_f64() * 1e3,
                kb("RssAnon:") >> 10,
                kb("RssFile:") >> 10,
                kb("VmHWM:") >> 10,
                crate::v3::mem::B8K.stats().0,
                crate::v3::mem::B8K.stats().1,
                crate::v3::mem::B8K.stats().2,
                crate::v3::mem::B8K.stats().2 >> 7
            );
            eprintln!(
                "timing v4.{what:<14} memory: 2K blocks fresh {} MB; region high water {} MB, live {:?}; heap mapped {} MB",
                crate::v3::mem::B2K.stats().2 >> 9,
                crate::v3::mem::REG.high_water() >> 20,
                crate::v3::mem::REG.live(),
                crate::alloc::mapped_bytes() >> 20
            );
            t = now;
        }
    };
    // The side tables (products, campaigns, manifest) are read and built on a thread of their own
    // while this one starts the pool and maps the inputs: they were 5 ms in front of the parse.
    let dir = data.to_path_buf();
    let side = std::thread::spawn(move || {
        let read = |name: &str| map_file(&dir.join(name));
        let (m, p, c) = (read("manifest.json"), read("products.csv"), read("campaigns.csv"));
        let t = (|| Ok((parselab::tables::manifest(&m)?, parselab::tables::products(&p)?, parselab::tables::campaigns(&c)?)))();
        lap("side tables");
        t
    });
    crate::par::init_pool(threads);
    let (ds, inp) = match crate::v3::open_rt_files(data)? {
        Ok(x) => x,
        Err(why) => return Ok(Outcome::Fallback(why)),
    };
    let input_bytes: u64 = [Stream::Events, Stream::Clicks, Stream::Links].iter().map(|&s| ds.bytes(s)).sum();
    lap("inputs mapped");
    if !crate::v3::mem::init((input_bytes as usize / 100) * 45) {
        return Ok(Outcome::Fallback("memory region could not be reserved".into()));
    }
    let Ok(tables) = side.join() else {
        return Ok(Outcome::Fallback("v4: the side-table thread failed".into()));
    };
    // As parselab::prepare: the file limits, then the tables, then the window.
    let ctx = match ds.check_limits().and_then(|_| tables).and_then(|(w, p, c)| parselab::Ctx::new(w, p, c)) {
        Ok(c) => c,
        Err(f) => return Ok(Outcome::Fallback(format!("{f:?}"))),
    };
    // Every check that needs no parse runs here, before the first parse byte: a decline later
    // costs the whole parse twice (RECORDS-V4.md section 9).
    if ctx.campaigns.ids.len() >= 4096 {
        return Ok(Outcome::Fallback("v4: 4,096 campaigns or more (I3 keeps 12 bits)".into()));
    }
    // Partition bits (RECORDS-V4.md section 2): computed once here, read by every stage.
    let est_events = ds.bytes(Stream::Events) as usize / 70 + 1;
    let (b1, b2) = keys::part_bits(ds.bytes(Stream::Events));
    let cx = Cx {
        ds: &ds,
        ctx: &ctx,
        inp,
        fp: parselab::fast::Fast::new(&ctx),
        isa: parselab::fast::Isa::detect(),
        plan: Plan { threads, est_events, b1, b2 },
        timing,
    };
    if cfg!(feature = "plant") && plant(out) {
        return Ok(Outcome::Fallback("v4: planted decline after partial output".into()));
    }
    mark("open");
    // One pool run for the link, event and click chunks (links first); the worker that completes
    // the last link chunk runs the links' union-find and layout itself, beside the rest.
    let lt = links::LinkTasks::new(&cx);
    lap("pool run");
    let (ev, ck) = sink::parse_all(&cx, &lt);
    let links = lt.result();
    drop(lt);
    let (links, ev, ck) = (stage!(links), stage!(ev), stage!(ck));
    mark("events+clicks");
    let sv = stage!(dedup::run(&cx, &ev));
    mark("dedup");
    if timing {
        let p2 = 1usize << cx.plan.b2;
        let (mut c, mut b, mut cw, mut bw) = (0usize, 0usize, 0usize, 0usize);
        for q in 0..p2 {
            for x in sv.short(q) {
                c += 1;
                b += x.len() * 8;
            }
            for x in sv.wide(q) {
                cw += 1;
                bw += x.len() * 8;
            }
        }
        eprintln!("timing v4.usage I2 short  {p2:5} parts: {c:7} chunks ({:5} MB) hold {:5} MB", (c * 8192) >> 20, b >> 20);
        eprintln!("timing v4.usage I2 wide   {p2:5} parts: {cw:7} chunks ({:5} MB) hold {:5} MB", (cw * 8192) >> 20, bw >> 20);
    }
    let blocks = stage!(p1::run(&cx, &sv, &ck, &links));
    mark("p1");
    stage!(p2::run(&cx, &blocks, &links, out));
    mark("p2");
    Ok(Outcome::Done)
}
