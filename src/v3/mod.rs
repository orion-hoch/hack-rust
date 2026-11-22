//! The v3 engine (RECORDS.md version 3): opaque ID keys to user tokens, exact dedup per move-1
//! partition, persons by union-find on tokens, person partitions sorted by (person, ts) and scanned
//! into sessions. Every person is handled the same way. Any failed internal check aborts to v1.

pub mod dedup;
pub mod ids;
pub mod mem;
pub mod people;
pub mod persons;
pub mod sink;
pub mod tokens;

use std::io;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use parselab::{Dataset, Stream};

use crate::fast::Outcome;
use crate::fast::agg::FastAgg;
use crate::fast::scan::Rows;
use crate::fast::sink::Bins;
use crate::sessions::CampaignAgg;
use crate::ts::DAY;

use self::dedup::{Ctx3, Move2, Tables};
use self::people::{PeopleCtx, Worker};
use self::persons::People;
use self::sink::{ClickSink, Special, V3Sink, part_of};
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
/// (user ms, system ms, minor faults, major faults) of the whole process so far.
fn usage() -> (f64, f64, i64, i64) {
    let mut r = Rusage::default();
    // SAFETY: `r` is a writable struct with the layout of Linux `struct rusage`.
    unsafe { getrusage(0, &mut r) };
    (
        r.utime[0] as f64 * 1e3 + r.utime[1] as f64 / 1e3,
        r.stime[0] as f64 * 1e3 + r.stime[1] as f64 / 1e3,
        r.rest[4],
        r.rest[5],
    )
}

struct Clock {
    on: bool,
    last: Instant,
    last_use: (f64, f64, i64, i64),
}

impl Clock {
    fn mark(&mut self, what: &str) {
        let now = Instant::now();
        if self.on {
            let u = usage();
            eprintln!(
                "timing v3.{what:<12} {:9.2} ms   user {:7.1} ms  sys {:7.1} ms  minflt {:7}  majflt {}",
                (now - self.last).as_secs_f64() * 1e3,
                u.0 - self.last_use.0,
                u.1 - self.last_use.1,
                u.2 - self.last_use.2,
                u.3 - self.last_use.3
            );
            self.last_use = u;
        }
        self.last = now;
    }
}

/// The inputs through rt: events, clicks and links mapped into one reservation (zero pages
/// around every file), registered for release-behind and for the parallel release at exit. The
/// same files as `Dataset::open` (regular `*.csv`, not starting with '.', in name order).
/// Err(why) when a folder holds more files than the parser's locator can number (named refusal,
/// decided before anything is mapped).
fn open_rt(dir: &Path) -> io::Result<Result<(Dataset, &'static rt::input::Inputs), String>> {
    use std::os::unix::ffi::OsStrExt;
    let errno = |e: rt::sys::Errno| io::Error::from_raw_os_error(e.0);
    let subs = ["events", "clicks", "links"];
    let mut ls = Vec::with_capacity(3);
    for sub in subs {
        let l = match rt::input::scan(dir.join(sub).as_os_str().as_bytes()) {
            Ok(l) => l,
            Err(rt::sys::Errno::ERANGE) => return Ok(Err(format!("more than {} *.csv files in {sub}/", rt::input::LISTING_MAX))),
            Err(e) => return Err(errno(e)),
        };
        if l.files().len() > parselab::Locator::MAX_FILES {
            return Ok(Err(format!("{} files in {sub}/, over the {} a locator numbers", l.files().len(), parselab::Locator::MAX_FILES)));
        }
        ls.push(l);
    }
    let ls: &'static [rt::input::Listing] = Box::leak(ls.into_boxed_slice());
    let streams: Vec<rt::input::Stream> = ls.iter().map(|l| rt::input::Stream { listing: l, dots: false }).collect();
    let slot = Box::leak(Box::new(std::mem::MaybeUninit::uninit()));
    let inp = rt::input::map(&streams, slot, true).map_err(errno)?;
    let files = [0usize, 1, 2].map(|s| {
        let base = dir.join(subs[s]);
        inp.stream(s)
            .iter()
            .zip(ls[s].files())
            .map(|(f, e)| {
                let path = base.join(std::ffi::OsStr::from_bytes(e.name()));
                // SAFETY: rt keeps the mapping for the whole run; this Mapped never unmaps it.
                unsafe { parselab::dataset::Mapped::from_raw(f.ptr, f.len, path) }
            })
            .collect()
    });
    let read = |name: &str| std::fs::read(dir.join(name)).unwrap_or_default();
    let ds = Dataset { dir: dir.to_path_buf(), files, products: read("products.csv"), campaigns: read("campaigns.csv"), manifest: read("manifest.json") };
    Ok(Ok((ds, inp)))
}

/// L2 bytes per hardware thread of CPU 0 (sysfs cache/index2 size over its thread siblings);
/// 512 KB when unreadable.
fn l2_per_thread() -> usize {
    let rd = |p: &str| std::fs::read_to_string(p).ok().map(|s| s.trim().to_string());
    let base = "/sys/devices/system/cpu/cpu0";
    let size = match (rd(&format!("{base}/cache/index2/level")), rd(&format!("{base}/cache/index2/size"))) {
        (Some(l), Some(s)) if l == "2" => {
            let (num, mul) = match s.strip_suffix('K') {
                Some(n) => (n, 1usize << 10),
                None => match s.strip_suffix('M') {
                    Some(n) => (n, 1 << 20),
                    None => (s.as_str(), 1),
                },
            };
            num.parse::<usize>().ok().map(|v| v * mul)
        }
        _ => None,
    };
    let Some(size) = size else { return 512 << 10 };
    // "0", "0,64", "0-1": count the listed threads.
    let siblings = rd(&format!("{base}/topology/thread_siblings_list")).map_or(1, |l| {
        l.split(',')
            .map(|r| match r.split_once('-') {
                Some((a, b)) => b.parse::<usize>().unwrap_or(0).saturating_sub(a.parse::<usize>().unwrap_or(0)) + 1,
                None => 1,
            })
            .sum::<usize>()
            .max(1)
    });
    size / siblings
}

/// `bits` moved by the signed integer in env `var` (measurement sweeps), kept in [lo, hi].
fn bits_delta(bits: u32, var: &str, lo: u32, hi: u32) -> u32 {
    let d: i32 = std::env::var(var).ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    (bits as i32 + d).clamp(lo as i32, hi as i32) as u32
}

/// Rows per user_daily file written during the session stage (about 1 MB of output).
const ROWS_PER_FILE: usize = 1 << 16;

/// Writes a session worker's rows as its next user_daily file and empties them.
fn flush_rows(w: &mut Worker, dir: &Path, cats: &[&[u8]]) {
    let path = dir.join(format!("part-{:03}-{:05}.parquet", w.index, w.files));
    w.write_err = crate::fast::write::user_rows(&path, &w.rows, cats).err();
    w.files += 1;
    w.wrote = true;
    w.rows.clear();
}

pub fn try_run(data: &Path, out: &Path, threads: usize, timing: bool) -> io::Result<Outcome> {
    let t0 = Instant::now();
    let mut clock = Clock { on: timing, last: t0, last_use: if timing { usage() } else { (0.0, 0.0, 0, 0) } };
    if timing {
        // The machine's caches as detected (partition sizes are set for half an L2 per thread).
        let c = rt::cpu::caches();
        let l = rt::cpu::topology(&rt::cpu::affinity(), b"/sys/devices/system/cpu");
        eprintln!(
            "v3: L2 {} KB shared by {} ({} KB per thread), L3 {} MB shared by {}, from {}; {} CPUs on {} cores",
            c.l2 >> 10,
            c.l2_shared,
            c.l2_per_thread() >> 10,
            c.l3 >> 20,
            c.l3_shared,
            match c.source { b's' => "sysfs", b'c' => "cpuid", _ => "nothing" },
            l.n,
            l.cores
        );
    }
    crate::par::init_pool(threads);
    if cfg!(feature = "measure") && std::env::var("ENGINE_BINPF").as_deref() == Ok("0") {
        crate::fast::sink::BINPF.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    let (ds, inp) = match open_rt(data)? {
        Ok(x) => x,
        Err(why) => return Ok(Outcome::Fallback(why)),
    };
    // The fast path's memory region (address space only until a stage takes from it).
    let input_bytes: u64 = [Stream::Events, Stream::Clicks, Stream::Links].iter().map(|&s| ds.bytes(s)).sum();
    crate::huge_pages(input_bytes);
    if !mem::init((input_bytes as usize / 100) * 45) {
        return Ok(Outcome::Fallback("memory region could not be reserved".into()));
    }
    let ev_bytes = ds.bytes(Stream::Events) as usize;
    let est_events = ev_bytes / 70 + 1;
    // User IDs come from all three inputs. In generator data every linked user is in a link line
    // (about one new ID per 18-byte line) and the users seen only in events or clicks are about
    // one per 30 of those lines; the estimate takes each input at that rate. The SHORT table
    // gets twice the estimate and the LONG table a quarter (one ID in eight is LONG), so
    // generator data sits below a third full; a table past 3/4 falls back to v1 (exact).
    let link_lines = ds.bytes(Stream::Links) as usize / 16 + 1;
    let click_lines = ds.bytes(Stream::Clicks) as usize / 40 + 1;
    let est_users = link_lines + (est_events + click_lines) / 30 + 4096;
    // An upper bound on distinct users from line-count bounds (shortest plausible lines): small
    // inputs size both tables from it, so an unusual mix of ID kinds does not fall back (tokens.rs).
    let bound_users = ds.bytes(Stream::Links) as usize / 8 * 2 + ev_bytes / 40 + ds.bytes(Stream::Clicks) as usize / 30;
    // The products and campaigns tables are read on a side thread while the pool builds and
    // faults the token table (neither needs the other).
    let (ctx, tok) = std::thread::scope(|sc| {
        let h = sc.spawn(|| parselab::prepare(&ds));
        let tok = Tokens::with_bound(est_users, bound_users);
        (h.join(), tok)
    });
    let ctx = match ctx {
        Ok(Ok(c)) => c,
        Ok(Err(f)) => return Ok(Outcome::Fallback(format!("{f:?}"))),
        Err(_) => return Ok(Outcome::Fallback("reading the tables failed".into())),
    };
    // The persons region (huge pages by the verdict for its own size), sized from the link
    // bytes: at most one link line per 16 bytes.
    if !mem::init_persons(persons::region_bytes(tok.capacity(), 2 * link_lines)) {
        return Ok(Outcome::Fallback("persons region could not be reserved".into()));
    }
    // Move-1 partitions of about 16K records: the dedup table (32K slots) stays in L2.
    // Move-1 partitions: one bit coarser than about 16K records each, and at most 2^11 (2^10 when
    // a thread has under 448 KB of L2): the sink's write fronts, one per partition and worker,
    // cost more than the larger dedup tables save. Swept on box 2 (Zen 2, 512 KB L2) and box 1:
    // at f60v5 2^11 instead of 2^13 took 6 to 9% off the whole run's CPU, at p20v5 2^10 instead
    // of 2^11 about 2%.
    let p1_raw = (est_events / 16_000).max(1).next_power_of_two().trailing_zeros();
    let p1_cap = if l2_per_thread() < (448 << 10) { 10 } else { 11 };
    let p1_bits = bits_delta(p1_raw.saturating_sub(1).min(p1_cap).clamp(4, 14), "ENGINE_P1_DELTA", 4, 14);
    clock.mark("open");

    // Parse A: links, then clicks.
    // Links only: clicks are parsed after the events and persons (ClickSink), straight into
    // person partitions, with read-only token lookups.
    let chunks = parselab::par::plan(&ds, &[Stream::Links], 4 << 20);
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
        // faults them back; keeping them mapped costs 70 ms at exit at practice size. Click lines
        // stay mapped: the session stage re-reads the click_id of every click in a ts tie
        // (about 180K reads at practice size), which faulted about 20K pages back.
        // rt batches whole 2 MB bins into 32 to 64 MB madvise calls (about 140 at p20v5 instead of
        // one per 4 MB chunk; each call interrupts every other running worker for its TLB flush).
        if c.stream != Stream::Clicks {
            inp.consumed(inp.first(c.stream as usize) + c.file as usize, c.from, c.to);
        }
    };
    // Links, persons and events in one pool run, no barrier between them: link chunks first;
    // a worker that moves on from links resolves its pending lookups and publishes its edges, and
    // the worker whose publication completes the links builds persons while the others parse
    // events.
    let plant = if cfg!(feature = "plant") { std::env::var("ENGINE_PLANT").unwrap_or_default() } else { String::new() };
    if plant == "full" {
        tok.full.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let ev_chunks = parselab::par::plan(&ds, &[Stream::Events], 4 << 20);
    let nlink = chunks.len();
    let people_slot: Mutex<Option<People>> = Mutex::new(None);
    let edges_pub: Mutex<Vec<Bins<u64>>> = Mutex::new(Vec::new());
    let links_published = std::sync::atomic::AtomicUsize::new(0);
    let link_bad = std::sync::atomic::AtomicBool::new(false);
    // Per participant: its sink, link chunks parsed and not yet published.
    let publish = |s: &mut V3Sink, done: &mut usize| {
        if *done == 0 {
            return;
        }
        s.drain();
        s.edges.finish();
        if s.bad {
            link_bad.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let e = std::mem::replace(&mut s.edges, Bins::with_chunk(1, 2, 2));
        edges_pub.lock().unwrap().push(e);
        let k = *done;
        *done = 0;
        if links_published.fetch_add(k, std::sync::atomic::Ordering::AcqRel) + k == nlink {
            // The last links are in: persons now, on this worker.
            let t = Instant::now();
            let ed = edges_pub.lock().unwrap();
            let refs: Vec<&[u64]> = ed.iter().flat_map(|b| b.all()).collect();
            let built = if link_bad.load(std::sync::atomic::Ordering::Relaxed) || tok.full.load(std::sync::atomic::Ordering::Relaxed) {
                None
            } else {
                People::build(&tok, &refs, timing)
            };
            *people_slot.lock().unwrap() = built;
            if timing {
                eprintln!("timing v3.  persons(task) {:9.2} ms", t.elapsed().as_secs_f64() * 1e3);
            }
        }
    };
    let st: Vec<(V3Sink, usize)> = crate::par::run_fin(
        threads,
        nlink + ev_chunks.len(),
        |t| (V3Sink::new(&tok, &ds, &ctx, p1_bits, per_thread, t), 0usize),
        |(s, done), i| {
            if i < nlink {
                parse(&chunks[i], s);
                *done += 1;
                return;
            }
            publish(s, done);
            parse(&ev_chunks[i - nlink], s);
        },
        |(s, done), _| publish(s, done),
    );
    let mut sinks: Vec<V3Sink> = st.into_iter().map(|(s, _)| s).collect();
    if nlink == 0 {
        // No link chunks: persons over no edges.
        *people_slot.lock().unwrap() = People::build(&tok, &[], timing);
    }
    let Some(people) = people_slot.into_inner().unwrap_or(None) else {
        return Ok(Outcome::Fallback("persons could not be built (bad token)".into()));
    };
    inp.flush();
    clock.mark("parse.events");
    if cfg!(feature = "measure") && std::env::var("ENGINE_STOP").is_ok() {
        std::process::exit(0);
    }
    // Finish first: the final drain resolves the last lookups and can raise the flags checked next.
    for s in sinks.iter_mut() {
        s.finish();
    }
    if tok.full.load(std::sync::atomic::Ordering::Relaxed) || sinks.iter().any(|s| s.bad) {
        return Ok(Outcome::Fallback("token table full or a record did not fit".into()));
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
    // A partition above 2^24 records (many copies of one event_id, say) goes to v1.
    if max_short.max(max_long) > 1 << 24 {
        return Ok(Outcome::Fallback(format!("a dedup partition of {} records", max_short.max(max_long))));
    }
    // Tables start at the typical partition size and grow only on the worker that meets an outlier.
    let typical_short = (sinks.iter().map(|s| s.short.len()).sum::<usize>() / parts) * 2 + 64;
    let typical_long = (sinks.iter().map(|s| s.long.len()).sum::<usize>() / parts) * 2 + 64;
    let est_surv = est_events;
    // Move-2 (person) partitions: likewise one bit coarser, at most 2^11 (swept with move 1).
    let p2_raw = (est_surv / 24_000).max(1).next_power_of_two().trailing_zeros();
    let p2_bits = bits_delta(p2_raw.saturating_sub(1).min(11).clamp(4, 13), "ENGINE_P2_DELTA", 4, 13);
    let cx3 = Ctx3 { people: &people, ingest_base_ms: ctx.ingest_base_s * 1000 };
    let sinks_ref = &sinks;
    let spec_ref = &spec_by_part;
    // Dedup partitions and the click parse in one pool run (clicks need only the token table and
    // the person map): the 4 MB click chunks first, the small dedup partitions fill the tail.
    let ck_chunks = parselab::par::plan(&ds, &[Stream::Clicks], 4 << 20);
    let nck = ck_chunks.len();
    let dd = crate::par::run(
        threads,
        nck + parts,
        |w| {
            (
                Tables::new((typical_short.min(max_short) * 3 / 2).next_power_of_two().max(16), (typical_long.min(max_long) * 3 / 2).next_power_of_two().max(16)),
                Move2::new_in(p2_bits, est_surv / threads.max(1), w),
                0usize,
                None::<Abort>,
                ClickSink::new(&tok, &people, &ds, &ctx, p2_bits, w),
            )
        },
        |(t, m2, n, err, cs), i| {
            if i < nck {
                let c = &ck_chunks[i];
                let bytes = ds.stream(c.stream)[c.file as usize].bytes();
                if reference {
                    parselab::reference::clicks(&ctx, c.file, bytes, c.from, c.to, cs);
                } else {
                    parselab::fast::parse(&fp, isa, parselab::fast::Kind::Clicks, c.file, parselab::fast::Buf::plain(bytes), c.from, c.to, cs);
                }
                return;
            }
            let p = i - nck;
            if plant == "panic" && p == 7 {
                panic!("planted panic in a dedup task");
            }
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
    let mut csinks: Vec<ClickSink> = Vec::with_capacity(dd.len());
    let mut survivors = 0usize;
    for (_, m2, n, err, cs) in dd {
        if let Some(e) = err {
            return Ok(Outcome::Fallback(format!("abort: {}", e.0)));
        }
        survivors += n;
        m2s.push(m2);
        csinks.push(cs);
    }
    if !foreign_specials.is_empty() {
        let refs: Vec<&Special> = foreign_specials.iter().collect();
        survivors += dedup::foreign(&refs, &cx3, &mut m2s[0]);
    }
    for m in m2s.iter_mut() {
        m.ev.finish();
        m.buys.finish();
    }
    let mut ck_parts: Vec<Bins<[u64; 2]>> = Vec::with_capacity(csinks.len());
    for s in csinks.iter_mut() {
        s.drain();
        if s.bad {
            return Ok(Outcome::Fallback("a click record did not fit".into()));
        }
        s.bins.finish();
    }
    for s in csinks {
        ck_parts.push(s.bins);
    }
    clock.mark("dedup+clicks");

    // Person partitions.
    let day0 = ctx.window_start_ms.div_euclid(DAY);
    let span = ((ctx.window_end_ms - 1).div_euclid(DAY) - day0 + 1).max(1) as usize;
    let campaigns_with_organic = ctx.campaigns.ids.len() + 1;
    let organic = ctx.campaigns.ids.len() as u32;
    // The session kernel: sorting network for small persons where AVX-512 exists, slots otherwise.
    // Test builds can force another (ENGINE_KERNEL=slots|sort|radix).
    let kernel = match if cfg!(feature = "plant") { std::env::var("ENGINE_KERNEL").unwrap_or_default() } else { String::new() }.as_str() {
        "slots" => sortlab::Kernel::Slots,
        "sort" => sortlab::Kernel::Sort,
        "radix" => sortlab::Kernel::Radix,
        _ => sortlab::Kernel::Net,
    };
    let walk = sortlab::walk::Params {
        base: ctx.click_base_ms - ctx.window_start_ms,
        day_shift: ctx.window_start_ms.rem_euclid(DAY) as u32,
        days: span,
        campaigns: campaigns_with_organic,
        organic,
    };
    let pcx = PeopleCtx { ds: &ds, ctx: &ctx, tok: &tok, organic, kernel, walk, day0: day0 as i32 };
    let (m2s_ref, ck_ref) = (&m2s, &ck_parts);
    // Each worker writes its own user_daily file as soon as the queue is empty for it, while the
    // others finish their partitions (no write stage for these rows).
    let user_dir = out.join("user_daily");
    let campaign_dir = out.join("campaign_daily");
    std::fs::create_dir_all(&user_dir)?;
    std::fs::create_dir_all(&campaign_dir)?;
    let cats: Vec<&[u8]> = ctx.products.categories.iter().map(|c| c.as_slice()).collect();
    let mut workers = crate::par::run_fin(
        threads,
        1 << p2_bits,
        |t| {
            // user-day rows are about 1 in 9 survivors on generator data; 1.5 times that per worker
            let mut w = Worker::new(FastAgg::new(campaigns_with_organic, day0, span), est_surv / 6 / threads.max(1));
            w.index = t;
            // About 0.11 user-day rows and 0.034 persons per event at practice size; a little more.
            let per = est_events / threads.max(1) + 1;
            w.rows.reserve(per / 7, per / 25, per / 25 * 12);
            w
        },
        |w, q| {
            let ev: Vec<&[u64]> = m2s_ref.iter().flat_map(|m| m.ev.slices(q)).collect();
            let buys: Vec<&[[u64; 2]]> = m2s_ref.iter().flat_map(|m| m.buys.slices(q)).collect();
            let cl: Vec<&[[u64; 2]]> = ck_ref.iter().flat_map(|b| b.slices(q)).collect();
            w.partition(&ev, &buys, &cl, &pcx);
            // A file per ROWS_PER_FILE rows, written by the worker as it goes, so that at the end
            // of the stage only each worker's remainder is left to write.
            if !w.bad && w.write_err.is_none() && w.rows.len() >= ROWS_PER_FILE {
                flush_rows(w, &user_dir, &cats);
            }
        },
        |w, _| {
            if !w.bad && w.write_err.is_none() && w.rows.len() > 0 {
                flush_rows(w, &user_dir, &cats);
            }
        },
    );
    clock.mark("sessions");
    if workers.iter().any(|w| w.bad) {
        return Ok(Outcome::Fallback("a purchase flag without its purchase record".into()));
    }

    // Write: the user_daily files are written; the campaign table from the merged counters.
    let mut camp = CampaignAgg::new(campaigns_with_organic, day0, span);
    let (mut sessions, mut events) = (0u64, 0u64);
    let mut wrote = false;
    for w in workers.iter_mut() {
        if let Some(e) = w.write_err.take() {
            return Err(e);
        }
        wrote |= w.wrote;
        w.agg.add_into(&mut camp);
        sessions += w.sessions;
        events += w.events;
    }
    if !wrote {
        crate::fast::write::user_rows(&user_dir.join("part-00000.parquet"), &Rows::new(), &cats)?;
    }
    crate::fast::write::campaigns(&campaign_dir.join("part-00000.parquet"), &camp, &ctx)?;
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
