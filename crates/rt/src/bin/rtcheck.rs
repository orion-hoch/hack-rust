//! Tests and measurements for the `rt` crate. Uses std (never shipped).
//!
//! rtcheck pool-test                 queue semantics, priorities, continuations, TLS on workers
//! rtcheck pool-bench [workers]      start-up, task cost, wake latency, context switches
//! rtcheck map DATA [mode] [rel]     list + map + checks + 8-worker touch (plain|seq|populate|pread)
//! rtcheck arena-test                recycling causes no new faults; allocator checks
//! rtcheck arena-bench MB huge pop   first-touch cost (huge: on|off|default; pop: touch|populate)
//! rtcheck exit MB huge inputs arenas [DATA]   tail from "work done" to exit
//! rtcheck fault KIND MB OUTDIR      plant a fault on a worker (child of fault-suite)
//! rtcheck fault-suite OUTDIR        runs every planted fault as a child and checks the re-exec
//! rtcheck launch ARGS...            runs `rtcheck ARGS` as a child with RT_ORIGIN_NS set; reports
//!                                   the child's T_ lines relative to the spawn and its end

use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::*};

use rt::{arena, clock, cpu, exit, fail, input, out, pool, sys};

fn mono() -> u64 {
    sys::mono_ns()
}

/// "T_<name> <mono ns>" on stdout, one write call (read by `launch`).
fn tline(name: &str) {
    let s = format!("T_{name} {}\n", mono());
    let _ = sys::write_all(1, s.as_bytes());
}

fn main() {
    if std::env::var("RT_NO_INIT").is_err() {
        rt::init(0);
    }
    // The pread reader and the content checks use the files' descriptors (map_all turns this off
    // for sets of 900 files or more: the soft limit of open files is often 1,024).
    input::set_keep_fds(true);
    let args: Vec<String> = std::env::args().collect();
    if std::env::var("ENGINE_PATH").as_deref() == Ok("v1") {
        return fallback_main(&args);
    }
    if let Ok(o) = std::env::var("RT_ORIGIN_NS") {
        clock::set_origin(o.parse().unwrap());
    }
    if std::env::var("RT_REPORT").is_ok() {
        clock::enable_report(true);
    }
    let a: Vec<&str> = args.iter().skip(2).map(|s| s.as_str()).collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("pool-test") => pool_test(),
        Some("pool-bench") => pool_bench(&a),
        Some("map") => map_cmd(&a),
        Some("arena-test") => arena_test(),
        Some("arena-bench") => arena_bench(&a),
        Some("exit") => exit_cmd(&a),
        Some("fault") => fault_cmd(&a),
        Some("fault-suite") => fault_suite(&a),
        Some("launch") => launch(&a),
        Some("info") => info(),
        Some("verdict") => verdict_cmd(&a),
        Some("topo-test") => topo_test(),
        Some("blocks-bench") => blocks_bench(&a),
        Some("fragment") => fragment(&a),
        Some("reader-test") => reader_test(&a),
        Some("release-bench") => release_bench(&a),
        _ => {
            eprintln!("usage: see the header of src/bin/rtcheck.rs");
            std::process::exit(2);
        }
    }
}

/// Keeps musl's internal locks on while raw workers run (see `pool` docs): one parked std thread.
fn libc_threaded_hint() {
    std::thread::spawn(|| {
        loop {
            std::thread::park();
        }
    });
}

fn start(workers: usize) -> usize {
    libc_threaded_hint();
    let mut cfg = pool::Config::DEFAULT;
    cfg.workers = workers;
    if let Ok(v) = std::env::var("RT_PIN") {
        cfg.pin = match v.as_str() {
            "yes" => pool::Pin::Yes,
            "no" => pool::Pin::No,
            _ => pool::Pin::Auto,
        };
    }
    if let Ok(v) = std::env::var("RT_SPIN_US") {
        cfg.spin_us = v.parse().unwrap();
    }
    pool::start(&cfg)
}

fn info() {
    let n = start(0);
    pool::wait_running();
    let (allowed, quota) = pool::allowed_cpus();
    println!(
        "workers {n} allowed {allowed} quota {quota} pinned {} topology_known {} musl_tcb {}",
        pool::pinned(),
        pool::topology_known(),
        rt::tls::musl_layout_ok()
    );
    for w in 0..n {
        println!("  worker {w}: cpu {} core {}", pool::cpu_of(w) as i64, pool::core_of(w));
    }
    let h = arena::huge_verdict(1 << 30);
    println!(
        "thp enabled '{}' defrag '{}' free_2m normal+movable {} all zones {} may_stall {} use_huge(1 GB) {}",
        h.enabled as char, h.defrag as char, h.free_2m, h.free_2m_all, h.may_stall, h.use_huge
    );
    let seg = rt::tls::segment();
    println!("tls segment {:?} area {}", seg, rt::tls::area_size(&seg));
}

/// What the engine's main.rs huge_pages() decides for DATA, and the numbers it decides from.
fn verdict_cmd(a: &[&str]) {
    let dir = a[0];
    let mut bytes = 0u64;
    for s in ["events", "clicks", "links"] {
        if let Ok(l) = input::scan(format!("{dir}/{s}").as_bytes()) {
            bytes += l.files().iter().chain(l.dots()).map(|e| e.size).sum::<u64>();
        }
    }
    let need = bytes as usize / 10 * 7;
    let v = arena::huge_verdict(need);
    let budget = if v.enabled == b'n' {
        0
    } else if v.use_huge {
        usize::MAX
    } else {
        (v.free_2m.max(0) as usize) * (2 << 20) / 5 * 4
    };
    let mi = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let g = |k: &str| mi.lines().find_map(|l| l.strip_prefix(k).map(|v| v.trim().trim_end_matches(" kB").trim().parse::<u64>().unwrap_or(0) >> 10)).unwrap_or(0);
    let normal = std::fs::read_to_string("/proc/buddyinfo").unwrap_or_default().lines().find(|l| l.contains("Normal")).unwrap_or("").to_string();
    println!(
        "input {} MB need {} MB (x1.25 = {} MB); thp '{}' defrag '{}' may_stall {}; free 2 MB blocks Normal+Movable {} ({} MB), all zones {}; MemFree {} MB MemAvailable {} MB; use_huge {} (reason '{}'); allocator budget {}",
        bytes >> 20,
        need >> 20,
        (need >> 20) * 5 / 4,
        v.enabled as char,
        v.defrag as char,
        v.may_stall,
        v.free_2m,
        v.free_2m * 2,
        v.free_2m_all,
        g("MemFree:"),
        g("MemAvailable:"),
        v.use_huge,
        v.reason as char,
        if budget == usize::MAX { "all".to_string() } else { format!("{} MB", budget >> 20) }
    );
    for req in [64usize << 20, 512 << 20] {
        println!("  verdict for a {} MB table: use_huge {}", req >> 20, arena::huge_verdict(req).use_huge);
    }
    println!("  buddyinfo {normal}");
}

/// Fragments free memory on purpose: maps GB with 4 KB pages, touches it all, then frees every
/// page but KEEP per 2 MB block and holds the rest until killed. Free memory stays large while
/// free 2 MB blocks vanish (the "16 GB free and the verdict says no" case).
fn fragment(a: &[&str]) {
    let gb: usize = a[0].parse().unwrap();
    let keep: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(1).clamp(1, 512);
    let n = start(0);
    pool::wait_running();
    let len = gb << 30;
    let p = sys::map_anon(len).unwrap() as usize;
    unsafe { sys::madvise(p, len, sys::MADV_NOHUGEPAGE).unwrap() };
    let parts = (len >> 26) as u32;
    let t0 = mono();
    let touch = |_w: usize, i: u32| arena::touch_range(p + ((i as usize) << 26), 1 << 26, true);
    pool::run(3, parts, &touch);
    let t1 = mono();
    let step = 512 / keep;
    let free = |_w: usize, i: u32| {
        let base = p + ((i as usize) << 26);
        for b in 0..32 {
            let blk = base + (b << 21);
            for k in 0..keep {
                let lo = blk + (k * step + 1) * 4096;
                let hi = blk + ((k + 1) * step) * 4096;
                if hi > lo {
                    unsafe {
                        let _ = sys::madvise(lo, hi - lo, sys::MADV_DONTNEED);
                    }
                }
            }
        }
    };
    pool::run(3, parts, &free);
    let t2 = mono();
    println!("fragmented {gb} GB on {n} workers: touched in {} ms, freed all but {keep} page(s) per 2 MB block in {} ms; holding {} MB; pid {}", (t1 - t0) / 1_000_000, (t2 - t1) / 1_000_000, (len / 512 * keep) >> 20, sys::getpid());
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

/// The topology reader on fabricated sysfs trees (bare-metal layouts a VM cannot show).
fn topo_test() {
    let root = "/tmp/rt-topo";
    let make = |name: &str, n: usize, sib: &dyn Fn(usize) -> String| -> String {
        let d = format!("{root}/{name}");
        let _ = std::fs::remove_dir_all(&d);
        for c in 0..n {
            std::fs::create_dir_all(format!("{d}/cpu{c}/topology")).unwrap();
            std::fs::write(format!("{d}/cpu{c}/topology/thread_siblings_list"), sib(c) + "\n").unwrap();
        }
        d
    };
    let set = |cpus: &[usize]| {
        let mut s = cpu::CpuSet::EMPTY;
        for &c in cpus {
            s.add(c);
        }
        s
    };
    let show = |t: &cpu::Topo| -> String { (0..t.n).map(|i| format!("{}:{}", t.cpu[i], t.core[i])).collect::<Vec<_>>().join(" ") };
    // Intel numbering, 16 CPUs on 8 cores: siblings i and i+8; a cpuset of 4 cores x 2 threads.
    let intel = make("intel", 16, &|c| format!("{},{}", c % 8, c % 8 + 8));
    let t = cpu::topology(&set(&[0, 1, 2, 3, 8, 9, 10, 11]), intel.as_bytes());
    println!("  intel 0-3,8-11: {} | cores {} known {}", show(&t), t.cores, t.known);
    check(t.cores == 4 && t.known && t.cpu[..8] == [0, 8, 1, 9, 2, 10, 3, 11] && t.core[..8] == [0, 0, 1, 1, 2, 2, 3, 3], "Intel-style siblings (i, i+8): 8 allowed CPUs are 4 cores, siblings adjacent");
    // AMD numbering: siblings 2k, 2k+1 (range list).
    let amd = make("amd", 16, &|c| format!("{}-{}", c & !1, (c & !1) + 1));
    let t = cpu::topology(&set(&[0, 1, 2, 3, 4, 5, 6, 7]), amd.as_bytes());
    println!("  amd 0-7: {} | cores {}", show(&t), t.cores);
    check(t.cores == 4 && t.core[..8] == [0, 0, 1, 1, 2, 2, 3, 3], "AMD-style siblings (2k, 2k+1): 8 allowed CPUs are 4 cores");
    // A cpuset that takes one thread of each core: 8 cores.
    let t = cpu::topology(&set(&[0, 1, 2, 3, 4, 5, 6, 7]), intel.as_bytes());
    println!("  intel 0-7: {} | cores {}", show(&t), t.cores);
    check(t.cores == 8, "Intel-style, one thread per core allowed: 8 cores (siblings outside the set ignored)");
    // No sysfs: ascending order, one core per CPU, known = false.
    let t = cpu::topology(&set(&[2, 3, 5]), b"/nonexistent");
    check(!t.known && t.cores == 3 && t.cpu[..3] == [2, 3, 5], "unreadable sysfs: ascending order, each CPU its own core, known = false");
    let _ = std::fs::remove_dir_all(root);
    println!("topo-test: all passed");
}

/// The dedup pattern on a block arena: per worker, PARTS partitions of input chunks; for each,
/// free its chunks, then allocate half as many output chunks and write them. 8 workers.
fn blocks_bench(a: &[&str]) {
    let bsize: usize = a.first().map(|s| s.parse().unwrap()).unwrap_or(8192);
    let per: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(12_500);
    let parts: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(64);
    let cap: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(256);
    let n = start(0);
    pool::wait_running();
    REG_A.init(32 << 30, arena::Huge::Auto(n * per * bsize * 2)).unwrap();
    BLOCKS.attach(&REG_A, bsize).unwrap();
    BLOCKS.set_local_max(cap);
    let input: Vec<Vec<AtomicUsize>> = (0..n).map(|_| (0..per).map(|_| AtomicUsize::new(0)).collect()).collect();
    let output: Vec<Vec<AtomicUsize>> = (0..n).map(|_| (0..per).map(|_| AtomicUsize::new(0)).collect()).collect();
    // One task per worker, each on its own worker (tasks wait for all to arrive).
    let each = |f: &(dyn Fn(usize) + Sync)| {
        let arrived = AtomicUsize::new(0);
        let g = |w: usize, _i: u32| {
            arrived.fetch_add(1, AcqRel);
            while arrived.load(Acquire) < n {
                std::hint::spin_loop();
            }
            f(w);
        };
        pool::run(3, n as u32, &g);
    };
    // The sink fills the input chunks (fresh memory, faulted once).
    each(&|w| unsafe {
        for k in 0..per {
            let p = BLOCKS.alloc(w);
            std::ptr::write_bytes(p, 1, bsize);
            input[w][k].store(p as usize, Relaxed);
        }
    });
    let (s0, r0) = BLOCKS.lock_ops();
    let f0 = sys::getrusage(sys::RUSAGE_SELF).minflt;
    let (_, _, carved0) = BLOCKS.stats();
    let t0 = mono();
    let kp = per / parts;
    each(&|w| unsafe {
        let mut o = 0;
        for p in 0..parts {
            for k in p * kp..(p + 1) * kp {
                BLOCKS.free(w, input[w][k].load(Relaxed) as *mut u8);
            }
            for _ in 0..kp / 2 {
                let q = BLOCKS.alloc(w);
                std::ptr::write_volatile(q, 2);
                output[w][o].store(q as usize, Relaxed);
                o += 1;
            }
        }
    });
    let t1 = mono();
    let f1 = sys::getrusage(sys::RUSAGE_SELF).minflt;
    let (s1, r1) = BLOCKS.lock_ops();
    let (_, _, carved1) = BLOCKS.stats();
    let ops = n * (parts * kp + parts * (kp / 2));
    println!(
        "block {} B, {} chunks per worker in {} partitions, cap {} blocks ({} KB): {:.2} ms for {} frees+allocs on {n} workers ({:.1} ns per op per worker); shared-list locks: {} spills {} refills; fresh blocks {}; faults {}",
        bsize,
        per,
        parts,
        cap,
        cap * bsize >> 10,
        (t1 - t0) as f64 / 1e6,
        ops,
        (t1 - t0) as f64 * n as f64 / ops as f64,
        s1 - s0,
        r1 - r0,
        carved1 - carved0,
        f1 - f0
    );
}

// ------------------------------------------------------------------------------------------
// Pool semantics.

thread_local! {
    static TL: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

fn check(ok: bool, what: &str) {
    if ok {
        println!("  ok   {what}");
    } else {
        println!("  FAIL {what}");
        std::process::exit(1);
    }
}

fn pool_test() {
    let n = start(0);
    pool::wait_running();
    println!("pool-test: {n} workers, pinned {}, topology known {}, musl tcb {}", pool::pinned(), pool::topology_known(), rt::tls::musl_layout_ok());

    // 1. Exactly once.
    let tasks: u32 = 1_000_003;
    let bits: Vec<AtomicU64> = (0..tasks.div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
    let dup = AtomicU64::new(0);
    let badw = AtomicU64::new(0);
    let f = |w: usize, i: u32| {
        if pool::me() != w {
            badw.fetch_add(1, Relaxed);
        }
        let b = 1u64 << (i % 64);
        if bits[(i / 64) as usize].fetch_or(b, Relaxed) & b != 0 {
            dup.fetch_add(1, Relaxed);
        }
    };
    let t0 = mono();
    pool::run(3, tasks, &f);
    let dt = mono() - t0;
    let missing: u64 = (0..tasks).filter(|&i| bits[(i / 64) as usize].load(Relaxed) >> (i % 64) & 1 == 0).count() as u64;
    check(dup.load(Relaxed) == 0 && missing == 0, &format!("1,000,003 tasks each ran exactly once ({:.1} ns per task)", dt as f64 / tasks as f64));
    check(badw.load(Relaxed) == 0, "me() equals the worker index passed to every task");
    let per: Vec<u64> = (0..n).map(|w| pool::stats(w).0).collect();
    println!("       tasks per worker {per:?}");

    // 2. Priorities: a high-priority queue pushed from inside a low one overtakes it.
    let l_started = AtomicU32::new(0);
    let at_push = AtomicU32::new(0);
    let h_max = AtomicU32::new(0);
    let hq = AtomicU64::new(u64::MAX);
    let spin = |us: u64| {
        let t = mono();
        while mono() - t < us * 1000 {
            std::hint::spin_loop();
        }
    };
    let h = |_w: usize, _i: u32| {
        h_max.fetch_max(l_started.load(Relaxed), Relaxed);
        spin(20);
    };
    let l = |_w: usize, i: u32| {
        l_started.fetch_add(1, Relaxed);
        if i == 100 {
            at_push.store(l_started.load(Relaxed), Relaxed);
            let q = unsafe { pool::push(1, 400, &h) };
            hq.store(unsafe { std::mem::transmute::<pool::Q, u64>(q) }, Release);
        }
        spin(20);
    };
    let lq = unsafe { pool::push(5, 2000, &l) };
    while hq.load(Acquire) == u64::MAX {
        std::hint::spin_loop();
    }
    pool::wait(unsafe { std::mem::transmute::<u64, pool::Q>(hq.load(Acquire)) });
    pool::wait(lq);
    let overtaken = h_max.load(Relaxed) - at_push.load(Relaxed);
    check(overtaken as usize <= 2 * n, &format!("high-priority queue of 400 pushed mid-drain: only {overtaken} low tasks started before its last task started (bound {})", 2 * n));

    // 3. Chained stages without a barrier: A's continuation pushes B, B's sets an event.
    let a_done = AtomicU32::new(0);
    let b_early = AtomicU32::new(0);
    let b_done = AtomicU32::new(0);
    let ev = pool::Event::new();
    let fb = |_w: usize, _i: u32| {
        if a_done.load(Acquire) != 64 {
            b_early.fetch_add(1, Relaxed);
        }
        b_done.fetch_add(1, Relaxed);
    };
    let db = |_w: usize| ev.set();
    let fa = |_w: usize, _i: u32| {
        spin(5);
        a_done.fetch_add(1, Release);
    };
    let da = |_w: usize| unsafe {
        pool::push_then(2, 64, &fb, &db);
    };
    unsafe { pool::push_then(2, 64, &fa, &da) };
    pool::wait_event(&ev);
    check(b_early.load(Relaxed) == 0 && b_done.load(Relaxed) == 64, "continuation chain: B started only after all of A, event set after B");

    // 4. Fork-join inside tasks.
    let inner_sum = AtomicU64::new(0);
    let outer = |_w: usize, i: u32| {
        let g = |_w: usize, j: u32| {
            inner_sum.fetch_add((i * 1000 + j) as u64, Relaxed);
        };
        pool::run(2, 100, &g);
    };
    pool::run(3, 16, &outer);
    let want: u64 = (0..16u64).map(|i| (0..100u64).map(|j| i * 1000 + j).sum::<u64>()).sum();
    check(inner_sum.load(Relaxed) == want, "nested push + wait inside tasks");

    // 5. Join of two queues.
    let join = pool::Join::new(2);
    let ev2 = pool::Event::new();
    let cnt = AtomicU32::new(0);
    let fj = |_w: usize, _i: u32| {
        cnt.fetch_add(1, Relaxed);
    };
    let dj = |_w: usize| {
        if join.arrive() {
            ev2.set();
        }
    };
    unsafe {
        pool::push_then(4, 300, &fj, &dj);
        pool::push_then(6, 500, &fj, &dj);
    }
    pool::wait_event(&ev2);
    check(cnt.load(Relaxed) == 800, "join of two queues sets the event once both finish");

    // 6. Zero tasks, more live queues than slots of one priority.
    let ran = AtomicU32::new(0);
    let dz = |_w: usize| {
        ran.fetch_add(1, Relaxed);
    };
    let fz = |_w: usize, _i: u32| {};
    let qz = unsafe { pool::push_then(0, 0, &fz, &dz) };
    check(pool::is_done(qz) && ran.load(Relaxed) == 1, "a queue of zero tasks runs its continuation at once");
    let many = AtomicU32::new(0);
    let fm = |_w: usize, _i: u32| {
        spin(2);
        many.fetch_add(1, Relaxed);
    };
    let qs: Vec<pool::Q> = (0..40).map(|_| unsafe { pool::push(7, 50, &fm) }).collect();
    for q in qs {
        pool::wait(q);
    }
    check(many.load(Relaxed) == 2000, "40 live queues at one priority (8 slots): pushes wait for slots by helping");

    // 7. Rust TLS with a destructor, heap use and errno on raw workers.
    let tl_bad = AtomicU32::new(0);
    let errno_ok = AtomicU32::new(0);
    let ft = |w: usize, i: u32| {
        TL.with(|v| {
            let mut v = v.borrow_mut();
            if let Some(&last) = v.last() {
                if (last >> 32) as usize != w {
                    tl_bad.fetch_add(1, Relaxed);
                }
            }
            v.push(((w as u64) << 32) | i as u64);
        });
        if i % 97 == 0 {
            match std::fs::metadata("/nonexistent-rt-check") {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    errno_ok.fetch_add(1, Relaxed);
                }
                _ => {}
            }
        }
    };
    pool::run(3, 9700, &ft);
    check(tl_bad.load(Relaxed) == 0, "thread_local! Vec on workers: each worker sees only its own");
    check(errno_ok.load(Relaxed) == 100, "errno on workers (std::fs::metadata of a missing path is NotFound)");

    // 8. Sleep and wake.
    std::thread::sleep(std::time::Duration::from_millis(30));
    let sleeps0: u64 = (0..n).map(|w| pool::stats(w).1).sum();
    let c8 = AtomicU32::new(0);
    let f8 = |_w: usize, _i: u32| {
        spin(50);
        c8.fetch_add(1, Relaxed);
    };
    let t0 = mono();
    pool::run(3, (n * 4) as u32, &f8);
    let dt = mono() - t0;
    check(c8.load(Relaxed) as usize == n * 4 && sleeps0 >= (n - 1) as u64, &format!("sleeping workers wake for new work ({} sleeps so far; {} tasks of 50 us took {} us)", sleeps0, n * 4, dt / 1000));
    println!("pool-test: all passed");
}

// ------------------------------------------------------------------------------------------
// Pool measurements.

fn pool_bench(a: &[&str]) {
    let want: usize = a.first().map(|s| s.parse().unwrap()).unwrap_or(0);
    let r0 = sys::getrusage(sys::RUSAGE_SELF);
    let t0 = mono();
    let n = start(want);
    let t1 = mono();
    pool::wait_running();
    let t2 = mono();
    tline("running");
    println!("workers {n}: start() returned after {} us, all running after {} us (ticks {})", (t1 - t0) / 1000, (t2 - t0) / 1000, clock::ticks_to_ns(pool::start_to_all_running_ticks()) / 1000);

    // Cost of taking a task: empty tasks.
    for &(label, tasks) in &[("empty tasks", 10_000_000u32)] {
        let f = |_w: usize, _i: u32| {};
        let mut best = u64::MAX;
        for _ in 0..5 {
            let t = mono();
            pool::run(3, tasks, &f);
            best = best.min(mono() - t);
        }
        println!("{label}: {} tasks in {} us best of 5: {:.1} ns per task wall, {:.1} ns per task per worker", tasks, best / 1000, best as f64 / tasks as f64, best as f64 * n as f64 / tasks as f64);
    }
    // One worker alone taking from the counter (uncontended): run on a queue while others are
    // busy elsewhere is hard to arrange; instead time the caller alone with a 1-worker pool via
    // `pool-bench 1`.

    if n < 2 {
        println!("fewer than 2 workers: wake-latency and queue tests skipped (they need a worker besides the caller)");
        return;
    }
    // Wake latency: push -> first task start, workers spinning vs sleeping.
    let lat = |label: &str, idle_ms: u64| {
        let mut v = Vec::new();
        for _ in 0..20 {
            if idle_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(idle_ms));
            } else {
                let f = |_w: usize, _i: u32| {};
                pool::run(3, 64, &f);
            }
            let first = AtomicU64::new(0);
            let all = AtomicU64::new(0);
            let started = AtomicUsize::new(0);
            let me0 = AtomicUsize::new(0);
            let f = |w: usize, _i: u32| {
                let t = mono();
                let _ = first.compare_exchange(0, t, Relaxed, Relaxed);
                if w == 0 {
                    me0.fetch_add(1, Relaxed);
                }
                if started.fetch_add(1, Relaxed) + 1 == n {
                    all.store(t, Relaxed);
                }
                let s = mono();
                while mono() - s < 200_000 {
                    std::hint::spin_loop();
                }
            };
            let t = mono();
            let q = unsafe { pool::push(3, n as u32, &f) };
            // The caller does not help, so the first task start measures a worker's wake.
            while !pool::is_done(q) {
                std::hint::spin_loop();
            }
            v.push(((first.load(Relaxed) - t) / 1000, (all.load(Relaxed).max(t) - t) / 1000));
        }
        v.sort();
        let firsts: Vec<u64> = v.iter().map(|x| x.0).collect();
        let mut alls: Vec<u64> = v.iter().map(|x| x.1).collect();
        alls.sort();
        println!("{label}: push to first task start us min {} median {} max {}; to all {n} tasks started (w0 not helping, so n-1 workers + reuse) median {} max {}", firsts[0], firsts[10], firsts[19], alls[10], alls[19]);
    };
    lat("workers spinning", 0);
    lat("workers asleep (idle 20 ms)", 20);

    // Context switches of a representative run: 50 queues of 1000 tasks of ~10 us.
    let ra = sys::getrusage(sys::RUSAGE_SELF);
    let t = mono();
    for _ in 0..50 {
        let f = |_w: usize, _i: u32| {
            let s = mono();
            while mono() - s < 10_000 {
                std::hint::spin_loop();
            }
        };
        pool::run(3, 1000, &f);
    }
    let dt = mono() - t;
    let rb = sys::getrusage(sys::RUSAGE_SELF);
    println!(
        "50 queues x 1000 tasks x 10 us: {} ms wall ({:.2} x ideal), voluntary cs {}, involuntary cs {}",
        dt / 1_000_000,
        dt as f64 / (50_000.0 * 10_000.0 / n as f64),
        rb.nvcsw - ra.nvcsw,
        rb.nivcsw - ra.nivcsw
    );
    let r1 = sys::getrusage(sys::RUSAGE_SELF);
    println!("whole bench: minflt {} vcsw {} ivcsw {}", r1.minflt - r0.minflt, r1.nvcsw - r0.nvcsw, r1.nivcsw - r0.nivcsw);
}

// ------------------------------------------------------------------------------------------
// Input.

static mut INPUTS: std::mem::MaybeUninit<input::Inputs> = std::mem::MaybeUninit::uninit();

fn map_all(dir: &str, register: bool) -> &'static input::Inputs {
    let ls: Vec<input::Listing> = ["events", "clicks", "links"].iter().map(|s| input::scan(format!("{dir}/{s}").as_bytes()).expect("scan")).collect();
    input::set_keep_fds(ls.iter().map(|l| l.files().len()).sum::<usize>() < 900);
    let ls: &'static [input::Listing] = Box::leak(ls.into_boxed_slice());
    let streams: Vec<input::Stream> = ls.iter().map(|l| input::Stream { listing: l, dots: false }).collect();
    #[allow(static_mut_refs)]
    let slot = unsafe { &mut INPUTS };
    input::map(&streams, slot, register).expect("map")
}

fn map_cmd(a: &[&str]) {
    let dir = a[0];
    let mode = a.get(1).copied().unwrap_or("plain");
    let rel = a.get(2).copied().unwrap_or("norelease") == "release";
    let premap = std::env::var("RT_PREMAP").is_ok();
    let t_entry = mono();
    let (n, inp, t_pool, t_mapped);
    if premap {
        let t = mono();
        inp = map_all(dir, true);
        t_mapped = mono();
        n = start(0);
        t_pool = t_entry + (mono() - t_mapped);
        let _ = t;
    } else {
        n = start(0);
        t_pool = mono();
        inp = map_all(dir, true);
        t_mapped = mono();
    }
    clock::mark("mapped");
    tline("mapped");
    let fa = input::probe_fault_around(inp);
    let t_probe = mono();
    let d = inp.detail.map(clock::ticks_to_ns);
    println!("map detail us: openat sum {} max {}, mmap sum {} max {}", d[0] / 1000, d[2] / 1000, d[1] / 1000, d[3] / 1000);

    println!(
        "{} files, {} MB; pool start {} us, scan+map {} us ({} us in map()), probe {} us: fault-around {:?}",
        inp.files().len(),
        inp.bytes() >> 20,
        (t_pool - t_entry) / 1000,
        (t_mapped - t_pool) / 1000,
        clock::ticks_to_ns(inp.map_ticks) / 1000,
        (t_probe - t_mapped) / 1000,
        fa
    );
    // Checks: content equals pread; guard zeros readable around every file; owned() partitions.
    let mut bad = 0;
    for f in inp.files() {
        let b = f.bytes();
        let mut tmp = vec![0u8; 4096];
        for &off in &[0usize, b.len() / 2, b.len().saturating_sub(4096)] {
            if f.fd < 0 {
                break;
            }
            let k = sys::pread_full(f.fd, &mut tmp, off as u64).unwrap();
            if b[off..off + k] != tmp[..k] {
                bad += 1;
            }
        }
        unsafe {
            for k in 0..4096 {
                if *f.ptr.add(f.len + k) != 0 || *f.ptr.sub(k + 1) != 0 && f.ptr as usize == inp.reservation().0 + (2 << 20) {
                    bad += 1;
                    break;
                }
            }
            // Bytes before a file (other than the lead) are the previous slot's zero tail.
            if f.ptr as usize % (2 << 20) != 0 {
                bad += 1;
            }
            let _ = std::ptr::read_volatile(f.ptr.sub(4096));
        }
    }
    check(bad == 0, "mapped bytes equal pread; 4096 zero bytes after every file, readable before; 2 MB aligned");
    let r_before = sys::getrusage(sys::RUSAGE_SELF);

    // 8-worker touch of every page, chunks of 4 MB.
    const CH: usize = 4 << 20;
    let mut chunks: Vec<(usize, usize, usize)> = Vec::new();
    for (g, f) in inp.files().iter().enumerate() {
        let mut s = 0;
        while s < f.len {
            let e = (s + CH).min(f.len);
            chunks.push((g, s, e));
            s = e;
        }
    }
    let reader = input::Reader::new(n, 128 << 10).unwrap();
    let sum = AtomicU64::new(0);
    let task = |w: usize, i: u32| {
        let (g, s, e) = chunks[i as usize];
        let f = inp.file(g);
        let mut acc = 0u64;
        match mode {
            "pread" => {
                let mut ls = reader.lines(w, f, s, e);
                while let Some(sp) = ls.next_span().unwrap() {
                    let b = sp.bytes();
                    let mut k = 0;
                    while k < b.len() {
                        acc += b[k] as u64;
                        k += 4096;
                    }
                }
            }
            _ => {
                if mode == "seq" {
                    let _ = inp.advise(g, s, e, input::Advice::Sequential);
                } else if mode == "populate" {
                    let _ = inp.advise(g, s, e, input::Advice::PopulateRead);
                }
                let mut p = s;
                while p < e {
                    acc += unsafe { std::ptr::read_volatile(f.ptr.add(p)) } as u64;
                    p += 4096;
                }
            }
        }
        if rel {
            inp.consumed(g, s, e);
        }
        sum.fetch_add(acc, Relaxed);
    };
    let t0 = mono();
    pool::run(3, chunks.len() as u32, &task);
    if rel {
        inp.flush();
    }
    let t1 = mono();
    let r_after = sys::getrusage(sys::RUSAGE_SELF);
    let (rb, rc) = inp.release_stats();
    println!(
        "touch {mode} release={rel}: {} chunks, {:.2} ms, minflt {}, majflt {}, sys {:.1} ms, user {:.1} ms; released {} MB in {} calls (sum {})",
        chunks.len(),
        (t1 - t0) as f64 / 1e6,
        r_after.minflt - r_before.minflt,
        r_after.majflt - r_before.majflt,
        (r_after.sys_us() - r_before.sys_us()) as f64 / 1000.0,
        (r_after.user_us() - r_before.user_us()) as f64 / 1000.0,
        rb >> 20,
        rc,
        sum.load(Relaxed) % 1000
    );
    tline("done");
    let how = a.get(3).copied().unwrap_or("all");
    let r = match how {
        "none" => exit::Release::NONE,
        _ => exit::Release::ALL,
    };
    let t2 = mono();
    exit::release(&r);
    let t3 = mono();
    println!("exit release ({how}): {:.2} ms", (t3 - t2) as f64 / 1e6);
    tline("released");
    sys::exit_group(0);
}

/// TLB shootdown interrupts so far: (sum over the CPUs in `cpus`, sum over all CPUs).
fn tlb_irqs(cpus: &[usize]) -> (u64, u64) {
    let t = std::fs::read_to_string("/proc/interrupts").unwrap_or_default();
    for l in t.lines() {
        if l.trim_start().starts_with("TLB:") {
            let v: Vec<u64> = l.split_whitespace().skip(1).filter_map(|x| x.parse().ok()).collect();
            let mine = cpus.iter().filter_map(|&c| v.get(c)).sum();
            return (mine, v.iter().sum());
        }
    }
    (0, 0)
}

/// Every byte of every file read (u64 sums) by 8 workers in 4 MB chunks, with release-behind
/// in runs of `min..max` bins (`off`: none, release at the end in parallel instead).
fn release_bench(a: &[&str]) {
    let dir = a[0];
    let (min, max): (u32, u32) = match a.get(1).copied().unwrap_or("16:32") {
        "off" => (0, 0),
        s => {
            let mut it = s.split(':');
            (it.next().unwrap().parse().unwrap(), it.next().unwrap().parse().unwrap())
        }
    };
    let n = start(0);
    pool::wait_running();
    let inp = map_all(dir, true);
    if min == 0 {
        inp.set_release(false);
    } else {
        inp.set_runs(min, max);
    }
    const CH: usize = 4 << 20;
    let mut chunks: Vec<(usize, usize, usize)> = Vec::new();
    for (g, f) in inp.files().iter().enumerate() {
        let mut s = 0;
        while s < f.len {
            let e = (s + CH).min(f.len);
            chunks.push((g, s, e));
            s = e;
        }
    }
    let cpus: Vec<usize> = (0..n).map(|w| pool::cpu_of(w) as usize).filter(|&c| c < 4096).collect();
    let sum = AtomicU64::new(0);
    let task = |_w: usize, i: u32| {
        let (g, s, e) = chunks[i as usize];
        let f = inp.file(g);
        let b = &f.bytes()[s..e];
        let (pre, mid, post) = unsafe { b.align_to::<u64>() };
        let mut acc = 0u64;
        for &x in mid {
            acc = acc.wrapping_add(x);
        }
        for &x in pre.iter().chain(post) {
            acc = acc.wrapping_add(x as u64);
        }
        inp.consumed(g, s, e);
        sum.fetch_add(acc, Relaxed);
    };
    let i0 = tlb_irqs(&cpus);
    let r0 = sys::getrusage(sys::RUSAGE_SELF);
    let t0 = mono();
    pool::run(3, chunks.len() as u32, &task);
    let t1 = mono();
    inp.flush();
    let t2 = mono();
    let i1 = tlb_irqs(&cpus);
    let r1 = sys::getrusage(sys::RUSAGE_SELF);
    let t3 = mono();
    let rel = exit::release(&exit::Release { inputs: min == 0, arenas: false, ..exit::Release::NONE });
    let t4 = mono();
    let i2 = tlb_irqs(&cpus);
    let (rb, rc) = inp.release_stats();
    println!(
        "runs {}: read {} MB in {:.2} ms + flush {:.2} ms + end release {:.2} ms ({} MB) = {:.2} ms; madvise calls {rc} ({} MB); TLB irqs on our cpus {} (all cpus {}) during read, {} at end; sys {:.1} ms user {:.1} ms minflt {} majflt {}",
        a.get(1).copied().unwrap_or("16:32"),
        inp.bytes() >> 20,
        (t1 - t0) as f64 / 1e6,
        (t2 - t1) as f64 / 1e6,
        (t4 - t3) as f64 / 1e6,
        rel >> 20,
        (t2 - t0 + t4 - t3) as f64 / 1e6,
        rb >> 20,
        i1.0 - i0.0,
        i1.1 - i0.1,
        i2.1 - i1.1,
        (r1.sys_us() - r0.sys_us()) as f64 / 1000.0,
        (r1.user_us() - r0.user_us()) as f64 / 1000.0,
        r1.minflt - r0.minflt,
        r1.majflt - r0.majflt
    );
    let _ = sum.load(Relaxed);
    sys::exit_group(0);
}

fn reader_test(a: &[&str]) {
    // Synthetic edge files next to the real data.
    let syn = "/tmp/rt-reader-syn";
    let _ = std::fs::remove_dir_all(syn);
    for s in ["events", "clicks", "links"] {
        std::fs::create_dir_all(format!("{syn}/{s}")).unwrap();
    }
    let mut long = String::from("h1,h2\n");
    for k in 0..50 {
        long.push_str(&"x".repeat(k * 997 % 300_000));
        long.push('\n');
    }
    std::fs::write(format!("{syn}/events/a-long.csv"), &long).unwrap();
    std::fs::write(format!("{syn}/events/b-nofinal.csv"), "h\nline1\nline2-no-newline").unwrap();
    std::fs::write(format!("{syn}/events/c-empty.csv"), "").unwrap();
    std::fs::write(format!("{syn}/events/d-header.csv"), "event_id,user_id\n").unwrap();
    std::fs::write(format!("{syn}/events/e-newlines.csv"), "\n\n\n\n").unwrap();
    std::fs::write(format!("{syn}/events/f-one.csv"), "x").unwrap();
    std::fs::write(format!("{syn}/events/.g-dot.csv"), "dot\n").unwrap();
    std::fs::write(format!("{syn}/events/h-notcsv.txt"), "no\n").unwrap();
    let n = start(0);
    pool::wait_running();
    let dir = a.first().copied().unwrap_or(syn);
    let ls: Vec<input::Listing> = [syn, dir].iter().flat_map(|d| ["events", "clicks", "links"].map(|s| input::scan(format!("{d}/{s}").as_bytes()).unwrap())).collect();
    check(ls[0].files().len() == 6 && ls[0].dots().len() == 1 && ls[0].dots()[0].name() == b".g-dot.csv", "scan: 6 csv files, the dotfile reported apart, the .txt ignored");
    let ls: &'static [input::Listing] = Box::leak(ls.into_boxed_slice());
    let streams: Vec<input::Stream> = ls.iter().map(|l| input::Stream { listing: l, dots: true }).collect();
    #[allow(static_mut_refs)]
    let inp = input::map(&streams, unsafe { &mut INPUTS }, false).unwrap();
    let readers = [input::Reader::new(n, 128 << 10).unwrap(), input::Reader::new(n, 4096).unwrap()];
    let bad = AtomicU64::new(0);
    let kinds: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
    let first = std::sync::Mutex::new(Vec::<String>::new());
    let note = |k: usize, m: String| {
        bad.fetch_add(1, Relaxed);
        if kinds[k].fetch_add(1, Relaxed) == 0 {
            first.lock().unwrap().push(m);
        }
    };
    let spans = AtomicU64::new(0);
    let splits = AtomicU64::new(0);
    let checked = AtomicU64::new(0);
    let task = |w: usize, g: u32| {
        let f = inp.file(g as usize);
        let data = f.bytes();
        let mut rng = 0x9e3779b97f4a7c15u64 ^ g as u64;
        for round in 0..4 {
            // A random partition of [0, len) into chunks of 1 byte to 8 MB.
            let mut cuts = vec![0usize];
            while *cuts.last().unwrap() < data.len() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let max = if round == 0 { 64 } else { 8 << 20 };
                let step = 1 + (rng as usize) % max;
                cuts.push((cuts.last().unwrap() + step).min(data.len()));
            }
            let mut covered = 0usize;
            for c in cuts.windows(2) {
                let (s, e) = input::owned(data, c[0], c[1]);
                if s != e {
                    if s != covered {
                        note(0, format!("tile: file {g} len {} chunk {:?} owned ({s},{e}) covered {covered}", data.len(), c));
                    }
                    covered = e;
                }
                let r = &readers[round % 2];
                let mut ls = r.lines(w, f, c[0], c[1]);
                let mut got: Vec<u8> = Vec::new();
                while let Some(sp) = ls.next_span().unwrap() {
                    spans.fetch_add(1, Relaxed);
                    if ls.split {
                        splits.fetch_add(1, Relaxed);
                    }
                    got.extend_from_slice(sp.bytes());
                    unsafe {
                        for k in 0..sp.post {
                            if *sp.ptr.add(sp.len + k) != 0 {
                                note(1, format!("guard: file {g} chunk {:?}", c));
                                break;
                            }
                        }
                        let _ = std::ptr::read_volatile(sp.ptr.sub(sp.pre));
                    }
                    if !ls.split && sp.len > 0 && sp.bytes()[sp.len - 1] != b'\n' && c[0] + 0 < data.len() && !(sp.ptr as usize == 0) {
                        // A span not ending in a newline must end at EOF.
                        if !data.ends_with(sp.bytes()) {
                            note(2, format!("noeol: file {g} chunk {:?} span {}", c, sp.len));
                        }
                    }
                }
                if got != data[s..e] {
                    note(3, format!("content: file {g} len {} chunk {:?} owned ({s},{e}) got {} bytes round {round}", data.len(), c, got.len()));
                }
                checked.fetch_add(1, Relaxed);
            }
            if covered != data.len() && !data.is_empty() {
                note(0, format!("tile end: file {g} len {} covered {covered}", data.len()));
            }
        }
    };
    pool::run(3, inp.files().len() as u32, &task);
    println!("  kinds tile/guard/noeol/content: {:?}", kinds.iter().map(|k| k.load(Relaxed)).collect::<Vec<_>>());
    for m in first.lock().unwrap().iter() {
        println!("  first: {m}");
    }
    check(bad.load(Relaxed) == 0, &format!("owned() tiles every file; reader spans equal the mapped bytes of {} chunks of {} files ({} spans, {} split pieces of long lines), zero guards after every span", checked.load(Relaxed), inp.files().len(), spans.load(Relaxed), splits.load(Relaxed)));
    println!("reader-test: all passed");
}

// ------------------------------------------------------------------------------------------
// Arenas.

static REG_A: arena::Region = arena::Region::new();
static REG_B: arena::Region = arena::Region::new();
static BLOCKS: arena::Blocks = arena::Blocks::new();
static BUMP: arena::Bump = arena::Bump::new();

fn arena_test() {
    let n = start(0);
    pool::wait_running();
    REG_A.init(8 << 30, arena::Huge::Off).unwrap();
    BLOCKS.attach(&REG_A, 64 << 10).unwrap();
    BUMP.attach(&REG_A);
    // Round 1: every worker allocates 64 blocks of 64 KB and touches every page.
    let per = 64;
    let ptrs: Vec<AtomicUsize> = (0..n * per).map(|_| AtomicUsize::new(0)).collect();
    let alloc = |w: usize, _i: u32| unsafe {
        for k in 0..per {
            let p = BLOCKS.alloc(w);
            arena::touch_range(p as usize, 64 << 10, false);
            ptrs[w * per + k].store(p as usize, Relaxed);
        }
    };
    let free = |w: usize, _i: u32| unsafe {
        for k in 0..per {
            BLOCKS.free(w, ptrs[w * per + k].load(Relaxed) as *mut u8);
        }
    };
    let one_each = |f: &(dyn Fn(usize, u32) + Sync)| {
        // One task per worker, each worker exactly one: tasks that wait for all to arrive.
        let arrived = AtomicUsize::new(0);
        let seen: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
        let g = |w: usize, i: u32| {
            arrived.fetch_add(1, AcqRel);
            while arrived.load(Acquire) < n {
                std::hint::spin_loop();
            }
            seen[w].fetch_add(1, Relaxed);
            f(w, i);
        };
        pool::run(3, n as u32, &g);
        assert!(seen.iter().all(|s| s.load(Relaxed) == 1));
    };
    one_each(&alloc);
    let distinct: std::collections::HashSet<usize> = ptrs.iter().map(|p| p.load(Relaxed)).collect();
    check(distinct.len() == n * per, "blocks are distinct");
    one_each(&free);
    let r0 = sys::getrusage(sys::RUSAGE_SELF);
    one_each(&alloc);
    let r1 = sys::getrusage(sys::RUSAGE_SELF);
    let (al, fr, carved) = BLOCKS.stats();
    check(r1.minflt - r0.minflt < 50, &format!("recycled blocks: {} new faults for {} MB re-touched (allocs {al}, frees {fr}, carved {carved})", r1.minflt - r0.minflt, n * per * 64 / 1024));
    one_each(&free);
    // Blocks freed on other workers than they were allocated on.
    let cross = |w: usize, _i: u32| unsafe {
        let v = (w + 1) % n;
        for k in 0..per {
            let p = BLOCKS.alloc(w);
            ptrs[v * per + k].store(p as usize, Relaxed);
        }
    };
    one_each(&cross);
    one_each(&free);
    let (_, _, carved2) = BLOCKS.stats();
    check(carved2 == carved, &format!("cross-worker frees recycle too (carved stays {carved})"));
    // Reset returns chunks to the region; a bump arena then reuses them with no new faults.
    unsafe { BLOCKS.reset() };
    let (live, peak) = REG_A.live();
    check(live == 0, &format!("Blocks::reset returns every chunk (peak live {} MB, high water {} MB)", peak >> 20, REG_A.high_water() >> 20));
    let hw = REG_A.high_water();
    let r2 = sys::getrusage(sys::RUSAGE_SELF);
    let bump = |w: usize, _i: u32| unsafe {
        let mut got = 0;
        while got < per * (64 << 10) - (1 << 20) {
            let p = BUMP.alloc(w, 4096, 64);
            std::ptr::write_volatile(p, 1);
            got += 4096;
        }
    };
    one_each(&bump);
    let r3 = sys::getrusage(sys::RUSAGE_SELF);
    check(REG_A.high_water() == hw && r3.minflt - r2.minflt < 50, &format!("bump arena reuses the freed chunks: high water unchanged, {} new faults", r3.minflt - r2.minflt));
    unsafe { BUMP.reset_all() };
    // Big bump allocations take their own ranges; alignment honoured.
    let big = unsafe { BUMP.alloc(0, 9 << 20, 1 << 21) };
    check(big as usize % (1 << 21) == 0, "large bump allocation is aligned");
    unsafe { BUMP.reset_all() };
    check(REG_A.live().0 == 0 && arena::free_ranges(&REG_A) == 1, "after resets the free list is one coalesced range");
    // Prefault puts fresh faulted memory at the front of the free list.
    REG_B.init(4 << 30, arena::Huge::Off).unwrap();
    REG_B.prefault(256 << 20, false);
    let r4 = sys::getrusage(sys::RUSAGE_SELF);
    let p = REG_B.take(256 << 20);
    arena::touch_range(p as usize, 256 << 20, false);
    let r5 = sys::getrusage(sys::RUSAGE_SELF);
    check(r5.minflt - r4.minflt < 50, &format!("prefaulted 256 MB then taken and touched: {} faults", r5.minflt - r4.minflt));
    println!("arena-test: all passed");
}

fn arena_bench(a: &[&str]) {
    let mb: usize = a.first().map(|s| s.parse().unwrap()).unwrap_or(1024);
    let huge = match a.get(1).copied().unwrap_or("off") {
        "on" => arena::Huge::On,
        "default" => arena::Huge::Default,
        "auto" => arena::Huge::Auto(mb << 20),
        _ => arena::Huge::Off,
    };
    let populate = a.get(2).copied() == Some("populate");
    let n = start(0);
    pool::wait_running();
    let v = arena::huge_verdict(mb << 20);
    REG_A.init(24 << 30, huge).unwrap();
    let (got_huge, probe_ns) = (false, 0u64);
    let vm = || -> (u64, u64, u64) {
        let t = std::fs::read_to_string("/proc/vmstat").unwrap_or_default();
        let g = |k: &str| t.lines().find_map(|l| l.strip_prefix(k).map(|v| v.trim().parse::<u64>().unwrap_or(0))).unwrap_or(0);
        (g("compact_stall "), g("thp_fault_alloc "), g("thp_fault_fallback "))
    };
    let v0 = vm();
    let r0 = sys::getrusage(sys::RUSAGE_SELF);
    let t0 = mono();
    REG_A.prefault(mb << 20, populate);
    let t1 = mono();
    let r1 = sys::getrusage(sys::RUSAGE_SELF);
    let v1 = vm();
    println!("  vmstat (system-wide) compact_stall +{} thp_fault_alloc +{} thp_fault_fallback +{}", v1.0 - v0.0, v1.1 - v0.1, v1.2 - v0.2);
    let ms = (t1 - t0) as f64 / 1e6;
    println!(
        "first touch {mb} MB huge={:?}->{:?} stalled={} populate={populate} on {n} workers: {:.2} ms ({:.1} ms per GB), minflt {}, sys {:.1} ms; verdict use_huge={} free_2m={} probe huge={} {} us",
        huge,
        REG_A.huge_mode(),
        REG_A.stalled(),
        ms,
        ms * 1024.0 / mb as f64,
        r1.minflt - r0.minflt,
        (r1.sys_us() - r0.sys_us()) as f64 / 1000.0,
        v.use_huge,
        v.free_2m,
        got_huge,
        probe_ns / 1000
    );
    // Recycle: take it back and touch again.
    let p = REG_A.take(mb << 20) as usize;
    let r2 = sys::getrusage(sys::RUSAGE_SELF);
    let t2 = mono();
    let parts = (n * 4) as u32;
    let step = (mb << 20) / parts as usize;
    let f = |_w: usize, i: u32| arena::touch_range(p + i as usize * step, step, false);
    pool::run(3, parts, &f);
    let t3 = mono();
    let r3 = sys::getrusage(sys::RUSAGE_SELF);
    println!("re-touch of recycled memory: {:.2} ms, minflt {}", (t3 - t2) as f64 / 1e6, r3.minflt - r2.minflt);
    let t4 = mono();
    exit::release(&exit::Release { arenas: true, ..exit::Release::NONE });
    let t5 = mono();
    println!("parallel release of {mb} MB: {:.2} ms", (t5 - t4) as f64 / 1e6);
}

// ------------------------------------------------------------------------------------------
// Exit tail.

fn exit_cmd(a: &[&str]) {
    let mb: usize = a[0].parse().unwrap();
    let huge = if a.get(1).copied() == Some("on") { arena::Huge::On } else { arena::Huge::Off };
    let rel_in = a.get(2).copied() == Some("1");
    let rel_ar = a.get(3).copied() == Some("1");
    let data = a.get(4).copied();
    let _n = start(0);
    pool::wait_running();
    if let Some(d) = data {
        let inp = map_all(d, true);
        let f = |_w: usize, g: u32| {
            let f = inp.file(g as usize);
            let mut p = 0;
            while p < f.len {
                unsafe { std::ptr::read_volatile(f.ptr.add(p)) };
                p += 4096;
            }
        };
        pool::run(3, inp.files().len() as u32, &f);
    }
    REG_A.init(24 << 30, huge).unwrap();
    REG_A.prefault(mb << 20, false);
    let r = sys::getrusage(sys::RUSAGE_SELF);
    let s = format!("T_done {}\nmaxrss_mb {}\n", mono(), r.maxrss / 1024);
    let _ = sys::write_all(1, s.as_bytes());
    exit::leave(0, &exit::Release { inputs: rel_in, arenas: rel_ar, ..exit::Release::NONE });
}

// ------------------------------------------------------------------------------------------
// Planted faults.

fn fallback_main(args: &[String]) {
    let t = mono();
    let outdir = std::env::var("RT_OUTDIR").unwrap_or_default();
    let mut left = 0;
    if !outdir.is_empty() {
        let swept = out::sweep(format!("{outdir}/user_daily").as_bytes());
        left = std::fs::read_dir(format!("{outdir}/user_daily")).map(|d| d.count()).unwrap_or(0);
        println!("FALLBACK swept {swept}");
    }
    let mask = cpu::affinity().count();
    println!("T_v1 {t}\nFALLBACK ok args {} left {left} cpus {mask}", args.len());
}

#[inline(never)]
fn recurse(d: u64) -> u64 {
    let mut a = [0u8; 4096];
    unsafe { std::ptr::write_volatile(&mut a[(d % 4096) as usize], 1) };
    recurse(d + 1) + a[(d % 7) as usize] as u64
}

fn fault_cmd(a: &[&str]) {
    let kind = a[0].to_string();
    let mb: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(0);
    let outdir = a.get(2).copied().unwrap_or("");
    let n = start(0);
    pool::wait_running();
    if !outdir.is_empty() {
        let d = out::make_dir(format!("{outdir}/user_daily").as_bytes()).unwrap();
        for k in 0..3 {
            let f = d.create(format!("part-{k}.parquet").as_bytes()).unwrap();
            f.write_all(b"partial").unwrap();
        }
    }
    if mb > 0 {
        REG_A.init(24 << 30, arena::Huge::Off).unwrap();
        REG_A.prefault(mb << 20, false);
    }
    let target = (n - 1).max(1) as u32 * 3; // a task late in the queue, on whichever worker
    let planted = AtomicU32::new(0);
    let task = |w: usize, i: u32| {
        if i == target && (w != 0 || n == 1) && planted.swap(1, AcqRel) == 0 {
            tline("fault");
            unsafe { plant(&kind) };
        }
        let s = mono();
        while mono() - s < 2_000_000 {
            std::hint::spin_loop();
        }
    };
    // Tasks spin 2 ms each so other workers are running when the fault hits.
    let q = unsafe { pool::push(3, (n * 8) as u32, &task) };
    // The caller does not help (worker 0 stays out, so the fault lands on another worker).
    while !pool::is_done(q) {
        std::hint::spin_loop();
    }
    println!("NOFAULT");
    sys::exit_group(3);
}

unsafe fn plant(kind: &str) {
    match kind {
        "segv" => std::ptr::write_volatile(16 as *mut u64, 1),
        "bus" => {
            let fd = sys::openat(sys::AT_FDCWD, b"/proc/self/exe\0".as_ptr(), sys::O_RDONLY | sys::O_CLOEXEC, 0).unwrap();
            let len = std::fs::metadata("/proc/self/exe").unwrap().len() as usize;
            let map = sys::up(len, 4096) + 4096;
            let p = sys::mmap(0, map, sys::PROT_READ, sys::MAP_PRIVATE, fd, 0).unwrap();
            std::ptr::read_volatile(p.add(map - 1));
        }
        "ill" => {
            #[cfg(target_arch = "x86_64")]
            std::arch::asm!("ud2");
            #[cfg(target_arch = "aarch64")]
            std::arch::asm!("udf #0");
        }
        "fpe" => {
            #[cfg(target_arch = "x86_64")]
            {
                let z: u64 = std::ptr::read_volatile(&0u64);
                std::arch::asm!("div {z}", z = in(reg) z, inout("rax") 1u64 => _, inout("rdx") 0u64 => _);
            }
            #[cfg(target_arch = "aarch64")]
            {
                // aarch64 integer division does not trap: deliver the signal to this thread.
                let _ = sys::tgkill(sys::getpid(), sys::gettid(), sys::SIGFPE);
            }
        }
        "abort" => std::process::abort(),
        "panic" => {
            let v: Vec<u8> = Vec::new();
            let i = std::hint::black_box(7);
            std::hint::black_box(v[i]);
        }
        "stack" => {
            std::hint::black_box(recurse(0));
        }
        "fail" => fail::fail(b"planted fail() from a worker"),
        "emfile" => {
            // Use up every descriptor, then fail: the re-exec must still read /proc.
            while sys::openat(sys::AT_FDCWD, b"/proc/self/exe\0".as_ptr(), sys::O_RDONLY | sys::O_CLOEXEC, 0).is_ok() {}
            fail::fail(b"planted descriptor exhaustion")
        }
        _ => panic!("unknown fault"),
    }
}

fn fault_suite(a: &[&str]) {
    let outdir = a.first().copied().unwrap_or("/tmp/rtfault");
    let mb = a.get(1).copied().unwrap_or("0");
    let exe = std::env::current_exe().unwrap();
    let kinds = ["segv", "bus", "ill", "fpe", "abort", "panic", "stack", "fail", "emfile"];
    let mut all_ok = true;
    for k in kinds {
        let _ = std::fs::remove_dir_all(outdir);
        std::fs::create_dir_all(outdir).unwrap();
        let t0 = mono();
        let o = std::process::Command::new(&exe).args(["fault", k, mb, outdir]).env("RT_OUTDIR", outdir).output().unwrap();
        let t1 = mono();
        let so = String::from_utf8_lossy(&o.stdout);
        let se = String::from_utf8_lossy(&o.stderr);
        let get = |name: &str| so.lines().find_map(|l| l.strip_prefix(name).map(|v| v.trim().parse::<u64>().unwrap()));
        let tf = get("T_fault ");
        let tv = get("T_v1 ");
        let ok = o.status.success() && so.contains("FALLBACK ok") && so.contains("left 0");
        let cpus_line = so.lines().find(|l| l.starts_with("FALLBACK ok")).unwrap_or("").to_string();
        all_ok &= ok;
        println!(
            "  {} {k:6}: exit {:?}, fault->v1 main {} us, whole child {} ms; {}; stderr: {}",
            if ok { "ok  " } else { "FAIL" },
            o.status.code(),
            match (tf, tv) {
                (Some(f), Some(v)) => ((v - f) / 1000).to_string(),
                _ => "?".into(),
            },
            (t1 - t0) / 1_000_000,
            cpus_line,
            se.lines().next().unwrap_or("")
        );
    }
    println!("fault-suite: {}", if all_ok { "all passed" } else { "FAILURES" });
    if !all_ok {
        std::process::exit(1);
    }
}

// ------------------------------------------------------------------------------------------
// Launcher.

fn launch(a: &[&str]) {
    let exe = std::env::current_exe().unwrap();
    let t0 = mono();
    let o = std::process::Command::new(&exe).args(a).env("RT_ORIGIN_NS", t0.to_string()).output().unwrap();
    let t1 = mono();
    let so = String::from_utf8_lossy(&o.stdout);
    for l in so.lines() {
        if let Some(rest) = l.strip_prefix("T_") {
            let mut it = rest.split(' ');
            let name = it.next().unwrap();
            let v: u64 = it.next().unwrap().parse().unwrap();
            println!("  T_{name}: {:.3} ms after spawn, {:.3} ms before exit seen", (v - t0) as f64 / 1e6, (t1 - v) as f64 / 1e6);
        } else {
            println!("{l}");
        }
    }
    print!("{}", String::from_utf8_lossy(&o.stderr));
    println!("  child exit {:?}, spawn to reaped {:.3} ms", o.status.code(), (t1 - t0) as f64 / 1e6);
}
