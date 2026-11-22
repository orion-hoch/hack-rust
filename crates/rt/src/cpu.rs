//! Which CPUs the process may use, how many it should use, and which share a core.
//!
//! - The allowed set comes from `sched_getaffinity` (inside Docker with `--cpuset-cpus` it is the
//!   cpuset; with only `--cpus` it is every host CPU).
//! - The CPU quota comes from cgroup v2 `cpu.max` (or v1 `cpu.cfs_quota_us`); with `--cpus 8` on
//!   a larger host it limits the useful worker count to 8 even though more CPUs are allowed.
//! - Siblings come from `/sys/devices/system/cpu/cpuN/topology/thread_siblings_list`. When that
//!   is not readable the order stays ascending CPU number and every CPU counts as its own core
//!   (`Topo::known == false`): no claim about siblings is made.

use crate::sys;

pub const MAX_CPUS: usize = 1024;

#[derive(Clone, Copy)]
pub struct CpuSet {
    pub bits: [u64; MAX_CPUS / 64],
}

impl CpuSet {
    pub const EMPTY: CpuSet = CpuSet { bits: [0; MAX_CPUS / 64] };
    pub fn has(&self, c: usize) -> bool {
        c < MAX_CPUS && self.bits[c / 64] >> (c % 64) & 1 != 0
    }
    pub fn add(&mut self, c: usize) {
        if c < MAX_CPUS {
            self.bits[c / 64] |= 1 << (c % 64);
        }
    }
    pub fn count(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }
    pub fn one(c: usize) -> CpuSet {
        let mut s = CpuSet::EMPTY;
        s.add(c);
        s
    }
}

/// The calling thread's allowed CPUs.
pub fn affinity() -> CpuSet {
    let mut s = CpuSet::EMPTY;
    unsafe {
        let _ = sys::sched_getaffinity(0, core::mem::size_of::<CpuSet>(), s.bits.as_mut_ptr() as *mut u8);
    }
    s
}

/// Restricts the calling thread (tid 0) or thread `tid` to `set`.
pub fn set_affinity(tid: i32, set: &CpuSet) -> sys::Res<()> {
    unsafe { sys::sched_setaffinity(tid, core::mem::size_of::<CpuSet>(), set.bits.as_ptr() as *const u8) }
}

/// Parses a kernel CPU list ("0-3,8,10-11\n") into `out`.
pub fn parse_list(text: &[u8], out: &mut CpuSet) -> bool {
    let mut i = 0;
    let mut any = false;
    let num = |i: &mut usize| -> Option<usize> {
        let s = *i;
        let mut v = 0usize;
        while *i < text.len() && text[*i].is_ascii_digit() {
            v = v * 10 + (text[*i] - b'0') as usize;
            *i += 1;
        }
        if *i == s { None } else { Some(v) }
    };
    while i < text.len() {
        let Some(a) = num(&mut i) else { break };
        let mut b = a;
        if i < text.len() && text[i] == b'-' {
            i += 1;
            match num(&mut i) {
                Some(x) => b = x,
                None => return false,
            }
        }
        for c in a..=b.min(MAX_CPUS - 1) {
            out.add(c);
            any = true;
        }
        if i < text.len() && text[i] == b',' {
            i += 1;
        } else {
            break;
        }
    }
    any
}

/// The CPU quota in whole CPUs, rounded up (None: no limit or not readable).
pub fn cgroup_quota() -> Option<usize> {
    let mut buf = [0u8; 128];
    if let Ok(n) = sys::read_small(b"/sys/fs/cgroup/cpu.max", &mut buf) {
        // "max 100000" or "800000 100000"
        let t = &buf[..n];
        if t.starts_with(b"max") {
            return None;
        }
        let mut i = 0;
        let q = num(t, &mut i)?;
        i += 1;
        let p = num(t, &mut i)?;
        return if p == 0 { None } else { Some(q.div_ceil(p).max(1)) };
    }
    let mut qb = [0u8; 64];
    let mut pb = [0u8; 64];
    let qn = sys::read_small(b"/sys/fs/cgroup/cpu/cpu.cfs_quota_us", &mut qb).ok()?;
    let pn = sys::read_small(b"/sys/fs/cgroup/cpu/cpu.cfs_period_us", &mut pb).ok()?;
    if qb[..qn].starts_with(b"-") {
        return None;
    }
    let q = num(&qb[..qn], &mut 0)?;
    let p = num(&pb[..pn], &mut 0)?;
    if p == 0 { None } else { Some(q.div_ceil(p).max(1)) }
}

fn num(t: &[u8], i: &mut usize) -> Option<usize> {
    let s = *i;
    let mut v = 0usize;
    while *i < t.len() && t[*i].is_ascii_digit() {
        v = v * 10 + (t[*i] - b'0') as usize;
        *i += 1;
    }
    if *i == s { None } else { Some(v) }
}

/// The allowed CPUs ordered so that hyperthread siblings are adjacent, with a core number each.
pub struct Topo {
    pub n: usize,
    pub cpu: [u16; MAX_CPUS],
    pub core: [u16; MAX_CPUS],
    pub cores: usize,
    /// False when the sibling lists could not be read: order is ascending, one core per CPU.
    pub known: bool,
}

/// Orders `set` by core using sysfs under `root` (normally `/sys/devices/system/cpu`).
pub fn topology(set: &CpuSet, root: &[u8]) -> Topo {
    let mut t = Topo { n: 0, cpu: [0; MAX_CPUS], core: [0; MAX_CPUS], cores: 0, known: true };
    let mut placed = CpuSet::EMPTY;
    for c in 0..MAX_CPUS {
        if !set.has(c) || placed.has(c) {
            continue;
        }
        let mut sib = CpuSet::EMPTY;
        let mut ok = false;
        let mut num = [0u8; 8];
        let nl = itoa(c, &mut num);
        if let Some(p) = sys::CPath::new(&[root, b"/cpu", &num[..nl], b"/topology/thread_siblings_list"]) {
            let mut buf = [0u8; 256];
            let fd = unsafe { sys::openat(sys::AT_FDCWD, p.ptr(), sys::O_RDONLY | sys::O_CLOEXEC, 0) };
            if let Ok(fd) = fd {
                if let Ok(n) = unsafe { sys::read(fd, buf.as_mut_ptr(), buf.len()) } {
                    ok = parse_list(&buf[..n], &mut sib) && sib.has(c);
                }
                let _ = sys::close(fd);
            }
        }
        if !ok {
            t.known = false;
            sib = CpuSet::one(c);
        }
        for s in c..MAX_CPUS {
            if sib.has(s) && set.has(s) && !placed.has(s) {
                placed.add(s);
                t.cpu[t.n] = s as u16;
                t.core[t.n] = t.cores as u16;
                t.n += 1;
            }
        }
        t.cores += 1;
    }
    t
}

fn itoa(mut v: usize, out: &mut [u8; 8]) -> usize {
    let mut t = [0u8; 8];
    let mut i = 8;
    loop {
        i -= 1;
        t[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    let n = 8 - i;
    out[..n].copy_from_slice(&t[i..]);
    n
}

/// Cache sizes as the machine reports them (bytes; 0 when unknown) and how many logical CPUs
/// share each. From sysfs (`/sys/devices/system/cpu/cpu0/cache/indexN`), else cpuid (Intel leaf
/// 4, AMD leaf 0x8000001D).
#[derive(Clone, Copy, Debug, Default)]
pub struct Caches {
    pub l2: usize,
    pub l2_shared: usize,
    pub l3: usize,
    pub l3_shared: usize,
    /// Where the numbers came from: b's' sysfs, b'c' cpuid, 0 neither.
    pub source: u8,
}

impl Caches {
    /// L2 bytes per logical CPU sharing it (the share of one hyperthread).
    pub fn l2_per_thread(&self) -> usize {
        self.l2 / self.l2_shared.max(1)
    }
}

pub fn caches() -> Caches {
    let mut c = Caches::default();
    for i in 0..10u8 {
        let base = [b"/sys/devices/system/cpu/cpu0/cache/index".as_slice(), &[b'0' + i]];
        let read = |leaf: &[u8], buf: &mut [u8; 64]| -> Option<usize> {
            let p = sys::CPath::new(&[base[0], base[1], b"/", leaf])?;
            let fd = unsafe { sys::openat(sys::AT_FDCWD, p.ptr(), sys::O_RDONLY | sys::O_CLOEXEC, 0) }.ok()?;
            let n = unsafe { sys::read(fd, buf.as_mut_ptr(), buf.len()) }.ok();
            let _ = sys::close(fd);
            n
        };
        let mut b = [0u8; 64];
        let Some(n) = read(b"level", &mut b) else { break };
        let level = b[..n].first().map(|&d| d.wrapping_sub(b'0')).unwrap_or(0);
        let Some(n) = read(b"type", &mut b) else { continue };
        if b[..n].starts_with(b"Instruction") {
            continue;
        }
        let Some(n) = read(b"size", &mut b) else { continue };
        let mut v = 0usize;
        let mut k = 0;
        while k < n && b[k].is_ascii_digit() {
            v = v * 10 + (b[k] - b'0') as usize;
            k += 1;
        }
        let size = match b.get(k) {
            Some(b'K') => v << 10,
            Some(b'M') => v << 20,
            Some(b'G') => v << 30,
            _ => v,
        };
        let mut shared = CpuSet::EMPTY;
        let nshared = match read(b"shared_cpu_list", &mut b) {
            Some(n) if parse_list(&b[..n], &mut shared) => shared.count(),
            _ => 1,
        };
        match level {
            2 => {
                c.l2 = size;
                c.l2_shared = nshared;
            }
            3 => {
                c.l3 = size;
                c.l3_shared = nshared;
            }
            _ => {}
        }
        c.source = b's';
    }
    if c.l2 == 0 {
        cpuid_caches(&mut c);
    }
    c
}

#[cfg(target_arch = "x86_64")]
fn cpuid_caches(c: &mut Caches) {
    use core::arch::x86_64::__cpuid_count;
    let v = unsafe { __cpuid_count(0, 0) };
    let amd = v.ebx == 0x6874_7541; // "Auth"
    let leaf = if amd { 0x8000_001d } else { 4 };
    if amd && unsafe { __cpuid_count(0x8000_0000, 0) }.eax < 0x8000_001d {
        return;
    }
    if !amd && v.eax < 4 {
        return;
    }
    for sub in 0..16 {
        let r = unsafe { __cpuid_count(leaf, sub) };
        let typ = r.eax & 0x1f;
        if typ == 0 {
            break;
        }
        if typ == 2 {
            continue; // instruction cache
        }
        let level = (r.eax >> 5) & 7;
        let ways = ((r.ebx >> 22) & 0x3ff) as usize + 1;
        let parts = ((r.ebx >> 12) & 0x3ff) as usize + 1;
        let line = (r.ebx & 0xfff) as usize + 1;
        let sets = r.ecx as usize + 1;
        let shared = ((r.eax >> 14) & 0xfff) as usize + 1;
        let size = ways * parts * line * sets;
        if level == 2 {
            c.l2 = size;
            c.l2_shared = shared;
        } else if level == 3 {
            c.l3 = size;
            c.l3_shared = shared;
        }
        c.source = b'c';
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn cpuid_caches(_c: &mut Caches) {}
