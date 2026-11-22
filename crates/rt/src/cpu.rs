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
