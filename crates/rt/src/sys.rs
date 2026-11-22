//! Linux system calls issued directly (x86-64 and aarch64): numbers, the trap instruction and thin
//! typed wrappers. Nothing in this crate calls libc.

use core::arch::asm;

pub type Res<T> = Result<T, Errno>;

/// A kernel error number (positive).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Errno(pub i32);

impl Errno {
    pub const ENOENT: Errno = Errno(2);
    pub const EINTR: Errno = Errno(4);
    pub const EAGAIN: Errno = Errno(11);
    pub const ENOMEM: Errno = Errno(12);
    pub const EEXIST: Errno = Errno(17);
    pub const ENOTDIR: Errno = Errno(20);
    pub const EINVAL: Errno = Errno(22);
    pub const ERANGE: Errno = Errno(34);
    pub const ENOSYS: Errno = Errno(38);
    pub const ENAMETOOLONG: Errno = Errno(36);
}

#[inline(always)]
fn res(r: isize) -> Res<usize> {
    if (r as usize) > (-4096isize) as usize { Err(Errno(-r as i32)) } else { Ok(r as usize) }
}

#[cfg(target_arch = "x86_64")]
pub mod nr {
    pub const READ: usize = 0;
    pub const WRITE: usize = 1;
    pub const CLOSE: usize = 3;
    pub const MMAP: usize = 9;
    pub const MPROTECT: usize = 10;
    pub const MUNMAP: usize = 11;
    pub const RT_SIGACTION: usize = 13;
    pub const RT_SIGPROCMASK: usize = 14;
    pub const RT_SIGRETURN: usize = 15;
    pub const PREAD64: usize = 17;
    pub const PWRITE64: usize = 18;
    pub const SCHED_YIELD: usize = 24;
    pub const MADVISE: usize = 28;
    pub const GETPID: usize = 39;
    pub const CLONE: usize = 56;
    pub const EXECVE: usize = 59;
    pub const EXIT: usize = 60;
    pub const KILL: usize = 62;
    pub const FTRUNCATE: usize = 77;
    pub const GETRUSAGE: usize = 98;
    pub const SIGALTSTACK: usize = 131;
    pub const GETTID: usize = 186;
    pub const FUTEX: usize = 202;
    pub const SCHED_SETAFFINITY: usize = 203;
    pub const SCHED_GETAFFINITY: usize = 204;
    pub const GETDENTS64: usize = 217;
    pub const CLOCK_GETTIME: usize = 228;
    pub const EXIT_GROUP: usize = 231;
    pub const TGKILL: usize = 234;
    pub const OPENAT: usize = 257;
    pub const MKDIRAT: usize = 258;
    pub const UNLINKAT: usize = 263;
    pub const FALLOCATE: usize = 285;
    pub const STATX: usize = 332;
    pub const CLONE3: usize = 435;
    pub const DUP3: usize = 292;
}

#[cfg(target_arch = "aarch64")]
pub mod nr {
    pub const READ: usize = 63;
    pub const WRITE: usize = 64;
    pub const CLOSE: usize = 57;
    pub const MMAP: usize = 222;
    pub const MPROTECT: usize = 226;
    pub const MUNMAP: usize = 215;
    pub const RT_SIGACTION: usize = 134;
    pub const RT_SIGPROCMASK: usize = 135;
    pub const RT_SIGRETURN: usize = 139;
    pub const PREAD64: usize = 67;
    pub const PWRITE64: usize = 68;
    pub const SCHED_YIELD: usize = 124;
    pub const MADVISE: usize = 233;
    pub const GETPID: usize = 172;
    pub const CLONE: usize = 220;
    pub const EXECVE: usize = 221;
    pub const EXIT: usize = 93;
    pub const KILL: usize = 129;
    pub const FTRUNCATE: usize = 46;
    pub const GETRUSAGE: usize = 165;
    pub const SIGALTSTACK: usize = 132;
    pub const GETTID: usize = 178;
    pub const FUTEX: usize = 98;
    pub const SCHED_SETAFFINITY: usize = 122;
    pub const SCHED_GETAFFINITY: usize = 123;
    pub const GETDENTS64: usize = 61;
    pub const CLOCK_GETTIME: usize = 113;
    pub const EXIT_GROUP: usize = 94;
    pub const TGKILL: usize = 131;
    pub const OPENAT: usize = 56;
    pub const MKDIRAT: usize = 34;
    pub const UNLINKAT: usize = 35;
    pub const FALLOCATE: usize = 47;
    pub const STATX: usize = 291;
    pub const CLONE3: usize = 435;
    pub const DUP3: usize = 24;
}

// ------------------------------------------------------------------------------------------
// The trap.

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn sc6(n: usize, a: usize, b: usize, c: usize, d: usize, e: usize, f: usize) -> isize {
    let r: isize;
    asm!("syscall", inlateout("rax") n as isize => r, in("rdi") a, in("rsi") b, in("rdx") c,
         in("r10") d, in("r8") e, in("r9") f, lateout("rcx") _, lateout("r11") _, options(nostack));
    r
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn sc3(n: usize, a: usize, b: usize, c: usize) -> isize {
    let r: isize;
    asm!("syscall", inlateout("rax") n as isize => r, in("rdi") a, in("rsi") b, in("rdx") c,
         lateout("rcx") _, lateout("r11") _, options(nostack));
    r
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn sc6(n: usize, a: usize, b: usize, c: usize, d: usize, e: usize, f: usize) -> isize {
    let r: isize;
    asm!("svc 0", in("x8") n, inlateout("x0") a as isize => r, in("x1") b, in("x2") c, in("x3") d,
         in("x4") e, in("x5") f, options(nostack));
    r
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn sc3(n: usize, a: usize, b: usize, c: usize) -> isize {
    let r: isize;
    asm!("svc 0", in("x8") n, inlateout("x0") a as isize => r, in("x1") b, in("x2") c, options(nostack));
    r
}

#[inline(always)]
pub unsafe fn sc4(n: usize, a: usize, b: usize, c: usize, d: usize) -> isize {
    sc6(n, a, b, c, d, 0, 0)
}

// ------------------------------------------------------------------------------------------
// Constants.

pub const AT_FDCWD: i32 = -100;
pub const AT_REMOVEDIR: i32 = 0x200;
pub const AT_EMPTY_PATH: i32 = 0x1000;

pub const O_RDONLY: i32 = 0;
pub const O_WRONLY: i32 = 1;
pub const O_RDWR: i32 = 2;
pub const O_CREAT: i32 = 0o100;
pub const O_EXCL: i32 = 0o200;
pub const O_TRUNC: i32 = 0o1000;
pub const O_CLOEXEC: i32 = 0o2000000;
#[cfg(target_arch = "x86_64")]
pub const O_DIRECTORY: i32 = 0o200000;
#[cfg(target_arch = "aarch64")]
pub const O_DIRECTORY: i32 = 0o40000;

pub const PROT_NONE: usize = 0;
pub const PROT_READ: usize = 1;
pub const PROT_WRITE: usize = 2;
pub const MAP_SHARED: usize = 1;
pub const MAP_PRIVATE: usize = 2;
pub const MAP_FIXED: usize = 0x10;
pub const MAP_ANONYMOUS: usize = 0x20;
pub const MAP_NORESERVE: usize = 0x4000;
pub const MAP_POPULATE: usize = 0x8000;
pub const MAP_STACK: usize = 0x20000;

pub const MADV_NORMAL: i32 = 0;
pub const MADV_RANDOM: i32 = 1;
pub const MADV_SEQUENTIAL: i32 = 2;
pub const MADV_WILLNEED: i32 = 3;
pub const MADV_DONTNEED: i32 = 4;
pub const MADV_HUGEPAGE: i32 = 14;
pub const MADV_NOHUGEPAGE: i32 = 15;
pub const MADV_POPULATE_READ: i32 = 22;
pub const MADV_POPULATE_WRITE: i32 = 23;

pub const FUTEX_WAIT_PRIVATE: usize = 128;
pub const FUTEX_WAKE_PRIVATE: usize = 129;

pub const CLONE_VM: usize = 0x100;
pub const CLONE_FS: usize = 0x200;
pub const CLONE_FILES: usize = 0x400;
pub const CLONE_SIGHAND: usize = 0x800;
pub const CLONE_THREAD: usize = 0x10000;
pub const CLONE_SYSVSEM: usize = 0x40000;
pub const CLONE_SETTLS: usize = 0x80000;
pub const CLONE_PARENT_SETTID: usize = 0x100000;

pub const SIGILL: i32 = 4;
pub const SIGTRAP: i32 = 5;
pub const SIGABRT: i32 = 6;
pub const SIGBUS: i32 = 7;
pub const SIGFPE: i32 = 8;
pub const SIGSEGV: i32 = 11;
pub const SA_SIGINFO: u64 = 4;
pub const SA_ONSTACK: u64 = 0x0800_0000;
pub const SA_RESTORER: u64 = 0x0400_0000;
pub const SA_NODEFER: u64 = 0x4000_0000;
pub const SIG_SETMASK: i32 = 2;

pub const CLOCK_MONOTONIC: i32 = 1;

pub const RUSAGE_SELF: i32 = 0;
pub const RUSAGE_THREAD: i32 = 1;

pub const PAGE: usize = 4096;
pub const HUGE: usize = 2 << 20;

#[inline(always)]
pub const fn up(x: usize, a: usize) -> usize {
    (x + a - 1) & !(a - 1)
}
#[inline(always)]
pub const fn down(x: usize, a: usize) -> usize {
    x & !(a - 1)
}

// ------------------------------------------------------------------------------------------
// Structures shared with the kernel.

/// The kernel's `struct sigaction` (x86-64 and aarch64 have the same layout).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KSigaction {
    pub handler: usize,
    pub flags: u64,
    pub restorer: usize,
    pub mask: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct StackT {
    pub sp: usize,
    pub flags: i32,
    pub size: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Rusage {
    pub utime_s: i64,
    pub utime_us: i64,
    pub stime_s: i64,
    pub stime_us: i64,
    pub maxrss: i64,
    pub ixrss: i64,
    pub idrss: i64,
    pub isrss: i64,
    pub minflt: i64,
    pub majflt: i64,
    pub nswap: i64,
    pub inblock: i64,
    pub oublock: i64,
    pub msgsnd: i64,
    pub msgrcv: i64,
    pub nsignals: i64,
    pub nvcsw: i64,
    pub nivcsw: i64,
}

impl Rusage {
    pub fn user_us(&self) -> i64 {
        self.utime_s * 1_000_000 + self.utime_us
    }
    pub fn sys_us(&self) -> i64 {
        self.stime_s * 1_000_000 + self.stime_us
    }
}

/// `struct statx` (256 bytes); only the fields we read are named.
#[repr(C)]
pub struct Statx {
    pub mask: u32,
    pub blksize: u32,
    pub attributes: u64,
    pub nlink: u32,
    pub uid: u32,
    pub gid: u32,
    pub mode: u16,
    _pad0: u16,
    pub ino: u64,
    pub size: u64,
    _rest: [u64; 26],
}

impl Statx {
    pub const fn zeroed() -> Statx {
        Statx { mask: 0, blksize: 0, attributes: 0, nlink: 0, uid: 0, gid: 0, mode: 0, _pad0: 0, ino: 0, size: 0, _rest: [0; 26] }
    }
}

pub const STATX_TYPE: u32 = 1;
pub const STATX_SIZE: u32 = 0x200;
pub const S_IFMT: u16 = 0o170000;
pub const S_IFREG: u16 = 0o100000;
pub const S_IFDIR: u16 = 0o040000;

pub const DT_UNKNOWN: u8 = 0;
pub const DT_DIR: u8 = 4;
pub const DT_REG: u8 = 8;
pub const DT_LNK: u8 = 10;

// ------------------------------------------------------------------------------------------
// Wrappers.

#[inline]
pub unsafe fn read(fd: i32, buf: *mut u8, len: usize) -> Res<usize> {
    res(sc3(nr::READ, fd as usize, buf as usize, len))
}
#[inline]
pub unsafe fn write(fd: i32, buf: *const u8, len: usize) -> Res<usize> {
    res(sc3(nr::WRITE, fd as usize, buf as usize, len))
}
#[inline]
pub unsafe fn pread(fd: i32, buf: *mut u8, len: usize, off: u64) -> Res<usize> {
    res(sc4(nr::PREAD64, fd as usize, buf as usize, len, off as usize))
}
#[inline]
pub unsafe fn pwrite(fd: i32, buf: *const u8, len: usize, off: u64) -> Res<usize> {
    res(sc4(nr::PWRITE64, fd as usize, buf as usize, len, off as usize))
}
/// `path` must be NUL-terminated.
#[inline]
pub unsafe fn openat(dirfd: i32, path: *const u8, flags: i32, mode: u32) -> Res<i32> {
    res(sc4(nr::OPENAT, dirfd as isize as usize, path as usize, flags as usize, mode as usize)).map(|r| r as i32)
}
#[inline]
pub fn close(fd: i32) -> Res<()> {
    unsafe { res(sc3(nr::CLOSE, fd as usize, 0, 0)).map(|_| ()) }
}
#[inline]
pub unsafe fn getdents64(fd: i32, buf: *mut u8, len: usize) -> Res<usize> {
    res(sc3(nr::GETDENTS64, fd as usize, buf as usize, len))
}
#[inline]
pub unsafe fn statx(dirfd: i32, path: *const u8, flags: i32, mask: u32, out: *mut Statx) -> Res<()> {
    res(sc6(nr::STATX, dirfd as isize as usize, path as usize, flags as usize, mask as usize, out as usize, 0)).map(|_| ())
}
#[inline]
pub unsafe fn mmap(addr: usize, len: usize, prot: usize, flags: usize, fd: i32, off: u64) -> Res<*mut u8> {
    res(sc6(nr::MMAP, addr, len, prot, flags, fd as isize as usize, off as usize)).map(|r| r as *mut u8)
}
#[inline]
pub unsafe fn munmap(addr: usize, len: usize) -> Res<()> {
    res(sc3(nr::MUNMAP, addr, len, 0)).map(|_| ())
}
#[inline]
pub unsafe fn madvise(addr: usize, len: usize, advice: i32) -> Res<()> {
    res(sc3(nr::MADVISE, addr, len, advice as usize)).map(|_| ())
}
#[inline]
pub unsafe fn mprotect(addr: usize, len: usize, prot: usize) -> Res<()> {
    res(sc3(nr::MPROTECT, addr, len, prot)).map(|_| ())
}
#[inline]
pub fn ftruncate(fd: i32, len: u64) -> Res<()> {
    unsafe { res(sc3(nr::FTRUNCATE, fd as usize, len as usize, 0)).map(|_| ()) }
}
#[inline]
pub fn fallocate(fd: i32, mode: i32, off: u64, len: u64) -> Res<()> {
    unsafe { res(sc4(nr::FALLOCATE, fd as usize, mode as usize, off as usize, len as usize)).map(|_| ()) }
}
#[inline]
pub unsafe fn mkdirat(dirfd: i32, path: *const u8, mode: u32) -> Res<()> {
    res(sc3(nr::MKDIRAT, dirfd as isize as usize, path as usize, mode as usize)).map(|_| ())
}
#[inline]
pub unsafe fn unlinkat(dirfd: i32, path: *const u8, flags: i32) -> Res<()> {
    res(sc3(nr::UNLINKAT, dirfd as isize as usize, path as usize, flags as usize)).map(|_| ())
}
/// Sleeps while `*addr == val` (private futex). Spurious returns are possible.
#[inline]
pub unsafe fn futex_wait(addr: *const u32, val: u32) -> Res<()> {
    res(sc4(nr::FUTEX, addr as usize, FUTEX_WAIT_PRIVATE, val as usize, 0)).map(|_| ())
}
#[inline]
pub unsafe fn futex_wake(addr: *const u32, n: i32) -> Res<usize> {
    res(sc3(nr::FUTEX, addr as usize, FUTEX_WAKE_PRIVATE, n as usize))
}
#[inline]
pub unsafe fn sched_getaffinity(tid: i32, len: usize, mask: *mut u8) -> Res<usize> {
    res(sc3(nr::SCHED_GETAFFINITY, tid as usize, len, mask as usize))
}
#[inline]
pub unsafe fn sched_setaffinity(tid: i32, len: usize, mask: *const u8) -> Res<()> {
    res(sc3(nr::SCHED_SETAFFINITY, tid as usize, len, mask as usize)).map(|_| ())
}
#[inline]
pub unsafe fn rt_sigaction(sig: i32, act: *const KSigaction, old: *mut KSigaction) -> Res<()> {
    res(sc4(nr::RT_SIGACTION, sig as usize, act as usize, old as usize, 8)).map(|_| ())
}
#[inline]
pub unsafe fn rt_sigprocmask(how: i32, set: *const u64, old: *mut u64) -> Res<()> {
    res(sc4(nr::RT_SIGPROCMASK, how as usize, set as usize, old as usize, 8)).map(|_| ())
}
#[inline]
pub unsafe fn sigaltstack(ss: *const StackT, old: *mut StackT) -> Res<()> {
    res(sc3(nr::SIGALTSTACK, ss as usize, old as usize, 0)).map(|_| ())
}
/// Returns only on failure.
#[inline]
pub unsafe fn execve(path: *const u8, argv: *const *const u8, envp: *const *const u8) -> Errno {
    match res(sc3(nr::EXECVE, path as usize, argv as usize, envp as usize)) {
        Err(e) => e,
        Ok(_) => Errno(0),
    }
}
#[inline]
pub fn exit_group(code: i32) -> ! {
    unsafe {
        sc3(nr::EXIT_GROUP, code as usize, 0, 0);
        sc3(nr::EXIT, code as usize, 0, 0);
    }
    loop {
        core::hint::spin_loop();
    }
}
/// Ends the calling thread only.
#[inline]
pub fn exit_thread(code: i32) -> ! {
    unsafe {
        sc3(nr::EXIT, code as usize, 0, 0);
    }
    loop {
        core::hint::spin_loop();
    }
}
#[inline]
pub fn getrusage(who: i32) -> Rusage {
    let mut r = Rusage::default();
    unsafe {
        sc3(nr::GETRUSAGE, who as usize, &mut r as *mut Rusage as usize, 0);
    }
    r
}
/// CLOCK_MONOTONIC in nanoseconds, through the system call (no vDSO).
#[inline]
pub fn mono_ns() -> u64 {
    let mut ts = [0i64; 2];
    unsafe {
        sc3(nr::CLOCK_GETTIME, CLOCK_MONOTONIC as usize, ts.as_mut_ptr() as usize, 0);
    }
    ts[0] as u64 * 1_000_000_000 + ts[1] as u64
}
#[inline]
pub fn gettid() -> i32 {
    unsafe { sc3(nr::GETTID, 0, 0, 0) as i32 }
}
#[inline]
pub fn getpid() -> i32 {
    unsafe { sc3(nr::GETPID, 0, 0, 0) as i32 }
}
#[inline]
pub fn tgkill(tgid: i32, tid: i32, sig: i32) -> Res<()> {
    unsafe { res(sc3(nr::TGKILL, tgid as usize, tid as usize, sig as usize)).map(|_| ()) }
}
#[inline]
pub fn sched_yield() {
    unsafe {
        sc3(nr::SCHED_YIELD, 0, 0, 0);
    }
}

#[inline]
pub fn dup3(old: i32, new: i32, flags: i32) -> Res<i32> {
    unsafe { res(sc3(nr::DUP3, old as usize, new as usize, flags as usize)).map(|r| r as i32) }
}

/// Grows the file-descriptor table to hold `n` descriptors now. Growing it later, once threads
/// share it, makes the kernel wait for an RCU grace period (measured 11 to 13 ms for one openat
/// on the loaded x86 box). Call while the process is still single-threaded.
pub fn reserve_fds(n: usize) -> bool {
    let fd = n.max(65) as i32 - 1;
    match dup3(2, fd, O_CLOEXEC) {
        Ok(_) => {
            let _ = close(fd);
            true
        }
        Err(_) => false,
    }
}

/// Writes all of `b` to `fd`, retrying short writes and EINTR.
pub fn write_all(fd: i32, mut b: &[u8]) -> Res<()> {
    while !b.is_empty() {
        match unsafe { write(fd, b.as_ptr(), b.len()) } {
            Ok(0) => return Err(Errno(5)),
            Ok(n) => b = &b[n..],
            Err(Errno::EINTR) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Writes all of `b` at `off`, retrying short writes and EINTR.
pub fn pwrite_all(fd: i32, mut b: &[u8], mut off: u64) -> Res<()> {
    while !b.is_empty() {
        match unsafe { pwrite(fd, b.as_ptr(), b.len(), off) } {
            Ok(0) => return Err(Errno(5)),
            Ok(n) => {
                b = &b[n..];
                off += n as u64;
            }
            Err(Errno::EINTR) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Reads up to `buf.len()` bytes at `off`, retrying short reads and EINTR. Returns the count
/// (less than requested only at end of file).
pub fn pread_full(fd: i32, buf: &mut [u8], off: u64) -> Res<usize> {
    let mut got = 0;
    while got < buf.len() {
        match unsafe { pread(fd, buf.as_mut_ptr().add(got), buf.len() - got, off + got as u64) } {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(Errno::EINTR) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(got)
}

/// A NUL-terminated path on the stack.
pub struct CPath {
    buf: [u8; 1024],
}

impl CPath {
    /// Concatenates `parts` (the caller supplies any `/`). None if too long or if a part holds NUL.
    pub fn new(parts: &[&[u8]]) -> Option<CPath> {
        let mut p = CPath { buf: [0; 1024] };
        let mut n = 0;
        for part in parts {
            if n + part.len() >= p.buf.len() || part.contains(&0) {
                return None;
            }
            p.buf[n..n + part.len()].copy_from_slice(part);
            n += part.len();
        }
        Some(p)
    }
    #[inline]
    pub fn ptr(&self) -> *const u8 {
        self.buf.as_ptr()
    }
}

/// Opens, reads whole (up to `buf.len()`), closes. Returns the byte count.
pub fn read_small(path: &[u8], buf: &mut [u8]) -> Res<usize> {
    let p = CPath::new(&[path]).ok_or(Errno::ENAMETOOLONG)?;
    let fd = unsafe { openat(AT_FDCWD, p.ptr(), O_RDONLY | O_CLOEXEC, 0)? };
    let mut got = 0;
    let r = loop {
        if got == buf.len() {
            break Ok(got);
        }
        match unsafe { read(fd, buf.as_mut_ptr().add(got), buf.len() - got) } {
            Ok(0) => break Ok(got),
            Ok(n) => got += n,
            Err(Errno::EINTR) => {}
            Err(e) => break Err(e),
        }
    };
    let _ = close(fd);
    r
}

/// Anonymous private read-write memory, not reserved against overcommit.
pub fn map_anon(len: usize) -> Res<*mut u8> {
    unsafe { mmap(0, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0) }
}
