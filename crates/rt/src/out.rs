//! Output files by direct system calls: create the table folders and files, write buffers as
//! given (nothing buffered behind the caller), and remember every file created so `fail` can
//! remove partial output before the general engine runs.
//!
//! Race with `fail`: a creator registers the name, then checks the failing flag, then creates;
//! `fail` sets the flag and then waits (bounded) for creators in flight before unlinking. After
//! the re-exec no fast-path thread exists, so `sweep` at the start of the fallback run removes
//! anything a lost race left behind.

use core::sync::atomic::{AtomicI32, AtomicU32, AtomicUsize, Ordering::*};

use crate::sys::{self, Errno, Res};

const MAX_OUT: usize = 512;
const NAME: usize = 64;

struct Reg {
    dirfd: AtomicI32,
    len: AtomicUsize,
    name: [AtomicU32; NAME / 4],
}

static REG: [Reg; MAX_OUT] = [const { Reg { dirfd: AtomicI32::new(-1), len: AtomicUsize::new(0), name: [const { AtomicU32::new(0) }; NAME / 4] } }; MAX_OUT];
static REG_N: AtomicUsize = AtomicUsize::new(0);
pub(crate) static FAILING: AtomicU32 = AtomicU32::new(0);
static INFLIGHT: AtomicU32 = AtomicU32::new(0);

/// An open directory.
#[derive(Clone, Copy, Debug)]
pub struct Dir {
    pub fd: i32,
}

/// Creates `path` (mode 0755; an existing folder is fine) and opens it.
pub fn make_dir(path: &[u8]) -> Res<Dir> {
    let p = sys::CPath::new(&[path]).ok_or(Errno::ENAMETOOLONG)?;
    match unsafe { sys::mkdirat(sys::AT_FDCWD, p.ptr(), 0o755) } {
        Ok(()) | Err(Errno::EEXIST) => {}
        Err(e) => return Err(e),
    }
    let fd = unsafe { sys::openat(sys::AT_FDCWD, p.ptr(), sys::O_RDONLY | sys::O_DIRECTORY | sys::O_CLOEXEC, 0)? };
    Ok(Dir { fd })
}

/// An output file open for writing.
#[derive(Clone, Copy, Debug)]
pub struct File {
    pub fd: i32,
}

impl Dir {
    /// Creates (or truncates) `name` (at most 63 bytes) in this folder, registered for `fail`.
    pub fn create(&self, name: &[u8]) -> Res<File> {
        if name.len() >= NAME || name.is_empty() {
            return Err(Errno::ENAMETOOLONG);
        }
        let i = REG_N.fetch_add(1, AcqRel);
        if i >= MAX_OUT {
            return Err(Errno::ERANGE);
        }
        let r = &REG[i];
        let mut words = [0u32; NAME / 4];
        for (k, &c) in name.iter().enumerate() {
            words[k / 4] |= (c as u32) << (8 * (k % 4));
        }
        for (k, w) in words.iter().enumerate() {
            r.name[k].store(*w, Relaxed);
        }
        r.len.store(name.len(), Relaxed);
        r.dirfd.store(self.fd, Release);
        INFLIGHT.fetch_add(1, SeqCst);
        if FAILING.load(SeqCst) != 0 {
            INFLIGHT.fetch_sub(1, SeqCst);
            return Err(Errno::EAGAIN);
        }
        let mut c = [0u8; NAME];
        c[..name.len()].copy_from_slice(name);
        let fd = unsafe { sys::openat(self.fd, c.as_ptr(), sys::O_WRONLY | sys::O_CREAT | sys::O_TRUNC | sys::O_CLOEXEC, 0o644) };
        INFLIGHT.fetch_sub(1, SeqCst);
        Ok(File { fd: fd? })
    }
}

impl File {
    /// Appends `b` at the file position (one or more write calls; no buffering).
    pub fn write_all(&self, b: &[u8]) -> Res<()> {
        sys::write_all(self.fd, b)
    }
    /// Writes `b` at byte `off` (for several writers into one file).
    pub fn write_at(&self, off: u64, b: &[u8]) -> Res<()> {
        sys::pwrite_all(self.fd, b, off)
    }
    /// Sets the length (for example to the final size before parallel `write_at`).
    pub fn set_len(&self, len: u64) -> Res<()> {
        sys::ftruncate(self.fd, len)
    }
    pub fn close(self) -> Res<()> {
        sys::close(self.fd)
    }
}

/// Unlinks every registered output file (called by `fail`). Waits up to about 20 ms for
/// creators in flight.
pub(crate) fn remove_partial() {
    FAILING.store(1, SeqCst);
    let t0 = sys::mono_ns();
    while INFLIGHT.load(SeqCst) != 0 && sys::mono_ns() - t0 < 20_000_000 {
        core::hint::spin_loop();
    }
    let n = REG_N.load(Acquire).min(MAX_OUT);
    for r in REG.iter().take(n) {
        let fd = r.dirfd.load(Acquire);
        if fd < 0 {
            continue;
        }
        let len = r.len.load(Relaxed);
        let mut c = [0u8; NAME];
        for k in 0..len {
            c[k] = (r.name[k / 4].load(Relaxed) >> (8 * (k % 4))) as u8;
        }
        unsafe {
            let _ = sys::unlinkat(fd, c.as_ptr(), 0);
        }
    }
}

/// Removes every entry of folder `path` whose name does not start with `.` or `_` (the folder
/// itself stays). For the start of a fallback run, after `fail`'s re-exec. Returns the count.
pub fn sweep(path: &[u8]) -> usize {
    let mut removed = 0;
    // Passes until one removes nothing (unlinking while listing may hide entries from a pass).
    for _ in 0..8 {
        let n = sweep_pass(path);
        removed += n;
        if n == 0 {
            break;
        }
    }
    removed
}

fn sweep_pass(path: &[u8]) -> usize {
    let Some(p) = sys::CPath::new(&[path]) else { return 0 };
    let Ok(fd) = (unsafe { sys::openat(sys::AT_FDCWD, p.ptr(), sys::O_RDONLY | sys::O_DIRECTORY | sys::O_CLOEXEC, 0) }) else {
        return 0;
    };
    let mut removed = 0;
    let mut buf = [0u8; 8192];
    loop {
        let got = match unsafe { sys::getdents64(fd, buf.as_mut_ptr(), buf.len()) } {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        let mut off = 0;
        while off < got {
            let reclen = u16::from_ne_bytes([buf[off + 16], buf[off + 17]]) as usize;
            let dtype = buf[off + 18];
            let first = buf[off + 19];
            let at = off + 19;
            off += reclen;
            if first == b'.' || first == b'_' || dtype == sys::DT_DIR {
                continue;
            }
            if unsafe { sys::unlinkat(fd, buf.as_ptr().add(at), 0) }.is_ok() {
                removed += 1;
            }
        }
    }
    let _ = sys::close(fd);
    removed
}
