//! The safety net: any failure in the fast path becomes a slow, correct run of the general
//! engine. `fail(reason)` (callable from any thread, while other workers run) and the handlers
//! for SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT (a Rust panic with panic=abort) and SIGTRAP do:
//!
//! 1. claim the failure (one thread wins; others sleep until the exec ends them; a second fault
//!    on the claiming thread exits with status 72 instead of looping);
//! 2. write the reason to stderr;
//! 3. unlink every output file `out` created;
//! 4. restore the start-up CPU mask on this thread (execve keeps the caller's mask) and unblock
//!    all signals (execve keeps the mask too);
//! 5. execve `/proc/self/exe` with the same argv and environment plus `ENGINE_PATH=v1`
//!    (replacing any ENGINE_PATH). If ENGINE_PATH=v1 is already set, this run already is the
//!    fallback: exit with status 70 rather than loop.
//!
//! execve kills every other thread before the new image runs, so nothing of the fast path can
//! write after it. The general engine should still call `out::sweep` on its output folders when
//! started with ENGINE_PATH=v1, which closes the one race left (a file created by a thread that
//! was between registering and creating when the failure began, after a 20 ms wait).

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicI32, AtomicU32, Ordering::*};

use crate::fmt::Buf;
use crate::sys::{self, KSigaction, StackT};

static FAILER: AtomicI32 = AtomicI32::new(0);
static PARK: AtomicU32 = AtomicU32::new(0);

const ARG_BYTES: usize = 1 << 18;
const MAX_PTRS: usize = 16384;

struct Scratch {
    cmd: UnsafeCell<[u8; ARG_BYTES]>,
    env: UnsafeCell<[u8; ARG_BYTES + 32]>,
    argv: UnsafeCell<[usize; MAX_PTRS]>,
    envp: UnsafeCell<[usize; MAX_PTRS]>,
}
unsafe impl Sync for Scratch {}

/// Only the thread that won `FAILER` touches this (in .bss: costs nothing until a failure).
static SCRATCH: Scratch = Scratch {
    cmd: UnsafeCell::new([0; ARG_BYTES]),
    env: UnsafeCell::new([0; ARG_BYTES + 32]),
    argv: UnsafeCell::new([0; MAX_PTRS]),
    envp: UnsafeCell::new([0; MAX_PTRS]),
};

/// Abandons the fast path: removes partial output and re-executes this binary with
/// ENGINE_PATH=v1. Never returns.
pub fn fail(reason: &[u8]) -> ! {
    claim();
    let mut b = Buf::<512>::new();
    b.s(b"rt: fast path failed: ").s(reason).s(b"; re-running the general engine\n");
    b.flush(2);
    reexec()
}

fn claim() {
    let tid = sys::gettid();
    match FAILER.compare_exchange(0, tid, AcqRel, Acquire) {
        Ok(_) => {}
        Err(t) if t == tid => {
            crate::fmt::eprint(&[b"rt: fault inside the fallback path; giving up\n"]);
            sys::exit_group(72);
        }
        Err(_) => loop {
            // Another thread is failing; its execve ends this thread.
            unsafe {
                let _ = sys::futex_wait(PARK.as_ptr(), 0);
            }
        },
    }
}

fn reexec() -> ! {
    crate::out::remove_partial();
    if crate::pool::ORIG_SAVED.load(Acquire) != 0 {
        let mut set = crate::cpu::CpuSet::EMPTY;
        for (i, w) in set.bits.iter_mut().enumerate() {
            *w = crate::pool::ORIG_MASK[i].load(Relaxed);
        }
        let _ = crate::cpu::set_affinity(0, &set);
    }
    let none: u64 = 0;
    unsafe {
        let _ = sys::rt_sigprocmask(sys::SIG_SETMASK, &none, core::ptr::null_mut());
    }
    let cmd = unsafe { &mut *SCRATCH.cmd.get() };
    let env = unsafe { &mut *SCRATCH.env.get() };
    let argv = unsafe { &mut *SCRATCH.argv.get() };
    let envp = unsafe { &mut *SCRATCH.envp.get() };
    let mut nc = read_proc(b"/proc/self/cmdline", &mut cmd[..ARG_BYTES - 1]);
    let mut ne = read_proc(b"/proc/self/environ", &mut env[..ARG_BYTES]);
    if nc.is_none() || ne.is_none() {
        // Out of descriptors (the failure itself may be that): close 3 and up, which execve closes
        // anyway (everything here is opened close-on-exec), and read again.
        for fd in 3..8192 {
            let _ = sys::close(fd);
        }
        nc = read_proc(b"/proc/self/cmdline", &mut cmd[..ARG_BYTES - 1]);
        ne = read_proc(b"/proc/self/environ", &mut env[..ARG_BYTES]);
    }
    let (Some(nc), Some(ne)) = (nc, ne) else {
        crate::fmt::eprint(&[b"rt: cannot read /proc/self/cmdline or environ\n"]);
        sys::exit_group(71);
    };
    let mut na = 0;
    split(cmd, nc, argv, &mut na);
    argv[na] = 0;
    let mut nv = 0;
    let mut i = 0;
    while i < ne && nv < MAX_PTRS - 2 {
        let s = i;
        while i < ne && env[i] != 0 {
            i += 1;
        }
        let var = &env[s..i];
        i += 1;
        if var.is_empty() {
            continue;
        }
        if var.starts_with(b"ENGINE_PATH=") {
            if var == b"ENGINE_PATH=v1" {
                crate::fmt::eprint(&[b"rt: already in the general engine; not re-executing\n"]);
                sys::exit_group(70);
            }
            continue;
        }
        envp[nv] = env.as_ptr() as usize + s;
        nv += 1;
    }
    // Append ENGINE_PATH=v1 after the copied block.
    let tail = ARG_BYTES;
    let v = b"ENGINE_PATH=v1\0";
    env[tail..tail + v.len()].copy_from_slice(v);
    envp[nv] = env.as_ptr() as usize + tail;
    envp[nv + 1] = 0;
    let e = unsafe { sys::execve(b"/proc/self/exe\0".as_ptr(), argv.as_ptr() as *const *const u8, envp.as_ptr() as *const *const u8) };
    let mut b = Buf::<128>::new();
    b.s(b"rt: execve failed, errno ").u(e.0 as u64).c(b'\n');
    b.flush(2);
    sys::exit_group(71)
}

fn read_proc(path: &[u8], buf: &mut [u8]) -> Option<usize> {
    let n = sys::read_small(path, buf).ok()?;
    if n == buf.len() {
        return None;
    }
    buf[n] = 0;
    Some(n)
}

fn split(b: &[u8], n: usize, out: &mut [usize; MAX_PTRS], k: &mut usize) {
    let mut i = 0;
    while i < n && *k < MAX_PTRS - 1 {
        out[*k] = b.as_ptr() as usize + i;
        *k += 1;
        while i < n && b[i] != 0 {
            i += 1;
        }
        i += 1;
    }
}

// ------------------------------------------------------------------------------------------
// Signals.

#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    ".pushsection .text.__rt_restorer,\"ax\",@progbits",
    ".globl __rt_restorer",
    ".hidden __rt_restorer",
    "__rt_restorer:",
    "mov eax, 15",
    "syscall",
    "ud2",
    ".popsection",
);

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    fn __rt_restorer();
}

pub const SIGNALS: [i32; 6] = [sys::SIGSEGV, sys::SIGBUS, sys::SIGILL, sys::SIGFPE, sys::SIGABRT, sys::SIGTRAP];

extern "C" fn on_signal(sig: i32, info: *const u8, ctx: *const u8) {
    claim();
    let addr = if info.is_null() { 0 } else { unsafe { *(info.add(16) as *const u64) } };
    // The faulting instruction (ucontext: x86-64 gregs[REG_RIP] at 168; aarch64 sigcontext.pc at
    // 176 + 264), reported as an offset into the executable for addr2line.
    let pc = if ctx.is_null() {
        0
    } else if cfg!(target_arch = "x86_64") {
        unsafe { *(ctx.add(168) as *const u64) }
    } else {
        unsafe { *(ctx.add(440) as *const u64) }
    };
    let (img, _) = crate::tls::image_range();
    // x86-64: gregs[REG_RSP] at 160. After a call into the null page, [rsp] is the caller's
    // return address; it is read only then (after a stack overflow rsp is in the guard page).
    let (sp, ret) = if cfg!(target_arch = "x86_64") && !ctx.is_null() {
        let sp = unsafe { *(ctx.add(160) as *const u64) };
        (sp, if pc < 4096 && sp != 0 { unsafe { *(sp as *const u64) } } else { 0 })
    } else {
        (0, 0)
    };
    let mut b = Buf::<320>::new();
    b.s(b"rt: fast path failed: signal ").u(sig as u64);
    b.s(match sig {
        sys::SIGSEGV => b" (SIGSEGV)" as &[u8],
        sys::SIGBUS => b" (SIGBUS)",
        sys::SIGILL => b" (SIGILL)",
        sys::SIGFPE => b" (SIGFPE)",
        sys::SIGABRT => b" (SIGABRT)",
        _ => b"",
    });
    b.s(b" addr ").hex(addr).s(b" pc +").hex(pc.wrapping_sub(img as u64)).s(b" ret +").hex(ret.wrapping_sub(img as u64)).s(b" sp ").hex(sp).s(b" worker ").i(crate::tls::widx() as i64 - 1).s(b" tid ").u(sys::gettid() as u64).s(b"; re-running the general engine\n");
    b.flush(2);
    reexec()
}

/// Installs the fault handlers process-wide (on the alternate stack of the faulting thread).
pub fn install() {
    #[cfg(target_arch = "x86_64")]
    let (flags, restorer) = (sys::SA_SIGINFO | sys::SA_ONSTACK | sys::SA_NODEFER | sys::SA_RESTORER, __rt_restorer as unsafe extern "C" fn() as usize);
    #[cfg(target_arch = "aarch64")]
    let (flags, restorer) = (sys::SA_SIGINFO | sys::SA_ONSTACK | sys::SA_NODEFER, 0usize);
    let act = KSigaction { handler: on_signal as extern "C" fn(i32, *const u8, *const u8) as usize, flags, restorer, mask: 0 };
    for s in SIGNALS {
        unsafe {
            let _ = sys::rt_sigaction(s, &act, core::ptr::null_mut());
        }
    }
}

/// Gives the calling thread its alternate signal stack `[base, base + pool::ALT_STACK)`.
pub fn thread_init(base: usize) {
    let ss = StackT { sp: base, flags: 0, size: crate::pool::ALT_STACK };
    unsafe {
        let _ = sys::sigaltstack(&ss, core::ptr::null_mut());
    }
}

/// True when this process runs as the fallback (ENGINE_PATH=v1 in /proc/self/environ).
/// Costs one small file read; the engine normally reads its environment itself.
pub fn in_fallback() -> bool {
    let mut buf = [0u8; 16384];
    let Ok(n) = sys::read_small(b"/proc/self/environ", &mut buf) else { return false };
    buf[..n].split(|&c| c == 0).any(|v| v == b"ENGINE_PATH=v1")
}
