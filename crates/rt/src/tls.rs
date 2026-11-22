//! Static thread-local storage for threads this crate creates with `clone`.
//!
//! A raw thread must have its own copy of the executable's TLS segment (Rust `thread_local!`
//! lives there, including the parser's scratch) and a thread control block that the C library
//! accepts. The segment is found through `__ehdr_start` (program headers, PT_TLS), copied from
//! its initialisation image, and placed where the linker's local-exec offsets expect it:
//!
//! - x86-64 (variant II): block ends at the thread pointer TP, `TP - align_up(memsz, align)`;
//!   `[TP]` holds TP itself (the ABI's self pointer).
//! - aarch64 (variant I): block starts at `TP + align_up(16, align)`.
//!
//! With musl (the shipped static build) the calling thread's `struct pthread` (200 bytes in musl
//! 1.2.5; at TP on x86-64, just below TP on aarch64) is copied and its self pointer and `tsd`
//! array are made the new thread's own, so errno, `pthread_getspecific` and Rust's TLS destructor
//! registration work on workers. The layout is checked at run time: `__rt_tls_magic` (an
//! initialised TLS word defined below) must read back on every worker, and `pool::start` falls
//! back to `fail()` if it does not.
//!
//! What a raw thread cannot do: call C library functions that rely on musl knowing the thread
//! exists. musl skips its internal locks (malloc included) until the first `pthread_create`;
//! see `pool` for the rule this implies.

use core::arch::{asm, global_asm};
use core::sync::atomic::{AtomicUsize, Ordering::*};

pub const MAGIC: u64 = 0x5254_4d41_4749_4331;

#[cfg(target_arch = "x86_64")]
global_asm!(
    ".pushsection .tdata,\"awT\",@progbits",
    ".p2align 3",
    ".globl __rt_tls_magic",
    ".hidden __rt_tls_magic",
    "__rt_tls_magic: .quad 0x52544d4147494331",
    ".popsection",
    ".pushsection .tbss,\"awT\",@nobits",
    ".p2align 3",
    ".globl __rt_tls_widx",
    ".hidden __rt_tls_widx",
    "__rt_tls_widx: .zero 8",
    ".popsection",
);

#[cfg(target_arch = "aarch64")]
global_asm!(
    ".pushsection .tdata,\"awT\",%progbits",
    ".p2align 3",
    ".globl __rt_tls_magic",
    ".hidden __rt_tls_magic",
    "__rt_tls_magic: .quad 0x52544d4147494331",
    ".popsection",
    ".pushsection .tbss,\"awT\",%nobits",
    ".p2align 3",
    ".globl __rt_tls_widx",
    ".hidden __rt_tls_widx",
    "__rt_tls_widx: .zero 8",
    ".popsection",
);

/// The thread pointer of the calling thread.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn tp() -> usize {
    let v: usize;
    unsafe { asm!("mov {}, qword ptr fs:[0]", out(reg) v, options(nostack, readonly, preserves_flags)) };
    v
}
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn tp() -> usize {
    let v: usize;
    unsafe { asm!("mrs {}, tpidr_el0", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

/// The per-thread worker word: worker index + 1, or 0 on threads the pool does not know.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn widx() -> usize {
    let v: usize;
    unsafe { asm!("movq %fs:__rt_tls_widx@tpoff, {}", out(reg) v, options(att_syntax, nostack, readonly, preserves_flags)) };
    v
}
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn set_widx(v: usize) {
    unsafe { asm!("movq {}, %fs:__rt_tls_widx@tpoff", in(reg) v, options(att_syntax, nostack, preserves_flags)) };
}
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn magic() -> u64 {
    let v: u64;
    unsafe { asm!("movq %fs:__rt_tls_magic@tpoff, {}", out(reg) v, options(att_syntax, nostack, readonly, preserves_flags)) };
    v
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn widx_addr() -> *mut usize {
    let a: usize;
    unsafe {
        asm!("mrs {a}, tpidr_el0",
             "add {a}, {a}, #:tprel_hi12:__rt_tls_widx, lsl #12",
             "add {a}, {a}, #:tprel_lo12_nc:__rt_tls_widx",
             a = out(reg) a, options(nomem, nostack, preserves_flags))
    };
    a as *mut usize
}
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn widx() -> usize {
    unsafe { core::ptr::read_volatile(widx_addr()) }
}
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn set_widx(v: usize) {
    unsafe { core::ptr::write_volatile(widx_addr(), v) }
}
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn magic() -> u64 {
    let a: usize;
    unsafe {
        asm!("mrs {a}, tpidr_el0",
             "add {a}, {a}, #:tprel_hi12:__rt_tls_magic, lsl #12",
             "add {a}, {a}, #:tprel_lo12_nc:__rt_tls_magic",
             a = out(reg) a, options(nomem, nostack, preserves_flags));
        core::ptr::read_volatile(a as *const u64)
    }
}

// ------------------------------------------------------------------------------------------
// The TLS segment.

unsafe extern "C" {
    static __ehdr_start: u8;
}

#[repr(C)]
struct Ehdr {
    ident: [u8; 16],
    typ: u16,
    machine: u16,
    version: u32,
    entry: u64,
    phoff: u64,
    shoff: u64,
    flags: u32,
    ehsize: u16,
    phentsize: u16,
    phnum: u16,
}

#[repr(C)]
struct Phdr {
    typ: u32,
    flags: u32,
    offset: u64,
    vaddr: u64,
    paddr: u64,
    filesz: u64,
    memsz: u64,
    align: u64,
}

const PT_LOAD: u32 = 1;
const PT_PHDR: u32 = 6;
const PT_TLS: u32 = 7;

#[derive(Clone, Copy, Debug, Default)]
pub struct Segment {
    /// Run-time address of the initialisation image.
    pub image: usize,
    pub filesz: usize,
    pub memsz: usize,
    pub align: usize,
}

/// The run-time address range covered by the executable's PT_LOAD segments (code, data, .bss).
pub fn image_range() -> (usize, usize) {
    unsafe {
        let eh = core::ptr::addr_of!(__ehdr_start) as usize;
        let e = &*(eh as *const Ehdr);
        let ph = eh + e.phoff as usize;
        let phs = |i: usize| &*((ph + i * e.phentsize as usize) as *const Phdr);
        let mut bias = None;
        for i in 0..e.phnum as usize {
            if phs(i).typ == PT_PHDR {
                bias = Some(ph.wrapping_sub(phs(i).vaddr as usize));
            }
        }
        let mut lo = usize::MAX;
        let mut hi = 0;
        for i in 0..e.phnum as usize {
            let p = phs(i);
            if p.typ == PT_LOAD {
                if bias.is_none() && p.offset == 0 {
                    bias = Some(eh.wrapping_sub(p.vaddr as usize));
                }
                lo = lo.min(p.vaddr as usize);
                hi = hi.max((p.vaddr + p.memsz) as usize);
            }
        }
        let b = bias.unwrap_or(0);
        (crate::sys::down(lo.wrapping_add(b), crate::sys::PAGE), crate::sys::up(hi.wrapping_add(b), crate::sys::PAGE))
    }
}

/// The executable's PT_TLS segment (memsz 0 if it has none).
pub fn segment() -> Segment {
    unsafe {
        let eh = core::ptr::addr_of!(__ehdr_start) as usize;
        let e = &*(eh as *const Ehdr);
        let ph = eh + e.phoff as usize;
        let phs = |i: usize| &*((ph + i * e.phentsize as usize) as *const Phdr);
        let mut bias: Option<usize> = None;
        for i in 0..e.phnum as usize {
            let p = phs(i);
            if p.typ == PT_PHDR {
                bias = Some(ph.wrapping_sub(p.vaddr as usize));
            }
        }
        if bias.is_none() {
            for i in 0..e.phnum as usize {
                let p = phs(i);
                if p.typ == PT_LOAD && p.offset == 0 {
                    bias = Some(eh.wrapping_sub(p.vaddr as usize));
                    break;
                }
            }
        }
        let bias = bias.unwrap_or(0);
        for i in 0..e.phnum as usize {
            let p = phs(i);
            if p.typ == PT_TLS {
                return Segment {
                    image: bias.wrapping_add(p.vaddr as usize),
                    filesz: p.filesz as usize,
                    memsz: p.memsz as usize,
                    align: (p.align as usize).max(1),
                };
            }
        }
        Segment::default()
    }
}

/// Bytes before the TLS block (x86-64) or after it (aarch64) kept zero for a C library's own
/// static TLS (glibc test builds only; musl static has none).
const SLACK: usize = 64 << 10;
/// Room for the thread control block and a private `tsd` array.
const TCB: usize = 8192;
/// musl 1.2.5 `sizeof(struct pthread)` on x86-64 and aarch64.
const MUSL_PTHREAD: usize = 200;

/// Per-thread area size for `init_area`.
pub fn area_size(seg: &Segment) -> usize {
    let a = seg.align.max(64);
    crate::sys::up(SLACK + crate::sys::up(seg.memsz, seg.align) + a + TCB + a, crate::sys::PAGE)
}

static PARENT_TP: AtomicUsize = AtomicUsize::new(0);
/// 1 when the template's musl `struct pthread` passed the layout check (self pointer and tid).
static MUSL_OK: AtomicUsize = AtomicUsize::new(0);

/// Remembers the calling thread's TP as the template for new threads (called by `pool::start`)
/// and checks the musl layout assumptions on it.
pub fn capture_template() {
    let t = tp();
    PARENT_TP.store(t, Relaxed);
    let ok = if cfg!(target_env = "musl") {
        unsafe {
            if cfg!(target_arch = "x86_64") {
                *(t as *const usize) == t && *((t + 48) as *const i32) == crate::sys::gettid()
            } else {
                let s = t - MUSL_PTHREAD;
                *(s as *const usize) == s && *((s + 32) as *const i32) == crate::sys::gettid()
            }
        }
    } else {
        false
    };
    MUSL_OK.store(ok as usize, Relaxed);
}

/// Whether the musl thread block was recognised (false on glibc builds).
pub fn musl_layout_ok() -> bool {
    MUSL_OK.load(Relaxed) != 0
}

/// Where the C library keeps the thread id in a new thread's block (musl), for
/// CLONE_PARENT_SETTID: musl's raise() and abort() signal that id, so it must be the worker's own.
pub fn tid_field(tp: usize) -> usize {
    if !musl_layout_ok() {
        return 0;
    }
    if cfg!(target_arch = "x86_64") { tp + 48 } else { tp - MUSL_PTHREAD + 32 }
}

/// Fills a zeroed area of `area_size` bytes at `base` and returns the new thread pointer.
///
/// # Safety
/// `base` is writable, zeroed and `area_size(seg)` bytes long.
#[cfg(target_arch = "x86_64")]
pub unsafe fn init_area(seg: &Segment, base: usize) -> usize {
    let a = seg.align.max(64);
    let blk = crate::sys::up(seg.memsz, seg.align);
    let tp = crate::sys::up(base + SLACK + blk, a);
    let start = tp - blk;
    core::ptr::copy_nonoverlapping(seg.image as *const u8, start as *mut u8, seg.filesz);
    let ptp = PARENT_TP.load(Relaxed);
    if musl_layout_ok() {
        core::ptr::copy_nonoverlapping(ptp as *const u8, tp as *mut u8, MUSL_PTHREAD);
        // tsd (offset 128): a private, zeroed array of PTHREAD_KEYS_MAX pointers.
        *((tp + 128) as *mut usize) = tp + 1024;
    } else if !cfg!(target_env = "musl") && ptp != 0 && *(ptp as *const usize) == ptp {
        // glibc: copy the tcbhead_t (stack and pointer guards); fix tcb and self.
        core::ptr::copy_nonoverlapping(ptp as *const u8, tp as *mut u8, 72);
        *((tp + 16) as *mut usize) = tp;
    }
    *(tp as *mut usize) = tp;
    tp
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn init_area(seg: &Segment, base: usize) -> usize {
    let a = seg.align.max(64);
    let tp = crate::sys::up(base + TCB, a);
    let start = tp + crate::sys::up(16, seg.align);
    core::ptr::copy_nonoverlapping(seg.image as *const u8, start as *mut u8, seg.filesz);
    let ptp = PARENT_TP.load(Relaxed);
    if musl_layout_ok() {
        let src = ptp - MUSL_PTHREAD;
        let dst = tp - MUSL_PTHREAD;
        core::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, MUSL_PTHREAD);
        *(dst as *mut usize) = dst;
        *((dst + 112) as *mut usize) = tp - 4096 - 1024;
    }
    tp
}
