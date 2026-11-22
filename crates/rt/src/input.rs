//! Input: list a directory, map every input file into one read-only reservation, release what
//! the parser has consumed, and a pread reader for kernels without fault-around.
//!
//! # The reservation
//! One anonymous PROT_READ mapping holds all files. Each file is mapped over it with MAP_FIXED
//! at a 2 MB-aligned address, largest file first, in a slot of `up(up(len, 4K) + 4K, 2M)` bytes,
//! after a 2 MB lead. So for every file:
//! - at least 4096 readable zero bytes lie after `ptr + len` (the rest of the last file page is
//!   zero-filled by the kernel, then the anonymous zero region);
//! - at least 4096 readable bytes lie before `ptr` (the lead, or the previous slot's zero tail);
//! - an empty file has `len == 0` and `ptr` pointing at zeros.
//!
//! # Releasing consumed input
//! `consumed(file, from, to)` takes ranges in any order from any thread. Each 2 MB bin of a file
//! counts its consumed bytes with one atomic add; the caller that completes a bin then releases
//! (MADV_DONTNEED) the run of complete bins starting at the file's release cursor, in runs of at
//! least `RUN_MIN` bins (or the file's last bins). Only file pages are released, never the zero
//! tail; a byte range must be reported at most once. `flush()` releases every complete bin left.

use core::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::*};

use crate::sys::{self, Errno, HUGE, PAGE, Res, Statx};

pub const MAX_FILES: usize = 4096;
pub const NAME_MAX: usize = 255;
const BIN: usize = HUGE;
const BIN_SHIFT: u32 = 21;
/// Release runs of at least this many complete bins (32 MB) except at a file's end.
pub const RUN_MIN: u32 = 16;
/// And at most this many per madvise call (64 MB: about 3 ms on one worker at 600 cycles/page).
pub const RUN_MAX: u32 = 32;

#[repr(C)]
pub struct Entry {
    pub size: u64,
    pub nlen: u16,
    pub dot: bool,
    name: [u8; NAME_MAX + 1],
}

impl Entry {
    pub fn name(&self) -> &[u8] {
        &self.name[..self.nlen as usize]
    }
}

/// The `*.csv` entries of one directory that are regular files (symlinks followed), sorted by
/// name bytes. Names starting with `.` are kept apart in `dots()`; the caller decides.
pub struct Listing {
    pub dirfd: i32,
    ents: *mut Entry,
    n: usize,
    ndots: usize,
    /// The directory did not exist (no files; not an error, as v1 treats it).
    pub missing: bool,
}

unsafe impl Send for Listing {}
unsafe impl Sync for Listing {}

impl Listing {
    /// Regular `*.csv` files not starting with `.`.
    pub fn files(&self) -> &[Entry] {
        if self.n == 0 { &[] } else { unsafe { core::slice::from_raw_parts(self.ents, self.n) } }
    }
    /// `*.csv` files whose name starts with `.`.
    pub fn dots(&self) -> &[Entry] {
        if self.ndots == 0 { &[] } else { unsafe { core::slice::from_raw_parts(self.ents.add(MAX_FILES), self.ndots) } }
    }
}

/// Lists `dir` (no trailing NUL needed) with getdents64 and statx.
pub fn scan(dir: &[u8]) -> Res<Listing> {
    let p = sys::CPath::new(&[dir]).ok_or(Errno::ENAMETOOLONG)?;
    let fd = match unsafe { sys::openat(sys::AT_FDCWD, p.ptr(), sys::O_RDONLY | sys::O_DIRECTORY | sys::O_CLOEXEC, 0) } {
        Ok(fd) => fd,
        Err(Errno::ENOENT) => return Ok(Listing { dirfd: -1, ents: core::ptr::null_mut(), n: 0, ndots: 0, missing: true }),
        Err(e) => return Err(e),
    };
    let ents = sys::map_anon(2 * MAX_FILES * core::mem::size_of::<Entry>())? as *mut Entry;
    let mut l = Listing { dirfd: fd, ents, n: 0, ndots: 0, missing: false };
    let mut buf = [0u8; 16384];
    loop {
        let got = unsafe { sys::getdents64(fd, buf.as_mut_ptr(), buf.len())? };
        if got == 0 {
            break;
        }
        let mut off = 0;
        while off < got {
            let reclen = u16::from_ne_bytes([buf[off + 16], buf[off + 17]]) as usize;
            let dtype = buf[off + 18];
            let name_at = off + 19;
            let mut nl = 0;
            while name_at + nl < off + reclen && buf[name_at + nl] != 0 {
                nl += 1;
            }
            let name = &buf[name_at..name_at + nl];
            off += reclen;
            if name.len() < 4 || !name.ends_with(b".csv") || dtype == sys::DT_DIR {
                continue;
            }
            // statx relative to the directory (follows symlinks): type and size.
            let mut cname = [0u8; NAME_MAX + 1];
            cname[..nl].copy_from_slice(name);
            let mut st = Statx::zeroed();
            if unsafe { sys::statx(fd, cname.as_ptr(), 0, sys::STATX_TYPE | sys::STATX_SIZE, &mut st) }.is_err() {
                continue;
            }
            if st.mode & sys::S_IFMT != sys::S_IFREG {
                continue;
            }
            let dot = name[0] == b'.';
            let (slot, count) = if dot { (MAX_FILES + l.ndots, &mut l.ndots) } else { (l.n, &mut l.n) };
            if *count >= MAX_FILES {
                return Err(Errno::ERANGE);
            }
            unsafe {
                let e = &mut *ents.add(slot);
                e.size = st.size;
                e.nlen = nl as u16;
                e.dot = dot;
                e.name[..nl].copy_from_slice(name);
            }
            *count += 1;
        }
    }
    unsafe {
        sort(core::slice::from_raw_parts_mut(ents, l.n));
        sort(core::slice::from_raw_parts_mut(ents.add(MAX_FILES), l.ndots));
    }
    Ok(l)
}

/// Insertion sort by name bytes (n is small; entries move by swapping their 272 bytes).
fn sort(e: &mut [Entry]) {
    for i in 1..e.len() {
        let mut j = i;
        while j > 0 && e[j - 1].name() > e[j].name() {
            e.swap(j - 1, j);
            j -= 1;
        }
    }
}

// ------------------------------------------------------------------------------------------
// The mapping.

/// One mapped input file.
#[repr(C)]
pub struct File {
    pub ptr: *const u8,
    pub len: usize,
    pub fd: i32,
    /// Which of the listings given to `map` it came from, and its position there.
    pub stream: u16,
    pub index: u16,
    bin0: u32,
    nbins: u32,
    cursor: AtomicU32,
}

// A File points at read-only mapped bytes that live until exit.
unsafe impl Send for File {}
unsafe impl Sync for File {}

impl File {
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        if self.len == 0 { &[] } else { unsafe { core::slice::from_raw_parts(self.ptr, self.len) } }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Advice {
    Normal,
    Sequential,
    WillNeed,
    /// MADV_POPULATE_READ: map every page now (Linux 5.14+).
    PopulateRead,
    DontNeed,
}

pub struct Inputs {
    base: usize,
    span: usize,
    files: *mut File,
    n: usize,
    /// Start of each stream's files in `files`.
    first: [u16; 9],
    nstreams: usize,
    bins: *mut AtomicU32,
    release: AtomicU32,
    run_min: AtomicU32,
    run_max: AtomicU32,
    released_bytes: AtomicU64,
    release_calls: AtomicU64,
    /// Ticks spent in `map` (for the report).
    pub map_ticks: u64,
    /// Ticks in openat (sum), mmap (sum), the longest openat, the longest mmap.
    pub detail: [u64; 4],
}

unsafe impl Send for Inputs {}
unsafe impl Sync for Inputs {}

static INPUTS: AtomicUsize = AtomicUsize::new(0);

/// The reservation registered for `exit::leave` (address, length; 0 when none).
pub(crate) fn registered() -> Option<&'static Inputs> {
    let p = INPUTS.load(Acquire);
    if p == 0 { None } else { Some(unsafe { &*(p as *const Inputs) }) }
}

/// What to map: a listing and whether its dotfiles are included (after its regular files).
pub struct Stream<'a> {
    pub listing: &'a Listing,
    pub dots: bool,
}

/// Maps every file of `streams` (at most 8) into one reservation. Files are numbered globally
/// in stream order, each stream's files in name order (dotfiles last when included).
///
/// The returned value must stay alive (and in place) until exit; `map` registers it with
/// `exit::leave` when `register` is true.
pub fn map(streams: &[Stream], out: &'static mut core::mem::MaybeUninit<Inputs>, register: bool) -> Res<&'static Inputs> {
    let t0 = crate::clock::ticks();
    assert!(streams.len() <= 8);
    let mut n = 0;
    for s in streams {
        n += s.listing.files().len() + if s.dots { s.listing.dots().len() } else { 0 };
    }
    if n > MAX_FILES {
        return Err(Errno::ERANGE);
    }
    let meta = sys::map_anon(sys::up(n.max(1) * core::mem::size_of::<File>(), PAGE))? as *mut File;
    let mut first = [0u16; 9];
    let mut total = HUGE * 2;
    let mut nbins_all = 0usize;
    let mut k = 0;
    for (si, s) in streams.iter().enumerate() {
        first[si] = k as u16;
        let l = s.listing;
        let dots = if s.dots { l.dots() } else { &[] };
        for (idx, e) in l.files().iter().chain(dots.iter()).enumerate() {
            let len = e.size as usize;
            let nb = len.div_ceil(BIN) as u32;
            unsafe {
                meta.add(k).write(File {
                    ptr: core::ptr::null(),
                    len,
                    fd: -1,
                    stream: si as u16,
                    index: idx as u16,
                    bin0: nbins_all as u32,
                    nbins: nb,
                    cursor: AtomicU32::new(0),
                });
            }
            nbins_all += nb as usize;
            total += slot_bytes(len);
            k += 1;
        }
    }
    first[streams.len()] = k as u16;
    let bins = sys::map_anon(sys::up(nbins_all.max(1) * 4, PAGE))? as *mut AtomicU32;
    let raw = unsafe { sys::mmap(0, total + HUGE, sys::PROT_READ, sys::MAP_PRIVATE | sys::MAP_ANONYMOUS | sys::MAP_NORESERVE, -1, 0)? } as usize;
    let base = sys::up(raw, HUGE);
    // Largest first.
    let mut order = [0u16; MAX_FILES];
    for (i, o) in order.iter_mut().enumerate().take(n) {
        *o = i as u16;
    }
    let files = unsafe { core::slice::from_raw_parts_mut(meta, n) };
    let ord = &mut order[..n];
    for i in 1..n {
        let mut j = i;
        while j > 0 && files[ord[j - 1] as usize].len < files[ord[j] as usize].len {
            ord.swap(j - 1, j);
            j -= 1;
        }
    }
    let mut at = base + HUGE;
    let mut detail = [0u64; 4];
    for &fi in ord.iter() {
        let f = &mut files[fi as usize];
        let s = &streams[f.stream as usize];
        let l = s.listing;
        let nf = l.files().len();
        let e = if (f.index as usize) < nf { &l.files()[f.index as usize] } else { &l.dots()[f.index as usize - nf] };
        let mut cname = [0u8; NAME_MAX + 1];
        cname[..e.name().len()].copy_from_slice(e.name());
        let ta = crate::clock::ticks();
        let fd = unsafe { sys::openat(l.dirfd, cname.as_ptr(), sys::O_RDONLY | sys::O_CLOEXEC, 0)? };
        let tb = crate::clock::ticks();
        f.fd = fd;
        if f.len > 0 {
            unsafe { sys::mmap(at, f.len, sys::PROT_READ, sys::MAP_PRIVATE | sys::MAP_FIXED, fd, 0)? };
        }
        let tc = crate::clock::ticks();
        detail[0] += tb - ta;
        detail[1] += tc - tb;
        detail[2] = detail[2].max(tb - ta);
        detail[3] = detail[3].max(tc - tb);
        f.ptr = at as *const u8;
        at += slot_bytes(f.len);
    }
    let inp = out.write(Inputs {
        base,
        span: at - base,
        files: meta,
        n,
        first,
        nstreams: streams.len(),
        bins,
        release: AtomicU32::new(1),
        run_min: AtomicU32::new(RUN_MIN),
        run_max: AtomicU32::new(RUN_MAX),
        released_bytes: AtomicU64::new(0),
        release_calls: AtomicU64::new(0),
        map_ticks: crate::clock::ticks() - t0,
        detail,
    });
    if register {
        INPUTS.store(inp as *const Inputs as usize, Release);
    }
    Ok(inp)
}

#[inline]
fn slot_bytes(len: usize) -> usize {
    sys::up(sys::up(len, PAGE) + PAGE, HUGE)
}

impl Inputs {
    #[inline]
    pub fn files(&self) -> &[File] {
        unsafe { core::slice::from_raw_parts(self.files, self.n) }
    }
    #[inline]
    pub fn file(&self, g: usize) -> &File {
        &self.files()[g]
    }
    /// The files of stream `s` (global numbers `first(s)..first(s) + len`).
    pub fn stream(&self, s: usize) -> &[File] {
        &self.files()[self.first[s] as usize..self.first[s + 1] as usize]
    }
    pub fn first(&self, s: usize) -> usize {
        self.first[s] as usize
    }
    pub fn streams(&self) -> usize {
        self.nstreams
    }
    /// Total bytes of all files.
    pub fn bytes(&self) -> u64 {
        self.files().iter().map(|f| f.len as u64).sum()
    }
    /// The reservation (for releasing at exit).
    pub fn reservation(&self) -> (usize, usize) {
        (self.base, self.span)
    }

    /// Advice over `[from, to)` of file `g`, widened to whole pages and clipped to the file.
    pub fn advise(&self, g: usize, from: usize, to: usize, a: Advice) -> Res<()> {
        let f = self.file(g);
        let end = sys::up(to.min(f.len), PAGE);
        let start = sys::down(from.min(f.len), PAGE);
        if f.len == 0 || start >= end {
            return Ok(());
        }
        let adv = match a {
            Advice::Normal => sys::MADV_NORMAL,
            Advice::Sequential => sys::MADV_SEQUENTIAL,
            Advice::WillNeed => sys::MADV_WILLNEED,
            Advice::PopulateRead => sys::MADV_POPULATE_READ,
            Advice::DontNeed => sys::MADV_DONTNEED,
        };
        unsafe { sys::madvise(f.ptr as usize + start, end - start, adv) }
    }

    /// Turns release-behind on or off (on by default). Off: `consumed` only counts nothing.
    pub fn set_release(&self, on: bool) {
        self.release.store(on as u32, Relaxed);
    }

    /// Release runs of `min..=max` complete 2 MB bins per madvise call (default 16..=32).
    pub fn set_runs(&self, min: u32, max: u32) {
        self.run_min.store(min.max(1), Relaxed);
        self.run_max.store(max.max(min.max(1)), Relaxed);
    }

    /// Reports `[from, to)` of file `g` as consumed. Any order, any thread, each byte at most
    /// once. Whole complete 2 MB bins are released in runs; the cost on the common path is one
    /// atomic add per bin the range touches.
    #[inline]
    pub fn consumed(&self, g: usize, from: usize, to: usize) {
        if self.release.load(Relaxed) == 0 {
            return;
        }
        let f = self.file(g);
        let to = to.min(f.len);
        if from >= to {
            return;
        }
        let mut any = false;
        let mut b = (from >> BIN_SHIFT) as u32;
        let last = ((to - 1) >> BIN_SHIFT) as u32;
        while b <= last {
            let lo = (b as usize) << BIN_SHIFT;
            let hi = (lo + BIN).min(f.len);
            let part = (to.min(hi) - from.max(lo)) as u32;
            let size = (hi - lo) as u32;
            let c = unsafe { &*self.bins.add((f.bin0 + b) as usize) };
            if c.fetch_add(part, AcqRel) + part == size {
                any = true;
            }
            b += 1;
        }
        if any {
            self.advance(f, false);
        }
    }

    #[inline]
    fn bin_done(&self, f: &File, b: u32) -> bool {
        let lo = (b as usize) << BIN_SHIFT;
        let size = ((lo + BIN).min(f.len) - lo) as u32;
        unsafe { (*self.bins.add((f.bin0 + b) as usize)).load(Acquire) == size }
    }

    /// Releases runs of complete bins from the file's cursor.
    fn advance(&self, f: &File, all: bool) {
        let (rmin, rmax) = (self.run_min.load(Relaxed), self.run_max.load(Relaxed));
        loop {
            let c = f.cursor.load(Acquire);
            let mut k = c;
            while k < f.nbins && k - c < rmax && self.bin_done(f, k) {
                k += 1;
            }
            let run = k - c;
            if run == 0 || (run < rmin && k < f.nbins && !all) {
                return;
            }
            if f.cursor.compare_exchange(c, k, AcqRel, Relaxed).is_err() {
                continue;
            }
            let lo = (c as usize) << BIN_SHIFT;
            let hi = sys::up(((k as usize) << BIN_SHIFT).min(f.len), PAGE);
            unsafe {
                let _ = sys::madvise(f.ptr as usize + lo, hi - lo, sys::MADV_DONTNEED);
            }
            self.released_bytes.fetch_add((hi - lo) as u64, Relaxed);
            self.release_calls.fetch_add(1, Relaxed);
        }
    }

    /// Releases every complete bin not yet released (call once the parse is over).
    pub fn flush(&self) {
        for f in self.files() {
            self.advance(f, true);
        }
    }

    /// Bytes released so far and the number of madvise calls that did it.
    pub fn release_stats(&self) -> (u64, u64) {
        (self.released_bytes.load(Relaxed), self.release_calls.load(Relaxed))
    }

    /// The largest file's global number.
    pub fn largest(&self) -> Option<usize> {
        (0..self.n).max_by_key(|&g| self.file(g).len)
    }
}

// ------------------------------------------------------------------------------------------
// Fault-around probe.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FaultAround {
    /// Touching the page after a faulted one cost no new fault.
    Working,
    /// It cost a fault: use the pread reader (or populate per range).
    Off,
    /// A major fault happened (file not cached) or the file is too small to tell.
    Unknown,
}

/// Touches the first two pages of the largest file and compares the fault counters.
pub fn probe_fault_around(inp: &Inputs) -> FaultAround {
    let Some(g) = inp.largest() else { return FaultAround::Unknown };
    let f = inp.file(g);
    if f.len < 3 * PAGE {
        return FaultAround::Unknown;
    }
    let r0 = sys::getrusage(sys::RUSAGE_THREAD);
    unsafe { core::ptr::read_volatile(f.ptr) };
    let r1 = sys::getrusage(sys::RUSAGE_THREAD);
    unsafe { core::ptr::read_volatile(f.ptr.add(PAGE)) };
    let r2 = sys::getrusage(sys::RUSAGE_THREAD);
    if r2.majflt != r0.majflt || r1.minflt == r0.minflt {
        return FaultAround::Unknown;
    }
    if r2.minflt == r1.minflt { FaultAround::Working } else { FaultAround::Off }
}

// ------------------------------------------------------------------------------------------
// Line ownership (the same rule for mapped bytes and the reader).

/// The lines a chunk `[from, to)` owns: every line whose first byte lies in `[from, to)`, a
/// line starting at 0 or just after a `\n`. Returns the byte range `[s, e)` they cover: `s` is
/// the first line start at or after `from`, `e` the end (just past its `\n`, or EOF) of the last
/// line starting before `to`. Consecutive chunks of a partition give consecutive ranges.
pub fn owned(data: &[u8], from: usize, to: usize) -> (usize, usize) {
    let len = data.len();
    let to = to.min(len);
    let s = if from == 0 { 0 } else { next_line(data, from - 1) };
    if s >= to {
        return (s.min(len), s.min(len));
    }
    let e = next_line(data, to - 1);
    (s, e)
}

/// Index just past the first `\n` at or after `i` (len if none).
#[inline]
fn next_line(data: &[u8], i: usize) -> usize {
    let mut k = i;
    while k < data.len() {
        if data[k] == b'\n' {
            return k + 1;
        }
        k += 1;
    }
    data.len()
}

// ------------------------------------------------------------------------------------------
// The pread reader.

/// Bytes kept zero (and readable) before and after a reader buffer's data.
pub const READER_PRE: usize = PAGE;
pub const READER_POST: usize = PAGE;

/// Readable bytes around a span: `[ptr - pre, ptr + len + post)`; the `post` bytes are zero.
#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub ptr: *const u8,
    pub len: usize,
    pub pre: usize,
    pub post: usize,
}

impl Span {
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Per-worker read buffers for kernels where fault-around is off. 128 KB measured best on the
/// x86 box (a buffer that stays in L2).
pub struct Reader {
    base: usize,
    per: usize,
    cap: usize,
}

unsafe impl Sync for Reader {}

impl Reader {
    pub fn new(workers: usize, cap: usize) -> Res<Reader> {
        let cap = sys::up(cap.max(PAGE), PAGE);
        let per = READER_PRE + cap + READER_POST + PAGE;
        let base = sys::map_anon(per * workers.max(1))? as usize;
        Ok(Reader { base, per, cap })
    }
    pub fn cap(&self) -> usize {
        self.cap
    }
    #[inline]
    fn buf(&self, w: usize) -> *mut u8 {
        (self.base + w * self.per + READER_PRE) as *mut u8
    }

    /// The owned lines of chunk `[from, to)` of `f` (see `owned`), as consecutive spans that
    /// end just after a `\n` (or at EOF). A line longer than the buffer arrives in pieces; then
    /// `LineStream::split` is true for each piece but the last.
    pub fn lines<'a>(&'a self, w: usize, f: &'a File, from: usize, to: usize) -> LineStream<'a> {
        LineStream { r: self, w, f, at: from, to: to.min(f.len), started: from == 0, ended: false, split: false }
    }
}

pub struct LineStream<'a> {
    r: &'a Reader,
    w: usize,
    f: &'a File,
    at: usize,
    to: usize,
    started: bool,
    ended: bool,
    /// The last span ended inside a line longer than the buffer.
    pub split: bool,
}

impl LineStream<'_> {
    /// The next span, or None when every owned line has been delivered. The span stays valid
    /// until the next call on this worker's reader.
    pub fn next_span(&mut self) -> Res<Option<Span>> {
        let buf = self.r.buf(self.w);
        let len = self.f.len;
        if self.ended {
            return Ok(None);
        }
        if !self.started {
            // The first line start at or after `at`: just past the first '\n' at or after at - 1.
            let mut probe = [0u8; 4096];
            let mut p = self.at - 1;
            loop {
                let got = sys::pread_full(self.f.fd, &mut probe, p as u64)?;
                if got == 0 {
                    self.at = len;
                    break;
                }
                if let Some(k) = probe[..got].iter().position(|&c| c == b'\n') {
                    self.at = p + k + 1;
                    break;
                }
                p += got;
            }
            self.started = true;
        }
        // Past `to` only while finishing the chunk's last line (the previous piece was split).
        if (self.at >= self.to && !self.split) || self.at >= len {
            self.ended = true;
            return Ok(None);
        }
        let at = self.at;
        let want = self.r.cap.min(len - at);
        let got = unsafe { sys::pread_full(self.f.fd, core::slice::from_raw_parts_mut(buf, want), at as u64)? };
        if got == 0 {
            self.ended = true;
            return Ok(None);
        }
        let data = unsafe { core::slice::from_raw_parts(buf, got) };
        let eof = at + got == len;
        // From here on, the first newline ends the chunk (0 when already past `to - 1`).
        let last = (self.to - 1).saturating_sub(at);
        let mut e = None;
        if last < got {
            if let Some(k) = data[last..].iter().position(|&c| c == b'\n') {
                e = Some(last + k + 1);
                self.ended = true;
            }
        }
        let e = match e {
            Some(e) => e,
            None if eof => {
                self.ended = true;
                got
            }
            None => match data.iter().rposition(|&c| c == b'\n') {
                Some(k) => k + 1,
                None => got,
            },
        };
        self.split = !self.ended && e == got && data[e - 1] != b'\n';
        unsafe { core::ptr::write_bytes(buf.add(e), 0, READER_POST) };
        self.at = at + e;
        Ok(Some(Span { ptr: buf, len: e, pre: READER_PRE, post: READER_POST }))
    }
}
