//! Bins: records binned by an index. A bin append is a compare of the bin's write pointer with its
//! chunk end, a store and an increment. Chunks (512 records) are bumped from large arena blocks
//! owned by the bins, so nothing is ever copied or reallocated; memory is the records plus at
//! most one partly filled chunk per bin.
//!
//! Bins made by `from_blocks` take each chunk as one block of an rt block class instead (the
//! fast path's region, `v3::mem`): the bins never free those blocks; whoever consumes a chunk
//! frees it (dedup, per partition) or the class is reset (the driver).


/// Measurement builds: ENGINE_BINPF=0 turns the next-line prefetch off.
pub static BINPF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

#[inline(always)]
fn pf_on() -> bool {
    !cfg!(feature = "measure") || BINPF.load(std::sync::atomic::Ordering::Relaxed)
}

/// Records binned by an index, in chunks bumped from arena blocks.
pub struct Bins<T: Copy> {
    /// Per bin: write pointer and end of the current chunk (one pair: one cache access per push).
    ce: Vec<[*mut T; 2]>,
    full: Vec<Vec<(*const T, u32)>>,
    blocks: Vec<Vec<T>>,
    bump: *mut T,
    bump_end: *mut T,
    block: usize,
    chunk: usize,
    /// Chunks from this block class, one block each, for pool participant `w` (`from_blocks`).
    src: Option<(&'static rt::arena::Blocks, usize)>,
    /// Prefetch each append's next line for writing (`next_line_prefetch`; off by default).
    pf: bool,
}

// SAFETY: the raw pointers point into `blocks`, which the struct owns; moving the struct to
// another thread moves that ownership with it (the blocks' heap memory does not move).
unsafe impl<T: Copy + Send> Send for Bins<T> {}
// SAFETY: shared references only read finished chunks (`slices`), never the write pointers.
unsafe impl<T: Copy + Sync> Sync for Bins<T> {}

impl<T: Copy> Bins<T> {
    /// `n` bins; arena blocks of `block` records; chunks of 512 records.
    pub fn new(n: usize, block: usize) -> Self {
        Self::with_chunk(n, block, 512)
    }

    /// `n` bins, arena blocks of `block` records, chunks of `chunk` records (small chunks for
    /// sparse streams: every bin touched costs one chunk).
    pub fn with_chunk(n: usize, block: usize, chunk: usize) -> Self {
        let chunk = chunk.max(1);
        Bins {
            ce: vec![[std::ptr::null_mut(); 2]; n],
            full: (0..n).map(|_| Vec::new()).collect(),
            blocks: Vec::new(),
            bump: std::ptr::null_mut(),
            bump_end: std::ptr::null_mut(),
            block: block.div_ceil(chunk).max(1) * chunk,
            chunk,
            src: None,
            pf: false,
        }
    }

    /// `n` bins whose chunks are blocks of `src` (one block per chunk, as many records as fit),
    /// taken for pool participant `w`: an index no other thread uses with `src` at the same time.
    pub fn from_blocks(n: usize, src: &'static rt::arena::Blocks, w: usize) -> Self {
        let (bs, sz) = (src.block_size(), std::mem::size_of::<T>());
        assert!(bs >= sz && bs % sz == 0, "block class of {bs} bytes for records of {sz}");
        let mut b = Self::with_chunk(n, 1, bs / sz);
        b.src = Some((src, w));
        b
    }

    /// Turns the next-line prefetch on appends on or off.
    pub fn next_line_prefetch(mut self, on: bool) -> Self {
        self.pf = on;
        self
    }

    #[inline(always)]
    pub fn push(&mut self, b: usize, x: T) {
        self.push_at(b, x);
    }

    /// Asks for the line `bytes` past bin `b`'s write pointer in a writable state (prefetchw), so
    /// an append there does not wait for ownership. A hint: it never faults, past a chunk's end
    /// included.
    #[inline(always)]
    pub fn prefetch_ahead(&self, b: usize, bytes: usize) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: prefetch never faults, whatever the address.
        unsafe {
            core::arch::x86_64::_mm_prefetch((self.ce[b][0] as *const i8).wrapping_add(bytes), core::arch::x86_64::_MM_HINT_ET0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (b, bytes);
    }

    /// Appends and returns where the record was written. The address stays valid for the life
    /// of the bins (chunks never move), so a field can be patched later.
    #[inline(always)]
    pub fn push_at(&mut self, b: usize, x: T) -> *mut T {
        let [mut p, e] = self.ce[b];
        if p == e {
            self.refill(b);
            p = self.ce[b][0];
        }
        // SAFETY: p lies in [chunk start, chunk end) of a chunk inside a live block.
        unsafe {
            p.write(x);
            self.ce[b][0] = p.add(1);
        }
        // The front's next line, owned before it is written: with thousands of write fronts the
        // hardware streamers follow none of them, and each fresh line's first store otherwise
        // waits at the head of the store buffer for a read from DRAM (box 1, p20v5, 2,048
        // partitions: store-buffer stalls 0.70G -> 0.36G cycles). At 8,192 partitions it only
        // competes with the token probes for fill buffers: the v3 sink turns it off there.
        if self.pf && pf_on() {
            let nx = (p as *const u8).wrapping_add(64);
            let q = if nx < e as *const u8 { nx } else { p as *const u8 };
            #[cfg(target_arch = "x86_64")]
            // SAFETY: prefetch never faults.
            unsafe {
                core::arch::x86_64::_mm_prefetch(q as *const i8, core::arch::x86_64::_MM_HINT_ET0);
            }
            let _ = q;
        }
        p
    }

    /// All finished chunks of every bin, in bin order.
    pub fn all(&self) -> impl Iterator<Item = &[T]> + '_ {
        (0..self.full.len()).flat_map(move |b| self.slices(b))
    }

    #[cold]
    #[inline(never)]
    fn refill(&mut self, b: usize) {
        let [cur, end] = self.ce[b];
        if !cur.is_null() {
            // SAFETY: end is `chunk` records past the start of the same chunk.
            let start = unsafe { end.sub(self.chunk) };
            self.full[b].push((start, self.chunk as u32));
        }
        if let Some((src, w)) = self.src {
            // SAFETY: `w` is this bins' participant index (from_blocks); the block holds `chunk`
            // records and is aligned to its size.
            let p = unsafe { src.alloc(w) } as *mut T;
            assert!(!p.is_null(), "block class exhausted");
            self.ce[b] = [p, unsafe { p.add(self.chunk) }];
            return;
        }
        // SAFETY (pointer arithmetic): bump and bump_end delimit the unused tail of the newest block.
        if self.bump.is_null() || (self.bump_end as usize - self.bump as usize) < self.chunk * std::mem::size_of::<T>() {
            let mut v: Vec<T> = Vec::with_capacity(self.block);
            self.bump = v.as_mut_ptr();
            // SAFETY: the block has capacity for `block` records.
            self.bump_end = unsafe { self.bump.add(self.block) };
            self.blocks.push(v);
        }
        // SAFETY: at least `chunk` records remain in the block (checked above).
        unsafe {
            self.ce[b] = [self.bump, self.bump.add(self.chunk)];
            self.bump = self.bump.add(self.chunk);
        }
    }

    /// Closes every partly filled chunk; call once, after the last push.
    pub fn finish(&mut self) {
        for b in 0..self.ce.len() {
            let [cur, end] = self.ce[b];
            if !cur.is_null() {
                // SAFETY: as in refill; cur is within [start, end].
                let start = unsafe { end.sub(self.chunk) };
                let len = (cur as usize - start as usize) / std::mem::size_of::<T>();
                if len > 0 {
                    self.full[b].push((start, len as u32));
                }
                self.ce[b] = [std::ptr::null_mut(); 2];
            }
        }
    }

    /// The finished chunks of bin `b` (after `finish`).
    pub fn slices(&self, b: usize) -> impl Iterator<Item = &[T]> + '_ {
        // SAFETY: each (ptr, len) is a written prefix of a chunk inside a block owned by self.
        self.full[b].iter().map(|&(p, n)| unsafe { std::slice::from_raw_parts(p, n as usize) })
    }

    pub fn bins(&self) -> usize {
        self.full.len()
    }

    pub fn len(&self) -> usize {
        self.full.iter().flatten().map(|&(_, n)| n as usize).sum()
    }
}

