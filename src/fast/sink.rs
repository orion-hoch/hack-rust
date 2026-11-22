//! Bins: records binned by an index. A bin append is a compare of the bin's write pointer with its
//! chunk end, a store and an increment. Chunks (512 records) are bumped from large arena blocks
//! owned by the bins, so nothing is ever copied or reallocated; memory is the records plus at
//! most one partly filled chunk per bin.



/// Records binned by an index, in chunks bumped from arena blocks.
pub struct Bins<T: Copy> {
    cur: Vec<*mut T>,
    end: Vec<*mut T>,
    full: Vec<Vec<(*const T, u32)>>,
    blocks: Vec<Vec<T>>,
    bump: *mut T,
    bump_end: *mut T,
    block: usize,
    chunk: usize,
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
            cur: vec![std::ptr::null_mut(); n],
            end: vec![std::ptr::null_mut(); n],
            full: (0..n).map(|_| Vec::new()).collect(),
            blocks: Vec::new(),
            bump: std::ptr::null_mut(),
            bump_end: std::ptr::null_mut(),
            block: block.div_ceil(chunk).max(1) * chunk,
            chunk,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, b: usize, x: T) {
        self.push_at(b, x);
    }

    /// Appends and returns where the record was written. The address stays valid for the life
    /// of the bins (chunks never move), so a field can be patched later.
    #[inline(always)]
    pub fn push_at(&mut self, b: usize, x: T) -> *mut T {
        let mut p = self.cur[b];
        if p == self.end[b] {
            self.refill(b);
            p = self.cur[b];
        }
        // SAFETY: p lies in [chunk start, chunk end) of a chunk inside a live block.
        unsafe {
            p.write(x);
            self.cur[b] = p.add(1);
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
        if !self.cur[b].is_null() {
            // SAFETY: end[b] is `chunk` records past the start of the same chunk.
            let start = unsafe { self.end[b].sub(self.chunk) };
            self.full[b].push((start, self.chunk as u32));
        }
        // SAFETY (pointer arithmetic): bump and bump_end delimit the unused tail of the newest block.
        if self.bump.is_null() || (self.bump_end as usize - self.bump as usize) < self.chunk * std::mem::size_of::<T>() {
            let mut v: Vec<T> = Vec::with_capacity(self.block);
            self.bump = v.as_mut_ptr();
            // SAFETY: the block has capacity for `block` records.
            self.bump_end = unsafe { self.bump.add(self.block) };
            self.blocks.push(v);
        }
        self.cur[b] = self.bump;
        // SAFETY: at least `chunk` records remain in the block (checked above).
        unsafe {
            self.end[b] = self.bump.add(self.chunk);
            self.bump = self.bump.add(self.chunk);
        }
    }

    /// Closes every partly filled chunk; call once, after the last push.
    pub fn finish(&mut self) {
        for b in 0..self.cur.len() {
            if !self.cur[b].is_null() {
                // SAFETY: as in refill; cur[b] is within [start, end].
                let start = unsafe { self.end[b].sub(self.chunk) };
                let len = (self.cur[b] as usize - start as usize) / std::mem::size_of::<T>();
                if len > 0 {
                    self.full[b].push((start, len as u32));
                }
                self.cur[b] = std::ptr::null_mut();
                self.end[b] = std::ptr::null_mut();
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

