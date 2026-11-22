//! User ID to token: one concurrent open-addressing table per key kind, built by hand. A token is
//! the slot index (SHORT slots first, then LONG slots, then FOREIGN interned IDs), so no counter
//! is shared and the key can always be read back from the slot.
//!
//! SHORT slot: one u64 = value + 1 (0 = empty); inserted by compare-and-swap from 0.
//! LONG slot: state word (0 empty, 1 being written, 2 ready) and the two key words; the writer
//! claims the slot by compare-and-swap, stores the words, then publishes with a release store;
//! a reader that sees "being written" waits for "ready" before comparing.
//! FOREIGN: a locked map from bytes to index (rare: dirt only).
//!
//! Probes for up to 8 IDs are batched: all hashes first, prefetch, then probe.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::{Acquire, Relaxed, Release}};

use super::ids::{Key, long_bytes, short_bytes};

pub struct Tokens<'a> {
    short: Vec<AtomicU64>,
    short_mask: usize,
    /// LONG slots, 4 words each: state, a, b, unused (two slots a cache line).
    long: Vec<AtomicU64>,
    long_cap: usize,
    long_mask: usize,
    foreign: Mutex<(HashMap<&'a [u8], u32>, Vec<&'a [u8]>)>,
    /// A table passed its load limit: the run must fall back to v1.
    pub full: AtomicBool,
    short_used: AtomicU64,
    long_used: AtomicU64,
}

pub const NO_TOKEN: u32 = u32::MAX;

/// A started lookup (see `Tokens::start`).
#[derive(Clone, Copy, Default, Debug)]
pub struct Pending {
    pub long: bool,
    pub a: u64,
    pub b: u64,
    pub slot: usize,
}

/// A zeroed vector of atomics from a zeroed allocation: pages are not touched until used.
pub fn zeroed_atomics(n: usize) -> Vec<AtomicU64> {
    let v: Vec<u64> = vec![0; n];
    let mut v = std::mem::ManuallyDrop::new(v);
    // SAFETY: AtomicU64 has the same size, alignment and bit validity as u64.
    unsafe { Vec::from_raw_parts(v.as_mut_ptr() as *mut AtomicU64, v.len(), v.capacity()) }
}

#[inline(always)]
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetch never faults, whatever the address.
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

impl<'a> Tokens<'a> {
    /// Tables for about `users` distinct user IDs (load at most 0.75 before falling back).
    pub fn new(users: usize) -> Self {
        let short_cap = (users * 2).next_power_of_two().max(1 << 12);
        let long_cap = (users / 4).next_power_of_two().max(1 << 10);
        Tokens {
            short: zeroed_atomics(short_cap),
            short_mask: short_cap - 1,
            long: zeroed_atomics(long_cap * 4),
            long_cap,
            long_mask: long_cap - 1,
            foreign: Mutex::new((HashMap::new(), Vec::new())),
            full: AtomicBool::new(false),
            short_used: AtomicU64::new(0),
            long_used: AtomicU64::new(0),
        }
    }

    pub fn short_used(&self) -> usize {
        self.short_used.load(Relaxed) as usize
    }

    pub fn long_used(&self) -> usize {
        self.long_used.load(Relaxed) as usize
    }

    /// A token that names a slot or interned ID (not NO_TOKEN, not past the tables).
    pub fn valid(&self, t: u32) -> bool {
        (t as usize) < self.capacity()
    }

    pub fn short_cap(&self) -> usize {
        self.short.len()
    }

    pub fn long_cap(&self) -> usize {
        self.long_cap
    }

    /// Upper bound of tokens handed out so far (FOREIGN ones are after both tables).
    pub fn capacity(&self) -> usize {
        self.short.len() + self.long_cap + self.foreign.lock().unwrap().1.len()
    }

    /// Token range reserved for FOREIGN IDs starts here.
    pub fn foreign_base(&self) -> u32 {
        (self.short.len() + self.long_cap) as u32
    }

    #[inline(always)]
    fn short_token(&self, v: u64, mut i: usize) -> u32 {
        let want = v + 1;
        loop {
            let cur = self.short[i].load(Relaxed);
            if cur == want {
                return i as u32;
            }
            if cur == 0 {
                match self.short[i].compare_exchange(0, want, Relaxed, Relaxed) {
                    Ok(_) => {
                        let used = self.short_used.fetch_add(1, Relaxed) + 1;
                        if used as usize > self.short.len() / 4 * 3 {
                            self.full.store(true, Relaxed);
                        }
                        return i as u32;
                    }
                    Err(now) if now == want => return i as u32,
                    Err(_) => {}
                }
            }
            if self.full.load(Relaxed) && self.short_used.load(Relaxed) as usize >= self.short.len() - 1 {
                return NO_TOKEN;
            }
            i = (i + 1) & self.short_mask;
        }
    }

    #[inline(always)]
    fn long_token(&self, a: u64, b: u64, mut i: usize) -> u32 {
        loop {
            let mut st = self.long[4 * i].load(Acquire);
            if st == 0 {
                match self.long[4 * i].compare_exchange(0, 1, Acquire, Acquire) {
                    Ok(_) => {
                        self.long[4 * i + 1].store(a, Relaxed);
                        self.long[4 * i + 2].store(b, Relaxed);
                        self.long[4 * i].store(2, Release);
                        let used = self.long_used.fetch_add(1, Relaxed) + 1;
                        if used as usize > self.long_cap / 4 * 3 {
                            self.full.store(true, Relaxed);
                        }
                        return (self.short.len() + i) as u32;
                    }
                    Err(now) => st = now,
                }
            }
            while st == 1 {
                std::hint::spin_loop();
                st = self.long[4 * i].load(Acquire);
            }
            if self.long[4 * i + 1].load(Relaxed) == a && self.long[4 * i + 2].load(Relaxed) == b {
                return (self.short.len() + i) as u32;
            }
            if self.full.load(Relaxed) && self.long_used.load(Relaxed) as usize >= self.long_cap - 1 {
                return NO_TOKEN;
            }
            i = (i + 1) & self.long_mask;
        }
    }

    /// Starts a lookup: computes the slot and prefetches its line; `finish` completes it later.
    #[inline(always)]
    pub fn start(&self, long: bool, a: u64, b: u64) -> Pending {
        if !long {
            let slot = (mix(a) as usize) & self.short_mask;
            prefetch(&self.short[slot]);
            Pending { long, a, b, slot }
        } else {
            let slot = (mix(a ^ b.rotate_left(29)) as usize) & self.long_mask;
            prefetch(&self.long[4 * slot]);
            Pending { long, a, b, slot }
        }
    }

    /// `start` for a Pending that carries only the key.
    #[inline(always)]
    pub fn start_pending(&self, p: Pending) -> Pending {
        self.start(p.long, p.a, p.b)
    }

    #[inline(always)]
    pub fn finish(&self, p: &Pending) -> u32 {
        if !p.long { self.short_token(p.a, p.slot) } else { self.long_token(p.a, p.b, p.slot) }
    }

    /// Tokens of up to 8 SHORT/LONG user IDs: lane i is LONG when bit i of `long` is set; lanes
    /// whose bit in `want` is clear are skipped (their output is NO_TOKEN).
    #[inline(always)]
    pub fn batch(&self, n: usize, want: u8, long: u8, a: &[u64; 8], b: &[u64; 8], out: &mut [u32; 8]) {
        let mut slot = [0usize; 8];
        for k in 0..n {
            if want >> k & 1 == 0 {
                continue;
            }
            if long >> k & 1 == 0 {
                slot[k] = (mix(a[k]) as usize) & self.short_mask;
                prefetch(&self.short[slot[k]]);
            } else {
                slot[k] = (mix(a[k] ^ b[k].rotate_left(29)) as usize) & self.long_mask;
                prefetch(&self.long[4 * slot[k]]);
            }
        }
        for k in 0..8 {
            out[k] = NO_TOKEN;
        }
        for k in 0..n {
            if want >> k & 1 == 0 {
                continue;
            }
            out[k] = if long >> k & 1 == 0 { self.short_token(a[k], slot[k]) } else { self.long_token(a[k], b[k], slot[k]) };
        }
    }

    /// Token of one ID of any kind (FOREIGN interned by its bytes).
    pub fn token(&self, key: Key, bytes: &'a [u8]) -> u32 {
        match key {
            Key::Short(v) => self.short_token(v, (mix(v) as usize) & self.short_mask),
            Key::Long(a, b) => self.long_token(a, b, (mix(a ^ b.rotate_left(29)) as usize) & self.long_mask),
            Key::Foreign => {
                let mut g = self.foreign.lock().unwrap();
                if let Some(&i) = g.0.get(bytes) {
                    return self.foreign_base() + i;
                }
                let i = g.1.len() as u32;
                g.1.push(bytes);
                g.0.insert(bytes, i);
                self.foreign_base() + i
            }
        }
    }

    /// Starts loading the key of a token (prefetch).
    #[inline(always)]
    pub fn prefetch_key(&self, t: u32) {
        let t = t as usize;
        if t < self.short.len() {
            prefetch(&self.short[t]);
        } else if t < self.short.len() + self.long_cap {
            prefetch(&self.long[4 * (t - self.short.len())]);
        }
    }

    /// Kind and key of a token (after parsing; FOREIGN returns its bytes).
    pub fn key_of(&self, t: u32) -> (Key, Option<&'a [u8]>) {
        let t = t as usize;
        if t < self.short.len() {
            (Key::Short(self.short[t].load(Relaxed) - 1), None)
        } else if t < self.short.len() + self.long_cap {
            let i = t - self.short.len();
            (Key::Long(self.long[4 * i + 1].load(Relaxed), self.long[4 * i + 2].load(Relaxed)), None)
        } else {
            let g = self.foreign.lock().unwrap_or_else(|e| e.into_inner());
            // An index past the interned IDs (only NO_TOKEN could be) reads as an empty ID; every
            // caller has rejected such tokens before (see `valid`).
            (Key::Foreign, Some(g.1.get(t - self.short.len() - self.long_cap).copied().unwrap_or(&[])))
        }
    }

    /// Appends the bytes of a user token.
    pub fn bytes(&self, t: u32, out: &mut Vec<u8>) {
        match self.key_of(t) {
            (Key::Short(v), _) => short_bytes(b'u', v, out),
            (Key::Long(a, b), _) => long_bytes(b'u', a, b, out),
            (Key::Foreign, Some(s)) => out.extend_from_slice(s),
            (Key::Foreign, None) => unreachable!(),
        }
    }

    /// Byte order of the user IDs behind two tokens.
    pub fn cmp(&self, x: u32, y: u32) -> std::cmp::Ordering {
        if x == y {
            return std::cmp::Ordering::Equal;
        }
        match (self.key_of(x), self.key_of(y)) {
            ((Key::Short(a), _), (Key::Short(b), _)) => super::ids::cmp_short(a, b),
            ((Key::Long(a1, b1), _), (Key::Long(a2, b2), _)) => (a1, b1).cmp(&(a2, b2)),
            _ => {
                let (mut p, mut q) = (Vec::new(), Vec::new());
                self.bytes(x, &mut p);
                self.bytes(y, &mut q);
                p.cmp(&q)
            }
        }
    }
}
