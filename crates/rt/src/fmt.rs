//! A fixed buffer for building short messages without core::fmt (reports and failure notes).

pub struct Buf<const N: usize> {
    b: [u8; N],
    n: usize,
}

impl<const N: usize> Default for Buf<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Buf<N> {
    pub const fn new() -> Self {
        Buf { b: [0; N], n: 0 }
    }
    pub fn s(&mut self, s: &[u8]) -> &mut Self {
        let k = s.len().min(N - self.n);
        self.b[self.n..self.n + k].copy_from_slice(&s[..k]);
        self.n += k;
        self
    }
    pub fn c(&mut self, c: u8) -> &mut Self {
        self.s(&[c])
    }
    pub fn u(&mut self, mut v: u64) -> &mut Self {
        let mut t = [0u8; 20];
        let mut i = 20;
        loop {
            i -= 1;
            t[i] = b'0' + (v % 10) as u8;
            v /= 10;
            if v == 0 {
                break;
            }
        }
        self.s(&t[i..])
    }
    pub fn i(&mut self, v: i64) -> &mut Self {
        if v < 0 {
            self.c(b'-').u(v.unsigned_abs())
        } else {
            self.u(v as u64)
        }
    }
    /// `v / 10^d` with `d` decimals (for example nanoseconds as milliseconds: `fix(ns / 1000, 3)`).
    pub fn fix(&mut self, v: u64, d: u32) -> &mut Self {
        let p = 10u64.pow(d);
        self.u(v / p);
        if d > 0 {
            self.c(b'.');
            let mut f = v % p;
            let mut t = [b'0'; 19];
            for k in (0..d as usize).rev() {
                t[k] = b'0' + (f % 10) as u8;
                f /= 10;
            }
            self.s(&t[..d as usize]);
        }
        self
    }
    /// Right-aligns the next `width` characters: pads with spaces before the value written by `f`.
    pub fn pad(&mut self, width: usize, f: impl FnOnce(&mut Buf<64>)) -> &mut Self {
        let mut t = Buf::<64>::new();
        f(&mut t);
        for _ in t.n..width {
            self.c(b' ');
        }
        self.s(t.as_bytes())
    }
    pub fn hex(&mut self, v: u64) -> &mut Self {
        let mut t = [0u8; 16];
        for (k, d) in t.iter_mut().enumerate() {
            let x = ((v >> (60 - 4 * k)) & 15) as u8;
            *d = if x < 10 { b'0' + x } else { b'a' + x - 10 };
        }
        self.s(b"0x").s(&t)
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.b[..self.n]
    }
    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    pub fn clear(&mut self) {
        self.n = 0;
    }
    /// Writes the buffer to `fd` and clears it.
    pub fn flush(&mut self, fd: i32) {
        let _ = crate::sys::write_all(fd, &self.b[..self.n]);
        self.n = 0;
    }
}

/// Writes a message to stderr directly.
pub fn eprint(parts: &[&[u8]]) {
    let mut b = Buf::<1024>::new();
    for p in parts {
        b.s(p);
    }
    b.flush(2);
}
