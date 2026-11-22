//! Value encoders. Everything appends to a Vec<u8>.
//!
//! Bit packing is LSB first, as in Parquet's RLE/bit-packed hybrid and in DELTA_BINARY_PACKED.

use crate::thrift::{varint, zigzag};

#[inline]
pub fn bits(v: u64) -> u32 {
    64 - v.leading_zeros()
}

#[inline]
fn varint_len(v: u64) -> usize {
    (bits(v | 1) as usize + 6) / 7
}

pub use crate::pack_gen::pack32;

/// Bit-pack `vals` (any length; the last group of 8 is zero-padded) at width w (0..=64), appending.
pub fn pack(out: &mut Vec<u8>, vals: &[u64], w: u32) {
    if w == 0 || vals.is_empty() {
        return;
    }
    let groups = (vals.len() + 7) / 8;
    let start = out.len();
    out.resize(start + groups * w as usize, 0);
    let mut tmp = [0u64; 32];
    let mut pos = start;
    for c in vals.chunks(32) {
        let src: &[u64; 32] = if c.len() == 32 {
            c.try_into().unwrap()
        } else {
            tmp[..c.len()].copy_from_slice(c);
            tmp[c.len()..].fill(0);
            &tmp
        };
        let nb = (c.len() + 7) / 8 * w as usize;
        if nb == 4 * w as usize {
            pack32(src, w, &mut out[pos..pos + nb]);
        } else {
            let mut full = [0u8; 256];
            pack32(src, w, &mut full);
            out[pos..pos + nb].copy_from_slice(&full[..nb]);
        }
        pos += nb;
    }
}

/// An integer column value, read with INT32 or INT64 semantics.
pub trait Int: Copy {
    fn i32(self) -> i32;
    fn i64(self) -> i64;
}
impl Int for i32 {
    #[inline(always)]
    fn i32(self) -> i32 {
        self
    }
    #[inline(always)]
    fn i64(self) -> i64 {
        self as i64
    }
}
impl Int for u32 {
    #[inline(always)]
    fn i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn i64(self) -> i64 {
        self as i64
    }
}
impl Int for i64 {
    #[inline(always)]
    fn i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn i64(self) -> i64 {
        self
    }
}

/// DELTA_BINARY_PACKED: blocks of 128 values, 4 miniblocks of 32 (the layout parquet-cpp writes).
/// `p32` selects INT32 semantics (deltas wrap in 32 bits), otherwise INT64.
pub fn delta_binary_packed<T: Int>(out: &mut Vec<u8>, v: &[T], p32: bool) {
    dbp_mb::<T, 1>(out, v, p32)
}

/// DELTA_BINARY_PACKED with miniblocks of 64 values (block 256), the layout parquet-cpp uses for INT64.
/// Only used to compare bytes with pyarrow.
pub fn delta_binary_packed_mb64<T: Int>(out: &mut Vec<u8>, v: &[T], p32: bool) {
    dbp_mb::<T, 2>(out, v, p32)
}

/// Miniblocks of 32*K values, blocks of 4 miniblocks.
fn dbp_mb<T: Int, const K: usize>(out: &mut Vec<u8>, v: &[T], p32: bool) {
    let n = v.len();
    varint(out, 128 * K as u64);
    varint(out, 4);
    varint(out, n as u64);
    if n == 0 {
        varint(out, 0);
        return;
    }
    varint(out, zigzag(if p32 { v[0].i32() as i64 } else { v[0].i64() }));
    let mut r = [0u64; 256];
    let mut i = 1;
    while i < n {
        let m = (n - i).min(128 * K);
        let blk = &v[i - 1..i + m];
        let mn: i64;
        if p32 {
            let mut d = [0i32; 256];
            for j in 0..m {
                d[j] = blk[j + 1].i32().wrapping_sub(blk[j].i32());
            }
            let mut lo = i32::MAX;
            for &x in &d[..m] {
                lo = lo.min(x);
            }
            for j in 0..m {
                r[j] = d[j].wrapping_sub(lo) as u32 as u64;
            }
            mn = lo as i64;
        } else {
            let mut d = [0i64; 256];
            for j in 0..m {
                d[j] = blk[j + 1].i64().wrapping_sub(blk[j].i64());
            }
            let mut lo = i64::MAX;
            for &x in &d[..m] {
                lo = lo.min(x);
            }
            for j in 0..m {
                r[j] = d[j].wrapping_sub(lo) as u64;
            }
            mn = lo;
        }
        let mbv = 32 * K;
        let used = (m + mbv - 1) / mbv;
        for x in &mut r[m..used * mbv] {
            *x = 0;
        }
        let mut w = [0u32; 4];
        let mut body = 0usize;
        for k in 0..used {
            let mut o = 0u64;
            for &x in &r[k * mbv..k * mbv + mbv] {
                o |= x;
            }
            w[k] = bits(o);
            body += 4 * K * w[k] as usize;
        }
        let zm = zigzag(mn);
        let st = out.len();
        let hl = varint_len(zm) + 4;
        out.resize(st + hl + body, 0);
        let blk = &mut out[st..];
        let mut p = 0;
        let mut z = zm;
        while z >= 0x80 {
            blk[p] = z as u8 | 0x80;
            z >>= 7;
            p += 1;
        }
        blk[p] = z as u8;
        p += 1;
        for k in 0..4 {
            blk[p + k] = w[k] as u8;
        }
        p += 4;
        for k in 0..used {
            for h in 0..K {
                let nb = 4 * w[k] as usize;
                let s0 = k * mbv + h * 32;
                pack32((&r[s0..s0 + 32]).try_into().unwrap(), w[k], &mut blk[p..p + nb]);
                p += nb;
            }
        }
        i += m;
    }
}

/// Longest bit-packed run, in groups of 8, that one header byte can describe (what parquet-cpp emits).
const MAX_LITERAL_GROUPS: usize = 63;

fn rle_run(out: &mut Vec<u8>, count: usize, v: u32, w: u32) {
    varint(out, (count as u64) << 1);
    let nb = ((w + 7) / 8) as usize;
    out.extend_from_slice(&v.to_le_bytes()[..nb]);
}

/// Smallest number of equal groups of 8 worth an RLE run at width w: an RLE run costs about
/// 2 + ceil(w/8) bytes and splits a literal run (one more header byte); bit-packed they cost g*w bytes.
fn rle_min_groups(w: u32) -> usize {
    (((4 + (w as usize + 7) / 8) + w as usize - 1) / w as usize).max(1)
}

/// The generic RLE/bit-packed hybrid driver. `uniform(g)` says whether group g (8 values) is all one
/// value, `val(i)` gives value i, `lit(out, a, b)` bit-packs values [a, b) (zero-padding the last group).
#[inline(always)]
fn hybrid_drive(
    out: &mut Vec<u8>,
    n: usize,
    w: u32,
    uniform: impl Fn(usize) -> bool,
    val: impl Fn(usize) -> u32,
    mut lit: impl FnMut(&mut Vec<u8>, usize, usize),
) {
    let ng = n / 8;
    let min_groups = rle_min_groups(w);
    let mut emit_lit = |out: &mut Vec<u8>, a: usize, b: usize| {
        let mut s = a;
        while s < b {
            let e = (s + MAX_LITERAL_GROUPS * 8).min(b);
            let groups = (e - s + 7) / 8;
            varint(out, ((groups as u64) << 1) | 1);
            lit(out, s, e);
            s = e;
        }
    };
    let mut done = 0usize; // values emitted so far
    let mut g = 0usize;
    while g < ng {
        if uniform(g) {
            let v = val(g * 8);
            let mut j = g + 1;
            while j < ng && val(j * 8) == v && uniform(j) {
                j += 1;
            }
            if j - g >= min_groups {
                emit_lit(out, done, g * 8);
                let mut end = j * 8;
                if j == ng && (end..n).all(|i| val(i) == v) {
                    end = n;
                }
                rle_run(out, end - g * 8, v, w);
                done = end;
            }
            g = j;
        } else {
            g += 1;
        }
    }
    if done < n {
        emit_lit(out, done, n);
    }
}

/// Pack 8 byte lanes (each < 2^W) into the low 8*W bits.
#[inline(always)]
fn pack8x8<const W: u32>(x: u64) -> u64 {
    let m = (1u64 << W) - 1;
    let mut r = 0u64;
    let mut i = 0;
    while i < 8 {
        r |= ((x >> (8 * i)) & m) << (W * i);
        i += 1;
    }
    r
}

#[inline(always)]
fn load8(v: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(v[i..i + 8].try_into().unwrap())
}

fn hybrid_u8_w<const W: u32>(out: &mut Vec<u8>, vals: &[u8]) {
    let n = vals.len();
    hybrid_drive(
        out,
        n,
        W,
        |g| {
            let x = load8(vals, g * 8);
            x == (x & 0xFF).wrapping_mul(0x0101_0101_0101_0101)
        },
        |i| vals[i] as u32,
        |out, a, b| {
            let groups = (b - a + 7) / 8;
            let st = out.len();
            out.resize(st + groups * W as usize + 8, 0);
            let mut p = st;
            let mut i = a;
            while i + 8 <= b {
                let r = pack8x8::<W>(load8(vals, i));
                out[p..p + 8].copy_from_slice(&r.to_le_bytes());
                p += W as usize;
                i += 8;
            }
            if i < b {
                let mut t = [0u8; 8];
                t[..b - i].copy_from_slice(&vals[i..b]);
                let r = pack8x8::<W>(u64::from_le_bytes(t));
                out[p..p + 8].copy_from_slice(&r.to_le_bytes());
                p += W as usize;
            }
            out.truncate(p);
        },
    );
}

/// RLE/bit-packed hybrid of byte values at width w (1..=8), no length prefix.
pub fn hybrid_u8(out: &mut Vec<u8>, vals: &[u8], w: u32) {
    match w {
        1 => hybrid_u8_w::<1>(out, vals),
        2 => hybrid_u8_w::<2>(out, vals),
        3 => hybrid_u8_w::<3>(out, vals),
        4 => hybrid_u8_w::<4>(out, vals),
        5 => hybrid_u8_w::<5>(out, vals),
        6 => hybrid_u8_w::<6>(out, vals),
        7 => hybrid_u8_w::<7>(out, vals),
        8 => hybrid_u8_w::<8>(out, vals),
        _ => panic!("hybrid_u8 width {w}"),
    }
}

/// RLE/bit-packed hybrid of u32 values at width w (1..=32), no length prefix.
pub fn hybrid(out: &mut Vec<u8>, vals: &[u32], w: u32) {
    debug_assert!((1..=32).contains(&w));
    let mut stage = vec![0u64; MAX_LITERAL_GROUPS * 8];
    hybrid_drive(
        out,
        vals.len(),
        w,
        |g| {
            let s = &vals[g * 8..g * 8 + 8];
            s.iter().all(|&x| x == s[0])
        },
        |i| vals[i],
        |out, a, b| {
            for (s, &v) in stage.iter_mut().zip(&vals[a..b]) {
                *s = v as u64;
            }
            pack(out, &stage[..b - a], w);
        },
    );
}

/// Definition levels (width 1) for byte codes where `null` marks a null, as RLE/bit-packed hybrid with
/// no length prefix. `bitmap` is scratch.
pub fn def_levels_u8(out: &mut Vec<u8>, codes: &[u8], null: u8, bitmap: &mut Vec<u8>) {
    let n = codes.len();
    bitmap.clear();
    bitmap.resize((n + 7) / 8, 0);
    let nn = (null as u64).wrapping_mul(0x0101_0101_0101_0101);
    let mut g = 0;
    while g * 8 + 8 <= n {
        // byte lane nonzero <=> code != null
        let y = load8(codes, g * 8) ^ nn;
        let t = (((y & 0x7F7F_7F7F_7F7F_7F7F) + 0x7F7F_7F7F_7F7F_7F7F) | y) & 0x8080_8080_8080_8080;
        bitmap[g] = ((t >> 7).wrapping_mul(0x0102_0408_1020_4080) >> 56) as u8;
        g += 1;
    }
    if g * 8 < n {
        let mut b = 0u8;
        for (k, &c) in codes[g * 8..].iter().enumerate() {
            b |= ((c != null) as u8) << k;
        }
        bitmap[g] = b;
    }
    let bm = &*bitmap;
    hybrid_drive(
        out,
        n,
        1,
        |g| bm[g] == 0 || bm[g] == 0xFF,
        |i| ((bm[i / 8] >> (i % 8)) & 1) as u32,
        |out, a, b| {
            // a is a multiple of 8; the padding bits of the last group are already zero
            out.extend_from_slice(&bm[a / 8..(b + 7) / 8]);
        },
    );
}

/// Common prefix length of two byte strings.
#[inline]
pub fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    let m = a.len().min(b.len());
    let mut i = 0;
    while i + 8 <= m {
        let x = u64::from_le_bytes(a[i..i + 8].try_into().unwrap())
            ^ u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        if x != 0 {
            return i + (x.trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < m && a[i] == b[i] {
        i += 1;
    }
    i
}

/// Append a 4-byte little-endian length then the bytes (PLAIN BYTE_ARRAY).
#[inline]
pub fn plain_bytes(out: &mut Vec<u8>, b: &[u8]) {
    out.extend_from_slice(&(b.len() as u32).to_le_bytes());
    out.extend_from_slice(b);
}

#[inline(always)]
fn load16(b: &[u8], i: usize) -> u128 {
    u128::from_le_bytes(b[i..i + 16].try_into().unwrap())
}

/// DELTA_BYTE_ARRAY for values bytes[o[j]..o[j+1]] (one page, so the first value has prefix 0):
/// fills prefix and suffix lengths and appends the suffix bytes to `sufb`. One pass; values of up to
/// 16 bytes are compared and copied with 16-byte loads.
pub fn prefix_suffix(bytes: &[u8], o: &[u32], pre: &mut Vec<i32>, suf: &mut Vec<i32>, sufb: &mut Vec<u8>) {
    let m = o.len() - 1;
    pre.clear();
    suf.clear();
    pre.resize(m, 0);
    suf.resize(m, 0);
    sufb.clear();
    sufb.resize((o[m] - o[0]) as usize + 16, 0);
    let blen = bytes.len();
    let mut p = 0usize;
    let mut pa = 0usize;
    let mut la = 0usize;
    let mut prev = 0u128;
    for j in 0..m {
        let cb = o[j] as usize;
        let cl = o[j + 1] as usize - cb;
        let k;
        if cl <= 16 && la <= 16 && cb + 16 <= blen {
            let cur = load16(bytes, cb);
            // prev was loaded the same way (or is 0 for the first value, with la = 0)
            let t = ((prev ^ cur).trailing_zeros() / 8) as usize;
            k = t.min(la).min(cl);
            let sh = ((8 * k) & 127) as u32;
            sufb[p..p + 16].copy_from_slice(&(cur >> sh).to_le_bytes());
            prev = cur;
        } else {
            let c = &bytes[cb..cb + cl];
            k = if j == 0 { 0 } else { common_prefix(&bytes[pa..pa + la], c) };
            sufb[p..p + cl - k].copy_from_slice(&c[k..]);
            prev = if cb + 16 <= blen { load16(bytes, cb) } else { 0 };
        }
        pre[j] = k as i32;
        suf[j] = (cl - k) as i32;
        p += cl - k;
        pa = cb;
        la = cl;
    }
    sufb.truncate(p);
}
