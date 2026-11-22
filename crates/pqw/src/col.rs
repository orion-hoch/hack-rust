//! Column chunk writers: pages for one column.

use super::*;

/// Emit header (in sc.hdr) + page body (sc.page) + an optional borrowed tail; return the page offset.
fn emit<W: Write>(sink: &mut Sink<W>, sc: &Scratch, tail: &[u8]) -> io::Result<i64> {
    let off = sink.off as i64;
    sink.put(&sc.hdr)?;
    sink.put(&sc.page)?;
    if !tail.is_empty() {
        sink.put(tail)?;
    }
    Ok(off)
}

fn data_page(sc: &mut Scratch, tail: usize, nvals: usize, enc: i32) -> io::Result<()> {
    let len = sc.page.len() + tail;
    if len > i32::MAX as usize {
        return Err(invalid("page larger than 2 GiB; lower page_rows".into()));
    }
    data_page_header(&mut sc.hdr, len, nvals, enc);
    Ok(())
}

use enc::Int;

pub(crate) trait Code: Copy + PartialEq {
    fn u(self) -> u32;
}
impl Code for u8 {
    #[inline]
    fn u(self) -> u32 {
        self as u32
    }
}
impl Code for u16 {
    #[inline]
    fn u(self) -> u32 {
        self as u32
    }
}

pub(crate) fn write_column<W: Write>(
    sink: &mut Sink<W>,
    c: &Col,
    n: usize,
    pr: usize,
    sc: &mut Scratch,
) -> io::Result<Chunk> {
    let start = sink.off as i64;
    let mut ch = match c.data {
        Data::Str(s) => write_str(sink, s, n, pr, c.enc, sc)?,
        Data::StrRuns { values, runs } if c.enc == Enc::DeltaLength => write_str_runs_dlba(sink, values, runs, n, pr, sc)?,
        Data::StrRuns { values, runs } => write_str_runs(sink, values, runs, n, pr, sc)?,
        Data::Dict { codes, dict, nullable } => write_dict(sink, codes, dict, nullable, n, pr, c.enc, sc)?,
        Data::Date(v) | Data::I32(v) => write_int(sink, v, true, n, pr, c.enc, sc)?,
        Data::U32(v) => {
            let p32 = v.iter().fold(0u32, |m, &x| m.max(x)) <= i32::MAX as u32;
            write_int(sink, v, p32, n, pr, c.enc, sc)?
        }
        Data::I64(v) => write_int(sink, v, false, n, pr, c.enc, sc)?,
    };
    ch.start = start;
    ch.size = sink.off as i64 - start;
    ch.num_values = n as i64;
    ch.encodings.sort_unstable();
    ch.encodings.dedup();
    Ok(ch)
}

fn chunk(phys: i32, encodings: Vec<i32>) -> Chunk {
    Chunk { phys, encodings, num_values: 0, start: 0, size: 0, data_off: -1, dict_off: None }
}

/// Page ranges [a, b): page_rows each; one empty page when n == 0.
fn pages(n: usize, pr: usize) -> impl Iterator<Item = (usize, usize)> {
    let k = if n == 0 { 1 } else { (n + pr - 1) / pr };
    (0..k).map(move |i| (i * pr, ((i + 1) * pr).min(n)))
}

fn write_str<W: Write>(sink: &mut Sink<W>, s: Strings, n: usize, pr: usize, enc: Enc, sc: &mut Scratch) -> io::Result<Chunk> {
    let e = match enc {
        Enc::Plain => E_PLAIN,
        Enc::DeltaLength => E_DELTA_LENGTH_BYTE_ARRAY,
        Enc::Auto | Enc::DeltaByteArray => E_DELTA_BYTE_ARRAY,
        _ => unreachable!(),
    };
    let mut ch = chunk(PT_BYTE_ARRAY, vec![e]);
    for (a, b) in pages(n, pr) {
        sc.page.clear();
        let o = &s.offsets[a..=b];
        let m = b - a;
        let mut tail: &[u8] = &[];
        match e {
            E_PLAIN => {
                let p = &mut sc.page;
                p.reserve((o[m] - o[0]) as usize + 4 * m);
                for i in a..b {
                    enc::plain_bytes(p, s.get(i));
                }
            }
            E_DELTA_LENGTH_BYTE_ARRAY => {
                sc.i32a.clear();
                sc.i32a.extend(o.windows(2).map(|w| (w[1] - w[0]) as i32));
                enc::delta_binary_packed(&mut sc.page, &sc.i32a, true);
                // the values are the caller's bytes, written straight from their buffer
                tail = &s.bytes[o[0] as usize..o[m] as usize];
            }
            _ => {
                enc::prefix_suffix(s.bytes, o, &mut sc.i32a, &mut sc.i32b, &mut sc.u8s);
                enc::delta_binary_packed(&mut sc.page, &sc.i32a, true);
                enc::delta_binary_packed(&mut sc.page, &sc.i32b, true);
                sc.page.extend_from_slice(&sc.u8s);
            }
        }
        data_page(sc, tail.len(), m, e)?;
        let off = emit(sink, sc, tail)?;
        if ch.data_off < 0 {
            ch.data_off = off;
        }
    }
    Ok(ch)
}

fn write_int<W: Write, T: Int>(
    sink: &mut Sink<W>,
    v: &[T],
    p32: bool,
    n: usize,
    pr: usize,
    enc: Enc,
    sc: &mut Scratch,
) -> io::Result<Chunk> {
    let phys = if p32 { PT_INT32 } else { PT_INT64 };
    let mut enc = if enc == Enc::Auto { Enc::Delta } else { enc };
    let (mut lo, mut hi) = (0i64, 0i64);
    if enc == Enc::Dict {
        if n == 0 {
            enc = Enc::Plain;
        } else {
            lo = i64::MAX;
            hi = i64::MIN;
            for x in v {
                lo = lo.min(x.i64());
                hi = hi.max(x.i64());
            }
            if (hi as i128 - lo as i128) >= 65536 {
                enc = Enc::Delta;
            }
        }
    }
    let e = match enc {
        Enc::Plain => E_PLAIN,
        Enc::Delta => E_DELTA_BINARY_PACKED,
        Enc::Dict => E_RLE_DICTIONARY,
        _ => unreachable!(),
    };
    let mut ch = chunk(phys, vec![e]);
    if enc == Enc::Dict {
        ch.encodings.push(E_PLAIN);
        sc.page.clear();
        let k = (hi - lo + 1) as usize;
        if p32 {
            sc.page.reserve(4 * k);
            for x in lo..=hi {
                sc.page.extend_from_slice(&(x as i32).to_le_bytes());
            }
        } else {
            sc.page.reserve(8 * k);
            for x in lo..=hi {
                sc.page.extend_from_slice(&x.to_le_bytes());
            }
        }
        dict_page_header(&mut sc.hdr, sc.page.len(), k);
        ch.dict_off = Some(emit(sink, sc, &[])?);
    }
    let width = enc::bits((hi - lo) as u64).max(1);
    for (a, b) in pages(n, pr) {
        sc.page.clear();
        let vs = &v[a..b];
        match enc {
            Enc::Plain => {
                let p = &mut sc.page;
                if p32 {
                    p.resize(4 * vs.len(), 0);
                    for (d, x) in p.chunks_exact_mut(4).zip(vs) {
                        d.copy_from_slice(&(x.i64() as i32).to_le_bytes());
                    }
                } else {
                    p.resize(8 * vs.len(), 0);
                    for (d, x) in p.chunks_exact_mut(8).zip(vs) {
                        d.copy_from_slice(&x.i64().to_le_bytes());
                    }
                }
            }
            Enc::Delta => enc::delta_binary_packed(&mut sc.page, vs, p32),
            Enc::Dict => {
                sc.page.push(width as u8);
                if width <= 8 {
                    sc.u8s.clear();
                    sc.u8s.extend(vs.iter().map(|x| (x.i64() - lo) as u8));
                    enc::hybrid_u8(&mut sc.page, &sc.u8s, width);
                } else {
                    sc.u32s.clear();
                    sc.u32s.extend(vs.iter().map(|x| (x.i64() - lo) as u32));
                    enc::hybrid(&mut sc.page, &sc.u32s, width);
                }
            }
            _ => unreachable!(),
        }
        data_page(sc, 0, b - a, e)?;
        let off = emit(sink, sc, &[])?;
        if ch.data_off < 0 {
            ch.data_off = off;
        }
    }
    Ok(ch)
}

#[allow(clippy::too_many_arguments)]
fn write_dict<W: Write>(
    sink: &mut Sink<W>,
    codes: Codes,
    dict: &[&[u8]],
    nullable: bool,
    n: usize,
    pr: usize,
    enc: Enc,
    sc: &mut Scratch,
) -> io::Result<Chunk> {
    let plain = enc == Enc::Plain;
    let e = if plain { E_PLAIN } else { E_RLE_DICTIONARY };
    let mut ch = chunk(PT_BYTE_ARRAY, vec![e]);
    if nullable {
        ch.encodings.push(E_RLE);
    }
    if !plain {
        ch.encodings.push(E_PLAIN);
        sc.page.clear();
        for d in dict {
            enc::plain_bytes(&mut sc.page, d);
        }
        dict_page_header(&mut sc.hdr, sc.page.len(), dict.len());
        ch.dict_off = Some(emit(sink, sc, &[])?);
    }
    // index width; 0 only for an empty dictionary (then every row is null)
    let width = if dict.is_empty() { 0 } else { enc::bits(dict.len() as u64 - 1).max(1) };
    for (a, b) in pages(n, pr) {
        sc.page.clear();
        if nullable {
            sc.page.extend_from_slice(&[0; 4]);
            match codes {
                Codes::U8(c) => enc::def_levels_u8(&mut sc.page, &c[a..b], NULL8, &mut sc.bits),
                Codes::U16(c) => {
                    sc.u32s.clear();
                    sc.u32s.extend(c[a..b].iter().map(|&x| (x != NULL16) as u32));
                    enc::hybrid(&mut sc.page, &sc.u32s, 1);
                }
            }
            let len = (sc.page.len() - 4) as u32;
            sc.page[..4].copy_from_slice(&len.to_le_bytes());
        }
        if plain {
            match codes {
                Codes::U8(c) => {
                    for &x in &c[a..b] {
                        if x != NULL8 {
                            enc::plain_bytes(&mut sc.page, dict[x as usize]);
                        }
                    }
                }
                Codes::U16(c) => {
                    for &x in &c[a..b] {
                        if x != NULL16 {
                            enc::plain_bytes(&mut sc.page, dict[x as usize]);
                        }
                    }
                }
            }
        } else {
            sc.page.push(width as u8);
            match codes {
                Codes::U8(c) if width <= 8 && !nullable => {
                    enc::hybrid_u8(&mut sc.page, &c[a..b], width);
                }
                Codes::U8(c) if width <= 8 => {
                    // keep non-null codes, branch-free
                    let c = &c[a..b];
                    sc.u8s.clear();
                    sc.u8s.resize(c.len(), 0);
                    let mut k = 0;
                    for &x in c {
                        sc.u8s[k] = x;
                        k += (x != NULL8) as usize;
                    }
                    sc.u8s.truncate(k);
                    if k > 0 {
                        enc::hybrid_u8(&mut sc.page, &sc.u8s, width);
                    }
                }
                _ => {
                    sc.u32s.clear();
                    match codes {
                        Codes::U8(c) => sc.u32s.extend(c[a..b].iter().filter(|&&x| x != NULL8).map(|&x| x as u32)),
                        Codes::U16(c) => sc.u32s.extend(c[a..b].iter().filter(|&&x| x != NULL16).map(|&x| x as u32)),
                    }
                    if !sc.u32s.is_empty() {
                        enc::hybrid(&mut sc.page, &sc.u32s, width);
                    }
                }
            }
        }
        data_page(sc, 0, b - a, e)?;
        let off = emit(sink, sc, &[])?;
        if ch.data_off < 0 {
            ch.data_off = off;
        }
    }
    Ok(ch)
}

/// DELTA_BYTE_ARRAY from runs: within a run every row after the first has prefix = its length and an
/// empty suffix; a run's first row shares a prefix with the previous run's value (prefix 0 at a page start).
fn write_str_runs<W: Write>(
    sink: &mut Sink<W>,
    s: Strings,
    runs: &[u32],
    n: usize,
    pr: usize,
    sc: &mut Scratch,
) -> io::Result<Chunk> {
    let mut ch = chunk(PT_BYTE_ARRAY, vec![E_DELTA_BYTE_ARRAY]);
    let mut r = 0usize; // current run
    let mut used = 0usize; // rows of run r already written
    for (a, b) in pages(n, pr) {
        let m = b - a;
        sc.page.clear();
        // 16 entries of slack so a run's repeats are written with one fixed-size store
        sc.i32a.clear();
        sc.i32a.resize(m + 16, 0);
        sc.i32b.clear();
        sc.i32b.resize(m, 0);
        let (pre, suf) = (&mut sc.i32a, &mut sc.i32b);
        // suffix bytes: at most one value per run touched by this page, so bound by the bytes of those runs
        let r0 = r;
        let mut cnt = 0usize;
        let mut r1 = r;
        let mut u1 = used;
        while cnt < m {
            let avail = runs[r1] as usize - u1;
            cnt += avail;
            r1 += 1;
            u1 = 0;
        }
        let span = (s.offsets[r1] - s.offsets[r0]) as usize;
        let sufb = &mut sc.u8s;
        sufb.clear();
        sufb.resize(span + 16, 0);
        let bytes = s.bytes;
        let blen = bytes.len();
        let mut p = 0usize;
        let mut row = 0usize;
        let (mut pa, mut la) = (0usize, 0usize);
        let mut prev = 0u128;
        let mut first = true;
        while row < m {
            while runs[r] as usize == used {
                r += 1;
                used = 0;
            }
            let cb = s.offsets[r] as usize;
            let cl = s.offsets[r + 1] as usize - cb;
            let take = (runs[r] as usize - used).min(m - row);
            let k;
            if used > 0 && !first {
                k = cl;
            } else if cl <= 16 && la <= 16 && cb + 16 <= blen {
                let cur = u128::from_le_bytes(bytes[cb..cb + 16].try_into().unwrap());
                let t = ((prev ^ cur).trailing_zeros() / 8) as usize;
                k = t.min(la).min(cl);
                sufb[p..p + 16].copy_from_slice(&(cur >> (((8 * k) & 127) as u32)).to_le_bytes());
                prev = cur;
            } else {
                let v = &bytes[cb..cb + cl];
                k = if first { 0 } else { enc::common_prefix(&bytes[pa..pa + la], v) };
                sufb[p..p + cl - k].copy_from_slice(&v[k..]);
                prev = if cb + 16 <= blen { u128::from_le_bytes(bytes[cb..cb + 16].try_into().unwrap()) } else { 0 };
            }
            if take <= 16 {
                pre[row..row + 16].copy_from_slice(&[cl as i32; 16]);
            } else {
                pre[row..row + take].fill(cl as i32);
            }
            pre[row] = k as i32;
            suf[row] = (cl - k) as i32;
            p += cl - k;
            // suf is already 0 for the repeats
            row += take;
            used += take;
            pa = cb;
            la = cl;
            first = false;
        }
        sufb.truncate(p);
        sc.i32a.truncate(m);
        enc::delta_binary_packed(&mut sc.page, &sc.i32a, true);
        enc::delta_binary_packed(&mut sc.page, &sc.i32b, true);
        sc.page.extend_from_slice(&sc.u8s);
        data_page(sc, 0, m, E_DELTA_BYTE_ARRAY)?;
        let off = emit(sink, sc, &[])?;
        if ch.data_off < 0 {
            ch.data_off = off;
        }
    }
    Ok(ch)
}

/// DELTA_LENGTH_BYTE_ARRAY from runs: every row spelled out, values copied 16 bytes at a time.
fn write_str_runs_dlba<W: Write>(
    sink: &mut Sink<W>,
    s: Strings,
    runs: &[u32],
    n: usize,
    pr: usize,
    sc: &mut Scratch,
) -> io::Result<Chunk> {
    let mut ch = chunk(PT_BYTE_ARRAY, vec![E_DELTA_LENGTH_BYTE_ARRAY]);
    let mut r = 0usize;
    let mut used = 0usize;
    let bytes = s.bytes;
    let blen = bytes.len();
    for (a, b) in pages(n, pr) {
        let m = b - a;
        // exact value bytes of this page
        let (mut r1, mut u1, mut left, mut total) = (r, used, m, 0usize);
        while left > 0 {
            let t = (runs[r1] as usize - u1).min(left);
            total += t * s.get(r1).len();
            left -= t;
            u1 += t;
            if u1 == runs[r1] as usize {
                r1 += 1;
                u1 = 0;
            }
        }
        sc.page.clear();
        sc.i32a.clear();
        sc.i32a.resize(m + 16, 0);
        sc.u8s.clear();
        sc.u8s.resize(total + 16, 0);
        let (lens, vb) = (&mut sc.i32a, &mut sc.u8s);
        let (mut row, mut p) = (0usize, 0usize);
        while row < m {
            while runs[r] as usize == used {
                r += 1;
                used = 0;
            }
            let cb = s.offsets[r] as usize;
            let cl = s.offsets[r + 1] as usize - cb;
            let take = (runs[r] as usize - used).min(m - row);
            if take <= 16 {
                lens[row..row + 16].copy_from_slice(&[cl as i32; 16]);
            } else {
                lens[row..row + take].fill(cl as i32);
            }
            if cl <= 16 && cb + 16 <= blen {
                let v: [u8; 16] = bytes[cb..cb + 16].try_into().unwrap();
                for _ in 0..take {
                    vb[p..p + 16].copy_from_slice(&v);
                    p += cl;
                }
            } else {
                for _ in 0..take {
                    vb[p..p + cl].copy_from_slice(&bytes[cb..cb + cl]);
                    p += cl;
                }
            }
            row += take;
            used += take;
        }
        debug_assert_eq!(p, total);
        sc.i32a.truncate(m);
        sc.u8s.truncate(total);
        enc::delta_binary_packed(&mut sc.page, &sc.i32a, true);
        let tail = std::mem::take(&mut sc.u8s);
        data_page(sc, tail.len(), m, E_DELTA_LENGTH_BYTE_ARRAY)?;
        let off = emit(sink, sc, &tail)?;
        sc.u8s = tail;
        if ch.data_off < 0 {
            ch.data_off = off;
        }
    }
    Ok(ch)
}
