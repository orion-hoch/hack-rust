//! Planting dirt for the equivalence proof: every dirt form in rules.md plus things the generator
//! never writes (\r, non-ASCII digits, spaces, stray quotes, short and long lines, leading zeros in
//! IDs and numbers, very long F3, offsets at the edges, window edges, a final line without \n,
//! empty and header-only files, lines longer than the 128-byte window, separators at bytes 63/64).

use std::fs;
use std::io;
use std::path::Path;

use crate::records::mix64;
use crate::ts::{DAY_MS, format_f1};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = mix64(self.0.wrapping_add(0x9E37_79B9_7F4A_7C15));
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn chance(&mut self, p: f64) -> bool {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 * 1.0 < p
    }
}

fn f2(ms: i64, sign: char, h: i64, m: i64) -> String {
    let off = (h * 60 + m) * 60_000 * if sign == '+' { 1 } else { -1 };
    let local = format_f1(ms + off);
    format!("{}{sign}{h:02}:{m:02}", &local[..19])
}

/// Timestamps: valid, bad, and at the edges of both windows.
fn ts_variant(r: &mut Rng, ws: i64, we: i64, near: i64) -> String {
    let near_s = near - near.rem_euclid(1000);
    // Edges of the date tables: both years of the direct table, every month, the last days of
    // months, the first and last table dates and one past each, NUL runs.
    if r.chance(0.4) {
        let y0: i64 = format_f1(ws - 9 * DAY_MS)[..4].parse().unwrap();
        let y = y0 + r.below(3) as i64 - (r.below(4) == 0) as i64;
        let m = 1 + r.below(13);
        let d = [1usize, 28, 29, 30, 31, 32, 0][r.below(7)];
        return match r.below(8) {
            0 => format_f1(ws - 9 * DAY_MS + r.below(86_400) as i64 * 1000),
            1 => format_f1(ws - 10 * DAY_MS + r.below(86_400) as i64 * 1000),
            2 => format_f1(we + 7 * DAY_MS + r.below(86_400) as i64 * 1000),
            3 => format_f1(we + 8 * DAY_MS + r.below(86_400) as i64 * 1000),
            4 => "\0\0\0\0\0\0\0\0\0\0T04:56:27Z".into(),
            5 => format!("{y:04}-{m:02}-{d:02}T10:00:00+0{}:00", r.below(10)),
            _ => format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", r.below(24), r.below(60), r.below(60)),
        };
    }
    match r.below(30) {
        0 => format!("{}{}", "0".repeat(1 + r.below(30)), near),
        1 => format!("{}", near as i128 * 1_000_000_000),
        2 => format!("{}", ws - 1),
        3 => format!("{ws}"),
        4 => format!("{}", we - 1),
        5 => format!("{we}"),
        6 => format!("{}", ws - 7 * DAY_MS - 1),
        7 => format!("{}", ws - 7 * DAY_MS),
        8 => f2(near_s, '+', 23, 59),
        9 => f2(near_s, '-', 23, 59),
        10 => f2(near_s, '-', 0, 0),
        11 => f2(near_s, '+', 0, 0),
        12 => format!("{}+24:00", &format_f1(near_s)[..19]),
        13 => format!("{}+05:60", &format_f1(near_s)[..19]),
        14 => format!("{}+0200", &format_f1(near_s)[..19]),
        15 => "2027-02-29T10:00:00Z".into(),
        16 => "2028-02-29T10:00:00Z".into(),
        17 => format!("{}-13-01T00:00:00Z", &format_f1(near_s)[..4]),
        18 => format!("{}-04-31T00:00:00Z", &format_f1(near_s)[..4]),
        19 => format!("{}T24:00:00Z", &format_f1(near_s)[..10]),
        20 => format!("{}T23:59:60Z", &format_f1(near_s)[..10]),
        21 => format_f1(ws - 1000),
        22 => format_f1(we),
        23 => format_f1(we - 1000),
        24 => format_f1(ws - 7 * DAY_MS),
        25 => format!("{}z", &format_f1(near_s)[..19]),
        26 => format!("{}.000Z", &format_f1(near_s)[..19]),
        27 => format!("{near}.0"),
        28 => format!("{}", near).replace('1', "١"),
        _ => format!("{}", near % 100_000_000),
    }
}

fn digit_swap(r: &mut Rng, s: &str) -> String {
    let positions: Vec<usize> = s.char_indices().filter(|(_, c)| c.is_ascii_digit()).map(|(i, _)| i).collect();
    if positions.is_empty() {
        return s.to_string();
    }
    let at = positions[r.below(positions.len())];
    let repl = if r.chance(0.5) { "٢" } else { "２" };
    format!("{}{}{}", &s[..at], repl, &s[at + 1..])
}

fn id_variant(r: &mut Rng, id: &str, prefix: char) -> String {
    let body = id.trim_matches('"');
    let rest = body.get(1..).unwrap_or("");
    match r.below(10) {
        0 => format!("{prefix}0{rest}"),
        1 => body.to_uppercase(),
        2 => format!("{body}a"),
        3 => format!("{prefix}zzzzzzzz"),
        4 => format!("{prefix}0"),
        5 => format!("{prefix}"),
        6 => format!(" {body}"),
        7 => format!("{body} "),
        8 => format!("{prefix}{}", &rest[..rest.len().min(1)]),
        _ => String::new(),
    }
}

fn boundary(s: &str, mut at: usize) -> usize {
    while !s.is_char_boundary(at) {
        at -= 1;
    }
    at
}

fn mutate_line(r: &mut Rng, kind: usize, line: &str, ws: i64, we: i64, out: &mut Vec<String>) {
    let mut f: Vec<String> = line.split(',').map(|s| s.to_string()).collect();
    let near = ws + (r.next() % (we - ws) as u64) as i64;
    let want = [7usize, 4, 2][kind];
    // Lines that are already odd only get the generic mutations.
    let pick = if f.len() == want { r.below(16) } else { r.below(10) };
    match (kind, pick) {
        (_, 0) => return out.push(format!("{line}\r")),
        (_, 1) => {
            let at = boundary(line, r.below(line.len() + 1));
            return out.push(format!("{}\r{}", &line[..at], &line[at..]));
        }
        (_, 2) => return out.push(digit_swap(r, line)),
        (_, 3) => {
            let i = r.below(f.len());
            f[i] = if r.chance(0.5) { format!(" {}", f[i]) } else { format!("{} ", f[i]) };
        }
        (_, 4) => {
            let at = boundary(line, r.below(line.len() + 1));
            return out.push(format!("{}\"{}", &line[..at], &line[at..]));
        }
        (_, 5) => {
            let i = r.below(f.len());
            let v = f[i].trim_matches('"').to_string();
            f[i] = match r.below(4) {
                0 => format!("\"{v}\""),
                1 => format!("\"\"{v}\"\""),
                2 => format!("\"{v}\"\r"),
                _ => "\"\"".into(),
            };
        }
        (_, 6) => {
            if r.chance(0.5) && f.len() > 1 {
                let i = r.below(f.len());
                f.remove(i);
            } else {
                f.push("x".into());
            }
        }
        (_, 7) => {
            out.push(String::new());
            return out.push(line.to_string());
        }
        (_, 8) => {
            // Long lines: pad the second-to-last field to put the line at 126..=130 bytes or > 128.
            let i = f.len().saturating_sub(2);
            let target = [126usize, 127, 128, 129, 130, 200][r.below(6)];
            let cur: usize = f.iter().map(|x| x.len() + 1).sum::<usize>() - 1;
            if cur < target {
                f[i].push_str(&"q".repeat(target - cur));
            }
        }
        (_, 9) => {
            // Put a separator or quote at byte 63 or 64 of the line by padding field 0's tail.
            let i = 1.min(f.len() - 1);
            let upto: usize = f[..=i].iter().map(|x| x.len() + 1).sum::<usize>() - 1;
            let target = [62usize, 63, 64, 65][r.below(4)];
            if upto < target {
                let pad = target - upto;
                let v = f[i].trim_matches('"').to_string();
                f[i] = if r.chance(0.5) { format!("\"{v}{}\"", "w".repeat(pad.saturating_sub(2))) } else { format!("{v}{}", "w".repeat(pad)) };
            }
        }
        (0, p) => match p {
            10 => f[0] = id_variant(r, &f[0], 'e'),
            11 => f[1] = id_variant(r, &f[1], 'u'),
            12 => f[2] = ts_variant(r, ws, we, near),
            13 => {
                f[3] = match r.below(7) {
                    0 => ts_variant(r, ws, we, near),
                    1 => "2099-01-01T00:00:00Z".into(),
                    2 => "1999-12-31T23:59:59Z".into(),
                    3 => format!("{near}"),
                    4 => "garbage".into(),
                    5 => format_f1(near - near.rem_euclid(1000) + DAY_MS),
                    _ => format_f1(ws - 30 * DAY_MS - 1000),
                }
            }
            14 => {
                f[4] = ["Purchase", "purchase ", "\"purchase\"", "purchase\r", "views", "", "cart", "view", "purchase"][r.below(9)].into();
                if f.len() > 6 {
                    f[6] = ["0", "10000000", "10000001", &format!("{}250", "0".repeat(40)), "1e3", " 250", "250 ", "+250", "-0", "00000000010000000", "99999999", "123456789", ""][r.below(13)].into();
                }
            }
            _ => {
                if f.len() > 5 {
                    f[5] = ["", "\"\"", " p1", "p0001", "P12", "p1", "p50000", "p50001", "p99999999999"][r.below(9)].into();
                }
            }
        },
        (1, p) => match p {
            10 | 11 => f[0] = id_variant(r, &f[0], 'k'),
            12 => f[1] = id_variant(r, &f[1], 'u'),
            13 => {
                if f.len() > 2 {
                    f[2] = [" c_1", "C_1", "c_01", "c_0", "c_2001", "c_1", "c_2000", "\"c_7\"", "c_"][r.below(9)].into();
                }
            }
            _ => {
                if f.len() > 3 {
                    let back = 7 * DAY_MS * (r.below(2) as i64);
                    f[3] = ts_variant(r, ws, we, near - back);
                }
            }
        },
        (_, p) => match p {
            10 => f[0] = id_variant(r, &f[0], 'u'),
            11 => {
                if f.len() > 1 {
                    f[1] = id_variant(r, &f[1], 'u')
                }
            }
            12 => {
                let a = f[0].clone();
                f = vec![a.clone(), a];
            }
            13 => return out.push(format!("{},\r", f[0])),
            _ => {
                if f.len() > 1 {
                    f[1] = f[1].to_uppercase();
                }
            }
        },
    }
    out.push(f.join(","));
}

/// Writes a mutated copy of the dataset at `src` to `dst`.
pub fn mutate(src: &Path, dst: &Path, seed: u64, rate: f64) -> io::Result<u64> {
    let mut r = Rng(seed ^ 0x6A09_E667_F3BC_C909);
    let manifest = fs::read(src.join("manifest.json"))?;
    let (ws, we) = crate::tables::manifest(&manifest).map_err(|e| io::Error::other(format!("{e:?}")))?;
    fs::create_dir_all(dst)?;
    for f in ["products.csv", "campaigns.csv", "manifest.json"] {
        fs::copy(src.join(f), dst.join(f))?;
    }
    let mut changed = 0u64;
    for (kind, sub, header) in [
        (0usize, "events", "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents"),
        (1, "clicks", "click_id,user_id,campaign_id,ts"),
        (2, "links", "user_id_a,user_id_b"),
    ] {
        let out_dir = dst.join(sub);
        fs::create_dir_all(&out_dir)?;
        for path in crate::dataset::csv_files(&src.join(sub))? {
            let text = String::from_utf8_lossy(&fs::read(&path)?).into_owned();
            let _ = text.len();
            let mut lines: Vec<&str> = text.split('\n').collect();
            if lines.last() == Some(&"") {
                lines.pop();
            }
            let mut out: Vec<String> = Vec::with_capacity(lines.len() + 16);
            for (i, line) in lines.iter().enumerate() {
                if i > 0 && r.chance(rate) {
                    changed += 1;
                    mutate_line(&mut r, kind, line, ws, we, &mut out);
                } else {
                    out.push(line.to_string());
                }
            }
            let mut body = out.join("\n").into_bytes();
            if !r.chance(0.3) {
                body.push(b'\n');
            }
            // Byte-level dirt: any byte value (NUL, DEL, bytes above 0x7F, \r) at any position of
            // a line, never a \n.
            let mut at = 0usize;
            let mut first = true;
            while at < body.len() {
                let end = body[at..].iter().position(|&c| c == b'\n').map_or(body.len(), |p| at + p);
                if !first && end > at && r.chance(rate / 3.0) {
                    let pos = at + r.below(end - at);
                    let v = [0x00u8, 0x7F, 0x80, 0xC3, 0xFF, 0x0D, 0x20, (r.next() & 0xFF) as u8][r.below(8)];
                    body[pos] = if v == b'\n' { 0 } else { v };
                }
                first = false;
                at = end + 1;
            }
            fs::write(out_dir.join(path.file_name().unwrap()), body)?;
        }
        fs::write(out_dir.join("zz-empty.csv"), "")?;
        fs::write(out_dir.join("zz-header.csv"), format!("{header}\n"))?;
        fs::write(out_dir.join("zz-header-no-newline.csv"), header)?;
        // A tiny header, so the first data lines start below the wide-load margin.
        let first = crate::dataset::csv_files(&src.join(sub))?;
        let sample: Vec<String> = match first.first() {
            Some(p) => fs::read_to_string(p)?.lines().skip(1).take(3).map(|l| l.to_string()).collect(),
            None => Vec::new(),
        };
        fs::write(out_dir.join("zz-short-header.csv"), format!("h\n{}", sample.join("\n")))?;
    }
    Ok(changed)
}
