//! Move-1 partitions, one task each: exact dedup by event key in reusable open-addressing tables
//! (never cleared: each slot's tag holds the partition's generation in its high 24 bits and flags
//! in the low 8; SHORT and LONG keys in separate tables since they never match), smallest ingest
//! wins, equal ingest means identical copies. Specials (ingest that does not fit, compared in
//! milliseconds) are merged after. The winner's purchase is found by key and ingest. Survivors go
//! to person partitions (move 2) as 8-byte records person << 32 | ts << 1 | purchase, purchases as
//! [person << 32 | ts, category | price << 8]; the person lookups are prefetched 16 ahead.

use super::persons::People;
use super::sink::{Special, mix};
use super::super::v3::Abort;
use crate::fast::sink::Bins;

const KEY40: u64 = (1 << 40) - 1;
const ING: u64 = (1 << 23) - 1;
/// Tag flag: the slot's winner is a special (its index is in the last word).
const SPEC: u32 = 1;

#[inline(always)]
fn h2(x: u64) -> u64 {
    let mut z = x.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z ^= z >> 29;
    z.wrapping_mul(0xbf58_476d_1ce4_e5b9) ^ (z >> 32)
}

/// One worker's reusable tables.
pub struct Tables {
    generation: u32,
    s_tag: Vec<u32>,
    s: Vec<[u64; 2]>,
    s_buy: Vec<u32>,
    l_tag: Vec<u32>,
    l: Vec<[u64; 4]>,
    l_buy: Vec<u32>,
    queue: Vec<(u32, u32, u32)>,
}

impl Tables {
    pub fn new(short_slots: usize, long_slots: usize) -> Self {
        Tables {
            generation: 0,
            s_tag: vec![0; short_slots],
            s: vec![[0; 2]; short_slots],
            s_buy: vec![0; short_slots],
            l_tag: vec![0; long_slots],
            l: vec![[0; 4]; long_slots],
            l_buy: vec![0; long_slots],
            queue: Vec::with_capacity(16),
        }
    }
}

pub struct Move2 {
    pub ev: Bins<u64>,
    pub buys: Bins<[u64; 2]>,
    pub bits: u32,
}

impl Move2 {
    pub fn new(bits: u32, hint: usize) -> Self {
        let parts = 1usize << bits;
        Move2 {
            ev: Bins::with_chunk(parts, hint.clamp(1 << 14, 1 << 21), (hint / parts / 2).clamp(32, 512)),
            buys: Bins::with_chunk(parts, 1 << 12, (hint / parts / 64).clamp(4, 64)),
            bits,
        }
    }
    #[inline(always)]
    pub fn part(&self, person: u32) -> usize {
        (mix(u64::from(person) | 0xabcd_0000_0000_0000) >> (64 - self.bits)) as usize
    }
    #[inline(always)]
    fn push(&mut self, person: u32, ts: u32, buy: u32) {
        let q = self.part(person);
        let purch = u64::from(buy != 0);
        self.ev.push(q, (u64::from(person) << 32) | (u64::from(ts) << 1) | purch);
        if buy != 0 {
            self.buys.push(q, [(u64::from(person) << 32) | u64::from(ts), u64::from(buy)]);
        }
    }
}

pub struct Ctx3<'p> {
    pub people: &'p People,
    pub ingest_base_ms: i64,
}

/// Survivor emission with the person lookup prefetched 16 survivors ahead.
struct Emitter<'x, 'p> {
    people: &'p People,
    q: &'x mut Vec<(u32, u32, u32)>,
    head: usize,
    out: usize,
}

impl Emitter<'_, '_> {
    #[inline(always)]
    fn add(&mut self, token: u32, ts: u32, buy: u32, m2: &mut Move2) {
        self.people.prefetch(token);
        if self.q.len() < 16 {
            self.q.push((token, ts, buy));
            return;
        }
        let (t, s, b) = self.q[self.head];
        m2.push(self.people.person_of(t), s, b);
        self.q[self.head] = (token, ts, buy);
        self.head = (self.head + 1) & 15;
        self.out += 1;
    }
    fn flush(&mut self, m2: &mut Move2) {
        let n = self.q.len();
        for k in 0..n {
            let (t, s, b) = self.q[(self.head + k) % n];
            m2.push(self.people.person_of(t), s, b);
            self.out += 1;
        }
        self.q.clear();
        self.head = 0;
    }
}

/// Dedups one partition and appends its survivors to `m2`. Returns the number of survivors.
#[allow(clippy::too_many_arguments)]
pub fn partition(
    short: &[&[[u64; 2]]],
    long: &[&[[u64; 4]]],
    pshort: &[&[[u64; 2]]],
    plong: &[&[[u64; 4]]],
    specials: &[Special],
    cx: &Ctx3,
    t: &mut Tables,
    m2: &mut Move2,
) -> Result<usize, Abort> {
    let ns: usize = short.iter().map(|c| c.len()).sum::<usize>() + specials.len();
    let nl: usize = long.iter().map(|c| c.len()).sum::<usize>() + specials.len();
    let s_slots = (ns + ns / 2).next_power_of_two().max(16);
    let l_slots = (nl + nl / 2).next_power_of_two().max(16);
    if s_slots > t.s_tag.len() || l_slots > t.l_tag.len() {
        return Err(Abort(format!("partition of {ns}/{nl} records exceeds the dedup table")));
    }
    t.generation = (t.generation + 1) & 0xff_ffff;
    if t.generation == 0 {
        t.s_tag.iter_mut().for_each(|x| *x = 0);
        t.l_tag.iter_mut().for_each(|x| *x = 0);
        t.generation = 1;
    }
    let g = t.generation << 8;
    let live = |tag: u32| tag & !0xff == g;
    let (sm, lm) = (s_slots - 1, l_slots - 1);
    for c in short {
        for r in c.iter() {
            let key = r[0] & KEY40;
            let ing = (r[0] >> 40) & ING;
            let mut i = (h2(key) as usize) & sm;
            loop {
                if !live(t.s_tag[i]) {
                    t.s_tag[i] = g;
                    t.s[i] = *r;
                    t.s_buy[i] = 0;
                    break;
                }
                let cur = t.s[i][0];
                if cur & KEY40 == key {
                    if ing < (cur >> 40) & ING {
                        t.s[i] = *r;
                    }
                    break;
                }
                i = (i + 1) & sm;
            }
        }
    }
    for c in long {
        for r in c.iter() {
            let ing = r[2] & ING;
            let mut i = (h2(r[0] ^ r[1].rotate_left(31)) as usize) & lm;
            loop {
                if !live(t.l_tag[i]) {
                    t.l_tag[i] = g;
                    t.l[i] = *r;
                    t.l_buy[i] = 0;
                    break;
                }
                let cur = &t.l[i];
                if cur[0] == r[0] && cur[1] == r[1] {
                    if ing < cur[2] & ING {
                        t.l[i] = *r;
                    }
                    break;
                }
                i = (i + 1) & lm;
            }
        }
    }
    // Specials: compare in milliseconds against the stored copy.
    let base = cx.ingest_base_ms;
    for (si, sp) in specials.iter().enumerate() {
        match sp.key {
            super::ids::Key::Short(key) => {
                let mut i = (h2(key) as usize) & sm;
                loop {
                    if !live(t.s_tag[i]) {
                        t.s_tag[i] = g | SPEC;
                        t.s[i] = [key, si as u64];
                        break;
                    }
                    if t.s[i][0] & KEY40 == key {
                        let cur = if t.s_tag[i] & SPEC != 0 {
                            specials[t.s[i][1] as usize].ingest_ms
                        } else {
                            base + (((t.s[i][0] >> 40) & ING) as i64) * 1000
                        };
                        if sp.ingest_ms < cur {
                            t.s_tag[i] = g | SPEC;
                            t.s[i] = [key, si as u64];
                        }
                        break;
                    }
                    i = (i + 1) & sm;
                }
            }
            super::ids::Key::Long(a, b) => {
                let mut i = (h2(a ^ b.rotate_left(31)) as usize) & lm;
                loop {
                    if !live(t.l_tag[i]) {
                        t.l_tag[i] = g | SPEC;
                        t.l[i] = [a, b, 0, si as u64];
                        break;
                    }
                    if t.l[i][0] == a && t.l[i][1] == b {
                        let cur = if t.l_tag[i] & SPEC != 0 {
                            specials[t.l[i][3] as usize].ingest_ms
                        } else {
                            base + ((t.l[i][2] & ING) as i64) * 1000
                        };
                        if sp.ingest_ms < cur {
                            t.l_tag[i] = g | SPEC;
                            t.l[i] = [a, b, 0, si as u64];
                        }
                        break;
                    }
                    i = (i + 1) & lm;
                }
            }
            super::ids::Key::Foreign => unreachable!("FOREIGN event IDs are deduplicated apart"),
        }
    }
    // Purchases of the winning copies: same key and same ingest.
    for c in pshort {
        for r in c.iter() {
            let key = r[0] & KEY40;
            let mut i = (h2(key) as usize) & sm;
            while live(t.s_tag[i]) {
                if t.s[i][0] & KEY40 == key {
                    if t.s_tag[i] & SPEC == 0 && t.s[i][0] & !(1u64 << 63) == r[0] {
                        t.s_buy[i] = r[1] as u32;
                    }
                    break;
                }
                i = (i + 1) & sm;
            }
        }
    }
    for c in plong {
        for r in c.iter() {
            let mut i = (h2(r[0] ^ r[1].rotate_left(31)) as usize) & lm;
            while live(t.l_tag[i]) {
                if t.l[i][0] == r[0] && t.l[i][1] == r[1] {
                    if t.l_tag[i] & SPEC == 0 && t.l[i][2] & ING == r[2] {
                        t.l_buy[i] = r[3] as u32;
                    }
                    break;
                }
                i = (i + 1) & lm;
            }
        }
    }
    // Survivors.
    let mut q = std::mem::take(&mut t.queue);
    q.clear();
    let mut em = Emitter { people: cx.people, q: &mut q, head: 0, out: 0 };
    for i in 0..s_slots {
        let tag = t.s_tag[i];
        if !live(tag) {
            continue;
        }
        if tag & SPEC != 0 {
            let sp = &specials[t.s[i][1] as usize];
            if sp.surv {
                em.add(sp.token, sp.ts, sp.buy, m2);
            }
            continue;
        }
        let [w0, w1] = t.s[i];
        if w0 >> 63 == 0 {
            continue;
        }
        let buy = if w1 >> 63 != 0 {
            if t.s_buy[i] == 0 {
                return Err(Abort("a surviving purchase has no purchase record".into()));
            }
            t.s_buy[i]
        } else {
            0
        };
        em.add(w1 as u32, ((w1 >> 32) & 0x7fff_ffff) as u32, buy, m2);
    }
    for i in 0..l_slots {
        let tag = t.l_tag[i];
        if !live(tag) {
            continue;
        }
        if tag & SPEC != 0 {
            let sp = &specials[t.l[i][3] as usize];
            if sp.surv {
                em.add(sp.token, sp.ts, sp.buy, m2);
            }
            continue;
        }
        let r = t.l[i];
        if r[2] >> 63 == 0 {
            continue;
        }
        let w3 = r[3];
        let buy = if w3 >> 63 != 0 {
            if t.l_buy[i] == 0 {
                return Err(Abort("a surviving purchase has no purchase record".into()));
            }
            t.l_buy[i]
        } else {
            0
        };
        em.add(w3 as u32, ((w3 >> 32) & 0x7fff_ffff) as u32, buy, m2);
    }
    em.flush(m2);
    let out = em.out;
    t.queue = q;
    Ok(out)
}

/// FOREIGN event IDs: copies only of each other; smallest ingest by bytes, then to move 2.
pub fn foreign(specials: &[&Special], cx: &Ctx3, m2: &mut Move2) -> usize {
    let mut best: std::collections::HashMap<&[u8], usize> = std::collections::HashMap::new();
    for (i, sp) in specials.iter().enumerate() {
        best.entry(sp.fid).and_modify(|b| if sp.ingest_ms < specials[*b].ingest_ms { *b = i }).or_insert(i);
    }
    let mut out = 0;
    for (_, i) in best {
        let sp = specials[i];
        if sp.surv {
            m2.push(cx.people.person_of(sp.token), sp.ts, sp.buy);
            out += 1;
        }
    }
    out
}
