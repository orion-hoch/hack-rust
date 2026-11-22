#!/usr/bin/env python3
"""Builds datasets that are valid by rules.md but break every locality hint the fast path uses.

  hintbreak.py BASE_DIR OUT_DIR CASE EVENT_KEY USER_KEY [SEED]

BASE_DIR is a generated dataset; EVENT_KEY and USER_KEY are its ID keys (the engine prints them
with --timing). IDs are renumbered with the same keys, so the new IDs still conform and decode to
the chosen numbers. Cases:

  events-shuffled   every event number permuted at random: no person owns a block of numbers
  events-in-person  event numbers permuted inside each person's block: time runs backwards
  users-random      every user number mapped to a random 32-bit number: user >> 9 says nothing
  links-rewired     a third of the persons linked to a random other person: persons span gps
  foreign-mix       some persons get a foreign ID (in links and in some of their rows)
  spill-split       copies whose ingest does not fit 23 bits, winning and losing
  time-backwards    ts values swapped between neighbouring events of a person
  purchase-disagree extra copies of purchases with other prices and products, earlier and later
  all               every case above at once
"""
import os
import random
import re
import sys

MASK = (1 << 40) - 1
M1 = 0x9E3779B97F4B
M2 = 0xD6E8FEB86659
M1_INV = pow(M1, -1, 1 << 40)
M2_INV = pow(M2, -1, 1 << 40)
DIG = "0123456789abcdefghijklmnopqrstuvwxyz"


def scramble(n, key):
    m = ((n & MASK) * M1 ^ key) & MASK
    m ^= m >> 17
    return (m * M2) & MASK


def unscramble(x, key):
    y = (x * M2_INV) & MASK
    y = y ^ (y >> 17) ^ (y >> 34)
    return (((y ^ key) & MASK) * M1_INV) & MASK


def b36(v):
    if v == 0:
        return "0"
    s = []
    while v:
        s.append(DIG[v % 36])
        v //= 36
    return "".join(reversed(s))


CONF = re.compile(rb"^([euk])(0|[1-9a-z][0-9a-z]{0,7})$")


def decode(idb, prefix, key):
    m = CONF.match(idb)
    if not m or m.group(1) != prefix:
        return None
    x = int(m.group(2), 36)
    if x > MASK:
        return None
    return unscramble(x, key)


def encode(prefix, n, key):
    return prefix + b36(scramble(n, key)).encode()


def split(line):
    """Fields with their quoting: [(value, quoted)], or None for a line we leave untouched."""
    out = []
    for f in line.split(b","):
        if b'"' in f:
            if len(f) >= 2 and f[0:1] == b'"' and f[-1:] == b'"' and b'"' not in f[1:-1]:
                out.append((f[1:-1], True))
            else:
                return None
        else:
            out.append((f, False))
    return out


def join(fields):
    return b",".join((b'"' + v + b'"') if q else v for v, q in fields)


def read_rows(folder):
    """{file name: (header, [line bytes])}"""
    rows = {}
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".csv"):
            continue
        data = open(os.path.join(folder, name), "rb").read()
        lines = data.split(b"\n")
        if lines and lines[-1] == b"":
            lines.pop()
        rows[name] = (lines[0] if lines else b"", lines[1:])
    return rows


def write_rows(folder, rows):
    os.makedirs(folder, exist_ok=True)
    for name, (header, lines) in rows.items():
        with open(os.path.join(folder, name), "wb") as f:
            f.write(b"\n".join([header] + lines) + b"\n")


def f1(ms):
    import datetime
    t = datetime.datetime(1970, 1, 1) + datetime.timedelta(milliseconds=ms)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ").encode()


def main():
    base, out, case, ekey, ukey = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 7
    rnd = random.Random(seed)
    cases = {case} if case != "all" else {
        "events-shuffled", "users-random", "links-rewired", "foreign-mix", "spill-split",
        "time-backwards", "purchase-disagree"}
    ev = read_rows(os.path.join(base, "events"))
    ck = read_rows(os.path.join(base, "clicks"))
    lk = read_rows(os.path.join(base, "links"))
    manifest = open(os.path.join(base, "manifest.json"), "rb").read()
    ws_text = re.search(rb'"window_start":\s*"([^"]+)"', manifest).group(1).decode()
    import datetime
    ws = int((datetime.datetime.strptime(ws_text, "%Y-%m-%dT%H:%M:%SZ") - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)

    # Event numbers and user numbers present.
    events = set()
    users = set()
    for _, (_, lines) in ev.items():
        for line in lines:
            f = split(line)
            if f and len(f) == 7:
                n = decode(f[0][0], b"e", ekey)
                if n is not None:
                    events.add(n)
                u = decode(f[1][0], b"u", ukey)
                if u is not None:
                    users.add(u)
    for rows in (ck, lk):
        for _, (_, lines) in rows.items():
            for line in lines:
                f = split(line)
                if not f:
                    continue
                for v, _ in f:
                    u = decode(v, b"u", ukey)
                    if u is not None:
                        users.add(u)
    emap = {}
    if "events-shuffled" in cases:
        nums = sorted(events)
        perm = nums[:]
        rnd.shuffle(perm)
        emap = dict(zip(nums, perm))
    if "events-in-person" in cases:
        # Event numbers of each person (by the user of its first copy) permuted among themselves.
        by_person = {}
        for _, (_, lines) in ev.items():
            for line in lines:
                f = split(line)
                if f and len(f) == 7:
                    n = decode(f[0][0], b"e", ekey)
                    u = decode(f[1][0], b"u", ukey)
                    if n is not None and u is not None:
                        by_person.setdefault(u >> 9, set()).add(n)
        seen = set()
        for _, nums in sorted(by_person.items()):
            nums = sorted(n for n in nums if n not in seen)
            seen.update(nums)
            perm = nums[:]
            rnd.shuffle(perm)
            emap.update(zip(nums, perm))
    umap = {}
    if "users-random" in cases:
        taken = set()
        for u in sorted(users):
            while True:
                v = rnd.randrange(1 << 32)
                if v not in taken:
                    taken.add(v)
                    umap[u] = v
                    break

    def map_event_id(v):
        n = decode(v, b"e", ekey)
        if n is None or n not in emap:
            return v
        return encode(b"e", emap[n], ekey)

    def map_user(v):
        u = decode(v, b"u", ukey)
        if u is None or u not in umap:
            return v
        return encode(b"u", umap[u], ukey)

    def foreign_of(v):
        return b"X" + v[1:] + b"-f"

    foreign_users = set()
    if "foreign-mix" in cases:
        pick = sorted(users)
        rnd.shuffle(pick)
        foreign_users = set(pick[: max(1, len(pick) // 20)])

    new_ev = {}
    for name, (header, lines) in ev.items():
        outl = []
        for line in lines:
            f = split(line)
            if not f or len(f) != 7:
                outl.append(line)
                continue
            f[0] = (map_event_id(f[0][0]), f[0][1])
            u = decode(f[1][0], b"u", ukey)
            if u in foreign_users and rnd.random() < 0.5:
                f[1] = (foreign_of(f[1][0]), f[1][1])
            else:
                f[1] = (map_user(f[1][0]), f[1][1])
            outl.append(join(f))
        new_ev[name] = (header, outl)

    if "spill-split" in cases or "purchase-disagree" in cases or "time-backwards" in cases:
        for name, (header, lines) in new_ev.items():
            outl = []
            prev = None
            for line in lines:
                f = split(line)
                if not f or len(f) != 7:
                    outl.append(line)
                    continue
                if "time-backwards" in cases and prev is not None and rnd.random() < 0.05:
                    # Swap ts with the previous line of this file (usually another person's event).
                    pf = split(outl[prev])
                    if pf and len(pf) == 7:
                        pf[2], f[2] = f[2], pf[2]
                        outl[prev] = join(pf)
                if "spill-split" in cases and rnd.random() < 0.01:
                    # A copy ingested 200 days later (does not fit, loses) and, for some, an
                    # earlier copy ingested 40 days before the window (does not fit, wins).
                    late = list(f)
                    late[3] = (f1(ws + 200 * 86_400_000 + rnd.randrange(86_400) * 1000), False)
                    outl.append(join(late))
                    if rnd.random() < 0.5:
                        early = list(f)
                        early[3] = (f1(ws - 40 * 86_400_000 - rnd.randrange(86_400) * 1000), True)
                        if rnd.random() < 0.5:
                            early[4] = (b"purchase", False)
                            early[6] = (b"777", False)
                        outl.append(join(early))
                if "purchase-disagree" in cases and f[4][0] == b"purchase" and rnd.random() < 0.3:
                    other = list(f)
                    other[6] = (str(rnd.randrange(1, 100000)).encode(), False)
                    other[5] = (b"p" + str(rnd.randrange(1, 50000)).encode(), False)
                    # An hour later loses; an hour earlier wins. (The generator's own copies are at
                    # most 10 minutes apart, so no two copies end up with equal ingest_ts.)
                    m = re.match(rb"^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)Z$", f[3][0])
                    if m:
                        t = datetime.datetime(*map(int, m.groups()))
                        delta = 3601 if rnd.random() < 0.5 else -3601
                        other[3] = ((t + datetime.timedelta(seconds=delta)).strftime("%Y-%m-%dT%H:%M:%SZ").encode(), False)
                        outl.append(join(other))
                prev = len(outl)
                outl.append(join(f))
            new_ev[name] = (header, outl)

    new_ck = {}
    for name, (header, lines) in ck.items():
        outl = []
        for line in lines:
            f = split(line)
            if not f or len(f) != 4:
                outl.append(line)
                continue
            u = decode(f[1][0], b"u", ukey)
            if u in foreign_users and rnd.random() < 0.5:
                f[1] = (foreign_of(f[1][0]), f[1][1])
            else:
                f[1] = (map_user(f[1][0]), f[1][1])
            outl.append(join(f))
        new_ck[name] = (header, outl)

    new_lk = {}
    all_users = sorted(users)
    for name, (header, lines) in lk.items():
        outl = []
        for line in lines:
            f = split(line)
            if not f or len(f) != 2:
                outl.append(line)
                continue
            f = [(map_user(v), q) for v, q in f]
            outl.append(join(f))
        new_lk[name] = (header, outl)
    first = next(iter(new_lk)) if new_lk else "part-00000.csv"
    header, extra = new_lk.get(first, (b"user_id_a,user_id_b", []))
    if "links-rewired" in cases:
        for u in all_users:
            if rnd.random() < 0.33:
                v = rnd.choice(all_users)
                extra.append(map_user(encode(b"u", u, ukey)) + b"," + map_user(encode(b"u", v, ukey)))
    for u in sorted(foreign_users):
        extra.append(foreign_of(encode(b"u", u, ukey)) + b"," + map_user(encode(b"u", u, ukey)))
    new_lk[first] = (header, extra)

    os.makedirs(out, exist_ok=True)
    write_rows(os.path.join(out, "events"), new_ev)
    write_rows(os.path.join(out, "clicks"), new_ck)
    write_rows(os.path.join(out, "links"), new_lk)
    for f in ("products.csv", "campaigns.csv", "manifest.json"):
        open(os.path.join(out, f), "wb").write(open(os.path.join(base, f), "rb").read())


if __name__ == "__main__":
    main()
