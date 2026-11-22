#!/usr/bin/env python3
"""Writes the engine's own edge-case fixtures (tests/fixtures/X*) in the starter's layout.

Each case's expected rows were derived by hand from rules.md; see the comments. Run:
  python3 tests/make_fixtures.py && python3 run_fixtures.py tests/fixtures -- target/release/engine
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "fixtures"
EVENTS_H = b"event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents"
CLICKS_H = b"click_id,user_id,campaign_id,ts"
LINKS_H = b"user_id_a,user_id_b"
USER_H = "user_id,day,sessions,events,revenue_cents,top_category"
CAMP_H = "campaign_id,channel,day,sessions,purchasing_sessions,revenue_cents,assisted_sessions"
MANIFEST = b'{\n  "window_start": "2026-10-01T00:00:00Z",\n  "window_end": "2026-10-15T00:00:00Z"\n}\n'


def case(name, events, clicks, links, user_rows, camp_rows):
    base = ROOT / name
    for sub, files in (("events", events), ("clicks", clicks), ("links", links)):
        d = base / "input" / sub
        d.mkdir(parents=True, exist_ok=True)
        for i, content in enumerate(files):
            (d / f"part-{i:05d}.csv").write_bytes(content)
    (base / "input" / "manifest.json").write_bytes(MANIFEST)
    (base / "input" / "products.csv").write_bytes(b"product_id,category\np1,shoes\np2,books\n")
    (base / "input" / "campaigns.csv").write_bytes(b"campaign_id,channel\nc_1,search\nc_2,social\n")
    (base / "expected").mkdir(parents=True, exist_ok=True)
    (base / "expected" / "user_daily.csv").write_text("\n".join([USER_H] + user_rows) + "\n", encoding="utf-8")
    (base / "expected" / "campaign_daily.csv").write_text("\n".join([CAMP_H] + camp_rows) + "\n", encoding="utf-8")


# X1 file endings: a last line without its \n counts; a zero-byte file, a header without a \n
# and an empty line after the header add nothing.
case(
    "X1",
    events=[
        EVENTS_H + b"\ne1,u1,2026-10-03T10:00:00Z,2026-10-03T12:00:00Z,purchase,p1,700",
        b"",
        EVENTS_H,
        EVENTS_H + b"\n\ne2,u1,2026-10-03T10:05:00Z,2026-10-03T12:00:00Z,view,p2,\n",
    ],
    clicks=[CLICKS_H + b"\nk1,u1,c_2,2026-10-03T09:00:00Z"],
    links=[LINKS_H],
    user_rows=["u1,2026-10-03,1,2,700,shoes"],
    camp_rows=["c_2,social,2026-10-03,1,1,700,1"],
)

# X2 dedup across files: identical copies with equal ingest_ts count once; the smaller
# ingest_ts wins even when it is in another file (e2's valid copy B); the earliest copy's user
# wins (e3 goes to u2); exactly 24 h late is kept (e4), 24 h + 1 s is dropped (e5), and an
# ingest_ts before ts is kept (e6). u1: e1, e2, e4, e6 at 10:00-10:30, one session.
case(
    "X2",
    events=[
        EVENTS_H + b"\n"
        b"e1,u1,2026-10-03T10:00:00Z,2026-10-03T10:01:00Z,view,p1,\n"
        b"e2,u1,2026-10-03T10:10:00Z,2026-10-03T10:11:00Z,Purchase,p1,500\n"
        b"e3,u2,2026-10-03T11:00:00Z,2026-10-03T11:01:00Z,purchase,p2,300\n",
        EVENTS_H + b"\n"
        b"e1,u1,2026-10-03T10:00:00Z,\"2026-10-03T10:01:00Z\",view,p1,\n"
        b"e2,u1,2026-10-03T10:10:00Z,2026-10-03T10:10:59Z,purchase,p1,500\n"
        b"e3,u1,2026-10-03T11:00:00Z,2026-10-03T11:05:00Z,purchase,p2,300\n"
        b"e4,u1,2026-10-03T10:20:00Z,2026-10-04T10:20:00Z,view,p1,\n"
        b"e5,u1,2026-10-03T10:25:00Z,2026-10-04T10:25:01Z,view,p1,\n"
        b"e6,u1,2026-10-03T10:30:00Z,2026-10-03T10:00:00Z,view,p1,\n",
    ],
    clicks=[CLICKS_H + b"\n"],
    links=[LINKS_H + b"\n"],
    user_rows=["u1,2026-10-03,1,4,500,shoes", "u2,2026-10-03,1,1,300,books"],
    camp_rows=["organic,organic,2026-10-03,2,2,800,0"],
)

# X3 IDs that are not plain [0-9A-Za-z_] strings, through every path. Links join " u1", "u1 "
# and "x-1"; the byte-wise smallest is " u1". Event "e 1" has two copies: the earlier one is a
# purchase of p1 for 900 by "x-1". e-2 is a purchase of p2 for 900 by " u1" 20 min later: one
# session, shoes and books tie at 900, books wins. Clicks "k 1" (c_1, by " u1") and "k1" (c_2,
# by "u1 ") share a ts: "k 1" < "k1", so c_1 credits and both assist. "\xc3\xbc1" (u-umlaut)
# is its own person, credited by k-9 for c_2.
case(
    "X3",
    events=[
        EVENTS_H + b"\n"
        b"\"e 1\",\"u1 \",2026-10-03T10:00:00Z,2026-10-03T10:01:00Z,view,p1,\n"
        b"\"e 1\",x-1,2026-10-03T10:00:00Z,2026-10-03T10:00:30Z,purchase,p1,900\n"
        b"e-2, u1,2026-10-03T10:20:00Z,2026-10-03T10:21:00Z,purchase,p2,900\n"
        b"e-3,\xc3\xbc1,2026-10-03T10:00:00Z,2026-10-03T10:01:00Z,view,p1,\n",
    ],
    clicks=[
        CLICKS_H + b"\n"
        b"\"k 1\", u1,c_1,2026-10-03T09:00:00Z\n"
        b"k1,u1 ,c_2,2026-10-03T09:00:00Z\n"
        b"k-9,\xc3\xbc1,c_2,2026-10-03T09:30:00Z\n",
    ],
    links=[LINKS_H + b"\n\" u1\",\"u1 \"\nu1 ,x-1\n"],
    user_rows=[" u1,2026-10-03,1,2,1800,books", "ü1,2026-10-03,1,1,0,"],
    camp_rows=["c_1,search,2026-10-03,1,1,1800,1", "c_2,social,2026-10-03,1,0,0,2"],
)
