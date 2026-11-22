# Shopping Sessions engine

A Rust program for the Databricks "Shopping Sessions" track. It reads the click-stream CSVs
under `DATA_DIR` (events, clicks, links, products, campaigns, manifest) and writes the two
Parquet tables `OUTPUT_DIR/user_daily/` and `OUTPUT_DIR/campaign_daily/` exactly as the
track's `rules.md` defines them.

It is built as separate stages with plain inputs and outputs.

## Run

```sh
cargo build --release
DATA_DIR=/path/to/data OUTPUT_DIR=/path/to/out NCPU=8 target/release/engine
# or: target/release/engine [--timing] DATA_DIR OUTPUT_DIR
```

- `NCPU` sets the number of worker threads. Without it, the engine uses all CPUs.
- `--timing` (or `ENGINE_TIMING=1`) prints the milliseconds of each stage and some counts to
  stderr. Nothing is ever written to stdout.
- The exit code is 0 on success and 1 on an I/O error.

## Docker

The `Dockerfile` has two stages. The first builds a fully static musl binary on
`rust:1.99-alpine3.22`; on x86-64 it compiles with `-C target-cpu=x86-64-v3`, which needs
AVX2. The second stage is `FROM scratch` and
holds only `/engine`, runs as uid 1000, has no shell, and needs no network at run time.

```sh
docker build -t shopping-sessions .
docker run --rm --network none --read-only -v "$PWD/data":/data:ro -v "$PWD/out":/output \
  --user 1000:1000 -e NCPU=8 shopping-sessions
```

## Tests

- `cargo test --release`: unit tests for the CSV dialect and newline search, timestamps,
  prices, ID codes, links, lookups, assist counting, the Parquet table split, and the allocator
  (size classes, and a threaded stress test that checks every block's contents).
- `tests/fixtures/X1..X3`: edge cases in the starter's fixture layout, with expected rows
  derived by hand from rules.md. They cover file endings, dedup across files, and IDs that
  are not plain alphanumerics on every path. `tests/make_fixtures.py` regenerates them.
  Run them with the starter's `scripts/run_fixtures.py tests/fixtures -- target/release/engine`.

## Dependencies

None. All code in this repository was written by this team; `Cargo.lock` lists no package
from crates.io. The workspace has three crates, all ours:

| Crate | Path | What it is |
| --- | --- | --- |
| `shopping-sessions-engine` | `.` | the engine, including its global allocator (`src/alloc.rs`), input mapping and newline search |
| `parselab` | `crates/parselab` | the input parser and the read-only file mapping (`mmap` by a plain `extern "C"` declaration) |
| `pqw` | `crates/pqw` | the Parquet writer |

System calls go through plain `extern "C"` declarations against the libc that the static
binary links (musl). Build image: `rust:1.99-alpine3.22` (its gcc only drives the link; no C
code is compiled).

## License

MIT, see [LICENSE](LICENSE).
