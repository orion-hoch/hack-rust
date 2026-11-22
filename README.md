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

- `cargo test --release`: unit tests for the CSV dialect, timestamps, prices, ID codes,
  links, lookups and assist counting.
- `tests/fixtures/X1..X3`: edge cases in the starter's fixture layout, with expected rows
  derived by hand from rules.md. They cover file endings, dedup across files, and IDs that
  are not plain alphanumerics on every path. `tests/make_fixtures.py` regenerates them.
  Run them with the starter's `scripts/run_fixtures.py tests/fixtures -- target/release/engine`.

## Dependencies

Direct dependencies, all from crates.io (versions pinned in `Cargo.lock`):

| Crate | Version | Use |
| --- | --- | --- |
| `memmap2` | 0.9.12 | memory-maps the input files |
| `memchr` | 2.8.3 | finds newlines |
| `mimalloc` | 0.1.52 (with `libmimalloc-sys` 0.1.49, which bundles Microsoft's mimalloc C library) | global allocator; musl's own allocator made the static build twice as slow |
| `parquet` | 60.2.0, default features off | writes the Parquet files (low-level column writer, no Arrow) |
| `bytes` | 1.12.2 | zero-copy string slices for the Parquet writer |

Transitive dependencies, pulled in by `parquet` and `mimalloc`: ahash, autocfg, bumpalo,
cc, cfg-if, chrono, const-random, const-random-macro, crunchy, find-msvc-tools,
futures-core, futures-task, futures-util, getrandom, half, hashbrown, iana-time-zone,
js-sys, libc, libm, log, num-bigint, num-integer, num-traits, once_cell, pin-project-lite,
proc-macro2, quote, rustversion, seq-macro, shlex, slab, syn, tiny-keccak, twox-hash,
unicode-ident, version_check, zerocopy, zerocopy-derive. A few platform crates
(android_system_properties, core-foundation-sys, iana-time-zone-haiku, r-efi, wasi,
wasip2, wasm-bindgen*, windows-*, wit-bindgen) are listed in `Cargo.lock` but are not
compiled on Linux.

Build images: `rust:1.99-alpine3.22`, plus the `musl-dev` package from Alpine.

## License

MIT, see [LICENSE](LICENSE).
