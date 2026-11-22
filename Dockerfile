# Shopping Sessions engine. The scorers build this natively on linux/amd64, so no FROM
# line pins a platform.
#
# Build stage: Alpine's Rust targets musl, so the binary is fully static. Every crate is this
# team's own; nothing is compiled by a C compiler (the image's gcc only drives the link) and
# nothing is fetched from crates.io.
FROM rust:1.99-alpine3.22 AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY src ./src
# x86-64-v3 = AVX2, BMI2, FMA. prefer-256-bit: the compiler vectorises ordinary loops and block
# fills with at most 256-bit registers, also inside functions that have AVX-512 enabled (the code
# written for 512-bit registers is unaffected). On Skylake-X and Cascade Lake any 512-bit
# instruction lowers the core's clock for about 2 ms, so stray ones outside the parser cost a
# whole stage its speed. rustc prints a warning for this LLVM-level feature name; that is expected.
# Branch padding: no jump, fused compare-and-jump, call or return crosses or ends on a 32-byte
# boundary (instruction prefixes are added to earlier instructions where possible, NOPs
# otherwise). The Skylake-family microcode erratum SKX102 keeps such branches out of the decoded
# instruction cache; without the padding about a fifth of the branches in the grouping and
# session stages sat on a boundary. Costs half a percent more instructions; CPUs without the
# erratum lose nothing measurable.
ARG X86_TARGET_CPU=x86-64-v3
RUN if [ "$(uname -m)" = "x86_64" ]; then export RUSTFLAGS="-C target-cpu=skylake-avx512 -C target-feature=-avx512f,+prefer-256-bit -C llvm-args=--x86-align-branch-boundary=32 -C llvm-args=--x86-align-branch=fused+jcc+jmp+call+ret+indirect -C llvm-args=--x86-pad-max-prefix-size=5"; fi; \
    cargo build --release --locked && \
    cp target/release/engine /engine

# Run stage: nothing but the binary. No shell, no libc, no network needed.
FROM scratch
COPY --from=build /engine /engine
USER 1000:1000
ENV DATA_DIR=/data OUTPUT_DIR=/output
ENTRYPOINT ["/engine"]
