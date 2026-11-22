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
# x86-64-v3 = AVX2, BMI2, FMA.
ARG X86_TARGET_CPU=x86-64-v3
RUN if [ "$(uname -m)" = "x86_64" ]; then export RUSTFLAGS="-C target-cpu=${X86_TARGET_CPU}"; fi; \
    cargo build --release --locked --features parselab/mul512 && \
    cp target/release/engine /engine

# Run stage: nothing but the binary. No shell, no libc, no network needed.
FROM scratch
COPY --from=build /engine /engine
USER 1000:1000
ENV DATA_DIR=/data OUTPUT_DIR=/output
ENTRYPOINT ["/engine"]
