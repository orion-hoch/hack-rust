# Shopping Sessions engine. The scorers build this natively on linux/amd64, so no FROM
# line pins a platform.
#
# Build stage: Alpine's Rust targets musl, so the binary is fully static.
FROM rust:1.99-alpine3.22 AS build
RUN apk add --no-cache musl-dev
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# x86-64-v3 = AVX2, BMI2, FMA.
ARG X86_TARGET_CPU=x86-64-v3
RUN if [ "$(uname -m)" = "x86_64" ]; then export RUSTFLAGS="-C target-cpu=${X86_TARGET_CPU}"; fi; \
    cargo build --release --locked && \
    cp target/release/engine /engine

# Run stage: nothing but the binary. No shell, no libc, no network needed.
FROM scratch
COPY --from=build /engine /engine
USER 1000:1000
ENV DATA_DIR=/data OUTPUT_DIR=/output
ENTRYPOINT ["/engine"]
