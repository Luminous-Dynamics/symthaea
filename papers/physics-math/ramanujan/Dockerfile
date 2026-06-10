# syntax=docker/dockerfile:1.6
# ═════════════════════════════════════════════════════════════════════════
# Ramanujan Protocol — reproducible showcase container.
#
# Pins: Rust stable, Z3 4.13, Debian 12 (bookworm).
# Build:   docker build -t ramanujan-repro papers/ramanujan/
# Run:     docker run --rm ramanujan-repro
# Output:  full pipeline stdout (the LaTeX table is appended at the end).
# ═════════════════════════════════════════════════════════════════════════

FROM rust:1.75-bookworm AS builder

ARG Z3_VERSION=4.13.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libssl-dev \
        curl \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Z3 at the pinned version. Using the prebuilt glibc binary to avoid
# recompiling Z3 from source (30+ minutes added).
RUN curl -sSL \
      "https://github.com/Z3Prover/z3/releases/download/z3-${Z3_VERSION}/z3-${Z3_VERSION}-x64-glibc-2.35.zip" \
      -o /tmp/z3.zip \
    && (command -v unzip || (apt-get update && apt-get install -y unzip)) \
    && unzip -q /tmp/z3.zip -d /opt \
    && ln -s "/opt/z3-${Z3_VERSION}-x64-glibc-2.35/bin/z3" /usr/local/bin/z3 \
    && rm /tmp/z3.zip \
    && z3 --version

WORKDIR /work

# Copy the entire symthaea workspace. The build context is assumed to be the
# symthaea/ directory when running from CI; for local docker build, invoke
# as:  docker build -f papers/ramanujan/Dockerfile -t ramanujan-repro .
#
# (the .dockerignore file should exclude target/, .git/, and large dataset
# directories.)
COPY . /work/

# Primary build + run. Release profile is mandatory per the paper's numeric
# claims (debug profile may change variance threshold outcomes).
ENV CARGO_HOME=/work/.cargo CARGO_TARGET_DIR=/work/target-docker
RUN cargo build --release -p symthaea-physics-bridge --example ramanujan_showcase

# Default command: rerun the showcase. Anything the reader needs to verify
# shows up on stdout; the LaTeX table is the last block printed.
CMD ["cargo", "run", "--release", "-p", "symthaea-physics-bridge", "--example", "ramanujan_showcase"]
