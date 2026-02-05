# Symthaea HLB - Multi-stage production build
#
# Build:   docker build -t symthaea-api .
# Run:     docker run -p 8080:8080 symthaea-api
# Health:  curl http://localhost:8080/health

# ── Stage 1: Build ──────────────────────────────────────────────
FROM rust:1.83-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Cache dependencies: copy manifests first
COPY Cargo.toml Cargo.lock ./
COPY symthaea-core/Cargo.toml symthaea-core/Cargo.toml
# Create stub crate dirs so cargo can resolve the workspace
COPY crates/symthaea-stt/Cargo.toml crates/symthaea-stt/Cargo.toml
COPY crates/symthaea-sentinel/Cargo.toml crates/symthaea-sentinel/Cargo.toml
COPY crates/symthaea-consciousness/Cargo.toml crates/symthaea-consciousness/Cargo.toml
COPY crates/symthaea-perception/Cargo.toml crates/symthaea-perception/Cargo.toml
COPY crates/symthaea-math/Cargo.toml crates/symthaea-math/Cargo.toml
COPY crates/symthaea-dynamics/Cargo.toml crates/symthaea-dynamics/Cargo.toml
COPY crates/symthaea-gym/Cargo.toml crates/symthaea-gym/Cargo.toml

# Create stub lib.rs files so cargo fetch works
RUN mkdir -p src symthaea-core/src \
    crates/symthaea-stt/src \
    crates/symthaea-sentinel/src \
    crates/symthaea-consciousness/src \
    crates/symthaea-perception/src \
    crates/symthaea-math/src \
    crates/symthaea-dynamics/src \
    crates/symthaea-gym/src \
    && echo "fn main() {}" > src/main.rs \
    && echo "" > src/lib.rs \
    && echo "" > symthaea-core/src/lib.rs \
    && for d in crates/*/src; do echo "" > "$d/lib.rs"; done

# Pre-fetch and compile dependencies (cached layer)
RUN cargo fetch

# Copy actual source
COPY . .

# Touch source files to invalidate stubs
RUN find src symthaea-core/src crates -name '*.rs' -exec touch {} +

# Build the API server binary
RUN cargo build --release --bin symthaea-api --features api_module \
    && strip /build/target/release/symthaea-api

# ── Stage 2: Runtime ────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r symthaea && useradd -r -g symthaea symthaea

COPY --from=builder /build/target/release/symthaea-api /usr/local/bin/symthaea-api

# Default configuration via environment
ENV SYMTHAEA_HOST=0.0.0.0
ENV SYMTHAEA_PORT=8080
ENV RUST_LOG=symthaea=info

EXPOSE 8080

USER symthaea

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/usr/local/bin/symthaea-api", "--health-check"] || exit 1

ENTRYPOINT ["/usr/local/bin/symthaea-api"]
