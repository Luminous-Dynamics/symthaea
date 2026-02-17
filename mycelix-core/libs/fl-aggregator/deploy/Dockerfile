# Dockerfile for fl-aggregator proof service
# Multi-stage build for minimal production image

# Build stage
FROM rust:1.83-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Copy manifest files for dependency caching
COPY Cargo.toml Cargo.lock ./

# Create dummy src to build dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn dummy() {}" > src/lib.rs

# Build dependencies only
RUN cargo build --release --features "proofs proofs-parallel proofs-telemetry"

# Remove dummy source
RUN rm -rf src

# Copy actual source
COPY src ./src
COPY benches ./benches

# Touch main.rs to trigger rebuild
RUN touch src/lib.rs

# Build the actual application
RUN cargo build --release --features "proofs proofs-parallel proofs-telemetry" --bin proof_server

# Runtime stage
FROM debian:bookworm-slim AS runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false proofservice

# Copy binary from builder
COPY --from=builder /app/target/release/proof_server /app/proof_server

# Copy configuration templates
COPY deploy/config /app/config

# Set ownership
RUN chown -R proofservice:proofservice /app

USER proofservice

# Environment variables
ENV RUST_LOG=info
ENV PROOF_SERVICE_HOST=0.0.0.0
ENV PROOF_SERVICE_PORT=50051
ENV METRICS_PORT=9090
ENV RAYON_NUM_THREADS=0

# Expose ports
# 50051: gRPC service
# 9090: Prometheus metrics
EXPOSE 50051 9090

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD /app/proof_server health || exit 1

# Run the service
ENTRYPOINT ["/app/proof_server"]
CMD ["serve"]
