# =============================================================================
# ZK Tax Proof API Server
# =============================================================================
# Multi-stage build for minimal production image

# Stage 1: Build
FROM rust:1.75-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for better caching
COPY Cargo.toml Cargo.lock ./
COPY methods/Cargo.toml methods/

# Create dummy main for dependency caching
RUN mkdir -p src/bin && \
    echo 'fn main() {}' > src/bin/zk-tax.rs && \
    mkdir -p methods/guest/src && \
    echo 'fn main() {}' > methods/guest/src/main.rs

# Build dependencies only
RUN cargo build --release --features server 2>/dev/null || true

# Copy actual source
COPY src ./src
COPY methods ./methods
COPY examples ./examples
COPY benches ./benches
COPY tests ./tests

# Build the actual binary
RUN touch src/lib.rs src/bin/zk-tax.rs && \
    cargo build --release --bin zk-tax --features server

# Build the server example
RUN cargo build --release --example axum_server --features server || true

# Stage 2: Runtime
FROM debian:bookworm-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -u 1000 -m zkuser
USER zkuser

# Copy binary from builder
COPY --from=builder /app/target/release/zk-tax /usr/local/bin/zk-tax

# Set environment variables
ENV PORT=3000
ENV DEV_MODE=true
ENV RUST_LOG=info

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health || exit 1

# Run the CLI in server mode (or use the axum_server example)
ENTRYPOINT ["zk-tax"]
CMD ["--help"]

# =============================================================================
# Usage:
#
#   # Build the image
#   docker build -t zk-tax-api .
#
#   # Run the CLI
#   docker run --rm zk-tax-api prove --income 85000 --jurisdiction US --year 2024
#
#   # Run with custom command
#   docker run --rm zk-tax-api brackets --jurisdiction US --year 2024
#
#   # For the API server (requires building axum_server example)
#   docker run -p 3000:3000 -e DEV_MODE=true zk-tax-api serve
#
# =============================================================================
