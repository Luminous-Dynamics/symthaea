# Multi-stage Dockerfile for Mycelix-DeSci
# Stage 1: Builder
FROM rust:1.75-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace files
COPY Cargo.toml ./
COPY src/core ./src/core
COPY src/api ./src/api

# Build release binary
RUN cargo build --release --package mycelix-desci-api

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 mycelix

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/mycelix-api /usr/local/bin/mycelix-api

# Change ownership
RUN chown -R mycelix:mycelix /app

USER mycelix

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the API server
CMD ["mycelix-api"]
