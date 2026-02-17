# Multi-stage Dockerfile for Mycelix Supply Chain Provenance Service
#
# Production-optimized with:
# - Security hardening (non-root user)
# - Health checks
# - Small image size
# - Build caching

# ============================================================================
# Stage 1: Build environment
# ============================================================================
FROM rust:1.75 as builder

WORKDIR /app

# Copy workspace files
COPY Cargo.toml ./
COPY claim-model ./claim-model
COPY crypto ./crypto
COPY service ./service

# Build release
WORKDIR /app/service
RUN cargo build --release

# ============================================================================
# Stage 2: Runtime environment
# ============================================================================
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash mycelix

WORKDIR /app

# Copy binary
COPY --from=builder /app/target/release/provenance-service /app/provenance-service

# Create data directory for SQLite
RUN mkdir -p /app/data && chown -R mycelix:mycelix /app

# Switch to non-root user for security
USER mycelix

# Expose port
EXPOSE 8080

# Health check using the /health/live endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

# Run service
CMD ["/app/provenance-service"]
