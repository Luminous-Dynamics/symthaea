# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Multi-stage build for the Mycelix FL Coordinator.
# Produces a minimal Debian-based image with just the release binary.

# Stage 1: Build
FROM rust:1.82-bookworm AS builder

WORKDIR /build

# Copy workspace files needed for dependency resolution
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/
COPY proto/ proto/

# Build release binary
RUN cargo build --release --bin mycelix-fl-coordinator --features std \
    && strip target/release/mycelix-fl-coordinator

# Stage 2: Minimal runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/mycelix-fl-coordinator /usr/local/bin/

# Non-root user
RUN useradd -r -s /bin/false mycelix
USER mycelix

EXPOSE 50051

# Health check — commented out until gRPC health endpoint is implemented
# HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
#     CMD ["/usr/local/bin/mycelix-fl-coordinator", "--health-check"] || exit 1

ENTRYPOINT ["/usr/local/bin/mycelix-fl-coordinator"]
