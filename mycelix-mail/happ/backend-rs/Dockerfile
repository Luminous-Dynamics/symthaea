# Mycelix-Mail Backend Dockerfile
# Multi-stage build for optimized production image

# ============================================================================
# Stage 1: Build
# ============================================================================
FROM rust:1.75-slim-bookworm as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy Cargo files first for dependency caching
COPY Cargo.toml Cargo.lock ./

# Create dummy src to build dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release && rm -rf src

# Copy actual source code
COPY src ./src

# Touch main.rs to ensure rebuild
RUN touch src/main.rs

# Build the actual application
RUN cargo build --release

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 mycelix
USER mycelix

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/mycelix-mail-backend /app/

# Copy .env.example as reference
COPY --chown=mycelix:mycelix .env.example /app/.env.example

# Expose port
EXPOSE 3001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:3001/health || exit 1

# Run the binary
CMD ["./mycelix-mail-backend"]
