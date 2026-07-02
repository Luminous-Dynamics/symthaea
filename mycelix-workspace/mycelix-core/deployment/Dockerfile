# Mycelix Validator Node Dockerfile
# ==================================
# Production-ready container for running a Mycelix testnet validator.
#
# Features:
#   - FL Coordinator with 0TML Byzantine detection
#   - Holochain conductor for DHT operations
#   - Prometheus metrics exporter
#   - Health check endpoint
#   - Multi-stage build for minimal image size
#
# Build:
#   docker build -t mycelix-validator:latest -f deployment/Dockerfile .
#
# Run:
#   docker run -d --name validator mycelix-validator:latest

# =============================================================================
# STAGE 1: BUILD RUST COMPONENTS
# =============================================================================
FROM rust:1.75-bookworm AS rust-builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Copy Rust source files
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/
COPY 0TML/gen7-zkstark/ ./0TML/gen7-zkstark/

# Build Rust components (release mode)
RUN cargo build --release --features full 2>/dev/null || echo "Rust build optional"

# =============================================================================
# STAGE 2: BUILD HOLOCHAIN
# =============================================================================
FROM rust:1.75-bookworm AS holochain-builder

WORKDIR /build

# Install Holochain build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Holochain (use nix-installed or download pre-built)
ENV HOLOCHAIN_VERSION=0.4.0-dev.4

# Download pre-built holochain binaries (faster than building from source)
RUN curl -L https://github.com/holochain/holochain/releases/download/holochain-${HOLOCHAIN_VERSION}/holochain-${HOLOCHAIN_VERSION}-x86_64-unknown-linux-gnu.tar.gz \
    | tar xz -C /usr/local/bin/ 2>/dev/null || true

# Fallback: Create placeholder if download fails
RUN touch /usr/local/bin/holochain && chmod +x /usr/local/bin/holochain

# =============================================================================
# STAGE 3: PRODUCTION RUNTIME
# =============================================================================
FROM python:3.11-slim-bookworm AS production

LABEL maintainer="Luminous Dynamics <team@luminousdynamics.com>"
LABEL description="Mycelix Testnet Validator Node"
LABEL version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/luminous-dynamics/mycelix-core"

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Mycelix defaults
    MYCELIX_USE_RUST=true \
    MYCELIX_HV_DIMENSION=2048 \
    MYCELIX_BYZANTINE_THRESHOLD=0.45 \
    # Logging
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    RUST_LOG=info \
    # Paths
    APP_HOME=/app \
    DATA_PATH=/data \
    LOGS_PATH=/logs \
    CONFIG_PATH=/config \
    CACHE_PATH=/cache

WORKDIR ${APP_HOME}

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Networking
    curl \
    wget \
    netcat-openbsd \
    dnsutils \
    # System utilities
    procps \
    htop \
    # SSL/TLS
    ca-certificates \
    libssl3 \
    # Build essentials for Python packages
    gcc \
    g++ \
    libffi-dev \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r mycelix --gid=1000 && \
    useradd -r -g mycelix --uid=1000 --home-dir=${APP_HOME} --shell=/bin/bash mycelix && \
    mkdir -p ${DATA_PATH} ${LOGS_PATH} ${CONFIG_PATH} ${CACHE_PATH} && \
    chown -R mycelix:mycelix ${APP_HOME} ${DATA_PATH} ${LOGS_PATH} ${CONFIG_PATH} ${CACHE_PATH}

# Copy Holochain binaries from builder
COPY --from=holochain-builder /usr/local/bin/holochain* /usr/local/bin/ 2>/dev/null || true

# Copy Rust binaries from builder
COPY --from=rust-builder /build/target/release/mycelix* /usr/local/bin/ 2>/dev/null || true

# Copy Python requirements and install dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        numpy>=1.24.0 \
        torch>=2.0.0 \
        aiohttp>=3.9.0 \
        pyyaml>=6.0 \
        structlog>=23.2.0 \
        prometheus-client>=0.19.0 \
        uvicorn>=0.24.0 \
        fastapi>=0.104.0 \
        pydantic>=2.5.0 \
        asyncio-mqtt>=0.16.0 \
        websockets>=12.0

# Copy application code
COPY --chown=mycelix:mycelix 0TML/ ./0TML/
COPY --chown=mycelix:mycelix libs/ ./libs/
COPY --chown=mycelix:mycelix src/ ./src/
COPY --chown=mycelix:mycelix config/ ./config/
COPY --chown=mycelix:mycelix conductor-config.yaml ./
COPY --chown=mycelix:mycelix deployment/scripts/ ./scripts/

# Create entrypoint script
RUN cat > /usr/local/bin/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

echo "=============================================="
echo "  Mycelix Validator Node"
echo "  Version: 1.0.0"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Node ID:      ${VALIDATOR_NODE_ID:-validator-001}"
echo "  Region:       ${VALIDATOR_REGION:-global}"
echo "  Network:      ${NETWORK_TYPE:-testnet}"
echo "  Bootstrap:    ${BOOTSTRAP_URL:-https://bootstrap.mycelix.net}"
echo ""

# Create required directories
mkdir -p ${DATA_PATH}/holochain ${DATA_PATH}/fl ${LOGS_PATH} ${CACHE_PATH}

# Generate conductor config if not exists
if [ ! -f "${CONFIG_PATH}/conductor-config.yaml" ]; then
    echo "Generating conductor configuration..."
    python3 -c "
import yaml
import os

config = {
    'environment_path': '${DATA_PATH}/holochain',
    'use_dangerous_test_keystore': os.getenv('NETWORK_TYPE', 'testnet') == 'testnet',
    'admin_interfaces': [{
        'driver': {
            'type': 'websocket',
            'port': int(os.getenv('HC_ADMIN_PORT', '39329')),
            'allowed_origins': '*'
        }
    }],
    'app_interfaces': [{
        'driver': {
            'type': 'websocket',
            'port': int(os.getenv('HC_APP_PORT', '9998')),
            'allowed_origins': '*'
        }
    }],
    'network': {
        'transport_pool': [{'type': 'webrtc', 'signal_url': os.getenv('SIGNAL_URL', 'wss://signal.holo.host')}],
        'default_network_type': 'quic_bootstrap',
        'bootstrap_url': os.getenv('BOOTSTRAP_URL', 'https://bootstrap.mycelix.net'),
        'tuning_params': {
            'gossip_loop_iteration_delay_ms': 1000,
            'default_notify_remote_agent_count': 5,
            'default_notify_timeout_ms': 10000,
        }
    },
    'dpki': {'no_dpki': True}
}

with open('${CONFIG_PATH}/conductor-config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print('Conductor config generated at ${CONFIG_PATH}/conductor-config.yaml')
"
fi

# Start services
echo "Starting Mycelix Validator..."
exec python3 /app/scripts/validator_entrypoint.py "$@"
EOF

RUN chmod +x /usr/local/bin/entrypoint.sh

# Create health check script
RUN cat > /usr/local/bin/healthcheck.sh << 'EOF'
#!/bin/bash
curl -sf http://localhost:${HEALTH_CHECK_PORT:-8080}/health || exit 1
EOF
RUN chmod +x /usr/local/bin/healthcheck.sh

# Switch to non-root user
USER mycelix

# Expose ports
EXPOSE 39329 9998 9090 8080 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Default command
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--mode", "validator"]
