#!/usr/bin/env bash
#
# Mycelix Validator Quick-Start Script
#
# This script helps new validators set up their environment quickly.
# It checks prerequisites, generates keys, and prepares the node.
#
# Usage: ./scripts/validator-quickstart.sh [--testnet|--mainnet] [--non-interactive]
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
NETWORK="testnet"
INTERACTIVE=true
MYCELIX_HOME="${MYCELIX_HOME:-$HOME/.mycelix}"
MIN_DISK_GB=250
MIN_RAM_GB=16
MIN_CORES=8

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --testnet)
            NETWORK="testnet"
            shift
            ;;
        --mainnet)
            NETWORK="mainnet"
            shift
            ;;
        --non-interactive)
            INTERACTIVE=false
            shift
            ;;
        --help)
            echo "Mycelix Validator Quick-Start"
            echo ""
            echo "Usage: $0 [--testnet|--mainnet] [--non-interactive]"
            echo ""
            echo "Options:"
            echo "  --testnet         Configure for testnet (default)"
            echo "  --mainnet         Configure for mainnet"
            echo "  --non-interactive Skip interactive prompts"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Helper functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }

confirm() {
    if $INTERACTIVE; then
        read -r -p "$1 [y/N] " response
        [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
    else
        return 0
    fi
}

# Header
echo ""
echo "=============================================="
echo "  Mycelix Validator Quick-Start"
echo "  Network: $NETWORK"
echo "=============================================="
echo ""

# =============================================================================
# Step 1: System Requirements Check
# =============================================================================
log_step "Step 1: Checking system requirements..."
echo ""

PREREQ_PASSED=true

# Check CPU cores
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "0")
if [[ $CORES -ge $MIN_CORES ]]; then
    log_pass "CPU: $CORES cores (minimum: $MIN_CORES)"
else
    log_fail "CPU: $CORES cores (minimum: $MIN_CORES required)"
    PREREQ_PASSED=false
fi

# Check RAM
if [[ -f /proc/meminfo ]]; then
    RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    RAM_GB=$((RAM_KB / 1024 / 1024))
elif command -v sysctl &>/dev/null; then
    RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
    RAM_GB=$((RAM_BYTES / 1024 / 1024 / 1024))
else
    RAM_GB=0
fi

if [[ $RAM_GB -ge $MIN_RAM_GB ]]; then
    log_pass "RAM: ${RAM_GB}GB (minimum: ${MIN_RAM_GB}GB)"
else
    log_fail "RAM: ${RAM_GB}GB (minimum: ${MIN_RAM_GB}GB required)"
    PREREQ_PASSED=false
fi

# Check disk space
DISK_AVAIL_KB=$(df -k "$HOME" | tail -1 | awk '{print $4}')
DISK_AVAIL_GB=$((DISK_AVAIL_KB / 1024 / 1024))
if [[ $DISK_AVAIL_GB -ge $MIN_DISK_GB ]]; then
    log_pass "Disk: ${DISK_AVAIL_GB}GB available (minimum: ${MIN_DISK_GB}GB)"
else
    log_fail "Disk: ${DISK_AVAIL_GB}GB available (minimum: ${MIN_DISK_GB}GB required)"
    PREREQ_PASSED=false
fi

echo ""

# =============================================================================
# Step 2: Software Dependencies Check
# =============================================================================
log_step "Step 2: Checking software dependencies..."
echo ""

REQUIRED_TOOLS=("docker" "curl" "openssl")
OPTIONAL_TOOLS=("age" "sops" "kubectl")

for tool in "${REQUIRED_TOOLS[@]}"; do
    if command -v "$tool" &>/dev/null; then
        VERSION=$($tool --version 2>/dev/null | head -1 || echo "installed")
        log_pass "$tool: $VERSION"
    else
        log_fail "$tool: not installed (REQUIRED)"
        PREREQ_PASSED=false
    fi
done

for tool in "${OPTIONAL_TOOLS[@]}"; do
    if command -v "$tool" &>/dev/null; then
        log_pass "$tool: installed"
    else
        log_warn "$tool: not installed (optional)"
    fi
done

# Check Docker is running
if docker info &>/dev/null; then
    log_pass "Docker daemon: running"
else
    log_fail "Docker daemon: not running"
    PREREQ_PASSED=false
fi

echo ""

if ! $PREREQ_PASSED; then
    log_fail "Prerequisites not met. Please fix the issues above and re-run."
    exit 1
fi

log_pass "All prerequisites met!"
echo ""

# =============================================================================
# Step 3: Create Directory Structure
# =============================================================================
log_step "Step 3: Creating directory structure..."
echo ""

mkdir -p "$MYCELIX_HOME"/{keys,data,logs,config}
chmod 700 "$MYCELIX_HOME/keys"

log_pass "Created: $MYCELIX_HOME/keys (mode 700)"
log_pass "Created: $MYCELIX_HOME/data"
log_pass "Created: $MYCELIX_HOME/logs"
log_pass "Created: $MYCELIX_HOME/config"

echo ""

# =============================================================================
# Step 4: Generate Validator Keys
# =============================================================================
log_step "Step 4: Generating validator keys..."
echo ""

KEY_DIR="$MYCELIX_HOME/keys"
VALIDATOR_KEY="$KEY_DIR/validator.key"
VALIDATOR_PUB="$KEY_DIR/validator.pub"

if [[ -f "$VALIDATOR_KEY" ]]; then
    log_warn "Validator key already exists: $VALIDATOR_KEY"
    if confirm "Overwrite existing key?"; then
        rm -f "$VALIDATOR_KEY" "$VALIDATOR_PUB"
    else
        log_info "Keeping existing key"
    fi
fi

if [[ ! -f "$VALIDATOR_KEY" ]]; then
    # Generate secp256k1 key pair
    openssl ecparam -name secp256k1 -genkey -noout -out "$VALIDATOR_KEY" 2>/dev/null
    openssl ec -in "$VALIDATOR_KEY" -pubout -out "$VALIDATOR_PUB" 2>/dev/null
    chmod 600 "$VALIDATOR_KEY"
    chmod 644 "$VALIDATOR_PUB"

    log_pass "Generated validator key pair"
    log_info "Private key: $VALIDATOR_KEY (keep this SECRET!)"
    log_info "Public key:  $VALIDATOR_PUB"

    # Display public key for registration
    echo ""
    echo -e "${CYAN}Your Validator Public Key:${NC}"
    echo "----------------------------------------"
    cat "$VALIDATOR_PUB"
    echo "----------------------------------------"
fi

echo ""

# =============================================================================
# Step 5: Generate Node Identity
# =============================================================================
log_step "Step 5: Generating node identity..."
echo ""

NODE_ID_FILE="$KEY_DIR/node-id"
if [[ ! -f "$NODE_ID_FILE" ]]; then
    # Generate a random node ID (32 bytes hex)
    NODE_ID=$(openssl rand -hex 32)
    echo "$NODE_ID" > "$NODE_ID_FILE"
    chmod 600 "$NODE_ID_FILE"
    log_pass "Generated node ID: ${NODE_ID:0:16}..."
else
    NODE_ID=$(cat "$NODE_ID_FILE")
    log_info "Using existing node ID: ${NODE_ID:0:16}..."
fi

echo ""

# =============================================================================
# Step 6: Create Configuration
# =============================================================================
log_step "Step 6: Creating configuration..."
echo ""

CONFIG_FILE="$MYCELIX_HOME/config/validator.toml"

if [[ -f "$CONFIG_FILE" ]]; then
    log_warn "Configuration already exists: $CONFIG_FILE"
    if ! confirm "Overwrite existing configuration?"; then
        log_info "Keeping existing configuration"
    else
        rm -f "$CONFIG_FILE"
    fi
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    cat > "$CONFIG_FILE" << EOF
# Mycelix Validator Configuration
# Network: $NETWORK
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

[node]
id = "$NODE_ID"
network = "$NETWORK"
data_dir = "$MYCELIX_HOME/data"
log_dir = "$MYCELIX_HOME/logs"

[validator]
key_file = "$VALIDATOR_KEY"
stake_amount = 10000  # Minimum stake in MYC
commission_rate = 0.10  # 10%

[network]
listen_addr = "0.0.0.0"
p2p_port = 9998
admin_port = 39329
metrics_port = 9090

[network.bootstrap]
$(if [[ "$NETWORK" == "testnet" ]]; then
echo 'nodes = ['
echo '    "bootstrap1.testnet.mycelix.network:9998",'
echo '    "bootstrap2.testnet.mycelix.network:9998",'
echo ']'
else
echo 'nodes = ['
echo '    "bootstrap1.mycelix.network:9998",'
echo '    "bootstrap2.mycelix.network:9998",'
echo '    "bootstrap3.mycelix.network:9998",'
echo ']'
fi)

[consensus]
# Byzantine fault tolerance settings
max_byzantine_fraction = 0.45
min_reputation_threshold = 0.3

[fl]
# Federated learning settings
max_model_size_mb = 100
gradient_batch_size = 32
aggregation_rounds = 10

[trust]
# MATL (Multi-factor Adaptive Trust Layer) weights
pogq_weight = 0.4   # Proof of Gradient Quality
tcdm_weight = 0.3   # Trust Contribution Decay Model
entropy_weight = 0.3  # Trust Entropy

[logging]
level = "info"
format = "json"
rotate_size_mb = 100
retain_days = 30
EOF

    log_pass "Created configuration: $CONFIG_FILE"
fi

echo ""

# =============================================================================
# Step 7: Pull Docker Images
# =============================================================================
log_step "Step 7: Pulling Docker images..."
echo ""

IMAGES=(
    "ghcr.io/mycelix/validator:v1.0.0-$NETWORK"
    "ghcr.io/mycelix/fl-node:v1.0.0-$NETWORK"
)

for image in "${IMAGES[@]}"; do
    log_info "Pulling $image..."
    if docker pull "$image" 2>/dev/null; then
        log_pass "Pulled: $image"
    else
        log_warn "Could not pull $image (may not be published yet)"
    fi
done

echo ""

# =============================================================================
# Step 8: Create Startup Script
# =============================================================================
log_step "Step 8: Creating startup script..."
echo ""

STARTUP_SCRIPT="$MYCELIX_HOME/start-validator.sh"
cat > "$STARTUP_SCRIPT" << 'EOF'
#!/usr/bin/env bash
#
# Start Mycelix Validator
#

set -euo pipefail

MYCELIX_HOME="${MYCELIX_HOME:-$HOME/.mycelix}"
CONFIG_FILE="$MYCELIX_HOME/config/validator.toml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration not found: $CONFIG_FILE"
    echo "Run validator-quickstart.sh first."
    exit 1
fi

echo "Starting Mycelix Validator..."
echo "Config: $CONFIG_FILE"
echo "Logs: $MYCELIX_HOME/logs/"
echo ""

# Run in Docker
docker run -d \
    --name mycelix-validator \
    --restart unless-stopped \
    -v "$MYCELIX_HOME:/root/.mycelix:ro" \
    -v "$MYCELIX_HOME/data:/data" \
    -v "$MYCELIX_HOME/logs:/logs" \
    -p 9998:9998 \
    -p 39329:39329 \
    -p 9090:9090 \
    ghcr.io/mycelix/validator:v1.0.0-testnet \
    --config /root/.mycelix/config/validator.toml

echo ""
echo "Validator started!"
echo "  View logs: docker logs -f mycelix-validator"
echo "  Stop:      docker stop mycelix-validator"
echo "  Metrics:   http://localhost:9090/metrics"
EOF

chmod +x "$STARTUP_SCRIPT"
log_pass "Created startup script: $STARTUP_SCRIPT"

echo ""

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "  Quick-Start Complete!"
echo "=============================================="
echo ""
echo -e "${GREEN}Your validator is configured for: $NETWORK${NC}"
echo ""
echo "Directory: $MYCELIX_HOME"
echo ""
echo "Files created:"
echo "  - $VALIDATOR_KEY (KEEP SECRET!)"
echo "  - $VALIDATOR_PUB (share for registration)"
echo "  - $CONFIG_FILE"
echo "  - $STARTUP_SCRIPT"
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo "  1. Submit your application at: validators@mycelix.network"
echo "     Include your public key from: $VALIDATOR_PUB"
echo ""
echo "  2. Once approved, start your validator:"
echo "     $STARTUP_SCRIPT"
echo ""
echo "  3. Monitor your node:"
echo "     docker logs -f mycelix-validator"
echo ""
echo -e "${YELLOW}Important:${NC}"
echo "  - Back up your private key: $VALIDATOR_KEY"
echo "  - Never share your private key with anyone"
echo "  - Ensure ports 9998, 39329, 9090 are accessible"
echo ""
echo "Documentation: https://docs.mycelix.network/validators"
echo "Support: #validators on Discord"
echo ""
