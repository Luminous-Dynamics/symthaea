#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Symthaea Test Suite Runner
# ═══════════════════════════════════════════════════════════════════════════════
#
# Organized test suites for the growing Symthaea codebase.
# Each suite can be run independently with clear output.
#
# Usage:
#   ./scripts/test-suites.sh              # Run all suites (default features)
#   ./scripts/test-suites.sh crypto       # Run just the crypto suite
#   ./scripts/test-suites.sh core         # Run just symthaea-core
#   ./scripts/test-suites.sh quick        # Fast smoke test (~30s)
#   ./scripts/test-suites.sh full         # All features, all suites
#   ./scripts/test-suites.sh list         # List available suites
#
# Exit code: 0 if all requested suites pass, 1 if any fail.
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

PASS=0
FAIL=0
SKIP=0
RESULTS=()

run_suite() {
    local name="$1"
    local cmd="$2"
    local description="$3"

    printf "${CYAN}[SUITE]${NC} %-30s %s\n" "$name" "$description"
    if eval "$cmd" > /tmp/symthaea-test-${name}.log 2>&1; then
        local count
        count=$(grep -c "^test .* ok$" /tmp/symthaea-test-${name}.log 2>/dev/null || echo "?")
        printf "${GREEN}  PASS${NC} (%s tests)\n" "$count"
        PASS=$((PASS + 1))
        RESULTS+=("PASS: $name ($count tests)")
    else
        printf "${RED}  FAIL${NC} (see /tmp/symthaea-test-${name}.log)\n"
        # Show last few lines of failure
        tail -5 /tmp/symthaea-test-${name}.log | sed 's/^/    /'
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL: $name")
    fi
}

skip_suite() {
    local name="$1"
    local reason="$2"
    printf "${YELLOW}[SKIP]${NC} %-30s %s\n" "$name" "$reason"
    SKIP=$((SKIP + 1))
    RESULTS+=("SKIP: $name ($reason)")
}

# ─── Suite Definitions ────────────────────────────────────────────────────────

suite_core() {
    run_suite "core-hdc-crypto" \
        "cargo test -p symthaea-core --lib -- hdc_crypto" \
        "HDC-MAC, threshold sharing, context keys, commitments"

    run_suite "core-hdc-fhe" \
        "cargo test -p symthaea-core --lib -- hdc_fhe" \
        "Encrypted HVs, homomorphic bind, collective wisdom pool"

    run_suite "core-proptests" \
        "cargo test -p symthaea-core --test proptest_hdc_crypto" \
        "Property tests: MAC, threshold, context key, commitment"

    run_suite "core-thresholds" \
        "cargo test -p symthaea-core --lib -- thresholds" \
        "Named constant validation (ordering, bounds)"
}

suite_crypto() {
    suite_core  # crypto lives in core

    run_suite "swarm-handshake" \
        "cargo test --lib -- swarm::handshake" \
        "Ed25519/BLAKE3 handshake, mutual auth, rate limiting"

    run_suite "swarm-types" \
        "cargo test --lib -- swarm::types" \
        "TrustLevel, SecurityTelemetry, ConsciousnessVector"

    run_suite "swarm-config" \
        "cargo test --lib -- swarm::config" \
        "SwarmConfig, CryptoConfig, BootstrapConfig"
}

suite_mesh() {
    run_suite "mesh-encryption" \
        "cargo test --lib --features mesh-encryption -- swarm::mesh" \
        "ChaCha20-Poly1305, epoch nonce, key rotation, HDC-MAC on WisdomPacket"

    run_suite "mesh-proptests" \
        "cargo test --lib --features mesh-encryption -- swarm::mesh::proptests" \
        "Fragment reassembly proptests"
}

suite_spore() {
    run_suite "spore-sensor" \
        "cargo test -p symthaea-spore --lib -- sensor_bridge" \
        "Sensor bridge, context key derivation"

    run_suite "spore-ble" \
        "cargo test -p symthaea-spore --lib -- ble_mesh" \
        "BLE mesh peer tracking"
}

suite_cognitive() {
    run_suite "civic-crisis" \
        "cargo test --lib -- civic_crisis_detector" \
        "Crisis detection pipeline (PE → emergency)"

    run_suite "motor-bridge" \
        "cargo test --lib -- motor_output_bridge" \
        "Motor output, robot actuator commands"

    run_suite "guardian" \
        "cargo test --lib -- guardian::tests" \
        "Guardian posture state machine"

    run_suite "thresholds" \
        "cargo test --lib -- cognitive_loop::thresholds" \
        "All named constant validation"
}

suite_integration() {
    run_suite "swarm-channel" \
        "cargo test --test swarm_channel_integration" \
        "Swarm channel event pipeline"

    run_suite "swarm-integration" \
        "cargo test --test swarm_integration" \
        "Swarm handshake integration"
}

suite_quick() {
    # ~30s smoke test: just the core + handshake + thresholds
    run_suite "core-hdc-crypto" \
        "cargo test -p symthaea-core --lib -- hdc_crypto" \
        "HDC crypto primitives (30 tests)"

    run_suite "core-hdc-fhe" \
        "cargo test -p symthaea-core --lib -- hdc_fhe" \
        "HDC-FHE (12 tests)"

    run_suite "swarm-handshake" \
        "cargo test --lib -- swarm::handshake" \
        "Handshake (12 tests)"
}

# ─── Main ─────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")/.."

SUITE="${1:-all}"

case "$SUITE" in
    list)
        echo "Available suites:"
        echo "  quick       - Smoke test (~30s): core crypto + handshake"
        echo "  core        - symthaea-core: HDC crypto, FHE, proptests, thresholds"
        echo "  crypto      - Full crypto stack: core + handshake + types + config"
        echo "  mesh        - Mesh encryption: ChaCha20, epoch, key rotation"
        echo "  spore       - Sensor bridge, BLE mesh"
        echo "  cognitive   - Crisis detector, motor bridge, guardian, thresholds"
        echo "  integration - Integration tests (swarm channel, swarm integration)"
        echo "  all         - All suites (default features)"
        echo "  full        - All suites with all crypto features"
        exit 0
        ;;
    quick)      suite_quick ;;
    core)       suite_core ;;
    crypto)     suite_crypto ;;
    mesh)       suite_mesh ;;
    spore)      suite_spore ;;
    cognitive)  suite_cognitive ;;
    integration) suite_integration ;;
    all)
        suite_core
        suite_crypto
        suite_spore
        suite_cognitive
        suite_integration
        ;;
    full)
        suite_core
        suite_crypto
        suite_mesh
        suite_spore
        suite_cognitive
        suite_integration
        ;;
    *)
        echo "Unknown suite: $SUITE"
        echo "Run '$0 list' for available suites"
        exit 1
        ;;
esac

# ─── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════"
printf "${GREEN}PASS: %d${NC}  ${RED}FAIL: %d${NC}  ${YELLOW}SKIP: %d${NC}\n" "$PASS" "$FAIL" "$SKIP"
echo "═══════════════════════════════════════════════════════════"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
