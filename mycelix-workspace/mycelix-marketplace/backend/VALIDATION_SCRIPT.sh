#!/usr/bin/env bash
# 🌐 Mycelix Phase 4 Infrastructure Validation
# Confirms all infrastructure is working and ready for MATL testing

set -e

echo "=== 🎯 Mycelix Phase 4 Infrastructure Validation ==="
echo "Date: $(date)"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

success_count=0
total_tests=0

test_item() {
    local name=$1
    local command=$2
    total_tests=$((total_tests + 1))

    echo -n "Testing: $name... "
    if eval "$command" &>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        success_count=$((success_count + 1))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        return 1
    fi
}

echo "📦 1. Binary Verification"
echo "---"
test_item "Holochain binary exists" "[ -f holochain ]"
test_item "Holochain binary executable" "[ -x holochain ]"
test_item "HC CLI exists" "[ -f hc ]"
test_item "HC CLI executable" "[ -x hc ]"
test_item "lair-keystore exists" "[ -f lair-keystore ]"
test_item "lair-keystore executable" "[ -x lair-keystore ]"
echo ""

echo "🐳 2. Docker Network"
echo "---"
test_item "Docker daemon running" "docker ps &>/dev/null"
test_item "mycelix-network exists" "docker network inspect mycelix-network &>/dev/null"
echo ""

echo "📄 3. hApp Bundle Validation"
echo "---"
test_item "hApp file exists" "[ -f mycelix_marketplace.happ ]"
test_item "hApp is valid ZIP" "unzip -t mycelix_marketplace.happ &>/dev/null"
test_item "hApp contains happ.yaml" "unzip -l mycelix_marketplace.happ | grep -q happ.yaml"
test_item "hApp contains dna.dna" "unzip -l mycelix_marketplace.happ | grep -q dna.dna"

# Extract and validate DNA bundle
unzip -q -o mycelix_marketplace.happ -d /tmp/happ-validation 2>/dev/null || true
test_item "DNA bundle is valid ZIP" "unzip -t /tmp/happ-validation/dna.dna &>/dev/null"
test_item "DNA contains dna.yaml" "unzip -l /tmp/happ-validation/dna.dna | grep -q dna.yaml"
test_item "DNA contains WASM files" "unzip -l /tmp/happ-validation/dna.dna | grep -q '.wasm'"

# Count WASM files
wasm_count=$(unzip -l /tmp/happ-validation/dna.dna | grep -c '.wasm' || true)
test_item "All 10 WASMs present" "[ $wasm_count -eq 10 ]"
echo ""

echo "🔧 4. Test Framework"
echo "---"
test_item "test-multi-agent.sh exists" "[ -f test-multi-agent.sh ]"
test_item "test-multi-agent.sh executable" "[ -x test-multi-agent.sh ]"
echo ""

echo "🎯 5. Container Verification"
echo "---"
# Check if test agents are running
if docker ps --filter "name=mycelix-agent-" --format "{{.Names}}" | grep -q "mycelix-agent"; then
    agent_count=$(docker ps --filter "name=mycelix-agent-" --format "{{.Names}}" | wc -l)
    echo -e "  Found $agent_count running agent(s): ${GREEN}✓${NC}"
    success_count=$((success_count + 1))
else
    echo -e "  No agents currently running: ${YELLOW}(expected if not testing)${NC}"
fi
total_tests=$((total_tests + 1))

# Test that we can spawn a test agent
echo -n "Testing: Can spawn test container... "
if docker run --rm --name mycelix-validation-test \
    --network mycelix-network \
    -v "$(pwd):/workspace" \
    -w /workspace \
    ubuntu:22.04 \
    /workspace/holochain --version &>/dev/null; then
    echo -e "${GREEN}✓ PASS${NC}"
    success_count=$((success_count + 1))
else
    echo -e "${RED}✗ FAIL${NC}"
fi
total_tests=$((total_tests + 1))

echo ""
echo "=== 📊 Results ==="
echo "---"
percentage=$((success_count * 100 / total_tests))

if [ $percentage -ge 90 ]; then
    color=$GREEN
    status="EXCELLENT"
elif [ $percentage -ge 70 ]; then
    color=$YELLOW
    status="GOOD"
else
    color=$RED
    status="NEEDS WORK"
fi

echo -e "Tests Passed: ${color}${success_count}/${total_tests} (${percentage}%)${NC}"
echo -e "Status: ${color}${status}${NC}"
echo ""

if [ $percentage -ge 90 ]; then
    echo "✨ Infrastructure is ready for MATL testing!"
    echo ""
    echo "🎯 Next Steps:"
    echo "1. Test on host system: Use NixOS host instead of containers"
    echo "2. Wait for Holochain 0.7+: Better container support expected"
    echo "3. Proceed with other phases: Infrastructure complete and reusable"
else
    echo "⚠️  Some infrastructure components need attention."
    echo "Review the failed tests above."
fi

echo ""
echo "📚 Documentation:"
echo "  - PHASE_4_STATUS_FINAL.md - Complete status and analysis"
echo "  - SESSION_SUMMARY_DEC31.md - Session achievements"
echo "  - test-multi-agent.sh - Multi-agent orchestration"
echo ""

# Cleanup
rm -rf /tmp/happ-validation

exit 0
