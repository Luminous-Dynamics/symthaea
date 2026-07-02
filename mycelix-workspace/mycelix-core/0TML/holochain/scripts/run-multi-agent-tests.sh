#!/usr/bin/env bash
#
# Multi-Agent Conductor Test Runner
# Byzantine-Robust Federated Learning System
#
# This script runs the multi-agent conductor tests with various configurations.
#
# Usage: ./scripts/run-multi-agent-tests.sh [OPTIONS]
#
# Options:
#   --agents N       Number of agents (default: 3)
#   --byzantine P    Byzantine percentage (default: 30)
#   --rounds N       Number of FL rounds (default: 5)
#   --conductor      Run with real conductors (requires holonix)
#   --simulation     Run simulation tests only (default)
#   --verbose        Verbose output
#
# Author: Luminous Dynamics Research Team
# Date: December 30, 2025

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOLOCHAIN_DIR="$(dirname "$SCRIPT_DIR")"
TESTS_DIR="$HOLOCHAIN_DIR/tests"

# Default values
NUM_AGENTS=3
BYZANTINE_PCT=30
NUM_ROUNDS=5
RUN_CONDUCTOR=false
VERBOSE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        --byzantine)
            BYZANTINE_PCT="$2"
            shift 2
            ;;
        --rounds)
            NUM_ROUNDS="$2"
            shift 2
            ;;
        --conductor)
            RUN_CONDUCTOR=true
            shift
            ;;
        --simulation)
            RUN_CONDUCTOR=false
            shift
            ;;
        --verbose|-v)
            VERBOSE="-v"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     Byzantine-Robust FL: Multi-Agent Test Runner                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Agents: $NUM_AGENTS"
echo "  Byzantine: $BYZANTINE_PCT%"
echo "  Rounds: $NUM_ROUNDS"
echo "  Mode: $([ "$RUN_CONDUCTOR" = true ] && echo "Real Conductor" || echo "Simulation")"
echo ""

# Set environment variables
export NUM_AGENTS
export BYZANTINE_PCT
export NUM_ROUNDS

cd "$HOLOCHAIN_DIR"

# Run simulation tests
echo "=== Running Simulation Tests ==="
echo ""

# Core Byzantine detection tests
echo "--- Byzantine Detection Tests ---"
pytest "$TESTS_DIR/test_multi_agent_conductor.py" \
    -k "TestMultiAgentByzantine" \
    $VERBOSE \
    --tb=short

# Reputation dynamics tests
echo ""
echo "--- Reputation Dynamics Tests ---"
pytest "$TESTS_DIR/test_multi_agent_conductor.py" \
    -k "TestReputationDynamics" \
    $VERBOSE \
    --tb=short

# 45% BFT tolerance tests
echo ""
echo "--- 45% BFT Tolerance Tests ---"
pytest "$TESTS_DIR/test_multi_agent_conductor.py" \
    -k "Test45PercentBFT" \
    $VERBOSE \
    --tb=short

# Cross-agent communication tests
echo ""
echo "--- Cross-Agent Communication Tests ---"
pytest "$TESTS_DIR/test_multi_agent_conductor.py" \
    -k "TestCrossAgentCommunication" \
    $VERBOSE \
    --tb=short

# Performance tests
echo ""
echo "--- Performance Tests ---"
pytest "$TESTS_DIR/test_multi_agent_conductor.py" \
    -k "TestPerformanceMetrics" \
    $VERBOSE \
    --tb=short

# DHT consistency tests
echo ""
echo "--- DHT Consistency Tests ---"
pytest "$TESTS_DIR/test_multi_agent_conductor.py" \
    -k "TestDHTConsistency" \
    $VERBOSE \
    --tb=short

# Run all tests together for summary
echo ""
echo "=== Full Test Suite Summary ==="
pytest "$TESTS_DIR/test_multi_agent_conductor.py" \
    --ignore="$TESTS_DIR/test_multi_agent_conductor.py::TestRealConductor" \
    $VERBOSE \
    --tb=short \
    2>&1 | tail -20

# Conductor tests (if requested)
if [ "$RUN_CONDUCTOR" = true ]; then
    echo ""
    echo "=== Real Conductor Tests ==="

    # Check for holochain
    if ! command -v holochain &> /dev/null; then
        echo "❌ holochain not found. Enter holonix first:"
        echo "   nix develop .#holonix"
        exit 1
    fi

    echo "✅ Holochain found: $(which holochain)"

    # Check for hApp
    HAPP_PATH="$HOLOCHAIN_DIR/happ/byzantine_defense/byzantine_defense.happ"
    if [ ! -f "$HAPP_PATH" ]; then
        echo "❌ hApp not found. Run pack first:"
        echo "   hc app pack $HOLOCHAIN_DIR/happ/byzantine_defense"
        exit 1
    fi

    echo "✅ hApp found: $HAPP_PATH"

    # Run conductor tests (would normally use pytest with asyncio)
    echo ""
    echo "Conductor integration tests require manual setup."
    echo "See CONDUCTOR_TESTING.md for instructions."
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    TEST RUN COMPLETE                              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
