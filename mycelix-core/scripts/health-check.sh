#!/usr/bin/env bash
#
# Mycelix Health Check Script
#
# Checks the health of a running Mycelix deployment.
# Can be run manually or integrated with monitoring systems.
#
# Usage: ./scripts/health-check.sh [--json] [--verbose]
#
# Exit codes:
#   0 - All checks passed
#   1 - Critical failure
#   2 - Warnings present
#

set -euo pipefail

# Configuration
VALIDATOR_URL="${VALIDATOR_URL:-http://localhost:9090}"
BOOTSTRAP_URL="${BOOTSTRAP_URL:-http://localhost:8080}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
REDIS_HOST="${REDIS_HOST:-localhost}"

# Options
JSON_OUTPUT=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --json)
            JSON_OUTPUT=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Mycelix Health Check"
            echo ""
            echo "Usage: $0 [--json] [--verbose]"
            echo ""
            echo "Options:"
            echo "  --json     Output results as JSON"
            echo "  --verbose  Show detailed output"
            echo ""
            echo "Environment variables:"
            echo "  VALIDATOR_URL   Validator metrics endpoint (default: http://localhost:9090)"
            echo "  BOOTSTRAP_URL   Bootstrap server URL (default: http://localhost:8080)"
            echo "  POSTGRES_HOST   PostgreSQL host (default: localhost)"
            echo "  REDIS_HOST      Redis host (default: localhost)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors (disabled for JSON output)
if $JSON_OUTPUT; then
    RED='' GREEN='' YELLOW='' BLUE='' NC=''
else
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
fi

# Results storage
declare -A RESULTS
CRITICAL=0
WARNINGS=0
PASSED=0

# Check function
check() {
    local name="$1"
    local status="$2"
    local message="$3"
    local details="${4:-}"

    RESULTS["$name"]="$status|$message|$details"

    if ! $JSON_OUTPUT; then
        case $status in
            pass)
                echo -e "${GREEN}[PASS]${NC} $name: $message"
                ((++PASSED))
                ;;
            warn)
                echo -e "${YELLOW}[WARN]${NC} $name: $message"
                ((++WARNINGS))
                ;;
            fail)
                echo -e "${RED}[FAIL]${NC} $name: $message"
                ((++CRITICAL))
                ;;
        esac
        if $VERBOSE && [[ -n "$details" ]]; then
            echo "       $details"
        fi
    else
        case $status in
            pass) ((++PASSED)) ;;
            warn) ((++WARNINGS)) ;;
            fail) ((++CRITICAL)) ;;
        esac
    fi
}

# Header
if ! $JSON_OUTPUT; then
    echo ""
    echo "=============================================="
    echo "  Mycelix Health Check"
    echo "  $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "=============================================="
    echo ""
fi

# =============================================================================
# Validator Health
# =============================================================================
if ! $JSON_OUTPUT; then
    echo "--- Validator ---"
fi

# Check validator metrics endpoint
if curl -sf "${VALIDATOR_URL}/metrics" >/dev/null 2>&1; then
    check "validator_endpoint" "pass" "Metrics endpoint responding"

    # Parse specific metrics
    METRICS=$(curl -sf "${VALIDATOR_URL}/metrics" 2>/dev/null || echo "")

    # Check peer count
    PEERS=$(echo "$METRICS" | grep -E "^mycelix_peers_connected " | awk '{print $2}' | head -1 || echo "0")
    if [[ -n "$PEERS" ]] && (( $(echo "$PEERS >= 3" | bc -l 2>/dev/null || echo 0) )); then
        check "peer_count" "pass" "$PEERS peers connected"
    elif [[ -n "$PEERS" ]] && (( $(echo "$PEERS > 0" | bc -l 2>/dev/null || echo 0) )); then
        check "peer_count" "warn" "Only $PEERS peers (minimum 3 recommended)"
    else
        check "peer_count" "warn" "No peer count metric found"
    fi

    # Check trust score
    TRUST=$(echo "$METRICS" | grep -E "^mycelix_trust_score " | awk '{print $2}' | head -1 || echo "")
    if [[ -n "$TRUST" ]] && (( $(echo "$TRUST >= 0.5" | bc -l 2>/dev/null || echo 0) )); then
        check "trust_score" "pass" "Trust score: $TRUST"
    elif [[ -n "$TRUST" ]]; then
        check "trust_score" "warn" "Low trust score: $TRUST"
    fi

    # Check FL rounds
    FL_ROUNDS=$(echo "$METRICS" | grep -E "^mycelix_fl_rounds_completed_total " | awk '{print $2}' | head -1 || echo "")
    if [[ -n "$FL_ROUNDS" ]]; then
        check "fl_rounds" "pass" "$FL_ROUNDS FL rounds completed"
    fi

else
    check "validator_endpoint" "fail" "Cannot reach ${VALIDATOR_URL}/metrics"
fi

if ! $JSON_OUTPUT; then
    echo ""
fi

# =============================================================================
# Bootstrap Server Health
# =============================================================================
if ! $JSON_OUTPUT; then
    echo "--- Bootstrap Server ---"
fi

if curl -sf "${BOOTSTRAP_URL}/health" >/dev/null 2>&1; then
    check "bootstrap_health" "pass" "Bootstrap server healthy"

    # Check node count from bootstrap
    NODE_COUNT=$(curl -sf "${BOOTSTRAP_URL}/api/nodes/count" 2>/dev/null | grep -oE '[0-9]+' | head -1 || echo "")
    if [[ -n "$NODE_COUNT" ]]; then
        if (( NODE_COUNT >= 3 )); then
            check "node_count" "pass" "$NODE_COUNT nodes in network"
        else
            check "node_count" "warn" "Only $NODE_COUNT nodes (minimum 3 for consensus)"
        fi
    fi
else
    check "bootstrap_health" "warn" "Cannot reach bootstrap server"
fi

if ! $JSON_OUTPUT; then
    echo ""
fi

# =============================================================================
# Database Health
# =============================================================================
if ! $JSON_OUTPUT; then
    echo "--- Database ---"
fi

# PostgreSQL
if command -v pg_isready &>/dev/null; then
    if pg_isready -h "$POSTGRES_HOST" -q 2>/dev/null; then
        check "postgres" "pass" "PostgreSQL is ready"
    else
        check "postgres" "fail" "PostgreSQL is not ready"
    fi
elif nc -z "$POSTGRES_HOST" 5432 2>/dev/null; then
    check "postgres" "pass" "PostgreSQL port open"
else
    check "postgres" "warn" "Cannot check PostgreSQL (pg_isready not available)"
fi

# Redis
if command -v redis-cli &>/dev/null; then
    if redis-cli -h "$REDIS_HOST" ping 2>/dev/null | grep -q PONG; then
        check "redis" "pass" "Redis responding"
    else
        check "redis" "fail" "Redis not responding"
    fi
elif nc -z "$REDIS_HOST" 6379 2>/dev/null; then
    check "redis" "pass" "Redis port open"
else
    check "redis" "warn" "Cannot check Redis (redis-cli not available)"
fi

if ! $JSON_OUTPUT; then
    echo ""
fi

# =============================================================================
# Docker Health
# =============================================================================
if ! $JSON_OUTPUT; then
    echo "--- Docker Containers ---"
fi

if command -v docker &>/dev/null && docker info &>/dev/null; then
    # Check for mycelix containers
    CONTAINERS=$(docker ps --filter "name=mycelix" --format "{{.Names}}:{{.Status}}" 2>/dev/null || echo "")

    if [[ -n "$CONTAINERS" ]]; then
        while IFS= read -r container; do
            NAME=$(echo "$container" | cut -d: -f1)
            STATUS=$(echo "$container" | cut -d: -f2-)

            if echo "$STATUS" | grep -q "Up"; then
                if echo "$STATUS" | grep -q "unhealthy"; then
                    check "container_$NAME" "warn" "$NAME is unhealthy"
                else
                    check "container_$NAME" "pass" "$NAME is running"
                fi
            else
                check "container_$NAME" "fail" "$NAME is not running"
            fi
        done <<< "$CONTAINERS"
    else
        check "containers" "warn" "No mycelix containers found"
    fi
else
    check "docker" "warn" "Docker not available"
fi

if ! $JSON_OUTPUT; then
    echo ""
fi

# =============================================================================
# System Resources
# =============================================================================
if ! $JSON_OUTPUT; then
    echo "--- System Resources ---"
fi

# Disk space
DISK_USAGE=$(df -h / | tail -1 | awk '{print $5}' | tr -d '%')
if (( DISK_USAGE < 80 )); then
    check "disk_space" "pass" "${DISK_USAGE}% used"
elif (( DISK_USAGE < 90 )); then
    check "disk_space" "warn" "${DISK_USAGE}% used (consider cleanup)"
else
    check "disk_space" "fail" "${DISK_USAGE}% used (critical!)"
fi

# Memory
if [[ -f /proc/meminfo ]]; then
    MEM_TOTAL=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    MEM_AVAIL=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    MEM_USED_PCT=$(( (MEM_TOTAL - MEM_AVAIL) * 100 / MEM_TOTAL ))

    if (( MEM_USED_PCT < 80 )); then
        check "memory" "pass" "${MEM_USED_PCT}% used"
    elif (( MEM_USED_PCT < 90 )); then
        check "memory" "warn" "${MEM_USED_PCT}% used"
    else
        check "memory" "fail" "${MEM_USED_PCT}% used (critical!)"
    fi
fi

# Load average
LOAD=$(cat /proc/loadavg 2>/dev/null | awk '{print $1}' || echo "0")
CORES=$(nproc 2>/dev/null || echo "1")
LOAD_PER_CORE=$(echo "scale=2; $LOAD / $CORES" | bc 2>/dev/null || echo "0")

if (( $(echo "$LOAD_PER_CORE < 0.7" | bc -l 2>/dev/null || echo 1) )); then
    check "load" "pass" "Load average: $LOAD ($LOAD_PER_CORE per core)"
elif (( $(echo "$LOAD_PER_CORE < 1.0" | bc -l 2>/dev/null || echo 1) )); then
    check "load" "warn" "Load average: $LOAD ($LOAD_PER_CORE per core)"
else
    check "load" "fail" "High load: $LOAD ($LOAD_PER_CORE per core)"
fi

if ! $JSON_OUTPUT; then
    echo ""
fi

# =============================================================================
# Summary
# =============================================================================
if $JSON_OUTPUT; then
    # Output JSON
    echo "{"
    echo "  \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\","
    echo "  \"summary\": {"
    echo "    \"passed\": $PASSED,"
    echo "    \"warnings\": $WARNINGS,"
    echo "    \"critical\": $CRITICAL"
    echo "  },"
    echo "  \"status\": \"$(if (( CRITICAL > 0 )); then echo "critical"; elif (( WARNINGS > 0 )); then echo "warning"; else echo "healthy"; fi)\","
    echo "  \"checks\": {"

    first=true
    for key in "${!RESULTS[@]}"; do
        IFS='|' read -r status message details <<< "${RESULTS[$key]}"
        if $first; then
            first=false
        else
            echo ","
        fi
        echo -n "    \"$key\": {\"status\": \"$status\", \"message\": \"$message\""
        if [[ -n "$details" ]]; then
            echo -n ", \"details\": \"$details\""
        fi
        echo -n "}"
    done
    echo ""
    echo "  }"
    echo "}"
else
    echo "=============================================="
    echo "  Summary"
    echo "=============================================="
    echo ""
    echo -e "  ${GREEN}Passed:${NC}   $PASSED"
    echo -e "  ${YELLOW}Warnings:${NC} $WARNINGS"
    echo -e "  ${RED}Critical:${NC} $CRITICAL"
    echo ""

    if (( CRITICAL > 0 )); then
        echo -e "${RED}Status: CRITICAL${NC}"
        exit 1
    elif (( WARNINGS > 0 )); then
        echo -e "${YELLOW}Status: WARNING${NC}"
        exit 2
    else
        echo -e "${GREEN}Status: HEALTHY${NC}"
        exit 0
    fi
fi
