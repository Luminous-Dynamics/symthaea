#!/usr/bin/env bash
# Mycelix Testnet - Monitoring Script
# This script provides health checks and monitoring for the testnet
# Usage: ./monitor-testnet.sh [command] [options]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default configuration
BOOTSTRAP_API="${BOOTSTRAP_API:-http://localhost:9001}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9091}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
REFRESH_INTERVAL="${REFRESH_INTERVAL:-5}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Show help
show_help() {
    echo ""
    echo "Mycelix Testnet Monitor"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  status        Show overall network status (default)"
    echo "  nodes         List all connected nodes"
    echo "  health        Run health checks on all services"
    echo "  metrics       Show key metrics"
    echo "  fl            Show federated learning status"
    echo "  logs          Show recent logs"
    echo "  watch         Continuous monitoring dashboard"
    echo "  alerts        Show active alerts"
    echo "  help          Show this help message"
    echo ""
    echo "Options:"
    echo "  --bootstrap URL    Bootstrap API URL (default: http://localhost:9001)"
    echo "  --prometheus URL   Prometheus URL (default: http://localhost:9091)"
    echo "  --interval N       Refresh interval in seconds (default: 5)"
    echo ""
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 watch --interval 10"
    echo "  $0 nodes --bootstrap https://testnet.mycelix.network:9001"
}

# Parse arguments
parse_args() {
    COMMAND="${1:-status}"
    shift || true

    while [[ $# -gt 0 ]]; do
        case $1 in
            --bootstrap)
                BOOTSTRAP_API="$2"
                shift 2
                ;;
            --prometheus)
                PROMETHEUS_URL="$2"
                shift 2
                ;;
            --interval)
                REFRESH_INTERVAL="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                shift
                ;;
        esac
    done
}

# Clear screen and show header
show_header() {
    clear
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║            Mycelix Testnet Monitor Dashboard                  ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "  ${BLUE}Time:${NC} $(date '+%Y-%m-%d %H:%M:%S')    ${BLUE}Refresh:${NC} ${REFRESH_INTERVAL}s"
    echo ""
}

# Check service health
check_service() {
    local name=$1
    local url=$2
    local endpoint=${3:-/health}

    if curl -sf "${url}${endpoint}" > /dev/null 2>&1; then
        log_success "$name"
        return 0
    else
        log_error "$name"
        return 1
    fi
}

# Get metric value from Prometheus
get_metric() {
    local query=$1
    local result

    result=$(curl -sf "${PROMETHEUS_URL}/api/v1/query?query=${query}" 2>/dev/null | \
        jq -r '.data.result[0].value[1] // "N/A"' 2>/dev/null) || echo "N/A"

    echo "$result"
}

# Show network status
show_status() {
    echo -e "${MAGENTA}=== Network Status ===${NC}"
    echo ""

    # Check bootstrap
    echo -n "  Bootstrap Node: "
    if curl -sf "${BOOTSTRAP_API}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}Online${NC}"

        # Get network info
        local network_info
        network_info=$(curl -sf "${BOOTSTRAP_API}/network/info" 2>/dev/null || echo "{}")

        if [ "$network_info" != "{}" ]; then
            local chain_id block_height peer_count
            chain_id=$(echo "$network_info" | jq -r '.chain_id // "N/A"')
            block_height=$(echo "$network_info" | jq -r '.block_height // "N/A"')
            peer_count=$(echo "$network_info" | jq -r '.peer_count // "N/A"')

            echo "  Chain ID:       $chain_id"
            echo "  Block Height:   $block_height"
            echo "  Connected Peers: $peer_count"
        fi
    else
        echo -e "${RED}Offline${NC}"
    fi

    echo ""
}

# Show all nodes
show_nodes() {
    echo -e "${MAGENTA}=== Connected Nodes ===${NC}"
    echo ""

    local nodes
    nodes=$(curl -sf "${BOOTSTRAP_API}/nodes" 2>/dev/null || echo "[]")

    if [ "$nodes" = "[]" ]; then
        log_warning "No nodes found or bootstrap unavailable"
        return
    fi

    echo "$nodes" | jq -r '.[] | "  \(.node_id)\t\(.node_type)\t\(.status)\t\(.last_seen)"' 2>/dev/null || \
        log_warning "Could not parse node list"

    echo ""
    local count
    count=$(echo "$nodes" | jq 'length' 2>/dev/null || echo "0")
    echo "  Total nodes: $count"
    echo ""
}

# Run health checks
run_health_checks() {
    echo -e "${MAGENTA}=== Health Checks ===${NC}"
    echo ""

    local failed=0

    echo "  Core Services:"
    check_service "    Bootstrap API" "$BOOTSTRAP_API" "/health" || ((failed++))

    # Check if running in Docker
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "mycelix"; then
        echo ""
        echo "  Docker Containers:"

        for container in mycelix-bootstrap mycelix-postgres mycelix-redis mycelix-prometheus mycelix-grafana; do
            if docker ps --format '{{.Names}}' | grep -q "$container"; then
                local status
                status=$(docker inspect --format '{{.State.Health.Status}}' "$container" 2>/dev/null || echo "unknown")
                if [ "$status" = "healthy" ]; then
                    log_success "    $container"
                elif [ "$status" = "unknown" ]; then
                    log_warning "    $container (no healthcheck)"
                else
                    log_error "    $container ($status)"
                    ((failed++))
                fi
            fi
        done
    fi

    echo ""
    echo "  Monitoring:"
    check_service "    Prometheus" "$PROMETHEUS_URL" "/-/healthy" || ((failed++))
    check_service "    Grafana" "$GRAFANA_URL" "/api/health" || ((failed++))

    echo ""
    if [ $failed -eq 0 ]; then
        log_success "All health checks passed!"
    else
        log_error "$failed health check(s) failed"
    fi
    echo ""

    return $failed
}

# Show key metrics
show_metrics() {
    echo -e "${MAGENTA}=== Key Metrics ===${NC}"
    echo ""

    # Network metrics
    echo "  Network:"
    echo "    Active Nodes:       $(get_metric 'count(up{job=~"mycelix-.*"}==1)')"
    echo "    Block Height:       $(get_metric 'max(mycelix_blocks_produced_total)')"
    echo "    Avg Block Time:     $(get_metric 'avg(mycelix_block_time_seconds)') s"
    echo "    Peer Connections:   $(get_metric 'sum(mycelix_peers_connected)')"

    echo ""
    echo "  Federated Learning:"
    echo "    FL Rounds:          $(get_metric 'max(mycelix_fl_rounds_completed_total)')"
    echo "    Active Participants: $(get_metric 'sum(mycelix_fl_active_participants)')"
    echo "    Gradients/min:      $(get_metric 'sum(rate(mycelix_fl_gradients_submitted_total[5m]))*60')"

    echo ""
    echo "  Resources:"
    echo "    CPU Usage:          $(get_metric 'avg(rate(container_cpu_usage_seconds_total{name=~"mycelix.*"}[5m]))*100')%"
    echo "    Memory Usage:       $(get_metric 'avg(container_memory_usage_bytes{name=~"mycelix.*"})/1024/1024') MB"

    echo ""
}

# Show FL status
show_fl_status() {
    echo -e "${MAGENTA}=== Federated Learning Status ===${NC}"
    echo ""

    local fl_status
    fl_status=$(curl -sf "${BOOTSTRAP_API}/fl/status" 2>/dev/null || echo "{}")

    if [ "$fl_status" = "{}" ]; then
        log_warning "Could not fetch FL status"
        return
    fi

    echo "$fl_status" | jq -r '
        "  Current Round:     \(.current_round // "N/A")",
        "  Round Status:      \(.round_status // "N/A")",
        "  Participants:      \(.participants_count // 0)/\(.min_participants // 3)",
        "  Gradients Received: \(.gradients_received // 0)",
        "  Model Version:     \(.model_version // "N/A")",
        "  Aggregation:       \(.aggregation_method // "fedavg")"
    ' 2>/dev/null || log_warning "Could not parse FL status"

    echo ""

    # Show recent rounds
    echo "  Recent Rounds:"
    local rounds
    rounds=$(curl -sf "${BOOTSTRAP_API}/fl/rounds?limit=5" 2>/dev/null || echo "[]")

    if [ "$rounds" != "[]" ]; then
        echo "$rounds" | jq -r '.[] | "    Round \(.round_number): \(.status) (\(.participants_count) participants)"' 2>/dev/null
    else
        echo "    No rounds yet"
    fi

    echo ""
}

# Show recent logs
show_logs() {
    echo -e "${MAGENTA}=== Recent Logs ===${NC}"
    echo ""

    local container="${1:-mycelix-bootstrap}"
    local lines="${2:-20}"

    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "$container"; then
        docker logs --tail "$lines" "$container" 2>&1
    else
        log_warning "Container $container not found"
        echo ""
        echo "Available containers:"
        docker ps --format '  {{.Names}}' 2>/dev/null | grep mycelix || echo "  None"
    fi

    echo ""
}

# Show active alerts
show_alerts() {
    echo -e "${MAGENTA}=== Active Alerts ===${NC}"
    echo ""

    local alerts
    alerts=$(curl -sf "${PROMETHEUS_URL}/api/v1/alerts" 2>/dev/null | jq '.data.alerts' 2>/dev/null || echo "[]")

    if [ "$alerts" = "[]" ] || [ -z "$alerts" ]; then
        log_success "No active alerts"
        echo ""
        return
    fi

    echo "$alerts" | jq -r '.[] | select(.state == "firing") |
        "  [\(.labels.severity | ascii_upcase)] \(.labels.alertname)",
        "    \(.annotations.summary)",
        ""' 2>/dev/null

    local count
    count=$(echo "$alerts" | jq '[.[] | select(.state == "firing")] | length' 2>/dev/null || echo "0")

    if [ "$count" -gt 0 ]; then
        log_warning "$count active alert(s)"
    fi
    echo ""
}

# Continuous monitoring dashboard
watch_dashboard() {
    while true; do
        show_header
        show_status
        show_metrics

        echo -e "${BLUE}Press Ctrl+C to exit${NC}"
        sleep "$REFRESH_INTERVAL"
    done
}

# Main function
main() {
    parse_args "$@"

    case "$COMMAND" in
        status)
            show_status
            show_metrics
            ;;
        nodes)
            show_nodes
            ;;
        health)
            run_health_checks
            ;;
        metrics)
            show_metrics
            ;;
        fl)
            show_fl_status
            ;;
        logs)
            show_logs "${2:-mycelix-bootstrap}" "${3:-50}"
            ;;
        alerts)
            show_alerts
            ;;
        watch)
            watch_dashboard
            ;;
        help)
            show_help
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
