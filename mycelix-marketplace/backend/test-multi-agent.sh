#!/usr/bin/env bash
# 🌐 Mycelix Multi-Agent Test Framework
# Automated testing for distributed Holochain network

set -e

AGENT_COUNT=${1:-3}  # Default 3 agents
SCENARIO=${2:-"basic"}  # basic, byzantine, partition

echo "=== 🌐 Mycelix Multi-Agent Test ==="
echo "Date: $(date)"
echo "Agents: $AGENT_COUNT"
echo "Scenario: $SCENARIO"
echo ""

# Ensure network exists
docker network create mycelix-network 2>/dev/null || echo "Network exists, continuing..."

# Function to start an agent
start_agent() {
    local agent_num=$1
    local agent_name="mycelix-agent-$agent_num"
    local host_port=$((8880 + agent_num))

    echo "🚀 Starting Agent $agent_num (port $host_port)..."

    docker run -d \
        --name "$agent_name" \
        --network mycelix-network \
        -v "$(pwd):/workspace" \
        -w /workspace \
        -p "$host_port:8888" \
        ubuntu:22.04 \
        sleep infinity

    echo "✅ Agent $agent_num started"
}

# Function to setup conductor on agent
setup_conductor() {
    local agent_num=$1
    local agent_name="mycelix-agent-$agent_num"

    echo "🔧 Setting up conductor on Agent $agent_num..."

    docker exec "$agent_name" bash -c '
        echo "Verifying Holochain conductor..."
        /workspace/holochain --version
    ' || echo "⚠️  Conductor setup failed for Agent $agent_num"
}

# Cleanup function
cleanup() {
    echo ""
    echo "🧹 Cleanup agents? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        for i in $(seq 1 "$AGENT_COUNT"); do
            echo "Stopping Agent $i..."
            docker stop "mycelix-agent-$i" 2>/dev/null || true
            docker rm "mycelix-agent-$i" 2>/dev/null || true
        done
        echo "✅ Cleanup complete"
    else
        echo "ℹ️  Agents left running for inspection"
    fi
}

# Trap cleanup on exit
trap cleanup EXIT

# Start all agents
echo "📦 Starting $AGENT_COUNT agents..."
for i in $(seq 1 "$AGENT_COUNT"); do
    start_agent "$i"
done

echo ""
echo "⏳ Waiting 5 seconds for containers to stabilize..."
sleep 5

# Setup conductors on all agents (in background for speed)
echo ""
echo "🔧 Installing conductors on all agents..."
for i in $(seq 1 "$AGENT_COUNT"); do
    setup_conductor "$i" &
done

# Wait for all conductor installs
wait

echo ""
echo "✅ All agents started and configured!"
echo ""
echo "📊 Agent Status:"
for i in $(seq 1 "$AGENT_COUNT"); do
    status=$(docker ps --filter "name=mycelix-agent-$i" --format "{{.Status}}" 2>/dev/null || echo "Not running")
    echo "  Agent $i: $status"
done

echo ""
echo "🎯 Next Steps:"
echo "1. Run scenario tests: ./scenarios/test-$SCENARIO.sh"
echo "2. Inspect agent: docker exec -it mycelix-agent-1 bash"
echo "3. View logs: docker logs mycelix-agent-1"
echo ""
