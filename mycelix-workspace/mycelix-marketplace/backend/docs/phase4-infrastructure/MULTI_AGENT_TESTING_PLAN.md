# 🌐 Multi-Agent Testing Plan - Docker Container Network

**Date**: December 31, 2025
**Insight**: Docker containers = perfect multi-user/multi-agent test environment!
**Goal**: Validate MATL (45% Byzantine fault tolerance) with real distributed network

---

## 🎯 Why Multi-Agent Testing with Docker is Perfect

### The Challenge
- Holochain is a **distributed peer-to-peer** network
- MATL requires **multiple agents** to validate trust scores
- 45% Byzantine fault tolerance means we need **≥3 agents minimum** to test
- Need to simulate **good actors** and **bad actors**

### The Solution: Docker
Each Docker container = One Holochain agent:
- ✅ **Isolated** - Each container has own conductor + keys
- ✅ **Networked** - Containers can communicate via Docker network
- ✅ **Reproducible** - Same environment for every agent
- ✅ **Scalable** - Easy to spin up 3, 5, 10, or 100 agents
- ✅ **Realistic** - True peer-to-peer network simulation

---

## 🏗️ Multi-Agent Architecture

### Docker Network Setup
```
┌─────────────────────────────────────────────────┐
│         Docker Network: mycelix-network         │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Agent 1  │  │ Agent 2  │  │ Agent 3  │     │
│  │ (Good)   │  │ (Good)   │  │ (Bad)    │     │
│  │          │  │          │  │          │     │
│  │ Conductor│  │ Conductor│  │ Conductor│     │
│  │  :8888   │  │  :8888   │  │  :8888   │     │
│  └──────────┘  └──────────┘  └──────────┘     │
│       │              │              │          │
│       └──────────────┴──────────────┘          │
│              P2P Communication                 │
└─────────────────────────────────────────────────┘
```

### Key Components
1. **Docker Network**: `mycelix-network` (bridge network)
2. **Shared Volume**: Our flake.nix + hApp bundle
3. **Individual Conductors**: Each container runs own conductor
4. **Unique Agent Keys**: Each conductor generates unique agent ID
5. **Port Mapping**: External ports for monitoring/testing

---

## 🧪 Test Scenarios

### Scenario 1: Three-Agent Minimum (MATL Baseline)
**Purpose**: Validate basic MATL functionality

**Setup**:
- **Agent A**: Good actor (creates listings, honest transactions)
- **Agent B**: Good actor (creates listings, honest transactions)
- **Agent C**: Attempted bad actor (tries to spam/scam)

**Expected Behavior**:
- Agent C starts with 0% trust score
- Agent C creates spam listing
- Agent A and B vote to reject (2/3 = 66% > 45%)
- Agent C's action blocked by MATL
- System remains functional

**Success Criteria**:
- ✅ All 3 agents connect to DHT
- ✅ Good agents can create listings
- ✅ Bad agent blocked at 45% threshold
- ✅ No agent can spam without trust

### Scenario 2: Five-Agent Trust Building
**Purpose**: Validate trust score evolution

**Setup**:
- **5 good agents** performing normal operations
- Track trust score progression over time

**Expected Behavior**:
- All start at 0%
- Perform valid transactions
- Trust scores increase with good behavior
- Eventually all reach 100% (full trust)

**Success Criteria**:
- ✅ Trust scores start at 0%
- ✅ Scores increase with good transactions
- ✅ Scores reach 100% after sufficient activity
- ✅ High-trust agents have unrestricted access

### Scenario 3: Ten-Agent Byzantine Fault Tolerance
**Purpose**: Validate 45% threshold with larger network

**Setup**:
- **6 good agents** (60%)
- **4 bad agents** (40% - below threshold)

**Expected Behavior**:
- Bad agents coordinate attack
- Try to approve each other's spam
- Good agents outvote them (60% > 45%)
- System remains functional
- Bad agents cannot gain trust

**Success Criteria**:
- ✅ Network with 10 agents stays functional
- ✅ 40% bad actors cannot take over
- ✅ Good actors maintain network integrity
- ✅ MATL threshold prevents spam
- ✅ Performance acceptable with 10 agents

### Scenario 4: Network Partition & Recovery
**Purpose**: Test network resilience

**Setup**:
- 5 agents, network split into 2 partitions
- Partition 1: 3 agents
- Partition 2: 2 agents

**Expected Behavior**:
- Both partitions continue operating
- DHT maintains data availability
- When partitions merge, data reconciles
- No data loss or corruption

**Success Criteria**:
- ✅ Partitions operate independently
- ✅ Data available in both partitions
- ✅ Successful merge after partition heals
- ✅ No conflicts or data corruption

---

## 🛠️ Implementation

### Step 1: Create Docker Network
```bash
docker network create mycelix-network
```

### Step 2: Agent Container Template
```bash
# Start Agent 1
docker run --name mycelix-agent-1 \
  --network mycelix-network \
  -v $(pwd):/workspace \
  -w /workspace \
  -p 8881:8888 \
  -p 8891:8889 \
  nixos/nix:latest \
  bash -c 'nix develop --command bash -c "
    nix profile install github:holochain/holochain#holochain
    holochain sandbox create --network-seed agent-1 /workspace
  "'

# Start Agent 2
docker run --name mycelix-agent-2 \
  --network mycelix-network \
  -v $(pwd):/workspace \
  -w /workspace \
  -p 8882:8888 \
  -p 8892:8889 \
  nixos/nix:latest \
  bash -c 'nix develop --command bash -c "
    nix profile install github:holochain/holochain#holochain
    holochain sandbox create --network-seed agent-2 /workspace
  "'

# Start Agent 3
docker run --name mycelix-agent-3 \
  --network mycelix-network \
  -v $(pwd):/workspace \
  -w /workspace \
  -p 8883:8888 \
  -p 8893:8889 \
  nixos/nix:latest \
  bash -c 'nix develop --command bash -c "
    nix profile install github:holochain/holochain#holochain
    holochain sandbox create --network-seed agent-3 /workspace
  "'
```

### Step 3: Install hApp on All Agents
```bash
# For each agent:
docker exec -it mycelix-agent-1 bash -c "hc app install /workspace/mycelix_marketplace.happ"
docker exec -it mycelix-agent-2 bash -c "hc app install /workspace/mycelix_marketplace.happ"
docker exec -it mycelix-agent-3 bash -c "hc app install /workspace/mycelix_marketplace.happ"
```

### Step 4: Run Test Scenarios
```bash
# Execute test script that:
# 1. Agent 1 creates listing
# 2. Agent 2 creates listing
# 3. Agent 3 tries to spam
# 4. Monitor MATL blocking
```

---

## 📊 Metrics to Capture

### Network Metrics
- **Agent Count**: Total agents in network
- **Connection Status**: Which agents connected to which
- **DHT Coverage**: % of DHT held by each agent
- **Network Latency**: Time for actions to propagate

### MATL Metrics
- **Trust Scores**: Track over time for each agent
- **Blocked Actions**: Count of MATL rejections
- **Threshold Violations**: When 45% threshold triggers
- **False Positives**: Good actors incorrectly blocked
- **False Negatives**: Bad actors incorrectly allowed

### Performance Metrics
- **Action Latency**: Time from action to confirmation
- **DHT Sync Time**: Time for all agents to see new data
- **Resource Usage**: CPU/memory per agent
- **Throughput**: Transactions per second network-wide

---

## 🔧 Test Automation Script Structure

```bash
#!/usr/bin/env bash
# test-multi-agent.sh

set -e

AGENT_COUNT=${1:-3}  # Default 3 agents
SCENARIO=${2:-"basic"}  # basic, byzantine, partition

echo "=== Mycelix Multi-Agent Test ==="
echo "Agents: $AGENT_COUNT"
echo "Scenario: $SCENARIO"

# 1. Create network
docker network create mycelix-network 2>/dev/null || true

# 2. Start agents
for i in $(seq 1 $AGENT_COUNT); do
  echo "Starting Agent $i..."
  docker run -d --name mycelix-agent-$i \
    --network mycelix-network \
    -v $(pwd):/workspace \
    -w /workspace \
    -p $((8880+$i)):8888 \
    nixos/nix:latest \
    bash -c "sleep infinity"  # Keep running
done

# 3. Install conductor on each
for i in $(seq 1 $AGENT_COUNT); do
  echo "Installing conductor on Agent $i..."
  docker exec mycelix-agent-$i bash -c \
    'nix develop --command bash -c "nix profile install github:holochain/holochain#holochain"'
done

# 4. Install hApp on each
for i in $(seq 1 $AGENT_COUNT); do
  echo "Installing hApp on Agent $i..."
  docker exec mycelix-agent-$i bash -c \
    'nix develop --command bash -c "hc app install /workspace/mycelix_marketplace.happ"'
done

# 5. Run scenario-specific tests
case $SCENARIO in
  "basic")
    ./scenarios/test-basic-matl.sh
    ;;
  "byzantine")
    ./scenarios/test-byzantine.sh
    ;;
  "partition")
    ./scenarios/test-partition.sh
    ;;
esac

# 6. Cleanup
echo "Cleanup? (y/n)"
read -r cleanup
if [ "$cleanup" = "y" ]; then
  for i in $(seq 1 $AGENT_COUNT); do
    docker stop mycelix-agent-$i
    docker rm mycelix-agent-$i
  done
  docker network rm mycelix-network
fi
```

---

## 📋 Test Scenarios Implementation

### Scenario Scripts

#### scenarios/test-basic-matl.sh
```bash
#!/usr/bin/env bash
# Basic MATL test with 3 agents

echo "=== Testing Basic MATL (3 agents) ==="

# Agent 1: Create valid listing
echo "Agent 1: Creating valid listing..."
docker exec mycelix-agent-1 bash -c \
  'nix develop --command bash -c "hc call listings create_listing \"{\"title\":\"Laptop\",\"price\":500}\""'

# Agent 2: Create valid listing
echo "Agent 2: Creating valid listing..."
docker exec mycelix-agent-2 bash -c \
  'nix develop --command bash -c "hc call listings create_listing \"{\"title\":\"Phone\",\"price\":300}\""'

# Agent 3: Try to spam (should be blocked by MATL)
echo "Agent 3: Attempting spam (should be blocked)..."
docker exec mycelix-agent-3 bash -c \
  'nix develop --command bash -c "hc call listings create_listing \"{\"title\":\"SPAM\",\"price\":1}\""' \
  || echo "✅ MATL blocked spam as expected!"

# Check trust scores
echo "Checking trust scores..."
for i in 1 2 3; do
  echo "Agent $i trust score:"
  docker exec mycelix-agent-$i bash -c \
    'nix develop --command bash -c "hc call reputation get_trust_score"'
done
```

---

## 🎯 Success Criteria Summary

### Minimum Requirements (Phase 4 Completion)
- [x] Build environment working
- [x] Docker + Nix integration proven
- [ ] **Conductor installed and working**
- [ ] **Single agent can run hApp**
- [ ] **3 agents can communicate**
- [ ] **MATL blocks untrusted agent**

### Ideal Goals (Phase 5 Quality)
- [ ] 10-agent network stable
- [ ] Byzantine fault tolerance proven
- [ ] Network partition recovery works
- [ ] Performance benchmarks met
- [ ] Automated test suite runs

### Stretch Goals (Production Ready)
- [ ] 100-agent network simulation
- [ ] Continuous testing in CI
- [ ] Performance under load
- [ ] Security audit passed

---

## 🚀 Next Steps

### Immediate (Today)
1. ⏳ **Wait for conductor installation** (in progress)
2. ✅ **Verify single conductor works**
3. 🎯 **Start 3-agent test**

### Short Term (This Week)
4. 🧪 **Run all 4 test scenarios**
5. 📊 **Capture metrics**
6. 📝 **Document results**

### Medium Term (This Month)
7. 🤖 **Automate test suite**
8. 📈 **Performance tuning**
9. 🔐 **Security validation**

---

## 💡 Why This Approach Wins

**Docker Benefits**:
- ✅ True multi-agent simulation
- ✅ Reproducible test environment
- ✅ Easy to scale (3 to 100 agents)
- ✅ Network isolation
- ✅ CI/CD integration ready

**MATL Validation**:
- ✅ Real distributed testing
- ✅ Byzantine scenarios possible
- ✅ Trust score evolution visible
- ✅ Network resilience proven
- ✅ Production-like conditions

**Development Velocity**:
- ✅ Fast iteration cycles
- ✅ Automated testing
- ✅ Clear pass/fail criteria
- ✅ Early bug detection
- ✅ Confidence in production readiness

---

**Status**: 📝 Plan complete, ready for implementation
**Dependencies**: Conductor installation (in progress)
**Confidence**: 🚀 Very High - Docker makes this possible!

🌊 **Multi-agent testing unlocks true distributed validation!** 🌊
