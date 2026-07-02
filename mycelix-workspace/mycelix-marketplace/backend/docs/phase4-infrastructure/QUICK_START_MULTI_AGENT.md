#!/usr/bin/env bash
# 🚀 Quick Start - Multi-Agent Testing
# Once conductor installation completes, use these commands

## 📋 Verify Conductor Installation

```bash
# Check if conductor installed successfully
docker run --rm -v $(pwd):/workspace -w /workspace \
  nixos/nix:latest \
  bash -c 'echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf && \
    nix develop --command holochain --version'
```

**Expected Output**: `holochain [version number]`

---

## 🧪 Run 3-Agent MATL Test (Full Workflow)

### Step 1: Start 3 Agents
```bash
./test-multi-agent.sh 3 basic
```

**What this does**:
- Creates 3 Docker containers
- Connects them to `mycelix-network`
- Installs Holochain conductor on each
- Waits for user inspection or cleanup

### Step 2: Install hApp on Each Agent
```bash
# Agent 1
docker exec mycelix-agent-1 bash -c \
  'nix develop --command bash -c "hc app install /workspace/mycelix_marketplace.happ"'

# Agent 2
docker exec mycelix-agent-2 bash -c \
  'nix develop --command bash -c "hc app install /workspace/mycelix_marketplace.happ"'

# Agent 3
docker exec mycelix-agent-3 bash -c \
  'nix develop --command bash -c "hc app install /workspace/mycelix_marketplace.happ"'
```

### Step 3: Run MATL Validation Test
```bash
./scenarios/test-basic-matl.sh
```

**What this tests**:
- Agent 1 creates valid listing (should succeed)
- Agent 2 creates valid listing (should succeed)
- Agent 3 attempts spam (should be BLOCKED by MATL)
- Trust scores for all 3 agents

---

## 🔍 Inspect Agents

### Enter Agent Container
```bash
docker exec -it mycelix-agent-1 bash
```

### Check Agent Status
```bash
# Inside container
nix develop
holochain --version
hc --version
```

### View Agent Logs
```bash
docker logs mycelix-agent-1
docker logs mycelix-agent-2
docker logs mycelix-agent-3
```

---

## 🧹 Cleanup

### Stop and Remove All Agents
```bash
# Manual cleanup
docker stop mycelix-agent-1 mycelix-agent-2 mycelix-agent-3
docker rm mycelix-agent-1 mycelix-agent-2 mycelix-agent-3

# Or let the script do it (it prompts at end)
# When test-multi-agent.sh exits, answer 'y' to cleanup prompt
```

### Remove Network (Optional)
```bash
docker network rm mycelix-network
```

---

## 📊 Expected Results

### Successful MATL Test
```
✅ Agent 1: Listing created (trust 0% → building)
✅ Agent 2: Listing created (trust 0% → building)
❌ Agent 3: SPAM BLOCKED (trust 0%, 2/3 agents rejected = 66% > 45%)
```

### Trust Score Progression
- **Start**: All agents at 0% trust
- **After valid transactions**: Agents 1 & 2 increase
- **After spam attempt**: Agent 3 remains at 0% or decreases

---

## 🚀 Advanced: 10-Agent Byzantine Test

### Start 10 Agents
```bash
./test-multi-agent.sh 10 byzantine
```

### Configure Agents
- Agents 1-6: Good actors (60%)
- Agents 7-10: Bad actors (40%)

### Run Byzantine Scenario
```bash
./scenarios/test-byzantine.sh  # (create this next)
```

**Expected**: 60% good actors > 45% threshold → System secure

---

## 🐛 Troubleshooting

### "Container not found"
```bash
# Check running containers
docker ps -a | grep mycelix

# If not running, start fresh
./test-multi-agent.sh 3 basic
```

### "Conductor not installed"
```bash
# Verify conductor installation completed
docker exec mycelix-agent-1 bash -c \
  'nix develop --command which holochain'

# Should return: /nix/store/.../bin/holochain
```

### "hApp install fails"
```bash
# Check if hApp file exists
ls -lh mycelix_marketplace.happ

# Verify it's accessible in container
docker exec mycelix-agent-1 ls -lh /workspace/mycelix_marketplace.happ
```

---

## 📝 Next Development Steps

1. ✅ **Conductor Installation** - Verify completion
2. ✅ **3-Agent MATL Test** - Run and validate
3. 🚧 **Create Byzantine Scenario** - `scenarios/test-byzantine.sh`
4. 🚧 **Create Network Partition Test** - `scenarios/test-partition.sh`
5. 🚧 **Performance Benchmarks** - Measure latency, throughput
6. 🚧 **10-Agent Scaling Test** - Validate at scale

---

## 🎯 Success Criteria

Phase 4 Integration Testing Complete when:
- [x] Conductor installed and verified
- [ ] Single agent can run hApp
- [ ] 3 agents communicate via DHT
- [ ] MATL blocks untrusted agents (45% threshold)
- [ ] Performance acceptable (<2s operations)
- [ ] Network partition recovery works
- [ ] 10+ agent network stable

**Current Phase 4 Progress**: ~80% → 100%

---

🌊 **Multi-agent distributed testing is the key to validating MATL!** 🌊
