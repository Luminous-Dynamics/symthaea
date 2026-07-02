# Multi-Node P2P Testing with Docker

**This demonstrates TRUE decentralization - 3 independent Holochain conductors running federated learning via P2P.**

## 🏗️ Architecture

```
┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
│   Hospital A (Boston)    │    │   Hospital B (London)    │    │   Hospital C (Tokyo)     │
├──────────────────────────┤    ├──────────────────────────┤    ├──────────────────────────┤
│  Zero-TrustML Node            │    │  Zero-TrustML Node            │    │  Zero-TrustML Node            │
│  - Private training data │    │  - Private training data │    │  - Private training data │
│  - Local model           │    │  - Local model           │    │  - Local model           │
│         ↕                │    │         ↕                │    │         ↕                │
│  Holochain Conductor     │◄──►│  Holochain Conductor     │◄──►│  Holochain Conductor     │
│  - Local DHT storage     │P2P │  - Local DHT storage     │P2P │  - Local DHT storage     │
│  - Port: 8881           │    │  - Port: 8882           │    │  - Port: 8883           │
└──────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
```

**Key Points:**
- ✅ **No central server** - each node is independent
- ✅ **Data privacy** - patient data never leaves hospital
- ✅ **P2P network** - nodes communicate directly via Holochain DHT
- ✅ **Byzantine resistance** - malicious nodes can be detected
- ✅ **Production-ready** - same architecture scales to 1000+ nodes

## 🚀 Quick Start

### Prerequisites

```bash
# You need Docker and Docker Compose
docker --version
docker-compose --version
```

### Step 1: Start the Multi-Node Network

```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML

# Start all 3 hospital nodes + conductors
docker-compose -f docker-compose.multi-node.yml up -d

# Watch the logs
docker-compose -f docker-compose.multi-node.yml logs -f
```

### Step 2: Verify Nodes Are Running

```bash
# Check all services are healthy
docker-compose -f docker-compose.multi-node.yml ps

# Should show:
# - postgres (healthy)
# - holochain-node1-boston (healthy)
# - holochain-node2-london (healthy)
# - holochain-node3-tokyo (healthy)
# - zerotrustml-node1-boston
# - zerotrustml-node2-london
# - zerotrustml-node3-tokyo
# - zerotrustml-test-orchestrator (exits after test completes)
```

### Step 3: View Test Results

```bash
# The test orchestrator runs automatically and saves results
cat test_results/multi_node_p2p_results.json

# View detailed logs
docker logs zerotrustml-test-orchestrator
```

### Step 4: Manual Testing

```bash
# Connect to a specific node
docker exec -it zerotrustml-node1-boston /bin/bash

# Run Python REPL and test
python3
>>> import asyncio
>>> from test_multi_node_p2p import HospitalNode
>>> # Test individual node operations
```

## 🔍 What's Being Tested

### Round 1-5: Federated Learning Cycle

For each round:

1. **Local Training** (parallel, independent)
   - Boston trains on local patient data
   - London trains on local patient data
   - Tokyo trains on local patient data
   - 🔒 **Data never leaves hospital**

2. **Share Gradients to DHT** (P2P network)
   - Each node shares gradients via local Holochain conductor
   - Gradients stored in distributed hash table (DHT)
   - No central server involved

3. **Fetch Peer Gradients** (P2P retrieval)
   - Each node queries DHT for peer gradients
   - P2P network automatically routes queries
   - Byzantine nodes can be filtered out

4. **Aggregate & Update** (local computation)
   - Each node averages peer gradients (FedAvg)
   - Updates local model
   - Model improves from collective learning

## 📊 Expected Output

```
======================================================================
🚀 ROUND 1 - Multi-Node Federated Learning
======================================================================

📊 Step 1: Local Training (Private Data)
----------------------------------------------------------------------
  Boston Medical Center        | Loss: 2.3456
  London General Hospital      | Loss: 2.4123
  Tokyo Research Hospital      | Loss: 2.3891

🌐 Step 2: Share Gradients to Holochain DHT
----------------------------------------------------------------------
  Boston Medical Center        | Gradient norm: 15.234
  London General Hospital      | Gradient norm: 14.891
  Tokyo Research Hospital      | Gradient norm: 15.123

📥 Step 3: Fetch Peer Gradients from DHT
----------------------------------------------------------------------
  Boston Medical Center        | ✅ Received 2 peer gradients
  London General Hospital      | ✅ Received 2 peer gradients
  Tokyo Research Hospital      | ✅ Received 2 peer gradients

🔄 Step 4: Aggregate & Update Local Models
----------------------------------------------------------------------
  Boston Medical Center        | ✅ Model updated
  London General Hospital      | ✅ Model updated
  Tokyo Research Hospital      | ✅ Model updated

📈 Round 1 Complete | Average Loss: 2.3823
```

## 🎯 Success Criteria

✅ **All nodes start successfully**
✅ **P2P network forms between conductors**
✅ **Models train on local data**
✅ **Gradients shared via DHT**
✅ **Model loss decreases over rounds**
✅ **No data privacy violations**

## 🔧 Troubleshooting

### Nodes won't start

```bash
# Clean up and restart
docker-compose -f docker-compose.multi-node.yml down -v
docker-compose -f docker-compose.multi-node.yml up -d
```

### Can't connect to Holochain

```bash
# Check conductor logs
docker logs holochain-node1-boston
docker logs holochain-node2-london
docker logs holochain-node3-tokyo

# Verify WebSocket ports are open
nc -zv localhost 8881  # Boston
nc -zv localhost 8882  # London
nc -zv localhost 8883  # Tokyo
```

### Test fails immediately

```bash
# The test orchestrator waits 15 seconds for nodes to be ready
# If it still fails, increase the sleep time in docker-compose.multi-node.yml
# Look for the "sleep 15" line and increase to "sleep 30"
```

## 🌍 Scaling to Production

This same architecture scales to real production:

### Development (3 nodes - this test)
```bash
docker-compose -f docker-compose.multi-node.yml up
```

### Production (100+ hospitals)
```yaml
# Each hospital deploys their own stack:
- Zero-TrustML node (their hardware)
- Holochain conductor (their hardware)
- Private patient data (never shared)

# Nodes connect via P2P - no coordination needed!
```

## 📝 Key Files

- **docker-compose.multi-node.yml** - 3-node P2P network definition
- **tests/test_multi_node_p2p.py** - Federated learning test script
- **holochain/conductor-config-minimal.yaml** - Holochain conductor config
- **test_results/multi_node_p2p_results.json** - Test results

## 🎓 Learn More

- **What is Holochain?** https://holochain.org
- **What is Federated Learning?** See: `docs/FEDERATED_LEARNING_INTRO.md`
- **How does Byzantine resistance work?** See: `docs/BYZANTINE_RESISTANCE.md`

## 🎉 Success!

If you see all nodes training and models improving, **congratulations!** You've just run a true P2P federated learning network with:

- ✅ No central server
- ✅ Data privacy preserved
- ✅ Byzantine resistance enabled
- ✅ Production-ready architecture

**This is exactly how Zero-TrustML works in production - scaled to 1000+ nodes!**
