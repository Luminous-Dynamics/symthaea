# Zero-TrustML CLI Reference Guide

Complete reference for Zero-TrustML Holochain command-line interface tools.

## Overview

Zero-TrustML provides three command-line tools for managing decentralized federated learning networks:

1. **`zerotrustml`** - Main CLI for initialization and basic node management
2. **`zerotrustml-node`** - Advanced node control and training
3. **`zerotrustml-monitor`** - Network monitoring and dashboard

## Installation

```bash
# Install from PyPI (after release)
pip install zerotrustml-holochain

# Or install from source
git clone https://github.com/Luminous-Dynamics/0TML
cd 0TML
pip install -e .
```

## 1. `zerotrustml` - Main CLI

Main command-line interface for Zero-TrustML node management.

### Commands

#### `zerotrustml init` - Initialize Node

Initialize a new Zero-TrustML node with configuration.

```bash
zerotrustml init --node-id hospital-1 --data-path /data/medical-images
```

**Options:**
- `--node-id` (required) - Unique identifier for this node
- `--data-path` (required) - Path to training data directory
- `--model-type` (default: `resnet18`) - Model architecture to use
- `--holochain-url` (default: `ws://localhost:8888`) - Holochain conductor URL
- `--aggregation` (default: `krum`) - Aggregation algorithm (`krum`, `fedavg`, `trimmed_mean`, `median`)
- `--batch-size` (default: `32`) - Training batch size
- `--learning-rate` (default: `0.01`) - Learning rate
- `--force` - Overwrite existing configuration

**Example:**
```bash
zerotrustml init \
  --node-id hospital-a \
  --data-path /data/mnist \
  --model-type resnet18 \
  --holochain-url ws://holochain:8888 \
  --aggregation krum \
  --batch-size 64 \
  --learning-rate 0.001
```

**Output:**
```
✅ Initialized Zero-TrustML node: hospital-a
   Config: /home/user/.zerotrustml/config.toml
   Data:   /data/mnist

Next steps:
  1. Review configuration: cat /home/user/.zerotrustml/config.toml
  2. Start node: zerotrustml start
```

#### `zerotrustml start` - Start Node

Start the Zero-TrustML node using saved configuration.

```bash
zerotrustml start [--rounds ROUNDS]
```

**Options:**
- `--rounds` (optional) - Number of training rounds to run before stopping

**Example:**
```bash
# Start node and keep running
zerotrustml start

# Start node and train for 50 rounds
zerotrustml start --rounds 50
```

**Output:**
```
🚀 Starting Zero-TrustML node: hospital-a
   Holochain: ws://holochain:8888
   Aggregation: krum

✅ Node started successfully
   Accuracy: 87.50%
   Credits:  1250
```

#### `zerotrustml status` - Node Status

Show current node configuration and status.

```bash
zerotrustml status
```

**Output:**
```
📊 Zero-TrustML Node Status
   Node ID:     hospital-a
   Data Path:   /data/mnist
   Holochain:   ws://holochain:8888
   Aggregation: krum

⚠️  Status check requires active node connection
   Run 'zerotrustml-node status' for live metrics
```

#### `zerotrustml credits` - Manage Credits

View and manage Zero-TrustML Credits balance.

```bash
zerotrustml credits balance [--node-id NODE_ID]
```

**Options:**
- `--node-id` (optional) - Node ID to check (defaults to current node)

**Output:**
```
💰 Credits Balance
   Node:    hospital-a
   Balance: ⚠️  Requires active node connection

   Run 'zerotrustml-node credits' for live balance
```

---

## 2. `zerotrustml-node` - Advanced Node Management

Advanced node control for detailed operations.

### Commands

#### `zerotrustml-node start` - Start with Full Control

Start a node with complete control over configuration.

```bash
zerotrustml-node start \
  --node-id hospital-1 \
  --data-path /data \
  [--model-type MODEL] \
  [--holochain-url URL] \
  [--aggregation ALGO] \
  [--batch-size SIZE] \
  [--learning-rate LR] \
  [--daemon]
```

**Options:**
- `--node-id` (required) - Node identifier
- `--data-path` (required) - Training data path
- `--model-type` (default: `resnet18`) - Model architecture
- `--holochain-url` (default: `ws://localhost:8888`) - Holochain URL
- `--aggregation` (default: `krum`) - Aggregation algorithm
- `--batch-size` (default: `32`) - Batch size
- `--learning-rate` (default: `0.01`) - Learning rate
- `--daemon` - Run as background daemon

**Example:**
```bash
zerotrustml-node start \
  --node-id hospital-a \
  --data-path /data/mnist \
  --aggregation krum \
  --batch-size 128 \
  --learning-rate 0.001
```

#### `zerotrustml-node status` - Detailed Status

Show detailed node status and metrics.

```bash
zerotrustml-node status
```

**Output:**
```
📊 Zero-TrustML Node Status
   Version: 1.0.0

⚠️  Live status requires active connection
   This would show:
   - Current round
   - Model accuracy
   - Credit balance
   - Peer connections
   - Recent gradients
```

#### `zerotrustml-node train` - Train Node

Run training for specified number of rounds.

```bash
zerotrustml-node train \
  --node-id hospital-1 \
  --data-path /data \
  --rounds ROUNDS \
  [--model-type MODEL] \
  [--holochain-url URL]
```

**Example:**
```bash
zerotrustml-node train \
  --node-id hospital-a \
  --data-path /data/mnist \
  --rounds 100
```

**Output:**
```
🔄 Training Node
   Rounds: 100
   Model:  resnet18

🔄 Starting training...

✅ Training complete
   Final accuracy: 92.35%
```

#### `zerotrustml-node credits` - Credit Balance

Show detailed credit balance and transaction history.

```bash
zerotrustml-node credits [--node-id NODE_ID]
```

**Output:**
```
💰 Zero-TrustML Credits
   Node: hospital-a

⚠️  Live credits require active connection
   This would show:
   - Current balance
   - Recent transactions
   - Reputation level
   - Pending rewards
```

#### `zerotrustml-node peers` - Connected Peers

Show network peers and their status.

```bash
zerotrustml-node peers
```

**Output:**
```
🌐 Connected Peers

⚠️  Live peer list requires active connection
   This would show:
   - Peer node IDs
   - Reputation scores
   - Recent gradients
   - Network topology
```

---

## 3. `zerotrustml-monitor` - Network Monitoring

Network monitoring and visualization dashboard.

### Commands

#### `zerotrustml-monitor dashboard` - Start Dashboard

Launch the web-based monitoring dashboard.

```bash
zerotrustml-monitor dashboard [--host HOST] [--port PORT]
```

**Options:**
- `--host` (default: `0.0.0.0`) - Host to bind to
- `--port` (default: `8000`) - Port to bind to

**Example:**
```bash
zerotrustml-monitor dashboard --host 0.0.0.0 --port 8000
```

**Output:**
```
📊 Starting Zero-TrustML Dashboard
   Listening on http://0.0.0.0:8000
   Monitoring 3 nodes

⚠️  Dashboard server not yet implemented
   Would show:
   - Network topology visualization
   - Real-time accuracy metrics
   - Credit flow visualization
   - Byzantine attack detection
```

#### `zerotrustml-monitor metrics` - Network Metrics

Display network-wide metrics.

```bash
zerotrustml-monitor metrics
```

**Output:**
```
📈 Network Metrics

⚠️  Live metrics require active network connection

This would show:
  - Total nodes online
  - Average accuracy
  - Total gradients shared
  - Byzantine attacks detected
  - Credit distribution
  - Network throughput
```

#### `zerotrustml-monitor health` - Network Health

Check overall network health status.

```bash
zerotrustml-monitor health
```

**Output:**
```
🏥 Network Health Check

   Status: healthy
   Score:  95.0%

⚠️  Full health check requires active network
   Would check:
   - Node connectivity
   - DHT synchronization
   - Model convergence
   - Byzantine resistance
```

#### `zerotrustml-monitor topology` - Network Topology

Visualize network topology and connections.

```bash
zerotrustml-monitor topology
```

**Output:**
```
🌐 Network Topology

⚠️  Live topology requires active network connection

This would show:
  - Node connections (graph)
  - Peer relationships
  - Geographic distribution
  - Communication patterns
```

#### `zerotrustml-monitor alerts` - Network Alerts

Show recent network alerts and warnings.

```bash
zerotrustml-monitor alerts
```

**Output:**
```
🚨 Network Alerts

⚠️  Live alerts require active connection

Recent alerts would include:
  - Byzantine attacks detected
  - Node failures
  - Model divergence
  - Performance degradation
```

---

## Environment Variables

All CLI tools respect these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ZEROTRUSTML_CONFIG_DIR` | Configuration directory | `~/.zerotrustml` |
| `DEBUG` | Enable debug output | `false` |
| `ZEROTRUSTML_NODE_ID` | Default node ID | - |
| `ZEROTRUSTML_DATA_PATH` | Default data path | - |
| `ZEROTRUSTML_HOLOCHAIN_URL` | Holochain conductor URL | `ws://localhost:8888` |

## Configuration File

Default location: `~/.zerotrustml/config.toml`

```toml
# Zero-TrustML Node Configuration
node_id = "hospital-a"
data_path = "/data/medical-images"
model_type = "resnet18"
holochain_url = "ws://localhost:8888"
aggregation = "krum"
batch_size = 32
learning_rate = 0.01
```

## Common Workflows

### Quick Start (Single Node)

```bash
# 1. Initialize node
zerotrustml init --node-id test-node --data-path ./data

# 2. Start node
zerotrustml start --rounds 10

# 3. Check status
zerotrustml status
```

### Production Deployment (Multi-Node)

```bash
# Hospital A
zerotrustml init --node-id hospital-a --data-path /data/site-a
zerotrustml start &

# Hospital B
zerotrustml init --node-id hospital-b --data-path /data/site-b
zerotrustml start &

# Hospital C
zerotrustml init --node-id hospital-c --data-path /data/site-c
zerotrustml start &

# Monitor all
zerotrustml-monitor dashboard --port 8000
```

### Docker Deployment

```bash
# Start network with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f node1

# Access dashboard
open http://localhost:8000
```

## Troubleshooting

### Command Not Found

If CLI commands are not found after installation:

```bash
# Check installation
pip show zerotrustml-holochain

# Verify scripts are installed
ls ~/.local/bin/zerotrustml*

# Add to PATH if needed
export PATH="$HOME/.local/bin:$PATH"
```

### Configuration Not Found

```bash
# Check config location
ls -la ~/.zerotrustml/

# Reinitialize if needed
zerotrustml init --node-id new-node --data-path ./data --force
```

### Connection Errors

```bash
# Check Holochain conductor is running
curl http://localhost:8888/health

# Test with different URL
zerotrustml init --holochain-url ws://192.168.1.100:8888 ...
```

## See Also

- [Installation Guide](INSTALLATION.md)
- [Docker Deployment Guide](DOCKER_DEPLOYMENT.md)
- [API Reference](API_REFERENCE.md)
- [Phase 9 Deployment Plan](PHASE9_DEPLOYMENT_PLAN.md)

---

*Zero-TrustML Holochain v1.0.0 - Decentralized Federated Learning*
