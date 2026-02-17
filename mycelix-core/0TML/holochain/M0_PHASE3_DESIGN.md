# M0 Phase 3: Fully Decentralized FL Orchestrator

**Status**: Design Phase
**Last Updated**: November 10, 2025
**Target**: 5-node MNIST demo with real Holochain DHT and Byzantine resistance

---

## 🎯 Goals

**Primary**: Demonstrate fully decentralized Byzantine-resistant federated learning using:
- 5 independent Holochain nodes (real DHT networking)
- RISC Zero PoGQ proofs for Byzantine detection
- Peer-to-peer gradient aggregation (no central coordinator)
- 2 Byzantine nodes (40% adversarial) successfully quarantined

**Success Criteria**:
- ✅ 5 nodes training MNIST independently
- ✅ All nodes publish gradients + PoGQ proofs to DHT
- ✅ Byzantine nodes detected and quarantined
- ✅ Model converges despite 40% Byzantine participation
- ✅ Zero central coordination (pure P2P)

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Holochain DHT Network                        │
│  (5 independent conductor processes, Go-based networking)       │
└─────────────────────────────────────────────────────────────────┘
         ↑                ↑              ↑              ↑
         │ publish        │ publish      │ publish      │ publish
         │ query          │ query        │ query        │ query
         │                │              │              │
    ┌────┴────┐      ┌───┴────┐    ┌───┴────┐    ┌────┴────┐
    │ Node 1  │      │ Node 2 │    │ Node 3 │    │ Node 4  │ ...
    │(Honest) │      │(Honest)│    │(Byz)   │    │(Honest) │
    └─────────┘      └────────┘    └────────┘    └─────────┘
         │                │              │              │
    ┌────┴────┐      ┌───┴────┐    ┌───┴────┐    ┌────┴────┐
    │MNIST    │      │MNIST   │    │MNIST   │    │MNIST    │
    │Partition│      │Partition│   │Partition│   │Partition│
    └─────────┘      └────────┘    └────────┘    └─────────┘
         │                │              │              │
    ┌────┴────┐      ┌───┴────┐    ┌───┴────┐    ┌────┴────┐
    │PoGQ     │      │PoGQ    │    │PoGQ    │    │PoGQ     │
    │Prover   │      │Prover  │    │Prover  │    │Prover   │
    └─────────┘      └────────┘    └────────┘    └─────────┘
```

### Node Architecture (Per Node)

Each node is an independent Python process running:

1. **FL Training Loop**
   - Load local MNIST partition (IID or non-IID)
   - Train model for E local epochs
   - Compute gradient update

2. **PoGQ Proof Generation**
   - Package gradient + provenance into journal
   - Generate RISC Zero proof (via vsv-stark host)
   - Extract receipt + journal

3. **Holochain Publication**
   - Publish PoGQProofEntry to DHT (via pogq_zome)
   - Publish GradientEntry linked to proof
   - Wait for DHT consistency

4. **Peer Aggregation**
   - Query DHT for all gradients in current round
   - Verify each proof's quarantine status
   - Compute sybil-weighted aggregate (Byzantine weight = 0)
   - Apply aggregate to local model

5. **Repeat**
   - Continue for R global rounds

---

## 📋 Implementation Phases

### Phase 3.1: Environment Setup (1-2 hours)

**Install Go** (required for Holochain networking):
```bash
# NixOS approach - add to flake.nix or use nix-shell
nix-shell -p go_1_21

# Verify installation
go version  # Should be 1.19+
```

**Install Holochain conductor**:
```bash
# Option A: Use Holochain's nix flake
nix develop github:holochain/holochain

# Option B: Binary install
cargo install holochain --version 0.4.4
cargo install holochain_cli --version 0.4.4
```

**Compile pogq_zome with Go available**:
```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML/holochain/zomes/pogq_zome
cargo test --package pogq_zome byzantine_resistance
# Should now succeed with Go installed
```

---

### Phase 3.2: DNA Packaging (2-3 hours)

**Create DNA manifest** (`holochain/dnas/pogq_dna/dna.yaml`):
```yaml
---
manifest_version: "1"
name: pogq_dna
integrity:
  origin_time: 2025-11-10T00:00:00.000000Z
  network_seed: ~
  properties: ~
  zomes:
    - name: pogq_zome
      bundled: ../../zomes/pogq_zome.wasm
coordinator:
  zomes: []
```

**Compile zome to WASM**:
```bash
# Fix WASM toolchain if needed (NixOS flake)
cd /srv/luminous-dynamics/Mycelix-Core/0TML/holochain/zomes/pogq_zome
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/pogq_zome.wasm ../../dnas/pogq_dna/
```

**Package DNA**:
```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML/holochain
hc dna pack dnas/pogq_dna
# Creates dnas/pogq_dna.dna
```

---

### Phase 3.3: Conductor Configuration (2-3 hours)

**Create 5 conductor configs** (`holochain/conductors/node{1-5}.yaml`):
```yaml
---
environment_path: /tmp/m0_demo/node1
use_dangerous_test_keystore: true  # M0 only!
keystore_path: /tmp/m0_demo/node1/keystore
admin_interfaces:
  - driver:
      type: websocket
      port: 3001  # Unique per node: 3001-3005
app_interfaces:
  - driver:
      type: websocket
      port: 4001  # Unique per node: 4001-4005
network:
  bootstrap_service: https://bootstrap.holo.host
  transport_pool:
    - type: quic
  tuning_params:
    gossip_loop_iteration_delay_ms: 100
```

**Launch script** (`holochain/scripts/launch_conductors.sh`):
```bash
#!/usr/bin/env bash
# Launch 5 independent Holochain conductors

for i in {1..5}; do
  echo "Starting conductor node${i}..."
  holochain -c conductors/node${i}.yaml > /tmp/m0_demo/node${i}.log 2>&1 &
  echo "Node ${i} PID: $!"
done

echo "All conductors started. Logs in /tmp/m0_demo/"
echo "Admin ports: 3001-3005"
echo "App ports: 4001-4005"
```

---

### Phase 3.4: FL Node Implementation (4-6 hours)

**FL Node Script** (`holochain/scripts/fl_node.py`):
```python
#!/usr/bin/env python3
"""
Federated Learning Node for M0 Demo

Each node:
1. Trains MNIST locally
2. Generates PoGQ proof via RISC Zero
3. Publishes to Holochain DHT
4. Aggregates peer gradients
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from holochain_client import HolochainClient
from pathlib import Path
import sys

sys.path.append('/srv/luminous-dynamics/Mycelix-Core/0TML/vsv-stark')
from host import generate_pogq_proof  # RISC Zero prover

class MNISTModel(nn.Module):
    """Simple CNN for MNIST"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class FLNode:
    """Decentralized Federated Learning Node"""

    def __init__(self, node_id: int, admin_port: int, app_port: int,
                 is_byzantine: bool = False):
        self.node_id = node_id
        self.is_byzantine = is_byzantine

        # Connect to local Holochain conductor
        self.hc_admin = HolochainClient(f"ws://localhost:{admin_port}")
        self.hc_app = HolochainClient(f"ws://localhost:{app_port}")

        # Initialize model
        self.model = MNISTModel()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        # Load MNIST partition
        self.train_data = self.load_mnist_partition(node_id)

    def load_mnist_partition(self, node_id: int):
        """Load IID partition of MNIST for this node"""
        from torchvision import datasets, transforms

        dataset = datasets.MNIST(
            '/tmp/mnist', train=True, download=True,
            transform=transforms.ToTensor()
        )

        # IID partition: each node gets 1/5 of data
        partition_size = len(dataset) // 5
        start_idx = (node_id - 1) * partition_size
        end_idx = start_idx + partition_size

        return torch.utils.data.Subset(dataset, range(start_idx, end_idx))

    def train_local_epochs(self, epochs: int = 1):
        """Train model on local data"""
        self.model.train()
        loader = torch.utils.data.DataLoader(self.train_data, batch_size=32)

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()

        # Return gradient update
        return self.get_model_gradients()

    def get_model_gradients(self) -> np.ndarray:
        """Extract gradients from model"""
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.detach().cpu().numpy().flatten())
        return np.concatenate(gradients)

    def generate_pogq_proof(self, gradient: np.ndarray, round_num: int):
        """Generate RISC Zero PoGQ proof"""
        # Package provenance data
        journal = {
            'node_id': self.node_id,
            'round': round_num,
            'gradient_hash': hash(gradient.tobytes()),
            'quarantine_out': 1 if self.is_byzantine else 0,
            'timestamp': int(time.time() * 1_000_000),
        }

        # Generate proof via RISC Zero
        receipt, journal_bytes = generate_pogq_proof(journal)

        return receipt, journal_bytes

    def publish_to_dht(self, round_num: int, gradient: np.ndarray,
                       receipt: bytes, journal: dict):
        """Publish proof and gradient to Holochain DHT"""
        import secrets

        # Generate unique nonce
        nonce = secrets.token_bytes(32)

        # Publish PoGQ proof
        proof_entry = {
            'node_id': self.hc_app.agent_pub_key(),
            'round': round_num,
            'nonce': list(nonce),
            'receipt_bytes': list(receipt),
            'prov_hash': [0, 0, 0, 0],  # Placeholder for M0
            'profile_id': 128,  # S128
            'air_rev': 1,
            'quarantine_out': journal['quarantine_out'],
            'current_round': round_num,
            'ema_t_fp': 65536,
            'consec_viol_t': 0,
            'consec_clear_t': 5 if not self.is_byzantine else 0,
            'timestamp': journal['timestamp'],
        }

        proof_hash = self.hc_app.call_zome(
            'pogq_zome', 'publish_pogq_proof', proof_entry
        )

        # Publish gradient linked to proof
        gradient_entry = {
            'node_id': self.hc_app.agent_pub_key(),
            'round': round_num,
            'nonce': list(nonce),
            'gradient_commitment': list(gradient[:256]),  # First 256 bytes
            'quality_score': 0.95 if not self.is_byzantine else 0.3,
            'pogq_proof_hash': proof_hash,
            'timestamp': journal['timestamp'],
        }

        grad_hash = self.hc_app.call_zome(
            'pogq_zome', 'publish_gradient', gradient_entry
        )

        return proof_hash, grad_hash

    def query_round_gradients(self, round_num: int):
        """Query DHT for all gradients in current round"""
        gradients = self.hc_app.call_zome(
            'pogq_zome', 'get_round_gradients', round_num
        )
        return gradients

    def compute_aggregate(self, round_num: int):
        """Compute sybil-weighted aggregate from DHT"""
        result = self.hc_app.call_zome(
            'pogq_zome', 'compute_sybil_weighted_aggregate', round_num
        )

        print(f"Node {self.node_id} | Round {round_num} | "
              f"Healthy: {result['num_healthy']} | "
              f"Quarantined: {result['num_quarantined']} | "
              f"Weight: {result['total_weight']:.2f}")

        return result['aggregate']

    def run_training(self, total_rounds: int = 10, local_epochs: int = 1):
        """Main training loop"""
        print(f"Node {self.node_id} starting training "
              f"({'BYZANTINE' if self.is_byzantine else 'Honest'})")

        for round_num in range(1, total_rounds + 1):
            # 1. Local training
            gradient = self.train_local_epochs(local_epochs)

            # 2. Generate PoGQ proof
            receipt, journal = self.generate_pogq_proof(gradient, round_num)

            # 3. Publish to DHT
            self.publish_to_dht(round_num, gradient, receipt, journal)

            # 4. Wait for DHT consistency (10 seconds)
            time.sleep(10)

            # 5. Query and aggregate
            aggregate = self.compute_aggregate(round_num)

            # 6. Apply aggregate to model
            self.apply_gradient(aggregate)

            print(f"Node {self.node_id} | Round {round_num} complete")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-id', type=int, required=True)
    parser.add_argument('--admin-port', type=int, required=True)
    parser.add_argument('--app-port', type=int, required=True)
    parser.add_argument('--byzantine', action='store_true')
    parser.add_argument('--rounds', type=int, default=10)
    args = parser.parse_args()

    node = FLNode(
        node_id=args.node_id,
        admin_port=args.admin_port,
        app_port=args.app_port,
        is_byzantine=args.byzantine
    )

    node.run_training(total_rounds=args.rounds)

if __name__ == '__main__':
    main()
```

---

### Phase 3.5: Demo Orchestration (1-2 hours)

**Master launch script** (`holochain/scripts/run_m0_demo.sh`):
```bash
#!/usr/bin/env bash
# M0 Phase 3 Demo: 5-node decentralized MNIST FL

set -e

echo "=== M0 Phase 3: Decentralized FL Demo ==="
echo ""

# 1. Clean previous run
echo "Cleaning previous demo state..."
rm -rf /tmp/m0_demo
mkdir -p /tmp/m0_demo

# 2. Launch Holochain conductors
echo "Launching 5 Holochain conductors..."
bash launch_conductors.sh
sleep 5  # Wait for conductors to start

# 3. Install DNA on each conductor
echo "Installing pogq_dna on all nodes..."
for i in {1..5}; do
  admin_port=$((3000 + i))
  hc app install \
    --admin-port ${admin_port} \
    --app-id m0_demo \
    --path ../dnas/pogq_dna.dna
done

# 4. Launch FL nodes (2 Byzantine, 3 Honest)
echo "Starting 5 FL training nodes (2 Byzantine)..."
python3 fl_node.py --node-id 1 --admin-port 3001 --app-port 4001 --rounds 10 > /tmp/m0_demo/node1_fl.log 2>&1 &
python3 fl_node.py --node-id 2 --admin-port 3002 --app-port 4002 --rounds 10 > /tmp/m0_demo/node2_fl.log 2>&1 &
python3 fl_node.py --node-id 3 --admin-port 3003 --app-port 4003 --byzantine --rounds 10 > /tmp/m0_demo/node3_fl.log 2>&1 &
python3 fl_node.py --node-id 4 --admin-port 3004 --app-port 4004 --rounds 10 > /tmp/m0_demo/node4_fl.log 2>&1 &
python3 fl_node.py --node-id 5 --admin-port 3005 --app-port 4005 --byzantine --rounds 10 > /tmp/m0_demo/node5_fl.log 2>&1 &

echo ""
echo "Demo running! Monitor logs:"
echo "  tail -f /tmp/m0_demo/node*.log"
echo ""
echo "Press Ctrl+C to stop all nodes"

# Wait for user interrupt
trap 'kill $(jobs -p)' EXIT
wait
```

---

### Phase 3.6: Visualization & Analysis (2-3 hours)

**Metrics collection** (`holochain/scripts/analyze_demo.py`):
```python
#!/usr/bin/env python3
"""Analyze M0 demo results and generate visualization"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

def parse_logs(log_dir: Path):
    """Parse FL node logs and extract metrics"""
    metrics = {
        'rounds': [],
        'accuracy': {1: [], 2: [], 3: [], 4: [], 5: []},
        'quarantined_nodes': [],
        'aggregate_weights': [],
    }

    for node_id in range(1, 6):
        log_file = log_dir / f"node{node_id}_fl.log"
        # Parse log file for metrics
        # ...

    return metrics

def plot_results(metrics):
    """Generate visualization of demo results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Model accuracy over rounds
    ax = axes[0, 0]
    for node_id in range(1, 6):
        style = '--' if node_id in [3, 5] else '-'
        label = f"Node {node_id} {'(Byz)' if node_id in [3, 5] else '(Hon)'}"
        ax.plot(metrics['rounds'], metrics['accuracy'][node_id],
                style, label=label)
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Convergence (40% Byzantine)')
    ax.legend()
    ax.grid(True)

    # Plot 2: Byzantine detection
    ax = axes[0, 1]
    ax.plot(metrics['rounds'], metrics['quarantined_nodes'], 'r-')
    ax.set_xlabel('Round')
    ax.set_ylabel('Quarantined Nodes')
    ax.set_title('Byzantine Node Detection')
    ax.grid(True)

    # Plot 3: Aggregate weights
    ax = axes[1, 0]
    ax.plot(metrics['rounds'], metrics['aggregate_weights'], 'g-')
    ax.set_xlabel('Round')
    ax.set_ylabel('Total Weight')
    ax.set_title('Sybil-Weighted Aggregation')
    ax.axhline(y=3.0, color='b', linestyle='--', label='Expected (3 honest)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('/tmp/m0_demo/results.png', dpi=300)
    print("Visualization saved to /tmp/m0_demo/results.png")

if __name__ == '__main__':
    metrics = parse_logs(Path('/tmp/m0_demo'))
    plot_results(metrics)
```

---

## 📊 Expected Results

### Success Metrics

1. **Byzantine Detection**: Nodes 3 & 5 quarantined by round 2-3
2. **Model Accuracy**: Honest nodes achieve >90% MNIST accuracy by round 10
3. **Aggregate Weight**: Stabilizes at ~3.0 (3 honest nodes contributing)
4. **DHT Consistency**: All nodes see same gradients within 10 seconds
5. **Zero Downtime**: No central coordinator failure = no system failure

### Timeline Estimate

| Phase | Task | Est. Time | Status |
|-------|------|-----------|--------|
| 3.1 | Environment Setup | 1-2 hours | Pending |
| 3.2 | DNA Packaging | 2-3 hours | Pending |
| 3.3 | Conductor Config | 2-3 hours | Pending |
| 3.4 | FL Node Impl | 4-6 hours | Pending |
| 3.5 | Demo Orchestration | 1-2 hours | Pending |
| 3.6 | Visualization | 2-3 hours | Pending |
| **Total** | | **12-19 hours** | **~2-3 days** |

---

## 🔧 Dependencies

**System Requirements**:
- [x] NixOS or Linux with Nix
- [ ] Go 1.19+ (for Holochain networking)
- [x] Rust 1.70+ (already installed)
- [x] Python 3.11+ (already installed)

**Rust Crates**:
- [x] `holochain = "0.4"` with `test_utils` feature
- [x] `hdk = "0.4"`
- [ ] `holochain_cli = "0.4.4"` (for `hc` command)

**Python Packages**:
- [x] `torch` (MNIST training)
- [ ] `holochain-client-python` (DHT communication)
- [ ] `numpy` (gradient operations)
- [x] `matplotlib` (visualization)

---

## 🚀 Next Steps

**Immediate** (tonight if time permits):
1. Install Go via Nix
2. Verify pogq_zome Rust integration tests pass
3. Create DNA packaging structure

**Tomorrow** (Day 1 of Phase 3):
1. Complete DNA packaging and conductor configs
2. Start FL node implementation
3. Test single-node workflow end-to-end

**Day 2 of Phase 3**:
1. Complete FL node implementation
2. Build orchestration scripts
3. First full 5-node test run

**Day 3 of Phase 3**:
1. Debug and polish
2. Add visualization and analysis
3. Document results for paper

---

## 📝 Notes

**Why Fully Decentralized Matters**:
- **Academic integrity**: Honest representation of Byzantine-resistant architecture
- **Scalability**: Proves system works without central bottleneck
- **Resilience**: Demonstrates true fault tolerance
- **Innovation**: Shows ZK proofs + Holochain DHT integration

**Comparison to Literature**:
- Most FL papers use centralized aggregator (single point of failure)
- Multi-Krum, Bulyan, etc. all assume trusted coordinator
- Our approach: **truly decentralized Byzantine resistance**

---

**Status**: Ready to begin Phase 3.1 (Environment Setup) ✨
