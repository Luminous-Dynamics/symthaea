# Mycelix Protocol Examples

This directory contains example scripts demonstrating key features of the Mycelix Protocol.

## Available Examples

### 1. Federated Learning with Holochain (`federated_learning_with_holochain.py`)

Demonstrates the core Proof-of-Gradient-Quality (PoGQ) mechanism integrated with Holochain's DHT for distributed coordination.

**What it shows:**
- Training local models on simulated data
- Computing and submitting gradients to Holochain DHT
- Validator nodes performing PoGQ validation
- Reputation-weighted gradient aggregation
- Byzantine node detection

**How to run:**
```bash
# Enter Nix development environment
nix develop

# Run the example
python examples/federated_learning_with_holochain.py
```

**Expected output:**
- 5 honest nodes and 2 Byzantine attackers
- PoGQ quality scores computed for each gradient
- Aggregated global model update
- Reputation score updates based on contribution quality

### 2. Multi-Node Trust Demo (`multi_node_trust_demo.py`)

Demonstrates the reputation system with multiple interacting nodes building trust over time.

**What it shows:**
- Node registration and DID creation
- Trust relationships forming through interactions
- Reputation scores evolving based on behavior
- Byzantine node penalization

**How to run:**
```bash
python examples/multi_node_trust_demo.py
```

**Expected output:**
- Network initialization with diverse node types
- Simulated interactions (gradients, attestations, reports)
- Reputation score updates over 10 rounds
- Honest nodes accumulating positive reputation
- Byzantine nodes being identified and penalized

### 3. Simple Client (`simple_client.py`)

A minimal client demonstrating basic interaction with the Mycelix Protocol backend.

**What it shows:**
- Connecting to the Mycelix backend
- Submitting basic requests
- Receiving and processing responses

**How to run:**
```bash
# Ensure backend services are running
./scripts/start-services.sh

# Run the simple client
python examples/simple_client.py
```

## Prerequisites

All examples require:
- **NixOS or Linux with Nix** installed
- **Nix development environment**: `nix develop`
- **Python dependencies**: Automatically available in Nix shell

Some examples may require:
- **Holochain conductor** running (for Holochain-integrated examples)
- **Backend services** running (for client examples)

## Understanding the Examples

### Core Concepts Demonstrated

**Proof-of-Gradient-Quality (PoGQ)**:
- Validators test gradients on private datasets
- Quality scores computed from validation loss improvement
- Low-quality gradients filtered out before aggregation
- Reputation scores updated based on contribution quality

**Reputation System**:
- Multi-dimensional scores (accuracy, reliability, availability)
- Time decay to encourage continuous participation
- Byzantine-resistant aggregation mechanisms

**Holochain Integration**:
- Agent-centric data model (each node has own source chain)
- DHT-based discovery and coordination
- Cryptographically signed gradients and validations

## Customizing Examples

Most examples accept command-line arguments:

```bash
# Run with custom parameters
python examples/federated_learning_with_holochain.py \
  --num_honest 10 \
  --num_byzantine 3 \
  --epochs 20

# See all options
python examples/federated_learning_with_holochain.py --help
```

## Troubleshooting

**Import errors:**
```bash
# Ensure you're in the Nix development environment
nix develop

# Or install dependencies manually (not recommended)
pip install -r requirements.txt
```

**Holochain connection errors:**
```bash
# Ensure Holochain conductor is running
hc sandbox create
hc sandbox run
```

**Performance issues:**
```bash
# For GPU acceleration (if available)
python examples/federated_learning_with_holochain.py --device cuda

# For smaller-scale testing
python examples/federated_learning_with_holochain.py --num_honest 3 --num_byzantine 1
```

## Next Steps

After exploring these examples:

1. **Run experiments**: See [Experimental Validation](../README.md#-experimental-results)
2. **Read architecture docs**: See [System Architecture](../docs/06-architecture/SYSTEM_ARCHITECTURE.md)
3. **Contribute**: See [Contributing Guide](../CONTRIBUTING.md)

## Questions?

- **GitHub Issues**: Report bugs or ask questions
- **Email**: tristan.stoltz@evolvingresonantcocreationism.com
- **Documentation**: See [README.md](../README.md) for comprehensive overview

---

**Note**: These examples are for educational and research purposes. For production use, see the full implementation in `src/`.
