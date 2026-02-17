# Getting Started with Mycelix-DeSci

Welcome to Mycelix-DeSci! This guide will help you get up and running quickly.

## Overview

Mycelix-DeSci provides infrastructure for:
- **Verifiable Data Sharing**: Upload and verify research datasets with cryptographic proofs
- **Federated Learning**: Collaborate on ML models without sharing raw data
- **IP Tokenization**: Convert research outputs into tradeable assets
- **DeSci Integration**: Connect with existing DeSci platforms

## Installation

### Prerequisites

Ensure you have:
- **Rust**: 1.75+ ([Install Rust](https://rustup.rs/))
- **Python**: 3.11+ ([Install Python](https://www.python.org/))
- **Node.js**: 20+ ([Install Node.js](https://nodejs.org/))

### Clone the Repository

```bash
git clone https://github.com/luminousdynamics/mycelix-desci.git
cd mycelix-desci
```

### Build Core Components

```bash
# Build Rust workspace
cargo build --release

# The binary will be at target/release/mycelix-desci-core
```

### Install Python ML Tools

```bash
cd src/ml
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

### Setup Frontend (Optional)

```bash
cd frontend
npm install
npm run dev
```

## Quick Start Examples

### 1. Upload a Verifiable Dataset

```bash
# Using the CLI (coming soon)
mycelix-desci upload \
  --file my_dataset.csv \
  --tier E2 \
  --category genomics \
  --description "CRISPR gene editing results" \
  --provenance "Lab Notebook ID:2024-001"
```

### 2. Query the Knowledge Graph

```rust
use mycelix_desci_core::{DesciClaim, EpistemicTier, ClaimContent};

// Search for claims
let claims = query_claims(QueryFilter {
    min_tier: EpistemicTier::E3,
    category: Some("longevity".to_string()),
    keywords: vec!["NAD+".to_string()],
});

for claim in claims {
    println!("Found: {}", claim.content.description);
}
```

### 3. Run Federated Learning

```python
from mycelix_desci_ml.fl import FederatedClient
import torch.nn as nn

# Define your model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Create FL client
client = FederatedClient(model, client_id="researcher_1")

# Train locally
metrics = client.train(train_loader, epochs=5)
print(f"Accuracy: {metrics['accuracy']:.2f}%")

# Share gradients (PoGQ validated)
gradients = client.get_gradients()
```

### 4. Validate Gradients with PoGQ

```python
from mycelix_desci_ml.pogq import PoGQValidator, GradientUpdate

# Create validator
validator = PoGQValidator(bft_threshold=0.45)

# Validate gradients from multiple participants
scores = validator.validate_gradients(gradient_updates)

# Detect Byzantine actors
byzantine = validator.detect_byzantine(gradient_updates, scores)
print(f"Byzantine actors detected: {byzantine}")

# Aggregate valid gradients
consensus = validator.aggregate_gradients(gradient_updates, scores)
```

## Understanding Epistemic Tiers

Mycelix-DeSci uses epistemic tiers (E0-E4) to classify claim verifiability:

| Tier | Description | Requirements |
|------|-------------|--------------|
| **E0** | Unverified claim | None |
| **E1** | Single-source verification | 1 verification |
| **E2** | Multi-source verification | 2+ verifications |
| **E3** | Reproducible methodology | 3+ verifications + documented methodology |
| **E4** | Peer-reviewed & reproduced | 5+ verifications + peer review |

Higher tiers = more trustworthy data for research.

## Next Steps

- **Read the [Architecture Guide](architecture.md)** to understand system design
- **Explore [Federated Learning Guide](federated-learning.md)** for collaborative ML
- **Check [Integration Guide](integrations.md)** to connect with DeSci platforms
- **Review [API Documentation](../api/)** for detailed references

## Common Tasks

### Verify a Dataset

```bash
# Calculate dataset hash
mycelix-desci hash --file dataset.csv

# Verify against claim
mycelix-desci verify --claim-id <UUID> --file dataset.csv
```

### Join a Federated Learning Network

```bash
# Start FL node
mycelix-desci fl-node start \
  --config config.toml \
  --data-path /path/to/local/data
```

### Query Claims by Category

```bash
mycelix-desci query \
  --category longevity \
  --min-tier E3 \
  --format json
```

## Troubleshooting

### Build Errors

**Issue**: Cargo build fails
**Solution**: Ensure Rust 1.75+ is installed: `rustc --version`

**Issue**: Python dependencies fail
**Solution**: Upgrade pip: `pip install --upgrade pip`

### Runtime Issues

**Issue**: Cannot connect to DHT
**Solution**: Check network configuration and firewall settings

**Issue**: PoGQ validation fails
**Solution**: Verify gradient dimensions match across participants

## Getting Help

- **Documentation**: Check [docs/](../)
- **Issues**: [GitHub Issues](https://github.com/luminousdynamics/mycelix-desci/issues)
- **Discussions**: [GitHub Discussions](https://github.com/luminousdynamics/mycelix-desci/discussions)
- **Discord**: Coming soon

## Contributing

Interested in contributing? See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

---

Happy building with Mycelix-DeSci! 🔬
