# 🛠️ Mycelix Protocol SDK

The Mycelix Protocol SDK provides developers with tools to build applications on top of the Mycelix Protocol.

## 🌟 Overview

The SDK enables:

- **Byzantine-Resistant Federated Learning** integration
- **Agent-Centric Application** development on Holochain
- **Constitutional Governance** implementation
- **Cross-Chain Value Transfer** capabilities

## 📖 SDK Documentation

- **[SDK Design Document](../architecture/Designing Mycelix Protocol SDK.md)** - Complete SDK architecture
- **[0TML Implementation](../0TML/docs/README.md)** - Zero-TrustML technical reference
- **[API Reference](../0TML/docs/04-api/README.md)** - Complete API documentation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Luminous-Dynamics/mycelix
cd Mycelix-Core

# Enter Nix development environment
nix develop

# Install dependencies
cd 0TML
poetry install
```

### Basic Usage

```python
from zerotrustml import FederatedLearning

# Create federated learning coordinator
fl = FederatedLearning(
    byzantine_tolerance=0.45,  # 45% tolerance
    backend="holochain"
)

# Run training
fl.train(dataset="your-data", epochs=10)
```

## 📚 Key Components

### 1. Zero-TrustML (0TML)

Production-grade federated learning with 45% Byzantine tolerance:

- Byzantine-resistant aggregation
- Reputation-weighted validation
- Multi-backend support (PostgreSQL, Holochain, Ethereum)
- Real-time monitoring and metrics

### 2. Agent-Centric DHT

Holochain-based distributed hash table:

- Source chains for personal data sovereignty
- Validation rules enforced by peers
- Currency exchange across networks

### 3. Constitutional Governance

Modular charter framework:

- Epistemic layer for truth classification
- Governance layer for decision-making
- Economic layer for value flows
- Commons layer for shared resources

## 🔗 Related Documentation

- **[MATL Integration Tutorial](../tutorials/matl_integration.md)** - 2-line integration guide
- **[Healthcare FL Tutorial](../tutorials/healthcare_federated_learning.md)** - HIPAA-compliant example
- **[Architecture](../03-architecture/README.md)** - System design
- **[Production Runbook](../0TML/docs/PRODUCTION_OPERATIONS_RUNBOOK.md)** - Operations guide
- **[Complete 0TML Docs](https://github.com/Luminous-Dynamics/mycelix/tree/main/0TML/docs)** - Full developer guide & examples

## 💡 Use Cases

### Healthcare Federated Learning
- HIPAA-compliant distributed training
- Hospital collaboration without data sharing
- Privacy-preserving medical research

### Energy Grid Optimization
- Distributed resource coordination
- Real-time load balancing
- Resilient to node failures

### Financial Systems
- Byzantine-resistant consensus
- Cross-border value transfer
- Regulatory compliance built-in

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Luminous-Dynamics/mycelix/issues)
- **Discussions**: [Community discussions](https://github.com/Luminous-Dynamics/mycelix/discussions)
- **Email**: tristan.stoltz@evolvingresonantcocreationism.com

---

**See also**: [Complete Documentation Hub](../index.md)
