# Mycelix Federated Learning Platform

## The Problem
Organizations need to train ML models collaboratively without sharing sensitive data. Existing solutions are either centralized (trust a server) or vulnerable to Byzantine attacks (malicious participants poison the model).

## The Solution
Mycelix FL is the first federated learning platform to achieve **45% Byzantine fault tolerance** — exceeding the theoretical 33% limit. A single Rust binary coordinates gradient aggregation with mathematically verifiable detection of malicious participants.

## Key Metrics
| Metric | Value |
|--------|-------|
| Byzantine Tolerance | **45%** (vs 33% classical limit) |
| Detection Accuracy | **99%+** across 35 attack types |
| False Positive Rate | **0%** in standard configurations |
| Detection Latency | **2-4 rounds** |
| Aggregation Speed | **7ms** (50 nodes, 10K parameters) |
| Proof Verification | **ZK-STARK** (128-bit security) |

## How It Works
1. **Nodes submit encrypted gradients** to the coordinator via gRPC
2. **PoGQ v4.1** scores each gradient using direction + magnitude analysis
3. **Adaptive thresholds** handle non-IID data distributions
4. **Hysteresis quarantine** prevents flapping (k violations to quarantine, m clears to release)
5. **ZK-STARK proofs** provide verifiable evidence of honest detection
6. **Differential privacy** guarantees (configurable epsilon, delta)

## Architecture
- **Coordinator**: Rust binary with gRPC API (Docker-ready)
- **Node Client**: Rust binary connecting to coordinator
- **Holochain Integration**: WASM-compatible for decentralized deployment
- **13 Defense Algorithms**: FedAvg, Krum, Bulyan, FLTrust, PoGQ v4.1, and 8 more

## Use Cases
- **Healthcare**: Multi-hospital model training (HIPAA-compliant privacy)
- **Finance**: Cross-institutional fraud detection
- **IoT/Edge**: Distributed sensor networks
- **Research**: Collaborative scientific computing

## Deployment
```bash
docker compose up coordinator  # Production-ready in minutes
```

## Licensing
- **Open Source**: AGPL-3.0 (copyleft)
- **Enterprise**: Commercial license available
- **Pricing**: Per-organization annual license

## Contact
Luminous Dynamics
tristan.stoltz@evolvingresonantcocreationism.com
https://luminousdynamics.org
