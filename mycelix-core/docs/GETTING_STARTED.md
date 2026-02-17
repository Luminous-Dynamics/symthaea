# Getting Started with Mycelix-Core

Welcome to Mycelix, the world's most advanced Byzantine-resistant federated learning system.

## Quick Start (30 seconds)

```bash
# Run the interactive demo
demo

# Or run the Byzantine resistance simulation
fl-demo

# Run with custom parameters
fl-demo --honest 10 --byzantine 5 --rounds 20
```

## What is Mycelix?

Mycelix is a complete federated learning infrastructure that enables:

- **Byzantine Resistance**: Tolerates up to 45% malicious participants (breaking the classical 33% limit)
- **Decentralized Coordination**: Runs on Holochain DHT with no single point of failure
- **Verifiable Contributions**: Every gradient is cryptographically proven valid
- **Fair Attribution**: Shapley-value based contribution scoring
- **Privacy Preservation**: Differential privacy and secure aggregation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Mycelix Stack                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   0TML      │  │  FL-Agg     │  │  Contracts  │             │
│  │  (Python)   │  │   (Rust)    │  │ (Solidity)  │             │
│  │             │  │             │  │             │             │
│  │ - Training  │  │ - Byzantine │  │ - Registry  │             │
│  │ - Encoding  │  │ - Proofs    │  │ - Payments  │             │
│  │ - Analysis  │  │ - Metrics   │  │ - Reputation│             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
├─────────┴────────────────┴────────────────┴─────────────────────┤
│                     Holochain DHT Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Agents    │  │  FL Zome    │  │   Bridge    │             │
│  │    Zome     │  │             │  │    Zome     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### FL-Aggregator (Rust)

The core aggregation engine with:

- **Byzantine Detection**: Multi-layer detection including Krum, Trimmed Mean, FoolsGold
- **Adaptive Defense**: Automatically escalates protection based on threat level
- **Phi Tracking**: Integrated Information Theory metrics for system coherence
- **Shapley Attribution**: Fair contribution measurement
- **zkSTARK Proofs**: Cryptographic gradient verification

```bash
# Run the FL demo
cargo run --bin fl-demo --release

# Run with specific attack type
cargo run --bin fl-demo --release -- --attack adaptive --byzantine 5

# Available attacks: scaling, flip, label, random, zero, noise, adaptive, freerider
```

### Holochain Zomes

Decentralized coordination layer:

- **Agents Zome**: Identity and credential management
- **FL Zome**: Gradient submission, aggregation rounds, Byzantine detection
- **Bridge Zome**: Cross-system reputation and payment integration

### 0TML (Python)

Training and encoding utilities:

- **HyperFeel Encoding**: 2000x gradient compression
- **Distributed Training**: Multi-node coordination
- **Analysis Tools**: Convergence and fairness metrics

### Smart Contracts (Solidity)

Ethereum integration:

- **MycelixRegistry**: Participant registration and staking
- **ReputationAnchor**: On-chain reputation scores
- **PaymentRouter**: Contribution-based payments
- **ModelRegistry**: FL model state tracking
- **ContributionRegistry**: Participant contribution records

## Running Tests

```bash
# Run all Rust tests
cargo test

# Run with nextest (faster, better output)
cargo nextest run

# Run specific test suite
cargo test -p fl-aggregator

# Run Python tests
cd 0TML && pytest

# Run contract tests
cd contracts && forge test
```

## Development Workflow

### Building

```bash
# Build all Rust components
cargo build --release

# Build WASM for Holochain
cd zomes && cargo build --release --target wasm32-unknown-unknown

# Build contracts
cd contracts && forge build
```

### Watching for Changes

```bash
# Auto-rebuild on file changes
cargo watch -x check

# Run tests on changes
cargo watch -x test
```

### Code Quality

```bash
# Format code
cargo fmt

# Run linter
cargo clippy

# Security audit
cargo audit
```

## Demo Modes

### Interactive Demo

```bash
demo
# Or: python -m mycelix_cli demo
```

Shows real-time:
- Byzantine attack detection
- Aggregation quality metrics
- Phi (system coherence) evolution
- Shapley contribution scores

### Benchmark Mode

```bash
# Run full benchmark suite
python -m mycelix_cli benchmark

# Run specific benchmark
python -m mycelix_cli benchmark --scenario byzantine
```

### Validation Mode

```bash
# Validate installation
python -m mycelix_cli validate

# Check all systems
python -m mycelix_cli status
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FL_COORDINATOR_HOST` | Coordinator bind address | `127.0.0.1` |
| `FL_COORDINATOR_PORT` | Coordinator port | `3000` |
| `RUST_LOG` | Logging level | `info` |
| `MYCELIX_ENV` | Environment (development/production) | `development` |

### FL Aggregator Config

Edit `config/fl-aggregator.toml`:

```toml
[byzantine]
detection_threshold = 0.7
max_byzantine_ratio = 0.45
enable_adaptive_defense = true

[aggregation]
method = "trimmed_mean"
trim_ratio = 0.1

[privacy]
differential_privacy = true
epsilon = 1.0
```

## Monitoring

### Prometheus Metrics

Access at `http://localhost:9090`:

- `fl_aggregation_rounds_total` - Total rounds completed
- `fl_byzantine_detected_total` - Detected malicious nodes
- `fl_aggregation_quality` - Current quality score
- `fl_phi_value` - System coherence metric

### Grafana Dashboards

Access at `http://localhost:3001`:

- FL Overview Dashboard
- Byzantine Detection Dashboard
- Node Health Dashboard
- Performance Dashboard

## Troubleshooting

### Build Errors

```bash
# Clean and rebuild
cargo clean && cargo build

# Update dependencies
cargo update
```

### Test Failures

```bash
# Run with verbose output
cargo test -- --nocapture

# Run specific failing test
cargo test test_name -- --nocapture
```

### Holochain Issues

```bash
# Check Holochain installation
hc --version

# Reset Holochain state
rm -rf .hc
```

## Next Steps

1. **Explore the Demo**: Run `fl-demo` with different attack types
2. **Read the Vision**: See `docs/ULTIMATE_FL_SYSTEM_VISION.md`
3. **Review the Plan**: Check `docs/COMPREHENSIVE_IMPROVEMENT_PLAN.md`
4. **Dive into Code**: Start with `libs/fl-aggregator/src/lib.rs`

## Resources

- [API Documentation](api/)
- [Whitepaper](../papers/)
- [Architecture Guide](ARCHITECTURE.md)
- [Contributing Guide](CONTRIBUTING.md)

## Support

- GitHub Issues: Report bugs and feature requests
- Discussions: Ask questions and share ideas

---

*Mycelix: Building the future of decentralized machine learning.*
