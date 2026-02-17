# FL Aggregator

High-performance Byzantine-resistant federated learning aggregator in Rust.

## Features

- **Byzantine Defense Algorithms**: Krum, MultiKrum, Median, TrimmedMean, GeometricMedian
- **Memory-Bounded Operation**: Configurable limits for large-scale deployments
- **Async Support**: Handle thousands of concurrent nodes with Tokio
- **Gradient Compression**: Top-k sparsification with error feedback for 40x bandwidth reduction
- **Python Bindings**: PyO3 integration for ML pipelines
- **HTTP API**: Optional Axum-based REST API
- **Prometheus Metrics**: Built-in observability

## Quick Start

```rust
use fl_aggregator::{Aggregator, AggregatorConfig, Defense};
use ndarray::Array1;

// Create aggregator with Krum defense (tolerates 1 Byzantine node)
let config = AggregatorConfig::default()
    .with_defense(Defense::Krum { f: 1 })
    .with_expected_nodes(5);

let mut aggregator = Aggregator::new(config);

// Submit gradients from nodes
aggregator.submit("node1", Array1::from(vec![1.0, 2.0, 3.0]))?;
aggregator.submit("node2", Array1::from(vec![1.1, 2.1, 3.1]))?;
// ... more nodes

// Get aggregated result
if aggregator.is_round_complete() {
    let result = aggregator.finalize_round()?;
}
```

## Defense Algorithms

| Algorithm | Description | Byzantine Tolerance |
|-----------|-------------|---------------------|
| FedAvg | Simple averaging | None |
| Krum | Select most central gradient | n >= 2f + 3 |
| MultiKrum | Average top-k central gradients | n >= 2f + 3 |
| Median | Coordinate-wise median | ~50% |
| TrimmedMean | Remove outliers before averaging | Configurable |
| GeometricMedian | Iterative geometric center | ~50% |

## Python Usage

```python
from fl_aggregator import PyAggregator, Defense
import numpy as np

aggregator = PyAggregator(
    expected_nodes=5,
    defense=Defense.krum(f=1)
)

aggregator.submit("node1", np.array([1.0, 2.0, 3.0]))
# ...

if aggregator.is_complete():
    result = aggregator.finalize()
```

## HTTP API

Run the server:

```bash
cargo run --bin fl-server --features http-api
```

Environment variables:
- `FL_EXPECTED_NODES`: Number of nodes per round (default: 10)
- `FL_DEFENSE`: Algorithm (fedavg, krum, multikrum, median, trimmedmean)
- `FL_BYZANTINE_F`: Byzantine tolerance parameter
- `FL_MAX_MEMORY_MB`: Memory limit (default: 2000)
- `FL_BIND_ADDR`: Bind address (default: 0.0.0.0:3000)

Endpoints:
- `GET /health` - Health check
- `GET /status` - Aggregator status
- `GET /metrics/prometheus` - Prometheus metrics
- `POST /nodes/:id` - Register node
- `POST /gradients` - Submit gradient
- `POST /aggregate` - Force aggregation
- `GET /result` - Get aggregated gradient

## Communication Architecture

The FL aggregator supports multiple communication transports. For Mycelix's decentralized
Byzantine-tolerant design, we recommend **HTTP + Holochain**, not GRPC.

### Transport Comparison

| Transport | Use Case | Decentralized? | Status |
|-----------|----------|----------------|--------|
| **HTTP REST** | Client gradient submission | Yes (stateless) | **Primary** |
| **Holochain** | P2P coordination, DHT storage | Yes (native P2P) | **Primary** |
| **GRPC** | Proof generation service | No (client-server) | Optional |

### Why Not GRPC for FL Coordination?

GRPC is fundamentally client-server, which contradicts Byzantine-tolerant design:

```
GRPC (centralized):              Mycelix (decentralized):
     ┌─────────┐                      ┌───┐
     │  Server │                  ┌───│ A │───┐
     └────┬────┘                  │   └───┘   │
          │                     ┌─┴─┐       ┌─┴─┐
    ┌─────┼─────┐               │ B │◄─────►│ C │
  ┌─┴─┐ ┌─┴─┐ ┌─┴─┐             └─┬─┘       └─┬─┘
  │ A │ │ B │ │ C │               └─────┬─────┘
  └───┘ └───┘ └───┘                   ┌─┴─┐
                                      │ D │
  Single point of failure             └───┘
                                  Byzantine-tolerant mesh
```

The GRPC feature is available for specialized services (like proof generation on port 50051)
but should not be used as the primary FL coordination path.

### Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FL Client (Node)                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ POST /gradients → HTTP API → Aggregator             │   │
│  │                      │                               │   │
│  │                      ▼                               │   │
│  │              Holochain DHT (P2P)                    │   │
│  │                      │                               │   │
│  │                      ▼                               │   │
│  │         Byzantine Detection & Aggregation            │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Feature Flags

```toml
[features]
default = ["parallel"]
http-api = ["axum", "tower", "tower-http"]  # Recommended
holochain = ["holochain_client"]             # Recommended for P2P
grpc = ["tonic", "prost"]                    # Optional, specialized use only
```

### Holochain Integration

For truly decentralized FL, use the Holochain zomes directly:

```rust
// In your Holochain coordinator zome
use federated_learning_coordinator::submit_gradient;

// Gradients are stored on the DHT, validated by all nodes
let hash = submit_gradient(SubmitGradientInput {
    node_id: my_node_id,
    gradient: gradient_bytes,
    round: current_round,
})?;
```

See `/zomes/federated_learning/` for the Holochain implementation.

## ZK Proof Integration

The aggregator includes a proof system for vote eligibility verification:

```rust
use fl_aggregator::proofs::integration::{
    ZomeBridgeClient, VoterProfileBuilder, VerifierService,
};

// Generate eligibility proof
let client = ZomeBridgeClient::new();
let profile = VoterProfileBuilder::new("did:mycelix:alice")
    .matl_score(0.75)
    .has_humanity_proof(true)
    .build();

let proof = client.generate_proof(&profile, ProofProposalType::Standard)?;

// Verify and attest
let verifier = VerifierService::with_signing_key(&key)?;
let attestation = verifier.verify_and_attest(&proof)?;
```

See `/src/proofs/integration/zome_bridge.rs` for the governance integration.

## Benchmarks

Run benchmarks:

```bash
cargo bench
```

Expected performance:
- 100x faster than Python for aggregation
- 16x less memory at 1000 nodes
- Scales to 10,000+ nodes

## License

MIT OR Apache-2.0
