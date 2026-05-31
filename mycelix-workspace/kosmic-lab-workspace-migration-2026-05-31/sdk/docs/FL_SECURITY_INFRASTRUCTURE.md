# Federated Learning Security Infrastructure

This document describes the security infrastructure for Byzantine-resistant federated learning in the Mycelix ecosystem.

## Overview

The FL security infrastructure provides three complementary layers of protection:

| Layer | Component | Protection |
|-------|-----------|------------|
| **Cryptographic** | ZK Proof Bridge | Gradient quality verified in zkVM |
| **Game-Theoretic** | RB-BFT Bridge | 45% Byzantine tolerance via reputation |
| **Combined** | Unified ZK-RBBFT | Both guarantees simultaneously |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FL Security Pipeline                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. GRADIENT SUBMISSION        2. VERIFICATION         3. AGGREGATION      │
│  ┌─────────────────────┐      ┌─────────────────┐     ┌─────────────────┐  │
│  │ Client generates    │      │ ZK Proof        │     │ Reputation²     │  │
│  │ gradient + ZK proof │ ───▶ │ Verified        │ ──▶ │ Weighted Avg    │  │
│  │ + K-Vector attached │      │ in zkVM         │     │ (45% tolerant)  │  │
│  └─────────────────────┘      └─────────────────┘     └─────────────────┘  │
│                                      │                        │             │
│                                      ▼                        ▼             │
│                              ┌─────────────────┐     ┌─────────────────┐   │
│                              │ Quality Check   │     │ Consensus       │   │
│                              │ (norm, bounds)  │     │ (67% quorum)    │   │
│                              └─────────────────┘     └─────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. RB-BFT Bridge (`rbbft_bridge.rs`)

Reputation-Based Byzantine Fault Tolerance achieves 45% Byzantine tolerance through quadratic reputation weighting.

#### Key Properties

- **45% Byzantine Tolerance**: Revolutionary improvement over classical 33%
- **Reputation² Weighting**: High-reputation participants have quadratically more influence
- **K-Vector Integration**: Trust scores from MATL drive aggregation weights
- **Consensus Voting**: 67% quorum threshold for round acceptance

#### Usage

```rust
use mycelix_sdk::fl::{RbbftFLBridge, RbbftFLConfig};
use mycelix_sdk::matl::KVector;

let mut bridge = RbbftFLBridge::new(RbbftFLConfig::default());

// Register participants with K-Vectors
bridge.register_participant("client-1", KVector::with_reputation(0.9));
bridge.register_participant("client-2", KVector::with_reputation(0.8));
bridge.register_participant("client-3", KVector::with_reputation(0.7));

// Start round
bridge.start_round([0u8; 32])?;

// Submit votes with gradients
bridge.submit_vote("client-1", true, Some(gradient), proof_valid)?;

// Check consensus and finalize
if bridge.check_consensus() {
    let result = bridge.finalize_round()?;
}
```

#### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_reputation` | 0.3 | Minimum K-Vector reputation to participate |
| `min_participants` | 3 | Minimum participants for consensus |
| `byzantine_threshold` | 0.45 | Maximum tolerable Byzantine fraction |
| `quorum_threshold` | 0.667 | Required weighted approval ratio |
| `use_quadratic_weighting` | true | Use reputation² for voting weight |

### 2. ZK Proof Bridge (`zkproof_bridge.rs`)

Zero-knowledge proofs provide cryptographic guarantees of gradient quality without revealing gradient values.

#### Key Properties

- **Privacy-Preserving**: Gradient values never leave the prover
- **Cryptographic Soundness**: Invalid gradients cannot produce valid proofs
- **Quality Verification**: Norm bounds, element magnitudes checked in zkVM
- **Hash Commitments**: Gradients bound to proofs cryptographically

#### Proof Statement

```
"I trained on my local data D_i for E epochs with learning rate η,
starting from global model W_t, and this gradient is the result."
```

The proof verifies:
1. Gradient has valid L2 norm (within min/max bounds)
2. No individual element exceeds magnitude threshold
3. Gradient dimension meets minimum requirement
4. All values are finite (no NaN/Inf)

#### Usage

```rust
use mycelix_sdk::fl::{ZKProofFLBridge, VerifiedAggregationMethod};

let mut bridge = ZKProofFLBridge::new()
    .with_aggregation(VerifiedAggregationMethod::FedAvg);

bridge.start_round(1, model_hash);

// Submit gradient with proof (proof generated internally)
bridge.submit_with_proof("client-1", &gradient, 5, 0.01, 32, 0.5)?;

// Aggregate only verified gradients
let result = bridge.aggregate_verified()?;
println!("Excluded {} invalid participants", result.excluded_count);
```

### 3. Unified ZK-RBBFT Bridge (`unified_zkrbbft_bridge.rs`)

Combines ZK proofs with RB-BFT consensus for maximum security.

#### Security Properties

| Property | Guarantee |
|----------|-----------|
| **Cryptographic Quality** | Gradients verified inside zkVM |
| **45% Byzantine Tolerance** | Reputation² weighting allows tolerating up to 45% malicious |
| **Privacy-Preserving** | Gradient values never leave the prover |
| **Accountability** | All actions tied to K-Vector trust profiles |

#### Usage

```rust
use mycelix_sdk::fl::{UnifiedZkRbbftBridge, UnifiedZkRbbftConfig};
use mycelix_sdk::matl::KVector;

let mut bridge = UnifiedZkRbbftBridge::new(UnifiedZkRbbftConfig::production());

// Register participants
bridge.register_participant("client-1", KVector::with_reputation(0.9));

// Start round
bridge.start_round(model_hash)?;

// Submit gradient with ZK proof
bridge.submit_proven_gradient(
    "client-1",
    &gradient,
    epochs,
    learning_rate,
    batch_size,
    loss,
)?;

// Check consensus and finalize
if bridge.check_consensus() {
    let result = bridge.finalize_round()?;
    println!("Aggregated {} valid submissions", result.valid_submissions);
    println!("Byzantine fraction: {:.2}%", result.byzantine_fraction * 100.0);
}
```

### 4. Prover Integration (`prover_integration.rs`)

Production deployment options for ZK proof generation.

#### Backends

| Backend | Use Case | Security |
|---------|----------|----------|
| `Simulation` | Testing only | **NONE** |
| `ExternalService` | Self-hosted prover | Full |
| `Bonsai` | Cloud proving | Full |
| `LocalRisc0` | Development | Full |

#### Environment Variables

```bash
# External prover service
export RISC0_PROVER_URL="http://prover.mycelix.network:3000"
export RISC0_PROVER_TIMEOUT_MS="300000"

# Bonsai cloud proving
export BONSAI_API_KEY="your-api-key"
export BONSAI_API_URL="https://api.bonsai.xyz"  # optional
```

#### Usage

```rust
use mycelix_sdk::fl::prover_integration::{ProverIntegration, ProverBackend};

// Auto-detect backend from environment
let mut prover = ProverIntegration::with_defaults();

// Or explicitly configure
let mut prover = ProverIntegration::new(ProverBackend::ExternalService {
    url: "http://localhost:3000".to_string(),
    timeout_ms: 300_000,
});

// Generate proof
let receipt = prover.prove_gradient(
    &gradient,
    &model_hash,
    epochs,
    learning_rate,
    client_id,
    round,
)?;

// Verify
assert!(prover.verify(&receipt));
```

## Threat Model

### Protected Against

1. **Byzantine Gradient Attacks**: Invalid gradients detected by ZK proofs
2. **Sybil Attacks**: Reputation² weighting limits influence of new identities
3. **Model Poisoning**: Quality constraints prevent extreme gradients
4. **Privacy Attacks**: Zero-knowledge proofs reveal nothing about training data
5. **Free-Riding**: Must prove actual training occurred

### Assumptions

1. **Honest Majority by Reputation**: Participants with >55% total reputation² are honest
2. **Cryptographic Soundness**: RISC-0 zkVM provides 128-bit security
3. **Reputation Accuracy**: K-Vector accurately reflects trustworthiness
4. **Network Liveness**: Messages eventually delivered (async safety)

### Limitations

1. **Adaptive Adversaries**: Sophisticated adversaries may game reputation over time
2. **Collusion**: Colluding high-reputation nodes could coordinate attacks
3. **Proof Overhead**: ZK proof generation adds computational cost
4. **Model Integrity**: Cannot prove model was correctly initialized

## Performance Characteristics

### Proof Generation Time

| Gradient Size | Simulation | Local RISC-0 | Bonsai |
|--------------|------------|--------------|--------|
| 1K params | <1ms | ~2s | ~5s |
| 10K params | <1ms | ~5s | ~10s |
| 100K params | <1ms | ~20s | ~30s |
| 1M params | <1ms | ~60s | ~90s |

### Aggregation Overhead

The reputation² weighting adds O(n) overhead where n is participant count:
- Weight calculation: O(n)
- Consensus check: O(n)
- Aggregation: O(n × d) where d is gradient dimension

## Production Deployment Checklist

- [ ] Deploy RISC-0 prover service or configure Bonsai
- [ ] Set `RISC0_PROVER_URL` or `BONSAI_API_KEY` environment variables
- [ ] Use `UnifiedZkRbbftConfig::production()` configuration
- [ ] Enable TLS for prover service communication
- [ ] Monitor Byzantine fraction in aggregation results
- [ ] Set appropriate timeout for proof generation
- [ ] Configure minimum reputation threshold based on network maturity

### Deploying the Prover Service

The ZK Prover Service provides an HTTP API for proof generation:

```bash
# Using Docker Compose
cd spike/zk-prover-service
docker-compose up -d

# With GPU acceleration (NVIDIA)
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Test the service
curl http://localhost:3000/health
```

Configure clients to use the service:

```bash
export RISC0_PROVER_URL="http://localhost:3000"
export RISC0_PROVER_TIMEOUT_MS="300000"  # 5 minutes
```

See `spike/zk-prover-service/README.md` for full API documentation.

## Feature Flags

```toml
[dependencies]
mycelix-sdk = { version = "0.1", features = ["risc0"] }  # Production
# or
mycelix-sdk = { version = "0.1", features = ["simulation"] }  # Testing only
```

**WARNING**: Never use `simulation` feature in production. It provides NO cryptographic guarantees.

## Related Documentation

- [MATL K-Vector Specification](./MATL_KVECTOR.md)
- [RB-BFT Consensus Protocol](./RBBFT_CONSENSUS.md)
- [ZK Gradient Proof Circuit](../zk-gradient-proof/README.md)
- [MIP-E-004 Agentic Economy Framework](../docs/06-mips/MIP-E-004_AGENTIC_ECONOMY_FRAMEWORK.md)
