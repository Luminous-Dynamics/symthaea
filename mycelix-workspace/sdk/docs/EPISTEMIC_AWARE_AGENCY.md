# Epistemic-Aware AI Agency Framework

Developer guide for building AI agents with verifiable trust profiles in the Mycelix ecosystem.

## Overview

The Epistemic-Aware AI Agency Framework enables AI agents to have verifiable "epistemic fingerprints" - trust profiles that prove reliability without revealing internal state. This is achieved through:

1. **K-Vector Trust Profiles** - 10-dimensional trust vectors that evolve based on behavior
2. **Epistemic Classification (E-N-M-H)** - All outputs carry verifiable epistemic levels
3. **Phi Coherence Measurement** - Consciousness-adjacent quality metrics
4. **GIS Uncertainty Handling** - Tripartite moral uncertainty with escalation
5. **KREDIT Allocation** - Resource allocation derived from trust scores
6. **UESS Storage Routing** - Data routed by epistemic classification

## Quick Start

```rust
use mycelix_sdk::agentic::{
    InstrumentalActor, AgentClass, AgentId, AgentConstraints, AgentStatus,
};
use mycelix_sdk::agentic::kvector_bridge::{
    analyze_behavior, compute_kvector_update, calculate_kredit_from_trust,
};
use mycelix_sdk::agentic::epistemic_classifier::{
    create_classified_output, ClassificationHints, calculate_epistemic_weight,
};
use mycelix_sdk::agentic::uncertainty::{
    MoralUncertainty, MoralActionGuidance, maybe_escalate,
};
use mycelix_sdk::matl::KVector;

// 1. Create agent with initial K-Vector
let mut agent = InstrumentalActor {
    agent_id: AgentId::generate(),
    sponsor_did: "did:example:sponsor".to_string(),
    agent_class: AgentClass::Supervised,
    k_vector: KVector::new_participant(),
    // ... other fields
};

// 2. Agent produces output with epistemic classification
let output = create_classified_output(
    agent.agent_id.as_str(),
    OutputContent::Text("Analysis result".to_string()),
    &ClassificationHints {
        has_crypto_proof: true,
        uses_public_data: true,
        ..Default::default()
    },
    timestamp,
);

// 3. Calculate epistemic weight
let weight = calculate_epistemic_weight(&output.classification);
println!("Epistemic weight: {:.4}", weight);

// 4. Express uncertainty
let uncertainty = MoralUncertainty::new(0.2, 0.3, 0.25);
let guidance = MoralActionGuidance::from_uncertainty(&uncertainty);

if guidance.requires_human() {
    let escalation = maybe_escalate(agent_id, &uncertainty, "action", None);
    // Handle escalation...
}

// 5. Update K-Vector from behavior
let analysis = analyze_behavior(&agent.behavior_log);
agent.k_vector = compute_kvector_update(&agent.k_vector, &analysis, &config, days);

// 6. Derive KREDIT from trust
let kredit_cap = calculate_kredit_from_trust(agent.k_vector.trust_score());
```

## K-Vector Trust Profiles

The K-Vector is a 10-dimensional trust profile:

| Dimension | Symbol | Description | Range |
|-----------|--------|-------------|-------|
| Reputation | k_r | Historical reliability | 0.0-1.0 |
| Activity | k_a | Recent engagement level | 0.0-1.0 |
| Integrity | k_i | Constraint adherence | 0.0-1.0 |
| Performance | k_p | Success rate | 0.0-1.0 |
| Membership | k_m | Network tenure | 0.0-1.0 |
| Stake | k_s | Collateral at risk | 0.0-1.0 |
| Historical | k_h | Long-term track record | 0.0-1.0 |
| Topology | k_topo | Network centrality | 0.0-1.0 |
| Verification | k_v | Identity verification | 0.0-1.0 |
| Coherence | k_coherence | Output consistency | 0.0-1.0 |

### Computing Trust Score

```rust
use mycelix_sdk::matl::KVector;

let kv = KVector::new(0.8, 0.6, 0.9, 0.7, 0.5, 0.4, 0.6, 0.5, 0.75, 0.7);
let trust = kv.trust_score(); // Weighted combination of dimensions
```

### Updating from Behavior

```rust
use mycelix_sdk::agentic::kvector_bridge::*;

// Analyze behavior log
let analysis = analyze_behavior(&agent.behavior_log);
println!("Success rate: {:.1}%", analysis.success_rate * 100.0);
println!("Violations: {}", analysis.constraint_violations);

// Compute update
let config = KVectorBridgeConfig::default();
let updated = compute_kvector_update(&old_kv, &analysis, &config, days_active);
```

## Epistemic Classification (E-N-M-H)

All agent outputs are classified along four dimensions:

### Empirical Level (E)

| Level | Name | Description |
|-------|------|-------------|
| E0 | Null | No empirical basis |
| E1 | Testimonial | First-hand account |
| E2 | Private Verify | Third-party verified |
| E3 | Cryptographic | ZK/signature proof |
| E4 | Public Repro | Publicly reproducible |

### Normative Level (N)

| Level | Name | Scope |
|-------|------|-------|
| N0 | Personal | Individual only |
| N1 | Communal | Shared group |
| N2 | Network | Network-wide |
| N3 | Axiomatic | Universal |

### Materiality Level (M)

| Level | Name | Duration |
|-------|------|----------|
| M0 | Ephemeral | Minutes to hours |
| M1 | Temporal | Days to weeks |
| M2 | Persistent | Months to years |
| M3 | Foundational | Indefinite |

### Harmonic Level (H)

| Level | Name | Impact Scope |
|-------|------|--------------|
| H0 | None | No external impact |
| H1 | Local | Immediate context |
| H2 | Network | Network effects |
| H3 | Civilizational | Societal impact |
| H4 | Kosmic | Universal impact |

### Classification Example

```rust
use mycelix_sdk::agentic::epistemic_classifier::*;

let hints = ClassificationHints {
    has_crypto_proof: true,
    uses_public_data: true,
    agreement_scope: Some(AgreementScope::Network),
    relevance_duration: Some(RelevanceDuration::LongTerm),
    affected_harmonies_count: 3,
    ..Default::default()
};

let output = create_classified_output(agent_id, content, &hints, timestamp);

// Classification: E4/N2/M3/H2
println!("E-level: {:?}", output.classification.empirical);
println!("Weight: {:.4}", calculate_epistemic_weight(&output.classification));
```

## Phi Coherence Measurement

Measures consistency of agent outputs using Integrated Information Theory concepts.

```rust
use mycelix_sdk::agentic::coherence_bridge::*;

let config = CoherenceMeasurementConfig::default();
let result = measure_coherence(&outputs, &config)?;

println!("Phi: {:.4}", result.phi);
println!("State: {:?}", result.coherence_state);

// Gate high-stakes actions on coherence
match check_coherence_for_action(result.coherence_state, is_high_stakes) {
    CoherenceCheckResult::Allowed => proceed(),
    CoherenceCheckResult::Blocked { reason, threshold } => {
        println!("Action blocked: {}", reason);
    }
}
```

### Coherence States

| State | Phi Range | Implication |
|-------|-----------|-------------|
| Critical | < 0.3 | Block all actions |
| Low | 0.3-0.5 | Block high-stakes |
| Moderate | 0.5-0.7 | Proceed with monitoring |
| Coherent | > 0.7 | Full autonomy |

## GIS Uncertainty Handling

Graceful Ignorance System (GIS) provides tripartite moral uncertainty:

```rust
use mycelix_sdk::agentic::uncertainty::*;

// Create uncertainty assessment
let uncertainty = MoralUncertainty::new(
    0.3,  // Epistemic: uncertainty about facts
    0.7,  // Axiological: uncertainty about values
    0.5,  // Deontic: uncertainty about right action
);

// Get guidance
let guidance = MoralActionGuidance::from_uncertainty(&uncertainty);

match guidance {
    MoralActionGuidance::ProceedConfidently => { /* Low uncertainty */ }
    MoralActionGuidance::ProceedWithMonitoring => { /* Moderate */ }
    MoralActionGuidance::PauseForReflection => { /* High in one dimension */ }
    MoralActionGuidance::SeekConsultation => { /* High overall */ }
    MoralActionGuidance::DeferAction => { /* Extreme uncertainty */ }
}

// Automatic escalation
if let Some(escalation) = maybe_escalate(agent_id, &uncertainty, action, context) {
    println!("Escalation: {}", escalation.blocked_action);
    println!("Recommendations: {:?}", escalation.recommendations);
}
```

### Calibration Tracking

```rust
let mut calibration = UncertaintyCalibration::default();

// Record outcomes
calibration.record(was_uncertain, was_good_outcome);

// Check calibration quality
let score = calibration.calibration_score();
if calibration.is_overconfident() {
    // Agent needs to express more uncertainty
}
```

## UESS Storage Routing

Universal Epistemic Storage System routes data based on E-N-M classification:

```rust
use mycelix_sdk::storage::{EpistemicStorage, StorageBackend};
use mycelix_sdk::epistemic::*;

let storage = EpistemicStorage::default_storage();
let router = storage.router();

// Route based on classification
let classification = EpistemicClassification::new(
    EmpiricalLevel::E3Cryptographic,
    NormativeLevel::N2Network,
    MaterialityLevel::M3Foundational,
);

let tier = router.route(&classification);
println!("Backend: {:?}", tier.backend);      // IPFS
println!("Mutability: {:?}", tier.mutability); // Immutable
println!("Encrypted: {}", tier.encrypted);     // false (N2+)
```

### Routing Rules

| Materiality | Backend | Use Case |
|-------------|---------|----------|
| M0 Ephemeral | Memory | Session data, caches |
| M1 Temporal | Local | User profiles, drafts |
| M2 Persistent | DHT | Verified documents |
| M3 Foundational | IPFS | Proofs, immutable records |

## Multi-Agent Coordination

Trust-weighted consensus for multi-agent decisions:

```rust
use mycelix_sdk::agentic::multi_agent::*;

// Collect votes
let votes = vec![
    AgentVote { agent_id: "agent-1", value: 0.8, confidence: 0.9, epistemic_level: 3, .. },
    AgentVote { agent_id: "agent-2", value: 0.7, confidence: 0.8, epistemic_level: 2, .. },
];

// Compute consensus weighted by trust
let result = compute_consensus(&votes, &agents, &ConsensusConfig::default());

println!("Consensus: {:.3}", result.consensus_value);
println!("Confidence: {:.3}", result.confidence);
println!("Reached: {}", result.consensus_reached);
```

## ZK Proof Integration

Generate and verify gradient quality proofs:

```rust
use mycelix_sdk::fl::prover_integration::*;

// Create prover (auto-detects backend from environment)
let mut prover = ProverIntegration::with_defaults();

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
assert!(receipt.is_valid());
```

### Production Deployment

```bash
# Start prover service
cd spike/zk-prover-service
docker-compose up -d

# Configure SDK
export RISC0_PROVER_URL=http://localhost:3000
export RISC0_PROVER_TIMEOUT_MS=300000
```

## Byzantine Fault Tolerance

RB-BFT achieves 45% Byzantine tolerance through reputation-squared weighting:

```rust
use mycelix_sdk::fl::{RbbftFLBridge, RbbftFLConfig};
use mycelix_sdk::matl::KVector;

let mut bridge = RbbftFLBridge::new(RbbftFLConfig::default());

// Register with K-Vector
bridge.register_participant("client-1", KVector::with_reputation(0.9));

// Submit votes
bridge.submit_vote("client-1", true, Some(gradient), proof_valid)?;

// Check consensus (67% quorum with reputation² weighting)
if bridge.check_consensus() {
    let result = bridge.finalize_round()?;
}
```

## Complete Integration Example

See `sdk/tests/e2e_epistemic_agent_full.rs` for a complete demonstration:

1. Agent creation with K-Vector
2. Output with E-N-M-H classification
3. Phi coherence measurement
4. GIS uncertainty expression
5. Outcome recording
6. K-Vector update
7. KREDIT adjustment
8. Storage routing

## API Reference

### Core Modules

| Module | Description |
|--------|-------------|
| `mycelix_sdk::agentic` | Agent lifecycle, K-Vector, KREDIT |
| `mycelix_sdk::agentic::kvector_bridge` | Behavior → K-Vector mapping |
| `mycelix_sdk::agentic::epistemic_classifier` | E-N-M-H classification |
| `mycelix_sdk::agentic::coherence_bridge` | Coherence measurement |
| `mycelix_sdk::agentic::uncertainty` | GIS uncertainty handling |
| `mycelix_sdk::agentic::multi_agent` | Multi-agent coordination |
| `mycelix_sdk::storage` | UESS storage routing |
| `mycelix_sdk::fl` | Federated learning, ZK proofs |
| `mycelix_sdk::matl` | K-Vector, RBBFT consensus |

### Feature Flags

```toml
[dependencies]
mycelix-sdk = { version = "0.1", features = ["simulation"] }  # Testing
mycelix-sdk = { version = "0.1", features = ["risc0"] }       # Production ZK
```

## Further Reading

- [FL Security Infrastructure](./FL_SECURITY_INFRASTRUCTURE.md)
- [K-Vector Specification](./MATL_KVECTOR.md)
- [RB-BFT Consensus](./RBBFT_CONSENSUS.md)
- [MIP-E-004 Agentic Economy](../docs/06-mips/MIP-E-004_AGENTIC_ECONOMY_FRAMEWORK.md)
