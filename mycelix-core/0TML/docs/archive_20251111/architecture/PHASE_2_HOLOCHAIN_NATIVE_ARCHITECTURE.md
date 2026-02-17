# 🏗️ Phase 2: Holochain-Native Architecture Roadmap

**Status**: PLANNING (Post Phase 1 Python implementation)
**Timeline**: 2-3 months after Phase 1 completion
**Goal**: Migrate Byzantine detection from Python orchestration to native Holochain zomes

---

## 🎯 Architectural Vision

### Current State (Phase 1)
```
┌─────────────────────────────────────┐
│   Python Harness (Test Framework)  │
│  - RB-BFT Logic                     │
│  - ML Detector                      │
│  - Gradient Analysis                │
│  - Aggregation                      │
└──────────────┬──────────────────────┘
               │ (RPC calls)
               ▼
┌─────────────────────────────────────┐
│   Holochain (DHT Storage Only)     │
│  - Store gradients                  │
│  - Retrieve gradients               │
│  - P2P gossip                       │
└─────────────────────────────────────┘
```

**Problem**: Holochain used as "dumb substrate", not leveraging intrinsic validation

### Target State (Phase 2)
```
┌─────────────────────────────────────────────────────────────┐
│            Holochain Native Architecture                    │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Rust Zome: Byzantine Detection                       │  │
│  │  - Intrinsic validation rules (PoGQ, cosine sim)      │  │
│  │  - Agent-centric reputation (source chain entries)    │  │
│  │  - Decentralized aggregation via gossip               │  │
│  └──────────────┬────────────────────────────────────────┘  │
│                 │                                             │
│                 ▼ (optional call_remote)                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  ML Service (Python/WASM)                             │  │
│  │  - Heavy ML inference                                 │  │
│  │  - Returns Byzantine probability                      │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  DHT: Intrinsic data validity enforced at network level     │
└─────────────────────────────────────────────────────────────┘
```

**Benefits**:
- Byzantine detection is *intrinsic* to the network
- No single point of failure (no central aggregator)
- Aligns with Holochain's agent-centric philosophy
- Scales naturally with network growth

---

## 📊 Component Migration Plan

### 1. Core Validation Rules → Zome Functions

**Python (Current)**:
```python
class PoGQDetector:
    def detect_byzantine(self, gradients):
        cosine_sims = compute_pairwise_cosine(gradients)
        return nodes_outside_range(cosine_sims, min_threshold, max_threshold)
```

**Rust Zome (Target)**:
```rust
#[hdk_extern]
pub fn validate_gradient(gradient: Gradient) -> ExternResult<ValidateCallbackResult> {
    // Get committee gradients from DHT
    let committee = get_committee_for_round(gradient.round)?;

    // Compute PoGQ score
    let pogq_score = compute_pogq_score(&gradient, &committee);

    // Check against thresholds (from gradient profile)
    let profile = get_gradient_profile(gradient.dataset)?;

    if pogq_score < profile.min_threshold || pogq_score > profile.max_threshold {
        return Ok(ValidateCallbackResult::Invalid(
            format!("PoGQ score {} outside acceptable range", pogq_score).into()
        ));
    }

    // Optionally query ML service for additional check
    if let Some(ml_score) = query_ml_service(&gradient)? {
        if ml_score > BYZANTINE_THRESHOLD {
            return Ok(ValidateCallbackResult::Invalid(
                format!("ML detector flagged: {:.2}", ml_score).into()
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}
```

**Benefits**:
- Validation happens *automatically* when gradients are gossiped
- Every peer in the DHT enforces these rules
- Byzantine gradients are rejected at the network level

---

### 2. Reputation System → Source Chain Entries

**Python (Current)**:
```python
class ReputationSystem:
    def __init__(self):
        self.reputation = {}  # node_id -> CIV score

    def update_reputation(self, node_id, is_byzantine):
        if is_byzantine:
            self.reputation[node_id] *= (1 - penalty)
        else:
            self.reputation[node_id] = min(1.0, reputation + recovery_bonus)
```

**Rust Zome (Target)**:
```rust
#[hdk_entry(id = "agent_reputation")]
#[derive(Clone)]
pub struct AgentReputation {
    agent: AgentPubKey,
    civ: f64,
    round: u64,
    behavior_history: Vec<RoundBehavior>,
    warranted_by: Vec<AgentPubKey>,  // Peer attestations
    timestamp: Timestamp,
}

#[hdk_entry(id = "reputation_update")]
pub struct ReputationUpdate {
    agent: AgentPubKey,
    old_civ: f64,
    new_civ: f64,
    reason: UpdateReason,
    evidence: Vec<EntryHash>,  // Links to gradient entries
}

#[hdk_extern]
pub fn update_my_reputation(behavior: RoundBehavior) -> ExternResult<ActionHash> {
    let my_pubkey = agent_info()?.agent_latest_pubkey;

    // Get current reputation
    let current_rep = get_my_latest_reputation()?;

    // Compute new CIV
    let new_civ = compute_new_civ(current_rep.civ, behavior);

    // Create reputation entry
    let new_rep = AgentReputation {
        agent: my_pubkey.clone(),
        civ: new_civ,
        round: behavior.round,
        behavior_history: vec![behavior],
        warranted_by: vec![],  // Peers can add warrants
        timestamp: sys_time()?,
    };

    // Commit to my source chain
    create_entry(&EntryTypes::AgentReputation(new_rep))
}

#[hdk_extern]
pub fn warrant_reputation(agent: AgentPubKey, round: u64) -> ExternResult<()> {
    // Peer attestation: "I observed this agent's behavior in round X"
    // Find their reputation entry
    let rep_entry = get_agent_reputation(agent.clone(), round)?;

    // Add my warrant (signature)
    let my_pubkey = agent_info()?.agent_latest_pubkey;
    create_link(
        rep_entry.hash,
        my_pubkey,
        LinkTypes::ReputationWarrant,
        ()
    )?;

    Ok(())
}
```

**Benefits**:
- Reputation is *publicly auditable* on each agent's source chain
- Peer attestations create accountability
- No central reputation database
- Tamper-evident (changing your chain invalidates all subsequent entries)

---

### 3. Gradient Dimensionality Analyzer → Zome Logic

**Python (Current)**:
```python
class GradientDimensionalityAnalyzer:
    def analyze_gradients(self, gradients):
        effective_dim = compute_pca(gradients)
        return recommend_parameters(effective_dim)
```

**Rust Zome (Target)**:
```rust
#[hdk_entry(id = "gradient_profile")]
pub struct GradientProfile {
    dataset: String,
    nominal_dim: usize,
    effective_dim: f64,
    recommended_thresholds: DetectionThresholds,
    computed_at_round: u64,
}

#[hdk_extern]
pub fn compute_gradient_profile(
    gradients: Vec<Gradient>
) -> ExternResult<GradientProfile> {
    // Perform PCA (or use pre-computed from calibration)
    let nominal_dim = gradients[0].data.len();
    let effective_dim = compute_effective_dimensionality(&gradients);

    // Determine strategy
    let strategy = if effective_dim < 100 {
        DetectionStrategy::LowDim
    } else if effective_dim < 1000 {
        DetectionStrategy::MidDim
    } else {
        DetectionStrategy::HighDim
    };

    // Recommend thresholds
    let thresholds = strategy.get_thresholds();

    let profile = GradientProfile {
        dataset: gradients[0].dataset.clone(),
        nominal_dim,
        effective_dim,
        recommended_thresholds: thresholds,
        computed_at_round: gradients[0].round,
    };

    // Store profile for this dataset
    create_entry(&EntryTypes::GradientProfile(profile.clone()))?;

    Ok(profile)
}
```

**Benefits**:
- Profile computed once per dataset, stored in DHT
- All peers can reference the same profile
- Automatic parameter adaptation without central config

---

### 4. Aggregation → Decentralized Gossip Protocol

**Python (Current)**:
```python
class RBBFTAggregator:
    def aggregate_round(self, gradients, reputations):
        # Single point aggregates all gradients
        trusted_gradients = filter_by_reputation(gradients, reputations)
        return compute_aggregate(trusted_gradients)
```

**Rust Zome (Target)**:
```rust
#[hdk_entry(id = "validation_score")]
pub struct ValidationScore {
    validator: AgentPubKey,
    target: AgentPubKey,
    gradient_hash: EntryHash,
    pogq_score: f64,
    ml_score: Option<f64>,
    is_trusted: bool,
    round: u64,
}

#[hdk_extern]
pub fn gossip_my_validation_scores(
    round: u64
) -> ExternResult<Vec<ActionHash>> {
    // I've validated other agents' gradients
    // Now gossip my assessment to the network
    let my_scores = compute_validation_scores_for_round(round)?;

    let mut created = vec![];
    for score in my_scores {
        let hash = create_entry(&EntryTypes::ValidationScore(score))?;
        created.push(hash);
    }

    Ok(created)
}

#[hdk_extern]
pub fn compute_local_aggregate(round: u64) -> ExternResult<Gradient> {
    // Decentralized aggregation: each agent computes their own

    // 1. Get all gradients for this round
    let all_gradients = get_gradients_for_round(round)?;

    // 2. Get validation scores from trusted peers
    let my_trusted_validators = get_my_trusted_validators()?;
    let validation_scores = get_validation_scores_from(my_trusted_validators, round)?;

    // 3. Filter gradients based on consensus of trusted validators
    let trusted_gradients: Vec<Gradient> = all_gradients
        .into_iter()
        .filter(|g| {
            let scores_for_gradient = validation_scores
                .iter()
                .filter(|s| s.gradient_hash == g.hash)
                .collect::<Vec<_>>();

            // Majority of my trusted validators say this gradient is good
            let trust_count = scores_for_gradient
                .iter()
                .filter(|s| s.is_trusted)
                .count();

            trust_count > (scores_for_gradient.len() / 2)
        })
        .collect();

    // 4. Compute aggregate using only trusted gradients
    Ok(coordinate_median(&trusted_gradients))
}
```

**Benefits**:
- No single aggregator (fully decentralized)
- Each agent computes their own aggregate based on their trust network
- Byzantine-resilient through gossip consensus
- Scales with network size

---

## 🔧 ML Detector Integration Options

### Option A: External Python Service (Pragmatic)

```rust
#[hdk_extern]
pub fn query_ml_service(gradient: &Gradient) -> ExternResult<Option<f64>> {
    // Call external ML service via HTTP
    let ml_endpoint = get_ml_service_endpoint()?;

    let response = http_client::post(
        format!("{}/detect", ml_endpoint),
        serde_json::to_string(gradient)?
    )?;

    let score: MLScore = serde_json::from_str(&response)?;
    Ok(Some(score.byzantine_probability))
}
```

**Pros**: Simple, keeps ML in Python
**Cons**: External dependency, potential SPOF

---

### Option B: WASM ML Model (Ambitious)

```rust
// Compile scikit-learn model to WASM using pyodide or similar

#[hdk_extern]
pub fn ml_detect_byzantine(gradient: &Gradient) -> ExternResult<f64> {
    // Load WASM ML model (embedded in zome)
    let model = load_ml_model_wasm()?;

    // Extract features
    let features = extract_features(gradient);

    // Run inference in WASM
    let score = model.predict(&features)?;

    Ok(score)
}
```

**Pros**: Fully self-contained, no external deps
**Cons**: Complex build process, limited ML library support in WASM

---

## 📅 Implementation Timeline

### Month 1: Core Validation Rules
- **Week 1-2**: Port PoGQ detector to Rust
- **Week 3-4**: Implement gradient profile computation
- **Testing**: Verify validation rules match Python behavior

### Month 2: Reputation & Aggregation
- **Week 5-6**: Implement agent-centric reputation on source chains
- **Week 7-8**: Build decentralized aggregation via gossip
- **Testing**: Multi-node network testing

### Month 3: ML Integration & Polish
- **Week 9-10**: Integrate ML service (Option A) or WASM model (Option B)
- **Week 11**: End-to-end testing with full Byzantine scenarios
- **Week 12**: Performance optimization, documentation

---

## ✅ Success Criteria

1. **Intrinsic Validation**: Byzantine gradients rejected automatically by DHT
2. **No Central Aggregator**: Each agent computes own aggregate
3. **Agent-Centric Reputation**: CIV scores on source chains with peer warrants
4. **Performance**: <100ms validation overhead per gradient
5. **Compatibility**: Matches Python implementation accuracy (85-90%+ detection)

---

## 🎯 Grant Narrative

**Phase 1 (Python)**: "Rapid prototyping validated our Byzantine detection algorithms, achieving 85-90% success rate across diverse datasets and attack scenarios."

**Phase 2 (Rust/Holochain)**: "Production deployment migrates core logic into native Holochain zomes, leveraging intrinsic data validation and agent-centric architecture for true decentralization."

This two-phase approach demonstrates:
- **Research rigor**: Algorithm validation before production implementation
- **Architectural maturity**: Understanding the proper Holochain integration patterns
- **Pragmatic development**: Fast iteration in Phase 1, robust implementation in Phase 2

---

**Status**: Ready to present to grant reviewers as the production roadmap post-Phase 1
