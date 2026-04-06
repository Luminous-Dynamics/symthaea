# Federated Learning Protocol: Deep Dive

A comprehensive guide to understanding Praxis's privacy-preserving federated learning protocol.

---

## Table of Contents

- [Overview](#overview)
- [The 6-Phase Lifecycle](#the-6-phase-lifecycle)
- [Phase Details & Code Walkthrough](#phase-details--code-walkthrough)
- [Aggregation Algorithms](#aggregation-algorithms)
- [Privacy Mechanisms](#privacy-mechanisms)
- [Security Considerations](#security-considerations)
- [Hands-On Example](#hands-on-example)
- [Advanced Topics](#advanced-topics)

---

## Overview

Praxis's Federated Learning (FL) protocol enables **collaborative model training** without centralizing data. Students train models locally on their devices, share only encrypted model updates, and collectively improve a shared model.

### Key Benefits

- **Privacy**: Raw data never leaves devices
- **Decentralization**: No single point of control or failure
- **Verifiable**: All contributions are cryptographically signed
- **Robust**: Byzantine-fault tolerant aggregation
- **Auditable**: Full provenance tracking on-chain

### Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Student A  │     │  Student B  │     │  Student C  │
│             │     │             │     │             │
│  Local Data │     │  Local Data │     │  Local Data │
│  📊 Train   │     │  📊 Train   │     │  📊 Train   │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       │ Model Updates     │                   │
       │ (gradients only)  │                   │
       └───────────────────┼───────────────────┘
                           │
                    ┌──────▼──────┐
                    │ Aggregator  │
                    │             │
                    │ 🔒 Secure   │
                    │ 📊 Robust   │
                    │ ✅ Verify   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Updated     │
                    │ Global Model│
                    └─────────────┘
```

---

## The 6-Phase Lifecycle

Each FL round progresses through six distinct phases:

```
DISCOVER → JOIN → ASSIGN → UPDATE → AGGREGATE → RELEASE
```

### Phase Summary

| Phase | Purpose | Duration | Who Acts |
|-------|---------|----------|----------|
| **DISCOVER** | Find eligible participants | ~5 min | Coordinator |
| **JOIN** | Students commit to participate | ~15 min | Students |
| **ASSIGN** | Distribute current model | ~2 min | Coordinator |
| **UPDATE** | Local training | ~30 min | Students |
| **AGGREGATE** | Combine updates securely | ~5 min | Aggregators |
| **RELEASE** | Publish new model | ~2 min | Coordinator |

**Total Round Time**: ~60 minutes

---

## Phase Details & Code Walkthrough

### Phase 1: DISCOVER

**Goal**: Find students eligible to participate in this round.

**Eligibility Criteria**:
- Enrolled in the course
- Device meets minimum specs (RAM, compute)
- Available network bandwidth
- Not in another active round

**Code** (`zomes/fl_zome/src/discover.rs`):

```rust
use hdk::prelude::*;
use crate::fl_round::{FlRound, RoundState};

/// Discover eligible participants for a new FL round
#[hdk_extern]
pub fn discover_participants(round_id: String, course_id: String) -> ExternResult<Vec<AgentPubKey>> {
    // Get all students enrolled in course
    let enrolled = get_enrolled_students(course_id)?;

    // Filter by eligibility
    let eligible: Vec<AgentPubKey> = enrolled
        .into_iter()
        .filter(|agent| {
            // Check device capabilities
            check_device_specs(agent) &&
            // Check not in another round
            !is_in_active_round(agent) &&
            // Check minimum data requirement
            has_minimum_data(agent, &course_id)
        })
        .collect();

    // Create round entry
    let round = FlRound {
        round_id: round_id.clone(),
        course_id,
        state: RoundState::Discover,
        discovered_participants: eligible.clone(),
        max_participants: 100,
        created_at: sys_time()?,
    };

    create_entry(&round)?;

    Ok(eligible)
}
```

**Outputs**:
- List of eligible `AgentPubKey`s
- Round record created with state `DISCOVER`

---

### Phase 2: JOIN

**Goal**: Students commit to participate and reserve capacity.

**What Happens**:
1. Students receive invitation signal
2. Review round parameters (model, privacy budget, time commitment)
3. Submit signed commitment
4. Coordinator confirms or rejects (if capacity reached)

**Code** (`zomes/fl_zome/src/join.rs`):

```rust
#[hdk_extern]
pub fn join_round(round_id: String) -> ExternResult<JoinResponse> {
    let my_agent = agent_info()?.agent_latest_pubkey;

    // Get round
    let round: FlRound = get_round(&round_id)?;

    // Verify eligibility
    if !round.discovered_participants.contains(&my_agent) {
        return Err(WasmError::Guest("Not eligible".into()));
    }

    // Check capacity
    if round.current_participants >= round.max_participants {
        return Err(WasmError::Guest("Round full".into()));
    }

    // Create participation record
    let participation = Participation {
        round_id: round_id.clone(),
        agent: my_agent.clone(),
        joined_at: sys_time()?,
        commitment_signature: sign_commitment(&round)?,
    };

    create_entry(&participation)?;

    // Update round participant count
    update_round_participants(&round_id)?;

    Ok(JoinResponse {
        accepted: true,
        model_hash: round.base_model_hash,
        deadline: round.update_deadline,
    })
}
```

**Security Note**: Signed commitments prevent agents from joining multiple rounds simultaneously (Sybil resistance).

---

### Phase 3: ASSIGN

**Goal**: Distribute the current global model to all participants.

**What Happens**:
1. Coordinator publishes model to DHT
2. Participants download and verify model hash
3. Participants initialize local training

**Code** (`zomes/fl_zome/src/assign.rs`):

```rust
#[hdk_extern]
pub fn assign_model(round_id: String, model_hash: EntryHash) -> ExternResult<()> {
    // Verify caller is coordinator
    verify_coordinator()?;

    // Get round
    let mut round: FlRound = get_round(&round_id)?;

    // Update round state
    round.state = RoundState::Assign;
    round.base_model_hash = Some(model_hash.clone());
    round.assign_timestamp = Some(sys_time()?);

    update_entry(round.hash()?, &round)?;

    // Signal all participants
    let participants = round.joined_participants.clone();
    for agent in participants {
        remote_signal(
            agent,
            SignalPayload::ModelAssigned {
                round_id: round_id.clone(),
                model_hash: model_hash.clone(),
            },
        )?;
    }

    Ok(())
}

/// Participants verify and download model
#[hdk_extern]
pub fn download_model(model_hash: EntryHash) -> ExternResult<ModelWeights> {
    let model: Model = get(model_hash, GetOptions::content())?
        .ok_or(WasmError::Guest("Model not found".into()))?
        .entry()
        .to_app_option()?
        .ok_or(WasmError::Guest("Invalid model entry".into()))?;

    // Verify hash matches
    if hash_entry(&model)? != model_hash {
        return Err(WasmError::Guest("Hash mismatch".into()));
    }

    Ok(model.weights)
}
```

---

### Phase 4: UPDATE

**Goal**: Students train locally and compute model updates.

**What Happens**:
1. Load local dataset
2. Initialize model from assigned weights
3. Train for N epochs
4. Compute update (gradient or weight delta)
5. Apply privacy mechanisms (clipping, noise)
6. Submit encrypted update

**Code** (Client-side, TypeScript):

```typescript
// apps/web/src/services/flTraining.ts

async function computeLocalUpdate(
  modelWeights: Float32Array,
  localData: Dataset,
  config: TrainingConfig
): Promise<ModelUpdate> {
  // Initialize model
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [784] }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
  model.setWeights(deserializeWeights(modelWeights));

  // Train locally
  const history = await model.fit(localData.x, localData.y, {
    epochs: config.epochs,
    batchSize: config.batchSize,
    validationSplit: 0.2,
    verbose: 0,
  });

  // Compute update (delta from initial weights)
  const newWeights = model.getWeights();
  const delta = computeWeightDelta(modelWeights, newWeights);

  // Apply gradient clipping (for privacy & robustness)
  const clippedDelta = clipGradients(delta, config.clipNorm);

  // Optional: Add differential privacy noise
  let privateDelta = clippedDelta;
  if (config.differentialPrivacy) {
    const noise = generateDPNoise(
      clippedDelta.length,
      config.epsilon,
      config.delta,
      config.clipNorm
    );
    privateDelta = addNoise(clippedDelta, noise);
  }

  // Compute update quality metrics
  const valLoss = history.history.val_loss[history.history.val_loss.length - 1];
  const updateNorm = computeL2Norm(privateDelta);

  return {
    roundId: config.roundId,
    delta: privateDelta,
    metadata: {
      sampleCount: localData.x.shape[0],
      valLoss,
      updateNorm,
      clipped: updateNorm > config.clipNorm,
    },
  };
}
```

**Rust Code** (Submit update):

```rust
#[hdk_extern]
pub fn submit_update(update: ModelUpdate) -> ExternResult<()> {
    let my_agent = agent_info()?.agent_latest_pubkey;

    // Verify participation
    verify_participant(&update.round_id, &my_agent)?;

    // Sign update for provenance
    let signature = sign(update.hash()?.as_ref())?;

    // Create update entry
    let signed_update = SignedModelUpdate {
        update,
        agent: my_agent.clone(),
        signature,
        submitted_at: sys_time()?,
    };

    create_entry(&signed_update)?;

    Ok(())
}
```

---

### Phase 5: AGGREGATE

**Goal**: Combine all model updates into a single improved model.

**Aggregation Methods** (see [next section](#aggregation-algorithms)):
- **Trimmed Mean**: Robust to outliers, trim top/bottom 10%
- **Median**: Most robust, but slower
- **Weighted Mean**: Weight by sample count or validation loss

**Code** (`crates/praxis-agg/src/lib.rs`):

```rust
use nalgebra::DVector;

/// Robust aggregation with trimmed mean
pub fn trimmed_mean_aggregation(
    updates: Vec<DVector<f32>>,
    trim_ratio: f32,
) -> DVector<f32> {
    let n_params = updates[0].len();
    let n_updates = updates.len();
    let trim_count = ((n_updates as f32) * trim_ratio).floor() as usize;

    let mut aggregated = DVector::zeros(n_params);

    for param_idx in 0..n_params {
        // Collect all values for this parameter
        let mut values: Vec<f32> = updates
            .iter()
            .map(|u| u[param_idx])
            .collect();

        // Sort
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Trim extremes
        let trimmed = &values[trim_count..(n_updates - trim_count)];

        // Mean of trimmed values
        let mean: f32 = trimmed.iter().sum::<f32>() / (trimmed.len() as f32);
        aggregated[param_idx] = mean;
    }

    aggregated
}
```

**Zome Function**:

```rust
#[hdk_extern]
pub fn aggregate_updates(round_id: String) -> ExternResult<EntryHash> {
    // Verify coordinator
    verify_coordinator()?;

    // Get all submitted updates
    let updates: Vec<SignedModelUpdate> = get_round_updates(&round_id)?;

    // Extract deltas
    let deltas: Vec<DVector<f32>> = updates
        .iter()
        .map(|u| deserialize_delta(&u.update.delta))
        .collect::<Result<_, _>>()?;

    // Aggregate with trimmed mean
    let aggregated = trimmed_mean_aggregation(deltas, 0.1);

    // Apply to base model
    let base_model = get_base_model(&round_id)?;
    let new_weights = apply_delta(&base_model.weights, &aggregated);

    // Create new model entry
    let new_model = Model {
        round_id: round_id.clone(),
        weights: serialize_weights(&new_weights),
        provenance: Provenance {
            contributor_count: updates.len(),
            aggregation_method: "trimmed_mean".into(),
            update_quality_metrics: compute_quality_metrics(&updates),
        },
        created_at: sys_time()?,
    };

    let model_hash = create_entry(&new_model)?;

    // Update round
    let mut round: FlRound = get_round(&round_id)?;
    round.state = RoundState::Aggregate;
    round.aggregated_model_hash = Some(model_hash.clone());
    update_entry(round.hash()?, &round)?;

    Ok(model_hash)
}
```

---

### Phase 6: RELEASE

**Goal**: Publish the new global model and issue credentials.

**What Happens**:
1. New model hash announced
2. Participants download and verify
3. Credentials issued to contributors
4. Round marked as COMPLETED

**Code** (`zomes/fl_zome/src/release.rs`):

```rust
#[hdk_extern]
pub fn release_model(round_id: String) -> ExternResult<()> {
    verify_coordinator()?;

    let mut round: FlRound = get_round(&round_id)?;

    // Ensure aggregation is complete
    let model_hash = round.aggregated_model_hash
        .ok_or(WasmError::Guest("No aggregated model".into()))?;

    // Update round state
    round.state = RoundState::Release;
    round.release_timestamp = Some(sys_time()?);
    update_entry(round.hash()?, &round)?;

    // Issue credentials to contributors
    let contributors = get_round_contributors(&round_id)?;
    for agent in contributors {
        issue_fl_credential(
            agent.clone(),
            &round_id,
            &round.course_id,
            &model_hash,
        )?;
    }

    // Signal all participants
    remote_signal_all(
        &round.joined_participants,
        SignalPayload::ModelReleased {
            round_id: round_id.clone(),
            model_hash,
        },
    )?;

    Ok(())
}
```

**Credential Issuance**:

```rust
// zomes/credential_zome/src/lib.rs

#[hdk_extern]
pub fn issue_fl_credential(
    agent: AgentPubKey,
    round_id: &str,
    course_id: &str,
    model_hash: &EntryHash,
) -> ExternResult<()> {
    let credential = VerifiableCredential {
        context: vec!["https://www.w3.org/2018/credentials/v1".into()],
        id: format!("urn:uuid:{}", uuid::Uuid::new_v4()),
        credential_type: vec!["VerifiableCredential".into(), "FLContribution".into()],
        issuer: "did:holo:praxis".into(),
        issuance_date: chrono::Utc::now().to_rfc3339(),
        credential_subject: CredentialSubject {
            id: agent_to_did(&agent),
            course_id: course_id.to_string(),
            achievement_type: "FL Round Contribution".into(),
            round_id: round_id.to_string(),
            provenance_hash: model_hash.to_string(),
        },
        proof: generate_proof(&agent, model_hash)?,
    };

    create_entry(&credential)?;

    Ok(())
}
```

---

## Aggregation Algorithms

Praxis supports multiple aggregation algorithms for different threat models.

### 1. Trimmed Mean (Default)

**How it works**:
- For each parameter, sort all updates
- Remove top/bottom 10% (configurable)
- Average the remaining middle 80%

**Pros**:
- Robust to outliers and poisoning attacks
- Faster than median
- Good balance of security and efficiency

**Cons**:
- Can lose information from legitimate extremes
- Still vulnerable to coordinated attacks (>10% malicious)

**Use when**: General-purpose FL with untrusted participants

### 2. Median Aggregation

**How it works**:
- For each parameter, take the median across all updates

**Pros**:
- Most robust to Byzantine attacks
- Works even with 49% malicious participants
- No hyperparameters to tune

**Cons**:
- Slower (O(n log n) sorting per parameter)
- Discards more information

**Use when**: High-stakes training with potential adversaries

### 3. Weighted Mean

**How it works**:
- Weight each update by sample count or validation loss
- Participants with more data or better performance have higher influence

**Pros**:
- Faster convergence
- Rewards high-quality contributions

**Cons**:
- Vulnerable to poisoning (malicious agents can claim high weight)
- Requires trust in reported weights

**Use when**: Trusted participants, optimize for speed

### Comparison

| Algorithm | Robustness | Speed | Byzantine Tolerance |
|-----------|------------|-------|---------------------|
| Trimmed Mean | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | <10% malicious |
| Median | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | <50% malicious |
| Weighted Mean | ⭐⭐ | ⭐⭐⭐⭐⭐ | None |

---

## Privacy Mechanisms

### 1. Gradient Clipping

**Goal**: Limit influence of any single participant.

**How it works**:
```rust
pub fn clip_gradients(grad: &DVector<f32>, max_norm: f32) -> DVector<f32> {
    let norm = grad.norm();
    if norm > max_norm {
        grad * (max_norm / norm)
    } else {
        grad.clone()
    }
}
```

**Why important**:
- Prevents large gradients from leaking sensitive info
- Required for differential privacy
- Bounds contribution to aggregated model

**Typical value**: `max_norm = 1.0` or `2.0`

### 2. Differential Privacy (Optional)

**Goal**: Formal privacy guarantee - no single data point can be identified.

**How it works** (Gaussian Mechanism):
```rust
pub fn add_dp_noise(
    grad: &DVector<f32>,
    epsilon: f64,
    delta: f64,
    clip_norm: f32,
) -> DVector<f32> {
    // Calculate noise scale from privacy budget
    let sigma = (2.0 * clip_norm.powi(2) * (1.25 / delta).ln()) / epsilon;

    // Sample Gaussian noise for each parameter
    let noise: DVector<f32> = DVector::from_fn(grad.len(), |_, _| {
        sample_gaussian(0.0, sigma)
    });

    grad + noise
}
```

**Privacy Budget**:
- **ε (epsilon)**: Lower = more private (typical: 0.1 - 10)
- **δ (delta)**: Failure probability (typical: 1e-5)

**Tradeoff**: Privacy ↔ Accuracy
- High privacy (low ε) → more noise → slower convergence
- Low privacy (high ε) → less noise → faster convergence

**Use when**: Training on highly sensitive data (medical, financial)

### 3. Secure Aggregation (Future)

**Goal**: Coordinator cannot see individual updates.

**How it works** (Multi-Party Computation):
1. Each participant secret-shares their update
2. Shares distributed to multiple aggregators
3. Aggregators compute on encrypted shares
4. Only sum is revealed, not individual updates

**Status**: Planned for Phase 6+ (requires MPC library integration)

---

## Security Considerations

### Threat Model

**Trusted**:
- Holochain DHT (data availability, integrity)
- Cryptographic primitives (signatures, hashes)

**Untrusted**:
- Participants (may submit malicious updates)
- Coordinators (should not learn individual updates)

### Attack Vectors & Defenses

| Attack | Description | Defense |
|--------|-------------|---------|
| **Model Poisoning** | Submit gradients that degrade model | Trimmed mean/median aggregation |
| **Backdoor Injection** | Embed trigger that causes misclassification | Update validation, clipping |
| **Gradient Leakage** | Infer training data from gradients | Differential privacy, secure aggregation |
| **Sybil Attack** | Single agent joins multiple times | Signed commitments, identity verification |
| **Free-Riding** | Join but don't train (submit random updates) | Update quality metrics, reputation |

### Best Practices

1. **Always clip gradients** - Even without DP, limits attack surface
2. **Use robust aggregation** - Trimmed mean or median
3. **Validate updates** - Check norms, loss improvements
4. **Monitor outliers** - Flag suspiciously large/small updates
5. **Reputation system** - Track participant quality over time
6. **Regular audits** - Review provenance data for anomalies

---

## Hands-On Example

Let's simulate a complete FL round locally.

### Setup

```bash
# Clone repository
git clone https://github.com/Luminous-Dynamics/mycelix-praxis.git
cd mycelix-praxis

# Build project
make build

# Run tests
make test
```

### Simulate Round

Create `examples/simulate_fl_round.rs`:

```rust
use praxis_core::fl_round::*;
use praxis_agg::trimmed_mean_aggregation;
use nalgebra::DVector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Simulating FL Round\n");

    // Phase 1: DISCOVER
    println!("Phase 1: DISCOVER");
    let participants = vec!["Alice", "Bob", "Charlie", "Diana", "Eve"];
    println!("  Discovered {} participants\n", participants.len());

    // Phase 2: JOIN
    println!("Phase 2: JOIN");
    let joined = participants[..4].to_vec(); // Eve doesn't join
    println!("  {} participants joined\n", joined.len());

    // Phase 3: ASSIGN
    println!("Phase 3: ASSIGN");
    let base_model = DVector::from_vec(vec![0.5, 0.3, 0.1, 0.2]);
    println!("  Assigned model: {:?}\n", base_model);

    // Phase 4: UPDATE
    println!("Phase 4: UPDATE");
    let updates = vec![
        DVector::from_vec(vec![0.1, 0.05, 0.02, 0.03]),  // Alice
        DVector::from_vec(vec![0.12, 0.04, 0.01, 0.04]), // Bob
        DVector::from_vec(vec![0.09, 0.06, 0.03, 0.02]), // Charlie
        DVector::from_vec(vec![0.11, 0.05, 0.02, 0.03]), // Diana
    ];
    println!("  Received {} updates\n", updates.len());

    // Phase 5: AGGREGATE
    println!("Phase 5: AGGREGATE");
    let aggregated = trimmed_mean_aggregation(updates, 0.0); // No trimming for small set
    println!("  Aggregated update: {:?}", aggregated);

    let new_model = base_model + aggregated;
    println!("  New model: {:?}\n", new_model);

    // Phase 6: RELEASE
    println!("Phase 6: RELEASE");
    println!("  Model released!");
    println!("  Issued credentials to {} contributors\n", joined.len());

    println!("✅ Round completed successfully!");

    Ok(())
}
```

**Run**:

```bash
cargo run --example simulate_fl_round
```

**Output**:

```
🚀 Simulating FL Round

Phase 1: DISCOVER
  Discovered 5 participants

Phase 2: JOIN
  4 participants joined

Phase 3: ASSIGN
  Assigned model: [0.5, 0.3, 0.1, 0.2]

Phase 4: UPDATE
  Received 4 updates

Phase 5: AGGREGATE
  Aggregated update: [0.105, 0.05, 0.02, 0.03]
  New model: [0.605, 0.35, 0.12, 0.23]

Phase 6: RELEASE
  Model released!
  Issued credentials to 4 contributors

✅ Round completed successfully!
```

---

## Advanced Topics

### Custom Aggregation

Implement your own aggregation algorithm:

```rust
use praxis_agg::Aggregator;

pub struct CustomAggregator {
    alpha: f32,
}

impl Aggregator for CustomAggregator {
    fn aggregate(&self, updates: Vec<DVector<f32>>) -> DVector<f32> {
        // Your logic here
        // Example: Exponentially weighted moving average
        updates.iter().enumerate().fold(
            DVector::zeros(updates[0].len()),
            |acc, (i, update)| {
                let weight = self.alpha.powi(i as i32);
                acc + update * weight
            }
        )
    }
}
```

### Model Checkpointing

Save intermediate models for analysis:

```rust
#[hdk_extern]
pub fn create_checkpoint(round_id: String, epoch: u32) -> ExternResult<EntryHash> {
    let model = get_current_model(&round_id)?;

    let checkpoint = Checkpoint {
        round_id,
        epoch,
        model_hash: model.hash()?,
        metrics: compute_metrics(&model)?,
        timestamp: sys_time()?,
    };

    create_entry(&checkpoint)
}
```

### Adaptive Privacy Budgets

Adjust ε dynamically based on model convergence:

```rust
fn compute_epsilon(round: u32, total_rounds: u32, initial_eps: f64) -> f64 {
    // Decrease privacy as training progresses (early rounds more private)
    let progress = round as f64 / total_rounds as f64;
    initial_eps * (1.0 + progress)
}
```

---

## Further Reading

- **Federated Learning**: [Google AI Blog](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
- **Differential Privacy**: [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- **Byzantine-Robust Aggregation**: [Krum Paper](https://papers.nips.cc/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf)
- **Holochain**: [Holochain Docs](https://developer.holochain.org/)

---

## Questions?

- 💬 [GitHub Discussions](https://github.com/Luminous-Dynamics/mycelix-praxis/discussions)
- 🐛 [Report Issues](https://github.com/Luminous-Dynamics/mycelix-praxis/issues)
- 📧 Contact: [praxis@luminous-dynamics.com](mailto:praxis@luminous-dynamics.com)

---

*Last updated: 2025-11-15*
