# Praxis Federated Learning Protocol v0.1

This document specifies the federated learning protocol for Mycelix Praxis.

## Overview

Praxis uses federated learning (FL) to enable privacy-preserving collaborative model training. Learners train models locally on their devices and contribute only gradient updates (not raw data) to a collective model.

## Protocol Phases

A federated learning round proceeds through these states:

```
DISCOVER → JOIN → ASSIGN → UPDATE → AGGREGATE → RELEASE → COMPLETED
```

### 1. DISCOVER

**Purpose**: Announce a new FL round

**Coordinator actions**:
- Create `FlRound` entry with:
  - `round_id`: Unique identifier
  - `model_id`: Model being trained
  - `base_model_hash`: Starting model parameters
  - `min_participants`: Minimum required
  - `max_participants`: Maximum allowed
  - `aggregation_method`: "trimmed_mean", "median", etc.
  - `clip_norm`: L2 norm clipping threshold
  - `state`: `DISCOVER`

**Broadcast**: Publish `FlRound` to DHT

**Duration**: 1-24 hours (configurable)

---

### 2. JOIN

**Purpose**: Participants signal intent to join

**Participant actions**:
- Read `FlRound` entry
- Verify capability (sufficient data, compute)
- Create `JoinRequest` link: `FlRound -> AgentPubKey`

**Coordinator actions**:
- Monitor join requests
- When `min_participants` reached, transition to `ASSIGN`
- When `max_participants` reached, close joins

**Validation**:
- Agent must not have joined already
- Round must be in `JOIN` state

**Duration**: Until min/max reached or timeout

---

### 3. ASSIGN

**Purpose**: Coordinator assigns tasks to participants

**Coordinator actions**:
- Select participants (if oversubscribed: random or reputation-based)
- Update `FlRound.current_participants`
- Set `state` to `ASSIGN`

**Participant actions**:
- Receive assignment confirmation
- Download base model (IPFS, Holochain file storage, or other)
- Prepare local training environment

**Duration**: 1-6 hours

---

### 4. UPDATE

**Purpose**: Participants train locally and submit updates

**Participant actions**:
1. Train model locally on private data
2. Compute gradient or parameter delta
3. **Clip gradient** to `clip_norm` using L2 norm clipping
4. (Optional) Add Gaussian noise for differential privacy
5. Compute commitment: `grad_commitment = BLAKE3(gradient)`
6. Create `FlUpdate` entry with:
   - `round_id`
   - `parent_model_hash`
   - `grad_commitment` (public)
   - `clipped_l2_norm`
   - `local_val_loss`
   - `sample_count`
7. Store actual gradient privately (off-DHT or encrypted)
8. Publish `FlUpdate` to DHT

**Validation**:
- `clipped_l2_norm` ≤ round's `clip_norm`
- `parent_model_hash` matches round's `base_model_hash`
- `sample_count` > 0

**Privacy guarantees**:
- Only commitment published (gradient kept private)
- Clipping bounds contribution
- Optional DP adds plausible deniability

**Duration**: 6-48 hours (depends on dataset size)

---

### 5. AGGREGATE

**Purpose**: Coordinator aggregates updates into new model

**Coordinator actions**:
1. Collect all `FlUpdate` entries for round
2. Request actual gradients from participants (peer-to-peer)
3. Verify gradients match commitments
4. Apply robust aggregation:
   - **Trimmed Mean**: Trim top/bottom 10% per dimension, average
   - **Median**: Median per dimension (max robustness)
   - **Weighted Mean**: Weight by `sample_count` or validation loss
5. Update base model with aggregated gradient
6. Compute `aggregated_model_hash`
7. Update `FlRound`:
   - `aggregated_model_hash`
   - `state` → `RELEASE`

**Security**:
- Robust aggregation mitigates poisoning attacks
- Outlier detection (e.g., high loss, abnormal norm)
- Multi-coordinator verification (future)

**Duration**: Minutes to hours (depends on participant count)

---

### 6. RELEASE

**Purpose**: Publish new model for validation and use

**Coordinator actions**:
- Publish new model (IPFS, file storage)
- Create `ModelProvenance` entry linking to FL round
- Update `FlRound`:
  - `state` → `RELEASE`
  - `completed_at` → timestamp

**Participant actions**:
- Download new model
- Validate locally (accuracy, size, hash)
- Optionally: run model on holdout set, report metrics

**Validation**:
- Model hash matches `aggregated_model_hash`
- Model improves over base (or DAO votes to accept regression)

**Duration**: Immediate (async validation by participants)

---

### 7. COMPLETED

**Purpose**: Archive round, prepare for next

**Actions**:
- Mark `FlRound.state` → `COMPLETED`
- Optionally start new round with `aggregated_model_hash` as base

---

## Message Types

### FlRound (Entry)

```rust
{
  round_id: RoundId,
  model_id: String,
  state: RoundState,
  base_model_hash: ModelHash,
  min_participants: u32,
  max_participants: u32,
  current_participants: u32,
  aggregation_method: String,
  clip_norm: f32,
  started_at: Timestamp,
  completed_at: Option<Timestamp>,
  aggregated_model_hash: Option<ModelHash>,
}
```

### FlUpdate (Entry)

```rust
{
  round_id: RoundId,
  model_id: String,
  parent_model_hash: ModelHash,
  grad_commitment: Vec<u8>,        // BLAKE3 hash
  clipped_l2_norm: f32,             // Norm after clipping
  local_val_loss: f32,              // Validation loss
  sample_count: u32,                // Local training samples
  timestamp: Timestamp,
}
```

### GradientRequest (P2P message)

```rust
{
  round_id: RoundId,
  requester: AgentPubKey,           // Coordinator
  update_hash: EntryHash,           // FlUpdate entry
}
```

### GradientResponse (P2P message)

```rust
{
  round_id: RoundId,
  gradient: Vec<f32>,               // Actual gradient
  signature: Signature,             // Proof it's from author
}
```

---

## Aggregation Methods

### Trimmed Mean (Default)

```
1. For each parameter dimension d:
   a. Collect all values: [v1, v2, ..., vn]
   b. Sort values
   c. Trim top/bottom p% (default p=10)
   d. Average remaining values
```

**Robustness**: Tolerates up to p% malicious participants

### Median

```
1. For each dimension d:
   a. Collect all values
   b. Compute median
```

**Robustness**: 50% breakdown point (max robustness)

### Weighted Mean

```
1. For each participant i:
   a. Weight w_i = sample_count_i / total_samples
2. Aggregate: sum(w_i * gradient_i)
```

**Use case**: Heterogeneous data distributions

---

## Privacy Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `clip_norm` | 1.0 | Bound individual contribution (prevent outliers) |
| `epsilon` | None | Differential privacy budget (optional) |
| `delta` | None | DP failure probability (optional) |
| `noise_multiplier` | None | Gaussian noise scale (if DP enabled) |

**Client-side DP** (optional):
```
1. Clip gradient to norm C
2. Add Gaussian noise: N(0, σ²) where σ = C * noise_multiplier
3. σ chosen to satisfy (ε, δ)-DP
```

---

## Security Considerations

### Threats

1. **Poisoning**: Malicious gradients degrade model
2. **Backdoor**: Inject hidden behavior (e.g., misclassify specific inputs)
3. **Sybil**: Multiple identities amplify attack
4. **Inference**: Infer training data from gradients

### Mitigations

1. **Robust aggregation**: Trim outliers
2. **Clipping**: Bound contribution magnitude
3. **Validation**: Coordinator tests model on holdout set
4. **Reputation**: Weight updates by past behavior (future)
5. **Bonding**: Stake tokens to participate (future)
6. **DP**: Add noise to gradients

---

## Future Extensions

- **Asynchronous FL**: Participants update at different rates
- **Hierarchical FL**: Multiple aggregation layers
- **Secure aggregation**: Coordinator can't see individual gradients (MPC)
- **Byzantine-robust methods**: Krum, Bulyan, etc.
- **Model compression**: Sparsification, quantization

---

## References

- McMahan et al. (2017): *Communication-Efficient Learning of Deep Networks from Decentralized Data*
- Kairouz et al. (2021): *Advances and Open Problems in Federated Learning*
- Yin et al. (2018): *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates*

---

**Version**: 0.1.0
**Last Updated**: 2025-11-15
**Status**: Draft
