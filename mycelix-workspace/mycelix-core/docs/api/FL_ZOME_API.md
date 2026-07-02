# Federated Learning Zome API Documentation

## Overview

The Federated Learning (FL) Zome implements Byzantine-resistant federated machine learning on Holochain 0.6. It integrates with the Mycelix SDK for Proof of Gradient Quality (PoGQ), hierarchical Byzantine detection, cartel detection, and HyperFeel gradient compression with zkSTARK proofs.

## Purpose

- **Byzantine-Resistant FL**: Detect and mitigate adversarial participants using MATL (Multi-Agent Trust Layer)
- **Gradient Submission**: Submit and retrieve model gradients for training rounds
- **Reputation Management**: Track node reputation with Bayesian smoothing
- **Coordinator Bootstrap**: Secure multi-signature coordinator authority (SEC-002)
- **Compressed Gradients**: 2000x bandwidth reduction via HyperFeel encoding
- **Cryptographic Proofs**: zkSTARK verification of honest computation

---

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `MAX_BYZANTINE_TOLERANCE` | `0.45` | Maximum Byzantine fraction (45%) |
| `MAX_NODE_ID_LENGTH` | `256` | Maximum length for node IDs |
| `MAX_SUBMISSIONS_PER_MINUTE` | `60` | Rate limit per agent |
| `MAX_SIGNAL_AGE_MICROSECONDS` | `300000000` | 5-minute signal validity window |
| `HV16_BYTES` | `2048` | HyperFeel hypervector size (2KB) |

---

## Entry Types (Integrity Zome)

### ModelGradient

Represents a gradient submission for federated learning.

```rust
pub struct ModelGradient {
    pub node_id: String,           // Node identifier
    pub round: u32,                // Training round
    pub gradient_hash: String,     // SHA-256 hash of gradient (64 hex chars)
    pub timestamp: i64,            // Unix timestamp (seconds)
    pub cpu_usage: f32,            // CPU utilization [0.0, 1.0]
    pub memory_mb: f32,            // Memory usage in MB
    pub network_latency_ms: f32,   // Network latency in ms
    pub trust_score: Option<f32>,  // Computed trust score [0.0, 1.0]
}
```

**Validation Rules:**
- `node_id`: Non-empty, max 256 chars, alphanumeric + `-_:`
- `gradient_hash`: Exactly 64 hexadecimal characters
- `cpu_usage`: Between 0.0 and 1.0, finite
- `memory_mb`: Non-negative, finite
- `network_latency_ms`: Non-negative, finite
- `trust_score`: If present, between 0.0 and 1.0, finite

---

### TrainingRound

Records the completion of a training round.

```rust
pub struct TrainingRound {
    pub round: u32,                  // Round number
    pub selected_gradient: ActionHash, // Chosen gradient for aggregation
    pub participant_count: u32,      // Total participants
    pub byzantine_detected: bool,    // Whether Byzantine nodes were found
    pub byzantine_count: u32,        // Number of Byzantine nodes
    pub accuracy: f32,               // Model accuracy [0.0, 1.0]
    pub completed_at: i64,           // Completion timestamp
}
```

**Validation Rules:**
- `participant_count`: Must be at least 1
- `byzantine_count`: Cannot exceed `participant_count`
- Byzantine fraction cannot exceed `MAX_BYZANTINE_TOLERANCE` (45%)
- `accuracy`: Between 0.0 and 1.0, finite

---

### ByzantineRecord

Records detected Byzantine behavior.

```rust
pub struct ByzantineRecord {
    pub node_id: String,           // Suspected node
    pub round: u32,                // Round when detected
    pub detection_method: String,  // e.g., "MATL_PoGQ", "hierarchical"
    pub confidence: f32,           // Detection confidence [0.0, 1.0]
    pub evidence_hash: String,     // Hash of evidence data
    pub detected_at: i64,          // Detection timestamp
}
```

---

### NodeReputation

Tracks reputation for a participating node.

```rust
pub struct NodeReputation {
    pub node_id: String,           // Node identifier
    pub successful_rounds: u32,    // Successful participations
    pub failed_rounds: u32,        // Failed/rejected participations
    pub reputation_score: f32,     // Computed score [0.0, 1.0]
    pub last_updated: i64,         // Last update timestamp
}
```

---

### Coordinator Bootstrap Types (SEC-002)

#### GenesisCoordinator
```rust
pub struct GenesisCoordinator {
    pub agent_pubkey: String,      // Coordinator's public key
    pub authority_proof: Vec<u8>,  // Ed25519 signature (min 64 bytes)
    pub dna_modifiers_hash: String, // SHA-256 of DNA modifiers
    pub established_at: i64,
    pub active: bool,
}
```

#### CoordinatorCredential
```rust
pub struct CoordinatorCredential {
    pub agent_pubkey: String,
    pub credential_type: String,   // "genesis", "voted", or "delegated"
    pub issued_at: i64,
    pub expires_at: i64,           // 0 = never expires
    pub guardian_signatures_json: String, // JSON array of signatures
    pub required_signatures: u32,
    pub active: bool,
    pub revoked_at: i64,           // 0 = not revoked
}
```

#### Guardian
```rust
pub struct Guardian {
    pub agent_pubkey: String,
    pub voting_weight: u32,        // Minimum 1
    pub added_at: i64,
    pub added_by: String,          // Who added this guardian
    pub active: bool,
}
```

---

## Link Types

| Link Type | Description |
|-----------|-------------|
| `RoundToGradients` | Round anchor to gradient entries |
| `NodeToGradients` | Node anchor to their gradient submissions |
| `NodeToReputation` | Node anchor to reputation entry |
| `RoundToByzantine` | Round anchor to Byzantine records |
| `GenesisCoordinators` | Path to genesis coordinator entries |
| `Guardians` | Path to guardian entries |
| `AgentToCredential` | Agent to their coordinator credential |
| `AuditLogChain` | Coordinator audit log chain |
| `BootstrapConfiguration` | Path to bootstrap config |

---

## Extern Functions

### Gradient Submission

#### submit_gradient

Submit a basic gradient to the DHT.

```rust
#[hdk_extern]
pub fn submit_gradient(input: SubmitGradientInput) -> ExternResult<ActionHash>
```

**Input:**
```rust
pub struct SubmitGradientInput {
    pub node_id: String,
    pub round: u32,
    pub gradient_hash: String,
    pub cpu_usage: f32,
    pub memory_mb: f32,
    pub network_latency_ms: f32,
    pub trust_score: Option<f32>,
}
```

**Example:**
```javascript
const result = await client.callZome({
  zome_name: "federated_learning",
  fn_name: "submit_gradient",
  payload: {
    node_id: "node-001",
    round: 5,
    gradient_hash: "a1b2c3...64_hex_chars",
    cpu_usage: 0.75,
    memory_mb: 2048.0,
    network_latency_ms: 15.0,
    trust_score: null
  }
});
```

**Security:**
- Rate limited to 60 submissions per minute
- Node ID validated for format and length

---

#### submit_gradient_with_pogq

Submit a gradient with automatic Proof of Gradient Quality computation.

```rust
#[hdk_extern]
pub fn submit_gradient_with_pogq(input: GradientWithPoGQInput) -> ExternResult<PoGQResult>
```

**Input:**
```rust
pub struct GradientWithPoGQInput {
    pub node_id: String,
    pub round: u32,
    pub gradient_hash: String,
    pub cpu_usage: f32,
    pub memory_mb: f32,
    pub network_latency_ms: f32,
    pub quality: f64,       // Gradient quality [0.0, 1.0]
    pub consistency: f64,   // Temporal consistency [0.0, 1.0]
    pub entropy: f64,       // Information entropy
}
```

**Output:**
```rust
pub struct PoGQResult {
    pub action_hash: ActionHash,
    pub pogq: PoGQData,
    pub composite_score: f64,
    pub is_byzantine: bool,
}
```

**Behavior:**
1. Computes PoGQ from quality, consistency, and entropy
2. Looks up node reputation (defaults to 0.5 for new nodes)
3. Calculates composite trust score
4. Automatically records Byzantine detection if threshold exceeded
5. Emits appropriate signals

---

#### submit_compressed_gradient

Submit a HyperFeel-compressed gradient with zkSTARK proof.

```rust
#[hdk_extern]
pub fn submit_compressed_gradient(input: SubmitCompressedGradientInput) -> ExternResult<CompressedGradientResult>
```

**Input:**
```rust
pub struct SubmitCompressedGradientInput {
    pub node_id: String,
    pub round: u32,
    pub hypervector: Vec<u8>,     // 2KB HyperFeel vector
    pub proof_bytes: Vec<u8>,     // zkSTARK proof
    pub original_size: usize,     // Original gradient size
    pub quality_score: f32,
    pub epochs: u32,
    pub learning_rate: f32,
    pub cpu_usage: f32,
    pub memory_mb: f32,
    pub network_latency_ms: f32,
}
```

**Output:**
```rust
pub struct CompressedGradientResult {
    pub action_hash: ActionHash,
    pub proof_verified: bool,
    pub trust_score: f32,
    pub compression_ratio: f32,
}
```

**Benefits:**
- 2000x bandwidth reduction via HyperFeel encoding
- Cryptographic proof of honest computation
- Automatic trust scoring

---

### Gradient Retrieval

#### get_round_gradients

Get all gradients for a training round.

```rust
#[hdk_extern]
pub fn get_round_gradients(round: u32) -> ExternResult<Vec<(ActionHash, ModelGradient)>>
```

**Example:**
```javascript
const gradients = await client.callZome({
  zome_name: "federated_learning",
  fn_name: "get_round_gradients",
  payload: 5
});
// gradients: [[ActionHash, {node_id: "...", round: 5, ...}], ...]
```

---

#### get_node_gradients

Get all gradients from a specific node.

```rust
#[hdk_extern]
pub fn get_node_gradients(node_id: String) -> ExternResult<Vec<(ActionHash, ModelGradient)>>
```

---

#### get_round_hypervectors

Get all HyperFeel hypervectors for a round.

```rust
#[hdk_extern]
pub fn get_round_hypervectors(round: u32) -> ExternResult<Vec<(String, Vec<u8>)>>
```

Returns: `Vec<(node_id, hypervector)>`

---

#### aggregate_round_hypervectors

Aggregate all hypervectors for a round using majority voting.

```rust
#[hdk_extern]
pub fn aggregate_round_hypervectors(round: u32) -> ExternResult<Vec<u8>>
```

Returns: A single 2KB aggregated hypervector.

---

### Round Management

#### complete_round

Complete a training round (coordinator only).

```rust
#[hdk_extern]
pub fn complete_round(input: CompleteRoundInput) -> ExternResult<ActionHash>
```

**Input:**
```rust
pub struct CompleteRoundInput {
    pub round: u32,
    pub selected_gradient: ActionHash,
    pub participant_count: u32,
    pub byzantine_detected: bool,
    pub byzantine_count: u32,
    pub accuracy: f32,
}
```

**Security:** Requires coordinator role.

---

#### get_round_matl_stats

Get comprehensive MATL statistics for a round.

```rust
#[hdk_extern]
pub fn get_round_matl_stats(round: u32) -> ExternResult<RoundMATLStats>
```

**Output:**
```rust
pub struct RoundMATLStats {
    pub round: u32,
    pub participant_count: usize,
    pub average_quality: f64,
    pub average_consistency: f64,
    pub average_composite: f64,
    pub byzantine_count: usize,
    pub byzantine_fraction: f64,
    pub within_tolerance: bool,
}
```

---

### Byzantine Detection

#### record_byzantine

Record a Byzantine detection (detector role only).

```rust
#[hdk_extern]
pub fn record_byzantine(input: ByzantineRecordInput) -> ExternResult<ActionHash>
```

**Input:**
```rust
pub struct ByzantineRecordInput {
    pub node_id: String,
    pub round: u32,
    pub detection_method: String,
    pub confidence: f32,
    pub evidence_hash: String,
}
```

**Security:** Requires detector role or coordinator role.

---

#### detect_byzantine_hierarchical

Detect Byzantine nodes using hierarchical clustering (O(n log n)).

```rust
#[hdk_extern]
pub fn detect_byzantine_hierarchical(input: DetectByzantineInput) -> ExternResult<ByzantineDetectionResult>
```

**Input:**
```rust
pub struct DetectByzantineInput {
    pub round: u32,
    pub levels: usize,           // Hierarchy levels
    pub min_cluster_size: usize, // Minimum cluster size
}
```

**Output:**
```rust
pub struct ByzantineDetectionResult {
    pub round: u32,
    pub suspected_nodes: Vec<String>,
    pub byzantine_fraction: f64,
    pub within_tolerance: bool,
    pub cluster_count: usize,
}
```

---

#### detect_cartels

Detect colluding nodes using graph-based clustering.

```rust
#[hdk_extern]
pub fn detect_cartels(input: DetectCartelInput) -> ExternResult<CartelDetectionResult>
```

**Input:**
```rust
pub struct DetectCartelInput {
    pub round: u32,
    pub correlation_threshold: f64, // Min correlation for collusion
    pub min_cartel_size: usize,
}
```

**Output:**
```rust
pub struct CartelDetectionResult {
    pub round: u32,
    pub cartels: Vec<Vec<String>>, // Groups of colluding nodes
    pub total_cartel_nodes: usize,
    pub cartel_fraction: f64,
}
```

---

#### get_round_byzantine_records

Get all Byzantine records for a round.

```rust
#[hdk_extern]
pub fn get_round_byzantine_records(round: u32) -> ExternResult<Vec<ByzantineRecord>>
```

---

### Reputation Management

#### update_reputation

Update node reputation (coordinator only).

```rust
#[hdk_extern]
pub fn update_reputation(input: UpdateReputationInput) -> ExternResult<ActionHash>
```

**Input:**
```rust
pub struct UpdateReputationInput {
    pub node_id: String,
    pub successful_rounds: u32,
    pub failed_rounds: u32,
    pub reputation_score: f32,
}
```

---

#### update_reputation_with_matl

Update reputation using MATL's Bayesian smoothing.

```rust
#[hdk_extern]
pub fn update_reputation_with_matl(input: UpdateReputationInput) -> ExternResult<ActionHash>
```

Uses SDK's `ReputationScore` for:
- Bayesian prior smoothing
- Automatic positive/negative interaction tracking
- Decay-resistant scoring

---

#### get_reputation

Get current reputation for a node.

```rust
#[hdk_extern]
pub fn get_reputation(node_id: String) -> ExternResult<Option<NodeReputation>>
```

---

### Role Management

#### grant_role

Grant a role to an agent (coordinator only).

```rust
#[hdk_extern]
pub fn grant_role(input: GrantRoleInput) -> ExternResult<()>
```

**Input:**
```rust
pub struct GrantRoleInput {
    pub agent: String,
    pub role: String,  // "fl_coordinator" or "byzantine_detector"
}
```

---

#### revoke_role

Revoke a role from an agent (coordinator only).

```rust
#[hdk_extern]
pub fn revoke_role(input: GrantRoleInput) -> ExternResult<()>
```

**Note:** Cannot revoke your own coordinator role.

---

### Coordinator Bootstrap (SEC-002)

#### initialize_bootstrap

Initialize the secure coordinator bootstrap ceremony.

```rust
#[hdk_extern]
pub fn initialize_bootstrap(input: InitializeBootstrapInput) -> ExternResult<BootstrapResult>
```

**Input:**
```rust
pub struct InitializeBootstrapInput {
    pub genesis_coordinators: Vec<String>,  // Public keys
    pub authority_proofs: Vec<Vec<u8>>,     // Ed25519 signatures
    pub initial_guardians: Vec<String>,
    pub window_duration_seconds: i64,       // 0 = 24h default
    pub min_votes: u32,                     // N of M for approval
}
```

**Output:**
```rust
pub struct BootstrapResult {
    pub success: bool,
    pub bootstrap_config_hash: ActionHash,
    pub genesis_coordinators_created: usize,
    pub guardians_created: usize,
    pub window_end: i64,
}
```

**Security:**
- Can only be called once (prevents re-initialization)
- Requires cryptographic authority proofs
- Creates immutable genesis coordinators

---

### Signal Handling

#### create_signed_signal

Create a cryptographically signed signal for remote transmission.

```rust
#[hdk_extern]
pub fn create_signed_signal(input: CreateSignedSignalInput) -> ExternResult<CreateSignedSignalOutput>
```

**Input:**
```rust
pub struct CreateSignedSignalInput {
    pub signal: Signal,
}
```

**Output:**
```rust
pub struct CreateSignedSignalOutput {
    pub signed_input: SignedRemoteSignalInput,
}
```

---

#### recv_remote_signal

Receive and verify remote signals.

```rust
#[hdk_extern]
pub fn recv_remote_signal(signal: ExternIO) -> ExternResult<()>
```

**Security:**
- Ed25519 signature verification
- Timestamp validation (5-minute window)
- Replay attack protection via nonce
- Rate limiting per source

---

#### verify_signed_signal_explicit

Verify a signed signal without processing.

```rust
#[hdk_extern]
pub fn verify_signed_signal_explicit(input: SignedRemoteSignalInput) -> ExternResult<bool>
```

---

### Utility Functions

#### compute_hypervector_similarity

Compute cosine similarity between two hypervectors.

```rust
#[hdk_extern]
pub fn compute_hypervector_similarity(input: (Vec<u8>, Vec<u8>)) -> ExternResult<f32>
```

Returns: Similarity score in [-1.0, 1.0]

---

## Signal Types

```rust
pub enum Signal {
    GradientSubmitted {
        node_id: String,
        round: u32,
        action_hash: ActionHash,
        source: Option<String>,
        signature: Option<String>,
    },
    RoundCompleted {
        round: u32,
        accuracy: f32,
        byzantine_count: u32,
        source: Option<String>,
        signature: Option<String>,
    },
    ByzantineDetected {
        node_id: String,
        round: u32,
        confidence: f32,
        source: Option<String>,
        signature: Option<String>,
    },
}
```

---

## Security Considerations

### Authorization (F-03)
- **Coordinator Role**: Required for `complete_round`, `update_reputation`, `grant_role`, `revoke_role`
- **Detector Role**: Required for `record_byzantine`
- Role verification uses path-based tracking with revocation support

### Input Validation (F-01)
- Node IDs: Non-empty, max 256 chars, alphanumeric + `-_:`
- Gradient hashes: Exactly 64 hex characters
- Scores: Finite, within [0.0, 1.0]

### Rate Limiting (F-06)
- 60 submissions per minute per agent
- 100 signals per minute per source

### Signal Authentication (SEC-005)
- Ed25519 signature verification for remote signals
- 5-minute timestamp window for replay protection
- Nonce-based replay prevention

### Byzantine Tolerance (F-07)
- Maximum 45% Byzantine participants allowed
- Validated at entry creation (integrity zome)
- Hierarchical detection is O(n log n)

### Coordinator Bootstrap (SEC-002)
- **No "first caller wins"**: Requires cryptographic proofs
- Multi-signature guardian approval for new coordinators
- Immutable genesis coordinators
- Audit log chain for all coordinator actions

---

## Error Conditions

| Error | Cause |
|-------|-------|
| `"Node ID cannot be empty"` | Empty node_id provided |
| `"Node ID exceeds maximum length"` | node_id > 256 chars |
| `"Node ID contains invalid characters"` | Non-alphanumeric chars (except `-_:`) |
| `"Rate limit exceeded"` | Too many requests per minute |
| `"Unauthorized: Coordinator role required"` | Missing coordinator credential |
| `"Unauthorized: Detector role required"` | Missing detector or coordinator role |
| `"Byzantine fraction exceeds tolerance"` | > 45% Byzantine participants |
| `"Invalid hypervector size"` | Hypervector != 2048 bytes |
| `"Signal signature verification failed"` | Invalid Ed25519 signature |
| `"Bootstrap already initialized"` | Attempted re-initialization |

---

## Related Documentation

- [Agents Zome API](./AGENTS_ZOME_API.md) - Basic agent registration
- [Bridge Zome API](./BRIDGE_ZOME_API.md) - Cross-hApp communication
- [MATL Specification](../MATL_SPECIFICATION.md) - Multi-Agent Trust Layer details
