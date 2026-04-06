# Holochain DHT Entry Definitions

This document describes the entry types stored in the Praxis Holochain DHT.

## Entry Types Overview

| Zome | Entry Type | Description |
|------|-----------|-------------|
| `learning_zome` | `Course` | Course metadata and structure |
| `learning_zome` | `LearnerProgress` | Individual learner progress tracking |
| `learning_zome` | `LearningActivity` | Privacy-preserving activity logs |
| `fl_zome` | `FlRound` | Federated learning round metadata |
| `fl_zome` | `FlUpdate` | Individual gradient/update submission |
| `credential_zome` | `VerifiableCredential` | W3C Verifiable Credential |
| `credential_zome` | `CredentialStatus` | Revocation status list |
| `dao_zome` | `Proposal` | Governance proposal |
| `dao_zome` | `Vote` | Individual vote on a proposal |

---

## Learning Zome

### Course

**Visibility**: Public
**Validation**: Creator must be author; title non-empty; tags max 10

```rust
struct Course {
    course_id: CourseId,
    title: String,
    description: String,
    creator: AgentPubKey,
    tags: Vec<String>,
    model_id: Option<String>,
    created_at: Timestamp,
    updated_at: Timestamp,
    metadata: Option<Value>,
}
```

**Links**:
- `AgentPubKey -> Course` (creator's courses)
- `Tag -> Course` (courses by tag)

---

### LearnerProgress

**Visibility**: Private (agent-only)
**Validation**: Learner must be author; progress 0-100

```rust
struct LearnerProgress {
    course_id: CourseId,
    learner: AgentPubKey,
    progress_percent: f32,
    completed_items: Vec<String>,
    model_version: Option<String>,
    last_active: Timestamp,
    metadata: Option<Value>,
}
```

**Links**:
- `AgentPubKey -> LearnerProgress` (learner's progress records)
- `Course -> LearnerProgress` (participants in course)

---

### LearningActivity

**Visibility**: Private (agent-only)
**Validation**: Activity type from enum; duration > 0

```rust
struct LearningActivity {
    course_id: CourseId,
    activity_type: String,
    item_id: String,
    outcome: Option<f32>,
    duration_secs: u32,
    timestamp: Timestamp,
}
```

**Links**:
- `AgentPubKey -> LearningActivity` (agent's activities)

---

## Federated Learning Zome

### FlRound

**Visibility**: Public
**Validation**: Min participants > 0; max >= min; clip norm > 0

```rust
struct FlRound {
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

**Links**:
- `ModelId -> FlRound` (rounds for a model)
- `RoundState -> FlRound` (rounds by state)

---

### FlUpdate

**Visibility**: Public (commitment only)
**Validation**: Clipped L2 norm <= round's clip norm; sample count > 0

```rust
struct FlUpdate {
    round_id: RoundId,
    model_id: String,
    parent_model_hash: ModelHash,
    grad_commitment: Vec<u8>,
    clipped_l2_norm: f32,
    local_val_loss: f32,
    sample_count: u32,
    timestamp: Timestamp,
}
```

**Links**:
- `RoundId -> FlUpdate` (updates for a round)
- `AgentPubKey -> FlUpdate` (agent's updates)

---

## Credential Zome

### VerifiableCredential

**Visibility**: Semi-public (holder can share)
**Validation**: Follows W3C VC spec; issuer signature valid

```rust
struct VerifiableCredential {
    context: Value,
    credential_type: Vec<String>,
    issuer: String,
    issuance_date: String,
    credential_subject: CredentialSubject,
    expiration_date: Option<String>,
    credential_status: Option<CredentialStatus>,
    proof: Proof,
}
```

**Links**:
- `AgentPubKey -> VerifiableCredential` (holder's credentials)
- `CourseId -> VerifiableCredential` (credentials for course)

---

### CredentialStatus

**Visibility**: Public
**Validation**: Revocation bitmap valid; issuer authorized

```rust
struct CredentialStatusList {
    id: String,
    issuer: AgentPubKey,
    status_purpose: String,
    encoded_list: String, // Base64 compressed bitmap
    updated_at: Timestamp,
}
```

**Links**:
- `Issuer -> CredentialStatusList` (issuer's status lists)

---

## DAO Zome

### Proposal

**Visibility**: Public
**Validation**: Proposer is author; voting deadline in future; type matches category constraints

```rust
struct Proposal {
    proposal_id: String,
    title: String,
    description: String,
    proposer: AgentPubKey,
    proposal_type: ProposalType,
    category: ProposalCategory,
    status: ProposalStatus,
    votes: VoteCount,
    voting_deadline: Timestamp,
    created_at: Timestamp,
    executed_at: Option<Timestamp>,
    actions: Vec<Value>,
}
```

**Links**:
- `ProposalType -> Proposal` (proposals by type)
- `ProposalStatus -> Proposal` (proposals by status)
- `AgentPubKey -> Proposal` (proposer's proposals)

---

### Vote

**Visibility**: Public
**Validation**: Voter is author; proposal exists; before deadline; one vote per agent per proposal

```rust
struct Vote {
    proposal_id: String,
    voter: AgentPubKey,
    choice: VoteChoice,
    justification: Option<String>,
    timestamp: Timestamp,
}
```

**Links**:
- `Proposal -> Vote` (votes for a proposal)
- `AgentPubKey -> Vote` (agent's votes)

---

## Validation Rules Summary

### Cross-Zome Rules

1. **Credential → Course**: Credential's `courseId` must reference existing Course
2. **FlUpdate → FlRound**: Update's `roundId` must reference existing FlRound
3. **Vote → Proposal**: Vote's `proposalId` must reference existing Proposal

### Privacy Rules

1. **LearnerProgress**: Only readable by learner (author)
2. **LearningActivity**: Only readable by learner (author)
3. **FlUpdate**: Only commitment published; actual gradients kept private
4. **VerifiableCredential**: Holder controls disclosure

### Integrity Rules

1. **Timestamps**: Must be within reasonable bounds (not future, not ancient)
2. **Hashes**: BLAKE3 format, 64 hex characters
3. **Signatures**: Ed25519, verified against agent key
4. **Links**: Bidirectional; cleanup on delete

---

## Schema Versioning

Current version: **v0.1.0**

When changing entry definitions:
1. Increment version in `Cargo.toml`
2. Create migration guide in `docs/migrations/`
3. Support backward compatibility for 1 major version

---

## Tools

- **Validation**: See `zomes/*/src/lib.rs` for HDK validation functions
- **Testing**: Integration tests in `tests/`
- **Schema export**: `make schema-export` (generates JSON schema from Rust types)
