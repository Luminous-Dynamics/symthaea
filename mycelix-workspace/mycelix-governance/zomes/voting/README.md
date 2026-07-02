# Voting Zome

The voting zome implements privacy-preserving, ZK-verified voting for the Mycelix governance system.

## Overview

This zome provides three voting mechanisms:

1. **Legacy Voting**: Simple weighted votes (backward compatibility)
2. **Φ-Weighted Voting**: Consciousness-integrated voting with Φ (phi) scores
3. **ZK-Verified Voting**: Privacy-preserving votes with STARK eligibility proofs

## Directory Structure

```
voting/
├── integrity/          # HDI validation rules
│   └── src/lib.rs     # Entry types, link types, validation
└── coordinator/        # HDK business logic
    └── src/lib.rs     # Zome functions
```

## Entry Types

### Core Vote Types

| Entry Type | Description |
|------------|-------------|
| `Vote` | Legacy simple vote |
| `PhiWeightedVote` | Consciousness-weighted vote |
| `QuadraticVote` | Quadratic voting with voice credits |
| `VerifiedVote` | ZK-verified vote with eligibility proof |

### ZK Infrastructure

| Entry Type | Description |
|------------|-------------|
| `EligibilityProof` | STARK proof of voter eligibility |
| `ProofAttestation` | External verifier's signature on proof validity |

### Support Types

| Entry Type | Description |
|------------|-------------|
| `VoiceCredits` | Quadratic voting credit balance |
| `Delegation` | Vote delegation record |
| `VoteTally` | Aggregated vote results |

## Coordinator Functions

### ZK-Verified Voting

```rust
// Store an eligibility proof
#[hdk_extern]
fn store_eligibility_proof(input: StoreProofInput) -> ExternResult<Record>;

// Store a verifier attestation
#[hdk_extern]
fn store_proof_attestation(input: StoreAttestationInput) -> ExternResult<Record>;

// Check if a proof has valid attestation
#[hdk_extern]
fn has_valid_attestation(proof_hash: ActionHash) -> ExternResult<bool>;

// Cast a vote with attestation check
#[hdk_extern]
fn cast_attested_vote(input: CastAttestedVoteInput) -> ExternResult<Record>;

// Cast a verified vote (without attestation requirement)
#[hdk_extern]
fn cast_verified_vote(input: CastVerifiedVoteInput) -> ExternResult<Record>;

// Tally verified votes for a proposal
#[hdk_extern]
fn tally_verified_votes(input: TallyVerifiedVotesInput) -> ExternResult<VerifiedVoteTally>;
```

### Legacy Voting

```rust
#[hdk_extern]
fn cast_vote(input: CastVoteInput) -> ExternResult<Record>;

#[hdk_extern]
fn get_proposal_votes(proposal_id: String) -> ExternResult<Vec<Record>>;

#[hdk_extern]
fn tally_votes(proposal_id: String) -> ExternResult<VoteTally>;
```

### Φ-Weighted Voting

```rust
#[hdk_extern]
fn cast_phi_vote(input: CastPhiVoteInput) -> ExternResult<Record>;

#[hdk_extern]
fn tally_phi_votes(input: TallyPhiVotesInput) -> ExternResult<PhiWeightedTally>;
```

### Query Functions

```rust
#[hdk_extern]
fn get_voter_proofs(voter_did: String) -> ExternResult<Vec<Record>>;

#[hdk_extern]
fn get_proof_attestations(proof_hash: ActionHash) -> ExternResult<Vec<Record>>;

#[hdk_extern]
fn get_voter_votes(voter_did: String) -> ExternResult<Vec<Record>>;
```

## Usage Examples

### Casting a ZK-Verified Vote

```rust
// 1. Client generates eligibility proof using fl-aggregator
let client = ZomeBridgeClient::new();
let profile = VoterProfileBuilder::new("did:mycelix:alice")
    .assurance_level(3)
    .matl_score(0.75)
    .stake(500.0)
    .has_humanity_proof(true)
    .build();

let proof = client.generate_proof(&profile, ProofProposalType::Standard)?;

// 2. Store proof on-chain
let proof_hash = call_zome("voting", "store_eligibility_proof", StoreProofInput {
    voter_did: "did:mycelix:alice".into(),
    voter_commitment: proof.voter_commitment.clone(),
    proposal_type: ZkProposalType::Standard,
    eligible: proof.eligible,
    requirements_met: proof.requirements_met,
    active_requirements: proof.active_requirements,
    proof_bytes: proof.proof_bytes.clone(),
    validity_hours: Some(24),
})?;

// 3. External verifier attests to proof validity
let verifier = VerifierService::with_signing_key(&signing_key)?;
let attestation = verifier.verify_and_attest(&proof)?;

// 4. Store attestation on-chain
call_zome("voting", "store_proof_attestation", StoreAttestationInput {
    proof_action_hash: proof_hash,
    proof_hash: attestation.proof_hash.to_vec(),
    voter_commitment: attestation.voter_commitment.to_vec(),
    proposal_type: ZkProposalType::Standard,
    verified: attestation.verified,
    verifier_pubkey: attestation.verifier_pubkey.to_vec(),
    signature: attestation.signature.to_vec(),
    security_level: attestation.security_level,
    verification_time_ms: attestation.verification_time_ms,
    validity_hours: Some(24),
})?;

// 5. Cast attested vote
call_zome("voting", "cast_attested_vote", CastAttestedVoteInput {
    proposal_id: "prop-001".into(),
    voter_did: "did:mycelix:alice".into(),
    tier: ProposalTier::Standard,
    choice: VoteChoice::Approve,
    eligibility_proof_hash: proof_hash,
    voter_commitment: proof.voter_commitment,
    reason: Some("Supporting this proposal".into()),
})?;
```

## Validation Rules

### EligibilityProof Validation

- `voter_did` must start with "did:"
- `voter_commitment` must be exactly 32 bytes
- `proof_bytes` must not be empty
- If `expires_at` is set, it must be after `generated_at`

### ProofAttestation Validation

- `proof_hash` must be exactly 32 bytes (Blake3)
- `voter_commitment` must be exactly 32 bytes
- `verifier_pubkey` must be exactly 32 bytes (Ed25519)
- `signature` must be exactly 64 bytes (Ed25519)
- `expires_at` must be after `verified_at`

### VerifiedVote Validation

- `voter` must be a valid DID
- `voter_commitment` must be exactly 32 bytes
- `effective_weight` must be between 0.0 and 2.0
- Voter cannot update votes once cast

## Link Types

| Link Type | Base | Target |
|-----------|------|--------|
| `ProposalToVote` | Proposal anchor | Vote entry |
| `VoterToVote` | Voter anchor | Vote entry |
| `VoterToEligibilityProof` | Voter anchor | EligibilityProof |
| `ProposalToVerifiedVote` | Proposal anchor | VerifiedVote |
| `ProofToAttestation` | EligibilityProof | ProofAttestation |
| `VerifierToAttestation` | Verifier anchor | ProofAttestation |

## Security

### Vote Weight Caps

Maximum effective weight is capped at 2.0 to prevent:
- Inflation attacks
- Sybil vote amplification
- Trust score manipulation

### Attestation Expiry

Attestations expire after 24 hours by default. This ensures:
- Proofs are re-verified periodically
- Stale proofs cannot be used indefinitely
- Verifier compromise has limited window

### Double-Vote Prevention

The zome checks for existing votes before allowing new ones:
```rust
let existing_votes = get_voter_verified_votes(&voter_did, &proposal_id)?;
if !existing_votes.is_empty() {
    return Err("Voter has already cast a verified vote".into());
}
```

## Testing

```bash
# Run integrity validation tests
cargo test -p voting_integrity

# Run coordinator tests
cargo test -p voting

# Run integration tests
cargo test --test zk_voting_tests
```

## Dependencies

```toml
[dependencies]
hdk = { workspace = true }
voting_integrity = { path = "../integrity" }
proposals_integrity = { path = "../../proposals/integrity" }
hex = "0.4"
```
