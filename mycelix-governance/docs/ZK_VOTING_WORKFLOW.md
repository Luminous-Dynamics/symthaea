# Zero-Knowledge Voting Workflow

This document describes the privacy-preserving voting system in Mycelix governance, which uses zero-knowledge proofs to verify voter eligibility without revealing sensitive trust metrics.

## Overview

The ZK voting system enables:
- **Privacy**: Voters prove eligibility without revealing exact K-Vector values
- **Verifiability**: Anyone can verify votes are from eligible participants
- **Trust-Weighted**: Vote weight derived from K-Vector trust metrics
- **Tiered Security**: Different proposal types require different security levels

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Voter Client                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ K-Vector     │───>│ Eligibility  │───>│ Verified     │          │
│  │ Trust Metrics│    │ Proof Gen    │    │ Vote Cast    │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                              │                      │
                              ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Holochain DHT                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Eligibility  │    │ Verified     │    │ Vote         │          │
│  │ Proofs       │    │ Votes        │    │ Tallies      │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. ZkProposalType

Defines the type of proposal, which determines security requirements:

```rust
pub enum ZkProposalType {
    Standard = 0,       // Regular proposals (96-bit security)
    Constitutional = 1, // Constitutional changes (264-bit security)
    ModelGovernance = 2,// ML model governance (96-bit security)
    Emergency = 3,      // Emergency proposals (96-bit security)
    Treasury = 4,       // Treasury operations (96-bit security)
    Membership = 5,     // Membership changes (84-bit security)
}
```

### 2. EligibilityProof

Proves a voter meets requirements without revealing exact values:

```rust
pub struct EligibilityProof {
    pub id: String,
    pub voter_did: String,
    pub voter_commitment: Vec<u8>,    // SHA3-256 commitment
    pub proposal_type: ZkProposalType,
    pub eligible: bool,
    pub requirements_met: u8,         // Number of requirements met
    pub active_requirements: u8,      // Total active requirements
    pub proof_bytes: Vec<u8>,         // STARK proof
    pub generated_at: Timestamp,
    pub expires_at: Option<Timestamp>,
}
```

### 3. VerifiedVote

A vote backed by a verified eligibility proof:

```rust
pub struct VerifiedVote {
    pub id: String,
    pub proposal_id: String,
    pub proposal_tier: ProposalTier,
    pub voter: String,
    pub choice: VoteChoice,           // Approve, Reject, Abstain
    pub eligibility_proof_hash: ActionHash,
    pub voter_commitment: Vec<u8>,
    pub effective_weight: f64,        // Trust-based weight
    pub reason: Option<String>,
    pub voted_at: Timestamp,
}
```

## Voting Workflow

### Step 1: Generate Eligibility Proof

The voter's client generates a ZK proof that their K-Vector meets the proposal requirements:

```rust
// Client-side proof generation
let input = GenerateEligibilityProofInput {
    voter_did: "did:mycelix:alice".to_string(),
    proposal_type: ZkProposalType::Standard,
    kvector: agent_kvector,          // Private K-Vector values
    proposal_thresholds: thresholds, // Public threshold requirements
};

let proof = generate_eligibility_proof(input)?;
```

**Requirements proven (without revealing values):**
1. Participation score ≥ minimum threshold
2. Reputation score ≥ minimum threshold
3. Account age ≥ minimum
4. No active sanctions
5. Stake requirements met (if applicable)
6. Trust decay not exceeded
7. Contribution threshold met

### Step 2: Store Eligibility Proof

Submit the proof to the DHT for verification:

```rust
let input = StoreEligibilityProofInput {
    voter_did: proof.voter_did,
    voter_commitment: proof.voter_commitment,
    proposal_type: proof.proposal_type,
    eligible: proof.eligible,
    requirements_met: proof.requirements_met,
    active_requirements: proof.active_requirements,
    proof_bytes: proof.proof_bytes,
    expires_at: Some(sys_time()? + Duration::from_secs(3600)),
};

let proof_record = store_eligibility_proof(input)?;
```

### Step 3: Cast Verified Vote

Cast a vote referencing the eligibility proof:

```rust
let input = CastVerifiedVoteInput {
    proposal_id: "prop-001".to_string(),
    proposal_tier: ProposalTier::Standard,
    voter: "did:mycelix:alice".to_string(),
    choice: VoteChoice::Approve,
    eligibility_proof_hash: proof_record.action_hashed().hash.clone(),
    voter_commitment: proof.voter_commitment,
    effective_weight: calculate_vote_weight(&agent_kvector),
    reason: Some("Supports ecosystem growth".to_string()),
};

let vote_record = cast_verified_vote(input)?;
```

### Step 4: Tally Votes

Aggregate verified votes for a proposal:

```rust
let input = TallyVerifiedVotesInput {
    proposal_id: "prop-001".to_string(),
    proposal_tier: ProposalTier::Standard,
};

let tally = tally_verified_votes(input)?;

// tally contains:
// - approve_weight: Total approval weight
// - reject_weight: Total rejection weight
// - abstain_weight: Total abstention weight
// - total_votes: Number of votes cast
// - valid_votes: Number of valid (verified) votes
```

## Security Levels

Different proposal types require different proof security:

| Proposal Type | Min Security | Use Case |
|--------------|--------------|----------|
| Standard | 96-bit | Regular governance |
| Constitutional | 264-bit | Constitutional amendments |
| Treasury | 96-bit | Treasury operations |
| Emergency | 96-bit | Emergency decisions |
| Membership | 84-bit | Membership changes |

## Proposal Tier Thresholds

Voting thresholds vary by proposal tier:

| Tier | Approval Threshold | Description |
|------|-------------------|-------------|
| Basic | 50% | Simple majority |
| Standard | 50% | Simple majority |
| Major | 60% | Supermajority |
| Constitutional | 67% | Two-thirds supermajority |

## Privacy Guarantees

The ZK voting system provides:

1. **Vote Privacy**: Individual votes cannot be linked to voters' identities
2. **Eligibility Privacy**: Exact K-Vector values are never revealed
3. **Commitment Binding**: Voters cannot change their vote after casting
4. **Proof Soundness**: Invalid proofs are rejected

## Integration Example

```rust
use voting_coordinator::*;
use voting_integrity::*;

// Full voting workflow
async fn participate_in_vote(
    voter_did: &str,
    proposal_id: &str,
    kvector: &KVector,
    choice: VoteChoice,
) -> Result<ActionHash, Error> {
    // 1. Generate eligibility proof
    let proof = generate_eligibility_proof(GenerateEligibilityProofInput {
        voter_did: voter_did.to_string(),
        proposal_type: ZkProposalType::Standard,
        kvector: kvector.clone(),
        proposal_thresholds: get_standard_thresholds(),
    })?;

    if !proof.eligible {
        return Err(Error::NotEligible);
    }

    // 2. Store proof on DHT
    let proof_record = store_eligibility_proof(StoreEligibilityProofInput {
        voter_did: voter_did.to_string(),
        voter_commitment: proof.voter_commitment.clone(),
        proposal_type: proof.proposal_type,
        eligible: proof.eligible,
        requirements_met: proof.requirements_met,
        active_requirements: proof.active_requirements,
        proof_bytes: proof.proof_bytes,
        expires_at: Some(sys_time()? + Duration::from_secs(3600)),
    })?;

    // 3. Cast verified vote
    let vote_record = cast_verified_vote(CastVerifiedVoteInput {
        proposal_id: proposal_id.to_string(),
        proposal_tier: ProposalTier::Standard,
        voter: voter_did.to_string(),
        choice,
        eligibility_proof_hash: proof_record.action_hashed().hash.clone(),
        voter_commitment: proof.voter_commitment,
        effective_weight: kvector.compute_vote_weight(),
        reason: None,
    })?;

    Ok(vote_record.action_hashed().hash.clone())
}
```

## Error Handling

Common errors and their causes:

| Error | Cause | Resolution |
|-------|-------|------------|
| `IneligibleVoter` | K-Vector doesn't meet requirements | Improve trust metrics |
| `ProofExpired` | Eligibility proof has expired | Generate new proof |
| `InvalidCommitment` | Vote commitment doesn't match proof | Ensure same voter identity |
| `DuplicateVote` | Voter already voted on proposal | Only one vote allowed |
| `SecurityLevelInsufficient` | Proof security too low | Regenerate with higher security |

## Testing

Run integration tests:

```bash
cd mycelix-governance
cargo test --test zk_voting_tests
```

Key test modules:
- `zk_proposal_type_tests`: Proposal type validation
- `eligibility_proof_tests`: Proof generation and validation
- `verified_vote_tests`: Vote casting and validation
- `vote_tallying_tests`: Tally calculation
- `zk_workflow_tests`: End-to-end workflow tests

## Performance Considerations

- **Proof Generation**: ~10-18 seconds depending on security level
- **Proof Verification**: ~100-500ms
- **Vote Casting**: ~50ms (after proof verification)
- **Tally Calculation**: O(n) where n = number of votes

## Related Documentation

- [K-Vector Trust Metrics](/libs/kvector-zkp/README.md)
- [Security Levels](/libs/proofs-config/src/lib.rs)
- [Commitment Schemes](/libs/proofs-commitment/src/lib.rs)
- [Governance Architecture](/docs/GOVERNANCE_ARCHITECTURE.md)
