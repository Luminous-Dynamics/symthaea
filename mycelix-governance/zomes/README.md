# Mycelix Governance Zomes

This directory contains the Holochain zomes that implement the Mycelix decentralized governance system. The system supports ZK-verified voting, consciousness-weighted decisions, and Byzantine-tolerant consensus.

## Architecture Overview

```
mycelix-governance/zomes/
├── proposals/          # Proposal lifecycle management
│   ├── integrity/     # Validation rules (HDI)
│   └── coordinator/   # Business logic (HDK)
├── voting/            # Vote casting and tallying
│   ├── integrity/     # Vote validation & ZK types
│   └── coordinator/   # Voting logic & attestations
├── execution/         # Proposal execution after approval
├── constitution/      # Constitutional amendments
├── threshold-signing/ # Multi-party signing ceremonies
└── deliberation/      # Discussion and feedback
```

## Zome Summary

| Zome | Purpose | Key Functions |
|------|---------|---------------|
| **proposals** | Create, amend, and manage proposals | `create_proposal`, `submit_proposal`, `get_proposal` |
| **voting** | Cast votes with ZK eligibility proofs | `cast_verified_vote`, `store_eligibility_proof`, `tally_verified_votes` |
| **execution** | Execute approved proposals | `execute_proposal`, `schedule_execution` |
| **constitution** | Manage constitutional documents | `propose_amendment`, `ratify_amendment` |
| **threshold-signing** | Coordinate multi-sig operations | `initiate_signing`, `submit_signature` |
| **deliberation** | Forum for proposal discussion | `post_comment`, `react_to_comment` |

## ZK-Verified Voting System

The voting zome implements privacy-preserving voting using ZK-STARK proofs:

### Workflow

1. **Proof Generation** (Client-side)
   ```
   VoterProfile → ZomeBridgeClient.generate_proof() → SerializableProof
   ```

2. **Proof Storage** (On-chain)
   ```rust
   store_eligibility_proof(proof) → ActionHash
   ```

3. **External Verification** (Off-chain oracle)
   ```
   VerifierService.verify_and_attest(proof) → ProofAttestation
   ```

4. **Attestation Storage** (On-chain)
   ```rust
   store_proof_attestation(attestation) → ActionHash
   ```

5. **Vote Casting** (On-chain)
   ```rust
   cast_attested_vote(vote_input) → VerifiedVote
   ```

### Proposal Types & Security Levels

| Proposal Type | Security Bits | Min MATL | Humanity Proof |
|---------------|---------------|----------|----------------|
| Standard | 96 | 0.3 | No |
| Constitutional | 264 | 0.7 | Yes |
| ModelGovernance | 96 | 0.5 | No |
| Emergency | 96 | 0.4 | No |
| Treasury | 96 | 0.5 | No |
| Membership | 84 | 0.1 | No |

### Proposal Tiers & Thresholds

| Tier | Approval Threshold | Quorum |
|------|-------------------|--------|
| Basic | 50% | 10% |
| Standard | 50% | 20% |
| Major | 60% | 30% |
| Constitutional | 67% | 50% |

## Key Entry Types

### EligibilityProof
```rust
pub struct EligibilityProof {
    pub voter_did: String,
    pub voter_commitment: Vec<u8>,  // Blake3 hash
    pub proposal_type: ZkProposalType,
    pub eligible: bool,
    pub requirements_met: u8,
    pub proof_bytes: Vec<u8>,       // STARK proof
    pub expires_at: Option<Timestamp>,
}
```

### ProofAttestation
```rust
pub struct ProofAttestation {
    pub proof_hash: Vec<u8>,
    pub proof_action_hash: ActionHash,
    pub verified: bool,
    pub verifier_pubkey: Vec<u8>,   // Ed25519
    pub signature: Vec<u8>,         // Ed25519
    pub expires_at: Timestamp,
}
```

### VerifiedVote
```rust
pub struct VerifiedVote {
    pub proposal_id: String,
    pub voter: String,              // DID
    pub choice: VoteChoice,
    pub eligibility_proof_hash: ActionHash,
    pub voter_commitment: Vec<u8>,
    pub effective_weight: f64,      // 0.0 - 1.0
}
```

## Link Types

| Link Type | From → To | Purpose |
|-----------|-----------|---------|
| `ProposalToVote` | Proposal → Vote | Track votes per proposal |
| `VoterToEligibilityProof` | Voter → Proof | Track voter's proofs |
| `ProofToAttestation` | Proof → Attestation | Link verified proofs |
| `ProposalToVerifiedVote` | Proposal → VerifiedVote | ZK-verified votes |

## Development

### Building

```bash
cd mycelix-governance
nix develop  # Enter dev shell
cargo build --release --target wasm32-unknown-unknown
```

### Testing

```bash
# Unit tests
cargo test -p voting_integrity
cargo test -p proposals_integrity

# Integration tests
cargo test --test zk_voting_tests
cargo test --test consciousness_metrics_tests
```

### Dependencies

- Holochain HDK 0.6.x / HDI 0.7.x
- fl-aggregator (for ZK proofs)
- ed25519-dalek (for attestation signatures)
- blake3 (for commitments)

## Integration with FL-Aggregator

The governance system integrates with the Federated Learning aggregator for:

1. **VoteEligibilityProof Generation**: Uses the `ZomeBridgeClient` from fl-aggregator
2. **STARK Proof Verification**: External verifiers use `VerifierService`
3. **MATL Trust Scores**: Trust metrics from MATL bridge influence voting weight

See `/Mycelix-Core/libs/fl-aggregator/src/proofs/integration/zome_bridge.rs` for the integration API.

## Security Considerations

1. **Proof Verification**: STARK proofs are verified off-chain due to WASM constraints
2. **Attestation Expiry**: Attestations expire after 24 hours by default
3. **Vote Weight Caps**: Maximum effective weight is 2.0 to prevent inflation
4. **Commitment Binding**: Voter commitments bind proofs to specific voters
5. **Double-Vote Prevention**: Voters can only cast one vote per proposal

## Related Documentation

- [ZK Voting Workflow](../docs/ZK_VOTING_WORKFLOW.md)
- [FL-Aggregator Proofs](../../Mycelix-Core/libs/fl-aggregator/README.md)
- [MATL Trust Bridge](../../Mycelix-Core/libs/matl-bridge/README.md)
