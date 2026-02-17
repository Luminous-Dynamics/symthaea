# LLM-SIMULATED USER ACTION SCENARIOS

## Overview

These simulations model various actor behaviors in the Mycelix network using defined personas and action patterns. Each simulation describes the actor's goals, strategies, and expected system responses.

---

## SIMULATION 1: Byzantine Attacker

### Persona: "Mallory" - Rational Byzantine Attacker

**Profile**:
- Objective: Maximize personal gain through protocol manipulation
- Resources: Controls 20% of network reputation weight
- Strategy: Exploit economic incentives without triggering slashing

**Simulated Action Sequence**:

```
[ROUND 1] Network Status: Normal operation
+-- Mallory has 3 validator nodes with reputation [0.82, 0.79, 0.85]
+-- Total network reputation weight: ~15.4 (100 validators)
+-- Mallory's weight: ~3.08 (20%)

[ROUND 2] Mallory Submits FL Contribution
+-- ACTION: Submit valid gradient update for Model "price-prediction-v3"
+-- Contribution: Legitimate but minimal (passes PoGQ threshold of 0.6)
+-- PoGQ Score: 0.62 (just above minimum)
+-- RESULT: Contribution accepted, receives payment split
+-- OBSERVATION: System accepts low-effort contributions

[ROUND 3] Mallory Attempts Free-Riding Attack
+-- ACTION: Submit gradient that is 95% copy of Round 2 with noise
+-- Contribution similarity: High cosine similarity to previous
+-- PoGQ Score: 0.58 (BELOW threshold)
+-- RESULT: Contribution REJECTED by pogq_validation zome
+-- Slashing: None (rejection != slashing offense)
+-- OBSERVATION: PoGQ detects derivative contributions

[ROUND 4] Mallory Attempts Sybil via Multiple Identities
+-- ACTION: Register new DID "mallory-node-4" in MycelixRegistry
+-- New node reputation: 0.10 (minimum starting reputation)
+-- New node weight: 0.01 (0.10^2 = 0.01)
+-- RESULT: Registration succeeds, but weight is negligible
+-- 100 sybils needed to match 1 established node
+-- OBSERVATION: Reputation^2 weighting makes sybils uneconomical

[ROUND 5] Mallory Attempts Vote Manipulation
+-- ACTION: Cast REJECT vote on valid governance proposal
+-- Mallory's combined voting weight: 3.08 (20%)
+-- Honest validators' weight: ~12.32 (80%)
+-- Threshold needed: 55% x 15.4 = 8.47
+-- RESULT: Proposal PASSES (honest weight exceeds threshold)
+-- OBSERVATION: 45% tolerance holds against 20% attacker

[ROUND 6] Mallory Attempts Double-Voting
+-- ACTION: Submit APPROVE vote, then REJECT vote for same proposal
+-- Vote 1: Recorded, weighted +2.46
+-- Vote 2: DETECTED as duplicate by receive_vote()
+-- RESULT: Double vote REJECTED
+-- SlashableOffense::DoubleVoting evidence created
+-- Slashing: -50% reputation penalty pending confirmation
+-- OBSERVATION: Double-vote detection works correctly

[ROUND 7] Post-Slashing Status
+-- Mallory's nodes: [0.41, 0.39, 0.43] (halved)
+-- New weight: 0.17 + 0.15 + 0.18 = 0.50 (down from 3.08)
+-- Network share: 3.2% (down from 20%)
+-- CONCLUSION: Byzantine behavior is economically punished
```

**Simulation Results**:

| Attack Vector | Success | Detection | Mitigation |
|---------------|---------|-----------|------------|
| Low-effort FL contributions | Partial | PoGQ | Rejected if below 0.6 |
| Gradient copying | Failed | Similarity check | Rejected |
| Sybil accounts | Failed | Rep^2 weighting | Economically infeasible |
| Vote manipulation (20% stake) | Failed | 55% threshold | 45% tolerance holds |
| Double voting | Failed | Vote dedup | Slashing applied |

---

## SIMULATION 2: Honest Federated Learning Participant

### Persona: "Alice" - Medical Research Collaborator

**Profile**:
- Objective: Contribute to shared ML models while preserving patient privacy
- Resources: Local hospital dataset (100K patient records), moderate compute
- Constraints: HIPAA compliance, cannot share raw data

**Simulated Action Sequence**:

```
[DAY 1] Onboarding to Mycelix Network
+-- ACTION: Generate Holochain keypair
+-- ACTION: Register DID in MycelixRegistry
+-- DID: did:mycelix:hc:uhCEk8zJ5yP2...
+-- Ethereum address linked: 0x742d35Cc6634C0532925a3b844Bc454e4438f44e
+-- Initial reputation: 0.10
+-- STATUS: Onboarding complete

[DAY 2] Join Federated Learning Model
+-- ACTION: Query ModelRegistry for health-related models
+-- Found: "cardiac-risk-predictor-v2" (round 47 active)
+-- ACTION: Read model requirements from federated_learning zome
|   +-- Required: PoGQ score >= 0.6
|   +-- Differential privacy: epsilon = 0.1
|   +-- Min local samples: 1000
+-- ACTION: Register as contributor via fl_coordinator
+-- STATUS: Registered for Round 47

[DAY 3] Local Training with Privacy Preservation
+-- ACTION: Download global model parameters
+-- ACTION: Apply differential privacy (using libs/differential-privacy)
|   +-- Gaussian mechanism with sigma = 0.5
|   +-- Privacy budget consumed: epsilon = 0.08
|   +-- Gradient clipping at L2 norm = 1.0
+-- ACTION: Train on local data (5 epochs)
|   +-- Local accuracy: 0.847 -> 0.892
|   +-- Privacy-preserving gradients generated
+-- ACTION: Generate K-Vector ZK proof
|   +-- kvector-zkp prover validates: k_r=0.1, k_a=0.3, k_i=0.9...
|   +-- Proof size: 12KB
+-- STATUS: Local training complete

[DAY 4] Submit Contribution
+-- ACTION: Call `submit_contribution` on fl_coordinator zome
+-- Payload:
|   +-- gradient_hash: 0x8f2e3c...
|   +-- zkp_proof: <binary>
|   +-- local_sample_count: 18,432
|   +-- dp_epsilon_spent: 0.08
+-- PoGQ Validation:
|   +-- Gradient quality score: 0.78
|   +-- Statistical validity: PASSED
|   +-- Similarity to previous: 0.12 (acceptable)
+-- RESULT: Contribution ACCEPTED
+-- STATUS: Awaiting aggregation

[DAY 5] Round Completion & Reward
+-- Aggregation complete: 47 contributors
+-- Alice's contribution weight: 2.1% (based on PoGQ x sample count)
+-- Round reward pool: 1000 MYC
+-- Alice's share: 21 MYC
+-- ACTION: PaymentRouter distributes via splits
|   +-- ETH equivalent: 0.021 ETH at current rate
+-- ACTION: Reputation updated
|   +-- k_a (activity): 0.3 -> 0.35 (+0.05)
|   +-- k_i (integrity): 0.9 -> 0.91 (+0.01)
|   +-- Overall reputation: 0.10 -> 0.23
+-- STATUS: Round 47 complete

[DAY 30] Accumulated Participation
+-- Rounds participated: 12
+-- Total MYC earned: 287 MYC
+-- Reputation growth: 0.10 -> 0.67
+-- PoGQ average: 0.81
+-- Privacy budget remaining: epsilon = 0.04 (of 0.10 annual budget)
+-- STATUS: Established contributor, eligible for governance
```

**Simulation Results**:

| Metric | Value | Notes |
|--------|-------|-------|
| Privacy preserved | Yes | DP epsilon=0.08 per round |
| Revenue generated | 287 MYC | ~$2,870 at $10/MYC |
| Reputation growth | 6.7x | 0.10 -> 0.67 over 30 days |
| Data leaked | 0 bytes | Only gradients + ZK proofs shared |

---

## SIMULATION 3: Governance Participant

### Persona: "Bob" - Long-term Stakeholder

**Profile**:
- Objective: Shape protocol development, ensure sustainability
- Resources: High reputation (0.85), significant MYC holdings
- Interest: Wants to propose increasing FL rewards

**Simulated Action Sequence**:

```
[GOVERNANCE CYCLE 1] Proposal Creation
+-- ACTION: Review current parameters
|   +-- Current FL reward: 1000 MYC per round
|   +-- Treasury balance: 5M MYC
|   +-- Recent participation: declining (47->35 contributors)
+-- ANALYSIS: Lower rewards correlating with decreased participation
+-- ACTION: Draft proposal using mycelix-governance zome
|   +-- Type: ParameterChange
|   +-- Parameter: fl_round_reward
|   +-- Current: 1000 MYC
|   +-- Proposed: 1500 MYC
|   +-- Rationale: "Increase contributor incentives"
+-- Consciousness Gate Check (Symthaea bridge):
|   +-- Bob's Phi (integrated information): 0.42
|   +-- Proposal threshold: 0.3
|   +-- RESULT: PASSED consciousness gate
|   +-- Trust signal emitted: score = 0.42 x 0.8 + 0.2 = 0.536
+-- STATUS: Proposal #127 created

[GOVERNANCE CYCLE 2] Discussion Period
+-- Duration: 7 days
+-- Forum activity:
|   +-- 23 comments
|   +-- 8 support, 12 oppose, 3 neutral
|   +-- Key opposition: "Treasury depletion risk"
+-- Bob's response:
|   +-- Amends proposal: Add sunset clause after 6 months
+-- Amended proposal:
|   +-- fl_round_reward: 1500 MYC
|   +-- expires_after: 180 days (auto-revert to 1000)
+-- STATUS: Discussion period complete

[GOVERNANCE CYCLE 3] Voting
+-- Voting period: 3 days
+-- Voting threshold: 0.4 Phi (consciousness gate for voters)
+-- Vote weights (reputation^2):
|   +-- Total eligible weight: 42.7
|   +-- Threshold (55%): 23.5
|   +-- Quorum requirement: 30% participation
+-- Vote tally:
|   +-- FOR: 28.3 weighted votes (66.3%)
|   +-- AGAINST: 11.2 weighted votes (26.2%)
|   +-- ABSTAIN: 3.2 weighted votes (7.5%)
|   +-- Participation: 42.7/52.1 = 82%
+-- Consciousness-weighted analysis:
|   +-- Average FOR voter Phi: 0.51
|   +-- Average AGAINST voter Phi: 0.44
|   +-- Higher-consciousness voters favored proposal
+-- STATUS: PROPOSAL PASSED

[GOVERNANCE CYCLE 4] Execution
+-- Timelock: 48 hours
+-- On-chain execution:
|   +-- fl_round_reward updated: 1000 -> 1500 MYC
|   +-- expiry_block set: current + 6 months
|   +-- Event emitted: ParameterChanged(...)
+-- Off-chain effects:
|   +-- Holochain DHT propagates new parameter
|   +-- fl_coordinator reads updated value
+-- STATUS: Parameter change live

[POST-EXECUTION] Monitoring
+-- Week 1: Contributors 35 -> 41 (+17%)
+-- Week 2: Contributors 41 -> 48 (+17%)
+-- Week 4: Contributors 48 -> 52 (+8%)
+-- Treasury impact: -26,000 MYC (vs baseline -17,333)
+-- Net assessment: Participation up, sustainable burn rate
+-- Proposal #127 SUCCESSFUL
```

**Simulation Results**:

| Governance Metric | Value |
|-------------------|-------|
| Proposal passage rate | 78% (for well-structured proposals) |
| Consciousness gating effectiveness | Filters low-Phi spam proposals |
| Execution latency | 48h timelock -> instant propagation |
| Parameter change impact | Measurable within 2 weeks |

---

## SIMULATION 4: Validator Node Operator

### Persona: "Carol" - Infrastructure Provider

**Profile**:
- Objective: Earn validation rewards, maintain network integrity
- Resources: Dedicated server (32GB RAM, 8 cores), stable uptime
- Stake: 10,000 MYC locked in validator contract

**Simulated Action Sequence**:

```
[SETUP] Validator Node Configuration
+-- ACTION: Deploy Holochain conductor
|   +-- Using: scripts/local-holochain-testnet.sh start
+-- ACTION: Generate Ed25519 validator keypair
|   +-- Using: rb-bft-consensus/ValidatorKeypair::generate()
+-- ACTION: Register validator in ValidatorSet
|   +-- validator_id: "carol-validator-1"
|   +-- stake: 10,000 MYC
|   +-- initial_reputation: 0.50 (based on stake)
|   +-- k_vector: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
+-- ACTION: Deploy Ethereum bridge
|   +-- Using: scripts/local-ethereum-testnet.sh start
+-- STATUS: Validator operational

[ROUND 1] Participate in Consensus
+-- Round started: 1705593600000 (Unix ms)
+-- Leader selection: VRF(round_number) % validators
+-- Carol is NOT leader (validator-3 selected)
+-- ACTION: Receive proposal from leader
|   +-- proposal_id: "prop-001-1705593600"
|   +-- content_hash: 0x7f3e...
|   +-- signature: valid Ed25519
+-- ACTION: Validate proposal
|   +-- Parent hash matches: YES
|   +-- Content hash valid: YES
|   +-- Signature valid: YES
|   +-- No double-propose detected: YES
+-- ACTION: Cast signed vote
|   +-- vote_signed(VoteDecision::Approve, keypair)
+-- Vote weight: 0.50^2 = 0.25
+-- Round outcome: ACCEPTED (weighted votes: 3.8 > threshold 2.1)
+-- Reputation update: 0.50 -> 0.51 (+0.01 for correct vote)
+-- STATUS: Round 1 complete

[ROUND 2] Carol Selected as Leader
+-- VRF output selects carol-validator-1
+-- ACTION: Propose block with FL aggregation results
|   +-- content: aggregated_gradients_round_47
|   +-- content_hash: 0x9a2b...
|   +-- parent_hash: 0x7f3e... (Round 1 result)
+-- ACTION: propose_signed(content, hash, parent, keypair)
+-- Broadcast proposal to network
+-- Collect votes:
|   +-- validator-1: APPROVE (0.64 weight)
|   +-- validator-2: APPROVE (0.49 weight)
|   +-- validator-3: APPROVE (0.36 weight)
|   +-- carol: APPROVE (0.26 weight) - self-vote
|   +-- validator-5: REJECT (0.16 weight) - always contrarian
+-- Total FOR: 1.75 > threshold 1.05
+-- RESULT: CONSENSUS REACHED
+-- Reputation update: 0.51 -> 0.53 (+0.02 for successful proposal)
+-- STATUS: Round 2 committed

[ROUNDS 3-100] Steady State Operation
+-- Participation rate: 98% (2 rounds missed due to network)
+-- Correct vote rate: 100%
+-- Leader selection: 19 times (expected: 20)
+-- Proposal acceptance: 19/19 (100%)
+-- Reputation trajectory: 0.50 -> 0.78
+-- Slashing events: 0
+-- Rewards earned: 347 MYC
+-- STATUS: Established validator

[INCIDENT] Network Partition
+-- Event: 30% of validators unreachable (network split)
+-- Carol's partition: 70% of validators
+-- Byzantine tolerance: 45%
+-- Threshold adjusted: 55% of available weight
+-- ACTION: Continue consensus with available validators
+-- RESULT: Network continues operating
+-- Missed rounds during partition: 3
+-- Recovery: Partitioned validators rejoin, sync state
+-- STATUS: Partition healed, no forks

[INCIDENT] Equivocation Detection
+-- Event: validator-5 submits two different votes for Round 87
+-- Carol detects: receive_vote() returns DuplicateVote error
+-- ACTION: Capture DoubleVoteEvidence
+-- ACTION: Report via slashing.report_offense()
+-- Confirmations needed: 3
+-- Carol confirms, validator-1 confirms, validator-2 confirms
+-- RESULT: validator-5 slashed
|   +-- Penalty: -50% reputation (0.40 -> 0.20)
|   +-- Carol reward for reporting: 5% of slash = 0.01 rep bonus
|   +-- validator-5 weight drops: 0.16 -> 0.04
+-- STATUS: Byzantine behavior punished
```

**Simulation Results**:

| Validator Metric | Carol's Performance |
|------------------|---------------------|
| Uptime | 98% |
| Correct votes | 100% |
| Proposals accepted | 100% |
| Reputation growth | 0.50 -> 0.78 (56% increase) |
| Slashing events (self) | 0 |
| Byzantine detection | 1 reported, confirmed |
| Net rewards | 347 MYC + reporting bonus |

---

## Simulation Summary

| Actor Type | Goals Achieved | Protocol Response |
|------------|----------------|-------------------|
| Byzantine Attacker | Mostly Failed | Slashing, PoGQ rejection, rep^2 defense |
| Honest FL Participant | Fully Achieved | Rewards, privacy preserved, reputation growth |
| Governance Participant | Fully Achieved | Proposal passed, parameter changed |
| Validator Operator | Fully Achieved | Rewards, uptime incentivized, detected byzantine |

---

## Key Protocol Observations

### Strengths Validated
1. **45% Byzantine tolerance** holds against 20% coordinated attacker
2. **Reputation^2 weighting** makes sybil attacks economically infeasible
3. **PoGQ validation** effectively filters low-quality/derivative contributions
4. **Double-vote detection** correctly identifies and slashes equivocation
5. **Consciousness gating** adds meaningful signal to governance decisions
6. **Differential privacy** integration preserves data while enabling ML

### Areas for Monitoring
1. **Abstention manipulation** (per security audit H-02) not tested
2. **Long-term reputation inflation** needs observation over many rounds
3. **Cross-round collusion** patterns may emerge in larger networks
4. **Privacy budget exhaustion** strategies need incentive alignment

---

**Disclaimer**: These are simulated scenarios for planning and educational purposes. Actual network behavior may vary based on implementation details and real-world conditions.
