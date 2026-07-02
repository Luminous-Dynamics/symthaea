# Voting Mechanics

**Version**: 1.0
**Status**: Production Ready
**Last Updated**: November 11, 2025

---

## Overview

Zero-TrustML's governance system implements **reputation-weighted quadratic voting** to achieve:

- **Sybil resistance**: New identities have low vote power
- **Byzantine resistance**: Reputation-weighted power prevents 51% attacks
- **Fair representation**: Quadratic voting prevents plutocracy
- **Progressive trust**: Vote power grows with verified contributions

This document explains the mathematical foundations and practical mechanics of the voting system.

---

## Core Voting Formula

### Effective Votes Calculation

```
effective_votes = sqrt(credits_spent) × vote_weight

Where:
  credits_spent = Credits allocated to this vote (from budget)
  vote_weight = Identity verification and reputation multiplier
```

**Key Properties**:
1. **Quadratic Cost**: Doubling vote power requires 4x credits
2. **Weighted Power**: High-reputation voters get more votes per credit
3. **Budget Constraints**: Total credits limited by vote_weight

---

## Vote Weight Calculation

Vote weight combines four factors to determine voting power:

```python
vote_weight = base_weight × (1 + sybil_resistance) × assurance_factor × reputation_factor

Components:
  base_weight = 1.0                          # Equal baseline
  sybil_resistance = 0.0-1.0                # Identity uniqueness
  assurance_factor = {                      # Identity verification level
      E0: 1.0,  # Unverified
      E1: 1.2,  # Testimonial
      E2: 1.5,  # Privately verifiable
      E3: 2.0,  # Cryptographically proven
      E4: 3.0   # Publicly reproducible
  }
  reputation_factor = 0.5 + (reputation × 0.5)  # 0.5-1.0 range
```

### Factor Breakdown

#### 1. Base Weight (Always 1.0)
All participants start with equal baseline. This ensures even low-reputation participants have meaningful voice.

#### 2. Sybil Resistance Bonus (0-100%)
Measures identity uniqueness to prevent Sybil attacks:

| Score | Interpretation | Bonus |
|-------|----------------|-------|
| 0.0 | Brand new identity | +0% |
| 0.3 | Recent identity with some activity | +30% |
| 0.6 | Established identity with history | +60% |
| 1.0 | Verified unique identity with strong signals | +100% |

**Calculated From**:
- Identity age (older = higher)
- Activity patterns (consistent = higher)
- Social graph (connections to established participants)
- Resource commitments (staked tokens, verified devices)

#### 3. Assurance Multiplier (1.0x-3.0x)
Based on identity verification level (Epistemic Charter E-axis):

| Level | Name | Multiplier | Example |
|-------|------|------------|---------|
| E0 | Null | 1.0x | Self-created identity |
| E1 | Testimonial | 1.2x | Email-verified account |
| E2 | Privately Verifiable | 1.5x | KYC with privacy |
| E3 | Cryptographically Proven | 2.0x | ZKP credential |
| E4 | Publicly Reproducible | 3.0x | Blockchain identity |

#### 4. Reputation Factor (0.5x-1.0x)
Based on contribution history:

```python
reputation_factor = 0.5 + (reputation × 0.5)

Examples:
  reputation = 0.0  →  factor = 0.5  # Minimum 50% effectiveness
  reputation = 0.5  →  factor = 0.75 # Average effectiveness
  reputation = 1.0  →  factor = 1.0  # Maximum effectiveness
```

**Reputation Sources**:
- FL gradient quality (PoGQ scores)
- Governance participation (proposals, votes)
- Network behavior (uptime, reliability)
- Guardian endorsements

---

## Example Vote Weights

### Alice: High-Trust Participant
```python
Identity: E4 (Publicly Reproducible blockchain identity)
Reputation: 0.9 (Excellent FL contributions)
Sybil Resistance: 0.8 (Established participant)

Calculation:
  base_weight = 1.0
  sybil_bonus = 1.0 + 0.8 = 1.8
  assurance_factor = 3.0 (E4)
  reputation_factor = 0.5 + (0.9 × 0.5) = 0.95

  vote_weight = 1.0 × 1.8 × 3.0 × 0.95 = 5.13

Budget: 5.13 × 100 = 513 credits
Max votes (single proposal): sqrt(513) × 5.13 = 116 effective votes
```

### Bob: Moderate Participant
```python
Identity: E2 (Privately Verifiable KYC)
Reputation: 0.7 (Good FL contributions)
Sybil Resistance: 0.6 (Established)

Calculation:
  base_weight = 1.0
  sybil_bonus = 1.0 + 0.6 = 1.6
  assurance_factor = 1.5 (E2)
  reputation_factor = 0.5 + (0.7 × 0.5) = 0.85

  vote_weight = 1.0 × 1.6 × 1.5 × 0.85 = 2.04

Budget: 2.04 × 100 = 204 credits
Max votes (single proposal): sqrt(204) × 2.04 = 29 effective votes
```

### Carol: New Participant
```python
Identity: E1 (Testimonial email verification)
Reputation: 0.5 (Average contributions)
Sybil Resistance: 0.4 (Recently joined)

Calculation:
  base_weight = 1.0
  sybil_bonus = 1.0 + 0.4 = 1.4
  assurance_factor = 1.2 (E1)
  reputation_factor = 0.5 + (0.5 × 0.5) = 0.75

  vote_weight = 1.0 × 1.4 × 1.2 × 0.75 = 1.26

Budget: 1.26 × 100 = 126 credits
Max votes (single proposal): sqrt(126) × 1.26 = 14 effective votes
```

### Dave: Suspected Sybil
```python
Identity: E0 (Unverified self-created)
Reputation: 0.0 (No contributions yet)
Sybil Resistance: 0.0 (Brand new)

Calculation:
  base_weight = 1.0
  sybil_bonus = 1.0 + 0.0 = 1.0
  assurance_factor = 1.0 (E0)
  reputation_factor = 0.5 + (0.0 × 0.5) = 0.5

  vote_weight = 1.0 × 1.0 × 1.0 × 0.5 = 0.5

Budget: 0.5 × 100 = 50 credits
Max votes (single proposal): sqrt(50) × 0.5 = 3.5 effective votes
```

**Comparison**:
- Alice has **33x more effective votes** than Dave (116 vs 3.5)
- Bob has **8x more effective votes** than Dave (29 vs 3.5)
- Carol has **4x more effective votes** than Dave (14 vs 3.5)

**Sybil Attack Prevention**: Even with 1000 Sybil identities like Dave, total effective votes = 1000 × 3.5 = 3500, which is less than 31 Alice-equivalent participants.

---

## Quadratic Voting

### Why Quadratic?

Quadratic voting prevents vote buying and plutocracy by making large vote concentrations expensive:

```
Cost(n votes) = n²

Examples:
  10 votes: 100 credits
  20 votes: 400 credits (4x cost for 2x votes)
  40 votes: 1600 credits (16x cost for 4x votes)
  100 votes: 10000 credits (100x cost for 10x votes)
```

### Effective Votes with Vote Weight

The actual formula combines quadratic voting with vote weight:

```python
effective_votes = sqrt(credits_spent) × vote_weight

Examples with vote_weight=2.0:
  100 credits → sqrt(100) × 2.0 = 10 × 2.0 = 20 effective votes
  400 credits → sqrt(400) × 2.0 = 20 × 2.0 = 40 effective votes

To double effective votes (20 → 40):
  Need to spend 4x credits (100 → 400)
```

### Credit Budgets

Each participant's budget is:

```python
budget = BASE_BUDGET × vote_weight

Where:
  BASE_BUDGET = 100 credits (configurable)

Examples:
  Alice (vote_weight=5.13): 513 credits
  Bob (vote_weight=2.04): 204 credits
  Carol (vote_weight=1.26): 126 credits
  Dave (vote_weight=0.5): 50 credits
```

Budgets can be split across multiple proposals:

```python
# Alice splits 513 credits across 3 proposals
Proposal A: 300 credits → sqrt(300) × 5.13 = 89 votes
Proposal B: 150 credits → sqrt(150) × 5.13 = 63 votes
Proposal C: 63 credits → sqrt(63) × 5.13 = 41 votes
Total: 193 votes across 3 proposals

# vs. spending all on one proposal
Proposal A: 513 credits → sqrt(513) × 5.13 = 116 votes
```

---

## Casting Votes

### Vote Structure

```python
@dataclass
class Vote:
    vote_id: str                   # Unique identifier
    proposal_id: str               # Which proposal
    voter_did: str                 # Voter's DID (W3C standard)
    voter_participant_id: str      # Voter's participant ID
    choice: VoteChoice             # FOR, AGAINST, or ABSTAIN
    credits_spent: int             # Credits allocated
    vote_weight: float             # Voter's weight at vote time
    effective_votes: float         # sqrt(credits) × vote_weight
    timestamp: int                 # When vote was cast
    signature: str                 # Cryptographic signature
```

### Voting Process

```python
from zerotrustml.governance import GovernanceCoordinator, VoteChoice

# Cast vote
success, message = await gov_coord.voting_engine.cast_vote(
    voter_participant_id="alice_id",
    proposal_id="prop_12345",
    choice=VoteChoice.FOR,        # or AGAINST, or ABSTAIN
    credits_spent=100              # From your budget
)

if success:
    print("Vote cast successfully!")
else:
    print(f"Vote failed: {message}")
```

### Vote Choices

1. **FOR**: Support the proposal
2. **AGAINST**: Oppose the proposal
3. **ABSTAIN**: Participate in quorum without affecting outcome

**Abstain votes**:
- Count toward quorum (participation requirement)
- Do not count toward approval threshold
- Useful when voter wants to let others decide but ensure quorum is met

---

## Vote Tallying

### Tallying Process

After voting period ends, the coordinator tallies votes:

```python
# Get vote results
results = await gov_coord.voting_engine.tally_votes(proposal_id)

print(f"For: {results['for_votes']}")
print(f"Against: {results['against_votes']}")
print(f"Abstain: {results['abstain_votes']}")
print(f"Total: {results['total_power']}")
print(f"Participation: {results['participation']:.1%}")
print(f"Approval Rate: {results['approval_rate']:.1%}")
print(f"Result: {'APPROVED' if results['approved'] else 'REJECTED'}")
```

### Approval Criteria

A proposal is **APPROVED** if both conditions are met:

#### 1. Quorum Requirement

```python
participation = total_voting_power / eligible_voting_power
quorum_met = participation >= proposal.quorum

Where:
  total_voting_power = sum of all effective votes (FOR + AGAINST + ABSTAIN)
  eligible_voting_power = total possible votes from all eligible participants
  proposal.quorum = minimum participation (default: 0.5 = 50%)
```

**Example**:
```
Eligible voting power: 10,000 votes
Total votes cast: 5,500 votes
Participation: 5,500 / 10,000 = 55%
Quorum (50%): MET ✅
```

#### 2. Approval Threshold

```python
approval_rate = votes_for / (votes_for + votes_against)
approved = approval_rate >= proposal.approval_threshold

Where:
  votes_for = sum of effective votes with choice=FOR
  votes_against = sum of effective votes with choice=AGAINST
  proposal.approval_threshold = required approval (default: 0.66 = 66%)

Note: ABSTAIN votes are excluded from approval calculation
```

**Example**:
```
Votes FOR: 3,800
Votes AGAINST: 1,200
Votes ABSTAIN: 500

Approval rate: 3,800 / (3,800 + 1,200) = 76%
Threshold (66%): MET ✅

Proposal APPROVED ✅
```

### Proposal Outcomes

| Quorum | Approval | Outcome | Next Step |
|--------|----------|---------|-----------|
| ✅ Met | ✅ Met | **APPROVED** | Automatic execution |
| ✅ Met | ❌ Not Met | **REJECTED** | No action, audit trail |
| ❌ Not Met | N/A | **EXPIRED** | No action, insufficient participation |

---

## Voting Strategies

### For Voters

#### Strategy 1: Concentrated Voting
Spend all credits on most important proposals:

**Pros**:
- Maximum impact on key issues
- Full voting power on priorities

**Cons**:
- No voice on other proposals
- Risk of wasted budget if outcome is decisive

**Best for**: Strong opinions on specific proposals

#### Strategy 2: Distributed Voting
Spread credits across multiple proposals:

**Pros**:
- Influence on more decisions
- Diversified impact

**Cons**:
- Lower impact per proposal
- May not reach threshold on any single proposal

**Best for**: Moderate opinions on many proposals

#### Strategy 3: Strategic Abstention
Vote ABSTAIN on proposals you don't feel strongly about:

**Pros**:
- Helps meet quorum
- Lets community decide
- Saves credits for other proposals

**Cons**:
- No influence on outcome
- Can't complain if outcome is unfavorable

**Best for**: Proposals outside your expertise

### For Proposers

#### Setting Quorum

- **High participation expected (60%+)**: Relevant to most participants
- **Standard (50%)**: Typical proposal
- **Lower (40%)**: Niche or technical proposal

#### Setting Approval Threshold

- **Minor changes (60%)**: Tweaks, reversible changes
- **Standard (66%)**: Most proposals, 2/3 majority
- **Major changes (75%)**: Significant impact, security-critical
- **Constitutional (80%+)**: Fundamental changes

---

## Security Properties

### Sybil Attack Resistance

**Attack**: Create many fake identities to manipulate votes.

**Defense**:
1. New identities have low vote_weight (~0.5)
2. Quadratic voting limits impact of many small votes
3. Reputation accumulation is slow

**Example**:
```
Attacker creates 1000 Sybil identities:
  Each Sybil: vote_weight = 0.5, budget = 50 credits
  Total budget: 1000 × 50 = 50,000 credits

Attacker's total effective votes:
  = sum(sqrt(50) × 0.5 for each Sybil)
  = 1000 × 3.5
  = 3,500 votes

vs. 31 high-reputation participants (like Alice):
  Each: vote_weight = 5.13, budget = 513 credits
  Max votes each: sqrt(513) × 5.13 = 116
  Total: 31 × 116 = 3,596 votes

Attacker's 1000 Sybils ≈ 31 honest participants
```

### Vote Buying Prevention

**Attack**: Buy votes from participants to manipulate outcome.

**Defense**: Quadratic voting makes large vote concentrations expensive.

**Example**:
```
Attacker wants to buy 1000 votes:
  At vote_weight=2.0, needs sqrt(credits) × 2.0 = 1000
  → sqrt(credits) = 500
  → credits = 250,000

If buying at $1 per credit: $250,000 cost

vs. buying 10 smaller votes (100 votes each):
  Each vote: sqrt(credits) × 2.0 = 100 → credits = 2,500
  Total: 10 × 2,500 = 25,000 credits → $25,000 cost

Quadratic voting makes concentration 10x more expensive!
```

### Byzantine Attack Resistance

**Attack**: Byzantine participants coordinate to manipulate governance.

**Defense**:
1. Reputation-weighted voting reduces Byzantine power
2. Byzantine participants have low reputation (by definition)
3. 45% Byzantine tolerance from MATL carries over to governance

**Example**:
```
Network: 100 participants
Byzantine: 45 participants (45%)

Honest participants:
  Average reputation: 0.8
  Average vote_weight: 4.0
  Total effective votes: 55 × 4.0 × sqrt(100) = 2,200

Byzantine participants:
  Average reputation: 0.3 (low by design)
  Average vote_weight: 1.2
  Total effective votes: 45 × 1.2 × sqrt(100) = 540

Byzantine power: 540 / 2,740 = 19.7% < 33% ✅

System remains secure even at 45% Byzantine participants!
```

---

## Advanced Topics

### Delegation (Future)

**Concept**: Delegate your voting power to a trusted representative.

**Benefits**:
- Participate without deep knowledge
- Leverage expert judgment
- Maintain ability to override

**Implementation** (not in Week 7-8):
```python
# Delegate to expert
await gov_coord.delegate_voting_power(
    delegator="alice_id",
    delegate="bob_id",
    percentage=0.5,  # Delegate 50% of voting power
    proposals=["security", "technical"]  # Only for tagged proposals
)

# Delegate votes on Alice's behalf
await gov_coord.voting_engine.cast_delegated_vote(
    delegate="bob_id",
    delegator="alice_id",
    proposal_id="prop_12345",
    choice=VoteChoice.FOR,
    credits_spent=50  # From Alice's delegated budget
)
```

### Conviction Voting (Future)

**Concept**: Vote weight increases over time based on commitment.

**Benefits**:
- Rewards long-term thinking
- Discourages tactical voting
- Amplifies strongly-held beliefs

**Implementation** (not in Week 7-8):
```python
# Cast conviction vote
await gov_coord.voting_engine.cast_conviction_vote(
    voter_participant_id="alice_id",
    proposal_id="prop_12345",
    choice=VoteChoice.FOR,
    credits_spent=100,
    conviction_days=30  # Vote weight increases over 30 days
)

# After 15 days: effective_votes = sqrt(100) × 5.13 × (15/30) = 58
# After 30 days: effective_votes = sqrt(100) × 5.13 = 116 (full power)
```

### Futarchy (Future)

**Concept**: Vote on values, bet on outcomes.

**Benefits**:
- Separates goals from methods
- Market-based outcome prediction
- Incentivizes accurate predictions

**Implementation** (not in Week 7-8):
```python
# Create futarchy proposal
await gov_coord.create_futarchy_proposal(
    proposer_participant_id="alice_id",
    goal="Maximize model accuracy",
    metric="validation_accuracy",
    options=[
        {"change": "learning_rate", "value": 0.02},
        {"change": "learning_rate", "value": 0.01}
    ],
    betting_duration_days=7
)

# Participants bet on which option achieves goal better
# Option with highest expected utility wins
```

---

## API Reference

### Cast Vote

```python
success, message = await gov_coord.voting_engine.cast_vote(
    voter_participant_id: str,
    proposal_id: str,
    choice: VoteChoice,        # FOR, AGAINST, or ABSTAIN
    credits_spent: int
) -> Tuple[bool, str]
```

### Get Vote Weight

```python
vote_weight = await gov_coord.gov_extensions.calculate_vote_weight(
    participant_id: str
) -> float
```

### Tally Votes

```python
results = await gov_coord.voting_engine.tally_votes(
    proposal_id: str
) -> Dict[str, Any]

# Returns:
# {
#     "for_votes": float,
#     "against_votes": float,
#     "abstain_votes": float,
#     "total_power": float,
#     "participation": float,  # 0.0-1.0
#     "quorum_met": bool,
#     "approval_rate": float,  # 0.0-1.0
#     "approved": bool
# }
```

### Get Voting History

```python
votes = await gov_coord.voting_engine.get_votes(
    proposal_id: str,
    voter_participant_id: Optional[str] = None
) -> List[Vote]
```

---

## Further Reading

- **[Governance System Architecture](./GOVERNANCE_SYSTEM_ARCHITECTURE.md)** - Complete system overview
- **[Capability Registry](./CAPABILITY_REGISTRY.md)** - Capability requirements
- **[Proposal Creation Guide](./PROPOSAL_CREATION_GUIDE.md)** - How to create proposals
- **[Guardian Authorization Guide](./GUARDIAN_AUTHORIZATION_GUIDE.md)** - Guardian approval workflows

---

**Document Status**: Phase 6 User Documentation Complete
**Version**: 1.0
**Last Updated**: November 11, 2025
