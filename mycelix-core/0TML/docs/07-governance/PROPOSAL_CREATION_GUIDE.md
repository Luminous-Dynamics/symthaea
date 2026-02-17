# Proposal Creation Guide

**Version**: 1.0
**Status**: Production Ready
**Last Updated**: November 11, 2025

---

## Overview

This guide walks through creating governance proposals in Zero-TrustML. Proposals enable community decision-making on:

- FL hyperparameter changes
- Participant management (bans, unbans)
- Capability requirement modifications
- Emergency actions
- Economic policy changes

---

## Prerequisites

Before creating a proposal, ensure you have:

1. **Identity Verification**: At least E2 assurance level
2. **Reputation**: At least 0.6 reputation score
3. **Capability**: `submit_mip` capability (checked automatically)
4. **Rate Limit**: Haven't exceeded 10 proposals in the last 24 hours

You can check your eligibility:
```python
# Check if you can create proposals
authorized, reason = await gov_coord.capability_enforcer.check_capability(
    participant_id="your_participant_id",
    capability_id="submit_mip"
)

if not authorized:
    print(f"Cannot create proposals: {reason}")
```

---

## Proposal Types

Zero-TrustML supports five proposal types:

### 1. PARAMETER_CHANGE
Change FL training hyperparameters (learning rate, batch size, etc.)

**Example Use Cases**:
- Increase learning rate for faster convergence
- Reduce batch size for memory-constrained devices
- Change aggregation method
- Modify differential privacy parameters

### 2. PARTICIPANT_MANAGEMENT
Add, remove, or modify participant permissions

**Example Use Cases**:
- Ban malicious participant (temporary or permanent)
- Unban previously banned participant
- Grant special permissions to trusted participants

### 3. CAPABILITY_UPDATE
Modify capability requirements

**Example Use Cases**:
- Lower reputation requirement for emergency stops
- Increase assurance level for treasury withdrawals
- Adjust rate limits for specific actions

### 4. ECONOMIC_PROPOSAL
Change reward distribution or economic parameters

**Example Use Cases**:
- Modify reward multipliers
- Change reward distribution schedule
- Update economic incentive structure

### 5. EMERGENCY_ACTION
Propose emergency actions (alternatively can use guardian authorization)

**Example Use Cases**:
- Emergency stop FL training
- Emergency parameter rollback
- Emergency ban of multiple participants

---

## Proposal Lifecycle

All proposals follow this lifecycle:

```
1. DRAFT      → Created, not yet submitted
2. REVIEW     → 1-hour review period for feedback (mandatory)
3. VOTING     → Active voting period (7 days default)
4. APPROVED   → Quorum met, approval threshold exceeded
    OR
4. REJECTED   → Quorum met, approval threshold not met
5. EXECUTED   → Approved proposal executed successfully
    OR
5. FAILED     → Execution failed (with error details)
```

---

## Creating a Proposal

### Step 1: Prepare Proposal Details

Gather the following information:

- **Title**: Clear, concise summary (max 100 characters)
- **Description**: Detailed explanation of what and why (max 5000 characters)
- **Proposal Type**: One of the 5 types above
- **Execution Parameters**: Action-specific parameters (JSON)
- **Voting Duration**: How long voting should last (default: 7 days)
- **Quorum**: Minimum participation required (default: 50%)
- **Approval Threshold**: Required approval percentage (default: 66%)

### Step 2: Create the Proposal

```python
from zerotrustml.governance import GovernanceCoordinator, ProposalType

# Initialize coordinator (assuming you have dht_client and identity_coordinator)
gov_coord = GovernanceCoordinator(
    dht_client=dht_client,
    identity_coordinator=identity_coordinator
)

# Create proposal
success, message, proposal_id = await gov_coord.create_governance_proposal(
    proposer_participant_id="your_participant_id",
    proposal_type=ProposalType.PARAMETER_CHANGE,  # or "PARAMETER_CHANGE"
    title="Increase learning rate to 0.02",
    description="""
## Motivation

Current learning rate (0.01) is too conservative, causing slow convergence.

## Proposal

Increase learning rate from 0.01 to 0.02.

## Analysis

- Convergence time reduced by ~30% in simulations
- No accuracy loss observed
- Tested on 3 different datasets

## Risks

- Slight increase in training instability for some models
- Mitigation: Participants can reduce local learning rate if needed
    """,
    execution_params={
        "parameter": "learning_rate",
        "current_value": 0.01,
        "new_value": 0.02,
        "justification": "Faster convergence with no accuracy loss"
    },
    voting_duration_days=7,  # Optional: default is 7
    quorum=0.5,              # Optional: default is 0.5
    approval_threshold=0.66, # Optional: default is 0.66
    tags=["hyperparameters", "performance"]  # Optional
)

if success:
    print(f"Proposal created successfully!")
    print(f"Proposal ID: {proposal_id}")
    print(f"Review period: 1 hour")
    print(f"Voting starts: {message}")
else:
    print(f"Failed to create proposal: {message}")
```

### Step 3: Share Proposal with Community

After creation, the proposal enters a **1-hour review period**. Use this time to:

1. **Announce the proposal**: Share proposal ID on community channels
2. **Provide context**: Explain motivation and expected impact
3. **Gather feedback**: Listen to community concerns
4. **Refine if needed**: Can withdraw and resubmit with changes

```python
# Get proposal details to share
proposal = await gov_coord.proposal_mgr.get_proposal(proposal_id)

print(f"Proposal: {proposal.title}")
print(f"View at: https://governance.mycelix.net/proposals/{proposal_id}")
print(f"Voting starts in: {proposal.voting_start - current_time} seconds")
```

### Step 4: Voting Period Begins

After the 1-hour review period, voting automatically begins. The proposal will be in `VOTING` status for the configured duration (default: 7 days).

During voting:
- **Monitor votes**: Track voting participation and sentiment
- **Answer questions**: Engage with community about the proposal
- **Provide updates**: Share any new information or clarifications

```python
# Check proposal status
proposal = await gov_coord.proposal_mgr.get_proposal(proposal_id)
print(f"Status: {proposal.status}")
print(f"Votes for: {proposal.total_votes_for}")
print(f"Votes against: {proposal.total_votes_against}")
print(f"Votes abstain: {proposal.total_votes_abstain}")
print(f"Total voting power: {proposal.total_voting_power}")
```

### Step 5: Voting Period Ends

When voting ends, the coordinator automatically tallies votes and determines outcome:

**Approved** if:
1. Quorum met: `total_voting_power >= quorum × eligible_voting_power`
2. Approval threshold met: `votes_for / (votes_for + votes_against) >= approval_threshold`

**Rejected** if quorum met but approval threshold not met.

**Expired** if quorum not met (proposal did not receive sufficient participation).

### Step 6: Execution (if Approved)

Approved proposals are automatically executed by the coordinator:

```python
# Manual execution trigger (if automatic execution is disabled)
success, message, result = await gov_coord.finalize_proposal(proposal_id)

if success:
    print(f"Proposal executed successfully!")
    print(f"Result: {result}")
else:
    print(f"Execution failed: {message}")
```

---

## Proposal Examples

### Example 1: Parameter Change Proposal

```python
success, message, proposal_id = await gov_coord.create_governance_proposal(
    proposer_participant_id="alice_id",
    proposal_type=ProposalType.PARAMETER_CHANGE,
    title="Increase minimum reputation to 0.7",
    description="""
## Problem

Recent influx of low-reputation participants has increased Byzantine attack risk.

## Solution

Increase minimum reputation threshold from 0.5 to 0.7 for FL participation.

## Impact

- Estimated 15% reduction in participant pool
- Estimated 40% reduction in Byzantine attack surface
- Existing participants (rep < 0.7) have 30-day grace period

## Data

Analysis of last 100 rounds shows participants with rep < 0.7 have:
- 3.2x higher Byzantine detection rate
- 1.8x lower gradient quality (PoGQ scores)
- 2.1x higher variance in contributions
    """,
    execution_params={
        "parameter": "min_reputation",
        "current_value": 0.5,
        "new_value": 0.7,
        "grace_period_days": 30
    },
    voting_duration_days=7,
    approval_threshold=0.66  # 66% approval required
)
```

### Example 2: Ban Participant Proposal

```python
success, message, proposal_id = await gov_coord.create_governance_proposal(
    proposer_participant_id="bob_id",
    proposal_type=ProposalType.PARTICIPANT_MANAGEMENT,
    title="Ban participant eve_id for Byzantine attacks",
    description="""
## Evidence of Malicious Behavior

Participant eve_id has consistently submitted Byzantine updates:

**Rounds 45-50 Analysis:**
- Round 45: PoGQ score 0.20 (threshold: 0.70)
- Round 46: PoGQ score 0.15 (threshold: 0.70)
- Round 47: PoGQ score 0.10 (threshold: 0.70)
- Round 48: PoGQ score 0.18 (threshold: 0.70)
- Round 49: PoGQ score 0.22 (threshold: 0.70)
- Round 50: PoGQ score 0.14 (threshold: 0.70)

6/6 rounds below threshold = 100% Byzantine detection rate

**Impact on Network:**
- Model accuracy reduced by 3.2% during this period
- 12 other participants flagged eve_id as suspicious
- Guardian network confirms malicious behavior

## Proposed Action

Temporary ban for 30 days to investigate and allow eve_id to improve behavior.

## Appeal Process

eve_id can submit appeal after 30 days with evidence of improved behavior.
    """,
    execution_params={
        "action": "ban_participant",
        "target_participant_id": "eve_id",
        "permanent": False,
        "ban_duration_seconds": 86400 * 30,  # 30 days
        "reason": "Consistent Byzantine attacks in rounds 45-50",
        "evidence": {
            "rounds": [45, 46, 47, 48, 49, 50],
            "pogq_scores": [0.20, 0.15, 0.10, 0.18, 0.22, 0.14],
            "detection_method": "PoGQ threshold violation"
        }
    },
    voting_duration_days=7,
    approval_threshold=0.75,  # Higher threshold for bans
    tags=["security", "participant_management", "byzantine"]
)
```

### Example 3: Capability Update Proposal

```python
success, message, proposal_id = await gov_coord.create_governance_proposal(
    proposer_participant_id="carol_id",
    proposal_type=ProposalType.CAPABILITY_UPDATE,
    title="Lower emergency_stop reputation requirement",
    description="""
## Current Situation

emergency_stop capability requires:
- Reputation: 0.8
- Assurance: E3
- Guardian approval: Yes (0.7 threshold)

## Problem

Only ~15% of participants meet reputation threshold (0.8).
During recent attack (round 67), no eligible participants were online.
Attack persisted for 2 hours before guardians manually intervened.

## Proposed Change

Lower reputation requirement from 0.8 to 0.7.

## Analysis

- ~35% of participants have reputation >= 0.7
- Guardian approval (0.7 threshold) still provides security
- False emergency stop rate estimated at <2% (guardian filter)

## Risks

- Slightly higher false positive rate for emergency stops
- Mitigation: Guardian approval still required, false stops can be quickly reversed
    """,
    execution_params={
        "capability_id": "emergency_stop",
        "field": "required_reputation",
        "current_value": 0.8,
        "new_value": 0.7
    },
    voting_duration_days=7,
    approval_threshold=0.66,
    tags=["capabilities", "security", "emergency"]
)
```

### Example 4: Economic Proposal

```python
success, message, proposal_id = await gov_coord.create_governance_proposal(
    proposer_participant_id="dave_id",
    proposal_type=ProposalType.ECONOMIC_PROPOSAL,
    title="Increase rewards for high-quality contributions",
    description="""
## Current Reward System

All participants receive equal rewards (100 tokens per round).
No differentiation based on contribution quality.

## Problem

High-quality participants (PoGQ > 0.9) are not incentivized more than marginal participants (PoGQ ~ 0.7).
Leads to mediocre contributions and participant churn.

## Proposed Change

Implement reputation-weighted rewards:
- Base reward: 100 tokens
- Multiplier: 0.5x to 2.0x based on vote_weight
- Formula: reward = base × (0.5 + vote_weight / max_vote_weight × 1.5)

## Expected Impact

- High-reputation participants: Up to 2x rewards
- Low-reputation participants: At least 0.5x rewards (50 tokens minimum)
- Estimated retention increase: +25% for high-quality participants
- Estimated quality increase: +15% average PoGQ scores

## Budget Impact

- Total rewards increase: ~20% (from redistribution to high performers)
- Funded by: Reduced rewards to low performers
- Net cost: Minimal (redistribution, not new issuance)
    """,
    execution_params={
        "policy": "reputation_weighted_rewards",
        "enabled": True,
        "multiplier_min": 0.5,
        "multiplier_max": 2.0,
        "base_reward": 100
    },
    voting_duration_days=10,  # Longer voting period for economic changes
    approval_threshold=0.66,
    tags=["economics", "rewards", "incentives"]
)
```

---

## Best Practices

### 1. Write Clear Proposals

**Do**:
- Use clear, descriptive titles
- Structure description with headings
- Provide data and analysis
- Explain risks and mitigations
- Use markdown formatting

**Don't**:
- Use vague or misleading titles
- Wall-of-text descriptions
- Make claims without evidence
- Ignore potential downsides
- Use ALL CAPS or excessive emphasis

### 2. Gather Community Input

**Before submitting**:
- Discuss informally on community channels
- Gauge sentiment and gather feedback
- Refine proposal based on input
- Identify potential concerns

**During review period**:
- Answer questions promptly
- Acknowledge concerns
- Provide additional data if requested
- Be open to withdrawing if needed

### 3. Set Appropriate Parameters

**Voting Duration**:
- Standard proposals: 7 days
- Urgent proposals: 3 days (minimum)
- Major changes: 10-14 days

**Approval Threshold**:
- Minor changes: 60% (0.6)
- Standard changes: 66% (0.66)
- Major changes: 75% (0.75)
- Constitutional changes: 80% (0.8)

**Quorum**:
- Standard: 50% (0.5)
- High participation expected: 60% (0.6)
- Emergency proposals: 40% (0.4)

### 4. Provide Execution Details

Execution parameters should be:
- **Complete**: All information needed to execute
- **Unambiguous**: Clear what action will be taken
- **Verifiable**: Easily checked if executed correctly
- **Reversible**: If possible, allow undo via another proposal

```python
# Good execution params
execution_params={
    "parameter": "learning_rate",
    "current_value": 0.01,
    "new_value": 0.02,
    "effective_round": 100,  # When to apply change
    "rollback_plan": "proposal_xxx"  # How to undo if needed
}

# Bad execution params
execution_params={
    "change": "make it better"  # Too vague!
}
```

### 5. Monitor Your Proposal

After submission:
- Check voting progress daily
- Respond to questions in community channels
- Provide updates if situation changes
- Withdraw if proposal becomes obsolete

```python
# Monitor proposal status
while True:
    proposal = await gov_coord.proposal_mgr.get_proposal(proposal_id)

    print(f"Status: {proposal.status}")
    print(f"Participation: {proposal.total_voting_power / eligible_power:.1%}")
    print(f"Approval: {proposal.total_votes_for / (proposal.total_votes_for + proposal.total_votes_against):.1%}")

    if proposal.status in ["APPROVED", "REJECTED", "FAILED"]:
        break

    await asyncio.sleep(3600)  # Check hourly
```

---

## Proposal Withdrawal

You can withdraw a proposal during the review period (before voting starts):

```python
# Withdraw proposal
success, message = await gov_coord.proposal_mgr.withdraw_proposal(
    proposal_id=proposal_id,
    proposer_participant_id="your_participant_id",
    reason="Community feedback suggests alternative approach"
)

if success:
    print("Proposal withdrawn successfully")
else:
    print(f"Cannot withdraw: {message}")
```

**Note**: Once voting starts, proposals cannot be withdrawn. This ensures voting integrity.

---

## Troubleshooting

### "Not authorized to submit proposals"

**Cause**: You don't meet capability requirements.

**Solution**:
1. Check your identity assurance level (need E2+)
2. Check your reputation (need 0.6+)
3. Check your Sybil resistance (need 0.5+)
4. Wait for identity verification to complete

### "Rate limit exceeded"

**Cause**: You've submitted 10+ proposals in the last 24 hours.

**Solution**:
- Wait for rate limit window to reset
- Consider combining multiple proposals if related

### "Proposal failed validation"

**Cause**: Execution parameters are invalid or incomplete.

**Solution**:
- Check execution_params match proposal type requirements
- Ensure all required fields are present
- Verify parameter values are valid (e.g., probabilities in 0-1 range)

### "Voting did not reach quorum"

**Cause**: Not enough participants voted.

**Solution**:
- Promote proposal more actively next time
- Consider lowering quorum for similar proposals
- Make sure proposal is relevant to community

### "Proposal rejected"

**Cause**: Approval threshold not met (not enough "for" votes).

**Solution**:
- Review feedback from "against" voters
- Refine proposal and resubmit with improvements
- Consider alternative approaches

---

## API Reference

### Create Proposal

```python
success, message, proposal_id = await gov_coord.create_governance_proposal(
    proposer_participant_id: str,
    proposal_type: Union[ProposalType, str],
    title: str,
    description: str,
    execution_params: Dict[str, Any],
    voting_duration_days: int = 7,
    quorum: float = 0.5,
    approval_threshold: float = 0.66,
    tags: Optional[List[str]] = None
) -> Tuple[bool, str, Optional[str]]
```

### Get Proposal

```python
proposal = await gov_coord.proposal_mgr.get_proposal(
    proposal_id: str
) -> ProposalData
```

### List Proposals

```python
proposals = await gov_coord.proposal_mgr.list_proposals(
    status: Optional[str] = None,
    limit: int = 50
) -> List[ProposalData]
```

### Withdraw Proposal

```python
success, message = await gov_coord.proposal_mgr.withdraw_proposal(
    proposal_id: str,
    proposer_participant_id: str,
    reason: str
) -> Tuple[bool, str]
```

---

## Further Reading

- **[Governance System Architecture](./GOVERNANCE_SYSTEM_ARCHITECTURE.md)** - Complete system overview
- **[Capability Registry](./CAPABILITY_REGISTRY.md)** - Capability requirements
- **[Voting Mechanics](./VOTING_MECHANICS.md)** - How voting works
- **[Guardian Authorization Guide](./GUARDIAN_AUTHORIZATION_GUIDE.md)** - Guardian approval process

---

**Document Status**: Phase 6 User Documentation Complete
**Version**: 1.0
**Last Updated**: November 11, 2025
