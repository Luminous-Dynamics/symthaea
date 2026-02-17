# Capability Registry Reference

**Version**: 1.0
**Status**: Production Ready
**Last Updated**: November 11, 2025

---

## Overview

The **Capability Registry** defines fine-grained permissions for governance and FL operations in Zero-TrustML. Each capability specifies:

- **Identity requirements**: Minimum assurance level (E0-E4)
- **Reputation thresholds**: Minimum reputation score (0.0-1.0)
- **Sybil resistance**: Minimum Sybil resistance score (0.0-1.0)
- **Guardian approval**: Whether guardian network approval is required
- **Rate limits**: Maximum invocations per time period

---

## Capability Structure

```python
@dataclass
class Capability:
    capability_id: str                   # Unique identifier
    name: str                            # Human-readable name
    description: str                     # Description of what this allows
    required_assurance: str              # E0, E1, E2, E3, or E4
    required_reputation: float           # 0.0-1.0
    required_sybil_resistance: float     # 0.0-1.0
    required_guardian_approval: bool     # True if guardians must approve
    guardian_threshold: float            # 0.0-1.0 (if guardian approval required)
    rate_limit: Optional[int]            # Max invocations per period (None = unlimited)
    time_period_seconds: Optional[int]   # Time period for rate limit
```

---

## Built-in Capabilities

### 1. Governance Capabilities

#### `submit_mip` - Submit Mycelix Improvement Proposal

**Purpose**: Create new governance proposals for community voting.

**Requirements**:
- **Assurance Level**: E2 (Privately Verifiable)
- **Reputation**: 0.6
- **Sybil Resistance**: 0.5
- **Guardian Approval**: No
- **Rate Limit**: 10 per 24 hours

**Rationale**: Proposers need moderate verification to prevent spam proposals. Rate limit prevents proposal flooding.

**Example Usage**:
```python
authorized, reason = await gov_coord.enforce_capability(
    participant_id="alice_id",
    capability_id="submit_mip"
)

if authorized:
    # Create proposal
    await gov_coord.create_governance_proposal(...)
```

---

#### `vote_on_mip` - Vote on Proposals

**Purpose**: Cast votes on existing governance proposals.

**Requirements**:
- **Assurance Level**: E1 (Testimonial)
- **Reputation**: 0.4
- **Sybil Resistance**: 0.4
- **Guardian Approval**: No
- **Rate Limit**: None

**Rationale**: Voting should be accessible to most verified participants. Vote weight (not capability) provides Sybil resistance.

**Example Usage**:
```python
authorized, reason = await gov_coord.enforce_capability(
    participant_id="bob_id",
    capability_id="vote_on_mip"
)

if authorized:
    # Cast vote
    await gov_coord.voting_engine.cast_vote(...)
```

---

#### `modify_capability` - Modify Capability Definitions

**Purpose**: Change capability requirements (advanced governance).

**Requirements**:
- **Assurance Level**: E4 (Publicly Reproducible)
- **Reputation**: 0.9
- **Sybil Resistance**: 0.8
- **Guardian Approval**: Yes (0.8 threshold)
- **Rate Limit**: 5 per 30 days

**Rationale**: Changing capabilities affects system security. Requires highest identity verification and guardian consensus.

**Example Usage**:
```python
# This would typically go through a proposal
authorized, reason = await gov_coord.enforce_capability(
    participant_id="admin_id",
    capability_id="modify_capability"
)
```

---

### 2. FL Coordinator Capabilities

#### `submit_update` - Submit Gradient Update

**Purpose**: Submit model gradient updates during FL training.

**Requirements**:
- **Assurance Level**: E1 (Testimonial)
- **Reputation**: 0.5
- **Sybil Resistance**: 0.4
- **Guardian Approval**: No
- **Rate Limit**: None (controlled by FL coordinator)

**Rationale**: FL participation should be accessible. Byzantine detection (PoGQ) provides safety, not capability gates.

**Example Usage**:
```python
# Automatically checked by GovernedFLCoordinator
authorized, reason = await fl_gov.verify_participant_for_round(
    participant_id="alice_id",
    round_number=42
)
```

---

#### `request_model` - Request Global Model

**Purpose**: Request current global model from FL coordinator.

**Requirements**:
- **Assurance Level**: E1 (Testimonial)
- **Reputation**: 0.5
- **Sybil Resistance**: 0.4
- **Guardian Approval**: No
- **Rate Limit**: 100 per hour

**Rationale**: Model requests are lightweight but should be rate-limited to prevent DDoS.

**Example Usage**:
```python
authorized, reason = await gov_coord.authorize_fl_action(
    participant_id="bob_id",
    action="request_model"
)
```

---

### 3. Emergency Action Capabilities

#### `emergency_stop` - Emergency Stop FL Training

**Purpose**: Halt FL training immediately in response to attacks or critical issues.

**Requirements**:
- **Assurance Level**: E3 (Cryptographically Proven)
- **Reputation**: 0.8
- **Sybil Resistance**: 0.7
- **Guardian Approval**: Yes (0.7 threshold)
- **Rate Limit**: 5 per 24 hours

**Rationale**: Emergency stops are disruptive and must be reserved for genuine emergencies. High identity requirements and guardian approval prevent abuse.

**Example Usage**:
```python
# Request emergency stop (creates guardian authorization request)
success, message, request_id = await fl_gov.request_emergency_stop(
    requester_participant_id="alice_id",
    reason="Byzantine attack detected",
    evidence={"byzantine_ratio": 0.48}
)

# Guardians approve
await gov_coord.guardian_auth_mgr.submit_approval(
    guardian_participant_id="guardian_bob",
    request_id=request_id,
    approve=True,
    approval_reason="Confirmed attack via independent analysis"
)
```

---

#### `ban_participant` - Ban Malicious Participant

**Purpose**: Remove participant from FL network (permanent or temporary).

**Requirements**:
- **Assurance Level**: E3 (Cryptographically Proven)
- **Reputation**: 0.8
- **Sybil Resistance**: 0.7
- **Guardian Approval**: Yes (0.8 threshold)
- **Rate Limit**: 10 per 7 days

**Rationale**: Banning affects participant livelihoods. Requires strong evidence and guardian consensus.

**Example Usage**:
```python
# Request ban (creates governance proposal + guardian authorization)
success, message, proposal_id = await fl_gov.request_participant_ban(
    requester_participant_id="alice_id",
    target_participant_id="eve_id",
    reason="Consistent Byzantine attacks in rounds 45-50",
    evidence={
        "rounds": [45, 46, 47, 48, 49, 50],
        "pogq_scores": [0.2, 0.15, 0.1, 0.18, 0.22, 0.14]
    },
    permanent=False,
    ban_duration_seconds=86400 * 30  # 30 days
)
```

---

#### `update_parameters` - Modify FL Hyperparameters

**Purpose**: Change FL training hyperparameters (learning rate, batch size, etc.).

**Requirements**:
- **Assurance Level**: E2 (Privately Verifiable)
- **Reputation**: 0.7
- **Sybil Resistance**: 0.6
- **Guardian Approval**: Yes (0.6 threshold)
- **Rate Limit**: 10 per 7 days

**Rationale**: Parameter changes affect all participants. Guardian approval ensures community agreement.

**Example Usage**:
```python
# Create parameter change proposal
success, message, proposal_id = await gov_coord.create_governance_proposal(
    proposer_participant_id="alice_id",
    proposal_type="PARAMETER_CHANGE",
    title="Increase learning rate",
    description="Increase from 0.01 to 0.02 for faster convergence",
    execution_params={
        "parameter": "learning_rate",
        "current_value": 0.01,
        "new_value": 0.02
    },
    voting_duration_days=7
)
```

---

### 4. Economic Capabilities

#### `distribute_rewards` - Distribute FL Rewards

**Purpose**: Trigger reward distribution to FL participants.

**Requirements**:
- **Assurance Level**: E2 (Privately Verifiable)
- **Reputation**: 0.7
- **Sybil Resistance**: 0.6
- **Guardian Approval**: No
- **Rate Limit**: None (controlled by FL coordinator)

**Rationale**: Reward distribution is automated but can be manually triggered. Moderate verification prevents unauthorized distributions.

**Example Usage**:
```python
# Typically automated, but can be manual
authorized, reason = await gov_coord.authorize_fl_action(
    participant_id="coordinator_id",
    action="distribute_rewards"
)

if authorized:
    await fl_coord.distribute_rewards(round_number=42)
```

---

#### `treasury_withdrawal` - Withdraw from Treasury

**Purpose**: Withdraw funds from community treasury (future capability).

**Requirements**:
- **Assurance Level**: E4 (Publicly Reproducible)
- **Reputation**: 0.9
- **Sybil Resistance**: 0.8
- **Guardian Approval**: Yes (0.9 threshold)
- **Rate Limit**: 5 per 30 days

**Rationale**: Treasury withdrawals affect entire community. Requires highest verification and near-unanimous guardian approval.

**Example Usage**:
```python
# Would require governance proposal + guardian approval
# (Treasury system not implemented in Week 7-8)
```

---

## Identity Assurance Levels

Capabilities reference identity assurance levels from the Mycelix Protocol Epistemic Charter v2.0:

| Level | Name | Description | Example |
|-------|------|-------------|---------|
| **E0** | Null | Unverifiable | Self-created identity |
| **E1** | Testimonial | Personal attestation | Email-verified account |
| **E2** | Privately Verifiable | Audit guild verification | KYC with privacy |
| **E3** | Cryptographically Proven | ZKP verification | Credential proof |
| **E4** | Publicly Reproducible | Open verification | Blockchain identity |

**Assurance Multipliers** (for vote weight):
- E0: 1.0x
- E1: 1.2x
- E2: 1.5x
- E3: 2.0x
- E4: 3.0x

---

## Reputation Scores

Reputation is calculated from:
1. **FL Contributions**: Gradient quality (PoGQ scores)
2. **Governance Participation**: Proposal submissions, voting activity
3. **Network Behavior**: Uptime, responsiveness, reliability
4. **Guardian Approvals**: Endorsements from trusted guardians

**Reputation Range**: 0.0-1.0

**Reputation Effects**:
- **Vote Weight**: `reputation_factor = 0.5 + (reputation × 0.5)` (maps 0.0→0.5, 1.0→1.0)
- **Capability Access**: Many capabilities have minimum reputation thresholds
- **Guardian Weight**: Guardian approvals are weighted by guardian reputation

---

## Sybil Resistance Scores

Sybil resistance measures identity uniqueness:
1. **Identity Age**: Older identities score higher
2. **Activity Patterns**: Consistent activity over time
3. **Social Graph**: Connections to established participants
4. **Resource Commitments**: Staked resources, verified devices

**Sybil Resistance Range**: 0.0-1.0

**Sybil Resistance Effects**:
- **Vote Weight**: `sybil_bonus = 1.0 + sybil_resistance` (adds 0-100% bonus)
- **New Identities**: Typically score 0.0-0.2, giving ~0.6x vote weight
- **Established Identities**: Score 0.7-1.0, giving ~1.7-2.0x vote weight

---

## Guardian Approval

Some capabilities require approval from the participant's guardian network.

### Guardian Approval Process

1. **Request Created**: Participant requests capability that requires guardian approval
2. **Guardians Notified**: System notifies participant's guardians
3. **Guardians Respond**: Each guardian approves or rejects
4. **Weighted Threshold**: System calculates weighted approval
5. **Execution**: If threshold met, action is authorized

### Guardian Weighting

```python
guardian_weight = assurance_factor × reputation

Where:
  assurance_factor = {E0: 1.0, E1: 1.2, E2: 1.5, E3: 2.0, E4: 3.0}
  reputation = 0.0-1.0

approval_weight = Σ(approving_guardian_weights)
total_weight = Σ(all_guardian_weights)
approval_ratio = approval_weight / total_weight

authorized = (approval_ratio >= required_threshold)
```

### Guardian Approval Thresholds

| Capability | Threshold | Interpretation |
|------------|-----------|----------------|
| `emergency_stop` | 0.7 | 70% of guardian weight |
| `ban_participant` | 0.8 | 80% of guardian weight |
| `update_parameters` | 0.6 | 60% of guardian weight |
| `treasury_withdrawal` | 0.9 | 90% of guardian weight |

**Example**:
```
Participant Alice has 5 guardians:
  Guardian 1 (E4, rep=0.9): weight = 3.0 × 0.9 = 2.70
  Guardian 2 (E3, rep=0.8): weight = 2.0 × 0.8 = 1.60
  Guardian 3 (E3, rep=0.7): weight = 2.0 × 0.7 = 1.40
  Guardian 4 (E2, rep=0.6): weight = 1.5 × 0.6 = 0.90
  Guardian 5 (E2, rep=0.5): weight = 1.5 × 0.5 = 0.75

Total weight: 7.35

For emergency_stop (0.7 threshold):
  Required: 0.7 × 7.35 = 5.145

If Guardians 1, 2, 3 approve: 2.70 + 1.60 + 1.40 = 5.70 > 5.145 ✅ APPROVED
If Guardians 1, 2 approve: 2.70 + 1.60 = 4.30 < 5.145 ❌ REJECTED
```

---

## Rate Limiting

Rate limits prevent abuse of sensitive capabilities.

### Rate Limit Enforcement

```python
# Check if participant exceeded rate limit
recent_invocations = await coordinator.count_recent_invocations(
    participant_id=participant_id,
    capability_id=capability_id,
    time_period_seconds=time_period_seconds
)

if recent_invocations >= rate_limit:
    return False, f"Rate limit exceeded: {recent_invocations}/{rate_limit}"
```

### Rate Limit Examples

| Capability | Rate Limit | Period | Rationale |
|------------|------------|--------|-----------|
| `submit_mip` | 10 | 24 hours | Prevent proposal spam |
| `emergency_stop` | 5 | 24 hours | Reserve for genuine emergencies |
| `ban_participant` | 10 | 7 days | Prevent ban abuse |
| `request_model` | 100 | 1 hour | Prevent DDoS on model requests |
| `vote_on_mip` | None | N/A | Voting should be unrestricted |

---

## Custom Capabilities

Communities can define custom capabilities for domain-specific actions.

### Example: Healthcare FL

```python
HEALTHCARE_CAPABILITIES = {
    "submit_patient_data": Capability(
        capability_id="submit_patient_data",
        name="Submit Patient Data",
        description="Submit encrypted patient data for FL training",
        required_assurance="E3",  # HIPAA compliance
        required_reputation=0.8,  # High trust required
        required_sybil_resistance=0.7,
        required_guardian_approval=False,
        guardian_threshold=0.0,
        rate_limit=1000,
        time_period_seconds=3600  # Max 1000 submissions per hour
    ),

    "request_model_weights": Capability(
        capability_id="request_model_weights",
        name="Request Model Weights",
        description="Request trained model weights (with differential privacy)",
        required_assurance="E2",
        required_reputation=0.7,
        required_sybil_resistance=0.6,
        required_guardian_approval=True,  # Model access is sensitive
        guardian_threshold=0.6,
        rate_limit=10,
        time_period_seconds=86400  # Max 10 requests per day
    )
}

# Register custom capabilities
gov_coord.capability_enforcer.register_capabilities(HEALTHCARE_CAPABILITIES)
```

---

## Capability Evolution

Capabilities can be modified via governance proposals.

### Modifying a Capability

```python
# Create proposal to modify capability
success, message, proposal_id = await gov_coord.create_governance_proposal(
    proposer_participant_id="alice_id",
    proposal_type="CAPABILITY_UPDATE",
    title="Reduce emergency_stop reputation requirement",
    description="Lower from 0.8 to 0.7 to allow more participants to request emergency stops",
    execution_params={
        "capability_id": "emergency_stop",
        "field": "required_reputation",
        "current_value": 0.8,
        "new_value": 0.7
    },
    voting_duration_days=7
)

# After proposal approved and executed, capability is updated
capability = gov_coord.capability_enforcer.get_capability("emergency_stop")
assert capability.required_reputation == 0.7
```

---

## API Reference

### Checking Capabilities

```python
# Check if participant has capability
authorized, reason = await gov_coord.capability_enforcer.check_capability(
    participant_id="alice_id",
    capability_id="submit_mip"
)

if authorized:
    # Proceed with action
    await perform_action()
else:
    logger.warning(f"Action blocked: {reason}")
```

### Listing Capabilities

```python
# Get all registered capabilities
capabilities = gov_coord.capability_enforcer.list_capabilities()

for cap in capabilities:
    print(f"{cap.capability_id}: {cap.name}")
    print(f"  Assurance: {cap.required_assurance}")
    print(f"  Reputation: {cap.required_reputation}")
    print(f"  Guardian Approval: {cap.required_guardian_approval}")
```

### Registering Custom Capabilities

```python
# Register new capability
custom_cap = Capability(
    capability_id="custom_action",
    name="Custom Action",
    description="Perform custom domain-specific action",
    required_assurance="E2",
    required_reputation=0.6,
    required_sybil_resistance=0.5,
    required_guardian_approval=False,
    guardian_threshold=0.0,
    rate_limit=50,
    time_period_seconds=3600
)

gov_coord.capability_enforcer.register_capability(custom_cap)
```

---

## Security Considerations

### Capability Design Principles

1. **Least Privilege**: Start with restrictive requirements, relax if needed
2. **Progressive Trust**: Higher impact actions require higher verification
3. **Defense in Depth**: Combine multiple mechanisms (assurance + reputation + guardians)
4. **Rate Limiting**: Always rate-limit sensitive actions
5. **Auditability**: Log all capability checks for security analysis

### Common Attack Vectors

1. **Sybil Attack on Capabilities**:
   - **Mitigation**: Sybil resistance scores reduce new identity power
   - **Effect**: Mass identity creation is ineffective

2. **Reputation Farming**:
   - **Mitigation**: Reputation accumulation is slow and requires genuine contributions
   - **Effect**: Cannot quickly gain high-trust capabilities

3. **Guardian Collusion**:
   - **Mitigation**: High guardian thresholds (70-90%) require broad consensus
   - **Effect**: Small guardian cartels cannot authorize actions alone

4. **Rate Limit Circumvention**:
   - **Mitigation**: Rate limits tied to participant identity, not IP
   - **Effect**: Cannot create new accounts to bypass limits (due to Sybil resistance)

---

## Further Reading

- **[Governance System Architecture](./GOVERNANCE_SYSTEM_ARCHITECTURE.md)** - Complete system overview
- **[Proposal Creation Guide](./PROPOSAL_CREATION_GUIDE.md)** - How to create proposals
- **[Voting Mechanics](./VOTING_MECHANICS.md)** - Deep dive into voting
- **[Guardian Authorization Guide](./GUARDIAN_AUTHORIZATION_GUIDE.md)** - Guardian workflows

---

**Document Status**: Phase 6 User Documentation Complete
**Version**: 1.0
**Last Updated**: November 11, 2025
