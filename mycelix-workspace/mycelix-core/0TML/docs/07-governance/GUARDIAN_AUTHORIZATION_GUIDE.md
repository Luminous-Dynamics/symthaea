# Guardian Authorization Guide

**Version**: 1.0
**Status**: Production Ready
**Last Updated**: November 11, 2025

---

## Overview

Guardian authorization provides a **multi-party approval mechanism** for critical actions in Zero-TrustML. This system ensures that sensitive operations (emergency stops, participant bans, parameter changes) require consensus from a trusted network of guardians before execution.

**Key Features**:
- **Weighted approval**: Guardian votes weighted by reputation and identity verification
- **Configurable thresholds**: 60-90% approval required depending on action severity
- **Time-bounded**: Authorization requests expire after 1 hour (configurable)
- **Transparent audit trail**: All requests and approvals stored on DHT
- **Decentralized**: No single guardian can authorize alone

---

## Guardian Networks

### What is a Guardian?

A **guardian** is a trusted participant who can authorize critical actions on behalf of another participant (the "subject"). Guardian relationships are established through the Identity DHT system (Week 5-6).

**Guardian Responsibilities**:
1. **Review authorization requests** promptly
2. **Verify evidence** before approving
3. **Act in subject's best interest** (or network's, for emergency actions)
4. **Maintain high reputation** to preserve guardian weight
5. **Reject malicious requests**

### Guardian Requirements

To become a guardian:
- **Identity Assurance**: E3+ (Cryptographically Proven)
- **Reputation**: 0.8+ (High trust)
- **Sybil Resistance**: 0.7+ (Verified uniqueness)
- **Uptime**: >95% availability over last 30 days
- **No Bans**: Never been banned or flagged for Byzantine behavior

### Guardian Weight Calculation

```python
guardian_weight = assurance_factor × reputation

Where:
  assurance_factor = {E0: 1.0, E1: 1.2, E2: 1.5, E3: 2.0, E4: 3.0}
  reputation = 0.0-1.0

Examples:
  Guardian with E4, rep=0.9: weight = 3.0 × 0.9 = 2.70
  Guardian with E3, rep=0.8: weight = 2.0 × 0.8 = 1.60
  Guardian with E2, rep=0.7: weight = 1.5 × 0.7 = 1.05
```

### Establishing Guardian Relationships

Guardian relationships are created through the Identity Coordinator:

```python
from zerotrustml.identity import IdentityCoordinator

# Subject requests guardian
success, message = await identity_coordinator.request_guardian(
    subject_participant_id="alice_id",
    guardian_participant_id="bob_id",
    relationship_type="recovery",  # or "authorization"
    evidence_of_relationship="We've worked together for 2 years"
)

# Guardian accepts request
success, message = await identity_coordinator.accept_guardian_request(
    guardian_participant_id="bob_id",
    subject_participant_id="alice_id",
    acceptance_evidence="Confirmed via video call"
)
```

---

## Authorization Request Flow

### Step 1: Create Authorization Request

When a critical action requires guardian approval:

```python
from zerotrustml.governance import GovernanceCoordinator

# Request emergency stop (requires guardian approval)
success, message, request_id = await gov_coord.guardian_auth_mgr.request_authorization(
    participant_id="alice_id",
    capability_id="emergency_stop",
    action_params={
        "reason": "Byzantine attack detected in round 42",
        "evidence": {
            "byzantine_ratio": 0.48,
            "affected_rounds": [40, 41, 42],
            "detection_method": "PoGQ threshold violation"
        }
    },
    timeout_seconds=3600  # 1 hour expiry
)

if success:
    print(f"Authorization request created: {request_id}")
    print("Guardians have been notified.")
    print("Awaiting guardian approval...")
else:
    print(f"Failed to create request: {message}")
```

**What Happens**:
1. System verifies participant meets minimum requirements (even for guardian approval)
2. Authorization request is created with status=PENDING
3. Request is stored on DHT for transparency
4. All guardians of the participant are notified
5. Request ID is returned for tracking

### Step 2: Guardians Review Request

Guardians receive notification with request details:

```python
# Guardian retrieves pending requests
requests = await gov_coord.guardian_auth_mgr.get_pending_requests(
    guardian_participant_id="bob_id"
)

for request in requests:
    print(f"Request: {request.request_id}")
    print(f"Subject: {request.subject_participant_id}")
    print(f"Action: {request.action}")
    print(f"Reason: {request.action_params['reason']}")
    print(f"Evidence: {request.action_params['evidence']}")
    print(f"Expires: {request.expires_at}")
    print("---")
```

**Guardian Review Checklist**:
1. **Verify subject identity**: Is this really my subject?
2. **Review evidence**: Is the evidence compelling?
3. **Assess necessity**: Is this action truly needed?
4. **Consider impact**: What are the consequences?
5. **Check alternatives**: Are there less drastic options?

### Step 3: Guardians Submit Approvals/Rejections

Each guardian submits their decision:

```python
# Guardian approves request
success, message = await gov_coord.guardian_auth_mgr.submit_approval(
    guardian_participant_id="bob_id",
    request_id=request_id,
    approve=True,
    approval_reason="Evidence is compelling. Independent analysis confirms Byzantine attack.",
    evidence={
        "independent_analysis": "Ran PoGQ on rounds 40-42, confirmed scores < 0.25",
        "cross_validation": "3 other guardians confirmed attack via different methods"
    }
)

if success:
    print("Approval submitted successfully")
else:
    print(f"Failed to submit approval: {message}")
```

Or reject:

```python
# Guardian rejects request
success, message = await gov_coord.guardian_auth_mgr.submit_approval(
    guardian_participant_id="bob_id",
    request_id=request_id,
    approve=False,
    approval_reason="Insufficient evidence. PoGQ scores within normal variance.",
    evidence={
        "counter_analysis": "Scores are low but not below emergency threshold",
        "recommendation": "Monitor for 2 more rounds before emergency action"
    }
)
```

### Step 4: Weighted Threshold Calculation

After each guardian response, the system calculates weighted approval:

```python
# Example: 5 guardians
guardians = [
    {"id": "guardian1", "weight": 2.70, "approved": True},   # E4, rep=0.9
    {"id": "guardian2", "weight": 1.60, "approved": True},   # E3, rep=0.8
    {"id": "guardian3", "weight": 1.40, "approved": False},  # E3, rep=0.7
    {"id": "guardian4", "weight": 0.90, "approved": None},   # E2, rep=0.6 (no response)
    {"id": "guardian5", "weight": 0.75, "approved": None},   # E2, rep=0.5 (no response)
]

total_weight = sum(g["weight"] for g in guardians) = 7.35
approval_weight = sum(g["weight"] for g in guardians if g["approved"]) = 2.70 + 1.60 = 4.30
rejection_weight = sum(g["weight"] for g in guardians if g["approved"] == False) = 1.40

required_threshold = 0.7  # 70% for emergency_stop
required_weight = total_weight × required_threshold = 7.35 × 0.7 = 5.145

approval_ratio = approval_weight / total_weight = 4.30 / 7.35 = 58.5%
authorized = (approval_ratio >= required_threshold) = False  # Not yet authorized
```

### Step 5: Execution (if Threshold Met)

Once the required threshold is met, the action is automatically executed:

```python
# System automatically executes when threshold met
# For emergency stop example:
await fl_coordinator.emergency_stop(
    reason="Guardian approval received",
    authorized_by=[list of approving guardians],
    request_id=request_id
)

# Execution result is recorded on DHT
await gov_coord.guardian_auth_mgr.record_execution(
    request_id=request_id,
    executed_at=int(time.time() * 1_000_000),
    execution_result={
        "success": True,
        "message": "FL training halted successfully",
        "affected_participants": 127
    }
)
```

### Step 6: Timeout Handling

If the request expires before threshold is met:

```python
# After 1 hour (or configured timeout)
if current_time > request.expires_at:
    # Mark request as EXPIRED
    await gov_coord.guardian_auth_mgr.expire_request(request_id)

    # Notify subject and guardians
    await gov_coord.guardian_auth_mgr.notify_expiration(
        request_id=request_id,
        reason="Insufficient guardian responses within timeout period",
        approval_weight=approval_weight,
        required_weight=required_weight
    )
```

---

## Approval Thresholds

Different capabilities require different guardian consensus levels:

| Capability | Threshold | Interpretation | Typical Guardians Needed |
|------------|-----------|----------------|--------------------------|
| `update_parameters` | 0.6 | 60% | 3 out of 5 |
| `emergency_stop` | 0.7 | 70% | 4 out of 5 |
| `ban_participant` | 0.8 | 80% | 4 out of 5 |
| `treasury_withdrawal` | 0.9 | 90% | 5 out of 5 (near unanimous) |

**Note**: Actual number of guardians needed depends on guardian weights. Higher-reputation guardians need fewer approvals to reach threshold.

---

## Guardian Authorization Examples

### Example 1: Emergency Stop During Attack

```python
# Alice detects Byzantine attack
# Alice's guardians: Bob (E4, 0.9), Carol (E3, 0.8), Dave (E3, 0.7), Eve (E2, 0.6), Frank (E2, 0.5)
# Total weight: 2.70 + 1.60 + 1.40 + 0.90 + 0.75 = 7.35
# Required (70%): 5.145

# Alice requests emergency stop
success, message, request_id = await fl_gov.request_emergency_stop(
    requester_participant_id="alice_id",
    reason="Byzantine attack detected",
    evidence={"byzantine_ratio": 0.48}
)

# Bob reviews and approves (weight=2.70, total=2.70/7.35=36.7%)
await gov_coord.guardian_auth_mgr.submit_approval(
    guardian_participant_id="bob_id",
    request_id=request_id,
    approve=True,
    approval_reason="Confirmed attack independently"
)
# Status: PENDING (need more approvals)

# Carol reviews and approves (weight=1.60, total=4.30/7.35=58.5%)
await gov_coord.guardian_auth_mgr.submit_approval(
    guardian_participant_id="carol_id",
    request_id=request_id,
    approve=True,
    approval_reason="PoGQ scores confirm Byzantine behavior"
)
# Status: PENDING (need more approvals)

# Dave reviews and approves (weight=1.40, total=5.70/7.35=77.6%)
await gov_coord.guardian_auth_mgr.submit_approval(
    guardian_participant_id="dave_id",
    request_id=request_id,
    approve=True,
    approval_reason="Attack pattern matches known Byzantine signatures"
)
# Status: APPROVED ✅ (threshold exceeded: 77.6% > 70%)

# System automatically executes emergency stop
# FL training halted
# All participants notified
```

### Example 2: Participant Ban Request

```python
# Bob requests ban for malicious participant
# Bob's guardians: Alice (E4, 0.9), Carol (E3, 0.8), Dave (E2, 0.7)
# Total weight: 2.70 + 1.60 + 1.05 = 5.35
# Required (80%): 4.28

# Bob requests ban
success, message, proposal_id = await fl_gov.request_participant_ban(
    requester_participant_id="bob_id",
    target_participant_id="eve_id",
    reason="Consistent Byzantine attacks",
    evidence={"rounds": [45, 46, 47, 48, 49, 50]},
    permanent=False,
    ban_duration_seconds=86400 * 30  # 30 days
)

# System creates both proposal and guardian authorization request

# Alice reviews and approves (weight=2.70, total=2.70/5.35=50.5%)
await gov_coord.guardian_auth_mgr.submit_approval(
    guardian_participant_id="alice_id",
    request_id=request_id,
    approve=True,
    approval_reason="Evidence is strong, ban is appropriate"
)

# Carol reviews and approves (weight=1.60, total=4.30/5.35=80.4%)
await gov_coord.guardian_auth_mgr.submit_approval(
    guardian_participant_id="carol_id",
    request_id=request_id,
    approve=True,
    approval_reason="Confirmed malicious pattern across multiple rounds"
)
# Status: APPROVED ✅ (threshold exceeded: 80.4% > 80%)

# Guardian authorization approved
# Proposal still needs community vote
# After proposal approved, ban is executed
```

### Example 3: Authorization Rejected

```python
# Carol requests parameter change
# Carol's guardians: Alice (E4, 0.9), Bob (E3, 0.8), Dave (E2, 0.7), Eve (E2, 0.6)
# Total weight: 2.70 + 1.60 + 1.05 + 0.90 = 6.25
# Required (60%): 3.75

# Carol requests parameter change
success, message, request_id = await gov_coord.guardian_auth_mgr.request_authorization(
    participant_id="carol_id",
    capability_id="update_parameters",
    action_params={
        "parameter": "min_reputation",
        "new_value": 0.9  # Very high threshold
    }
)

# Alice reviews and rejects (weight=2.70, rejection=2.70)
await gov_coord.guardian_auth_mgr.submit_approval(
    guardian_participant_id="alice_id",
    request_id=request_id,
    approve=False,
    approval_reason="Threshold too high, would exclude 80% of participants"
)

# Bob reviews and rejects (weight=1.60, rejection=4.30)
await gov_coord.guardian_auth_mgr.submit_approval(
    guardian_participant_id="bob_id",
    request_id=request_id,
    approve=False,
    approval_reason="Alternative approaches available, no urgency"
)

# Dave reviews and approves (weight=1.05, approval=1.05)
await gov_coord.guardian_auth_mgr.submit_approval(
    guardian_participant_id="dave_id",
    request_id=request_id,
    approve=True,
    approval_reason="I support higher standards"
)

# Eve doesn't respond (timeout after 1 hour)

# Final tally:
# Approval weight: 1.05 / 6.25 = 16.8%
# Rejection weight: 4.30 / 6.25 = 68.8%
# Status: REJECTED ❌ (approval < 60% threshold)

# Carol is notified of rejection
# Can revise proposal and resubmit with lower threshold
```

---

## Best Practices for Guardians

### 1. Respond Promptly

Authorization requests expire after 1 hour. Guardians should:
- Enable notifications for authorization requests
- Check pending requests at least once per hour
- Set up backup guardians if unavailable

### 2. Verify Evidence Independently

Never blindly approve requests. Always:
- Review provided evidence carefully
- Cross-check with independent sources
- Run your own analysis if possible
- Ask subject for clarification if needed

### 3. Document Your Decision

Provide clear reasoning for approvals/rejections:
```python
# Good approval reason
approval_reason="""
Verified evidence via independent analysis:
1. Ran PoGQ on affected rounds: confirmed scores < 0.25
2. Checked with 2 other guardians: consensus on attack
3. Reviewed network logs: detected coordinated Byzantine behavior
4. Assessment: Emergency stop is necessary and proportionate
"""

# Bad approval reason
approval_reason="Looks good"  # Too vague!
```

### 4. Consider Alternatives

Before approving drastic actions:
- Is there a less severe alternative?
- Can the issue be resolved without guardian authorization?
- What are the consequences of approval vs. rejection?
- Is this the right time for this action?

### 5. Maintain Impartiality

As a guardian:
- Act in subject's (or network's) best interest
- Avoid conflicts of interest
- Recuse yourself if conflicted
- Don't approve requests for personal gain

### 6. Coordinate with Other Guardians

For complex decisions:
- Discuss with other guardians before voting
- Share analysis and perspectives
- Reach consensus when possible
- Document disagreements for transparency

---

## Best Practices for Subjects

### 1. Build a Strong Guardian Network

Choose guardians who:
- Have high reputation and assurance levels
- Are available and responsive
- Have relevant expertise for your activities
- Are trustworthy and impartial
- Are geographically/organizationally diverse

**Recommended Guardian Network Size**: 5-7 guardians

### 2. Provide Complete Evidence

When requesting authorization:
- Include all relevant data and logs
- Provide clear reasoning and justification
- Anticipate questions and address them upfront
- Make evidence independently verifiable

```python
# Good authorization request
evidence={
    "byzantine_ratio": 0.48,
    "affected_rounds": [40, 41, 42],
    "detection_method": "PoGQ threshold violation",
    "pogq_scores": [0.22, 0.18, 0.15],
    "threshold": 0.70,
    "affected_model_accuracy": -0.032,
    "logs": "https://logs.example.com/rounds/40-42",
    "independent_verification": "3 other participants confirmed attack"
}

# Bad authorization request
evidence={"issue": "something wrong"}  # Too vague!
```

### 3. Communicate with Guardians

- Notify guardians of urgent requests via external channels (email, chat)
- Explain context and urgency
- Answer questions promptly
- Accept feedback gracefully

### 4. Respect Guardian Decisions

If authorization is rejected:
- Understand the reasoning
- Address concerns
- Revise and resubmit if appropriate
- Don't pressure guardians to approve

### 5. Plan for Emergencies

- Establish emergency communication channels with guardians
- Test authorization workflows periodically
- Have backup guardians in different time zones
- Document emergency procedures

---

## Monitoring Authorization Requests

### For Subjects

```python
# Check status of your authorization request
status = await gov_coord.guardian_auth_mgr.get_authorization_status(request_id)

print(f"Status: {status['status']}")
print(f"Approval weight: {status['approval_weight']:.2f} / {status['total_weight']:.2f}")
print(f"Approval ratio: {status['approval_ratio']:.1%}")
print(f"Required threshold: {status['required_threshold']:.1%}")
print(f"Time remaining: {status['time_remaining_seconds']}s")
print(f"Approvals: {len(status['approving_guardians'])}")
print(f"Rejections: {len(status['rejecting_guardians'])}")
print(f"No response: {len(status['pending_guardians'])}")
```

### For Guardians

```python
# List pending requests you need to review
requests = await gov_coord.guardian_auth_mgr.get_pending_requests(
    guardian_participant_id="your_guardian_id"
)

print(f"{len(requests)} pending authorization requests:")
for request in requests:
    print(f"  {request.request_id}: {request.action} for {request.subject_participant_id}")
    print(f"    Expires in: {request.expires_at - current_time}s")
```

### For Network Administrators

```python
# Get all active authorization requests
requests = await gov_coord.guardian_auth_mgr.list_active_requests()

# Get statistics
stats = await gov_coord.guardian_auth_mgr.get_authorization_stats()
print(f"Total requests (last 30 days): {stats['total_requests']}")
print(f"Approved: {stats['approved']} ({stats['approval_rate']:.1%})")
print(f"Rejected: {stats['rejected']} ({stats['rejection_rate']:.1%})")
print(f"Expired: {stats['expired']} ({stats['expiration_rate']:.1%})")
print(f"Average response time: {stats['avg_response_time_seconds']}s")
```

---

## Security Considerations

### Guardian Collusion Attack

**Attack**: Multiple guardians collude to approve malicious requests.

**Defense**:
1. High approval thresholds (70-90%) require broad consensus
2. Guardian relationships are transparent on DHT
3. Subjects can revoke compromised guardians
4. Guardian cartel detection via graph analysis

**Example**:
```
Attacker controls 3 out of 5 guardians:
  Malicious guardians: weight = 3.0 (60% of total)
  Honest guardians: weight = 2.0 (40% of total)

For 70% threshold capability:
  Malicious guardians alone: 60% < 70% ❌ Cannot approve

For 80% threshold capability:
  Malicious guardians alone: 60% < 80% ❌ Cannot approve

Guardian collusion requires near-total control to be effective.
```

### Guardian Reputation Attack

**Attack**: Attacker builds high reputation to become guardian, then approves malicious requests.

**Defense**:
1. Reputation accumulation is slow (takes months)
2. Guardian status requires sustained good behavior
3. Single malicious guardian has limited impact (weighted voting)
4. Suspicious guardian patterns trigger alerts

### Guardian Unavailability

**Problem**: Guardians offline during emergency, authorization times out.

**Mitigation**:
1. Shorter timeout for urgent actions (15-30 minutes)
2. Backup guardians in different time zones
3. Fallback to coordinator override after timeout
4. Emergency notification system (SMS, phone, email)

---

## API Reference

### Request Authorization

```python
success, message, request_id = await gov_coord.guardian_auth_mgr.request_authorization(
    participant_id: str,
    capability_id: str,
    action_params: Dict[str, Any],
    timeout_seconds: int = 3600
) -> Tuple[bool, str, Optional[str]]
```

### Submit Guardian Approval

```python
success, message = await gov_coord.guardian_auth_mgr.submit_approval(
    guardian_participant_id: str,
    request_id: str,
    approve: bool,
    approval_reason: str,
    evidence: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]
```

### Check Authorization Status

```python
status = await gov_coord.guardian_auth_mgr.get_authorization_status(
    request_id: str
) -> Dict[str, Any]

# Returns:
# {
#     "request_id": str,
#     "status": str,  # PENDING, APPROVED, REJECTED, EXPIRED
#     "approval_weight": float,
#     "rejection_weight": float,
#     "total_weight": float,
#     "approval_ratio": float,
#     "required_threshold": float,
#     "time_remaining_seconds": int,
#     "approving_guardians": List[str],
#     "rejecting_guardians": List[str],
#     "pending_guardians": List[str]
# }
```

### List Pending Requests

```python
requests = await gov_coord.guardian_auth_mgr.get_pending_requests(
    guardian_participant_id: str
) -> List[AuthorizationRequest]
```

---

## Further Reading

- **[Governance System Architecture](./GOVERNANCE_SYSTEM_ARCHITECTURE.md)** - Complete system overview
- **[Capability Registry](./CAPABILITY_REGISTRY.md)** - Which capabilities require guardian approval
- **[Proposal Creation Guide](./PROPOSAL_CREATION_GUIDE.md)** - Creating proposals
- **[Voting Mechanics](./VOTING_MECHANICS.md)** - Voting system details

---

**Document Status**: Phase 6 User Documentation Complete
**Version**: 1.0
**Last Updated**: November 11, 2025
