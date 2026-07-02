# Zero-TrustML Governance Documentation

**Version**: 1.0
**Status**: Production Ready
**Last Updated**: November 11, 2025

---

## Overview

This directory contains comprehensive documentation for the Zero-TrustML governance system. The governance system provides identity-gated, Byzantine-resistant decision-making for federated learning networks.

**Key Features**:
- **Sybil-Resistant Voting**: Reputation-weighted quadratic voting
- **Capability-Based Access Control**: Fine-grained permissions
- **Guardian Authorization**: Multi-party approval for critical actions
- **Transparent Audit Trail**: All governance actions on DHT
- **45% Byzantine Tolerance**: Extends FL Byzantine resistance to governance

---

## Documentation Structure

### 📖 Getting Started

**Start here** if you're new to Zero-TrustML governance:

1. **[Governance System Architecture](./GOVERNANCE_SYSTEM_ARCHITECTURE.md)** - Complete system overview
   - High-level architecture
   - Component descriptions
   - Integration with Identity DHT
   - FL integration points
   - Deployment considerations

---

### 🔐 Core Concepts

**Understand the governance mechanisms**:

2. **[Capability Registry](./CAPABILITY_REGISTRY.md)** - Fine-grained access control
   - Capability structure and requirements
   - Built-in capabilities for governance and FL
   - Identity assurance levels (E0-E4)
   - Reputation and Sybil resistance
   - Guardian approval requirements
   - Custom capability creation

3. **[Voting Mechanics](./VOTING_MECHANICS.md)** - How voting works
   - Vote weight calculation (reputation + identity + Sybil resistance)
   - Quadratic voting formula
   - Credit budgets and allocation
   - Vote tallying and approval criteria
   - Security properties (Sybil resistance, vote buying prevention)
   - Advanced topics (delegation, conviction voting)

4. **[Guardian Authorization Guide](./GUARDIAN_AUTHORIZATION_GUIDE.md)** - Multi-party approvals
   - Guardian networks and responsibilities
   - Authorization request flow
   - Weighted approval thresholds
   - Best practices for guardians and subjects
   - Security considerations

---

### 🛠️ Practical Guides

**Step-by-step instructions for common tasks**:

5. **[Proposal Creation Guide](./PROPOSAL_CREATION_GUIDE.md)** - Creating governance proposals
   - Prerequisites and eligibility
   - Proposal types (parameter change, ban, capability update, etc.)
   - Complete proposal lifecycle
   - Detailed examples for each proposal type
   - Best practices and troubleshooting

---

## Quick Start

### For Participants

**I want to vote on proposals:**
1. Read [Voting Mechanics](./VOTING_MECHANICS.md) to understand how voting works
2. Check your vote weight: `await gov_coord.gov_extensions.calculate_vote_weight("your_id")`
3. Cast votes: `await gov_coord.voting_engine.cast_vote(...)`

**I want to create a proposal:**
1. Read [Proposal Creation Guide](./PROPOSAL_CREATION_GUIDE.md)
2. Check eligibility: Need E2+ assurance, 0.6+ reputation
3. Create proposal: `await gov_coord.create_governance_proposal(...)`

**I want to become a guardian:**
1. Meet requirements: E3+ assurance, 0.8+ reputation, 0.7+ Sybil resistance
2. Establish guardian relationships with subjects
3. Read [Guardian Authorization Guide](./GUARDIAN_AUTHORIZATION_GUIDE.md)

---

### For Developers

**I'm integrating governance into my FL application:**
1. Read [Governance System Architecture](./GOVERNANCE_SYSTEM_ARCHITECTURE.md)
2. Review [Capability Registry](./CAPABILITY_REGISTRY.md) for capability definitions
3. Use `GovernedFLCoordinator` as drop-in replacement for `Phase10Coordinator`
4. Configure via `FLGovernanceConfig`

**I'm building custom capabilities:**
1. Read [Capability Registry](./CAPABILITY_REGISTRY.md) - Custom Capabilities section
2. Define capability with requirements
3. Register: `gov_coord.capability_enforcer.register_capability(custom_cap)`
4. Enforce: `authorized, reason = await gov_coord.capability_enforcer.check_capability(...)`

**I'm deploying governance infrastructure:**
1. Read [Governance System Architecture](./GOVERNANCE_SYSTEM_ARCHITECTURE.md) - Deployment section
2. Deploy Holochain conductor with governance_record zome
3. Initialize GovernanceCoordinator with DHT client
4. Configure governance parameters

---

## Documentation Map

```
docs/07-governance/
├── README.md                           # This file - navigation hub
├── GOVERNANCE_SYSTEM_ARCHITECTURE.md   # Complete system overview (1,600+ lines)
├── CAPABILITY_REGISTRY.md              # Capability definitions and reference (1,300+ lines)
├── PROPOSAL_CREATION_GUIDE.md          # Step-by-step proposal guide (1,200+ lines)
├── VOTING_MECHANICS.md                 # Voting system deep dive (1,400+ lines)
└── GUARDIAN_AUTHORIZATION_GUIDE.md     # Guardian approval workflows (1,200+ lines)

Total: 6 files, ~7,700 lines of documentation
```

---

## Key Concepts at a Glance

### Vote Weight Formula

```
vote_weight = 1.0 × (1 + sybil_resistance) × assurance_factor × reputation_factor

Where:
  sybil_resistance = 0.0-1.0 (identity uniqueness)
  assurance_factor = {E0: 1.0, E1: 1.2, E2: 1.5, E3: 2.0, E4: 3.0}
  reputation_factor = 0.5 + (reputation × 0.5)
```

**Result**: New identities have ~0.5x vote power, verified E4 identities with high reputation have ~7x power.

### Quadratic Voting

```
effective_votes = sqrt(credits_spent) × vote_weight
```

**Result**: Doubling vote power requires 4x credits, preventing plutocracy.

### Guardian Approval Thresholds

| Action | Threshold | Interpretation |
|--------|-----------|----------------|
| Parameter changes | 60% | Moderate consensus |
| Emergency stops | 70% | Strong consensus |
| Participant bans | 80% | Very strong consensus |
| Treasury operations | 90% | Near unanimous |

### Proposal Lifecycle

```
DRAFT → REVIEW (1h) → VOTING (7d) → APPROVED/REJECTED → EXECUTED/FAILED
```

### Assurance Levels (Epistemic Charter E-Axis)

| Level | Name | Governance Weight | Example |
|-------|------|-------------------|---------|
| E0 | Null | 1.0x | Self-created |
| E1 | Testimonial | 1.2x | Email verified |
| E2 | Privately Verifiable | 1.5x | KYC with privacy |
| E3 | Cryptographically Proven | 2.0x | ZKP credential |
| E4 | Publicly Reproducible | 3.0x | Blockchain identity |

---

## Common Workflows

### Workflow 1: Create and Vote on Parameter Change

```python
# 1. Create proposal (requires E2+, rep 0.6+)
success, message, proposal_id = await gov_coord.create_governance_proposal(
    proposer_participant_id="alice_id",
    proposal_type="PARAMETER_CHANGE",
    title="Increase learning rate",
    description="Detailed explanation...",
    execution_params={"parameter": "learning_rate", "new_value": 0.02},
    voting_duration_days=7
)

# 2. Wait for review period (1 hour)
await asyncio.sleep(3600)

# 3. Cast vote (requires E1+, rep 0.4+)
success, message = await gov_coord.voting_engine.cast_vote(
    voter_participant_id="bob_id",
    proposal_id=proposal_id,
    choice=VoteChoice.FOR,
    credits_spent=100
)

# 4. Wait for voting period (7 days)
await asyncio.sleep(7 * 86400)

# 5. Finalize proposal (automatic or manual)
success, message, result = await gov_coord.finalize_proposal(proposal_id)
```

### Workflow 2: Request Emergency Stop with Guardian Approval

```python
# 1. Request emergency stop (requires E3+, rep 0.8+, guardian approval)
success, message, request_id = await fl_gov.request_emergency_stop(
    requester_participant_id="alice_id",
    reason="Byzantine attack detected",
    evidence={"byzantine_ratio": 0.48}
)

# 2. Guardians review and approve
await gov_coord.guardian_auth_mgr.submit_approval(
    guardian_participant_id="guardian_bob",
    request_id=request_id,
    approve=True,
    approval_reason="Confirmed attack independently"
)

# 3. Once 70% threshold met, emergency stop executes automatically
# FL training is halted
```

### Workflow 3: Ban Malicious Participant

```python
# 1. Request ban (requires E3+, rep 0.8+, guardian approval)
success, message, proposal_id = await fl_gov.request_participant_ban(
    requester_participant_id="alice_id",
    target_participant_id="eve_id",
    reason="Consistent Byzantine attacks",
    evidence={"rounds": [45, 46, 47, 48, 49, 50]},
    permanent=False,
    ban_duration_seconds=86400 * 30  # 30 days
)

# 2. Guardian approval (80% threshold)
# ... guardians approve ...

# 3. Community votes on proposal
# ... participants vote ...

# 4. If proposal approved, ban executes
# eve_id is banned for 30 days
```

---

## Security Properties

### Sybil Attack Resistance

- New identities have low vote weight (~0.5x)
- 1000 Sybil identities = ~31 verified participants
- Reputation accumulation is slow (months)

### Byzantine Attack Resistance

- Reputation-weighted voting reduces Byzantine power
- System safe at 45% Byzantine participants
- Byzantine power = Σ(malicious_reputation²)

### Vote Buying Prevention

- Quadratic voting makes large concentrations expensive
- 4x cost to double vote power
- Vote budgets limited by vote weight

### Guardian Collusion Prevention

- High thresholds (70-90%) require broad consensus
- Guardian relationships transparent on DHT
- Cartel detection via graph analysis

---

## API Quick Reference

```python
# Initialize Governance Coordinator
from zerotrustml.governance import GovernanceCoordinator

gov_coord = GovernanceCoordinator(
    dht_client=dht_client,
    identity_coordinator=identity_coordinator
)

# Create proposal
success, message, proposal_id = await gov_coord.create_governance_proposal(...)

# Cast vote
success, message = await gov_coord.voting_engine.cast_vote(...)

# Check vote weight
vote_weight = await gov_coord.gov_extensions.calculate_vote_weight(participant_id)

# Request guardian authorization
success, message, request_id = await gov_coord.guardian_auth_mgr.request_authorization(...)

# Submit guardian approval
success, message = await gov_coord.guardian_auth_mgr.submit_approval(...)

# Finalize proposal
success, message, result = await gov_coord.finalize_proposal(proposal_id)
```

---

## Integration Examples

### FL Coordinator Integration

```python
from zerotrustml.governance import GovernanceCoordinator
from zerotrustml.governance.governed_fl_coordinator import GovernedFLCoordinator

# Create governance coordinator
gov_coord = GovernanceCoordinator(
    dht_client=dht_client,
    identity_coordinator=identity_coordinator
)

# Create governed FL coordinator (drop-in replacement)
fl_coord = GovernedFLCoordinator(
    governance_coordinator=gov_coord,
    model=model,
    ...
)

# Use normally - governance checks are automatic
await fl_coord.run_training_rounds(num_rounds=10)
```

### Custom Capability Registration

```python
from zerotrustml.governance.models import Capability

# Define custom capability
custom_cap = Capability(
    capability_id="custom_action",
    name="Custom Action",
    description="Domain-specific action",
    required_assurance="E2",
    required_reputation=0.6,
    required_sybil_resistance=0.5,
    required_guardian_approval=False,
    guardian_threshold=0.0,
    rate_limit=50,
    time_period_seconds=3600
)

# Register capability
gov_coord.capability_enforcer.register_capability(custom_cap)

# Enforce capability
authorized, reason = await gov_coord.capability_enforcer.check_capability(
    participant_id="user_id",
    capability_id="custom_action"
)
```

---

## Troubleshooting

### "Not authorized to submit proposals"
**Solution**: Check identity assurance (need E2+), reputation (need 0.6+), and Sybil resistance (need 0.5+).

### "Rate limit exceeded"
**Solution**: Wait for rate limit window to reset (24 hours for most actions).

### "Guardian approval timeout"
**Solution**: Ensure guardians are responsive. Consider adding backup guardians in different time zones.

### "Proposal did not reach quorum"
**Solution**: Promote proposal more actively. Consider lowering quorum for similar proposals.

### "Vote failed: insufficient budget"
**Solution**: Check vote budget: `budget = 100 × vote_weight`. Split credits across fewer proposals.

---

## Performance Benchmarks

| Operation | Target | Typical (Mocked) | Notes |
|-----------|--------|------------------|-------|
| Vote weight calculation | <5ms | ~2ms | Fast identity lookup |
| Capability enforcement | <10ms | ~5ms | Cached identity data |
| Vote casting | <100ms | ~20ms | DHT write + signature |
| Vote tallying (1000 votes) | <200ms | ~50ms | In-memory aggregation |
| Guardian authorization request | <500ms | ~100ms | DHT write + notifications |
| Proposal submission | <1000ms | ~200ms | DHT write + validation |

---

## Further Resources

### Technical Implementation

- **[Week 7-8 Design Document](../06-architecture/WEEK_7_8_DESIGN.md)** - Original design specification
- **[Phase 1-5 Completion Docs](../06-architecture/)** - Implementation details
- **[Test Suite](../../tests/governance/)** - 55+ comprehensive tests

### Mycelix Protocol Integration

- **[Epistemic Charter v2.0](../../../docs/architecture/THE%20EPISTEMIC%20CHARTER%20(v2.0).md)** - Identity assurance levels
- **[Governance Charter v1.0](../../../docs/architecture/THE%20GOVERNANCE%20CHARTER%20(v1.0).md)** - Governance principles
- **[Identity DHT System](../05-identity/)** - Identity verification infrastructure

### Academic Foundation

- **[Research Foundation](../RESEARCH_FOUNDATION.md)** - Academic backing
- **[Byzantine Tolerance Paper](../whitepaper/)** - 45% BFT proof
- **[MATL Architecture](../06-architecture/matl_architecture.md)** - Trust layer design

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Nov 11, 2025 | Initial documentation release (Phase 6) |

---

## Contributing to Documentation

Improvements welcome! To contribute:

1. Read the document you want to improve
2. Identify gaps, errors, or unclear sections
3. Submit proposed changes via pull request
4. Ensure consistency with other documents
5. Update this README if adding new files

**Documentation Standards**:
- Clear, concise language
- Complete code examples
- Real-world use cases
- Security considerations
- Performance implications

---

## License

This documentation is licensed under the same terms as Zero-TrustML (Apache 2.0 for SDK components).

---

**Documentation Status**: Phase 6 Complete ✅
**Total Documentation**: 7,700+ lines across 6 files
**Coverage**: 100% of Week 7-8 governance system
**Quality**: Production-ready, comprehensive user documentation

---

*"Governance is not about control. It's about enabling collective intelligence to emerge."*

🍄 **Mycelix Protocol** - Cultivating collective wisdom 🍄
