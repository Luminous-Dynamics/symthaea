# Governance System Architecture

**Version**: 1.0
**Status**: Production Ready
**Last Updated**: November 11, 2025

---

## Overview

The Zero-TrustML Governance System provides **identity-gated, Byzantine-resistant decision-making** for federated learning networks. It integrates with the Mycelix Protocol's Identity DHT system to enable:

- **Sybil-resistant voting** through reputation weighting
- **Capability-based access control** tied to identity verification
- **Guardian-authorized emergency actions** for critical operations
- **Transparent audit trails** via Holochain DHT storage

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Zero-TrustML FL Coordinator                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │              Governance Coordinator                         │    │
│  │  ┌──────────────────────────────────────────────────────┐  │    │
│  │  │ Proposal Manager                                      │  │    │
│  │  │ - Create proposals                                    │  │    │
│  │  │ - Validate proposals                                  │  │    │
│  │  │ - Manage proposal lifecycle                           │  │    │
│  │  └──────────────────────────────────────────────────────┘  │    │
│  │  ┌──────────────────────────────────────────────────────┐  │    │
│  │  │ Voting Engine                                         │  │    │
│  │  │ - Cast votes with quadratic voting                    │  │    │
│  │  │ - Tally votes with reputation weighting               │  │    │
│  │  │ - Finalize proposal outcomes                          │  │    │
│  │  └──────────────────────────────────────────────────────┘  │    │
│  │  ┌──────────────────────────────────────────────────────┐  │    │
│  │  │ Capability Enforcer                                   │  │    │
│  │  │ - Verify identity assurance levels                    │  │    │
│  │  │ - Check reputation thresholds                         │  │    │
│  │  │ - Enforce rate limits                                 │  │    │
│  │  └──────────────────────────────────────────────────────┘  │    │
│  │  ┌──────────────────────────────────────────────────────┐  │    │
│  │  │ Guardian Authorization Manager                        │  │    │
│  │  │ - Request guardian approvals                          │  │    │
│  │  │ - Collect guardian responses                          │  │    │
│  │  │ - Execute authorized actions                          │  │    │
│  │  └──────────────────────────────────────────────────────┘  │    │
│  └───────────────────┬────────────────────────────────────────┘    │
│                      │                                              │
│                      │ Identity Verification                        │
│                      │ Reputation Queries                           │
│                      │ Guardian Authorization                       │
│                      ▼                                              │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │           Identity Coordinator (Week 5-6)                   │    │
│  │  - verify_identity_for_governance()                         │    │
│  │  - calculate_vote_weight()                                  │    │
│  │  - authorize_capability()                                   │    │
│  │  - request_guardian_authorization()                         │    │
│  └───────────────────┬────────────────────────────────────────┘    │
│                      │                                              │
└──────────────────────┼──────────────────────────────────────────────┘
                       │
                       │ DHT Read/Write
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Holochain DHT                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ DID Registry │  │ Identity     │  │ Reputation   │             │
│  │ Zome         │  │ Store Zome   │  │ Sync Zome    │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│  ┌──────────────┐  ┌──────────────┐                                │
│  │ Guardian     │  │ Governance   │  ← NEW                         │
│  │ Graph Zome   │  │ Record Zome  │                                │
│  └──────────────┘  └──────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Governance Coordinator

**Location**: `src/zerotrustml/governance/coordinator.py`

The main orchestration layer for all governance operations.

**Responsibilities**:
- Initialize and manage sub-components (Proposal Manager, Voting Engine, etc.)
- Provide unified API for governance operations
- Coordinate between identity system and governance components
- Manage governance state and configuration

**Key Methods**:
```python
class GovernanceCoordinator:
    async def create_governance_proposal(...) -> Tuple[bool, str, Optional[str]]
    async def cast_vote(...) -> Tuple[bool, str]
    async def finalize_proposal(...) -> Tuple[bool, str, Dict[str, Any]]
    async def authorize_fl_action(...) -> Tuple[bool, str]
    async def request_guardian_authorization(...) -> Tuple[bool, str, Optional[str]]
```

**Dependencies**:
- `IdentityCoordinator` (from Week 5-6)
- `DHT Client` (Holochain connection)
- `GovernanceExtensions` (identity integration)

---

### 2. Proposal Manager

**Location**: `src/zerotrustml/governance/coordinator.py` (ProposalManager class)

Manages the complete proposal lifecycle from creation to execution.

**Responsibilities**:
- Validate proposal submissions
- Store proposals on DHT
- Track proposal status (DRAFT → REVIEW → VOTING → APPROVED/REJECTED → EXECUTED)
- Execute approved proposals
- Maintain proposal audit trail

**Proposal Lifecycle**:
```
1. DRAFT      → Proposal created, not yet submitted
2. REVIEW     → 1-hour review period for community feedback
3. VOTING     → Active voting period (7 days default)
4. APPROVED   → Quorum met, approval threshold exceeded
5. REJECTED   → Quorum met, approval threshold not met
6. EXECUTED   → Approved proposal executed successfully
7. FAILED     → Execution failed (with error details)
```

**Key Features**:
- **Proposal Types**: Parameter change, participant management, emergency actions
- **Validation**: Check proposer capability before submission
- **Review Period**: 1-hour mandatory review before voting starts
- **Quorum Requirements**: Configurable minimum participation (default 50%)
- **Approval Thresholds**: Configurable approval percentage (default 60-66%)

---

### 3. Voting Engine

**Location**: `src/zerotrustml/governance/coordinator.py` (VotingEngine class)

Implements reputation-weighted quadratic voting with Sybil resistance.

**Responsibilities**:
- Cast votes with credit spending
- Calculate effective votes (quadratic formula)
- Apply vote weights based on identity verification
- Tally votes for proposal outcomes
- Prevent double voting

**Voting Mechanics**:

**Vote Weight Formula**:
```python
vote_weight = base_weight × (1 + sybil_resistance) × assurance_factor × reputation_factor

Where:
  base_weight = 1.0
  sybil_resistance = 0.0-1.0 (from identity verification)
  assurance_factor = {E0: 1.0, E1: 1.2, E2: 1.5, E3: 2.0, E4: 3.0}
  reputation_factor = 0.5 + (reputation × 0.5)  # 0.5-1.0 range
```

**Quadratic Voting**:
```python
effective_votes = sqrt(credits_spent) × vote_weight

Example:
  Alice (vote_weight=5.0): 100 credits → sqrt(100)×5.0 = 50 effective votes
  Bob (vote_weight=2.0): 100 credits → sqrt(100)×2.0 = 20 effective votes

  Doubling vote power requires 4x credits (quadratic cost)
```

**Vote Budget**:
```python
vote_budget = BASE_BUDGET × vote_weight

Where:
  BASE_BUDGET = 100 credits (configurable)

Example:
  Alice (vote_weight=5.0): 500 credits total
  Bob (vote_weight=2.0): 200 credits total
```

---

### 4. Capability Enforcer

**Location**: `src/zerotrustml/governance/coordinator.py` (CapabilityEnforcer class)

Enforces fine-grained capability-based access control tied to identity verification.

**Responsibilities**:
- Check if participant has required identity assurance
- Verify reputation meets threshold
- Enforce rate limits on sensitive actions
- Determine if guardian approval is required

**Capability Definition**:
```python
@dataclass
class Capability:
    capability_id: str
    name: str
    description: str
    required_assurance: str          # E0-E4
    required_reputation: float       # 0.0-1.0
    required_sybil_resistance: float # 0.0-1.0
    required_guardian_approval: bool
    guardian_threshold: float        # 0.0-1.0
    rate_limit: Optional[int]        # Max invocations per period
    time_period_seconds: Optional[int]
```

**Built-in Capabilities**:

| Capability | Assurance | Reputation | Guardian | Description |
|------------|-----------|------------|----------|-------------|
| `submit_mip` | E2 | 0.6 | No | Submit Mycelix Improvement Proposal |
| `vote_on_mip` | E1 | 0.4 | No | Vote on existing proposals |
| `emergency_stop` | E3 | 0.8 | Yes (0.7) | Emergency stop FL training |
| `update_parameters` | E2 | 0.7 | Yes (0.6) | Modify FL hyperparameters |
| `ban_participant` | E3 | 0.8 | Yes (0.8) | Ban malicious participant |

**Verification Flow**:
```
1. Check identity assurance level (E0-E4)
2. Check reputation threshold
3. Check sybil resistance score
4. Check rate limit (if applicable)
5. Request guardian approval (if required)
6. Return (authorized: bool, reason: str)
```

---

### 5. Guardian Authorization Manager

**Location**: `src/zerotrustml/governance/coordinator.py` (GuardianAuthorizationManager class)

Manages guardian approval for critical actions requiring multi-party authorization.

**Responsibilities**:
- Create authorization requests
- Collect guardian approvals/rejections
- Calculate weighted approval threshold
- Execute actions when threshold met
- Handle timeouts and expirations

**Authorization Request Flow**:
```
1. Participant requests critical action
2. System creates authorization request
3. Guardians are notified
4. Guardians submit approvals/rejections
5. System calculates weighted approval
6. If threshold met: execute action
7. If timeout: expire request
```

**Guardian Approval Weighting**:
```python
approval_weight = Σ(guardian_reputation × guardian_identity_weight)
threshold = required_threshold × total_guardian_weight

authorized = (approval_weight >= threshold)

Example (70% threshold):
  Guardian Alice (E4, rep=0.9): weight = 3.0 × 0.95 = 2.85
  Guardian Bob (E3, rep=0.8): weight = 2.0 × 0.9 = 1.80
  Guardian Carol (E2, rep=0.7): weight = 1.5 × 0.85 = 1.275

  Total weight: 5.925
  Required threshold: 0.7 × 5.925 = 4.1475

  If Alice + Bob approve: 2.85 + 1.80 = 4.65 > 4.1475 ✅ APPROVED
```

---

## DHT Storage Layer

### Governance Record Zome

**Location**: `zerotrustml-identity-dna/zomes/governance_record/`

Rust zome providing immutable, distributed storage for governance records.

**Entry Types**:
1. **Proposal**: Proposal details and status
2. **Vote**: Individual vote records
3. **ExecutionRecord**: Execution results and audit trail
4. **AuthorizationRequest**: Guardian approval requests
5. **GuardianApproval**: Guardian approval/rejection records

**Zome Functions**:
- `store_proposal(proposal)` → ActionHash
- `get_proposal(proposal_id)` → Proposal
- `list_proposals(status, limit)` → Vec<Proposal>
- `store_vote(vote)` → ActionHash
- `get_votes(proposal_id)` → Vec<Vote>
- `store_execution_record(record)` → ActionHash
- `get_execution_record(proposal_id)` → ExecutionRecord
- `store_authorization_request(request)` → ActionHash
- `get_authorization_request(request_id)` → AuthorizationRequest
- `store_guardian_approval(approval)` → ActionHash
- `get_guardian_approvals(request_id)` → Vec<GuardianApproval>

**Path-Based Resolution**:
For efficient lookups, proposals are indexed using Holochain paths:
```rust
// O(1) lookup by proposal ID
proposal.{proposal_id}

// Query by status
proposals.by_status.{status}

// Query by type
proposals.by_type.{proposal_type}

// Query by proposer
proposals.by_proposer.{proposer_did}
```

---

## Integration with Identity System

### Identity DHT (Week 5-6)

The governance system depends on the Identity DHT system for:

1. **Identity Verification**: `verify_identity_for_governance(participant_id, required_assurance)`
2. **Reputation Queries**: Extract reputation scores from identity records
3. **Sybil Resistance**: Use identity signals to calculate Sybil resistance scores
4. **Guardian Networks**: Query guardian relationships for authorization

### Governance Extensions

**Location**: `src/zerotrustml/identity/governance_extensions.py`

Provides governance-specific methods on the Identity Coordinator:

```python
class IdentityGovernanceExtensions:
    async def verify_identity_for_governance(
        participant_id: str,
        required_assurance: str
    ) -> Dict[str, Any]

    async def calculate_vote_weight(
        participant_id: str
    ) -> float

    async def authorize_capability(
        participant_id: str,
        capability_id: str
    ) -> Tuple[bool, str]

    async def request_guardian_authorization(
        participant_id: str,
        action: str,
        required_threshold: float
    ) -> str  # Returns request_id
```

---

## FL Integration

### Governed FL Coordinator

**Location**: `src/zerotrustml/governance/governed_fl_coordinator.py`

Extended FL coordinator with governance integration.

**Integration Points**:

1. **Pre-Round Capability Checks**:
   - Verify participant authorized before accepting gradients
   - Check emergency stop status
   - Check participant ban status

2. **Parameter Change Proposals**:
   - Hyperparameter changes require governance approval
   - Automatic proposal creation for FL parameter changes

3. **Emergency Stop Integration**:
   - FL coordinator can request emergency stop
   - Guardian approval required
   - Automatic halt of training when approved

4. **Participant Management**:
   - Ban/unban participants via governance proposals
   - Automatic rejection of banned participant submissions

5. **Reputation-Based Rewards**:
   - Reward multipliers based on identity verification
   - High-reputation participants earn more rewards

**Usage**:
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

---

## Security Properties

### Sybil Attack Resistance

**Mechanism**: Reputation-weighted voting + quadratic voting

**Effect**: New Sybil identities have low vote weight (~0.6x), making mass Sybil creation ineffective.

**Bound**: Even with 1000 Sybil identities (E0, 0.2 rep each):
```
Total Sybil power = 1000 × 0.66 = 660 credits
Effective votes = sqrt(660) × 0.66 = 16.9 votes

vs. 1 verified E4 identity (vote_weight=7.24):
Budget = 724 credits
Effective votes = sqrt(724) × 7.24 = 195 votes
```

### Vote Buying Prevention

**Mechanism**: Quadratic voting

**Effect**: Cost of buying votes increases quadratically, making vote buying economically infeasible.

**Example**:
- 10 votes: 100 credits
- 20 votes: 400 credits (4x cost for 2x votes)
- 40 votes: 1600 credits (16x cost for 4x votes)

### Byzantine Attack Tolerance

**Mechanism**: Reputation-weighted power, not raw participant count

**Effect**: 51% of participants ≠ 51% of voting power

**Bound**: Byzantine power limited by reputation accumulation rate. New attackers start with low reputation and cannot easily gain trust.

### Collusion Prevention

**Mechanism**: Guardian networks with diversity requirements

**Effect**: Circular guardian networks (cartels) are detectable and can be excluded from critical actions.

**Mitigation**: Governance requires independent guardians for high-stakes actions (E3+, 0.8+ reputation threshold).

---

## Performance Characteristics

| Operation | Target | Typical (Mocked) | Notes |
|-----------|--------|------------------|-------|
| Vote weight calculation | <5ms | ~2ms | Fast identity lookup |
| Capability enforcement | <10ms | ~5ms | Cached identity data |
| Vote casting | <100ms | ~20ms | DHT write + signature |
| Vote tallying (1000 votes) | <200ms | ~50ms | In-memory aggregation |
| Guardian authorization request | <500ms | ~100ms | DHT write + notifications |
| Proposal submission | <1000ms | ~200ms | DHT write + validation |

**Note**: Performance with real Holochain conductor will be slower than mocked tests. Expect 2-5x latency for DHT operations.

---

## Configuration

### Governance Coordinator Config

```python
@dataclass
class GovernanceConfig:
    # Proposal defaults
    default_voting_duration_days: int = 7
    default_review_period_hours: int = 1
    default_quorum: float = 0.5
    default_approval_threshold: float = 0.66

    # Voting
    base_vote_budget: int = 100
    enable_quadratic_voting: bool = True

    # Guardian authorization
    guardian_approval_timeout_seconds: int = 3600  # 1 hour
    default_guardian_threshold: float = 0.7

    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_window_seconds: int = 86400  # 24 hours
```

### FL Integration Config

```python
@dataclass
class FLGovernanceConfig:
    # Pre-round checks
    require_capability_for_submit: bool = True
    check_emergency_stop: bool = True
    check_ban_status: bool = True

    # Emergency actions
    emergency_stop_requires_guardian: bool = True
    emergency_stop_capability: str = "emergency_stop"

    # Participant management
    ban_requires_proposal: bool = True
    ban_requires_guardian: bool = True

    # Parameter changes
    parameter_change_requires_proposal: bool = True
    parameter_change_requires_guardian: bool = False

    # Rewards
    reputation_weighted_rewards: bool = True
    reward_multiplier_min: float = 0.5
    reward_multiplier_max: float = 2.0
```

---

## Deployment Considerations

### DHT Deployment

1. **Holochain Conductor**: Run Holochain conductor with governance_record zome
2. **DNA Hash**: Register DNA hash with Identity Coordinator
3. **Bootstrap Nodes**: Ensure sufficient DHT bootstrap nodes for reliability
4. **Persistence**: Configure DHT persistence for long-term proposal/vote storage

### Identity System Requirements

- Identity DHT (Week 5-6) must be fully operational
- Identity Coordinator must be initialized with DHT connection
- Guardian networks must be established for critical actions

### FL Coordinator Integration

- Use `GovernedFLCoordinator` as drop-in replacement for `Phase10Coordinator`
- Configure FL governance settings via `FLGovernanceConfig`
- Enable governance checks in production (disable for testing only)

### Monitoring & Observability

- **Proposal Metrics**: Creation rate, approval rate, execution success rate
- **Voting Metrics**: Participation rate, vote distribution, quorum achievement
- **Capability Metrics**: Authorization success/failure rates, rate limit hits
- **Guardian Metrics**: Response times, approval rates, timeout frequency
- **Performance Metrics**: DHT latency, vote tallying time, proposal execution time

---

## Example Workflows

### Creating a Proposal

```python
success, message, proposal_id = await gov_coord.create_governance_proposal(
    proposer_participant_id="alice_id",
    proposal_type="PARAMETER_CHANGE",
    title="Increase minimum reputation threshold",
    description="Increase from 0.5 to 0.7 for better security",
    execution_params={
        "parameter": "min_reputation",
        "current_value": 0.5,
        "new_value": 0.7,
    },
    voting_duration_days=7,
)

if success:
    print(f"Proposal created: {proposal_id}")
else:
    print(f"Failed: {message}")
```

### Casting a Vote

```python
success, message = await gov_coord.voting_engine.cast_vote(
    voter_participant_id="bob_id",
    proposal_id=proposal_id,
    choice=VoteChoice.FOR,
    credits_spent=100,
)

if success:
    print("Vote cast successfully")
else:
    print(f"Vote failed: {message}")
```

### Requesting Emergency Stop

```python
from zerotrustml.governance.fl_integration import FLGovernanceIntegration

fl_gov = FLGovernanceIntegration(governance_coordinator=gov_coord)

success, message, request_id = await fl_gov.request_emergency_stop(
    requester_participant_id="alice_id",
    reason="Byzantine attack detected in round 42",
    evidence={"byzantine_ratio": 0.48, "affected_rounds": [40, 41, 42]},
)

if success:
    print(f"Emergency stop requested: {request_id}")
    print("Waiting for guardian approval...")
else:
    print(f"Failed: {message}")
```

---

## Further Reading

- **[Capability Registry Reference](./CAPABILITY_REGISTRY.md)** - Complete capability definitions
- **[Proposal Creation Guide](./PROPOSAL_CREATION_GUIDE.md)** - Step-by-step proposal creation
- **[Voting Mechanics](./VOTING_MECHANICS.md)** - Deep dive into voting system
- **[Guardian Authorization Guide](./GUARDIAN_AUTHORIZATION_GUIDE.md)** - Guardian approval workflows
- **[FL Integration Examples](./FL_INTEGRATION_EXAMPLES.md)** - Governed FL coordinator usage

---

**Document Status**: Phase 6 User Documentation Complete
**Version**: 1.0
**Last Updated**: November 11, 2025
