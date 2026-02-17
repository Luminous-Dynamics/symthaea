# Identity-Coordinator Integration Design

**Date**: November 11, 2025
**Version**: v1.0
**Status**: Implementation Phase

---

## Executive Summary

This document describes the integration of the Multi-Factor Decentralized Identity System with the Zero-TrustML Phase 10 Coordinator to enable identity-enhanced Byzantine resistance.

**Key Goals:**
1. Integrate identity assurance levels with MATL trust scoring
2. Apply identity boost to reputation scores
3. Validate 45% Byzantine tolerance with identity verification
4. Measure performance impact on FL workflows

---

## Integration Architecture

### High-Level Flow

```
Node Registration
    ↓
Present DID + Identity Factors
    ↓
Coordinator verifies identity
    ↓
Identity signals computed (IdentityMATLBridge)
    ↓
Initial reputation assigned based on assurance level
    ↓
Gradient submission with identity-enhanced trust
    ↓
MATL score = Base Score + Identity Boost
    ↓
Byzantine-resistant aggregation (45% tolerance)
```

### Integration Points

#### 1. Node Registration/Initialization
**Location**: `Phase10Coordinator.__init__()` or new `register_node()` method
**Changes**:
- Add `identity_bridge: IdentityMATLBridge` instance
- Store node DID → identity signals mapping
- Compute initial reputation from assurance level

#### 2. Reputation Retrieval
**Location**: `Phase10Coordinator._get_reputation()`
**Changes**:
- Query identity signals for node
- Apply identity boost to base reputation
- Return enhanced score

#### 3. Gradient Submission
**Location**: `Phase10Coordinator.handle_gradient_submission()`
**Changes**:
- Include identity-enhanced reputation in validation
- Log identity assurance level for metrics
- Apply identity-weighted Byzantine resistance

#### 4. Storage Backend
**Location**: Backend implementations
**Changes**:
- Store DID → identity signals mapping
- Store identity-enhanced reputation history
- Query identity factors when needed

---

## Implementation Plan

### Phase 1: Core Integration (Week 3, Days 1-3)

#### Step 1: Add IdentityMATLBridge to Coordinator
```python
from zerotrustml.identity import IdentityMATLBridge

class Phase10Coordinator:
    def __init__(self, config):
        # ... existing init ...

        # Identity integration
        self.identity_bridge = IdentityMATLBridge()
        self.node_identities: Dict[str, IdentityTrustSignal] = {}
```

#### Step 2: Create `register_node_identity()` method
```python
async def register_node_identity(
    self,
    node_id: str,
    did: str,
    factors: List[IdentityFactor],
    credentials: List[VerifiableCredential],
    guardian_graph: Optional[Dict] = None
) -> IdentityTrustSignal:
    """
    Register node identity and compute trust signals

    Args:
        node_id: Node identifier
        did: Decentralized identifier
        factors: Identity verification factors
        credentials: Verifiable credentials
        guardian_graph: Optional guardian network graph

    Returns:
        Computed identity trust signal
    """
    # Compute identity signals
    signals = self.identity_bridge.compute_identity_signals(
        did=did,
        factors=factors,
        credentials=credentials,
        guardian_graph=guardian_graph
    )

    # Store in coordinator
    self.node_identities[node_id] = signals

    # Store in backends
    await self._store_with_strategy(
        "store_identity",
        {
            "node_id": node_id,
            "did": did,
            "identity_signals": signals.to_dict()
        }
    )

    # Set initial reputation based on identity
    initial_rep = self.identity_bridge.get_initial_reputation(did)
    await self._store_with_strategy(
        "store_reputation",
        {
            "node_id": node_id,
            "score": initial_rep,
            "source": "identity_verified",
            "assurance_level": signals.assurance_level.value
        }
    )

    logger.info(
        f"Node {node_id} registered: {signals.assurance_level.value}, "
        f"initial reputation: {initial_rep:.2f}"
    )

    return signals
```

#### Step 3: Enhance `_get_reputation()` with identity boost
```python
async def _get_reputation(self, node_id: str) -> float:
    """
    Get node reputation with identity enhancement

    Flow:
    1. Get base reputation from storage
    2. Get identity signals for node
    3. Apply identity boost
    4. Return enhanced score
    """
    # Get base reputation (PoGQ history, etc.)
    base_rep_data = await self._get_with_strategy("get_reputation", node_id)
    base_reputation = base_rep_data.get("score", 0.5) if base_rep_data else 0.5

    # Get identity signals
    identity_signals = self.node_identities.get(node_id)

    if not identity_signals:
        # No identity verification - use base only
        logger.warning(f"Node {node_id}: No identity verification, using base reputation")
        return base_reputation

    # Calculate identity boost
    identity_boost = self.identity_bridge._calculate_identity_boost(identity_signals)

    # Apply boost (clamped to [0.3, 1.0])
    enhanced_reputation = max(0.3, min(1.0, base_reputation + identity_boost))

    logger.debug(
        f"Node {node_id}: base={base_reputation:.2f}, "
        f"boost={identity_boost:+.2f}, "
        f"enhanced={enhanced_reputation:.2f}"
    )

    return enhanced_reputation
```

#### Step 4: Update `handle_gradient_submission()` to log identity
```python
async def handle_gradient_submission(self, ...):
    # ... existing validation ...

    # Get identity-enhanced reputation
    reputation = await self._get_reputation(node_id)

    # Log identity assurance for metrics
    identity_signals = self.node_identities.get(node_id)
    if identity_signals:
        result["identity_assurance"] = identity_signals.assurance_level.value
        result["sybil_resistance"] = identity_signals.sybil_resistance_score
    else:
        result["identity_assurance"] = "E0_Anonymous"
        result["sybil_resistance"] = 0.0

    # ... rest of submission logic ...
```

### Phase 2: FL Workload Testing (Week 3, Days 4-5)

#### Test Scenarios

**Test 1: Identity Levels Comparison**
- Create 10 nodes: 2 E0, 2 E1, 2 E2, 2 E3, 2 E4
- Train MNIST for 20 rounds
- Measure: Initial reputation, gradient acceptance, convergence

**Test 2: Byzantine Attack with Mixed Identity**
- Create 20 nodes:
  - 10 honest (5 E3, 5 E4)
  - 10 Byzantine (10 E0)
- Inject label-flipping attacks
- Measure: Attack detection rate, model accuracy

**Test 3: Sybil Attack with Identity**
- Create 30 nodes:
  - 10 honest (E4 verified humans)
  - 20 Sybil (E0 anonymous, same guardian network)
- Attempt to dominate voting
- Measure: Cartel detection, effective Byzantine power

#### Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Identity overhead | <50ms per gradient | Acceptable latency |
| E0 initial reputation | 0.30 | Low starting trust |
| E4 initial reputation | 0.70 | High starting trust |
| Byzantine detection (E0 attackers) | >95% | Identity flags Sybils |
| Byzantine detection (E4 attackers) | >80% | Harder to detect verified |
| Attack cost differential | >5x | E4 requires real identity |

### Phase 3: Performance Validation (Week 3-4 Transition)

#### Benchmarks to Run

1. **Identity Computation Time**
   - Measure `compute_identity_signals()` latency
   - Target: <10ms for typical case

2. **Reputation Lookup Time**
   - Measure `_get_reputation()` with identity boost
   - Target: <5ms (should be cached)

3. **Storage Overhead**
   - Measure identity data size per node
   - Target: <10KB per node

4. **FL Convergence Impact**
   - Compare convergence with/without identity
   - Target: No degradation in honest case

---

## Data Structures

### Node Identity Storage

```python
{
    "node_id": "node_001",
    "did": "did:mycelix:z5Cw...",
    "identity_signals": {
        "assurance_level": "E3_CryptographicallyProven",
        "assurance_score": 0.75,
        "active_factors": 3,
        "factor_categories": 3,
        "verified_human": true,
        "gitcoin_score": 45.0,
        "guardian_count": 5,
        "sybil_resistance_score": 0.82,
        "risk_level": "Low"
    },
    "registered_at": "2025-11-11T10:00:00Z"
}
```

### Enhanced Reputation Record

```python
{
    "node_id": "node_001",
    "base_reputation": 0.65,
    "identity_boost": 0.12,
    "enhanced_reputation": 0.77,
    "last_updated": "2025-11-11T10:05:00Z"
}
```

---

## Risk Mitigation

### Risk 1: Identity Verification Overhead
**Mitigation**: Cache identity signals, only recompute on factor changes

### Risk 2: Honest Nodes Penalized by Low Identity
**Mitigation**: Default base reputation still allows participation (0.3 minimum)

### Risk 3: Identity Spoofing
**Mitigation**: W3C DID + Ed25519 signatures prevent impersonation

### Risk 4: Guardian Cartel Detection False Positives
**Mitigation**: Use conservative threshold (>0.8 connection density) for cartel flag

---

## Testing Plan

### Unit Tests (Already Complete ✅)
- 105 tests passing
- All integration points covered

### Integration Tests (New)
1. **test_coordinator_identity_registration.py**
   - Test node registration with different identity levels
   - Verify initial reputation assignment
   - Check storage backend integration

2. **test_identity_enhanced_reputation.py**
   - Test reputation boost calculation
   - Verify identity caching
   - Test reputation updates

3. **test_byzantine_resistance_with_identity.py**
   - Test mixed identity Byzantine attack
   - Verify detection rates
   - Measure attack cost differential

### E2E Tests (New)
1. **test_mnist_with_identity.py**
   - Full FL workflow with identity
   - Multiple assurance levels
   - Convergence validation

2. **test_byzantine_scenarios.py**
   - Label flipping with E0 attackers
   - Model poisoning with mixed identity
   - Sybil attacks with cartel detection

---

## Success Criteria

**Week 3-4 Integration Complete When:**
- ✅ Identity bridge integrated into coordinator
- ✅ Node registration with DID working
- ✅ Identity-enhanced reputation functional
- ✅ MNIST training completes with identity
- ✅ Byzantine resistance validated (>45% tolerance)
- ✅ Performance acceptable (<50ms identity overhead)
- ✅ All integration tests passing

**Go/No-Go for Week 5-6:**
- All success criteria met
- Performance within targets
- No critical bugs
- Documentation complete

---

## Next Steps After Integration

1. **Week 5-6**: Holochain DHT integration for decentralized DID resolution
2. **Week 7-8**: Governance integration with assurance-gated capabilities
3. **Week 9+**: Production deployment and monitoring

---

**Status**: Ready for Implementation
**Estimated Effort**: 3-5 days
**Dependencies**: Identity system (✅ Complete), Phase 10 coordinator (✅ Available)
