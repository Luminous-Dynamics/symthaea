# Comprehensive BFT Testing Plan: All Trust Models, All Attack Types

**Date**: October 30, 2025
**Objective**: Rigorous validation of Hybrid-Trust Architecture across all scenarios
**Status**: Planning complete, implementation in progress

---

## 🎯 Testing Philosophy

**Core Principle**: **Test the boundaries, not just the sweet spot**

Our testing must be:
1. **Comprehensive**: All trust models, all BFT percentages, all attack types
2. **Modular**: Parameterized tests that can run any configuration
3. **Rigorous**: Multi-seed validation, statistical significance
4. **Practical**: Reflects real-world deployment scenarios

---

## 📊 BFT Testing Matrix

### Dimension 1: Trust Models (3 modes)

| Mode | Name | Primary Defense | BFT Limit | Status |
|------|------|----------------|-----------|--------|
| **Mode 0** | Public Trust | 0TML Hybrid Detector | ~35% | ✅ Tested (Week 3) |
| **Mode 1** | Intra-Federation | PoGQ + RB-BFT | >50% | ✅ Tested (40-50% BFT) |
| **Mode 2** | Inter-Federation | TEE Attestation | >50% | ⏳ Not implemented |

### Dimension 2: BFT Percentages (7 levels)

| BFT % | Honest | Byzantine | Expected Behavior | Test Status |
|-------|--------|-----------|------------------|-------------|
| **20%** | 16 | 4 | All modes succeed | ✅ Mode 0, Mode 1 tested |
| **30%** | 14 | 6 | All modes succeed | ✅ Mode 0, Mode 1 tested |
| **35%** | 13 | 7 | Mode 0 boundary, Mode 1/2 succeed | 🚧 Needs Mode 0 test |
| **40%** | 12 | 8 | Mode 0 FAILS, Mode 1/2 succeed | ✅ Mode 1 tested, ⏳ Mode 0 fail-safe |
| **50%** | 10 | 10 | Mode 0 FAILS, Mode 1/2 boundary | ✅ Mode 1 tested, ⏳ Mode 0 fail-safe |
| **60%** | 8 | 12 | Mode 0 FAILS, Mode 1/2 degrade | ⏳ Needs test |
| **70%** | 6 | 14 | All modes likely fail | ⏳ Needs test |

### Dimension 3: Attack Types (10+ varieties)

| Attack Type | Sophistication | Description | Test Status |
|-------------|----------------|-------------|-------------|
| **Random Noise** | Low | Gaussian noise gradients | ✅ Baseline |
| **Sign Flip** | Low | Negate honest gradient | ✅ Tested |
| **Scaling Attack** | Medium | Multiply gradient by large constant | ✅ Magnitude detector |
| **Noise Masked** | Medium | Poison masked by random noise | ✅ test_40_50 |
| **Targeted Neuron** | Medium-High | Backdoor-style (modify 5% of weights) | ✅ test_40_50 |
| **Adaptive Stealth** | High | Learn to evade detection thresholds | ✅ test_40_50 |
| **Coordinated Collusion** | High | Multiple Byzantine nodes collaborate | ✅ test_40_50 |
| **Sleeper Agent** | Very High | Honest → Byzantine after N rounds | ⏳ **NEEDS IMPLEMENTATION** |
| **Gradient Inversion** | Very High | Extract training data from gradients | ⏳ Privacy attack |
| **Model Poisoning** | Very High | Inject persistent backdoor | ⏳ Advanced |

**Additional Sophisticated Attacks to Implement**:
1. **Sybil Attack**: Single adversary controls multiple Byzantine identities
2. **Byzantine Coordination**: Sophisticated collusion with adaptive strategies
3. **Reputation Manipulation**: Byzantine nodes try to boost/tank reputations
4. **Data Poisoning**: Corrupt local training data (not just gradients)

---

## 🧪 Comprehensive Test Matrix (90 test combinations)

### Mode 0: Public Trust (0TML Hybrid Detector)

**BFT Range**: 20-35% (graceful, expected success) + 40-50% (fail-safe testing)

| BFT % | Attack Type | Expected Result | Test Priority |
|-------|-------------|----------------|---------------|
| 20% | Random Noise | ✅ Detect 90%+ | P0 - Baseline |
| 20% | Sign Flip | ✅ Detect 95%+ | P0 - Baseline |
| 20% | Scaling | ✅ Detect 90%+ | P0 - Magnitude signal |
| 20% | Noise Masked | ✅ Detect 70-80% | P1 - Ensemble test |
| 20% | Targeted Neuron | ✅ Detect 60-70% | P1 - Sophisticated |
| 20% | Adaptive Stealth | ✅ Detect 50-60% | P1 - Sophisticated |
| 20% | Coordinated Collusion | ✅ Detect 70-80% | P1 - Ensemble test |
| 20% | Sleeper Agent | ❓ Detect 40-60% | **P0 - NEW TEST** |
| | | | |
| 30% | [All 8 attacks] | ✅ Similar to 20% | P0 - Week 3 validation |
| | | | |
| 35% | [All 8 attacks] | ✅ Boundary case | **P0 - CRITICAL** |
| | | | |
| **40%** | [All 8 attacks] | ❌ **HALT NETWORK** | **P0 - FAIL-SAFE TEST** |
| **50%** | [All 8 attacks] | ❌ **HALT NETWORK** | **P0 - FAIL-SAFE TEST** |

**Total Mode 0 Tests**: ~56 test combinations (7 BFT % × 8 attack types)

**Critical Tests**:
1. **35% BFT**: Validate peer-comparison still works at boundary
2. **40% BFT**: Validate fail-safe halts network gracefully
3. **Sleeper Agent**: Test temporal consistency signal effectiveness

---

### Mode 1: Intra-Federation (PoGQ + RB-BFT)

**BFT Range**: 20-60% (expected to work beyond classical 33% limit)

| BFT % | Attack Type | Expected Result | Test Priority |
|-------|-------------|----------------|---------------|
| 20% | Random Noise | ✅ Detect 95%+ | P0 - Baseline |
| 20% | Sign Flip | ✅ Detect 99%+ | P0 - PoGQ strong |
| 20% | Scaling | ✅ Detect 99%+ | P0 - PoGQ strong |
| 20% | Noise Masked | ✅ Detect 90%+ | P1 - PoGQ test |
| 20% | Targeted Neuron | ✅ Detect 80-90% | P1 - Subtle attack |
| 20% | Adaptive Stealth | ✅ Detect 70-80% | P1 - Sophisticated |
| 20% | Coordinated Collusion | ✅ Detect 85%+ | P1 - PoGQ resilience |
| 20% | Sleeper Agent | ✅ Detect 90%+ | P1 - PoGQ+Temporal |
| | | | |
| 30-40% | [All 8 attacks] | ✅ Similar performance | P0 - Core validation |
| | | | |
| **50%** | [All 8 attacks] | ⚠️ **BOUNDARY** (degraded but functional) | **P0 - CRITICAL** |
| **60%** | [All 8 attacks] | ❌ Expected failure | P1 - Limit finding |

**Total Mode 1 Tests**: ~48 test combinations (6 BFT % × 8 attack types)

**Critical Tests**:
1. **40% BFT**: Demonstrate exceeding classical 33% limit
2. **50% BFT**: Find exact boundary of PoGQ effectiveness
3. **Sleeper Agent**: Validate PoGQ detects even delayed attacks

---

### Mode 2: Inter-Federation (TEE Attestation)

**BFT Range**: 20-70% (expected to work regardless of Byzantine %)

**Status**: ⏳ **NOT YET IMPLEMENTED** - Future work

| BFT % | Attack Type | Expected Result | Test Priority |
|-------|-------------|----------------|---------------|
| 20-60% | Any | ✅ TEE proves correct execution | P2 - Future |
| 70% | Any | ✅ TEE cryptographically enforces | P2 - Future |
| Any | TEE Bypass Attempt | ❌ Detect invalid attestation | **P0 - SECURITY** |

**Total Mode 2 Tests**: ~16 test combinations (2 BFT ranges × 8 attack types)

**Critical Tests**:
1. **TEE Attestation Validation**: Cryptographic verification
2. **TEE Bypass Detection**: Invalid attestation rejection
3. **Performance Overhead**: Measure computational cost

---

## 🦹 Comprehensive Attack Taxonomy

### Category 1: Gradient Manipulation (Basic)

**1.1 Random Noise**
```python
attack_gradient = np.random.normal(0, 1.0, gradient.shape)
```
- **Sophistication**: Low
- **Detectability**: Very High (magnitude + similarity signals)
- **Real-World**: Accidental corruption, network errors

**1.2 Sign Flip**
```python
attack_gradient = -honest_gradient
```
- **Sophistication**: Low
- **Detectability**: Very High (similarity signal)
- **Real-World**: Simple malicious attack

**1.3 Scaling Attack**
```python
attack_gradient = honest_gradient * 100.0
```
- **Sophistication**: Low
- **Detectability**: Very High (magnitude signal)
- **Real-World**: Amplification attack

---

### Category 2: Obfuscated Attacks (Medium)

**2.1 Noise-Masked Poisoning**
```python
poison = -honest_gradient
noise = np.random.normal(0, 0.4, gradient.shape)
attack_gradient = 0.5 * honest_gradient + 0.3 * poison + 0.2 * noise
```
- **Sophistication**: Medium
- **Detectability**: Medium (ensemble needed)
- **Real-World**: Sophisticated attacker hiding malicious intent

**2.2 Targeted Neuron Attack** (Backdoor)
```python
attack_grad = honest_gradient.copy()
num_to_modify = int(len(attack_grad) * 0.05)  # Only 5%
indices = np.random.choice(len(attack_grad), num_to_modify, replace=False)
attack_grad[indices] *= -10
```
- **Sophistication**: Medium-High
- **Detectability**: Medium (subtle, only 5% modified)
- **Real-World**: Backdoor injection, trigger-based attacks

**2.3 Adaptive Stealth**
```python
perturbation = np.random.normal(0, 0.1, gradient.shape)
attack_gradient = honest_gradient * 0.7 + perturbation
```
- **Sophistication**: High
- **Detectability**: Low-Medium (learns to stay below thresholds)
- **Real-World**: Adaptive adversary

---

### Category 3: Coordinated Attacks (High)

**3.1 Coordinated Collusion**
```python
poison_direction = -honest_gradient / np.linalg.norm(honest_gradient)
attack_gradient = honest_gradient * 0.3 + poison_direction * 0.7
```
- **Sophistication**: High
- **Detectability**: Medium (multiple attackers send similar gradients)
- **Real-World**: Organized Byzantine coalition

**3.2 Sybil Attack**
```python
# Single adversary controls multiple identities
# Submits coordinated gradients from N identities
for identity in controlled_identities:
    submit_gradient(identity, coordinated_attack_gradient)
```
- **Sophistication**: High
- **Detectability**: Low (requires identity verification)
- **Real-World**: Single powerful adversary

---

### Category 4: Stateful/Adaptive Attacks (Very High)

**4.1 Sleeper Agent** ⭐ **NEW - MUST IMPLEMENT**
```python
class SleeperAgent:
    def __init__(self, activation_round: int = 5):
        self.activation_round = activation_round
        self.current_round = 0

    def get_gradient(self, honest_gradient: np.ndarray) -> np.ndarray:
        self.current_round += 1

        if self.current_round < self.activation_round:
            # HONEST BEHAVIOR (build reputation)
            return honest_gradient
        else:
            # BYZANTINE BEHAVIOR (after reputation built)
            return -honest_gradient  # Sign flip after activation
```
- **Sophistication**: Very High
- **Detectability**: Medium (temporal signal should catch transition)
- **Real-World**: Long-term infiltration, delayed attack
- **Test Objective**: Validate temporal consistency detector catches state changes

**4.2 Gradient Inversion Attack** (Privacy)
```python
# Attacker tries to reconstruct training data from gradients
reconstructed_data = gradient_inversion_algorithm(observed_gradients)
```
- **Sophistication**: Very High
- **Detectability**: N/A (privacy attack, not Byzantine)
- **Real-World**: Privacy breach, sensitive data extraction
- **Defense**: Differential privacy, secure aggregation

**4.3 Model Poisoning** (Persistent Backdoor)
```python
# Inject persistent backdoor that survives aggregation
backdoor_gradient = craft_persistent_backdoor(honest_gradient, trigger_pattern)
```
- **Sophistication**: Very High
- **Detectability**: Very Low (subtle, persistent)
- **Real-World**: Advanced persistent threat (APT)

**4.4 Reputation Manipulation**
```python
# Try to boost Byzantine reputation or tank honest reputation
if target_is_honest:
    accuse_as_byzantine(target_node)  # False accusation
else:
    vouch_for_byzantine(target_node)  # Boost malicious actor
```
- **Sophistication**: Very High
- **Detectability**: Medium (requires reputation system integrity)
- **Real-World**: Social engineering, trust manipulation

---

## 🏗️ Modular Test Framework Design

### Architecture

```
tests/
├── test_bft_comprehensive.py          # NEW - Unified modular test
│   ├── BFTTestFramework (base class)
│   ├── TrustModelMode (enum: Public, GroundTruth, TEE)
│   ├── AttackType (enum: all 10+ attacks)
│   ├── BFTConfiguration (dataclass: %, nodes, rounds)
│   └── run_bft_test(mode, bft_%, attack, seeds)
│
├── byzantine_attacks/                  # NEW - Attack implementations
│   ├── __init__.py
│   ├── basic_attacks.py               # Noise, sign flip, scaling
│   ├── obfuscated_attacks.py          # Noise-masked, targeted neuron
│   ├── coordinated_attacks.py         # Collusion, sybil
│   └── stateful_attacks.py            # ⭐ Sleeper agent, adaptive
│
├── test_30_bft_validation.py          # ARCHIVE - Week 3 peer-comparison
└── test_40_50_bft_breakthrough.py     # ARCHIVE - PoGQ + RB-BFT
```

### Modular Test Example

```python
# Example: Run comprehensive BFT sweep across all configurations

@pytest.mark.parametrize("bft_percent", [20, 30, 35, 40, 50, 60])
@pytest.mark.parametrize("attack_type", AttackType)
@pytest.mark.parametrize("trust_mode", [TrustModelMode.PUBLIC, TrustModelMode.GROUND_TRUTH])
async def test_bft_comprehensive(bft_percent, attack_type, trust_mode):
    """
    Comprehensive BFT test across all dimensions

    This single test runs 6 BFT % × 10 attacks × 2 trust models = 120 combinations
    """

    # Configure test
    config = BFTConfiguration(
        total_nodes=20,
        byzantine_percentage=bft_percent,
        attack_type=attack_type,
        trust_model=trust_mode,
        num_rounds=10,
        seeds=[42, 123, 456]  # Multi-seed validation
    )

    # Run test
    results = await run_bft_test(config)

    # Validate results based on trust model and BFT %
    if trust_mode == TrustModelMode.PUBLIC:
        if bft_percent <= 35:
            assert results.fpr < 0.05, f"Mode 0 FPR too high at {bft_percent}% BFT"
            assert results.detection > 0.68, f"Mode 0 detection too low"
        else:
            assert results.network_halted, f"Mode 0 should halt at {bft_percent}% BFT"

    elif trust_mode == TrustModelMode.GROUND_TRUTH:
        if bft_percent <= 50:
            assert results.fpr < 0.10, f"Mode 1 FPR too high at {bft_percent}% BFT"
            assert results.detection > 0.80, f"Mode 1 detection too low"
```

---

## 📋 Implementation Checklist

### Phase 1: Refactor and Consolidate (Current)

- [x] Document Hybrid-Trust Architecture
- [x] Create comprehensive testing plan
- [ ] **Implement Sleeper Agent attack** ⭐ **CRITICAL**
- [ ] Create `byzantine_attacks/` module
- [ ] Create `test_bft_comprehensive.py` framework
- [ ] Migrate Mode 0 tests from `test_30_bft_validation.py`
- [ ] Migrate Mode 1 tests from `test_40_50_bft_breakthrough.py`
- [ ] Archive old tests with migration guide

### Phase 2: Comprehensive Validation (Week 4)

- [ ] Run Mode 0: 20%, 30%, 35% BFT across all attacks
- [ ] Run Mode 0: 40%, 50% BFT fail-safe tests
- [ ] Run Mode 1: 20%, 30%, 40%, 50% BFT across all attacks
- [ ] Multi-seed validation (5 seeds minimum)
- [ ] Generate comprehensive results report

### Phase 3: Advanced Attacks (Month 1)

- [ ] Implement Model Poisoning attack
- [ ] Implement Gradient Inversion (privacy) attack
- [ ] Implement Reputation Manipulation attack
- [ ] Implement Data Poisoning (corrupt local data)
- [ ] Validate all advanced attacks

### Phase 4: TEE Integration (Month 2)

- [ ] Implement Mode 2 (TEE Attestation)
- [ ] Test Mode 2 across all BFT percentages
- [ ] Validate cryptographic attestation
- [ ] Measure performance overhead

---

## 🎯 Success Criteria

### Overall Test Coverage

- **Trust Models**: 3/3 modes tested (Mode 0 ✅, Mode 1 ✅, Mode 2 ⏳)
- **BFT Range**: 20-60% tested across 6 levels
- **Attack Types**: 10+ attack varieties implemented
- **Seeds**: 5+ random seeds for statistical validation
- **Total Combinations**: 150+ test scenarios

### Performance Targets

**Mode 0 (Public Trust)**: 0-35% BFT
- FPR: <5% (target: <2%)
- Detection: >68% (target: >80%)
- Network Halt: Required at >35% BFT

**Mode 1 (Ground Truth)**: 0-50% BFT
- FPR: <10% (target: <5%)
- Detection: >80% (target: >90%)
- Graceful Degradation: 50-60% BFT

**Mode 2 (TEE)**: Any BFT %
- Attestation Verification: 100%
- Invalid Attestation Rejection: 100%
- Performance Overhead: <20%

---

## 🚀 Immediate Next Steps

1. **Implement Sleeper Agent Attack** ⭐ **TOP PRIORITY**
   - Create `byzantine_attacks/stateful_attacks.py`
   - Implement `SleeperAgentAttacker` class
   - Test against Mode 0 (temporal signal) and Mode 1 (PoGQ)

2. **Create Modular Test Framework**
   - Design `test_bft_comprehensive.py` structure
   - Implement parameterized test runner
   - Add multi-seed validation

3. **Run Critical Boundary Tests**
   - Mode 0: 35% BFT (boundary case)
   - Mode 0: 40% BFT (fail-safe validation)
   - Mode 1: 50% BFT (performance limit)

4. **Archive Old Tests**
   - Move `test_30_bft_validation.py` → `archive/`
   - Move `test_40_50_bft_breakthrough.py` → `archive/`
   - Create migration guide

---

**Status**: Comprehensive plan documented, ready for implementation
**Impact**: Rigorous validation of all trust models across all realistic scenarios
**Next**: Implement Sleeper Agent attack and modular test framework

---

*"Test the boundaries. That's where you learn what the system is truly capable of."*
