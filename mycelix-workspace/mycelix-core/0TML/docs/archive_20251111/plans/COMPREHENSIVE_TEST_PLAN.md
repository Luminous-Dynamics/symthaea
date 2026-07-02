# Comprehensive Test Plan: Validating Hybrid-Trust Architecture

**Created**: November 4, 2025
**Purpose**: Complete empirical validation of all architecture claims
**Timeline**: 2-4 weeks for core tests, 2-3 months for complete matrix

---

## 🎯 Primary Objective: Validate PoGQ's 45% BFT Claim

### Test Suite 1: Mode 1 (Ground Truth) Boundary Validation

**Goal**: Empirically confirm that PoGQ (Mode 1) succeeds where peer-comparison (Mode 0) fails.

#### Test 1.1: Mode 1 at 35% BFT ⭐ **CRITICAL**
```python
# Configuration
num_clients = 20
num_byzantine = 7  # 35% BFT
detector = GroundTruthDetector(validation_set=mnist_val)  # Mode 1
attack = SignFlipAttack()
training_rounds = 10

# Expected Result
# - Detection: >90% (using validation loss as quality metric)
# - FPR: <5% (honest gradients improve validation loss)
# - Network Status: Operational (no fail-safe trigger)
```

**Why Critical**: Direct comparison with boundary test that had 100% FPR with Mode 0.

#### Test 1.2: Mode 1 at 40% BFT
```python
# Configuration
num_byzantine = 8  # 40% BFT
# Same setup as 1.1

# Expected Result
# - Detection: >85%
# - FPR: <10%
# - Network Status: Operational
```

#### Test 1.3: Mode 1 at 45% BFT ⭐ **VALIDATES WHITEPAPER CLAIM**
```python
# Configuration
num_byzantine = 9  # 45% BFT
# Same setup as 1.1

# Expected Result (Per PoGQ Whitepaper)
# - Detection: >80%
# - Final Accuracy: >95%
# - Network Status: Operational
```

#### Test 1.4: Mode 1 at 50% BFT (Boundary of Mode 1)
```python
# Configuration
num_byzantine = 10  # 50% BFT
# Same setup as 1.1

# Expected Result
# - Should fail or degrade significantly
# - Validates Mode 1 ceiling (50% is limit)
```

**Multi-Seed Validation**: Run all tests with seeds [42, 123, 456] for statistical robustness.

---

## 🔬 Test Suite 2: Full 0TML Detector at Critical Boundaries

**Goal**: Prove that temporal + reputation signals extend Mode 0 ceiling.

#### Test 2.1: Full 0TML Detector at 35% BFT ⭐ **HIGH VALUE**
```python
# Configuration
detector = HybridByzantineDetector(
    temporal_window_size=5,
    similarity_weight=0.5,
    temporal_weight=0.3,
    magnitude_weight=0.2,
    reputation_enabled=True
)
num_byzantine = 7  # 35% BFT

# Expected Result
# - Detection: >80% (vs 100% with simplified)
# - FPR: <10% (vs 100% with simplified) ⭐
# - Network Status: Operational (vs Halted with simplified)
```

**Value**: Direct comparison showing temporal + reputation are ESSENTIAL, not optional.

**Comparison Table We'll Generate:**
| Detector Type | Detection Rate | FPR | Network Status |
|---------------|----------------|-----|----------------|
| Simplified (Week boundary test) | 100% | 100% ❌ | Halted |
| **Full 0TML** | >80% | **<10%** ✅ | **Operational** |

#### Test 2.2: Full 0TML at 30% BFT (Baseline)
Repeat Week 3 test with label skew (Dirichlet α=0.1) for fair comparison.

---

## 🧪 Test Suite 3: Ablation Studies

**Goal**: Prove every component is necessary by removing each and measuring degradation.

#### Test 3.1: Remove Temporal Signal
```python
detector = HybridByzantineDetector(
    temporal_weight=0.0,  # REMOVED
    similarity_weight=0.7,
    magnitude_weight=0.3
)
attack = SleeperAgent(activation_round=5)

# Expected Result
# - Sleeper detection: <50% (vs 100% with temporal)
# - Proves temporal signal is critical for stateful attacks
```

#### Test 3.2: Remove Magnitude Signal
```python
detector = HybridByzantineDetector(
    magnitude_weight=0.0,  # REMOVED
    similarity_weight=0.7,
    temporal_weight=0.3
)
attack = ScalingAttack(multiplier=100)

# Expected Result
# - Scaling attack detection: <50% (vs >90% with magnitude)
# - Proves magnitude signal is critical for scaling attacks
```

#### Test 3.3: Remove Reputation System
```python
detector = HybridByzantineDetector(
    reputation_enabled=False  # REMOVED
)
num_byzantine = 7  # 35% BFT

# Expected Result
# - FPR: >50% (vs <10% with reputation)
# - Proves reputation filters transient false positives
```

#### Test 3.4: Remove Fail-Safe
```python
# Run 35% BFT test WITHOUT fail-safe
# Expected Result
# - Silent corruption (accepts all Byzantine, rejects all honest)
# - Proves fail-safe prevents catastrophic failure
```

---

## 🎭 Test Suite 4: Attack Type Coverage

**Goal**: Demonstrate generalization beyond sign flip and Sleeper Agent.

#### Test 4.1: Scaling Attack
```python
attack = ScalingAttack(multiplier=100)
# Test at 30%, 35%, 40% BFT with Mode 0 and Mode 1
```

#### Test 4.2: Noise Attack
```python
attack = GaussianNoiseAttack(noise_level=0.5)
# Test at 30%, 35%, 40% BFT with Mode 0 and Mode 1
```

#### Test 4.3: Adaptive Stealth Attack
```python
attack = AdaptiveStealth(
    magnitude_multiplier=0.8,  # Stay under threshold
    similarity_threshold=0.75   # Appear similar to honest
)
# Test at 30% BFT (should be harder to detect)
```

#### Test 4.4: Model Poisoning
```python
attack = ModelPoisoning(
    target_class=7,
    poison_percentage=0.1
)
# Test at 30% BFT with Mode 1 (requires validation set)
```

---

## 📊 Test Suite 5: Scalability and Robustness

#### Test 5.1: Larger Networks (50 nodes)
```python
num_clients = 50
num_byzantine = [15, 18, 23]  # 30%, 36%, 46% BFT
# Test Mode 0 and Mode 1 at scale
```

#### Test 5.2: Different Datasets
- **CIFAR-10**: More complex (32×32 RGB, 10 classes)
- **Fashion-MNIST**: Similar complexity to MNIST
- **CIFAR-100**: High complexity (100 classes)

#### Test 5.3: Different Model Architectures
- **SimpleCNN**: Current baseline (MNIST)
- **ResNet-18**: Deeper network (CIFAR-10)
- **MobileNetV2**: Efficient architecture
- **Transformer**: Modern architecture

#### Test 5.4: Extreme Data Heterogeneity
```python
# Test with various Dirichlet parameters
alpha_values = [0.01, 0.1, 0.5, 1.0]  # More skew → harder detection
# Measure impact on FPR and detection rate
```

---

## 🌐 Test Suite 6: Mode 2 (Holochain) Preliminary Tests

**Note**: These are architectural validation tests, not full implementation.

#### Test 6.1: P2P Validation Logic
```python
# Simulate Holochain validation rules
# Each node validates gradients from peers
# No central aggregator

def validate_gradient(gradient, peer_reputation, dht_state):
    """
    Holochain validation function (runs on each node's source chain)
    """
    # Check gradient quality against local validation set
    quality = compute_local_quality(gradient)

    # Query DHT for peer reputation
    rep = dht_state.get_reputation(peer_id)

    # Accept/reject based on quality + reputation
    return quality > threshold and rep > min_reputation
```

#### Test 6.2: Distributed Fail-Safe
```python
# Test consensus-based halt trigger
# Network halts when >67% of nodes detect unsafe conditions
# No single point of failure
```

#### Test 6.3: DHT-Based Reputation Propagation
```python
# Test reputation updates via Holochain DHT
# Every node maintains local view, gossips updates
# Measure convergence time and consistency
```

---

## ⏱️ Timeline and Priorities

### Phase 1 (Week 1): Critical Validations ⭐
**Focus**: Tests that directly validate whitepaper claims
1. Test 1.1: Mode 1 at 35% BFT (2 days)
2. Test 1.3: Mode 1 at 45% BFT (2 days)
3. Test 2.1: Full 0TML at 35% BFT (2 days)

**Deliverable**: Confirm PoGQ claim empirically

### Phase 2 (Week 2): Ablation Studies
**Focus**: Prove every component is necessary
1. Test 3.1-3.4: All ablation tests (4 days)
2. Multi-seed validation for all Phase 1 tests (2 days)

**Deliverable**: Component necessity proof

### Phase 3 (Week 3): Attack Coverage
**Focus**: Generalization beyond simple attacks
1. Test 4.1-4.4: Four additional attack types (4 days)
2. Generate comparison matrix (2 days)

**Deliverable**: Comprehensive attack matrix

### Phase 4 (Week 4): Scalability & Dataset Diversity
**Focus**: Real-world applicability
1. Test 5.1: 50-node network (2 days)
2. Test 5.2: CIFAR-10 validation (2 days)
3. Test 5.3: ResNet-18 architecture (2 days)

**Deliverable**: Scalability evidence

### Phase 5 (Weeks 5-8): Mode 2 Preliminary Tests (Optional)
**Focus**: Holochain architectural validation
1. Design Holochain validation logic (1 week)
2. Simulate DHT-based aggregation (1 week)
3. Test distributed fail-safe (1 week)
4. Write up results (1 week)

**Deliverable**: Mode 2 feasibility study

---

## 📈 Expected Outcomes

### Hypothesis 1: PoGQ Exceeds 33% (Mode 1)
**Expected**: Mode 1 succeeds at 35%, 40%, 45% BFT where Mode 0 fails
**Evidence**: Test Suite 1
**Impact**: Validates whitepaper core claim ✅

### Hypothesis 2: Temporal + Reputation Extend Mode 0
**Expected**: Full 0TML detector operational at 35% vs simplified halt
**Evidence**: Test Suite 2
**Impact**: Proves architectural design is essential ✅

### Hypothesis 3: Every Component Necessary
**Expected**: Removing any component causes >20% degradation
**Evidence**: Test Suite 3
**Impact**: Justifies complexity ✅

### Hypothesis 4: Generalizes Beyond Sign Flip
**Expected**: >80% detection across 4+ attack types
**Evidence**: Test Suite 4
**Impact**: Practical applicability ✅

### Hypothesis 5: Scales to Production
**Expected**: Performance maintained at 50 nodes, CIFAR-10
**Evidence**: Test Suite 5
**Impact**: Deployment readiness ✅

### Hypothesis 6: Holochain Enables True P2P
**Expected**: Distributed validation without central server
**Evidence**: Test Suite 6
**Impact**: Decentralization feasibility ✅

---

## 💻 Implementation Checklist

### Code to Create:
- [ ] `src/ground_truth_detector.py` (Mode 1 implementation)
- [ ] `tests/test_mode1_boundaries.py` (Test Suite 1)
- [ ] `tests/test_full_0tml_35bft.py` (Test Suite 2)
- [ ] `tests/test_ablation_studies.py` (Test Suite 3)
- [ ] `tests/test_attack_matrix.py` (Test Suite 4)
- [ ] `tests/test_scalability.py` (Test Suite 5)
- [ ] `tests/test_holochain_validation.py` (Test Suite 6, architectural only)

### Data to Generate:
- [ ] All test results with statistical analysis
- [ ] Comparison tables (Mode 0 vs Mode 1)
- [ ] Ablation study graphs
- [ ] Attack coverage matrix
- [ ] Scalability charts

---

## 📊 Metrics to Report

For every test, record:
1. **Detection Rate**: TP / (TP + FN)
2. **False Positive Rate**: FP / (FP + TN)
3. **Final Model Accuracy**: On test set
4. **Network Status**: Operational / Halted
5. **BFT Estimate**: ρ̂ from fail-safe
6. **Overhead**: Time per round (ms)

**Statistical Robustness**: All results with mean ± std dev across 3 seeds

---

## 🎓 Publication Strategy

### Current Paper (Hybrid-Trust Architecture)
**Focus**: Mode 0 + Mode 1 + Fail-Safe
**Length**: 12 pages (IEEE S&P / USENIX format)
**Test Coverage**: Test Suites 1-5 (complete validation)

**Sections**:
1. Introduction (PoGQ context + fail-safe innovation)
2. Related Work (BFT + FL + reputation systems)
3. Hybrid-Trust Architecture (3 modes, focus on 0 and 1)
4. Experimental Validation (Test Suites 1-5)
5. Discussion (Holochain as future work)
6. Conclusion

### Future Paper (Decentralized Byzantine-Robust FL)
**Focus**: Mode 2 (Holochain) + Full implementation
**Length**: 12 pages (MLSys / ICML format)
**Test Coverage**: Test Suite 6 + production deployment

**Sections**:
1. Introduction (centralization problem)
2. Related Work (P2P ML + DHT)
3. Holochain Integration (DHT-based aggregation)
4. Implementation (production code)
5. Evaluation (real-world deployment)
6. Conclusion

**Rationale for Two Papers**:
- Each paper has clear, focused contribution
- Reviewers won't ask "where's the full implementation?" for Paper 1
- More citations (papers reference each other)
- Doubles publication count (PhD/tenure metrics)

---

## 🚀 Immediate Next Steps (This Week)

1. **Implement Mode 1 Detector** (ground_truth_detector.py)
   - Use validation loss as quality metric
   - Integrate with existing reputation system
   - Add to fail-safe mechanism

2. **Run Critical Tests** (Test Suite 1)
   - Mode 1 at 35% BFT (compare with boundary test)
   - Mode 1 at 45% BFT (validate whitepaper claim)
   - Multi-seed validation (3 seeds)

3. **Generate Results Table**
   ```markdown
   | Test | BFT % | Mode | Detection | FPR | Network Status |
   |------|-------|------|-----------|-----|----------------|
   | Boundary (simplified) | 35% | 0 | 100% | 100% | Halted |
   | **Mode 1** | **35%** | **1** | **>90%** | **<5%** | **Operational** ✅ |
   | **Mode 1** | **45%** | **1** | **>80%** | **<10%** | **Operational** ✅ |
   ```

4. **Update Paper Draft**
   - Add Mode 1 results to Section 5
   - Create comparison figures
   - Update discussion with validated claims

---

**Status**: Test plan complete
**Ready for**: Implementation + execution
**Estimated completion**: 4 weeks for Phase 1-4, 8 weeks for Phase 1-5

---

*"Empirical validation transforms theoretical claims into scientific truth."*
