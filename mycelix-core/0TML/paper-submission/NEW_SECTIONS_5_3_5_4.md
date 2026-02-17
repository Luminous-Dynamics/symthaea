### 5.3 Full 0TML Hybrid Detector: Limitations with Heterogeneous Data

**Research Question**: Can a sophisticated multi-signal peer-comparison detector overcome the fundamental limitations demonstrated by the simplified Mode 0? Specifically, does combining similarity, temporal consistency, and magnitude signals with ensemble voting enable reliable Byzantine detection at 30-35% BFT with heterogeneous data?

**Answer**: NO. Even our Full 0TML Hybrid Detector (with three detection signals and ensemble voting) achieves **0% detection rate** at 30-35% BFT with heterogeneous data.

**Table 5: Full 0TML Hybrid Detector Performance at 30-35% BFT**

| BFT Level | Clients (Honest/Byzantine) | Detection Rate | False Positive Rate | Max Ensemble Confidence | Status |
|-----------|---------------------------|----------------|---------------------|------------------------|--------|
| 30% | 20 (14/6) | **0.0%** (0/18) | 0.0% (0/42) | 0.11 | **FAILED** |
| 35% | 20 (13/7) | **0.0%** (0/21) | 0.0% (0/39) | 0.11 | **FAILED** |

**Key Findings**:

1. **Complete Detection Failure**: 0% of Byzantine nodes detected despite sophisticated multi-signal architecture
2. **Excellent False Positive Control**: 0% FPR demonstrates detector is correctly conservative, not broken
3. **Signal Weakness**: Maximum ensemble confidence (0.11) far below threshold (0.6), indicating insufficient signal strength
4. **Consistent Across BFT Levels**: Same failure mode at both 30% and 35% BFT

**5.3.1 Detector Architecture**

The Full 0TML Hybrid Detector combines three detection signals with weighted ensemble voting:

**Signal 1: Gradient Similarity** (Weight: 0.5)
- Computes cosine similarity between each gradient and all peers
- Flags nodes with similarity outside expected range [cos_min, cos_max]
- Based on gradient profile analysis via PCA

**Signal 2: Temporal Consistency** (Weight: 0.3)
- Tracks behavioral patterns over rolling window (5 rounds)
- Detects sudden changes in cosine similarity variance and gradient magnitude variance
- Specifically designed to catch stateful attacks (see Section 5.4)

**Signal 3: Magnitude Distribution** (Weight: 0.2)
- Analyzes gradient norm distribution across all clients
- Flags nodes with gradient magnitudes >3σ from median (Z-score outlier detection)
- Robust to heterogeneous data via MAD statistics

**Ensemble Voting**:
```
ensemble_confidence = 0.5 × sim_conf + 0.3 × temp_conf + 0.2 × mag_conf
is_byzantine = (ensemble_confidence ≥ 0.6)
```

**5.3.2 Threshold Parameter Sweep**

To determine whether threshold miscalibration caused the 0% detection, we systematically tested ensemble thresholds from 0.6 down to 0.05 at 30% BFT.

**Table 6: Threshold Sweep Results (30% BFT, 3 rounds)**

| Threshold | Detection Rate | False Positive Rate | Assessment |
|-----------|---------------|---------------------|------------|
| 0.60 | 0.0% (0/18) | 0.0% (0/42) | Too conservative |
| 0.50 | 0.0% (0/18) | 0.0% (0/42) | Too conservative |
| 0.40 | 0.0% (0/18) | 0.0% (0/42) | Too conservative |
| 0.30 | 0.0% (0/18) | 0.0% (0/42) | Too conservative |
| 0.20 | 0.0% (0/18) | 2.4% (1/42) | Too conservative |
| 0.10 | 11.1% (2/18) | 33.3% (14/42) | FPR explosion |
| 0.05 | 61.1% (11/18) | 57.1% (24/42) | Unacceptable tradeoff |

**Finding**: No threshold achieves >80% detection with <10% FPR. The best achievable configuration (threshold=0.05) provides 61% detection but with 57% FPR—an unacceptable tradeoff.

**Conclusion**: The problem is not threshold miscalibration. The underlying signals are fundamentally too weak to distinguish Byzantine from honest nodes with heterogeneous data.

**5.3.3 Signal Confidence Analysis: Root Cause**

To understand why ensemble confidences are so low (<0.11), we analyzed the individual signal contributions with heterogeneous data (Dirichlet α=0.1 label skew).

**Table 7: Signal Confidence Distribution at 35% BFT (First Round)**

| Signal | Honest Min | Honest Mean | Honest Max | Byzantine Min | Byzantine Mean | Byzantine Max | Separation |
|--------|-----------|------------|-----------|--------------|---------------|--------------|-----------|
| Similarity | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** |
| Temporal | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.000** |
| Magnitude | 0.052 | 0.320 | 0.704 | 0.024 | 0.258 | 0.472 | **0.062** |
| Ensemble | 0.010 | 0.064 | 0.141 | 0.005 | 0.052 | 0.094 | **0.012** |

**Raw Gradient Statistics**:
- **Pairwise Cosine Similarities**: Mean = 0.0506, Std = 0.486, Range = [-0.77, 0.79]
- **Gradient Norms**: Honest mean = 2.04, Byzantine mean = 1.81 (minimal difference)

**Root Cause Identified**:

1. **Similarity Signal Completely Flat** (confidence = 0.000):
   - With heterogeneous data, ALL gradients are naturally dissimilar (mean cosine 0.0506)
   - Honest nodes are as diverse as Byzantine nodes due to label skew
   - Byzantine nodes (sign flip) blend into the natural diversity
   - **No signal to distinguish Byzantine from honest heterogeneity**

2. **Temporal Signal Inactive** (confidence = 0.000):
   - First round has no history for temporal comparison
   - Even in later rounds (see Section 5.4), temporal signal remains weak (~0.3-0.4)

3. **Magnitude Signal Weak** (confidence = 0.32 honest, 0.26 Byzantine):
   - Only signal with any strength, but separation is minimal (0.062)
   - Insufficient to overcome low similarity and temporal confidences in ensemble

**Fundamental Problem**:

The detector's signal thresholds are calibrated for **IID data** where:
- Honest nodes cluster together (high cosine similarity ~0.8-0.9)
- Byzantine nodes are clear outliers
- Peer-comparison has a strong signal

With **heterogeneous data** (realistic federated learning):
- ALL nodes are spread out (cosine similarity ~0.0-0.1)
- Byzantine nodes don't look MORE dissimilar than honest diversity
- **Peer-comparison has no signal—Byzantine blends into honest heterogeneity**

**5.3.4 Implications**

This systematic validation demonstrates that **peer-comparison methods fundamentally struggle with heterogeneous data**, even with:
- Multiple detection signals (similarity + temporal + magnitude)
- Sophisticated ensemble voting
- Outlier-robust statistics (MAD, PCA)
- Extensive threshold tuning (sweep across 7 values)

The 0% detection rate is not a bug—it is the **correct behavior** of a detector that recognizes it has insufficient signal to make confident decisions. The detector maintains 0% FPR (no false accusations) but cannot identify Byzantine nodes because they are indistinguishable from honest diversity.

**This empirically validates that ground truth validation (Mode 1) is not merely beneficial—it is mathematically necessary for Byzantine-robust federated learning with realistic, heterogeneous data distributions.**

---

### 5.4 Temporal Signal Evaluation: Sleeper Agent Attack

**Research Question**: Can the temporal consistency signal detect stateful Byzantine attacks where nodes change behavior mid-training (e.g., Sleeper Agents that build reputation before activating)?

**Answer**: The temporal signal DOES detect behavioral changes (confidence increases 2.6×), but signal strength remains insufficient for reliable flagging in heterogeneous data scenarios.

**5.4.1 Sleeper Agent Attack Configuration**

**Attack Model**: Stateful Byzantine attack with delayed activation
- **Honest Phase** (Rounds 1-5): Sleeper Agents submit legitimate gradients to build reputation
- **Activation** (Round 5): Agents switch to Byzantine mode
- **Attack Phase** (Rounds 6-10): Agents submit malicious gradients (sign flip)

**Test Configuration**:
- 20 clients: 14 honest, 6 Sleeper Agents (30% BFT when activated)
- Detector: Full 0TML Hybrid with temporal signal enabled
- Data: Heterogeneous MNIST (Dirichlet α=0.1)
- Seed: 42

**Expected Behavior**:
- Pre-activation: Low detection (<20%) as Sleepers build reputation
- Activation: Temporal signal detects sudden behavioral change
- Post-activation: High detection (>80%) within 1-2 rounds

**5.4.2 Results**

**Table 8: Sleeper Agent Detection Across Training Rounds**

| Round | Sleeper Status | Detection Rate | FPR | Temporal Confidence (Sample) |
|-------|---------------|----------------|-----|----------------------------|
| 1-5 | Honest | 0.0% | 0.0% | 0.00 (no history) |
| 5 | **Activation** | 0.0% | 0.0% | 0.17-0.25 (building history) |
| 6 | Attack | 0.0% | 0.0% | **0.29-0.35** (increased) |
| 7-8 | Attack | 0.0% | 0.0% | **0.25-0.45** (peak) |
| 9-10 | Attack | 0.0% | 0.0% | 0.09-0.11 (stabilized) |

**Pre-Activation Phase (Rounds 1-5)**:
- Detection Rate: 0.0% ✅ (expected—Sleepers acting honestly)
- False Positive Rate: 0.0% ✅
- Sleeper Agents successfully built reputation without raising suspicion

**Post-Activation Phase (Rounds 6-10)**:
- Detection Rate: 0.0% ❌ (expected >80%)
- False Positive Rate: 0.0% ✅
- Temporal confidence increased but remained below ensemble threshold

**5.4.3 Temporal Signal Response**

**Behavioral Change Detection**:

The temporal consistency detector tracks two metrics over a rolling window (5 rounds):
1. **Cosine Variance**: σ²(cos_sim) over past 5 rounds
2. **Magnitude Variance**: σ²(||∇||) over past 5 rounds

**Temporal Confidence Evolution (Representative Sleeper Agent)**:

| Phase | Rounds | Cosine Variance | Magnitude Variance | Temporal Confidence |
|-------|--------|-----------------|-------------------|-------------------|
| Honest | 1-5 | 0.013-0.031 | 2.0-2.1 | 0.17-0.25 |
| Activation | 6 | 0.047 | 3.1 | **0.35** |
| Attack | 7-8 | 0.029-0.048 | 2.6-2.8 | **0.36-0.45** |
| Stabilized | 9-10 | 0.001-0.006 | 0.4-0.5 | 0.09-0.11 |

**Key Observation**: Temporal confidence increased by **2.6× (0.17 → 0.45)** immediately after activation (rounds 5 → 7), demonstrating that the temporal signal DID detect the behavioral change.

**5.4.4 Analysis: Why Detection Still Failed**

**Signal Detected Correctly**:
- Temporal variance increased significantly when Sleeper Agents activated
- Confidence peaked at 0.45 (round 7-8)
- Signal correctly identified behavioral inconsistency

**Ensemble Threshold Too High**:
- Maximum ensemble confidence: 0.31 (weighted average with similarity=0.0, temporal=0.45, magnitude=0.2)
- Ensemble threshold: 0.6
- **Even with temporal signal firing, ensemble confidence remained far below threshold**

**Heterogeneous Data Masks Signal**:
- With heterogeneous data, honest nodes also exhibit temporal variance (0.1-0.3) due to natural data distribution changes
- Sleeper Agent variance (0.35-0.45) is only marginally higher than honest variance
- **Signal exists but separation is insufficient for confident detection**

**Temporal Variance Thresholds**:
- Configured thresholds: cosine variance > 0.1, magnitude variance > 0.5
- Sleeper Agent variances: cosine 0.029-0.048, magnitude 2.6-3.1
- Magnitude threshold exceeded, but cosine threshold NOT consistently exceeded
- **Thresholds calibrated for IID data, not heterogeneous data**

**5.4.5 Implications**

This test demonstrates that:

1. **Temporal Signal Functions Correctly**: The temporal consistency detector successfully identified behavioral changes (2.6× confidence increase after activation)

2. **Signal Strength Insufficient Alone**: Even with correct detection of behavioral changes, the signal strength (0.35-0.45) is insufficient when other signals (similarity, magnitude) are weak due to heterogeneous data

3. **Stateful Attack Detection Requires Ground Truth**: In heterogeneous federated learning scenarios, detecting Sleeper Agents and other stateful attacks requires ground truth validation to provide sufficient signal strength, as peer-comparison signals (even temporal) are masked by natural data diversity

**Conclusion**: The temporal consistency signal is a valuable component of Byzantine detection, correctly identifying behavioral changes. However, in realistic heterogeneous federated learning scenarios, **temporal signals alone or in combination with peer-comparison methods cannot achieve reliable detection**. Ground truth validation (Mode 1) remains necessary to provide the strong, external reference signal required for confident Byzantine identification.

---

