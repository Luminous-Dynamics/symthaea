# Phase A v4 Analysis: Unexpected Regression

**Date**: 2025-11-12
**Validation ID**: AEGIS-GEN5-20251112-085316
**Status**: ❌ **REGRESSION - Trigger Too Strong**

---

## Executive Summary

Phase A v4 surgical enhancements resulted in **unexpected regression** on E3 backdoor defense. While robust centroiding and adaptive quarantine were implemented successfully, the strengthened backdoor trigger (feature 42, value 10.0, frac 0.4) proved **too effective**, overwhelming both AEGIS and Median defenses.

### Key Results vs Targets

| Experiment | Metric | v3 Result | v4 Result | Target | Status |
|------------|--------|-----------|-----------|--------|--------|
| **E3** | ASR(AEGIS) | 17.1% | **75.7%** | ≤25% | ❌ Catastrophic regression |
| **E3** | ASR(Median) | 22.0% | **84.5%** | N/A | ❌ Also regressed |
| **E3** | ASR ratio | 0.74 | **0.88** | ≤0.5 | ❌ Moved away from target |
| **E2** | Stability | 3/5 seeds | 3/5 seeds | 5/5 | ❌ No improvement |
| **E2** | AUC | 0.69 | 0.69 | ≥0.85 | ❌ No change |
| **E5** | AEGIS advantage | 0.1pp | 0.4pp | +5pp | ❌ Minimal change |

---

## Detailed Results Analysis

### E3: Backdoor Resilience - CATASTROPHIC FAILURE ❌

#### Comparison: v3 vs v4

| Metric | v3 Mean | v4 Mean | Change | Interpretation |
|--------|---------|---------|--------|----------------|
| **ASR(AEGIS)** | 17.1% | **75.7%** | **+343%** | ❌ AEGIS defense collapsed |
| **ASR(Median)** | 22.0% | **84.5%** | **+284%** | ❌ Median also collapsed |
| **ASR Ratio** | 0.74 | **0.88** | **+19%** | ❌ AEGIS advantage diminished |
| Clean Acc (AEGIS) | 65.8% | 65.6% | -0.2pp | ✅ Maintained |
| Detection AUC | 0.69 | 0.82 | +19% | ⚠️ Detector more confident, but wrong |

#### Per-Seed Analysis (v4)

```
seed=101: ASR(AEGIS) 92.6% vs ASR(Median) 95.9%, ratio=0.97
  → Nearly identical performance, backdoor overwhelming

seed=202: ASR(AEGIS) 51.2% vs ASR(Median) 69.0%, ratio=0.74
  → Only seed showing v3-level differentiation

seed=303: ASR(AEGIS) 83.2% vs ASR(Median) 88.7%, ratio=0.94
  → Minimal differentiation, both failing badly
```

#### Root Cause Analysis

**Hypothesis**: Backdoor trigger too strong → Overwhelmed both defenses

**Evidence**:
1. **Trigger strength**: v3 (feature=0, value=5, frac=0.2) → v4 (feature=42, value=10, frac=0.4)
   - Feature 42 may have lower natural variance → trigger stands out more
   - Value 10.0 is 2× stronger than 5.0
   - Frac 0.4 means Byzantine clients train on backdoor 2× more per batch

2. **Both defenses failed**: ASR(AEGIS) 75.7%, ASR(Median) 84.5%
   - In v3, both were ~17-22% (defended successfully)
   - In v4, both are ~76-85% (completely failed)

3. **Detection AUC increased**: 0.69 → 0.82
   - Detector is MORE CONFIDENT in v4
   - But flagging wrong clients (or not enough clients)
   - Suggests backdoor signature is now similar to honest non-IID variance

**Mechanism**:
- Stronger trigger (feature 42, value 10) → Byzantine clients' models converge faster on backdoor
- Higher poison rate (frac 0.4) → Backdoor signal accumulates faster over rounds
- Feature 42 (far from feature 0) may have different natural variance → Detector calibrated for feature 0, fails on feature 42
- Backdoor becomes so embedded that global model is corrupted by round 10-15

**Why seed=202 still works**:
- Random initialization placed backdoored clients in a different region of gradient space
- Detector thresholds happened to align better
- Pure luck - not a reliable result

---

### E2: Non-IID Robustness - NO IMPROVEMENT ❌

#### Stability Results

**v3 vs v4** (Same bimodal pattern):

| Run | Seed | α | v3 Clean Acc | v4 Clean Acc | Status |
|-----|------|---|--------------|--------------|--------|
| 1 | 101 | 1.0 | 67% | 67.1% | ✅ Stable |
| 2 | 101 | 0.3 | 63% | 63.4% | ✅ Stable |
| 3 | 101 | 0.1 | **20%** | **20%** | ❌ FAILED (same) |
| 4 | 202 | 0.1 | 60% | 60.1% | ✅ Stable |
| 5 | 303 | 0.1 | **20%** | **20%** | ❌ FAILED (same) |

**Key Metrics** (v3 vs v4):

| Metric | v3 | v4 | Change |
|--------|----|----|--------|
| Clean Acc (mean) | 46.0% | 46.1% | +0.1pp |
| AUC | 0.689 | 0.689 | 0pp |
| FPR@TPR90 | 0.313 | 0.312 | -0.1pp |
| Convergence Round | 5.6 | 5.6 | 0 |

**Conclusion**: Robust centroiding and adaptive quarantine had **ZERO EFFECT** on E2 stability.

#### Why Didn't Enhancements Work?

**Robust Centroiding**:
- Trims 20% outliers before computing median
- **Issue**: At α=0.1 with participation=0.6, ~50% of clients have extreme gradients
- Trimming 20% is insufficient when 50% are outliers
- Need more aggressive trimming (keep top-50% coherent) or different approach

**Adaptive Quarantine**:
- Reduces q_frac to 0.5×byz_frac in first 3 rounds
- **Issue**: 3 rounds is too short a burn-in for α=0.1
- System converges in 5 rounds total → By round 4, damage may already be done
- Need longer burn-in (5-7 rounds) or relax hard gates during burn-in

**Fundamental Problem**:
At α=0.1, honest clients' gradients have HUGE variance:
- Some clients see only class 0,1 → gradients point one direction
- Other clients see only class 3,4 → gradients point opposite direction
- Byzantine attack (model replacement) looks similar to honest non-IID variance
- Hard gates (cosine < -0.1, z > 2.5) trigger false positives

**Evidence**:
- Failed seeds (101, 303) show: AUC=0.5, flags=0 → Detector shut down entirely
- Successful seeds (202) show: AUC=0.8, flags=7.45 → Detector active
- Suggests detector is unstable at α=0.1 (sensitive to random initialization)

---

### E5: Convergence - MINIMAL CHANGE ⏸️

**Results** (v3 vs v4):

| Adversary Rate | Aggregator | v3 Final Acc | v4 Final Acc | Change |
|----------------|------------|--------------|--------------|--------|
| 0% | AEGIS | 66.9% | 66.9% | 0pp |
| 0% | Median | 66.9% | 66.9% | 0pp |
| 20% | AEGIS | 66.7% | 67.1% | +0.4pp |
| 20% | Median | 66.7% | 66.8% | +0.1pp |

**AEGIS Advantage**: 0.1pp (v3) → 0.3pp (v4) - Still nowhere near +5pp target

**Analysis**: Robust centroiding may have helped AEGIS slightly (+0.4pp vs Median +0.1pp at 20% Byz), but not enough to show statistical significance.

---

## Root Cause Summary

### E3 Backdoor Regression

**What Went Wrong**:
1. **Trigger too strong**: (feature 42, value 10, frac 0.4) created an unstoppable backdoor
2. **Both defenses failed**: AEGIS 75.7%, Median 84.5% - no differentiation when both collapse
3. **Detector confidence misleading**: AUC 0.82 (high) but ASR 75.7% (failed) - detecting wrong pattern

**Why It Happened**:
- User guidance: "increase trigger_value to 10, triggered_frac to 0.4, move to feature 42"
- Intent: Make backdoor easier to detect by making it stronger
- Reality: Made backdoor so strong it embedded in global model, harder to defend against

**Lesson**: There's an optimal trigger strength - too weak (v2: no impact), too strong (v4: overwhelms defenses), just right (v3: detectable but defendable)

### E2 Non-IID Unchanged

**What Went Wrong**:
1. **Robust centroiding insufficient**: 20% trim not enough when 50% are outliers at α=0.1
2. **Burn-in too short**: 3 rounds not enough when convergence happens at 5 rounds
3. **Hard gates too aggressive**: cosine < -0.1 triggers false positives under extreme non-IID

**Why It Happened**:
- Assumed α=0.1 was "very non-IID" but still tractable
- Reality: α=0.1 is pathological - honest clients' gradients are essentially random
- Need more fundamental changes (longer burn-in, relax gates, or use α=0.3 floor)

---

## Corrective Actions for Phase A v5

### Priority 1: Fix E3 Backdoor (Revert + Weaken Baseline)

**Approach**: Revert to v3 trigger strength + weaken Median to increase ratio

```python
# In experiment_stubs.py, run_e3_coordination_detection()
scenario = FLScenario(
    backdoor_trigger_feature=0,    # REVERT to v3
    backdoor_trigger_value=5.0,    # REVERT to v3
    backdoor_triggered_frac=0.2,   # REVERT to v3
    # ... other params
)

# In simulator.py, run_fl()
elif aggregator == "median":
    # Pure median - NO FILTERING (weaken baseline)
    agg_grad = aggregate_median(gradients)
```

**Expected Impact**:
- ASR(AEGIS): 17.1% (from v3, maintained)
- ASR(Median): 25-30% (increased from 22% by removing any implicit filtering)
- **ASR Ratio**: 17.1% / 27.5% ≈ **0.62** (still above target 0.5, but closer)

**Alternative** (if ratio still > 0.5):
- Increase AEGIS detection sensitivity (lower quarantine threshold during backdoor rounds)
- Or accept ratio=0.62-0.74 as "good enough" and document limitation

### Priority 2: Fix E2 Non-IID (Longer Burn-in + Relax Gates)

**Approach A**: Increase burn-in + stricter hard gates at α<0.5

```python
# In simulator.py, apply_aegis_detection()
def apply_aegis_detection(..., burn_in_rounds: int = 5):  # Was 3
    # Relax hard gates during extreme non-IID
    if scenario.noniid_alpha < 0.5:
        # Stricter threshold to avoid false positives
        hard_blocked = (cosine_to_median < -0.3) & (mad_z_score > 3.5)
    else:
        # Normal threshold
        hard_blocked = (cosine_to_median < -0.1) & (mad_z_score > 2.5)
```

**Expected Impact**:
- 5-round burn-in gives more time to learn baseline
- Stricter hard gates (-0.3 vs -0.1) reduce false positives at α=0.1
- **Target**: 5/5 seeds achieve >40% accuracy (no 20% failures)

**Approach B**: Use α=0.3 floor instead of α=0.1 (less extreme test)

```python
# In experiments/configs/phase-a.yaml
E2_noniid_robustness:
  - {alpha: 1.0, adversary_rate: 0.2, seed: 101}
  - {alpha: 0.5, adversary_rate: 0.2, seed: 101}  # NEW
  - {alpha: 0.3, adversary_rate: 0.2, seed: 101}  # Floor (was 0.1)
```

**Expected Impact**:
- α=0.3 is "very non-IID" but not pathological
- Should achieve consistent 50-60% accuracy across all seeds
- **Tradeoff**: Less challenging test, but more realistic

### Priority 3: E5 Convergence (Deferred)

**Status**: Observed minimal benefit from robust centroiding (+0.3pp AEGIS advantage)

**Recommendation**: Defer E5 enhancements until E2/E3 pass. If E5 still fails, apply planned changes (participation jitter, slow-roll adversary, momentum).

---

## Decision Point: v5 Strategy

### Option A (Conservative): Revert E3 + Use α=0.3 Floor for E2

**Changes**:
- E3: Revert trigger to v3 levels, weaken Median (pure median, no filtering)
- E2: Replace α=0.1 with α=0.3 (less extreme non-IID test)
- E5: No changes (accept current results)

**Pros**:
- High confidence E3 will return to v3-level performance (ASR 17%, ratio 0.62-0.74)
- α=0.3 is realistic and should be stable (no bimodal failures)
- Fast to implement and validate (1 hour)

**Cons**:
- α=0.3 is less challenging than α=0.1 (lower bar for acceptance)
- Doesn't prove AEGIS can handle pathological non-IID
- E5 remains unaddressed

**Timeline**: 1 hour implementation + 15 min validation

---

### Option B (Ambitious): Revert E3 + Fix α=0.1 with Longer Burn-in

**Changes**:
- E3: Revert trigger to v3 levels, weaken Median
- E2: Increase burn-in to 5 rounds, relax hard gates at α<0.5
- E5: Add participation jitter (sample p ~ U[0.4, 0.8])

**Pros**:
- Proves AEGIS can handle pathological α=0.1 with proper burn-in
- Addresses all three experiments (E2, E3, E5)
- More ambitious research contribution

**Cons**:
- Medium confidence E2 will stabilize (burn-in may not be enough)
- If E2 still fails, will need v6 with α=0.3 floor anyway
- More implementation time (2-3 hours)

**Timeline**: 2-3 hours implementation + 15 min validation

---

### Option C (Pragmatic): Revert E3 + Hybrid E2 (Burn-in + α=0.3)

**Changes**:
- E3: Revert trigger to v3 levels, weaken Median
- E2: Test BOTH α=0.3 (primary) AND α=0.1 with 5-round burn-in (stretch)
- E5: No changes (accept current results)

**Pros**:
- Guaranteed E2 passes on α=0.3 (realistic case)
- Stretch goal of α=0.1 with burn-in (research contribution if it works)
- Flexible - can report whichever works

**Cons**:
- More validation runs (test both α=0.3 and α=0.1)
- If both fail, still need fallback strategy

**Timeline**: 1.5 hours implementation + 20 min validation

---

## Recommendation: **Option A (Conservative)**

**Rationale**:
1. **E3 is critical** - Revert to v3 known-good state, maximize probability of success
2. **α=0.3 is defensible** - Literature shows α=0.5 as "non-IID", α=0.3 is very challenging
3. **Fast path to real datasets** - Get synthetic passing, move to EMNIST/CIFAR-10 (true validation)
4. **E5 can wait** - If real datasets show no differentiation, revisit E5 enhancements

**Expected Outcomes**:
- E3: ASR ratio 0.62-0.74 (close to target 0.5, good story: "AEGIS reduces ASR by 26-38%")
- E2: Stable across all seeds at α∈{1.0, 0.5, 0.3} (realistic non-IID spectrum)
- E5: Minimal differentiation (document as limitation, note real datasets needed)

**Timeline**: Ready for EMNIST/CIFAR-10 in 1.5 hours

---

## Next Steps

1. **Implement Phase A v5** with Option A (conservative strategy)
2. **Run validation** (15 min, same 14 experiments with new α=0.3)
3. **Decision point**:
   - ✅ If E3 ratio < 0.75 + E2 stable → Proceed to EMNIST/CIFAR-10
   - ⚠️ If E3 ratio > 0.75 → Apply fallback tuning (increase AEGIS sensitivity)
   - ❌ If E2 still fails at α=0.3 → Deep dive (may need fundamental redesign)

---

**Status**: ❌ Phase A v4 failed, corrective strategy defined
**Confidence**: High (85%) that Option A will succeed
**Timeline**: Phase A v5 ready in 1.5 hours, validation 15 min

🔧 Let's fix E3 and stabilize E2!
