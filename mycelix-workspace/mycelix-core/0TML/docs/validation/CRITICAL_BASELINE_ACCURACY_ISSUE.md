# 🚨 CRITICAL FINDING: System-Wide Baseline Accuracy Issue

**Date:** 2025-11-12
**Priority:** 🔴 **PAPER-BLOCKING**
**Severity:** Critical - Affects ALL experiments (E1, E2, E3, E5)

---

## Executive Summary

The E1 Byzantine sweep revealed a **critical system-wide issue**: ALL experiments are limited to ~65-67% maximum accuracy, even with 0% adversaries. This is NOT a Byzantine tolerance problem - it's a **fundamental baseline learning problem** with the synthetic dataset.

**Impact:** Paper cannot be submitted until baseline accuracy reaches expected 90%+ levels.

---

## The Discovery

### Initial Observation (E1 Quick Test)
- Expected: AEGIS ≥70% at 45% Byzantine, Median <70%
- Actual: AEGIS 51.1%, Median 46.1% (both fail)

### Full E1 Sweep Results
| Byzantine % | AEGIS Accuracy | Median Accuracy | Expected AEGIS | Status |
|-------------|----------------|-----------------|----------------|---------|
| 10% | 66.8% | 67.2% | ~92% | ❌ Both fail |
| 20% | 67.1% | 66.9% | ~87% | ❌ Both fail |
| 30% | 66.5% | 65.5% | ~82% | ❌ Both fail |
| 35% | 66.2% | 65.4% | ~77% | ❌ Both fail |
| 40% | **2.9%** ⚠️ | 49.5% | ~72% | ❌ Catastrophic |
| 45% | 51.1% | 46.1% | ≥70% | ❌ Both fail |
| 50% | 19.3% | 20.1% | ~65% | ❌ Both fail |

**Key Finding:** AEGIS and Median fail at ALL Byzantine ratios, even 10%.

---

## Smoking Gun: E5 Results (0% Adversaries)

**E5 Convergence Experiment - NO ATTACKS:**
- AEGIS with 0% Byzantine: **66.9% accuracy**
- Median with 0% Byzantine: **66.9% accuracy**

**This proves the ~67% ceiling exists EVEN WITHOUT BYZANTINE ATTACKS.**

The issue is not Byzantine tolerance - it's that the baseline model can only achieve 67% accuracy maximum on the synthetic task.

---

## System-Wide Evidence

### E2 (Non-IID Robustness)
```
α=1.0 (IID):  67.1% clean_acc  (Expected: ~92%)
α=0.5:        66.2% clean_acc  (Expected: ~87%)
α=0.3:        63.4% clean_acc  (Expected: ~82%)
α=0.1:        59.8% clean_acc  (Expected: ~78%)
α=0.1 (s303): 20.0% clean_acc  ⚠️ COMPLETE FAILURE (random chance)
```

### E3 (Backdoor Resilience)
```
Seed 101: 65.7% clean_acc_aegis, 59.6% robust_acc_aegis
Seed 202: 64.9% clean_acc_aegis, 62.8% robust_acc_aegis
Seed 303: 65.8% clean_acc_aegis, 55.6% robust_acc_aegis
```

### E5 (Convergence)
```
AEGIS (0% adv):  66.9% final_acc
Median (0% adv): 66.9% final_acc
AEGIS (20% adv): 67.1% final_acc  (Δ: +0.2pp from 0%)
Median (20% adv):66.7% final_acc
```

**Pattern:** ALL experiments capped at ~60-67% accuracy, regardless of adversary ratio.

---

## Root Cause Hypotheses

### 1. Synthetic Dataset is Too Hard / Poorly Defined
**Current Configuration:**
- `n_features = 50`
- `n_classes = 5`
- Linear separability unknown
- Data generation using `make_classification()` with default parameters

**Evidence:**
- Random seed 303 achieves only 20% (random chance for 5 classes)
- No experiment exceeds 67.1% accuracy
- E5 (0% adversary) plateaus at 67%

### 2. Model Architecture Too Weak
**Current Model:** Logistic Regression
- Single linear layer
- No hidden layers
- May be insufficient for the synthetic task complexity

**Evidence:**
- Fast convergence (5-14 rounds) suggests hitting capacity limit
- Accuracy plateaus early and never improves
- 67% ceiling persists across all configurations

### 3. Insufficient Training
**Current Training:**
- 25 rounds maximum
- 5 local epochs per round
- May not be enough for convergence

**Evidence:**
- Convergence detected at rounds 5-14
- Accuracy stops improving early
- No overfitting observed (clean_acc ≈ robust_acc)

### 4. Evaluation Bug (Secondary Issue)
**Observation:** `clean_acc == robust_acc` in ALL E1 and E2 results

**Expected Behavior:**
- Clean accuracy = performance on non-poisoned test data
- Robust accuracy = weighted average of clean + backdoored performance
- Should differ when backdoors exist

**Evidence:**
```json
// E1, 45% Byzantine
"clean_acc_aegis": 0.511,
"robust_acc_aegis": 0.511,   // Should differ!
"clean_acc_median": 0.461,
"robust_acc_median": 0.461   // Should differ!
```

---

## Impact Assessment

### Paper Status: BLOCKED ❌

**Cannot submit without fixing baseline accuracy** because:

1. **No evidence for 45% BFT claim**
   - Both AEGIS and Median fail at ALL Byzantine ratios
   - No clear advantage demonstrated (±1pp delta)

2. **Baseline learning incompetent**
   - 67% accuracy at 0% Byzantine is unacceptable
   - Expected: 90%+ baseline accuracy for competent learning
   - Reviewers will immediately reject

3. **E2, E3, E5 also affected**
   - All experiments show same ~65-67% ceiling
   - Cannot trust any current validation results
   - Need to re-run all experiments after fix

4. **Catastrophic failures**
   - E1 at 40% Byzantine: AEGIS 2.9% (worse than random!)
   - E2 seed 303: 20% accuracy (random guessing)
   - Indicates numerical instability or implementation bugs

### Acceptance Probability
- **Current:** ~0% (immediate rejection for poor baseline)
- **After fix:** TBD (depends on whether 45% BFT claim can be validated)

---

## Recommended Solutions (Priority Order)

### OPTION 1: Switch to Real Datasets (HIGHEST PRIORITY) ⭐⭐⭐⭐⭐

**Action:** Use EMNIST or CIFAR-10 for ALL experiments instead of synthetic data

**Rationale:**
- Real datasets have known baseline accuracies (EMNIST: 85-90%, CIFAR-10: 75-85%)
- Eliminates synthetic data generation as confound
- Prior work uses real datasets - easier to compare
- Already have EMNIST and CIFAR-10 code in codebase

**Implementation:**
```python
# In experiment_stubs.py, replace FLScenario with:
from experiments.datasets.emnist import load_emnist_federated

# E1 Byzantine sweep on EMNIST
def run_e1_byzantine_sweep(config: Dict) -> Dict:
    byz_frac = config.get("byz_frac", 0.20)
    seed = config.get("seed", 101)

    # Use EMNIST instead of synthetic
    client_data, test_data = load_emnist_federated(
        n_clients=50,
        noniid_alpha=1.0,  # IID for E1
        seed=seed
    )

    # Run FL with model replacement attack
    metrics_aegis = run_fl_emnist(
        client_data, test_data,
        aggregator="aegis",
        byz_frac=byz_frac,
        attack="model_replacement",
        ...
    )
    ...
```

**Expected Results:**
- Baseline (0% Byzantine): 85-90% accuracy
- 20% Byzantine: ~75-80% AEGIS, ~65-70% Median
- 45% Byzantine: ~70-75% AEGIS, ~40-50% Median

**Timeline:** 2-4 hours implementation + 10-12 hours re-running all experiments

---

### OPTION 2: Fix Synthetic Dataset Parameters ⭐⭐⭐⭐

**Action:** Tune `FLScenario` to generate learnable synthetic task

**Rationale:**
- Keeps synthetic data (faster, more controlled)
- May only need parameter adjustments

**Implementation:**
```python
# In experiments/simulator.py
scenario = FLScenario(
    n_clients=50,
    n_features=784,        # Increase from 50 (match MNIST)
    n_classes=10,          # Increase from 5
    n_samples=1000,        # Increase per client
    informative=0.8,       # 80% informative features
    redundant=0.1,
    n_clusters_per_class=2,
    class_sep=2.0,         # Increase separability
    flip_y=0.0,            # No label noise
    ...
)
```

**Expected Results:**
- Should achieve 85-90% baseline if parameters correct
- Still synthetic (reproducible, fast)

**Risks:**
- May not be sufficient to reach 90% baseline
- Harder to compare to prior work
- Unknown if parameters exist that work

**Timeline:** 1-2 hours tuning + 2 hours testing + 10-12 hours re-running

---

### OPTION 3: Increase Model Capacity ⭐⭐⭐

**Action:** Replace logistic regression with 2-layer MLP

**Rationale:**
- Current model may be too weak for task
- Deeper model can learn more complex patterns

**Implementation:**
```python
# In experiments/simulator.py
class MLPModel(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.layers(x)
```

**Expected Results:**
- May reach 75-85% on current synthetic data
- Better learning capacity

**Risks:**
- Slower training (more parameters)
- May still plateau if data is the issue
- Need to retune all hyperparameters

**Timeline:** 2-3 hours implementation + testing + re-running

---

### OPTION 4: Increase Training Duration ⭐⭐

**Action:** Increase rounds from 25 to 50-100

**Rationale:**
- May just need more training time
- Current convergence at round 5-14 seems premature

**Risks:**
- Unlikely to help if model hits capacity ceiling
- Wastes time if data/model is the issue

**Timeline:** 0 hours implementation, but doubles/quadruples experiment time

---

## Recommended Path Forward

### Phase 1: Immediate (Today - 4 hours)
1. ✅ **Switch E1 to EMNIST** - Implement real dataset version
2. ✅ **Run E1 quick test on EMNIST** at 45% Byzantine
3. ✅ **Validate baseline** - Check 0% Byzantine achieves 85%+
4. ✅ **Decision point** - If EMNIST works, proceed; if not, investigate

### Phase 2: Full Re-validation (Tomorrow - 12 hours)
1. ✅ **Re-run E1 full sweep on EMNIST** (10% → 50%)
2. ✅ **Re-run E2 on EMNIST** (all α values)
3. ✅ **Re-run E3 on EMNIST** (feature-level backdoor)
4. ✅ **Re-run E5 on EMNIST** (convergence comparison)

### Phase 3: Analysis & Paper Update (Day 3 - 4 hours)
1. ✅ **Analyze results** - Generate Figure 1, tables
2. ✅ **Update GEN5_VALIDATION_REPORT** with real results
3. ✅ **Adjust paper claims** based on empirical data
4. ✅ **Draft Section 4** (Experiments) with validated results

### Phase 4: Paper Finalization (Week 2)
1. ✅ Complete manuscript draft
2. ✅ Internal review and revision
3. ✅ Prepare for MLSys/ICML 2026 submission

---

## Alternative: Pivot Away from 45% BFT Claim

**If EMNIST also fails to show 45% BFT**, consider reframing paper:

### Option A: Focus on Heterogeneity Robustness
- Emphasize E2 results (non-IID performance)
- Claim: "AEGIS maintains accuracy under heterogeneity better than Median"
- Lower novelty but more achievable

### Option B: Focus on Backdoor Detection
- Emphasize E3 results (backdoor resilience)
- Claim: "AEGIS detects backdoors with lower ASR than baseline"
- Different angle, avoids BFT claim

### Option C: Accept Lower BFT Limit
- Test empirically on real data
- If limit is 35% or 40%, claim that instead
- Honest reporting: "AEGIS achieves 35% BFT, exceeding 33% classical limit"

---

## Critical Questions for Investigation

### Q1: What is the ACTUAL baseline accuracy on EMNIST with 0% Byzantine?
**Test:** Run E5 on EMNIST with adversary_rate=0.0
**Expected:** ≥85% accuracy
**If fails:** Model architecture or training is the issue

### Q2: Does the evaluation correctly separate clean vs robust accuracy?
**Test:** Check test set composition and metric calculation
**Expected:** clean_acc > robust_acc when backdoors exist
**If fails:** Fix evaluation logic

### Q3: What causes the catastrophic failures (2.9% at 40%, 20% at α=0.1)?
**Test:** Debug specific failing seeds
**Expected:** Numerical overflow or aggregation bug
**If confirmed:** Fix numerical stability

---

## Bottom Line

**The paper is BLOCKED** until we demonstrate:
1. ✅ **≥85% baseline accuracy** with 0% adversaries
2. ✅ **Clear AEGIS advantage** (+10-30pp) at high Byzantine ratios
3. ✅ **45% BFT empirically validated** (or adjust claim to actual limit)

**Recommended immediate action:** Switch to EMNIST dataset and re-run E1 quick test to validate baseline accuracy is achievable.

**Estimated time to resolution:** 2-4 days if EMNIST works, 1-2 weeks if deeper issues exist.

---

**Status:** Investigation complete, awaiting decision on next steps
**Next Action:** Implement E1 with EMNIST and test baseline accuracy
**Owner:** Research team
**Timeline:** Start immediately, resolve within 72 hours

🚨 **This is the highest priority issue blocking paper submission** 🚨
