# P1c + P1d Empirical Findings - Label Skew Tuning

**Date**: 2025-10-27  
**Test Iterations**: 3 (baseline, v1 [0.3, 0.8], v2 [-0.3, 0.95])

## Executive Summary

Implemented and tested two label skew fixes with mixed results:

- **P1c (Robust Anchor)**: Top-k reputation-weighted centroid
- **P1d (Two-Sided Cosine Guard)**: Range-based Byzantine detection

**Key Finding**: Threshold tuning provides **modest short-term improvement** but **reputation collapse** remains the fundamental blocker.

## Test Results Comparison

| Metric | Baseline | V1 [0.3, 0.8] | V2 [-0.3, 0.95] | Target |
|--------|----------|---------------|-----------------|--------|
| **Round 1-2 FP Rate** | 35-42% | 64-100% ❌ | 0-14% ✅ | ≤5% |
| **Round 3-10 FP Rate** | 28-42% | 92-100% ❌ | 28-71% ❌ | ≤5% |
| **Final Honest Rep** | 0.43 | 0.007 ❌ | 0.29 ❌ | ≥0.70 |
| **Detection Rate** | 66-100% | 100% ✅ | 83-100% ✅ | ≥95% |

### V2 Results (Best Attempt)
```
Round  | FP Rate | Honest Rep | Detection Rate
-------|---------|------------|---------------
1      | 0%      | 0.821      | 0% (too early)
2      | 14%     | 0.684      | 83% ✅
3      | 29%     | 0.529      | 100% ✅
4-6    | 57-64%  | 0.434-0.387| 100% ✅
7-10   | 64-71%  | 0.360-0.289| 100% ✅
```

## Root Cause Analysis

### Threshold Analysis

**Honest Node Cosine Distribution** (Round 0, baseline):
- Min: -0.207
- Median: -0.064
- Max: +0.999

**Byzantine Attack Patterns**:
- Noise: cos ≈ -1.0 (perfect negative)
- Sign-flip: cos ≈ 0.0-0.1 (orthogonal/weak positive)

**Threshold Effectiveness**:

| Threshold | Honest Protected | False Positives | Byzantine Caught |
|-----------|------------------|-----------------|------------------|
| [0.3, 0.8] | 43% (6/14) | 64-100% ❌ | 100% ✅ |
| [-0.3, 0.95] | 86% (12/14) | 14-71% ⚠️ | 83-100% ✅ |
| [-0.5, 0.99] | ~95% (estimated) | 10-60% (est) | 75-100% (est) |

**Conclusion**: **[-0.3, 0.95] is near-optimal** for threshold alone, but insufficient.

### Reputation Collapse Mechanism

**The Vicious Cycle**:
1. Round 1-2: Label skew causes **legitimate gradient diversity**
2. 2-4 honest nodes flagged (14-29% FP) due to low cosine or committee rejection
3. **Reputation decays** (×0.5 per detection): 1.0 → 0.5 → 0.25 → 0.125...
4. Once rep < 0.3, node is excluded even if behaving honestly
5. **No recovery mechanism**: Honest behavior doesn't restore reputation
6. Reputation collapse accelerates: 0.82 → 0.68 → 0.53 → 0.29 (by round 10)

**Evidence**:
- Rounds 4-10: Same 9-10 nodes repeatedly flagged
- No "rehabilitation" - once Byzantine, always Byzantine
- System can't distinguish:
  - "Node is Byzantine" (persistent bad actor)
  - "Node had different label distribution" (one-time misclassification)

## What We Learned

### ✅ P1c: Robust Anchor (Qualified Success)
**Implementation**: Top-3 reputation-weighted centroid

**Pro**:
- More stable than single node in later rounds (when reputations diverge)
- Reduces dependence on potentially-skewed Node 0

**Con**:
- In round 0, all reputations are 1.0 → simple averaging
- Centroid may be LESS representative than well-chosen single node
- Averaging "washes out" distinct gradient patterns

**Recommendation**: **Keep but modify**
- Use **median** instead of mean (robust to outliers)
- OR use **highest-reputation single node** (simpler, may work better)

### ⚠️ P1d: Two-Sided Cosine Guard (Partial Success)
**Implementation**: Range [-0.3, 0.95]

**Pro**:
- **14% FP in round 2** vs 42% baseline ✅
- Correctly protects honest diversity while catching extreme attacks
- Conceptually sound

**Con**:
- **FP grows to 71% by round 10** due to reputation collapse
- Can't compensate for broken reputation system
- Threshold alone is insufficient

**Recommendation**: **Keep [-0.3, 0.95]** but pair with reputation fixes

### ❌ Missing: P1e - Cumulative Tracking (CRITICAL)

**The Real Problem**:
Current system tracks **"Is node Byzantine RIGHT NOW?"**  
Should track: **"Has node EVER been reliably identified as Byzantine?"**

**Why Current Approach Fails**:
- False positive in round 2 → reputation×0.5
- Node behaves honestly rounds 3-10 → still flagged (rep too low)
- **No path to redemption**

**Proposed Fix** (from LABEL_SKEW_ANALYSIS_AND_RECOMMENDATIONS.md):
1. Track **cumulative detections** vs current state
2. Require **N consecutive** Byzantine rounds before permanent flagging (N=3?)
3. Allow **reputation recovery** for sustained honest behavior
4. Separate **"suspicious"** from **"confirmed Byzantine"**

## Recommendations

### Immediate Actions

**1. Revert P1c to simpler approach**:
```python
# Instead of top-k centroid:
anchor_idx = max(honest_candidates, key=lambda x: x[1])[0]  # Highest reputation
anchor_vector = flattened_gradients[anchor_idx]
```

**2. Keep P1d with [-0.3, 0.95]**:
- Empirically validated
- Protects 86% of honest nodes
- Catches extreme attacks

**3. Implement P1e: Reputation Recovery System** (CRITICAL):
```python
class ReputationSystem:
    def __init__(self):
        self.consecutive_honest = {}  # Track honest streaks
        self.consecutive_byzantine = {}  # Track Byzantine streaks
        
    def update(self, node_id, is_byzantine):
        if is_byzantine:
            self.consecutive_byzantine[node_id] = self.consecutive_byzantine.get(node_id, 0) + 1
            self.consecutive_honest[node_id] = 0
        else:
            self.consecutive_honest[node_id] = self.consecutive_honest.get(node_id, 0) + 1
            # Recovery: 3 consecutive honest rounds → restore some reputation
            if self.consecutive_honest[node_id] >= 3:
                self.reputation[node_id] = min(1.0, self.reputation[node_id] * 1.2)
```

### Success Criteria (Revised)

Given reputation collapse is the blocker, realistic targets:

| Metric | Baseline | Current Best | Achievable with P1e | Ultimate Goal |
|--------|----------|--------------|---------------------|---------------|
| Round 2-3 FP | 35-42% | 14-29% | **≤10%** | ≤5% |
| Round 10 FP | 28% | 71% | **≤20%** | ≤5% |
| Honest Rep (final) | 0.43 | 0.29 | **≥0.65** | ≥0.80 |
| Detection Rate | 66-100% | 83-100% | **≥90%** | ≥95% |

## Next Steps

1. **Simplify P1c**: Use single highest-reputation node (not centroid)
2. **Keep P1d**: Threshold [-0.3, 0.95] is validated
3. **Implement P1e**: Reputation recovery + cumulative tracking
4. **Test P1c+P1d+P1e**: Should hit achievable targets
5. **Then P1f**: Adaptive thresholds for final optimization

## Files

- **Test Results**:
  - Baseline: `results/label_skew_trace.jsonl` (28-42% FP)
  - V1: `results/label_skew_trace_fixed.jsonl` (64-100% FP ❌)
  - V2: `results/label_skew_trace_fixed_v2.jsonl` (14-71% FP ⚠️)

- **Code**: `tests/test_30_bft_validation.py`
  - Lines 724-750: P1c (robust anchor)
  - Lines 477-481, 800-807, 822-836, 862-870: P1d (two-sided guard)

- **Documentation**:
  - `LABEL_SKEW_DIAGNOSTIC_REPORT.md` - Initial analysis
  - `P1C_P1D_IMPLEMENTATION_COMPLETE.md` - Implementation details
  - `P1C_P1D_EMPIRICAL_FINDINGS.md` - This document

---

**Conclusion**: Threshold tuning alone provides **modest improvement** (42% → 14% FP in round 2) but **reputation collapse** drives FP back to 71% by round 10. **P1e (reputation recovery) is the critical missing piece**.
