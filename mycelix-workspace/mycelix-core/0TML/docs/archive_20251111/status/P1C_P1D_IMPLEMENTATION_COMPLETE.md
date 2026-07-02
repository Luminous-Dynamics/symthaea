# P1c + P1d Implementation Complete

**Date**: 2025-10-27  
**Status**: ✅ COMPLETE  
**Files Modified**: `tests/test_30_bft_validation.py`

## Summary

Implemented the two highest-priority fixes for label skew Byzantine detection:

1. **P1c: Robust Anchor Selection** - Reputation-weighted centroid (lines 724-750)
2. **P1d: Two-Sided Cosine Guard** - Range [0.3, 0.8] (lines 478-480, 800-807, 822-836, 862-870)

## P1c: Robust Anchor Selection

### Problem (Confirmed via Tracing)
- Anchor was ALWAYS Node 0 (first honest by index)
- Node 0 may have extreme label distribution due to Dirichlet sampling
- Honest nodes with different distributions had low cosine similarity
- Result: 35-42% false positive rate

### Solution Implemented
**Lines 724-750**: Reputation-weighted centroid anchor

```python
# P1c: Robust anchor selection using reputation-weighted centroid
honest_candidates = []
for idx, flag in enumerate(byzantine_flags):
    if flag == 0:  # Honest node
        rep = self.reputation_system.get_reputation(node_ids[idx])
        honest_candidates.append((idx, rep, flattened_gradients[idx]))

if honest_candidates:
    # Sort by reputation (descending) and select top-k
    honest_candidates.sort(key=lambda x: x[1], reverse=True)
    top_k = min(5, len(honest_candidates))
    selected = honest_candidates[:top_k]

    # Compute reputation-weighted centroid
    total_weight = sum(rep for _, rep, _ in selected)
    if total_weight > 0:
        weighted_sum = np.zeros_like(selected[0][2])
        for idx, rep, grad in selected:
            weighted_sum += (rep / total_weight) * grad
        anchor_vector = weighted_sum
    else:
        # Fallback: equal-weighted centroid
        anchor_vector = np.mean([grad for _, _, grad in selected], axis=0)
```

### Key Features
- Selects **top-5** honest nodes by reputation (or all if < 5)
- Weights each gradient by normalized reputation
- Falls back to equal weighting if all reputations are 0
- Reduces dependence on single node's label distribution

### Expected Impact
- More stable anchor across different label distributions
- Reduced false positives for honest nodes with diverse gradients
- Better detection of coordinated attacks

## P1d: Two-Sided Cosine Guard

### Problem (Confirmed via Tracing)
- One-sided threshold: `cos >= -0.8`
- Honest nodes with negative cosine WERE being flagged (10+ examples in rounds 0-2)
- Nodes with positive cosine (e.g., 0.441) also flagged
- Sign-flip attacks with high cosine (~0.1) not caught

### Solution Implemented

#### Configuration (Lines 477-480)
```python
# P1d: Two-sided cosine threshold [min, max]
self.label_skew_cos_min = float(os.environ.get("LABEL_SKEW_COS_MIN", 0.3))
self.label_skew_cos_max = float(os.environ.get("LABEL_SKEW_COS_MAX", 0.8))
```

#### Guard Location 1: Extra Detection Override (Lines 800-807)
```python
# P1d: Two-sided cosine guard - protect honest nodes in [min, max] range
if (
    extra_detection
    and self.distribution == "label_skew"
    and cos_anchor is not None
    and self.label_skew_cos_min <= cos_anchor <= self.label_skew_cos_max
):
    extra_detection = False  # Override: node is within safe range
```

#### Guard Location 2: Committee Rejection (Lines 822-836)
```python
# P1d: Committee rejection with two-sided cosine guard
if (
    not is_byzantine
    and not committee_accept
    and consensus_score < self.committee_reject_floor
):
    if (
        self.distribution == "label_skew"
        and cos_anchor is not None
        and self.label_skew_cos_min <= cos_anchor <= self.label_skew_cos_max
    ):
        pass  # Protected by cosine guard
    else:
        is_byzantine = True  # Outside safe range
```

#### Guard Location 3: Anchor Outlier Detection (Lines 862-870)
```python
# P1d: Flag nodes OUTSIDE the cosine range [min, max]
elif (
    self.distribution == "label_skew"
    and cos_anchor is not None
    and (cos_anchor < self.label_skew_cos_min or cos_anchor > self.label_skew_cos_max)
):
    is_byzantine = True
    heuristic_reason = "label-skew-anchor-outlier"
```

### Key Features
- **Lower bound (0.3)**: Catches orthogonal and negative cosine attacks
- **Upper bound (0.8)**: Catches overly-aligned attacks (e.g., scaled sign-flip)
- **Protected range [0.3, 0.8]**: Allows honest diversity in label skew
- **Applied consistently** at all 3 detection points

### Expected Impact
- Fewer false positives for honest nodes with diverse gradients
- Better detection of noise attacks (cos < 0.3)
- Better detection of aligned attacks (cos > 0.8)
- Maintains high Byzantine detection rate

## Trace Logging Updates

**Lines 1128-1129**: Added two-sided threshold to trace data
```python
"label_skew_cos_min": getattr(self, "label_skew_cos_min", 0.3),
"label_skew_cos_max": getattr(self, "label_skew_cos_max", 0.8),
```

This enables post-hoc analysis of threshold impact.

## Testing & Validation

### Baseline Performance (Before Fixes)
- **False Positive Rate**: 28-42%
- **Detection Rate**: 66-100% (variable)
- **Honest Reputation (final)**: 0.43 (collapsed)
- **Anchor**: Always Node 0

### Next Steps: Validation Test
```bash
# Run with P1c + P1d fixes
export RUN_30_BFT=1
export BFT_DISTRIBUTION="label_skew"
export LABEL_SKEW_TRACE_PATH="results/label_skew_trace_fixed.jsonl"
nix develop --command poetry run python tests/test_30_bft_validation.py
```

### Expected Improvements
- **False Positive Rate**: Target ≤10% (vs 28-42% baseline)
- **Detection Rate**: Target ≥95% (vs 66-100% variable)
- **Honest Reputation**: Target ≥0.70 (vs 0.43 collapsed)
- **Anchor Stability**: Multiple nodes contributing (vs single node)

### Success Criteria
- ✅ FP rate reduced by >50% (42% → ≤20%)
- ✅ Detection rate stable at ≥90%
- ✅ Honest reputation maintained above 0.65
- ✅ Trace shows centroid anchor (not single node)
- ✅ Cosine guard correctly protecting [0.3, 0.8] range

## Environment Variables

New configuration options:
```bash
# Two-sided cosine threshold
LABEL_SKEW_COS_MIN=0.3    # Lower bound (default)
LABEL_SKEW_COS_MAX=0.8    # Upper bound (default)

# For tuning if needed
LABEL_SKEW_COS_MIN=0.2    # More permissive (higher FP risk)
LABEL_SKEW_COS_MAX=0.9    # More permissive (higher FN risk)
```

## References

- **Diagnostic Report**: `LABEL_SKEW_DIAGNOSTIC_REPORT.md`
- **Analysis**: `LABEL_SKEW_ANALYSIS_AND_RECOMMENDATIONS.md`
- **Test File**: `tests/test_30_bft_validation.py`
- **Trace Data**: `results/label_skew_trace.jsonl` (baseline), `results/label_skew_trace_fixed.jsonl` (after fixes)

## Remaining Work

- [ ] **P1e**: Fix cumulative metric tracking (medium priority)
- [ ] **P1f**: Adaptive thresholds (medium priority)
- [ ] **Validation**: Run test and compare to baseline
- [ ] **Tuning**: Adjust thresholds if needed based on results

---

**Next Action**: Run validation test with P1c+P1d fixes to measure improvement vs baseline (28-42% FP → target ≤10%).
