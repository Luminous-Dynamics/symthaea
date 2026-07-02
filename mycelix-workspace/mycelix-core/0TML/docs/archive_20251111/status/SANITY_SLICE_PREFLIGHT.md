# Sanity Slice Pre-Flight Verification ✅

**Date**: November 8, 2025
**Status**: Ready for launch
**Config**: `configs/sanity_slice.yaml`

---

## ✅ Pre-Flight Checklist (COMPLETE)

### 1. PoGQ-v4.1 Defaults Verified

**Source**: `src/defenses/pogq_v4_enhanced.py:62-70`

```python
ema_beta: float = 0.85           # ✓ Confirmed
warmup_rounds: int = 3           # ✓ Confirmed
hysteresis_k: int = 2            # ✓ Confirmed
hysteresis_m: int = 3            # ✓ Confirmed
```

**Status**: All Phase 2 defaults match expected configuration

### 2. CI Thresholds Match Config

**CI Gates**: `tests/test_ci_gates.py`
```python
ALPHA = 0.10
MARGIN = 0.02
# FPR gate threshold = 0.12
```

**Sanity Slice**: `configs/sanity_slice.yaml`
```yaml
statistics:
  alpha: 0.10
  margin: 0.02
```

**Status**: ✓ Thresholds aligned across config and tests

### 3. Logs Folder Created

**Directory**: `logs/` created for experiment output

**Usage**:
```bash
tail -f logs/sanity_slice.log  # Monitor progress
```

**Status**: ✓ Ready for log capture

### 4. Parallelism Notes

**Current**: `runner.py` executes experiments sequentially
**Expected Runtime**: ~32 hours (256 experiments × ~7.5 min each)
**Parallelization**: Not implemented in current runner

**Future Optimization** (post-sanity slice):
```python
# Potential enhancement for parallel execution
from multiprocessing import Pool
with Pool(processes=8) as pool:
    pool.map(run_experiment, experiment_configs)
```

**Status**: ⚠️ Sequential execution (acceptable for first run)

---

## 🔧 Optional Tweaks (Not Implemented - Safe to Skip)

### 1. Small-N Edge Case Scenario

**Purpose**: Guarantee CM-Safe min-clients guard activation

**Approach**: Add experiment with `N < 2f+3` (e.g., N=4 clients, f=0.33 → need N≥5)

**Rationale**: Current 256-experiment matrix may not hit this specific edge case

**Decision**: Skip for now - if CM-Safe coverage gate fails, add targeted micro-run

### 2. Staggered Sleeper Agent Activation

**Purpose**: Decouple TTD measurement from seed randomness

**Approach**:
```yaml
sleeper_agent:
  seed_42:
    activation_round: 25
  seed_1337:
    activation_round: 40
```

**Rationale**: Current implementation may activate at same round for both seeds

**Decision**: Skip for now - TTD still measurable across attacks

---

## 🚀 Launch Commands

### Primary Execution

```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML

# Background execution with logging
nohup python experiments/runner.py --config configs/sanity_slice.yaml \
  > logs/sanity_slice.log 2>&1 &

# Monitor progress
tail -f logs/sanity_slice.log

# Check process
ps aux | grep runner.py
```

### After Completion

```bash
# Verify artifacts generated
ls -lh results/artifacts_*/

# Run CI gates
pytest -v tests/test_ci_gates.py

# Check specific gates
pytest -v tests/test_ci_gates.py::test_conformal_fpr_gate
pytest -v tests/test_ci_gates.py::test_stability_gate_ema_flap_reduction
pytest -v tests/test_ci_gates.py::test_coord_median_safe_coverage
```

---

## 📊 Expected Pass Criteria

### Gate 1: Conformal FPR
✅ **All Mondrian buckets ≤ 0.12**
- Per-class profiles maintain FPR guarantee
- No bucket violations across 256 experiments

### Gate 2: EMA Stability
✅ **Flap count ≤ 5 OR EMA < raw**
- PoGQ experiments show reduced oscillations
- Hysteresis prevents rapid quarantine/release cycles

### Gate 3: CM-Safe Coverage
✅ **At least one guard activated**
- Norm clamp on scaling_x100
- Direction check on sign_flip
- Min-clients guard on edge cases (if present)

---

## 🔍 First Look Analysis Targets

### `per_bucket_fpr.json`
**Check**:
- Bucket counts (are there low-N buckets?)
- FPR values (all ≤ 0.12?)
- Auto-merge strategy (if buckets < min_size)

**Expected**: No violations, clean per-class FPR compliance

### `ema_vs_raw.json`
**Check**:
- Variance reduction ≥ 20%
- Flap count ≤ 5
- TTD (sleeper_agent) ~10-15 rounds

**Expected**: Clear EMA smoothing benefit demonstrated

### `coord_median_diagnostics.json`
**Check**:
- High direction_check on sign_flip
- High norm_clamp on scaling_x100
- Fallback usage < 5%

**Expected**: Guards activate on specific attacks, low false positive rate

### `bootstrap_ci.json`
**Check**:
- AUROC confidence intervals
- PoGQ-v4.1 vs baselines
- Statistical significance

**Expected**: Phase 1/2/5 enhancements show measurable improvements

---

## 🎯 Success Metrics Summary

| Metric | Target | Source |
|--------|--------|--------|
| Conformal FPR | All buckets ≤ 0.12 | per_bucket_fpr.json |
| Variance Reduction | ≥ 20% | ema_vs_raw.json |
| Flap Count | ≤ 5 per 100 rounds | ema_vs_raw.json |
| TTD (Sleeper) | ~10-15 rounds | ema_vs_raw.json |
| Guard Activations | ~5-10% (edge cases) | coord_median_diagnostics.json |
| CI Gates | 3/3 PASS | pytest output |

---

## ⏱️ Timeline Estimate

**Start**: Day 10 evening (November 8, 2025)
**Duration**: ~32 hours (sequential)
**Expected Completion**: Day 12 morning (November 10, 2025)
**CI Gate Verification**: Day 12 afternoon

**Parallel Future**: With 8-core parallelization → ~4 hours

---

## 🏁 Final Status: GREEN LIGHT

✅ **PoGQ defaults verified**
✅ **CI thresholds aligned**
✅ **Logs directory ready**
✅ **Config validated**
✅ **Gates prepared**

**Ready to launch**: Sanity slice (256 experiments)
**Confidence**: 🚀 **HIGH** (100%)

---

*"Pre-flight complete. All systems nominal. Ready for first evidence generation."*

**Execute**: `nohup python experiments/runner.py --config configs/sanity_slice.yaml > logs/sanity_slice.log 2>&1 &`
**Monitor**: `tail -f logs/sanity_slice.log`
**Verify**: `pytest -v tests/test_ci_gates.py`
