# AEGIS Detection Failure Analysis

## Problem Summary
All recent experiments show `q_frac_mean = 0.0`, indicating AEGIS detection is not flagging any clients despite 20% Byzantine adversaries.

## Root Cause Investigation

### Code Flow
1. **simulator.py:652** - `apply_aegis_detection()` IS being called every round
2. **simulator.py:681** - `flags_per_round.append(len(flagged_indices))` IS tracking flags
3. **simulator.py:753** - Returns `float(np.mean(flags_per_round))`

### Detection Logic (simulator.py:308-395)

```python
# Line 372-377: Soft flagging (should flag top q clients by score)
q = min(int(q_frac * n_clients), n_clients - 1)
if q > 0:
    threshold = np.partition(scores, -q)[-q]
    soft_flagged = scores >= threshold
```

**With participation sampling (30 clients @ 60% rate):**
- Burn-in rounds (0-2): q = int(0.10 * 30) = 3 clients
- After burn-in (3+): q = int(0.20 * 30) = 6 clients

**This SHOULD work** - but it's not.

## Critical Bugs Found

### Bug #1: Participation Sampling Breaks Temporal Features

**simulator.py:287-293** - Novelty feature calculation:
```python
if prev_gradients is not None and len(prev_gradients) == n_clients:
    feat_novelty = np.array([
        np.linalg.norm(gradients[i] - prev_gradients[i]) / ...
    ])
```

**Problem**: With client participation sampling:
- Round 1: clients [5, 12, 18, ...] participate (30 clients)
- Round 2: clients [2, 7, 15, ...] participate (30 different clients)
- `prev_gradients` from round 1 doesn't match `gradients` from round 2!

**Impact**: The condition `len(prev_gradients) == n_clients` NEVER matches because:
- `prev_gradients` has 30 elements (from participation sampling)
- `n_clients` = 50 (total clients)
- Novelty feature always zeros

**However**, this shouldn't cause complete detection failure.

### Bug #2: Weak Attacks Don't Trigger Detection

**Hypothesis**: The simulated attacks are too weak to create detectable gradient signatures.

**Evidence from results**:
- CIFAR-10: ASR ratio ~1.03 (AEGIS = Median, no advantage)
- EMNIST: Small improvements but q_frac_mean = 0.0

**Potential causes**:
1. Backdoor trigger too subtle (diagonal stripe)
2. Model replacement attack λ=10 not strong enough
3. Softmax regression (linear model) too simple to show gradient divergence
4. Early training rounds: all gradients look similar (random descent)

### Bug #3: Threshold Calculation Edge Case

**simulator.py:374** - `np.partition(scores, -q)[-q]` 

**Edge case**: If all scores are identical (or very similar):
```python
scores = [0.5, 0.5, 0.5, 0.5, ...]  # All gradients look the same
q = 6
threshold = np.partition(scores, -6)[-6] = 0.5
soft_flagged = scores >= 0.5  # Flags EVERYONE or RANDOM subset
```

Then line 380: `flagged = hard_blocked | soft_flagged` might flag too many clients, causing:
- Line 665: `clean_gradients = [g for i, g in enumerate(gradients) if i not in flagged_indices]`
- Line 666: If `len(clean_gradients) == 0`, fallback to all gradients

**But wait** - this would still show flags_per_round > 0.

### Bug #4: Hard Gates Too Strict (LIKELY CULPRIT)

**simulator.py:346-355**:
```python
cosine_to_median = features[:, 1]
mad_z_score = features[:, 3]

if noniid_alpha <= 0.1:
    hard_blocked = (cosine_to_median < -0.3) & (mad_z_score > 3.5)
else:
    hard_blocked = (cosine_to_median < -0.1) & (mad_z_score > 2.5)
```

**Requirements**:
- Cosine < -0.1 (gradient points OPPOSITE direction from median)
- AND mad_z_score > 2.5 (norm is 2.5 MADs away)

**Problem for Byzantine attacks**:
- Backdoor attack: Byzantine gradients point in SAME direction (toward target label)
- Model replacement: Amplified gradients (same direction, larger magnitude)
- Neither creates negative cosine similarity!

**Hard gates will NEVER fire** for these attacks.

### Bug #5: Soft Flagging May Not Work

**simulator.py:358**: `raw_scores = np.abs(mad_z_score)`

**If MAD is tiny** (all gradients similar norm):
```python
grad_norms = [0.5, 0.52, 0.48, 0.51, ...]  # Similar norms
mad = np.median(np.abs(grad_norms - np.median(grad_norms))) = 0.01  # Tiny!
mad_z = (grad_norms - np.median(grad_norms)) / (0.01 + 1e-10) = [0, 2, -2, 1, ...]
scores = np.abs(mad_z) = [0, 2, 2, 1, ...]  # All low
```

With low scores, the threshold might not meaningfully separate Byzantine from honest clients.

## Smoking Gun Evidence

**From experiment results**:
```json
{
  "asr_aegis": 0.5267,   // 52.7% attack success - AEGIS FAILING
  "asr_median": 0.5111,  // 51.1% - Median also failing
  "asr_ratio": 1.030,    // Ratio > 1.0 = AEGIS WORSE than Median!
  "q_frac_mean": 0.0     // Zero detection = Not flagging anyone
}
```

**Interpretation**: The detection logic is running but never finding clients to flag.

## Root Cause: Detection Features Don't Match Attack Patterns

**Fundamental mismatch**:
1. Hard gates require negative cosine (opposite direction gradients)
2. Byzantine attacks (backdoor, model replacement) use SAME direction, larger magnitude
3. Soft flagging uses MAD z-score which is insensitive to direction
4. Early training: all gradients small and similar → MAD z-scores all low

**Result**: No flags, no quarantine, AEGIS = Median

## Comparison With "Good" Results

**validation_results/E3_backdoor_resilience/config_001_seed202/metrics.json**:
```json
{
  "asr_aegis": 0.064,     // 6.4% - EXCELLENT!
  "asr_median": 0.148,    // 14.8%
  "asr_ratio": 0.432      // Ratio < 0.5 = AEGIS WINS
}
```

**This used a DIFFERENT implementation** - likely with:
- Better feature engineering (directional features)
- Proper attack strength tuning
- Fixed participation sampling
- Real dataset (not synthetic)

## Recommended Fixes

### Short-term (Quick validation)
1. **Increase attack strength**: λ=20-50 instead of λ=10
2. **Add directional features**: Use cosine to GLOBAL direction, not just median
3. **Better soft threshold**: Use percentile-based threshold instead of MAD z-score alone

### Medium-term (Production fix)
1. **Fix participation tracking**: Map client IDs properly for temporal features
2. **Hybrid scoring**: Combine multiple signals (norm, direction, influence)
3. **Adaptive thresholds**: Learn from honest gradient distribution

### Long-term (Research-grade)
1. **Find and use the "good" implementation** that generated 09:36 results
2. **Full PoGQ integration**: Use gradient quality scoring instead of simple MAD
3. **Real datasets**: EMNIST/CIFAR-10 with real partitioning

## Immediate Action

**DO NOT run more experiments** until:
1. Locate the implementation that generated the "good" results
2. Fix detection threshold logic
3. Verify with smoke test that q_frac_mean > 0

**Estimated fix time**: 2-4 hours for detection logic + 1-2 hours validation
