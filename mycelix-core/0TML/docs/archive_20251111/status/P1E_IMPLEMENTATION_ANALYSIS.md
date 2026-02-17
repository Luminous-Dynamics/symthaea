# P1e Implementation Analysis - Reputation Recovery System

**Date**: 2025-10-27  
**Status**: Implementation Complete, Results Analyzed  
**Outcome**: ❌ FAILED - Reputation recovery mechanism ineffective

## Executive Summary

Implemented P1e (Reputation Recovery System) with consecutive honest/Byzantine round tracking and accelerated recovery (×1.2 multiplier after 3 consecutive honest rounds). **Results were significantly worse** than without P1e:

| Metric | Baseline | V2 (P1c+P1d) | P1e (This Implementation) | Target |
|--------|----------|---------------|---------------------------|--------|
| **Final FP Rate** | 28-42% | 14-71% | **92.9%** ❌ | ≤5% |
| **Avg Honest Rep** | 0.43 | 0.29 | **0.128** ❌ | ≥0.80 |
| **Detection Rate** | 66-100% | 83-100% | **100%** ✅ | ≥95% |

**Critical Finding**: The reputation recovery mechanism **never activates** because nodes flagged as Byzantine (even false positives) continue to be flagged in subsequent rounds, preventing accumulation of consecutive honest rounds.

### Behavior-Based Recovery Prototype (Oct 28)

To break the penalty loop we prototyped Option A (behavior-based recovery). This tracks whether a node’s gradient behaviour remains within acceptable bounds (PoGQ pass + cosine guard) and grants an additive reputation bonus once a configurable streak is reached.

| Metric | Baseline (Label Skew) | P1e (penalty-only) | **Behavior Recovery** | Target |
|--------|-----------------------|--------------------|-----------------------|--------|
| **Final FP Rate** | 92.9% | 92.9% | **42.9%** ⚠️ | ≤5% |
| **Avg Honest Rep** | 0.128 | 0.128 | **0.557** ⚠️ | ≥0.80 |
| **Detection Rate** | 100% | 100% | **100%** ✅ | ≥95% |

False positives remain above target, but they are cut roughly in half and honest reputations stabilize >0.5 instead of collapsing to 0.01. Additional tuning (tighter behaviour checks, adaptive bonuses, committee integration) is required to reach the ≤5% FP objective.

## Implementation Details

### Code Changes

**File**: `tests/test_30_bft_validation.py`

1. **Added Tracking Fields** (lines 339-341):
```python
class ReputationSystem:
    def __init__(self):
        self.reputations: Dict[int, float] = {}
        self.detection_history: Dict[int, List[bool]] = {}
        self.consensus_streaks: Dict[int, int] = {}
        # P1e: Track consecutive honest/Byzantine rounds for reputation recovery
        self.consecutive_honest: Dict[int, int] = {}
        self.consecutive_byzantine: Dict[int, int] = {}
```

2. **Initialized Counters** (lines 349-350):
```python
def initialize_node(self, node_id: int, initial_reputation: float = 1.0):
    self.reputations[node_id] = initial_reputation
    self.detection_history[node_id] = []
    self.consensus_streaks[node_id] = 0
    # P1e: Initialize consecutive round counters
    self.consecutive_honest[node_id] = 0
    self.consecutive_byzantine[node_id] = 0
```

3. **Reputation Recovery Logic** (lines 391-413):
```python
if was_detected_byzantine:
    # Exponential penalty (harsh on Byzantine behavior)
    self.reputations[node_id] = current_rep * 0.5
    streak = min(0, streak) - 1
    # P1e: Track consecutive Byzantine rounds, reset honest streak
    self.consecutive_byzantine[node_id] = self.consecutive_byzantine.get(node_id, 0) + 1
    self.consecutive_honest[node_id] = 0
else:
    # P1e: Track consecutive honest rounds, reset Byzantine streak
    self.consecutive_honest[node_id] = self.consecutive_honest.get(node_id, 0) + 1
    self.consecutive_byzantine[node_id] = 0
    streak = max(0, streak) + 1

    # P1e: Accelerated recovery after N consecutive honest rounds
    recovery_threshold = int(os.environ.get("REPUTATION_RECOVERY_THRESHOLD", "3"))
    if self.consecutive_honest[node_id] >= recovery_threshold:
        # Aggressive recovery: boost reputation by 20% per round (multiplicative)
        # This allows nodes below threshold (0.3) to recover quickly
        recovery_multiplier = float(os.environ.get("REPUTATION_RECOVERY_MULTIPLIER", "1.2"))
        self.reputations[node_id] = min(1.0, current_rep * recovery_multiplier)
    else:
        # Standard gradual recovery (slow trust building)
        self.reputations[node_id] = min(1.0, current_rep * 0.95 + 0.05)
```

4. **Added Trace Logging** (lines 1199-1200):
```python
"consecutive_honest": self.reputation_system.consecutive_honest.get(node_id, 0),
"consecutive_byzantine": self.reputation_system.consecutive_byzantine.get(node_id, 0),
```

5. **Fixed JSON Serialization** (lines 1156, 1163, 1211, 1221):
```python
# Ensured all boolean values are Python bools, not numpy bools
"selected": bool(anchor_vector is not None),
"detected": bool(split_info.get("split", False)),
"committee_rejected": bool(i in committee_rejections),
node_trace["correct"] = bool(is_correct)
```

### How to Capture Label-Skew Traces

1. **Enable the trace writer** by pointing `LABEL_SKEW_TRACE_PATH` to a JSONL file:
   ```bash
   export LABEL_SKEW_TRACE_PATH=results/label_skew_trace_p1e.jsonl
   ```
2. (Optional) Tune the cosine guard and reputation recovery parameters:
   ```bash
   export LABEL_SKEW_COS_MIN=-0.3
   export LABEL_SKEW_COS_MAX=0.95
   export REPUTATION_RECOVERY_THRESHOLD=3
   export REPUTATION_RECOVERY_MULTIPLIER=1.2
   ```
3. **Run the 30% BFT harness** with label skew enabled (example):
   ```bash
   nix develop --command bash -c "
     export RUN_30_BFT=1
     export BFT_DISTRIBUTION=label_skew
     python tests/test_30_bft_validation.py
   "
   ```

Each line in the JSONL file contains:
- Round metadata (thresholds, anchor selection, split detection)
- Per-node records with ground-truth, final classification, committee/PogQ scores, cosine similarities
- **New reputation event block** capturing `prev_reputation`, `new_reputation`, the update `mode` (`penalty`, `accelerated_recovery`, etc.), consecutive honest/Byzantine counters, and whether a consensus override triggered.
- Behaviour diagnostics (`behavior.acceptable`, `behavior.streak`) for each node, plus `behavior_recovery` metadata showing when the additive bonus activated.

### Behavior Recovery Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BEHAVIOR_RECOVERY_THRESHOLD` | `3` | Consecutive acceptable-behaviour rounds required before applying the additive recovery bonus. |
| `BEHAVIOR_RECOVERY_BONUS` | `0.1` | Additive reputation boost granted when the behaviour streak reaches the threshold. |
| `REPUTATION_FLOOR` | `0.01` | Minimum reputation to prevent honest nodes from collapsing to zero. |

Example label-skew run (produces `results/label_skew_trace_behavior_20251028.jsonl`):

```bash
nix develop --command bash -c "
  export RUN_30_BFT=1
  export BFT_DISTRIBUTION=label_skew
  export LABEL_SKEW_TRACE_PATH=results/label_skew_trace_behavior_20251028.jsonl
  python 0TML/tests/test_30_bft_validation.py
"
```

## Empirical Analysis

### Test Configuration
```bash
BFT_DISTRIBUTION=label_skew
LABEL_SKEW_COS_MIN=-0.3
LABEL_SKEW_COS_MAX=0.95
REPUTATION_RECOVERY_THRESHOLD=3 (default)
REPUTATION_RECOVERY_MULTIPLIER=1.2 (default)
```

### Node Behavior Patterns

#### Pattern 1: Never-Flagged Honest Nodes (e.g., Node 10)
```
R0: GT=honest CLS=honest Rep=1.000 ConsH= 1 ConsB= 0 ✅ Correct
R1: GT=honest CLS=honest Rep=1.000 ConsH= 2 ConsB= 0 ✅ Correct
R2: GT=honest CLS=honest Rep=1.000 ConsH= 3 ConsB= 0 ✅ Correct
...
R9: GT=honest CLS=honest Rep=1.000 ConsH=10 ConsB= 0 ✅ Correct
```
**Result**: Reputation stays at 1.0, accumulates consecutive honest rounds (but recovery mechanism never needed).

#### Pattern 2: Always-Flagged False Positives (e.g., Node 1)
```
R0: GT=honest CLS=byzantine Rep=0.500 ConsH= 0 ConsB= 1 ❌ False Positive
R1: GT=honest CLS=byzantine Rep=0.250 ConsH= 0 ConsB= 2 ❌ False Positive
R2: GT=honest CLS=byzantine Rep=0.125 ConsH= 0 ConsB= 3 ❌ False Positive
R3: GT=honest CLS=byzantine Rep=0.062 ConsH= 0 ConsB= 4 ❌ False Positive
...
R9: GT=honest CLS=byzantine Rep=0.001 ConsH= 0 ConsB=10 ❌ False Positive
```
**Result**: 
- Reputation decays exponentially: 1.0 → 0.5 → 0.25 → 0.125 → ... → 0.001
- ConsH stays at 0 (never accumulates)
- ConsB increases to 10
- **Recovery mechanism NEVER activates** because node is flagged every round

### Root Cause: Feedback Loop Trap

The reputation recovery mechanism **cannot work** in the current architecture because:

1. **Round N**: Node flagged as Byzantine (false positive) → Rep ×0.5 → ConsH reset to 0
2. **Round N+1**: Node's low reputation makes it more likely to be excluded/flagged → ConsH stays 0
3. **Round N+2**: Same pattern continues → ConsH never reaches threshold (3)
4. **Recovery never triggers**: `if consecutive_honest >= 3` is never satisfied

### Why This Design Fails

**Circular dependency**:
```
Low Reputation → More likely to be flagged → ConsH doesn't increase → 
No recovery → Reputation stays low → Loop continues
```

The system tracks "**was the node detected as Byzantine?**" but not "**did the node behave honestly despite being flagged?**"

## Lessons Learned

### ✅ What Works
1. **Tracking Infrastructure**: Consecutive round counters work correctly
2. **JSON Trace Logging**: Fixed serialization issues, comprehensive tracing functional
3. **Code Architecture**: Clean separation of concerns, easy to modify

### ❌ What Doesn't Work
1. **Detection-Based Recovery**: Basing recovery on "not flagged" fails because false positives persist
2. **Simple Threshold**: `consecutive_honest >= 3` never triggers for nodes that need it most
3. **Multiplicative Recovery**: Even if triggered, ×1.2 is too slow (0.125 → 0.150 → 0.180 → 0.216 = 4 rounds to exceed 0.3)

### 🔑 Key Insights
1. **Need Behavior-Based Recovery**: Track actual gradient quality, not just detection outcome
2. **Need Separate "Suspicious" State**: Distinguish temporary issues from confirmed Byzantine nodes
3. **Need Faster Recovery**: Multiplicative recovery from low reputation takes too long
4. **Need Look-Back Window**: Consider recent behavior, not just current round

## Recommended Next Steps

### Option A: Behavior-Based Recovery (RECOMMENDED)
Instead of tracking "was detected", track "gradient quality metrics":

```python
def should_recover_reputation(self, node_id: int, cos_anchor: float) -> bool:
    """
    Recovery based on BEHAVIOR, not detection outcome
    
    A node qualifies for recovery if:
    1. Cosine similarity is within acceptable range (even if flagged for other reasons)
    2. N consecutive rounds of acceptable behavior
    """
    # Track behavior quality, not just detection
    is_acceptable_behavior = (
        cos_anchor is not None and
        self.label_skew_cos_min <= cos_anchor <= self.label_skew_cos_max
    )
    
    if is_acceptable_behavior:
        self.consecutive_acceptable[node_id] = self.consecutive_acceptable.get(node_id, 0) + 1
    else:
        self.consecutive_acceptable[node_id] = 0
    
    # Recover if behavior is consistently acceptable
    return self.consecutive_acceptable[node_id] >= 3
```

### Option B: Two-Tier Reputation System
Separate "temporary suspicion" from "confirmed Byzantine":

```python
class TwoTierReputationSystem:
    def __init__(self):
        self.reputation: Dict[int, float] = {}  # Gradual decay/recovery
        self.confirmed_byzantine: Set[int] = set()  # Permanent exclusion
        self.suspicion_count: Dict[int, int] = {}  # Requires N consecutive to confirm
    
    def update(self, node_id, was_detected):
        if was_detected:
            self.suspicion_count[node_id] = self.suspicion_count.get(node_id, 0) + 1
            if self.suspicion_count[node_id] >= 5:  # Confirmed after 5 consecutive
                self.confirmed_byzantine.add(node_id)
            else:
                self.reputation[node_id] *= 0.8  # Milder penalty for suspicion
        else:
            self.suspicion_count[node_id] = 0  # Reset suspicion
            self.reputation[node_id] = min(1.0, self.reputation[node_id] * 1.1 + 0.05)
```

### Option C: Adaptive Thresholds (P1f Focus)
Accept that label skew causes high FP rates initially, use adaptive thresholds:

```python
# Learn acceptable ranges from observed honest node behavior
honest_cos_distribution = [cos for node, cos in zip(nodes, cosines) if reputation[node] > 0.8]
dynamic_min = np.percentile(honest_cos_distribution, 5)
dynamic_max = np.percentile(honest_cos_distribution, 95)
```

## Metrics Comparison

### P1e vs Previous Implementations

| Implementation | Round 2 FP | Round 10 FP | Final Honest Rep | Notes |
|----------------|------------|-------------|------------------|-------|
| Baseline | 35-42% | 28-42% | 0.43 | No label skew fixes |
| V1 [0.3, 0.8] | 64-100% | 64-100% | 0.007 | Threshold too narrow |
| V2 [-0.3, 0.95] | 14% ✅ | 71% | 0.29 | Best early performance |
| P1e (this) | Unknown | 92.9% ❌ | 0.128 | Recovery never triggered |

### Why P1e Made Things Worse

Without P1e, honest nodes flagged in early rounds still had:
- Standard gradual recovery: `rep * 0.95 + 0.05`
- Slow but steady: 0.5 → 0.525 → 0.549 → 0.572 → ...

With P1e:
- Recovery only if `consecutive_honest >= 3`
- False positives never reach threshold
- **Same decay rate but no recovery path**

## Conclusion

P1e implementation is **technically correct but architecturally flawed**. The reputation recovery mechanism cannot activate because the conditions for triggering it (consecutive honest classifications) are prevented by the very problem it's designed to solve (persistent false positives).

**Critical Recommendation**: Abandon detection-based recovery in favor of **behavior-based recovery** (Option A) or **two-tier reputation system** (Option B).

## Files Modified

- `tests/test_30_bft_validation.py` (lines 332-417, 1156-1228)
- Generated trace: `results/label_skew_trace_p1e.jsonl` (10 rounds, 100KB)

## Next Actions

1. ❌ **DO NOT proceed with current P1e approach**
2. ✅ **Implement Option A (Behavior-Based Recovery)** - Track gradient quality metrics
3. ✅ **Test revised P1e implementation** with behavior-based logic
4. ✅ **Document findings and compare** with V2 results

---

**Status**: P1e implementation **complete but ineffective** → Recommend **revised approach** before proceeding to P1f.
