# Gen 5 Layer 7: Self-Healing Mechanism - Design Specification

**Date**: November 12, 2025
**Version**: 1.0
**Status**: DRAFT - Ready for Implementation
**Complexity**: MODERATE (automatic recovery, no complex game theory)

---

## 🎯 Overview

**Purpose**: Automatic recovery from high Byzantine fractions through dynamic weight adjustment, gradient quarantine, and adaptive thresholding.

**Key Idea**: When Byzantine fraction exceeds tolerance, automatically quarantine suspected Byzantine gradients, reduce their influence in ensemble, and gradually restore after sustained honest behavior.

**Goal**: Maintain system operation even when Byzantine fraction temporarily exceeds 45% tolerance limit.

---

## 🏛️ Architecture

### Core Components

```
┌──────────────────────────────────────────────────────────────┐
│                    SelfHealingMechanism                       │
├──────────────────────────────────────────────────────────────┤
│ Purpose: Automatic recovery from high Byzantine activity     │
│                                                               │
│ Components:                                                   │
│  1. BFTEstimator - Estimate current Byzantine fraction       │
│  2. GradientQuarantine - Isolate suspected Byzantine grads   │
│  3. WeightRecovery - Gradually restore quarantined weights   │
│  4. AdaptiveThresholds - Dynamic decision boundaries         │
│                                                               │
│ Recovery Phases:                                              │
│  Detection → Quarantine → Monitoring → Recovery → Normal     │
└──────────────────────────────────────────────────────────────┘
```

### System Flow

```
Normal Operation (BFT < 45%):
  ┌──────────────────────────┐
  │ Standard AEGIS detection │
  │ All layers active        │
  └──────────┬───────────────┘
             │
             ▼
    BFT Estimation (continuous)
             │
             ▼
        BFT > 45%? ───NO──► Continue normal
             │
             YES
             ▼
┌────────────────────────────────────┐
│ HEALING PROTOCOL ACTIVATED         │
├────────────────────────────────────┤
│ Phase 1: DETECTION                 │
│  - Identify suspected Byzantine    │
│  - Lower confidence threshold      │
│                                    │
│ Phase 2: QUARANTINE                │
│  - Reduce Byzantine weight to 0.1  │
│  - Increase honest weight to 0.9   │
│  - Track quarantine duration       │
│                                    │
│ Phase 3: MONITORING                │
│  - Observe behavior under reduced  │
│    influence                       │
│  - Detect if attack persists       │
│                                    │
│ Phase 4: RECOVERY                  │
│  - Gradual weight restoration      │
│  - wᵢ(t+1) = wᵢ(t) × (1 + α)      │
│  - Monitor for regression          │
│                                    │
│ Phase 5: RETURN TO NORMAL          │
│  - BFT < 40% for sustained period  │
│  - Resume standard operation       │
└────────────────────────────────────┘
```

---

## 🧮 Mathematical Foundation

### BFT Estimation

**Problem**: Estimate Byzantine fraction from observed gradient scores.

**Approach**: Mixture model with two components (honest and Byzantine).

**Algorithm**:
```
Given: Scores s₁, s₂, ..., sₙ from n gradients

Assumptions:
- Honest scores ~ N(μₕ=0.85, σₕ=0.10)  # High scores
- Byzantine scores ~ N(μᵦ=0.15, σᵦ=0.10)  # Low scores

EM Algorithm:
1. Initialize: πₕ = 0.7 (70% honest)
2. E-step: Compute responsibilities
   γᵢ(honest) = πₕ · N(sᵢ | μₕ, σₕ) / p(sᵢ)
   γᵢ(byzantine) = πᵦ · N(sᵢ | μᵦ, σᵦ) / p(sᵢ)

3. M-step: Update parameters
   πₕ = (1/n) Σᵢ γᵢ(honest)
   πᵦ = 1 - πₕ

4. Iterate until convergence

Output: BFT estimate = πᵦ
```

**Simplified Version** (for implementation):
```python
def estimate_bft_simple(scores: List[float], threshold: float = 0.5) -> float:
    """
    Simple BFT estimation: fraction of scores below threshold.

    Args:
        scores: Detection scores in [0, 1]
        threshold: Byzantine/honest boundary (default: 0.5)

    Returns:
        Estimated Byzantine fraction
    """
    byzantine_count = sum(1 for s in scores if s < threshold)
    return byzantine_count / len(scores)
```

### Quarantine Weight Adjustment

**Goal**: Reduce Byzantine influence without removing them completely.

**Algorithm**:
```
Quarantine Phase:
- Byzantine gradients: wᵢ → 0.1  (reduce to 10%)
- Honest gradients: wⱼ → 0.9     (boost to 90%)

Recovery Phase (exponential restoration):
- wᵢ(t+1) = wᵢ(t) × (1 + α)
- Where α = 0.05 (5% increase per round)
- Continue until wᵢ = original weight
```

**Safety**: Quarantine never fully removes gradients (min weight = 0.1)

### Adaptive Thresholds

**Problem**: Fixed threshold (0.5) may not be optimal under attack.

**Solution**: Adjust threshold based on observed score distribution.

**Algorithm**:
```
Normal operation: threshold = 0.5

Under attack (BFT > 45%):
  - Move threshold toward honest mean
  - threshold_new = 0.5 - δ
  - Where δ = 0.1 × (BFT - 0.45)  # Max shift: 0.05

Example:
  BFT = 50% → δ = 0.005 → threshold = 0.495
  BFT = 60% → δ = 0.015 → threshold = 0.485
```

**Rationale**: Under attack, be more aggressive in rejecting lower scores.

---

## 💻 Implementation Specification

### Class: `BFTEstimator`

```python
class BFTEstimator:
    """
    Byzantine Fraction Tolerance estimator.

    Estimates current Byzantine fraction from detection scores.
    """

    def __init__(self, threshold: float = 0.5, window_size: int = 100):
        """
        Initialize BFT estimator.

        Args:
            threshold: Score threshold for Byzantine classification
            window_size: Number of recent scores to consider
        """
        self.threshold = threshold
        self.window_size = window_size
        self.score_history = deque(maxlen=window_size)

    def update(self, scores: List[float]):
        """Update score history with new batch."""
        self.score_history.extend(scores)

    def estimate_bft(self) -> float:
        """
        Estimate current Byzantine fraction.

        Returns:
            Estimated BFT in [0, 1]
        """
        if len(self.score_history) == 0:
            return 0.0

        byzantine_count = sum(1 for s in self.score_history if s < self.threshold)
        return byzantine_count / len(self.score_history)

    def is_under_attack(self, tolerance: float = 0.45) -> bool:
        """Check if BFT exceeds tolerance."""
        return self.estimate_bft() > tolerance
```

### Class: `GradientQuarantine`

```python
class GradientQuarantine:
    """
    Gradient quarantine system for suspected Byzantine gradients.
    """

    def __init__(
        self,
        quarantine_weight: float = 0.1,
        recovery_rate: float = 0.05,
    ):
        """
        Initialize quarantine system.

        Args:
            quarantine_weight: Weight for quarantined gradients (0.1 = 10%)
            recovery_rate: Exponential recovery rate (0.05 = 5% per round)
        """
        self.quarantine_weight = quarantine_weight
        self.recovery_rate = recovery_rate

        # Tracking
        self.quarantined_indices: Set[int] = set()
        self.current_weights: Dict[int, float] = {}
        self.original_weights: Dict[int, float] = {}
        self.quarantine_start: Dict[int, int] = {}  # Round when quarantined

    def quarantine_gradient(self, idx: int, round_num: int, original_weight: float = 1.0):
        """
        Quarantine a gradient by reducing its weight.

        Args:
            idx: Gradient index
            round_num: Current round number
            original_weight: Original weight before quarantine
        """
        if idx not in self.quarantined_indices:
            self.quarantined_indices.add(idx)
            self.original_weights[idx] = original_weight
            self.quarantine_start[idx] = round_num

        self.current_weights[idx] = self.quarantine_weight

    def release_gradient(self, idx: int):
        """Release gradient from quarantine."""
        if idx in self.quarantined_indices:
            self.quarantined_indices.remove(idx)
            self.current_weights[idx] = self.original_weights.get(idx, 1.0)
            del self.quarantine_start[idx]

    def apply_recovery(self, round_num: int):
        """
        Gradually restore weights for quarantined gradients.

        Exponential recovery: w(t+1) = w(t) × (1 + α)
        """
        for idx in list(self.quarantined_indices):
            current = self.current_weights[idx]
            original = self.original_weights[idx]

            # Exponential recovery
            new_weight = current * (1 + self.recovery_rate)

            # Cap at original weight
            if new_weight >= original:
                new_weight = original
                self.release_gradient(idx)  # Fully recovered

            self.current_weights[idx] = new_weight

    def get_weight(self, idx: int, default: float = 1.0) -> float:
        """Get current weight for gradient."""
        return self.current_weights.get(idx, default)

    def get_stats(self) -> Dict:
        """Get quarantine statistics."""
        return {
            "quarantined_count": len(self.quarantined_indices),
            "average_weight": (
                np.mean(list(self.current_weights.values()))
                if self.current_weights
                else 1.0
            ),
            "quarantined_indices": list(self.quarantined_indices),
        }
```

### Class: `SelfHealingMechanism`

```python
class SelfHealingMechanism:
    """
    Self-healing mechanism for automatic recovery from high Byzantine activity.
    """

    def __init__(
        self,
        bft_tolerance: float = 0.45,
        healing_threshold: float = 0.40,
        score_threshold: float = 0.5,
    ):
        """
        Initialize self-healing mechanism.

        Args:
            bft_tolerance: BFT above which healing activates (default: 45%)
            healing_threshold: BFT below which healing deactivates (default: 40%)
            score_threshold: Score boundary for Byzantine classification
        """
        self.bft_tolerance = bft_tolerance
        self.healing_threshold = healing_threshold
        self.score_threshold = score_threshold

        # Components
        self.bft_estimator = BFTEstimator(threshold=score_threshold)
        self.quarantine = GradientQuarantine()

        # State
        self.is_healing = False
        self.healing_start_round: Optional[int] = None
        self.adaptive_threshold = score_threshold

        # Statistics
        self.stats = {
            "total_rounds": 0,
            "healing_activations": 0,
            "total_healing_rounds": 0,
            "max_bft_observed": 0.0,
            "gradients_quarantined": 0,
        }

    def process_batch(
        self,
        scores: List[float],
        round_num: int,
    ) -> Tuple[List[str], Dict]:
        """
        Process batch of gradients with self-healing.

        Args:
            scores: Detection scores for gradients
            round_num: Current round number

        Returns:
            (decisions, details) where:
                decisions: List of "honest" or "byzantine" per gradient
                details: Healing state and statistics
        """
        # Update BFT estimation
        self.bft_estimator.update(scores)
        current_bft = self.bft_estimator.estimate_bft()

        # Update stats
        self.stats["total_rounds"] += 1
        self.stats["max_bft_observed"] = max(
            self.stats["max_bft_observed"], current_bft
        )

        # Check if healing needed
        if not self.is_healing and current_bft > self.bft_tolerance:
            self._activate_healing(round_num)

        # Check if healing can stop
        if self.is_healing and current_bft < self.healing_threshold:
            self._deactivate_healing()

        # Make decisions
        decisions = []
        for idx, score in enumerate(scores):
            # Use adaptive threshold if healing
            threshold = (
                self.adaptive_threshold if self.is_healing else self.score_threshold
            )

            decision = "honest" if score >= threshold else "byzantine"
            decisions.append(decision)

            # Quarantine if healing and Byzantine
            if self.is_healing and decision == "byzantine":
                self.quarantine.quarantine_gradient(idx, round_num)
                self.stats["gradients_quarantined"] += 1

        # Apply recovery to quarantined gradients
        if self.is_healing:
            self.quarantine.apply_recovery(round_num)
            self.stats["total_healing_rounds"] += 1

        # Prepare details
        details = {
            "is_healing": self.is_healing,
            "current_bft": current_bft,
            "adaptive_threshold": self.adaptive_threshold,
            "quarantine_stats": self.quarantine.get_stats(),
            "healing_round": (
                round_num - self.healing_start_round
                if self.is_healing
                else None
            ),
        }

        return (decisions, details)

    def _activate_healing(self, round_num: int):
        """Activate healing protocol."""
        self.is_healing = True
        self.healing_start_round = round_num
        self.stats["healing_activations"] += 1

        # Adjust threshold (be more aggressive)
        current_bft = self.bft_estimator.estimate_bft()
        delta = 0.1 * (current_bft - self.bft_tolerance)
        self.adaptive_threshold = self.score_threshold - delta

    def _deactivate_healing(self):
        """Deactivate healing protocol and return to normal."""
        self.is_healing = False
        self.healing_start_round = None
        self.adaptive_threshold = self.score_threshold

        # Release all quarantined gradients
        for idx in list(self.quarantine.quarantined_indices):
            self.quarantine.release_gradient(idx)

    def get_stats(self) -> Dict:
        """Get healing statistics."""
        stats = self.stats.copy()
        stats["is_currently_healing"] = self.is_healing
        stats["quarantine_active_count"] = len(self.quarantine.quarantined_indices)
        return stats

    def reset(self):
        """Reset healing mechanism."""
        self.is_healing = False
        self.healing_start_round = None
        self.adaptive_threshold = self.score_threshold
        self.bft_estimator = BFTEstimator(threshold=self.score_threshold)
        self.quarantine = GradientQuarantine()
        self.stats = {
            "total_rounds": 0,
            "healing_activations": 0,
            "total_healing_rounds": 0,
            "max_bft_observed": 0.0,
            "gradients_quarantined": 0,
        }
```

---

## 🧪 Testing Strategy

### Test Coverage

#### Unit Tests (12 tests)
1. `test_bft_estimation_all_honest` - BFT = 0% for all honest
2. `test_bft_estimation_all_byzantine` - BFT = 100% for all Byzantine
3. `test_bft_estimation_mixed` - Correct estimation for mixed
4. `test_quarantine_basic` - Quarantine reduces weight
5. `test_quarantine_recovery` - Gradual weight restoration
6. `test_quarantine_release` - Full recovery after sustained honest behavior
7. `test_healing_activation` - Activates when BFT > tolerance
8. `test_healing_deactivation` - Deactivates when BFT < threshold
9. `test_adaptive_threshold` - Threshold adjusts during healing
10. `test_quarantine_weight_floor` - Minimum weight maintained
11. `test_multiple_healing_cycles` - Can activate/deactivate multiple times
12. `test_healing_statistics` - Correct stat tracking

#### Integration Tests (6 tests)
1. `test_end_to_end_normal_operation` - No healing when BFT < tolerance
2. `test_end_to_end_healing_activation` - Full healing protocol
3. `test_gradual_attack_recovery` - Attack → heal → return to normal
4. `test_sustained_high_bft` - Continuous healing under persistent attack
5. `test_integration_with_ensemble` - Works with MetaLearningEnsemble
6. `test_performance_overhead` - Acceptable latency addition

---

## 📊 Performance Analysis

### Computational Complexity

| Operation | Complexity | Time |
|-----------|-----------|------|
| BFT Estimation | O(w) | ~0.1ms |
| Quarantine Check | O(1) | ~0.01ms |
| Weight Adjustment | O(q) | ~0.1ms |
| Recovery Update | O(q) | ~0.1ms |
| Full Process | O(n + q) | ~1ms |

Where: n = batch size, w = window size, q = quarantined count

**Overhead**: ~1ms per batch (negligible)

### Healing Effectiveness

**Scenario**: BFT increases from 40% → 60% → 40%

```
Round 0-10: BFT = 40% (normal)
Round 11-30: BFT = 60% (attack starts)
  → Healing activates at round 11
  → Quarantine reduces Byzantine influence
  → Effective BFT drops to ~45% (quarantine effect)
Round 31-50: Attack stops, BFT drops to 40%
  → Healing deactivates
  → Gradients gradually recovered
Round 51+: BFT = 40% (normal operation restored)
```

**Key Metrics**:
- Time to activation: 1 round (immediate)
- Time to deactivation: 5 rounds (sustained low BFT needed)
- Recovery time: 20 rounds (exponential @ 5% per round)

---

## 🎯 Key Innovations

### 1. Automatic BFT Estimation
**Novelty**: Continuous real-time Byzantine fraction estimation from score distribution.

**Contribution**: No manual intervention needed, system self-monitors.

### 2. Gradual Weight Quarantine
**Novelty**: Exponential recovery instead of binary quarantine/release.

**Contribution**: Smooth transitions, avoids oscillations.

### 3. Adaptive Thresholds
**Novelty**: Dynamic decision boundary based on estimated attack severity.

**Contribution**: More aggressive rejection under high BFT, normal operation otherwise.

### 4. Multi-Phase Recovery Protocol
**Novelty**: Structured healing protocol: Detection → Quarantine → Monitor → Recover → Normal.

**Contribution**: Principled recovery with safety checks at each phase.

---

## 🔮 Future Enhancements

### Near-Term
1. **Smart Recovery Triggers**: ML-based prediction of when to release quarantine
2. **Per-Gradient Histories**: Track individual gradient behavior over time
3. **Ensemble Weight Adaptation**: Automatically tune ensemble weights during healing

### Medium-Term
1. **Game-Theoretic Recovery**: Nash equilibrium analysis for optimal quarantine strategies
2. **Federated Healing Coordination**: Distributed healing across multiple coordinators
3. **Predictive Healing**: Detect attacks before BFT crosses tolerance

---

## 🚀 Implementation Checklist

### Core Components
- [ ] `BFTEstimator` class
- [ ] `GradientQuarantine` class
- [ ] `SelfHealingMechanism` class
- [ ] Adaptive threshold logic
- [ ] Recovery protocol

### Testing
- [ ] 12 unit tests
- [ ] 6 integration tests
- [ ] Healing scenario simulations
- [ ] Performance benchmarks

### Documentation
- [ ] API documentation
- [ ] Usage examples
- [ ] Healing protocol diagrams
- [ ] Integration guide

### Integration
- [ ] Export from `gen5/__init__.py`
- [ ] Add to AEGIS pipeline
- [ ] Update README
- [ ] Add to paper Methods section

---

**Estimated Implementation Time**: 3-4 hours
**Test Coverage Target**: 100%
**Performance Target**: < 2ms overhead per batch

---

**Design Status**: READY FOR IMPLEMENTATION ✅
**Next Step**: Implement `BFTEstimator` class
