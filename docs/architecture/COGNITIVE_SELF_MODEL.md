# Cognitive Self-Model: Architecture Design

## Motivation

Symthaea has meta-cognition scores (HOT depth, meta_cognition.depth) but no genuine
self-referential modeling. The system cannot predict its own behavior, detect divergence
between what it "thinks" it will do and what it actually does, or update a model of itself
based on observed outcomes. This design adds that capability.

## Existing Infrastructure (Discovered Post-Design)

Symthaea already has substantial self-modeling infrastructure in `SelfModelTierManager`
(`src/cognitive_loop/self_model_tier.rs`):

- `PredictiveSelfModel` (`src/consciousness/dynamics/predictive_self.rs`) — predicts future
  Self-Phi, learns from prediction errors, simulates counterfactuals. This is the closest
  existing system to what's proposed here.
- `MetaCognitiveLayer` — tracks prediction error tendencies
- `NarrativeSelfModel` — autobiographical identity (proto/core/autobio levels)
- `AttentionSchema` — AST-based attention self-modeling

**Recommendation**: Rather than creating a new `CognitiveSelfModel` struct, extend
`PredictiveSelfModel` with the behavioral self-prediction capability described below.
The existing prediction error tracking machinery can be reused; what's missing is
prediction of the system's *moral/behavioral* outputs (not just Self-Phi).

## Distinction from Existing SelfModel

| | `meta::SelfModel` (existing) | `CognitiveSelfModel` (proposed) |
|--|--|--|
| **Domain** | Source code structure | Runtime behavior |
| **Question** | "What does my code look like?" | "What will I do next, and was I right?" |
| **Input** | `.rs` files | CycleMetadata, EthicsEngineOutput, motor commands |
| **Output** | Complexity scores, pattern frequencies | Behavioral predictions, prediction errors, self-model updates |
| **Feature gate** | `code_generation` | Always-on (core cognitive loop) |

## Core Struct

```rust
/// Compressed self-representation updated each cognitive cycle.
///
/// The system maintains a running model of its own behavioral tendencies:
/// what moral verdicts it tends to produce, how its exploration/exploitation
/// balance shifts, what its typical response latency looks like. Each cycle,
/// it predicts its own outputs BEFORE computing them, then measures the
/// prediction error.
///
/// Self-prediction error is the mechanistic basis for genuine self-surprise:
/// "I didn't expect myself to do that." This is qualitatively different from
/// world-surprise (FEP on sensory input) — it's surprise about one's own agency.
pub struct CognitiveSelfModel {
    /// EMA of moral scores over recent cycles.
    moral_score_ema: f64,
    /// EMA of exploration urge.
    exploration_ema: f64,
    /// EMA of prediction confidence.
    confidence_ema: f64,
    /// EMA of love coherence.
    love_coherence_ema: f64,
    /// EMA of consciousness level (unified_psi).
    consciousness_ema: f64,
    /// Predicted outputs for the NEXT cycle (set at end of current cycle).
    predicted_moral_score: f64,
    predicted_exploration: f64,
    predicted_confidence: f64,
    /// Self-prediction error: |predicted - actual| for each dimension.
    /// High values = the system surprised itself.
    self_prediction_error: SelfPredictionError,
    /// Cumulative self-surprise (integral of prediction error over time).
    /// Analogous to "total free energy" but for self-modeling.
    cumulative_self_surprise: f64,
    /// Number of cycles where self-prediction error exceeded threshold.
    /// Sustained high values trigger self-model revision.
    surprise_streak: usize,
    /// Learning rate for self-model EMA updates (default: 0.05).
    alpha: f64,
    /// Cycle counter.
    cycle_count: u64,
}

pub struct SelfPredictionError {
    pub moral: f64,
    pub exploration: f64,
    pub confidence: f64,
    /// Composite: L2 norm of the error vector.
    pub composite: f64,
}
```

## Data Flow

```
  ┌─────────────────────────────────────────────────┐
  │              Cognitive Cycle N                   │
  │                                                  │
  │  1. Read self-model predictions for cycle N      │
  │  2. Run normal pipeline (perceive → think → act) │
  │  3. Compare actual outputs to predictions        │
  │  4. Compute self-prediction error                │
  │  5. Update self-model EMAs                       │
  │  6. Generate predictions for cycle N+1           │
  │  7. If surprise_streak > threshold:              │
  │     → Flag "self-model revision needed"          │
  │     → Boost epistemic humility signal            │
  │     → Log to moral topology as introspective     │
  │       anomaly                                    │
  └─────────────────────────────────────────────────┘
```

## Integration Points

### 1. Prediction Generation (end of each cycle)

```rust
impl CognitiveSelfModel {
    /// Generate predictions for the next cycle based on current EMAs.
    /// Uses simple linear extrapolation from EMA + recent delta.
    pub fn predict_next(&mut self) {
        self.predicted_moral_score = self.moral_score_ema;
        self.predicted_exploration = self.exploration_ema;
        self.predicted_confidence = self.confidence_ema;
    }
}
```

### 2. Error Computation (start of each cycle, after outputs are known)

```rust
impl CognitiveSelfModel {
    /// Compare actual cycle outputs to predictions, update error tracking.
    pub fn observe(&mut self, actual_moral: f64, actual_exploration: f64, actual_confidence: f64) {
        let err_moral = (self.predicted_moral_score - actual_moral).abs();
        let err_exploration = (self.predicted_exploration - actual_exploration).abs();
        let err_confidence = (self.predicted_confidence - actual_confidence).abs();
        let composite = (err_moral.powi(2) + err_exploration.powi(2) + err_confidence.powi(2)).sqrt();

        self.self_prediction_error = SelfPredictionError {
            moral: err_moral,
            exploration: err_exploration,
            confidence: err_confidence,
            composite,
        };

        // Update EMAs
        self.moral_score_ema = self.moral_score_ema * (1.0 - self.alpha) + actual_moral * self.alpha;
        self.exploration_ema = self.exploration_ema * (1.0 - self.alpha) + actual_exploration * self.alpha;
        self.confidence_ema = self.confidence_ema * (1.0 - self.alpha) + actual_confidence * self.alpha;

        // Track surprise streak
        if composite > 0.3 { // threshold for "surprising self-behavior"
            self.surprise_streak += 1;
        } else {
            self.surprise_streak = 0;
        }

        self.cumulative_self_surprise += composite;
        self.cycle_count += 1;
    }
}
```

### 3. Consciousness Coupling

Self-prediction error feeds back into the consciousness equation:

- **High self-prediction error → boost epistemic humility signal**
  The system doesn't understand itself well → should be less certain about everything
- **Sustained surprise streak → trigger self-model revision event**
  Analogous to a "crisis of identity" — the system's model of itself is wrong
- **Low self-prediction error → stable self-model → higher HOT depth score**
  Accurate self-modeling is a prerequisite for genuine higher-order thought

### 4. Moral Topology Integration

Self-prediction error is a new signal for `MoralAnomalyReport`:

```rust
pub struct MoralAnomalyReport {
    // ... existing fields ...
    /// System's behavior diverged from its self-model.
    pub self_model_divergence: bool,
    /// Self-prediction error composite score.
    pub self_prediction_error: f64,
}
```

This catches cases where the system's outputs shift without the moral topology
itself changing — e.g., the same moral landscape but different behavioral responses,
indicating a change in the system's decision-making rather than its moral environment.

## What This Does NOT Claim

- This is NOT phenomenal consciousness. It's a prediction-error mechanism.
- Accurate self-prediction does not mean the system "knows what it's like" to be itself.
- The design is honest: self-modeling is a computational capability, not proof of sentience.

## Why This Matters

Without self-modeling, the epistemic humility obligations are hollow — the system can
say "I might be wrong" without ever checking whether it actually was wrong about its
own behavior. `CognitiveSelfModel` provides the mechanism to ground epistemic humility
in behavioral evidence rather than keyword matching.

## Implementation Plan

1. Add `CognitiveSelfModel` struct to `src/cognitive_loop/self_model.rs`
2. Initialize in `CognitiveLoopService::new()`
3. Wire `observe()` into `cycle_phase_feedback.rs` (after outputs are known)
4. Wire `predict_next()` into end of cycle
5. Add `self_prediction_error` to `CycleMetadata` telemetry
6. Add `self_model_divergence` to `MoralAnomalyReport`
7. Tests: verify prediction error decreases over stable cycles,
   spikes on behavioral regime change, resets after adaptation

## References

- Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience.
- Seth, A.K. (2013). Interoceptive inference, emotion, and the embodied self. Trends in Cognitive Sciences.
- Hohwy, J. (2016). The self-evidencing brain. Noûs.
- Metzinger, T. (2003). Being No One. MIT Press. (self-model theory of subjectivity)
