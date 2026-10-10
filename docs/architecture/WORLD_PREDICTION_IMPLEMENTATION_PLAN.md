# World Prediction + Calibration Pipeline Implementation Plan

**Priority**: 1 (Critical - Truth Backbone)
**Status**: Phase 1-3.5 IMPLEMENTED
**Estimated Complexity**: Medium-High
**Key Insight**: "The system predicts itself but not the world"
**Related**: `MAGI_LOOP_SPECIFICATION.md` (AGI Crossing Criterion)
**Implementation Date**: 2026-01-20

### Implementation Status

| Phase | Status | Files Created |
|-------|--------|---------------|
| Phase 1: WorldPrediction | COMPLETE | `world_prediction.rs` |
| Phase 1.5: ResolutionContract | COMPLETE | `world_prediction.rs` |
| Phase 2: BrierScoreTracker | COMPLETE | `calibration.rs` |
| Phase 3.5: ConstraintGate | COMPLETE | `constraint_gate.rs` |
| Integration | COMPLETE | `magi_integration.rs` |
| Phase 3: EFE Integration | PENDING | - |
| Phase 4: CausalAttribution Integration | PENDING | - |

---

## Executive Summary

Symthaea has sophisticated self-modeling infrastructure but lacks world-grounded prediction. The system predicts its own Φ and latency but not external outcomes. This plan closes that loop by:

1. Extending the existing `PredictionRecord` pattern to world outcomes
2. Adding Brier score calibration for proper scoring rules
3. **[UPGRADE A]** Adding Resolution Contracts to make resolution ungameable
4. **[UPGRADE B]** Adding Constraint Gate before autonomous actions
5. Integrating world predictions into the Active Inference EFE computation
6. Creating a causal attribution mechanism for learning from outcomes

### MAGI Loop Integration

This implementation enables the **Minimum AGI Loop** (MAGI Loop):

```
PREDICT (world) → RESOLVE (calibrate) → SELECT (action) → OBSERVE (reality) → ATTRIBUTE (causal) → UPDATE (safe)
     ↑                                                                                                    │
     └────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

The crossing criterion: **Complete this loop across 2+ domains without hard-coded fixes.**

---

## Gap Analysis

### What EXISTS (Strong Foundation)

| Component | Location | Capability |
|-----------|----------|------------|
| `PredictionRecord` | `self_model.rs:211-241` | Tracks predicted vs actual Φ/latency |
| `record_outcome()` | `lookahead.rs:305-322` | Records predicted vs actual for learning |
| `CalibrationStats` | `self_model.rs:416-440` | Mean error tracking |
| `ActiveInferenceRouter` | `active_inference.rs` | Full EFE with pragmatic/epistemic |
| `GenerativeModel` | `active_inference.rs:132-248` | State transition + likelihood matrices |
| `AdaptiveThresholds` | `phi_attention.rs:300-399` | Success/failure tracking per action type |

### What's MISSING (Critical Gaps)

| Gap | Impact | Solution |
|-----|--------|----------|
| **World outcome predictions** | Can't verify actions work | `WorldPrediction` struct |
| **Brier scores** | Poor calibration signal | `BrierScoreTracker` |
| **Action → Outcome mapping** | No causal learning | `CausalAttribution` |
| **EFE world grounding** | Routing ignores reality | `WorldGroundedEFE` |
| **Prediction horizons** | Only immediate Φ | `TemporalPrediction` |

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EXISTING INFRASTRUCTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│  SelfModel          │  LookaheadEngine     │  ActiveInferenceRouter │
│  - PredictionRecord │  - record_outcome()  │  - compute_efe()       │
│  - calibration_stats│  - confidence calc   │  - pragmatic/epistemic │
└─────────┬───────────┴─────────┬────────────┴──────────┬─────────────┘
          │                     │                       │
          ▼                     ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    NEW: WORLD PREDICTION LAYER                      │
├─────────────────────────────────────────────────────────────────────┤
│  WorldPrediction        │  BrierScoreTracker   │  CausalAttribution │
│  - predicted_outcome    │  - brier_score()     │  - attribute_cause()│
│  - actual_outcome       │  - calibration_curve │  - update_model()   │
│  - action_context       │  - reliability_bands │  - learn_mapping()  │
│  - prediction_horizon   │                      │                     │
└─────────┬───────────────┴──────────┬───────────┴──────────┬─────────┘
          │                          │                      │
          └──────────────────────────┴──────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    INTEGRATION: WorldGroundedEFE                    │
│  - Uses real outcome predictions (not just self-predictions)        │
│  - Calibrated by Brier scores (proper scoring rule)                 │
│  - Causal model learns action→outcome mappings                      │
│  - Feeds back into action selection loop                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: WorldPrediction Struct (Week 1)

**File**: `src/consciousness/recursive_improvement/world_prediction.rs` (NEW)

```rust
//! World-grounded prediction tracking with Brier score calibration

use std::collections::VecDeque;
use serde::{Serialize, Deserialize};

/// A prediction about a world outcome (not self-state)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldPrediction {
    /// Unique prediction ID
    pub id: u64,
    /// What action was taken
    pub action: ActionContext,
    /// Predicted probability of outcome categories
    pub predicted_probs: Vec<f64>,  // Sum to 1.0
    /// Which outcome actually occurred (one-hot index)
    pub actual_outcome: Option<usize>,
    /// Confidence in prediction (0-1)
    pub confidence: f64,
    /// Prediction horizon (how far ahead)
    pub horizon_steps: usize,
    /// When prediction was made
    pub timestamp: std::time::Instant,
    /// Has this been resolved?
    pub resolved: bool,
}

/// Context for an action being predicted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionContext {
    /// Action type/ID
    pub action_type: String,
    /// Pre-action state summary
    pub pre_state: StateSummary,
    /// Expected post-action state
    pub expected_post_state: StateSummary,
    /// Risk tier of action
    pub risk_tier: RiskTier,
}

/// Outcome categories for predictions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutcomeCategory {
    /// Action succeeded as expected
    Success,
    /// Action succeeded but differently than expected
    PartialSuccess,
    /// Action had no effect
    NoEffect,
    /// Action failed safely (rollback worked)
    SafeFailure,
    /// Action failed with side effects
    UnsafeFailure,
}

impl OutcomeCategory {
    pub fn all() -> Vec<Self> {
        vec![
            Self::Success,
            Self::PartialSuccess,
            Self::NoEffect,
            Self::SafeFailure,
            Self::UnsafeFailure,
        ]
    }

    pub fn to_index(&self) -> usize {
        match self {
            Self::Success => 0,
            Self::PartialSuccess => 1,
            Self::NoEffect => 2,
            Self::SafeFailure => 3,
            Self::UnsafeFailure => 4,
        }
    }
}
```

**Integration point**: Extend `self_model.rs:SelfModel` to hold `VecDeque<WorldPrediction>`.

---

### Phase 1.5: Resolution Contract (UPGRADE A - CRITICAL)

**Why This Upgrade Is Critical**: Without explicit resolution authority, the system can drift into "self-graded success." This makes the loop ungameable.

**File**: Add to `world_prediction.rs`

```rust
/// Contract defining how an action type's outcome gets resolved
/// This makes resolution ungameable - the system can't grade itself
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionContract {
    /// Action type this contract applies to
    pub action_type: String,

    /// What counts as Success
    pub success_criteria: String,
    /// What counts as PartialSuccess
    pub partial_criteria: String,
    /// What counts as NoEffect
    pub no_effect_criteria: String,
    /// What counts as SafeFailure
    pub safe_failure_criteria: String,
    /// What counts as UnsafeFailure
    pub unsafe_failure_criteria: String,

    /// Who/what resolves it (the resolution authority)
    pub resolver: ResolutionAuthority,

    /// Time horizon (when "NoEffect" is decided)
    pub timeout: std::time::Duration,
}

/// The authority that resolves a prediction - NOT the system itself
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionAuthority {
    /// Unit test pass/fail
    TestSuite {
        test_pattern: String,  // e.g., "tests/test_*.rs"
        pass_threshold: f64,   // e.g., 1.0 for all tests must pass
    },

    /// Process exit code
    ExitCode {
        success_codes: Vec<i32>,  // e.g., [0] for typical success
    },

    /// Diff-based verification
    DiffVerifier {
        expected_path: std::path::PathBuf,
        tolerance: DiffTolerance,
    },

    /// External API check
    ExternalAPI {
        endpoint: String,
        expected_status: u16,
        expected_body_contains: Option<String>,
    },

    /// Human confirmation required
    HumanConfirmation {
        prompt: String,
        timeout: std::time::Duration,
    },

    /// File/resource state check
    ResourceState {
        path: std::path::PathBuf,
        expected_state: ResourceExpectation,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiffTolerance {
    Exact,
    IgnoreWhitespace,
    IgnoreComments,
    SemanticEquivalence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceExpectation {
    Exists,
    NotExists,
    ContainsText(String),
    MatchesPattern(String),
    SizeInRange { min: u64, max: u64 },
}

impl ResolutionContract {
    /// Create a contract for code execution actions
    pub fn for_code_execution(test_pattern: &str) -> Self {
        Self {
            action_type: "CodeExecution".to_string(),
            success_criteria: "All specified tests pass".to_string(),
            partial_criteria: "Some tests pass, some fail".to_string(),
            no_effect_criteria: "Tests unchanged from before action".to_string(),
            safe_failure_criteria: "Tests fail but no side effects".to_string(),
            unsafe_failure_criteria: "Tests fail with unexpected side effects".to_string(),
            resolver: ResolutionAuthority::TestSuite {
                test_pattern: test_pattern.to_string(),
                pass_threshold: 1.0,
            },
            timeout: std::time::Duration::from_secs(300),
        }
    }

    /// Create a contract for shell command actions
    pub fn for_shell_command(success_codes: Vec<i32>) -> Self {
        Self {
            action_type: "ShellCommand".to_string(),
            success_criteria: format!("Exit code in {:?}", success_codes),
            partial_criteria: "Command completes with warnings".to_string(),
            no_effect_criteria: "Command produces no output or change".to_string(),
            safe_failure_criteria: "Command fails with known error".to_string(),
            unsafe_failure_criteria: "Command fails with unknown side effects".to_string(),
            resolver: ResolutionAuthority::ExitCode { success_codes },
            timeout: std::time::Duration::from_secs(60),
        }
    }
}
```

**Registry of Contracts**: Add to `SelfModel` or create dedicated `ContractRegistry`:

```rust
pub struct ContractRegistry {
    contracts: HashMap<String, ResolutionContract>,
}

impl ContractRegistry {
    pub fn new() -> Self {
        let mut contracts = HashMap::new();

        // Pre-register common action types
        contracts.insert(
            "CodeExecution".to_string(),
            ResolutionContract::for_code_execution("tests/**/*.rs"),
        );
        contracts.insert(
            "ShellCommand".to_string(),
            ResolutionContract::for_shell_command(vec![0]),
        );

        Self { contracts }
    }

    pub fn resolve(&self, action_type: &str) -> Option<&ResolutionContract> {
        self.contracts.get(action_type)
    }
}
```

---

### Phase 2: Brier Score Calibration (Week 1-2)

**File**: Add to `world_prediction.rs`

```rust
/// Proper scoring rule for probability calibration
pub struct BrierScoreTracker {
    /// Recent predictions for rolling calibration
    predictions: VecDeque<WorldPrediction>,
    /// Maximum history size
    max_history: usize,
    /// Calibration bins for reliability diagram
    calibration_bins: Vec<CalibrationBin>,
}

#[derive(Debug, Clone, Default)]
pub struct CalibrationBin {
    /// Predicted probability range center
    pub bin_center: f64,
    /// Sum of predicted probabilities in this bin
    pub predicted_sum: f64,
    /// Sum of actual outcomes (1 if occurred, 0 otherwise)
    pub actual_sum: f64,
    /// Number of predictions in this bin
    pub count: usize,
}

impl BrierScoreTracker {
    pub fn new(max_history: usize) -> Self {
        // 10 bins: 0-0.1, 0.1-0.2, ..., 0.9-1.0
        let calibration_bins = (0..10)
            .map(|i| CalibrationBin {
                bin_center: (i as f64 + 0.5) / 10.0,
                ..Default::default()
            })
            .collect();

        Self {
            predictions: VecDeque::with_capacity(max_history),
            max_history,
            calibration_bins,
        }
    }

    /// Compute Brier score for a single prediction
    /// Brier = (1/N) * Σ(predicted - actual)²
    /// Lower is better. Perfect = 0.0, Worst = 1.0 (for binary)
    pub fn brier_score(&self, prediction: &WorldPrediction) -> Option<f64> {
        let actual_idx = prediction.actual_outcome?;

        let mut score = 0.0;
        for (i, &prob) in prediction.predicted_probs.iter().enumerate() {
            let actual = if i == actual_idx { 1.0 } else { 0.0 };
            score += (prob - actual).powi(2);
        }

        Some(score / prediction.predicted_probs.len() as f64)
    }

    /// Compute rolling average Brier score
    pub fn rolling_brier(&self) -> f64 {
        let resolved: Vec<_> = self.predictions.iter()
            .filter(|p| p.resolved && p.actual_outcome.is_some())
            .collect();

        if resolved.is_empty() {
            return 0.25; // Prior for 4 outcomes
        }

        let total: f64 = resolved.iter()
            .filter_map(|p| self.brier_score(p))
            .sum();

        total / resolved.len() as f64
    }

    /// Get calibration error (ECE - Expected Calibration Error)
    pub fn expected_calibration_error(&self) -> f64 {
        let total_samples: usize = self.calibration_bins.iter().map(|b| b.count).sum();
        if total_samples == 0 {
            return 0.0;
        }

        let mut ece = 0.0;
        for bin in &self.calibration_bins {
            if bin.count > 0 {
                let avg_predicted = bin.predicted_sum / bin.count as f64;
                let avg_actual = bin.actual_sum / bin.count as f64;
                ece += (bin.count as f64 / total_samples as f64)
                     * (avg_predicted - avg_actual).abs();
            }
        }
        ece
    }

    /// Record a resolved prediction
    pub fn record(&mut self, prediction: WorldPrediction) {
        if prediction.resolved && prediction.actual_outcome.is_some() {
            // Update calibration bins
            let actual_idx = prediction.actual_outcome.unwrap();
            for (i, &prob) in prediction.predicted_probs.iter().enumerate() {
                let bin_idx = (prob * 10.0).min(9.0) as usize;
                self.calibration_bins[bin_idx].predicted_sum += prob;
                self.calibration_bins[bin_idx].actual_sum +=
                    if i == actual_idx { 1.0 } else { 0.0 };
                self.calibration_bins[bin_idx].count += 1;
            }
        }

        self.predictions.push_back(prediction);
        while self.predictions.len() > self.max_history {
            self.predictions.pop_front();
        }
    }

    /// Are we well-calibrated? (ECE < threshold)
    pub fn is_calibrated(&self, threshold: f64) -> bool {
        self.expected_calibration_error() < threshold
    }
}
```

**Integration point**: Add `brier_tracker: BrierScoreTracker` to `SelfModel`.

---

### Phase 3: Causal Attribution (Week 2)

**File**: Add to `world_prediction.rs`

```rust
/// Maps actions to outcomes for causal learning
pub struct CausalAttribution {
    /// Action type → outcome distribution (learned)
    action_outcome_model: HashMap<String, Vec<f64>>,
    /// State-action-outcome triples for learning
    history: VecDeque<(ActionContext, OutcomeCategory)>,
    /// Learning rate for model updates
    learning_rate: f64,
}

impl CausalAttribution {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            action_outcome_model: HashMap::new(),
            history: VecDeque::with_capacity(1000),
            learning_rate,
        }
    }

    /// Predict outcome distribution for an action
    pub fn predict_outcome(&self, action: &ActionContext) -> Vec<f64> {
        self.action_outcome_model
            .get(&action.action_type)
            .cloned()
            .unwrap_or_else(|| {
                // Prior: 60% success, 20% partial, 10% no-effect, 5% safe-fail, 5% unsafe
                vec![0.60, 0.20, 0.10, 0.05, 0.05]
            })
    }

    /// Learn from observed outcome
    pub fn learn(&mut self, action: &ActionContext, outcome: OutcomeCategory) {
        let current = self.action_outcome_model
            .entry(action.action_type.clone())
            .or_insert_with(|| vec![0.60, 0.20, 0.10, 0.05, 0.05]);

        // Bayesian update toward observed outcome
        let outcome_idx = outcome.to_index();
        for (i, prob) in current.iter_mut().enumerate() {
            let target = if i == outcome_idx { 1.0 } else { 0.0 };
            *prob = *prob * (1.0 - self.learning_rate) + target * self.learning_rate;
        }

        // Renormalize
        let sum: f64 = current.iter().sum();
        for prob in current.iter_mut() {
            *prob /= sum;
        }

        self.history.push_back((action.clone(), outcome));
        if self.history.len() > 1000 {
            self.history.pop_front();
        }
    }

    /// Get confidence in prediction (based on evidence count)
    pub fn confidence(&self, action_type: &str) -> f64 {
        let count = self.history.iter()
            .filter(|(a, _)| a.action_type == action_type)
            .count();

        // Confidence grows with evidence: 1 - e^(-count/20)
        1.0 - (-count as f64 / 20.0).exp()
    }
}
```

**Integration point**: Add to `SelfModel` for unified access.

---

### Phase 3.5: Constraint Gate (UPGRADE B - CRITICAL)

**Why This Upgrade Is Critical**: The EFE loop can become capable quickly. The first strong behavior must not be "unsafe cleverness." This gate controls execution mode based on risk and calibration.

**File**: `src/consciousness/recursive_improvement/constraint_gate.rs` (NEW)

```rust
//! Constraint Gate: Safety control before autonomous action execution
//!
//! This gate ensures that:
//! 1. High-risk actions require supervision
//! 2. Poorly-calibrated predictions force exploration/dry-run mode
//! 3. The system can't "cleverly" bypass safety through EFE optimization

use super::world_prediction::{BrierScoreTracker, ActionContext, RiskTier};

/// Execution mode determined by the constraint gate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Full autonomous execution - system is trusted
    Autonomous,
    /// Dry-run first, then confirm - system needs verification
    DryRun,
    /// Human must approve before execution - high risk or poor calibration
    Supervised,
}

/// Configuration for the constraint gate
#[derive(Debug, Clone)]
pub struct ConstraintGateConfig {
    /// Risk tier at or above which supervision is required
    pub supervision_threshold: RiskTier,
    /// ECE threshold above which curious/dry-run mode is forced
    pub calibration_threshold: f64,
    /// Minimum predictions before trusting calibration
    pub min_predictions_for_trust: usize,
    /// Whether to allow override in emergency
    pub allow_emergency_override: bool,
}

impl Default for ConstraintGateConfig {
    fn default() -> Self {
        Self {
            supervision_threshold: RiskTier::High,
            calibration_threshold: 0.15,  // ECE > 15% = poorly calibrated
            min_predictions_for_trust: 20,
            allow_emergency_override: false,
        }
    }
}

/// The constraint gate that controls execution mode
pub struct ConstraintGate {
    config: ConstraintGateConfig,
    /// Number of actions gated so far
    actions_checked: usize,
    /// Number of actions that required supervision
    supervision_required: usize,
    /// Number of actions forced to dry-run
    dry_run_forced: usize,
}

impl ConstraintGate {
    pub fn new(config: ConstraintGateConfig) -> Self {
        Self {
            config,
            actions_checked: 0,
            supervision_required: 0,
            dry_run_forced: 0,
        }
    }

    /// Check an action and determine execution mode
    pub fn check(
        &mut self,
        action: &ActionContext,
        calibration: &BrierScoreTracker,
    ) -> ExecutionMode {
        self.actions_checked += 1;

        // Rule 1: High risk → require supervision
        if action.risk_tier >= self.config.supervision_threshold {
            self.supervision_required += 1;
            tracing::info!(
                "ConstraintGate: Action '{}' requires supervision (risk_tier = {:?})",
                action.action_type, action.risk_tier
            );
            return ExecutionMode::Supervised;
        }

        // Rule 2: Not enough predictions → force dry-run (cold start protection)
        let prediction_count = calibration.prediction_count();
        if prediction_count < self.config.min_predictions_for_trust {
            self.dry_run_forced += 1;
            tracing::info!(
                "ConstraintGate: Forcing dry-run for '{}' (only {} predictions, need {})",
                action.action_type, prediction_count, self.config.min_predictions_for_trust
            );
            return ExecutionMode::DryRun;
        }

        // Rule 3: Poor calibration → force dry-run (epistemic humility)
        let ece = calibration.expected_calibration_error();
        if ece > self.config.calibration_threshold {
            self.dry_run_forced += 1;
            tracing::info!(
                "ConstraintGate: Forcing dry-run for '{}' (ECE = {:.3} > threshold {:.3})",
                action.action_type, ece, self.config.calibration_threshold
            );
            return ExecutionMode::DryRun;
        }

        // All checks passed → autonomous execution allowed
        ExecutionMode::Autonomous
    }

    /// Get statistics about gate decisions
    pub fn stats(&self) -> ConstraintGateStats {
        ConstraintGateStats {
            actions_checked: self.actions_checked,
            supervision_required: self.supervision_required,
            dry_run_forced: self.dry_run_forced,
            autonomous_allowed: self.actions_checked
                - self.supervision_required
                - self.dry_run_forced,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConstraintGateStats {
    pub actions_checked: usize,
    pub supervision_required: usize,
    pub dry_run_forced: usize,
    pub autonomous_allowed: usize,
}
```

**Integration with EFE**: The constraint gate runs BEFORE action execution, after EFE selects the action:

```rust
// In action execution pipeline:
impl ActionExecutor {
    pub fn execute(&mut self, action: ActionContext) -> ExecutionResult {
        // 1. Constraint gate determines mode
        let mode = self.constraint_gate.check(&action, &self.calibration);

        match mode {
            ExecutionMode::Autonomous => {
                // Execute directly
                self.execute_action(&action)
            }
            ExecutionMode::DryRun => {
                // Show what would happen, then ask for confirmation
                let preview = self.dry_run(&action)?;
                if self.confirm_execution(&preview)? {
                    self.execute_action(&action)
                } else {
                    ExecutionResult::Cancelled
                }
            }
            ExecutionMode::Supervised => {
                // Require human approval
                let approval = self.request_human_approval(&action)?;
                if approval.approved {
                    self.execute_action(&action)
                } else {
                    ExecutionResult::Rejected { reason: approval.reason }
                }
            }
        }
    }
}
```

**Key Insight**: This gate makes the calibration penalty in EFE *behaviorally binding*, not just a preference weight.

---

### Phase 4: World-Grounded EFE Integration (Week 2-3)

**File**: Modify `active_inference.rs`

```rust
// Add to ActiveInferenceRouter struct:
/// World prediction tracker
world_predictions: BrierScoreTracker,
/// Causal model for action-outcome mapping
causal_model: CausalAttribution,

// Modify compute_efe() to use world predictions:
impl ActiveInferenceRouter {
    /// Compute EFE with world-grounded predictions
    pub fn compute_world_grounded_efe(
        &self,
        strategy: RoutingStrategy,
        proposed_action: &ActionContext,
    ) -> ExpectedFreeEnergy {
        let mut efe = ExpectedFreeEnergy::new(strategy);

        // ---- PRAGMATIC: Expected utility of action ----
        let outcome_probs = self.causal_model.predict_outcome(proposed_action);

        // Utility weights per outcome category
        let utilities = vec![1.0, 0.5, 0.0, -0.2, -1.0]; // Success→UnsafeFail
        efe.pragmatic = outcome_probs.iter()
            .zip(utilities.iter())
            .map(|(p, u)| p * u)
            .sum();

        // ---- EPISTEMIC: Information gain from action ----
        let causal_confidence = self.causal_model.confidence(&proposed_action.action_type);
        // Low confidence = high epistemic value (learning opportunity)
        efe.epistemic = 1.0 - causal_confidence;

        // ---- CALIBRATION PENALTY: Reduce trust if poorly calibrated ----
        let calibration_error = self.world_predictions.expected_calibration_error();
        let calibration_penalty = calibration_error * 0.5;

        // ---- NOVELTY: Standard exploration bonus ----
        let recent_uses = self.strategy_history.iter()
            .filter(|s| **s == strategy)
            .count();
        efe.novelty = 1.0 / (1.0 + recent_uses as f64);

        // ---- TOTAL EFE (lower = better) ----
        // Negate pragmatic (maximize utility)
        // Add epistemic (value learning)
        // Subtract novelty (encourage exploration)
        // Add calibration penalty (penalize when poorly calibrated)
        efe.total = -self.config.pragmatic_weight * efe.pragmatic
                  + self.config.epistemic_weight * efe.epistemic
                  - self.config.novelty_weight * efe.novelty
                  + calibration_penalty;

        efe
    }

    /// Record action outcome and update models
    pub fn record_world_outcome(
        &mut self,
        prediction: WorldPrediction,
        actual_outcome: OutcomeCategory,
    ) {
        let mut resolved = prediction;
        resolved.actual_outcome = Some(actual_outcome.to_index());
        resolved.resolved = true;

        // Update Brier score tracker
        self.world_predictions.record(resolved.clone());

        // Update causal model
        self.causal_model.learn(&resolved.action, actual_outcome);

        // Log calibration if needed
        if !self.world_predictions.is_calibrated(0.15) {
            tracing::warn!(
                "World prediction poorly calibrated: ECE = {:.3}",
                self.world_predictions.expected_calibration_error()
            );
        }
    }
}
```

---

### Phase 5: Integration with ActionIR (Week 3)

**File**: Modify `action/mod.rs` to emit predictions

```rust
impl ActionIR {
    /// Create a world prediction for this action
    pub fn to_world_prediction(&self, confidence: f64) -> WorldPrediction {
        WorldPrediction {
            id: generate_prediction_id(),
            action: ActionContext {
                action_type: self.action_type_string(),
                pre_state: StateSummary::current(),
                expected_post_state: self.expected_state_change(),
                risk_tier: self.risk_tier(),
            },
            predicted_probs: vec![0.7, 0.15, 0.10, 0.03, 0.02], // Default prior
            actual_outcome: None,
            confidence,
            horizon_steps: 1,
            timestamp: std::time::Instant::now(),
            resolved: false,
        }
    }
}
```

---

## Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `src/consciousness/recursive_improvement/world_prediction.rs` | **CREATE** | Core world prediction types + ResolutionContract |
| `src/consciousness/recursive_improvement/constraint_gate.rs` | **CREATE** | [UPGRADE B] Safety gate before execution |
| `src/consciousness/recursive_improvement/resolution.rs` | **CREATE** | ResolutionAuthority implementations |
| `src/consciousness/recursive_improvement/self_model.rs` | **MODIFY** | Add WorldPrediction + ContractRegistry |
| `crates/symthaea-consciousness/src/recursive_improvement/routers/active_inference.rs` | **MODIFY** | Add world-grounded EFE |
| `src/action/mod.rs` | **MODIFY** | Add prediction emission + gate check |
| `src/consciousness/recursive_improvement/mod.rs` | **MODIFY** | Re-export new types |

---

## Success Metrics (Phased)

### Early Phase (Cold Start)

| Metric | Target | Focus |
|--------|--------|-------|
| Resolution coverage | 100% of actions | No self-grading |
| Domain coverage | 2+ domains | MAGI crossing requirement |
| Constraint gate active | Always | Safety enforcement |

### Mid Phase (Stabilization)

| Metric | Target | Measurement |
|--------|--------|-------------|
| World predictions tracked | 100% of actions | Count in logs |
| Brier score | < 0.25 | Rolling average |
| ECE (calibration error) | < 0.15 | Per 100 predictions |
| Resolution authority used | 100% | No self-grading |

### Late Phase (Maturity)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Brier score | < 0.20 | Rolling average |
| ECE (calibration error) | < 0.10 | Per 100 predictions |
| Action→Outcome accuracy | > 70% | Top-1 prediction |
| EFE-action correlation | r > 0.5 | Correlation analysis |
| Autonomous execution rate | Increasing | As calibration improves |

### MAGI Crossing Metrics

| Metric | Requirement |
|--------|-------------|
| Loop completions | > 100 per domain |
| Domains validated | ≥ 2 unrelated domains |
| Self-grading | 0% (all external resolution) |
| Behavior change demonstrated | Calibration improves over time |

---

## Rollout Strategy (Updated)

1. **Week 1**: Implement WorldPrediction + ResolutionContract (Phase 1 + 1.5)
2. **Week 2**: Add BrierScoreTracker + ConstraintGate (Phase 2 + 3.5)
3. **Week 3**: Add CausalAttribution + integrate with SelfModel (Phase 3)
4. **Week 4**: Modify ActiveInferenceRouter for world-grounded EFE (Phase 4)
5. **Week 5**: Integration with ActionIR + end-to-end testing (Phase 5)
6. **Week 6**: First domain validation (CodeExecution)
7. **Week 7**: Second domain validation (ShellCommand)
8. **Week 8**: MAGI crossing verification

---

## Risk Mitigation (Updated)

| Risk | Mitigation |
|------|------------|
| Performance regression from tracking | Async logging, bounded queues |
| Calibration never converges | Fallback to fixed priors after N failures |
| Circular dependency with EFE | Clear separation: predict → act → observe → learn |
| Cold start problem | ConstraintGate forces dry-run until min predictions met |
| Self-grading drift | ResolutionContract mandates external authority |
| Unsafe cleverness | ConstraintGate controls execution mode |
| Resolution authority unavailable | Fallback to HumanConfirmation |

---

## Dependencies

- **Existing**: `self_model.rs`, `active_inference.rs`, `action/mod.rs`, `lookahead.rs`
- **New crates**: None (uses existing ndarray, serde)
- **Testing**: Property-based tests for calibration convergence

---

*"The system that doesn't check its predictions against reality will eventually act on hallucinations." — Design principle*

**Next Step**: Implement Phase 1 (WorldPrediction struct) in new file.
