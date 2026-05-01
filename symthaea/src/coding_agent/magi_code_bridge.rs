// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # MAGI Code Bridge — Self-Improving Code Generation
//!
//! Connects the Code Orchestrator to the MAGI Loop's world-grounded prediction
//! system. Before generating code, the system predicts whether it will compile
//! and pass tests. After execution, it resolves the prediction with the actual
//! outcome. Over time, BrierScoreTracker calibration improves, making the
//! system increasingly accurate at predicting its own coding success.
//!
//! This is the core mechanism for self-improving code generation:
//!
//! ```text
//! MAGI predict("this code will compile", confidence=0.7)
//!     ↓
//! Orchestrator generates code
//!     ↓
//! CodeExecutor compiles + tests
//!     ↓
//! MAGI resolve(prediction_id, actual_outcome)
//!     ↓
//! BrierScoreTracker adjusts domain calibration
//!     ↓
//! Next generation: adjusted confidence → better routing
//! ```

use crate::consciousness::recursive_improvement::world_prediction::{
    OutcomeCategory, PredictionDomain, ResolutionAuthority, ResolutionContract, WorldActionContext,
    WorldPrediction,
};

/// Bridge between code generation and MAGI world predictions.
///
/// Tracks predictions about code generation outcomes and provides
/// the feedback loop for self-improvement.
pub struct MagiCodeBridge {
    /// Pending predictions awaiting resolution
    pending: Vec<PendingCodePrediction>,
    /// History of resolved predictions for analysis
    history: Vec<ResolvedCodePrediction>,
    /// Running statistics
    stats: CodePredictionStats,
}

/// A prediction about a code generation attempt, awaiting resolution.
#[derive(Debug, Clone)]
pub struct PendingCodePrediction {
    /// The MAGI prediction
    pub prediction: WorldPrediction,
    /// What backend was selected
    pub backend: String,
    /// Request name for tracking
    pub request_name: String,
}

/// A resolved code prediction with outcome.
#[derive(Debug, Clone)]
pub struct ResolvedCodePrediction {
    /// Original prediction confidence
    pub predicted_confidence: f64,
    /// Whether the prediction was correct
    pub was_correct: bool,
    /// Actual outcome category
    pub actual_outcome: OutcomeCategory,
    /// Backend that generated the code
    pub backend: String,
    /// Brier score contribution
    pub brier_score: f64,
}

/// Running statistics on code prediction accuracy.
#[derive(Debug, Clone, Default)]
pub struct CodePredictionStats {
    /// Total predictions made
    pub total_predictions: usize,
    /// Predictions resolved as correct
    pub correct_predictions: usize,
    /// Running Brier score (lower = better calibrated)
    pub running_brier: f64,
    /// Compilation success predictions that matched reality
    pub compile_accuracy: f64,
    /// Test pass predictions that matched reality
    pub test_accuracy: f64,
    /// Per-backend accuracy
    pub native_accuracy: f64,
    pub llm_accuracy: f64,
}

impl CodePredictionStats {
    /// Overall accuracy ratio
    pub fn accuracy(&self) -> f64 {
        if self.total_predictions == 0 {
            0.5 // Prior: 50% before any data
        } else {
            self.correct_predictions as f64 / self.total_predictions as f64
        }
    }
}

impl MagiCodeBridge {
    /// Create a new MAGI code bridge.
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            history: Vec::new(),
            stats: CodePredictionStats::default(),
        }
    }

    /// Create a prediction about a code generation attempt.
    ///
    /// Call this BEFORE generating code. The returned prediction should be
    /// resolved after compilation/testing via `resolve_prediction()`.
    ///
    /// # Arguments
    /// * `request_name` — Name of the function/type being generated
    /// * `backend` — Which backend will attempt generation
    /// * `raw_confidence` — The system's raw confidence (0.0-1.0) that this will succeed
    /// * `claim` — What specifically is being predicted (e.g., "code will compile")
    pub fn predict_generation(
        &mut self,
        request_name: &str,
        backend: &str,
        raw_confidence: f64,
        claim: &str,
    ) -> String {
        let action_context = WorldActionContext::new(
            "compile",
            format!(
                "Generate {} via {} — code should compile and pass tests",
                request_name, backend
            ),
        );

        let resolution_contract = ResolutionContract::new(
            "compile",
            ResolutionAuthority::ExitCode {
                success_codes: vec![0],
            },
        );
        let prediction = WorldPrediction::new(
            claim,
            OutcomeCategory::Success,
            raw_confidence,
            action_context,
            resolution_contract,
        );

        let prediction_id = prediction.id.clone();

        self.pending.push(PendingCodePrediction {
            prediction,
            backend: backend.to_string(),
            request_name: request_name.to_string(),
        });

        prediction_id
    }

    /// Resolve a pending prediction with the actual outcome.
    ///
    /// Call this AFTER compilation/testing. Updates statistics and
    /// returns whether the prediction was correct.
    ///
    /// # Arguments
    /// * `prediction_id` — ID returned from `predict_generation()`
    /// * `compiled` — Whether the code compiled successfully
    /// * `tests_passed` — Whether all tests passed (None if no tests run)
    pub fn resolve_prediction(
        &mut self,
        prediction_id: &str,
        compiled: bool,
        tests_passed: Option<bool>,
    ) -> Option<bool> {
        // Find and remove the pending prediction
        let idx = self
            .pending
            .iter()
            .position(|p| p.prediction.id == prediction_id)?;
        let mut pending = self.pending.remove(idx);

        // Determine actual outcome
        let actual_outcome = if compiled && tests_passed.unwrap_or(true) {
            OutcomeCategory::Success
        } else if compiled && tests_passed == Some(false) {
            OutcomeCategory::Partial
        } else {
            OutcomeCategory::SafeFailure
        };

        // Resolve the prediction
        let predicted_success = pending.prediction.confidence > 0.5;
        let actual_success = actual_outcome.is_positive();
        let was_correct = predicted_success == actual_success;

        if actual_success {
            pending.prediction.resolve_true(actual_outcome, 1.0);
        } else {
            pending.prediction.resolve_false(actual_outcome, 1.0);
        }

        // Calculate Brier score: (predicted_prob - actual_binary)^2
        let actual_binary = if actual_success { 1.0 } else { 0.0 };
        let brier = (pending.prediction.confidence - actual_binary).powi(2);

        // Record in history
        self.history.push(ResolvedCodePrediction {
            predicted_confidence: pending.prediction.confidence,
            was_correct,
            actual_outcome,
            backend: pending.backend.clone(),
            brier_score: brier,
        });

        // Update statistics
        self.stats.total_predictions += 1;
        if was_correct {
            self.stats.correct_predictions += 1;
        }

        // Update running Brier (exponential moving average, alpha=0.1)
        let alpha = 0.1;
        self.stats.running_brier = self.stats.running_brier * (1.0 - alpha) + brier * alpha;

        // Update per-backend accuracy
        let backend_history: Vec<&ResolvedCodePrediction> = self
            .history
            .iter()
            .filter(|h| h.backend == pending.backend)
            .collect();
        if !backend_history.is_empty() {
            let correct = backend_history.iter().filter(|h| h.was_correct).count();
            let accuracy = correct as f64 / backend_history.len() as f64;
            if pending.backend.contains("Native") || pending.backend.contains("CodeGenerator") {
                self.stats.native_accuracy = accuracy;
            } else {
                self.stats.llm_accuracy = accuracy;
            }
        }

        Some(was_correct)
    }

    /// Get adjusted confidence for a code generation attempt.
    ///
    /// Uses historical accuracy to calibrate raw confidence.
    /// If the system has been overconfident historically, this returns
    /// a lower confidence than the raw estimate.
    pub fn adjusted_confidence(&self, raw_confidence: f64, backend: &str) -> f64 {
        if self.stats.total_predictions < 5 {
            // Not enough data — use raw confidence with mild pessimism
            return raw_confidence * 0.8;
        }

        // Backend-specific adjustment
        let backend_accuracy = if backend.contains("Native") || backend.contains("CodeGenerator") {
            self.stats.native_accuracy
        } else {
            self.stats.llm_accuracy
        };

        // Blend raw confidence with historical accuracy
        // More history → more trust in historical data
        let history_weight = (self.stats.total_predictions as f64 / 20.0).min(0.7);
        let adjusted = raw_confidence * (1.0 - history_weight) + backend_accuracy * history_weight;

        adjusted.clamp(0.05, 0.95) // Never 0% or 100% confident
    }

    /// Get current prediction statistics.
    pub fn stats(&self) -> &CodePredictionStats {
        &self.stats
    }

    /// Get number of pending (unresolved) predictions.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get resolved prediction history.
    pub fn history(&self) -> &[ResolvedCodePrediction] {
        &self.history
    }

    /// Check if the system is well-calibrated for code generation.
    ///
    /// Well-calibrated means Brier score < 0.15 with at least 10 predictions.
    pub fn is_well_calibrated(&self) -> bool {
        self.stats.total_predictions >= 10 && self.stats.running_brier < 0.15
    }

    /// Get a human-readable summary of calibration quality.
    pub fn calibration_summary(&self) -> String {
        if self.stats.total_predictions == 0 {
            return "No code predictions yet — calibration pending".to_string();
        }

        format!(
            "Code calibration: {}/{} correct ({:.0}%), Brier: {:.3}, Native: {:.0}%, LLM: {:.0}%{}",
            self.stats.correct_predictions,
            self.stats.total_predictions,
            self.stats.accuracy() * 100.0,
            self.stats.running_brier,
            self.stats.native_accuracy * 100.0,
            self.stats.llm_accuracy * 100.0,
            if self.is_well_calibrated() {
                " [CALIBRATED]"
            } else {
                ""
            },
        )
    }
}

impl Default for MagiCodeBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_and_resolve_success() {
        let mut bridge = MagiCodeBridge::new();

        let pred_id = bridge.predict_generation("add", "CodeGenerator", 0.8, "code will compile");
        assert_eq!(bridge.pending_count(), 1);

        let was_correct = bridge.resolve_prediction(&pred_id, true, Some(true));
        assert_eq!(was_correct, Some(true)); // Predicted success, got success
        assert_eq!(bridge.pending_count(), 0);
        assert_eq!(bridge.stats().total_predictions, 1);
        assert_eq!(bridge.stats().correct_predictions, 1);
    }

    #[test]
    fn test_predict_and_resolve_failure() {
        let mut bridge = MagiCodeBridge::new();

        let pred_id =
            bridge.predict_generation("complex_algo", "CodeGenerator", 0.9, "code will compile");

        let was_correct = bridge.resolve_prediction(&pred_id, false, None);
        assert_eq!(was_correct, Some(false)); // Predicted success (0.9), got failure
        assert_eq!(bridge.stats().correct_predictions, 0);
    }

    #[test]
    fn test_resolve_unknown_id() {
        let mut bridge = MagiCodeBridge::new();
        let result = bridge.resolve_prediction("nonexistent", true, None);
        assert_eq!(result, None);
    }

    #[test]
    fn test_adjusted_confidence_initial() {
        let bridge = MagiCodeBridge::new();
        // With no history, should apply mild pessimism (0.8x)
        let adjusted = bridge.adjusted_confidence(0.9, "CodeGenerator");
        assert!((adjusted - 0.72).abs() < 0.01);
    }

    #[test]
    fn test_calibration_improves_with_data() {
        let mut bridge = MagiCodeBridge::new();

        // Simulate 10 predictions where native succeeds 80% of the time
        for i in 0..10 {
            let id = bridge.predict_generation(
                &format!("func_{i}"),
                "CodeGenerator",
                0.7,
                "will compile",
            );
            let compiled = i < 8; // 80% success rate
            bridge.resolve_prediction(&id, compiled, Some(compiled));
        }

        assert_eq!(bridge.stats().total_predictions, 10);
        assert!(bridge.stats().accuracy() > 0.5);

        // Adjusted confidence should reflect the 80% reality
        let adjusted = bridge.adjusted_confidence(0.9, "CodeGenerator");
        // Should be pulled toward historical accuracy (~0.8)
        assert!(
            adjusted < 0.9,
            "Should be pulled down from 0.9: got {adjusted}"
        );
        assert!(adjusted > 0.5, "Should still be positive: got {adjusted}");
    }

    #[test]
    fn test_partial_success_tracking() {
        let mut bridge = MagiCodeBridge::new();

        let id = bridge.predict_generation("sort", "LLM", 0.8, "will compile and pass tests");
        // Compiles but tests fail
        let was_correct = bridge.resolve_prediction(&id, true, Some(false));
        // Predicted success (0.8 > 0.5), got partial (is_positive=true via Partial)
        // Partial is_positive() = true, so prediction was correct
        assert_eq!(was_correct, Some(true));
    }

    #[test]
    fn test_calibration_summary() {
        let bridge = MagiCodeBridge::new();
        let summary = bridge.calibration_summary();
        assert!(summary.contains("No code predictions yet"));
    }

    #[test]
    fn test_well_calibrated_detection() {
        let mut bridge = MagiCodeBridge::new();
        assert!(!bridge.is_well_calibrated()); // Need >= 10 predictions

        // Add 10 perfectly calibrated predictions (0.8 confidence, 80% success)
        for i in 0..10 {
            let id = bridge.predict_generation(&format!("f{i}"), "CodeGenerator", 0.8, "compile");
            bridge.resolve_prediction(&id, i < 8, Some(i < 8));
        }

        // With good calibration, Brier should be low
        assert!(bridge.stats().running_brier < 0.3);
    }

    #[test]
    fn test_per_backend_accuracy() {
        let mut bridge = MagiCodeBridge::new();

        // Native: 90% success
        for i in 0..10 {
            let id = bridge.predict_generation(&format!("n{i}"), "CodeGenerator", 0.8, "compile");
            bridge.resolve_prediction(&id, i < 9, Some(i < 9));
        }

        // LLM: 60% success
        for i in 0..10 {
            let id = bridge.predict_generation(&format!("l{i}"), "LLM", 0.8, "compile");
            bridge.resolve_prediction(&id, i < 6, Some(i < 6));
        }

        // Native should show higher accuracy than LLM
        assert!(bridge.stats().native_accuracy > bridge.stats().llm_accuracy);
    }
}
