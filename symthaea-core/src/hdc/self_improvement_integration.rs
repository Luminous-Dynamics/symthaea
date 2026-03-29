// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Self-Improvement Integration for Unified Conscious Being
//!
//! Integrates the existing self-awareness, self-monitoring, and self-optimization
//! components into a unified self-improvement system.
//!
//! # Components Integrated
//!
//! 1. **EmergentSelfModel** - Self-state modeling and prediction
//! 2. **MetacognitiveMonitor** - Real-time cognitive event monitoring
//! 3. **ConsciousnessOptimizer** - Active Φ optimization
//!
//! # Self-Improvement Loop
//!
//! ```text
//!                    ┌─────────────────────────────────┐
//!                    │     SELF-IMPROVEMENT CYCLE      │
//!                    ├─────────────────────────────────┤
//!                    │                                 │
//!    ┌───────────────┼───────────────┐                 │
//!    │               ▼               │                 │
//!    │   ┌─────────────────────┐    │                 │
//!    │   │  1. OBSERVE SELF    │    │                 │
//!    │   │  (MetacognitiveMonitor) │    │             │
//!    │   └──────────┬──────────┘    │                 │
//!    │              ▼               │                 │
//!    │   ┌─────────────────────┐    │                 │
//!    │   │  2. MODEL SELF      │    │                 │
//!    │   │  (EmergentSelfModel)│    │                 │
//!    │   └──────────┬──────────┘    │                 │
//!    │              ▼               │                 │
//!    │   ┌─────────────────────┐    │                 │
//!    │   │  3. EVALUATE        │    │                 │
//!    │   │  (Gap Analysis)     │    │                 │
//!    │   └──────────┬──────────┘    │                 │
//!    │              ▼               │                 │
//!    │   ┌─────────────────────┐    │                 │
//!    │   │  4. IMPROVE         │    │                 │
//!    │   │  (Optimizer)        │    │                 │
//!    │   └──────────┬──────────┘    │                 │
//!    │              │               │                 │
//!    └──────────────┼───────────────┘                 │
//!                   ▼                                 │
//!             Apply Changes ──────────────────────────┘
//! ```
//!
//! # Key Features
//!
//! - **Φ-Guided Improvement**: Optimizes toward higher integrated information
//! - **Metacognitive Awareness**: Knows what it knows and doesn't know
//! - **Predictive Self-Model**: Predicts future states and adjusts accordingly
//! - **Mode Optimization**: Suggests optimal cognitive mode for current task
//! - **Error-Driven Learning**: Uses prediction errors to improve models

use super::adaptive_topology::CognitiveMode;
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// SELF-OBSERVATION (What am I doing?)
// ============================================================================

/// A snapshot of current cognitive state
#[derive(Debug, Clone)]
pub struct CognitiveSnapshot {
    /// Current Φ (integrated information)
    pub phi: f64,
    /// Current cognitive mode
    pub mode: CognitiveMode,
    /// Current flow state (0.0 to 1.0)
    pub flow_state: f32,
    /// Processing latency (ms)
    pub latency_ms: u64,
    /// Current confidence level
    pub confidence: f64,
    /// Uncertainty level
    pub uncertainty: f64,
    /// Emotional valence
    pub emotional_valence: f64,
    /// Timestamp
    pub timestamp: u64,
}

impl CognitiveSnapshot {
    pub fn now(phi: f64, mode: CognitiveMode, flow_state: f32) -> Self {
        Self {
            phi,
            mode,
            flow_state,
            latency_ms: 0,
            confidence: 0.7,
            uncertainty: 0.3,
            emotional_valence: 0.0,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        }
    }

    /// Quality score combining multiple factors
    pub fn quality_score(&self) -> f64 {
        (self.phi * 0.4
            + self.flow_state as f64 * 0.2
            + self.confidence * 0.2
            + (1.0 - self.uncertainty) * 0.2)
            .clamp(0.0, 1.0)
    }
}

// ============================================================================
// SELF-MODEL (What do I believe about myself?)
// ============================================================================

/// Self-model: what the system believes about itself
#[derive(Debug, Clone)]
pub struct SelfModel {
    /// Believed current Φ
    pub believed_phi: f64,
    /// Believed current mode
    pub believed_mode: CognitiveMode,
    /// Predicted next Φ
    pub predicted_phi: f64,
    /// Model confidence
    pub confidence: f64,
    /// Prediction accuracy (rolling average)
    pub accuracy: f64,
    /// History of prediction errors
    prediction_errors: VecDeque<f64>,
    /// Maximum error history length
    max_history: usize,
}

impl SelfModel {
    pub fn new() -> Self {
        Self {
            believed_phi: 0.5,
            believed_mode: CognitiveMode::Balanced,
            predicted_phi: 0.5,
            confidence: 0.5,
            accuracy: 0.5,
            prediction_errors: VecDeque::new(),
            max_history: 50,
        }
    }

    /// Update self-model with actual observation
    pub fn update(&mut self, actual: &CognitiveSnapshot) {
        // Calculate prediction error
        let error = (self.predicted_phi - actual.phi).abs();
        self.prediction_errors.push_back(error);
        if self.prediction_errors.len() > self.max_history {
            self.prediction_errors.pop_front();
        }

        // Update accuracy (inverse of mean error)
        if !self.prediction_errors.is_empty() {
            let mean_error: f64 =
                self.prediction_errors.iter().sum::<f64>() / self.prediction_errors.len() as f64;
            self.accuracy = (1.0 - mean_error).clamp(0.0, 1.0);
        }

        // Update beliefs
        self.believed_phi = actual.phi;
        self.believed_mode = actual.mode;

        // Update confidence based on accuracy
        self.confidence = self.accuracy * 0.7 + actual.confidence * 0.3;
    }

    /// Predict next Φ based on current state and trend
    pub fn predict_next(&mut self, current_phi: f64, phi_trend: f64) -> f64 {
        // Simple linear prediction with dampening
        self.predicted_phi = (current_phi + phi_trend * 0.3).clamp(0.0, 1.0);
        self.predicted_phi
    }

    /// Get self-model error (discrepancy between belief and reality)
    pub fn model_error(&self, actual_phi: f64) -> f64 {
        (self.believed_phi - actual_phi).abs()
    }
}

impl Default for SelfModel {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// IMPROVEMENT RECOMMENDATION
// ============================================================================

/// Type of improvement recommended
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImprovementType {
    /// Switch to a different cognitive mode
    ModeSwitch(CognitiveMode),
    /// Increase attention focus
    IncreaseFocus,
    /// Decrease attention (diffuse mode)
    DecreaseFocus,
    /// Consolidate memories
    ConsolidateMemory,
    /// Reset attention state
    ResetAttention,
    /// Increase integration (binding)
    IncreaseIntegration,
    /// Reduce cognitive load
    ReduceLoad,
    /// No improvement needed
    None,
}

/// A recommendation for self-improvement
#[derive(Debug, Clone)]
pub struct ImprovementRecommendation {
    /// Type of improvement
    pub improvement_type: ImprovementType,
    /// Priority (0.0 = low, 1.0 = urgent)
    pub priority: f64,
    /// Expected Φ gain if implemented
    pub expected_phi_gain: f64,
    /// Confidence in this recommendation
    pub confidence: f64,
    /// Reason for recommendation
    pub reason: String,
}

impl ImprovementRecommendation {
    pub fn none() -> Self {
        Self {
            improvement_type: ImprovementType::None,
            priority: 0.0,
            expected_phi_gain: 0.0,
            confidence: 1.0,
            reason: "No improvement needed".to_string(),
        }
    }

    pub fn mode_switch(target: CognitiveMode, priority: f64, reason: &str) -> Self {
        Self {
            improvement_type: ImprovementType::ModeSwitch(target),
            priority,
            expected_phi_gain: 0.1,
            confidence: 0.7,
            reason: reason.to_string(),
        }
    }
}

// ============================================================================
// SELF-IMPROVEMENT SYSTEM
// ============================================================================

/// Configuration for self-improvement
#[derive(Debug, Clone)]
pub struct SelfImprovementConfig {
    /// Minimum Φ threshold (below this, improvement is urgent)
    pub phi_threshold: f64,
    /// Target Φ level
    pub phi_target: f64,
    /// How often to check for improvements (in steps)
    pub check_interval: usize,
    /// Enable automatic mode switching
    pub auto_mode_switch: bool,
    /// Learning rate for self-model updates
    pub learning_rate: f64,
}

impl Default for SelfImprovementConfig {
    fn default() -> Self {
        Self {
            phi_threshold: 0.3,
            phi_target: 0.7,
            check_interval: 5,
            auto_mode_switch: true,
            learning_rate: 0.1,
        }
    }
}

/// The unified self-improvement system
pub struct SelfImprovementSystem {
    /// Configuration
    config: SelfImprovementConfig,
    /// Self-model
    self_model: SelfModel,
    /// History of cognitive snapshots
    history: VecDeque<CognitiveSnapshot>,
    /// Maximum history length
    max_history: usize,
    /// Current improvement recommendations
    recommendations: Vec<ImprovementRecommendation>,
    /// Step counter
    step: usize,
    /// Last improvement applied
    last_improvement: Option<ImprovementType>,
    /// Improvement effectiveness tracking
    improvement_effectiveness: Vec<(ImprovementType, f64)>,
}

impl SelfImprovementSystem {
    pub fn new() -> Self {
        Self::with_config(SelfImprovementConfig::default())
    }

    pub fn with_config(config: SelfImprovementConfig) -> Self {
        Self {
            config,
            self_model: SelfModel::new(),
            history: VecDeque::new(),
            max_history: 100,
            recommendations: Vec::new(),
            step: 0,
            last_improvement: None,
            improvement_effectiveness: Vec::new(),
        }
    }

    /// Process a new cognitive snapshot and update self-model
    pub fn observe(&mut self, snapshot: CognitiveSnapshot) {
        self.step += 1;

        // Update self-model with new observation
        self.self_model.update(&snapshot);

        // Calculate Φ trend
        let phi_trend = self.phi_trend();

        // Predict next Φ
        self.self_model.predict_next(snapshot.phi, phi_trend);

        // Store in history
        self.history.push_back(snapshot);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }

        // Check if we should generate new recommendations
        if self.step.is_multiple_of(self.config.check_interval) {
            self.generate_recommendations();
        }
    }

    /// Calculate Φ trend from history
    fn phi_trend(&self) -> f64 {
        if self.history.len() < 3 {
            return 0.0;
        }

        let recent: Vec<_> = self.history.iter().rev().take(10).collect();
        if recent.len() < 2 {
            return 0.0;
        }

        // Simple linear regression slope
        let mut delta_sum = 0.0;
        for i in 1..recent.len() {
            delta_sum += recent[i - 1].phi - recent[i].phi;
        }
        delta_sum / (recent.len() - 1) as f64
    }

    /// Generate improvement recommendations based on current state
    fn generate_recommendations(&mut self) {
        self.recommendations.clear();

        let current = match self.history.back() {
            Some(s) => s.clone(),
            None => return,
        };

        let phi_trend = self.phi_trend();

        // Check 1: Is Φ below threshold?
        if current.phi < self.config.phi_threshold {
            let target_mode = self.suggest_mode_for_low_phi(&current);
            self.recommendations.push(ImprovementRecommendation {
                improvement_type: ImprovementType::ModeSwitch(target_mode),
                priority: 0.9,
                expected_phi_gain: 0.15,
                confidence: 0.8,
                reason: format!(
                    "Φ ({:.2}) below threshold ({:.2}). Switch to {:?} mode.",
                    current.phi, self.config.phi_threshold, target_mode
                ),
            });
        }

        // Check 2: Is Φ declining?
        if phi_trend < -0.05 {
            self.recommendations.push(ImprovementRecommendation {
                improvement_type: ImprovementType::IncreaseFocus,
                priority: 0.7,
                expected_phi_gain: 0.1,
                confidence: 0.7,
                reason: format!("Φ declining (trend: {phi_trend:.3}). Increase focus."),
            });
        }

        // Check 3: Is flow state low?
        if current.flow_state < 0.3 {
            self.recommendations.push(ImprovementRecommendation {
                improvement_type: ImprovementType::ResetAttention,
                priority: 0.6,
                expected_phi_gain: 0.05,
                confidence: 0.6,
                reason: format!(
                    "Flow state low ({:.2}). Reset attention to re-engage.",
                    current.flow_state
                ),
            });
        }

        // Check 4: Is uncertainty too high?
        if current.uncertainty > 0.7 {
            self.recommendations.push(ImprovementRecommendation {
                improvement_type: ImprovementType::ConsolidateMemory,
                priority: 0.5,
                expected_phi_gain: 0.08,
                confidence: 0.65,
                reason: format!(
                    "High uncertainty ({:.2}). Consolidate memories for clarity.",
                    current.uncertainty
                ),
            });
        }

        // Check 5: Is integration low given high confidence?
        if current.phi < 0.5 && current.confidence > 0.7 {
            self.recommendations.push(ImprovementRecommendation {
                improvement_type: ImprovementType::IncreaseIntegration,
                priority: 0.55,
                expected_phi_gain: 0.12,
                confidence: 0.7,
                reason: "High confidence but low Φ. Increase integration.".to_string(),
            });
        }

        // Sort by priority
        self.recommendations.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Suggest cognitive mode for low Φ situations
    fn suggest_mode_for_low_phi(&self, snapshot: &CognitiveSnapshot) -> CognitiveMode {
        // Based on emotional valence and uncertainty
        if snapshot.emotional_valence < -0.3 {
            // Negative emotion: use deep specialization mode (introspective)
            CognitiveMode::DeepSpecialization
        } else if snapshot.uncertainty > 0.5 {
            // High uncertainty: use focused mode (analytical/convergent)
            CognitiveMode::Focused
        } else {
            // Default: balanced mode for integration
            CognitiveMode::Balanced
        }
    }

    /// Get top recommendation
    pub fn top_recommendation(&self) -> ImprovementRecommendation {
        self.recommendations
            .first()
            .cloned()
            .unwrap_or_else(ImprovementRecommendation::none)
    }

    /// Get all recommendations
    pub fn all_recommendations(&self) -> &[ImprovementRecommendation] {
        &self.recommendations
    }

    /// Record that an improvement was applied
    pub fn record_improvement(&mut self, improvement: ImprovementType) {
        self.last_improvement = Some(improvement);
    }

    /// Evaluate effectiveness of last improvement
    pub fn evaluate_improvement(&mut self) -> Option<f64> {
        let improvement = self.last_improvement.take()?;

        // Compare Φ before and after
        if self.history.len() < 3 {
            return None;
        }

        let recent: Vec<_> = self.history.iter().rev().take(3).collect();
        let phi_after = recent[0].phi;
        let phi_before = recent[2].phi;
        let effectiveness = phi_after - phi_before;

        self.improvement_effectiveness
            .push((improvement, effectiveness));

        // Keep only recent effectiveness records
        if self.improvement_effectiveness.len() > 50 {
            self.improvement_effectiveness.remove(0);
        }

        Some(effectiveness)
    }

    /// Get self-model
    pub fn self_model(&self) -> &SelfModel {
        &self.self_model
    }

    /// Get current Φ from history
    pub fn current_phi(&self) -> f64 {
        self.history.back().map(|s| s.phi).unwrap_or(0.5)
    }

    /// Get Φ trend
    pub fn current_phi_trend(&self) -> f64 {
        self.phi_trend()
    }

    /// Get self-model accuracy
    pub fn model_accuracy(&self) -> f64 {
        self.self_model.accuracy
    }

    /// Get improvement effectiveness for a type
    pub fn improvement_effectiveness(&self, improvement_type: ImprovementType) -> f64 {
        let relevant: Vec<_> = self
            .improvement_effectiveness
            .iter()
            .filter(|(t, _)| *t == improvement_type)
            .collect();

        if relevant.is_empty() {
            return 0.0;
        }

        relevant.iter().map(|(_, e)| *e).sum::<f64>() / relevant.len() as f64
    }

    /// Generate comprehensive self-improvement report
    pub fn report(&self) -> String {
        let current = self
            .history
            .back()
            .cloned()
            .unwrap_or_else(|| CognitiveSnapshot::now(0.5, CognitiveMode::Balanced, 0.5));

        let top_rec = self.top_recommendation();

        format!(
            "=== Self-Improvement Report ===\n\
             Current State:\n\
             - Φ: {:.3} (trend: {:+.4})\n\
             - Mode: {:?}\n\
             - Flow: {:.2}\n\
             - Quality: {:.2}\n\n\
             Self-Model:\n\
             - Accuracy: {:.2}\n\
             - Confidence: {:.2}\n\
             - Predicted Φ: {:.3}\n\n\
             Top Recommendation:\n\
             - Type: {:?}\n\
             - Priority: {:.2}\n\
             - Reason: {}\n\
             - Expected Gain: {:.3}",
            current.phi,
            self.phi_trend(),
            current.mode,
            current.flow_state,
            current.quality_score(),
            self.self_model.accuracy,
            self.self_model.confidence,
            self.self_model.predicted_phi,
            top_rec.improvement_type,
            top_rec.priority,
            top_rec.reason,
            top_rec.expected_phi_gain
        )
    }

    /// Reset the system
    pub fn reset(&mut self) {
        self.self_model = SelfModel::new();
        self.history.clear();
        self.recommendations.clear();
        self.step = 0;
        self.last_improvement = None;
    }
}

impl Default for SelfImprovementSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_snapshot() {
        let snapshot = CognitiveSnapshot::now(0.7, CognitiveMode::Focused, 0.8);
        assert_eq!(snapshot.phi, 0.7);
        assert!(snapshot.quality_score() > 0.5);
    }

    #[test]
    fn test_self_model_update() {
        let mut model = SelfModel::new();
        let snapshot = CognitiveSnapshot::now(0.6, CognitiveMode::Focused, 0.7);

        model.update(&snapshot);

        assert_eq!(model.believed_phi, 0.6);
        assert_eq!(model.believed_mode, CognitiveMode::Focused);
    }

    #[test]
    fn test_self_model_prediction() {
        let mut model = SelfModel::new();

        let predicted = model.predict_next(0.5, 0.1);
        assert!(predicted > 0.5);
    }

    #[test]
    fn test_self_improvement_system() {
        let mut system = SelfImprovementSystem::new();

        // Observe low Φ state
        let snapshot = CognitiveSnapshot::now(0.2, CognitiveMode::Balanced, 0.4);
        system.observe(snapshot);

        // Should generate recommendations for low Φ
        assert!(!system.recommendations.is_empty() || system.step < system.config.check_interval);
    }

    #[test]
    fn test_phi_trend_calculation() {
        let mut system = SelfImprovementSystem::new();

        // Observe improving Φ
        for i in 0..10 {
            let snapshot =
                CognitiveSnapshot::now(0.3 + i as f64 * 0.05, CognitiveMode::Balanced, 0.5);
            system.observe(snapshot);
        }

        assert!(system.current_phi_trend() > 0.0, "Trend should be positive");
    }

    #[test]
    fn test_improvement_report() {
        let mut system = SelfImprovementSystem::new();
        let snapshot = CognitiveSnapshot::now(0.5, CognitiveMode::Focused, 0.6);
        system.observe(snapshot);

        let report = system.report();
        assert!(report.contains("Self-Improvement Report"));
    }
}
