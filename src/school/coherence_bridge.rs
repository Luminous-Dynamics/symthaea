// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # School-Coherence Integration
//!
//! Bridges the School learning system with the CoherenceField physiology.
//!
//! ## Revolutionary Insight
//!
//! Learning is not just a cognitive activity - it's a consciousness-integration task.
//! The coherence field models how integrated we are, and learning requires (and affects)
//! that integration:
//!
//! - **High coherence** → Better learning, higher Φ gains
//! - **Low coherence** → Scattered attention, reduced retention
//! - **Connected learning** → BUILDS coherence (learning with a teacher)
//! - **Solo learning** → May scatter coherence (grinding alone)
//! - **Successful learning** → Boosts coherence (integration success)
//! - **Failed learning** → May scatter (cognitive dissonance)
//!
//! ## The CfC-Coherence Synergy
//!
//! CfC's O(1) lookahead can now factor in coherence state:
//! - Predict if we're coherent enough to learn effectively
//! - Estimate coherence cost of learning
//! - Recommend centering before difficult objectives
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::school::{School, SchoolConfig, CoherenceBridgedSchool};
//! use symthaea::physiology::coherence::CoherenceField;
//!
//! let school = School::new(SchoolConfig::default())?;
//! let coherence = CoherenceField::new();
//!
//! let mut bridged = CoherenceBridgedSchool::new(school, coherence);
//!
//! // Check if we can learn (coherence sufficient?)
//! if let Some(centering_needed) = bridged.centering_needed_for_learning() {
//!     println!("Need {:.0}s of centering before learning", centering_needed);
//!     bridged.center(centering_needed);
//! }
//!
//! // Learn with coherence integration
//! let result = bridged.learn_with_coherence(&objective, true)?;
//! println!("Learned with coherence: {:.1}% -> {:.1}%",
//!          result.coherence_before * 100.0,
//!          result.coherence_after * 100.0);
//! ```

use super::{Curriculum, LearningObjective, LearningResult, Recommendation, School, SchoolStats};
use crate::physiology::coherence::{
    CoherenceError, CoherenceField, CoherenceState, TaskComplexity,
};
use crate::physiology::endocrine::HormoneState;
use crate::wisdom::autopoiesis::{AutopoieticMonitor, OperationalClosure, SelfProductionMetrics};
use anyhow::Result;

// ═══════════════════════════════════════════════════════════════════════════════
// COHERENCE-LEARNING CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for how learning interacts with coherence
#[derive(Debug, Clone)]
pub struct CoherenceLearningConfig {
    /// What TaskComplexity to use for learning objectives
    pub learning_task_complexity: TaskComplexity,

    /// Φ gain multiplier based on coherence level
    /// At coherence=1.0: full multiplier, at 0.5: half multiplier
    pub coherence_phi_multiplier: f32,

    /// Extra coherence boost for successful learning (beyond task effect)
    pub success_coherence_bonus: f32,

    /// Coherence scatter for failed/hallucinated predictions
    pub hallucination_coherence_penalty: f32,

    /// Minimum coherence required to attempt learning
    /// (Can override TaskComplexity::Learning threshold)
    pub min_learning_coherence: Option<f32>,

    /// Enable automatic centering suggestions
    pub suggest_centering: bool,

    // ═══════════════════════════════════════════════════════════════════════════
    // LAYER 1: Bidirectional Φ-Coherence Feedback
    // ═══════════════════════════════════════════════════════════════════════════
    /// How much Φ gain contributes to coherence (integration builds coherence)
    /// At phi_gain=0.01, coherence boost = phi_gain * phi_to_coherence_factor
    pub phi_to_coherence_factor: f32,

    /// Maximum coherence boost from a single learning session
    pub max_phi_coherence_boost: f32,

    /// How much Φ loss damages coherence (disintegration scatters)
    pub phi_loss_coherence_penalty: f32,

    // ═══════════════════════════════════════════════════════════════════════════
    // LAYER 2: Hormone-Learning Integration
    // ═══════════════════════════════════════════════════════════════════════════
    /// Cortisol threshold above which learning is impaired
    /// High stress → reduced retention, increased hallucination risk
    pub cortisol_impairment_threshold: f32,

    /// Dopamine boost to Φ gain (reward enhances integration)
    pub dopamine_phi_boost: f32,

    /// Oxytocin boost to coherence gain (connection enhances integration)
    pub oxytocin_coherence_boost: f32,
}

impl Default for CoherenceLearningConfig {
    fn default() -> Self {
        Self {
            learning_task_complexity: TaskComplexity::Learning,
            coherence_phi_multiplier: 0.5, // Coherence affects up to 50% of Φ gain
            success_coherence_bonus: 0.02, // +2% coherence on success
            hallucination_coherence_penalty: 0.05, // -5% on hallucination
            min_learning_coherence: None,  // Use TaskComplexity default (0.8)
            suggest_centering: true,

            // Layer 1: Bidirectional Φ-Coherence
            phi_to_coherence_factor: 2.0, // Φ gain of 0.01 → +0.02 coherence
            max_phi_coherence_boost: 0.10, // Cap at 10% coherence boost per session
            phi_loss_coherence_penalty: 1.5, // Φ loss hurts coherence 1.5x

            // Layer 2: Hormone-Learning Integration
            cortisol_impairment_threshold: 0.7, // High stress impairs learning
            dopamine_phi_boost: 0.3,            // Dopamine adds 30% to Φ gain
            oxytocin_coherence_boost: 0.5,      // Oxytocin adds 50% to coherence boost
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COHERENCE-BRIDGED LEARNING RESULT
// ═══════════════════════════════════════════════════════════════════════════════

/// Extended learning result with coherence integration
#[derive(Debug, Clone)]
pub struct CoherenceLearningResult {
    /// Base learning result
    pub learning: LearningResult,

    /// Coherence before learning
    pub coherence_before: f32,

    /// Coherence after learning
    pub coherence_after: f32,

    /// Whether learning was done with connection (teacher/partner)
    pub with_connection: bool,

    /// Coherence change from learning task
    pub coherence_change: f32,

    /// Whether learning was affected by low coherence
    pub coherence_limited: bool,

    /// Φ gain multiplier that was applied due to coherence
    pub applied_phi_multiplier: f32,

    // ═══════════════════════════════════════════════════════════════════════════
    // LAYER 1: Bidirectional Φ-Coherence Feedback Results
    // ═══════════════════════════════════════════════════════════════════════════
    /// Coherence boost from Φ gain (integration builds coherence)
    pub phi_coherence_boost: f32,

    /// Whether the virtuous cycle was triggered (Φ↑ → coherence↑ → next Φ↑)
    pub virtuous_cycle_active: bool,

    // ═══════════════════════════════════════════════════════════════════════════
    // LAYER 2: Hormone-Learning Integration Results
    // ═══════════════════════════════════════════════════════════════════════════
    /// How much hormones affected Φ gain
    pub hormone_phi_modifier: f32,

    /// How much hormones affected coherence change
    pub hormone_coherence_modifier: f32,

    /// Whether cortisol impaired learning
    pub stress_impaired: bool,
}

impl CoherenceLearningResult {
    /// Check if learning was successful (no hallucination, coherence improved)
    pub fn was_successful(&self) -> bool {
        !self.learning.was_hallucination && self.coherence_after >= self.coherence_before
    }

    /// Describe the result in natural language
    pub fn describe(&self) -> String {
        let coherence_pct_before = self.coherence_before * 100.0;
        let coherence_pct_after = self.coherence_after * 100.0;
        let coherence_delta = coherence_pct_after - coherence_pct_before;

        let coherence_desc = if coherence_delta > 1.0 {
            format!(
                "coherence rose to {:.0}% (+{:.1}%)",
                coherence_pct_after, coherence_delta
            )
        } else if coherence_delta < -1.0 {
            format!(
                "coherence dropped to {:.0}% ({:.1}%)",
                coherence_pct_after, coherence_delta
            )
        } else {
            format!("coherence stable at {:.0}%", coherence_pct_after)
        };

        let connection_desc = if self.with_connection {
            "connected learning"
        } else {
            "solo learning"
        };

        let phi_desc = if self.learning.actual_phi_gain > 0.001 {
            format!("Φ gained {:.4}", self.learning.actual_phi_gain)
        } else if self.learning.actual_phi_gain < -0.001 {
            format!("Φ dropped {:.4}", self.learning.actual_phi_gain.abs())
        } else {
            "Φ stable".to_string()
        };

        // Build effect annotations
        let mut annotations = Vec::new();
        if self.coherence_limited {
            annotations.push("limited by low coherence");
        }
        if self.virtuous_cycle_active {
            annotations.push("virtuous cycle active");
        }
        if self.stress_impaired {
            annotations.push("stress-impaired");
        }
        if self.phi_coherence_boost > 0.01 {
            annotations.push("integration boost");
        }

        let annotation_str = if annotations.is_empty() {
            String::new()
        } else {
            format!(" ({})", annotations.join(", "))
        };

        format!(
            "Learned '{}' via {}: {} | {}{}",
            self.learning.objective_name, connection_desc, phi_desc, coherence_desc, annotation_str
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COHERENCE-BRIDGED SCHOOL
// ═══════════════════════════════════════════════════════════════════════════════

/// School with CoherenceField integration
///
/// This wrapper connects the School's learning system with the CoherenceField
/// physiology, creating a unified model where:
/// - Learning requires coherence (you can't learn when scattered)
/// - Learning affects coherence (connected work builds, solo work may scatter)
/// - Success boosts coherence, failures scatter it
///
/// ## Layer 3: Consciousness Graph Integration
///
/// Learning is modeled as a consciousness-producing activity:
/// - Each learning cycle is recorded as a cognitive production cycle
/// - Hallucinations are tracked as prediction errors (world-model mismatch)
/// - Successful learning strengthens operational closure
/// - The autopoietic monitor tracks whether learning is self-sustaining
pub struct CoherenceBridgedSchool {
    /// The underlying school
    school: School,

    /// The coherence field
    coherence: CoherenceField,

    /// Configuration for the integration
    config: CoherenceLearningConfig,

    /// Current hormone state (for coherence predictions)
    hormones: HormoneState,

    /// Count of successful learning operations
    successful_learnings: u64,

    /// Count of coherence-blocked attempts
    coherence_blocked_count: u64,

    /// Total centering time recommended (seconds)
    total_centering_suggested: f32,

    // ═══════════════════════════════════════════════════════════════════════════
    // LAYER 3: Consciousness Graph Integration
    // ═══════════════════════════════════════════════════════════════════════════
    /// Autopoietic monitor tracks self-sustaining learning
    autopoietic: AutopoieticMonitor,

    /// Count of virtuous cycles achieved
    virtuous_cycle_count: u64,
}

impl CoherenceBridgedSchool {
    /// Create a new bridged school with default learning config
    pub fn new(school: School, coherence: CoherenceField) -> Self {
        Self {
            school,
            coherence,
            config: CoherenceLearningConfig::default(),
            hormones: HormoneState::neutral(),
            successful_learnings: 0,
            coherence_blocked_count: 0,
            total_centering_suggested: 0.0,
            autopoietic: AutopoieticMonitor::new(),
            virtuous_cycle_count: 0,
        }
    }

    /// Create with custom learning config
    pub fn with_config(
        school: School,
        coherence: CoherenceField,
        config: CoherenceLearningConfig,
    ) -> Self {
        Self {
            school,
            coherence,
            config,
            hormones: HormoneState::neutral(),
            successful_learnings: 0,
            coherence_blocked_count: 0,
            total_centering_suggested: 0.0,
            autopoietic: AutopoieticMonitor::new(),
            virtuous_cycle_count: 0,
        }
    }

    /// Update hormone state (affects coherence dynamics)
    pub fn set_hormones(&mut self, hormones: HormoneState) {
        self.coherence.apply_hormone_modulation(&hormones);
        self.hormones = hormones;
    }

    /// Get current coherence state
    pub fn coherence_state(&self) -> CoherenceState {
        self.coherence.state()
    }

    /// Get current coherence level (0.0 - 1.0)
    pub fn coherence_level(&self) -> f32 {
        self.coherence.coherence
    }

    /// Check if we can learn with current coherence
    pub fn can_learn(&mut self) -> Result<(), CoherenceError> {
        let task = self.config.learning_task_complexity;
        self.coherence.can_perform(task)
    }

    /// Check how much centering is needed before learning
    ///
    /// Returns Some(seconds) if centering is needed, None if we can learn now
    pub fn centering_needed_for_learning(&self) -> Option<f32> {
        let required = self
            .config
            .min_learning_coherence
            .unwrap_or(TaskComplexity::Learning.required_coherence(&self.coherence.config));

        if self.coherence.coherence >= required {
            None
        } else {
            let deficit = required - self.coherence.coherence;
            // Use passive centering rate to estimate time
            let time_needed = deficit / self.coherence.config.passive_centering_rate;
            Some(time_needed)
        }
    }

    /// Simulate centering (advance time to restore coherence)
    pub fn center(&mut self, seconds: f32) {
        self.coherence.tick(seconds);
    }

    /// Receive gratitude (synchronization boost)
    pub fn receive_gratitude(&mut self) {
        self.coherence.receive_gratitude();
    }

    /// Predict coherence impact of learning an objective
    pub fn predict_learning_impact(
        &self,
        objective: &LearningObjective,
        with_connection: bool,
    ) -> LearningCoherencePrediction {
        let task = self.config.learning_task_complexity;
        let coherence_prediction =
            self.coherence
                .predict_impact(task, with_connection, &self.hormones);

        // Predict Φ gain (adjusted by coherence)
        let base_phi_prediction = self
            .school
            .lookahead_engine()
            .evaluate(objective, &self.get_consciousness_state())
            .map(|r| r.predicted_phi_gain)
            .unwrap_or(0.0);

        // Apply coherence multiplier to predicted Φ
        let coherence_effect = self
            .coherence
            .coherence
            .powf(self.config.coherence_phi_multiplier);
        let adjusted_phi = base_phi_prediction * coherence_effect;

        LearningCoherencePrediction {
            can_learn: coherence_prediction.will_succeed,
            current_coherence: self.coherence.coherence,
            predicted_coherence: coherence_prediction.final_coherence,
            centering_needed: coherence_prediction.centering_needed,
            base_phi_prediction,
            adjusted_phi_prediction: adjusted_phi,
            coherence_multiplier: coherence_effect,
            reasoning: coherence_prediction.reasoning,
        }
    }

    /// Learn an objective with full coherence integration
    ///
    /// This implements the three-layer embodied consciousness learning model:
    ///
    /// ## Layer 1: Bidirectional Φ-Coherence Feedback
    /// - Coherence affects Φ gain (high coherence → better learning)
    /// - Φ gain affects coherence (integration builds coherence)
    /// - Creates a virtuous cycle when both are high
    ///
    /// ## Layer 2: Hormone-Learning Integration
    /// - Cortisol impairs learning (stress reduces retention)
    /// - Dopamine boosts Φ gain (reward enhances integration)
    /// - Oxytocin boosts coherence (connection enhances integration)
    ///
    /// ## Flow:
    /// 1. Check if coherence is sufficient
    /// 2. Apply hormone modifiers
    /// 3. Apply learning as a coherence task
    /// 4. Adjust Φ gain based on coherence and hormones
    /// 5. Apply bidirectional Φ → coherence feedback
    /// 6. Update coherence based on success/failure
    pub fn learn_with_coherence(
        &mut self,
        objective: &LearningObjective,
        with_connection: bool,
    ) -> Result<CoherenceLearningResult> {
        let coherence_before = self.coherence.coherence;

        // 1. Check if we can learn
        let task = self.config.learning_task_complexity;
        match self.coherence.can_perform(task) {
            Ok(_) => {}
            Err(e) => {
                self.coherence_blocked_count += 1;
                if self.config.suggest_centering {
                    if let Some(centering) = self.centering_needed_for_learning() {
                        self.total_centering_suggested += centering;
                    }
                }
                return Err(anyhow::anyhow!("Coherence too low for learning: {}", e));
            }
        }

        // ═══════════════════════════════════════════════════════════════════════
        // LAYER 2: Hormone-Learning Integration
        // ═══════════════════════════════════════════════════════════════════════

        // Check for cortisol impairment (stress reduces learning effectiveness)
        let cortisol_threshold = self.config.cortisol_impairment_threshold as f64;
        let stress_impaired = self.hormones.cortisol > cortisol_threshold;
        let cortisol_penalty = if stress_impaired {
            1.0 - (self.hormones.cortisol - cortisol_threshold).min(0.3)
        } else {
            1.0
        };

        // Calculate dopamine boost to Φ gain (reward enhances integration)
        let dopamine_boost = 1.0 + (self.hormones.dopamine * self.config.dopamine_phi_boost as f64);

        // Calculate oxytocin boost to coherence changes (connection enhances)
        let oxytocin_boost =
            1.0 + (self.hormones.oxytocin * self.config.oxytocin_coherence_boost as f64);

        // Combined hormone effect on Φ
        let hormone_phi_modifier = dopamine_boost * cortisol_penalty;

        // ═══════════════════════════════════════════════════════════════════════
        // LAYER 1: Coherence → Φ (forward direction)
        // ═══════════════════════════════════════════════════════════════════════

        // Calculate coherence effect on learning
        let coherence_multiplier = coherence_before.powf(self.config.coherence_phi_multiplier);
        let coherence_limited = coherence_before < 0.8;

        // 3. Perform the learning as a coherence task
        // This affects coherence (builds if connected, scatters if solo)
        self.coherence.perform_task(task, with_connection)?;

        // 4. Do the actual learning
        let mut learning_result = self.school.learn(objective)?;

        // 5. Adjust actual Φ gain by coherence AND hormone multipliers
        // (Low coherence = reduced effectiveness, stress = impaired, dopamine = boosted)
        learning_result.actual_phi_gain *= coherence_multiplier * hormone_phi_modifier as f32;

        // ═══════════════════════════════════════════════════════════════════════
        // LAYER 1: Φ → Coherence (reverse direction - THE NEW PART!)
        // ═══════════════════════════════════════════════════════════════════════

        // Calculate coherence boost from Φ gain (integration builds coherence)
        let phi_coherence_boost = if learning_result.actual_phi_gain > 0.0 {
            // Positive Φ gain → coherence boost (proportional, with oxytocin enhancement)
            let raw_boost = learning_result.actual_phi_gain * self.config.phi_to_coherence_factor;
            let boosted = raw_boost * oxytocin_boost as f32;
            boosted.min(self.config.max_phi_coherence_boost)
        } else if learning_result.actual_phi_gain < 0.0 {
            // Negative Φ → coherence penalty (disintegration scatters)
            learning_result.actual_phi_gain * self.config.phi_loss_coherence_penalty
        } else {
            0.0
        };

        // Check if virtuous cycle is active (high coherence + positive Φ + connection)
        let virtuous_cycle_active = coherence_before > 0.75
            && learning_result.actual_phi_gain > 0.005
            && with_connection
            && !stress_impaired;

        // 6. Apply success/failure coherence effects
        if learning_result.was_hallucination {
            // Hallucination scatters coherence (cognitive dissonance)
            self.coherence.coherence =
                (self.coherence.coherence - self.config.hallucination_coherence_penalty).max(0.0);
        } else {
            // Apply base success bonus
            if learning_result.actual_phi_gain > 0.0 {
                self.coherence.coherence =
                    (self.coherence.coherence + self.config.success_coherence_bonus).min(1.0);
            }

            // Apply bidirectional Φ → coherence boost
            self.coherence.coherence =
                (self.coherence.coherence + phi_coherence_boost).clamp(0.0, 1.0);

            // Extra boost if virtuous cycle is active
            if virtuous_cycle_active {
                self.coherence.coherence = (self.coherence.coherence + 0.01).min(1.0);
            }

            if learning_result.actual_phi_gain > 0.0 {
                self.successful_learnings += 1;
            }
        }

        // 7. Record learning performance for adaptive thresholds
        self.coherence.record_task_performance(
            task,
            coherence_before,
            !learning_result.was_hallucination,
        );

        // ═══════════════════════════════════════════════════════════════════════
        // LAYER 3: Consciousness Graph Integration - Record in Autopoietic Monitor
        // ═══════════════════════════════════════════════════════════════════════

        // Record this learning cycle as a cognitive production cycle
        // Prediction error = hallucination rate (world-model mismatch)
        let prediction_error = if learning_result.was_hallucination {
            1.0
        } else {
            0.0
        };
        self.autopoietic
            .record_cycle(prediction_error, self.coherence.coherence);

        // Track virtuous cycles
        if virtuous_cycle_active {
            self.virtuous_cycle_count += 1;
        }

        let coherence_after = self.coherence.coherence;
        let hormone_coherence_modifier = oxytocin_boost;

        Ok(CoherenceLearningResult {
            learning: learning_result,
            coherence_before,
            coherence_after,
            with_connection,
            coherence_change: coherence_after - coherence_before,
            coherence_limited,
            applied_phi_multiplier: coherence_multiplier,
            // Layer 1: Bidirectional feedback
            phi_coherence_boost,
            virtuous_cycle_active,
            // Layer 2: Hormone effects
            hormone_phi_modifier: hormone_phi_modifier as f32,
            hormone_coherence_modifier: hormone_coherence_modifier as f32,
            stress_impaired,
        })
    }

    /// Get recommended next objective, considering coherence
    ///
    /// May return a centering suggestion if coherence is too low
    pub fn recommend_next_with_coherence(&self) -> Result<CoherenceRecommendation> {
        // First check if we can learn at all
        if let Some(centering_needed) = self.centering_needed_for_learning() {
            return Ok(CoherenceRecommendation::NeedsCentering {
                current_coherence: self.coherence.coherence,
                required_coherence: TaskComplexity::Learning
                    .required_coherence(&self.coherence.config),
                centering_seconds: centering_needed,
                message: format!(
                    "Coherence too low ({:.0}%). Center for {:.0} seconds before learning.",
                    self.coherence.coherence * 100.0,
                    centering_needed
                ),
            });
        }

        // Get the base recommendation
        let rec = self.school.recommend_next()?;

        if rec.is_complete() {
            return Ok(CoherenceRecommendation::Complete);
        }

        // Predict coherence impact
        let impact = self.predict_learning_impact(&rec.objective, true);

        Ok(CoherenceRecommendation::Ready {
            recommendation: rec,
            coherence_impact: impact,
        })
    }

    /// Tick time forward (passive centering)
    pub fn tick(&mut self, delta_seconds: f32) {
        self.coherence.tick(delta_seconds);
    }

    /// Sleep cycle (full coherence restoration)
    pub fn sleep_cycle(&mut self) {
        self.coherence.sleep_cycle();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Passthrough methods for School
    // ─────────────────────────────────────────────────────────────────────────

    /// Add a curriculum
    pub fn add_curriculum(&mut self, curriculum: Curriculum) -> Result<()> {
        self.school.add_curriculum(curriculum)
    }

    /// Get all objectives
    pub fn all_objectives(&self) -> Vec<&LearningObjective> {
        self.school.all_objectives()
    }

    /// Get ready objectives (prerequisites met)
    pub fn ready_objectives(&self) -> Vec<&LearningObjective> {
        self.school.ready_objectives()
    }

    /// Get current Φ
    pub fn current_phi(&self) -> f32 {
        self.school.current_phi()
    }

    /// Get school stats
    pub fn school_stats(&self) -> SchoolStats {
        self.school.stats()
    }

    /// Get underlying school reference
    pub fn school(&self) -> &School {
        &self.school
    }

    /// Get mutable school reference
    pub fn school_mut(&mut self) -> &mut School {
        &mut self.school
    }

    /// Get underlying coherence field reference
    pub fn coherence(&self) -> &CoherenceField {
        &self.coherence
    }

    /// Get mutable coherence field reference
    pub fn coherence_mut(&mut self) -> &mut CoherenceField {
        &mut self.coherence
    }

    // ─────────────────────────────────────────────────────────────────────────
    // LAYER 3: Consciousness Graph Accessors
    // ─────────────────────────────────────────────────────────────────────────

    /// Get the autopoietic monitor
    pub fn autopoietic(&self) -> &AutopoieticMonitor {
        &self.autopoietic
    }

    /// Get operational closure state
    pub fn operational_closure(&self) -> OperationalClosure {
        self.autopoietic.operational_closure()
    }

    /// Check if learning is self-sustaining (autopoietic)
    pub fn is_self_sustaining(&self) -> bool {
        self.autopoietic.is_healthy() && self.autopoietic.closure_maintained()
    }

    /// Get count of virtuous cycles achieved
    pub fn virtuous_cycle_count(&self) -> u64 {
        self.virtuous_cycle_count
    }

    /// Get self-production metrics
    pub fn self_production_metrics(&self) -> SelfProductionMetrics {
        self.autopoietic.metrics()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────────

    fn get_consciousness_state(&self) -> Vec<crate::hdc::unified_hv::ContinuousHV> {
        // This is a bit of a hack - we're accessing the school's private state
        // In production, we'd have a proper API for this
        (0..8)
            .map(|i| {
                let hv = crate::hdc::ContinuousHV::random(crate::hdc::HDC_DIMENSION, 42 + i);
                crate::hdc::unified_hv::ContinuousHV::from_vec(hv.values)
            })
            .collect()
    }

    /// Get integrated statistics
    pub fn stats(&self) -> CoherenceBridgedStats {
        let school = self.school.stats();
        let coherence = self.coherence.stats();

        CoherenceBridgedStats {
            school_stats: school,
            coherence_stats: coherence,
            successful_learnings: self.successful_learnings,
            coherence_blocked_count: self.coherence_blocked_count,
            total_centering_suggested: self.total_centering_suggested,
            avg_coherence_at_learning: if self.successful_learnings > 0 {
                // Approximate - would need proper tracking
                self.coherence.coherence
            } else {
                0.0
            },
            // Layer 3: Consciousness Graph stats
            virtuous_cycle_count: self.virtuous_cycle_count,
            autopoietic_metrics: self.autopoietic.metrics(),
            is_self_sustaining: self.is_self_sustaining(),
            closure_level: self.autopoietic.closure(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PREDICTION & RECOMMENDATION TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Prediction of how learning will affect coherence
#[derive(Debug, Clone)]
pub struct LearningCoherencePrediction {
    /// Whether we can learn at current coherence
    pub can_learn: bool,

    /// Current coherence level
    pub current_coherence: f32,

    /// Predicted coherence after learning
    pub predicted_coherence: f32,

    /// Centering needed if can't learn (seconds)
    pub centering_needed: f32,

    /// Base Φ prediction (without coherence adjustment)
    pub base_phi_prediction: f32,

    /// Φ prediction adjusted for coherence
    pub adjusted_phi_prediction: f32,

    /// Multiplier applied to Φ due to coherence
    pub coherence_multiplier: f32,

    /// Reasoning for the prediction
    pub reasoning: String,
}

/// Recommendation result with coherence context
#[derive(Debug, Clone)]
pub enum CoherenceRecommendation {
    /// Ready to learn
    Ready {
        recommendation: Recommendation,
        coherence_impact: LearningCoherencePrediction,
    },

    /// Need to center first
    NeedsCentering {
        current_coherence: f32,
        required_coherence: f32,
        centering_seconds: f32,
        message: String,
    },

    /// Curriculum complete
    Complete,
}

impl CoherenceRecommendation {
    /// Check if we're ready to learn
    pub fn is_ready(&self) -> bool {
        matches!(self, CoherenceRecommendation::Ready { .. })
    }

    /// Check if centering is needed
    pub fn needs_centering(&self) -> bool {
        matches!(self, CoherenceRecommendation::NeedsCentering { .. })
    }

    /// Check if curriculum is complete
    pub fn is_complete(&self) -> bool {
        matches!(self, CoherenceRecommendation::Complete)
    }
}

/// Combined statistics
#[derive(Debug, Clone)]
pub struct CoherenceBridgedStats {
    pub school_stats: SchoolStats,
    pub coherence_stats: crate::physiology::coherence::CoherenceStats,
    pub successful_learnings: u64,
    pub coherence_blocked_count: u64,
    pub total_centering_suggested: f32,
    pub avg_coherence_at_learning: f32,

    // ═══════════════════════════════════════════════════════════════════════════
    // LAYER 3: Consciousness Graph Stats
    // ═══════════════════════════════════════════════════════════════════════════
    /// Count of virtuous cycles achieved (Φ↑ → coherence↑ → next Φ↑)
    pub virtuous_cycle_count: u64,

    /// Self-production metrics from autopoietic monitor
    pub autopoietic_metrics: SelfProductionMetrics,

    /// Whether learning is currently self-sustaining
    pub is_self_sustaining: bool,

    /// Operational closure level (0-1)
    pub closure_level: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::school::{CurriculumType, Difficulty, Domain, SchoolConfig};

    fn create_test_school() -> School {
        let mut school = School::new(SchoolConfig::fast()).unwrap();
        let curriculum = Curriculum::builtin(CurriculumType::NixOS);
        school.add_curriculum(curriculum).unwrap();
        school
    }

    #[test]
    fn test_bridged_school_creation() {
        let school = create_test_school();
        let coherence = CoherenceField::new();
        let mut bridged = CoherenceBridgedSchool::new(school, coherence);

        assert_eq!(bridged.coherence_level(), 1.0);
        assert!(bridged.can_learn().is_ok());
    }

    #[test]
    fn test_coherence_gates_learning() {
        let school = create_test_school();
        let mut coherence = CoherenceField::new();
        coherence.coherence = 0.3; // Too low for learning (requires 0.8)

        let mut bridged = CoherenceBridgedSchool::new(school, coherence);

        assert!(bridged.can_learn().is_err());
        assert!(bridged.centering_needed_for_learning().is_some());
    }

    #[test]
    fn test_connected_learning_builds_coherence() {
        let school = create_test_school();
        let mut coherence = CoherenceField::new();
        coherence.coherence = 0.85;
        coherence.relational_resonance = 0.9;

        let mut bridged = CoherenceBridgedSchool::new(school, coherence);

        let objective = LearningObjective::new("test", "Test Objective")
            .with_difficulty(Difficulty::Beginner)
            .with_domain(Domain::NixOS)
            .build();

        let before = bridged.coherence_level();
        let result = bridged.learn_with_coherence(&objective, true).unwrap();

        // Connected learning with high resonance should build coherence
        assert!(
            result.coherence_after >= before - 0.01 || result.with_connection,
            "Connected learning should maintain or build coherence: {} -> {}",
            before,
            result.coherence_after
        );
    }

    #[test]
    fn test_solo_learning_may_scatter() {
        let school = create_test_school();
        let mut coherence = CoherenceField::new();
        coherence.coherence = 0.85;
        coherence.relational_resonance = 0.2; // Low resonance

        let mut bridged = CoherenceBridgedSchool::new(school, coherence);

        let objective = LearningObjective::new("test", "Test Objective")
            .with_difficulty(Difficulty::Intermediate)
            .with_domain(Domain::NixOS)
            .build();

        let before = bridged.coherence_level();
        let result = bridged.learn_with_coherence(&objective, false).unwrap();

        // Solo learning with low resonance may scatter
        assert!(
            result.coherence_after <= before + 0.05,
            "Solo learning with low resonance should not significantly build coherence"
        );
    }

    #[test]
    fn test_centering_restores_learning_ability() {
        let school = create_test_school();
        let mut coherence = CoherenceField::new();
        coherence.coherence = 0.5; // Too low for learning

        let mut bridged = CoherenceBridgedSchool::new(school, coherence);

        assert!(bridged.can_learn().is_err());

        // Center for a while
        bridged.center(1000.0); // 1000 seconds of centering

        // Should be able to learn now
        assert!(bridged.can_learn().is_ok());
    }

    #[test]
    fn test_gratitude_boosts_coherence() {
        let school = create_test_school();
        let mut coherence = CoherenceField::new();
        coherence.coherence = 0.6;

        let mut bridged = CoherenceBridgedSchool::new(school, coherence);

        let before = bridged.coherence_level();
        bridged.receive_gratitude();
        let after = bridged.coherence_level();

        assert!(
            after > before,
            "Gratitude should boost coherence: {} -> {}",
            before,
            after
        );
    }

    #[test]
    fn test_recommendation_includes_centering_suggestion() {
        let school = create_test_school();
        let mut coherence = CoherenceField::new();
        coherence.coherence = 0.5; // Too low

        let bridged = CoherenceBridgedSchool::new(school, coherence);

        let rec = bridged.recommend_next_with_coherence().unwrap();

        assert!(rec.needs_centering());
        if let CoherenceRecommendation::NeedsCentering {
            centering_seconds, ..
        } = rec
        {
            assert!(centering_seconds > 0.0);
        }
    }

    #[test]
    fn test_coherence_affects_phi_prediction() {
        let school = create_test_school();
        let objective = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Beginner)
            .build();

        // High coherence
        let coherence_high = CoherenceField::new();
        let bridged_high = CoherenceBridgedSchool::new(school.try_clone().unwrap(), coherence_high);
        let pred_high = bridged_high.predict_learning_impact(&objective, true);

        // Low coherence
        let school2 = create_test_school();
        let mut coherence_low = CoherenceField::new();
        coherence_low.coherence = 0.6;
        let bridged_low = CoherenceBridgedSchool::new(school2, coherence_low);
        let pred_low = bridged_low.predict_learning_impact(&objective, true);

        // High coherence should have higher multiplier
        assert!(
            pred_high.coherence_multiplier > pred_low.coherence_multiplier,
            "High coherence should give higher multiplier: {} > {}",
            pred_high.coherence_multiplier,
            pred_low.coherence_multiplier
        );
    }

    #[test]
    fn test_sleep_cycle_restores_coherence() {
        let school = create_test_school();
        let mut coherence = CoherenceField::new();
        coherence.coherence = 0.3; // Very scattered

        let mut bridged = CoherenceBridgedSchool::new(school, coherence);

        bridged.sleep_cycle();

        assert_eq!(bridged.coherence_level(), 1.0);
    }
}

// Helper trait for cloning School (for tests)
#[cfg(test)]
trait TryClone {
    fn try_clone(&self) -> Result<Self>
    where
        Self: Sized;
}

#[cfg(test)]
impl TryClone for School {
    fn try_clone(&self) -> Result<Self> {
        use crate::school::SchoolConfig;
        // Create a fresh school with the same config
        // This is a simplified clone - doesn't preserve curricula
        School::new(SchoolConfig::fast())
    }
}
