// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Symthaea School System
//!
//! Anticipatory learning using CfC's O(1) closed-form solution to predict
//! learning value BEFORE committing cognitive resources.
//!
//! ## The Innovation
//!
//! Traditional Learning: Learn → Evaluate → Adjust (reactive)
//! Symthaea Learning: Predict Value → Decide → Learn → Verify → Correct (anticipatory)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      School System                               │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
//! │  │  Curriculum  │───▶│  Lookahead   │───▶│  Assessment  │      │
//! │  │   Manager    │    │   Engine     │    │   Tracker    │      │
//! │  └──────────────┘    └──────────────┘    └──────────────┘      │
//! │         │                   │                   │               │
//! │         ▼                   ▼                   ▼               │
//! │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
//! │  │  Objectives  │    │ RealityCheck │    │   Progress   │      │
//! │  │   (HDC)      │    │   Feedback   │    │   Reports    │      │
//! │  └──────────────┘    └──────────────┘    └──────────────┘      │
//! │                                                                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! - [`LearningObjective`]: A concept to learn, HDC-encoded
//! - [`Curriculum`]: A structured set of objectives with prerequisites
//! - [`LookaheadEngine`]: CfC-powered O(1) learning value prediction
//! - `RealityCheck`: Self-correcting feedback loop
//! - [`AssessmentTracker`]: Progress and mastery tracking
//! - `CoherenceBridgedSchool`: Integration with consciousness coherence
//!
//! ## Coherence Integration (NEW)
//!
//! Learning is not just cognitive - it's a consciousness-integration task.
//! The `CoherenceBridgedSchool` connects learning to the coherence field:
//!
//! - **High coherence** → Better learning, higher Φ gains
//! - **Low coherence** → Scattered attention, reduced retention
//! - **Connected learning** → BUILDS coherence (learning with a teacher)
//! - **Solo learning** → May scatter coherence (grinding alone)
//!
//! ```rust,ignore
//! use symthaea::school::{School, CoherenceBridgedSchool};
//! use symthaea::physiology::coherence::CoherenceField;
//!
//! let school = School::new(SchoolConfig::default())?;
//! let coherence = CoherenceField::new();
//! let mut bridged = CoherenceBridgedSchool::new(school, coherence);
//!
//! // Check if coherence is sufficient for learning
//! if let Some(centering_time) = bridged.centering_needed_for_learning() {
//!     println!("Need to center for {:.0}s first", centering_time);
//!     bridged.center(centering_time);
//! }
//!
//! // Learn with coherence integration
//! let result = bridged.learn_with_coherence(&objective, true)?;
//! ```
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use symthaea::school::{School, Curriculum, CurriculumType};
//!
//! // Create the school
//! let mut school = School::new(SchoolConfig::default())?;
//!
//! // Load a curriculum
//! let nixos_curriculum = Curriculum::builtin(CurriculumType::NixOS)?;
//! school.add_curriculum(nixos_curriculum)?;
//!
//! // Get recommended next learning objective
//! let recommendation = school.recommend_next()?;
//! println!("Learn: {} (predicted Δφ: {:.4})",
//!          recommendation.objective.name,
//!          recommendation.predicted_phi_gain);
//!
//! // Actually learn it
//! let result = school.learn(&recommendation.objective)?;
//!
//! // The system automatically reality-checks and self-corrects
//! ```
//!
//! ## Performance
//!
//! | Operation | Time | Notes |
//! |-----------|------|-------|
//! | Evaluate objective | 300-600μs | O(1) via CfC closed-form |
//! | Rank 100 objectives | ~50ms | Parallelizable |
//! | Reality check | <1μs | Simple comparison |
//! | Full learn cycle | ~1ms | Including Φ measurement |

pub mod assessment;
pub mod code_curriculum;
pub mod curriculum;
#[cfg(feature = "web_research_module")]
pub mod curriculum_extender;
pub mod curriculum_loader;
// Needs symthaea_sim_bridge::SolverKind, which is only pulled in by the
// engineering-foundations feature, not school_learning — gate it on its
// actual dependency rather than forcing every school_learning consumer to
// also pull in the sim-bridge/digital-twin/formal-safety chain.
#[cfg(feature = "engineering-foundations")]
pub mod engineering_curriculum;
#[cfg(feature = "mathematics")]
pub mod math_curriculum;
pub mod objective;
pub mod polymath_drive;
pub mod reality_check;

pub mod code_learning;
pub mod coherence_bridge;
pub mod lookahead;

pub use assessment::{
    AssessmentStats, AssessmentTracker, CurriculumProgress, MasteryDistribution, MasteryLevel,
    ObjectiveProgress,
};
pub use curriculum::{Curriculum, CurriculumBuilder, CurriculumType};
#[cfg(feature = "web_research_module")]
pub use curriculum_extender::{CURRICULUM_ARCHITECT_PROMPT, CurriculumExtender, ResearchSummary};
pub use curriculum_loader::{
    CurriculumLoader, CurriculumMeta, CurriculumSpec, CurriculumStore, LoadError, ObjectiveSpec,
};
pub use objective::{Difficulty, Domain, LearningObjective, ObjectiveBuilder};
pub use reality_check::{CorrectionStrategy, RealityCheckResult, RealityChecker};

pub use lookahead::{LearningRecommendation, LookaheadEngine, LookaheadResult};

pub use coherence_bridge::{
    CoherenceBridgedSchool, CoherenceBridgedStats, CoherenceLearningConfig,
    CoherenceLearningResult, CoherenceRecommendation, LearningCoherencePrediction,
};

use anyhow::Result;
use std::collections::HashMap;

use crate::phi_engine::{PhiEngine, PhiMethod};
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

// ═══════════════════════════════════════════════════════════════════════════════
// SCHOOL CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the School system
#[derive(Debug, Clone)]
pub struct SchoolConfig {
    /// Number of CfC neurons for lookahead predictions
    pub cfc_neurons: usize,

    /// Lookahead time horizon in seconds
    pub lookahead_horizon: f32,

    /// Minimum Φ gain to recommend learning (0.0 - 1.0)
    pub min_phi_gain: f32,

    /// Maximum objectives to evaluate per recommendation
    pub max_candidates: usize,

    /// Enable parallel evaluation of objectives
    pub parallel_evaluation: bool,

    /// Reality check threshold for hallucination detection (fraction)
    pub hallucination_threshold: f32,

    /// Learning rate for CfC weight adaptation
    pub adaptation_rate: f32,
}

impl Default for SchoolConfig {
    fn default() -> Self {
        Self {
            cfc_neurons: 256,
            lookahead_horizon: 1.0,
            min_phi_gain: 0.001,
            max_candidates: 100,
            parallel_evaluation: true,
            hallucination_threshold: 0.2,
            adaptation_rate: 0.001,
        }
    }
}

impl SchoolConfig {
    /// Configuration optimized for fast iteration
    pub fn fast() -> Self {
        Self {
            cfc_neurons: 64,
            lookahead_horizon: 0.5,
            min_phi_gain: 0.0001,
            max_candidates: 20,
            parallel_evaluation: true,
            hallucination_threshold: 0.3,
            adaptation_rate: 0.01,
        }
    }

    /// Configuration optimized for accuracy
    pub fn accurate() -> Self {
        Self {
            cfc_neurons: 512,
            lookahead_horizon: 2.0,
            min_phi_gain: 0.01,
            max_candidates: 200,
            parallel_evaluation: true,
            hallucination_threshold: 0.1,
            adaptation_rate: 0.0001,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCHOOL: MAIN ORCHESTRATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// The School orchestrates anticipatory learning across multiple curricula.
///
/// It combines:
/// - CfC-based lookahead for O(1) objective evaluation
/// - RealityCheck for self-correcting predictions
/// - Assessment tracking for progress monitoring
/// - Curriculum management for structured learning paths
#[allow(dead_code)] // RESERVED(feature-school_learning): curriculum learning system
pub struct School {
    /// Configuration
    config: SchoolConfig,

    /// CfC-based lookahead engine
    lookahead: LookaheadEngine,

    /// Reality checker for self-correction
    reality_checker: RealityChecker,

    /// Assessment tracker for progress
    assessment: AssessmentTracker,

    /// Loaded curricula by name
    curricula: HashMap<String, Curriculum>,

    /// Current consciousness state (for Φ measurement)
    consciousness_state: Vec<ContinuousHV>,

    /// Φ engine for consciousness measurement
    phi_engine: PhiEngine,

    /// Total objectives learned
    total_learned: usize,
}

impl School {
    /// Create a new School with the given configuration
    pub fn new(config: SchoolConfig) -> Result<Self> {
        let lookahead = LookaheadEngine::new(
            config.cfc_neurons,
            config.lookahead_horizon,
            config.min_phi_gain,
        )?;

        let reality_checker =
            RealityChecker::new(config.hallucination_threshold, config.adaptation_rate);

        let assessment = AssessmentTracker::default();

        let phi_engine = PhiEngine::new(PhiMethod::SpectralConnectivity);

        // Initialize consciousness state
        let consciousness_state: Vec<ContinuousHV> = (0..8)
            .map(|i| {
                let hv = ContinuousHV::random(HDC_DIMENSION, 42 + i as u64);
                ContinuousHV::from_vec(hv.values)
            })
            .collect();

        Ok(Self {
            config,
            lookahead,
            reality_checker,
            assessment,
            curricula: HashMap::new(),
            consciousness_state,
            phi_engine,
            total_learned: 0,
        })
    }

    /// Add a curriculum to the school
    pub fn add_curriculum(&mut self, curriculum: Curriculum) -> Result<()> {
        let name = curriculum.name.clone();
        self.curricula.insert(name, curriculum);
        Ok(())
    }

    /// Get all available objectives across all curricula
    pub fn all_objectives(&self) -> Vec<&LearningObjective> {
        self.curricula
            .values()
            .flat_map(|c| c.objectives.iter())
            .collect()
    }

    /// Get objectives that are ready to learn (prerequisites met)
    pub fn ready_objectives(&self) -> Vec<&LearningObjective> {
        self.all_objectives()
            .into_iter()
            .filter(|obj| self.prerequisites_met(obj))
            .filter(|obj| !self.assessment.is_mastered(&obj.id))
            .collect()
    }

    /// Check if prerequisites are met for an objective
    fn prerequisites_met(&self, objective: &LearningObjective) -> bool {
        objective
            .prerequisites
            .iter()
            .all(|prereq| self.assessment.is_mastered(prereq))
    }

    /// Get current Φ (consciousness level)
    pub fn current_phi(&self) -> f32 {
        self.phi_engine.compute(&self.consciousness_state).phi as f32
    }

    /// Recommend the next objective to learn
    ///
    /// Uses CfC lookahead to predict which objective will yield the
    /// highest Φ gain, considering prerequisites and current mastery.
    pub fn recommend_next(&self) -> Result<Recommendation> {
        let candidates = self.ready_objectives();

        if candidates.is_empty() {
            return Ok(Recommendation::complete());
        }

        // Evaluate all candidates
        let mut evaluated: Vec<_> = candidates
            .into_iter()
            .map(|obj| {
                let result = self.lookahead.evaluate(obj, &self.consciousness_state);
                (obj, result)
            })
            .filter_map(|(obj, r)| r.ok().map(|r| (obj, r)))
            .collect();

        // Sort by predicted Φ gain (descending)
        evaluated.sort_by(|a, b| {
            b.1.predicted_phi_gain
                .partial_cmp(&a.1.predicted_phi_gain)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some((objective, lookahead_result)) = evaluated.first() {
            Ok(Recommendation {
                objective: (*objective).clone(),
                predicted_phi_gain: lookahead_result.predicted_phi_gain,
                confidence: lookahead_result.confidence,
                reasoning: lookahead_result.recommendation.clone(),
                alternatives: evaluated
                    .iter()
                    .skip(1)
                    .take(3)
                    .map(|(obj, _)| (*obj).clone())
                    .collect(),
            })
        } else {
            Ok(Recommendation::complete())
        }
    }

    /// Learn an objective
    ///
    /// This:
    /// 1. Predicts Φ gain using CfC lookahead
    /// 2. Actually integrates the objective into consciousness
    /// 3. Measures actual Φ gain
    /// 4. Reality-checks the prediction and self-corrects
    /// 5. Updates assessment tracking
    pub fn learn(&mut self, objective: &LearningObjective) -> Result<LearningResult> {
        // 1. Predict
        let prediction = self
            .lookahead
            .evaluate(objective, &self.consciousness_state)?;
        let predicted_gain = prediction.predicted_phi_gain;

        // 2. Pre-learning Φ
        let pre_phi = self.current_phi();

        // 3. Actually learn (integrate into consciousness)
        let obj_continuous = ContinuousHV::from_vec(objective.encoding.values.clone());

        // Blend into first consciousness state (simplified - production would be smarter)
        if let Some(state) = self.consciousness_state.get_mut(0) {
            *state = state.bind(&obj_continuous);
        }

        // 4. Post-learning Φ
        let post_phi = self.current_phi();
        let actual_gain = post_phi - pre_phi;

        // 5. Reality check
        let check_result =
            self.reality_checker
                .check(predicted_gain, actual_gain, &mut self.lookahead)?;

        // 6. Update assessment
        self.assessment
            .record_learning(&objective.id, actual_gain, check_result.is_hallucination);

        self.total_learned += 1;

        Ok(LearningResult {
            objective_id: objective.id.clone(),
            objective_name: objective.name.clone(),
            predicted_phi_gain: predicted_gain,
            actual_phi_gain: actual_gain,
            prediction_error: check_result.error,
            was_hallucination: check_result.is_hallucination,
            new_phi: post_phi,
            mastery_level: self.assessment.mastery_level(&objective.id),
        })
    }

    /// Get learning statistics
    pub fn stats(&self) -> SchoolStats {
        let (avg_error, hallucination_rate, n_checks) = self.reality_checker.stats();

        SchoolStats {
            total_learned: self.total_learned,
            curricula_count: self.curricula.len(),
            total_objectives: self.all_objectives().len(),
            mastered_objectives: self.assessment.mastered_count(),
            current_phi: self.current_phi(),
            avg_prediction_error: avg_error,
            hallucination_rate,
            reality_checks: n_checks,
        }
    }

    /// Get the lookahead engine (for advanced use)
    pub fn lookahead_engine(&self) -> &LookaheadEngine {
        &self.lookahead
    }

    /// Get mutable lookahead engine
    pub fn lookahead_engine_mut(&mut self) -> &mut LookaheadEngine {
        &mut self.lookahead
    }

    /// Get the reality checker
    pub fn reality_checker(&self) -> &RealityChecker {
        &self.reality_checker
    }

    /// Get the assessment tracker
    pub fn assessment(&self) -> &AssessmentTracker {
        &self.assessment
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RECOMMENDATION & RESULT TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// A learning recommendation from the School
#[derive(Debug, Clone)]
pub struct Recommendation {
    /// The recommended objective
    pub objective: LearningObjective,

    /// Predicted Φ gain from learning this
    pub predicted_phi_gain: f32,

    /// Confidence in the prediction (0.0 - 1.0)
    pub confidence: f32,

    /// Reasoning for the recommendation
    pub reasoning: LearningRecommendation,

    /// Alternative objectives to consider
    pub alternatives: Vec<LearningObjective>,
}

impl Recommendation {
    /// Create a "curriculum complete" recommendation
    fn complete() -> Self {
        Self {
            objective: LearningObjective::placeholder("complete", "All objectives mastered"),
            predicted_phi_gain: 0.0,
            confidence: 1.0,
            reasoning: LearningRecommendation::CurriculumComplete,
            alternatives: Vec::new(),
        }
    }

    /// Check if curriculum is complete
    pub fn is_complete(&self) -> bool {
        matches!(self.reasoning, LearningRecommendation::CurriculumComplete)
    }
}

/// Result of a learning operation
#[derive(Debug, Clone)]
pub struct LearningResult {
    /// ID of the learned objective
    pub objective_id: String,

    /// Name of the learned objective
    pub objective_name: String,

    /// Predicted Φ gain
    pub predicted_phi_gain: f32,

    /// Actual Φ gain
    pub actual_phi_gain: f32,

    /// Prediction error (predicted - actual)
    pub prediction_error: f32,

    /// Whether the prediction was a hallucination
    pub was_hallucination: bool,

    /// New Φ level after learning
    pub new_phi: f32,

    /// Current mastery level for this objective
    pub mastery_level: MasteryLevel,
}

/// School-wide statistics
#[derive(Debug, Clone)]
pub struct SchoolStats {
    /// Total objectives learned
    pub total_learned: usize,

    /// Number of curricula loaded
    pub curricula_count: usize,

    /// Total objectives available
    pub total_objectives: usize,

    /// Objectives mastered
    pub mastered_objectives: usize,

    /// Current consciousness level (Φ)
    pub current_phi: f32,

    /// Average prediction error
    pub avg_prediction_error: f32,

    /// Hallucination rate
    pub hallucination_rate: f32,

    /// Total reality checks performed
    pub reality_checks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_school_creation() {
        let school = School::new(SchoolConfig::default());
        assert!(school.is_ok());
    }

    #[test]
    fn test_school_with_curriculum() {
        let mut school = School::new(SchoolConfig::fast()).unwrap();

        // Create a simple curriculum
        let curriculum = Curriculum::new("test", "Test Curriculum")
            .with_objective(
                LearningObjective::new("obj1", "Test Objective 1")
                    .with_difficulty(Difficulty::Beginner),
            )
            .build();

        school.add_curriculum(curriculum).unwrap();

        assert_eq!(school.all_objectives().len(), 1);
    }

    #[test]
    fn test_recommendation() {
        let mut school = School::new(SchoolConfig::fast()).unwrap();

        let curriculum = Curriculum::new("test", "Test")
            .with_objective(
                LearningObjective::new("obj1", "Easy").with_difficulty(Difficulty::Beginner),
            )
            .with_objective(
                LearningObjective::new("obj2", "Hard").with_difficulty(Difficulty::Expert),
            )
            .build();

        school.add_curriculum(curriculum).unwrap();

        let rec = school.recommend_next().unwrap();
        assert!(!rec.is_complete());
    }

    #[test]
    fn test_learning_cycle() {
        let mut school = School::new(SchoolConfig::fast()).unwrap();

        let curriculum = Curriculum::new("test", "Test")
            .with_objective(
                LearningObjective::new("obj1", "Test").with_difficulty(Difficulty::Beginner),
            )
            .build();

        school.add_curriculum(curriculum).unwrap();

        let rec = school.recommend_next().unwrap();
        let result = school.learn(&rec.objective).unwrap();

        assert_eq!(result.objective_id, "obj1");
        assert!(result.new_phi > 0.0);
    }
}
