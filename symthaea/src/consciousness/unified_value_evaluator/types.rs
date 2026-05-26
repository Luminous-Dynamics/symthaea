// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Configuration, context, and decision types for the Unified Value Evaluator.

use super::super::affective_consciousness::CoreAffect;
use super::super::contextual_weights::ActionDomain;
use super::super::eight_harmonies::AlignmentResult;
use serde::{Deserialize, Serialize};

/// Minimum consciousness level for different action types
#[derive(Debug, Clone, Copy)]
pub struct ConsciousnessThresholds {
    /// Minimum Φ for any action
    pub basic_action: f64,
    /// Minimum Φ for governance proposals
    pub governance: f64,
    /// Minimum Φ for voting on proposals
    pub voting: f64,
    /// Minimum Φ for constitutional changes
    pub constitutional: f64,
}

impl Default for ConsciousnessThresholds {
    fn default() -> Self {
        Self {
            basic_action: 0.2,
            governance: 0.3,
            voting: 0.4,
            constitutional: 0.6,
        }
    }
}

/// Configuration for the unified evaluator
#[derive(Debug, Clone)]
pub struct EvaluatorConfig {
    /// Consciousness thresholds
    pub consciousness_thresholds: ConsciousnessThresholds,
    /// Minimum CARE activation for actions involving others
    pub min_care_activation: f64,
    /// Minimum alignment score before warning
    pub warning_threshold: f64,
    /// Maximum negative alignment before veto
    pub veto_threshold: f64,
    /// Whether to require affective grounding
    pub require_affective_grounding: bool,
    /// Weight for semantic alignment (vs affective)
    pub semantic_weight: f64,
    /// Weight for affective alignment (vs semantic)
    pub affective_weight: f64,
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        Self {
            consciousness_thresholds: ConsciousnessThresholds::default(),
            min_care_activation: 0.3,
            warning_threshold: 0.1,
            veto_threshold: -0.3,
            require_affective_grounding: true,
            semantic_weight: 0.6,
            affective_weight: 0.4,
        }
    }
}

/// Context for evaluation
#[derive(Debug, Clone)]
pub struct EvaluationContext {
    /// Current consciousness level (Φ)
    pub consciousness_level: f64,
    /// Current affective state
    pub affective_state: CoreAffect,
    /// Affective systems activation levels
    pub affective_systems: AffectiveSystemsState,
    /// Type of action being evaluated
    pub action_type: ActionType,
    /// Domain of the action (financial, creative, social, etc.)
    /// If None, will be auto-detected from action text
    pub action_domain: Option<ActionDomain>,
    /// Whether action involves other beings
    pub involves_others: bool,
}

impl Default for EvaluationContext {
    fn default() -> Self {
        Self {
            consciousness_level: 0.5,
            affective_state: CoreAffect::neutral(),
            affective_systems: AffectiveSystemsState::default(),
            action_type: ActionType::Basic,
            action_domain: None, // Will be auto-detected
            involves_others: false,
        }
    }
}

/// Activation levels for each affective system
#[derive(Debug, Clone, Default)]
pub struct AffectiveSystemsState {
    pub seeking: f64,
    pub rage: f64,
    pub fear: f64,
    pub lust: f64,
    pub care: f64,
    pub panic: f64,
    pub play: f64,
}

impl AffectiveSystemsState {
    /// Create from array (matching PrimaryAffectSystem order)
    pub fn from_array(values: [f64; 7]) -> Self {
        Self {
            seeking: values[0],
            rage: values[1],
            fear: values[2],
            lust: values[3],
            care: values[4],
            panic: values[5],
            play: values[6],
        }
    }

    /// Get CARE activation level
    pub fn care_level(&self) -> f64 {
        self.care
    }

    /// Get positive affect (CARE + PLAY + SEEKING)
    pub fn positive_affect(&self) -> f64 {
        (self.care + self.play + self.seeking) / 3.0
    }

    /// Get negative affect (RAGE + FEAR + PANIC)
    pub fn negative_affect(&self) -> f64 {
        (self.rage + self.fear + self.panic) / 3.0
    }

    /// Check if affective state is benevolent
    pub fn is_benevolent(&self) -> bool {
        self.care > 0.3 && self.rage < 0.3 && self.fear < 0.5
    }
}

/// Type of action being evaluated (re-exported from contextual_weights)
pub use super::super::contextual_weights::ActionType;

/// The decision outcome
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Decision {
    /// Action is allowed
    Allow,
    /// Action is allowed but with warnings
    Warn(Vec<String>),
    /// Action is vetoed
    Veto(VetoReason),
}

/// Reason for veto
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VetoReason {
    /// Consciousness level too low
    InsufficientConsciousness {
        current: f64,
        required: f64,
        action_type: String,
    },
    /// Value violation detected
    ValueViolation { harmony: String, alignment: f64 },
    /// Lacking genuine caring (CARE system inactive)
    InauthenicBenevolence { care_level: f64, required: f64 },
    /// Negative affect dominant (RAGE/FEAR too high)
    NegativeAffectDominant { rage: f64, fear: f64 },
    /// Multiple minor issues compound to veto
    CompoundedWarnings { warnings: Vec<String>, count: usize },
}

/// Complete evaluation result
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// The decision
    pub decision: Decision,
    /// Harmony alignment result
    pub harmony_alignment: AlignmentResult,
    /// Authenticity score (0-1)
    pub authenticity: f64,
    /// Consciousness adequacy (0-1)
    pub consciousness_adequacy: f64,
    /// Affective grounding score (0-1)
    pub affective_grounding: f64,
    /// Overall score combining all factors
    pub overall_score: f64,
    /// Detailed breakdown
    pub breakdown: EvaluationBreakdown,
}

/// Detailed breakdown of the evaluation
#[derive(Debug, Clone)]
pub struct EvaluationBreakdown {
    /// Semantic alignment with each harmony
    pub harmony_scores: Vec<(String, f64)>,
    /// CARE system contribution
    pub care_contribution: f64,
    /// PLAY system contribution
    pub play_contribution: f64,
    /// SEEKING system contribution
    pub seeking_contribution: f64,
    /// Negative affect penalty
    pub negative_affect_penalty: f64,
    /// Consciousness boost
    pub consciousness_boost: f64,
}

/// Evaluator statistics
#[derive(Debug, Clone)]
pub struct EvaluatorStats {
    pub total_evaluations: usize,
    pub vetoes: usize,
    pub warnings: usize,
    pub allows: usize,
    pub veto_rate: f64,
}
