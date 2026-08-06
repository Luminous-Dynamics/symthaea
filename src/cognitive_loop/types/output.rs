// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Output types — CycleResult, ActionHint, MoralJudgmentSummary.

use serde::{Deserialize, Serialize};

use super::scheduling::CycleUrgency;
use super::telemetry::CycleMetadata;

// ═══════════════════════════════════════════════════════════════════════════════
// PHI ATTESTATION RECORD — for governance bridge consumption
// ═══════════════════════════════════════════════════════════════════════════════

/// Record of a Phi measurement from a cognitive cycle, ready for attestation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsiAttestationRecord {
    pub psi: f64,
    pub cycle_id: u64,
    pub captured_at_us: u64,
    pub prediction_error: f32,
    pub urgency: CycleUrgency,
}

impl PsiAttestationRecord {
    pub fn sign_message(&self, agent_did: &str) -> Vec<u8> {
        format!(
            "symthaea-phi-attestation:v1:{}:{}:{}:{}",
            agent_did, self.psi, self.cycle_id, self.captured_at_us
        )
        .into_bytes()
    }
}

/// Result of a single cognitive cycle
#[derive(Debug, Clone)]
pub struct CycleResult {
    pub output: Vec<f32>,
    pub prediction_error: f32,
    pub peak_attention: f32,
    pub detected_primitives: Vec<String>,
    pub learning_occurred: bool,
    pub training_loss: Option<f32>,
    /// Relative predictive "bits saved" vs the persistence baseline
    /// (measurement-only diagnostic; see
    /// docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md §2 + Amendment 1).
    /// None until the shortest-horizon prediction, baseline, and shared
    /// residual variance are all warm.
    pub bits_saved_persist: Option<f32>,
    /// Same, vs the zero-prediction floor.
    pub bits_saved_zero: Option<f32>,
    /// Shared concentration κ behind the bits fields — Δcos = bits·ln2/κ
    /// (the quantity C1 verdicts use; Amendment 3).
    pub bits_kappa: Option<f32>,
    /// Predictive Compression C3 (§7): whether episodic recall blended into
    /// the prediction this cycle. Always false when
    /// `enable_episodic_recall_prediction` is off (the default).
    pub recall_fired: bool,
    /// C3: top-1 recall similarity this cycle, when a recall was attempted.
    pub recall_similarity: Option<f32>,
    /// C3c (§7 Experiment C3c): write-cycle number of the matched episode,
    /// when a recall was attempted — lets an external harness that knows
    /// what each cycle number "was" look up the matched episode's provenance.
    pub recall_matched_timestamp: Option<u64>,
    pub cycle_time_us: u64,
    pub metadata: CycleMetadata,
    pub thought_vector: Vec<f32>,
    pub wisdom_hv: symthaea_core::hdc::BinaryHV,
    pub language_output: Option<String>,
    pub language_source: Option<String>,
    #[cfg(feature = "canvas")]
    pub canvas_svg: Option<String>,
    #[cfg(feature = "identity")]
    pub signed_output: Option<crate::identity::SignedOutput>,
    #[cfg(feature = "identity")]
    pub assurance_level: crate::identity::AssuranceLevel,

    /// Mental movie generated from geodesic mental simulation, if requested.
    #[cfg(feature = "vision-manifold")]
    pub mental_movie: Option<MentalMovie>,
}

/// Mental movie generated from geodesic mental simulation.
#[cfg(feature = "vision-manifold")]
#[derive(Clone, Debug, Default)]
pub struct MentalMovie {
    pub frames: Vec<Vec<u8>>,
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub path_length: usize,
    pub semantic_coherence: f32,
    /// Raw mathematical trajectory (HDC vectors) used to generate the frames.
    pub trajectory: Vec<symthaea_core::core::ContinuousHV>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// MORAL JUDGMENT SUMMARY
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoralJudgmentSummary {
    pub input: String,
    pub verdict: String,
    pub deontological_verdict: String,
    pub violations: Vec<String>,
    pub satisfactions: Vec<String>,
    pub consent_violation: bool,
    pub moral_score: f32,
    pub confidence: f32,
}

impl Default for MoralJudgmentSummary {
    fn default() -> Self {
        Self {
            input: String::new(),
            verdict: "Neutral".to_string(),
            deontological_verdict: "Permissible".to_string(),
            violations: Vec::new(),
            satisfactions: Vec::new(),
            consent_violation: false,
            moral_score: 0.0,
            confidence: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADAPTIVE BEHAVIOR
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveBehavior {
    pub learning_rate_multiplier: f32,
    pub speech_rate_multiplier: f32,
    pub pause_multiplier: f32,
    pub attention_sensitivity: f32,
    pub exploration_factor: f32,
    pub confidence: f32,
    pub pause_learning: bool,
    pub action_hint: ActionHint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionHint {
    Continue,
    SlowDown,
    SpeedUp,
    Stabilize,
    Explore,
    SeekInput,
    Inhibit,
}

impl ActionHint {
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Continue => "Continue",
            Self::SlowDown => "SlowDown",
            Self::SpeedUp => "SpeedUp",
            Self::Stabilize => "Stabilize",
            Self::Explore => "Explore",
            Self::SeekInput => "SeekInput",
            Self::Inhibit => "Inhibit",
        }
    }
}

impl Default for AdaptiveBehavior {
    fn default() -> Self {
        Self {
            learning_rate_multiplier: 1.0,
            speech_rate_multiplier: 1.0,
            pause_multiplier: 1.0,
            attention_sensitivity: 1.0,
            exploration_factor: 0.3,
            confidence: 0.5,
            pause_learning: false,
            action_hint: ActionHint::Continue,
        }
    }
}

impl AdaptiveBehavior {
    pub fn from_consciousness_state(
        pattern: crate::dynamics::temporal_signatures::ConsciousnessPattern,
        pattern_confidence: f32,
        coherence: f32,
        voice_confidence: f32,
    ) -> Self {
        use crate::dynamics::temporal_signatures::ConsciousnessPattern;
        let confidence =
            (pattern_confidence * 0.4 + coherence * 0.3 + voice_confidence * 0.3).clamp(0.0, 1.0);
        match pattern {
            ConsciousnessPattern::Focused => Self {
                learning_rate_multiplier: 1.3 + confidence * 0.4,
                speech_rate_multiplier: 1.05 + confidence * 0.15,
                pause_multiplier: 0.7,
                attention_sensitivity: 0.7,
                exploration_factor: 0.1,
                confidence,
                pause_learning: false,
                action_hint: ActionHint::SpeedUp,
            },
            ConsciousnessPattern::Contemplative => Self {
                learning_rate_multiplier: 0.8,
                speech_rate_multiplier: 0.85,
                pause_multiplier: 1.5,
                attention_sensitivity: 1.0,
                exploration_factor: 0.2,
                confidence,
                pause_learning: false,
                action_hint: ActionHint::SlowDown,
            },
            ConsciousnessPattern::Excited => Self {
                learning_rate_multiplier: 1.1,
                speech_rate_multiplier: 1.15,
                pause_multiplier: 0.6,
                attention_sensitivity: 1.3,
                exploration_factor: 0.4,
                confidence,
                pause_learning: false,
                action_hint: ActionHint::Continue,
            },
            ConsciousnessPattern::Exploratory => Self {
                learning_rate_multiplier: 1.0,
                speech_rate_multiplier: 0.95,
                pause_multiplier: 1.0,
                attention_sensitivity: 1.4,
                exploration_factor: 0.7,
                confidence: confidence * 0.8,
                pause_learning: false,
                action_hint: ActionHint::Explore,
            },
            ConsciousnessPattern::Resting => Self {
                learning_rate_multiplier: 0.6,
                speech_rate_multiplier: 0.9,
                pause_multiplier: 1.2,
                attention_sensitivity: 0.8,
                exploration_factor: 0.2,
                confidence,
                pause_learning: false,
                action_hint: ActionHint::Continue,
            },
            ConsciousnessPattern::Transitioning => Self {
                learning_rate_multiplier: 0.3,
                speech_rate_multiplier: 0.8,
                pause_multiplier: 1.8,
                attention_sensitivity: 1.0,
                exploration_factor: 0.3,
                confidence: confidence * 0.5,
                pause_learning: true,
                action_hint: ActionHint::Stabilize,
            },
            ConsciousnessPattern::Uncertain => Self {
                learning_rate_multiplier: 0.4,
                speech_rate_multiplier: 0.75,
                pause_multiplier: 2.0,
                attention_sensitivity: 1.2,
                exploration_factor: 0.5,
                confidence: confidence * 0.3,
                pause_learning: false,
                action_hint: ActionHint::SeekInput,
            },
        }
    }

    pub fn effective_learning_rate(&self, base_rate: f32) -> f32 {
        if self.pause_learning {
            0.0
        } else {
            base_rate * self.learning_rate_multiplier
        }
    }

    pub fn should_seek_input(&self) -> bool {
        self.action_hint == ActionHint::SeekInput || self.confidence < 0.3
    }

    pub fn is_confident(&self) -> bool {
        self.confidence > 0.6 && !self.pause_learning
    }

    pub fn description(&self) -> &'static str {
        match self.action_hint {
            ActionHint::Continue => "Operating normally",
            ActionHint::SlowDown => "Deliberating carefully",
            ActionHint::SpeedUp => "Confident and focused",
            ActionHint::Stabilize => "Stabilizing state",
            ActionHint::Explore => "Exploring possibilities",
            ActionHint::SeekInput => "Seeking clarification",
            ActionHint::Inhibit => "Inhibiting response",
        }
    }
}
