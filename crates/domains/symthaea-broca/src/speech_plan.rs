// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed speech-production planning for Broca.
//!
//! The planner is an additive boundary between semantic/epistemic readout and
//! language or voice realization. It does not generate text and does not
//! replace the trained CfC/HDC decoder. Instead it preserves the production
//! contract explicitly:
//!
//! `preverbal structure -> linguistic delivery mode -> prosodic intent -> monitor/repair policy`.
//!
//! This separation is motivated by established speech-production models that
//! distinguish conceptualization/formulation/articulation/monitoring, while
//! deliberately avoiding any claim that this software reproduces human Broca's
//! area anatomically.

use serde::{Deserialize, Serialize};

use crate::decoder::StructuredReadout;
use crate::encoder::ThoughtChannels;

/// Stable identity for the typed speech-plan representation.
pub const SPEECH_PLAN_VERSION: &str = "broca-speech-plan-v1";

/// High-level clause realization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClauseMode {
    /// Ordinary declarative delivery.
    Statement,
    /// Information-seeking delivery.
    Question,
    /// Action-oriented / directive delivery.
    Directive,
    /// Metacognitive or affective reflection.
    Reflective,
    /// Relationship-oriented delivery.
    Relational,
    /// No assertive linguistic realization is authorized.
    Abstention,
}

/// Epistemic delivery policy derived from the cognitive state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EpistemicDelivery {
    /// High-confidence, grounded assertion is permitted.
    Assertive,
    /// Content may be stated only with explicit qualification.
    Qualified,
    /// The planner must not manufacture an answer from the state.
    NonAssertive,
}

/// Intonation family consumed by a downstream vocal realization layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntonationIntent {
    Statement,
    Question,
    Exclamation,
}

/// A typed role/filler slot retained from the deterministic semantic readout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechPlanRole {
    pub role: String,
    pub prime: String,
    /// Relative prominence in the utterance plan, normalized to [0, 1].
    pub salience: f32,
}

/// Prosodic intent before phoneme-level realization.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProsodicIntent {
    /// Speech-rate multiplier. 1.0 is neutral.
    pub rate: f32,
    /// Relative pitch-range multiplier. 1.0 is neutral.
    pub pitch_range: f32,
    /// Desired prominence of the focus constituent.
    pub prominence: f32,
    /// Relative pause pressure. Higher means more deliberate spacing.
    pub pause_weight: f32,
    pub intonation: IntonationIntent,
}

/// Closed-loop monitoring policy for the produced utterance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpeechMonitorPlan {
    /// Whether the produced realization should be compared with the intended plan.
    pub compare_intent_to_output: bool,
    /// Whether an acoustic/self-perceptual listen-back should be requested.
    pub auditory_self_monitor: bool,
    /// Normalized discrepancy above which repair should be considered.
    pub repair_threshold: f32,
    /// Unknown/out-of-domain state must remain non-assertive.
    pub fail_closed_on_unknown: bool,
}

/// Explicit production boundary between cognition and linguistic/voice realization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechPlan {
    pub version: String,
    pub intent: String,
    pub clause_mode: ClauseMode,
    pub epistemic_delivery: EpistemicDelivery,
    pub roles: Vec<SpeechPlanRole>,
    /// Optional role that should receive the main information-structural prominence.
    pub focus_role: Option<String>,
    pub confidence: f32,
    pub warmth: f32,
    pub arousal: f32,
    pub social_context: f32,
    pub time_pressure: f32,
    pub prosody: ProsodicIntent,
    pub monitor: SpeechMonitorPlan,
}

impl SpeechPlan {
    /// Build a deterministic speech-production plan from the existing structured readout.
    ///
    /// This method is intentionally loss-aware: role/filler data is copied rather than
    /// re-inferred downstream, while delivery controls are derived from the same source
    /// ThoughtChannels.
    pub fn from_readout(channels: &ThoughtChannels, readout: &StructuredReadout) -> Self {
        let intent = readout.intent.clone();
        let epistemic_delivery = epistemic_delivery(channels.epistemic_ordinal(), readout.confidence);
        let clause_mode = clause_mode_for(&intent, epistemic_delivery);

        let focus_role = focus_role_for(&intent, epistemic_delivery);
        let roles = readout
            .roles
            .iter()
            .map(|role| SpeechPlanRole {
                role: role.role.clone(),
                prime: role.prime.clone(),
                salience: role_salience(&role.role, focus_role.as_deref()),
            })
            .collect::<Vec<_>>();

        let confidence = readout.confidence.clamp(0.0, 1.0);
        let warmth = channels.warmth().clamp(0.0, 1.0);
        let arousal = channels.arousal().clamp(0.0, 1.0);
        let social_context = channels.channels.get(22).copied().unwrap_or(0.5).clamp(0.0, 1.0);
        let time_pressure = channels.channels.get(20).copied().unwrap_or(0.0).clamp(0.0, 1.0);

        let prosody = derive_prosody(
            &intent,
            confidence,
            warmth,
            arousal,
            time_pressure,
            epistemic_delivery,
        );

        let monitor = SpeechMonitorPlan {
            compare_intent_to_output: true,
            auditory_self_monitor: true,
            repair_threshold: (0.35
                + 0.35 * (1.0 - confidence)
                + 0.20 * (1.0 - channels.coherence().clamp(0.0, 1.0)))
                .clamp(0.2, 0.9),
            fail_closed_on_unknown: matches!(epistemic_delivery, EpistemicDelivery::NonAssertive),
        };

        Self {
            version: SPEECH_PLAN_VERSION.to_string(),
            intent,
            clause_mode,
            epistemic_delivery,
            roles,
            focus_role,
            confidence,
            warmth,
            arousal,
            social_context,
            time_pressure,
            prosody,
            monitor,
        }
    }

    /// Deterministic compact surface for tracing/evidence capture.
    ///
    /// This is not a prose realization. It is a stable inspection surface that can be
    /// stored beside generation/auditory evidence without depending on tokenizer state.
    pub fn grounding_surface(&self) -> String {
        let roles = self
            .roles
            .iter()
            .map(|r| format!("{}={}", r.role, r.prime))
            .collect::<Vec<_>>()
            .join("|");
        format!(
            "{};intent={};clause={:?};epistemic={:?};focus={};roles={};confidence={:.4};rate={:.4};pitch={:.4};prominence={:.4};pause={:.4};intonation={:?}",
            self.version,
            self.intent,
            self.clause_mode,
            self.epistemic_delivery,
            self.focus_role.as_deref().unwrap_or("NONE"),
            roles,
            self.confidence,
            self.prosody.rate,
            self.prosody.pitch_range,
            self.prosody.prominence,
            self.prosody.pause_weight,
            self.prosody.intonation,
        )
    }

    /// True when this plan explicitly refuses an assertive answer.
    pub fn is_non_assertive(&self) -> bool {
        matches!(self.epistemic_delivery, EpistemicDelivery::NonAssertive)
    }
}

fn epistemic_delivery(ordinal: f32, confidence: f32) -> EpistemicDelivery {
    let ordinal = if ordinal.is_finite() {
        ordinal.clamp(0.0, 4.0)
    } else {
        3.0
    };
    let confidence = if confidence.is_finite() {
        confidence.clamp(0.0, 1.0)
    } else {
        0.0
    };

    if ordinal >= 3.0 {
        EpistemicDelivery::NonAssertive
    } else if ordinal >= 2.0 || confidence < 0.55 {
        EpistemicDelivery::Qualified
    } else {
        EpistemicDelivery::Assertive
    }
}

fn clause_mode_for(intent: &str, epistemic: EpistemicDelivery) -> ClauseMode {
    if matches!(epistemic, EpistemicDelivery::NonAssertive) {
        return ClauseMode::Abstention;
    }

    match intent {
        "question" => ClauseMode::Question,
        "create" | "propose" => ClauseMode::Directive,
        "reflect" => ClauseMode::Reflective,
        "relate" => ClauseMode::Relational,
        _ => ClauseMode::Statement,
    }
}

fn focus_role_for(intent: &str, epistemic: EpistemicDelivery) -> Option<String> {
    if matches!(epistemic, EpistemicDelivery::NonAssertive) {
        return None;
    }

    let role = match intent {
        "question" => "PATIENT",
        "explain" | "answer" => "PREDICATE",
        "create" | "propose" => "ACTION",
        "reflect" => "EVALUATOR",
        "relate" => "PATIENT",
        "analyze" => "PREDICATE",
        _ => return None,
    };
    Some(role.to_string())
}

fn role_salience(role: &str, focus_role: Option<&str>) -> f32 {
    if Some(role) == focus_role {
        1.0
    } else {
        match role {
            "AGENT" => 0.70,
            "ACTION" => 0.85,
            "PATIENT" => 0.80,
            "PREDICATE" => 0.75,
            "EVALUATOR" => 0.60,
            "TIME" => 0.45,
            "REASON" => 0.50,
            _ => 0.40,
        }
    }
}

fn derive_prosody(
    intent: &str,
    confidence: f32,
    warmth: f32,
    arousal: f32,
    time_pressure: f32,
    epistemic: EpistemicDelivery,
) -> ProsodicIntent {
    let deliberate = match epistemic {
        EpistemicDelivery::Assertive => 0.0,
        EpistemicDelivery::Qualified => 0.20,
        EpistemicDelivery::NonAssertive => 0.45,
    };

    let rate = (0.92 + 0.30 * time_pressure + 0.08 * arousal - deliberate).clamp(0.55, 1.35);
    let pitch_range = (0.88 + 0.38 * arousal + 0.10 * warmth).clamp(0.65, 1.45);
    let prominence = (0.35 + 0.55 * confidence + 0.10 * arousal).clamp(0.0, 1.0);
    let pause_weight =
        (0.18 + 0.52 * (1.0 - confidence) + 0.20 * deliberate + 0.10 * (1.0 - warmth))
            .clamp(0.0, 1.0);

    let intonation = match intent {
        "question" => IntonationIntent::Question,
        "create" | "propose" if arousal > 0.8 => IntonationIntent::Exclamation,
        _ => IntonationIntent::Statement,
    };

    ProsodicIntent {
        rate,
        pitch_range,
        prominence,
        pause_weight,
        intonation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::StructuredDecoder;
    use symthaea_core::genesis::GenesisSeed;

    #[test]
    fn answer_produces_assertive_statement_plan() {
        let genesis = GenesisSeed::from_phrase("broca-speech-plan-answer");
        let decoder = StructuredDecoder::new(&genesis);
        let mut channels = ThoughtChannels::with_intent(4);
        channels.set_epistemic(0.0);
        channels.set_consciousness(0.8, 0.8, 0.9);

        let readout = decoder.decode(&channels);
        let plan = SpeechPlan::from_readout(&channels, &readout);

        assert_eq!(plan.clause_mode, ClauseMode::Statement);
        assert_eq!(plan.epistemic_delivery, EpistemicDelivery::Assertive);
        assert_eq!(plan.focus_role.as_deref(), Some("PREDICATE"));
        assert!(!plan.is_non_assertive());
    }

    #[test]
    fn question_gets_question_intonation_and_patient_focus() {
        let genesis = GenesisSeed::from_phrase("broca-speech-plan-question");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(3);

        let readout = decoder.decode(&channels);
        let plan = SpeechPlan::from_readout(&channels, &readout);

        assert_eq!(plan.clause_mode, ClauseMode::Question);
        assert_eq!(plan.prosody.intonation, IntonationIntent::Question);
        assert_eq!(plan.focus_role.as_deref(), Some("PATIENT"));
    }

    #[test]
    fn unknown_is_terminally_non_assertive() {
        let genesis = GenesisSeed::from_phrase("broca-speech-plan-unknown");
        let decoder = StructuredDecoder::new(&genesis);
        let mut channels = ThoughtChannels::with_intent(7);
        channels.set_epistemic(3.0);

        let readout = decoder.decode(&channels);
        let plan = SpeechPlan::from_readout(&channels, &readout);

        assert_eq!(plan.clause_mode, ClauseMode::Abstention);
        assert_eq!(plan.epistemic_delivery, EpistemicDelivery::NonAssertive);
        assert!(plan.focus_role.is_none());
        assert!(plan.monitor.fail_closed_on_unknown);
        assert!(plan.is_non_assertive());
    }

    #[test]
    fn planning_is_sensitive_to_time_pressure_and_arousal() {
        let genesis = GenesisSeed::from_phrase("broca-speech-plan-prosody");
        let decoder = StructuredDecoder::new(&genesis);
        let mut channels = ThoughtChannels::with_intent(4);
        channels.set_epistemic(0.0);
        channels.set_emotion(0.2, 0.2, 0.5);
        let readout = decoder.decode(&channels);
        let calm = SpeechPlan::from_readout(&channels, &readout);

        channels.channels[20] = 1.0;
        channels.set_emotion(0.2, 1.0, 0.5);
        let readout = decoder.decode(&channels);
        let urgent = SpeechPlan::from_readout(&channels, &readout);

        assert!(urgent.prosody.rate > calm.prosody.rate);
        assert!(urgent.prosody.pitch_range > calm.prosody.pitch_range);
    }

    #[test]
    fn grounding_surface_is_deterministic_and_contains_plan_identity() {
        let genesis = GenesisSeed::from_phrase("broca-speech-plan-surface");
        let decoder = StructuredDecoder::new(&genesis);
        let channels = ThoughtChannels::with_intent(2);
        let readout = decoder.decode(&channels);
        let plan = SpeechPlan::from_readout(&channels, &readout);

        let first = plan.grounding_surface();
        let second = plan.grounding_surface();

        assert_eq!(first, second);
        assert!(first.starts_with("broca-speech-plan-v1;"));
        assert!(first.contains("intent=explain"));
    }
}
