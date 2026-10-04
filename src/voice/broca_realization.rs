// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Broca -> vocal-tract realization adapter.
//!
//! Kept in the root voice layer to avoid a dependency cycle between the Broca
//! and vocal-tract domain crates. Broca owns linguistic intent; the vocal tract
//! owns articulatory realization. This bridge translates only the shared,
//! versioned production intent.

#![cfg(all(feature = "ssm_language", feature = "vocal-tract"))]

use symthaea_broca::{IntonationIntent, ProsodicIntent, SpeechPlan, SpeechSensoryTarget};
use symthaea_vocal_tract::pipeline::{Intonation, PitchAccent, ProsodyContext, SourceType};

/// Timing/articulation state supplied by the phonological/realization layer.
#[derive(Debug, Clone, Copy)]
pub struct BrocaFramePosition {
    pub utterance_progress: f32,
    pub phoneme_progress: f32,
    pub stress: u8,
    pub phrase_index: u8,
    pub phrase_progress: f32,
    pub is_focus: bool,
    pub is_syllable_onset: bool,
    pub syllable_progress: f32,
    pub prev_source_type: Option<SourceType>,
    pub next_source_type: Option<SourceType>,
}

impl Default for BrocaFramePosition {
    fn default() -> Self {
        Self {
            utterance_progress: 0.0,
            phoneme_progress: 0.5,
            stress: 0,
            phrase_index: 0,
            phrase_progress: 0.0,
            is_focus: false,
            is_syllable_onset: true,
            syllable_progress: 0.0,
            prev_source_type: None,
            next_source_type: None,
        }
    }
}

/// Root-layer adapter from Broca's production plan into the existing vocal tract API.
#[derive(Debug, Clone, Copy)]
pub struct BrocaProsodyAdapter {
    plan: SpeechPlan,
}

impl BrocaProsodyAdapter {
    pub fn new(plan: SpeechPlan) -> Self {
        Self { plan }
    }

    pub fn plan(&self) -> &SpeechPlan {
        &self.plan
    }

    /// Return the realization target used by closed-loop feedback.
    pub fn sensory_target(&self) -> SpeechSensoryTarget {
        SpeechSensoryTarget::from_plan(&self.plan)
    }

    /// Convert a plan into one vocal-tract prosody frame.
    ///
    /// base_f0 remains owned by the voice identity/configuration. Broca contributes
    /// its affective state and intonation intent rather than overwriting speaker anatomy.
    pub fn context(&self, base_f0: f32, position: BrocaFramePosition) -> ProsodyContext {
        ProsodyContext {
            utterance_progress: sanitize_unit(position.utterance_progress),
            phoneme_progress: sanitize_unit(position.phoneme_progress),
            stress: position.stress.min(2),
            base_f0: if base_f0.is_finite() && base_f0 > 0.0 {
                base_f0
            } else {
                120.0
            },
            arousal: self.plan.arousal.clamp(0.0, 1.0),
            intonation: map_intonation(self.plan.prosody.intonation),
            phrase_index: position.phrase_index,
            phrase_progress: sanitize_unit(position.phrase_progress),
            is_focus: position.is_focus || focus_matches(position, &self.plan),
            pitch_accent: pitch_accent_for(&self.plan.prosody, position),
            is_syllable_onset: position.is_syllable_onset,
            syllable_progress: sanitize_unit(position.syllable_progress),
            prev_source_type: position.prev_source_type,
            next_source_type: position.next_source_type,
        }
    }

    /// Whether the plan asks the realization layer to slow down relative to neutral.
    pub fn deliberate(&self) -> bool {
        self.plan.prosody.pause_weight > 0.45
            || matches!(
                self.plan.epistemic_delivery,
                symthaea_broca::EpistemicDelivery::NonAssertive
            )
    }

    /// Relative rate target for the caller's duration/timing subsystem.
    pub fn rate_target(&self) -> f32 {
        self.plan.prosody.rate
    }
}

fn sanitize_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn map_intonation(intent: IntonationIntent) -> Intonation {
    match intent {
        IntonationIntent::Statement => Intonation::Statement,
        IntonationIntent::Question => Intonation::Question,
        IntonationIntent::Exclamation => Intonation::Exclamation,
    }
}

fn pitch_accent_for(prosody: &ProsodicIntent, position: BrocaFramePosition) -> PitchAccent {
    if !position.is_syllable_onset && position.syllable_progress <= 0.0 {
        return PitchAccent::None;
    }

    if position.is_focus || prosody.prominence >= 0.82 {
        PitchAccent::RiseHigh
    } else if prosody.prominence >= 0.62 {
        PitchAccent::High
    } else if prosody.pause_weight >= 0.65 {
        PitchAccent::FallLow
    } else {
        PitchAccent::None
    }
}

fn focus_matches(position: BrocaFramePosition, plan: &SpeechPlan) -> bool {
    position.is_focus && plan.focus_role.is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_broca::{ClauseMode, EpistemicDelivery, StructuredDecoder, SpeechPlan};
    use symthaea_core::genesis::GenesisSeed;

    fn plan_for(intent: usize, epistemic: f32) -> SpeechPlan {
        let genesis = GenesisSeed::from_phrase("broca-prosody-adapter");
        let decoder = StructuredDecoder::new(&genesis);
        let mut channels = symthaea_broca::ThoughtChannels::with_intent(intent);
        channels.set_epistemic(epistemic);
        let readout = decoder.decode(&channels);
        SpeechPlan::from_readout(&channels, &readout)
    }

    #[test]
    fn directive_plan_maps_to_statement_intonation_by_default() {
        let plan = plan_for(3, 0.0);
        assert_eq!(plan.clause_mode, ClauseMode::Directive);

        let adapter = BrocaProsodyAdapter::new(plan);
        let context = adapter.context(150.0, BrocaFramePosition::default());

        assert_eq!(context.intonation, Intonation::Statement);
        assert!(context.base_f0 > 0.0);
    }

    #[test]
    fn question_intent_maps_to_question_intonation() {
        let plan = plan_for(3, 0.0);
        let mut question_plan = plan;
        question_plan.intent = "question".to_string();
        question_plan.clause_mode = ClauseMode::Question;
        question_plan.prosody.intonation = IntonationIntent::Question;

        let adapter = BrocaProsodyAdapter::new(question_plan);
        let context = adapter.context(150.0, BrocaFramePosition::default());

        assert_eq!(context.intonation, Intonation::Question);
    }

    #[test]
    fn plan_preserves_speaker_base_f0_and_plan_arousal() {
        let mut plan = plan_for(4, 0.0);
        plan.arousal = 0.9;

        let adapter = BrocaProsodyAdapter::new(plan);
        let context = adapter.context(
            187.0,
            BrocaFramePosition {
                utterance_progress: 0.5,
                ..Default::default()
            },
        );

        assert!((context.base_f0 - 187.0).abs() < f32::EPSILON);
        assert!((context.arousal - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn focused_high_prominence_gets_rise_high_accent() {
        let mut plan = plan_for(4, 0.0);
        plan.prosody.prominence = 0.9;

        let adapter = BrocaProsodyAdapter::new(plan);
        let context = adapter.context(
            120.0,
            BrocaFramePosition {
                is_focus: true,
                is_syllable_onset: true,
                syllable_progress: 0.1,
                ..Default::default()
            },
        );

        assert_eq!(context.pitch_accent, PitchAccent::RiseHigh);
    }

    #[test]
    fn invalid_base_f0_falls_back_safely() {
        let plan = plan_for(4, 0.0);
        let adapter = BrocaProsodyAdapter::new(plan);
        let context = adapter.context(f32::NAN, BrocaFramePosition::default());

        assert_eq!(context.base_f0, 120.0);
    }

    #[test]
    fn rate_target_exposes_plan_without_rewriting_timing() {
        let plan = plan_for(4, 0.0);
        let adapter = BrocaProsodyAdapter::new(plan.clone());

        assert!((adapter.rate_target() - plan.prosody.rate).abs() < f32::EPSILON);
        assert_eq!(
            adapter.deliberate(),
            plan.prosody.pause_weight > 0.45
                || matches!(plan.epistemic_delivery, EpistemicDelivery::NonAssertive)
        );
    }
}
