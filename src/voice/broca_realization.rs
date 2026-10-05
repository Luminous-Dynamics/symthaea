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

use symthaea_broca::{IntonationIntent, PhonemeSlot, ProsodicIntent, SpeechPlan, SpeechSensoryTarget};
use symthaea_vocal_tract::pipeline::{Intonation, PitchAccent, ProsodyContext, SourceType, predict_duration};

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

impl BrocaFramePosition {
    /// Construct frame metadata directly from an explicit phonological slot.
    ///
    /// Timing and neighboring source types remain owned by the realization scheduler;
    /// stress, onset, and information-structural focus come from the phonological plan.
    pub fn from_phoneme_slot(
        slot: &PhonemeSlot,
        utterance_progress: f32,
        phoneme_progress: f32,
        phrase_index: u8,
        phrase_progress: f32,
        syllable_progress: f32,
        prev_source_type: Option<SourceType>,
        next_source_type: Option<SourceType>,
    ) -> Self {
        Self {
            utterance_progress,
            phoneme_progress,
            stress: slot.stress.ordinal().min(2),
            phrase_index,
            phrase_progress,
            is_focus: slot.is_focus,
            is_syllable_onset: slot.is_syllable_onset,
            syllable_progress,
            prev_source_type,
            next_source_type,
        }
    }
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
            is_focus: position.is_focus,
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

    /// Compute scheduler-owned duration frames using the exact Broca rate target.
    ///
    /// This is a typed timing projection, not proof that every live voice path currently
    /// consumes it. Word/utterance-final and stress effects remain explicit inputs to the
    /// realization primitive rather than being folded into the Broca rate field.
    pub fn duration_frames(
        &self,
        phoneme: &str,
        stress: u8,
        is_word_final: bool,
        is_utterance_final: bool,
    ) -> usize {
        predict_duration(
            phoneme,
            stress.min(2),
            is_word_final,
            is_utterance_final,
            self.rate_target(),
        )
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
    fn phonological_slot_binds_stress_focus_and_onset_metadata() {
        let slot = PhonemeSlot::new(
            "AH1",
            0,
            symthaea_broca::SyllableStress::Primary,
            true,
            true,
            false,
        );
        let position = BrocaFramePosition::from_phoneme_slot(
            &slot,
            0.25,
            0.5,
            0,
            0.25,
            0.1,
            None,
            None,
        );

        assert_eq!(position.stress, 1);
        assert!(position.is_focus);
        assert!(position.is_syllable_onset);
    }

    #[test]
    fn rate_target_maps_monotonically_to_scheduler_duration() {
        let mut slow_plan = plan_for(4, 0.0);
        slow_plan.prosody.rate = 0.70;
        let slow = BrocaProsodyAdapter::new(slow_plan);

        let mut fast_plan = plan_for(4, 0.0);
        fast_plan.prosody.rate = 1.30;
        let fast = BrocaProsodyAdapter::new(fast_plan);

        let slow_frames = slow.duration_frames("AH", 1, false, false);
        let fast_frames = fast.duration_frames("AH", 1, false, false);

        assert!(
            slow_frames > fast_frames,
            "lower speaking rate must produce longer scheduler duration: slow={slow_frames}, fast={fast_frames}"
        );
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
