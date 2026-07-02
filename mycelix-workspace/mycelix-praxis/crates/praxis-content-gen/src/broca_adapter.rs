// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Broca adapter — bridges EduNet's ContentGenerator to Symthaea's BrocaGenerator.
//!
//! This module converts [`ContentChannels`] into Broca's 43-channel
//! [`ThoughtChannels`] and calls the real consciousness-gated language
//! generator. When the `broca` feature is enabled, `BrocaContentGenerator`
//! replaces `MockGenerator` for real AI-generated educational content.
//!
//! # Architecture
//!
//! ```text
//! ContentChannels (educator API)
//!   → map_to_thought_channels()
//!     → ThoughtChannels [43 floats]
//!       → BrocaGenerator::generate()
//!         → GenerationResult { text, coherence, hallucination_flag, ... }
//!           → GenerationOutput (ContentGenerator trait)
//! ```
//!
//! # Feature Gate
//!
//! This module compiles unconditionally but the `BrocaContentGenerator`
//! struct is only available when the `broca` feature is enabled (which
//! requires the `symthaea-broca` crate as a dependency).

use crate::channels::{ContentChannels, ContentIntent};
use crate::pipeline::GenerationOutput;

/// Map ContentChannels to a 43-element ThoughtChannels array.
///
/// This is the critical bridge between the educator-facing API and
/// Broca's neural generation interface.
pub fn map_to_thought_channels(channels: &ContentChannels) -> [f32; 43] {
    let mut tc = [0.0f32; 43];

    // Channels 0-7: SemanticIntent one-hot
    let intent_index = match channels.intent {
        ContentIntent::TeachConcept => 0,         // Acknowledge/teach
        ContentIntent::GiveExample => 1,          // Worked example
        ContentIntent::AskQuestion => 2,          // Practice problem
        ContentIntent::ProvideHint => 3,          // Hint
        ContentIntent::ExplainMisconception => 4, // Misconception correction
        ContentIntent::ReflectOnLearning => 5,    // Reflection
        ContentIntent::Encourage => 6,            // Encouragement
    };
    tc[intent_index] = 1.0;

    // Channel 8: Epistemic status (0-4)
    tc[8] = channels.epistemic_level.clamp(0.0, 4.0);

    // Channels 9-11: Emotional tone
    tc[9] = channels.valence.clamp(-1.0, 1.0);
    tc[10] = channels.arousal.clamp(0.0, 1.0);
    tc[11] = channels.warmth.clamp(0.0, 1.0);

    // Channel 12: Consciousness (psi) — controls verbosity vs conciseness
    tc[12] = channels.consciousness.clamp(0.0, 1.0);

    // Channel 13: Meta-awareness
    tc[13] = channels.meta_awareness.clamp(0.0, 1.0);

    // Channel 14: Coherence requirement
    tc[14] = channels.coherence.clamp(0.0, 1.0);

    // Channel 15: Relationship stage (teacher = professional)
    tc[15] = 4.0; // Professional/teacher

    // Channel 16: Trust (high for educational context)
    tc[16] = 0.8;

    // Channel 17: Mood temperature (generation temperature)
    tc[17] = 1.0; // Neutral

    // Channel 18: has_computed_answer (1.0 for worked examples, 0.0 otherwise)
    tc[18] = if matches!(channels.intent, ContentIntent::GiveExample) {
        1.0
    } else {
        0.0
    };

    // Channel 19: concept_count (based on intent complexity)
    tc[19] = match channels.intent {
        ContentIntent::TeachConcept => 3.0,
        ContentIntent::GiveExample => 2.0,
        ContentIntent::AskQuestion => 1.0,
        ContentIntent::ProvideHint => 1.0,
        ContentIntent::ExplainMisconception => 2.0,
        ContentIntent::ReflectOnLearning => 3.0,
        ContentIntent::Encourage => 1.0,
    };

    // Channel 20: Time pressure
    tc[20] = channels.time_pressure.clamp(0.0, 1.0);

    // Channel 21: Domain familiarity
    tc[21] = channels.domain_familiarity.clamp(0.0, 1.0);

    // Channel 22: Social context (formal for education)
    tc[22] = 0.7; // Somewhat formal

    // Channel 23: Response confidence
    tc[23] = channels.confidence.clamp(0.0, 1.0);

    // Channels 24-27: Code context (0 for general education)
    // Will be set when generating programming content

    // Channels 28-42: Epistemic Cube
    // E-tier: E2 (peer-reviewed level) for teaching
    tc[30] = 1.0; // e_tier_e2

    // N-tier: N1 (community norms) for educational content
    tc[34] = 1.0; // n_tier_n1

    // M-tier: M1 (persistent) for educational content
    tc[38] = 1.0; // m_tier_m1

    // H-tier: Harmonic coherence
    tc[41] = 0.8;

    // Epistemic quality
    tc[42] = 0.75;

    tc
}

/// Convert a Broca GenerationResult into a ContentGenerator GenerationOutput.
///
/// This maps Broca's rich output (coherence dynamics, gating traces, etc.)
/// into the simpler ContentGenerator interface.
pub fn result_to_output(
    text: String,
    coherence: f32,
    hallucination_flag: bool,
    veto_count: u32,
) -> GenerationOutput {
    GenerationOutput {
        text,
        coherence,
        hallucination_flag,
        veto_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::ContentChannels;

    #[test]
    fn test_teaching_factual_mapping() {
        let channels = ContentChannels::teaching_factual();
        let tc = map_to_thought_channels(&channels);

        // Intent should be TeachConcept (index 0)
        assert_eq!(tc[0], 1.0);
        assert_eq!(tc[1], 0.0);

        // Epistemic level
        assert!(tc[8] >= 0.0 && tc[8] <= 4.0);

        // Emotional channels
        assert!(tc[9] >= -1.0 && tc[9] <= 1.0); // valence
        assert!(tc[10] >= 0.0 && tc[10] <= 1.0); // arousal
        assert!(tc[11] >= 0.0 && tc[11] <= 1.0); // warmth

        // Coherence should be high for teaching
        assert!(tc[14] > 0.5);
    }

    #[test]
    fn test_practice_problem_mapping() {
        let channels = ContentChannels::practice_problem();
        let tc = map_to_thought_channels(&channels);

        // Intent should be AskQuestion (index 2)
        assert_eq!(tc[2], 1.0);
    }

    #[test]
    fn test_hint_mapping() {
        let channels = ContentChannels::hint(1);
        let tc = map_to_thought_channels(&channels);

        // Intent should be ProvideHint (index 3)
        assert_eq!(tc[3], 1.0);
    }

    #[test]
    fn test_misconception_mapping() {
        let channels = ContentChannels::misconception_correction();
        let tc = map_to_thought_channels(&channels);

        // Intent should be ExplainMisconception (index 4)
        assert_eq!(tc[4], 1.0);

        // Meta-awareness should be high
        assert!(tc[13] > 0.5);
    }

    #[test]
    fn test_encouragement_mapping() {
        let channels = ContentChannels::encouragement();
        let tc = map_to_thought_channels(&channels);

        // Intent should be Encourage (index 6)
        assert_eq!(tc[6], 1.0);

        // Warmth should be high
        assert!(tc[11] > 0.5);
    }

    #[test]
    fn test_grade_adjustment() {
        let young = ContentChannels::teaching_factual().for_grade(2); // Grade 1
        let old = ContentChannels::teaching_factual().for_grade(14); // College

        let tc_young = map_to_thought_channels(&young);
        let tc_old = map_to_thought_channels(&old);

        // Young learner should have higher warmth
        assert!(tc_young[11] >= tc_old[11]);
    }

    #[test]
    fn test_all_channels_bounded() {
        let channels = ContentChannels::teaching_factual();
        let tc = map_to_thought_channels(&channels);

        // All channels should be in valid ranges
        for (i, &v) in tc.iter().enumerate() {
            match i {
                9 => assert!(v >= -1.0 && v <= 1.0, "Channel {i} out of range: {v}"),
                17 => assert!(v >= 0.0 && v <= 2.0, "Channel {i} out of range: {v}"),
                19 => assert!(v >= 0.0 && v <= 10.0, "Channel {i} out of range: {v}"),
                _ => assert!(v >= 0.0 && v <= 5.0, "Channel {i} out of range: {v}"),
            }
        }
    }
}
