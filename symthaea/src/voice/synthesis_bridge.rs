// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Synthesis Bridge: CycleResult to ThoughtChannels Conversion
//!
//! Converts the cognitive loop's `CycleResult` into Broca's `ThoughtChannels`
//! for neural language generation. This is the bridge between cognition and speech.
//!
//! ## Pipeline Position
//!
//! ```text
//! CognitiveLoopService::cycle()
//!     │
//!     ▼
//! CycleResult  ──► cycle_result_to_thought_channels() ──► ThoughtChannels
//!                                                              │
//!                                                              ▼
//!                                              BrocaGenerator::generate()
//! ```
//!
//! ## Channel Mapping
//!
//! | CycleResult field        | ThoughtChannels channel       |
//! |--------------------------|-------------------------------|
//! | detected_primitives      | intent (one-hot, channels 0-7)|
//! | metadata.affective_*     | valence, arousal, warmth      |
//! | prediction_error         | epistemic_status              |
//! | metadata.psi_*           | psi, meta_awareness, coherence|

#[cfg(feature = "ssm_language")]
use symthaea_broca::encoder::ThoughtChannels;

#[cfg(feature = "ssm_language")]
use crate::cognitive_loop::types::CycleResult;

/// Map a detected-primitive keyword to a Broca intent index.
///
/// Intent channels (indices 0-7):
/// 0=acknowledge, 1=answer, 2=clarify, 3=propose,
/// 4=uncertainty, 5=reflect, 6=continue, 7=unknown.
///
/// Falls back to 7 (unknown) if no keyword matches.
#[allow(dead_code)] // Used by cfg(ssm_language) cycle_result_to_thought_channels + tests
fn primitive_to_intent_index(primitives: &[String]) -> usize {
    for p in primitives {
        let lower = p.to_lowercase();
        if lower.contains("assert") || lower.contains("acknowledge") || lower.contains("confirm") {
            return 0; // acknowledge
        }
        if lower.contains("answer") || lower.contains("respond") || lower.contains("explain") {
            return 1; // answer
        }
        // Check "continue" before "clarify" because "task" contains "ask"
        if lower.contains("continu") || lower.contains("proceed") || lower.contains("next") {
            return 6; // continue
        }
        if lower.contains("query") || lower.contains("clarif") || lower.contains("ask") {
            return 2; // clarify
        }
        if lower.contains("propos") || lower.contains("suggest") || lower.contains("recommend") {
            return 3; // propose
        }
        if lower.contains("uncertain") || lower.contains("maybe") || lower.contains("doubt") {
            return 4; // uncertainty
        }
        if lower.contains("reflect") || lower.contains("consider") || lower.contains("think") {
            return 5; // reflect
        }
    }
    7 // unknown
}

/// Extract `ThoughtChannels` from a `CycleResult` for Broca generation.
///
/// Maps cognitive loop outputs to the 20-dimensional `ThoughtChannels` struct:
///
/// - **Intent**: Derived from `detected_primitives` keywords (one-hot encoding).
/// - **Epistemic status**: Mapped from `prediction_error` (low error = Certain,
///   high error = Unknown).
/// - **Emotion**: Taken from `metadata.affective_valence` and `metadata.affective_arousal`.
/// - **Consciousness**: Psi from `metadata.attention.psi_attention_avg`, coherence from
///   thought vector norm.
///
/// # Arguments
///
/// * `result` - The output of `CognitiveLoopService::cycle()`.
///
/// # Example
///
/// ```ignore
/// let result = cognitive_loop.cycle("Hello");
/// let channels = cycle_result_to_thought_channels(&result);
/// let text = broca.generate(&channels);
/// ```
#[cfg(feature = "ssm_language")]
pub fn cycle_result_to_thought_channels(result: &CycleResult) -> ThoughtChannels {
    let intent_idx = primitive_to_intent_index(&result.detected_primitives);
    let mut channels = ThoughtChannels::with_intent(intent_idx);

    // Epistemic status from prediction error:
    // 0=Certain (<0.1), 1=Probable (<0.3), 2=Uncertain (<0.6), 3=Unknown (>=0.6)
    let epistemic = if result.prediction_error < 0.1 {
        0.0 // Certain
    } else if result.prediction_error < 0.3 {
        1.0 // Probable
    } else if result.prediction_error < 0.6 {
        2.0 // Uncertain
    } else {
        3.0 // Unknown
    };
    channels.set_epistemic(epistemic);

    // Emotion from CycleMetadata affective bridge
    let valence = result.metadata.embodied.affective_valence;
    let arousal = result.metadata.embodied.affective_arousal;
    let warmth = 0.5; // Default neutral warmth (no direct source in CycleResult)
    channels.set_emotion(valence, arousal, warmth);

    // Consciousness metrics
    let psi = result.metadata.attention.psi_attention_avg;
    // Use thought_vector norm as a proxy for meta-awareness (bounded to [0, 1])
    let meta_awareness = if result.thought_vector.is_empty() {
        0.5
    } else {
        let norm: f32 = result
            .thought_vector
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        (norm / result.thought_vector.len() as f32).clamp(0.0, 1.0)
    };
    // Coherence from prediction confidence (1 - error, as a rough proxy)
    let coherence = (1.0 - result.prediction_error).clamp(0.0, 1.0);
    channels.set_consciousness(psi, meta_awareness, coherence);

    // has_computed_answer: true if learning occurred (the system produced a response)
    channels.channels[18] = if result.learning_occurred { 1.0 } else { 0.0 };

    // concept_count: number of detected primitives
    channels.channels[19] = (result.detected_primitives.len() as f32).min(10.0);

    channels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_to_intent_acknowledge() {
        assert_eq!(primitive_to_intent_index(&["assert".to_string()]), 0);
        assert_eq!(
            primitive_to_intent_index(&["confirm_result".to_string()]),
            0
        );
    }

    #[test]
    fn test_primitive_to_intent_answer() {
        assert_eq!(primitive_to_intent_index(&["answer_user".to_string()]), 1);
        assert_eq!(
            primitive_to_intent_index(&["explain_concept".to_string()]),
            1
        );
    }

    #[test]
    fn test_primitive_to_intent_clarify() {
        assert_eq!(primitive_to_intent_index(&["query_db".to_string()]), 2);
        assert_eq!(primitive_to_intent_index(&["ask_question".to_string()]), 2);
    }

    #[test]
    fn test_primitive_to_intent_propose() {
        assert_eq!(
            primitive_to_intent_index(&["suggest_action".to_string()]),
            3
        );
    }

    #[test]
    fn test_primitive_to_intent_uncertainty() {
        assert_eq!(primitive_to_intent_index(&["maybe_true".to_string()]), 4);
    }

    #[test]
    fn test_primitive_to_intent_reflect() {
        assert_eq!(primitive_to_intent_index(&["reflect_on".to_string()]), 5);
    }

    #[test]
    fn test_primitive_to_intent_continue() {
        assert_eq!(primitive_to_intent_index(&["continue_task".to_string()]), 6);
    }

    #[test]
    fn test_primitive_to_intent_unknown_fallback() {
        assert_eq!(primitive_to_intent_index(&["random_word".to_string()]), 7);
        assert_eq!(primitive_to_intent_index(&[]), 7);
    }

    #[test]
    fn test_primitive_priority() {
        // First matching keyword wins
        assert_eq!(
            primitive_to_intent_index(&[
                "some_noise".to_string(),
                "explain_thing".to_string(), // answer (1)
                "query_data".to_string(),    // clarify (2) — but answer came first
            ]),
            1
        );
    }
}
