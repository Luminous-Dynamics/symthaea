// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for Voice Pipeline

Fuzzes the vocal tract pipeline with random cognitive states, phoneme sequences,
and FEP observations to verify:

1. All FormantFrame fields remain finite (no NaN/Inf)
2. Formant frequencies stay within physical bounds
3. FEP free energy and prediction error remain finite
4. Repeated identical states don't cause divergence
*/

#![cfg(feature = "vocal-tract")]

use proptest::prelude::*;
use symthaea_core::genesis::GenesisSeed;
use symthaea_vocal_tract::encoder::VoiceCognitiveState;
use symthaea_vocal_tract::fep::{VocalTractFepAgent, VocalTractObservation};
use symthaea_vocal_tract::pipeline::VocalTractPipeline;

// ═══════════════════════════════════════════════════════════════════════════════
// Strategies
// ═══════════════════════════════════════════════════════════════════════════════

/// Random VoiceCognitiveState within documented channel ranges.
fn voice_state_strategy() -> impl Strategy<Value = VoiceCognitiveState> {
    (
        0.0f32..2.0,                // prediction_error
        -1.0f32..1.0,               // emotional_valence
        0.0f32..1.0,                // emotional_arousal
        0.0f32..1.0,                // unified_quality
        0.0f32..1.0,                // epistemic_confidence
        -1.0f32..1.0,               // coherence_velocity
        0.0f32..1.0,                // cross_agreement
        0.0f32..1.0,                // consciousness_level
        0.0f32..1.0,                // articulation_quality
        0.0f32..1.0,                // rate_stability
        (0.0f32..2.0, 0.0f32..5.0), // integrated_phi, expected_free_energy
    )
        .prop_map(|(pe, val, aro, uq, ec, cv, ca, cl, aq, rs, (phi, efe))| {
            VoiceCognitiveState {
                prediction_error: pe,
                emotional_valence: val,
                emotional_arousal: aro,
                unified_quality: uq,
                epistemic_confidence: ec,
                coherence_velocity: cv,
                cross_agreement: ca,
                consciousness_level: cl,
                articulation_quality: aq,
                rate_stability: rs,
                integrated_phi: phi,
                expected_free_energy: efe,
            }
        })
}

/// Random ARPABET phoneme name.
fn phoneme_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("AH".to_string()),
        Just("IY".to_string()),
        Just("UW".to_string()),
        Just("AE".to_string()),
        Just("AA".to_string()),
        Just("EH".to_string()),
        Just("IH".to_string()),
        Just("OW".to_string()),
        Just("P".to_string()),
        Just("T".to_string()),
        Just("K".to_string()),
        Just("S".to_string()),
        Just("M".to_string()),
        Just("N".to_string()),
        Just("SH".to_string()),
        Just("L".to_string()),
    ]
}

/// Random phoneme sequence.
fn phoneme_sequence(min: usize, max: usize) -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(phoneme_strategy(), min..=max)
}

/// Random VocalTractObservation (6D, all in [0, 1]).
fn observation_strategy() -> impl Strategy<Value = VocalTractObservation> {
    (
        0.0f64..1.0,
        0.0f64..1.0,
        0.0f64..1.0,
        0.0f64..1.0,
        0.0f64..1.0,
        0.0f64..1.0,
    )
        .prop_map(
            |(art, form, pitch, coart, dur, energy)| VocalTractObservation {
                articulation_score: art,
                formant_accuracy: form,
                pitch_stability: pitch,
                coarticulation_smoothness: coart,
                duration_accuracy: dur,
                energy_consistency: energy,
            },
        )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Properties
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    /// All FormantFrame fields must be finite (no NaN/Inf) for any cognitive state + phoneme.
    #[test]
    fn prop_formant_frame_always_finite(
        states in prop::collection::vec(voice_state_strategy(), 20..=40),
        phonemes in phoneme_sequence(20, 40),
    ) {
        let genesis = GenesisSeed::from_phrase("proptest-finite");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let dt = 0.005;

        for (i, state) in states.iter().enumerate() {
            let ph = phonemes.get(i).map(|s| s.as_str());
            let frame = pipeline.tick_phoneme(state, None, dt, ph);
            prop_assert!(frame.f1.is_finite(), "f1 NaN/Inf at frame {i}");
            prop_assert!(frame.f2.is_finite(), "f2 NaN/Inf at frame {i}");
            prop_assert!(frame.f3.is_finite(), "f3 NaN/Inf at frame {i}");
            prop_assert!(frame.b1.is_finite(), "b1 NaN/Inf at frame {i}");
            prop_assert!(frame.b2.is_finite(), "b2 NaN/Inf at frame {i}");
            prop_assert!(frame.b3.is_finite(), "b3 NaN/Inf at frame {i}");
            prop_assert!(frame.f0.is_finite(), "f0 NaN/Inf at frame {i}");
            prop_assert!(frame.energy.is_finite(), "energy NaN/Inf at frame {i}");
            prop_assert!(frame.voicing.is_finite(), "voicing NaN/Inf at frame {i}");
        }
    }

    /// FormantFrame values must stay within physical bounds.
    #[test]
    fn prop_formant_frame_within_bounds(
        states in prop::collection::vec(voice_state_strategy(), 20..=40),
        phonemes in phoneme_sequence(20, 40),
    ) {
        let genesis = GenesisSeed::from_phrase("proptest-bounds");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let dt = 0.005;

        for (i, state) in states.iter().enumerate() {
            let ph = phonemes.get(i).map(|s| s.as_str());
            let frame = pipeline.tick_phoneme(state, None, dt, ph);
            prop_assert!(frame.f1 >= 200.0 && frame.f1 <= 1000.0,
                "f1={} out of [200,1000] at frame {i}", frame.f1);
            prop_assert!(frame.f2 >= 600.0 && frame.f2 <= 3000.0,
                "f2={} out of [600,3000] at frame {i}", frame.f2);
            prop_assert!(frame.f3 >= 1500.0 && frame.f3 <= 5000.0,
                "f3={} out of [1500,5000] at frame {i}", frame.f3);
            prop_assert!(frame.energy >= 0.0 && frame.energy <= 1.0,
                "energy={} out of [0,1] at frame {i}", frame.energy);
            prop_assert!(frame.voicing >= 0.0 && frame.voicing <= 1.0,
                "voicing={} out of [0,1] at frame {i}", frame.voicing);
        }
    }

    /// FEP free energy and prediction error must remain finite for random observations.
    #[test]
    fn prop_fep_free_energy_finite(
        observations in prop::collection::vec(observation_strategy(), 20..=50),
    ) {
        let mut agent = VocalTractFepAgent::new();

        for (i, obs) in observations.iter().enumerate() {
            let result = agent.tick(obs);
            prop_assert!(result.free_energy.is_finite(),
                "free_energy NaN/Inf at tick {i}");
            prop_assert!(result.prediction_error.is_finite(),
                "prediction_error NaN/Inf at tick {i}");
            prop_assert!(result.tau_factor > 0.0,
                "tau_factor non-positive at tick {i}: {}", result.tau_factor);
        }
    }

    /// Repeated identical state must not cause divergence.
    #[test]
    fn prop_repeated_state_stable(
        state in voice_state_strategy(),
        n_frames in 40usize..=100,
    ) {
        let genesis = GenesisSeed::from_phrase("proptest-stable");
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let dt = 0.005;

        let mut prev_f1: Option<f32> = None;
        for i in 0..n_frames {
            let frame = pipeline.tick_phoneme(&state, None, dt, Some("AH"));
            prop_assert!(frame.f1.is_finite(), "f1 diverged at frame {i}");

            // After settling (20 frames), formants should be relatively stable
            if i > 20 {
                if let Some(prev) = prev_f1 {
                    let delta: f32 = frame.f1 - prev;
                    prop_assert!(delta.abs() < 50.0,
                        "f1 jumped {}Hz at frame {i} (divergence)", delta.abs());
                }
            }
            prev_f1 = Some(frame.f1);
        }
    }
}