// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bridge from behavioral biometrics to symthaea-muse audio synthesis.
//!
//! Converts player stress state into `MusicalState` for real-time audio.
//! This is the "viral mechanic": as the player panics, the music becomes
//! hostile. As they calm down, the music heals.
//!
//! # The Feedback Loop
//!
//! ```text
//! Panic → erratic mouse → high velocity_surprise → high arousal
//!   → low serotonin + high dopamine + high NE
//!     → harsh FM, fast tempo, minor 7th chords
//!       → player hears hostile music → more panic → amplifies
//!
//! Calm → smooth mouse → low surprise → low arousal
//!   → allostatic_load decays → serotonin recovers
//!     → soft overtones, slow tempo, consonant intervals
//!       → music becomes soothing → stress decreases
//! ```

use crate::input_telemetry::StressVector;
use crate::stress_model::PlayerStressModel;
use symthaea_muse::MusicalState;

/// Convert player stress into a `MusicalState` for audio synthesis.
///
/// Maps psychological dimensions to neuromodulator levels and harmony
/// activations. The mapping is designed so that increasing stress produces
/// increasingly hostile, dissonant, fast-tempo audio.
pub fn stress_to_musical_state(stress: &StressVector, model: &PlayerStressModel) -> MusicalState {
    // Dopamine: arousal drives FM modulation depth (metallic/harsh at high stress)
    let dopamine = (stress.arousal * 0.8 + model.allostatic_load * 0.2).clamp(0.0, 1.0);

    // Serotonin: inversely tracks stress. Low serotonin = harsh timbre.
    // Also depressed by burnout via model.sht_baseline.
    let serotonin_raw = (1.0 + stress.valence) * 0.5 * (1.0 - model.allostatic_load * 0.5);
    let serotonin = (serotonin_raw * model.sht_baseline * 2.0).clamp(0.05, 1.0);

    // Noradrenaline: arousal + cortisol → edgy overtones, dissonant FM ratio
    let noradrenaline = (stress.arousal * 0.7 + model.cortisol_proxy * 0.3).clamp(0.0, 1.0);

    // Consciousness level: degrades under burnout (fewer polyphonic voices)
    let consciousness_level = (0.7 - model.allostatic_load * 0.5).clamp(0.1, 0.9);

    // Prediction error: cortisol proxy drives musical dissonance
    let prediction_error = (model.cortisol_proxy * 0.6 + stress.arousal * 0.2).clamp(0.0, 1.0);

    // Harmony activations: stress shifts tonal center toward tension
    let harmonies = compute_stress_harmonies(stress, model);

    MusicalState {
        harmony_activations: harmonies,
        dopamine,
        serotonin,
        noradrenaline,
        arousal: stress.arousal,
        valence: stress.valence,
        consciousness_level,
        prediction_error,
    }
}

/// Compute Eight Harmonies activations from stress state.
///
/// | Harmony | Index | Stress Behavior |
/// |---------|-------|-----------------|
/// | ResonantCoherence | 0 | Base consonance, fades with load |
/// | PanSentientFlourishing | 1 | Major 3rd: warmth, only when calm |
/// | IntegralWisdom | 2 | Perfect 5th: stable anchor |
/// | InfinitePlay | 3 | Minor 7th: tension, rises with arousal |
/// | UniversalInterconnectedness | 4 | Perfect 4th: unresolved, rises with load |
/// | SacredReciprocity | 5 | Major 6th: warmth, fades under stress |
/// | EvolutionaryProgression | 6 | Ascending: urgency, rises with arousal |
/// | SacredStillness | 7 | Drone: contemplation, only when deeply calm |
fn compute_stress_harmonies(stress: &StressVector, model: &PlayerStressModel) -> [f32; 8] {
    let a = stress.arousal;
    let load = model.allostatic_load;

    [
        // 0: ResonantCoherence — consonance erodes under stress
        (0.5 * (1.0 - a) * (1.0 - load)).max(0.05),
        // 1: PanSentientFlourishing — major 3rd only when calm and positive valence
        ((1.0 + stress.valence) * 0.25 * (1.0 - a)).max(0.0),
        // 2: IntegralWisdom — perfect 5th provides stable anchor
        0.4,
        // 3: InfinitePlay — minor 7th = tension, primary stress interval
        (a * 0.8 + load * 0.2).min(0.9),
        // 4: UniversalInterconnectedness — perfect 4th = unresolved
        (load * 0.6).min(0.7),
        // 5: SacredReciprocity — warmth fades with arousal
        ((1.0 - a) * 0.3).max(0.0),
        // 6: EvolutionaryProgression — ascending = urgency
        (a * 0.5).min(0.6),
        // 7: SacredStillness — drone only when deeply calm, no load
        ((1.0 - a) * (1.0 - load)).max(0.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calm_produces_warm_audio() {
        let stress = StressVector {
            arousal: 0.1,
            valence: 0.3,
            dominance: 0.2,
        };
        let model = PlayerStressModel::new();
        let state = stress_to_musical_state(&stress, &model);

        // Calm = high serotonin (soft timbre)
        assert!(state.serotonin > 0.3, "serotonin should be high when calm");
        // Calm = low dopamine (gentle FM)
        assert!(state.dopamine < 0.3, "dopamine should be low when calm");
        // Calm = SacredStillness drone active
        assert!(
            state.harmony_activations[7] > 0.5,
            "SacredStillness should be high when calm"
        );
        // Calm = low InfinitePlay (no tension)
        assert!(
            state.harmony_activations[3] < 0.3,
            "InfinitePlay should be low when calm"
        );
    }

    #[test]
    fn panic_produces_hostile_audio() {
        let stress = StressVector {
            arousal: 0.9,
            valence: -0.7,
            dominance: 0.1,
        };
        let mut model = PlayerStressModel::new();
        model.allostatic_load = 0.5;
        model.cortisol_proxy = 0.6;

        let state = stress_to_musical_state(&stress, &model);

        // Panic = high dopamine (deep FM modulation)
        assert!(state.dopamine > 0.6, "dopamine should be high in panic");
        // Panic = low serotonin (harsh timbre)
        assert!(state.serotonin < 0.4, "serotonin should be low in panic");
        // Panic = high NE (edgy overtones)
        assert!(state.noradrenaline > 0.5, "NE should be high in panic");
        // Panic = InfinitePlay (minor 7th tension)
        assert!(
            state.harmony_activations[3] > 0.6,
            "InfinitePlay should be high in panic"
        );
        // Panic = low SacredStillness
        assert!(
            state.harmony_activations[7] < 0.15,
            "SacredStillness should be low in panic"
        );
    }

    #[test]
    fn burnout_depresses_consciousness() {
        let stress = StressVector {
            arousal: 0.5,
            valence: -0.3,
            dominance: 0.0,
        };
        let mut model = PlayerStressModel::new();
        model.allostatic_load = 0.9;
        model.sht_baseline = 0.35; // burnout-depressed

        let state = stress_to_musical_state(&stress, &model);

        assert!(
            state.consciousness_level < 0.3,
            "consciousness should be low under burnout, got {}",
            state.consciousness_level
        );
        assert!(
            state.serotonin < 0.25,
            "serotonin should be very low with depressed baseline, got {}",
            state.serotonin
        );
    }

    #[test]
    fn all_fields_bounded() {
        // Test with extreme values
        for &arousal in &[0.0, 0.5, 1.0] {
            for &valence in &[-1.0, 0.0, 1.0] {
                let stress = StressVector {
                    arousal,
                    valence,
                    dominance: 0.0,
                };
                let mut model = PlayerStressModel::new();
                model.allostatic_load = arousal;
                model.cortisol_proxy = arousal;
                let state = stress_to_musical_state(&stress, &model);

                assert!(state.dopamine >= 0.0 && state.dopamine <= 1.0);
                assert!(state.serotonin >= 0.0 && state.serotonin <= 1.0);
                assert!(state.noradrenaline >= 0.0 && state.noradrenaline <= 1.0);
                assert!(state.arousal >= 0.0 && state.arousal <= 1.0);
                assert!(state.valence >= -1.0 && state.valence <= 1.0);
                assert!(state.consciousness_level >= 0.0 && state.consciousness_level <= 1.0);
                assert!(state.prediction_error >= 0.0 && state.prediction_error <= 1.0);
                for h in &state.harmony_activations {
                    assert!(*h >= 0.0 && *h <= 1.0, "harmony out of bounds: {h}");
                }
            }
        }
    }

    #[test]
    fn stress_ramp_shifts_audio_continuously() {
        let model = PlayerStressModel::new();
        let mut prev_dopamine = 0.0;
        let mut prev_infinite_play = 0.0;

        // Ramp arousal from 0 to 1
        for i in 0..=10 {
            let arousal = i as f32 / 10.0;
            let stress = StressVector {
                arousal,
                valence: -arousal * 0.5,
                dominance: 0.0,
            };
            let state = stress_to_musical_state(&stress, &model);

            // Dopamine and tension should monotonically increase
            assert!(
                state.dopamine >= prev_dopamine - 0.01,
                "dopamine should increase with arousal"
            );
            assert!(
                state.harmony_activations[3] >= prev_infinite_play - 0.01,
                "InfinitePlay should increase with arousal"
            );

            prev_dopamine = state.dopamine;
            prev_infinite_play = state.harmony_activations[3];
        }
    }
}
