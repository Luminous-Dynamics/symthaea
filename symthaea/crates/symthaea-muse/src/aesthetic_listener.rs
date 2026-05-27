// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Aesthetic self-listener: the system judges its own musical beauty.
//!
//! Goes beyond raw audio features (RMS, ZCR) to understand musical semantics:
//! - **Consonance**: are the simultaneous frequencies harmonically related?
//! - **Harshness**: is the spectral content abrasive (high ZCR + high energy)?
//! - **Breathiness**: is there enough silence/space between notes?
//! - **Dynamic balance**: is the loudness appropriate for the consciousness phase?
//! - **Beauty score**: composite aesthetic judgment [0, 1]
//!
//! When beauty is low, the listener attenuates parameters that cause harshness:
//! reduces FM depth, lowers master gain, widens note spacing.

use crate::MusicalState;
use crate::audio_feedback::AudioFeatures;

/// Aesthetic quality assessment of generated audio.
#[derive(Debug, Clone, Copy, Default)]
pub struct AestheticAssessment {
    /// Consonance [0, 1]: how harmonically pure the sound is.
    pub consonance: f32,
    /// Harshness [0, 1]: 0 = smooth and warm, 1 = abrasive.
    pub harshness: f32,
    /// Breathiness [0, 1]: space between notes, room to breathe.
    pub breathiness: f32,
    /// Dynamic appropriateness [0, 1]: is loudness right for the mood?
    pub dynamic_balance: f32,
    /// Composite beauty score [0, 1].
    pub beauty: f32,
}

/// Self-listening aesthetic engine.
pub struct AestheticListener {
    /// Target RMS for each consciousness level bracket.
    target_rms_by_psi: [f32; 4],
    /// Maximum acceptable harshness before intervention.
    harshness_threshold: f32,
    /// Smoothed assessment (EMA).
    smoothed: AestheticAssessment,
    ema_alpha: f32,
}

impl AestheticListener {
    pub fn new() -> Self {
        Self {
            // Target RMS: whisper at low psi, gentle at peak
            target_rms_by_psi: [0.01, 0.02, 0.04, 0.06],
            harshness_threshold: 0.2, // intervene early
            smoothed: AestheticAssessment::default(),
            ema_alpha: 0.15, // respond faster
        }
    }

    /// Assess the aesthetic quality of the current audio output.
    pub fn assess(
        &mut self,
        features: &AudioFeatures,
        state: &MusicalState,
    ) -> AestheticAssessment {
        // Consonance: inverse of harmonic_tension × spectral noise
        // Low ZCR + low tension = consonant
        let consonance = (1.0 - features.harmonic_tension) * (1.0 - features.zero_crossing_rate);

        // Harshness: high ZCR × high energy = harsh
        let harshness = features.zero_crossing_rate * features.rms_energy * 2.0;
        let harshness = harshness.clamp(0.0, 1.0);

        // Breathiness: how much silence/space exists
        // Low RMS = more space
        let breathiness = (1.0 - features.rms_energy * 3.0).clamp(0.0, 1.0);

        // Dynamic balance: is loudness appropriate for consciousness level?
        let psi_bracket = ((state.consciousness_level * 3.0) as usize).min(3);
        let target_rms = self.target_rms_by_psi[psi_bracket];
        let rms_error = (features.rms_energy - target_rms).abs();
        let dynamic_balance = (1.0 - rms_error * 5.0).clamp(0.0, 1.0);

        // Beauty: weighted composite
        let beauty = consonance * 0.35
            + (1.0 - harshness) * 0.30
            + breathiness * 0.15
            + dynamic_balance * 0.20;

        let assessment = AestheticAssessment {
            consonance,
            harshness,
            breathiness,
            dynamic_balance,
            beauty: beauty.clamp(0.0, 1.0),
        };

        // EMA smoothing
        let a = self.ema_alpha;
        self.smoothed.consonance += a * (assessment.consonance - self.smoothed.consonance);
        self.smoothed.harshness += a * (assessment.harshness - self.smoothed.harshness);
        self.smoothed.breathiness += a * (assessment.breathiness - self.smoothed.breathiness);
        self.smoothed.dynamic_balance +=
            a * (assessment.dynamic_balance - self.smoothed.dynamic_balance);
        self.smoothed.beauty += a * (assessment.beauty - self.smoothed.beauty);

        assessment
    }

    /// Get EMA-smoothed assessment.
    pub fn smoothed_assessment(&self) -> &AestheticAssessment {
        &self.smoothed
    }

    /// Apply aesthetic corrections to the musical state.
    ///
    /// When the system sounds harsh, this gently pulls parameters back toward beauty:
    /// - Reduce FM depth (softer timbre)
    /// - Reduce arousal (calmer dynamics)
    /// - Increase serotonin (warmer rolloff)
    /// - Reduce noradrenaline (less edge)
    pub fn apply_corrections(&self, state: &mut MusicalState) {
        let harshness = self.smoothed.harshness;
        let beauty = self.smoothed.beauty;

        // Only intervene when harshness exceeds threshold or beauty is low
        if harshness > self.harshness_threshold {
            let correction = (harshness - self.harshness_threshold) * 0.8; // strong correction

            // Aggressively soften
            state.dopamine = (state.dopamine - correction * 0.8).max(0.05);
            state.noradrenaline = (state.noradrenaline - correction * 0.7).max(0.05);
            state.serotonin = (state.serotonin + correction * 0.5).min(0.95);

            // Pull back dynamics
            state.arousal = (state.arousal - correction * 0.4).max(0.05);
        }

        // If beauty is low, stronger structural correction
        if beauty < 0.4 {
            let urgency = (0.4 - beauty) * 0.6;
            state.prediction_error = (state.prediction_error - urgency * 0.4).max(0.0);
            state.harmony_activations[0] = (state.harmony_activations[0] + urgency * 0.15).min(1.0);
            state.harmony_activations[7] = (state.harmony_activations[7] + urgency * 0.1).min(1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiet_smooth_audio_is_beautiful() {
        let mut listener = AestheticListener::new();
        let features = AudioFeatures {
            spectral_centroid: 0.3,
            spectral_flux: 0.05,
            rhythm_entropy: 0.2,
            harmonic_tension: 0.1,
            rms_energy: 0.05,
            zero_crossing_rate: 0.03,
        };
        let state = MusicalState {
            consciousness_level: 0.5,
            ..Default::default()
        };
        let assessment = listener.assess(&features, &state);
        assert!(
            assessment.beauty > 0.6,
            "quiet smooth should be beautiful: {:.3}",
            assessment.beauty
        );
        assert!(
            assessment.harshness < 0.2,
            "should not be harsh: {:.3}",
            assessment.harshness
        );
    }

    #[test]
    fn loud_noisy_audio_is_harsh() {
        let mut listener = AestheticListener::new();
        let features = AudioFeatures {
            spectral_centroid: 0.8,
            spectral_flux: 0.6,
            rhythm_entropy: 0.7,
            harmonic_tension: 0.6,
            rms_energy: 0.8,
            zero_crossing_rate: 0.5,
        };
        let state = MusicalState {
            consciousness_level: 0.3,
            ..Default::default()
        };
        let assessment = listener.assess(&features, &state);
        assert!(
            assessment.harshness > 0.4,
            "loud noisy should be harsh: {:.3}",
            assessment.harshness
        );
        assert!(
            assessment.beauty < 0.5,
            "harsh audio should score low: {:.3}",
            assessment.beauty
        );
    }

    #[test]
    fn corrections_reduce_harshness_params() {
        let mut listener = AestheticListener::new();
        // Simulate harsh audio assessment
        listener.smoothed.harshness = 0.7;
        listener.smoothed.beauty = 0.2;

        let mut state = MusicalState {
            dopamine: 0.8,
            noradrenaline: 0.7,
            serotonin: 0.3,
            arousal: 0.8,
            ..Default::default()
        };
        let orig_dopamine = state.dopamine;
        let orig_ne = state.noradrenaline;

        listener.apply_corrections(&mut state);

        assert!(state.dopamine < orig_dopamine, "should reduce dopamine/FM");
        assert!(state.noradrenaline < orig_ne, "should reduce NE/edge");
        assert!(state.serotonin > 0.3, "should increase serotonin/warmth");
    }

    #[test]
    fn no_correction_when_beautiful() {
        let mut listener = AestheticListener::new();
        listener.smoothed.harshness = 0.1;
        listener.smoothed.beauty = 0.8;

        let mut state = MusicalState {
            dopamine: 0.5,
            ..Default::default()
        };
        let orig = state.dopamine;
        listener.apply_corrections(&mut state);
        assert!(
            (state.dopamine - orig).abs() < 0.01,
            "should not intervene when beautiful"
        );
    }

    #[test]
    fn smoothing_converges() {
        let mut listener = AestheticListener::new();
        let features = AudioFeatures {
            rms_energy: 0.1,
            harmonic_tension: 0.2,
            zero_crossing_rate: 0.05,
            ..Default::default()
        };
        let state = MusicalState::default();
        for _ in 0..50 {
            listener.assess(&features, &state);
        }
        let smoothed = listener.smoothed_assessment();
        assert!(smoothed.beauty > 0.0, "smoothed beauty should converge");
    }
}
