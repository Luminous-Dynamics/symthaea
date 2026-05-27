// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ambient drone: continuous low pad that fills silence between notes.
//!
//! Always present at low volume, tuned to the current key. Creates a bed
//! of sound that prevents dead silence and gives the composition continuity.
//! Volume scales inversely with note activity — louder when fewer notes are
//! playing, quieter when the synth is active.

use crate::MusicalState;

/// Continuous ambient drone generator with organic LFO modulation.
pub struct AmbientDrone {
    /// Phase accumulators for the drone partials.
    phases: [f32; 4],
    /// Current drone frequency (root of key).
    frequency: f32,
    /// Target frequency (smoothed toward).
    target_freq: f32,
    /// Drone volume (scales with silence).
    volume: f32,
    /// Target volume.
    target_volume: f32,
    sample_rate: f32,
    /// Asynchronous LFO phases for organic movement (incommensurable rates).
    lfo_phases: [f32; 3],
    /// LFO rates in Hz — deliberately irrational ratios to prevent repetition.
    lfo_rates: [f32; 3],
}

impl AmbientDrone {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            phases: [0.0; 4],
            frequency: 65.41,
            target_freq: 65.41,
            volume: 0.0,
            target_volume: 0.015,
            sample_rate: sample_rate as f32,
            lfo_phases: [0.0; 3],
            // Incommensurable rates (irrational ratios) — never repeat
            // Inspired by Eno's tape loop technique
            lfo_rates: [0.037, 0.071, 0.113], // ~27s, ~14s, ~9s cycles
        }
    }

    /// Update drone parameters from consciousness state.
    pub fn update_state(&mut self, state: &MusicalState, active_notes: usize) {
        // Drone frequency follows the key (first harmony = root)
        // Use a very low octave for warmth
        let base_freq = 65.41; // C2
        // Modulate by valence: positive = major feel (slight sharp), negative = minor (slight flat)
        let detune = state.valence * 2.0; // ±2 cents
        self.target_freq = base_freq * 2.0f32.powf(detune / 1200.0);

        // Volume: louder when fewer notes, quieter when synth is active
        let note_factor = 1.0 / (1.0 + active_notes as f32 * 0.3);
        // Consciousness level scales drone presence
        let psi_factor = 0.5 + state.consciousness_level * 0.5;
        // Stillness harmony boosts drone
        let stillness_boost = 1.0 + state.harmony_activations[7] * 0.5;

        // Drone volume: present as foundation but not filling all silence.
        // Very quiet during low-arousal non-stillness states (let the space breathe).
        let arousal_factor = 0.3 + state.arousal * 0.7; // quiet when calm
        self.target_volume = 0.02 * note_factor * psi_factor * stillness_boost * arousal_factor;
        self.target_volume = self.target_volume.clamp(0.003, 0.08);
    }

    /// Generate stereo drone samples with organic LFO modulation.
    pub fn render(&mut self, chunk_len: usize) -> Vec<[f32; 2]> {
        let sr = self.sample_rate;
        let mut output = Vec::with_capacity(chunk_len);

        for _ in 0..chunk_len {
            // Smooth frequency and volume transitions
            self.frequency += (self.target_freq - self.frequency) * 0.0001;
            self.volume += (self.target_volume - self.volume) * 0.0005;

            // Advance asynchronous LFOs (incommensurable rates → never repeats)
            for i in 0..3 {
                self.lfo_phases[i] += self.lfo_rates[i] / sr;
                if self.lfo_phases[i] > 1.0 {
                    self.lfo_phases[i] -= 1.0;
                }
            }

            // LFO outputs: smooth sine waves at different timescales
            let lfo0 = (self.lfo_phases[0] * std::f32::consts::TAU).sin(); // ~27s cycle: volume breathing
            let lfo1 = (self.lfo_phases[1] * std::f32::consts::TAU).sin(); // ~14s cycle: stereo drift
            let lfo2 = (self.lfo_phases[2] * std::f32::consts::TAU).sin(); // ~9s cycle: pitch micro-drift

            // Volume breathing: ±20% modulation (organic pulse)
            let vol_mod = 1.0 + lfo0 * 0.2;

            // Pitch micro-drift: ±3 cents (barely perceptible, adds life)
            let pitch_mod = 2.0f32.powf(lfo2 * 3.0 / 1200.0);

            // 4 partials with modulated frequency
            let base = self.frequency * pitch_mod;
            let freqs = [base, base * 2.0, base * 1.5, base * 4.0];
            // Per-partial amplitude modulation (each partial breathes independently)
            let amps = [
                1.0 + lfo0 * 0.1,          // fundamental: gentle pulse
                0.3 + lfo1 * 0.08,         // octave: different cycle
                0.15 + lfo2 * 0.05,        // fifth: yet another
                0.05 + lfo0 * lfo1 * 0.02, // 2oct: product of two LFOs (complex)
            ];

            let mut sample = 0.0f32;
            for i in 0..4 {
                self.phases[i] += freqs[i] / sr * std::f32::consts::TAU;
                if self.phases[i] > std::f32::consts::TAU {
                    self.phases[i] -= std::f32::consts::TAU;
                }
                sample += self.phases[i].sin() * amps[i];
            }

            let out = sample * self.volume * vol_mod;
            // Stereo drift: slow L/R panning movement
            let pan = lfo1 * 0.15; // ±15% stereo drift
            output.push([out * (0.95 - pan), out * (0.95 + pan)]);
        }

        output
    }

    pub fn reset(&mut self) {
        self.phases = [0.0; 4];
        self.lfo_phases = [0.0; 3];
        self.volume = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drone_produces_signal() {
        let mut drone = AmbientDrone::new(44100);
        drone.update_state(&MusicalState::default(), 0);
        let output = drone.render(4410);
        assert!(
            output.iter().any(|s| s[0].abs() > 0.0001),
            "drone should produce signal"
        );
    }

    #[test]
    fn drone_quieter_with_active_notes() {
        let mut drone = AmbientDrone::new(44100);
        let state = MusicalState::default();

        drone.update_state(&state, 0); // no notes
        let vol_none = drone.target_volume;

        drone.update_state(&state, 10); // many notes
        let vol_many = drone.target_volume;

        assert!(
            vol_none > vol_many,
            "drone should be quieter with notes: none={vol_none} many={vol_many}"
        );
    }

    #[test]
    fn drone_is_always_present() {
        let mut drone = AmbientDrone::new(44100);
        drone.update_state(
            &MusicalState {
                consciousness_level: 0.1,
                ..Default::default()
            },
            20,
        );
        // Even with many notes and low consciousness, drone should be > 0
        assert!(
            drone.target_volume > 0.001,
            "drone should always be present"
        );
    }

    #[test]
    fn drone_samples_are_finite() {
        let mut drone = AmbientDrone::new(44100);
        drone.update_state(&MusicalState::default(), 0);
        for pair in drone.render(44100) {
            assert!(pair[0].is_finite() && pair[1].is_finite());
        }
    }
}
