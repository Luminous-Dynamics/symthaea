// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ambient drone: continuous low pad that fills silence between notes.
//!
//! Always present at low volume, tuned to the current key. Creates a bed
//! of sound that prevents dead silence and gives the composition continuity.
//! Volume scales inversely with note activity — louder when fewer notes are
//! playing, quieter when the synth is active.

use crate::MusicalState;

/// Continuous ambient drone generator.
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
}

impl AmbientDrone {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            phases: [0.0; 4],
            frequency: 65.41, // C2 — low, warm root
            target_freq: 65.41,
            volume: 0.0,
            target_volume: 0.015,
            sample_rate: sample_rate as f32,
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

        self.target_volume = 0.01 * note_factor * psi_factor * stillness_boost;
        self.target_volume = self.target_volume.clamp(0.003, 0.03); // always barely present
    }

    /// Generate stereo drone samples for one chunk.
    pub fn render(&mut self, chunk_len: usize) -> Vec<[f32; 2]> {
        let sr = self.sample_rate;
        let mut output = Vec::with_capacity(chunk_len);

        for _ in 0..chunk_len {
            // Smooth frequency and volume transitions
            self.frequency += (self.target_freq - self.frequency) * 0.0001;
            self.volume += (self.target_volume - self.volume) * 0.0005;

            // 4 partials: fundamental + octave + fifth + 2 octaves
            let freqs = [self.frequency, self.frequency * 2.0, self.frequency * 1.5, self.frequency * 4.0];
            let amps = [1.0f32, 0.3, 0.15, 0.05]; // fundamental dominant

            let mut sample = 0.0f32;
            for i in 0..4 {
                self.phases[i] += freqs[i] / sr * std::f32::consts::TAU;
                if self.phases[i] > std::f32::consts::TAU {
                    self.phases[i] -= std::f32::consts::TAU;
                }
                sample += self.phases[i].sin() * amps[i];
            }

            let out = sample * self.volume;
            // Slight stereo width via phase offset
            output.push([out * 0.95, out * 1.05]);
        }

        output
    }

    pub fn reset(&mut self) {
        self.phases = [0.0; 4];
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
        assert!(output.iter().any(|s| s[0].abs() > 0.0001), "drone should produce signal");
    }

    #[test]
    fn drone_quieter_with_active_notes() {
        let mut drone = AmbientDrone::new(44100);
        let state = MusicalState::default();

        drone.update_state(&state, 0); // no notes
        let vol_none = drone.target_volume;

        drone.update_state(&state, 10); // many notes
        let vol_many = drone.target_volume;

        assert!(vol_none > vol_many, "drone should be quieter with notes: none={vol_none} many={vol_many}");
    }

    #[test]
    fn drone_is_always_present() {
        let mut drone = AmbientDrone::new(44100);
        drone.update_state(&MusicalState { consciousness_level: 0.1, ..Default::default() }, 20);
        // Even with many notes and low consciousness, drone should be > 0
        assert!(drone.target_volume > 0.001, "drone should always be present");
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
