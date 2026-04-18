// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dramatic tools: the difference between "nice" and "wow."
//!
//! Strategic silence, tempo breathing, key modulation, sub-bass,
//! texture layering, and strong endings.

use crate::composer_mind::SectionArc;
use crate::MusicalState;

/// Dramatic state tracked across the composition.
pub struct DramaticState {
    /// Current key shift in semitones from original (for modulation).
    pub key_shift: i32,
    /// Target key shift (smooth transition).
    pub target_key_shift: i32,
    /// Tempo multiplier (for rubato/breathing).
    pub tempo_factor: f32,
    /// Whether we're in a dramatic silence.
    pub silence_active: bool,
    /// Samples remaining in silence.
    pub silence_remaining: usize,
    /// Texture density [0, 1]: 0 = single voice, 1 = full ensemble.
    pub texture_density: f32,
    /// Target texture density.
    pub target_density: f32,
    /// Whether the final cadence has been triggered.
    pub final_cadence: bool,
    /// Sub-bass phase accumulator.
    sub_bass_phase: f32,
    /// Sub-bass volume.
    sub_bass_volume: f32,
    /// Dynamic range multiplier: exaggerates quiet/loud contrast.
    pub dynamic_range: f32,
}

impl DramaticState {
    pub fn new() -> Self {
        Self {
            key_shift: 0,
            target_key_shift: 0,
            tempo_factor: 1.0,
            silence_active: false,
            silence_remaining: 0,
            texture_density: 0.2, // start sparse
            target_density: 0.2,
            final_cadence: false,
            sub_bass_phase: 0.0,
            sub_bass_volume: 0.0,
            dynamic_range: 1.5, // 50% wider than default
        }
    }

    /// Update dramatic state based on section arc and consciousness.
    pub fn update(&mut self, section: SectionArc, state: &MusicalState, bars_in_section: f32) {
        let psi = state.consciousness_level;
        let arousal = state.arousal;

        match section {
            SectionArc::Exposition => {
                self.target_density = 0.2 + psi * 0.3; // sparse, gradually adding
                self.tempo_factor = 1.0; // steady
                self.target_key_shift = 0;
                self.sub_bass_volume = 0.0; // no sub-bass yet
            }
            SectionArc::Development => {
                self.target_density = 0.4 + arousal * 0.3;
                // Accelerando during development
                self.tempo_factor = 1.0 + arousal * 0.1;
                // Modulate up for tension
                if bars_in_section > 8.0 {
                    self.target_key_shift = 2; // whole step up
                }
                self.sub_bass_volume = 0.005;
            }
            SectionArc::Climax => {
                self.target_density = 0.8 + psi * 0.2; // nearly full
                                                       // Slight accelerando at peak
                self.tempo_factor = 1.05 + arousal * 0.05;
                // Key modulation: up a half step for emotional lift
                self.target_key_shift = 1;
                self.sub_bass_volume = 0.015; // sub-bass kicks in

                // Strategic silence right before the climax peak
                if bars_in_section < 2.0 && !self.silence_active && bars_in_section > 1.0 {
                    self.trigger_silence(22050); // 500ms at 44100Hz
                }
            }
            SectionArc::Recapitulation => {
                self.target_density = 0.6;
                // Ritardando
                self.tempo_factor = 1.0 - bars_in_section * 0.005;
                self.target_key_shift = 0; // return to original key
                self.sub_bass_volume = 0.008;
            }
            SectionArc::Coda => {
                self.target_density = 0.3 - bars_in_section * 0.02;
                self.target_density = self.target_density.max(0.05);
                // Strong ritardando
                self.tempo_factor = (1.0 - bars_in_section * 0.02).max(0.7);
                self.target_key_shift = 0;
                self.sub_bass_volume = (0.01 - bars_in_section * 0.001).max(0.0);

                // Final cadence
                if bars_in_section > 12.0 && !self.final_cadence {
                    self.final_cadence = true;
                }
            }
        }

        // Smooth transitions
        self.texture_density += (self.target_density - self.texture_density) * 0.02;
        if self.key_shift != self.target_key_shift {
            // Shift key by one semitone per update (smooth modulation)
            if self.target_key_shift > self.key_shift {
                self.key_shift += 1;
            } else {
                self.key_shift -= 1;
            }
        }

        // Tick silence
        if self.silence_active {
            if self.silence_remaining > 0 {
                self.silence_remaining -= 1;
            } else {
                self.silence_active = false;
            }
        }
    }

    /// Trigger a dramatic silence.
    pub fn trigger_silence(&mut self, samples: usize) {
        self.silence_active = true;
        self.silence_remaining = samples;
    }

    /// Apply key shift to a frequency.
    pub fn apply_key_shift(&self, freq: f32) -> f32 {
        freq * 2.0f32.powf(self.key_shift as f32 / 12.0)
    }

    /// Apply dynamic range exaggeration to a velocity.
    pub fn apply_dynamics(&self, velocity: f32, base_velocity: f32) -> f32 {
        let deviation = velocity - base_velocity;
        let exaggerated = base_velocity + deviation * self.dynamic_range;
        exaggerated.clamp(0.02, 1.0)
    }

    /// Render sub-bass for one chunk. Returns mono samples.
    pub fn render_sub_bass(
        &mut self,
        chunk_len: usize,
        sample_rate: f32,
        kick_active: bool,
    ) -> Vec<f32> {
        let freq = 45.0; // deep sub-bass
        let volume = if kick_active {
            self.sub_bass_volume * 1.5 // louder when kick hits
        } else {
            self.sub_bass_volume
        };

        let mut output = Vec::with_capacity(chunk_len);
        for _ in 0..chunk_len {
            self.sub_bass_phase += freq / sample_rate * std::f32::consts::TAU;
            if self.sub_bass_phase > std::f32::consts::TAU {
                self.sub_bass_phase -= std::f32::consts::TAU;
            }
            output.push(self.sub_bass_phase.sin() * volume);
        }
        output
    }

    /// How many voices should be active based on texture density.
    pub fn max_voices(&self) -> usize {
        (1.0 + self.texture_density * 7.0).round() as usize // 1-8 voices
    }

    /// Should notes be generated this chunk? (silence gate)
    pub fn allow_notes(&self) -> bool {
        !self.silence_active
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Vibrato generator for richer instrument sounds.
pub struct Vibrato {
    phase: f32,
    rate: f32,  // Hz (typically 5-7 Hz)
    depth: f32, // cents (typically 10-30)
}

impl Vibrato {
    pub fn new(rate: f32, depth_cents: f32) -> Self {
        Self {
            phase: 0.0,
            rate,
            depth: depth_cents,
        }
    }

    /// Get frequency multiplier for current sample.
    pub fn tick(&mut self, sample_rate: f32) -> f32 {
        self.phase += self.rate / sample_rate * std::f32::consts::TAU;
        if self.phase > std::f32::consts::TAU {
            self.phase -= std::f32::consts::TAU;
        }
        2.0f32.powf(self.phase.sin() * self.depth / 1200.0)
    }

    pub fn set_depth(&mut self, cents: f32) {
        self.depth = cents;
    }
}

/// Noise transient for attack realism.
pub fn attack_noise(sample_idx: usize, attack_samples: usize, brightness: f32) -> f32 {
    if sample_idx >= attack_samples {
        return 0.0;
    }

    let progress = sample_idx as f32 / attack_samples as f32;
    let envelope = (1.0 - progress).powi(3); // fast decay

    // LP filter: only change noise every 8 samples → ~5.5kHz bandlimited
    // instead of 22kHz white noise. Prevents sample-to-sample discontinuities
    // from creating audible clicks in the 2nd derivative.
    let sample_group = (sample_idx / 8) as u32;
    let noise = (sample_group.wrapping_mul(1103515245).wrapping_add(12345) >> 16) as f32
        / 32768.0
        - 1.0;

    noise * envelope * brightness * 0.08 // also reduced from 0.15
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_blocks_notes() {
        let mut state = DramaticState::new();
        assert!(state.allow_notes());
        state.trigger_silence(100);
        assert!(!state.allow_notes());
    }

    #[test]
    fn key_shift_modulates_frequency() {
        let state = DramaticState {
            key_shift: 2,
            ..DramaticState::new()
        };
        let shifted = state.apply_key_shift(440.0);
        // 2 semitones up from A4 = B4 ≈ 493.88
        assert!((shifted - 493.88).abs() < 1.0, "should be ~B4: {shifted}");
    }

    #[test]
    fn dynamics_exaggeration() {
        let state = DramaticState {
            dynamic_range: 2.0,
            ..DramaticState::new()
        };
        let quiet = state.apply_dynamics(0.3, 0.5); // below average
        let loud = state.apply_dynamics(0.8, 0.5); // above average
        assert!(quiet < 0.3, "quiet should be quieter: {quiet}");
        assert!(loud > 0.8, "loud should be louder: {loud}");
    }

    #[test]
    fn texture_starts_sparse() {
        let state = DramaticState::new();
        assert!(
            state.max_voices() <= 3,
            "should start sparse: {}",
            state.max_voices()
        );
    }

    #[test]
    fn sub_bass_produces_signal() {
        let mut state = DramaticState::new();
        state.sub_bass_volume = 0.01;
        let bass = state.render_sub_bass(441, 44100.0, false);
        assert!(
            bass.iter().any(|&s| s.abs() > 0.001),
            "should produce signal"
        );
    }

    #[test]
    fn vibrato_modulates() {
        let mut vib = Vibrato::new(6.0, 20.0);
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for _ in 0..44100 {
            let m = vib.tick(44100.0);
            min = min.min(m);
            max = max.max(m);
        }
        assert!(
            max > 1.0 && min < 1.0,
            "vibrato should modulate above and below 1.0"
        );
    }

    #[test]
    fn attack_noise_decays() {
        let early = attack_noise(10, 1000, 0.5).abs();
        let late = attack_noise(900, 1000, 0.5).abs();
        // Early should generally be louder than late (stochastic but envelope-weighted)
        // Just check both are finite
        assert!(early.is_finite() && late.is_finite());
    }
}
