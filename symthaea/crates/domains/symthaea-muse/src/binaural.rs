// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Binaural consciousness rendering: map consciousness topology to 3D auditory space.
//!
//! High Phi = sounds converge to center (integrated experience).
//! Low Phi = sounds scatter to periphery (fragmented experience).
//! Eight Harmonies map to fixed spatial positions around the listener.
//!
//! Uses ITD (Interaural Time Difference) and ILD (Interaural Level Difference)
//! for HRTF-based spatialization without external dependencies.

use crate::MusicalState;

/// Head radius in meters (average human, Woodworth model).
const HEAD_RADIUS: f32 = 0.0875;
/// Speed of sound in m/s.
const SPEED_OF_SOUND: f32 = 343.0;
/// Maximum ITD in samples at 44.1kHz (~0.63ms).
const MAX_ITD_SAMPLES_44K: usize = 28;

/// Spatial position in spherical coordinates.
#[derive(Debug, Clone, Copy)]
pub struct SpatialPosition {
    /// Azimuth angle in radians [-π, π]. 0 = front, π/2 = right, -π/2 = left.
    pub azimuth: f32,
    /// Elevation angle in radians [-π/2, π/2]. 0 = level, π/2 = above.
    pub elevation: f32,
    /// Distance from listener in meters.
    pub distance: f32,
}

impl SpatialPosition {
    pub fn front(distance: f32) -> Self {
        Self {
            azimuth: 0.0,
            elevation: 0.0,
            distance,
        }
    }
}

/// Eight Harmony spatial positions (fixed reference points in auditory space).
const HARMONY_POSITIONS: [SpatialPosition; 8] = [
    // ResonantCoherence: center, close
    SpatialPosition {
        azimuth: 0.0,
        elevation: 0.0,
        distance: 1.0,
    },
    // PanSentientFlourishing: right-front, warm
    SpatialPosition {
        azimuth: 0.5,
        elevation: 0.1,
        distance: 1.5,
    },
    // IntegralWisdom: left-front, grounded
    SpatialPosition {
        azimuth: -0.4,
        elevation: -0.1,
        distance: 1.5,
    },
    // InfinitePlay: left-rear, playful
    SpatialPosition {
        azimuth: -2.3,
        elevation: -0.15,
        distance: 2.5,
    },
    // UniversalInterconnectedness: wide right
    SpatialPosition {
        azimuth: 1.2,
        elevation: 0.0,
        distance: 2.0,
    },
    // SacredReciprocity: behind, reciprocal
    SpatialPosition {
        azimuth: 3.0,
        elevation: 0.0,
        distance: 2.0,
    },
    // EvolutionaryProgression: ascending front-right
    SpatialPosition {
        azimuth: 0.3,
        elevation: 0.4,
        distance: 1.8,
    },
    // SacredStillness: directly above
    SpatialPosition {
        azimuth: 0.0,
        elevation: 1.2,
        distance: 1.5,
    },
];

/// A binaural source with delay-line ITD and gain-based ILD.
struct BinauralSource {
    delay_l: DelayLine,
    delay_r: DelayLine,
    gain_l: f32,
    gain_r: f32,
    itd_l: usize,
    itd_r: usize,
    // Smoothing state: targets vs current for glitch-free updates
    target_gain_l: f32,
    target_gain_r: f32,
    target_itd_l: usize,
    target_itd_r: usize,
}

struct DelayLine {
    buffer: Vec<f32>,
    write_pos: usize,
}

impl DelayLine {
    fn new(max_samples: usize) -> Self {
        Self {
            buffer: vec![0.0; max_samples.max(1)],
            write_pos: 0,
        }
    }

    fn write_and_read(&mut self, input: f32, delay: usize) -> f32 {
        self.buffer[self.write_pos] = input;
        let read = (self.write_pos + self.buffer.len() - delay.min(self.buffer.len() - 1))
            % self.buffer.len();
        let output = self.buffer[read];
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        output
    }
}

impl BinauralSource {
    fn new(sample_rate: u32) -> Self {
        let max_delay = (MAX_ITD_SAMPLES_44K as f32 * sample_rate as f32 / 44100.0) as usize + 2;
        Self {
            delay_l: DelayLine::new(max_delay),
            delay_r: DelayLine::new(max_delay),
            gain_l: 0.707,
            gain_r: 0.707,
            itd_l: 0,
            itd_r: 0,
            target_gain_l: 0.707,
            target_gain_r: 0.707,
            target_itd_l: 0,
            target_itd_r: 0,
        }
    }

    /// Update ITD and ILD from a spatial position (Woodworth model).
    fn set_position(&mut self, pos: &SpatialPosition, sample_rate: u32) {
        let az = pos.azimuth;
        let sr = sample_rate as f32;

        // ITD via Woodworth formula: Δt = (r/c)(θ + sin(θ)) for |θ| ≤ π/2
        let theta = az.clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        let itd_seconds = HEAD_RADIUS / SPEED_OF_SOUND * (theta.abs() + theta.abs().sin());
        let itd_samples = (itd_seconds * sr) as usize;

        // FIX: Write TARGETS, not current values. process() smooths to them
        // over samples, preventing the phase-jump/amplitude-pop clicks caused
        // by instantaneous ITD/gain changes.
        if az >= 0.0 {
            self.target_itd_l = itd_samples;
            self.target_itd_r = 0;
        } else {
            self.target_itd_l = 0;
            self.target_itd_r = itd_samples;
        }

        let shadow_db = 6.0 * az.abs().sin();
        let shadow_gain = 10.0f32.powf(-shadow_db / 20.0);
        let (mut g_l, mut g_r) = if az >= 0.0 {
            (shadow_gain, 1.0)
        } else {
            (1.0, shadow_gain)
        };

        let dist_gain = 1.0 / pos.distance.max(0.1);
        g_l *= dist_gain;
        g_r *= dist_gain;

        let elev_boost = 1.0 + pos.elevation.abs() * 0.15;
        g_l *= elev_boost;
        g_r *= elev_boost;

        self.target_gain_l = g_l;
        self.target_gain_r = g_r;
    }

    /// Process a mono sample through ITD delays and ILD gains.
    ///
    /// Smooths gain changes per-sample to prevent click artifacts when
    /// consciousness state shifts position. ITD changes one sample at a
    /// time — a gradual slew that sounds like natural head motion.
    fn process(&mut self, input: f32) -> (f32, f32) {
        // Smooth gains: 1-pole filter, ~10ms time constant at 44.1kHz
        let alpha = 0.003_f32;
        self.gain_l += alpha * (self.target_gain_l - self.gain_l);
        self.gain_r += alpha * (self.target_gain_r - self.gain_r);

        // Slew ITD by at most 1 sample per call — prevents phase jumps
        if self.itd_l < self.target_itd_l {
            self.itd_l += 1;
        } else if self.itd_l > self.target_itd_l {
            self.itd_l -= 1;
        }
        if self.itd_r < self.target_itd_r {
            self.itd_r += 1;
        } else if self.itd_r > self.target_itd_r {
            self.itd_r -= 1;
        }

        let l = self.delay_l.write_and_read(input, self.itd_l) * self.gain_l;
        let r = self.delay_r.write_and_read(input, self.itd_r) * self.gain_r;
        (l, r)
    }
}

/// Binaural consciousness renderer.
///
/// Maps each voice to a 3D position derived from consciousness state,
/// then applies ITD/ILD spatialization.
pub struct BinauralConsciousnessRenderer {
    sources: Vec<BinauralSource>,
    sample_rate: u32,
    phi_spread: f32,
    positions: Vec<SpatialPosition>,
}

impl BinauralConsciousnessRenderer {
    /// Create renderer for a given number of voices.
    pub fn new(num_voices: usize, sample_rate: u32) -> Self {
        let sources = (0..num_voices)
            .map(|_| BinauralSource::new(sample_rate))
            .collect();
        let positions = vec![SpatialPosition::front(1.5); num_voices];
        Self {
            sources,
            sample_rate,
            phi_spread: 0.5,
            positions,
        }
    }

    /// Update spatial positions from consciousness state.
    ///
    /// Phi controls spatial spread: high Phi → voices converge to center.
    /// Each voice's position is a weighted blend of harmony positions.
    pub fn update_state(&mut self, state: &MusicalState) {
        let phi = state.consciousness_level;
        // Spread: high consciousness = narrow (0.1), low = wide (1.0)
        self.phi_spread = 1.0 - phi * 0.9;

        for (i, source) in self.sources.iter_mut().enumerate() {
            // Each voice maps to a blend of harmony positions
            let base_pos = if i < HARMONY_POSITIONS.len() {
                HARMONY_POSITIONS[i]
            } else {
                HARMONY_POSITIONS[i % HARMONY_POSITIONS.len()]
            };

            // Apply Phi-based contraction toward center
            let pos = SpatialPosition {
                azimuth: base_pos.azimuth * self.phi_spread,
                elevation: base_pos.elevation * self.phi_spread,
                distance: base_pos.distance * (0.5 + self.phi_spread * 0.5),
            };

            // Modulate by harmony activation for that position
            let harmony_idx = i.min(7);
            let activation = state.harmony_activations[harmony_idx];
            let final_pos = SpatialPosition {
                azimuth: pos.azimuth * (0.5 + activation * 0.5),
                elevation: pos.elevation * (0.5 + activation * 0.5),
                distance: pos.distance / (0.5 + activation * 0.5),
            };

            if i < self.positions.len() {
                self.positions[i] = final_pos;
            }
            source.set_position(&final_pos, self.sample_rate);
        }
    }

    /// Spatialize mono voice buffers into binaural stereo.
    ///
    /// `mono_voices`: one buffer per voice (same length).
    /// Returns interleaved stereo output.
    pub fn render(&mut self, mono_voices: &[Vec<f32>]) -> Vec<[f32; 2]> {
        if mono_voices.is_empty() {
            return Vec::new();
        }

        let chunk_len = mono_voices[0].len();
        let mut output = vec![[0.0f32; 2]; chunk_len];

        for (voice_idx, voice_buf) in mono_voices.iter().enumerate() {
            if voice_idx >= self.sources.len() {
                break;
            }
            for (i, &sample) in voice_buf.iter().enumerate() {
                let (l, r) = self.sources[voice_idx].process(sample);
                output[i][0] += l;
                output[i][1] += r;
            }
        }

        output
    }

    /// Get current spatial positions (for visualization).
    pub fn positions(&self) -> &[SpatialPosition] {
        &self.positions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_creates_stereo_output() {
        let mut renderer = BinauralConsciousnessRenderer::new(2, 44100);
        renderer.update_state(&MusicalState::default());

        let voices = vec![vec![0.5f32; 1024], vec![0.3f32; 1024]];
        let output = renderer.render(&voices);
        assert_eq!(output.len(), 1024);
        assert!(output.iter().any(|s| s[0].abs() > 0.01));
        assert!(output.iter().any(|s| s[1].abs() > 0.01));
    }

    #[test]
    fn high_phi_narrows_spread() {
        let mut renderer = BinauralConsciousnessRenderer::new(4, 44100);

        let high_phi = MusicalState {
            consciousness_level: 0.95,
            ..Default::default()
        };
        renderer.update_state(&high_phi);
        let high_spread = renderer.phi_spread;

        let low_phi = MusicalState {
            consciousness_level: 0.1,
            ..Default::default()
        };
        renderer.update_state(&low_phi);
        let low_spread = renderer.phi_spread;

        assert!(
            high_spread < low_spread,
            "high Phi should narrow: {high_spread} vs {low_spread}"
        );
        assert!(high_spread < 0.2, "high Phi spread should be narrow");
    }

    #[test]
    fn itd_creates_stereo_difference() {
        let mut renderer = BinauralConsciousnessRenderer::new(1, 44100);
        // Place source hard right
        let state = MusicalState {
            consciousness_level: 0.1, // wide spread
            harmony_activations: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        renderer.update_state(&state);

        // Impulse
        let mut voice = vec![0.0f32; 256];
        voice[0] = 1.0;
        let output = renderer.render(&[voice]);

        // Right ear should get signal before left (ITD)
        let first_r = output.iter().position(|s| s[1].abs() > 0.01).unwrap_or(256);
        let first_l = output.iter().position(|s| s[0].abs() > 0.01).unwrap_or(256);
        // With source on right, right ear is closer
        assert!(
            first_r <= first_l,
            "right should hear first: R={first_r}, L={first_l}"
        );
    }

    #[test]
    fn empty_input_empty_output() {
        let mut renderer = BinauralConsciousnessRenderer::new(2, 44100);
        let output = renderer.render(&[]);
        assert!(output.is_empty());
    }

    #[test]
    fn harmony_positions_defined() {
        assert_eq!(HARMONY_POSITIONS.len(), 8);
        // Sacred Stillness should be elevated
        assert!(HARMONY_POSITIONS[7].elevation > 0.5);
        // ResonantCoherence should be centered
        assert!(HARMONY_POSITIONS[0].azimuth.abs() < 0.01);
    }
}
