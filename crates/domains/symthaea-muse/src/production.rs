// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Production quality: the fixes that separate amateur from professional sound.
//!
//! Based on analysis of output + music production research:
//! 1. Stochastic residual (filtered noise on every voice — #1 fix for thin sound)
//! 2. Velocity→timbre mapping (louder = brighter, not just louder)
//! 3. Phrase dynamic envelopes (2-4dB crescendo/diminuendo per phrase)
//! 4. Pitch range constraint (melodies within ~14 semitones, not 119)
//! 5. Mud cut (remove 200-500Hz buildup)
//! 6. Production EQ per genre

/// Stochastic residual: filtered noise added to pitched synthesis.
///
/// Real instruments produce both periodic (pitch) and stochastic (noise)
/// components. Missing the noise is the #1 reason additive sounds thin.
/// (IRCAM SMS framework, Serra & Smith 1990)
pub struct StochasticResidual {
    noise_state: u32,
    /// Low-pass filter state for shaping noise spectrum.
    lp_state: f32,
    /// Band-pass filter state for body resonance.
    bp_state: [f32; 2],
}

impl StochasticResidual {
    pub fn new(seed: u32) -> Self {
        Self {
            noise_state: seed,
            lp_state: 0.0,
            bp_state: [0.0; 2],
        }
    }

    /// Generate one sample of filtered noise appropriate for the instrument.
    ///
    /// `brightness`: 0-1, controls noise spectrum (higher = more HF)
    /// `body_freq`: resonant body frequency in Hz (violin ~700, guitar ~200)
    /// `amount`: mix level (0.02-0.10 typical)
    /// `sample_rate`: audio sample rate
    pub fn tick(&mut self, brightness: f32, body_freq: f32, amount: f32, sample_rate: f32) -> f32 {
        // Generate white noise
        self.noise_state = self
            .noise_state
            .wrapping_mul(1103515245)
            .wrapping_add(12345);
        let noise = (self.noise_state >> 16) as f32 / 32768.0 - 1.0;

        // Low-pass filter (brightness control)
        let lp_coeff =
            (std::f32::consts::TAU * (500.0 + brightness * 4000.0) / sample_rate).min(0.99);
        self.lp_state += lp_coeff * (noise - self.lp_state);

        // Body resonance (bandpass at body_freq)
        let bp_coeff = (std::f32::consts::TAU * body_freq / sample_rate).min(0.99);
        let q = 2.0; // moderate resonance
        self.bp_state[0] += bp_coeff * (self.lp_state - self.bp_state[0] - self.bp_state[1] / q);
        self.bp_state[1] += bp_coeff * self.bp_state[0];

        (self.lp_state * 0.3 + self.bp_state[1] * 0.7) * amount
    }
}

/// Velocity→timbre mapping: louder notes have more high-frequency energy.
///
/// In real instruments, hitting harder doesn't just increase volume —
/// it tilts the spectral slope, adding more upper harmonics.
/// (Fletcher & Rossing 1998: piano hammer hardness increases with velocity)
pub fn velocity_to_spectral_tilt(velocity: f32, harmonic_index: usize) -> f32 {
    // Higher harmonics get more boost at higher velocities
    let harmonic_boost = if harmonic_index > 2 {
        velocity * 0.3 * (harmonic_index as f32 / 8.0).min(1.0)
    } else {
        0.0
    };
    1.0 + harmonic_boost
}

/// Constrain pitch to a musically useful range.
///
/// Analysis showed notes scattered across 119 semitones (full piano range).
/// Memorable melodies stay within ~10-14 semitones (Jakubowski 2017).
pub fn constrain_pitch_range(freq: f32, center_freq: f32, max_range_semitones: f32) -> f32 {
    let half_range = max_range_semitones / 2.0;
    let min_freq = center_freq * 2.0f32.powf(-half_range / 12.0);
    let max_freq = center_freq * 2.0f32.powf(half_range / 12.0);

    if freq >= min_freq && freq <= max_freq {
        freq
    } else {
        // Hard clamp to boundary then octave-fold inward
        let mut f = freq.clamp(min_freq * 0.5, max_freq * 2.0);
        while f > max_freq {
            f /= 2.0;
        }
        while f < min_freq {
            f *= 2.0;
        }
        f
    }
}

/// Body resonance frequencies for instruments (Hz).
/// These are the formant peaks that give each instrument its characteristic "body."
pub fn body_resonance(instrument: &str) -> f32 {
    match instrument {
        "piano" => 250.0,
        "violin" => 700.0,
        "cello" => 250.0,
        "guitar" => 200.0,
        "flute" => 1500.0,
        "bell" => 3000.0,
        _ => 400.0,
    }
}

/// Noise amount by instrument (how much stochastic residual).
pub fn noise_amount(instrument: &str) -> f32 {
    match instrument {
        "piano" => 0.03,  // hammer noise
        "violin" => 0.06, // bow scrape
        "cello" => 0.05,
        "guitar" => 0.04, // pick/finger noise
        "flute" => 0.08,  // breath noise (significant)
        "bell" => 0.01,   // minimal
        _ => 0.03,
    }
}

/// Production EQ: per-genre frequency corrections.
#[derive(Debug, Clone)]
pub struct ProductionEQ {
    /// High-pass frequency (remove rumble below this).
    pub highpass_hz: f32,
    /// Mud cut: dB reduction at 200-500Hz.
    pub mud_cut_db: f32,
    /// Presence boost: dB at 1.5-4kHz.
    pub presence_boost_db: f32,
    /// Air: dB shelf boost at 10kHz+.
    pub air_boost_db: f32,
}

impl ProductionEQ {
    pub fn for_genre(genre: &str) -> Self {
        match genre {
            "prog_metal" => Self {
                highpass_hz: 80.0,
                mud_cut_db: -3.0,
                presence_boost_db: 3.0, // pick attack, articulation
                air_boost_db: 1.0,
            },
            "jazz" => Self {
                highpass_hz: 60.0,
                mud_cut_db: -1.5, // less aggressive
                presence_boost_db: 1.0,
                air_boost_db: 2.0, // warmth + air
            },
            "ambient" => Self {
                highpass_hz: 40.0,
                mud_cut_db: -1.0,
                presence_boost_db: 0.0, // no harsh presence
                air_boost_db: 3.0,      // lots of air
            },
            "funk" => Self {
                highpass_hz: 60.0,
                mud_cut_db: -2.5,
                presence_boost_db: 2.0,
                air_boost_db: 1.5,
            },
            _ => Self {
                highpass_hz: 80.0,
                mud_cut_db: -2.0,
                presence_boost_db: 1.5,
                air_boost_db: 1.5,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual_produces_signal() {
        let mut res = StochasticResidual::new(42);
        let mut has_signal = false;
        for _ in 0..1000 {
            let s = res.tick(0.5, 400.0, 0.05, 44100.0);
            if s.abs() > 0.001 {
                has_signal = true;
            }
        }
        assert!(has_signal, "residual should produce signal");
    }

    #[test]
    fn velocity_tilt_increases_with_velocity() {
        let soft = velocity_to_spectral_tilt(0.3, 5);
        let loud = velocity_to_spectral_tilt(0.9, 5);
        assert!(loud > soft, "louder should be brighter");
    }

    #[test]
    fn pitch_constraint_folds() {
        let center = 440.0; // A4
        let too_high = 1760.0; // A6, 24 semitones up
        let constrained = constrain_pitch_range(too_high, center, 14.0);
        let distance = (constrained / center).log2() * 12.0;
        assert!(
            distance.abs() <= 7.5,
            "should be within range: {distance} semitones"
        );
    }

    #[test]
    fn pitch_constraint_preserves_in_range() {
        let center = 440.0;
        let in_range = 493.88; // B4, 2 semitones up
        let result = constrain_pitch_range(in_range, center, 14.0);
        assert!(
            (result - in_range).abs() < 1.0,
            "should not change in-range pitch"
        );
    }
}
