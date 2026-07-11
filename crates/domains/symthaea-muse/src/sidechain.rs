// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sidechain compression / ducking for polyphonic voice mixing.
//!
//! Dynamic voice coupling: bass ducks when lead peaks, harmony recedes
//! for lead sustains. Uses an envelope follower with configurable attack/release
//! and gain reduction ratio.

use crate::voice::VoiceRole;

/// Sidechain ducking processor for a single voice pair.
pub struct SidechainDucker {
    envelope: f32,
    attack_coeff: f32,
    release_coeff: f32,
    threshold: f32,
    ratio: f32,
}

impl SidechainDucker {
    /// Create a ducker.
    /// - `attack_ms`: envelope attack time (~1ms for percussive ducking)
    /// - `release_ms`: envelope release time (~50-100ms for smooth recovery)
    /// - `threshold`: linear amplitude above which ducking engages (0.1-0.3)
    /// - `ratio`: gain reduction ratio (4.0 = 4:1 ducking)
    pub fn new(
        sample_rate: u32,
        attack_ms: f32,
        release_ms: f32,
        threshold: f32,
        ratio: f32,
    ) -> Self {
        let sr = sample_rate as f32;
        Self {
            envelope: 0.0,
            attack_coeff: (-1.0 / (attack_ms * 0.001 * sr)).exp(),
            release_coeff: (-1.0 / (release_ms * 0.001 * sr)).exp(),
            threshold,
            ratio,
        }
    }

    /// Update envelope from sidechain source sample and compute gain reduction.
    /// Returns gain multiplier [0, 1] to apply to the ducked voice.
    pub fn process(&mut self, sidechain_sample: f32) -> f32 {
        let level = sidechain_sample.abs();
        let coeff = if level > self.envelope {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope = crate::synth::flush_denormal(coeff * self.envelope + (1.0 - coeff) * level);

        if self.envelope > self.threshold {
            let excess_db = 20.0 * (self.envelope / self.threshold).log10();
            let reduced_db = excess_db / self.ratio;
            let gain_reduction_db = excess_db - reduced_db;
            10.0f32.powf(-gain_reduction_db / 20.0)
        } else {
            1.0
        }
    }

    /// Reset envelope state.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
    }
}

/// Matrix of ducking relationships between voices.
pub struct DuckingMatrix {
    entries: Vec<DuckingEntry>,
}

struct DuckingEntry {
    source: VoiceRole,
    target: VoiceRole,
    ducker: SidechainDucker,
}

impl DuckingMatrix {
    /// Create the default ducking matrix.
    /// Lead ducks Bass (4:1, threshold 0.15), Lead ducks Harmony (2:1, threshold 0.2).
    pub fn default_matrix(sample_rate: u32) -> Self {
        Self {
            entries: vec![
                DuckingEntry {
                    source: VoiceRole::Lead,
                    target: VoiceRole::Bass,
                    ducker: SidechainDucker::new(sample_rate, 3.0, 150.0, 0.15, 2.0), // gentler: 2:1, slower release
                },
                DuckingEntry {
                    source: VoiceRole::Lead,
                    target: VoiceRole::Harmony,
                    ducker: SidechainDucker::new(sample_rate, 5.0, 200.0, 0.2, 1.5), // very gentle: 1.5:1
                },
            ],
        }
    }

    /// Apply ducking: given per-voice mono buffers, modify target buffers in-place.
    ///
    /// `voice_buffers` is indexed by voice index in the arrangement.
    /// `voice_roles` maps index → VoiceRole.
    pub fn apply(
        &mut self,
        voice_buffers: &mut [Vec<f32>],
        voice_roles: &[VoiceRole],
        chunk_len: usize,
    ) {
        for entry in &mut self.entries {
            let source_idx = voice_roles.iter().position(|r| *r == entry.source);
            let target_idx = voice_roles.iter().position(|r| *r == entry.target);

            if let (Some(si), Some(ti)) = (source_idx, target_idx) {
                if si == ti || si >= voice_buffers.len() || ti >= voice_buffers.len() {
                    continue;
                }
                for i in 0..chunk_len
                    .min(voice_buffers[si].len())
                    .min(voice_buffers[ti].len())
                {
                    let gain = entry.ducker.process(voice_buffers[si][i]);
                    voice_buffers[ti][i] *= gain;
                }
            }
        }
    }

    /// Reset all duckers.
    pub fn reset(&mut self) {
        for entry in &mut self.entries {
            entry.ducker.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ducker_reduces_gain_above_threshold() {
        let mut d = SidechainDucker::new(44100, 0.1, 10.0, 0.1, 4.0);
        // Feed loud sidechain signal
        for _ in 0..100 {
            d.process(0.5);
        }
        let gain = d.process(0.5);
        assert!(gain < 1.0, "should reduce gain: {gain}");
        assert!(gain > 0.0, "should not silence: {gain}");
    }

    #[test]
    fn ducker_unity_below_threshold() {
        let mut d = SidechainDucker::new(44100, 1.0, 50.0, 0.3, 4.0);
        let gain = d.process(0.05);
        assert!(
            (gain - 1.0).abs() < 0.01,
            "below threshold should be unity: {gain}"
        );
    }

    #[test]
    fn ducking_matrix_applies() {
        let mut matrix = DuckingMatrix::default_matrix(44100);
        let roles = vec![VoiceRole::Lead, VoiceRole::Bass];
        let mut bufs = vec![
            vec![0.5f32; 100], // lead: loud
            vec![0.3f32; 100], // bass: moderate
        ];
        matrix.apply(&mut bufs, &roles, 100);
        // Bass should be reduced
        let bass_avg: f32 = bufs[1].iter().sum::<f32>() / 100.0;
        assert!(bass_avg < 0.3, "bass should be ducked: avg={bass_avg}");
    }
}
