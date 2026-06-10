// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Performance expression: humanize notes with Phi-dependent timing,
//! FEP-driven dynamics, rubato, and legato.
//!
//! Research basis:
//! - Human pianists deviate ±7ms (4%) in controlled performance (Weber's Law)
//! - SD scales linearly with IOI duration above 300ms
//! - Beat 1 timing is more precise (humans anchor to downbeats)
//! - Velocity curves follow phrase arcs (crescendo/diminuendo)

use crate::Note;

/// Phi-dependent Gaussian timing jitter standard deviation (milliseconds).
///
/// High Phi (integrated consciousness) → tight timing (~8ms SD)
/// Low Phi (fragmented) → erratic timing (~22ms SD)
fn timing_sd_ms(phi: f32) -> f32 {
    7.0 + (1.0 - phi.clamp(0.0, 1.0)) * 15.0
}

/// Box-Muller Gaussian approximation from a seed.
/// Returns a value with mean 0 and SD ~1.
fn gaussian(seed: u32) -> f32 {
    let u1 = (seed.wrapping_mul(2654435761) >> 8) as f32 / 16777216.0;
    let u2 = (seed.wrapping_mul(1103515245).wrapping_add(12345) >> 8) as f32 / 16777216.0;
    let u1 = u1.max(0.0001);
    let r = (-2.0 * u1.ln()).sqrt();
    r * (std::f32::consts::TAU * u2).cos()
}

/// Apply humanization to a note (backward-compatible signature).
pub fn humanize(note: &mut Note, beat_position: f32, phrase_position: f32, seed: u32) {
    humanize_with_consciousness(note, beat_position, phrase_position, seed, 0.5, 0.5);
}

/// Full humanization with consciousness parameters.
///
/// - Phi-dependent Gaussian timing jitter (±7ms at high Phi, ±22ms at low Phi)
/// - Beat accent: beat 1 tighter + louder, beat 3 softer
/// - Phrase dynamics: crescendo to 60%, then diminuendo
/// - Arousal-dependent dynamic range compression
/// - Ghost notes at low arousal
/// - Legato overlap
pub fn humanize_with_consciousness(
    note: &mut Note,
    beat_position: f32,
    phrase_position: f32,
    seed: u32,
    phi: f32,
    arousal: f32,
) {
    // ── Timing: Phi-dependent Gaussian jitter ──
    let sd_ms = timing_sd_ms(phi);
    let beat_in_bar = beat_position % 4.0;
    // Beat 1 is tighter (humans anchor to downbeats)
    let beat_tightness = if beat_in_bar < 0.5 { 0.5 } else { 1.0 };
    let jitter_s = gaussian(seed) * sd_ms * beat_tightness / 1000.0;
    note.start_time = (note.start_time + jitter_s).max(0.0);

    // ── Velocity: beat accent ──
    if beat_in_bar < 0.5 {
        note.velocity = (note.velocity * 1.12).min(1.0);
    } else if (beat_in_bar - 2.0).abs() < 0.5 {
        note.velocity *= 0.93;
    }

    // ── Velocity: phrase dynamics ──
    let peak = 0.6;
    let dynamic_curve = if phrase_position < peak {
        0.82 + 0.18 * (phrase_position / peak)
    } else {
        1.0 - 0.18 * ((phrase_position - peak) / (1.0 - peak))
    };
    note.velocity *= dynamic_curve;

    // ── Velocity: arousal-dependent dynamic range ──
    let range_factor = 0.6 + arousal * 0.4;
    let center = 0.55;
    note.velocity = center + (note.velocity - center) * range_factor;

    // ── Velocity: Gaussian variation (±8%) ──
    let vel_jitter = gaussian(seed.wrapping_mul(7)) * 0.08;
    note.velocity = (note.velocity + vel_jitter).clamp(0.05, 1.0);

    // ── Ghost notes: occasional very soft notes at low arousal ──
    if arousal < 0.3 && seed % 7 == 0 {
        note.velocity *= 0.4;
    }

    // ── Legato: 10% duration extension ──
    note.duration *= 1.10;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_sd_high_phi() {
        let sd = timing_sd_ms(0.9);
        assert!(sd < 10.0, "high phi should have tight timing: {sd}ms");
    }

    #[test]
    fn timing_sd_low_phi() {
        let sd = timing_sd_ms(0.2);
        assert!(sd > 15.0, "low phi should have erratic timing: {sd}ms");
    }

    #[test]
    fn gaussian_distribution() {
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        let n = 1000;
        for i in 0..n {
            let g = gaussian(i as u32 * 12345 + 67890);
            sum += g;
            sum_sq += g * g;
        }
        let mean = sum / n as f32;
        let variance = sum_sq / n as f32 - mean * mean;
        assert!(mean.abs() < 0.3, "mean should be near 0: {mean}");
        assert!(
            variance > 0.2 && variance < 3.0,
            "variance should be near 1: {variance}"
        );
    }

    #[test]
    fn beat_one_accent() {
        let mut note = Note {
            frequency: 440.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.7,
        };
        humanize(&mut note, 0.0, 0.5, 42);
        assert!(note.velocity > 0.6, "beat 1 should be accented");
    }

    #[test]
    fn legato_extends_duration() {
        let mut note = Note {
            frequency: 440.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.7,
        };
        let orig_dur = note.duration;
        humanize(&mut note, 1.0, 0.5, 42);
        assert!(note.duration > orig_dur, "legato should extend duration");
    }

    #[test]
    fn velocity_stays_bounded() {
        for seed in 0..100 {
            let mut note = Note {
                frequency: 440.0,
                start_time: 0.0,
                duration: 0.5,
                velocity: 0.9,
            };
            humanize(&mut note, 0.0, 0.5, seed);
            assert!(note.velocity >= 0.05 && note.velocity <= 1.0);
        }
    }
}
