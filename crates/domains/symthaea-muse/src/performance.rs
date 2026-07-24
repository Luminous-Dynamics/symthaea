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

/// Humanize a note realized from a symbolic theory `Score`. Deliberately a
/// SUBSET of [`humanize_with_consciousness`]: a theory score's velocities
/// already carry phrase arch, section intensity, and climax accents
/// (`symthaea-music-theory`'s composer), so applying the full phrase-dynamics
/// curve here would double-apply structural dynamics. What a score CANNOT
/// represent — and what this adds — is human motor variance:
/// - Φ-scaled Gaussian onset jitter (downbeats anchored tighter, matching
///   measured pianist behavior), meter-aware via `beats_per_bar` instead of
///   the 4/4 hardcoded in the older path (a Waltz has 3 beats to a bar);
/// - ±8% Gaussian velocity noise (small enough not to fight the score's
///   structural dynamics);
/// - optional legato overlap for sustained melodic/bass lines (block chords
///   should NOT get it — extending every chord tone smears the harmonic
///   rhythm into mud).
///
/// `timing_scale` scales the TIMING jitter SD only (velocity noise is
/// dynamics, not clock) — the performance-dialect hook: a groove dialect
/// tightens it (e.g. 0.6), a session dialect loosens it (e.g. 1.2). Pass
/// 1.0 for the historical behavior (bit-identical — ×1.0 is exact).
pub fn humanize_score_note(
    note: &mut Note,
    beat_position: f32,
    beats_per_bar: f32,
    seed: u32,
    phi: f32,
    legato: bool,
    timing_scale: f32,
) {
    let sd_ms = timing_sd_ms(phi) * timing_scale.max(0.0);
    let beat_in_bar = beat_position % beats_per_bar.max(1.0);
    // Downbeats anchor tighter (humans lock to beat 1).
    let beat_tightness = if beat_in_bar < 0.5 { 0.5 } else { 1.0 };
    let jitter_s = gaussian(seed) * sd_ms * beat_tightness / 1000.0;
    note.start_time = (note.start_time + jitter_s).max(0.0);

    let vel_jitter = gaussian(seed.wrapping_mul(7)) * 0.08;
    note.velocity = (note.velocity + vel_jitter).clamp(0.05, 1.0);

    if legato {
        note.duration *= 1.08;
    }
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

    #[test]
    fn score_humanize_is_deterministic_and_bounded() {
        let make = || Note {
            frequency: 440.0,
            start_time: 2.0,
            duration: 0.5,
            velocity: 0.7,
        };
        for seed in 0..200 {
            let mut a = make();
            let mut b = make();
            humanize_score_note(&mut a, 1.0, 4.0, seed, 0.5, true, 1.0);
            humanize_score_note(&mut b, 1.0, 4.0, seed, 0.5, true, 1.0);
            assert_eq!(a.start_time, b.start_time, "same seed → same jitter");
            assert_eq!(a.velocity, b.velocity);
            // Jitter must stay in the tens-of-milliseconds regime — audible
            // as human looseness, never as a wrong rhythm.
            assert!(
                (a.start_time - 2.0).abs() < 0.15,
                "jitter too large: {}",
                a.start_time - 2.0
            );
            assert!(a.velocity >= 0.05 && a.velocity <= 1.0);
        }
    }

    #[test]
    fn score_humanize_legato_only_when_asked() {
        let mut melodic = Note {
            frequency: 440.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.7,
        };
        let mut chordal = melodic;
        humanize_score_note(&mut melodic, 0.0, 4.0, 3, 0.5, true, 1.0);
        humanize_score_note(&mut chordal, 0.0, 4.0, 3, 0.5, false, 1.0);
        assert!(melodic.duration > 0.5, "legato should extend");
        assert_eq!(chordal.duration, 0.5, "chords keep their written length");
    }

    #[test]
    fn score_humanize_timing_scale_scales_only_the_clock() {
        // The dialect hook: timing_scale shrinks (or grows) the onset
        // jitter proportionally, leaves velocity noise untouched, and at
        // 0.0 pins the note to its written onset exactly.
        for seed in 1..100 {
            let make = || Note {
                frequency: 440.0,
                start_time: 2.0,
                duration: 0.5,
                velocity: 0.7,
            };
            let (mut full, mut tight, mut frozen) = (make(), make(), make());
            humanize_score_note(&mut full, 1.0, 4.0, seed, 0.5, false, 1.0);
            humanize_score_note(&mut tight, 1.0, 4.0, seed, 0.5, false, 0.6);
            humanize_score_note(&mut frozen, 1.0, 4.0, seed, 0.5, false, 0.0);
            let full_dev = full.start_time - 2.0;
            let tight_dev = tight.start_time - 2.0;
            assert!(
                (tight_dev - full_dev * 0.6).abs() < 1e-6,
                "seed {seed}: jitter must scale linearly ({full_dev} vs {tight_dev})"
            );
            assert_eq!(frozen.start_time, 2.0, "scale 0.0 must freeze the clock");
            // Velocity noise is NOT the clock — identical across scales.
            assert_eq!(full.velocity, tight.velocity);
            assert_eq!(full.velocity, frozen.velocity);
        }
    }

    #[test]
    fn score_humanize_does_not_reshape_structural_dynamics() {
        // The velocity change must be small noise, not a phrase curve: over
        // many seeds the mean velocity stays near the written value.
        let mut sum = 0.0f32;
        let n = 500;
        for seed in 0..n {
            let mut note = Note {
                frequency: 440.0,
                start_time: 1.0,
                duration: 0.5,
                velocity: 0.6,
            };
            humanize_score_note(&mut note, 2.0, 4.0, seed, 0.5, false, 1.0);
            sum += note.velocity;
        }
        let mean = sum / n as f32;
        assert!(
            (mean - 0.6).abs() < 0.03,
            "velocity noise must be zero-mean around the written value: {mean}"
        );
    }
}
