// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Performance expression primitives: legacy Phi-linked humanization and an
//! explicit, versioned profile API for parameters that can be calibrated from
//! instrument-specific performer evidence.
//!
//! The legacy Phi-to-jitter relation and the default profile values are
//! heuristics, not validated measures of musicianship or universal estimates
//! of human motor variability. In particular, Phi must not be interpreted as
//! instrumental skill. Product-level performance claims require role-specific
//! held-out measurements and listener evaluation.

use crate::Note;

/// Legacy heuristic mapping from Phi to Gaussian timing jitter SD (ms).
///
/// Retained for compatibility with existing callers; this is not a validated
/// relationship between Phi, consciousness, expressive timing, or performance skill.
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

/// Legacy humanization heuristic with consciousness-related inputs.
///
/// These input-to-performance mappings are engineering choices, not evidence that
/// Phi or arousal measure musicianship or predict the best expressive timing.
///
/// - Phi-dependent Gaussian timing jitter (approximately 7–22ms SD)
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
/// - legacy Φ-scaled Gaussian onset jitter (downbeat weighting is a heuristic,
///   not a calibrated cross-instrument performance rule), meter-aware via
///   `beats_per_bar` instead of the 4/4 hardcoded in the older path;
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

/// Explicit, versioned low-level performance parameters that can be calibrated from
/// performer data rather than inferred from a cognitive-state variable.
///
/// This is a per-note realization primitive, not a complete groove or
/// musicianship model: it does not coordinate multiple parts, generate fills,
/// model articulation techniques such as bends/slides, or establish quality.
/// Defaults are conservative engineering priors, not population estimates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerformanceProfileV1 {
    /// Standard deviation of onset perturbation in milliseconds, in [0, 50].
    pub timing_stddev_ms: f32,
    /// Multiplier for onset perturbation on the first beat of the bar, in [0, 1].
    pub downbeat_timing_scale: f32,
    /// Standard deviation of per-note velocity perturbation, in [0, 0.25].
    pub velocity_stddev: f32,
    /// Written duration multiplier, in [0.5, 1.5].
    pub duration_scale: f32,
}

impl Default for PerformanceProfileV1 {
    fn default() -> Self {
        Self {
            timing_stddev_ms: 7.0,
            downbeat_timing_scale: 0.5,
            velocity_stddev: 0.08,
            duration_scale: 1.08,
        }
    }
}

impl PerformanceProfileV1 {
    /// Check the versioned parameter bounds before applying the profile.
    pub fn validate(self) -> Result<(), &'static str> {
        if !self.timing_stddev_ms.is_finite()
            || !(0.0..=50.0).contains(&self.timing_stddev_ms)
        {
            return Err("timing_stddev_ms must be finite and within 0..=50");
        }
        if !self.downbeat_timing_scale.is_finite()
            || !(0.0..=1.0).contains(&self.downbeat_timing_scale)
        {
            return Err("downbeat_timing_scale must be finite and within 0..=1");
        }
        if !self.velocity_stddev.is_finite() || !(0.0..=0.25).contains(&self.velocity_stddev) {
            return Err("velocity_stddev must be finite and within 0..=0.25");
        }
        if !self.duration_scale.is_finite() || !(0.5..=1.5).contains(&self.duration_scale) {
            return Err("duration_scale must be finite and within 0.5..=1.5");
        }
        Ok(())
    }
}

/// Apply explicit performance controls to one note without using Phi as a
/// proxy for musical skill. The seed makes the perturbation deterministic;
/// jitter is capped at three standard deviations to avoid pathological
/// outliers. This function adds controlled variation only—it does not decide
/// whether the resulting performance is musically effective.
///
/// Invalid profiles, timing context, or note state are rejected before the
/// note is mutated.
pub fn humanize_score_note_with_profile(
    note: &mut Note,
    beat_position: f32,
    beats_per_bar: f32,
    seed: u32,
    profile: PerformanceProfileV1,
) -> Result<(), &'static str> {
    profile.validate()?;
    if !beat_position.is_finite() || !beats_per_bar.is_finite() || beats_per_bar <= 0.0 {
        return Err("beat_position must be finite and beats_per_bar must be finite and positive");
    }
    if !note.frequency.is_finite()
        || note.frequency <= 0.0
        || !note.start_time.is_finite()
        || !note.duration.is_finite()
        || !note.velocity.is_finite()
        || note.start_time < 0.0
        || note.duration < 0.0
        || note.duration > f32::MAX / profile.duration_scale
    {
        return Err("note frequency/timing/velocity must be finite and valid, and scaled duration representable");
    }

    let beat_in_bar = beat_position.rem_euclid(beats_per_bar);
    let timing_scale = if beat_in_bar < 0.5 {
        profile.downbeat_timing_scale
    } else {
        1.0
    };
    let max_jitter = 3.0 * profile.timing_stddev_ms;
    let jitter_ms = (gaussian(seed) * profile.timing_stddev_ms * timing_scale)
        .clamp(-max_jitter, max_jitter);
    note.start_time = (note.start_time + jitter_ms / 1000.0).max(0.0);

    let velocity_jitter = (gaussian(seed.wrapping_mul(7)) * profile.velocity_stddev)
        .clamp(-3.0 * profile.velocity_stddev, 3.0 * profile.velocity_stddev);
    note.velocity = (note.velocity + velocity_jitter).clamp(0.0, 1.0);
    note.duration *= profile.duration_scale;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_profile_can_disable_variation_without_changing_the_note() {
        let mut note = Note {
            frequency: 110.0,
            start_time: 2.0,
            duration: 0.25,
            velocity: 0.6,
        };
        let original = note;
        let profile = PerformanceProfileV1 {
            timing_stddev_ms: 0.0,
            downbeat_timing_scale: 0.0,
            velocity_stddev: 0.0,
            duration_scale: 1.0,
        };
        humanize_score_note_with_profile(&mut note, 1.0, 4.0, 17, profile).unwrap();
        assert_eq!(note.start_time, original.start_time);
        assert_eq!(note.duration, original.duration);
        assert_eq!(note.velocity, original.velocity);
    }

    #[test]
    fn explicit_profile_is_seed_deterministic_and_bounded() {
        let make = || Note {
            frequency: 110.0,
            start_time: 1.0,
            duration: 0.5,
            velocity: 0.98,
        };
        let profile = PerformanceProfileV1::default();
        let (mut a, mut b) = (make(), make());
        humanize_score_note_with_profile(&mut a, 2.0, 4.0, 77, profile).unwrap();
        humanize_score_note_with_profile(&mut b, 2.0, 4.0, 77, profile).unwrap();
        assert_eq!(a.start_time, b.start_time);
        assert_eq!(a.duration, b.duration);
        assert_eq!(a.velocity, b.velocity);
        assert!(a.start_time >= 0.0);
        assert!((0.0..=1.0).contains(&a.velocity));
        assert!((a.start_time - 1.0).abs() <= 0.150001);
        assert_eq!(a.duration, 0.54);
    }

    #[test]
    fn explicit_profile_rejects_bad_parameters_before_mutating_note() {
        let original = Note {
            frequency: 220.0,
            start_time: 1.0,
            duration: 0.25,
            velocity: 0.5,
        };
        let mut note = original;
        let mut invalid = PerformanceProfileV1::default();
        invalid.timing_stddev_ms = f32::NAN;
        assert!(humanize_score_note_with_profile(&mut note, 1.0, 4.0, 0, invalid).is_err());
        assert_eq!(note.start_time, original.start_time);
        assert_eq!(note.duration, original.duration);
        assert_eq!(note.velocity, original.velocity);

        assert!(humanize_score_note_with_profile(
            &mut note,
            f32::INFINITY,
            4.0,
            0,
            PerformanceProfileV1::default()
        ).is_err());
        assert!(humanize_score_note_with_profile(
            &mut note,
            1.0,
            0.0,
            0,
            PerformanceProfileV1::default()
        ).is_err());

        let mut oversized = original;
        oversized.duration = f32::MAX;
        assert!(humanize_score_note_with_profile(
            &mut oversized,
            1.0,
            4.0,
            0,
            PerformanceProfileV1::default()
        ).is_err());
        assert_eq!(oversized.duration, f32::MAX, "invalid input must not mutate the note");

        let mut nonfinite_frequency = original;
        nonfinite_frequency.frequency = f32::NAN;
        assert!(humanize_score_note_with_profile(
            &mut nonfinite_frequency,
            1.0,
            4.0,
            0,
            PerformanceProfileV1::default()
        ).is_err());
        assert_eq!(nonfinite_frequency.start_time, original.start_time);
    }

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
