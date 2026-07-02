// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Melodic grammar: rules that make notes sing instead of wander.
//!
//! Implements the fundamental principles of melodic writing:
//! - Stepwise motion 70-80% of the time (conjunct motion)
//! - Leaps resolved by step in the opposite direction
//! - Chord tones on strong beats, passing/neighbor tones on weak beats
//! - Approach target notes from a half-step below (leading tones)
//! - Tension-resolution arcs within phrases
//!
//! References:
//! - Huron (2006), Sweet Anticipation
//! - Narmour (1990), Implication-Realization Model

/// Melodic context for making the next note decision.
pub struct MelodicContext {
    /// Previous note frequency (None if first note).
    pub prev_freq: Option<f32>,
    /// Direction of last interval: -1 descending, 0 unison, 1 ascending.
    pub last_direction: i32,
    /// Size of last interval in semitones.
    pub last_interval: f32,
    /// Position within phrase [0, 1]: 0 = start, 1 = end.
    pub phrase_position: f32,
    /// Current beat position within bar [0, 4).
    pub beat_position: f32,
    /// Whether previous note was a leap (> 2 semitones).
    pub prev_was_leap: bool,
}

impl Default for MelodicContext {
    fn default() -> Self {
        Self {
            prev_freq: None,
            last_direction: 0,
            last_interval: 0.0,
            phrase_position: 0.0,
            beat_position: 0.0,
            prev_was_leap: false,
        }
    }
}

/// Apply melodic grammar to select the best note from available pitches.
///
/// `candidate`: the raw generated frequency
/// `chord_tones`: frequencies of current chord tones (sorted)
/// `scale_tones`: all scale frequencies available (sorted)
/// `ctx`: melodic context from previous notes
/// `seed`: for deterministic randomness
///
/// Returns the grammatically corrected frequency.
pub fn apply_grammar(
    candidate: f32,
    chord_tones: &[f32],
    scale_tones: &[f32],
    ctx: &MelodicContext,
    seed: u32,
) -> f32 {
    if scale_tones.is_empty() {
        return candidate;
    }

    let prev = match ctx.prev_freq {
        Some(f) => f,
        None => return nearest(candidate, chord_tones, scale_tones), // first note: snap to chord
    };

    // Rule 1: After a leap, resolve by step in opposite direction
    if ctx.prev_was_leap {
        let step = if ctx.last_direction > 0 { -1 } else { 1 }; // opposite
        if let Some(resolved) = step_from(prev, step, scale_tones) {
            return resolved;
        }
    }

    // Rule 2: Strong beats (1, 3) prefer chord tones
    let on_strong_beat = ctx.beat_position < 0.5 || (ctx.beat_position - 2.0).abs() < 0.5;
    if on_strong_beat && !chord_tones.is_empty() {
        // Find nearest chord tone that's a step away from prev
        if let Some(ct) = nearest_step(prev, chord_tones, scale_tones) {
            return ct;
        }
    }

    // Rule 3: Phrase ending → approach target from half-step below (leading tone)
    if ctx.phrase_position > 0.85 && !chord_tones.is_empty() {
        let root = chord_tones[0]; // resolve to root
        let leading_tone = root / 2.0f32.powf(1.0 / 12.0); // half step below
        // If we're close to phrase end, move toward the leading tone
        if ctx.phrase_position > 0.92 {
            return nearest(root, chord_tones, scale_tones); // final resolution
        }
        return nearest(leading_tone, scale_tones, scale_tones);
    }

    // Rule 4: Prefer stepwise motion (70% step, 20% skip, 10% leap)
    let r = (seed.wrapping_mul(2654435761) >> 16) % 100;
    if r < 70 {
        // Step: move by 1-2 semitones
        let direction = preferred_direction(ctx, seed);
        step_from(prev, direction, scale_tones).unwrap_or(candidate)
    } else if r < 90 {
        // Skip: move by 3-4 semitones (third)
        let direction = preferred_direction(ctx, seed);
        skip_from(prev, direction, scale_tones).unwrap_or(candidate)
    } else {
        // Leap: larger interval (for expression)
        // Snap to chord tone (leaps sound best to chord tones)
        nearest(candidate, chord_tones, scale_tones)
    }
}

/// Build tension-resolution arc for a phrase.
///
/// Returns a tension value [0, 1] for the given phrase position.
/// Peak tension at 60-70% through the phrase, then resolves.
pub fn tension_curve(phrase_position: f32) -> f32 {
    let peak = 0.65;
    if phrase_position < peak {
        // Build tension
        (phrase_position / peak).powi(2) * 0.8
    } else {
        // Resolve
        let release = (phrase_position - peak) / (1.0 - peak);
        0.8 * (1.0 - release.powi(2))
    }
}

/// Should we use a passing tone (non-chord tone) based on tension level?
pub fn use_passing_tone(tension: f32, seed: u32) -> bool {
    let threshold = (tension * 60.0) as u32; // up to 60% chance at peak tension
    let r = (seed.wrapping_mul(1103515245) >> 16) % 100;
    r < threshold
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn semitone_distance(a: f32, b: f32) -> f32 {
    (b / a).log2() * 12.0
}

fn nearest(target: f32, primary: &[f32], fallback: &[f32]) -> f32 {
    let pool = if primary.is_empty() {
        fallback
    } else {
        primary
    };
    pool.iter()
        .min_by(|a, b| {
            ((**a - target).abs())
                .partial_cmp(&((**b - target).abs()))
                .unwrap()
        })
        .copied()
        .unwrap_or(target)
}

fn preferred_direction(ctx: &MelodicContext, seed: u32) -> i32 {
    // Narmour: after a step, tend to continue in same direction
    // After a leap, reverse direction (already handled by leap resolution)
    if ctx.last_interval.abs() <= 2.0 && ctx.last_direction != 0 {
        ctx.last_direction // continue
    } else {
        // Random with slight ascending bias (melodies tend to rise then fall)
        if ctx.phrase_position < 0.5 {
            if (seed >> 8) % 3 < 2 { 1 } else { -1 } // 67% ascending in first half
        } else {
            if (seed >> 8) % 3 < 2 { -1 } else { 1 } // 67% descending in second half
        }
    }
}

fn step_from(freq: f32, direction: i32, scale: &[f32]) -> Option<f32> {
    // Find nearest scale tone to freq, then move one position in direction
    let idx = scale
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            ((**a - freq).abs())
                .partial_cmp(&((**b - freq).abs()))
                .unwrap()
        })
        .map(|(i, _)| i)?;

    let target_idx = (idx as i32 + direction).max(0) as usize;
    if target_idx < scale.len() && target_idx != idx {
        Some(scale[target_idx])
    } else {
        None
    }
}

fn skip_from(freq: f32, direction: i32, scale: &[f32]) -> Option<f32> {
    let idx = scale
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            ((**a - freq).abs())
                .partial_cmp(&((**b - freq).abs()))
                .unwrap()
        })
        .map(|(i, _)| i)?;

    let target_idx = (idx as i32 + direction * 2).max(0) as usize; // skip one scale degree
    if target_idx < scale.len() && target_idx != idx {
        Some(scale[target_idx])
    } else {
        None
    }
}

fn nearest_step(from: f32, targets: &[f32], _scale: &[f32]) -> Option<f32> {
    // Among targets, find one that's a step (1-2 scale degrees) from 'from'
    targets
        .iter()
        .filter(|&&t| {
            let dist = semitone_distance(from, t).abs();
            dist >= 0.5 && dist <= 4.0 // within a major third
        })
        .min_by(|a, b| {
            let da = semitone_distance(from, **a).abs();
            let db = semitone_distance(from, **b).abs();
            da.total_cmp(&db)
        })
        .copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c_major_scale() -> Vec<f32> {
        // C4 to C5
        vec![
            261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25,
        ]
    }

    fn c_major_chord() -> Vec<f32> {
        vec![261.63, 329.63, 392.00, 523.25] // C E G C
    }

    #[test]
    fn first_note_snaps_to_chord() {
        let ctx = MelodicContext::default();
        let result = apply_grammar(300.0, &c_major_chord(), &c_major_scale(), &ctx, 42);
        // Should snap to nearest chord tone (E4 = 329.63 or C4 = 261.63)
        assert!(
            c_major_chord().iter().any(|&c| (c - result).abs() < 1.0),
            "first note should be a chord tone: {result}"
        );
    }

    #[test]
    fn leap_resolves_by_step() {
        let ctx = MelodicContext {
            prev_freq: Some(523.25), // C5
            last_direction: 1,       // was ascending
            last_interval: 7.0,      // big leap
            prev_was_leap: true,
            ..Default::default()
        };
        let result = apply_grammar(523.25, &c_major_chord(), &c_major_scale(), &ctx, 42);
        // After ascending leap, should step DOWN
        assert!(result < 523.25, "should resolve leap downward: {result}");
    }

    #[test]
    fn phrase_ending_resolves() {
        let ctx = MelodicContext {
            prev_freq: Some(493.88), // B4 (leading tone)
            phrase_position: 0.95,   // near end
            ..Default::default()
        };
        let result = apply_grammar(440.0, &c_major_chord(), &c_major_scale(), &ctx, 42);
        // Should resolve to C (root) at phrase end
        let dist_to_root = (result - 261.63).abs().min((result - 523.25).abs());
        assert!(
            dist_to_root < 35.0,
            "phrase end should resolve near root: {result}"
        );
    }

    #[test]
    fn tension_curve_peaks_then_resolves() {
        let t_start = tension_curve(0.0);
        let t_peak = tension_curve(0.65);
        let t_end = tension_curve(1.0);

        assert!(t_peak > t_start, "tension should build");
        assert!(t_peak > t_end, "tension should resolve");
        assert!(t_end < 0.2, "end should be resolved: {t_end}");
    }

    #[test]
    fn stepwise_motion_dominates() {
        let scale = c_major_scale();
        let chord = c_major_chord();
        let mut steps = 0;
        let mut total = 0;

        let mut prev = 329.63; // E4
        for seed in 0..100 {
            let ctx = MelodicContext {
                prev_freq: Some(prev),
                last_direction: 1,
                last_interval: 2.0,
                prev_was_leap: false,
                phrase_position: 0.3,
                beat_position: 1.0,
            };
            let next = apply_grammar(prev, &chord, &scale, &ctx, seed * 7919);
            let interval = semitone_distance(prev, next).abs();
            if interval <= 3.0 {
                steps += 1;
            }
            total += 1;
            prev = next;
        }

        let step_ratio = steps as f32 / total as f32;
        assert!(
            step_ratio > 0.5,
            "stepwise motion should dominate: {step_ratio:.2} ({steps}/{total})"
        );
    }
}
