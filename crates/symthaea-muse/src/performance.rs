// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Performance expression: humanize notes with dynamics, rubato, and legato.

use crate::Note;

/// Apply humanization to a note.
///
/// - Beat accent: beat 1 slightly louder
/// - Velocity shaping: phrase arc (crescendo to peak, diminuendo after)
/// - Timing jitter: ±10ms randomized (via seed)
/// - Duration adjustment: legato overlap for smooth transitions
pub fn humanize(note: &mut Note, beat_position: f32, phrase_position: f32, seed: u32) {
    // Beat accent: beat 1 = +10% velocity, beat 3 = -5%
    let beat_in_bar = beat_position % 4.0;
    if beat_in_bar < 0.5 {
        note.velocity = (note.velocity * 1.1).min(1.0);
    } else if (beat_in_bar - 2.0).abs() < 0.5 {
        note.velocity *= 0.95;
    }

    // Phrase dynamics: crescendo to 60% through phrase, then diminuendo
    let peak = 0.6;
    let dynamic_curve = if phrase_position < peak {
        0.85 + 0.15 * (phrase_position / peak) // crescendo
    } else {
        1.0 - 0.15 * ((phrase_position - peak) / (1.0 - peak)) // diminuendo
    };
    note.velocity *= dynamic_curve;

    // Timing jitter: ±8ms (subtle, human-like)
    let jitter_ms = ((seed.wrapping_mul(2654435761) >> 16) as f32 / 32768.0 - 1.0) * 0.008;
    note.start_time = (note.start_time + jitter_ms).max(0.0);

    // Legato: extend duration by 15% for overlap with next note
    note.duration *= 1.15;

    // Velocity variation: ±5%
    let vel_jitter = ((seed.wrapping_mul(1103515245) >> 16) as f32 / 32768.0 - 1.0) * 0.05;
    note.velocity = (note.velocity + vel_jitter).clamp(0.05, 1.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beat_one_accent() {
        let mut note = Note { frequency: 440.0, start_time: 0.0, duration: 0.5, velocity: 0.7 };
        humanize(&mut note, 0.0, 0.5, 42); // beat 1
        assert!(note.velocity > 0.7, "beat 1 should be accented");
    }

    #[test]
    fn legato_extends_duration() {
        let mut note = Note { frequency: 440.0, start_time: 0.0, duration: 0.5, velocity: 0.7 };
        let orig_dur = note.duration;
        humanize(&mut note, 1.0, 0.5, 42);
        assert!(note.duration > orig_dur, "legato should extend duration");
    }

    #[test]
    fn velocity_stays_bounded() {
        for seed in 0..100 {
            let mut note = Note { frequency: 440.0, start_time: 0.0, duration: 0.5, velocity: 0.9 };
            humanize(&mut note, 0.0, 0.5, seed);
            assert!(note.velocity >= 0.05 && note.velocity <= 1.0);
        }
    }
}
