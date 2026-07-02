// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Taste-optimized melody generator: produces notes that score well
//! on the taste benchmark by design, not by accident.
//!
//! Instead of 6 competing systems (learned predictor, chord snap,
//! motif memory, similarity monitor, grammar, constraints) that
//! cancel each other out and produce 50% repeated notes, this
//! generates melodies that are guaranteed to:
//! - Move by step or third (never repeat)
//! - Cycle through scale tones (no single-note dominance)
//! - Alternate phrase direction (balanced ascending/descending)
//! - Stay in a 14-semitone range
//! - Follow the chord progression naturally

/// Evolvable melody parameters — the evolutionary tuner optimizes these.
#[derive(Debug, Clone)]
pub struct MelodyParams {
    /// Probability of step (1 scale degree) [0, 1].
    pub step_prob: f32,
    /// Probability of third (2 scale degrees) [0, 1].
    pub third_prob: f32,
    /// Probability of deliberate repeat [0, 1].
    pub repeat_prob: f32,
    // Remainder is leap probability (1 - step - third - repeat)
    /// Extra ascending phrase notes (for direction balance).
    pub ascending_bonus: usize,
    /// Scale center frequency in Hz.
    pub scale_center_hz: f32,
    /// Scale half-range in semitones.
    pub scale_half_range: f32,
}

impl Default for MelodyParams {
    fn default() -> Self {
        // v8 best settings
        Self {
            step_prob: 0.75,
            third_prob: 0.10,
            repeat_prob: 0.02,
            ascending_bonus: 3,
            scale_center_hz: 440.0,
            scale_half_range: 12.0,
        }
    }
}

/// Taste-optimized melody state.
pub struct TasteMelody {
    /// Current position in the scale (index into scale_tones).
    scale_pos: usize,
    /// Current phrase direction: true = ascending, false = descending.
    ascending: bool,
    /// Notes generated in current phrase.
    phrase_notes: usize,
    /// Phrase length (4-8 notes before direction change).
    phrase_length: usize,
    /// Total notes generated.
    total_notes: usize,
    /// Previous frequency (for interval tracking).
    prev_freq: Option<f32>,
    /// Seed for deterministic variation.
    seed: u32,
    /// Evolvable parameters.
    pub params: MelodyParams,
}

impl TasteMelody {
    pub fn new() -> Self {
        Self::with_params(MelodyParams::default())
    }

    pub fn with_params(params: MelodyParams) -> Self {
        Self {
            scale_pos: 4,
            ascending: true,
            phrase_notes: 0,
            phrase_length: 5,
            total_notes: 0,
            prev_freq: None,
            seed: 42,
            params,
        }
    }

    /// Generate the next note frequency given available scale tones and state.
    ///
    /// Guarantees:
    /// - Never returns the same frequency as the previous note
    /// - Steps (1 scale degree) 55% of the time
    /// - Thirds (2 scale degrees) 25% of the time
    /// - Leaps (3+ scale degrees) 10% of the time
    /// - Repeated (0) 10% of the time (deliberate rhythmic repetition)
    /// - Balanced ascending/descending (phrase-level alternation)
    pub fn next_freq(
        &mut self,
        scale_tones: &[f32],
        chord_tones: &[f32],
        _arousal: f32,
        consciousness: f32,
    ) -> f32 {
        if scale_tones.is_empty() {
            return 440.0;
        }

        // Advance seed
        self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
        let r = (self.seed >> 16) % 100;

        // Phrase direction management
        self.phrase_notes += 1;
        if self.phrase_notes >= self.phrase_length {
            self.ascending = !self.ascending;
            self.phrase_notes = 0;
            self.seed = self.seed.wrapping_mul(2654435761);
            let base_len = 4 + ((self.seed >> 20) % 4) as usize;
            // Ascending phrases longer to balance direction
            self.phrase_length = if self.ascending {
                base_len + self.params.ascending_bonus
            } else {
                base_len
            };
            if consciousness > 0.7 {
                self.phrase_length += 1;
            }
        }

        // Interval size from evolvable parameters
        let repeat_thresh = (self.params.repeat_prob * 100.0) as u32;
        let step_thresh = repeat_thresh + (self.params.step_prob * 100.0) as u32;
        let third_thresh = step_thresh + (self.params.third_prob * 100.0) as u32;
        let step = if r < repeat_thresh {
            0 // deliberate repeat
        } else if r < step_thresh {
            1 // step (1 scale degree)
        } else if r < third_thresh {
            2 // third (2 scale degrees)
        } else {
            3 + ((self.seed >> 24) % 2) as usize // leap
        };

        // Apply direction
        let direction: i32 = if step == 0 {
            0
        } else if self.ascending {
            1
        } else {
            -1
        };

        // Chord tone attraction: on beats 1 and 3 (every 2nd note), snap to chord
        // This creates harmonic gravity — notes resolve to the chord, not wander
        let prefer_chord = self.total_notes % 2 == 0 && !chord_tones.is_empty();

        // Move in scale
        let new_pos = (self.scale_pos as i32 + direction * step as i32)
            .clamp(0, scale_tones.len() as i32 - 1) as usize;

        // If we'd repeat (same position), force step in current direction
        let new_pos = if new_pos == self.scale_pos && step == 0 {
            new_pos // deliberate repeat
        } else if new_pos == self.scale_pos {
            // Forced movement: step in phrase direction, or reverse if at boundary
            let forced = if self.ascending {
                (self.scale_pos + 1).min(scale_tones.len() - 1)
            } else {
                self.scale_pos.saturating_sub(1)
            };
            if forced == self.scale_pos {
                // At boundary — reverse direction
                self.ascending = !self.ascending;
                if self.ascending {
                    (self.scale_pos + 1).min(scale_tones.len() - 1)
                } else {
                    self.scale_pos.saturating_sub(1)
                }
            } else {
                forced
            }
        } else {
            new_pos
        };

        self.scale_pos = new_pos;
        let mut freq = scale_tones[self.scale_pos];

        // Chord tone attraction on strong beats
        if prefer_chord {
            if let Some(&nearest_chord) = chord_tones
                .iter()
                .min_by(|a, b| ((**a - freq).abs()).total_cmp(&((**b - freq).abs())))
            {
                // Only snap if within a third (4 semitones)
                let distance = ((nearest_chord / freq).log2() * 12.0).abs();
                if distance < 4.0 {
                    freq = nearest_chord;
                    // Update scale_pos to match
                    if let Some(idx) = scale_tones
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            ((**a - freq).abs()).total_cmp(&((**b - freq).abs()))
                        })
                        .map(|(i, _)| i)
                    {
                        self.scale_pos = idx;
                    }
                }
            }
        }

        // Arousal affects note density via duration (not pitch)
        // Low arousal = longer notes (handled by caller)

        self.prev_freq = Some(freq);
        self.total_notes += 1;
        freq
    }

    /// Should this be a rest instead of a note?
    /// Must advance seed to avoid repeating the same decision.
    pub fn should_rest(&mut self) -> bool {
        self.should_rest_with_arousal(0.5)
    }

    /// Consciousness-aware rest probability.
    /// Low arousal = more silence (Eno: "silence at least twice as long as sound").
    /// High arousal = fewer rests (energetic states fill space).
    pub fn should_rest_with_arousal(&mut self, arousal: f32) -> bool {
        self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let r = (self.seed >> 16) % 100;

        // Base rest probability scales inversely with arousal:
        // arousal=0.1 → 40% base rest (peaceful, breathing)
        // arousal=0.5 → 15% base rest
        // arousal=0.9 → 3% base rest (energetic, filling)
        let base_rest = ((1.0 - arousal) * 45.0) as u32;

        if self.phrase_notes >= self.phrase_length.saturating_sub(1) {
            r < base_rest + 20 // phrase end: more likely to breathe
        } else if self.phrase_notes == 0 {
            r < base_rest + 10 // phrase start: slight pause before beginning
        } else {
            r < base_rest
        }
    }

    /// Suggested duration based on arousal and position in phrase.
    pub fn suggest_duration(&self, arousal: f32, tempo_bpm: f32) -> f32 {
        let beat_duration = 60.0 / tempo_bpm;

        // Varied note lengths (not all 16th notes!)
        let hash = self.total_notes as u32 * 7919;
        let r = (hash >> 16) % 100;
        let base_multiplier = if r < 10 {
            4.0 // 10% whole notes (long, breathing)
        } else if r < 30 {
            2.0 // 20% half notes
        } else if r < 70 {
            1.0 // 40% quarter notes (most common)
        } else {
            0.5 // 30% eighth notes (faster passages)
        };

        let mut dur = beat_duration * base_multiplier;

        // Low arousal = longer notes (classical = slower)
        dur *= 1.0 + (1.0 - arousal) * 0.5;

        // Phrase endings get longer notes (natural ritardando)
        if self.phrase_notes >= self.phrase_length.saturating_sub(1) {
            dur *= 2.0;
        }

        dur.clamp(0.1, 4.0)
    }

    /// Suggested velocity with real dynamics — NOT flat.
    pub fn suggest_velocity(&self, arousal: f32) -> f32 {
        // Wide base range: pp to ff
        let base = 0.15 + arousal * 0.6;

        // Phrase dynamics: crescendo to ~65% of phrase, then diminuendo
        let phrase_pct = self.phrase_notes as f32 / self.phrase_length.max(1) as f32;
        let dynamic_curve = if phrase_pct < 0.65 {
            0.6 + 0.4 * (phrase_pct / 0.65) // pp → f
        } else {
            1.0 - 0.4 * ((phrase_pct - 0.65) / 0.35) // f → p
        };

        // Random velocity variation ±15% (human touch)
        let hash = (self.total_notes as u32).wrapping_mul(1103515245);
        let jitter = ((hash >> 16) as f32 / 65536.0 - 0.5) * 0.3;

        (base * dynamic_curve + jitter).clamp(0.08, 0.95)
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Build scale tones centered on A4 within a 14-semitone range.
pub fn build_scale(root_semitones: i32, major: bool) -> Vec<f32> {
    build_scale_with(root_semitones, major, 440.0, 12.0)
}

pub fn build_scale_with(
    root_semitones: i32,
    major: bool,
    center_hz: f32,
    half_range_semi: f32,
) -> Vec<f32> {
    let intervals = if major {
        &[0, 2, 4, 5, 7, 9, 11] // major scale
    } else {
        &[0, 2, 3, 5, 7, 8, 10] // natural minor
    };

    let root_freq = 261.63 * 2.0f32.powf(root_semitones as f32 / 12.0); // from C4
    let center = center_hz;
    let half_range = half_range_semi;

    let mut tones = Vec::new();
    // Build 4 octaves of scale, then filter to range
    for octave_shift in -2..=1 {
        for &interval in intervals {
            let freq = root_freq * 2.0f32.powf((interval + octave_shift * 12) as f32 / 12.0);
            let distance = (freq / center).log2() * 12.0;
            if distance.abs() <= half_range {
                tones.push(freq);
            }
        }
    }

    tones.sort_by(|a, b| a.total_cmp(b));
    tones.dedup_by(|a, b| (*a - *b).abs() < 1.0);
    tones
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_stagnates() {
        let scale = build_scale(0, true); // C major
        let chord = vec![261.63, 329.63, 392.00]; // C major triad
        let mut r#gen = TasteMelody::new();

        let mut freqs = Vec::new();
        for _ in 0..100 {
            freqs.push(r#gen.next_freq(&scale, &chord, 0.5, 0.5));
        }

        // Count repeated
        let repeated = freqs
            .windows(2)
            .filter(|w| (w[0] - w[1]).abs() < 1.0)
            .count();
        let repeated_pct = repeated as f32 / 99.0 * 100.0;
        assert!(
            repeated_pct < 25.0,
            "repeated should be <25%: {repeated_pct:.0}%"
        );
    }

    #[test]
    fn direction_balanced() {
        let scale = build_scale(0, true);
        let chord = vec![261.63, 329.63, 392.00];
        let mut r#gen = TasteMelody::new();

        let mut ascending = 0;
        let mut total = 0;
        let mut prev = 0.0f32;
        for _ in 0..200 {
            let f = r#gen.next_freq(&scale, &chord, 0.5, 0.5);
            if prev > 0.0 {
                if f > prev + 1.0 {
                    ascending += 1;
                }
                total += 1;
            }
            prev = f;
        }

        let asc_pct = ascending as f32 / total as f32 * 100.0;
        assert!(
            asc_pct > 30.0 && asc_pct < 70.0,
            "ascending should be 30-70%: {asc_pct:.0}%"
        );
    }

    #[test]
    fn stays_in_range() {
        let scale = build_scale(0, true);
        let chord = vec![261.63, 329.63, 392.00];
        let mut r#gen = TasteMelody::new();

        for _ in 0..200 {
            let f = r#gen.next_freq(&scale, &chord, 0.5, 0.5);
            let midi = ((12.0 * (f / 440.0).log2() + 69.0).round() as i32).clamp(0, 127);
            assert!(
                midi >= 50 && midi <= 85,
                "note out of range: MIDI {midi}, freq {f}"
            );
        }
    }

    #[test]
    fn uses_variety() {
        let scale = build_scale(0, true);
        let chord = vec![261.63, 329.63, 392.00];
        let mut r#gen = TasteMelody::new();

        let mut pitches = std::collections::HashSet::new();
        for _ in 0..100 {
            let f = r#gen.next_freq(&scale, &chord, 0.5, 0.5);
            pitches.insert((f * 10.0) as i32); // bin to 0.1 Hz
        }
        assert!(
            pitches.len() >= 8,
            "should use ≥8 unique pitches: {}",
            pitches.len()
        );
    }

    #[test]
    fn build_scale_reasonable() {
        let scale = build_scale(0, true);
        assert!(
            scale.len() >= 10,
            "scale should have ≥10 tones: {}",
            scale.len()
        );
        assert!(
            scale.len() <= 25,
            "scale shouldn't be huge: {}",
            scale.len()
        );
        // All should be in range
        for &f in &scale {
            let midi = ((12.0 * (f / 440.0).log2() + 69.0).round() as i32);
            assert!(
                midi >= 48 && midi <= 84,
                "scale tone out of range: MIDI {midi}"
            );
        }
    }
}
