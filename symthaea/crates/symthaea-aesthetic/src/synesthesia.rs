// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-modal synesthesia: bidirectional mappings between sound and vision.
//!
//! Implements psychoacoustic and synesthetic research linking pitch to color,
//! loudness to brightness, and timbre to texture. Grounded in Scriabin's
//! *clavier à lumières* (1911) with modern perceptual adjustments.
//!
//! # Audio → Visual Mappings
//!
//! | Sound Property | Visual Property | Basis |
//! |----------------|-----------------|-------|
//! | Pitch (Hz) | Hue (0-360°) | Scriabin chromatic circle |
//! | Loudness [0,1] | Lightness [0,1] | Psychoacoustic brightness |
//! | Timbre [4] | Texture params | Harmonic spectrum → visual grain |
//! | Tempo (BPM) | Motion speed | Rhythmic entrainment |
//!
//! # Visual → Audio Mappings
//!
//! | Visual Property | Sound Property | Basis |
//! |-----------------|----------------|-------|
//! | Hue (0-360°) | Pitch (Hz) | Inverse Scriabin |
//! | Saturation [0,1] | Timbre brightness | Color intensity ↔ spectral centroid |
//! | Complexity [0,1] | Note density | Visual busyness → rhythmic density |

use serde::{Deserialize, Serialize};

// ─── Scriabin Pitch-Color Mapping ───────────────────────────────────────────

/// Scriabin's pitch-class to hue mapping (degrees on the color wheel).
///
/// Based on his *Prometheus: The Poem of Fire* (1910), adjusted for
/// perceptual uniformity in HSL color space.
///
/// C=0 (Red), C#=30, D=60 (Yellow), D#=90, E=120 (Green),
/// F=150, F#=180 (Cyan), G=210 (Blue), G#=240, A=270 (Violet),
/// A#=300 (Magenta), B=330 (Rose)
const SCRIABIN_HUE: [f32; 12] = [
    0.0,   // C  → Red
    30.0,  // C# → Orange-Red
    60.0,  // D  → Yellow
    90.0,  // D# → Yellow-Green
    120.0, // E  → Green
    150.0, // F  → Cyan-Green
    180.0, // F# → Cyan
    210.0, // G  → Blue
    240.0, // G# → Blue-Violet
    270.0, // A  → Violet
    300.0, // A# → Magenta
    330.0, // B  → Rose
];

// ─── Audio → Visual ─────────────────────────────────────────────────────────

/// Convert a frequency (Hz) to a hue (0-360 degrees) via Scriabin's mapping.
///
/// Uses the pitch class (chromatic position modulo 12) with interpolation
/// for frequencies between semitones.
pub fn pitch_to_hue(freq_hz: f32) -> f32 {
    if freq_hz <= 0.0 {
        return 0.0;
    }
    // MIDI note = 69 + 12 * log2(freq / 440)
    let midi = 69.0 + 12.0 * (freq_hz / 440.0).log2();
    // Pitch class (0-12, fractional for interpolation)
    let pitch_class = ((midi % 12.0) + 12.0) % 12.0;
    let lower = pitch_class.floor() as usize % 12;
    let upper = (lower + 1) % 12;
    let frac = pitch_class - pitch_class.floor();

    // Interpolate on the color wheel (handle wrap-around)
    let h0 = SCRIABIN_HUE[lower];
    let h1 = SCRIABIN_HUE[upper];
    let diff = h1 - h0;
    let hue = if diff.abs() <= 180.0 {
        h0 + frac * diff
    } else if diff > 0.0 {
        ((h0 + frac * (diff - 360.0)) + 360.0) % 360.0
    } else {
        ((h0 + frac * (diff + 360.0)) + 360.0) % 360.0
    };
    hue % 360.0
}

/// Convert loudness [0,1] to visual lightness [0,1].
///
/// Louder sounds → brighter visuals. Slight gamma curve for perceptual uniformity.
pub fn loudness_to_lightness(loudness: f32) -> f32 {
    let l = loudness.clamp(0.0, 1.0);
    // Gamma 0.8: slight boost to make quiet sounds visible
    0.15 + 0.7 * l.powf(0.8)
}

/// Convert timbre (4 harmonic partial amplitudes) to visual texture parameters.
///
/// Returns (grain_size, roughness, warmth):
/// - grain_size [0,1]: smaller when more upper partials (bright timbre)
/// - roughness [0,1]: higher when harmonics are unbalanced
/// - warmth [0,1]: higher when fundamental dominates
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TextureParams {
    /// Visual grain size: large for simple tones, small for complex.
    pub grain_size: f32,
    /// Surface roughness: smooth for pure tones, rough for complex.
    pub roughness: f32,
    /// Color warmth: warm for fundamental-heavy, cool for overtone-heavy.
    pub warmth: f32,
}

pub fn timbre_to_texture(timbre: &[f32; 4]) -> TextureParams {
    let total: f32 = timbre.iter().sum::<f32>().max(0.001);
    let fundamental_ratio = timbre[0] / total;
    let upper_energy: f32 = timbre[1..].iter().sum::<f32>() / total;

    // Spectral centroid (weighted average of partial index)
    let centroid: f32 = timbre
        .iter()
        .enumerate()
        .map(|(i, &a)| (i + 1) as f32 * a)
        .sum::<f32>()
        / total;

    TextureParams {
        grain_size: (1.0 - upper_energy * 0.8).clamp(0.1, 1.0),
        roughness: (1.0 - fundamental_ratio).clamp(0.0, 1.0),
        warmth: (1.0 - (centroid - 1.0) / 3.0).clamp(0.0, 1.0),
    }
}

/// Convert tempo (BPM) to motion speed [0,1].
///
/// Maps 30-300 BPM to 0-1 with a log-like curve (perceptual tempo scaling).
pub fn tempo_to_motion(bpm: f32) -> f32 {
    let bpm = bpm.clamp(30.0, 300.0);
    ((bpm - 30.0) / 270.0).powf(0.7)
}

// ─── Visual → Audio ─────────────────────────────────────────────────────────

/// Convert a hue (0-360 degrees) back to a frequency (Hz).
///
/// Inverse of Scriabin mapping. Returns a frequency in octave 4 (middle register).
pub fn hue_to_pitch(hue: f32) -> f32 {
    let h = ((hue % 360.0) + 360.0) % 360.0;

    // Find the two bracketing pitch classes
    let mut best_lower = 0usize;
    let mut best_dist = 360.0f32;
    for (i, &sh) in SCRIABIN_HUE.iter().enumerate() {
        let dist = ((h - sh) + 360.0) % 360.0;
        if dist < best_dist {
            best_dist = dist;
            best_lower = i;
        }
    }

    // Pitch class → MIDI note in octave 4 (C4=60)
    let midi_note = 60.0 + best_lower as f32;
    // MIDI → frequency
    440.0 * 2.0_f32.powf((midi_note - 69.0) / 12.0)
}

/// Convert saturation [0,1] to timbre brightness.
///
/// Returns a 4-element partial array where higher saturation = more upper partials.
pub fn saturation_to_timbre(saturation: f32) -> [f32; 4] {
    let s = saturation.clamp(0.0, 1.0);
    [
        1.0,
        s * 0.5,
        s * s * 0.3,
        s * s * s * 0.15,
    ]
}

/// Convert visual complexity [0,1] to note density multiplier.
///
/// More visually complex scenes → more notes per beat.
pub fn complexity_to_density(complexity: f32) -> f32 {
    let c = complexity.clamp(0.0, 1.0);
    0.3 + c * 1.7 // range: 0.3 to 2.0 notes per beat
}

// ─── Synesthetic Frame ──────────────────────────────────────────────────────

/// Per-beat synesthetic features extracted from a musical composition.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SynestheticFrame {
    /// Dominant hue for this beat (0-360 degrees).
    pub hue: f32,
    /// Lightness for this beat (0-1).
    pub lightness: f32,
    /// Texture parameters.
    pub texture: TextureParams,
    /// Motion speed (0-1).
    pub motion: f32,
    /// Beat time in seconds.
    pub time: f32,
}

/// Extract synesthetic visual features from a note sequence.
///
/// Groups notes into time windows (one per beat) and computes the
/// dominant visual parameters for each window.
pub fn extract_synesthetic_features(
    notes: &[(f32, f32, f32, f32)], // (frequency, start_time, duration, velocity)
    tempo_bpm: f32,
    duration_secs: f32,
) -> Vec<SynestheticFrame> {
    if notes.is_empty() {
        return vec![];
    }

    let beat_duration = 60.0 / tempo_bpm.max(30.0);
    let num_beats = (duration_secs / beat_duration).ceil() as usize;
    let mut frames = Vec::with_capacity(num_beats);

    for beat_idx in 0..num_beats {
        let beat_start = beat_idx as f32 * beat_duration;
        let beat_end = beat_start + beat_duration;

        // Collect notes active during this beat
        let active: Vec<&(f32, f32, f32, f32)> = notes
            .iter()
            .filter(|&&(_, start, dur, _)| {
                start < beat_end && (start + dur) > beat_start
            })
            .collect();

        if active.is_empty() {
            // Rest beat: dim, neutral
            frames.push(SynestheticFrame {
                hue: frames.last().map_or(0.0, |f: &SynestheticFrame| f.hue),
                lightness: 0.15,
                texture: TextureParams {
                    grain_size: 1.0,
                    roughness: 0.0,
                    warmth: 0.5,
                },
                motion: tempo_to_motion(tempo_bpm) * 0.3,
                time: beat_start,
            });
            continue;
        }

        // Weighted average frequency (by velocity)
        let total_vel: f32 = active.iter().map(|n| n.3).sum();
        let avg_freq = if total_vel > 0.0 {
            active.iter().map(|n| n.0 * n.3).sum::<f32>() / total_vel
        } else {
            active[0].0
        };

        // Loudness: max velocity in this beat
        let max_vel = active.iter().map(|n| n.3).fold(0.0f32, f32::max);

        frames.push(SynestheticFrame {
            hue: pitch_to_hue(avg_freq),
            lightness: loudness_to_lightness(max_vel),
            texture: timbre_to_texture(&saturation_to_timbre(max_vel)),
            motion: tempo_to_motion(tempo_bpm),
            time: beat_start,
        });
    }

    frames
}

// ─── Blend Feedback ─────────────────────────────────────────────────────────

/// Blend aesthetic feedbacks from multiple modalities.
///
/// Used for synesthetic works that produce both visual and audio output.
pub fn blend_feedbacks(
    feedbacks: &[super::AestheticFeedback],
) -> super::AestheticFeedback {
    if feedbacks.is_empty() {
        return super::AestheticFeedback::neutral();
    }
    let n = feedbacks.len() as f32;
    let mut result = super::AestheticFeedback::neutral();
    for f in feedbacks {
        result.dopamine_delta += f.dopamine_delta / n;
        result.serotonin_delta += f.serotonin_delta / n;
        result.surprise_signal += f.surprise_signal / n;
        for i in 0..8 {
            result.harmony_projection[i] += f.harmony_projection[i] / n;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Pitch-to-Hue tests ──

    #[test]
    fn c4_maps_to_red() {
        let hue = pitch_to_hue(261.63); // C4
        assert!(hue < 15.0 || hue > 345.0, "C4 should be near red (0°), got {hue}");
    }

    #[test]
    fn a4_maps_to_violet() {
        let hue = pitch_to_hue(440.0); // A4
        assert!(
            (hue - 270.0).abs() < 15.0,
            "A4 should be near violet (270°), got {hue}"
        );
    }

    #[test]
    fn e4_maps_to_green() {
        let hue = pitch_to_hue(329.63); // E4
        assert!(
            (hue - 120.0).abs() < 15.0,
            "E4 should be near green (120°), got {hue}"
        );
    }

    #[test]
    fn pitch_to_hue_monotonic_within_octave() {
        // C through B should traverse the color wheel in order
        let freqs = [261.63, 277.18, 293.66, 311.13, 329.63, 349.23,
                     369.99, 392.00, 415.30, 440.00, 466.16, 493.88];
        let hues: Vec<f32> = freqs.iter().map(|&f| pitch_to_hue(f)).collect();
        // Each should be roughly 30° apart
        for w in hues.windows(2) {
            let diff = ((w[1] - w[0]) + 360.0) % 360.0;
            assert!(
                diff > 10.0 && diff < 50.0,
                "adjacent hues should be ~30° apart, got {diff} ({} → {})",
                w[0], w[1]
            );
        }
    }

    #[test]
    fn octave_equivalence() {
        // C4 and C5 should map to the same hue
        let h4 = pitch_to_hue(261.63);
        let h5 = pitch_to_hue(523.25);
        assert!(
            (h4 - h5).abs() < 5.0 || (h4 - h5).abs() > 355.0,
            "octaves should share hue: C4={h4}, C5={h5}"
        );
    }

    // ── Loudness-to-Lightness tests ──

    #[test]
    fn silence_is_dim() {
        let l = loudness_to_lightness(0.0);
        assert!(l < 0.25, "silence should be dim, got {l}");
    }

    #[test]
    fn loud_is_bright() {
        let l = loudness_to_lightness(1.0);
        assert!(l > 0.7, "loud should be bright, got {l}");
    }

    #[test]
    fn lightness_monotonic() {
        let l1 = loudness_to_lightness(0.2);
        let l2 = loudness_to_lightness(0.5);
        let l3 = loudness_to_lightness(0.8);
        assert!(l1 < l2 && l2 < l3, "lightness should increase: {l1} < {l2} < {l3}");
    }

    // ── Timbre-to-Texture tests ──

    #[test]
    fn pure_tone_is_smooth() {
        let tex = timbre_to_texture(&[1.0, 0.0, 0.0, 0.0]);
        assert!(tex.roughness < 0.1, "pure tone should be smooth, got {}", tex.roughness);
        assert!(tex.warmth > 0.8, "pure tone should be warm, got {}", tex.warmth);
    }

    #[test]
    fn complex_tone_is_rough() {
        let tex = timbre_to_texture(&[0.25, 0.25, 0.25, 0.25]);
        assert!(tex.roughness > 0.5, "complex tone should be rough, got {}", tex.roughness);
        assert!(tex.grain_size < 0.5, "complex tone should have small grain, got {}", tex.grain_size);
    }

    // ── Tempo-to-Motion tests ──

    #[test]
    fn slow_tempo_low_motion() {
        assert!(tempo_to_motion(40.0) < 0.15);
    }

    #[test]
    fn fast_tempo_high_motion() {
        assert!(tempo_to_motion(250.0) > 0.7);
    }

    // ── Inverse mapping tests ──

    #[test]
    fn hue_to_pitch_red_gives_c() {
        let freq = hue_to_pitch(0.0);
        // C4 = 261.63 Hz
        assert!(
            (freq - 261.63).abs() < 5.0,
            "red (0°) should give C4 ≈ 261.63, got {freq}"
        );
    }

    #[test]
    fn hue_to_pitch_blue_gives_g() {
        let freq = hue_to_pitch(210.0);
        // G4 = 392.0 Hz
        assert!(
            (freq - 392.0).abs() < 10.0,
            "blue (210°) should give G4 ≈ 392, got {freq}"
        );
    }

    // ── Synesthetic frame extraction ──

    #[test]
    fn empty_notes_empty_frames() {
        let frames = extract_synesthetic_features(&[], 120.0, 4.0);
        assert!(frames.is_empty());
    }

    #[test]
    fn frames_cover_duration() {
        let notes = vec![
            (261.63, 0.0, 0.5, 0.8),
            (329.63, 0.5, 0.5, 0.7),
            (392.0, 1.0, 0.5, 0.9),
        ];
        let frames = extract_synesthetic_features(&notes, 120.0, 2.0);
        assert!(!frames.is_empty());
        assert!(frames.last().unwrap().time < 2.0);
    }

    #[test]
    fn high_notes_cool_colors() {
        // High frequency → blue/violet hue range (180-330°)
        let hue = pitch_to_hue(880.0); // A5
        assert!(
            hue > 250.0 && hue < 290.0,
            "high pitch should be violet, got {hue}"
        );
    }

    #[test]
    fn low_notes_warm_colors() {
        // Low frequency → red/yellow hue range (0-90°)
        let hue = pitch_to_hue(130.81); // C3
        assert!(
            hue < 15.0 || hue > 345.0,
            "low C should be near red, got {hue}"
        );
    }

    // ── Blend feedback ──

    #[test]
    fn blend_empty_is_neutral() {
        let blended = blend_feedbacks(&[]);
        assert_eq!(blended.dopamine_delta, 0.0);
    }

    #[test]
    fn blend_averages() {
        let f1 = super::super::AestheticFeedback {
            dopamine_delta: 0.1,
            serotonin_delta: 0.0,
            surprise_signal: 0.0,
            harmony_projection: [0.0; 8],
        };
        let f2 = super::super::AestheticFeedback {
            dopamine_delta: 0.3,
            serotonin_delta: 0.0,
            surprise_signal: 0.0,
            harmony_projection: [0.0; 8],
        };
        let blended = blend_feedbacks(&[f1, f2]);
        assert!((blended.dopamine_delta - 0.2).abs() < 0.001);
    }

    // ── Round-trip tests ──

    #[test]
    fn saturation_to_timbre_bounded() {
        for s in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let t = saturation_to_timbre(s);
            for &v in &t {
                assert!(v >= 0.0 && v <= 1.0, "timbre value {v} out of [0,1] for sat={s}");
            }
        }
    }

    #[test]
    fn complexity_to_density_range() {
        let low = complexity_to_density(0.0);
        let high = complexity_to_density(1.0);
        assert!(low >= 0.3);
        assert!(high <= 2.0);
        assert!(high > low);
    }
}
