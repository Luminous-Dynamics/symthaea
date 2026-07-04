// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-modal synesthesia: bidirectional audio-visual mappings.
//!
//! Implements Scriabin's pitch-to-hue correspondences and inverse mappings
//! for unified aesthetic evaluation across sensory modalities.
//!
//! # References
//!
//! - Scriabin, A. (1911). *Prometheus: The Poem of Fire*. Pitch-color associations.
//! - Cytowic, R. (2002). *Synesthesia: A Union of the Senses*. MIT Press.
//! - Ward, J. (2013). Synesthesia. *Annual Review of Psychology*, 64, 49-75.

use crate::AestheticFeedback;

/// Scriabin's pitch-class-to-hue mapping (degrees, C=0 through B=11).
///
/// C=red(0), C#=orange(30), D=yellow(60), D#=yellow-green(90),
/// E=green(120), F=cyan(150), F#=blue(180), G=blue-violet(210),
/// G#=violet(240), A=purple(270), A#=magenta(300), B=rose(330).
const SCRIABIN_HUE: [f32; 12] = [
    0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0,
];

/// Texture parameters derived from timbral analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextureParams {
    /// Spatial grain size (0.0 = smooth, 1.0 = coarse).
    pub grain_size: f32,
    /// Surface roughness (0.0 = polished, 1.0 = rough).
    pub roughness: f32,
    /// Color temperature (0.0 = cool, 1.0 = warm).
    pub warmth: f32,
}

/// A single frame of synesthetic visual data derived from audio.
#[derive(Debug, Clone)]
pub struct SynestheticFrame {
    /// Dominant hue (0-360 degrees).
    pub hue: f32,
    /// Lightness (0.0-1.0).
    pub lightness: f32,
    /// Texture derived from timbre.
    pub texture: TextureParams,
    /// Motion intensity (0.0-1.0).
    pub motion: f32,
    /// Time in seconds.
    pub time: f32,
}

// ─── Audio → Visual ─────────────────────────────────────────────────────────

/// Map a frequency (Hz) to a Scriabin hue (0-360 degrees).
///
/// Converts frequency to MIDI note, extracts pitch class, then interpolates
/// between adjacent Scriabin hues for smooth chromatic transitions.
pub fn pitch_to_hue(freq_hz: f32) -> f32 {
    if freq_hz <= 0.0 {
        return 0.0;
    }
    // MIDI note number (A4=440Hz=69)
    let midi = 12.0 * (freq_hz / 440.0).ln() / 2.0_f32.ln() + 69.0;
    // Continuous pitch class [0, 12)
    let pitch_class = ((midi % 12.0) + 12.0) % 12.0;
    let pc_floor = pitch_class.floor() as usize % 12;
    let pc_ceil = (pc_floor + 1) % 12;
    let frac = pitch_class - pitch_class.floor();

    let hue_a = SCRIABIN_HUE[pc_floor];
    let hue_b = SCRIABIN_HUE[pc_ceil];

    // Handle wraparound (330 -> 0)
    let diff = hue_b - hue_a;
    let adjusted_diff = if diff > 180.0 {
        diff - 360.0
    } else if diff < -180.0 {
        diff + 360.0
    } else {
        diff
    };

    let result = hue_a + frac * adjusted_diff;
    ((result % 360.0) + 360.0) % 360.0
}

/// Map loudness (0.0-1.0) to visual lightness (0.0-1.0).
///
/// Uses a power curve (gamma 0.8) to match perceptual brightness,
/// with a floor of 0.15 so silence is still faintly visible.
pub fn loudness_to_lightness(loudness: f32) -> f32 {
    let l = loudness.clamp(0.0, 1.0);
    (0.15 + 0.7 * l.powf(0.8)).clamp(0.0, 1.0)
}

/// Map a 4-element timbre vector (spectral centroid, spectral spread,
/// spectral flatness, spectral rolloff) to visual texture parameters.
pub fn timbre_to_texture(timbre: &[f32; 4]) -> TextureParams {
    let centroid = timbre[0].clamp(0.0, 1.0);
    let spread = timbre[1].clamp(0.0, 1.0);
    let flatness = timbre[2].clamp(0.0, 1.0);
    let rolloff = timbre[3].clamp(0.0, 1.0);

    TextureParams {
        grain_size: spread * 0.7 + flatness * 0.3,
        roughness: flatness * 0.6 + (1.0 - rolloff) * 0.4,
        warmth: (1.0 - centroid) * 0.8 + rolloff * 0.2,
    }
}

/// Map tempo (BPM) to visual motion intensity (0.0-1.0).
///
/// 30 BPM maps to 0.0, 300 BPM maps to 1.0, linear interpolation.
pub fn tempo_to_motion(bpm: f32) -> f32 {
    let clamped = bpm.clamp(30.0, 300.0);
    (clamped - 30.0) / 270.0
}

// ─── Visual → Audio ─────────────────────────────────────────────────────────

/// Map a hue (0-360 degrees) to a pitch frequency (Hz) in octave 4.
///
/// Inverse of `pitch_to_hue`: finds the bracketing Scriabin hues,
/// interpolates the fractional pitch class, converts to frequency.
pub fn hue_to_pitch(hue: f32) -> f32 {
    let h = ((hue % 360.0) + 360.0) % 360.0;

    // Find bracketing hues
    let mut best_below = 0usize;
    let mut best_dist = f32::MAX;

    for (i, &scriabin_h) in SCRIABIN_HUE.iter().enumerate() {
        // Distance going clockwise from scriabin_h to h
        let dist = ((h - scriabin_h) % 360.0 + 360.0) % 360.0;
        if dist < best_dist {
            best_dist = dist;
            best_below = i;
        }
    }

    let next = (best_below + 1) % 12;
    let hue_a = SCRIABIN_HUE[best_below];
    let hue_b = SCRIABIN_HUE[next];

    // Compute the span between these two hues (clockwise)
    let span = ((hue_b - hue_a) % 360.0 + 360.0) % 360.0;
    let frac = if span > 0.001 {
        (((h - hue_a) % 360.0 + 360.0) % 360.0) / span
    } else {
        0.0
    };

    let pitch_class = best_below as f32 + frac.clamp(0.0, 1.0);

    // Convert to frequency in octave 4 (MIDI 60-71)
    let midi = 60.0 + pitch_class;
    440.0 * 2.0_f32.powf((midi - 69.0) / 12.0)
}

/// Map saturation (0.0-1.0) to a 4-element timbre vector.
///
/// High saturation produces bright, focused timbres (high centroid, low flatness).
/// Low saturation produces dull, diffuse timbres.
pub fn saturation_to_timbre(saturation: f32) -> [f32; 4] {
    let s = saturation.clamp(0.0, 1.0);
    [
        s * 0.8 + 0.1,         // spectral centroid: saturated = bright
        (1.0 - s) * 0.6 + 0.1, // spectral spread: saturated = focused
        (1.0 - s) * 0.7,       // spectral flatness: saturated = tonal
        s * 0.7 + 0.2,         // spectral rolloff: saturated = extended
    ]
}

/// Map visual complexity (0.0-1.0) to sonic density (0.0-2.0).
///
/// Low complexity = sparse (0.3), high complexity = dense (2.0).
pub fn complexity_to_density(complexity: f32) -> f32 {
    let c = complexity.clamp(0.0, 1.0);
    0.3 + c * 1.7
}

// ─── Composite Operations ───────────────────────────────────────────────────

/// Extract synesthetic visual frames from a sequence of notes.
///
/// Each note is `(frequency_hz, velocity, start_time, duration)`.
/// Notes are grouped by beat, and per-beat features are computed.
pub fn extract_synesthetic_features(
    notes: &[(f32, f32, f32, f32)],
    tempo_bpm: f32,
    duration: f32,
) -> Vec<SynestheticFrame> {
    if notes.is_empty() || duration <= 0.0 {
        return Vec::new();
    }

    let beat_duration = 60.0 / tempo_bpm.max(1.0);
    let num_beats = (duration / beat_duration).ceil() as usize;
    let mut frames = Vec::with_capacity(num_beats);

    for beat_idx in 0..num_beats {
        let beat_start = beat_idx as f32 * beat_duration;
        let beat_end = beat_start + beat_duration;

        // Collect notes active during this beat
        let beat_notes: Vec<&(f32, f32, f32, f32)> = notes
            .iter()
            .filter(|(_, _, start, dur)| {
                let end = start + dur;
                *start < beat_end && end > beat_start
            })
            .collect();

        if beat_notes.is_empty() {
            frames.push(SynestheticFrame {
                hue: 0.0,
                lightness: 0.15,
                texture: TextureParams {
                    grain_size: 0.0,
                    roughness: 0.0,
                    warmth: 0.5,
                },
                motion: tempo_to_motion(tempo_bpm),
                time: beat_start,
            });
            continue;
        }

        // Average hue (circular mean), loudness, and derive texture
        let mut sin_sum = 0.0f32;
        let mut cos_sum = 0.0f32;
        let mut loudness_sum = 0.0f32;

        for &(freq, vel, _, _) in &beat_notes {
            let h = pitch_to_hue(*freq).to_radians();
            sin_sum += h.sin();
            cos_sum += h.cos();
            loudness_sum += vel;
        }

        let n = beat_notes.len() as f32;
        let avg_hue = sin_sum.atan2(cos_sum).to_degrees();
        let avg_hue = ((avg_hue % 360.0) + 360.0) % 360.0;
        let avg_loudness = loudness_sum / n;

        // Simple timbre estimate from frequency spread
        let freq_mean = beat_notes.iter().map(|(f, _, _, _)| f).sum::<f32>() / n;
        let freq_spread = if n > 1.0 {
            (beat_notes
                .iter()
                .map(|(f, _, _, _)| (f - freq_mean).powi(2))
                .sum::<f32>()
                / n)
                .sqrt()
                / 1000.0
        } else {
            0.0
        };
        let timbre = [
            (freq_mean / 2000.0).clamp(0.0, 1.0),
            freq_spread.clamp(0.0, 1.0),
            0.3,
            0.5,
        ];

        frames.push(SynestheticFrame {
            hue: avg_hue,
            lightness: loudness_to_lightness(avg_loudness),
            texture: timbre_to_texture(&timbre),
            motion: tempo_to_motion(tempo_bpm),
            time: beat_start,
        });
    }

    frames
}

/// Blend multiple aesthetic feedbacks into a single combined feedback.
///
/// All fields are averaged across the input feedbacks.
pub fn blend_feedbacks(feedbacks: &[AestheticFeedback]) -> AestheticFeedback {
    if feedbacks.is_empty() {
        return AestheticFeedback::neutral();
    }

    let n = feedbacks.len() as f32;
    let mut result = AestheticFeedback::neutral();

    for fb in feedbacks {
        result.dopamine_delta += fb.dopamine_delta;
        result.serotonin_delta += fb.serotonin_delta;
        result.surprise_signal += fb.surprise_signal;
        for i in 0..8 {
            result.harmony_projection[i] += fb.harmony_projection[i];
        }
    }

    result.dopamine_delta /= n;
    result.serotonin_delta /= n;
    result.surprise_signal /= n;
    for i in 0..8 {
        result.harmony_projection[i] /= n;
    }

    result
}

// ── Cross-Modal Coherence ────────────────────────────────────────────────────

/// Score cross-modal coherence between music and visual art.
///
/// Measures whether the emotional/aesthetic qualities expressed in the music
/// align with those in the visual output. High coherence = the system is
/// "emotionally honest" across modalities. Low coherence = the system is
/// producing sad music with happy visuals (or vice versa).
///
/// Returns a score in [0, 1] where 1.0 = perfectly aligned.
///
/// # Dimensions Checked
///
/// 1. **Hue-valence alignment**: warm musical hues (red/yellow, low pitch) should
///    match positive visual valence, cool hues (blue/violet, high pitch) should
///    match negative/contemplative visual qualities
/// 2. **Energy alignment**: high-tempo music (motion) should correspond to
///    high-arousal visuals (complexity, contrast)
/// 3. **Texture-timbre alignment**: rough timbres should produce rough textures,
///    pure tones should produce smooth surfaces
pub fn coherence_score(
    syn_frames: &[SynestheticFrame],
    visual_valence: f32,
    visual_arousal: f32,
    visual_complexity: f32,
) -> f32 {
    if syn_frames.is_empty() {
        return 0.5; // neutral when no music
    }

    let n = syn_frames.len() as f32;

    // Average musical features
    let avg_hue: f32 = syn_frames.iter().map(|f| f.hue).sum::<f32>() / n;
    let avg_lightness: f32 = syn_frames.iter().map(|f| f.lightness).sum::<f32>() / n;
    let avg_motion: f32 = syn_frames.iter().map(|f| f.motion).sum::<f32>() / n;
    let avg_roughness: f32 = syn_frames.iter().map(|f| f.texture.roughness).sum::<f32>() / n;

    // 1. Hue-valence alignment
    // Warm hues (0-120°) → positive valence expected
    // Cool hues (180-360°) → negative/contemplative valence expected
    let music_warmth = if avg_hue < 180.0 {
        1.0 - avg_hue / 180.0 // 1.0 at red, 0.0 at cyan
    } else {
        -(avg_hue - 180.0) / 180.0 // -1.0 at rose, 0.0 at cyan
    };
    // Visual valence in [-1, 1]: positive = warm, negative = cool
    let hue_valence_alignment = 1.0 - (music_warmth - visual_valence).abs();

    // 2. Energy alignment
    // High motion (tempo) → high visual arousal expected
    let energy_alignment = 1.0 - (avg_motion - visual_arousal).abs();

    // 3. Lightness-complexity alignment
    // Loud music (bright) → visually complex expected
    let lightness_complexity_alignment = 1.0 - (avg_lightness - visual_complexity).abs();

    // 4. Texture roughness alignment
    // This is a softer signal — not all visual styles expose roughness
    let texture_alignment = 1.0 - (avg_roughness - visual_complexity * 0.5).abs();

    // Weighted composite

    (0.35 * hue_valence_alignment
        + 0.30 * energy_alignment
        + 0.20 * lightness_complexity_alignment
        + 0.15 * texture_alignment)
        .clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c4_maps_to_red() {
        let hue = pitch_to_hue(261.63); // C4
        assert!(
            hue < 5.0 || hue > 355.0,
            "C4 should be red (near 0), got {hue}"
        );
    }

    #[test]
    fn a4_maps_to_purple() {
        let hue = pitch_to_hue(440.0); // A4
        assert!(
            (hue - 270.0).abs() < 5.0,
            "A4 should be purple (~270), got {hue}"
        );
    }

    #[test]
    fn e4_maps_to_green() {
        let hue = pitch_to_hue(329.63); // E4
        assert!(
            (hue - 120.0).abs() < 5.0,
            "E4 should be green (~120), got {hue}"
        );
    }

    #[test]
    fn pitch_to_hue_monotonic_within_octave() {
        // Within a single octave, hues should progress through the spectrum
        let freqs = [
            261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16,
            493.88,
        ];
        let hues: Vec<f32> = freqs.iter().map(|&f| pitch_to_hue(f)).collect();
        // Each should be approximately 30 degrees apart
        for i in 0..11 {
            let diff = ((hues[i + 1] - hues[i]) % 360.0 + 360.0) % 360.0;
            assert!(
                (diff - 30.0).abs() < 5.0,
                "Step {i}: expected ~30 deg, got {diff} (hues: {} -> {})",
                hues[i],
                hues[i + 1]
            );
        }
    }

    #[test]
    fn octave_equivalence() {
        let c4 = pitch_to_hue(261.63);
        let c5 = pitch_to_hue(523.25);
        let diff = (c4 - c5).abs();
        let circular_diff = diff.min(360.0 - diff);
        assert!(
            circular_diff < 2.0,
            "C4 and C5 should map to same hue, circular_diff={circular_diff}"
        );
    }

    #[test]
    fn loudness_bounds() {
        assert!(loudness_to_lightness(0.0) >= 0.14);
        assert!(loudness_to_lightness(0.0) <= 0.16);
        assert!(loudness_to_lightness(1.0) >= 0.84);
        assert!(loudness_to_lightness(1.0) <= 0.86);
        // Clamping
        assert!(loudness_to_lightness(-1.0) >= 0.0);
        assert!(loudness_to_lightness(2.0) <= 1.0);
    }

    #[test]
    fn timbre_texture_mapping() {
        let bright = timbre_to_texture(&[1.0, 0.0, 0.0, 1.0]);
        let dull = timbre_to_texture(&[0.0, 1.0, 1.0, 0.0]);
        assert!(bright.warmth < dull.warmth, "bright should be cooler");
        assert!(
            bright.roughness < dull.roughness,
            "bright should be smoother"
        );
    }

    #[test]
    fn tempo_bounds() {
        assert!((tempo_to_motion(30.0)).abs() < 0.001);
        assert!((tempo_to_motion(300.0) - 1.0).abs() < 0.001);
        assert!(tempo_to_motion(165.0) > 0.4);
        assert!(tempo_to_motion(165.0) < 0.6);
    }

    #[test]
    fn hue_to_pitch_inverse_c() {
        // Red (0 deg) -> C4
        let freq = hue_to_pitch(0.0);
        assert!(
            (freq - 261.63).abs() < 2.0,
            "hue 0 should map to C4 (~261.63), got {freq}"
        );
    }

    #[test]
    fn hue_to_pitch_inverse_a() {
        // Purple (270 deg) -> A4
        let freq = hue_to_pitch(270.0);
        assert!(
            (freq - 440.0).abs() < 5.0,
            "hue 270 should map to A4 (~440), got {freq}"
        );
    }

    #[test]
    fn hue_to_pitch_roundtrip() {
        // Test that pitch -> hue -> pitch is approximately identity
        for midi in 60..72 {
            let freq = 440.0 * 2.0_f32.powf((midi as f32 - 69.0) / 12.0);
            let hue = pitch_to_hue(freq);
            let recovered = hue_to_pitch(hue);
            assert!(
                (recovered - freq).abs() < 3.0,
                "Roundtrip failed for MIDI {midi}: {freq} -> hue {hue} -> {recovered}"
            );
        }
    }

    #[test]
    fn synesthetic_frame_extraction() {
        let notes = vec![
            (261.63, 0.8, 0.0, 0.5),  // C4 first beat
            (329.63, 0.6, 0.0, 0.5),  // E4 first beat
            (392.00, 0.7, 0.75, 0.5), // G4 second beat
        ];
        let frames = extract_synesthetic_features(&notes, 120.0, 1.5);
        assert!(!frames.is_empty());
        assert!(
            frames[0].lightness > 0.15,
            "should have non-silent first beat"
        );
    }

    #[test]
    fn synesthetic_frame_empty_input() {
        let frames = extract_synesthetic_features(&[], 120.0, 1.0);
        assert!(frames.is_empty());
    }

    #[test]
    fn blend_feedbacks_averages() {
        let fb1 = AestheticFeedback {
            dopamine_delta: 0.1,
            serotonin_delta: 0.2,
            surprise_signal: 0.3,
            harmony_projection: [1.0; 8],
        };
        let fb2 = AestheticFeedback {
            dopamine_delta: 0.3,
            serotonin_delta: 0.4,
            surprise_signal: 0.5,
            harmony_projection: [0.0; 8],
        };
        let blended = blend_feedbacks(&[fb1, fb2]);
        assert!((blended.dopamine_delta - 0.2).abs() < 0.001);
        assert!((blended.serotonin_delta - 0.3).abs() < 0.001);
        assert!((blended.surprise_signal - 0.4).abs() < 0.001);
        assert!((blended.harmony_projection[0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn blend_empty_returns_neutral() {
        let blended = blend_feedbacks(&[]);
        assert_eq!(blended.dopamine_delta, 0.0);
    }

    #[test]
    fn saturation_to_timbre_bounds() {
        let low = saturation_to_timbre(0.0);
        let high = saturation_to_timbre(1.0);
        for &v in &low {
            assert!(v >= 0.0 && v <= 1.0);
        }
        for &v in &high {
            assert!(v >= 0.0 && v <= 1.0);
        }
        // High saturation should have higher centroid
        assert!(high[0] > low[0]);
    }

    #[test]
    fn complexity_to_density_range() {
        assert!((complexity_to_density(0.0) - 0.3).abs() < 0.001);
        assert!((complexity_to_density(1.0) - 2.0).abs() < 0.001);
        assert!(complexity_to_density(0.5) > 0.3);
        assert!(complexity_to_density(0.5) < 2.0);
    }

    #[test]
    fn pitch_to_hue_negative_freq() {
        assert_eq!(pitch_to_hue(0.0), 0.0);
        assert_eq!(pitch_to_hue(-100.0), 0.0);
    }

    // ── Coherence tests ──

    #[test]
    fn warm_music_warm_visuals_coherent() {
        // Low pitch (warm hue) + positive valence = high coherence
        let frames = vec![SynestheticFrame {
            hue: 30.0, // warm (orange)
            lightness: 0.7,
            texture: TextureParams {
                grain_size: 0.5,
                roughness: 0.3,
                warmth: 0.8,
            },
            motion: 0.6,
            time: 0.0,
        }];
        let score = coherence_score(&frames, 0.7, 0.6, 0.5);
        assert!(
            score > 0.6,
            "warm music + warm visuals should be coherent: {score}"
        );
    }

    #[test]
    fn cool_music_warm_visuals_incoherent() {
        // High pitch (cool hue) + positive valence = mismatch
        let frames = vec![SynestheticFrame {
            hue: 270.0, // cool (violet)
            lightness: 0.3,
            texture: TextureParams {
                grain_size: 0.8,
                roughness: 0.7,
                warmth: 0.2,
            },
            motion: 0.2,
            time: 0.0,
        }];
        let score = coherence_score(&frames, 0.8, 0.8, 0.8);
        // Mismatch: cool music but very warm/active visuals
        assert!(
            score < 0.7,
            "cool music + hot visuals should be less coherent: {score}"
        );
    }

    #[test]
    fn no_frames_neutral() {
        let score = coherence_score(&[], 0.5, 0.5, 0.5);
        assert_eq!(score, 0.5);
    }

    #[test]
    fn coherence_bounded() {
        for hue in [0.0, 90.0, 180.0, 270.0, 359.0] {
            for val in [-1.0, -0.5, 0.0, 0.5, 1.0] {
                let frames = vec![SynestheticFrame {
                    hue,
                    lightness: 0.5,
                    texture: TextureParams {
                        grain_size: 0.5,
                        roughness: 0.5,
                        warmth: 0.5,
                    },
                    motion: 0.5,
                    time: 0.0,
                }];
                let score = coherence_score(&frames, val, 0.5, 0.5);
                assert!(
                    score >= 0.0 && score <= 1.0,
                    "hue={hue}, val={val}: coherence={score}"
                );
            }
        }
    }

    #[test]
    fn high_energy_alignment() {
        // Fast tempo (high motion) should align with high arousal visuals
        let frames = vec![SynestheticFrame {
            hue: 60.0,
            lightness: 0.8,
            texture: TextureParams {
                grain_size: 0.3,
                roughness: 0.2,
                warmth: 0.6,
            },
            motion: 0.9, // very fast
            time: 0.0,
        }];
        let high_arousal = coherence_score(&frames, 0.3, 0.9, 0.7);
        let low_arousal = coherence_score(&frames, 0.3, 0.1, 0.2);
        assert!(
            high_arousal > low_arousal,
            "fast music should align better with high arousal: {high_arousal} vs {low_arousal}"
        );
    }
}
