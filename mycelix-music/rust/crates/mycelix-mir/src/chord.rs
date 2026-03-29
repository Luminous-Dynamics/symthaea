// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Chord detection and key estimation

use crate::{Chord, ChordQuality, Chromagram, Key, NoteName, Result};

/// Chord templates for matching
struct ChordTemplate {
    quality: ChordQuality,
    intervals: Vec<u8>,
    weights: Vec<f32>,
}

impl ChordTemplate {
    fn major() -> Self {
        Self {
            quality: ChordQuality::Major,
            intervals: vec![0, 4, 7],
            weights: vec![1.0, 0.8, 0.8],
        }
    }

    fn minor() -> Self {
        Self {
            quality: ChordQuality::Minor,
            intervals: vec![0, 3, 7],
            weights: vec![1.0, 0.8, 0.8],
        }
    }

    fn dominant7() -> Self {
        Self {
            quality: ChordQuality::Dominant7,
            intervals: vec![0, 4, 7, 10],
            weights: vec![1.0, 0.7, 0.7, 0.6],
        }
    }

    fn major7() -> Self {
        Self {
            quality: ChordQuality::Major7,
            intervals: vec![0, 4, 7, 11],
            weights: vec![1.0, 0.7, 0.7, 0.6],
        }
    }

    fn minor7() -> Self {
        Self {
            quality: ChordQuality::Minor7,
            intervals: vec![0, 3, 7, 10],
            weights: vec![1.0, 0.7, 0.7, 0.6],
        }
    }

    fn diminished() -> Self {
        Self {
            quality: ChordQuality::Diminished,
            intervals: vec![0, 3, 6],
            weights: vec![1.0, 0.8, 0.8],
        }
    }

    fn augmented() -> Self {
        Self {
            quality: ChordQuality::Augmented,
            intervals: vec![0, 4, 8],
            weights: vec![1.0, 0.8, 0.8],
        }
    }

    fn sus4() -> Self {
        Self {
            quality: ChordQuality::Sus4,
            intervals: vec![0, 5, 7],
            weights: vec![1.0, 0.8, 0.8],
        }
    }

    fn sus2() -> Self {
        Self {
            quality: ChordQuality::Sus2,
            intervals: vec![0, 2, 7],
            weights: vec![1.0, 0.8, 0.8],
        }
    }

    /// Generate template chroma for a given root
    fn to_chroma(&self, root: u8) -> [f32; 12] {
        let mut chroma = [0.0f32; 12];
        for (&interval, &weight) in self.intervals.iter().zip(self.weights.iter()) {
            let pc = (root + interval) % 12;
            chroma[pc as usize] = weight;
        }

        // Normalize
        let sum: f32 = chroma.iter().sum();
        if sum > 0.0 {
            for c in &mut chroma {
                *c /= sum;
            }
        }

        chroma
    }
}

/// Chord detector
pub struct ChordDetector {
    templates: Vec<ChordTemplate>,
    smoothing_window: usize,
}

impl ChordDetector {
    pub fn new() -> Self {
        Self {
            templates: vec![
                ChordTemplate::major(),
                ChordTemplate::minor(),
                ChordTemplate::dominant7(),
                ChordTemplate::major7(),
                ChordTemplate::minor7(),
                ChordTemplate::diminished(),
                ChordTemplate::augmented(),
                ChordTemplate::sus4(),
                ChordTemplate::sus2(),
            ],
            smoothing_window: 5,
        }
    }

    /// Detect chord at a single frame
    pub fn detect_frame(&self, chroma: &[f32; 12]) -> Chord {
        let mut best_chord = Chord::new(NoteName::C, ChordQuality::Unknown);
        let mut best_score = -1.0f32;

        for root in 0..12u8 {
            for template in &self.templates {
                let template_chroma = template.to_chroma(root);
                let score = self.correlation(chroma, &template_chroma);

                if score > best_score {
                    best_score = score;
                    best_chord = Chord {
                        root: NoteName::from_pitch_class(root),
                        quality: template.quality,
                        bass: None,
                        confidence: score,
                    };
                }
            }
        }

        // Detect bass note (lowest energy pitch class)
        let bass_pc = self.detect_bass(chroma);
        if bass_pc != best_chord.root.to_pitch_class() {
            best_chord.bass = Some(NoteName::from_pitch_class(bass_pc));
        }

        best_chord
    }

    /// Detect chord sequence from chromagram
    pub fn detect_sequence(&self, chromagram: &Chromagram) -> Vec<(f32, Chord)> {
        let num_frames = chromagram.num_frames();
        if num_frames == 0 {
            return vec![];
        }

        // Detect chord for each frame
        let mut frame_chords: Vec<Chord> = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            let chroma = chromagram.get_frame(i);
            frame_chords.push(self.detect_frame(&chroma));
        }

        // Apply smoothing / HMM (simplified: just median filtering)
        let smoothed = self.smooth_sequence(&frame_chords);

        // Convert to timed chord list (merge consecutive same chords)
        let mut result: Vec<(f32, Chord)> = Vec::new();

        for (i, chord) in smoothed.into_iter().enumerate() {
            let time = chromagram.frame_to_time(i);

            if result.is_empty() {
                result.push((time, chord));
            } else {
                let last = result.last().unwrap();
                // Check if same chord
                if last.1.root == chord.root && last.1.quality == chord.quality {
                    // Same chord, don't add
                    continue;
                }
                result.push((time, chord));
            }
        }

        result
    }

    fn correlation(&self, a: &[f32; 12], b: &[f32; 12]) -> f32 {
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        for i in 0..12 {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        let denom = (norm_a * norm_b).sqrt();
        if denom > 1e-6 {
            dot / denom
        } else {
            0.0
        }
    }

    fn detect_bass(&self, chroma: &[f32; 12]) -> u8 {
        // In a full implementation, this would use low-frequency analysis
        // For now, just find the strongest pitch class
        chroma
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u8)
            .unwrap_or(0)
    }

    fn smooth_sequence(&self, chords: &[Chord]) -> Vec<Chord> {
        if chords.len() <= self.smoothing_window {
            return chords.to_vec();
        }

        let mut smoothed = Vec::with_capacity(chords.len());
        let half_window = self.smoothing_window / 2;

        for i in 0..chords.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(chords.len());
            let window = &chords[start..end];

            // Vote for most common chord in window
            let mut votes: std::collections::HashMap<(u8, u8), usize> = std::collections::HashMap::new();
            for chord in window {
                let key = (chord.root.to_pitch_class(), chord.quality as u8);
                *votes.entry(key).or_insert(0) += 1;
            }

            let winner = votes
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|((root, quality_idx), _)| {
                    let quality = match quality_idx {
                        0 => ChordQuality::Major,
                        1 => ChordQuality::Minor,
                        2 => ChordQuality::Dominant7,
                        3 => ChordQuality::Major7,
                        4 => ChordQuality::Minor7,
                        5 => ChordQuality::Diminished,
                        6 => ChordQuality::Augmented,
                        7 => ChordQuality::Sus4,
                        8 => ChordQuality::Sus2,
                        _ => ChordQuality::Unknown,
                    };
                    Chord::new(NoteName::from_pitch_class(root), quality)
                })
                .unwrap_or_else(|| chords[i].clone());

            smoothed.push(winner);
        }

        smoothed
    }
}

impl Default for ChordDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Key detector using Krumhansl-Kessler profiles
pub struct KeyDetector {
    major_profile: [f32; 12],
    minor_profile: [f32; 12],
}

impl KeyDetector {
    pub fn new() -> Self {
        // Krumhansl-Kessler key profiles
        Self {
            major_profile: [
                6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
            ],
            minor_profile: [
                6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
            ],
        }
    }

    /// Detect key from chromagram
    pub fn detect(&self, chromagram: &Chromagram) -> Key {
        // Average chroma across all frames
        let mut avg_chroma = [0.0f32; 12];
        let num_frames = chromagram.num_frames();

        if num_frames == 0 {
            return Key {
                tonic: NoteName::C,
                is_major: true,
                confidence: 0.0,
            };
        }

        for i in 0..num_frames {
            let frame = chromagram.get_frame(i);
            for (j, &c) in frame.iter().enumerate() {
                avg_chroma[j] += c;
            }
        }

        for c in &mut avg_chroma {
            *c /= num_frames as f32;
        }

        self.detect_from_chroma(&avg_chroma)
    }

    /// Detect key from single chroma vector
    pub fn detect_from_chroma(&self, chroma: &[f32; 12]) -> Key {
        let mut best_key = Key {
            tonic: NoteName::C,
            is_major: true,
            confidence: 0.0,
        };
        let mut best_correlation = -1.0f32;

        // Test all keys
        for root in 0..12u8 {
            // Major key
            let major_corr = self.correlate_profile(chroma, &self.major_profile, root);
            if major_corr > best_correlation {
                best_correlation = major_corr;
                best_key = Key {
                    tonic: NoteName::from_pitch_class(root),
                    is_major: true,
                    confidence: major_corr,
                };
            }

            // Minor key
            let minor_corr = self.correlate_profile(chroma, &self.minor_profile, root);
            if minor_corr > best_correlation {
                best_correlation = minor_corr;
                best_key = Key {
                    tonic: NoteName::from_pitch_class(root),
                    is_major: false,
                    confidence: minor_corr,
                };
            }
        }

        best_key
    }

    fn correlate_profile(&self, chroma: &[f32; 12], profile: &[f32; 12], root: u8) -> f32 {
        // Rotate profile to match root
        let mut rotated_profile = [0.0f32; 12];
        for i in 0..12 {
            rotated_profile[i] = profile[(12 + i - root as usize) % 12];
        }

        // Normalize profile
        let profile_mean: f32 = rotated_profile.iter().sum::<f32>() / 12.0;
        let chroma_mean: f32 = chroma.iter().sum::<f32>() / 12.0;

        // Pearson correlation
        let mut num = 0.0;
        let mut denom_a = 0.0;
        let mut denom_b = 0.0;

        for i in 0..12 {
            let a = chroma[i] - chroma_mean;
            let b = rotated_profile[i] - profile_mean;
            num += a * b;
            denom_a += a * a;
            denom_b += b * b;
        }

        let denom = (denom_a * denom_b).sqrt();
        if denom > 1e-6 {
            num / denom
        } else {
            0.0
        }
    }
}

impl Default for KeyDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Chord sequence analyzer for harmony analysis
pub struct HarmonyAnalyzer {
    key_detector: KeyDetector,
}

impl HarmonyAnalyzer {
    pub fn new() -> Self {
        Self {
            key_detector: KeyDetector::new(),
        }
    }

    /// Analyze chord progression relative to key
    pub fn analyze_progression(&self, chords: &[(f32, Chord)], key: &Key) -> Vec<RomanNumeral> {
        let tonic_pc = key.tonic.to_pitch_class();

        chords
            .iter()
            .map(|(_, chord)| {
                let interval = (12 + chord.root.to_pitch_class() - tonic_pc) % 12;
                RomanNumeral::from_interval(interval, chord.quality, key.is_major)
            })
            .collect()
    }
}

impl Default for HarmonyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Roman numeral chord representation
#[derive(Debug, Clone)]
pub struct RomanNumeral {
    pub degree: u8,
    pub quality: ChordQuality,
    pub symbol: String,
}

impl RomanNumeral {
    fn from_interval(interval: u8, quality: ChordQuality, is_major_key: bool) -> Self {
        let degree = match interval {
            0 => 1,
            2 => 2,
            3 | 4 => 3,
            5 => 4,
            7 => 5,
            8 | 9 => 6,
            10 | 11 => 7,
            _ => 1,
        };

        let numeral = match degree {
            1 => "I",
            2 => "II",
            3 => "III",
            4 => "IV",
            5 => "V",
            6 => "VI",
            7 => "VII",
            _ => "?",
        };

        let symbol = match quality {
            ChordQuality::Major => numeral.to_string(),
            ChordQuality::Minor => numeral.to_lowercase(),
            ChordQuality::Diminished => format!("{}°", numeral.to_lowercase()),
            ChordQuality::Augmented => format!("{}+", numeral),
            ChordQuality::Dominant7 => format!("{}7", numeral),
            ChordQuality::Major7 => format!("{}maj7", numeral),
            ChordQuality::Minor7 => format!("{}7", numeral.to_lowercase()),
            _ => format!("{}{}", numeral, quality.symbol()),
        };

        Self {
            degree,
            quality,
            symbol,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chord_template() {
        let template = ChordTemplate::major();
        let chroma = template.to_chroma(0); // C major

        // Should have energy at C, E, G (0, 4, 7)
        assert!(chroma[0] > 0.0);
        assert!(chroma[4] > 0.0);
        assert!(chroma[7] > 0.0);
    }

    #[test]
    fn test_chord_detector() {
        let detector = ChordDetector::new();

        // Create C major chroma
        let mut chroma = [0.0f32; 12];
        chroma[0] = 1.0; // C
        chroma[4] = 0.8; // E
        chroma[7] = 0.8; // G

        let chord = detector.detect_frame(&chroma);
        assert_eq!(chord.root, NoteName::C);
        assert_eq!(chord.quality, ChordQuality::Major);
    }

    #[test]
    fn test_key_detector() {
        let detector = KeyDetector::new();

        // C major scale emphasis
        let chroma = [1.0, 0.0, 0.5, 0.0, 0.7, 0.5, 0.0, 0.8, 0.0, 0.4, 0.0, 0.3];
        let key = detector.detect_from_chroma(&chroma);

        // Should detect C major or A minor (relative)
        assert!(key.tonic == NoteName::C || key.tonic == NoteName::A);
    }
}
