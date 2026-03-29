// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Song structure segmentation

use crate::{Chromagram, Result, Segment, SegmentType};
use std::collections::HashMap;

/// Structure analyzer using self-similarity matrices
pub struct StructureAnalyzer {
    sample_rate: u32,
    min_segment_length: f32,
    feature_window: usize,
}

impl StructureAnalyzer {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            min_segment_length: 4.0, // Minimum 4 seconds
            feature_window: 8,       // Frames for feature averaging
        }
    }

    pub fn with_min_segment_length(mut self, seconds: f32) -> Self {
        self.min_segment_length = seconds;
        self
    }

    /// Analyze song structure
    pub fn analyze(&self, audio: &[f32], chromagram: &Chromagram) -> Vec<Segment> {
        let num_frames = chromagram.num_frames();
        if num_frames < 10 {
            return vec![];
        }

        // Build feature vectors
        let features = self.build_features(audio, chromagram);

        // Compute self-similarity matrix
        let ssm = self.compute_ssm(&features);

        // Find segment boundaries using novelty curve
        let boundaries = self.detect_boundaries(&ssm);

        // Label segments based on similarity
        let labeled = self.label_segments(&ssm, &boundaries, chromagram);

        // Convert to Segment structs
        self.create_segments(&labeled, chromagram)
    }

    fn build_features(&self, audio: &[f32], chromagram: &Chromagram) -> Vec<Vec<f32>> {
        let num_frames = chromagram.num_frames();
        let mut features = Vec::with_capacity(num_frames);

        for i in 0..num_frames {
            let mut feature = Vec::with_capacity(24); // Chroma + timbre

            // Add chroma features
            let chroma = chromagram.get_frame(i);
            feature.extend_from_slice(&chroma);

            // Add simple spectral features
            let start = i * 512;
            let end = (start + 2048).min(audio.len());
            if end > start {
                let frame = &audio[start..end];

                // RMS energy
                let rms: f32 = (frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32).sqrt();
                feature.push(rms);

                // Zero crossing rate
                let zcr = frame
                    .windows(2)
                    .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                    .count() as f32
                    / frame.len() as f32;
                feature.push(zcr);

                // Spectral centroid approximation
                let sum_energy: f32 = frame.iter().map(|x| x.abs()).sum();
                if sum_energy > 0.0 {
                    let weighted_sum: f32 = frame
                        .iter()
                        .enumerate()
                        .map(|(i, x)| i as f32 * x.abs())
                        .sum();
                    let centroid = weighted_sum / sum_energy / frame.len() as f32;
                    feature.push(centroid);
                } else {
                    feature.push(0.0);
                }
            } else {
                feature.extend_from_slice(&[0.0, 0.0, 0.0]);
            }

            features.push(feature);
        }

        features
    }

    fn compute_ssm(&self, features: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = features.len();
        let mut ssm = vec![vec![0.0f32; n]; n];

        for i in 0..n {
            for j in i..n {
                let sim = cosine_similarity(&features[i], &features[j]);
                ssm[i][j] = sim;
                ssm[j][i] = sim;
            }
        }

        ssm
    }

    fn detect_boundaries(&self, ssm: &[Vec<f32>]) -> Vec<usize> {
        let n = ssm.len();
        if n < 10 {
            return vec![0, n - 1];
        }

        // Compute novelty curve using checkerboard kernel
        let kernel_size = 8;
        let mut novelty = vec![0.0f32; n];

        for i in kernel_size..n - kernel_size {
            // Compare quadrants around diagonal
            let mut on_diag = 0.0;
            let mut off_diag = 0.0;
            let mut count = 0;

            for di in 0..kernel_size {
                for dj in 0..kernel_size {
                    // Upper-left and lower-right (same segment)
                    on_diag += ssm[i - di - 1][i - dj - 1];
                    on_diag += ssm[i + di][i + dj];

                    // Upper-right and lower-left (different segments)
                    off_diag += ssm[i - di - 1][i + dj];
                    off_diag += ssm[i + di][i - dj - 1];

                    count += 4;
                }
            }

            novelty[i] = on_diag / count as f32 - off_diag / count as f32;
        }

        // Peak picking on novelty curve
        let mut boundaries = vec![0];
        let threshold = self.compute_adaptive_threshold(&novelty);

        for i in 1..n - 1 {
            if novelty[i] > threshold && novelty[i] > novelty[i - 1] && novelty[i] > novelty[i + 1]
            {
                // Check minimum segment length
                if let Some(&last) = boundaries.last() {
                    let min_frames = (self.min_segment_length * self.sample_rate as f32 / 512.0) as usize;
                    if i - last >= min_frames {
                        boundaries.push(i);
                    }
                }
            }
        }

        boundaries.push(n - 1);
        boundaries
    }

    fn compute_adaptive_threshold(&self, novelty: &[f32]) -> f32 {
        let mean: f32 = novelty.iter().sum::<f32>() / novelty.len() as f32;
        let variance: f32 =
            novelty.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / novelty.len() as f32;
        mean + variance.sqrt() * 0.5
    }

    fn label_segments(
        &self,
        ssm: &[Vec<f32>],
        boundaries: &[usize],
        chromagram: &Chromagram,
    ) -> Vec<(usize, usize, char, f32)> {
        if boundaries.len() < 2 {
            return vec![];
        }

        let mut labeled = Vec::new();
        let mut label_assignments: HashMap<usize, char> = HashMap::new();
        let mut next_label = 'A';

        // Compare each segment to previous segments
        for i in 0..boundaries.len() - 1 {
            let start = boundaries[i];
            let end = boundaries[i + 1];

            // Compute average similarity to each previous segment
            let mut best_match: Option<usize> = None;
            let mut best_sim = 0.6; // Threshold for considering segments similar

            for j in 0..i {
                let prev_start = boundaries[j];
                let prev_end = boundaries[j + 1];

                let sim = self.segment_similarity(ssm, start, end, prev_start, prev_end);
                if sim > best_sim {
                    best_sim = sim;
                    best_match = Some(j);
                }
            }

            let label = if let Some(match_idx) = best_match {
                *label_assignments.get(&match_idx).unwrap_or(&next_label)
            } else {
                let l = next_label;
                next_label = (next_label as u8 + 1) as char;
                l
            };

            label_assignments.insert(i, label);
            labeled.push((start, end, label, best_sim));
        }

        labeled
    }

    fn segment_similarity(
        &self,
        ssm: &[Vec<f32>],
        start1: usize,
        end1: usize,
        start2: usize,
        end2: usize,
    ) -> f32 {
        let mut sum = 0.0;
        let mut count = 0;

        for i in start1..end1 {
            for j in start2..end2 {
                sum += ssm[i][j];
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f32
        } else {
            0.0
        }
    }

    fn create_segments(
        &self,
        labeled: &[(usize, usize, char, f32)],
        chromagram: &Chromagram,
    ) -> Vec<Segment> {
        let mut segments = Vec::new();
        let mut label_counts: HashMap<char, usize> = HashMap::new();

        for &(start, end, label, confidence) in labeled {
            *label_counts.entry(label).or_insert(0) += 1;
            let occurrence = label_counts[&label];

            let segment_type = self.classify_segment_type(labeled, label, occurrence);
            let label_str = if occurrence == 1 {
                format!("{}", label)
            } else {
                format!("{}{}", label, occurrence)
            };

            segments.push(Segment {
                start: chromagram.frame_to_time(start),
                end: chromagram.frame_to_time(end),
                segment_type,
                label: label_str,
                confidence,
            });
        }

        segments
    }

    fn classify_segment_type(
        &self,
        labeled: &[(usize, usize, char, f32)],
        label: char,
        occurrence: usize,
    ) -> SegmentType {
        // Count total occurrences of this label
        let total_occurrences = labeled.iter().filter(|(_, _, l, _)| *l == label).count();

        // Heuristic classification
        let position = labeled
            .iter()
            .position(|(_, _, l, _)| *l == label)
            .unwrap_or(0);

        let total_segments = labeled.len();

        // First segment is likely intro
        if position == 0 && total_segments > 3 {
            return SegmentType::Intro;
        }

        // Last segment is likely outro
        if position == total_segments - 1 && total_segments > 3 {
            return SegmentType::Outro;
        }

        // Most repeated section is likely chorus
        if total_occurrences >= 3 {
            return SegmentType::Chorus;
        }

        // Second most repeated is likely verse
        if total_occurrences == 2 {
            return SegmentType::Verse;
        }

        // Single occurrence in middle could be bridge
        if total_occurrences == 1 && position > total_segments / 3 {
            return SegmentType::Bridge;
        }

        SegmentType::Unknown
    }
}

impl Default for StructureAnalyzer {
    fn default() -> Self {
        Self::new(44100)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom > 1e-6 {
        dot / denom
    } else {
        0.0
    }
}

/// Repetition detector for finding recurring patterns
pub struct RepetitionDetector {
    min_pattern_length: usize,
    max_pattern_length: usize,
}

impl RepetitionDetector {
    pub fn new() -> Self {
        Self {
            min_pattern_length: 16,
            max_pattern_length: 128,
        }
    }

    /// Find repeating patterns in features
    pub fn find_patterns(&self, features: &[Vec<f32>]) -> Vec<Pattern> {
        let n = features.len();
        let mut patterns = Vec::new();

        // Compute diagonal lag matrix
        for lag in self.min_pattern_length..self.max_pattern_length.min(n / 2) {
            let mut matches = Vec::new();
            let mut in_match = false;
            let mut match_start = 0;

            for i in 0..n - lag {
                let sim = cosine_similarity(&features[i], &features[i + lag]);

                if sim > 0.8 {
                    if !in_match {
                        in_match = true;
                        match_start = i;
                    }
                } else if in_match {
                    in_match = false;
                    let length = i - match_start;
                    if length >= self.min_pattern_length / 2 {
                        matches.push((match_start, length, lag));
                    }
                }
            }

            // Merge overlapping matches
            for (start, length, lag) in matches {
                if length >= self.min_pattern_length / 2 {
                    patterns.push(Pattern {
                        start_frame: start,
                        length_frames: length,
                        repeat_frame: start + lag,
                        similarity: 0.8,
                    });
                }
            }
        }

        patterns
    }
}

impl Default for RepetitionDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct Pattern {
    pub start_frame: usize,
    pub length_frames: usize,
    pub repeat_frame: usize,
    pub similarity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }

    #[test]
    fn test_structure_analyzer() {
        let analyzer = StructureAnalyzer::new(44100);
        // Would need actual audio for meaningful test
        assert_eq!(analyzer.min_segment_length, 4.0);
    }
}
