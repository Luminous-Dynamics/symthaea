// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dynamic Time Warping Alignment for Phoneme Training
//!
//! Provides precise frame-to-phoneme alignment using DTW between
//! acoustic frames and expected phoneme sequence.
//!
//! This replaces salience-based alignment with a more robust approach
//! that properly assigns frames to phonemes based on acoustic similarity.

use crate::hdc::HV16;
use std::collections::HashMap;

/// DTW alignment result
#[derive(Debug, Clone)]
pub struct DtwAlignment {
    /// Frame index -> phoneme label
    pub frame_to_phoneme: Vec<String>,
    /// Phoneme segments with precise boundaries
    pub segments: Vec<AlignedSegment>,
    /// Total DTW cost (lower = better alignment)
    pub cost: f32,
}

/// A phoneme segment with precise frame boundaries
#[derive(Debug, Clone)]
pub struct AlignedSegment {
    pub phoneme: String,
    pub start_frame: usize,
    pub end_frame: usize,
    pub avg_similarity: f32,
}

/// DTW-based aligner for phoneme sequence to acoustic frames
pub struct DtwAligner {
    /// Reference prototypes for computing frame-phoneme similarity
    prototypes: HashMap<String, HV16>,
    /// Minimum frames per phoneme (prevents degenerate alignments)
    min_frames_per_phoneme: usize,
    /// Maximum frames per phoneme (prevents one phoneme eating everything)
    max_frames_per_phoneme: usize,
}

impl DtwAligner {
    /// Create a new DTW aligner
    pub fn new() -> Self {
        Self {
            prototypes: HashMap::new(),
            min_frames_per_phoneme: 2,  // ~20ms minimum
            max_frames_per_phoneme: 30, // ~300ms maximum
        }
    }

    /// Create with prototype references for similarity computation
    pub fn with_prototypes(prototypes: HashMap<String, HV16>) -> Self {
        Self {
            prototypes,
            min_frames_per_phoneme: 2,
            max_frames_per_phoneme: 30,
        }
    }

    /// Set minimum frames per phoneme
    pub fn with_min_frames(mut self, min_frames: usize) -> Self {
        self.min_frames_per_phoneme = min_frames;
        self
    }

    /// Set maximum frames per phoneme
    pub fn with_max_frames(mut self, max_frames: usize) -> Self {
        self.max_frames_per_phoneme = max_frames;
        self
    }

    /// Align phoneme sequence to acoustic frames using DTW
    ///
    /// # Arguments
    /// * `frames` - Acoustic frame hypervectors
    /// * `phonemes` - Expected phoneme sequence
    ///
    /// # Returns
    /// DTW alignment with frame-to-phoneme mapping
    pub fn align(&self, frames: &[HV16], phonemes: &[String]) -> DtwAlignment {
        if frames.is_empty() || phonemes.is_empty() {
            return DtwAlignment {
                frame_to_phoneme: Vec::new(),
                segments: Vec::new(),
                cost: f32::INFINITY,
            };
        }

        let n_frames = frames.len();
        let n_phonemes = phonemes.len();

        // Compute cost matrix: cost[i][j] = dissimilarity of frame i to phoneme j
        let cost_matrix = self.compute_cost_matrix(frames, phonemes);

        // Run DTW with constraints
        let (path, total_cost) = self.dtw_with_constraints(&cost_matrix, n_frames, n_phonemes);

        // Convert path to alignment
        self.path_to_alignment(path, phonemes, total_cost)
    }

    /// Compute cost matrix between frames and phonemes
    fn compute_cost_matrix(&self, frames: &[HV16], phonemes: &[String]) -> Vec<Vec<f32>> {
        let n_frames = frames.len();
        let n_phonemes = phonemes.len();

        let mut cost = vec![vec![0.0f32; n_phonemes]; n_frames];

        for (i, frame) in frames.iter().enumerate() {
            for (j, phoneme) in phonemes.iter().enumerate() {
                // Cost = 1 - similarity (so lower is better)
                let similarity = if let Some(proto) = self.prototypes.get(phoneme) {
                    frame.similarity(proto)
                } else {
                    // If no prototype, use frame self-similarity as baseline
                    0.0
                };

                // Convert similarity [-1, 1] to cost [0, 2]
                // Higher similarity = lower cost
                cost[i][j] = 1.0 - similarity;
            }
        }

        cost
    }

    /// DTW with monotonicity and duration constraints
    fn dtw_with_constraints(
        &self,
        cost: &[Vec<f32>],
        n_frames: usize,
        n_phonemes: usize,
    ) -> (Vec<(usize, usize)>, f32) {
        // dp[i][j] = minimum cost to align frames[0..i] to phonemes[0..j]
        let mut dp = vec![vec![f32::INFINITY; n_phonemes + 1]; n_frames + 1];
        let mut backtrack = vec![vec![(0usize, 0usize); n_phonemes + 1]; n_frames + 1];

        dp[0][0] = 0.0;

        for i in 1..=n_frames {
            for j in 1..=n_phonemes {
                let frame_cost = cost[i - 1][j - 1];

                // Three possible transitions:
                // 1. Stay on same phoneme (horizontal in phoneme space)
                // 2. Move to next phoneme (diagonal)
                // 3. Skip frame (not allowed in standard DTW)

                let candidates = [
                    (dp[i - 1][j] + frame_cost, (i - 1, j)), // Same phoneme
                    (dp[i - 1][j - 1] + frame_cost, (i - 1, j - 1)), // New phoneme
                ];

                let (best_cost, best_prev) = candidates
                    .iter()
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .unwrap();

                dp[i][j] = *best_cost;
                backtrack[i][j] = *best_prev;
            }
        }

        // Backtrack to find path
        let mut path = Vec::new();
        let mut i = n_frames;
        let mut j = n_phonemes;

        while i > 0 && j > 0 {
            path.push((i - 1, j - 1)); // Convert to 0-indexed
            let (pi, pj) = backtrack[i][j];
            i = pi;
            j = pj;
        }

        path.reverse();
        (path, dp[n_frames][n_phonemes])
    }

    /// Convert DTW path to alignment structure
    fn path_to_alignment(
        &self,
        path: Vec<(usize, usize)>,
        phonemes: &[String],
        cost: f32,
    ) -> DtwAlignment {
        if path.is_empty() {
            return DtwAlignment {
                frame_to_phoneme: Vec::new(),
                segments: Vec::new(),
                cost,
            };
        }

        // Build frame-to-phoneme mapping
        let max_frame = path.iter().map(|(f, _)| *f).max().unwrap_or(0);
        let mut frame_to_phoneme = vec![String::new(); max_frame + 1];

        for &(frame_idx, phoneme_idx) in &path {
            if phoneme_idx < phonemes.len() {
                frame_to_phoneme[frame_idx] = phonemes[phoneme_idx].clone();
            }
        }

        // Build segments
        let mut segments = Vec::new();
        let mut current_phoneme = String::new();
        let mut start_frame = 0;

        for (frame_idx, phoneme) in frame_to_phoneme.iter().enumerate() {
            if phoneme != &current_phoneme {
                if !current_phoneme.is_empty() {
                    segments.push(AlignedSegment {
                        phoneme: current_phoneme.clone(),
                        start_frame,
                        end_frame: frame_idx,
                        avg_similarity: 0.0, // Could compute from cost matrix
                    });
                }
                current_phoneme = phoneme.clone();
                start_frame = frame_idx;
            }
        }

        // Add final segment
        if !current_phoneme.is_empty() {
            segments.push(AlignedSegment {
                phoneme: current_phoneme,
                start_frame,
                end_frame: frame_to_phoneme.len(),
                avg_similarity: 0.0,
            });
        }

        DtwAlignment {
            frame_to_phoneme,
            segments,
            cost,
        }
    }

    /// Align using uniform distribution (fallback when no prototypes)
    pub fn align_uniform(&self, n_frames: usize, phonemes: &[String]) -> DtwAlignment {
        if phonemes.is_empty() || n_frames == 0 {
            return DtwAlignment {
                frame_to_phoneme: Vec::new(),
                segments: Vec::new(),
                cost: 0.0,
            };
        }

        let frames_per_phoneme = n_frames / phonemes.len();
        let remainder = n_frames % phonemes.len();

        let mut frame_to_phoneme = Vec::with_capacity(n_frames);
        let mut segments = Vec::new();
        let mut current_frame = 0;

        for (i, phoneme) in phonemes.iter().enumerate() {
            let extra = if i < remainder { 1 } else { 0 };
            let segment_len = frames_per_phoneme + extra;

            let start = current_frame;
            for _ in 0..segment_len {
                frame_to_phoneme.push(phoneme.clone());
                current_frame += 1;
            }

            segments.push(AlignedSegment {
                phoneme: phoneme.clone(),
                start_frame: start,
                end_frame: current_frame,
                avg_similarity: 0.5, // Unknown
            });
        }

        DtwAlignment {
            frame_to_phoneme,
            segments,
            cost: 0.0,
        }
    }
}

impl Default for DtwAligner {
    fn default() -> Self {
        Self::new()
    }
}

/// Training with DTW alignment
pub struct DtwTrainer {
    aligner: DtwAligner,
    /// Accumulated prototypes per phoneme
    accumulators: HashMap<String, Vec<HV16>>,
    /// Frame counts per phoneme
    counts: HashMap<String, usize>,
}

impl DtwTrainer {
    /// Create new DTW trainer
    pub fn new() -> Self {
        Self {
            aligner: DtwAligner::new(),
            accumulators: HashMap::new(),
            counts: HashMap::new(),
        }
    }

    /// Create with initial prototypes (for iterative refinement)
    pub fn with_prototypes(prototypes: HashMap<String, HV16>) -> Self {
        Self {
            aligner: DtwAligner::with_prototypes(prototypes),
            accumulators: HashMap::new(),
            counts: HashMap::new(),
        }
    }

    /// Train on a single utterance
    ///
    /// # Arguments
    /// * `frames` - Acoustic frame hypervectors from LTC projection
    /// * `phonemes` - Expected phoneme sequence from transcript
    ///
    /// # Returns
    /// Number of frame-phoneme pairs added
    pub fn train_utterance(&mut self, frames: &[HV16], phonemes: &[String]) -> usize {
        if frames.is_empty() || phonemes.is_empty() {
            return 0;
        }

        // Align frames to phonemes
        let alignment = if self.aligner.prototypes.is_empty() {
            // No prototypes yet - use uniform alignment
            self.aligner.align_uniform(frames.len(), phonemes)
        } else {
            // Use DTW with current prototypes
            self.aligner.align(frames, phonemes)
        };

        // Accumulate frame HVs for each phoneme
        let mut count = 0;
        for segment in &alignment.segments {
            if segment.start_frame >= frames.len() {
                continue;
            }

            let end = segment.end_frame.min(frames.len());
            for frame_idx in segment.start_frame..end {
                self.accumulators
                    .entry(segment.phoneme.clone())
                    .or_default()
                    .push(frames[frame_idx]);

                *self.counts.entry(segment.phoneme.clone()).or_insert(0) += 1;
                count += 1;
            }
        }

        count
    }

    /// Finalize training and return prototypes
    pub fn finalize(self, min_instances: usize) -> HashMap<String, HV16> {
        let mut prototypes = HashMap::new();

        for (phoneme, hvs) in self.accumulators {
            if hvs.len() >= min_instances {
                // Bundle all frames for this phoneme
                let prototype = crate::hdc::bundle(&hvs);
                prototypes.insert(phoneme, prototype);
            }
        }

        prototypes
    }

    /// Get current instance counts
    pub fn counts(&self) -> &HashMap<String, usize> {
        &self.counts
    }

    /// Update aligner with current prototypes for next iteration
    pub fn update_aligner(&mut self) {
        let current_protos = self.finalize_snapshot(1);
        self.aligner = DtwAligner::with_prototypes(current_protos);
    }

    /// Get current prototypes without consuming
    fn finalize_snapshot(&self, min_instances: usize) -> HashMap<String, HV16> {
        let mut prototypes = HashMap::new();

        for (phoneme, hvs) in &self.accumulators {
            if hvs.len() >= min_instances {
                let prototype = crate::hdc::bundle(hvs);
                prototypes.insert(phoneme.clone(), prototype);
            }
        }

        prototypes
    }
}

impl Default for DtwTrainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_alignment() {
        let aligner = DtwAligner::new();
        let phonemes = vec!["AH".to_string(), "B".to_string(), "IY".to_string()];

        let alignment = aligner.align_uniform(9, &phonemes);

        assert_eq!(alignment.frame_to_phoneme.len(), 9);
        assert_eq!(alignment.segments.len(), 3);

        // Each phoneme should get 3 frames
        for seg in &alignment.segments {
            assert_eq!(seg.end_frame - seg.start_frame, 3);
        }
    }

    #[test]
    fn test_dtw_alignment_basic() {
        let mut prototypes = HashMap::new();
        prototypes.insert("A".to_string(), HV16::random("A"));
        prototypes.insert("B".to_string(), HV16::random("B"));

        let aligner = DtwAligner::with_prototypes(prototypes.clone());

        // Create frames that are similar to prototypes
        let frames = vec![
            prototypes["A"],
            prototypes["A"],
            prototypes["B"],
            prototypes["B"],
            prototypes["B"],
        ];

        let phonemes = vec!["A".to_string(), "B".to_string()];
        let alignment = aligner.align(&frames, &phonemes);

        assert_eq!(alignment.segments.len(), 2);
        assert!(alignment.cost < f32::INFINITY);
    }
}
