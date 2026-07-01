// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Novelty scoring relative to a history of past aesthetic evaluations.
//!
//! Instead of comparing only to the EMA (a single scalar), this module
//! compares a new score's multi-dimensional profile against the full
//! distribution of recent scores, rewarding departure from the norm.

use crate::AestheticScore;
use serde::{Deserialize, Serialize};

/// Sliding window of recent aesthetic score profiles for novelty computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoveltyTracker {
    /// Recent score profiles (order, complexity, harmony, birkhoff).
    history: Vec<[f32; 4]>,
    /// Maximum history size.
    max_history: usize,
}

impl NoveltyTracker {
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::new(),
            max_history: max_history.max(2),
        }
    }

    /// Record a new score and compute its novelty relative to history.
    ///
    /// Returns a novelty value in [0, 1] where:
    /// - 0.0 = identical to the average of recent scores
    /// - 1.0 = maximally different from all recent scores
    pub fn record_and_score(&mut self, score: &AestheticScore) -> f32 {
        let profile = score_to_profile(score);
        let novelty = if self.history.is_empty() {
            0.5 // neutral for first score
        } else {
            compute_novelty(&profile, &self.history)
        };

        self.history.push(profile);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        novelty
    }

    /// Compute novelty without recording (peek).
    pub fn peek_novelty(&self, score: &AestheticScore) -> f32 {
        if self.history.is_empty() {
            return 0.5;
        }
        let profile = score_to_profile(score);
        compute_novelty(&profile, &self.history)
    }

    /// Number of scores in history.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Reset the history.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

impl Default for NoveltyTracker {
    fn default() -> Self {
        Self::new(50)
    }
}

/// Extract the 4D profile from an AestheticScore.
fn score_to_profile(score: &AestheticScore) -> [f32; 4] {
    [score.order, score.complexity, score.harmony, score.birkhoff]
}

/// Compute novelty as average Euclidean distance from history centroid,
/// normalized to [0, 1].
fn compute_novelty(profile: &[f32; 4], history: &[[f32; 4]]) -> f32 {
    let n = history.len() as f32;
    if n == 0.0 {
        return 0.5;
    }

    // Compute centroid
    let mut centroid = [0.0f32; 4];
    for h in history {
        for i in 0..4 {
            centroid[i] += h[i] / n;
        }
    }

    // Euclidean distance from centroid
    let dist: f32 = profile
        .iter()
        .zip(centroid.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();

    // Also compute average pairwise distance in history for normalization
    let mut avg_dist = 0.0f32;
    let mut count = 0;
    for h in history {
        let d: f32 = h
            .iter()
            .zip(centroid.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        avg_dist += d;
        count += 1;
    }
    let baseline = if count > 0 {
        (avg_dist / count as f32).max(0.01)
    } else {
        0.5
    };

    // Novelty: how many baseline-distances away from centroid
    // Sigmoid to [0, 1]
    let ratio = dist / baseline;
    sigmoid(ratio - 1.0) // centered at 1.0 = average distance
}

/// Sigmoid function mapping (-inf, inf) to (0, 1).
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x * 2.0).exp())
}

/// A simple linear aesthetic preference model trained from gallery history.
///
/// Learns which combinations of (order, complexity, harmony, birkhoff)
/// the system has historically rated highest, enabling personalized taste.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TasteModel {
    /// Learned weights for each dimension.
    weights: [f32; 4],
    /// Bias term.
    bias: f32,
    /// Number of training examples seen.
    examples_seen: usize,
    /// Learning rate.
    lr: f32,
}

impl TasteModel {
    pub fn new() -> Self {
        Self {
            // Start with equal weights (neutral preference)
            weights: [0.25, 0.25, 0.25, 0.25],
            bias: 0.0,
            examples_seen: 0,
            lr: 0.01,
        }
    }

    /// Train on a single example: (score, actual composite).
    ///
    /// Uses simple online gradient descent to learn which dimensions
    /// best predict the composite score.
    pub fn train(&mut self, score: &AestheticScore) {
        let profile = score_to_profile(score);
        let predicted = self.predict_raw(&profile);
        let error = score.composite - predicted;

        // Gradient descent
        for i in 0..4 {
            self.weights[i] += self.lr * error * profile[i];
        }
        self.bias += self.lr * error;

        // Keep weights non-negative and normalized
        for w in &mut self.weights {
            *w = w.max(0.0);
        }
        let sum: f32 = self.weights.iter().sum();
        if sum > 0.01 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }

        self.examples_seen += 1;
    }

    /// Predict aesthetic quality from a score's profile.
    pub fn predict(&self, score: &AestheticScore) -> f32 {
        let profile = score_to_profile(score);
        self.predict_raw(&profile).clamp(0.0, 1.0)
    }

    fn predict_raw(&self, profile: &[f32; 4]) -> f32 {
        let mut sum = self.bias;
        for i in 0..4 {
            sum += self.weights[i] * profile[i];
        }
        sum
    }

    /// Current learned weights (for introspection).
    pub fn weights(&self) -> &[f32; 4] {
        &self.weights
    }

    /// How many examples the model has been trained on.
    pub fn examples_seen(&self) -> usize {
        self.examples_seen
    }
}

impl Default for TasteModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_score(order: f32, complexity: f32, harmony: f32, composite: f32) -> AestheticScore {
        AestheticScore {
            order,
            complexity,
            surprise: 0.0,
            harmony,
            birkhoff: if complexity > 0.01 {
                order / complexity
            } else {
                0.0
            },
            composite,
        }
    }

    #[test]
    fn novelty_first_score_neutral() {
        let mut tracker = NoveltyTracker::default();
        let score = make_score(0.5, 0.5, 0.5, 0.5);
        let novelty = tracker.record_and_score(&score);
        assert!(
            (novelty - 0.5).abs() < 0.1,
            "first score novelty = {novelty}"
        );
    }

    #[test]
    fn similar_scores_low_novelty() {
        let mut tracker = NoveltyTracker::default();
        for _ in 0..10 {
            tracker.record_and_score(&make_score(0.5, 0.5, 0.5, 0.5));
        }
        // One more similar score should have low novelty
        let novelty = tracker.peek_novelty(&make_score(0.5, 0.5, 0.5, 0.5));
        assert!(novelty < 0.6, "similar novelty = {novelty}");
    }

    #[test]
    fn different_score_high_novelty() {
        let mut tracker = NoveltyTracker::default();
        for _ in 0..10 {
            tracker.record_and_score(&make_score(0.5, 0.5, 0.5, 0.5));
        }
        // Very different score should have higher novelty
        let novelty = tracker.peek_novelty(&make_score(0.1, 0.9, 0.1, 0.3));
        assert!(novelty > 0.5, "different novelty = {novelty}");
    }

    #[test]
    fn tracker_caps_history() {
        let mut tracker = NoveltyTracker::new(5);
        for _ in 0..10 {
            tracker.record_and_score(&make_score(0.5, 0.5, 0.5, 0.5));
        }
        assert_eq!(tracker.len(), 5);
    }

    #[test]
    fn taste_model_learns() {
        let mut model = TasteModel::new();

        // Train on scores where high harmony = high composite
        for _ in 0..100 {
            model.train(&make_score(0.3, 0.5, 0.9, 0.8));
            model.train(&make_score(0.3, 0.5, 0.1, 0.2));
        }

        // Model should predict higher for high-harmony scores
        let high_h = model.predict(&make_score(0.3, 0.5, 0.9, 0.0));
        let low_h = model.predict(&make_score(0.3, 0.5, 0.1, 0.0));
        assert!(high_h > low_h, "high harmony {high_h} should > low {low_h}");
    }

    #[test]
    fn taste_model_weights_normalized() {
        let mut model = TasteModel::new();
        for _ in 0..50 {
            model.train(&make_score(0.7, 0.3, 0.8, 0.7));
        }
        let sum: f32 = model.weights().iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "weights sum = {sum}");
    }

    #[test]
    fn novelty_bounded() {
        let mut tracker = NoveltyTracker::default();
        for i in 0..20 {
            let v = i as f32 / 20.0;
            let novelty = tracker.record_and_score(&make_score(v, 1.0 - v, v, 0.5));
            assert!(
                novelty >= 0.0 && novelty <= 1.0,
                "novelty {novelty} out of bounds"
            );
        }
    }
}
