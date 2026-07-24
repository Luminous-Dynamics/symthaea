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
    [
        finite_unit(score.order),
        finite_unit(score.complexity),
        finite_unit(score.harmony),
        finite_unit(score.birkhoff),
    ]
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

/// Provenance for a preference observation.
///
/// Source counts remain separate so dashboards and evidence bundles can
/// distinguish grounded human taste from synthetic or self-critic labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreferenceSource {
    Human,
    Population,
    Analyst,
    SelfCritic,
}

impl PreferenceSource {
    fn index(self) -> usize {
        match self {
            Self::Human => 0,
            Self::Population => 1,
            Self::Analyst => 2,
            Self::SelfCritic => 3,
        }
    }
}

/// Explicit target used to train aesthetic preference rather than re-learning
/// the crate's own hand-authored composite formula.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PreferenceObservation {
    /// Aesthetic evidence being judged.
    pub score: AestheticScore,
    /// Observed preference in [0, 1].
    pub target: f32,
    /// Reliability or agreement weight in [0, 1].
    pub confidence: f32,
    /// Origin of the judgement.
    pub source: PreferenceSource,
}

impl PreferenceObservation {
    pub fn human(score: AestheticScore, target: f32, confidence: f32) -> Self {
        Self {
            score,
            target: finite_unit(target),
            confidence: finite_unit(confidence),
            source: PreferenceSource::Human,
        }
    }
}

/// Online logistic preference model over the aesthetic evidence profile.
///
/// The model accepts explicit absolute labels and pairwise comparisons. It no
/// longer silently trains against `score.composite`, which only teaches the
/// model to imitate the formula it was meant to evaluate independently.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TasteModel {
    /// Signed learned coefficients for order, complexity, harmony, and Birkhoff.
    weights: [f32; 4],
    /// Bias term in logit space.
    bias: f32,
    /// Number of observations seen.
    examples_seen: usize,
    /// Learning rate.
    lr: f32,
    /// Counts by [`PreferenceSource`] for auditability.
    #[serde(default)]
    source_counts: [usize; 4],
}

impl TasteModel {
    pub fn new() -> Self {
        Self {
            weights: [0.0; 4],
            bias: 0.0,
            examples_seen: 0,
            lr: 0.05,
            source_counts: [0; 4],
        }
    }

    /// Train from an explicit preference observation using online logistic
    /// regression. Non-finite inputs are treated as zero evidence.
    pub fn train_observation(&mut self, observation: PreferenceObservation) {
        let profile = centered_profile(&observation.score);
        let target = finite_unit(observation.target);
        let confidence = finite_unit(observation.confidence);
        let predicted = self.predict_centered(&profile);
        let error = target - predicted;
        let step = self.lr * confidence * error;

        for (weight, feature) in self.weights.iter_mut().zip(profile) {
            *weight = (*weight + step * feature).clamp(-8.0, 8.0);
        }
        self.bias = (self.bias + step).clamp(-8.0, 8.0);
        self.examples_seen = self.examples_seen.saturating_add(1);
        let index = observation.source.index();
        self.source_counts[index] = self.source_counts[index].saturating_add(1);
    }

    /// Train a Bradley-Terry-style pairwise preference: `preferred` should be
    /// ranked above `rejected`.
    pub fn train_pairwise(
        &mut self,
        preferred: &AestheticScore,
        rejected: &AestheticScore,
        confidence: f32,
        source: PreferenceSource,
    ) {
        let preferred = centered_profile(preferred);
        let rejected = centered_profile(rejected);
        let difference: [f32; 4] = std::array::from_fn(|index| preferred[index] - rejected[index]);
        let logit: f32 = self
            .weights
            .iter()
            .zip(difference)
            .map(|(weight, feature)| weight * feature)
            .sum();
        let predicted = logistic(logit);
        let step = self.lr * finite_unit(confidence) * (1.0 - predicted);

        for (weight, feature) in self.weights.iter_mut().zip(difference) {
            *weight = (*weight + step * feature).clamp(-8.0, 8.0);
        }
        self.examples_seen = self.examples_seen.saturating_add(1);
        let index = source.index();
        self.source_counts[index] = self.source_counts[index].saturating_add(1);
    }

    /// Legacy formula-imitation path retained for source compatibility.
    ///
    /// Prefer [`Self::train_observation`] or [`Self::train_pairwise`]. This
    /// method records the source as `SelfCritic` so synthetic training cannot be
    /// mistaken for grounded human evidence.
    #[deprecated(note = "use train_observation or train_pairwise with explicit labels")]
    pub fn train(&mut self, score: &AestheticScore) {
        self.train_observation(PreferenceObservation {
            score: *score,
            target: finite_unit(score.composite),
            confidence: 1.0,
            source: PreferenceSource::SelfCritic,
        });
    }

    /// Predict aesthetic preference from a score's evidence profile.
    pub fn predict(&self, score: &AestheticScore) -> f32 {
        self.predict_centered(&centered_profile(score))
    }

    fn predict_centered(&self, profile: &[f32; 4]) -> f32 {
        let logit = self.bias
            + self
                .weights
                .iter()
                .zip(profile.iter())
                .map(|(weight, feature)| weight * feature)
                .sum::<f32>();
        logistic(logit)
    }

    /// Current signed learned coefficients (for introspection).
    pub fn weights(&self) -> &[f32; 4] {
        &self.weights
    }

    /// Normalized absolute feature importance in [0, 1], summing to one when
    /// at least one coefficient is non-zero.
    pub fn feature_importance(&self) -> [f32; 4] {
        let magnitude: f32 = self.weights.iter().map(|weight| weight.abs()).sum();
        if magnitude <= f32::EPSILON {
            [0.0; 4]
        } else {
            std::array::from_fn(|index| self.weights[index].abs() / magnitude)
        }
    }

    /// How many examples the model has been trained on.
    pub fn examples_seen(&self) -> usize {
        self.examples_seen
    }

    /// Number of observations received from a particular provenance class.
    pub fn source_count(&self, source: PreferenceSource) -> usize {
        self.source_counts[source.index()]
    }
}

fn centered_profile(score: &AestheticScore) -> [f32; 4] {
    let profile = score_to_profile(score);
    std::array::from_fn(|index| finite_unit(profile[index]) - 0.5)
}

fn finite_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn logistic(value: f32) -> f32 {
    1.0 / (1.0 + (-value.clamp(-20.0, 20.0)).exp())
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
    fn taste_model_learns_from_explicit_human_labels() {
        let mut model = TasteModel::new();
        let high = make_score(0.3, 0.5, 0.9, 0.0);
        let low = make_score(0.3, 0.5, 0.1, 1.0);

        for _ in 0..150 {
            model.train_observation(PreferenceObservation::human(high, 0.95, 1.0));
            model.train_observation(PreferenceObservation::human(low, 0.05, 1.0));
        }

        assert!(model.predict(&high) > model.predict(&low));
        assert_eq!(model.source_count(PreferenceSource::Human), 300);
        assert_eq!(model.source_count(PreferenceSource::SelfCritic), 0);
    }

    #[test]
    fn pairwise_training_learns_ranking_without_absolute_scores() {
        let mut model = TasteModel::new();
        let preferred = make_score(0.8, 0.5, 0.8, 0.0);
        let rejected = make_score(0.2, 0.5, 0.2, 1.0);
        for _ in 0..100 {
            model.train_pairwise(&preferred, &rejected, 1.0, PreferenceSource::Human);
        }
        assert!(model.predict(&preferred) > model.predict(&rejected));
    }

    #[test]
    fn feature_importance_is_normalized() {
        let mut model = TasteModel::new();
        let preferred = make_score(0.9, 0.2, 0.8, 0.0);
        let rejected = make_score(0.1, 0.8, 0.2, 1.0);
        model.train_pairwise(&preferred, &rejected, 1.0, PreferenceSource::Population);
        let sum: f32 = model.feature_importance().iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "importance sum = {sum}");
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
