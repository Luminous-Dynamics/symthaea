// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-aesthetic
//!
//! Modality-independent aesthetic evaluation and feedback for Symthaea's creative systems.
//!
//! Provides the "beauty signal" — a composite aesthetic score that feeds back into
//! the neuromodulator bath, driving the consciousness-aware creative loop.
//!
//! # Architecture
//!
//! ```text
//! Artifact → AestheticEvaluator → AestheticScore → AestheticFeedback → NeuromodulatorBath
//! ```
//!
//! The evaluator trait is modality-independent: visual art, music, and poetry each
//! implement it for their respective artifact types. The feedback conversion is shared.
//!
//! # Theoretical Foundations
//!
//! - **Birkhoff (1933)**: Aesthetic measure M = O/C (order / complexity)
//! - **Shannon (1948)**: Information-theoretic complexity via entropy
//! - **Livio (2002)**: Golden ratio and Fibonacci aesthetics
//! - **Berlyne (1971)**: Arousal potential — optimal complexity for aesthetic pleasure

#![deny(unsafe_code)]

pub mod birkhoff;
pub mod feedback;
pub mod golden;
pub mod information;
pub mod novelty;
pub mod session;
pub mod synesthesia;
pub mod valence_arousal;

pub use valence_arousal::{MusicalParams, ValenceArousal, from_core_affect, lerp_va};

use serde::{Deserialize, Serialize};

// ─── Core Trait ───────────────────────────────────────────────────────────────

/// Modality-independent aesthetic evaluation.
///
/// Implement this for each creative modality (visual, musical, poetic).
/// The evaluator produces an `AestheticScore` from an artifact of type `A`.
pub trait AestheticEvaluator<A> {
    /// Evaluate the aesthetic quality of an artifact.
    ///
    /// Returns a score with all dimensions in [0.0, 1.0].
    fn evaluate(&self, artifact: &A) -> AestheticScore;
}

// ─── Core Types ──────────────────────────────────────────────────────────────

/// Multi-dimensional aesthetic score.
///
/// Each dimension captures a different aspect of beauty, allowing the creative
/// system to optimize for specific aesthetic qualities.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AestheticScore {
    /// Symmetry, balance, repetition, structural regularity.
    /// High order = pleasing regularity (Birkhoff's "O").
    pub order: f32,

    /// Information content, variety, richness.
    /// High complexity = more elements to process (Birkhoff's "C").
    pub complexity: f32,

    /// Deviation from aesthetic expectation (the EMA of recent scores).
    /// High surprise = novel aesthetic territory.
    pub surprise: f32,

    /// Consonance with the Eight Harmonies projection.
    /// High harmony = alignment with the system's value basis.
    pub harmony: f32,

    /// Birkhoff's aesthetic measure: order / complexity.
    /// The classical beauty metric.
    pub birkhoff: f32,

    /// Weighted composite of all dimensions.
    /// Default weighting: 0.25 × birkhoff + 0.20 × harmony + 0.20 × surprise
    ///                   + 0.20 × information_balance + 0.15 × golden_ratio
    pub composite: f32,
}

impl AestheticScore {
    /// Create a score with all dimensions set to the same value.
    pub fn uniform(value: f32) -> Self {
        let v = value.clamp(0.0, 1.0);
        Self {
            order: v,
            complexity: v,
            surprise: v,
            harmony: v,
            birkhoff: v,
            composite: v,
        }
    }

    /// Create a zero score (no aesthetic value).
    pub fn zero() -> Self {
        Self::uniform(0.0)
    }

    /// Compute the composite score from individual dimensions.
    ///
    /// Weights: birkhoff=0.30, harmony=0.25, surprise=0.20, order=0.15, complexity_balance=0.10
    /// where complexity_balance = 1.0 - |complexity - 0.5| * 2 (peaks at medium complexity,
    /// per Berlyne's arousal theory).
    pub fn compute_composite(&mut self) {
        // Berlyne: moderate complexity is most pleasing
        let complexity_balance = 1.0 - (self.complexity - 0.5).abs() * 2.0;
        let complexity_balance = complexity_balance.clamp(0.0, 1.0);

        self.composite = (0.30 * self.birkhoff
            + 0.25 * self.harmony
            + 0.20 * self.surprise
            + 0.15 * self.order
            + 0.10 * complexity_balance)
            .clamp(0.0, 1.0);
    }
}

impl Default for AestheticScore {
    fn default() -> Self {
        Self::zero()
    }
}

/// Aesthetic feedback signal for the cognitive loop.
///
/// Converts an `AestheticScore` into neuromodulator deltas that can be injected
/// into the `NeuromodulatorBath`. This is the key loop-closing mechanism:
/// beautiful outputs reinforce creative exploration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AestheticFeedback {
    /// Dopamine delta: reward for exceeding the aesthetic EMA.
    /// Positive when current score > running average (reward prediction error).
    pub dopamine_delta: f32,

    /// Serotonin delta: satisfaction from harmonic alignment.
    /// Proportional to the harmony dimension of the score.
    pub serotonin_delta: f32,

    /// Noradrenaline signal: surprise/novelty for the exploration system.
    /// High when aesthetic surprise is high.
    pub surprise_signal: f32,

    /// Projection of the artifact's aesthetic character onto the Eight Harmonies.
    /// Used by the creative loop to bias future generation toward resonant harmonies.
    pub harmony_projection: [f32; 8],
}

impl AestheticFeedback {
    /// No-feedback signal (all zeros).
    pub fn neutral() -> Self {
        Self {
            dopamine_delta: 0.0,
            serotonin_delta: 0.0,
            surprise_signal: 0.0,
            harmony_projection: [0.0; 8],
        }
    }
}

impl Default for AestheticFeedback {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Configuration for the aesthetic evaluation system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AestheticConfig {
    /// EMA alpha for tracking the aesthetic running average.
    /// Lower = smoother, higher = more responsive.
    pub ema_alpha: f32,

    /// Dopamine reward scaling factor.
    pub dopamine_scale: f32,

    /// Serotonin satisfaction scaling factor.
    pub serotonin_scale: f32,

    /// Surprise signal scaling factor.
    pub surprise_scale: f32,

    /// Minimum score delta (vs EMA) to trigger dopamine reward.
    pub reward_threshold: f32,
}

impl Default for AestheticConfig {
    fn default() -> Self {
        Self {
            ema_alpha: 0.15,
            dopamine_scale: 0.10,
            serotonin_scale: 0.05,
            surprise_scale: 0.10,
            reward_threshold: 0.02,
        }
    }
}

/// Persisted aesthetic memory — survives across sessions.
///
/// Stores the EMA baseline and accumulated harmony bias so Symthaea
/// develops a genuine long-term aesthetic identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AestheticMemory {
    /// EMA of composite scores accumulated across all sessions.
    pub ema: f32,
    /// Accumulated harmony bias (which harmonies have historically scored well).
    pub harmony_bias: [f32; 8],
    /// Total evaluations ever performed.
    pub total_evaluations: u64,
    /// Sessions completed.
    pub session_count: u32,
}

impl AestheticMemory {
    pub fn new() -> Self {
        Self {
            ema: 0.5,
            harmony_bias: [0.0; 8],
            total_evaluations: 0,
            session_count: 0,
        }
    }

    /// Load from a JSON file, or return a fresh memory if missing/corrupt.
    pub fn load(path: &std::path::Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Persist to a JSON file. Silently no-ops on write failure.
    pub fn save(&self, path: &std::path::Path) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, json);
        }
    }
}

impl Default for AestheticMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Stateful aesthetic tracker that maintains a running EMA of scores
/// and produces feedback signals relative to that expectation.
#[derive(Debug, Clone)]
pub struct AestheticTracker {
    config: AestheticConfig,
    /// Exponential moving average of composite scores.
    ema: f32,
    /// Accumulated harmony bias from historical high-scoring works.
    harmony_bias: [f32; 8],
    /// Total evaluations performed.
    evaluation_count: u64,
}

impl AestheticTracker {
    pub fn new(config: AestheticConfig) -> Self {
        Self {
            config,
            ema: 0.5, // neutral starting expectation
            harmony_bias: [0.0; 8],
            evaluation_count: 0,
        }
    }

    /// Create a tracker pre-warmed from persisted memory.
    ///
    /// The EMA and harmony bias from previous sessions seed this session,
    /// so Symthaea's aesthetic expectations carry forward over time.
    pub fn from_memory(config: AestheticConfig, memory: &AestheticMemory) -> Self {
        Self {
            config,
            ema: memory.ema,
            harmony_bias: memory.harmony_bias,
            evaluation_count: memory.total_evaluations,
        }
    }

    /// Snapshot current state into an `AestheticMemory` for persistence.
    pub fn to_memory(&self, previous: &AestheticMemory) -> AestheticMemory {
        AestheticMemory {
            ema: self.ema,
            harmony_bias: self.harmony_bias,
            total_evaluations: self.evaluation_count,
            session_count: previous.session_count + 1,
        }
    }

    /// The accumulated harmony bias: which harmonies have historically scored well.
    /// Values in [0, 1] — higher means this harmony correlates with beautiful output.
    pub fn harmony_bias(&self) -> &[f32; 8] {
        &self.harmony_bias
    }

    /// Process an aesthetic score and produce feedback.
    ///
    /// Updates the EMA and computes neuromodulator deltas relative to expectation.
    pub fn process(
        &mut self,
        score: &AestheticScore,
        harmony_activations: &[f32; 8],
    ) -> AestheticFeedback {
        self.evaluation_count += 1;

        // Compute surprise as deviation from expectation
        let delta = score.composite - self.ema;

        // Update EMA
        self.ema =
            self.ema * (1.0 - self.config.ema_alpha) + score.composite * self.config.ema_alpha;

        // Dopamine: reward prediction error (positive when exceeding expectation)
        let dopamine_delta = if delta > self.config.reward_threshold {
            (delta * self.config.dopamine_scale).min(0.15)
        } else if delta < -self.config.reward_threshold {
            // Mild negative signal for disappointing output (not as strong as reward)
            (delta * self.config.dopamine_scale * 0.5).max(-0.05)
        } else {
            0.0
        };

        // Serotonin: proportional to harmony alignment
        let serotonin_delta = score.harmony * self.config.serotonin_scale;

        // Surprise signal for exploration system
        let surprise_signal = score.surprise * self.config.surprise_scale;

        // Project aesthetic quality onto harmonies using the system's current activations.
        // Harmonies that are active AND the artwork scored well get reinforced.
        let harmony_projection: [f32; 8] =
            std::array::from_fn(|i| harmony_activations[i] * score.composite);

        // Accumulate long-term harmony bias: EMA toward high-scoring harmony activations.
        // Alpha 0.01 = very slow drift, building aesthetic identity over many sessions.
        let bias_alpha = 0.01_f32;
        for (bias, &activation) in self.harmony_bias.iter_mut().zip(harmony_activations.iter()) {
            let target = activation * score.composite;
            *bias = *bias * (1.0 - bias_alpha) + target * bias_alpha;
        }

        AestheticFeedback {
            dopamine_delta,
            serotonin_delta,
            surprise_signal,
            harmony_projection,
        }
    }

    /// Current EMA value (the system's aesthetic expectation).
    pub fn expectation(&self) -> f32 {
        self.ema
    }

    /// Total number of evaluations performed.
    pub fn evaluation_count(&self) -> u64 {
        self.evaluation_count
    }

    /// Reset the tracker to initial state (preserves harmony_bias).
    pub fn reset(&mut self) {
        self.ema = 0.5;
        self.evaluation_count = 0;
        // harmony_bias intentionally preserved — it's long-term taste, not session state
    }

    // ── Human Feedback API ───────────────────────────────────────────────────

    /// Incorporate human feedback into the aesthetic system.
    ///
    /// `rating` is a scalar from -1.0 (terrible) through 0.0 (neutral) to +1.0 (beautiful).
    /// `harmony_activations` is the harmony state at the time the rated piece was generated.
    ///
    /// Human feedback carries 10x more weight than self-evaluation because humans
    /// are the ground truth for aesthetic quality. The system recalibrates its
    /// EMA and harmony bias toward the human's judgement.
    ///
    /// # Example
    ///
    /// ```
    /// # use symthaea_aesthetic::{AestheticTracker, AestheticConfig};
    /// let mut tracker = AestheticTracker::new(AestheticConfig::default());
    /// let harmonies = [0.5, 0.6, 0.4, 0.7, 0.3, 0.5, 0.8, 0.2];
    /// tracker.human_feedback(0.8, &harmonies); // "that was beautiful"
    /// ```
    pub fn human_feedback(
        &mut self,
        rating: f32,
        harmony_activations: &[f32; 8],
    ) -> AestheticFeedback {
        let rating = rating.clamp(-1.0, 1.0);
        // Map rating [-1, 1] to score [0, 1]
        let score = (rating + 1.0) * 0.5;

        self.evaluation_count += 1;

        let delta = score - self.ema;

        // Human feedback EMA alpha is 10x stronger than self-evaluation.
        // One "beautiful" from a human shifts expectations more than 10 self-scores.
        let human_alpha = (self.config.ema_alpha * 10.0).min(0.5);
        self.ema = self.ema * (1.0 - human_alpha) + score * human_alpha;

        // Harmony bias: human-rated works get 5x stronger bias update.
        // This is how Symthaea learns "what sounds she likes = what humans like."
        let human_bias_alpha = 0.05_f32;
        for (bias, &activation) in self.harmony_bias.iter_mut().zip(harmony_activations.iter()) {
            let target = activation * score;
            *bias = *bias * (1.0 - human_bias_alpha) + target * human_bias_alpha;
        }

        // Strong dopamine signal: human approval is the ultimate reward
        let dopamine_delta = delta * self.config.dopamine_scale * 2.0;
        let serotonin_delta = if rating > 0.3 {
            rating * self.config.serotonin_scale * 1.5 // warm glow from approval
        } else if rating < -0.3 {
            rating * self.config.serotonin_scale * 0.5 // mild serotonin dip
        } else {
            0.0
        };

        let harmony_projection: [f32; 8] = std::array::from_fn(|i| harmony_activations[i] * score);

        AestheticFeedback {
            dopamine_delta: dopamine_delta.clamp(-0.2, 0.3),
            serotonin_delta: serotonin_delta.clamp(-0.1, 0.2),
            surprise_signal: delta.abs() * self.config.surprise_scale,
            harmony_projection,
        }
    }

    /// Batch human feedback: apply multiple ratings at once.
    ///
    /// Useful for catching up on feedback from a listening session.
    pub fn human_feedback_batch(&mut self, ratings: &[(f32, [f32; 8])]) -> Vec<AestheticFeedback> {
        ratings
            .iter()
            .map(|(rating, harmonies)| self.human_feedback(*rating, harmonies))
            .collect()
    }
}

impl Default for AestheticTracker {
    fn default() -> Self {
        Self::new(AestheticConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_composite_moderate_complexity_preferred() {
        let mut low = AestheticScore {
            order: 0.8,
            complexity: 0.1, // too simple
            surprise: 0.5,
            harmony: 0.7,
            birkhoff: 0.8,
            composite: 0.0,
        };
        let mut mid = AestheticScore {
            order: 0.8,
            complexity: 0.5, // just right (Berlyne peak)
            surprise: 0.5,
            harmony: 0.7,
            birkhoff: 0.8,
            composite: 0.0,
        };
        let mut high = AestheticScore {
            order: 0.8,
            complexity: 0.9, // too complex
            surprise: 0.5,
            harmony: 0.7,
            birkhoff: 0.8,
            composite: 0.0,
        };
        low.compute_composite();
        mid.compute_composite();
        high.compute_composite();
        assert!(mid.composite > low.composite, "mid {mid:?} > low {low:?}");
        assert!(
            mid.composite > high.composite,
            "mid {mid:?} > high {high:?}"
        );
    }

    #[test]
    fn score_bounded() {
        let mut score = AestheticScore {
            order: 1.0,
            complexity: 0.5,
            surprise: 1.0,
            harmony: 1.0,
            birkhoff: 1.0,
            composite: 0.0,
        };
        score.compute_composite();
        assert!(score.composite >= 0.0 && score.composite <= 1.0);

        let mut zero_score = AestheticScore::zero();
        zero_score.compute_composite();
        assert!(zero_score.composite >= 0.0);
    }

    #[test]
    fn tracker_ema_updates() {
        let mut tracker = AestheticTracker::default();
        let harmonies = [0.5; 8];

        let score = AestheticScore::uniform(0.8);
        tracker.process(&score, &harmonies);

        // EMA should move toward 0.8 from initial 0.5
        assert!(tracker.expectation() > 0.5);
        assert!(tracker.expectation() < 0.8);
    }

    #[test]
    fn tracker_reward_on_exceeding_expectation() {
        let mut tracker = AestheticTracker::default();
        let harmonies = [0.5; 8];

        // First: set baseline low
        for _ in 0..10 {
            tracker.process(&AestheticScore::uniform(0.3), &harmonies);
        }
        let ema_before = tracker.expectation();

        // Now: provide a high score — should produce positive dopamine
        let high_score = AestheticScore::uniform(0.9);
        let feedback = tracker.process(&high_score, &harmonies);

        assert!(
            feedback.dopamine_delta > 0.0,
            "should reward exceeding EMA: delta={}",
            feedback.dopamine_delta
        );
        assert!(tracker.expectation() > ema_before, "EMA should increase");
    }

    #[test]
    fn tracker_disappointment_mild_negative() {
        let mut tracker = AestheticTracker::default();
        let harmonies = [0.5; 8];

        // Set baseline high
        for _ in 0..10 {
            tracker.process(&AestheticScore::uniform(0.9), &harmonies);
        }

        // Now: disappointing output
        let low_score = AestheticScore::uniform(0.2);
        let feedback = tracker.process(&low_score, &harmonies);

        assert!(
            feedback.dopamine_delta < 0.0,
            "should have mild negative dopamine: {}",
            feedback.dopamine_delta
        );
        // But not too negative (clamped)
        assert!(feedback.dopamine_delta >= -0.05);
    }

    #[test]
    fn feedback_neutral_is_zero() {
        let feedback = AestheticFeedback::neutral();
        assert_eq!(feedback.dopamine_delta, 0.0);
        assert_eq!(feedback.serotonin_delta, 0.0);
        assert_eq!(feedback.surprise_signal, 0.0);
        assert_eq!(feedback.harmony_projection, [0.0; 8]);
    }

    #[test]
    fn harmony_projection_scales_with_activation() {
        let mut tracker = AestheticTracker::default();
        let mut harmonies = [0.0f32; 8];
        harmonies[3] = 1.0; // Only InfinitePlay active

        let score = AestheticScore::uniform(0.8);
        let feedback = tracker.process(&score, &harmonies);

        assert!(feedback.harmony_projection[3] > 0.0);
        assert_eq!(feedback.harmony_projection[0], 0.0); // Inactive harmony
    }

    #[test]
    fn tracker_evaluation_count() {
        let mut tracker = AestheticTracker::default();
        assert_eq!(tracker.evaluation_count(), 0);

        tracker.process(&AestheticScore::uniform(0.5), &[0.5; 8]);
        assert_eq!(tracker.evaluation_count(), 1);

        tracker.process(&AestheticScore::uniform(0.5), &[0.5; 8]);
        assert_eq!(tracker.evaluation_count(), 2);
    }

    #[test]
    fn tracker_reset() {
        let mut tracker = AestheticTracker::default();
        tracker.process(&AestheticScore::uniform(0.9), &[0.5; 8]);
        tracker.reset();
        assert_eq!(tracker.expectation(), 0.5);
        assert_eq!(tracker.evaluation_count(), 0);
    }
}
