// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ensemble self-critic: the system evaluates its own creative output.
//!
//! Orchestrates multiple existing scoring dimensions into a single verdict.
//! Weights shift with the neuromodulator state (moral-aesthetic binding):
//! care → harmony + golden ratio; stress → novelty + entropy.
//!
//! Detects Sacred Stillness: when improvement drops below threshold, the
//! work is declared complete. The system knows when to stop.

use symthaea_aesthetic::{
    golden, information, novelty, AestheticConfig, AestheticScore, AestheticTracker,
};
use symthaea_canvas::CognitiveSnapshot;

/// Input to the self-critic: perceptual features from generated artwork.
///
/// Decoupled from `self_perception::PerceptualInput` to avoid requiring
/// pixel_canvas/neural_canvas dependencies. Can be constructed from any
/// perception source (PerceptualEncoder, SigLIP, manual features).
#[derive(Debug, Clone)]
pub struct PerceptualInput {
    /// Creative surprise (0-1). How unexpected was the output?
    pub creative_surprise: f32,
    /// Intention-perception coherence (0-1). How faithfully was intent expressed?
    pub perceptual_coherence: f32,
    /// Visual feature vector (variable length). Color, spatial, structural features.
    pub visual_features: Vec<f32>,
}

/// Score improvement below this triggers Sacred Stillness (work is done).
pub const STILLNESS_THRESHOLD: f32 = 0.01;

/// Verdict from the ensemble self-critic.
#[derive(Debug, Clone)]
pub struct CriticVerdict {
    /// Full aesthetic score (6 dimensions).
    pub aesthetic: AestheticScore,
    /// Novelty vs recent history (0-1).
    pub novelty: f32,
    /// Golden ratio spatial balance (0-1).
    pub golden_ratio: f32,
    /// Information balance (0-1). Peaks at moderate entropy.
    pub information_balance: f32,
    /// Self-surprise from perception (0-1).
    pub self_surprise: f32,
    /// Intention-perception coherence (0-1).
    pub intention_coherence: f32,
    /// Taste alignment from learned preferences (0-1).
    pub taste_alignment: f32,
    /// Weighted composite score.
    pub composite: f32,
    /// Whether the work has reached stillness (done).
    pub reached_stillness: bool,
}

/// Ensemble self-critic with moral-aesthetic binding.
pub struct SelfCritic {
    novelty_tracker: novelty::NoveltyTracker,
    taste_model: novelty::TasteModel,
    tracker: AestheticTracker,
    prev_composite: f32,
    eval_count: u64,
}

impl SelfCritic {
    pub fn new() -> Self {
        Self {
            novelty_tracker: novelty::NoveltyTracker::new(50),
            taste_model: novelty::TasteModel::new(),
            tracker: AestheticTracker::new(AestheticConfig::default()),
            prev_composite: 0.0,
            eval_count: 0,
        }
    }

    /// Evaluate creative output using the full ensemble of scorers.
    ///
    /// Weights shift based on neuromodulators:
    /// - High serotonin/oxytocin (CARE) → emphasize harmony + golden ratio
    /// - High noradrenaline/allostatic load (STRESS) → emphasize novelty + entropy
    pub fn evaluate(
        &mut self,
        perception: &PerceptualInput,
        snapshot: &CognitiveSnapshot,
    ) -> CriticVerdict {
        self.eval_count += 1;

        let features = &perception.visual_features;
        let self_surprise = perception.creative_surprise;
        let intention_coherence = perception.perceptual_coherence;

        // Harmony from Eight Harmonies activation mean
        let harmony_mean = snapshot.harmony_activations.iter().sum::<f32>() / 8.0;

        // Structural complexity from feature variance
        let complexity = compute_complexity(features);

        // Golden ratio: ratio of first two features (if available)
        let golden = if features.len() >= 2 && features[1].abs() > 0.001 {
            golden::golden_ratio_score(features[0] / features[1])
        } else {
            0.5
        };

        // Information balance (peaks at moderate entropy, Berlyne)
        let info_balance = if !features.is_empty() {
            information::information_balance(features)
        } else {
            0.5
        };

        // Build preliminary AestheticScore for novelty + taste tracking
        let mut score = AestheticScore {
            order: intention_coherence,
            complexity,
            surprise: self_surprise,
            harmony: harmony_mean,
            birkhoff: if complexity > 0.01 {
                (intention_coherence / complexity).clamp(0.0, 1.0)
            } else {
                0.5
            },
            composite: 0.0,
        };
        score.compute_composite();

        // Novelty vs recent history
        let novelty_score = self.novelty_tracker.record_and_score(&score);

        // Taste alignment from learned model
        let taste_alignment = self.taste_model.predict(&score).clamp(0.0, 1.0);
        self.taste_model.train(&score);

        // ── Moral-Aesthetic Binding ──

        let care = ((snapshot.serotonin + snapshot.oxytocin) / 2.0).clamp(0.0, 1.0);
        let stress = ((snapshot.noradrenaline + snapshot.allostatic_load) / 2.0).clamp(0.0, 1.0);

        // Base weights
        let mut w = [0.20f32, 0.10, 0.15, 0.10, 0.15, 0.10, 0.10, 0.10];
        // [harmony, golden, novelty, entropy, coherence, surprise, taste, birkhoff]

        // Care → harmony + golden + coherence
        w[0] += care * 0.10;
        w[1] += care * 0.05;
        w[4] += care * 0.05;
        // Stress → novelty + entropy + surprise
        w[2] += stress * 0.10;
        w[3] += stress * 0.05;
        w[5] += stress * 0.05;

        let total: f32 = w.iter().sum();
        for wi in &mut w {
            *wi /= total;
        }

        let composite = (w[0] * harmony_mean
            + w[1] * golden
            + w[2] * novelty_score
            + w[3] * info_balance
            + w[4] * intention_coherence
            + w[5] * self_surprise
            + w[6] * taste_alignment
            + w[7] * score.birkhoff)
            .clamp(0.0, 1.0);

        // ── Stillness detection ──
        let delta = (composite - self.prev_composite).abs();
        let reached_stillness = self.eval_count > 3 && delta < STILLNESS_THRESHOLD;
        self.prev_composite = composite;

        // Update tracker for downstream feedback
        let _ = self.tracker.process(&score, &snapshot.harmony_activations);

        CriticVerdict {
            aesthetic: AestheticScore { composite, ..score },
            novelty: novelty_score,
            golden_ratio: golden,
            information_balance: info_balance,
            self_surprise,
            intention_coherence,
            taste_alignment,
            composite,
            reached_stillness,
        }
    }

    /// Number of evaluations performed.
    pub fn eval_count(&self) -> u64 {
        self.eval_count
    }

    /// Reset critic state.
    pub fn reset(&mut self) {
        self.novelty_tracker = novelty::NoveltyTracker::new(50);
        self.taste_model = novelty::TasteModel::new();
        self.prev_composite = 0.0;
        self.eval_count = 0;
    }
}

impl Default for SelfCritic {
    fn default() -> Self {
        Self::new()
    }
}

fn compute_complexity(features: &[f32]) -> f32 {
    if features.is_empty() {
        return 0.0;
    }
    let mean: f32 = features.iter().sum::<f32>() / features.len() as f32;
    let variance: f32 =
        features.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / features.len() as f32;
    variance.sqrt().clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_perception() -> PerceptualInput {
        PerceptualInput {
            creative_surprise: 0.6,
            perceptual_coherence: 0.7,
            visual_features: vec![0.3, 0.5, 0.2, 0.7, 0.4, 0.6, 0.1, 0.8],
        }
    }

    fn test_snapshot() -> CognitiveSnapshot {
        CognitiveSnapshot {
            consciousness_level: 0.7,
            harmony_activations: [0.5, 0.6, 0.4, 0.7, 0.3, 0.5, 0.8, 0.2],
            dopamine: 0.6,
            serotonin: 0.5,
            noradrenaline: 0.3,
            oxytocin: 0.4,
            allostatic_load: 0.1,
            arousal: 0.5,
            valence: 0.2,
            betti_0: 3,
            betti_1: 1,
            betti_2: 0,
            ..CognitiveSnapshot::dormant()
        }
    }

    #[test]
    fn verdict_bounded() {
        let mut critic = SelfCritic::new();
        let v = critic.evaluate(&test_perception(), &test_snapshot());
        assert!(v.composite >= 0.0 && v.composite <= 1.0);
        assert!(v.novelty >= 0.0);
        assert!(v.information_balance >= 0.0 && v.information_balance <= 1.0);
    }

    #[test]
    fn care_shifts_weights() {
        let mut critic = SelfCritic::new();
        let perception = test_perception();

        let neutral = test_snapshot();
        let v1 = critic.evaluate(&perception, &neutral);
        critic.reset();

        let mut care_snap = test_snapshot();
        care_snap.serotonin = 0.95;
        care_snap.oxytocin = 0.95;
        care_snap.noradrenaline = 0.05;
        care_snap.allostatic_load = 0.0;
        let v2 = critic.evaluate(&perception, &care_snap);

        // Different weights → different composite (same perception)
        assert!(
            (v1.composite - v2.composite).abs() > 0.001,
            "care should shift weights: neutral={}, care={}",
            v1.composite, v2.composite
        );
    }

    #[test]
    fn stillness_detected() {
        let mut critic = SelfCritic::new();
        let perception = test_perception();
        let snapshot = test_snapshot();

        // Run identical evaluations until stillness
        let mut found_stillness = false;
        for _ in 0..10 {
            let v = critic.evaluate(&perception, &snapshot);
            if v.reached_stillness {
                found_stillness = true;
                break;
            }
        }
        assert!(found_stillness, "identical inputs should reach stillness");
    }

    #[test]
    fn different_inputs_different_verdicts() {
        let mut critic = SelfCritic::new();
        let snapshot = test_snapshot();

        let p1 = PerceptualInput {
            creative_surprise: 0.2,
            perceptual_coherence: 0.9,
            visual_features: vec![0.1, 0.2, 0.3, 0.4],
        };
        let v1 = critic.evaluate(&p1, &snapshot);

        let p2 = PerceptualInput {
            creative_surprise: 0.9,
            perceptual_coherence: 0.2,
            visual_features: vec![0.9, 0.8, 0.7, 0.6],
        };
        let v2 = critic.evaluate(&p2, &snapshot);

        assert!(
            (v1.composite - v2.composite).abs() > 0.01,
            "different inputs should differ: {} vs {}",
            v1.composite, v2.composite
        );
    }

    #[test]
    fn empty_features_safe() {
        let mut critic = SelfCritic::new();
        let p = PerceptualInput {
            creative_surprise: 0.5,
            perceptual_coherence: 0.5,
            visual_features: vec![],
        };
        let v = critic.evaluate(&p, &test_snapshot());
        assert!(v.composite >= 0.0 && v.composite <= 1.0);
    }

    #[test]
    fn eval_count_tracks() {
        let mut critic = SelfCritic::new();
        assert_eq!(critic.eval_count(), 0);
        critic.evaluate(&test_perception(), &test_snapshot());
        assert_eq!(critic.eval_count(), 1);
        critic.evaluate(&test_perception(), &test_snapshot());
        assert_eq!(critic.eval_count(), 2);
    }
}
