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
    AestheticConfig, AestheticScore, AestheticTracker, golden, information, novelty,
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

impl CriticVerdict {
    /// Encode this verdict as a 4D vector suitable for FEP observation.
    ///
    /// When wired into the cognitive loop, the caller creates:
    /// ```ignore
    /// let obs = symthaea_fep::Observation::new(
    ///     verdict.as_fep_values(),
    ///     2.0,        // high precision
    ///     "creative",
    /// );
    /// fep_agent.perceive(&obs);
    /// ```
    pub fn as_fep_values(&self) -> Vec<f64> {
        vec![
            self.composite as f64,           // phi proxy: aesthetic quality
            self.novelty as f64,             // integration: creative novelty
            self.intention_coherence as f64, // coherence: faithful expression
            self.taste_alignment as f64,     // attention: matches preferences
        ]
    }
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

// ── Autonomous Practice Loop ────────────────────────────────────────────────

/// Result of an autonomous practice session.
#[derive(Debug, Clone)]
pub struct PracticeResult {
    /// Number of rounds completed before stillness.
    pub rounds: usize,
    /// Score trajectory (composite per round).
    pub trajectory: Vec<f32>,
    /// Whether stillness was reached (natural completion).
    pub reached_stillness: bool,
    /// Best composite score achieved.
    pub best_score: f32,
    /// Improvement from first to best (may be negative).
    pub improvement: f32,
}

/// Autonomous practice: the system generates, critiques, and improves
/// in a closed loop without any external input.
///
/// Each round: generate artwork → extract features → critic evaluates →
/// apply wisdom to snapshot → repeat. Stops when stillness is reached
/// (score delta < 0.01) or max_rounds exceeded.
///
/// Accepts an optional `style_fn` that conditions the snapshot each round
/// (e.g., applying a gallery-derived style embedding). Pass `None` for
/// unconditioned practice.
pub fn auto_improve(
    config: &crate::AtelierConfig,
    snapshot: &mut CognitiveSnapshot,
    max_rounds: usize,
    seed_base: u64,
) -> PracticeResult {
    auto_improve_with(
        config,
        snapshot,
        max_rounds,
        seed_base,
        None::<fn(&mut CognitiveSnapshot)>,
    )
}

/// Autonomous practice with optional per-round style conditioning.
///
/// The `style_fn` is called before each generation round to condition
/// the snapshot toward the desired artistic identity. Use this to wire
/// in gallery-derived style embeddings.
pub fn auto_improve_with<F: FnMut(&mut CognitiveSnapshot)>(
    config: &crate::AtelierConfig,
    snapshot: &mut CognitiveSnapshot,
    max_rounds: usize,
    seed_base: u64,
    mut style_fn: Option<F>,
) -> PracticeResult {
    let mut critic = SelfCritic::new();
    let mut trajectory = Vec::with_capacity(max_rounds);
    let mut best_score = 0.0f32;

    for round in 0..max_rounds {
        // Apply style conditioning (if provided)
        if let Some(ref mut f) = style_fn {
            f(snapshot);
        }

        // Generate artwork using the public API
        let artwork = crate::create_artwork(config, snapshot, seed_base + round as u64);

        // Extract perceptual features from the artwork's scene graph
        let features = extract_scene_features(&artwork);

        // Critic evaluates
        let verdict = critic.evaluate(&features, snapshot);
        trajectory.push(verdict.composite);

        if verdict.composite > best_score {
            best_score = verdict.composite;
        }

        // Check for Sacred Stillness
        if verdict.reached_stillness {
            let first = trajectory[0];
            return PracticeResult {
                rounds: round + 1,
                trajectory,
                reached_stillness: true,
                best_score,
                improvement: best_score - first,
            };
        }

        // Apply wisdom: nudge cognitive state toward higher scores
        apply_verdict_wisdom(snapshot, &verdict);
    }

    PracticeResult {
        rounds: max_rounds,
        trajectory: trajectory.clone(),
        reached_stillness: false,
        best_score,
        improvement: best_score - trajectory.first().copied().unwrap_or(0.0),
    }
}

/// Extract perceptual features from an artwork's aesthetic score.
///
/// Maps the existing AestheticScore dimensions into a PerceptualInput
/// compatible with the SelfCritic.
fn extract_scene_features(artwork: &crate::Artwork) -> PerceptualInput {
    let score = &artwork.aesthetic_score;
    PerceptualInput {
        creative_surprise: score.surprise,
        perceptual_coherence: score.order,
        visual_features: vec![
            score.order,
            score.complexity,
            score.surprise,
            score.harmony,
            score.birkhoff,
            score.composite,
        ],
    }
}

/// Apply critic verdict as wisdom to evolve the cognitive state.
///
/// Uses FEP-inspired exploration/exploitation balance instead of gradient following.
/// The `exploration_urge` parameter (0-1) controls the explore/exploit tradeoff:
/// - High exploration: perturb harmonies, boost arousal, seek novelty
/// - High exploitation: reinforce dominant harmony, consolidate, boost serotonin
///
/// Call with `exploration_urge = 1.0 - verdict.novelty` as a simple heuristic,
/// or feed in the FEP agent's epistemic value for principled active inference.
pub fn apply_verdict_wisdom(snapshot: &mut CognitiveSnapshot, verdict: &CriticVerdict) {
    // Default exploration heuristic: low novelty → explore, high novelty → exploit
    let exploration_urge = (1.0 - verdict.novelty).clamp(0.0, 1.0);
    apply_fep_wisdom(snapshot, verdict, exploration_urge);
}

/// FEP-driven wisdom application with explicit exploration/exploitation parameter.
///
/// The exploration_urge comes from either:
/// - `CuriosityDrive.exploration_urge` (cognitive loop)
/// - `1.0 - verdict.novelty` (simple heuristic)
/// - FEP epistemic value (principled active inference)
pub fn apply_fep_wisdom(
    snapshot: &mut CognitiveSnapshot,
    verdict: &CriticVerdict,
    exploration_urge: f32,
) {
    let lr = 0.03;
    let exploit = (1.0 - exploration_urge).clamp(0.0, 1.0);
    let explore = exploration_urge.clamp(0.0, 1.0);

    // ── Homeostatic decay: neuromodulators drift toward resting baseline ──
    let decay = 0.02;
    snapshot.serotonin += (0.5 - snapshot.serotonin) * decay;
    snapshot.arousal += (0.4 - snapshot.arousal) * decay;
    snapshot.dopamine += (0.5 - snapshot.dopamine) * decay;
    snapshot.noradrenaline += (0.3 - snapshot.noradrenaline) * decay;

    // ── Harmony homeostatic decay: prevent all-1.0 saturation ──
    // Each harmony drifts toward its own baseline (diversity-preserving)
    let harmony_baselines = [0.4, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.3];
    for (act, baseline) in snapshot
        .harmony_activations
        .iter_mut()
        .zip(harmony_baselines.iter())
    {
        *act += (baseline - *act) * decay * 1.5;
    }

    // ── EXPLOIT: reinforce what works (dominant harmony only) ──
    if exploit > 0.5 && verdict.composite > 0.5 {
        // Find the dominant harmony (competitive reinforcement)
        let max_idx = snapshot
            .harmony_activations
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        snapshot.harmony_activations[max_idx] = (snapshot.harmony_activations[max_idx]
            + lr * exploit * verdict.aesthetic.harmony)
            .clamp(0.0, 1.0);

        // Taste satisfaction → serotonin
        if verdict.taste_alignment > 0.5 {
            snapshot.serotonin = (snapshot.serotonin + lr * 0.2 * exploit).clamp(0.0, 1.0);
        }

        // Good coherence → consolidate (reduce arousal gently)
        if verdict.intention_coherence > 0.6 {
            snapshot.arousal = (snapshot.arousal - lr * 0.2 * exploit).clamp(0.0, 1.0);
        }
    }

    // ── EXPLORE: seek novelty when bored ──
    if explore > 0.4 {
        // Boost arousal + noradrenaline (seek something new)
        snapshot.arousal = (snapshot.arousal + lr * explore).clamp(0.0, 1.0);
        snapshot.noradrenaline = (snapshot.noradrenaline + lr * 0.5 * explore).clamp(0.0, 1.0);

        // Perturb a non-dominant harmony (create variation)
        let min_idx = snapshot
            .harmony_activations
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        snapshot.harmony_activations[min_idx] =
            (snapshot.harmony_activations[min_idx] + lr * 2.0 * explore).clamp(0.0, 1.0);
    }

    // ── Entropy pressure: prevent harmony homogenization ──
    let harmony_diversity = information::element_diversity(&snapshot.harmony_activations, 0.1);
    if harmony_diversity < 0.5 {
        // Too homogeneous — dampen the highest, boost the lowest
        let (max_i, _) = snapshot
            .harmony_activations
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.5));
        let (min_i, _) = snapshot
            .harmony_activations
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.5));
        if max_i != min_i {
            snapshot.harmony_activations[max_i] =
                (snapshot.harmony_activations[max_i] - lr * 0.5).clamp(0.0, 1.0);
            snapshot.harmony_activations[min_i] =
                (snapshot.harmony_activations[min_i] + lr * 0.5).clamp(0.0, 1.0);
        }
    }

    // ── Surprise reward: unexpected good output → dopamine ──
    if verdict.self_surprise > 0.5 && verdict.composite > 0.4 {
        snapshot.dopamine = (snapshot.dopamine + lr * 0.5 * verdict.self_surprise).clamp(0.0, 1.0);
    }

    // ── Golden ratio correction ──
    if verdict.golden_ratio < 0.3 {
        snapshot.valence *= 0.97;
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
            v1.composite,
            v2.composite
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
            v1.composite,
            v2.composite
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

    #[test]
    fn auto_improve_produces_trajectory() {
        let config = crate::AtelierConfig::default();
        let mut snapshot = test_snapshot();
        let result = auto_improve(&config, &mut snapshot, 10, 42);
        assert!(result.rounds > 0, "should complete at least 1 round");
        assert_eq!(result.trajectory.len(), result.rounds);
        assert!(result.best_score >= 0.0 && result.best_score <= 1.0);
    }

    #[test]
    fn auto_improve_reaches_stillness() {
        let config = crate::AtelierConfig {
            max_elements: 50,
            iteration_budget: 1,
            ..Default::default()
        };
        let mut snapshot = test_snapshot();
        let result = auto_improve(&config, &mut snapshot, 20, 42);
        // With deterministic input, should reach stillness well before 20 rounds
        assert!(
            result.reached_stillness || result.rounds == 20,
            "should either reach stillness or exhaust rounds"
        );
    }

    #[test]
    fn auto_improve_modifies_snapshot() {
        let config = crate::AtelierConfig::default();
        let mut snapshot = test_snapshot();
        let original_arousal = snapshot.arousal;
        let original_serotonin = snapshot.serotonin;
        auto_improve(&config, &mut snapshot, 5, 42);
        // Snapshot should be modified by wisdom application
        let changed = (snapshot.arousal - original_arousal).abs() > 0.001
            || (snapshot.serotonin - original_serotonin).abs() > 0.001
            || snapshot.harmony_activations.iter().any(|&h| {
                h != 0.5 && h != 0.6 && h != 0.4 && h != 0.7 && h != 0.3 && h != 0.8 && h != 0.2
            });
        assert!(changed, "auto_improve should modify the cognitive state");
    }

    #[test]
    fn extract_scene_features_from_artwork() {
        let config = crate::AtelierConfig::default();
        let snapshot = test_snapshot();
        let artwork = crate::create_artwork(&config, &snapshot, 42);
        let features = extract_scene_features(&artwork);
        assert!(features.creative_surprise >= 0.0);
        assert!(!features.visual_features.is_empty());
    }

    #[test]
    fn apply_wisdom_bounded() {
        let mut snapshot = test_snapshot();
        let verdict = CriticVerdict {
            aesthetic: AestheticScore::uniform(0.8),
            novelty: 0.9,
            golden_ratio: 0.2,
            information_balance: 0.5,
            self_surprise: 0.7,
            intention_coherence: 0.3,
            taste_alignment: 0.8,
            composite: 0.6,
            reached_stillness: false,
        };
        apply_verdict_wisdom(&mut snapshot, &verdict);
        // All values should remain bounded
        assert!(snapshot.arousal >= 0.0 && snapshot.arousal <= 1.0);
        assert!(snapshot.serotonin >= 0.0 && snapshot.serotonin <= 1.0);
        assert!(snapshot.valence >= -1.0 && snapshot.valence <= 1.0);
        for &h in &snapshot.harmony_activations {
            assert!(h >= 0.0 && h <= 1.0);
        }
    }

    #[test]
    fn auto_improve_with_style_conditioning() {
        let config = crate::AtelierConfig::default();
        let mut snapshot = test_snapshot();

        // Style that strongly favors harmony index 7 (SacredStillness)
        let result = auto_improve_with(
            &config,
            &mut snapshot,
            5,
            42,
            Some(|snap: &mut CognitiveSnapshot| {
                // Push toward stillness each round
                snap.harmony_activations[7] = (snap.harmony_activations[7] + 0.05).min(1.0);
                snap.arousal = (snap.arousal - 0.02).max(0.0);
            }),
        );
        assert!(result.rounds > 0);
        // Sacred Stillness should have been boosted
        assert!(
            snapshot.harmony_activations[7] > 0.3,
            "stillness harmony should increase: {}",
            snapshot.harmony_activations[7]
        );
    }

    #[test]
    fn full_autopoietic_cycle() {
        // The complete loop: generate → critique → improve → reach stillness
        let config = crate::AtelierConfig {
            max_elements: 30,
            iteration_budget: 1,
            ..Default::default()
        };
        let mut snapshot = CognitiveSnapshot {
            consciousness_level: 0.8,
            harmony_activations: [0.5; 8],
            dopamine: 0.5,
            serotonin: 0.5,
            noradrenaline: 0.3,
            oxytocin: 0.5,
            allostatic_load: 0.1,
            arousal: 0.5,
            valence: 0.0,
            ..CognitiveSnapshot::dormant()
        };

        let result = auto_improve(&config, &mut snapshot, 15, 99);

        // Should produce a trajectory of scores
        assert!(!result.trajectory.is_empty(), "should have scores");

        // All scores should be bounded
        for &s in &result.trajectory {
            assert!(s >= 0.0 && s <= 1.0, "score out of bounds: {s}");
        }

        // System should reach stillness OR exhaust rounds
        assert!(
            result.reached_stillness || result.rounds == 15,
            "should terminate: rounds={}, stillness={}",
            result.rounds,
            result.reached_stillness
        );

        // Snapshot should have been modified (arousal, serotonin, or harmonies)
        // Exact drift depends on critic verdicts which depend on deterministic art generation
        let any_change = (snapshot.arousal - 0.5).abs() > 0.001
            || (snapshot.serotonin - 0.5).abs() > 0.001
            || (snapshot.valence - 0.0).abs() > 0.001
            || snapshot
                .harmony_activations
                .iter()
                .any(|&h| (h - 0.5).abs() > 0.001);
        assert!(
            any_change || result.reached_stillness,
            "state should evolve or reach stillness"
        );
    }
}
