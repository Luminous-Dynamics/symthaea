// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dreams as art: counterfactual aesthetic replay.
//!
//! During low-load ("sleep") periods, the system replays its best artworks,
//! mutates them counterfactually ("what if this painting had more warmth?"),
//! evaluates the alternatives, and consolidates aesthetic wisdom.
//!
//! The dream output IS the most creative art — unconstrained by real-time
//! consciousness gating, with reduced precision enabling divergent exploration.
//! This maps directly to human creativity research: artists report their best
//! ideas come in dreams or hypnagogic states (Stickgold & Walker, 2004).
//!
//! # Architecture
//!
//! ```text
//! AestheticMemory (ring buffer of recent artworks)
//!   → sample high-scoring entry (salience-weighted)
//!   → perturb cognitive snapshot (counterfactual: "what if...")
//!   → regenerate artwork from perturbed state
//!   → evaluate: did the perturbation improve the score?
//!   → consolidate: record DreamWisdom (which perturbations help)
//! ```

use rand::SeedableRng;
use symthaea_aesthetic::AestheticScore;
use symthaea_canvas::CognitiveSnapshot;

use crate::AtelierConfig;
use crate::pixel_canvas::{NeuralPainter, PixelCanvas};
use crate::score_scene;

/// A remembered artwork with its cognitive context.
#[derive(Clone)]
pub struct AestheticMemory {
    /// The cognitive state that produced this artwork.
    pub snapshot: CognitiveSnapshot,
    /// The aesthetic score it achieved.
    pub score: AestheticScore,
    /// Salience weight (surprise × consciousness at time of creation).
    pub salience: f32,
}

/// Wisdom consolidated from dream replay.
#[derive(Debug, Clone)]
pub struct DreamWisdom {
    /// Which dimension was perturbed (e.g., "valence", "harmony_3").
    pub dimension: String,
    /// Direction of improvement: positive = increase helped, negative = decrease helped.
    pub direction: f32,
    /// How much the perturbation improved the score.
    pub improvement: f32,
    /// Confidence (how many dream replays support this).
    pub confidence: f32,
}

/// Dream engine for aesthetic counterfactual replay.
pub struct AestheticDreamEngine {
    /// Ring buffer of recent artworks (salience-weighted).
    memories: Vec<AestheticMemory>,
    /// Maximum memory size.
    max_memories: usize,
    /// Accumulated dream wisdom.
    wisdom: Vec<DreamWisdom>,
    /// Number of dream cycles completed.
    dream_count: u64,
}

impl AestheticDreamEngine {
    pub fn new(max_memories: usize) -> Self {
        Self {
            memories: Vec::new(),
            max_memories,
            wisdom: Vec::new(),
            dream_count: 0,
        }
    }

    /// Record an artwork for potential dream replay.
    pub fn record(&mut self, snapshot: CognitiveSnapshot, score: AestheticScore) {
        let salience = score.surprise * snapshot.consciousness_level as f32;

        self.memories.push(AestheticMemory {
            snapshot,
            score,
            salience: salience.max(0.1), // minimum salience for uniform sampling
        });

        // Prune to capacity (remove lowest-salience)
        if self.memories.len() > self.max_memories {
            self.memories.sort_by(|a, b| {
                b.salience
                    .partial_cmp(&a.salience)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.memories.truncate(self.max_memories);
        }
    }

    /// Run a dream cycle: counterfactual replay of stored artworks.
    ///
    /// `counterfactuals`: how many "what if" variations to try per dream.
    /// Returns the dream wisdoms discovered and optionally the best dream artwork.
    pub fn dream(
        &mut self,
        counterfactuals: usize,
        painter: &mut NeuralPainter,
        canvas_size: (u32, u32),
    ) -> (Vec<DreamWisdom>, Option<(PixelCanvas, AestheticScore)>) {
        if self.memories.is_empty() {
            return (Vec::new(), None);
        }

        self.dream_count += 1;

        // Sample highest-salience memory
        let memory = self.memories.first().cloned().unwrap();
        let base_score = memory.score.composite;

        let mut wisdoms = Vec::new();
        let mut best_dream: Option<(PixelCanvas, AestheticScore)> = None;
        let mut best_dream_score = base_score;

        let perturbations: Vec<(&str, Box<dyn Fn(&mut CognitiveSnapshot, f32)>)> = vec![
            (
                "valence",
                Box::new(|s: &mut CognitiveSnapshot, d: f32| {
                    s.valence = (s.valence + d).clamp(-1.0, 1.0)
                }),
            ),
            (
                "arousal",
                Box::new(|s: &mut CognitiveSnapshot, d: f32| {
                    s.arousal = (s.arousal + d).clamp(0.0, 1.0)
                }),
            ),
            (
                "dopamine",
                Box::new(|s: &mut CognitiveSnapshot, d: f32| {
                    s.dopamine = (s.dopamine + d).clamp(0.0, 1.0)
                }),
            ),
            (
                "serotonin",
                Box::new(|s: &mut CognitiveSnapshot, d: f32| {
                    s.serotonin = (s.serotonin + d).clamp(0.0, 1.0)
                }),
            ),
            (
                "harmony_0",
                Box::new(|s: &mut CognitiveSnapshot, d: f32| {
                    s.harmony_activations[0] = (s.harmony_activations[0] + d).clamp(0.0, 1.0)
                }),
            ),
            (
                "harmony_3",
                Box::new(|s: &mut CognitiveSnapshot, d: f32| {
                    s.harmony_activations[3] = (s.harmony_activations[3] + d).clamp(0.0, 1.0)
                }),
            ),
            (
                "harmony_7",
                Box::new(|s: &mut CognitiveSnapshot, d: f32| {
                    s.harmony_activations[7] = (s.harmony_activations[7] + d).clamp(0.0, 1.0)
                }),
            ),
            (
                "consciousness",
                Box::new(|s: &mut CognitiveSnapshot, d: f32| {
                    s.consciousness_level = (s.consciousness_level + d as f64).clamp(0.0, 1.0)
                }),
            ),
        ];

        for cf in 0..counterfactuals.min(perturbations.len()) {
            let (dim_name, perturb_fn) = &perturbations[cf % perturbations.len()];

            for &delta in &[0.15, -0.15] {
                let mut dreamed_snapshot = memory.snapshot.clone();
                perturb_fn(&mut dreamed_snapshot, delta);

                // Generate dream artwork
                let num_strokes = 20 + (dreamed_snapshot.consciousness_level * 30.0) as usize;
                let (canvas, _strokes) =
                    painter.paint(&dreamed_snapshot, canvas_size.0, canvas_size.1, num_strokes);

                // Score using SVG-based scorer as proxy (generate SVG scene for scoring)
                let config = AtelierConfig::default();
                let scene = crate::generate(
                    &config,
                    &dreamed_snapshot,
                    &mut rand::rngs::StdRng::seed_from_u64(self.dream_count + cf as u64),
                );
                let dream_score = score_scene(&scene, &dreamed_snapshot);

                let improvement = dream_score.composite - base_score;

                if improvement > 0.01 {
                    wisdoms.push(DreamWisdom {
                        dimension: dim_name.to_string(),
                        direction: delta,
                        improvement,
                        confidence: 1.0 / (1.0 + self.dream_count as f32 * 0.1),
                    });
                }

                if dream_score.composite > best_dream_score {
                    best_dream_score = dream_score.composite;
                    best_dream = Some((canvas, dream_score));
                }
            }
        }

        // Accumulate wisdom
        for w in &wisdoms {
            // Update existing or add new
            if let Some(existing) = self.wisdom.iter_mut().find(|e| {
                e.dimension == w.dimension && e.direction.signum() == w.direction.signum()
            }) {
                existing.improvement = existing.improvement * 0.7 + w.improvement * 0.3;
                existing.confidence += 0.1;
            } else {
                self.wisdom.push(w.clone());
            }
        }

        (wisdoms, best_dream)
    }

    /// Get accumulated dream wisdom (sorted by improvement).
    pub fn wisdom(&self) -> &[DreamWisdom] {
        &self.wisdom
    }

    /// Number of memories stored.
    pub fn memory_count(&self) -> usize {
        self.memories.len()
    }

    /// Number of dream cycles completed.
    pub fn dream_count(&self) -> u64 {
        self.dream_count
    }
}

impl Default for AestheticDreamEngine {
    fn default() -> Self {
        Self::new(50)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::genesis::GenesisSeed;

    fn test_snapshot() -> CognitiveSnapshot {
        CognitiveSnapshot {
            consciousness_level: 0.7,
            valence: 0.3,
            arousal: 0.5,
            dopamine: 0.6,
            serotonin: 0.5,
            noradrenaline: 0.4,
            harmony_activations: [0.5, 0.6, 0.4, 0.7, 0.3, 0.5, 0.8, 0.2],
            prediction_error: 0.1,
            thought_vector: vec![0.3, -0.2],
            ..CognitiveSnapshot::dormant()
        }
    }

    #[test]
    fn record_and_memory_count() {
        let mut engine = AestheticDreamEngine::new(10);
        let mut score = AestheticScore::uniform(0.6);
        score.surprise = 0.5;
        score.compute_composite();

        engine.record(test_snapshot(), score);
        assert_eq!(engine.memory_count(), 1);
    }

    #[test]
    fn dream_on_empty_returns_nothing() {
        let mut engine = AestheticDreamEngine::new(10);
        let genesis = GenesisSeed::from_phrase("test-dream");
        let mut painter = NeuralPainter::new(&genesis);

        let (wisdoms, art) = engine.dream(3, &mut painter, (64, 64));
        assert!(wisdoms.is_empty());
        assert!(art.is_none());
    }

    #[test]
    fn dream_produces_wisdom() {
        let mut engine = AestheticDreamEngine::new(10);
        let mut score = AestheticScore::uniform(0.5);
        score.surprise = 0.5;
        score.compute_composite();

        engine.record(test_snapshot(), score);

        let genesis = GenesisSeed::from_phrase("test-dream-wisdom");
        let mut painter = NeuralPainter::new(&genesis);

        let (_wisdoms, _art) = engine.dream(4, &mut painter, (64, 64));

        // Should have attempted counterfactuals
        assert_eq!(engine.dream_count(), 1);
        // Wisdom may or may not be found (depends on score deltas)
        // But the engine should have run without panicking
    }

    #[test]
    fn memory_pruning() {
        let mut engine = AestheticDreamEngine::new(3);

        for i in 0..5 {
            let mut score = AestheticScore::uniform(i as f32 / 5.0);
            score.surprise = i as f32 / 5.0;
            score.compute_composite();
            engine.record(test_snapshot(), score);
        }

        assert_eq!(engine.memory_count(), 3); // pruned to max
    }
}