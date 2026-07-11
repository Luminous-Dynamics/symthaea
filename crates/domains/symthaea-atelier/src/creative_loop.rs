// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! The closed creative loop: paint → perceive → dream → improve.
//!
//! This module wires together the four paradigm-shifting components:
//!
//! ```text
//! 1. NeuralPainter creates art from CognitiveSnapshot
//! 2. PerceptualEncoder perceives the art (self-perception)
//! 3. AestheticDreamEngine records + counterfactually replays
//! 4. DreamWisdom feeds back to evolve the cognitive state
//! 5. Repeat → the system's art improves through its own aesthetic experience
//! ```
//!
//! This is not a training loop (gradient descent). It's a *living* creative
//! process: the system paints, sees what it painted, is surprised or satisfied,
//! dreams about alternatives, and wakes up changed.

use symthaea_aesthetic::{AestheticConfig, AestheticTracker};
use symthaea_canvas::CognitiveSnapshot;
use symthaea_core::genesis::GenesisSeed;

use crate::dream_art::{AestheticDreamEngine, DreamWisdom};
use crate::pixel_canvas::{NeuralPainter, PixelCanvas};
use crate::self_perception::{PerceptualEncoder, SelfPerception};

/// Result of one creative cycle.
#[derive(Debug)]
pub struct CreativeCycleResult {
    /// The artwork produced this cycle.
    pub canvas: PixelCanvas,
    /// Self-perception of the artwork.
    pub perception: SelfPerception,
    /// Aesthetic composite score.
    pub aesthetic_score: f32,
    /// Creative surprise from self-perception.
    pub creative_surprise: f32,
    /// Perceptual coherence (intention vs result).
    pub perceptual_coherence: f32,
    /// Dopamine feedback signal.
    pub dopamine_delta: f32,
}

/// The closed creative loop: all components wired together.
pub struct CreativeLoop {
    painter: NeuralPainter,
    encoder: PerceptualEncoder,
    dream_engine: AestheticDreamEngine,
    tracker: AestheticTracker,
    genesis: GenesisSeed,
    /// Canvas dimensions.
    width: u32,
    height: u32,
    /// Strokes per painting.
    strokes: usize,
    /// Cycles completed.
    cycle_count: u64,
}

impl CreativeLoop {
    pub fn new(width: u32, height: u32, strokes: usize) -> Self {
        let genesis = GenesisSeed::from_phrase("creative-loop");
        Self {
            painter: NeuralPainter::new(&genesis),
            encoder: PerceptualEncoder::new(&genesis),
            dream_engine: AestheticDreamEngine::new(30),
            tracker: AestheticTracker::new(AestheticConfig::default()),
            genesis,
            width,
            height,
            strokes,
            cycle_count: 0,
        }
    }

    /// Run one creative cycle: paint → perceive → evaluate → record.
    pub fn cycle(&mut self, snapshot: &CognitiveSnapshot) -> CreativeCycleResult {
        self.cycle_count += 1;

        // 1. Paint
        let (canvas, _strokes) =
            self.painter
                .paint(snapshot, self.width, self.height, self.strokes);

        // 2. Perceive own art
        let intention = NeuralPainter::encode_snapshot(snapshot, &self.genesis);
        let perception = self.encoder.perceive_own_art(&canvas, &intention);

        // 3. Score aesthetically
        // Use the self-perception features to build a score
        let mean_color = canvas.mean_color();
        let color_diversity = (mean_color[0] - mean_color[1]).abs()
            + (mean_color[1] - mean_color[2]).abs()
            + (mean_color[0] - mean_color[2]).abs();

        let mut aesthetic_score = symthaea_aesthetic::AestheticScore {
            order: perception.perceptual_coherence,
            complexity: (color_diversity / 3.0).clamp(0.0, 1.0),
            surprise: perception.creative_surprise,
            harmony: snapshot.harmony_activations.iter().sum::<f32>() / 8.0,
            birkhoff: 0.0,
            composite: 0.0,
        };
        aesthetic_score.birkhoff = if aesthetic_score.complexity > 0.01 {
            (aesthetic_score.order / aesthetic_score.complexity).clamp(0.0, 1.0)
        } else {
            0.0
        };
        aesthetic_score.compute_composite();

        // 4. Feedback
        let feedback = self
            .tracker
            .process(&aesthetic_score, &snapshot.harmony_activations);

        // 5. Record for dream replay
        self.dream_engine.record(snapshot.clone(), aesthetic_score);

        let creative_surprise = perception.creative_surprise;
        let perceptual_coherence = perception.perceptual_coherence;

        CreativeCycleResult {
            canvas,
            perception,
            aesthetic_score: aesthetic_score.composite,
            creative_surprise,
            perceptual_coherence,
            dopamine_delta: feedback.dopamine_delta,
        }
    }

    /// Run a dream cycle: counterfactual replay of past artworks.
    /// Returns wisdoms learned and optionally the best dream artwork.
    pub fn dream(&mut self, counterfactuals: usize) -> Vec<DreamWisdom> {
        let (wisdoms, _best) = self.dream_engine.dream(
            counterfactuals,
            &mut self.painter,
            (self.width, self.height),
        );
        wisdoms
    }

    /// Apply dream wisdom to evolve a cognitive snapshot.
    ///
    /// This is how the dream loop closes: wisdom discovered during sleep
    /// modifies the cognitive state for the next waking cycle.
    pub fn apply_wisdom(snapshot: &mut CognitiveSnapshot, wisdoms: &[DreamWisdom]) {
        for w in wisdoms {
            let delta = w.direction * w.improvement * w.confidence * 0.5;
            match w.dimension.as_str() {
                "valence" => snapshot.valence = (snapshot.valence + delta).clamp(-1.0, 1.0),
                "arousal" => snapshot.arousal = (snapshot.arousal + delta).clamp(0.0, 1.0),
                "dopamine" => snapshot.dopamine = (snapshot.dopamine + delta).clamp(0.0, 1.0),
                "serotonin" => snapshot.serotonin = (snapshot.serotonin + delta).clamp(0.0, 1.0),
                "consciousness" => {
                    snapshot.consciousness_level =
                        (snapshot.consciousness_level + delta as f64).clamp(0.0, 1.0)
                }
                dim if dim.starts_with("harmony_") => {
                    if let Ok(idx) = dim[8..].parse::<usize>() {
                        if idx < 8 {
                            snapshot.harmony_activations[idx] =
                                (snapshot.harmony_activations[idx] + delta).clamp(0.0, 1.0);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Current aesthetic expectation (EMA).
    pub fn aesthetic_expectation(&self) -> f32 {
        self.tracker.expectation()
    }

    /// Total cycles completed.
    pub fn cycle_count(&self) -> u64 {
        self.cycle_count
    }

    /// Dream wisdoms accumulated.
    pub fn accumulated_wisdom(&self) -> &[DreamWisdom] {
        self.dream_engine.wisdom()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn single_cycle_produces_result() {
        let mut loop_ = CreativeLoop::new(64, 64, 20);
        let snap = test_snapshot();
        let result = loop_.cycle(&snap);

        assert!(result.aesthetic_score >= 0.0 && result.aesthetic_score <= 1.0);
        assert!(result.creative_surprise >= 0.0);
        assert!(result.perceptual_coherence >= 0.0);
        assert_eq!(result.canvas.width, 64);
        assert_eq!(loop_.cycle_count(), 1);
    }

    #[test]
    fn multi_cycle_tracks_ema() {
        let mut loop_ = CreativeLoop::new(64, 64, 15);
        let snap = test_snapshot();

        let r1 = loop_.cycle(&snap);
        let ema1 = loop_.aesthetic_expectation();

        let r2 = loop_.cycle(&snap);
        let ema2 = loop_.aesthetic_expectation();

        // EMA should update
        assert_ne!(ema1, ema2);
        assert_eq!(loop_.cycle_count(), 2);

        let _ = (r1, r2);
    }

    #[test]
    fn dream_after_cycles() {
        let mut loop_ = CreativeLoop::new(64, 64, 15);
        let snap = test_snapshot();

        // Create some memories
        for _ in 0..3 {
            loop_.cycle(&snap);
        }

        // Dream
        let wisdoms = loop_.dream(3);
        // May or may not find improvements, but shouldn't panic
        let _ = wisdoms;
    }

    #[test]
    fn apply_wisdom_modifies_snapshot() {
        let mut snap = test_snapshot();
        let original_valence = snap.valence;

        let wisdoms = vec![DreamWisdom {
            dimension: "valence".to_string(),
            direction: 1.0,
            improvement: 0.2,
            confidence: 0.8,
        }];

        CreativeLoop::apply_wisdom(&mut snap, &wisdoms);
        assert!(
            snap.valence > original_valence,
            "wisdom should increase valence"
        );
    }

    #[test]
    fn full_loop_paint_perceive_dream_evolve() {
        let mut loop_ = CreativeLoop::new(64, 64, 15);
        let mut snap = test_snapshot();

        let mut scores = Vec::new();

        // 5 waking cycles
        for _ in 0..5 {
            let result = loop_.cycle(&snap);
            scores.push(result.aesthetic_score);
        }

        // Dream
        let wisdoms = loop_.dream(4);

        // Apply wisdom
        CreativeLoop::apply_wisdom(&mut snap, &wisdoms);

        // 5 more waking cycles with evolved state
        for _ in 0..5 {
            let result = loop_.cycle(&snap);
            scores.push(result.aesthetic_score);
        }

        // System should have 10 total cycles
        assert_eq!(loop_.cycle_count(), 10);
        assert_eq!(scores.len(), 10);

        // Scores should be bounded
        for s in &scores {
            assert!(*s >= 0.0 && *s <= 1.0, "score out of bounds: {s}");
        }
    }
}
