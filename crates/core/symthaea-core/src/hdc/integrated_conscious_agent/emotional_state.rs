// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Emotional State - Valence-Arousal Model

use std::collections::VecDeque;

/// Emotional state using the valence-arousal model
#[derive(Clone, Debug)]
pub struct EmotionalState {
    /// Valence: -1 (negative) to +1 (positive)
    pub valence: f64,
    /// Arousal: 0 (calm) to 1 (excited)
    pub arousal: f64,
    /// Dominance: 0 (submissive) to 1 (dominant)
    pub dominance: f64,
    /// Emotional momentum (how quickly emotions change)
    momentum: f64,
    /// Recent emotional history
    history: VecDeque<(f64, f64)>,
}

impl EmotionalState {
    pub fn new() -> Self {
        Self {
            valence: 0.0,   // Neutral
            arousal: 0.3,   // Slightly calm
            dominance: 0.5, // Balanced
            momentum: 0.1,  // Slow emotional changes
            history: VecDeque::with_capacity(20),
        }
    }

    /// Update emotional state based on experience
    pub fn update(&mut self, phi: f64, prediction_error: f64, goal_progress: f64) {
        // Store current state in history
        self.history.push_back((self.valence, self.arousal));
        if self.history.len() > 20 {
            self.history.pop_front();
        }

        // Compute target emotional state
        // High Φ and goal progress → positive valence
        let target_valence = (phi - 0.4) * 2.0 + (goal_progress - 0.5) * 0.5;

        // High prediction error → high arousal (surprise)
        let target_arousal = 0.3 + prediction_error * 0.7;

        // High Φ → higher dominance (sense of control)
        let target_dominance = 0.3 + phi * 0.5;

        // Smooth transition based on momentum
        self.valence += (target_valence.clamp(-1.0, 1.0) - self.valence) * self.momentum;
        self.arousal += (target_arousal.clamp(0.0, 1.0) - self.arousal) * self.momentum;
        self.dominance += (target_dominance.clamp(0.0, 1.0) - self.dominance) * self.momentum;
    }

    /// Get the current emotional label
    pub fn label(&self) -> &'static str {
        // Valence-Arousal quadrant mapping
        match (self.valence > 0.0, self.arousal > 0.5) {
            (true, true) => "excited/happy",
            (true, false) => "calm/content",
            (false, true) => "stressed/anxious",
            (false, false) => "sad/bored",
        }
    }

    /// Get emotional stability (how consistent emotions have been)
    pub fn stability(&self) -> f64 {
        if self.history.len() < 2 {
            return 1.0;
        }

        let variance: f64 = self
            .history
            .iter()
            .map(|(v, a)| (v - self.valence).powi(2) + (a - self.arousal).powi(2))
            .sum::<f64>()
            / self.history.len() as f64;

        (1.0 - variance.sqrt()).max(0.0)
    }

    /// Check if emotional state is conducive to deep processing
    pub fn conducive_to_processing(&self) -> bool {
        // Moderate arousal and positive valence are best for cognition
        self.arousal > 0.2 && self.arousal < 0.8 && self.valence > -0.5
    }

    /// Apply hormone-based modulation to emotional state
    ///
    /// This integrates EndocrineSystem chemical signals into the
    /// agent's felt emotional experience
    pub fn apply_hormone_modulation(
        &mut self,
        valence_effect: f32,
        arousal_effect: f32,
        focus_effect: f32,
    ) {
        // Hormones are slow-moving, so use gentle integration
        let hormone_weight = 0.3; // 30% hormone influence per cycle

        // Blend hormone effects with current emotional state
        self.valence = (self.valence * (1.0 - hormone_weight as f64)
            + valence_effect as f64 * hormone_weight as f64)
            .clamp(-1.0, 1.0);

        self.arousal = (self.arousal * (1.0 - hormone_weight as f64)
            + (self.arousal + arousal_effect as f64) * hormone_weight as f64)
            .clamp(0.0, 1.0);

        // Focus affects dominance (sense of control)
        self.dominance = (self.dominance * (1.0 - hormone_weight as f64)
            + focus_effect as f64 * hormone_weight as f64)
            .clamp(0.0, 1.0);
    }

    /// Get the emotional quadrant based on valence and arousal
    ///
    /// Returns one of: "excited", "calm", "stressed", "sad"
    pub fn get_emotion_quadrant(&self) -> &'static str {
        match (self.valence > 0.0, self.arousal > 0.5) {
            (true, true) => "excited",
            (true, false) => "calm",
            (false, true) => "stressed",
            (false, false) => "sad",
        }
    }
}

impl Default for EmotionalState {
    fn default() -> Self {
        Self::new()
    }
}