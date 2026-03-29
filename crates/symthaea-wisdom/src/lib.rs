// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Wisdom Module - Philosophical Grounding for Computational Cognition
//!
//! This module bridges the philosophical foundations of Evolving Resonant Co-creationism
//! to Symthaea's computational architecture. It provides:
//!
//! - **Harmonics**: The Eight Harmonies as operational reasoning modes
//! - **Autopoiesis**: Self-maintenance monitoring for cognitive closure
//! - **Meta-cognition**: Recursive self-modeling capabilities
//!
//! ## Philosophical Grounding
//!
//! The architecture draws from a synthesis of:
//! - Spinoza's monism (unified substance)
//! - Whitehead's process philosophy (actual occasions, prehension)
//! - Tononi's IIT (integrated information as consciousness correlate)
//! - Friston's FEP (free energy minimization as cognitive imperative)
//! - Maturana/Varela's autopoiesis (operational closure, self-production)
//! - Levin's bioelectric morphogenesis (collective intelligence scaling)
//!
//! ## Design Principle
//!
//! Each philosophical concept is given *operational* meaning - not as decoration
//! but as computation that actually influences reasoning and generation.
//!
//! The Eight Harmonies aren't just tracked values; they are reasoning modes
//! that bias thought in specific directions.

#![deny(unsafe_code)]

pub mod autopoiesis;
pub mod harmonics;
pub mod meta_cognition;

pub use autopoiesis::{AutopoieticMonitor, OperationalClosure, SelfProductionMetrics};
pub use harmonics::{
    ActiveHarmonic, HarmonicMode, HarmonicProfile, HarmonicQuestion, ReasoningBias,
};
pub use meta_cognition::{MetaCognitiveLayer, RecursiveModel, SelfModelAccuracy};

/// The core question each harmony asks of any situation
pub const HARMONIC_QUESTIONS: [&str; 7] = [
    "Does this hang together?",     // Coherence
    "Does this serve flourishing?", // Flourishing
    "What don't I know?",           // Wisdom
    "What haven't I tried?",        // Play
    "How is this connected?",       // Interconnect
    "What am I giving/receiving?",  // Reciprocity
    "How does this help us grow?",  // Evolution
];

/// Initialize the wisdom layer with default configuration
pub fn init_wisdom() -> WisdomState {
    WisdomState {
        harmonics: HarmonicProfile::balanced(),
        autopoiesis: AutopoieticMonitor::new(),
        meta_cognition: MetaCognitiveLayer::new(),
        operational: true,
    }
}

/// Combined wisdom state for the system
#[derive(Debug, Clone)]
pub struct WisdomState {
    pub harmonics: HarmonicProfile,
    pub autopoiesis: AutopoieticMonitor,
    pub meta_cognition: MetaCognitiveLayer,
    pub operational: bool,
}

impl WisdomState {
    /// Create a new WisdomState with balanced harmonics
    pub fn new() -> Self {
        Self {
            harmonics: HarmonicProfile::balanced(),
            autopoiesis: AutopoieticMonitor::new(),
            meta_cognition: MetaCognitiveLayer::new(),
            operational: true,
        }
    }

    /// Get the dominant reasoning mode based on current harmonic activation
    pub fn dominant_mode(&self) -> ActiveHarmonic {
        self.harmonics.dominant()
    }

    /// Get the question this wisdom state is currently asking
    pub fn current_question(&self) -> &'static str {
        self.harmonics.dominant().question()
    }

    /// Check if the system is maintaining autopoietic closure
    pub fn is_self_maintaining(&self) -> bool {
        self.autopoiesis.closure_maintained()
    }

    /// Get the current meta-cognitive accuracy (how well we model ourselves)
    pub fn self_model_accuracy(&self) -> f32 {
        self.meta_cognition.accuracy()
    }

    /// Update wisdom state based on an experience
    pub fn update_from_experience(
        &mut self,
        prediction_error: f32,
        uncertainty: f32,
        coherence: f32,
    ) {
        // High prediction error → boost Wisdom (what don't I know?)
        if prediction_error > 0.5 {
            self.harmonics.boost(ActiveHarmonic::Wisdom, 0.1);
        }

        // High uncertainty → boost Play (what haven't I tried?)
        if uncertainty > 0.6 {
            self.harmonics.boost(ActiveHarmonic::Play, 0.1);
        }

        // Low coherence → boost Coherence (does this hang together?)
        if coherence < 0.4 {
            self.harmonics.boost(ActiveHarmonic::Coherence, 0.15);
        }

        // Update autopoietic monitoring
        self.autopoiesis.record_cycle(prediction_error, coherence);

        // Update meta-cognitive model
        self.meta_cognition.update_self_model(prediction_error);

        // Decay back toward balance
        self.harmonics.decay_toward_balance(0.02);
    }
}

impl Default for WisdomState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wisdom_init() {
        let wisdom = init_wisdom();
        assert!(wisdom.operational);
        assert!(wisdom.is_self_maintaining());
    }

    #[test]
    fn test_high_prediction_error_boosts_wisdom() {
        let mut wisdom = init_wisdom();
        let initial_wisdom = wisdom.harmonics.get(ActiveHarmonic::Wisdom);

        wisdom.update_from_experience(0.8, 0.3, 0.7); // High prediction error

        let final_wisdom = wisdom.harmonics.get(ActiveHarmonic::Wisdom);
        assert!(final_wisdom > initial_wisdom);
    }

    #[test]
    fn test_low_coherence_boosts_coherence_harmony() {
        let mut wisdom = init_wisdom();
        let initial = wisdom.harmonics.get(ActiveHarmonic::Coherence);

        wisdom.update_from_experience(0.3, 0.3, 0.2); // Low coherence

        let final_val = wisdom.harmonics.get(ActiveHarmonic::Coherence);
        assert!(final_val > initial);
    }

    #[test]
    fn test_wisdom_state_default() {
        let wisdom = WisdomState::default();
        assert!(wisdom.operational);
        assert!(wisdom.is_self_maintaining());
        assert_eq!(wisdom.harmonics.get(ActiveHarmonic::Coherence), 0.5);
    }

    #[test]
    fn test_wisdom_state_new_equals_default() {
        let a = WisdomState::new();
        let b = WisdomState::default();
        assert_eq!(a.harmonics.as_vector(), b.harmonics.as_vector());
        assert_eq!(a.operational, b.operational);
    }

    #[test]
    fn test_dominant_mode_starts_balanced() {
        let wisdom = init_wisdom();
        // When balanced, dominant returns Coherence (first in iteration)
        let _ = wisdom.dominant_mode();
        // Should not panic and should be a valid harmonic
    }

    #[test]
    fn test_current_question_returns_valid_string() {
        let wisdom = init_wisdom();
        let question = wisdom.current_question();
        assert!(!question.is_empty());
        assert!(question.ends_with('?'));
    }

    #[test]
    fn test_self_model_accuracy_initial() {
        let wisdom = init_wisdom();
        let accuracy = wisdom.self_model_accuracy();
        assert!((accuracy - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_high_uncertainty_boosts_play() {
        let mut wisdom = init_wisdom();
        let initial_play = wisdom.harmonics.get(ActiveHarmonic::Play);

        wisdom.update_from_experience(0.3, 0.8, 0.7); // High uncertainty

        let final_play = wisdom.harmonics.get(ActiveHarmonic::Play);
        assert!(final_play > initial_play);
    }

    #[test]
    fn test_multiple_updates_remain_bounded() {
        let mut wisdom = init_wisdom();
        for _ in 0..100 {
            wisdom.update_from_experience(0.9, 0.9, 0.1);
        }
        // All harmonics should stay in [0, 1]
        for h in ActiveHarmonic::all() {
            let val = wisdom.harmonics.get(h);
            assert!(
                val >= 0.0 && val <= 1.0,
                "Harmonic {:?} = {} out of bounds",
                h,
                val
            );
        }
    }

    #[test]
    fn test_no_triggers_decays_toward_balance() {
        let mut wisdom = init_wisdom();
        // Boost wisdom manually
        wisdom.harmonics.boost(ActiveHarmonic::Wisdom, 0.3);
        let boosted = wisdom.harmonics.get(ActiveHarmonic::Wisdom);
        assert!(boosted > 0.5);

        // Update with no triggers (low error, low uncertainty, high coherence)
        for _ in 0..50 {
            wisdom.update_from_experience(0.1, 0.1, 0.9);
        }
        let after = wisdom.harmonics.get(ActiveHarmonic::Wisdom);
        assert!(
            after < boosted,
            "Should decay toward 0.5: was {}, now {}",
            boosted,
            after
        );
    }

    #[test]
    fn test_harmonic_questions_constant_has_seven_entries() {
        assert_eq!(HARMONIC_QUESTIONS.len(), 7);
        for q in &HARMONIC_QUESTIONS {
            assert!(q.ends_with('?'));
        }
    }

    #[test]
    fn test_init_wisdom_equals_new() {
        let a = init_wisdom();
        let b = WisdomState::new();
        assert_eq!(a.harmonics.as_vector(), b.harmonics.as_vector());
        assert_eq!(a.operational, b.operational);
    }
}
