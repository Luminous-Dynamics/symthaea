//! # Wisdom Module - Philosophical Grounding for Computational Cognition
//!
//! This module bridges the philosophical foundations of Evolving Resonant Co-creationism
//! to Symthaea's computational architecture. It provides:
//!
//! - **Harmonics**: The Seven Harmonies as operational reasoning modes
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
//! The Seven Harmonies aren't just tracked values; they are reasoning modes
//! that bias thought in specific directions.

pub mod harmonics;
pub mod autopoiesis;
pub mod meta_cognition;

pub use harmonics::{
    HarmonicMode, HarmonicProfile, ReasoningBias,
    ActiveHarmonic, HarmonicQuestion,
};
pub use autopoiesis::{
    AutopoieticMonitor, OperationalClosure, SelfProductionMetrics,
};
pub use meta_cognition::{
    MetaCognitiveLayer, RecursiveModel, SelfModelAccuracy,
};

/// The core question each harmony asks of any situation
pub const HARMONIC_QUESTIONS: [&str; 7] = [
    "Does this hang together?",           // Coherence
    "Does this serve flourishing?",       // Flourishing
    "What don't I know?",                 // Wisdom
    "What haven't I tried?",              // Play
    "How is this connected?",             // Interconnect
    "What am I giving/receiving?",        // Reciprocity
    "How does this help us grow?",        // Evolution
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
    pub fn update_from_experience(&mut self, prediction_error: f32, uncertainty: f32, coherence: f32) {
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
}
