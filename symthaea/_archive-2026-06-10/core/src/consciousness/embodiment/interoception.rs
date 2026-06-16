// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Interoceptive tags and inference engine for internal-state memory.
//!
//! ## Interoceptive Inference (Seth 2021)
//!
//! Consciousness is fundamentally about predicting the body's internal states.
//! When `interoceptive_inference` is enabled, interoceptive signals (valence,
//! arousal, physiological stress) are elevated to first-class FEP observations
//! that modulate precision and the epistemic/pragmatic tradeoff in expected
//! free energy.
//!
//! ## References
//!
//! - Seth, A.K. (2021). *Being You: A New Science of Consciousness*. Dutton.
//! - Seth, A.K., & Tsakiris, M. (2018). "Being a Beast Machine." *Trends Cog Sci*.
//! - Barrett, L.F. (2017). *How Emotions Are Made*. Houghton Mifflin.

/// Canonical interoceptive tags used for memory topics and recall routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InteroceptionTag {
    /// CPU/memory metabolic load — maps from cycle throughput and error trend
    ThermodynamicLoad,
    /// Infrastructure stress — maps from lock poisoning, task panics, DB failures
    InfrastructureStress,
    /// Arousal level — activation state of the system
    #[cfg(feature = "interoceptive_inference")]
    Arousal,
    /// Emotional valence — pleasure/displeasure dimension
    #[cfg(feature = "interoceptive_inference")]
    Valence,
    /// Gut feeling — fast interoceptive signal (Barrett 2017)
    #[cfg(feature = "interoceptive_inference")]
    GutFeeling,
}

impl InteroceptionTag {
    pub fn as_topic(self) -> &'static str {
        match self {
            InteroceptionTag::ThermodynamicLoad => "interoception:thermodynamic_load",
            InteroceptionTag::InfrastructureStress => "interoception:infrastructure_stress",
            #[cfg(feature = "interoceptive_inference")]
            InteroceptionTag::Arousal => "interoception:arousal",
            #[cfg(feature = "interoceptive_inference")]
            InteroceptionTag::Valence => "interoception:valence",
            #[cfg(feature = "interoceptive_inference")]
            InteroceptionTag::GutFeeling => "interoception:gut_feeling",
        }
    }

    pub fn as_label(self) -> &'static str {
        match self {
            InteroceptionTag::ThermodynamicLoad => "ThermodynamicLoad",
            InteroceptionTag::InfrastructureStress => "InfrastructureStress",
            #[cfg(feature = "interoceptive_inference")]
            InteroceptionTag::Arousal => "Arousal",
            #[cfg(feature = "interoceptive_inference")]
            InteroceptionTag::Valence => "Valence",
            #[cfg(feature = "interoceptive_inference")]
            InteroceptionTag::GutFeeling => "GutFeeling",
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTEROCEPTIVE INFERENCE ENGINE (Seth 2021)
// ═══════════════════════════════════════════════════════════════════════════════

/// Interoceptive inference engine.
///
/// Computes interoceptive prediction error — the mismatch between predicted
/// body state and actual body state. This PE feeds into the hierarchical free
/// energy framework as an additional precision-modulating signal.
///
/// The engine also computes modulation factors for:
/// - **Precision modulation** (`precision_factor`): valence-driven confidence adjustment
/// - **Epistemic/pragmatic balance** (`epistemic_weight_shift`): arousal shifts exploration
/// - **Blanket permeability** (`blanket_stress_signal`): physiological stress closes the blanket
///
/// Science: Seth (2021) argues that consciousness is fundamentally about
/// *interoceptive inference* — predicting and controlling the body's internal
/// milieu. The "beast machine" theory places interoception on equal footing
/// with exteroception in the brain's generative model.
#[cfg(feature = "interoceptive_inference")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InteroceptiveInferenceEngine {
    /// Predicted valence [-1, 1]
    predicted_valence: f64,
    /// Predicted arousal [0, 1]
    predicted_arousal: f64,
    /// Interoceptive prediction error (EMA smoothed)
    pub interoceptive_pe: f64,
    /// Precision modulation factor: positive valence → higher precision
    pub precision_factor: f64,
    /// Epistemic weight shift: high arousal → more pragmatic (less exploratory)
    pub epistemic_weight_shift: f64,
    /// Blanket stress signal: physiological stress → close the blanket
    pub blanket_stress_signal: f64,
    /// EMA smoothing alpha
    alpha: f64,
}

#[cfg(feature = "interoceptive_inference")]
impl Default for InteroceptiveInferenceEngine {
    fn default() -> Self {
        Self {
            predicted_valence: 0.0,
            predicted_arousal: 0.5,
            interoceptive_pe: 0.0,
            precision_factor: 1.0,
            epistemic_weight_shift: 0.0,
            blanket_stress_signal: 0.0,
            alpha: 0.1, // Smooth interoceptive signals
        }
    }
}

#[cfg(feature = "interoceptive_inference")]
impl InteroceptiveInferenceEngine {
    /// Update interoceptive inference with current body state.
    ///
    /// # Arguments
    /// - `actual_valence`: Current emotional valence [-1, 1]
    /// - `actual_arousal`: Current arousal level [0, 1]
    /// - `physiological_stress`: Aggregate stress signal [0, 1]
    pub fn update(&mut self, actual_valence: f64, actual_arousal: f64, physiological_stress: f64) {
        // Interoceptive prediction error
        let valence_pe = (actual_valence - self.predicted_valence).abs();
        let arousal_pe = (actual_arousal - self.predicted_arousal).abs();
        let raw_pe = (valence_pe + arousal_pe) / 2.0;

        // EMA smooth
        self.interoceptive_pe = self.alpha * raw_pe + (1.0 - self.alpha) * self.interoceptive_pe;

        // Update predictions (simple EMA predictor — learns body patterns)
        self.predicted_valence =
            self.alpha * actual_valence + (1.0 - self.alpha) * self.predicted_valence;
        self.predicted_arousal =
            self.alpha * actual_arousal + (1.0 - self.alpha) * self.predicted_arousal;

        // Compute modulation factors

        // Precision: positive valence → boost confidence; negative → reduce
        // Basis: Barrett (2017) — affect is a primary signal about body budget
        self.precision_factor = 1.0 + 0.2 * actual_valence.clamp(-1.0, 1.0);

        // Epistemic/pragmatic balance: high arousal → more pragmatic (goal-directed)
        // Low arousal → more epistemic (exploratory)
        // Basis: Seth (2021) — interoceptive arousal shapes the exploration/exploitation tradeoff
        self.epistemic_weight_shift = -0.3 * (actual_arousal - 0.5);

        // Blanket stress: physiological stress closes the Markov blanket
        // Basis: Sterling (2012) — allostasis adjusts boundaries under stress
        self.blanket_stress_signal = physiological_stress.clamp(0.0, 1.0);
    }
}
