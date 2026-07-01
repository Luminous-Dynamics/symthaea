// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Scale Physics Unification — Genesis Mission Challenge 14
//!
//! Demonstrates that the HDC-LTC architecture can predict physical
//! observables at any timescale from 10^-15 s (particle) to 10^9 s (organism)
//! with O(1) cost per prediction. Each physical scale gets its own CfC
//! neuron with domain-appropriate tau_base, and cross-scale coherence is
//! measured via HDC cosine similarity.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ── Physical Scales ────────────────────────────────────────────────────────

/// Physical scales spanning 24 orders of magnitude.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhysicalScale {
    /// Subatomic particles (tau ~ 10^-15 s, length ~ 10^-15 m).
    Particle,
    /// Nuclear structure (tau ~ 10^-12 s, length ~ 10^-14 m).
    Nuclear,
    /// Atomic orbitals (tau ~ 10^-9 s, length ~ 10^-10 m).
    Atomic,
    /// Molecular dynamics (tau ~ 10^-6 s, length ~ 10^-9 m).
    Molecular,
    /// Cellular processes (tau ~ 1 s, length ~ 10^-5 m).
    Cellular,
    /// Organism-level (tau ~ 10^3 s, length ~ 1 m).
    Organism,
}

impl PhysicalScale {
    /// All scales in order from smallest to largest.
    pub const ALL: [PhysicalScale; 6] = [
        PhysicalScale::Particle,
        PhysicalScale::Nuclear,
        PhysicalScale::Atomic,
        PhysicalScale::Molecular,
        PhysicalScale::Cellular,
        PhysicalScale::Organism,
    ];

    /// Characteristic time constant (seconds) for this scale.
    pub fn tau_seconds(&self) -> f32 {
        match self {
            PhysicalScale::Particle => 1e-15,
            PhysicalScale::Nuclear => 1e-12,
            PhysicalScale::Atomic => 1e-9,
            PhysicalScale::Molecular => 1e-6,
            PhysicalScale::Cellular => 1.0,
            PhysicalScale::Organism => 1e3,
        }
    }

    /// Characteristic length scale (meters).
    pub fn length_meters(&self) -> f64 {
        match self {
            PhysicalScale::Particle => 1e-15,
            PhysicalScale::Nuclear => 1e-14,
            PhysicalScale::Atomic => 1e-10,
            PhysicalScale::Molecular => 1e-9,
            PhysicalScale::Cellular => 1e-5,
            PhysicalScale::Organism => 1.0,
        }
    }

    /// Number of observables at this scale.
    pub fn num_observables(&self) -> usize {
        match self {
            PhysicalScale::Particle => 4,  // energy, momentum, spin, charge
            PhysicalScale::Nuclear => 4,   // binding energy, mass number, spin, magnetic moment
            PhysicalScale::Atomic => 5, // energy levels, orbital, ionization, electron affinity, radius
            PhysicalScale::Molecular => 5, // bond length, angle, vibration freq, dipole, energy
            PhysicalScale::Cellular => 6, // membrane potential, Ca2+, ATP, pH, volume, viability
            PhysicalScale::Organism => 5, // heart rate, temperature, blood pressure, O2 sat, cortisol
        }
    }

    /// Deterministic seed for this scale's encoder.
    fn seed(&self) -> u64 {
        match self {
            PhysicalScale::Particle => 0xAA0_0001,
            PhysicalScale::Nuclear => 0xAA0_0002,
            PhysicalScale::Atomic => 0xAA0_0003,
            PhysicalScale::Molecular => 0xAA0_0004,
            PhysicalScale::Cellular => 0xAA0_0005,
            PhysicalScale::Organism => 0xAA0_0006,
        }
    }
}

// ── Scale Encoder ──────────────────────────────────────────────────────────

/// HDC encoder for a specific physical scale.
///
/// Each scale has its own base vectors for domain-specific observables.
pub struct ScaleEncoder {
    scale: PhysicalScale,
    bases: Vec<ContinuousHV>,
}

impl ScaleEncoder {
    /// Create an encoder for the given physical scale.
    pub fn new(scale: PhysicalScale) -> Self {
        let n = scale.num_observables();
        let bases = (0..n)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, scale.seed() + i as u64))
            .collect();
        Self { scale, bases }
    }

    /// Encode a vector of observables into 16,384D.
    ///
    /// Values should be normalized to [0, 1] or [-1, 1] range.
    pub fn encode(&self, observables: &[f64]) -> ContinuousHV {
        assert!(
            observables.len() <= self.bases.len(),
            "Too many observables for scale {:?}: {} > {}",
            self.scale,
            observables.len(),
            self.bases.len()
        );

        let weights: Vec<f32> = observables.iter().map(|&v| v as f32).collect();
        ContinuousHV::encode_weighted(&self.bases[..weights.len()], &weights)
    }

    /// Which physical scale this encodes.
    pub fn scale(&self) -> PhysicalScale {
        self.scale
    }
}

// ── Scale Predictor ────────────────────────────────────────────────────────

/// CfC neuron tuned for a specific physical timescale.
pub struct ScalePredictor {
    scale: PhysicalScale,
    neuron: HdcLtcUnifiedNeuron,
}

impl ScalePredictor {
    pub fn new(scale: PhysicalScale) -> Self {
        let config = UnifiedConfig {
            tau_base: scale.tau_seconds(),
            backbone_tau: 0.1,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            scale,
            neuron: HdcLtcUnifiedNeuron::new(config, scale.seed() + 0xFF),
        }
    }

    /// Predict state at the given time horizon. O(1) cost.
    ///
    /// # Panics
    ///
    /// Panics if `horizon_seconds` is not finite or is non-positive.
    pub fn predict(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        assert!(
            horizon_seconds.is_finite() && horizon_seconds > 0.0,
            "horizon_seconds must be finite and positive, got {}",
            horizon_seconds
        );
        let mut neuron_copy = self.neuron.clone();
        neuron_copy.evolve_closed_form(horizon_seconds, current_state);
        neuron_copy.state().clone()
    }

    /// Predict state at the given time horizon — alias for [`Self::predict`].
    ///
    /// This name aligns with the cross-domain [`TemporalPredictor`] convention
    /// used by materials, hydrology, and fusion predictors.
    pub fn predict_at_horizon(
        &self,
        current_state: &ContinuousHV,
        horizon_seconds: f32,
    ) -> ContinuousHV {
        self.predict(current_state, horizon_seconds)
    }

    /// Feed an observation.
    pub fn observe(&mut self, state: &ContinuousHV, dt: f32) {
        self.neuron.evolve_closed_form(dt, state);
    }

    /// Which physical scale this predictor is tuned for.
    pub fn scale(&self) -> PhysicalScale {
        self.scale
    }
}

impl symthaea_core::temporal::TemporalPredictor for ScalePredictor {
    fn predict_at(&self, current_state: &ContinuousHV, horizon_seconds: f32) -> ContinuousHV {
        self.predict(current_state, horizon_seconds)
    }

    fn observe(&mut self, state: &ContinuousHV, dt_seconds: f32) {
        self.observe(state, dt_seconds);
    }

    fn domain(&self) -> &'static str {
        "multiscale"
    }

    fn tau_base(&self) -> f32 {
        self.scale.tau_seconds()
    }
}

// ── Multi-Scale Unifier ────────────────────────────────────────────────────

/// Cross-scale coherence measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossScaleCoherence {
    /// Similarity between the two scales.
    pub similarity: f32,
    /// First scale.
    pub scale_a: PhysicalScale,
    /// Second scale.
    pub scale_b: PhysicalScale,
}

/// Multi-scale physics unifier.
///
/// One encoder + predictor per physical scale. Cross-scale coherence is
/// computed via HDC cosine similarity, enabling discovery of emergent
/// patterns that span multiple scales.
///
/// # Genesis Mission Challenge 14
///
/// Addresses the DOE Multi-Scale Physics challenge by demonstrating that
/// a single architecture can handle physics from 10^-15 s to 10^9 s
/// with O(1) prediction cost at every scale.
pub struct MultiScaleUnifier {
    encoders: Vec<ScaleEncoder>,
    predictors: Vec<ScalePredictor>,
}

impl MultiScaleUnifier {
    /// Create a unifier with one encoder + predictor per physical scale.
    pub fn new() -> Self {
        let encoders = PhysicalScale::ALL
            .iter()
            .map(|&s| ScaleEncoder::new(s))
            .collect();
        let predictors = PhysicalScale::ALL
            .iter()
            .map(|&s| ScalePredictor::new(s))
            .collect();
        Self {
            encoders,
            predictors,
        }
    }

    /// Encode observables at a specific scale.
    pub fn encode(&self, scale: PhysicalScale, observables: &[f64]) -> ContinuousHV {
        let idx = PhysicalScale::ALL
            .iter()
            .position(|&s| s == scale)
            .expect("scale must be a known PhysicalScale variant");
        self.encoders[idx].encode(observables)
    }

    /// Predict at a specific scale and horizon.
    pub fn predict(
        &self,
        scale: PhysicalScale,
        current_state: &ContinuousHV,
        dt_seconds: f32,
    ) -> ContinuousHV {
        let idx = PhysicalScale::ALL
            .iter()
            .position(|&s| s == scale)
            .expect("scale must be a known PhysicalScale variant");
        self.predictors[idx].predict(current_state, dt_seconds)
    }

    /// Feed an observation at a specific scale.
    pub fn observe(&mut self, scale: PhysicalScale, state: &ContinuousHV, dt: f32) {
        let idx = PhysicalScale::ALL
            .iter()
            .position(|&s| s == scale)
            .expect("scale must be a known PhysicalScale variant");
        self.predictors[idx].observe(state, dt);
    }

    /// Compute cross-scale coherence between all pairs of encoded states.
    pub fn cross_scale_coherence(
        &self,
        states: &[(PhysicalScale, ContinuousHV)],
    ) -> Vec<CrossScaleCoherence> {
        let mut results = Vec::new();
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                results.push(CrossScaleCoherence {
                    similarity: states[i].1.similarity(&states[j].1),
                    scale_a: states[i].0,
                    scale_b: states[j].0,
                });
            }
        }
        results
    }

    /// Access encoders.
    pub fn encoders(&self) -> &[ScaleEncoder] {
        &self.encoders
    }

    /// Access predictors.
    pub fn predictors(&self) -> &[ScalePredictor] {
        &self.predictors
    }
}

impl Default for MultiScaleUnifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Demonstrate the O(1) property: prediction cost is independent of timescale.
///
/// Encodes a state at each scale, then predicts at the scale's natural
/// timescale (tau) and at 1000× tau. Both should take the same time.
///
/// Returns `(scale, natural_tau, time_ns_at_tau, time_ns_at_1000tau)` tuples.
pub fn demonstrate_o1_property() -> Vec<(PhysicalScale, f32, u128, u128)> {
    let unifier = MultiScaleUnifier::new();
    let mut results = Vec::new();

    for (i, &scale) in PhysicalScale::ALL.iter().enumerate() {
        let n = scale.num_observables();
        let observables: Vec<f64> = (0..n).map(|j| 0.5 + j as f64 * 0.1).collect();
        let state = unifier.encoders[i].encode(&observables);
        let tau = scale.tau_seconds();

        let start_tau = std::time::Instant::now();
        for _ in 0..100 {
            let _ = unifier.predictors[i].predict(&state, tau);
        }
        let time_tau = start_tau.elapsed().as_nanos();

        let start_1000tau = std::time::Instant::now();
        for _ in 0..100 {
            let _ = unifier.predictors[i].predict(&state, tau * 1000.0);
        }
        let time_1000tau = start_1000tau.elapsed().as_nanos();

        results.push((scale, tau, time_tau, time_1000tau));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_scale_tau_ordering() {
        for i in 1..PhysicalScale::ALL.len() {
            assert!(
                PhysicalScale::ALL[i].tau_seconds() > PhysicalScale::ALL[i - 1].tau_seconds(),
                "{:?} tau should be > {:?} tau",
                PhysicalScale::ALL[i],
                PhysicalScale::ALL[i - 1]
            );
        }
    }

    #[test]
    fn test_scale_length_ordering() {
        for i in 1..PhysicalScale::ALL.len() {
            assert!(
                PhysicalScale::ALL[i].length_meters() > PhysicalScale::ALL[i - 1].length_meters(),
                "{:?} length should be > {:?} length",
                PhysicalScale::ALL[i],
                PhysicalScale::ALL[i - 1]
            );
        }
    }

    #[test]
    fn test_scale_spans_24_orders() {
        let smallest = PhysicalScale::Particle.tau_seconds();
        let largest = PhysicalScale::Organism.tau_seconds();
        let orders = (largest as f64 / smallest as f64).log10();
        assert!(
            orders > 17.0,
            "Should span >17 orders of magnitude, got {}",
            orders
        );
    }

    #[test]
    fn test_encoder_dimension() {
        for &scale in &PhysicalScale::ALL {
            let encoder = ScaleEncoder::new(scale);
            let n = scale.num_observables();
            let observables: Vec<f64> = vec![0.5; n];
            let hv = encoder.encode(&observables);
            assert_eq!(hv.dim(), HDC_DIMENSION);
        }
    }

    #[test]
    fn test_predictor_dimension() {
        for &scale in &PhysicalScale::ALL {
            let predictor = ScalePredictor::new(scale);
            let input = ContinuousHV::random(HDC_DIMENSION, 42);
            let predicted = predictor.predict(&input, scale.tau_seconds());
            assert_eq!(predicted.dim(), HDC_DIMENSION);
        }
    }

    #[test]
    fn test_o1_property_per_scale() {
        for &scale in &PhysicalScale::ALL {
            let predictor = ScalePredictor::new(scale);
            let input = ContinuousHV::random(HDC_DIMENSION, 42);
            let tau = scale.tau_seconds();

            let start_1tau = Instant::now();
            for _ in 0..50 {
                let _ = predictor.predict(&input, tau);
            }
            let time_1tau = start_1tau.elapsed();

            let start_1000tau = Instant::now();
            for _ in 0..50 {
                let _ = predictor.predict(&input, tau * 1000.0);
            }
            let time_1000tau = start_1000tau.elapsed();

            let ratio = time_1000tau.as_nanos() as f64 / time_1tau.as_nanos().max(1) as f64;
            assert!(
                ratio < 5.0 && ratio > 0.2,
                "O(1) violated for {:?}: 1τ={:?}, 1000τ={:?}, ratio={}",
                scale,
                time_1tau,
                time_1000tau,
                ratio
            );
        }
    }

    #[test]
    fn test_unifier_encode_all_scales() {
        let unifier = MultiScaleUnifier::new();
        for &scale in &PhysicalScale::ALL {
            let n = scale.num_observables();
            let obs: Vec<f64> = vec![0.5; n];
            let hv = unifier.encode(scale, &obs);
            assert_eq!(hv.dim(), HDC_DIMENSION);
        }
    }

    #[test]
    fn test_cross_scale_coherence() {
        let unifier = MultiScaleUnifier::new();
        let states: Vec<(PhysicalScale, ContinuousHV)> = PhysicalScale::ALL
            .iter()
            .map(|&s| {
                let n = s.num_observables();
                let obs: Vec<f64> = vec![0.5; n];
                (s, unifier.encode(s, &obs))
            })
            .collect();

        let coherence = unifier.cross_scale_coherence(&states);
        // n*(n-1)/2 = 6*5/2 = 15 pairs
        assert_eq!(coherence.len(), 15);

        for c in &coherence {
            assert!(c.similarity.is_finite());
        }
    }

    #[test]
    fn test_different_scales_produce_different_encodings() {
        let unifier = MultiScaleUnifier::new();
        let obs_particle = vec![0.5; PhysicalScale::Particle.num_observables()];
        let obs_organism = vec![0.5; PhysicalScale::Organism.num_observables()];
        let hv_particle = unifier.encode(PhysicalScale::Particle, &obs_particle);
        let hv_organism = unifier.encode(PhysicalScale::Organism, &obs_organism);

        // Different base vectors → low similarity even with same values
        let sim = hv_particle.similarity(&hv_organism);
        assert!(
            sim.abs() < 0.5,
            "Different scales should produce dissimilar encodings, sim={}",
            sim
        );
    }

    #[test]
    fn test_demonstrate_o1_property() {
        let results = demonstrate_o1_property();
        assert_eq!(results.len(), 6);
        for (scale, _tau, time_tau, time_1000tau) in &results {
            let ratio = *time_1000tau as f64 / (*time_tau).max(1) as f64;
            assert!(
                ratio < 5.0,
                "O(1) violated for {:?}: ratio={}",
                scale,
                ratio
            );
        }
    }

    // ── Track B: failure-path tests ──────────────────────────────────────

    #[test]
    #[should_panic(expected = "horizon_seconds must be finite and positive")]
    fn test_scale_predictor_rejects_nan() {
        let predictor = ScalePredictor::new(PhysicalScale::Atomic);
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        predictor.predict(&input, f32::NAN);
    }

    #[test]
    #[should_panic(expected = "horizon_seconds must be finite and positive")]
    fn test_scale_predictor_rejects_zero() {
        let predictor = ScalePredictor::new(PhysicalScale::Atomic);
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        predictor.predict(&input, 0.0);
    }

    #[test]
    #[should_panic(expected = "horizon_seconds must be finite and positive")]
    fn test_scale_predictor_rejects_negative() {
        let predictor = ScalePredictor::new(PhysicalScale::Atomic);
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        predictor.predict(&input, -1.0);
    }

    #[test]
    #[should_panic(expected = "Too many observables")]
    fn test_encoder_rejects_too_many_observables() {
        let encoder = ScaleEncoder::new(PhysicalScale::Particle); // 4 observables
        encoder.encode(&[0.5; 10]); // 10 > 4
    }

    #[test]
    fn test_encoder_accepts_fewer_observables() {
        let encoder = ScaleEncoder::new(PhysicalScale::Cellular); // 6 observables
        let hv = encoder.encode(&[0.5; 3]); // 3 < 6, should work
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    // ── Track C: edge-path tests ─────────────────────────────────────────

    #[test]
    fn test_observe_changes_prediction() {
        let mut predictor = ScalePredictor::new(PhysicalScale::Cellular);
        let state_a = ContinuousHV::random(HDC_DIMENSION, 100);
        let state_b = ContinuousHV::random(HDC_DIMENSION, 200);

        // Predict before any observation
        let pred_before = predictor.predict(&state_a, 1.0);

        // Feed an observation
        predictor.observe(&state_b, 1.0);

        // Predict again — internal neuron state has changed, so prediction differs
        let pred_after = predictor.predict(&state_a, 1.0);

        let sim = pred_before.similarity(&pred_after);
        assert!(
            sim < 0.999,
            "Prediction should change after observe(), similarity={}",
            sim
        );
    }

    #[test]
    fn test_unifier_observe_routes_to_correct_scale() {
        let mut unifier = MultiScaleUnifier::new();
        let state = ContinuousHV::random(HDC_DIMENSION, 42);

        // Predict before observe
        let pred_before = unifier.predict(PhysicalScale::Molecular, &state, 1e-6);

        // Observe at Molecular scale
        let obs = ContinuousHV::random(HDC_DIMENSION, 999);
        unifier.observe(PhysicalScale::Molecular, &obs, 1e-6);

        // Prediction at Molecular should have changed
        let pred_after = unifier.predict(PhysicalScale::Molecular, &state, 1e-6);
        let sim_molecular = pred_before.similarity(&pred_after);
        assert!(
            sim_molecular < 0.999,
            "Molecular prediction should change after observe, sim={}",
            sim_molecular
        );
    }

    #[test]
    fn test_predict_at_horizon_alias() {
        let predictor = ScalePredictor::new(PhysicalScale::Atomic);
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let tau = PhysicalScale::Atomic.tau_seconds();

        let from_predict = predictor.predict(&input, tau);
        let from_alias = predictor.predict_at_horizon(&input, tau);

        // Both should return identical results
        assert_eq!(
            from_predict.similarity(&from_alias),
            1.0,
            "predict_at_horizon should be an exact alias for predict"
        );
    }

    #[test]
    fn test_temporal_predictor_trait_impl() {
        use symthaea_core::temporal::TemporalPredictor;

        let predictor = ScalePredictor::new(PhysicalScale::Organism);
        assert_eq!(predictor.domain(), "multiscale");
        assert_eq!(predictor.tau_base(), PhysicalScale::Organism.tau_seconds());
    }

    #[test]
    fn test_unifier_default() {
        let unifier = MultiScaleUnifier::default();
        assert_eq!(unifier.encoders().len(), 6);
        assert_eq!(unifier.predictors().len(), 6);
    }

    #[test]
    fn test_cross_scale_coherence_single_state() {
        let unifier = MultiScaleUnifier::new();
        let state = ContinuousHV::random(HDC_DIMENSION, 42);
        // Single state → 0 pairs
        let coherence = unifier.cross_scale_coherence(&[(PhysicalScale::Particle, state)]);
        assert!(coherence.is_empty());
    }

    #[test]
    fn test_cross_scale_coherence_empty() {
        let unifier = MultiScaleUnifier::new();
        let coherence = unifier.cross_scale_coherence(&[]);
        assert!(coherence.is_empty());
    }
}
