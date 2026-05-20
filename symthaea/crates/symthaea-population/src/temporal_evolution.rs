// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! O(1) population trajectory prediction via CfC closed-form.
//!
//! Instead of simulating N generations explicitly, this module uses an
//! HdcLtcUnifiedNeuron to predict population state (heterozygosity, Ne, F, load)
//! at any future generation in O(1) time via closed-form temporal jumps.
//!
//! The population state is encoded as a 16,384D hypervector where different
//! dimensions encode heterozygosity, effective population size, inbreeding
//! coefficient, and genetic load metrics.

use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNeuron, UnifiedConfig};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::types::Population;

/// Average human generation time in seconds (~25 years).
const GENERATION_TIME_SECONDS: f32 = 25.0 * 365.25 * 24.0 * 3600.0;

/// Seed for heterozygosity metric encoding.
const _SEED_HETEROZYGOSITY: u64 = 0xBE7A_0001_0000_0001;
/// Seed for effective Ne metric encoding.
const _SEED_EFFECTIVE_NE: u64 = 0xBE7A_0002_0000_0002;
/// Seed for inbreeding coefficient encoding.
const _SEED_INBREEDING: u64 = 0xBE7A_0003_0000_0003;
/// Seed for genetic load encoding.
const _SEED_GENETIC_LOAD: u64 = 0xBE7A_0004_0000_0004;

/// O(1) population trajectory predictor using CfC closed-form temporal jumps.
///
/// Encodes population metrics into a 16,384D hypervector state and evolves
/// it through LTC dynamics. Because of the closed-form solution, predicting
/// state at generation 1 costs the same as generation 1,000.
pub struct PopulationTrajectoryPredictor {
    /// The unified HDC-LTC neuron performing temporal evolution.
    neuron: HdcLtcUnifiedNeuron,
    /// Time per generation in seconds.
    generation_time_seconds: f32,
}

impl PopulationTrajectoryPredictor {
    /// Create a new predictor with default configuration.
    pub fn new() -> Self {
        let config = UnifiedConfig {
            tau_base: 1.0, // Longer time constant for population-scale dynamics
            backbone_tau: 0.3,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0xA0A0_1A71_0000_5EED),
            generation_time_seconds: GENERATION_TIME_SECONDS,
        }
    }

    /// Create a predictor with a custom generation time.
    pub fn with_generation_time(generation_time_seconds: f32) -> Self {
        let config = UnifiedConfig {
            tau_base: 1.0,
            backbone_tau: 0.3,
            dimension: HDC_DIMENSION,
            ..UnifiedConfig::default()
        };
        Self {
            neuron: HdcLtcUnifiedNeuron::new(config, 0x5EED_5EED_5EED_5EED),
            generation_time_seconds,
        }
    }

    /// Predict population state at a target generation.
    ///
    /// Uses O(1) closed-form temporal jump regardless of how far into the future.
    /// Returns the predicted state as a 16,384D hypervector.
    pub fn predict_at_generation(
        &mut self,
        initial_state: &ContinuousHV,
        target_generation: u32,
    ) -> ContinuousHV {
        // Set neuron to initial state
        self.neuron.set_state(initial_state.clone());

        // O(1) temporal jump: same cost for generation 1 or generation 1000
        let dt = target_generation as f32 * self.generation_time_seconds;
        self.neuron.evolve_closed_form(dt, initial_state);

        // Return the evolved state
        self.neuron.state().clone()
    }

    /// Encode population metrics into a 16,384D state vector.
    ///
    /// Metrics are encoded by binding each metric value to a metric-specific
    /// basis HV and bundling the results.
    pub fn encode_population_state(population: &Population) -> ContinuousHV {
        let n = population.size();
        if n == 0 {
            return ContinuousHV::zero(HDC_DIMENSION);
        }

        // Compute metrics
        let ne = crate::effective_population::ne_sex_ratio(
            population.count_females(),
            population.count_males(),
        );
        let load = crate::genetic_load::population_genetic_load(population);

        // Mean heterozygosity from genotypes
        let heterozygosity = if population.individuals.is_empty()
            || population.individuals[0].genotypes.is_empty()
        {
            0.0
        } else {
            let n_loci = population.individuals[0].genotypes.len();
            let total_ho: f64 = (0..n_loci)
                .map(|li| {
                    let genos: Vec<crate::types::Genotype> = population
                        .individuals
                        .iter()
                        .filter_map(|i| i.genotypes.get(li).cloned())
                        .collect();
                    crate::diversity::observed_heterozygosity(&genos)
                })
                .sum();
            total_ho / n_loci as f64
        };

        // Encode each metric as a weighted basis HV
        let he_basis = ContinuousHV::random(HDC_DIMENSION, 0xBE7A_0001_0000_0001);
        let ne_basis = ContinuousHV::random(HDC_DIMENSION, 0xBE7A_0002_0000_0002);
        let load_basis = ContinuousHV::random(HDC_DIMENSION, 0xBE7A_0003_0000_0003);
        let size_basis = ContinuousHV::random(HDC_DIMENSION, 0xBE7A_0004_0000_0004);

        // Normalize Ne to [0, 1] range (cap at 1000)
        let ne_normalized = (ne / 1000.0).min(1.0) as f32;
        let n_normalized = (n as f64 / 1000.0).min(1.0) as f32;

        ContinuousHV::weighted_bundle(
            &[&he_basis, &ne_basis, &load_basis, &size_basis],
            &[
                heterozygosity as f32,
                ne_normalized,
                load as f32,
                n_normalized,
            ],
        )
    }

    /// Decode a predicted metric from the state vector.
    ///
    /// Returns the similarity of the state with each metric basis HV,
    /// giving approximate metric values.
    pub fn decode_heterozygosity(state: &ContinuousHV) -> f64 {
        let he_basis = ContinuousHV::random(HDC_DIMENSION, 0xBE7A_0001_0000_0001);
        // Similarity gives a rough estimate of the encoded metric weight
        ((state.similarity(&he_basis) + 1.0) / 2.0) as f64
    }

    /// Decode effective population size estimate from state vector.
    pub fn decode_effective_ne(state: &ContinuousHV) -> f64 {
        let ne_basis = ContinuousHV::random(HDC_DIMENSION, 0xBE7A_0002_0000_0002);
        let raw = ((state.similarity(&ne_basis) + 1.0) / 2.0) as f64;
        raw * 1000.0 // Un-normalize
    }

    /// Reset the predictor state.
    pub fn reset(&mut self) {
        self.neuron.reset();
    }

    // ── Calibrated Prediction API ──────────────────────────────────────────

    /// O(1) calibrated prediction of heterozygosity under neutral drift.
    ///
    /// Uses the analytical formula H(t) = H(0) * (1 - 1/(2*Ne))^t which is
    /// itself O(1) — a single exponentiation regardless of generation count.
    pub fn predict_heterozygosity_neutral(h0: f64, ne: f64, generations: u32) -> f64 {
        crate::diversity::heterozygosity_after_generations(h0, ne, generations)
    }

    /// Predict the *relative diversity advantage* of a breeding strategy.
    ///
    /// Runs CfC evolution from two states and returns the ratio of their
    /// decoded heterozygosities. Values > 1.0 mean `state_a` retains more
    /// diversity than `state_b`.
    pub fn predict_relative_advantage(
        &mut self,
        state_a: &ContinuousHV,
        state_b: &ContinuousHV,
        target_generation: u32,
    ) -> f64 {
        let evolved_a = self.predict_at_generation(state_a, target_generation);
        self.reset();
        let evolved_b = self.predict_at_generation(state_b, target_generation);
        self.reset();

        let het_a = Self::decode_heterozygosity(&evolved_a);
        let het_b = Self::decode_heterozygosity(&evolved_b);

        if het_b > 1e-10 { het_a / het_b } else { 1.0 }
    }

    /// CfC-calibrated heterozygosity prediction blending neural and analytical.
    ///
    /// Uses CfC temporal evolution to capture state trajectory *direction*
    /// and analytical drift to anchor the *magnitude*.
    pub fn predict_heterozygosity_calibrated(
        &mut self,
        population: &Population,
        ne: f64,
        target_generation: u32,
    ) -> f64 {
        if population.individuals.is_empty() {
            return 0.0;
        }

        let initial_state = Self::encode_population_state(population);
        let initial_het = Self::decode_heterozygosity(&initial_state);

        if initial_het < 1e-10 || target_generation == 0 {
            return initial_het;
        }

        // CfC trajectory
        let evolved = self.predict_at_generation(&initial_state, target_generation);
        self.reset();
        let cfc_het = Self::decode_heterozygosity(&evolved);
        let cfc_ratio = cfc_het / initial_het;

        // Analytical baseline: neutral drift
        let analytical = Self::predict_heterozygosity_neutral(initial_het, ne, target_generation);
        let analytical_ratio = analytical / initial_het;

        // Blend: anchor on analytical, modulated by CfC trajectory shape
        let modulation = if analytical_ratio > 1e-10 {
            cfc_ratio / analytical_ratio
        } else {
            1.0
        };

        let modulation = modulation.clamp(0.5, 2.0);
        (analytical * modulation).clamp(0.0, 1.0)
    }
}

impl Default for PopulationTrajectoryPredictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Allele, BiologicalSex, Genotype, Individual};

    fn make_test_population(n: usize) -> Population {
        let individuals: Vec<Individual> = (0..n)
            .map(|i| {
                let sex = if i % 2 == 0 {
                    BiologicalSex::Female
                } else {
                    BiologicalSex::Male
                };
                Individual {
                    id: i as u64,
                    sex,
                    genotypes: vec![
                        Genotype {
                            locus_id: 0,
                            allele_a: Allele::neutral(1),
                            allele_b: Allele::neutral(if i % 3 == 0 { 1 } else { 2 }),
                        },
                        Genotype {
                            locus_id: 1,
                            allele_a: Allele::neutral(3),
                            allele_b: Allele::neutral(4),
                        },
                    ],
                    genome_hv: ContinuousHV::random(HDC_DIMENSION, i as u64 * 100),
                    generation: 0,
                    parent_ids: (None, None),
                    fitness: 1.0,
                }
            })
            .collect();

        Population {
            individuals,
            generation: 0,
            founding_size: n,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        }
    }

    #[test]
    fn test_predictor_creation() {
        let pred = PopulationTrajectoryPredictor::new();
        assert_eq!(pred.generation_time_seconds, GENERATION_TIME_SECONDS);
    }

    #[test]
    fn test_predictor_default() {
        let pred = PopulationTrajectoryPredictor::default();
        assert_eq!(pred.generation_time_seconds, GENERATION_TIME_SECONDS);
    }

    #[test]
    fn test_predictor_custom_generation_time() {
        let pred = PopulationTrajectoryPredictor::with_generation_time(100.0);
        assert_eq!(pred.generation_time_seconds, 100.0);
    }

    #[test]
    fn test_encode_population_state_nonzero() {
        let pop = make_test_population(20);
        let state = PopulationTrajectoryPredictor::encode_population_state(&pop);
        assert_eq!(state.dim(), HDC_DIMENSION);
        let norm: f32 = state.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.0, "encoded state should be non-zero");
    }

    #[test]
    fn test_encode_empty_population() {
        let pop = Population {
            individuals: vec![],
            generation: 0,
            founding_size: 0,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        let state = PopulationTrajectoryPredictor::encode_population_state(&pop);
        let norm: f32 = state.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm < 1e-10, "empty population state should be zero");
    }

    #[test]
    fn test_predict_at_generation_zero_returns_near_initial() {
        let mut pred = PopulationTrajectoryPredictor::with_generation_time(1.0);
        let initial = ContinuousHV::random(HDC_DIMENSION, 42);

        let predicted = pred.predict_at_generation(&initial, 0);
        let sim = initial.similarity(&predicted);
        assert!(
            sim > 0.5,
            "prediction at gen 0 should be similar to initial: sim={sim}"
        );
    }

    #[test]
    fn test_predict_gen1_and_gen1000_same_cost() {
        // Both should complete in approximately the same time (O(1))
        // We can't easily measure time in a unit test, but we can verify
        // that both produce valid results.
        let mut pred = PopulationTrajectoryPredictor::with_generation_time(1.0);
        let initial = ContinuousHV::random(HDC_DIMENSION, 42);

        let pred_1 = pred.predict_at_generation(&initial, 1);
        pred.reset();
        let pred_1000 = pred.predict_at_generation(&initial, 1000);

        assert_eq!(pred_1.dim(), HDC_DIMENSION);
        assert_eq!(pred_1000.dim(), HDC_DIMENSION);

        // Both should be valid (non-NaN) vectors
        assert!(
            pred_1.values.iter().all(|v| v.is_finite()),
            "gen 1 prediction should have finite values"
        );
        assert!(
            pred_1000.values.iter().all(|v| v.is_finite()),
            "gen 1000 prediction should have finite values"
        );
    }

    #[test]
    fn test_prediction_diverges_over_time() {
        let mut pred = PopulationTrajectoryPredictor::with_generation_time(1.0);
        let initial = ContinuousHV::random(HDC_DIMENSION, 42);

        let pred_10 = pred.predict_at_generation(&initial, 10);
        pred.reset();
        let pred_100 = pred.predict_at_generation(&initial, 100);

        // Predictions at different times should generally differ
        let sim = pred_10.similarity(&pred_100);
        // They might be similar (depending on dynamics), but not identical
        assert!(
            sim < 1.0 - 1e-6,
            "predictions at different times should differ: sim={sim}"
        );
    }

    #[test]
    fn test_decode_heterozygosity_in_range() {
        let pop = make_test_population(20);
        let state = PopulationTrajectoryPredictor::encode_population_state(&pop);
        let he = PopulationTrajectoryPredictor::decode_heterozygosity(&state);
        assert!(
            he >= 0.0 && he <= 1.0,
            "decoded heterozygosity should be in [0,1], got {he}"
        );
    }

    #[test]
    fn test_decode_effective_ne_positive() {
        let pop = make_test_population(20);
        let state = PopulationTrajectoryPredictor::encode_population_state(&pop);
        let ne = PopulationTrajectoryPredictor::decode_effective_ne(&state);
        assert!(ne >= 0.0, "decoded Ne should be non-negative, got {ne}");
    }

    #[test]
    fn test_predictor_reset() {
        let mut pred = PopulationTrajectoryPredictor::new();
        let input = ContinuousHV::random(HDC_DIMENSION, 42);
        let _ = pred.predict_at_generation(&input, 10);

        pred.reset();
        let zero_state = pred.neuron.state();
        let norm: f32 = zero_state.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm < 1e-6, "after reset, state should be near zero");
    }

    #[test]
    fn test_different_populations_encode_differently() {
        let pop_small = make_test_population(10);
        let pop_large = make_test_population(100);

        let state_small = PopulationTrajectoryPredictor::encode_population_state(&pop_small);
        let state_large = PopulationTrajectoryPredictor::encode_population_state(&pop_large);

        let sim = state_small.similarity(&state_large);
        assert!(
            sim < 0.99,
            "different populations should encode differently: sim={sim}"
        );
    }

    // ── Calibration tests ──────────────────────────────────────────────────

    #[test]
    fn test_neutral_prediction_matches_analytical() {
        let h0 = 0.75;
        let ne = 100.0;
        for r#gen in [1, 10, 50, 100, 500] {
            let predicted =
                PopulationTrajectoryPredictor::predict_heterozygosity_neutral(h0, ne, r#gen);
            let expected = crate::diversity::heterozygosity_after_generations(h0, ne, r#gen);
            assert!(
                (predicted - expected).abs() < 1e-12,
                "neutral prediction at gen {gen}: {predicted} vs {expected}"
            );
        }
    }

    #[test]
    fn test_neutral_prediction_monotonic_decay() {
        let h0 = 0.8;
        let ne = 50.0;
        let mut prev = h0;
        for r#gen in 1..=20 {
            let h = PopulationTrajectoryPredictor::predict_heterozygosity_neutral(h0, ne, r#gen);
            assert!(h <= prev + 1e-12, "should decay: gen {gen}, {h} > {prev}");
            assert!(h >= 0.0);
            prev = h;
        }
    }

    #[test]
    fn test_calibrated_prediction_reasonable_range() {
        let pop = make_test_population(50);
        let ne = 25.0;
        let mut pred = PopulationTrajectoryPredictor::new();
        for r#gen in [1, 5, 10, 20] {
            let h = pred.predict_heterozygosity_calibrated(&pop, ne, r#gen);
            pred.reset();
            assert!(
                (0.0..=1.0).contains(&h),
                "calibrated at gen {gen} should be in [0,1]: got {h}"
            );
        }
    }

    #[test]
    fn test_calibrated_gen_zero_returns_initial() {
        let pop = make_test_population(50);
        let mut pred = PopulationTrajectoryPredictor::new();
        let h0 = pred.predict_heterozygosity_calibrated(&pop, 25.0, 0);
        pred.reset();
        let initial_state = PopulationTrajectoryPredictor::encode_population_state(&pop);
        let initial_het = PopulationTrajectoryPredictor::decode_heterozygosity(&initial_state);
        assert!(
            (h0 - initial_het).abs() < 1e-10,
            "gen 0 should equal initial: {h0} vs {initial_het}"
        );
    }

    #[test]
    fn test_relative_advantage_same_state_near_one() {
        let mut pred = PopulationTrajectoryPredictor::with_generation_time(1.0);
        let state = ContinuousHV::random(HDC_DIMENSION, 42);
        let ratio = pred.predict_relative_advantage(&state, &state, 10);
        assert!(
            (ratio - 1.0).abs() < 0.1,
            "same state ratio should be ~1.0, got {ratio}"
        );
    }

    #[test]
    fn test_relative_advantage_finite() {
        let mut pred = PopulationTrajectoryPredictor::with_generation_time(1.0);
        let state_a = ContinuousHV::random(HDC_DIMENSION, 100);
        let state_b = ContinuousHV::random(HDC_DIMENSION, 200);
        let ratio = pred.predict_relative_advantage(&state_a, &state_b, 50);
        assert!(ratio.is_finite());
        assert!(ratio >= 0.0);
    }

    #[test]
    fn test_calibrated_empty_population_zero() {
        let pop = Population {
            individuals: vec![],
            generation: 0,
            founding_size: 0,
            diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
        };
        let mut pred = PopulationTrajectoryPredictor::new();
        let h = pred.predict_heterozygosity_calibrated(&pop, 50.0, 10);
        assert!(h.abs() < 1e-10, "empty pop should be ~0: got {h}");
    }
}
