// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Thermodynamic analogies and model diagnostics for the cognitive loop.
//!
//! This module maps selected cognitive telemetry into quantities inspired by
//! statistical mechanics and thermodynamics. It currently computes heuristic
//! proxies in effective or dimensionless units; it does not simulate a
//! physical heat engine, perform a measured bit-erasure protocol, or estimate
//! Onsager response coefficients from conjugate flux/force perturbations.
//!
//! # Claim boundary
//!
//! - Attention and prediction-error signals are not measurements of physical
//!   memory erasure. Their information-theoretic values are model proxies.
//! - The "Carnot" and "actual efficiency" fields are normalized cognitive
//!   diagnostics, not heat-engine efficiencies.
//! - Insight probability is a heuristic mapping, not a fluctuation-theorem
//!   path-probability ratio.
//! - The current covariance-symmetry diagnostic is not an Onsager reciprocity
//!   test: covariance is symmetric by construction.
//! - The current work statistic is not a Jarzynski work ensemble. A physical
//!   free-energy estimate requires a defined protocol and forward/reverse work
//!   trajectories under the equality's assumptions.
//! - The "epistemic W2" value is currently a scalar prediction-error proxy,
//!   not a Wasserstein distance between probability distributions.
//!
//! Keep these exploratory signals separate from physical measurements,
//! calibrated implementation costs, and validated thermodynamic laws. Stronger
//! interpretations are tracked in
//! [PHYS-THERMO-COMPUTING-001](https://github.com/Luminous-Dynamics/symthaea/issues/10428).

use serde::{Deserialize, Serialize};

use super::thresholds;

/// Cognitive thermodynamic-analogy diagnostics. These values are heuristic
/// model outputs unless a field explicitly documents a measured physical input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicPhysicsBridge {
    // ═══════════════════════════════════════════════════════════════════════
    // Maxwell's Demon as Attention
    // ═══════════════════════════════════════════════════════════════════════
    /// Attention-information proxy derived from cognitive telemetry; not a
    /// measured count of physical bit erasures.
    pub attention_demon_bits: f64,
    /// Heuristic information-to-action score in model units; not measured work.
    pub attention_work: f64,
    /// Landauer-inspired model cost from the attention-information proxy.
    /// No physical memory reset operation is measured by this field.
    pub attention_erasure_cost: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Landauer Limit for Memory
    // ═══════════════════════════════════════════════════════════════════════
    /// Prediction-error-derived memory-consolidation proxy in effective units.
    /// This is not a Landauer lower bound for a measured erasure operation.
    pub memory_landauer_cost: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Carnot Efficiency
    // ═══════════════════════════════════════════════════════════════════════
    /// Carnot-shaped normalization computed from configured effective
    /// temperatures; not a physical heat-engine bound for this cognitive loop.
    pub carnot_efficiency: f64,
    /// Normalized cognitive diagnostic; numerator and denominator are not
    /// measured heat-engine work and heat flows.
    pub actual_efficiency: f64,
    /// Ratio of two model diagnostics; not a thermodynamic efficiency ratio.
    pub efficiency_ratio: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Heuristic insight score
    // ═══════════════════════════════════════════════════════════════════════
    /// Heuristic insight score derived from order/entropy proxies; not a
    /// probability obtained from a fluctuation-theorem path ensemble.
    pub insight_probability: f64,
    /// Whether an insight event was detected this cycle.
    pub insight_detected: bool,

    // ═══════════════════════════════════════════════════════════════════════
    // Onsager Cross-Coupling Health
    // ═══════════════════════════════════════════════════════════════════════
    /// Legacy covariance-symmetry diagnostic. The current implementation
    /// subtracts a covariance matrix from its transpose; that is zero up to
    /// floating-point roundoff by construction. It does not estimate Onsager
    /// reciprocity or measure integration; retained for compatibility.
    pub onsager_asymmetry: f64,

    /// Rolling cognitive telemetry used by the legacy covariance-symmetry
    /// diagnostic; these are not conjugate thermodynamic flux/force pairs.
    #[serde(skip)]
    history: std::collections::VecDeque<[f64; 4]>,

    // ═══════════════════════════════════════════════════════════════════════
    // Cognitive thermodynamic-analogy diagnostics (Phase 6)
    // ═══════════════════════════════════════════════════════════════════════
    /// Rolling model-derived "work" proxies. These are not work measurements
    /// from a defined nonequilibrium protocol and must not be treated as a
    /// Jarzynski estimator.
    #[serde(skip)]
    pub(crate) work_samples: std::collections::VecDeque<f64>,

    /// Legacy exponential-work proxy, not a validated Jarzynski free-energy
    /// estimate: its inputs are not sampled protocol work trajectories.
    pub jarzynski_free_energy: f64,

    /// Difference between two model diagnostics. It is not a comparison
    /// between independently valid thermodynamic and variational free energies.
    pub jarzynski_hfe_divergence: f64,

    /// Heuristic trend flag for the configured LinearNonEquilibrium regime.
    /// It does not establish violation of Prigogine's minimum entropy production
    /// principle, which has restricted near-equilibrium assumptions.
    pub prigogine_violated: bool,

    /// Previous entropy production rate (for Prigogine trend detection).
    #[serde(skip)]
    pub(crate) prev_entropy_production: f64,

    /// Previous order parameter (for England signal detection).
    #[serde(skip)]
    pub(crate) prev_order_parameter: f64,

    /// England dissipation-driven adaptation signal.
    /// True when entropy production correlates with order growth AND prediction
    /// error reduction — indicating that dissipation is driving useful learning.
    /// Science: England (2013) — matter driven by external energy self-organizes
    /// to maximize entropy production while building internal order.
    /// Only active when `england_dissipation` feature is enabled.
    pub england_exploration_boost: bool,

    // ═══════════════════════════════════════════════════════════════════════
    // Epistemic model diagnostics (feature-gated)
    // ═══════════════════════════════════════════════════════════════════════
    /// Heuristic epistemic score formed from scalar prediction error and
    /// entropy-production proxies. It is not a measured thermodynamic free
    /// energy or a validated result of the cited learning theory.
    #[cfg(feature = "epistemic")]
    pub epistemic_free_energy: f64,

    /// Heuristic ratio using a scalar prediction-error proxy in place of W2.
    /// Values above 1 are internal diagnostic thresholds, not proof of a
    /// physical thermodynamic speed-limit violation. The cited learning theory
    /// motivates a future distribution-based test but is not validated by this
    /// scalar implementation.
    #[cfg(feature = "epistemic")]
    pub epistemic_speed_limit_violation: f64,

    /// Absolute scalar prediction-error change; not a Wasserstein-2 distance
    /// unless replaced by a distribution-to-distribution transport calculation.
    #[cfg(feature = "epistemic")]
    pub belief_wasserstein_distance: f64,

    /// Abstract per-cycle model budget (default 1.0 arbitrary units). It is
    /// not an SI joule budget until a calibrated mapping is supplied.
    #[cfg(feature = "epistemic")]
    pub total_energy_budget: f64,
}

impl Default for ThermodynamicPhysicsBridge {
    fn default() -> Self {
        Self {
            attention_demon_bits: 0.0,
            attention_work: 0.0,
            attention_erasure_cost: 0.0,
            memory_landauer_cost: 0.0,
            carnot_efficiency: 0.0,
            actual_efficiency: 0.0,
            efficiency_ratio: 0.0,
            insight_probability: 0.0,
            insight_detected: false,
            onsager_asymmetry: 1.0,
            history: std::collections::VecDeque::with_capacity(
                thresholds::THERMO_ONSAGER_WINDOW + 1,
            ),
            work_samples: std::collections::VecDeque::with_capacity(
                thresholds::THERMO_JARZYNSKI_WINDOW + 1,
            ),
            jarzynski_free_energy: 0.0,
            jarzynski_hfe_divergence: 0.0,
            prigogine_violated: false,
            prev_entropy_production: 0.0,
            prev_order_parameter: 0.5,
            england_exploration_boost: false,
            #[cfg(feature = "epistemic")]
            epistemic_free_energy: 0.0,
            #[cfg(feature = "epistemic")]
            epistemic_speed_limit_violation: 0.0,
            #[cfg(feature = "epistemic")]
            belief_wasserstein_distance: 0.0,
            #[cfg(feature = "epistemic")]
            total_energy_budget: 1.0,
        }
    }
}

impl ThermodynamicPhysicsBridge {
    /// Compute cognitive diagnostics inspired by thermodynamic concepts.
    /// Most fields are model proxies in effective units, not physical observables.
    ///
    /// Called after all three thermodynamic modules have written to the unified state.
    pub(crate) fn compute(
        &mut self,
        attention_sensitivity: f64,
        information_processed: f64,
        coherence: f64,
        prediction_error: f64,
        effective_temperature: f64,
        entropy_production_rate: f64,
        order_parameter: f64,
        energy_per_cycle: f64,
        prev_order_parameter: f64,
        hfe_total: f64,
        regime: &crate::consciousness::dissipative_consciousness::ThermodynamicRegime,
    ) {
        // ═══════════════════════════════════════════════════════════════════
        // Maxwell's Demon as Attention (Szilard 1929)
        // ═══════════════════════════════════════════════════════════════════
        // Model heuristic: scale the attention-information proxy into an
        // effective work/erasure-cost diagnostic. No physical memory-erasure
        // protocol or work extraction is represented here.
        let k_eff = thresholds::K_CONSCIOUSNESS_BOLTZMANN;
        let t_eff = effective_temperature.max(0.01); // Avoid division by zero
        let ln2 = std::f64::consts::LN_2;

        self.attention_demon_bits = (attention_sensitivity * information_processed).max(0.0);
        let demon_efficiency = thresholds::THERMO_ATTENTION_DEMON_EFFICIENCY;
        self.attention_work = self.attention_demon_bits * k_eff * t_eff * ln2 * demon_efficiency;
        self.attention_erasure_cost = self.attention_demon_bits * k_eff * t_eff * ln2;

        // ═══════════════════════════════════════════════════════════════════
        // Landauer-inspired memory-cost proxy
        // ═══════════════════════════════════════════════════════════════════
        // Model heuristic: prediction error is mapped to a nominal
        // information quantity. This is not a measured bit count, memory reset,
        // or Landauer lower-bound calculation for the implementation.
        let bits_to_consolidate = prediction_error.abs() as f64 * 10.0; // Scale factor
        self.memory_landauer_cost = bits_to_consolidate * k_eff * t_eff * ln2;

        // ═══════════════════════════════════════════════════════════════════
        // Carnot-shaped effective-temperature normalization (model proxy)
        // ═══════════════════════════════════════════════════════════════════
        let t_cold = thresholds::THERMO_CARNOT_T_COLD;
        self.carnot_efficiency = if t_eff > t_cold {
            1.0 - t_cold / t_eff
        } else {
            0.0
        };

        // Normalized cognitive diagnostic: order × information proxy divided
        // by a scaled energy input. Neither term is a measured engine work/heat.
        let useful_work = order_parameter * information_processed;
        self.actual_efficiency = if energy_per_cycle > 0.0 {
            (useful_work / (energy_per_cycle * 1e15)).clamp(0.0, 1.0) // Scale to [0,1]
        } else {
            order_parameter // Fallback: order as efficiency proxy
        };

        self.efficiency_ratio = if self.carnot_efficiency > 0.01 {
            (self.actual_efficiency / self.carnot_efficiency).min(1.0)
        } else {
            0.0
        };

        // ═══════════════════════════════════════════════════════════════════
        // Heuristic insight score (not a fluctuation-theorem estimator)
        // ═══════════════════════════════════════════════════════════════════
        // Map an order-change / entropy-production proxy to a bounded score.
        // No forward/reverse path probabilities or stochastic entropy
        // production are sampled by this heuristic.
        let order_delta = order_parameter - prev_order_parameter;
        let entropy_decrease = if entropy_production_rate < -0.01 {
            entropy_production_rate.abs()
        } else if order_delta > 0.1 {
            // Sudden increase in the configured order proxy; this is not
            // an independently measured entropy decrease.
            order_delta
        } else {
            0.0
        };

        // The exponent uses the configured dimensionless model scale. It is
        // not a physical Boltzmann factor unless entropy and temperature are
        // calibrated to compatible physical units.
        self.insight_probability = if entropy_decrease > 0.0 {
            (-entropy_decrease / t_eff).exp()
        } else {
            0.0
        };

        self.insight_detected =
            self.insight_probability > thresholds::THERMO_INSIGHT_PROBABILITY_THRESHOLD;

        // ═══════════════════════════════════════════════════════════════════
        // Legacy covariance-symmetry diagnostic (not Onsager reciprocity)
        // ═══════════════════════════════════════════════════════════════════
        // These four telemetry channels are not conjugate flux/force pairs.
        let sample = [
            entropy_production_rate,
            energy_per_cycle,
            information_processed,
            coherence,
        ];
        self.history.push_back(sample);
        while self.history.len() > thresholds::THERMO_ONSAGER_WINDOW {
            self.history.pop_front();
        }

        if self.history.len() >= 3 {
            self.onsager_asymmetry = self.compute_covariance_symmetry_residual();
        }

        // ═══════════════════════════════════════════════════════════════════
        // Phase 6: Nonequilibrium Physics Integration
        // ═══════════════════════════════════════════════════════════════════

        // 6a. Legacy exponential-work proxy. The samples below are constructed
        // from cognitive telemetry, not measured work along a defined protocol.
        let work = energy_per_cycle - (energy_per_cycle * self.actual_efficiency);
        if work.is_finite() {
            self.work_samples.push_back(work);
            while self.work_samples.len() > thresholds::THERMO_JARZYNSKI_WINDOW {
                self.work_samples.pop_front();
            }
        }

        if self.work_samples.len() >= 3 {
            let beta = 1.0 / (k_eff * t_eff);
            let exp_sum: f64 = self.work_samples.iter().map(|&w| (-beta * w).exp()).sum();
            let avg_exp = exp_sum / self.work_samples.len() as f64;
            if avg_exp > 0.0 && avg_exp.is_finite() {
                self.jarzynski_free_energy = -avg_exp.ln() / beta;
            }
        }

        // Compare with HFE — divergence indicates model inaccuracy
        if hfe_total.is_finite() && self.jarzynski_free_energy.is_finite() {
            self.jarzynski_hfe_divergence = (self.jarzynski_free_energy - hfe_total).abs();
        }

        // 6c. Heuristic entropy-production trend flag for the configured regime.
        // A rising proxy is not by itself a violation of the restricted
        // near-equilibrium minimum-entropy-production result.
        use crate::consciousness::dissipative_consciousness::ThermodynamicRegime;
        self.prigogine_violated = matches!(regime, ThermodynamicRegime::LinearNonEquilibrium)
            && entropy_production_rate > self.prev_entropy_production + 0.01
            && self.prev_entropy_production > 0.001; // Only after warm-up

        // 6d. England dissipation-driven adaptation (England 2013)
        // When entropy production is high AND order is growing AND we're at
        // or near the edge of chaos, the dissipation is driving useful
        // self-organization — boost exploration rather than dampen it.
        #[cfg(feature = "england_dissipation")]
        {
            let order_delta = order_parameter - self.prev_order_parameter;
            let high_entropy = entropy_production_rate > 0.15;
            let order_growing = order_delta > 0.01;
            let prediction_error_decreasing = prediction_error < 0.3;
            let at_edge = matches!(
                regime,
                ThermodynamicRegime::EdgeOfChaos | ThermodynamicRegime::FarFromEquilibrium
            );
            self.england_exploration_boost =
                high_entropy && order_growing && prediction_error_decreasing && at_edge;
        }
        #[cfg(not(feature = "england_dissipation"))]
        {
            self.england_exploration_boost = false;
        }

        self.prev_order_parameter = order_parameter;
        self.prev_entropy_production = entropy_production_rate;

        // ═══════════════════════════════════════════════════════════════════
        // Epistemic Free Energy (arXiv 2601.17607)
        // ═══════════════════════════════════════════════════════════════════
        #[cfg(feature = "epistemic")]
        {
            // Use prediction_error as a proxy for belief_delta: high PE means
            // the model's beliefs shifted significantly this cycle.
            let belief_delta = prediction_error as f64;
            self.compute_epistemic_thermodynamics(
                effective_temperature,
                entropy_production_rate,
                belief_delta,
            );
        }
    }

    /// Compute heuristic epistemic diagnostics for this cycle.
    ///
    /// The current inputs are effective model scales and a scalar prediction-error
    /// displacement. This calculation does not compute a Wasserstein-2 distance
    /// between belief distributions, estimate trajectory entropy production, or
    /// test a physical thermodynamic speed limit. It is retained as a cognitive
    /// heuristic pending a distribution-based, dimensionally specified estimator.
    #[cfg(feature = "epistemic")]
    pub(crate) fn compute_epistemic_thermodynamics(
        &mut self,
        effective_temperature: f64,
        entropy_production: f64,
        belief_delta: f64,
    ) {
        let t_eff = effective_temperature.max(0.01);
        let sigma = entropy_production.max(0.001);

        // Heuristic score: scalar prediction-error displacement minus a
        // configured temperature-scaled entropy-production proxy. This is not
        // a distributional free-energy calculation.
        self.epistemic_free_energy = belief_delta.abs() - t_eff * sigma;

        // Legacy scalar displacement proxy; no probability distributions or
        // transport plan are computed, so this value is not Wasserstein-2.
        self.belief_wasserstein_distance = belief_delta.abs();

        // Heuristic ratio: scalar displacement proxy squared divided by the
        // configured temperature/entropy-production scale. Crossing 1.0 only
        // crosses an internal diagnostic threshold, not a physical bound.
        self.epistemic_speed_limit_violation =
            self.belief_wasserstein_distance.powi(2) / (t_eff * sigma);
    }

    /// Estimate the number of claim-validation model operations allowed by
    /// the abstract per-cycle budget.
    ///
    /// This subtracts the current attention/memory cost proxies and divides by a
    /// configured per-claim model cost. The budget and costs are effective model
    /// units, not joules, and this is not a Landauer-derived physical throughput
    /// bound. Actual work/energy measurement and logical bit-erasure accounting
    /// are separate requirements.
    ///
    /// Returns zero when the abstract budget is exhausted or the configured
    /// model cost is non-positive.
    #[cfg(feature = "epistemic")]
    pub fn epistemic_energy_budget(&self) -> usize {
        let available =
            (self.total_energy_budget - self.memory_landauer_cost - self.attention_erasure_cost)
                .max(0.0);
        // Configured model cost per claim proxy = k_eff * T_eff * ln(2).
        let k_eff = thresholds::K_CONSCIOUSNESS_BOLTZMANN;
        // Use the reference effective-temperature scale as a floor.
        let t_eff = if self.carnot_efficiency > 0.01 {
            // Recover the configured effective scale from the normalization.
            thresholds::THERMO_CARNOT_T_COLD / (1.0 - self.carnot_efficiency)
        } else {
            thresholds::THERMO_CARNOT_T_COLD
        }
        .max(0.01);
        let cost_per_claim = k_eff * t_eff * std::f64::consts::LN_2;
        if cost_per_claim <= 0.0 {
            return 0;
        }
        (available / cost_per_claim).floor() as usize
    }

    /// Legacy covariance transpose-difference diagnostic.
    ///
    /// The covariance matrix C_ij = mean((x_i - mean_i)(x_j - mean_j)) is
    /// symmetric by construction, so max |C_ij - C_ji| is zero apart from
    /// floating-point roundoff. This cannot test Onsager reciprocity.
    fn compute_covariance_symmetry_residual(&self) -> f64 {
        let n = self.history.len();
        if n < 2 {
            return 1.0;
        }

        // Compute means
        let mut means = [0.0f64; 4];
        for s in &self.history {
            for (i, v) in s.iter().enumerate() {
                means[i] += v;
            }
        }
        for m in &mut means {
            *m /= n as f64;
        }

        // Compute the covariance of cognitive telemetry channels.
        let mut cov = [[0.0f64; 4]; 4];
        for s in &self.history {
            for i in 0..4 {
                for j in 0..4 {
                    cov[i][j] += (s[i] - means[i]) * (s[j] - means[j]);
                }
            }
        }
        for row in &mut cov {
            for v in row.iter_mut() {
                *v /= n as f64;
            }
        }

        // Max asymmetry: max |L_ij - L_ji|
        let mut max_asym = 0.0f64;
        for i in 0..4 {
            for j in (i + 1)..4 {
                let asym = (cov[i][j] - cov[j][i]).abs();
                if asym.is_finite() && asym > max_asym {
                    max_asym = asym;
                }
            }
        }
        max_asym
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    use crate::consciousness::dissipative_consciousness::ThermodynamicRegime;

    fn make_bridge() -> ThermodynamicPhysicsBridge {
        ThermodynamicPhysicsBridge::default()
    }

    /// Default regime for tests
    fn default_regime() -> ThermodynamicRegime {
        ThermodynamicRegime::EdgeOfChaos
    }

    #[test]
    fn test_default_values() {
        let b = make_bridge();
        assert_eq!(b.attention_demon_bits, 0.0);
        assert_eq!(b.carnot_efficiency, 0.0);
        assert!(!b.insight_detected);
        assert_eq!(b.onsager_asymmetry, 1.0);
    }

    #[test]
    fn test_maxwell_demon_accounting() {
        let mut b = make_bridge();
        b.compute(
            0.8,   // attention_sensitivity
            100.0, // information_processed
            0.7,   // coherence
            0.3,   // prediction_error
            0.5,   // effective_temperature
            0.2,   // entropy_production_rate
            0.6,   // order_parameter
            1e-10, // energy_per_cycle
            0.5,   // prev_order_parameter
            0.5,   // hfe_total
            &default_regime(),
        );
        // Attention-information proxy = sensitivity × modeled information.
        assert!((b.attention_demon_bits - 80.0).abs() < 0.01);
        // Configured erasure-cost proxy >= heuristic work proxy by policy.
        assert!(b.attention_erasure_cost >= b.attention_work);
        // This ratio checks the configured model policy; neither field is a
        // measurement of physical work or measured bit-erasure heat.
        let ratio = b.attention_work / b.attention_erasure_cost;
        let expected = thresholds::THERMO_ATTENTION_DEMON_EFFICIENCY;
        assert!(
            (ratio - expected).abs() < 0.01,
            "expected ratio ~{expected}, got {ratio}",
        );
    }

    #[test]
    fn test_memory_cost_proxy_positive() {
        let mut b = make_bridge();
        b.compute(
            0.5,
            50.0,
            0.5,
            0.5,
            0.4,
            0.1,
            0.5,
            1e-10,
            0.5,
            0.5,
            &default_regime(),
        );
        assert!(b.memory_landauer_cost > 0.0);
        assert!(b.memory_landauer_cost.is_finite());
    }

    #[test]
    fn test_effective_temperature_normalization_bounded() {
        let mut b = make_bridge();
        // Larger configured effective-temperature scale raises the normalization.
        b.compute(
            0.5,
            50.0,
            0.5,
            0.3,
            1.0,
            0.1,
            0.5,
            1e-10,
            0.5,
            0.5,
            &default_regime(),
        );
        assert!(b.carnot_efficiency > 0.0);
        assert!(b.carnot_efficiency < 1.0);
        // Reference effective scale: eta_proxy = 1 - T_cold / T_eff.
        let t_cold = thresholds::THERMO_CARNOT_T_COLD;
        let expected = 1.0 - t_cold / 1.0;
        assert!(
            (b.carnot_efficiency - expected).abs() < 0.01,
            "expected ~{expected}, got {}",
            b.carnot_efficiency,
        );

        // Low temperature → low efficiency
        // Smaller configured effective-temperature scale approaches zero normalization.
        b.compute(
            0.5,
            50.0,
            0.5,
            0.3,
            0.25,
            0.1,
            0.5,
            1e-10,
            0.5,
            0.5,
            &default_regime(),
        );
        // With effective T below the reference scale, the normalization is zero.
        assert!(
            b.carnot_efficiency < 0.01,
            "expected ~0 below the effective reference scale, got {}",
            b.carnot_efficiency,
        );
    }

    #[test]
    fn test_insight_detection() {
        let mut b = make_bridge();
        // A moderate order-proxy increase crosses the heuristic event threshold.
        // The returned score is deterministic model arithmetic, not a sampled
        // fluctuation-theorem probability.
        b.compute(
            0.5,
            50.0,
            0.5,
            0.3,
            0.5,
            0.1,
            0.45,
            1e-10,
            0.3,
            0.5,
            &default_regime(),
        );
        assert!(b.insight_probability > 0.0);
        assert!(
            b.insight_probability > thresholds::THERMO_INSIGHT_PROBABILITY_THRESHOLD,
            "expected > {}, got {}",
            thresholds::THERMO_INSIGHT_PROBABILITY_THRESHOLD,
            b.insight_probability,
        );
        assert!(b.insight_detected);
    }

    #[test]
    fn test_covariance_symmetry_residual_is_near_zero() {
        let mut b = make_bridge();
        // Feed identical samples → perfectly symmetric → asymmetry ≈ 0
        for _ in 0..5 {
            b.compute(
                0.5,
                50.0,
                0.5,
                0.3,
                0.5,
                0.1,
                0.5,
                1e-10,
                0.5,
                0.5,
                &default_regime(),
            );
        }
        // Covariance of identical inputs = 0 variance → asymmetry = 0
        assert!(b.onsager_asymmetry < 0.01);
    }

    #[cfg(feature = "epistemic")]
    #[test]
    fn test_epistemic_free_energy_computation() {
        let mut b = make_bridge();
        // belief_delta=0.5, temperature=0.5, entropy_production=0.1
        b.compute_epistemic_thermodynamics(0.5, 0.1, 0.5);

        // F = |0.5| - 0.5 * 0.1 = 0.5 - 0.05 = 0.45
        assert!(
            (b.epistemic_free_energy - 0.45).abs() < 1e-9,
            "expected 0.45, got {}",
            b.epistemic_free_energy
        );

        // W2 = |0.5| = 0.5
        assert!((b.belief_wasserstein_distance - 0.5).abs() < 1e-9);

        // ESL = 0.5^2 / (0.5 * 0.1) = 0.25 / 0.05 = 5.0
        assert!(
            (b.epistemic_speed_limit_violation - 5.0).abs() < 1e-9,
            "expected ESL 5.0, got {}",
            b.epistemic_speed_limit_violation
        );
        // Ratio > 1.0 crosses the configured heuristic threshold.
        assert!(b.epistemic_speed_limit_violation > 1.0);
    }

    #[cfg(feature = "epistemic")]
    #[test]
    fn test_epistemic_speed_limit_respects_temperature() {
        let mut b = make_bridge();

        // Larger model scales lower this heuristic ratio.
        b.compute_epistemic_thermodynamics(2.0, 1.0, 0.3);
        let esl_hot = b.epistemic_speed_limit_violation;
        // ESL = 0.3^2 / (2.0 * 1.0) = 0.09 / 2.0 = 0.045
        assert!(
            esl_hot < 1.0,
            "larger model scales should lower the heuristic ratio, got {esl_hot}"
        );

        // Smaller model scales raise this heuristic ratio.
        b.compute_epistemic_thermodynamics(0.05, 0.01, 0.3);
        let esl_cold = b.epistemic_speed_limit_violation;
        // ESL = 0.09 / (0.05 * 0.01) = 0.09 / 0.0005 = 180.0
        assert!(
            esl_cold > 1.0,
            "smaller model scales should raise the heuristic ratio, got {esl_cold}"
        );
        assert!(esl_cold > esl_hot, "the small-scale heuristic ratio should exceed the large-scale ratio");

        // Zero belief delta → ESL = 0
        b.compute_epistemic_thermodynamics(0.5, 0.1, 0.0);
        assert!(
            b.epistemic_speed_limit_violation.abs() < 1e-12,
            "zero belief delta → zero ESL"
        );
        assert!(
            b.epistemic_free_energy < 0.0,
            "zero loss - positive T*H should be negative"
        );
    }

    #[test]
    fn test_all_values_finite() {
        let mut b = make_bridge();
        // Edge case: very small values
        b.compute(
            0.0,
            0.0,
            0.0,
            0.0,
            0.01,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            &default_regime(),
        );
        assert!(b.attention_demon_bits.is_finite());
        assert!(b.attention_work.is_finite());
        assert!(b.memory_landauer_cost.is_finite());
        assert!(b.carnot_efficiency.is_finite());
        assert!(b.insight_probability.is_finite());
        assert!(b.onsager_asymmetry.is_finite());

        // Edge case: very large values
        b.compute(
            1.0,
            1e6,
            1.0,
            1.0,
            2.0,
            1.0,
            1.0,
            1e-5,
            0.0,
            1.0,
            &default_regime(),
        );
        assert!(b.attention_demon_bits.is_finite());
        assert!(b.carnot_efficiency.is_finite());
        assert!(b.efficiency_ratio.is_finite());
    }

    #[cfg(feature = "epistemic")]
    #[test]
    fn test_epistemic_energy_budget() {
        let mut b = make_bridge();
        // After compute, the information-cost proxy fields are populated.
        b.compute(
            0.5,
            50.0,
            0.5,
            0.3,
            0.5,
            0.1,
            0.5,
            1e-10,
            0.5,
            0.5,
            &default_regime(),
        );

        // With default total_energy_budget=1.0, we should get a positive claim budget
        let budget = b.epistemic_energy_budget();
        assert!(
            budget > 0,
            "should be able to validate at least one claim, got {budget}"
        );

        // Exhausted budget → 0 claims
        b.total_energy_budget = 0.0;
        assert_eq!(
            b.epistemic_energy_budget(),
            0,
            "zero energy budget should yield zero claims"
        );

        // Large budget → many claims
        b.total_energy_budget = 1e6;
        let large_budget = b.epistemic_energy_budget();
        assert!(
            large_budget > budget,
            "larger energy budget should allow more claims: {large_budget} vs {budget}"
        );
    }
}
