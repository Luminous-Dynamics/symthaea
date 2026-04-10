// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Thermodynamic Physics Bridge: Maps classical thermodynamic concepts
//! into cognitive operations within the consciousness loop.
//!
//! Bridges the gap between symthaea-core's physics encodings (ThermoEncoder,
//! Maxwell's Demon, Landauer limit, Carnot efficiency, fluctuation theorem,
//! Onsager coefficients) and the live cognitive cycle.
//!
//! # Concepts Mapped
//!
//! | Physics | Cognition | Mechanism |
//! |---------|-----------|-----------|
//! | Maxwell's Demon | Attention | Selective info extraction costs energy |
//! | Landauer Limit | Memory consolidation | Minimum energy to write memory |
//! | Carnot Efficiency | Consciousness efficiency | Work output / energy input |
//! | Fluctuation Theorem | Spontaneous insight | Rare entropy-decreasing events |
//! | Onsager Coefficients | Cross-coupling health | Transport symmetry = integration |
//!
//! # Science
//!
//! - Landauer (1961) — information erasure has minimum thermodynamic cost
//! - Szilard (1929) — Maxwell's demon extracts work from information
//! - Carnot (1824) — maximum heat engine efficiency
//! - Jarzynski (1997) — rare fluctuations carry free energy information
//! - Onsager (1931) — reciprocal relations in near-equilibrium transport

use serde::{Deserialize, Serialize};

use super::thresholds;

/// Thermodynamic physics bridge: computes physics-derived cognitive metrics each cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicPhysicsBridge {
    // ═══════════════════════════════════════════════════════════════════════
    // Maxwell's Demon as Attention
    // ═══════════════════════════════════════════════════════════════════════
    /// Bits of information gathered by attention this cycle.
    pub attention_demon_bits: f64,
    /// Work extracted by attention (information → action) in effective joules.
    pub attention_work: f64,
    /// Landauer cost of erasing attention memory (effective joules).
    pub attention_erasure_cost: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Landauer Limit for Memory
    // ═══════════════════════════════════════════════════════════════════════
    /// Minimum energy to consolidate memory this cycle (effective joules).
    pub memory_landauer_cost: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Carnot Efficiency
    // ═══════════════════════════════════════════════════════════════════════
    /// Theoretical maximum efficiency (1 - T_cold/T_hot).
    pub carnot_efficiency: f64,
    /// Actual consciousness efficiency (useful work / total energy).
    pub actual_efficiency: f64,
    /// Ratio actual/Carnot — performance relative to theoretical max.
    pub efficiency_ratio: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Fluctuation Theorem for Insight
    // ═══════════════════════════════════════════════════════════════════════
    /// Probability of spontaneous insight (rare entropy-decreasing event).
    pub insight_probability: f64,
    /// Whether an insight event was detected this cycle.
    pub insight_detected: bool,

    // ═══════════════════════════════════════════════════════════════════════
    // Onsager Cross-Coupling Health
    // ═══════════════════════════════════════════════════════════════════════
    /// Onsager symmetry measure (0 = perfectly symmetric, higher = asymmetric).
    /// Lower is better — symmetric transport indicates well-integrated consciousness.
    pub onsager_asymmetry: f64,

    /// Rolling history for Onsager computation (last N cycles of [entropy, energy, info, coherence]).
    #[serde(skip)]
    history: std::collections::VecDeque<[f64; 4]>,

    // ═══════════════════════════════════════════════════════════════════════
    // Nonequilibrium Physics (Phase 6)
    // ═══════════════════════════════════════════════════════════════════════
    /// Rolling work samples for Jarzynski free energy estimation.
    #[serde(skip)]
    pub(crate) work_samples: std::collections::VecDeque<f64>,

    /// Jarzynski free energy estimate ΔF (from nonequilibrium work).
    /// Science: Jarzynski (1997) — ⟨exp(-W/k_BT)⟩ = exp(-ΔF/k_BT).
    pub jarzynski_free_energy: f64,

    /// Divergence between Jarzynski estimate and HFE total free energy.
    /// High divergence indicates model inaccuracy.
    pub jarzynski_hfe_divergence: f64,

    /// Whether Prigogine minimum entropy production was violated this cycle.
    /// In LinearNonEquilibrium regime, entropy production should decrease toward steady state.
    /// Science: Prigogine (1947) — steady states minimize entropy production.
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
    // Epistemic Free Energy (arXiv 2601.17607)
    // ═══════════════════════════════════════════════════════════════════════
    /// Epistemic free energy: F = E[loss] - T*H[beliefs].
    /// Positive = beliefs are costly relative to entropy; negative = beliefs are cheap.
    /// Science: Thermodynamic Theory of Learning (Still et al., arXiv 2601.17607).
    #[cfg(feature = "epistemic")]
    pub epistemic_free_energy: f64,

    /// Epistemic speed limit violation: ESL = W2^2 / (T * sigma).
    /// >1.0 means beliefs moved faster than thermodynamically permitted — suspicious.
    /// Science: T * sigma >= W_2(q_0, q_1)^2 (arXiv 2601.17607).
    #[cfg(feature = "epistemic")]
    pub epistemic_speed_limit_violation: f64,

    /// Wasserstein-2 proxy: how far beliefs moved this cycle (|belief_delta|).
    #[cfg(feature = "epistemic")]
    pub belief_wasserstein_distance: f64,

    /// Total energy budget available per cycle (effective joules).
    /// Constrains how many epistemic operations (claim validations, memory writes)
    /// can be performed. Default 1.0 (arbitrary units).
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
    /// Compute all physics bridge metrics for this cycle.
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
        // Attention selectively focuses on information = demon gathering bits.
        // Each bit gathered can extract k_B_eff * T * ln(2) work.
        // Erasing that information costs the same (Landauer 1961).
        let k_eff = thresholds::K_CONSCIOUSNESS_BOLTZMANN;
        let t_eff = effective_temperature.max(0.01); // Avoid division by zero
        let ln2 = std::f64::consts::LN_2;

        self.attention_demon_bits = (attention_sensitivity * information_processed).max(0.0);
        let demon_efficiency = thresholds::THERMO_ATTENTION_DEMON_EFFICIENCY;
        self.attention_work = self.attention_demon_bits * k_eff * t_eff * ln2 * demon_efficiency;
        self.attention_erasure_cost = self.attention_demon_bits * k_eff * t_eff * ln2;

        // ═══════════════════════════════════════════════════════════════════
        // Landauer Limit for Memory (Landauer 1961)
        // ═══════════════════════════════════════════════════════════════════
        // Memory consolidation writes bits. Minimum cost = bits * k_B_eff * T * ln(2).
        // Use prediction_error as proxy for bits needing consolidation
        // (high error = more to learn = more bits to write).
        let bits_to_consolidate = prediction_error.abs() as f64 * 10.0; // Scale factor
        self.memory_landauer_cost = bits_to_consolidate * k_eff * t_eff * ln2;

        // ═══════════════════════════════════════════════════════════════════
        // Carnot Efficiency (Carnot 1824)
        // ═══════════════════════════════════════════════════════════════════
        let t_cold = thresholds::THERMO_CARNOT_T_COLD;
        self.carnot_efficiency = if t_eff > t_cold {
            1.0 - t_cold / t_eff
        } else {
            0.0
        };

        // Actual efficiency: useful work (order × info) / total energy
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
        // Fluctuation Theorem for Insight (Jarzynski 1997)
        // ═══════════════════════════════════════════════════════════════════
        // Rare entropy-decreasing events = spontaneous insight.
        // P(insight) = exp(-|delta_entropy|) for negative entropy fluctuations.
        let order_delta = order_parameter - prev_order_parameter;
        let entropy_decrease = if entropy_production_rate < -0.01 {
            entropy_production_rate.abs()
        } else if order_delta > 0.1 {
            // Sudden order increase = entropy-decreasing event
            order_delta
        } else {
            0.0
        };

        // Use temperature alone as fluctuation scale (not k_eff, which scales energy).
        // The Boltzmann factor exp(-ΔS/(k_B T)) in consciousness space uses T directly
        // since entropy_decrease is already in dimensionless consciousness units.
        self.insight_probability = if entropy_decrease > 0.0 {
            (-entropy_decrease / t_eff).exp()
        } else {
            0.0
        };

        self.insight_detected =
            self.insight_probability > thresholds::THERMO_INSIGHT_PROBABILITY_THRESHOLD;

        // ═══════════════════════════════════════════════════════════════════
        // Onsager Cross-Coupling Health (Onsager 1931)
        // ═══════════════════════════════════════════════════════════════════
        // Track 4 thermodynamic forces over time, compute transport matrix symmetry.
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
            self.onsager_asymmetry = self.compute_onsager_asymmetry();
        }

        // ═══════════════════════════════════════════════════════════════════
        // Phase 6: Nonequilibrium Physics Integration
        // ═══════════════════════════════════════════════════════════════════

        // 6a. Jarzynski free energy estimation from work samples
        // Science: Jarzynski (1997) — ⟨exp(-W/k_BT)⟩ = exp(-ΔF/k_BT)
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

        // 6c. Prigogine minimum entropy production principle
        // Science: Prigogine (1947) — in LinearNonEquilibrium, steady state minimizes σ.
        // Violation: entropy production increasing when it should be decreasing.
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

    /// Compute epistemic thermodynamic metrics for this cycle.
    ///
    /// # Arguments
    /// - `effective_temperature` — system effective temperature (k_B_eff units)
    /// - `entropy_production` — entropy production rate this cycle
    /// - `belief_delta` — how far beliefs moved this cycle (Wasserstein-2 proxy)
    ///
    /// # Science
    ///
    /// The Thermodynamic Theory of Learning (arXiv 2601.17607) establishes that
    /// learning has an intrinsic thermodynamic cost. The Epistemic Speed Limit
    /// (ESL) bounds how fast beliefs can change:
    ///
    ///   T * sigma >= W_2(q_0, q_1)^2
    ///
    /// where T is temperature, sigma is entropy production, and W_2 is the
    /// Wasserstein-2 distance between consecutive belief distributions.
    /// Violation (ESL > 1.0) indicates beliefs changed faster than
    /// thermodynamically permitted — a red flag for hallucination or instability.
    #[cfg(feature = "epistemic")]
    pub(crate) fn compute_epistemic_thermodynamics(
        &mut self,
        effective_temperature: f64,
        entropy_production: f64,
        belief_delta: f64,
    ) {
        let t_eff = effective_temperature.max(0.01);
        let sigma = entropy_production.max(0.001);

        // F = E[loss] - T * H[beliefs]
        // We use |belief_delta| as proxy for expected loss and sigma as proxy for
        // belief entropy (higher entropy production ≈ higher belief entropy).
        self.epistemic_free_energy = belief_delta.abs() - t_eff * sigma;

        // W_2 proxy: absolute belief displacement
        self.belief_wasserstein_distance = belief_delta.abs();

        // ESL violation = W_2^2 / (T * sigma)
        // >1.0 means beliefs moved faster than the thermodynamic bound allows
        self.epistemic_speed_limit_violation =
            self.belief_wasserstein_distance.powi(2) / (t_eff * sigma);
    }

    /// Compute the maximum number of knowledge claims that can be validated
    /// this cycle given the available energy budget.
    ///
    /// Available energy = total budget - memory Landauer cost - attention erasure cost.
    /// Each claim validation costs approximately one Landauer bit erasure at the
    /// current effective temperature: k_B_eff * T * ln(2).
    ///
    /// Returns 0 when energy is exhausted or temperature is negligible.
    ///
    /// # Science
    ///
    /// Landauer (1961) — each irreversible bit operation (claim evaluation)
    /// dissipates at least k_B * T * ln(2) energy. This bounds epistemic
    /// throughput by thermodynamic cost.
    #[cfg(feature = "epistemic")]
    pub fn epistemic_energy_budget(&self) -> usize {
        let available =
            (self.total_energy_budget - self.memory_landauer_cost - self.attention_erasure_cost)
                .max(0.0);
        // Cost per claim = k_B_eff * T_eff * ln(2)
        let k_eff = thresholds::K_CONSCIOUSNESS_BOLTZMANN;
        // Use Carnot T_cold as floor to avoid division issues when actual_efficiency is 0
        let t_eff = if self.carnot_efficiency > 0.01 {
            // Recover T_hot from Carnot: η = 1 - T_cold/T_hot → T_hot = T_cold / (1 - η)
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

    /// Compute Onsager reciprocal relation asymmetry from transport history.
    /// Returns max |L_ij - L_ji| across all pairs.
    fn compute_onsager_asymmetry(&self) -> f64 {
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

        // Compute covariance matrix (proxy for Onsager L_ij)
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
        // Demon gathers bits = sensitivity × info
        assert!((b.attention_demon_bits - 80.0).abs() < 0.01);
        // Erasure cost ≥ work extracted (Landauer bound)
        assert!(b.attention_erasure_cost >= b.attention_work);
        // Work = bits × k × T × ln2 × efficiency (THERMO_ATTENTION_DEMON_EFFICIENCY)
        // Erasure = bits × k × T × ln2
        let ratio = b.attention_work / b.attention_erasure_cost;
        let expected = thresholds::THERMO_ATTENTION_DEMON_EFFICIENCY;
        assert!(
            (ratio - expected).abs() < 0.01,
            "expected ratio ~{expected}, got {ratio}",
        );
    }

    #[test]
    fn test_landauer_cost_positive() {
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
    fn test_carnot_efficiency_bounded() {
        let mut b = make_bridge();
        // High temperature → high Carnot efficiency
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
        // T_hot=1.0, T_cold=THERMO_CARNOT_T_COLD → η = 1 - T_cold/1.0
        let t_cold = thresholds::THERMO_CARNOT_T_COLD;
        let expected = 1.0 - t_cold / 1.0;
        assert!(
            (b.carnot_efficiency - expected).abs() < 0.01,
            "expected ~{expected}, got {}",
            b.carnot_efficiency,
        );

        // Low temperature → low efficiency
        // T_hot=0.25+, T_cold=THERMO_CARNOT_T_COLD → approaches 0
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
        // With T_hot=0.25 < T_cold=0.3, Carnot = 0.0 (can't extract work)
        assert!(
            b.carnot_efficiency < 0.01,
            "expected ~0 with T_hot < T_cold, got {}",
            b.carnot_efficiency,
        );
    }

    #[test]
    fn test_insight_detection() {
        let mut b = make_bridge();
        // Moderate order increase (0.3 → 0.45) = entropy-decreasing event
        // order_delta = 0.15 > 0.1 threshold, t_eff = 0.5
        // P(insight) = exp(-0.15 / 0.5) = exp(-0.3) ≈ 0.741 > THRESHOLD (0.7)
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
    fn test_onsager_symmetry() {
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
        // ESL > 1.0 → beliefs moved too fast
        assert!(b.epistemic_speed_limit_violation > 1.0);
    }

    #[cfg(feature = "epistemic")]
    #[test]
    fn test_epistemic_speed_limit_respects_temperature() {
        let mut b = make_bridge();

        // High temperature + high entropy production → ESL low (thermodynamically affordable)
        b.compute_epistemic_thermodynamics(2.0, 1.0, 0.3);
        let esl_hot = b.epistemic_speed_limit_violation;
        // ESL = 0.3^2 / (2.0 * 1.0) = 0.09 / 2.0 = 0.045
        assert!(
            esl_hot < 1.0,
            "hot system should allow fast belief change, got ESL={esl_hot}"
        );

        // Low temperature + low entropy production → ESL high (thermodynamically expensive)
        b.compute_epistemic_thermodynamics(0.05, 0.01, 0.3);
        let esl_cold = b.epistemic_speed_limit_violation;
        // ESL = 0.09 / (0.05 * 0.01) = 0.09 / 0.0005 = 180.0
        assert!(
            esl_cold > 1.0,
            "cold system should flag fast belief change, got ESL={esl_cold}"
        );
        assert!(esl_cold > esl_hot, "cold ESL should exceed hot ESL");

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
        // After compute, memory_landauer_cost and attention_erasure_cost are populated
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
