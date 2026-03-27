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
            history: std::collections::VecDeque::with_capacity(thresholds::THERMO_ONSAGER_WINDOW + 1),
            work_samples: std::collections::VecDeque::with_capacity(thresholds::THERMO_JARZYNSKI_WINDOW + 1),
            jarzynski_free_energy: 0.0,
            jarzynski_hfe_divergence: 0.0,
            prigogine_violated: false,
            prev_entropy_production: 0.0,
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

        self.attention_demon_bits = (attention_sensitivity * information_processed)
            .max(0.0);
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

        self.insight_probability = if entropy_decrease > 0.0 {
            (-entropy_decrease / (k_eff * t_eff)).exp()
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
            let exp_sum: f64 = self
                .work_samples
                .iter()
                .map(|&w| (-beta * w).exp())
                .sum();
            let avg_exp = exp_sum / self.work_samples.len() as f64;
            if avg_exp > 0.0 && avg_exp.is_finite() {
                self.jarzynski_free_energy = -avg_exp.ln() / beta;
            }
        }

        // Compare with HFE — divergence indicates model inaccuracy
        if hfe_total.is_finite() && self.jarzynski_free_energy.is_finite() {
            self.jarzynski_hfe_divergence =
                (self.jarzynski_free_energy - hfe_total).abs();
        }

        // 6c. Prigogine minimum entropy production principle
        // Science: Prigogine (1947) — in LinearNonEquilibrium, steady state minimizes σ.
        // Violation: entropy production increasing when it should be decreasing.
        use crate::consciousness::dissipative_consciousness::ThermodynamicRegime;
        self.prigogine_violated = matches!(regime, ThermodynamicRegime::LinearNonEquilibrium)
            && entropy_production_rate > self.prev_entropy_production + 0.01
            && self.prev_entropy_production > 0.001; // Only after warm-up

        self.prev_entropy_production = entropy_production_rate;
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
            0.8,  // attention_sensitivity
            100.0, // information_processed
            0.7,  // coherence
            0.3,  // prediction_error
            0.5,  // effective_temperature
            0.2,  // entropy_production_rate
            0.6,  // order_parameter
            1e-10, // energy_per_cycle
            0.5,  // prev_order_parameter
            0.5,  // hfe_total
            &default_regime(),
        );
        // Demon gathers bits = sensitivity × info
        assert!((b.attention_demon_bits - 80.0).abs() < 0.01);
        // Erasure cost ≥ work extracted (Landauer bound)
        assert!(b.attention_erasure_cost >= b.attention_work);
        // Work = bits × k × T × ln2 × efficiency (0.7)
        // Erasure = bits × k × T × ln2
        let ratio = b.attention_work / b.attention_erasure_cost;
        assert!((ratio - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_landauer_cost_positive() {
        let mut b = make_bridge();
        b.compute(0.5, 50.0, 0.5, 0.5, 0.4, 0.1, 0.5, 1e-10, 0.5, 0.5, &default_regime());
        assert!(b.memory_landauer_cost > 0.0);
        assert!(b.memory_landauer_cost.is_finite());
    }

    #[test]
    fn test_carnot_efficiency_bounded() {
        let mut b = make_bridge();
        // High temperature → high Carnot efficiency
        b.compute(0.5, 50.0, 0.5, 0.3, 1.0, 0.1, 0.5, 1e-10, 0.5, 0.5, &default_regime());
        assert!(b.carnot_efficiency > 0.0);
        assert!(b.carnot_efficiency < 1.0);
        // T_hot=1.0, T_cold=0.2 → η = 1 - 0.2/1.0 = 0.8
        assert!((b.carnot_efficiency - 0.8).abs() < 0.01);

        // Low temperature → low efficiency
        b.compute(0.5, 50.0, 0.5, 0.3, 0.25, 0.1, 0.5, 1e-10, 0.5, 0.5, &default_regime());
        assert!((b.carnot_efficiency - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_insight_detection() {
        let mut b = make_bridge();
        // Sudden order increase (0.3 → 0.8) = entropy-decreasing event
        b.compute(0.5, 50.0, 0.5, 0.3, 0.5, 0.1, 0.8, 1e-10, 0.3, 0.5, &default_regime());
        assert!(b.insight_probability > 0.0);
        // The delta is 0.5, so exp(-0.5/(1.0*0.5)) = exp(-1.0) ≈ 0.37
        assert!(b.insight_probability > thresholds::THERMO_INSIGHT_PROBABILITY_THRESHOLD);
        assert!(b.insight_detected);
    }

    #[test]
    fn test_onsager_symmetry() {
        let mut b = make_bridge();
        // Feed identical samples → perfectly symmetric → asymmetry ≈ 0
        for _ in 0..5 {
            b.compute(0.5, 50.0, 0.5, 0.3, 0.5, 0.1, 0.5, 1e-10, 0.5, 0.5, &default_regime());
        }
        // Covariance of identical inputs = 0 variance → asymmetry = 0
        assert!(b.onsager_asymmetry < 0.01);
    }

    #[test]
    fn test_all_values_finite() {
        let mut b = make_bridge();
        // Edge case: very small values
        b.compute(0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, &default_regime());
        assert!(b.attention_demon_bits.is_finite());
        assert!(b.attention_work.is_finite());
        assert!(b.memory_landauer_cost.is_finite());
        assert!(b.carnot_efficiency.is_finite());
        assert!(b.insight_probability.is_finite());
        assert!(b.onsager_asymmetry.is_finite());

        // Edge case: very large values
        b.compute(1.0, 1e6, 1.0, 1.0, 2.0, 1.0, 1.0, 1e-5, 0.0, 1.0, &default_regime());
        assert!(b.attention_demon_bits.is_finite());
        assert!(b.carnot_efficiency.is_finite());
        assert!(b.efficiency_ratio.is_finite());
    }
}
