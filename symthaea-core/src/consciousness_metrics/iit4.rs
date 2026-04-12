// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! IIT 4.0 Measures
//!
//! Reference: Albantakis et al. (2023) "Integrated Information Theory (IIT) 4.0"

use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

use super::{ContinuousEntropyEstimator, TruePhiCalculator};

/// IIT 4.0 result containing intrinsic difference measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IIT4Result {
    /// Intrinsic difference (id) - fundamental measure
    pub intrinsic_difference: f64,
    /// Small phi (φ) - irreducibility of a mechanism
    pub small_phi: f64,
    /// Big Phi (Φ) - integrated information of the system
    pub big_phi: f64,
    /// Intrinsic information (ii) - how much the state specifies itself
    pub intrinsic_information: f64,
    /// Number of concepts (mechanisms with φ > 0)
    pub concept_count: usize,
    /// Concept structure: irreducible cause-effect mechanisms.
    /// Only populated when `iit4` feature is enabled.
    #[cfg(feature = "iit4")]
    pub concepts: Vec<ConceptStructure>,
}

/// A concept in IIT 4.0: an irreducible cause-effect mechanism.
///
/// Each mechanism in the system has a cause repertoire (constraints on past)
/// and an effect repertoire (constraints on future). The mechanism's phi (φ)
/// is the irreducibility of this cause-effect structure under all partitions.
///
/// Science: Albantakis et al. (2023) — concepts are the "atoms of experience".
/// A concept with φ > 0 specifies an irreducible cause-effect distinction.
#[cfg(feature = "iit4")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptStructure {
    /// Index of the mechanism component
    pub mechanism_index: usize,
    /// Small phi (irreducibility) of this concept
    pub phi: f64,
    /// Cause information: how much this mechanism constrains its past
    pub cause_info: f64,
    /// Effect information: how much this mechanism constrains its future
    pub effect_info: f64,
    /// Combined cause-effect power: min(cause_info, effect_info)
    /// IIT 4.0 uses the minimum to ensure both directions are specified.
    pub cause_effect_power: f64,
}

/// IIT 4.0 Calculator
///
/// Implements the updated IIT 4.0 formalism with:
/// - Intrinsic difference as the fundamental measure
/// - Improved irreducibility computation
/// - Intrinsic information quantification
///
/// ## Entropy estimation
///
/// Intrinsic difference uses histogram-based JSD (Jensen-Shannon divergence).
/// The bin count controls the bias-variance tradeoff:
/// - Fewer bins (8-16): lower variance, higher bias (smooths differences)
/// - More bins (32-64): lower bias, higher variance (noisy with few samples)
///
/// Default: 16 bins. For HV dimensions ≥ 1000, the bias from 16 bins is
/// small relative to the signal. For smaller dimensions, consider Sturges'
/// rule: `num_bins = ceil(log2(n)) + 1`.
///
/// Max approximation error for JSD with k bins and n samples:
/// O(k / (2n)) + O(1 / (k × bin_width)) (discretization + finite-sample).
#[derive(Debug, Clone)]
pub struct IIT4Calculator {
    /// Base entropy estimator
    estimator: ContinuousEntropyEstimator,
    /// Threshold for considering φ > 0
    phi_threshold: f64,
    /// Number of histogram bins for intrinsic difference computation
    num_bins: usize,
}

impl Default for IIT4Calculator {
    fn default() -> Self {
        Self::new()
    }
}

impl IIT4Calculator {
    /// Create a new IIT 4.0 calculator with default 16 bins
    pub fn new() -> Self {
        Self {
            estimator: ContinuousEntropyEstimator::fast(),
            phi_threshold: 0.001,
            num_bins: 16,
        }
    }

    /// Create with custom bin count for entropy estimation
    pub fn with_bins(num_bins: usize) -> Self {
        Self {
            estimator: ContinuousEntropyEstimator::fast(),
            phi_threshold: 0.001,
            num_bins: num_bins.max(2),
        }
    }

    /// Compute intrinsic difference between two distributions
    ///
    /// id(p, q) = D_KL(p || m) + D_KL(q || m)
    /// where m = (p + q) / 2 (Jensen-Shannon style)
    pub fn intrinsic_difference(&self, p: &ContinuousHV, q: &ContinuousHV) -> f64 {
        let n = p.values.len().min(q.values.len());
        if n == 0 {
            return 0.0;
        }

        // Compute histogram distributions (bin count configurable via with_bins())
        let num_bins = self.num_bins;
        let mut p_counts = vec![0usize; num_bins];
        let mut q_counts = vec![0usize; num_bins];

        for i in 0..n {
            let p_bin = (((p.values[i] + 1.0) / 2.0).clamp(0.0, 0.9999) * num_bins as f32) as usize;
            let q_bin = (((q.values[i] + 1.0) / 2.0).clamp(0.0, 0.9999) * num_bins as f32) as usize;
            p_counts[p_bin] += 1;
            q_counts[q_bin] += 1;
        }

        let total = n as f64;
        let mut id = 0.0;

        for i in 0..num_bins {
            let p_prob = p_counts[i] as f64 / total;
            let q_prob = q_counts[i] as f64 / total;
            let m_prob = (p_prob + q_prob) / 2.0;

            if m_prob > 1e-10 {
                if p_prob > 1e-10 {
                    id += p_prob * (p_prob / m_prob).log2();
                }
                if q_prob > 1e-10 {
                    id += q_prob * (q_prob / m_prob).log2();
                }
            }
        }

        id / 2.0 // Normalize to [0, 1] range for identical distributions
    }

    /// Compute small phi (φ) - irreducibility of a mechanism
    ///
    /// φ measures how much a mechanism's cause-effect is reduced
    /// when the system is partitioned.
    pub fn small_phi(&self, mechanism: &ContinuousHV, context: &[ContinuousHV]) -> f64 {
        if context.is_empty() {
            return 0.0;
        }

        // Compute unpartitioned cause-effect
        let refs: Vec<&ContinuousHV> = context.iter().collect();
        let whole = ContinuousHV::bundle(&refs);
        let whole_mi = self.estimator.mutual_information_fast(mechanism, &whole);

        // Find minimum over all partitions
        let mut min_mi = f64::INFINITY;

        // For each possible partition of context
        let n = context.len();
        for mask in 1..(1 << n) - 1 {
            let mut part_a = Vec::new();
            let mut part_b = Vec::new();

            for i in 0..n {
                if (mask & (1 << i)) != 0 {
                    part_a.push(&context[i]);
                } else {
                    part_b.push(&context[i]);
                }
            }

            if part_a.is_empty() || part_b.is_empty() {
                continue;
            }

            // Compute MI for partitioned system
            let bundle_a = ContinuousHV::bundle(&part_a);
            let bundle_b = ContinuousHV::bundle(&part_b);
            let mi_a = self.estimator.mutual_information_fast(mechanism, &bundle_a);
            let mi_b = self.estimator.mutual_information_fast(mechanism, &bundle_b);

            let partition_mi = mi_a + mi_b;
            min_mi = min_mi.min(partition_mi);
        }

        // φ = whole - min(partitioned)
        (whole_mi - min_mi).max(0.0)
    }

    /// Compute intrinsic information
    ///
    /// How much the current state specifies about itself
    /// (self-information under the mechanism)
    pub fn intrinsic_information(&self, state: &ContinuousHV) -> f64 {
        // Use entropy as a proxy for self-information
        // Lower entropy = more specific state = higher intrinsic info
        let h = self.estimator.entropy(state);
        let max_h = (self.num_bins as f64).log2(); // Max entropy for uniform distribution
        (max_h - h).max(0.0)
    }

    /// Compute full IIT 4.0 analysis
    pub fn analyze(&self, components: &[ContinuousHV]) -> IIT4Result {
        let n = components.len();
        if n < 2 {
            return IIT4Result {
                intrinsic_difference: 0.0,
                small_phi: 0.0,
                big_phi: 0.0,
                intrinsic_information: 0.0,
                concept_count: 0,
                #[cfg(feature = "iit4")]
                concepts: Vec::new(),
            };
        }

        // Compute pairwise intrinsic differences
        let mut total_id = 0.0;
        let mut pair_count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                total_id += self.intrinsic_difference(&components[i], &components[j]);
                pair_count += 1;
            }
        }
        let avg_id = if pair_count > 0 {
            total_id / pair_count as f64
        } else {
            0.0
        };

        // Compute small phi for each component and build concept structures
        let mut total_phi = 0.0;
        let mut concept_count = 0;
        #[cfg(feature = "iit4")]
        let mut concepts = Vec::new();

        for i in 0..n {
            let context: Vec<ContinuousHV> = components
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, c)| c.clone())
                .collect();

            let phi_i = self.small_phi(&components[i], &context);
            total_phi += phi_i;
            if phi_i > self.phi_threshold {
                concept_count += 1;
            }

            #[cfg(feature = "iit4")]
            {
                // Compute cause and effect info for this mechanism
                // Cause info: MI between mechanism and context (past → present)
                let refs: Vec<&ContinuousHV> = context.iter().collect();
                let bundle = ContinuousHV::bundle(&refs);
                let cause_info = self
                    .estimator
                    .mutual_information_fast(&components[i], &bundle);

                // Effect info: use intrinsic information as proxy for future constraint
                let effect_info = self.intrinsic_information(&components[i]);

                // IIT 4.0: cause-effect power = min(cause, effect)
                let cause_effect_power = cause_info.min(effect_info);

                if phi_i > self.phi_threshold {
                    concepts.push(ConceptStructure {
                        mechanism_index: i,
                        phi: phi_i,
                        cause_info,
                        effect_info,
                        cause_effect_power,
                    });
                }
            }
        }

        // Compute big Phi using TruePhiCalculator
        let calc = TruePhiCalculator::new();
        let phi_result = calc.compute_true_phi(components);

        // Compute total intrinsic information
        let total_ii: f64 = components
            .iter()
            .map(|c| self.intrinsic_information(c))
            .sum();

        IIT4Result {
            intrinsic_difference: avg_id,
            small_phi: total_phi / n as f64,
            big_phi: phi_result.phi,
            intrinsic_information: total_ii / n as f64,
            concept_count,
            #[cfg(feature = "iit4")]
            concepts,
        }
    }
}
