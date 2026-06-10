// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal IIT (Cause-Effect Information)
//!
//! Based on IIT 3.0's cause-effect repertoire framework.
//! Reference: Oizumi et al. (2014) - "From the Phenomenology to the Mechanisms"

use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

use super::{ContinuousEntropyEstimator, TemporalTransition, TruePartition, TruePhiCalculator};

/// Cause-effect information result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CauseEffectInfo {
    /// Cause information: how much current state specifies past (ci)
    pub cause_info: f64,
    /// Effect information: how much current state specifies future (ei)
    pub effect_info: f64,
    /// Integrated cause information (φ_cause)
    pub integrated_cause: f64,
    /// Integrated effect information (φ_effect)
    pub integrated_effect: f64,
    /// Total cause-effect information (min of cause and effect)
    pub phi_cause_effect: f64,
    /// Cause repertoire entropy
    pub cause_entropy: f64,
    /// Effect repertoire entropy
    pub effect_entropy: f64,
}

/// Temporal Φ calculator for cause-effect analysis
///
/// Extends IIT with temporal dynamics:
/// - Cause information: I(current; past)
/// - Effect information: I(current; future)
/// - Integrated cause: φ_cause = I(current; past | partition)
/// - Integrated effect: φ_effect = I(current; future | partition)
///
/// Reference: Oizumi et al. (2014) - "From the Phenomenology to the Mechanisms"
#[derive(Debug, Clone)]
pub struct TemporalPhiCalculator {
    /// Base calculator for entropy computations (reserved for future use)
    #[allow(dead_code)]
    base: TruePhiCalculator,
    /// Continuous entropy estimator for MI
    estimator: ContinuousEntropyEstimator,
}

impl Default for TemporalPhiCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalPhiCalculator {
    /// Create a new temporal calculator
    pub fn new() -> Self {
        Self {
            base: TruePhiCalculator::new(),
            estimator: ContinuousEntropyEstimator::fast(),
        }
    }

    /// Create with custom estimator
    pub fn with_estimator(estimator: ContinuousEntropyEstimator) -> Self {
        Self {
            base: TruePhiCalculator::new(),
            estimator,
        }
    }

    /// Compute cause information I(current; past)
    ///
    /// How much does the current state tell us about what caused it?
    /// Uses mutual information between current state and the prior state.
    pub fn cause_information(&self, transition: &TemporalTransition) -> f64 {
        // Use the fast MI method
        self.estimator
            .mutual_information_fast(&transition.next, &transition.current)
    }

    /// Compute effect information I(current; future)
    ///
    /// How much does the current state tell us about what will happen?
    pub fn effect_information(&self, transition: &TemporalTransition) -> f64 {
        // Effect info is the same MI but conceptually different
        self.estimator
            .mutual_information_fast(&transition.current, &transition.next)
    }

    /// Compute cause repertoire entropy
    ///
    /// The entropy of the cause repertoire represents the uncertainty
    /// about past states given the current mechanism.
    pub fn cause_repertoire_entropy(&self, past_states: &[ContinuousHV]) -> f64 {
        if past_states.is_empty() {
            return 0.0;
        }

        // Bundle past states and compute entropy
        let refs: Vec<&ContinuousHV> = past_states.iter().collect();
        let bundled = ContinuousHV::bundle(&refs);
        self.estimator.entropy(&bundled)
    }

    /// Compute effect repertoire entropy
    ///
    /// The entropy of possible future states given the current mechanism.
    pub fn effect_repertoire_entropy(&self, future_states: &[ContinuousHV]) -> f64 {
        if future_states.is_empty() {
            return 0.0;
        }

        let refs: Vec<&ContinuousHV> = future_states.iter().collect();
        let bundled = ContinuousHV::bundle(&refs);
        self.estimator.entropy(&bundled)
    }

    /// Compute integrated cause information for a system
    ///
    /// φ_cause = min over partitions of I(M_A; past_A) + I(M_B; past_B)
    /// where M is the mechanism and A,B partition the system
    pub fn integrated_cause_info(&self, components: &[TemporalTransition]) -> f64 {
        let n = components.len();
        if n < 2 {
            return 0.0;
        }

        // Compute whole system cause info
        let current_bundle =
            self.bundle_states(&components.iter().map(|t| &t.current).collect::<Vec<_>>());
        let past_bundle =
            self.bundle_states(&components.iter().map(|t| &t.next).collect::<Vec<_>>());
        let system_cause = self
            .estimator
            .mutual_information_fast(&current_bundle, &past_bundle);

        // Find MIP for cause
        let mut min_partition_cause = f64::INFINITY;

        // Try all non-trivial bipartitions
        for mask in 1..(1 << n) - 1 {
            let partition = TruePartition::from_mask(mask, n);
            if partition.part_a.is_empty() || partition.part_b.is_empty() {
                continue;
            }

            // Compute cause info for each partition
            let current_a = self.bundle_indices(
                &components.iter().map(|t| &t.current).collect::<Vec<_>>(),
                &partition.part_a,
            );
            let past_a = self.bundle_indices(
                &components.iter().map(|t| &t.next).collect::<Vec<_>>(),
                &partition.part_a,
            );
            let cause_a = self.estimator.mutual_information_fast(&current_a, &past_a);

            let current_b = self.bundle_indices(
                &components.iter().map(|t| &t.current).collect::<Vec<_>>(),
                &partition.part_b,
            );
            let past_b = self.bundle_indices(
                &components.iter().map(|t| &t.next).collect::<Vec<_>>(),
                &partition.part_b,
            );
            let cause_b = self.estimator.mutual_information_fast(&current_b, &past_b);

            let partition_cause = cause_a + cause_b;
            min_partition_cause = min_partition_cause.min(partition_cause);
        }

        // φ_cause = system cause - MIP cause
        (system_cause - min_partition_cause).max(0.0)
    }

    /// Compute integrated effect information for a system
    ///
    /// φ_effect = min over partitions of I(M_A; future_A) + I(M_B; future_B)
    pub fn integrated_effect_info(&self, components: &[TemporalTransition]) -> f64 {
        let n = components.len();
        if n < 2 {
            return 0.0;
        }

        // Compute whole system effect info
        let current_bundle =
            self.bundle_states(&components.iter().map(|t| &t.current).collect::<Vec<_>>());
        let future_bundle =
            self.bundle_states(&components.iter().map(|t| &t.next).collect::<Vec<_>>());
        let system_effect = self
            .estimator
            .mutual_information_fast(&current_bundle, &future_bundle);

        // Find MIP for effect
        let mut min_partition_effect = f64::INFINITY;

        for mask in 1..(1 << n) - 1 {
            let partition = TruePartition::from_mask(mask, n);
            if partition.part_a.is_empty() || partition.part_b.is_empty() {
                continue;
            }

            let current_a = self.bundle_indices(
                &components.iter().map(|t| &t.current).collect::<Vec<_>>(),
                &partition.part_a,
            );
            let future_a = self.bundle_indices(
                &components.iter().map(|t| &t.next).collect::<Vec<_>>(),
                &partition.part_a,
            );
            let effect_a = self
                .estimator
                .mutual_information_fast(&current_a, &future_a);

            let current_b = self.bundle_indices(
                &components.iter().map(|t| &t.current).collect::<Vec<_>>(),
                &partition.part_b,
            );
            let future_b = self.bundle_indices(
                &components.iter().map(|t| &t.next).collect::<Vec<_>>(),
                &partition.part_b,
            );
            let effect_b = self
                .estimator
                .mutual_information_fast(&current_b, &future_b);

            let partition_effect = effect_a + effect_b;
            min_partition_effect = min_partition_effect.min(partition_effect);
        }

        (system_effect - min_partition_effect).max(0.0)
    }

    /// Compute full cause-effect information for a transition
    ///
    /// Returns comprehensive cause-effect analysis including:
    /// - Cause and effect information
    /// - Integrated cause and effect
    /// - φ_cause_effect (minimum of integrated cause and effect)
    pub fn compute_cause_effect(&self, transition: &TemporalTransition) -> CauseEffectInfo {
        let cause_info = self.cause_information(transition);
        let effect_info = self.effect_information(transition);
        let cause_entropy = self.estimator.entropy(&transition.current);
        let effect_entropy = self.estimator.entropy(&transition.next);

        // For single transition, integrated info is just the MI
        let integrated_cause = cause_info;
        let integrated_effect = effect_info;

        // φ_cause_effect is the minimum (IIT 3.0 definition)
        let phi_cause_effect = cause_info.min(effect_info);

        CauseEffectInfo {
            cause_info,
            effect_info,
            integrated_cause,
            integrated_effect,
            phi_cause_effect,
            cause_entropy,
            effect_entropy,
        }
    }

    /// Compute cause-effect for a system of components
    pub fn compute_system_cause_effect(
        &self,
        components: &[TemporalTransition],
    ) -> CauseEffectInfo {
        if components.is_empty() {
            return CauseEffectInfo {
                cause_info: 0.0,
                effect_info: 0.0,
                integrated_cause: 0.0,
                integrated_effect: 0.0,
                phi_cause_effect: 0.0,
                cause_entropy: 0.0,
                effect_entropy: 0.0,
            };
        }

        // Bundle all states
        let current_bundle =
            self.bundle_states(&components.iter().map(|t| &t.current).collect::<Vec<_>>());
        let next_bundle =
            self.bundle_states(&components.iter().map(|t| &t.next).collect::<Vec<_>>());

        let cause_info = self
            .estimator
            .mutual_information_fast(&next_bundle, &current_bundle);
        let effect_info = self
            .estimator
            .mutual_information_fast(&current_bundle, &next_bundle);
        let cause_entropy = self.estimator.entropy(&current_bundle);
        let effect_entropy = self.estimator.entropy(&next_bundle);

        let integrated_cause = self.integrated_cause_info(components);
        let integrated_effect = self.integrated_effect_info(components);
        let phi_cause_effect = integrated_cause.min(integrated_effect);

        CauseEffectInfo {
            cause_info,
            effect_info,
            integrated_cause,
            integrated_effect,
            phi_cause_effect,
            cause_entropy,
            effect_entropy,
        }
    }

    /// Helper: Bundle multiple states into one
    fn bundle_states(&self, states: &[&ContinuousHV]) -> ContinuousHV {
        if states.is_empty() {
            return ContinuousHV::zero(16384);
        }
        ContinuousHV::bundle(states)
    }

    /// Helper: Bundle states at specific indices
    fn bundle_indices(&self, states: &[&ContinuousHV], indices: &[usize]) -> ContinuousHV {
        let selected: Vec<&ContinuousHV> = indices
            .iter()
            .filter_map(|&i| states.get(i).copied())
            .collect();
        self.bundle_states(&selected)
    }
}
