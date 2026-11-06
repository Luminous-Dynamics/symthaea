// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paired common-random-number contrast diagnostics.
//!
//! This module checks exact replication-index / coupling-identity pairing and
//! computes empirical pairwise contrast variance. It does NOT create a
//! confidence sequence, prove a shared additive shock model, authenticate a
//! seed receipt, or change any selection/stopping decision. A caller-supplied
//! qualification flag and digest are scope declarations, not proof by
//! themselves.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const PAIRED_CONTRAST_DIAGNOSTIC_METHOD_ID: &str =
    "spectra-paired-crn-contrast-variance-diagnostic-v1";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairedUtilityObservation {
    pub action_id: u32,
    pub scenario_id: u16,
    pub replication_index: u32,
    /// Identity of the shared random stream / coupled exogenous draw. This is
    /// not treated as cryptographic evidence.
    pub coupling_id: u64,
    pub target_utility: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedContrastContract {
    pub policy_version: String,
    pub coupling_contract_digest: String,
    /// Caller declares that the data generator supports the shared additive
    /// shock / matched-coupling interpretation for this exact scope.
    pub shared_coupling_qualified: bool,
    pub action_ids: Vec<u32>,
    pub scenario_ids: Vec<u16>,
    pub utility_lower: f64,
    pub utility_upper: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairwiseContrastDiagnostic {
    pub scenario_id: u16,
    pub left_action_id: u32,
    pub right_action_id: u32,
    pub sample_count: usize,
    pub mean_paired_difference: f64,
    pub left_sample_variance: f64,
    pub right_sample_variance: f64,
    pub paired_difference_sample_variance: f64,
    /// Variance sum implied by an independence approximation.
    pub independent_sum_sample_variance: f64,
    pub paired_to_independent_variance_ratio: Option<f64>,
    pub empirical_pair_correlation: Option<f64>,
    /// Descriptive sqrt(sample variance / n); not a confidence bound.
    pub paired_standard_error_diagnostic: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedContrastDiagnosticReceipt {
    pub method_id: &'static str,
    pub policy_version: String,
    pub coupling_contract_digest: String,
    pub action_ids: Vec<u32>,
    pub scenario_ids: Vec<u16>,
    pub utility_lower: f64,
    pub utility_upper: f64,
    pub diagnostics: Vec<PairwiseContrastDiagnostic>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairedContrastError {
    MissingPolicyVersion,
    MissingCouplingContractDigest,
    UnqualifiedSharedCoupling,
    EmptyActionSet,
    EmptyScenarioSet,
    DuplicateActionId,
    DuplicateScenarioId,
    InvalidUtilityRange,
    EmptyObservations,
    UnknownActionId,
    UnknownScenarioId,
    InvalidObservation,
    DuplicateObservation,
    IncompletePairing,
    CouplingIdentityMismatch,
    NonContiguousReplicationIndices,
    InsufficientReplications,
    PairCountOverflow,
    NumericFailure,
    ReplayMismatch,
}

impl fmt::Display for PairedContrastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::MissingPolicyVersion => "paired contrast policy version must not be empty",
            Self::MissingCouplingContractDigest => "coupling contract digest must not be empty",
            Self::UnqualifiedSharedCoupling => "shared coupling / additive-shock interpretation is not qualified",
            Self::EmptyActionSet => "paired contrasts require at least two frozen actions",
            Self::EmptyScenarioSet => "paired contrasts require a nonempty frozen scenario set",
            Self::DuplicateActionId => "frozen action IDs must be unique",
            Self::DuplicateScenarioId => "frozen scenario IDs must be unique",
            Self::InvalidUtilityRange => "utility range must be finite and ordered",
            Self::EmptyObservations => "paired contrast diagnostics require raw observations",
            Self::UnknownActionId => "observation references an action outside the frozen action set",
            Self::UnknownScenarioId => "observation references a scenario outside the frozen scenario set",
            Self::InvalidObservation => "utility observation is nonfinite or outside its declared range",
            Self::DuplicateObservation => "duplicate action × scenario × replication observation",
            Self::IncompletePairing => "every scenario × replication index must contain exactly one observation for every action",
            Self::CouplingIdentityMismatch => "actions at a matched scenario × replication index have different coupling identities",
            Self::NonContiguousReplicationIndices => "replication indices must be contiguous from zero within each scenario",
            Self::InsufficientReplications => "each scenario requires at least two complete paired replications",
            Self::PairCountOverflow => "action-pair diagnostic count overflowed usize",
            Self::NumericFailure => "paired contrast moment computation was nonfinite or numerically invalid",
            Self::ReplayMismatch => "paired contrast diagnostic receipt does not reproduce",
        };
        f.write_str(message)
    }
}
impl std::error::Error for PairedContrastError {}

/// Validate complete pairing and compute descriptive empirical contrast
/// moments for every action pair within every scenario.
///
/// For pair (a,b), the series is d_r = Y[a,s,r] - Y[b,s,r]. The independent
/// approximation is Var(Y[a]) + Var(Y[b]); the paired variance is Var(d_r).
/// A ratio below one indicates an empirical variance reduction under pairing.
/// It is only diagnostic and does not justify an inferential interval.
pub fn diagnose_paired_contrasts(
    contract: &PairedContrastContract,
    observations: &[PairedUtilityObservation],
) -> Result<PairedContrastDiagnosticReceipt, PairedContrastError> {
    validate_contract(contract)?;
    if observations.is_empty() {
        return Err(PairedContrastError::EmptyObservations);
    }

    let mut actions = contract.action_ids.clone();
    actions.sort_unstable();
    let mut scenarios = contract.scenario_ids.clone();
    scenarios.sort_unstable();

    let action_set: BTreeSet<u32> = actions.iter().copied().collect();
    let scenario_set: BTreeSet<u16> = scenarios.iter().copied().collect();
    let mut grouped = BTreeMap::<(u16, u32), BTreeMap<u32, PairedUtilityObservation>>::new();

    for observation in observations {
        if !action_set.contains(&observation.action_id) {
            return Err(PairedContrastError::UnknownActionId);
        }
        if !scenario_set.contains(&observation.scenario_id) {
            return Err(PairedContrastError::UnknownScenarioId);
        }
        if !observation.target_utility.is_finite()
            || observation.target_utility < contract.utility_lower
            || observation.target_utility > contract.utility_upper
        {
            return Err(PairedContrastError::InvalidObservation);
        }
        let replication_group = grouped
            .entry((observation.scenario_id, observation.replication_index))
            .or_default();
        if replication_group
            .insert(observation.action_id, *observation)
            .is_some()
        {
            return Err(PairedContrastError::DuplicateObservation);
        }
    }

    let mut scenario_indices = BTreeMap::<u16, Vec<u32>>::new();
    for ((scenario_id, replication_index), group) in &grouped {
        if group.len() != actions.len() {
            return Err(PairedContrastError::IncompletePairing);
        }
        if actions.iter().any(|action_id| !group.contains_key(action_id)) {
            return Err(PairedContrastError::IncompletePairing);
        }
        let mut coupling_ids = group.values().map(|row| row.coupling_id);
        let first_coupling_id = coupling_ids
            .next()
            .ok_or(PairedContrastError::IncompletePairing)?;
        if coupling_ids.any(|coupling_id| coupling_id != first_coupling_id) {
            return Err(PairedContrastError::CouplingIdentityMismatch);
        }
        scenario_indices
            .entry(*scenario_id)
            .or_default()
            .push(*replication_index);
    }

    let mut diagnostics = Vec::new();
    let pair_count_per_scenario = actions
        .len()
        .checked_mul(actions.len().saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or(PairedContrastError::PairCountOverflow)?;
    let expected_diagnostic_count = pair_count_per_scenario
        .checked_mul(scenarios.len())
        .ok_or(PairedContrastError::PairCountOverflow)?;
    diagnostics.reserve(expected_diagnostic_count);

    for scenario_id in &scenarios {
        let mut indices = scenario_indices
            .remove(scenario_id)
            .ok_or(PairedContrastError::IncompletePairing)?;
        indices.sort_unstable();
        if indices.len() < 2 {
            return Err(PairedContrastError::InsufficientReplications);
        }
        if indices
            .iter()
            .enumerate()
            .any(|(expected, observed)| *observed as usize != expected)
        {
            return Err(PairedContrastError::NonContiguousReplicationIndices);
        }

        for left_index in 0..actions.len() {
            for right_index in (left_index + 1)..actions.len() {
                let left_action_id = actions[left_index];
                let right_action_id = actions[right_index];
                let mut moments = BivariateMoments::default();
                let mut contrast_moments = UnivariateMoments::default();

                for replication_index in &indices {
                    let group = grouped
                        .get(&(*scenario_id, *replication_index))
                        .ok_or(PairedContrastError::IncompletePairing)?;
                    let left = group
                        .get(&left_action_id)
                        .ok_or(PairedContrastError::IncompletePairing)?
                        .target_utility;
                    let right = group
                        .get(&right_action_id)
                        .ok_or(PairedContrastError::IncompletePairing)?
                        .target_utility;
                    let difference = left - right;
                    if !difference.is_finite() {
                        return Err(PairedContrastError::NumericFailure);
                    }
                    moments.push(left, right)?;
                    contrast_moments.push(difference)?;
                }

                let (mean_left, mean_right, variance_left, variance_right, covariance) =
                    moments.finish()?;
                let (mean_difference, paired_variance) = contrast_moments.finish()?;
                let independent_sum_variance = variance_left + variance_right;
                let variance_ratio = if independent_sum_variance > 0.0 {
                    Some(paired_variance / independent_sum_variance)
                } else {
                    None
                };
                let correlation = if variance_left > 0.0 && variance_right > 0.0 {
                    Some((covariance / (variance_left.sqrt() * variance_right.sqrt())).clamp(-1.0, 1.0))
                } else {
                    None
                };
                let paired_standard_error_diagnostic =
                    (paired_variance / indices.len() as f64).sqrt();

                if !mean_left.is_finite()
                    || !mean_right.is_finite()
                    || !independent_sum_variance.is_finite()
                    || !paired_standard_error_diagnostic.is_finite()
                    || variance_ratio.is_some_and(|ratio| !ratio.is_finite())
                {
                    return Err(PairedContrastError::NumericFailure);
                }

                diagnostics.push(PairwiseContrastDiagnostic {
                    scenario_id: *scenario_id,
                    left_action_id,
                    right_action_id,
                    sample_count: indices.len(),
                    mean_paired_difference: mean_difference,
                    left_sample_variance: variance_left,
                    right_sample_variance: variance_right,
                    paired_difference_sample_variance: paired_variance,
                    independent_sum_sample_variance: independent_sum_variance,
                    paired_to_independent_variance_ratio: variance_ratio,
                    empirical_pair_correlation: correlation,
                    paired_standard_error_diagnostic,
                });
            }
        }
    }

    Ok(PairedContrastDiagnosticReceipt {
        method_id: PAIRED_CONTRAST_DIAGNOSTIC_METHOD_ID,
        policy_version: contract.policy_version.clone(),
        coupling_contract_digest: contract.coupling_contract_digest.clone(),
        action_ids: actions,
        scenario_ids: scenarios,
        utility_lower: contract.utility_lower,
        utility_upper: contract.utility_upper,
        diagnostics,
    })
}

/// Deterministically recompute every pairing and moment field.
pub fn verify_paired_contrast_replay(
    contract: &PairedContrastContract,
    observations: &[PairedUtilityObservation],
    recorded: &PairedContrastDiagnosticReceipt,
) -> Result<(), PairedContrastError> {
    let expected = diagnose_paired_contrasts(contract, observations)?;
    if &expected == recorded {
        Ok(())
    } else {
        Err(PairedContrastError::ReplayMismatch)
    }
}

fn validate_contract(contract: &PairedContrastContract) -> Result<(), PairedContrastError> {
    if contract.policy_version.trim().is_empty() {
        return Err(PairedContrastError::MissingPolicyVersion);
    }
    if contract.coupling_contract_digest.trim().is_empty() {
        return Err(PairedContrastError::MissingCouplingContractDigest);
    }
    if !contract.shared_coupling_qualified {
        return Err(PairedContrastError::UnqualifiedSharedCoupling);
    }
    if contract.action_ids.len() < 2 {
        return Err(PairedContrastError::EmptyActionSet);
    }
    if contract.scenario_ids.is_empty() {
        return Err(PairedContrastError::EmptyScenarioSet);
    }
    let mut actions = contract.action_ids.clone();
    actions.sort_unstable();
    if actions.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(PairedContrastError::DuplicateActionId);
    }
    let mut scenarios = contract.scenario_ids.clone();
    scenarios.sort_unstable();
    if scenarios.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(PairedContrastError::DuplicateScenarioId);
    }
    if !contract.utility_lower.is_finite()
        || !contract.utility_upper.is_finite()
        || contract.utility_lower > contract.utility_upper
    {
        return Err(PairedContrastError::InvalidUtilityRange);
    }
    Ok(())
}

#[derive(Debug, Default)]
struct UnivariateMoments {
    count: usize,
    mean: f64,
    sum_squared_deviations: f64,
}

impl UnivariateMoments {
    fn push(&mut self, value: f64) -> Result<(), PairedContrastError> {
        self.count = self.count.checked_add(1).ok_or(PairedContrastError::NumericFailure)?;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta_after = value - self.mean;
        self.sum_squared_deviations += delta * delta_after;
        if !self.mean.is_finite() || !self.sum_squared_deviations.is_finite() {
            return Err(PairedContrastError::NumericFailure);
        }
        Ok(())
    }

    fn finish(&self) -> Result<(f64, f64), PairedContrastError> {
        if self.count < 2 {
            return Err(PairedContrastError::InsufficientReplications);
        }
        let variance = self.sum_squared_deviations / (self.count - 1) as f64;
        if !variance.is_finite() || variance < 0.0 {
            return Err(PairedContrastError::NumericFailure);
        }
        Ok((self.mean, variance))
    }
}

#[derive(Debug, Default)]
struct BivariateMoments {
    count: usize,
    mean_left: f64,
    mean_right: f64,
    sum_squared_left: f64,
    sum_squared_right: f64,
    co_moment: f64,
}

impl BivariateMoments {
    fn push(&mut self, left: f64, right: f64) -> Result<(), PairedContrastError> {
        self.count = self.count.checked_add(1).ok_or(PairedContrastError::NumericFailure)?;
        let delta_left = left - self.mean_left;
        let delta_right = right - self.mean_right;
        self.mean_left += delta_left / self.count as f64;
        self.mean_right += delta_right / self.count as f64;
        self.sum_squared_left += delta_left * (left - self.mean_left);
        self.sum_squared_right += delta_right * (right - self.mean_right);
        self.co_moment += delta_left * (right - self.mean_right);
        if !self.mean_left.is_finite()
            || !self.mean_right.is_finite()
            || !self.sum_squared_left.is_finite()
            || !self.sum_squared_right.is_finite()
            || !self.co_moment.is_finite()
        {
            return Err(PairedContrastError::NumericFailure);
        }
        Ok(())
    }

    fn finish(&self) -> Result<(f64, f64, f64, f64, f64), PairedContrastError> {
        if self.count < 2 {
            return Err(PairedContrastError::InsufficientReplications);
        }
        let denominator = (self.count - 1) as f64;
        let variance_left = self.sum_squared_left / denominator;
        let variance_right = self.sum_squared_right / denominator;
        let covariance = self.co_moment / denominator;
        if !variance_left.is_finite()
            || !variance_right.is_finite()
            || !covariance.is_finite()
            || variance_left < 0.0
            || variance_right < 0.0
        {
            return Err(PairedContrastError::NumericFailure);
        }
        Ok((self.mean_left, self.mean_right, variance_left, variance_right, covariance))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> PairedContrastContract {
        PairedContrastContract {
            policy_version: "paired-contrast-test-v1".to_owned(),
            coupling_contract_digest: "test-coupling-contract-sha256".to_owned(),
            shared_coupling_qualified: true,
            action_ids: vec![1, 2, 3],
            scenario_ids: vec![7],
            utility_lower: -10.0,
            utility_upper: 10.0,
        }
    }

    fn observations() -> Vec<PairedUtilityObservation> {
        let common = [0.0, 1.0, -0.5, 2.0, -1.0, 0.25];
        let offsets = [(1_u32, 0.6), (2, 0.2), (3, -0.1)];
        let mut rows = Vec::new();
        for (replication_index, shock) in common.iter().enumerate() {
            for (action_id, offset) in offsets {
                rows.push(PairedUtilityObservation {
                    action_id,
                    scenario_id: 7,
                    replication_index: replication_index as u32,
                    coupling_id: 10_000 + replication_index as u64,
                    target_utility: offset + shock,
                });
            }
        }
        rows
    }

    #[test]
    fn common_additive_shock_cancels_and_pairing_variance_is_zero() {
        let result = diagnose_paired_contrasts(&contract(), &observations()).unwrap();
        assert_eq!(result.diagnostics.len(), 3);
        for pair in &result.diagnostics {
            assert!(pair.paired_difference_sample_variance.abs() < 1.0e-12);
            assert!(pair.independent_sum_sample_variance > 0.0);
            assert_eq!(pair.paired_to_independent_variance_ratio, Some(0.0));
            assert!(pair.empirical_pair_correlation.unwrap() > 0.999_999);
        }
        assert!((result.diagnostics[0].mean_paired_difference - 0.4).abs() < 1.0e-12);
    }

    #[test]
    fn arbitrary_per_replication_offsets_leave_every_contrast_unchanged() {
        let original = observations();
        let mut shifted = original.clone();
        let shifts = [11.0, -8.0, 0.125, 23.0, -4.0, 3.5];
        for row in &mut shifted {
            row.target_utility += shifts[row.replication_index as usize];
        }
        let first = diagnose_paired_contrasts(&contract(), &original).unwrap();
        let second = diagnose_paired_contrasts(&contract(), &shifted).unwrap();
        for (left, right) in first.diagnostics.iter().zip(second.diagnostics.iter()) {
            assert!((left.mean_paired_difference - right.mean_paired_difference).abs() < 1.0e-12);
            assert!((left.paired_difference_sample_variance - right.paired_difference_sample_variance).abs() < 1.0e-12);
        }
    }

    #[test]
    fn input_order_does_not_change_diagnostic_receipt() {
        let rows = observations();
        let first = diagnose_paired_contrasts(&contract(), &rows).unwrap();
        let mut reversed = rows;
        reversed.reverse();
        let second = diagnose_paired_contrasts(&contract(), &reversed).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn pairing_requires_every_action_and_the_same_coupling_identity() {
        let mut missing = observations();
        missing.pop();
        assert_eq!(
            diagnose_paired_contrasts(&contract(), &missing),
            Err(PairedContrastError::IncompletePairing)
        );

        let mut mismatch = observations();
        mismatch[0].coupling_id += 1;
        assert_eq!(
            diagnose_paired_contrasts(&contract(), &mismatch),
            Err(PairedContrastError::CouplingIdentityMismatch)
        );

        let mut gap = observations();
        for row in &mut gap {
            if row.replication_index == 2 {
                row.replication_index = 3;
            } else if row.replication_index == 3 {
                row.replication_index = 4;
            } else if row.replication_index == 4 {
                row.replication_index = 5;
            } else if row.replication_index == 5 {
                row.replication_index = 6;
            }
        }
        assert_eq!(
            diagnose_paired_contrasts(&contract(), &gap),
            Err(PairedContrastError::NonContiguousReplicationIndices)
        );
    }

    #[test]
    fn invalid_scope_range_and_observations_fail_closed() {
        let rows = observations();
        let mut unqualified = contract();
        unqualified.shared_coupling_qualified = false;
        assert_eq!(
            diagnose_paired_contrasts(&unqualified, &rows),
            Err(PairedContrastError::UnqualifiedSharedCoupling)
        );

        let mut missing_digest = contract();
        missing_digest.coupling_contract_digest.clear();
        assert_eq!(
            diagnose_paired_contrasts(&missing_digest, &rows),
            Err(PairedContrastError::MissingCouplingContractDigest)
        );

        let mut outside = observations();
        outside[0].target_utility = 100.0;
        assert_eq!(
            diagnose_paired_contrasts(&contract(), &outside),
            Err(PairedContrastError::InvalidObservation)
        );
    }

    #[test]
    fn replay_accepts_exact_receipt_and_rejects_mutated_diagnostics() {
        let rows = observations();
        let c = contract();
        let receipt = diagnose_paired_contrasts(&c, &rows).unwrap();
        assert_eq!(verify_paired_contrast_replay(&c, &rows, &receipt), Ok(()));
        let mut changed = receipt.clone();
        changed.diagnostics[0].paired_difference_sample_variance = 1.0;
        assert_eq!(
            verify_paired_contrast_replay(&c, &rows, &changed),
            Err(PairedContrastError::ReplayMismatch)
        );
    }
}
