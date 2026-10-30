// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Grouped split-conformal calibration for retrieval value bounds.
//!
//! The VOI gate requires conservative lower bounds for correctness and
//! lifecycle-risk improvement, and a conservative upper bound for retrieval
//! harm. This module derives those bounds from paired retrieval/no-retrieval
//! observations on a held-out calibration split.
//!
//! Independent units are calibration groups (for example source documents,
//! users or conversations), not individual questions. Within each group we
//! take the worst residual for each component: minimum residual for benefits,
//! maximum residual for harm. A one-sided split-conformal order statistic is
//! then applied across groups. Bonferroni allocation gives at least 90%
//! familywise marginal coverage across the three one-sided bounds, under the
//! exchangeability assumption for held-out groups.
//!
//! This is intentionally conservative. Fewer than 29 independent groups are
//! rejected rather than silently emitting an invalid finite 90% joint bound.
//! The guarantee is marginal over exchangeable groups, not a per-query,
//! per-domain or distribution-free conditional guarantee. Dataset shift,
//! group leakage, treatment non-compliance and judge error invalidate the
//! interpretation and must be audited separately.

use crate::memory_retrieval_value::{
    RETRIEVAL_UTILITY_CALIBRATION_VERSION, RetrievalUtilityEstimate,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const MIN_INDEPENDENT_GROUPS: usize = 29;
const COMPONENT_ALPHA_NUMERATOR: usize = 1;
const COMPONENT_ALPHA_DENOMINATOR: usize = 30;
const JOINT_COVERAGE_PERCENT: u8 = 90;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalUtilityCalibrationObservation {
    pub observation_id: String,
    /// Independent unit used for exchangeability and leakage-safe splitting.
    pub calibration_group_id: String,
    pub query_fingerprint_sha256: String,
    pub predicted_correctness_gain_micro: i64,
    pub observed_correctness_gain_micro: i64,
    pub predicted_lifecycle_risk_reduction_micro: i64,
    pub observed_lifecycle_risk_reduction_micro: i64,
    pub predicted_retrieval_harm_micro: i64,
    pub observed_retrieval_harm_micro: i64,
}

impl RetrievalUtilityCalibrationObservation {
    fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("observation_id", self.observation_id.as_str()),
            ("calibration_group_id", self.calibration_group_id.as_str()),
        ] {
            require_non_empty(value, field)?;
        }
        validate_sha256(&self.query_fingerprint_sha256, "query_fingerprint_sha256")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibratedRetrievalUtilityBounds {
    pub calibration_method_version: String,
    pub calibration_method_sha256: String,
    pub calibration_split_sha256: String,
    pub calibration_joint_coverage_percent: u8,
    pub independent_group_count: usize,
    pub observation_count: usize,
    /// One-sided lower residual offset; added to the runtime point estimate.
    pub correctness_gain_lcb_residual_micro: i64,
    /// One-sided lower residual offset; added to the runtime point estimate.
    pub lifecycle_risk_reduction_lcb_residual_micro: i64,
    /// One-sided upper residual offset; added to the runtime point estimate.
    pub retrieval_harm_ucb_residual_micro: i64,
}

impl CalibratedRetrievalUtilityBounds {
    pub fn apply_to_estimate(
        &self,
        estimate: &mut RetrievalUtilityEstimate,
    ) -> Result<(), String> {
        self.verify_internal_consistency()?;
        estimate.calibration_method_sha256 = self.calibration_method_sha256.clone();
        estimate.calibration_joint_coverage_percent = self.calibration_joint_coverage_percent;
        estimate.calibration_split_sha256 = self.calibration_split_sha256.clone();
        estimate.correctness_gain_lcb_micro = i64_from_i128(
            i128::from(estimate.correctness_gain_prediction_micro)
                + i128::from(self.correctness_gain_lcb_residual_micro),
        )?;
        estimate.lifecycle_risk_reduction_lcb_micro = i64_from_i128(
            i128::from(estimate.lifecycle_risk_reduction_prediction_micro)
                + i128::from(self.lifecycle_risk_reduction_lcb_residual_micro),
        )?;
        estimate.retrieval_harm_ucb_micro = i64_from_i128(
            i128::from(estimate.retrieval_harm_prediction_micro)
                + i128::from(self.retrieval_harm_ucb_residual_micro),
        )?;
        estimate.validate()
    }

    pub fn verify_internal_consistency(&self) -> Result<(), String> {
        if self.calibration_method_version != RETRIEVAL_UTILITY_CALIBRATION_VERSION {
            return Err("unsupported retrieval calibration method version".into());
        }
        validate_sha256(&self.calibration_method_sha256, "calibration_method_sha256")?;
        validate_sha256(&self.calibration_split_sha256, "calibration_split_sha256")?;
        if self.calibration_joint_coverage_percent != JOINT_COVERAGE_PERCENT {
            return Err("calibration result does not carry the pinned 90% joint coverage target".into());
        }
        if self.independent_group_count < MIN_INDEPENDENT_GROUPS {
            return Err("calibration result has too few independent groups".into());
        }
        if self.observation_count < self.independent_group_count {
            return Err("observation count cannot be below independent group count".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GroupResiduals {
    correctness_gain_min: i128,
    lifecycle_reduction_min: i128,
    retrieval_harm_max: i128,
}

const METHOD_DESCRIPTION: &str =
    "BROCA-GROUP-CONFORMAL-VOI-V1; split-conformal; group worst-case residuals; three one-sided bounds; Bonferroni alpha=1/30 each; 90% familywise marginal coverage; min 29 independent groups";

/// Calibrate conservative utility bounds from held-out paired observations.
///
/// Each observed delta must be measured under the same evaluator and utility
/// schema as the runtime estimate. Correctness and lifecycle-risk components
/// must be disjoint to prevent double counting. The caller supplies the
/// calibration rows produced by frozen-model paired execution; this function
/// does not create counterfactual outcomes or train the predictor.
pub fn calibrate_retrieval_utility_bounds(
    observations: &[RetrievalUtilityCalibrationObservation],
) -> Result<CalibratedRetrievalUtilityBounds, String> {
    if observations.is_empty() {
        return Err("calibration observations must not be empty".into());
    }

    let mut observation_ids = BTreeSet::new();
    let mut query_fingerprints = BTreeSet::new();
    let mut grouped = BTreeMap::<String, GroupResiduals>::new();

    for observation in observations {
        observation.validate()?;
        if !observation_ids.insert(observation.observation_id.as_str()) {
            return Err(format!(
                "duplicate calibration observation ID {:?}",
                observation.observation_id
            ));
        }
        if !query_fingerprints.insert(observation.query_fingerprint_sha256.as_str()) {
            return Err(format!(
                "duplicate query fingerprint {:?}; one paired result per query is required",
                observation.query_fingerprint_sha256
            ));
        }

        let correctness_residual = i128::from(observation.observed_correctness_gain_micro)
            - i128::from(observation.predicted_correctness_gain_micro);
        let lifecycle_residual =
            i128::from(observation.observed_lifecycle_risk_reduction_micro)
                - i128::from(observation.predicted_lifecycle_risk_reduction_micro);
        let harm_residual = i128::from(observation.observed_retrieval_harm_micro)
            - i128::from(observation.predicted_retrieval_harm_micro);

        grouped
            .entry(observation.calibration_group_id.clone())
            .and_modify(|residuals| {
                residuals.correctness_gain_min =
                    residuals.correctness_gain_min.min(correctness_residual);
                residuals.lifecycle_reduction_min =
                    residuals.lifecycle_reduction_min.min(lifecycle_residual);
                residuals.retrieval_harm_max =
                    residuals.retrieval_harm_max.max(harm_residual);
            })
            .or_insert(GroupResiduals {
                correctness_gain_min: correctness_residual,
                lifecycle_reduction_min: lifecycle_residual,
                retrieval_harm_max: harm_residual,
            });
    }

    if grouped.len() < MIN_INDEPENDENT_GROUPS {
        return Err(format!(
            "at least {MIN_INDEPENDENT_GROUPS} independent calibration groups are required; received {}",
            grouped.len()
        ));
    }

    let groups: Vec<GroupResiduals> = grouped.values().copied().collect();
    let correctness_residuals: Vec<i128> =
        groups.iter().map(|group| group.correctness_gain_min).collect();
    let lifecycle_residuals: Vec<i128> =
        groups.iter().map(|group| group.lifecycle_reduction_min).collect();
    let harm_residuals: Vec<i128> =
        groups.iter().map(|group| group.retrieval_harm_max).collect();

    let correctness_lcb_offset = lower_conformal_quantile(&correctness_residuals)?;
    let lifecycle_lcb_offset = lower_conformal_quantile(&lifecycle_residuals)?;
    let harm_ucb_offset = upper_conformal_quantile(&harm_residuals)?;

    let mut canonical_observations = observations.to_vec();
    canonical_observations.sort_by(|left, right| {
        left.observation_id.cmp(&right.observation_id)
    });
    let split_bytes = serde_json::to_vec(&canonical_observations)
        .map_err(|error| format!("cannot serialize calibration split: {error}"))?;
    let calibration_split_sha256 = sha256_hex(&split_bytes);
    let calibration_method_sha256 = sha256_hex(METHOD_DESCRIPTION.as_bytes());

    let result = CalibratedRetrievalUtilityBounds {
        calibration_method_version: RETRIEVAL_UTILITY_CALIBRATION_VERSION.into(),
        calibration_method_sha256,
        calibration_split_sha256,
        calibration_joint_coverage_percent: JOINT_COVERAGE_PERCENT,
        independent_group_count: grouped.len(),
        observation_count: observations.len(),
        correctness_gain_lcb_residual_micro: i64_from_i128(correctness_lcb_offset)?,
        lifecycle_risk_reduction_lcb_residual_micro: i64_from_i128(lifecycle_lcb_offset)?,
        retrieval_harm_ucb_residual_micro: i64_from_i128(harm_ucb_offset)?,
    };
    result.verify_internal_consistency()?;
    Ok(result)
}

/// Add the lower one-sided residual quantile to a point prediction. For alpha
/// 1/30 the one-based finite-sample rank is floor(alpha * (n + 1)).
fn lower_conformal_quantile(residuals: &[i128]) -> Result<i128, String> {
    let rank = (residuals.len() + 1) * COMPONENT_ALPHA_NUMERATOR
        / COMPONENT_ALPHA_DENOMINATOR;
    if rank == 0 || rank > residuals.len() {
        return Err("insufficient calibration groups for finite lower conformal bound".into());
    }
    let mut sorted = residuals.to_vec();
    sorted.sort_unstable();
    Ok(sorted[rank - 1])
}

/// Add the upper one-sided residual quantile to a point prediction. For alpha
/// 1/30 the one-based finite-sample rank is ceil((n + 1) * (1 - alpha)).
fn upper_conformal_quantile(residuals: &[i128]) -> Result<i128, String> {
    let numerator = (residuals.len() + 1)
        * (COMPONENT_ALPHA_DENOMINATOR - COMPONENT_ALPHA_NUMERATOR);
    let rank = numerator.div_ceil(COMPONENT_ALPHA_DENOMINATOR);
    if rank == 0 || rank > residuals.len() {
        return Err("insufficient calibration groups for finite upper conformal bound".into());
    }
    let mut sorted = residuals.to_vec();
    sorted.sort_unstable();
    Ok(sorted[rank - 1])
}

fn i64_from_i128(value: i128) -> Result<i64, String> {
    i64::try_from(value).map_err(|_| "conformal residual exceeds i64 utility range".into())
}

fn require_non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(format!("{field} must be non-empty and trimmed"));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 64
        || !value.bytes().all(|byte| {
            byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
        })
    {
        return Err(format!("{field} must be a 64-character lowercase SHA-256 digest"));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: &str) -> String {
        sha256_hex(value.as_bytes())
    }

    fn observation(
        index: usize,
        group: usize,
        correctness_residual: i64,
        lifecycle_residual: i64,
        harm_residual: i64,
    ) -> RetrievalUtilityCalibrationObservation {
        RetrievalUtilityCalibrationObservation {
            observation_id: format!("observation-{index}"),
            calibration_group_id: format!("document-group-{group}"),
            query_fingerprint_sha256: digest(&format!("query-{index}")),
            predicted_correctness_gain_micro: 100,
            observed_correctness_gain_micro: 100 + correctness_residual,
            predicted_lifecycle_risk_reduction_micro: 200,
            observed_lifecycle_risk_reduction_micro: 200 + lifecycle_residual,
            predicted_retrieval_harm_micro: 50,
            observed_retrieval_harm_micro: 50 + harm_residual,
        }
    }

    fn calibration_rows() -> Vec<RetrievalUtilityCalibrationObservation> {
        (0..MIN_INDEPENDENT_GROUPS)
            .map(|index| {
                let residual = index as i64 - 14;
                observation(index, index, residual, residual - 2, residual + 1)
            })
            .collect()
    }

    #[derive(Deserialize)]
    struct CalibrationFixture {
        schema_version: String,
        calibration_method_version: String,
        joint_coverage_percent: u8,
        independent_group_count: usize,
        observation_count: usize,
        observations: Vec<RetrievalUtilityCalibrationObservation>,
        expected_residual_offsets: ExpectedResidualOffsets,
    }

    #[derive(Deserialize)]
    struct ExpectedResidualOffsets {
        correctness_gain_lcb_residual_micro: i64,
        lifecycle_risk_reduction_lcb_residual_micro: i64,
        retrieval_harm_ucb_residual_micro: i64,
    }

    #[test]
    fn versioned_group_conformal_fixture_matches_calibration_algorithm() {
        let raw = include_str!(
            "../../../../research/broca-bitemporal-evaluation-v1/retrieval-value-calibration-fixtures-v1.json"
        );
        let fixture: CalibrationFixture = serde_json::from_str(raw).unwrap();
        assert_eq!(
            fixture.schema_version,
            "broca-retrieval-value-calibration-fixtures-v1"
        );
        assert_eq!(
            fixture.calibration_method_version,
            RETRIEVAL_UTILITY_CALIBRATION_VERSION
        );
        assert_eq!(fixture.joint_coverage_percent, JOINT_COVERAGE_PERCENT);
        assert_eq!(fixture.independent_group_count, 29);
        assert_eq!(fixture.observation_count, fixture.observations.len());

        let bounds = calibrate_retrieval_utility_bounds(&fixture.observations).unwrap();
        assert_eq!(bounds.independent_group_count, fixture.independent_group_count);
        assert_eq!(bounds.observation_count, fixture.observation_count);
        assert_eq!(
            bounds.correctness_gain_lcb_residual_micro,
            fixture.expected_residual_offsets.correctness_gain_lcb_residual_micro
        );
        assert_eq!(
            bounds.lifecycle_risk_reduction_lcb_residual_micro,
            fixture.expected_residual_offsets.lifecycle_risk_reduction_lcb_residual_micro
        );
        assert_eq!(
            bounds.retrieval_harm_ucb_residual_micro,
            fixture.expected_residual_offsets.retrieval_harm_ucb_residual_micro
        );
    }

    #[test]
    fn uses_group_conformal_lower_bounds_for_benefits_and_upper_bound_for_harm() {
        let rows = calibration_rows();
        let bounds = calibrate_retrieval_utility_bounds(&rows).unwrap();

        // 29 groups => lower rank 1 and upper rank 29 at alpha=1/30.
        assert_eq!(bounds.correctness_gain_lcb_residual_micro, -14);
        assert_eq!(bounds.lifecycle_risk_reduction_lcb_residual_micro, -16);
        assert_eq!(bounds.retrieval_harm_ucb_residual_micro, 15);
        assert_eq!(bounds.independent_group_count, 29);
        assert_eq!(bounds.observation_count, 29);
        assert_eq!(bounds.calibration_joint_coverage_percent, 90);
        bounds.verify_internal_consistency().unwrap();
    }

    #[test]
    fn takes_worst_residual_within_each_independent_group() {
        let mut rows = calibration_rows();
        rows.push(observation(29, 0, -100, -90, 100));
        let bounds = calibrate_retrieval_utility_bounds(&rows).unwrap();
        assert_eq!(bounds.correctness_gain_lcb_residual_micro, -100);
        assert_eq!(bounds.lifecycle_risk_reduction_lcb_residual_micro, -92);
        assert_eq!(bounds.retrieval_harm_ucb_residual_micro, 101);
        assert_eq!(bounds.independent_group_count, 29);
        assert_eq!(bounds.observation_count, 30);
    }

    #[test]
    fn rejects_too_few_independent_groups() {
        let mut rows = calibration_rows();
        rows.pop();
        assert!(calibrate_retrieval_utility_bounds(&rows)
            .unwrap_err()
            .contains("at least 29 independent calibration groups"));
    }

    #[test]
    fn rejects_duplicate_observation_and_query_fingerprints() {
        let mut duplicate_id = calibration_rows();
        duplicate_id[1].observation_id = duplicate_id[0].observation_id.clone();
        assert!(calibrate_retrieval_utility_bounds(&duplicate_id)
            .unwrap_err()
            .contains("duplicate calibration observation"));

        let mut duplicate_query = calibration_rows();
        duplicate_query[1].query_fingerprint_sha256 =
            duplicate_query[0].query_fingerprint_sha256.clone();
        assert!(calibrate_retrieval_utility_bounds(&duplicate_query)
            .unwrap_err()
            .contains("duplicate query fingerprint"));
    }

    #[test]
    fn calibration_digest_is_order_invariant_and_commits_to_rows() {
        let rows = calibration_rows();
        let first = calibrate_retrieval_utility_bounds(&rows).unwrap();
        let mut reversed = rows.clone();
        reversed.reverse();
        let second = calibrate_retrieval_utility_bounds(&reversed).unwrap();
        assert_eq!(first.calibration_split_sha256, second.calibration_split_sha256);
        assert_eq!(first.calibration_method_sha256, second.calibration_method_sha256);

        let mut changed = rows;
        changed[0].observed_correctness_gain_micro += 1;
        let third = calibrate_retrieval_utility_bounds(&changed).unwrap();
        assert_ne!(first.calibration_split_sha256, third.calibration_split_sha256);
    }

    #[test]
    fn applies_signed_bounds_and_split_receipt_to_runtime_estimate() {
        let rows = calibration_rows();
        let bounds = calibrate_retrieval_utility_bounds(&rows).unwrap();
        let mut estimate = RetrievalUtilityEstimate {
            estimate_id: "estimate-1".into(),
            decision_id: "decision-1".into(),
            principal_id: "principal-1".into(),
            scope_id: "scope-1".into(),
            valid_at_unix_ms: 1,
            known_at_unix_ms: 2,
            evidence_index_state_sha256: digest("index"),
            unresolved_claims_sha256: digest("claims"),
            unresolved_required_claim_ids: BTreeSet::from(["C1".into()]),
            calibration_method_sha256: digest("placeholder method"),
            calibration_joint_coverage_percent: 90,
            estimator_id: "estimator".into(),
            estimator_version: "1".into(),
            estimator_sha256: digest("estimator"),
            calibration_split_sha256: digest("placeholder split"),
            feature_schema_sha256: digest("features"),
            utility_schema_sha256: digest("utility"),
            correctness_gain_prediction_micro: 100,
            correctness_gain_lcb_micro: 5,
            lifecycle_risk_reduction_prediction_micro: 200,
            lifecycle_risk_reduction_lcb_micro: 5,
            retrieval_harm_prediction_micro: 50,
            retrieval_harm_ucb_micro: 5,
            retrieval_cost_micro: 1,
            remaining_calls: 1,
            remaining_budget_micro: 10,
            minimum_net_value_micro: 1,
        };
        estimate.unresolved_claims_sha256 =
            RetrievalUtilityEstimate::unresolved_claims_digest(
                &estimate.unresolved_required_claim_ids
            ).unwrap();
        bounds.apply_to_estimate(&mut estimate).unwrap();

        assert_eq!(estimate.correctness_gain_lcb_micro, 86);
        assert_eq!(estimate.lifecycle_risk_reduction_lcb_micro, 184);
        assert_eq!(estimate.retrieval_harm_ucb_micro, 65);
        assert_eq!(estimate.calibration_split_sha256, bounds.calibration_split_sha256);
        assert_eq!(estimate.calibration_method_sha256, bounds.calibration_method_sha256);
        assert_eq!(estimate.calibration_joint_coverage_percent, 90);
    }
}
