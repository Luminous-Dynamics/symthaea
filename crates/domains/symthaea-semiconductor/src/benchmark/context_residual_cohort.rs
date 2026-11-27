// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-context residual cohort decomposition.
//!
//! A single context residual cannot identify common-mode and local drift. This
//! module therefore estimates common mode only from a closed cohort containing
//! enough distinct, declared common-cause groups. It is a pure decomposition
//! oracle: it does not mutate a streaming adapter or assert physical root cause.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::gated_streaming_adapter::StreamingPredictionReceiptV1;
use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
};

pub const CONTEXT_RESIDUAL_COHORT_SCHEMA_V1: &str =
    "luminous.semi-bench.context-residual-cohort.v1";
pub const CONTEXT_RESIDUAL_OBSERVATION_SCHEMA_V1: &str =
    "luminous.semi-bench.context-residual-observation.v1";
pub const CONTEXT_RESIDUAL_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.context-residual-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextResidualCohortStatusV1 {
    Complete,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextResidualObservationV1 {
    pub schema: String,
    pub observation_id: String,
    pub prediction_receipt: StreamingPredictionReceiptV1,
    pub label_id: String,
    pub label_revision: u64,
    pub label_payload_sha256: String,
    pub label_value: f64,
    pub label_available_at_unix_seconds: i64,
    pub eligibility_report_available_at_unix_seconds: i64,
    pub delivered_at_unix_seconds: i64,
    pub common_cause_group_id: String,
    pub common_cause_membership_receipt_sha256: String,
    pub calibration_epoch_id: String,
    pub eligibility_report: LabelConsumptionEligibilityReportV1,
    pub observation_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextResidualCohortInputV1 {
    pub schema: String,
    pub cohort_id: String,
    pub clock_domain_id: String,
    pub calibration_epoch_id: String,
    pub window_start_unix_seconds: i64,
    pub window_end_unix_seconds: i64,
    pub closed_at_unix_seconds: i64,
    pub minimum_independent_groups: usize,
    pub inventory_complete: bool,
    pub cohort_closed: bool,
    pub common_cause_registry_sha256: String,
    pub observations: Vec<ContextResidualObservationV1>,
    pub cohort_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextResidualCohortReportV1 {
    pub schema: String,
    pub cohort_id: String,
    pub cohort_sha256: String,
    pub status: ContextResidualCohortStatusV1,
    pub reason_codes: Vec<String>,
    pub eligible_observation_count: usize,
    pub independent_common_cause_group_count: usize,
    pub independent_common_cause_group_ids: Vec<String>,
    pub context_keys: Vec<String>,
    pub global_common_mode_residual: Option<f64>,
    /// Per-context residual median minus the common-mode cohort median.
    pub local_context_residual_offsets: BTreeMap<String, f64>,
    pub group_residual_medians: BTreeMap<String, f64>,
    pub source_authenticity_proven: bool,
    pub physical_independence_proven: bool,
    pub adapter_state_mutated: bool,
    pub process_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct ContextResidualCohortErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), ContextResidualCohortErrorV1> {
    if value.trim().is_empty() {
        return Err(ContextResidualCohortErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, ContextResidualCohortErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        ContextResidualCohortErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn eligibility_report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, ContextResidualCohortErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn prediction_receipt_digest(
    receipt: &StreamingPredictionReceiptV1,
) -> Result<String, ContextResidualCohortErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn observation_digest(
    observation: &ContextResidualObservationV1,
) -> Result<String, ContextResidualCohortErrorV1> {
    let mut canonical = observation.clone();
    canonical.observation_sha256.clear();
    json_sha256(&canonical)
}

/// Seals an observation over the complete prediction and K1 label evidence.
pub fn seal_context_residual_observation_v1(
    observation: &ContextResidualObservationV1,
) -> Result<String, ContextResidualCohortErrorV1> {
    observation_digest(observation)
}

/// Seals the cohort independent of caller-provided observation vector order.
pub fn seal_context_residual_cohort_v1(
    cohort: &ContextResidualCohortInputV1,
) -> Result<String, ContextResidualCohortErrorV1> {
    let mut canonical = cohort.clone();
    canonical.cohort_sha256.clear();
    canonical
        .observations
        .sort_by(|left, right| left.observation_id.cmp(&right.observation_id));
    json_sha256(&canonical)
}

fn median(values: &mut [f64]) -> Result<f64, ContextResidualCohortErrorV1> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(ContextResidualCohortErrorV1(
            "median requires a non-empty finite observation set".into(),
        ));
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    let result = if values.len() % 2 == 0 {
        values[middle - 1] / 2.0 + values[middle] / 2.0
    } else {
        values[middle]
    };
    if !result.is_finite() {
        return Err(ContextResidualCohortErrorV1(
            "median calculation produced a non-finite result".into(),
        ));
    }
    Ok(result)
}

fn report_digest(
    report: &ContextResidualCohortReportV1,
) -> Result<String, ContextResidualCohortErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn unknown_report(
    cohort: &ContextResidualCohortInputV1,
    reason: &str,
    eligible_observation_count: usize,
    independent_groups: Vec<String>,
    contexts: Vec<String>,
) -> Result<ContextResidualCohortReportV1, ContextResidualCohortErrorV1> {
    let mut report = ContextResidualCohortReportV1 {
        schema: CONTEXT_RESIDUAL_REPORT_SCHEMA_V1.into(),
        cohort_id: cohort.cohort_id.clone(),
        cohort_sha256: cohort.cohort_sha256.clone(),
        status: ContextResidualCohortStatusV1::Unknown,
        reason_codes: vec![reason.into()],
        eligible_observation_count,
        independent_common_cause_group_count: independent_groups.len(),
        independent_common_cause_group_ids: independent_groups,
        context_keys: contexts,
        global_common_mode_residual: None,
        local_context_residual_offsets: BTreeMap::new(),
        group_residual_medians: BTreeMap::new(),
        source_authenticity_proven: false,
        physical_independence_proven: false,
        adapter_state_mutated: false,
        process_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

/// Estimates global and local residual components from a sealed, complete
/// cohort. Every common-cause group contributes exactly one robust median to
/// the global estimate, regardless of how many chambers/observations share it.
///
/// The result is statistical decomposition only. A declared group ID is not
/// independent proof that its members are physically independent.
pub fn evaluate_context_residual_cohort_v1(
    cohort: &ContextResidualCohortInputV1,
) -> Result<ContextResidualCohortReportV1, ContextResidualCohortErrorV1> {
    if cohort.schema != CONTEXT_RESIDUAL_COHORT_SCHEMA_V1 {
        return Err(ContextResidualCohortErrorV1(
            "context residual cohort schema is unsupported".into(),
        ));
    }
    require_text(&cohort.cohort_id, "cohort.cohort_id")?;
    require_text(&cohort.clock_domain_id, "cohort.clock_domain_id")?;
    require_text(&cohort.calibration_epoch_id, "cohort.calibration_epoch_id")?;
    if !valid_sha256(&cohort.common_cause_registry_sha256)
        || !valid_sha256(&cohort.cohort_sha256)
        || seal_context_residual_cohort_v1(cohort)? != cohort.cohort_sha256
    {
        return Err(ContextResidualCohortErrorV1(
            "cohort registry or content digest is missing/mismatched".into(),
        ));
    }
    if cohort.window_start_unix_seconds > cohort.window_end_unix_seconds
        || cohort.closed_at_unix_seconds < cohort.window_end_unix_seconds
        || cohort.minimum_independent_groups == 0
    {
        return Err(ContextResidualCohortErrorV1(
            "cohort window, close time or independent-group minimum is invalid".into(),
        ));
    }

    let mut observation_ids = BTreeSet::new();
    let mut label_revision_keys = BTreeSet::new();
    let mut measurement_event_ids = BTreeSet::new();
    let mut grouped_residuals: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut context_residuals: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut eligible_observation_count = 0_usize;

    for observation in &cohort.observations {
        if observation.schema != CONTEXT_RESIDUAL_OBSERVATION_SCHEMA_V1 {
            return Err(ContextResidualCohortErrorV1(format!(
                "observation {} schema is unsupported",
                observation.observation_id
            )));
        }
        require_text(&observation.observation_id, "observation.observation_id")?;
        require_text(&observation.label_id, "observation.label_id")?;
        require_text(
            &observation.common_cause_group_id,
            "observation.common_cause_group_id",
        )?;
        require_text(
            &observation.calibration_epoch_id,
            "observation.calibration_epoch_id",
        )?;
        let context_key = observation
            .prediction_receipt
            .context_key
            .as_deref()
            .ok_or_else(|| {
                ContextResidualCohortErrorV1(format!(
                    "observation {} has no context key",
                    observation.observation_id
                ))
            })?;
        require_text(context_key, "prediction.context_key")?;

        if !observation_ids.insert(observation.observation_id.as_str()) {
            return Err(ContextResidualCohortErrorV1(format!(
                "duplicate observation ID {}",
                observation.observation_id
            )));
        }
        if !label_revision_keys.insert((observation.label_id.as_str(), observation.label_revision))
        {
            return Err(ContextResidualCohortErrorV1(format!(
                "duplicate label/revision {}:{}",
                observation.label_id, observation.label_revision
            )));
        }
        if !measurement_event_ids
            .insert(observation.prediction_receipt.measurement_event_id.as_str())
        {
            return Err(ContextResidualCohortErrorV1(format!(
                "duplicate measurement event {}",
                observation.prediction_receipt.measurement_event_id
            )));
        }
        if !valid_sha256(&observation.common_cause_membership_receipt_sha256)
            || !valid_sha256(&observation.label_payload_sha256)
            || !valid_sha256(&observation.prediction_receipt.global_model_artifact_sha256)
            || !valid_sha256(&observation.prediction_receipt.feature_schema_sha256)
            || !valid_sha256(&observation.prediction_receipt.policy_sha256)
            || observation_digest(observation)? != observation.observation_sha256
            || prediction_receipt_digest(&observation.prediction_receipt)?
                != observation.prediction_receipt.receipt_sha256
        {
            return Err(ContextResidualCohortErrorV1(format!(
                "observation {} contains a malformed or tampered receipt",
                observation.observation_id
            )));
        }

        let report = &observation.eligibility_report;
        if eligibility_report_digest(report)? != report.report_sha256
            || report.frontier != LabelEvidenceFrontierV1::CurrentAtK1
            || report.eligibility != LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
            || !report.downstream_consumption_allowed
            || report.training_snapshot_mutated
            || report.equipment_control_authority
            || report.label_id != observation.label_id
            || report.label_revision != observation.label_revision
            || report.measurement_event_id != observation.prediction_receipt.measurement_event_id
            || report.label_sha256 != observation.label_payload_sha256
            || !valid_sha256(&report.measurement_sha256)
            || !valid_sha256(&report.qualification_report_sha256)
            || !valid_sha256(&report.requalification_report_sha256)
            || !valid_sha256(&report.requalification_frontier_sha256)
            || report.qualified_standard_uncertainty_upper_bound.is_none()
            || report
                .qualified_standard_uncertainty_upper_bound
                .is_some_and(|value| !value.is_finite() || value < 0.0)
            || report.qualified_corrected_value != observation.label_value
        {
            return Err(ContextResidualCohortErrorV1(format!(
                "observation {} lacks an eligible, identity/value-bound current K1 report",
                observation.observation_id
            )));
        }

        let prediction = &observation.prediction_receipt;
        if prediction.clock_domain_id != cohort.clock_domain_id
            || observation.calibration_epoch_id != cohort.calibration_epoch_id
            || prediction.event_time_unix_seconds < cohort.window_start_unix_seconds
            || prediction.event_time_unix_seconds > cohort.window_end_unix_seconds
            || observation.label_available_at_unix_seconds < prediction.event_time_unix_seconds
            || observation.eligibility_report_available_at_unix_seconds
                < observation.label_available_at_unix_seconds
            || observation.delivered_at_unix_seconds
                < observation.eligibility_report_available_at_unix_seconds
            || observation.delivered_at_unix_seconds < prediction.prediction_at_unix_seconds
            || observation.delivered_at_unix_seconds > cohort.closed_at_unix_seconds
        {
            return Err(ContextResidualCohortErrorV1(format!(
                "observation {} is outside cohort clock, epoch or availability bounds",
                observation.observation_id
            )));
        }

        if !observation.label_value.is_finite() || !prediction.prediction_value.is_finite() {
            return Err(ContextResidualCohortErrorV1(format!(
                "observation {} contains a non-finite value",
                observation.observation_id
            )));
        }
        let residual = observation.label_value - prediction.prediction_value;
        if !residual.is_finite() {
            return Err(ContextResidualCohortErrorV1(format!(
                "observation {} residual is non-finite",
                observation.observation_id
            )));
        }
        grouped_residuals
            .entry(observation.common_cause_group_id.clone())
            .or_default()
            .push(residual);
        context_residuals
            .entry(context_key.to_string())
            .or_default()
            .push(residual);
        eligible_observation_count += 1;
    }

    let mut context_keys: Vec<String> = context_residuals.keys().cloned().collect();
    context_keys.sort();
    let mut independent_group_ids: Vec<String> = grouped_residuals.keys().cloned().collect();
    independent_group_ids.sort();

    if !cohort.inventory_complete {
        return unknown_report(
            cohort,
            "common_cause_inventory_incomplete",
            eligible_observation_count,
            independent_group_ids,
            context_keys,
        );
    }
    if !cohort.cohort_closed {
        return unknown_report(
            cohort,
            "cohort_not_closed",
            eligible_observation_count,
            independent_group_ids,
            context_keys,
        );
    }
    if independent_group_ids.len() < cohort.minimum_independent_groups {
        return unknown_report(
            cohort,
            "insufficient_independent_common_cause_groups",
            eligible_observation_count,
            independent_group_ids,
            context_keys,
        );
    }

    let mut group_residual_medians = BTreeMap::new();
    let mut group_values = Vec::with_capacity(grouped_residuals.len());
    for (group_id, mut residuals) in grouped_residuals {
        let estimate = median(&mut residuals)?;
        group_residual_medians.insert(group_id, estimate);
        group_values.push(estimate);
    }
    let global_common_mode_residual = median(&mut group_values)?;

    let mut local_context_residual_offsets = BTreeMap::new();
    for (context, mut residuals) in context_residuals {
        let local_median = median(&mut residuals)?;
        local_context_residual_offsets.insert(context, local_median - global_common_mode_residual);
    }

    let mut report = ContextResidualCohortReportV1 {
        schema: CONTEXT_RESIDUAL_REPORT_SCHEMA_V1.into(),
        cohort_id: cohort.cohort_id.clone(),
        cohort_sha256: cohort.cohort_sha256.clone(),
        status: ContextResidualCohortStatusV1::Complete,
        reason_codes: Vec::new(),
        eligible_observation_count,
        independent_common_cause_group_count: independent_group_ids.len(),
        independent_common_cause_group_ids: independent_group_ids,
        context_keys,
        global_common_mode_residual: Some(global_common_mode_residual),
        local_context_residual_offsets,
        group_residual_medians,
        source_authenticity_proven: false,
        physical_independence_proven: false,
        adapter_state_mutated: false,
        process_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn report(
        label_id: &str,
        revision: u64,
        event_id: &str,
        payload: &str,
        value: f64,
    ) -> LabelConsumptionEligibilityReportV1 {
        let mut value_report = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label_id.into(),
            label_revision: revision,
            measurement_event_id: event_id.into(),
            calibration_certificate_id: "cal-cohort-1".into(),
            frontier: LabelEvidenceFrontierV1::CurrentAtK1,
            eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            downstream_consumption_allowed: true,
            reasons: Vec::new(),
            label_sha256: payload.into(),
            measurement_sha256: digest('a'),
            qualified_corrected_value: value,
            qualified_standard_uncertainty_upper_bound: Some(0.1),
            qualification_report_sha256: digest('b'),
            requalification_report_sha256: digest('c'),
            requalification_frontier_sha256: digest('d'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        value_report.report_sha256 = eligibility_report_digest(&value_report).unwrap();
        value_report
    }

    fn observation(
        id: &str,
        context: &str,
        group: &str,
        residual: f64,
    ) -> ContextResidualObservationV1 {
        let label_value = 10.0 + residual;
        let event_id = format!("event-{id}");
        let label_id = format!("label-{id}");
        let payload = digest('e');
        let mut prediction = StreamingPredictionReceiptV1 {
            schema: super::super::gated_streaming_adapter::STREAMING_PREDICTION_RECEIPT_SCHEMA_V1
                .into(),
            sequence: id.parse::<u64>().unwrap_or(0),
            sample_id: format!("sample-{id}"),
            measurement_event_id: event_id.clone(),
            event_time_unix_seconds: 100,
            feature_available_at_unix_seconds: 101,
            prediction_at_unix_seconds: 102,
            clock_domain_id: "fab-clock".into(),
            global_model_artifact_sha256: digest('f'),
            feature_schema_sha256: digest('a'),
            policy_sha256: digest('b'),
            context_key: Some(context.into()),
            feature_vector_sha256: digest('c'),
            selected_feature_indices: vec![0],
            selected_feature_vector: vec![1.0],
            global_baseline_prediction: 10.0,
            feature_adapter_prediction: 0.0,
            context_correction: 0.0,
            temporal_correction: 0.0,
            base_prediction: 10.0,
            prediction_value: 10.0,
            adapter_state_before_sha256: digest('d'),
            adapter_state_after_sha256: digest('d'),
            receipt_sha256: String::new(),
        };
        prediction.receipt_sha256 = prediction_receipt_digest(&prediction).unwrap();

        let mut value = ContextResidualObservationV1 {
            schema: CONTEXT_RESIDUAL_OBSERVATION_SCHEMA_V1.into(),
            observation_id: id.into(),
            prediction_receipt: prediction,
            label_id: label_id.clone(),
            label_revision: 1,
            label_payload_sha256: payload.clone(),
            label_value,
            label_available_at_unix_seconds: 110,
            eligibility_report_available_at_unix_seconds: 111,
            delivered_at_unix_seconds: 112,
            common_cause_group_id: group.into(),
            common_cause_membership_receipt_sha256: digest('f'),
            calibration_epoch_id: "cal-epoch-1".into(),
            eligibility_report: report(&label_id, 1, &event_id, &payload, label_value),
            observation_sha256: String::new(),
        };
        value.observation_sha256 = observation_digest(&value).unwrap();
        value
    }

    fn cohort(
        observations: Vec<ContextResidualObservationV1>,
        min_groups: usize,
        inventory_complete: bool,
        closed: bool,
    ) -> ContextResidualCohortInputV1 {
        let mut value = ContextResidualCohortInputV1 {
            schema: CONTEXT_RESIDUAL_COHORT_SCHEMA_V1.into(),
            cohort_id: "cohort-001".into(),
            clock_domain_id: "fab-clock".into(),
            calibration_epoch_id: "cal-epoch-1".into(),
            window_start_unix_seconds: 90,
            window_end_unix_seconds: 105,
            closed_at_unix_seconds: 120,
            minimum_independent_groups: min_groups,
            inventory_complete,
            cohort_closed: closed,
            common_cause_registry_sha256: digest('1'),
            observations,
            cohort_sha256: String::new(),
        };
        value.cohort_sha256 = seal_context_residual_cohort_v1(&value).unwrap();
        value
    }

    #[test]
    fn single_context_group_cannot_identify_global_common_mode() {
        let input = cohort(
            vec![
                observation("1", "tool-a/ch-1", "cal-ref-a", 4.0),
                observation("2", "tool-a/ch-2", "cal-ref-a", 4.2),
            ],
            2,
            true,
            true,
        );
        let report = evaluate_context_residual_cohort_v1(&input).unwrap();
        assert_eq!(report.status, ContextResidualCohortStatusV1::Unknown);
        assert!(report.global_common_mode_residual.is_none());
        assert_eq!(report.independent_common_cause_group_count, 1);
    }

    #[test]
    fn coherent_independent_contexts_identify_common_mode_and_local_offsets() {
        let input = cohort(
            vec![
                observation("1", "tool-a/ch-1", "cal-a", 2.0),
                observation("2", "tool-b/ch-1", "cal-b", 2.2),
                observation("3", "tool-c/ch-1", "cal-c", 1.8),
            ],
            3,
            true,
            true,
        );
        let report = evaluate_context_residual_cohort_v1(&input).unwrap();
        assert_eq!(report.status, ContextResidualCohortStatusV1::Complete);
        assert!((report.global_common_mode_residual.unwrap() - 2.0).abs() < 1.0e-12);
        assert!(
            report
                .local_context_residual_offsets
                .values()
                .all(|v| v.abs() <= 0.2 + 1.0e-12)
        );
        assert!(!report.physical_independence_proven);
        assert!(!report.adapter_state_mutated);
    }

    #[test]
    fn a_single_outlier_group_does_not_dominate_robust_global_median() {
        let input = cohort(
            vec![
                observation("1", "tool-a/ch-1", "cal-a", 2.0),
                observation("2", "tool-b/ch-1", "cal-b", 2.1),
                observation("3", "tool-c/ch-1", "cal-c", 80.0),
            ],
            3,
            true,
            true,
        );
        let report = evaluate_context_residual_cohort_v1(&input).unwrap();
        assert!((report.global_common_mode_residual.unwrap() - 2.1).abs() < 1.0e-12);
    }

    #[test]
    fn common_cause_group_counts_once_even_for_many_chambers() {
        let input = cohort(
            vec![
                observation("1", "tool-a/ch-1", "shared-ref", 2.0),
                observation("2", "tool-a/ch-2", "shared-ref", 2.2),
                observation("3", "tool-b/ch-1", "independent-ref", 2.1),
            ],
            2,
            true,
            true,
        );
        let report = evaluate_context_residual_cohort_v1(&input).unwrap();
        assert_eq!(report.independent_common_cause_group_count, 2);
        assert_eq!(report.status, ContextResidualCohortStatusV1::Complete);
    }

    #[test]
    fn incomplete_inventory_and_open_cohort_return_unknown() {
        for (inventory, closed, reason) in [
            (false, true, "common_cause_inventory_incomplete"),
            (true, false, "cohort_not_closed"),
        ] {
            let input = cohort(
                vec![
                    observation("1", "tool-a/ch-1", "cal-a", 2.0),
                    observation("2", "tool-b/ch-1", "cal-b", 2.0),
                ],
                2,
                inventory,
                closed,
            );
            let report = evaluate_context_residual_cohort_v1(&input).unwrap();
            assert_eq!(report.status, ContextResidualCohortStatusV1::Unknown);
            assert_eq!(report.reason_codes, vec![reason]);
            assert!(report.global_common_mode_residual.is_none());
        }
    }

    #[test]
    fn duplicate_label_revision_cannot_count_twice_across_contexts() {
        let mut first = observation("1", "tool-a/ch-1", "cal-a", 2.0);
        let mut second = observation("2", "tool-b/ch-1", "cal-b", 2.1);
        second.label_id = first.label_id.clone();
        second.label_revision = first.label_revision;
        second.eligibility_report.label_id = second.label_id.clone();
        second.eligibility_report.label_revision = second.label_revision;
        second.eligibility_report.report_sha256 =
            eligibility_report_digest(&second.eligibility_report).unwrap();
        second.observation_sha256 = observation_digest(&second).unwrap();
        first.observation_sha256 = observation_digest(&first).unwrap();
        let input = cohort(vec![first, second], 2, true, true);
        assert!(evaluate_context_residual_cohort_v1(&input).is_err());
    }

    #[test]
    fn mismatched_clock_domain_cannot_enter_one_cohort() {
        let first = observation("1", "tool-a/ch-1", "cal-a", 2.0);
        let mut second = observation("2", "tool-b/ch-1", "cal-b", 2.1);
        second.prediction_receipt.clock_domain_id = "other-clock".into();
        second.prediction_receipt.receipt_sha256 =
            prediction_receipt_digest(&second.prediction_receipt).unwrap();
        second.observation_sha256 = observation_digest(&second).unwrap();
        let input = cohort(vec![first, second], 2, true, true);
        assert!(evaluate_context_residual_cohort_v1(&input).is_err());
    }

    #[test]
    fn input_order_does_not_change_cohort_or_report_digest() {
        let mut input = cohort(
            vec![
                observation("1", "tool-a/ch-1", "cal-a", 2.0),
                observation("2", "tool-b/ch-1", "cal-b", 2.1),
                observation("3", "tool-c/ch-1", "cal-c", 1.9),
            ],
            3,
            true,
            true,
        );
        let first = evaluate_context_residual_cohort_v1(&input).unwrap();
        input.observations.reverse();
        input.cohort_sha256 = seal_context_residual_cohort_v1(&input).unwrap();
        let second = evaluate_context_residual_cohort_v1(&input).unwrap();
        assert_eq!(first.cohort_sha256, second.cohort_sha256);
        assert_eq!(first.report_sha256, second.report_sha256);
    }

    #[test]
    fn late_or_wrong_epoch_observation_is_rejected() {
        let mut input = cohort(
            vec![
                observation("1", "tool-a/ch-1", "cal-a", 2.0),
                observation("2", "tool-b/ch-1", "cal-b", 2.1),
            ],
            2,
            true,
            true,
        );
        input.observations[0].delivered_at_unix_seconds = 121;
        input.observations[0].observation_sha256 =
            observation_digest(&input.observations[0]).unwrap();
        input.cohort_sha256 = seal_context_residual_cohort_v1(&input).unwrap();
        assert!(evaluate_context_residual_cohort_v1(&input).is_err());

        let mut other_epoch = cohort(
            vec![
                observation("1", "tool-a/ch-1", "cal-a", 2.0),
                observation("2", "tool-b/ch-1", "cal-b", 2.1),
            ],
            2,
            true,
            true,
        );
        other_epoch.observations[0].calibration_epoch_id = "other-epoch".into();
        other_epoch.observations[0].observation_sha256 =
            observation_digest(&other_epoch.observations[0]).unwrap();
        other_epoch.cohort_sha256 = seal_context_residual_cohort_v1(&other_epoch).unwrap();
        assert!(evaluate_context_residual_cohort_v1(&other_epoch).is_err());
    }
}
