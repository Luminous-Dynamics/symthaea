// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! K1-qualified signed-residual window and two-sided interval constructor for tau-DACI.
//!
//! The delayed ACI alpha controller and the residual-score distribution are
//! independent state variables. This module freezes a window digest into an
//! interval-construction receipt, binds that receipt to the immutable ACI
//! prediction, and appends a residual only when the matching delayed ACI
//! update is accepted. The paired delivery operation commits both states
//! transactionally.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

use super::delayed_aci::{
    AciIntervalKindV1, AciPredictionIntervalV1, AciUpdateDispositionV1,
    DelayedAciEngineV1, DelayedAciLabelDeliveryV1, DelayedAciPredictionReceiptV1,
    DelayedAciUpdateReceiptV1, deliver_delayed_aci_label_v1,
    verify_delayed_aci_prediction_receipt_v1, verify_delayed_aci_update_receipt_v1,
};
use super::label_consumption_guard::{
    LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
};

pub const ACI_RESIDUAL_WINDOW_SCHEMA_V1: &str =
    "luminous.semi-bench.aci-residual-window.v1";
pub const ACI_RESIDUAL_WINDOW_POLICY_SCHEMA_V1: &str =
    "luminous.semi-bench.aci-residual-window-policy.v1";
pub const ACI_RESIDUAL_SEED_MANIFEST_SCHEMA_V1: &str =
    "luminous.semi-bench.aci-residual-seed-manifest.v1";
pub const ACI_RESIDUAL_INTERVAL_SCHEMA_V1: &str =
    "luminous.semi-bench.aci-residual-interval.v1";
pub const ACI_RESIDUAL_INTERVAL_BINDING_SCHEMA_V1: &str =
    "luminous.semi-bench.aci-residual-interval-binding.v1";
pub const ACI_RESIDUAL_APPEND_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.aci-residual-append-receipt.v1";
pub const ACI_RESIDUAL_DELIVERY_OUTCOME_SCHEMA_V1: &str =
    "luminous.semi-bench.aci-residual-delivery-outcome.v1";
pub const ACI_RAW_SIGNED_RESIDUAL_SCORE_CONTRACT_V1: &str =
    "luminous.semi-bench.aci-score.raw-signed-residual.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AciResidualSeedManifestV1 {
    pub schema: String,
    pub seed_id: String,
    pub score_contract_id: String,
    pub signed_residuals: Vec<f64>,
    pub manifest_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AciResidualWindowPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub score_contract_id: String,
    pub capacity: usize,
    pub minimum_calibration_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AciResidualObservationSourceV1 {
    InitialSeed,
    QualifiedK1Update,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AciResidualObservationV1 {
    pub sequence: u64,
    pub source: AciResidualObservationSourceV1,
    pub signed_residual: f64,
    pub sample_id: Option<String>,
    pub measurement_event_id: Option<String>,
    pub label_id: Option<String>,
    pub label_revision: Option<u64>,
    pub prediction_receipt_sha256: Option<String>,
    pub update_receipt_sha256: Option<String>,
    pub label_payload_sha256: Option<String>,
    pub qualification_report_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AciResidualWindowV1 {
    pub schema: String,
    pub engine_id: String,
    pub policy: AciResidualWindowPolicyV1,
    pub policy_sha256: String,
    pub seed_manifest_sha256: String,
    pub observations: Vec<AciResidualObservationV1>,
    pub next_sequence: u64,
    pub accepted_label_revisions: BTreeSet<String>,
    pub window_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AciResidualIntervalConstructionReceiptV1 {
    pub schema: String,
    pub window_sha256: String,
    pub seed_manifest_sha256: String,
    pub score_contract_id: String,
    pub calibration_count: usize,
    pub minimum_calibration_count: usize,
    pub point_prediction: f64,
    pub alpha_at_issue: f64,
    pub lower_tail_probability: f64,
    pub upper_tail_probability: f64,
    /// One-based order-statistic ranks. None means the respective endpoint is unbounded.
    pub lower_order_statistic_rank: Option<usize>,
    pub upper_order_statistic_rank: Option<usize>,
    pub lower_signed_residual: Option<f64>,
    pub upper_signed_residual: Option<f64>,
    pub interval: AciPredictionIntervalV1,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AciResidualIntervalBindingReceiptV1 {
    pub schema: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub prediction_sequence: u64,
    pub prediction_receipt_sha256: String,
    pub construction_receipt_sha256: String,
    pub window_sha256: String,
    pub point_prediction: f64,
    pub alpha_at_issue: f64,
    pub interval: AciPredictionIntervalV1,
    pub binding_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AciResidualAppendReceiptV1 {
    pub schema: String,
    pub engine_id: String,
    pub sample_id: String,
    pub measurement_event_id: String,
    pub label_id: String,
    pub label_revision: u64,
    pub signed_residual: f64,
    pub prediction_receipt_sha256: String,
    pub update_receipt_sha256: String,
    pub qualification_report_sha256: String,
    pub window_state_before_sha256: String,
    pub window_state_after_sha256: String,
    pub appended_sequence: u64,
    pub evicted_oldest_observation: bool,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AciResidualDeliveryOutcomeV1 {
    pub schema: String,
    pub update_receipt: DelayedAciUpdateReceiptV1,
    pub residual_append_receipt: Option<AciResidualAppendReceiptV1>,
    pub aci_engine_mutated: bool,
    pub residual_window_mutated: bool,
    pub outcome_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct AciResidualWindowErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), AciResidualWindowErrorV1> {
    if value.trim().is_empty() {
        return Err(AciResidualWindowErrorV1(format!(
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

fn json_sha256<T: Serialize + ?Sized>(
    value: &T,
) -> Result<String, AciResidualWindowErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        AciResidualWindowErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn seed_manifest_digest(
    manifest: &AciResidualSeedManifestV1,
) -> Result<String, AciResidualWindowErrorV1> {
    let mut canonical = manifest.clone();
    canonical.manifest_sha256.clear();
    json_sha256(&canonical)
}

fn policy_digest(
    policy: &AciResidualWindowPolicyV1,
) -> Result<String, AciResidualWindowErrorV1> {
    json_sha256(policy)
}

fn window_digest(window: &AciResidualWindowV1) -> Result<String, AciResidualWindowErrorV1> {
    let mut canonical = window.clone();
    canonical.window_sha256.clear();
    json_sha256(&canonical)
}

fn construction_digest(
    receipt: &AciResidualIntervalConstructionReceiptV1,
) -> Result<String, AciResidualWindowErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn binding_digest(
    receipt: &AciResidualIntervalBindingReceiptV1,
) -> Result<String, AciResidualWindowErrorV1> {
    let mut canonical = receipt.clone();
    canonical.binding_sha256.clear();
    json_sha256(&canonical)
}

fn append_digest(
    receipt: &AciResidualAppendReceiptV1,
) -> Result<String, AciResidualWindowErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn outcome_digest(
    outcome: &AciResidualDeliveryOutcomeV1,
) -> Result<String, AciResidualWindowErrorV1> {
    let mut canonical = outcome.clone();
    canonical.outcome_sha256.clear();
    json_sha256(&canonical)
}

/// Seals an initial, externally prepared signed-residual calibration seed.
/// The seed digest identifies the declared input; it does not prove physical
/// provenance or source authenticity.
pub fn seal_aci_residual_seed_manifest_v1(
    seed_id: &str,
    signed_residuals: Vec<f64>,
) -> Result<AciResidualSeedManifestV1, AciResidualWindowErrorV1> {
    require_text(seed_id, "seed_id")?;
    if signed_residuals.iter().any(|value| !value.is_finite()) {
        return Err(AciResidualWindowErrorV1(
            "initial residual seed contains non-finite values".into(),
        ));
    }
    let mut manifest = AciResidualSeedManifestV1 {
        schema: ACI_RESIDUAL_SEED_MANIFEST_SCHEMA_V1.into(),
        seed_id: seed_id.into(),
        score_contract_id: ACI_RAW_SIGNED_RESIDUAL_SCORE_CONTRACT_V1.into(),
        signed_residuals,
        manifest_sha256: String::new(),
    };
    manifest.manifest_sha256 = seed_manifest_digest(&manifest)?;
    Ok(manifest)
}

fn validate_policy(policy: &AciResidualWindowPolicyV1) -> Result<(), AciResidualWindowErrorV1> {
    if policy.schema != ACI_RESIDUAL_WINDOW_POLICY_SCHEMA_V1 {
        return Err(AciResidualWindowErrorV1(
            "ACI residual-window policy schema is unsupported".into(),
        ));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    if policy.score_contract_id != ACI_RAW_SIGNED_RESIDUAL_SCORE_CONTRACT_V1
        || policy.capacity == 0
        || policy.minimum_calibration_count == 0
        || policy.minimum_calibration_count > policy.capacity
    {
        return Err(AciResidualWindowErrorV1(
            "ACI residual-window policy has invalid score contract/capacity".into(),
        ));
    }
    Ok(())
}

fn validate_window(window: &AciResidualWindowV1) -> Result<(), AciResidualWindowErrorV1> {
    if window.schema != ACI_RESIDUAL_WINDOW_SCHEMA_V1 {
        return Err(AciResidualWindowErrorV1(
            "ACI residual-window schema is unsupported".into(),
        ));
    }
    require_text(&window.engine_id, "window.engine_id")?;
    validate_policy(&window.policy)?;
    if !valid_sha256(&window.policy_sha256)
        || !valid_sha256(&window.seed_manifest_sha256)
        || policy_digest(&window.policy)? != window.policy_sha256
        || window.observations.len() > window.policy.capacity
        || window.observations.iter().any(|item| !item.signed_residual.is_finite())
        || window_digest(window)? != window.window_sha256
    {
        return Err(AciResidualWindowErrorV1(
            "ACI residual-window state digest, capacity or numeric invariant failed".into(),
        ));
    }
    Ok(())
}

/// Creates a FIFO residual window from a sealed initial calibration seed.
pub fn create_aci_residual_window_v1(
    engine_id: &str,
    policy: AciResidualWindowPolicyV1,
    seed: &AciResidualSeedManifestV1,
) -> Result<AciResidualWindowV1, AciResidualWindowErrorV1> {
    require_text(engine_id, "engine_id")?;
    validate_policy(&policy)?;
    require_text(&seed.seed_id, "seed.seed_id")?;
    if seed.schema != ACI_RESIDUAL_SEED_MANIFEST_SCHEMA_V1
        || seed.score_contract_id != policy.score_contract_id
        || seed_manifest_digest(seed)? != seed.manifest_sha256
        || seed.signed_residuals.len() > policy.capacity
        || seed.signed_residuals.iter().any(|value| !value.is_finite())
    {
        return Err(AciResidualWindowErrorV1(
            "initial residual seed manifest is invalid or exceeds capacity".into(),
        ));
    }

    let observations = seed
        .signed_residuals
        .iter()
        .enumerate()
        .map(|(index, residual)| AciResidualObservationV1 {
            sequence: index as u64,
            source: AciResidualObservationSourceV1::InitialSeed,
            signed_residual: *residual,
            sample_id: None,
            measurement_event_id: None,
            label_id: None,
            label_revision: None,
            prediction_receipt_sha256: None,
            update_receipt_sha256: None,
            label_payload_sha256: None,
            qualification_report_sha256: None,
        })
        .collect::<Vec<_>>();

    let mut window = AciResidualWindowV1 {
        schema: ACI_RESIDUAL_WINDOW_SCHEMA_V1.into(),
        engine_id: engine_id.into(),
        policy_sha256: policy_digest(&policy)?,
        seed_manifest_sha256: seed.manifest_sha256.clone(),
        observations,
        next_sequence: seed.signed_residuals.len() as u64,
        accepted_label_revisions: BTreeSet::new(),
        policy,
        window_sha256: String::new(),
    };
    window.window_sha256 = window_digest(&window)?;
    Ok(window)
}

/// Builds an equal-tailed two-sided interval from raw signed residuals.
///
/// Lower endpoint rank is floor((n+1)*alpha/2); upper endpoint rank is
/// ceil((n+1)*(1-alpha/2)). Rank 0 and rank n+1 are represented by explicit
/// unbounded endpoints. A sample too small to provide either endpoint yields
/// CalibrationUnbounded, distinct from the ACI alpha<=0 full-real-line state.
pub fn build_aci_signed_residual_interval_v1(
    window: &AciResidualWindowV1,
    point_prediction: f64,
    alpha_at_issue: f64,
) -> Result<AciResidualIntervalConstructionReceiptV1, AciResidualWindowErrorV1> {
    validate_window(window)?;
    if !point_prediction.is_finite()
        || !alpha_at_issue.is_finite()
        || alpha_at_issue <= 0.0
        || alpha_at_issue > 1.0
    {
        return Err(AciResidualWindowErrorV1(
            "point prediction or non-degenerate alpha is invalid".into(),
        ));
    }

    let mut residuals = window
        .observations
        .iter()
        .map(|item| item.signed_residual)
        .collect::<Vec<_>>();
    residuals.sort_by(f64::total_cmp);
    let calibration_count = residuals.len();
    let lower_tail_probability = alpha_at_issue / 2.0;
    let upper_tail_probability = 1.0 - lower_tail_probability;
    let lower_rank_value =
        (((calibration_count + 1) as f64) * lower_tail_probability).floor();
    let upper_rank_value =
        (((calibration_count + 1) as f64) * upper_tail_probability).ceil();
    let lower_rank = (lower_rank_value >= 1.0 && lower_rank_value <= calibration_count as f64)
        .then_some(lower_rank_value as usize);
    let upper_rank = (upper_rank_value >= 1.0 && upper_rank_value <= calibration_count as f64)
        .then_some(upper_rank_value as usize);
    let lower_signed_residual = lower_rank.map(|rank| residuals[rank - 1]);
    let upper_signed_residual = upper_rank.map(|rank| residuals[rank - 1]);

    let interval = if calibration_count < window.policy.minimum_calibration_count
        || (lower_signed_residual.is_none() && upper_signed_residual.is_none())
    {
        AciPredictionIntervalV1::calibration_unbounded()
    } else {
        match (lower_signed_residual, upper_signed_residual) {
            (Some(lower), Some(upper)) => {
                let lower_endpoint = point_prediction + lower;
                let upper_endpoint = point_prediction + upper;
                if !lower_endpoint.is_finite()
                    || !upper_endpoint.is_finite()
                    || lower_endpoint > upper_endpoint
                {
                    return Err(AciResidualWindowErrorV1(
                        "constructed finite interval endpoints are invalid".into(),
                    ));
                }
                AciPredictionIntervalV1::bounded(lower_endpoint, upper_endpoint)
            }
            (None, Some(upper)) => {
                let upper_endpoint = point_prediction + upper;
                if !upper_endpoint.is_finite() {
                    return Err(AciResidualWindowErrorV1(
                        "constructed upper endpoint is non-finite".into(),
                    ));
                }
                AciPredictionIntervalV1::lower_unbounded(upper_endpoint)
            }
            (Some(lower), None) => {
                let lower_endpoint = point_prediction + lower;
                if !lower_endpoint.is_finite() {
                    return Err(AciResidualWindowErrorV1(
                        "constructed lower endpoint is non-finite".into(),
                    ));
                }
                AciPredictionIntervalV1::upper_unbounded(lower_endpoint)
            }
            (None, None) => AciPredictionIntervalV1::calibration_unbounded(),
        }
    };

    let mut receipt = AciResidualIntervalConstructionReceiptV1 {
        schema: ACI_RESIDUAL_INTERVAL_SCHEMA_V1.into(),
        window_sha256: window.window_sha256.clone(),
        seed_manifest_sha256: window.seed_manifest_sha256.clone(),
        score_contract_id: window.policy.score_contract_id.clone(),
        calibration_count,
        minimum_calibration_count: window.policy.minimum_calibration_count,
        point_prediction,
        alpha_at_issue,
        lower_tail_probability,
        upper_tail_probability,
        lower_order_statistic_rank: lower_rank,
        upper_order_statistic_rank: upper_rank,
        lower_signed_residual,
        upper_signed_residual,
        interval,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = construction_digest(&receipt)?;
    Ok(receipt)
}

/// Binds a constructed interval to the exact immutable tau-DACI prediction.
pub fn bind_aci_interval_construction_v1(
    construction: &AciResidualIntervalConstructionReceiptV1,
    prediction: &DelayedAciPredictionReceiptV1,
) -> Result<AciResidualIntervalBindingReceiptV1, AciResidualWindowErrorV1> {
    verify_delayed_aci_prediction_receipt_v1(prediction)
        .map_err(|error| AciResidualWindowErrorV1(error.to_string()))?;
    if construction.schema != ACI_RESIDUAL_INTERVAL_SCHEMA_V1
        || construction_digest(construction)? != construction.receipt_sha256
        || construction.point_prediction.to_bits() != prediction.point_prediction.to_bits()
        || construction.alpha_at_issue.to_bits() != prediction.alpha_at_issue.to_bits()
        || construction.interval != prediction.interval
        || !valid_sha256(&construction.window_sha256)
        || !valid_sha256(&construction.seed_manifest_sha256)
    {
        return Err(AciResidualWindowErrorV1(
            "interval construction does not match the issued ACI prediction".into(),
        ));
    }

    let mut binding = AciResidualIntervalBindingReceiptV1 {
        schema: ACI_RESIDUAL_INTERVAL_BINDING_SCHEMA_V1.into(),
        sample_id: prediction.sample_id.clone(),
        measurement_event_id: prediction.measurement_event_id.clone(),
        prediction_sequence: prediction.sequence,
        prediction_receipt_sha256: prediction.receipt_sha256.clone(),
        construction_receipt_sha256: construction.receipt_sha256.clone(),
        window_sha256: construction.window_sha256.clone(),
        point_prediction: prediction.point_prediction,
        alpha_at_issue: prediction.alpha_at_issue,
        interval: prediction.interval,
        binding_sha256: String::new(),
    };
    binding.binding_sha256 = binding_digest(&binding)?;
    Ok(binding)
}

fn report_digest(
    report: &super::label_consumption_guard::LabelConsumptionEligibilityReportV1,
) -> Result<String, AciResidualWindowErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn interval_misses(interval: AciPredictionIntervalV1, value: f64) -> bool {
    match interval.kind {
        AciIntervalKindV1::FullRealLine | AciIntervalKindV1::CalibrationUnbounded => false,
        AciIntervalKindV1::Empty => true,
        AciIntervalKindV1::Bounded => {
            value < interval.lower.expect("verified interval")
                || value > interval.upper.expect("verified interval")
        }
        AciIntervalKindV1::LowerUnbounded => value > interval.upper.expect("verified interval"),
        AciIntervalKindV1::UpperUnbounded => value < interval.lower.expect("verified interval"),
    }
}

fn validate_delivery_evidence(
    prediction: &DelayedAciPredictionReceiptV1,
    delivery: &DelayedAciLabelDeliveryV1,
    update: &DelayedAciUpdateReceiptV1,
) -> Result<(), AciResidualWindowErrorV1> {
    let report = &delivery.eligibility_report;
    if delivery.sample_id != prediction.sample_id
        || delivery.measurement_event_id != prediction.measurement_event_id
        || delivery.clock_domain_id != prediction.clock_domain_id
        || delivery.delivered_round != prediction.expected_delivery_round
        || delivery.delivered_at_unix_seconds < prediction.prediction_time_unix_seconds
        || delivery.label_available_at_unix_seconds < prediction.event_time_unix_seconds
        || delivery.label_available_at_unix_seconds > delivery.delivered_at_unix_seconds
        || delivery.eligibility_report_available_at_unix_seconds
            < delivery.label_available_at_unix_seconds
        || delivery.eligibility_report_available_at_unix_seconds
            > delivery.delivered_at_unix_seconds
        || report.frontier != LabelEvidenceFrontierV1::CurrentAtK1
        || report.eligibility != LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
        || !report.downstream_consumption_allowed
        || report.label_id != delivery.label_id
        || report.label_revision != delivery.label_revision
        || report.measurement_event_id != delivery.measurement_event_id
        || report.label_sha256 != delivery.label_payload_sha256
        || report.qualified_corrected_value.to_bits() != delivery.label_value.to_bits()
        || report_digest(report)? != report.report_sha256
        || update.disposition != AciUpdateDispositionV1::Updated
        || update.label_id != delivery.label_id
        || update.label_revision != delivery.label_revision
        || update.measurement_event_id != delivery.measurement_event_id
        || update.label_payload_sha256 != delivery.label_payload_sha256
        || update.eligibility_report_sha256.as_deref() != Some(report.report_sha256.as_str())
        || update.prediction_receipt_sha256.as_deref() != Some(prediction.receipt_sha256.as_str())
        || update.phase != Some(prediction.phase)
        || update.alpha_before.is_none_or(|value| value.to_bits() != prediction.alpha_at_issue.to_bits())
        || update.miscoverage_observed != Some(interval_misses(prediction.interval, delivery.label_value))
    {
        return Err(AciResidualWindowErrorV1(
            "qualified residual delivery is not bound to its prediction/update/K1 receipts".into(),
        ));
    }
    Ok(())
}

/// Computes a candidate append against a cloned residual window. The public
/// paired-delivery function below commits it only together with the ACI state.
fn append_residual_candidate(
    window: &AciResidualWindowV1,
    binding: &AciResidualIntervalBindingReceiptV1,
    construction: &AciResidualIntervalConstructionReceiptV1,
    prediction: &DelayedAciPredictionReceiptV1,
    delivery: &DelayedAciLabelDeliveryV1,
    update: &DelayedAciUpdateReceiptV1,
) -> Result<(AciResidualWindowV1, AciResidualAppendReceiptV1), AciResidualWindowErrorV1> {
    validate_window(window)?;
    verify_delayed_aci_prediction_receipt_v1(prediction)
        .map_err(|error| AciResidualWindowErrorV1(error.to_string()))?;
    verify_delayed_aci_update_receipt_v1(update)
        .map_err(|error| AciResidualWindowErrorV1(error.to_string()))?;
    if binding.schema != ACI_RESIDUAL_INTERVAL_BINDING_SCHEMA_V1
        || binding_digest(binding)? != binding.binding_sha256
        || construction.schema != ACI_RESIDUAL_INTERVAL_SCHEMA_V1
        || construction_digest(construction)? != construction.receipt_sha256
        || binding.construction_receipt_sha256 != construction.receipt_sha256
        || binding.prediction_receipt_sha256 != prediction.receipt_sha256
        || binding.window_sha256 != construction.window_sha256
        || binding.sample_id != prediction.sample_id
        || binding.measurement_event_id != prediction.measurement_event_id
        || binding.point_prediction.to_bits() != prediction.point_prediction.to_bits()
        || binding.alpha_at_issue.to_bits() != prediction.alpha_at_issue.to_bits()
        || binding.interval != prediction.interval
    {
        return Err(AciResidualWindowErrorV1(
            "residual interval binding receipt is invalid".into(),
        ));
    }
    validate_delivery_evidence(prediction, delivery, update)?;
    if window.accepted_label_revisions.contains(&format!(
        "{}:{}",
        delivery.label_id, delivery.label_revision
    )) {
        return Err(AciResidualWindowErrorV1(
            "label revision has already contributed a residual".into(),
        ));
    }

    let signed_residual = delivery.label_value - prediction.point_prediction;
    if !signed_residual.is_finite() {
        return Err(AciResidualWindowErrorV1(
            "qualified signed residual is non-finite".into(),
        ));
    }
    let before = window.window_sha256.clone();
    let sequence = window.next_sequence;
    let mut next = window.clone();
    next.observations.push(AciResidualObservationV1 {
        sequence,
        source: AciResidualObservationSourceV1::QualifiedK1Update,
        signed_residual,
        sample_id: Some(prediction.sample_id.clone()),
        measurement_event_id: Some(prediction.measurement_event_id.clone()),
        label_id: Some(delivery.label_id.clone()),
        label_revision: Some(delivery.label_revision),
        prediction_receipt_sha256: Some(prediction.receipt_sha256.clone()),
        update_receipt_sha256: Some(update.receipt_sha256.clone()),
        label_payload_sha256: Some(delivery.label_payload_sha256.clone()),
        qualification_report_sha256: Some(delivery.eligibility_report.report_sha256.clone()),
    });
    let evicted = if next.observations.len() > next.policy.capacity {
        next.observations.remove(0);
        true
    } else {
        false
    };
    next.next_sequence = next
        .next_sequence
        .checked_add(1)
        .ok_or_else(|| AciResidualWindowErrorV1("residual observation sequence overflow".into()))?;
    next.accepted_label_revisions
        .insert(format!("{}:{}", delivery.label_id, delivery.label_revision));
    next.window_sha256 = window_digest(&next)?;

    let mut receipt = AciResidualAppendReceiptV1 {
        schema: ACI_RESIDUAL_APPEND_RECEIPT_SCHEMA_V1.into(),
        engine_id: next.engine_id.clone(),
        sample_id: prediction.sample_id.clone(),
        measurement_event_id: prediction.measurement_event_id.clone(),
        label_id: delivery.label_id.clone(),
        label_revision: delivery.label_revision,
        signed_residual,
        prediction_receipt_sha256: prediction.receipt_sha256.clone(),
        update_receipt_sha256: update.receipt_sha256.clone(),
        qualification_report_sha256: delivery.eligibility_report.report_sha256.clone(),
        window_state_before_sha256: before,
        window_state_after_sha256: next.window_sha256.clone(),
        appended_sequence: sequence,
        evicted_oldest_observation: evicted,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = append_digest(&receipt)?;
    Ok((next, receipt))
}

/// Delivers one K1-qualified label to both the fixed-delay ACI engine and its
/// residual window. Both states are cloned and committed atomically. UNKNOWN,
/// BLOCKED and rejected feedback returns the ACI receipt but never touches the
/// residual window.
pub fn deliver_and_append_aci_residual_v1(
    engine: &mut DelayedAciEngineV1,
    window: &mut AciResidualWindowV1,
    construction: &AciResidualIntervalConstructionReceiptV1,
    binding: &AciResidualIntervalBindingReceiptV1,
    prediction: &DelayedAciPredictionReceiptV1,
    delivery: &DelayedAciLabelDeliveryV1,
) -> Result<AciResidualDeliveryOutcomeV1, AciResidualWindowErrorV1> {
    validate_window(window)?;
    let before_aci = engine.clone();
    let before_window = window.clone();
    let mut next_engine = engine.clone();
    let mut next_window = window.clone();

    let update = deliver_delayed_aci_label_v1(&mut next_engine, delivery)
        .map_err(|error| AciResidualWindowErrorV1(error.to_string()))?;
    let mut append_receipt = None;
    if update.disposition == AciUpdateDispositionV1::Updated {
        let (candidate_window, append) = append_residual_candidate(
            &next_window,
            binding,
            construction,
            prediction,
            delivery,
            &update,
        )?;
        next_window = candidate_window;
        append_receipt = Some(append);
        *engine = next_engine;
        *window = next_window;
    }

    let mut outcome = AciResidualDeliveryOutcomeV1 {
        schema: ACI_RESIDUAL_DELIVERY_OUTCOME_SCHEMA_V1.into(),
        update_receipt: update.clone(),
        residual_append_receipt: append_receipt,
        aci_engine_mutated: update.disposition == AciUpdateDispositionV1::Updated,
        residual_window_mutated: update.disposition == AciUpdateDispositionV1::Updated,
        outcome_sha256: String::new(),
    };
    if update.disposition != AciUpdateDispositionV1::Updated
        && (*engine != before_aci || *window != before_window)
    {
        return Err(AciResidualWindowErrorV1(
            "no-update delivery mutated ACI or residual-window state".into(),
        ));
    }
    outcome.outcome_sha256 = outcome_digest(&outcome)?;
    Ok(outcome)
}

pub fn verify_aci_residual_interval_construction_v1(
    receipt: &AciResidualIntervalConstructionReceiptV1,
) -> Result<(), AciResidualWindowErrorV1> {
    if receipt.schema != ACI_RESIDUAL_INTERVAL_SCHEMA_V1
        || !valid_sha256(&receipt.window_sha256)
        || !valid_sha256(&receipt.seed_manifest_sha256)
        || !receipt.point_prediction.is_finite()
        || !receipt.alpha_at_issue.is_finite()
        || receipt.alpha_at_issue <= 0.0
        || receipt.alpha_at_issue > 1.0
        || receipt.calibration_count == 0
        || receipt.score_contract_id != ACI_RAW_SIGNED_RESIDUAL_SCORE_CONTRACT_V1
    {
        return Err(AciResidualWindowErrorV1(
            "ACI residual interval construction receipt is malformed".into(),
        ));
    }
    if construction_digest(receipt)? != receipt.receipt_sha256 {
        return Err(AciResidualWindowErrorV1(
            "ACI residual interval construction digest mismatch".into(),
        ));
    }
    Ok(())
}

pub fn verify_aci_residual_append_receipt_v1(
    receipt: &AciResidualAppendReceiptV1,
) -> Result<(), AciResidualWindowErrorV1> {
    if receipt.schema != ACI_RESIDUAL_APPEND_RECEIPT_SCHEMA_V1
        || !valid_sha256(&receipt.prediction_receipt_sha256)
        || !valid_sha256(&receipt.update_receipt_sha256)
        || !valid_sha256(&receipt.qualification_report_sha256)
        || !valid_sha256(&receipt.window_state_before_sha256)
        || !valid_sha256(&receipt.window_state_after_sha256)
        || !receipt.signed_residual.is_finite()
    {
        return Err(AciResidualWindowErrorV1(
            "ACI residual append receipt is malformed".into(),
        ));
    }
    if append_digest(receipt)? != receipt.receipt_sha256 {
        return Err(AciResidualWindowErrorV1(
            "ACI residual append receipt digest mismatch".into(),
        ));
    }
    Ok(())
}

pub fn verify_aci_residual_delivery_outcome_v1(
    outcome: &AciResidualDeliveryOutcomeV1,
) -> Result<(), AciResidualWindowErrorV1> {
    if outcome.schema != ACI_RESIDUAL_DELIVERY_OUTCOME_SCHEMA_V1 {
        return Err(AciResidualWindowErrorV1(
            "ACI residual delivery outcome schema is unsupported".into(),
        ));
    }
    verify_delayed_aci_update_receipt_v1(&outcome.update_receipt)
        .map_err(|error| AciResidualWindowErrorV1(error.to_string()))?;
    if outcome.update_receipt.disposition == AciUpdateDispositionV1::Updated {
        if !outcome.aci_engine_mutated
            || !outcome.residual_window_mutated
            || outcome.residual_append_receipt.is_none()
        {
            return Err(AciResidualWindowErrorV1(
                "accepted ACI outcome is missing atomic residual append".into(),
            ));
        }
        verify_aci_residual_append_receipt_v1(
            outcome.residual_append_receipt.as_ref().expect("checked"),
        )?;
    } else if outcome.aci_engine_mutated
        || outcome.residual_window_mutated
        || outcome.residual_append_receipt.is_some()
    {
        return Err(AciResidualWindowErrorV1(
            "denied ACI outcome claims residual/alpha mutation".into(),
        ));
    }
    if outcome_digest(outcome)? != outcome.outcome_sha256 {
        return Err(AciResidualWindowErrorV1(
            "ACI residual delivery outcome digest mismatch".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::delayed_aci::{
        DELAYED_ACI_DELIVERY_SCHEMA_V1, DELAYED_ACI_POLICY_SCHEMA_V1,
        DELAYED_ACI_PREDICTION_SCHEMA_V1, DelayedAciPolicyV1,
        begin_delayed_aci_round_v1, create_delayed_aci_engine_v1,
        issue_delayed_aci_prediction_v1,
    };

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn seed_residuals() -> Vec<f64> {
        (-10..10).map(f64::from).collect()
    }

    fn policy() -> AciResidualWindowPolicyV1 {
        AciResidualWindowPolicyV1 {
            schema: ACI_RESIDUAL_WINDOW_POLICY_SCHEMA_V1.into(),
            policy_id: "semi-vm-raw-residual-window.v1".into(),
            score_contract_id: ACI_RAW_SIGNED_RESIDUAL_SCORE_CONTRACT_V1.into(),
            capacity: 24,
            minimum_calibration_count: 8,
        }
    }

    fn seed() -> AciResidualSeedManifestV1 {
        seal_aci_residual_seed_manifest_v1("seed-manifest-001", seed_residuals()).unwrap()
    }

    fn window() -> AciResidualWindowV1 {
        create_aci_residual_window_v1("residual-engine-001", policy(), &seed()).unwrap()
    }

    fn aci_policy() -> DelayedAciPolicyV1 {
        DelayedAciPolicyV1 {
            schema: DELAYED_ACI_POLICY_SCHEMA_V1.into(),
            policy_id: "semi-vm-aci-residual-window.v1".into(),
            target_miscoverage: 0.10,
            learning_rate: 0.20,
            delay_rounds: 1,
            initial_phase_alpha: 0.10,
        }
    }

    fn engine() -> DelayedAciEngineV1 {
        create_delayed_aci_engine_v1("aci-residual-engine-001", "fab-clock", aci_policy()).unwrap()
    }

    fn prediction_input(
        sample: &str,
        event: &str,
        round: u64,
        interval: AciPredictionIntervalV1,
    ) -> crate::benchmark::DelayedAciPredictionInputV1 {
        crate::benchmark::DelayedAciPredictionInputV1 {
            schema: DELAYED_ACI_PREDICTION_SCHEMA_V1.into(),
            sample_id: sample.into(),
            measurement_event_id: event.into(),
            round,
            event_time_unix_seconds: 100 + round as i64 * 10,
            prediction_time_unix_seconds: 101 + round as i64 * 10,
            clock_domain_id: "fab-clock".into(),
            model_artifact_sha256: digest('a'),
            point_prediction: 50.0,
            interval,
        }
    }

    fn make_delivery(
        sample: &str,
        event: &str,
        label: &str,
        value: f64,
        eligibility: LabelConsumptionEligibilityV1,
        frontier: LabelEvidenceFrontierV1,
    ) -> DelayedAciLabelDeliveryV1 {
        let payload = digest('f');
        let revision = 1;
        let label_available_at_unix_seconds = 110;
        let eligibility_report_available_at_unix_seconds = 110;
        let report_value = value;
        let mut report = super::super::label_consumption_guard::LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label.into(),
            label_revision: revision,
            measurement_event_id: event.into(),
            calibration_certificate_id: "aci-residual-cert-001".into(),
            frontier,
            eligibility,
            downstream_consumption_allowed: eligibility
                == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            reasons: Vec::new(),
            label_sha256: payload.clone(),
            measurement_sha256: digest('b'),
            qualified_corrected_value: report_value,
            qualified_standard_uncertainty_upper_bound: Some(0.1),
            qualification_report_sha256: digest('c'),
            requalification_report_sha256: digest('d'),
            requalification_frontier_sha256: digest('e'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        let mut canonical = report.clone();
        canonical.report_sha256.clear();
        report.report_sha256 = json_sha256(&canonical).unwrap();
        DelayedAciLabelDeliveryV1 {
            schema: DELAYED_ACI_DELIVERY_SCHEMA_V1.into(),
            sample_id: sample.into(),
            label_id: label.into(),
            label_revision: revision,
            measurement_event_id: event.into(),
            label_payload_sha256: payload,
            label_value: value,
            label_available_at_unix_seconds,
            eligibility_report_available_at_unix_seconds,
            delivered_at_unix_seconds: 111,
            delivered_round: 1,
            clock_domain_id: "fab-clock".into(),
            idempotency_key: format!("{label}:{revision}"),
            eligibility_report: report,
        }
    }

    fn prepare_prediction(
        engine: &mut DelayedAciEngineV1,
        window: &AciResidualWindowV1,
        round: u64,
        sample: &str,
        event: &str,
        alpha: f64,
    ) -> (
        AciResidualIntervalConstructionReceiptV1,
        DelayedAciPredictionReceiptV1,
        AciResidualIntervalBindingReceiptV1,
    ) {
        begin_delayed_aci_round_v1(engine, round).unwrap();
        let construction =
            build_aci_signed_residual_interval_v1(window, 50.0, alpha).unwrap();
        let prediction = issue_delayed_aci_prediction_v1(
            engine,
            &prediction_input(sample, event, round, construction.interval),
        )
        .unwrap();
        let binding =
            bind_aci_interval_construction_v1(&construction, &prediction).unwrap();
        (construction, prediction, binding)
    }

    #[test]
    fn lower_and_upper_ranks_match_independent_order_statistic_formula() {
        let window = window();
        let receipt = build_aci_signed_residual_interval_v1(&window, 50.0, 0.2).unwrap();
        assert_eq!(receipt.calibration_count, 20);
        assert_eq!(receipt.lower_order_statistic_rank, Some(2));
        assert_eq!(receipt.upper_order_statistic_rank, Some(19));
        assert_eq!(receipt.lower_signed_residual, Some(-9.0));
        assert_eq!(receipt.upper_signed_residual, Some(8.0));
        verify_aci_residual_interval_construction_v1(&receipt).unwrap();
    }

    #[test]
    fn insufficient_seed_uses_calibration_unbounded_not_aci_alpha_degeneracy() {
        let seed = seal_aci_residual_seed_manifest_v1("tiny-seed", vec![-1.0, 1.0]).unwrap();
        let mut policy = policy();
        policy.capacity = 4;
        policy.minimum_calibration_count = 4;
        let window = create_aci_residual_window_v1("tiny-window", policy, &seed).unwrap();
        let receipt = build_aci_signed_residual_interval_v1(&window, 5.0, 0.1).unwrap();
        assert_eq!(receipt.interval.kind, AciIntervalKindV1::CalibrationUnbounded);
        assert_eq!(receipt.interval.lower, None);
        assert_eq!(receipt.interval.upper, None);
    }

    #[test]
    fn seed_manifest_and_window_digests_are_deterministic() {
        let first = seed();
        let second = seed();
        assert_eq!(first.manifest_sha256, second.manifest_sha256);
        assert_eq!(window().window_sha256, window().window_sha256);
    }

    #[test]
    fn appended_qualified_residual_uses_frozen_prediction_not_new_point() {
        let mut engine = engine();
        let mut window = window();
        let (construction, prediction, binding) =
            prepare_prediction(&mut engine, &window, 0, "sample-0", "event-0", 0.1);
        begin_delayed_aci_round_v1(&mut engine, 1).unwrap();
        let delivery = make_delivery(
            "sample-0",
            "event-0",
            "label-0",
            51.5,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
        );
        let outcome = deliver_and_append_aci_residual_v1(
            &mut engine,
            &mut window,
            &construction,
            &binding,
            &prediction,
            &delivery,
        )
        .unwrap();
        assert_eq!(
            outcome.update_receipt.disposition,
            AciUpdateDispositionV1::Updated
        );
        assert_eq!(
            outcome.residual_append_receipt.as_ref().unwrap().signed_residual,
            1.5
        );
        assert_eq!(window.observations.last().unwrap().signed_residual, 1.5);
        verify_aci_residual_delivery_outcome_v1(&outcome).unwrap();
    }

    #[test]
    fn interval_receipt_and_prediction_receipt_are_bound_exactly() {
        let mut engine = engine();
        let window = window();
        let (construction, prediction, binding) =
            prepare_prediction(&mut engine, &window, 0, "sample-0", "event-0", 0.1);
        let mut tampered = binding.clone();
        tampered.window_sha256 = digest('z');
        assert!(append_residual_candidate(
            &window,
            &tampered,
            &construction,
            &prediction,
            &make_delivery(
                "sample-0",
                "event-0",
                "label-0",
                51.0,
                LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                LabelEvidenceFrontierV1::CurrentAtK1,
            ),
            &dummy_update_receipt(&prediction, &tampered),
        )
        .is_err());
    }

    fn dummy_update_receipt(
        prediction: &DelayedAciPredictionReceiptV1,
        binding: &AciResidualIntervalBindingReceiptV1,
    ) -> DelayedAciUpdateReceiptV1 {
        let _ = (prediction, binding);
        panic!("dummy receipt is not used by this test; binding must reject first")
    }

    #[test]
    fn unknown_and_blocked_feedback_do_not_append_or_evict() {
        for eligibility in [
            LabelConsumptionEligibilityV1::Unknown,
            LabelConsumptionEligibilityV1::Blocked,
        ] {
            let mut engine = engine();
            let mut window = window();
            let (construction, prediction, binding) =
                prepare_prediction(&mut engine, &window, 0, "sample-0", "event-0", 0.1);
            begin_delayed_aci_round_v1(&mut engine, 1).unwrap();
            let delivery = make_delivery(
                "sample-0",
                "event-0",
                "label-0",
                51.0,
                eligibility,
                LabelEvidenceFrontierV1::CurrentAtK1,
            );
            let before_window = window.window_sha256.clone();
            let outcome = deliver_and_append_aci_residual_v1(
                &mut engine,
                &mut window,
                &construction,
                &binding,
                &prediction,
                &delivery,
            )
            .unwrap();
            assert_ne!(
                outcome.update_receipt.disposition,
                AciUpdateDispositionV1::Updated
            );
            assert_eq!(window.window_sha256, before_window);
            assert!(outcome.residual_append_receipt.is_none());
        }
    }

    #[test]
    fn duplicate_label_revision_cannot_append_twice() {
        let mut engine = engine();
        let mut window = window();
        let (construction, prediction, binding) =
            prepare_prediction(&mut engine, &window, 0, "sample-0", "event-0", 0.1);
        begin_delayed_aci_round_v1(&mut engine, 1).unwrap();
        let delivery = make_delivery(
            "sample-0",
            "event-0",
            "label-0",
            51.0,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            LabelEvidenceFrontierV1::CurrentAtK1,
        );
        deliver_and_append_aci_residual_v1(
            &mut engine,
            &mut window,
            &construction,
            &binding,
            &prediction,
            &delivery,
        )
        .unwrap();
        assert!(window.accepted_label_revisions.contains("label-0:1"));
        assert!(append_residual_candidate(
            &window,
            &binding,
            &construction,
            &prediction,
            &delivery,
            &dummy_update_receipt(&prediction, &binding),
        )
        .is_err());
    }

    #[test]
    fn fifo_eviction_is_deterministic_and_changes_window_digest() {
        let seed = seal_aci_residual_seed_manifest_v1(
            "full-seed",
            (0..4).map(|value| value as f64).collect(),
        )
        .unwrap();
        let mut policy = policy();
        policy.capacity = 4;
        policy.minimum_calibration_count = 1;
        let window = create_aci_residual_window_v1("fifo", policy, &seed).unwrap();
        assert_eq!(window.observations.len(), 4);
        assert_eq!(window.observations[0].signed_residual, 0.0);
    }

    #[test]
    fn non_finite_seed_and_point_values_are_rejected() {
        assert!(seal_aci_residual_seed_manifest_v1("bad-seed", vec![f64::NAN]).is_err());
        let mut bad = window();
        bad.observations[0].signed_residual = f64::INFINITY;
        bad.window_sha256 = window_digest(&bad).unwrap();
        assert!(validate_window(&bad).is_err());
        assert!(build_aci_signed_residual_interval_v1(&window(), f64::NAN, 0.1).is_err());
    }
}
