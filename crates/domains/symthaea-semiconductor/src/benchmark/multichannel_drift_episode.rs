// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-channel, evidence-qualified drift monitoring for semiconductor VM.
//!
//! This module reports statistical change and evidence-backed associations.
//! It does not infer physical root cause. Residual channels accept only current
//! K1-qualified metrology evidence; denied labels do not mutate detector state.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::calibration_requalification::{
    CALIBRATION_REQUALIFICATION_REPORT_SCHEMA_V1, CalibrationRequalificationFrontierReportV1,
    CalibrationRequalificationReportV1,
};
use super::label_consumption_guard::{
    LABEL_CONSUMPTION_REPORT_SCHEMA_V1, LabelConsumptionEligibilityReportV1,
    LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
};
use super::metrology_qualification_gate::MeasurementQualificationStatusV1;

pub const DRIFT_MONITOR_SCHEMA_V1: &str = "luminous.semi-bench.drift-monitor.v1";
pub const DRIFT_POLICY_SCHEMA_V1: &str = "luminous.semi-bench.drift-policy.v1";
pub const DRIFT_OBSERVATION_SCHEMA_V1: &str = "luminous.semi-bench.drift-observation.v1";
pub const DRIFT_OBSERVATION_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.drift-observation-receipt.v1";
pub const DRIFT_EPISODE_SCHEMA_V1: &str = "luminous.semi-bench.drift-episode.v1";
pub const SHARED_RESOURCE_GRAPH_SCHEMA_V1: &str = "luminous.semi-bench.shared-resource-graph.v1";
pub const COMMON_CAUSE_ASSESSMENT_SCHEMA_V1: &str =
    "luminous.semi-bench.common-cause-assessment.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftEvidencePlaneV1 {
    InputFeature,
    QualifiedResidual,
    MeasurementSystem,
    ProcessContext,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftDetectorKindV1 {
    Ewma,
    TwoSidedCusum,
    MissingnessRate,
    EventChange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftEvidenceKindV1 {
    FeatureWindow,
    QualifiedMetrologyLabel,
    CalibrationQualification,
    ProcessEvent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftQualificationV1 {
    NotApplicable,
    CurrentK1Eligible,
    QualifiedSystemEvidence,
    Unknown,
    Blocked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftObservationDispositionV1 {
    Updated,
    HeldUnknown,
    Blocked,
    Rejected,
    Duplicate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftAlarmStateV1 {
    InControl,
    Alert,
    InsufficientEvidence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalAttributionStateV1 {
    NotEstablished,
    ExternalReviewAttached,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommonCauseAssessmentStateV1 {
    Unknown,
    NoDeclaredSharedResource,
    SharedResourceAssociationCandidate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftObservationReasonV1 {
    EvidenceKindPlaneMismatch,
    QualificationNotCurrentK1,
    QualificationUnknown,
    QualificationBlocked,
    InvalidClockDomain,
    InvalidChronology,
    ObservationDelayExceeded,
    FeatureSchemaMismatch,
    InvalidSignal,
    SignalOutOfPolicy,
    DuplicateEvidenceReceipt,
    ChannelIdentityChanged,
    InsufficientSampleCount,
    MissingEvidenceDigest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriftMonitorPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    pub ewma_lambda: f64,
    pub cusum_slack: f64,
    pub alarm_threshold: f64,
    pub min_observations: u64,
    pub maximum_abs_standardized_signal: f64,
    pub maximum_observation_delay_seconds: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriftChannelStateV1 {
    pub channel_id: String,
    pub plane: DriftEvidencePlaneV1,
    pub detector: DriftDetectorKindV1,
    pub scope_id: String,
    pub feature_schema_sha256: String,
    pub baseline_sha256: String,
    pub observations: u64,
    pub missing_observations: u64,
    pub ewma: f64,
    pub positive_cusum: f64,
    pub negative_cusum: f64,
    pub last_event_time_unix_seconds: i64,
    pub last_observed_at_unix_seconds: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriftMonitorEngineV1 {
    pub schema: String,
    pub engine_id: String,
    pub clock_domain_id: String,
    pub policy: DriftMonitorPolicyV1,
    pub policy_sha256: String,
    pub channels: BTreeMap<String, DriftChannelStateV1>,
    pub seen_evidence_receipt_sha256: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriftObservationInputV1 {
    pub schema: String,
    pub observation_id: String,
    pub channel_id: String,
    pub plane: DriftEvidencePlaneV1,
    pub detector: DriftDetectorKindV1,
    pub scope_id: String,
    pub event_time_unix_seconds: i64,
    pub observed_at_unix_seconds: i64,
    pub knowledge_frontier_unix_seconds: i64,
    pub clock_domain_id: String,
    pub feature_schema_sha256: String,
    pub baseline_sha256: String,
    pub evidence_kind: DriftEvidenceKindV1,
    pub qualification: DriftQualificationV1,
    pub measurement_event_id: Option<String>,
    pub evidence_receipt_sha256: String,
    pub qualification_report: Option<LabelConsumptionEligibilityReportV1>,
    pub calibration_report: Option<CalibrationRequalificationReportV1>,
    pub prediction_receipt_sha256: Option<String>,
    pub source_label_ids: Vec<String>,
    /// Standardized scalar for EWMA/CUSUM; missingness is represented separately.
    pub standardized_signal: Option<f64>,
    pub is_missing: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriftObservationReceiptV1 {
    pub schema: String,
    pub observation_id: String,
    pub channel_id: String,
    pub plane: DriftEvidencePlaneV1,
    pub detector: DriftDetectorKindV1,
    pub scope_id: String,
    pub disposition: DriftObservationDispositionV1,
    pub reason: Option<DriftObservationReasonV1>,
    pub alarm_state: DriftAlarmStateV1,
    pub observation_count: u64,
    pub missing_count: u64,
    pub statistic: f64,
    pub threshold: f64,
    pub event_time_unix_seconds: i64,
    pub observed_at_unix_seconds: i64,
    pub knowledge_frontier_unix_seconds: i64,
    pub evidence_receipt_sha256: String,
    pub measurement_event_id: Option<String>,
    pub feature_schema_sha256: String,
    pub baseline_sha256: String,
    pub clock_domain_id: String,
    pub qualification_report_sha256: Option<String>,
    pub calibration_report_sha256: Option<String>,
    pub prediction_receipt_sha256: Option<String>,
    pub source_label_ids: Vec<String>,
    pub policy_sha256: String,
    pub channel_state_before_sha256: String,
    pub channel_state_after_sha256: String,
    pub causal_attribution: CausalAttributionStateV1,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriftEpisodeReceiptV1 {
    pub schema: String,
    pub episode_id: String,
    pub window_start_unix_seconds: i64,
    pub window_end_unix_seconds: i64,
    pub observation_receipt_sha256s: Vec<String>,
    pub active_planes: Vec<DriftEvidencePlaneV1>,
    pub alert_channel_ids: Vec<String>,
    pub denied_observation_count: u64,
    pub causal_attribution: CausalAttributionStateV1,
    pub episode_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SharedResourceBindingV1 {
    pub context_id: String,
    pub resource_id: String,
    pub resource_kind: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SharedResourceGraphV1 {
    pub schema: String,
    pub graph_id: String,
    pub inventory_complete: bool,
    pub bindings: Vec<SharedResourceBindingV1>,
    pub graph_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriftAlertContextV1 {
    pub alert_id: String,
    pub context_id: String,
    pub observation_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CommonCauseAssessmentV1 {
    pub schema: String,
    pub graph_id: String,
    pub graph_sha256: String,
    pub state: CommonCauseAssessmentStateV1,
    pub alert_ids: Vec<String>,
    pub shared_resource_ids: Vec<String>,
    pub reason_codes: Vec<String>,
    /// An association supported by a declared graph is not a causal finding.
    pub causal_attribution: CausalAttributionStateV1,
    pub assessment_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct DriftMonitorErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), DriftMonitorErrorV1> {
    if value.trim().is_empty() {
        return Err(DriftMonitorErrorV1(format!(
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

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, DriftMonitorErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        DriftMonitorErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn policy_digest(policy: &DriftMonitorPolicyV1) -> Result<String, DriftMonitorErrorV1> {
    json_sha256(policy)
}

fn channel_state_digest(
    state: Option<&DriftChannelStateV1>,
) -> Result<String, DriftMonitorErrorV1> {
    json_sha256(&state)
}

fn observation_receipt_digest(
    receipt: &DriftObservationReceiptV1,
) -> Result<String, DriftMonitorErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn validate_policy(policy: &DriftMonitorPolicyV1) -> Result<(), DriftMonitorErrorV1> {
    if policy.schema != DRIFT_POLICY_SCHEMA_V1 {
        return Err(DriftMonitorErrorV1(
            "drift policy schema is unsupported".into(),
        ));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    if !policy.ewma_lambda.is_finite()
        || !(0.0..=1.0).contains(&policy.ewma_lambda)
        || !policy.cusum_slack.is_finite()
        || policy.cusum_slack < 0.0
        || !policy.alarm_threshold.is_finite()
        || policy.alarm_threshold <= 0.0
        || policy.min_observations == 0
        || !policy.maximum_abs_standardized_signal.is_finite()
        || policy.maximum_abs_standardized_signal <= 0.0
        || policy.maximum_observation_delay_seconds < 0
    {
        return Err(DriftMonitorErrorV1(
            "drift policy contains invalid parameters".into(),
        ));
    }
    Ok(())
}

fn qualification_report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, DriftMonitorErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn validate_engine(engine: &DriftMonitorEngineV1) -> Result<(), DriftMonitorErrorV1> {
    if engine.schema != DRIFT_MONITOR_SCHEMA_V1 {
        return Err(DriftMonitorErrorV1(
            "drift monitor engine schema is unsupported".into(),
        ));
    }
    validate_policy(&engine.policy)?;
    if policy_digest(&engine.policy)? != engine.policy_sha256 {
        return Err(DriftMonitorErrorV1(
            "drift policy digest does not match its content".into(),
        ));
    }
    Ok(())
}

fn evidence_kind_matches(input: &DriftObservationInputV1) -> bool {
    matches!(
        (input.plane, input.evidence_kind, input.qualification),
        (
            DriftEvidencePlaneV1::InputFeature,
            DriftEvidenceKindV1::FeatureWindow,
            DriftQualificationV1::NotApplicable
        ) | (
            DriftEvidencePlaneV1::QualifiedResidual,
            DriftEvidenceKindV1::QualifiedMetrologyLabel,
            DriftQualificationV1::CurrentK1Eligible
        ) | (
            DriftEvidencePlaneV1::MeasurementSystem,
            DriftEvidenceKindV1::CalibrationQualification,
            DriftQualificationV1::QualifiedSystemEvidence
        ) | (
            DriftEvidencePlaneV1::ProcessContext,
            DriftEvidenceKindV1::ProcessEvent,
            DriftQualificationV1::NotApplicable
        )
    )
}

fn denied_disposition(input: &DriftObservationInputV1) -> DriftObservationDispositionV1 {
    match input.qualification {
        DriftQualificationV1::Unknown => DriftObservationDispositionV1::HeldUnknown,
        DriftQualificationV1::Blocked => DriftObservationDispositionV1::Blocked,
        _ => DriftObservationDispositionV1::Rejected,
    }
}

fn denied_reason(input: &DriftObservationInputV1) -> DriftObservationReasonV1 {
    match input.qualification {
        DriftQualificationV1::Unknown => DriftObservationReasonV1::QualificationUnknown,
        DriftQualificationV1::Blocked => DriftObservationReasonV1::QualificationBlocked,
        _ if input.plane == DriftEvidencePlaneV1::QualifiedResidual => {
            DriftObservationReasonV1::QualificationNotCurrentK1
        }
        _ => DriftObservationReasonV1::EvidenceKindPlaneMismatch,
    }
}

fn make_receipt(
    engine: &DriftMonitorEngineV1,
    input: &DriftObservationInputV1,
    disposition: DriftObservationDispositionV1,
    reason: Option<DriftObservationReasonV1>,
    alarm_state: DriftAlarmStateV1,
    observations: u64,
    missing: u64,
    statistic: f64,
    before: &str,
    after: &str,
) -> Result<DriftObservationReceiptV1, DriftMonitorErrorV1> {
    let mut label_ids = input.source_label_ids.clone();
    label_ids.sort();
    label_ids.dedup();
    let mut receipt = DriftObservationReceiptV1 {
        schema: DRIFT_OBSERVATION_RECEIPT_SCHEMA_V1.into(),
        observation_id: input.observation_id.clone(),
        channel_id: input.channel_id.clone(),
        plane: input.plane,
        detector: input.detector,
        scope_id: input.scope_id.clone(),
        disposition,
        reason,
        alarm_state,
        observation_count: observations,
        missing_count: missing,
        statistic,
        threshold: engine.policy.alarm_threshold,
        event_time_unix_seconds: input.event_time_unix_seconds,
        observed_at_unix_seconds: input.observed_at_unix_seconds,
        knowledge_frontier_unix_seconds: input.knowledge_frontier_unix_seconds,
        evidence_receipt_sha256: input.evidence_receipt_sha256.clone(),
        measurement_event_id: input.measurement_event_id.clone(),
        feature_schema_sha256: input.feature_schema_sha256.clone(),
        baseline_sha256: input.baseline_sha256.clone(),
        clock_domain_id: input.clock_domain_id.clone(),
        qualification_report_sha256: input
            .qualification_report
            .as_ref()
            .map(|report| report.report_sha256.clone()),
        calibration_report_sha256: input
            .calibration_report
            .as_ref()
            .map(|report| report.report_sha256.clone()),
        prediction_receipt_sha256: input.prediction_receipt_sha256.clone(),
        source_label_ids: label_ids,
        policy_sha256: engine.policy_sha256.clone(),
        channel_state_before_sha256: before.into(),
        channel_state_after_sha256: after.into(),
        causal_attribution: CausalAttributionStateV1::NotEstablished,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = observation_receipt_digest(&receipt)?;
    Ok(receipt)
}

pub fn create_drift_monitor_engine_v1(
    engine_id: &str,
    clock_domain_id: &str,
    policy: DriftMonitorPolicyV1,
) -> Result<DriftMonitorEngineV1, DriftMonitorErrorV1> {
    require_text(engine_id, "engine_id")?;
    require_text(clock_domain_id, "clock_domain_id")?;
    validate_policy(&policy)?;
    let policy_sha256 = policy_digest(&policy)?;
    Ok(DriftMonitorEngineV1 {
        schema: DRIFT_MONITOR_SCHEMA_V1.into(),
        engine_id: engine_id.into(),
        clock_domain_id: clock_domain_id.into(),
        policy,
        policy_sha256,
        channels: BTreeMap::new(),
        seen_evidence_receipt_sha256: BTreeSet::new(),
    })
}

fn calibration_report_digest(
    report: &CalibrationRequalificationReportV1,
) -> Result<String, DriftMonitorErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn validate_input(
    engine: &DriftMonitorEngineV1,
    input: &DriftObservationInputV1,
) -> Result<(), DriftObservationReasonV1> {
    if input.schema != DRIFT_OBSERVATION_SCHEMA_V1
        || input.clock_domain_id != engine.clock_domain_id
    {
        return Err(DriftObservationReasonV1::InvalidClockDomain);
    }
    if input.event_time_unix_seconds > input.observed_at_unix_seconds
        || input.observed_at_unix_seconds > input.knowledge_frontier_unix_seconds
    {
        return Err(DriftObservationReasonV1::InvalidChronology);
    }
    if input.observed_at_unix_seconds - input.event_time_unix_seconds
        > engine.policy.maximum_observation_delay_seconds
    {
        return Err(DriftObservationReasonV1::ObservationDelayExceeded);
    }
    if !valid_sha256(&input.feature_schema_sha256) || !valid_sha256(&input.baseline_sha256) {
        return Err(DriftObservationReasonV1::FeatureSchemaMismatch);
    }
    if !valid_sha256(&input.evidence_receipt_sha256) {
        return Err(DriftObservationReasonV1::MissingEvidenceDigest);
    }
    if !evidence_kind_matches(input) {
        return Err(denied_reason(input));
    }
    if input.plane == DriftEvidencePlaneV1::QualifiedResidual {
        let Some(report) = input.qualification_report.as_ref() else {
            return Err(DriftObservationReasonV1::QualificationNotCurrentK1);
        };
        if report.schema != LABEL_CONSUMPTION_REPORT_SCHEMA_V1
            || report.frontier != LabelEvidenceFrontierV1::CurrentAtK1
        {
            return Err(DriftObservationReasonV1::QualificationNotCurrentK1);
        }
        match report.eligibility {
            LabelConsumptionEligibilityV1::Unknown => {
                return Err(DriftObservationReasonV1::QualificationUnknown);
            }
            LabelConsumptionEligibilityV1::Blocked => {
                return Err(DriftObservationReasonV1::QualificationBlocked);
            }
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy => {}
        }
        if !report.downstream_consumption_allowed
            || qualification_report_digest(report).ok().as_deref()
                != Some(report.report_sha256.as_str())
            || report.report_sha256 != input.evidence_receipt_sha256
            || !valid_sha256(&report.label_sha256)
            || input.source_label_ids.len() != 1
            || input.source_label_ids.first() != Some(&report.label_id)
            || input.measurement_event_id.as_deref() != Some(report.measurement_event_id.as_str())
            || input
                .prediction_receipt_sha256
                .as_deref()
                .is_none_or(|value| !valid_sha256(value))
        {
            return Err(DriftObservationReasonV1::MissingEvidenceDigest);
        }
    } else if input.qualification_report.is_some() || input.prediction_receipt_sha256.is_some() {
        return Err(DriftObservationReasonV1::EvidenceKindPlaneMismatch);
    }

    if input.plane == DriftEvidencePlaneV1::MeasurementSystem {
        let Some(report) = input.calibration_report.as_ref() else {
            return Err(DriftObservationReasonV1::EvidenceKindPlaneMismatch);
        };
        if report.schema != CALIBRATION_REQUALIFICATION_REPORT_SCHEMA_V1
            || input.measurement_event_id.as_deref() != Some(report.measurement_event_id.as_str())
            || calibration_report_digest(report).ok().as_deref()
                != Some(report.report_sha256.as_str())
            || report.report_sha256 != input.evidence_receipt_sha256
            || !valid_sha256(&report.current_at_k1.frontier_input_sha256)
            || report.k1_unix_seconds > input.knowledge_frontier_unix_seconds
            || report.equipment_control_authority
        {
            return Err(DriftObservationReasonV1::MissingEvidenceDigest);
        }
        match report.current_at_k1.status {
            MeasurementQualificationStatusV1::Qualified
                if report
                    .current_at_k1
                    .downstream_consumption_allowed_at_frontier => {}
            MeasurementQualificationStatusV1::Unknown => {
                return Err(DriftObservationReasonV1::QualificationUnknown);
            }
            MeasurementQualificationStatusV1::Rejected
            | MeasurementQualificationStatusV1::Qualified => {
                return Err(DriftObservationReasonV1::QualificationBlocked);
            }
        }
    } else if input.calibration_report.is_some() {
        return Err(DriftObservationReasonV1::EvidenceKindPlaneMismatch);
    }

    if input.plane != DriftEvidencePlaneV1::QualifiedResidual
        && (!input.source_label_ids.is_empty()
            || input.qualification_report.is_some()
            || input.prediction_receipt_sha256.is_some())
    {
        return Err(DriftObservationReasonV1::EvidenceKindPlaneMismatch);
    }
    if input.plane != DriftEvidencePlaneV1::MeasurementSystem && input.calibration_report.is_some()
    {
        return Err(DriftObservationReasonV1::EvidenceKindPlaneMismatch);
    }
    if input
        .source_label_ids
        .iter()
        .any(|label| label.trim().is_empty())
    {
        return Err(DriftObservationReasonV1::MissingEvidenceDigest);
    }
    if input.plane != DriftEvidencePlaneV1::QualifiedResidual
        && (!input.source_label_ids.is_empty()
            || input.qualification_report.is_some()
            || input.prediction_receipt_sha256.is_some())
    {
        return Err(DriftObservationReasonV1::EvidenceKindPlaneMismatch);
    }
    match input.detector {
        DriftDetectorKindV1::MissingnessRate => {}
        DriftDetectorKindV1::EventChange => {
            let Some(signal) = input.standardized_signal else {
                return Err(DriftObservationReasonV1::InvalidSignal);
            };
            if !signal.is_finite() {
                return Err(DriftObservationReasonV1::InvalidSignal);
            }
            if signal.abs() > engine.policy.maximum_abs_standardized_signal {
                return Err(DriftObservationReasonV1::SignalOutOfPolicy);
            }
        }
        DriftDetectorKindV1::Ewma | DriftDetectorKindV1::TwoSidedCusum => {
            let Some(signal) = input.standardized_signal else {
                return Err(DriftObservationReasonV1::InvalidSignal);
            };
            if !signal.is_finite() {
                return Err(DriftObservationReasonV1::InvalidSignal);
            }
            if signal.abs() > engine.policy.maximum_abs_standardized_signal {
                return Err(DriftObservationReasonV1::SignalOutOfPolicy);
            }
        }
    }
    Ok(())
}

/// Apply one observation to its detector channel.
///
/// Residual observations can enter only with a CurrentK1Eligible metrology
/// receipt. Unknown/blocked evidence returns a denial receipt without touching
/// detector state or the seen-evidence set.
pub fn observe_drift_channel_v1(
    engine: &mut DriftMonitorEngineV1,
    input: &DriftObservationInputV1,
) -> Result<DriftObservationReceiptV1, DriftMonitorErrorV1> {
    validate_engine(engine)?;
    require_text(&input.observation_id, "input.observation_id")?;
    require_text(&input.channel_id, "input.channel_id")?;
    require_text(&input.scope_id, "input.scope_id")?;

    let existing = engine.channels.get(&input.channel_id);
    let before = channel_state_digest(existing)?;
    if let Err(reason) = validate_input(engine, input) {
        let disposition = match reason {
            DriftObservationReasonV1::QualificationUnknown => {
                DriftObservationDispositionV1::HeldUnknown
            }
            DriftObservationReasonV1::QualificationBlocked => {
                DriftObservationDispositionV1::Blocked
            }
            _ => DriftObservationDispositionV1::Rejected,
        };
        return make_receipt(
            engine,
            input,
            disposition,
            Some(reason),
            DriftAlarmStateV1::InsufficientEvidence,
            existing.map_or(0, |state| state.observations),
            existing.map_or(0, |state| state.missing_observations),
            0.0,
            &before,
            &before,
        );
    }

    if engine
        .seen_evidence_receipt_sha256
        .contains(&input.evidence_receipt_sha256)
    {
        return make_receipt(
            engine,
            input,
            DriftObservationDispositionV1::Duplicate,
            Some(DriftObservationReasonV1::DuplicateEvidenceReceipt),
            DriftAlarmStateV1::InsufficientEvidence,
            existing.map_or(0, |state| state.observations),
            existing.map_or(0, |state| state.missing_observations),
            0.0,
            &before,
            &before,
        );
    }

    if let Some(state) = existing {
        if state.plane != input.plane
            || state.detector != input.detector
            || state.scope_id != input.scope_id
            || state.feature_schema_sha256 != input.feature_schema_sha256
            || state.baseline_sha256 != input.baseline_sha256
        {
            return make_receipt(
                engine,
                input,
                DriftObservationDispositionV1::Rejected,
                Some(DriftObservationReasonV1::ChannelIdentityChanged),
                DriftAlarmStateV1::InsufficientEvidence,
                state.observations,
                state.missing_observations,
                0.0,
                &before,
                &before,
            );
        }
        if input.event_time_unix_seconds < state.last_event_time_unix_seconds
            || input.observed_at_unix_seconds < state.last_observed_at_unix_seconds
        {
            return make_receipt(
                engine,
                input,
                DriftObservationDispositionV1::Rejected,
                Some(DriftObservationReasonV1::InvalidChronology),
                DriftAlarmStateV1::InsufficientEvidence,
                state.observations,
                state.missing_observations,
                0.0,
                &before,
                &before,
            );
        }
    }

    let mut next = existing.cloned().unwrap_or_else(|| DriftChannelStateV1 {
        channel_id: input.channel_id.clone(),
        plane: input.plane,
        detector: input.detector,
        scope_id: input.scope_id.clone(),
        feature_schema_sha256: input.feature_schema_sha256.clone(),
        baseline_sha256: input.baseline_sha256.clone(),
        observations: 0,
        missing_observations: 0,
        ewma: 0.0,
        positive_cusum: 0.0,
        negative_cusum: 0.0,
        last_event_time_unix_seconds: input.event_time_unix_seconds,
        last_observed_at_unix_seconds: input.observed_at_unix_seconds,
    });

    let signal = match input.detector {
        DriftDetectorKindV1::MissingnessRate => {
            if input.is_missing {
                1.0
            } else {
                0.0
            }
        }
        DriftDetectorKindV1::EventChange => input.standardized_signal.unwrap_or(0.0),
        DriftDetectorKindV1::Ewma | DriftDetectorKindV1::TwoSidedCusum => {
            input.standardized_signal.unwrap_or(0.0)
        }
    };
    if input.is_missing {
        next.missing_observations = next.missing_observations.saturating_add(1);
    }
    next.observations = next
        .observations
        .checked_add(1)
        .ok_or_else(|| DriftMonitorErrorV1("drift observation count overflow".into()))?;
    next.last_event_time_unix_seconds = input.event_time_unix_seconds;
    next.last_observed_at_unix_seconds = input.observed_at_unix_seconds;

    let statistic = match input.detector {
        DriftDetectorKindV1::Ewma => {
            let lambda = engine.policy.ewma_lambda;
            next.ewma = lambda * signal + (1.0 - lambda) * next.ewma;
            next.ewma.abs()
        }
        DriftDetectorKindV1::TwoSidedCusum => {
            let slack = engine.policy.cusum_slack;
            next.positive_cusum = (next.positive_cusum + signal - slack).max(0.0);
            next.negative_cusum = (next.negative_cusum - signal - slack).max(0.0);
            next.positive_cusum.max(next.negative_cusum)
        }
        DriftDetectorKindV1::MissingnessRate => {
            let rate = next.missing_observations as f64 / next.observations as f64;
            next.ewma =
                engine.policy.ewma_lambda * rate + (1.0 - engine.policy.ewma_lambda) * next.ewma;
            next.ewma
        }
        DriftDetectorKindV1::EventChange => signal.abs(),
    };
    if !statistic.is_finite() {
        return Err(DriftMonitorErrorV1(
            "drift detector produced a non-finite statistic".into(),
        ));
    }
    let alarm_state = if next.observations < engine.policy.min_observations {
        DriftAlarmStateV1::InsufficientEvidence
    } else if statistic >= engine.policy.alarm_threshold {
        DriftAlarmStateV1::Alert
    } else {
        DriftAlarmStateV1::InControl
    };
    let after = channel_state_digest(Some(&next))?;

    // Build the receipt before committing either detector state or evidence ID.
    let receipt = make_receipt(
        engine,
        input,
        DriftObservationDispositionV1::Updated,
        None,
        alarm_state,
        next.observations,
        next.missing_observations,
        statistic,
        &before,
        &after,
    )?;
    engine.channels.insert(input.channel_id.clone(), next);
    engine
        .seen_evidence_receipt_sha256
        .insert(input.evidence_receipt_sha256.clone());
    Ok(receipt)
}

/// Seal a deterministic multi-plane episode from immutable observation receipts.
/// The episode records association across channels; it does not declare cause.
pub fn seal_drift_episode_v1(
    episode_id: &str,
    window_start_unix_seconds: i64,
    window_end_unix_seconds: i64,
    receipts: &[DriftObservationReceiptV1],
) -> Result<DriftEpisodeReceiptV1, DriftMonitorErrorV1> {
    require_text(episode_id, "episode_id")?;
    if window_start_unix_seconds > window_end_unix_seconds {
        return Err(DriftMonitorErrorV1("episode window is reversed".into()));
    }
    let mut accepted = Vec::new();
    let mut planes = BTreeSet::new();
    let mut alerts = BTreeSet::new();
    let mut denied = 0_u64;
    let mut receipt_ids = BTreeSet::new();
    for receipt in receipts {
        if observation_receipt_digest(receipt)? != receipt.receipt_sha256 {
            return Err(DriftMonitorErrorV1(format!(
                "observation receipt {} digest mismatch",
                receipt.observation_id
            )));
        }
        if receipt.causal_attribution != CausalAttributionStateV1::NotEstablished {
            return Err(DriftMonitorErrorV1(
                "episode cannot claim causal attribution without an external review receipt".into(),
            ));
        }
        if receipt.event_time_unix_seconds < window_start_unix_seconds
            || receipt.event_time_unix_seconds > window_end_unix_seconds
        {
            return Err(DriftMonitorErrorV1(format!(
                "observation receipt {} is outside episode window",
                receipt.observation_id
            )));
        }
        if !receipt_ids.insert(receipt.receipt_sha256.as_str()) {
            return Err(DriftMonitorErrorV1(
                "duplicate observation receipt in episode".into(),
            ));
        }
        accepted.push(receipt.receipt_sha256.clone());
        if receipt.disposition == DriftObservationDispositionV1::Updated {
            planes.insert(receipt.plane);
            if receipt.alarm_state == DriftAlarmStateV1::Alert {
                alerts.insert(receipt.channel_id.clone());
            }
        } else {
            denied = denied.saturating_add(1);
        }
    }
    accepted.sort();
    let mut episode = DriftEpisodeReceiptV1 {
        schema: DRIFT_EPISODE_SCHEMA_V1.into(),
        episode_id: episode_id.into(),
        window_start_unix_seconds,
        window_end_unix_seconds,
        observation_receipt_sha256s: accepted,
        active_planes: planes.into_iter().collect(),
        alert_channel_ids: alerts.into_iter().collect(),
        denied_observation_count: denied,
        causal_attribution: CausalAttributionStateV1::NotEstablished,
        episode_sha256: String::new(),
    };
    let mut canonical = episode.clone();
    canonical.episode_sha256.clear();
    episode.episode_sha256 = json_sha256(&canonical)?;
    Ok(episode)
}

fn shared_resource_graph_digest(
    graph: &SharedResourceGraphV1,
) -> Result<String, DriftMonitorErrorV1> {
    let mut canonical = graph.clone();
    canonical.graph_sha256.clear();
    canonical.bindings.sort_by(|a, b| {
        (&a.context_id, &a.resource_kind, &a.resource_id).cmp(&(
            &b.context_id,
            &b.resource_kind,
            &b.resource_id,
        ))
    });
    json_sha256(&canonical)
}

pub fn seal_shared_resource_graph_v1(
    graph: &SharedResourceGraphV1,
) -> Result<String, DriftMonitorErrorV1> {
    shared_resource_graph_digest(graph)
}

/// Produce an evidence-backed common-resource association candidate.
/// Coincident alerts without a declared shared-resource edge never produce one.
pub fn assess_declared_common_resource_association_v1(
    graph: &SharedResourceGraphV1,
    alerts: &[DriftAlertContextV1],
) -> Result<CommonCauseAssessmentV1, DriftMonitorErrorV1> {
    if graph.schema != SHARED_RESOURCE_GRAPH_SCHEMA_V1
        || !valid_sha256(&graph.graph_sha256)
        || shared_resource_graph_digest(graph)? != graph.graph_sha256
    {
        return Err(DriftMonitorErrorV1(
            "shared-resource graph digest or schema invalid".into(),
        ));
    }
    require_text(&graph.graph_id, "graph.graph_id")?;
    let mut alert_ids = BTreeSet::new();
    let mut contexts = BTreeSet::new();
    for alert in alerts {
        require_text(&alert.alert_id, "alert.alert_id")?;
        require_text(&alert.context_id, "alert.context_id")?;
        if !valid_sha256(&alert.observation_receipt_sha256) {
            return Err(DriftMonitorErrorV1(
                "alert observation receipt digest invalid".into(),
            ));
        }
        if !alert_ids.insert(alert.alert_id.as_str()) {
            return Err(DriftMonitorErrorV1("duplicate alert ID".into()));
        }
        contexts.insert(alert.context_id.as_str());
    }

    let mut bindings: BTreeMap<&str, BTreeSet<(&str, &str)>> = BTreeMap::new();
    let mut unique_bindings = BTreeSet::new();
    for binding in &graph.bindings {
        require_text(&binding.context_id, "binding.context_id")?;
        require_text(&binding.resource_id, "binding.resource_id")?;
        require_text(&binding.resource_kind, "binding.resource_kind")?;
        if !unique_bindings.insert((
            binding.context_id.as_str(),
            binding.resource_id.as_str(),
            binding.resource_kind.as_str(),
        )) {
            return Err(DriftMonitorErrorV1(
                "duplicate shared-resource binding".into(),
            ));
        }
        bindings
            .entry(binding.context_id.as_str())
            .or_default()
            .insert((binding.resource_id.as_str(), binding.resource_kind.as_str()));
    }

    let mut shared = BTreeSet::new();
    let mut reason_codes = Vec::new();
    let unrepresented_contexts: Vec<&str> = contexts
        .iter()
        .filter(|context| !bindings.contains_key(**context))
        .copied()
        .collect();
    let state = if !graph.inventory_complete || !unrepresented_contexts.is_empty() {
        reason_codes.push("shared_resource_inventory_incomplete".into());
        CommonCauseAssessmentStateV1::Unknown
    } else if contexts.len() < 2 {
        reason_codes.push("fewer_than_two_distinct_contexts".into());
        CommonCauseAssessmentStateV1::NoDeclaredSharedResource
    } else {
        let mut resource_counts: BTreeMap<(&str, &str), usize> = BTreeMap::new();
        for context in &contexts {
            for resource in bindings.get(context).into_iter().flatten() {
                *resource_counts.entry(*resource).or_default() += 1;
            }
        }
        shared.extend(
            resource_counts
                .into_iter()
                .filter(|(_, count)| *count >= 2)
                .map(|((resource, _kind), _)| resource.to_string()),
        );
        if shared.is_empty() {
            reason_codes.push("no_declared_shared_resource".into());
            CommonCauseAssessmentStateV1::NoDeclaredSharedResource
        } else {
            reason_codes.push("shared_resource_association_only_not_causality".into());
            CommonCauseAssessmentStateV1::SharedResourceAssociationCandidate
        }
    };

    let mut assessment = CommonCauseAssessmentV1 {
        schema: COMMON_CAUSE_ASSESSMENT_SCHEMA_V1.into(),
        graph_id: graph.graph_id.clone(),
        graph_sha256: graph.graph_sha256.clone(),
        state,
        alert_ids: alert_ids.into_iter().map(str::to_string).collect(),
        shared_resource_ids: shared.into_iter().collect(),
        reason_codes,
        causal_attribution: CausalAttributionStateV1::NotEstablished,
        assessment_sha256: String::new(),
    };
    let mut canonical = assessment.clone();
    canonical.assessment_sha256.clear();
    assessment.assessment_sha256 = json_sha256(&canonical)?;
    Ok(assessment)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn policy() -> DriftMonitorPolicyV1 {
        DriftMonitorPolicyV1 {
            schema: DRIFT_POLICY_SCHEMA_V1.into(),
            policy_id: "drift-policy-v1".into(),
            revision: 1,
            ewma_lambda: 0.25,
            cusum_slack: 0.25,
            alarm_threshold: 1.5,
            min_observations: 3,
            maximum_abs_standardized_signal: 10.0,
            maximum_observation_delay_seconds: 100,
        }
    }

    fn engine() -> DriftMonitorEngineV1 {
        create_drift_monitor_engine_v1("drift-001", "fab-clock", policy()).unwrap()
    }

    fn eligible_calibration_report(
        event_id: &str,
        evidence: char,
    ) -> CalibrationRequalificationReportV1 {
        let frontier = CalibrationRequalificationFrontierReportV1 {
            frontier_input_sha256: digest(evidence),
            status: MeasurementQualificationStatusV1::Qualified,
            downstream_consumption_allowed_at_frontier: true,
            reason_codes: Vec::new(),
            potentially_applicable_event_ids: Vec::new(),
            applied_event_ids: Vec::new(),
        };
        let mut report = CalibrationRequalificationReportV1 {
            schema: CALIBRATION_REQUALIFICATION_REPORT_SCHEMA_V1.into(),
            input_id: format!("cal-{event_id}"),
            measurement_event_id: event_id.into(),
            calibration_certificate_id: "cal-cert-1".into(),
            original_k0_qualification_sha256: digest('1'),
            owner_event_inventory_sha256: digest('2'),
            canonical_input_sha256: digest('3'),
            k0_unix_seconds: 99,
            k1_unix_seconds: 101,
            as_known_at_k0: frontier.clone(),
            current_at_k1: frontier,
            source_authenticity_proven: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        let mut canonical = report.clone();
        canonical.report_sha256.clear();
        report.report_sha256 = json_sha256(&canonical).unwrap();
        report
    }

    fn eligible_label_report(
        label_id: &str,
        evidence: char,
    ) -> LabelConsumptionEligibilityReportV1 {
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: LABEL_CONSUMPTION_REPORT_SCHEMA_V1.into(),
            label_id: label_id.into(),
            label_revision: 1,
            measurement_event_id: format!("event-{label_id}"),
            calibration_certificate_id: "calibration-1".into(),
            frontier: LabelEvidenceFrontierV1::CurrentAtK1,
            eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            downstream_consumption_allowed: true,
            reasons: Vec::new(),
            label_sha256: digest(evidence),
            measurement_sha256: digest('d'),
            qualified_corrected_value: 1.0,
            qualified_standard_uncertainty_upper_bound: Some(0.1),
            qualification_report_sha256: digest('e'),
            requalification_report_sha256: digest('f'),
            requalification_frontier_sha256: digest('a'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        let mut canonical = report.clone();
        canonical.report_sha256.clear();
        report.report_sha256 = json_sha256(&canonical).unwrap();
        report
    }

    fn input(
        id: &str,
        channel: &str,
        plane: DriftEvidencePlaneV1,
        detector: DriftDetectorKindV1,
        evidence_kind: DriftEvidenceKindV1,
        qualification: DriftQualificationV1,
        signal: Option<f64>,
        time: i64,
        receipt: char,
    ) -> DriftObservationInputV1 {
        let qualification_report = if plane == DriftEvidencePlaneV1::QualifiedResidual {
            Some(eligible_label_report(&format!("label-{id}"), receipt))
        } else {
            None
        };
        let calibration_report = if plane == DriftEvidencePlaneV1::MeasurementSystem {
            Some(eligible_calibration_report(&format!("event-{id}"), receipt))
        } else {
            None
        };
        let evidence_receipt_sha256 = qualification_report
            .as_ref()
            .map(|report| report.report_sha256.clone())
            .or_else(|| {
                calibration_report
                    .as_ref()
                    .map(|report| report.report_sha256.clone())
            })
            .unwrap_or_else(|| digest(receipt));
        let measurement_event_id = qualification_report
            .as_ref()
            .map(|report| report.measurement_event_id.clone())
            .or_else(|| {
                calibration_report
                    .as_ref()
                    .map(|report| report.measurement_event_id.clone())
            });
        DriftObservationInputV1 {
            schema: DRIFT_OBSERVATION_SCHEMA_V1.into(),
            observation_id: id.into(),
            channel_id: channel.into(),
            plane,
            detector,
            scope_id: "tool-1/chamber-1".into(),
            event_time_unix_seconds: time,
            observed_at_unix_seconds: time + 1,
            knowledge_frontier_unix_seconds: time + 2,
            clock_domain_id: "fab-clock".into(),
            feature_schema_sha256: digest('a'),
            baseline_sha256: digest('b'),
            evidence_kind,
            qualification,
            measurement_event_id,
            evidence_receipt_sha256,
            qualification_report,
            calibration_report,
            prediction_receipt_sha256: if plane == DriftEvidencePlaneV1::QualifiedResidual {
                Some(digest('p'))
            } else {
                None
            },
            source_label_ids: if plane == DriftEvidencePlaneV1::QualifiedResidual {
                vec![format!("label-{id}")]
            } else {
                Vec::new()
            },
            standardized_signal: signal,
            is_missing: false,
        }
    }

    #[test]
    fn feature_channel_updates_and_emits_alert_after_minimum_support() {
        let mut engine = engine();
        let base = input(
            "x1",
            "feature-pressure",
            DriftEvidencePlaneV1::InputFeature,
            DriftDetectorKindV1::TwoSidedCusum,
            DriftEvidenceKindV1::FeatureWindow,
            DriftQualificationV1::NotApplicable,
            Some(2.0),
            100,
            'c',
        );
        let mut final_receipt = None;
        for index in 0..3 {
            let mut observation = base.clone();
            observation.observation_id = format!("x{index}");
            observation.evidence_receipt_sha256 = digest(char::from(b'c' + index as u8));
            observation.event_time_unix_seconds += index;
            observation.observed_at_unix_seconds += index;
            observation.knowledge_frontier_unix_seconds += index;
            final_receipt = Some(observe_drift_channel_v1(&mut engine, &observation).unwrap());
        }
        let receipt = final_receipt.unwrap();
        assert_eq!(receipt.alarm_state, DriftAlarmStateV1::Alert);
        assert_eq!(receipt.disposition, DriftObservationDispositionV1::Updated);
    }

    #[test]
    fn unknown_and_blocked_residuals_do_not_mutate_detector_state() {
        let mut engine = engine();
        let mut clean = input(
            "r1",
            "residual-tool-1",
            DriftEvidencePlaneV1::QualifiedResidual,
            DriftDetectorKindV1::TwoSidedCusum,
            DriftEvidenceKindV1::QualifiedMetrologyLabel,
            DriftQualificationV1::CurrentK1Eligible,
            Some(0.2),
            100,
            'c',
        );
        clean.evidence_receipt_sha256 = clean
            .qualification_report
            .as_ref()
            .unwrap()
            .report_sha256
            .clone();
        observe_drift_channel_v1(&mut engine, &clean).unwrap();
        let before = channel_state_digest(engine.channels.get("residual-tool-1")).unwrap();
        let mut unknown = clean.clone();
        unknown.observation_id = "r2".into();
        unknown.qualification = DriftQualificationV1::Unknown;
        unknown.evidence_receipt_sha256 = digest('d');
        let receipt = observe_drift_channel_v1(&mut engine, &unknown).unwrap();
        assert_eq!(
            receipt.disposition,
            DriftObservationDispositionV1::HeldUnknown
        );
        assert_eq!(
            channel_state_digest(engine.channels.get("residual-tool-1")).unwrap(),
            before
        );

        let mut blocked = clean;
        blocked.observation_id = "r3".into();
        blocked.qualification = DriftQualificationV1::Blocked;
        blocked.evidence_receipt_sha256 = digest('e');
        let receipt = observe_drift_channel_v1(&mut engine, &blocked).unwrap();
        assert_eq!(receipt.disposition, DriftObservationDispositionV1::Blocked);
        assert_eq!(
            channel_state_digest(engine.channels.get("residual-tool-1")).unwrap(),
            before
        );
    }

    #[test]
    fn tampered_k1_label_report_is_rejected_from_residual_monitor() {
        let mut engine = engine();
        let mut observation = input(
            "r1",
            "residual",
            DriftEvidencePlaneV1::QualifiedResidual,
            DriftDetectorKindV1::TwoSidedCusum,
            DriftEvidenceKindV1::QualifiedMetrologyLabel,
            DriftQualificationV1::CurrentK1Eligible,
            Some(1.0),
            100,
            'c',
        );
        let report = observation.qualification_report.as_mut().unwrap();
        report.label_revision = 2;
        let receipt = observe_drift_channel_v1(&mut engine, &observation).unwrap();
        assert_eq!(receipt.disposition, DriftObservationDispositionV1::Rejected);
        assert_eq!(receipt.alarm_state, DriftAlarmStateV1::InsufficientEvidence);
        assert!(engine.channels.is_empty());
    }

    #[test]
    fn non_k1_residual_evidence_is_rejected() {
        let mut engine = engine();
        let mut observation = input(
            "r1",
            "residual",
            DriftEvidencePlaneV1::QualifiedResidual,
            DriftDetectorKindV1::TwoSidedCusum,
            DriftEvidenceKindV1::QualifiedMetrologyLabel,
            DriftQualificationV1::NotApplicable,
            Some(1.0),
            100,
            'c',
        );
        observation.qualification = DriftQualificationV1::NotApplicable;
        let receipt = observe_drift_channel_v1(&mut engine, &observation).unwrap();
        assert_eq!(receipt.disposition, DriftObservationDispositionV1::Rejected);
        assert_eq!(engine.channels.len(), 0);
    }

    #[test]
    fn duplicate_evidence_receipt_is_idempotent() {
        let mut engine = engine();
        let first = input(
            "x1",
            "feature",
            DriftEvidencePlaneV1::InputFeature,
            DriftDetectorKindV1::Ewma,
            DriftEvidenceKindV1::FeatureWindow,
            DriftQualificationV1::NotApplicable,
            Some(1.0),
            100,
            'c',
        );
        observe_drift_channel_v1(&mut engine, &first).unwrap();
        let before = channel_state_digest(engine.channels.get("feature")).unwrap();
        let mut duplicate = first;
        duplicate.observation_id = "x2".into();
        let receipt = observe_drift_channel_v1(&mut engine, &duplicate).unwrap();
        assert_eq!(
            receipt.disposition,
            DriftObservationDispositionV1::Duplicate
        );
        assert_eq!(
            channel_state_digest(engine.channels.get("feature")).unwrap(),
            before
        );
    }

    #[test]
    fn delayed_or_out_of_order_observation_is_rejected_without_state_mutation() {
        let mut engine = engine();
        let first = input(
            "x1",
            "feature",
            DriftEvidencePlaneV1::InputFeature,
            DriftDetectorKindV1::Ewma,
            DriftEvidenceKindV1::FeatureWindow,
            DriftQualificationV1::NotApplicable,
            Some(0.1),
            100,
            'c',
        );
        observe_drift_channel_v1(&mut engine, &first).unwrap();
        let before = channel_state_digest(engine.channels.get("feature")).unwrap();
        let mut out_of_order = first.clone();
        out_of_order.observation_id = "x2".into();
        out_of_order.event_time_unix_seconds = 99;
        out_of_order.observed_at_unix_seconds = 101;
        out_of_order.knowledge_frontier_unix_seconds = 102;
        out_of_order.evidence_receipt_sha256 = digest('d');
        let receipt = observe_drift_channel_v1(&mut engine, &out_of_order).unwrap();
        assert_eq!(receipt.disposition, DriftObservationDispositionV1::Rejected);
        assert_eq!(
            channel_state_digest(engine.channels.get("feature")).unwrap(),
            before
        );
    }

    #[test]
    fn feature_schema_or_baseline_change_requires_a_new_channel() {
        let mut engine = engine();
        let first = input(
            "x1",
            "feature",
            DriftEvidencePlaneV1::InputFeature,
            DriftDetectorKindV1::Ewma,
            DriftEvidenceKindV1::FeatureWindow,
            DriftQualificationV1::NotApplicable,
            Some(0.1),
            100,
            'c',
        );
        observe_drift_channel_v1(&mut engine, &first).unwrap();
        let mut changed = first;
        changed.observation_id = "x2".into();
        changed.evidence_receipt_sha256 = digest('d');
        changed.baseline_sha256 = digest('f');
        let receipt = observe_drift_channel_v1(&mut engine, &changed).unwrap();
        assert_eq!(receipt.disposition, DriftObservationDispositionV1::Rejected);
        assert_eq!(
            receipt.reason,
            Some(DriftObservationReasonV1::ChannelIdentityChanged)
        );
    }

    #[test]
    fn missingness_monitor_tracks_dropout_rate() {
        let mut engine = engine();
        let mut last = None;
        for index in 0..4 {
            let mut observation = input(
                &format!("m{index}"),
                "sensor-missing",
                DriftEvidencePlaneV1::InputFeature,
                DriftDetectorKindV1::MissingnessRate,
                DriftEvidenceKindV1::FeatureWindow,
                DriftQualificationV1::NotApplicable,
                None,
                100 + index,
                char::from(b'c' + index as u8),
            );
            observation.is_missing = index >= 2;
            last = Some(observe_drift_channel_v1(&mut engine, &observation).unwrap());
        }
        assert!(last.unwrap().statistic > 0.0);
    }

    #[test]
    fn process_and_calibration_events_remain_separate_planes() {
        let mut engine = engine();
        let process = input(
            "p1",
            "recipe-event",
            DriftEvidencePlaneV1::ProcessContext,
            DriftDetectorKindV1::EventChange,
            DriftEvidenceKindV1::ProcessEvent,
            DriftQualificationV1::NotApplicable,
            Some(1.0),
            100,
            'c',
        );
        let measurement = input(
            "m1",
            "calibration-event",
            DriftEvidencePlaneV1::MeasurementSystem,
            DriftDetectorKindV1::EventChange,
            DriftEvidenceKindV1::CalibrationQualification,
            DriftQualificationV1::QualifiedSystemEvidence,
            Some(1.0),
            100,
            'd',
        );
        let p = observe_drift_channel_v1(&mut engine, &process).unwrap();
        let m = observe_drift_channel_v1(&mut engine, &measurement).unwrap();
        assert_ne!(p.plane, m.plane);
        assert_ne!(p.channel_id, m.channel_id);
    }

    #[test]
    fn episode_digest_is_insertion_order_invariant_and_never_claims_cause() {
        let mut engine = engine();
        let mut receipts = Vec::new();
        for (id, channel, plane, kind, qualification, signal, evidence) in [
            (
                "x1",
                "feature",
                DriftEvidencePlaneV1::InputFeature,
                DriftEvidenceKindV1::FeatureWindow,
                DriftQualificationV1::NotApplicable,
                0.5,
                'c',
            ),
            (
                "r1",
                "residual",
                DriftEvidencePlaneV1::QualifiedResidual,
                DriftEvidenceKindV1::QualifiedMetrologyLabel,
                DriftQualificationV1::CurrentK1Eligible,
                0.7,
                'd',
            ),
        ] {
            let detector = DriftDetectorKindV1::Ewma;
            let observation = input(
                id,
                channel,
                plane,
                detector,
                kind,
                qualification,
                Some(signal),
                100,
                evidence,
            );
            receipts.push(observe_drift_channel_v1(&mut engine, &observation).unwrap());
        }
        let first = seal_drift_episode_v1("episode-1", 100, 102, &receipts).unwrap();
        receipts.reverse();
        let second = seal_drift_episode_v1("episode-1", 100, 102, &receipts).unwrap();
        assert_eq!(first.episode_sha256, second.episode_sha256);
        assert_eq!(
            first.causal_attribution,
            CausalAttributionStateV1::NotEstablished
        );
        assert_eq!(first.active_planes.len(), 2);
    }

    #[test]
    fn coincidence_without_declared_shared_resource_is_not_common_cause() {
        let mut graph = SharedResourceGraphV1 {
            schema: SHARED_RESOURCE_GRAPH_SCHEMA_V1.into(),
            graph_id: "resources-1".into(),
            inventory_complete: true,
            bindings: vec![
                SharedResourceBindingV1 {
                    context_id: "tool-1/ch-1".into(),
                    resource_id: "reference-1".into(),
                    resource_kind: "calibration_reference".into(),
                },
                SharedResourceBindingV1 {
                    context_id: "tool-2/ch-2".into(),
                    resource_id: "reference-2".into(),
                    resource_kind: "calibration_reference".into(),
                },
            ],
            graph_sha256: String::new(),
        };
        graph.graph_sha256 = seal_shared_resource_graph_v1(&graph).unwrap();
        let alerts = vec![
            DriftAlertContextV1 {
                alert_id: "a1".into(),
                context_id: "tool-1/ch-1".into(),
                observation_receipt_sha256: digest('a'),
            },
            DriftAlertContextV1 {
                alert_id: "a2".into(),
                context_id: "tool-2/ch-2".into(),
                observation_receipt_sha256: digest('b'),
            },
        ];
        let report = assess_declared_common_resource_association_v1(&graph, &alerts).unwrap();
        assert_eq!(
            report.state,
            CommonCauseAssessmentStateV1::NoDeclaredSharedResource
        );
        assert_eq!(
            report.causal_attribution,
            CausalAttributionStateV1::NotEstablished
        );
    }

    #[test]
    fn unrepresented_context_in_claimed_complete_inventory_returns_unknown() {
        let mut graph = SharedResourceGraphV1 {
            schema: SHARED_RESOURCE_GRAPH_SCHEMA_V1.into(),
            graph_id: "resources-1".into(),
            inventory_complete: true,
            bindings: vec![SharedResourceBindingV1 {
                context_id: "tool-1/ch-1".into(),
                resource_id: "cal-ref-1".into(),
                resource_kind: "calibration_reference".into(),
            }],
            graph_sha256: String::new(),
        };
        graph.graph_sha256 = seal_shared_resource_graph_v1(&graph).unwrap();
        let alerts = vec![
            DriftAlertContextV1 {
                alert_id: "a1".into(),
                context_id: "tool-1/ch-1".into(),
                observation_receipt_sha256: digest('a'),
            },
            DriftAlertContextV1 {
                alert_id: "a2".into(),
                context_id: "tool-2/ch-2".into(),
                observation_receipt_sha256: digest('b'),
            },
        ];
        let report = assess_declared_common_resource_association_v1(&graph, &alerts).unwrap();
        assert_eq!(report.state, CommonCauseAssessmentStateV1::Unknown);
        assert!(report.shared_resource_ids.is_empty());
    }

    #[test]
    fn calibration_unknown_does_not_update_measurement_system_drift_channel() {
        let mut engine = engine();
        let mut observation = input(
            "m1",
            "calibration",
            DriftEvidencePlaneV1::MeasurementSystem,
            DriftDetectorKindV1::EventChange,
            DriftEvidenceKindV1::CalibrationQualification,
            DriftQualificationV1::Unknown,
            Some(1.0),
            100,
            'd',
        );
        let report = observation.calibration_report.as_mut().unwrap();
        report.current_at_k1.status = MeasurementQualificationStatusV1::Unknown;
        report.report_sha256 = calibration_report_digest(report).unwrap();
        observation.evidence_receipt_sha256 = observation
            .calibration_report
            .as_ref()
            .unwrap()
            .report_sha256
            .clone();
        let receipt = observe_drift_channel_v1(&mut engine, &observation).unwrap();
        assert_eq!(
            receipt.disposition,
            DriftObservationDispositionV1::HeldUnknown
        );
        assert!(engine.channels.is_empty());
    }

    #[test]
    fn incomplete_dependency_inventory_returns_unknown_association() {
        let mut graph = SharedResourceGraphV1 {
            schema: SHARED_RESOURCE_GRAPH_SCHEMA_V1.into(),
            graph_id: "resources-1".into(),
            inventory_complete: false,
            bindings: vec![SharedResourceBindingV1 {
                context_id: "tool-1/ch-1".into(),
                resource_id: "cal-ref-1".into(),
                resource_kind: "calibration_reference".into(),
            }],
            graph_sha256: String::new(),
        };
        graph.graph_sha256 = seal_shared_resource_graph_v1(&graph).unwrap();
        let alerts = vec![
            DriftAlertContextV1 {
                alert_id: "a1".into(),
                context_id: "tool-1/ch-1".into(),
                observation_receipt_sha256: digest('a'),
            },
            DriftAlertContextV1 {
                alert_id: "a2".into(),
                context_id: "tool-2/ch-2".into(),
                observation_receipt_sha256: digest('b'),
            },
        ];
        let report = assess_declared_common_resource_association_v1(&graph, &alerts).unwrap();
        assert_eq!(report.state, CommonCauseAssessmentStateV1::Unknown);
        assert!(report.shared_resource_ids.is_empty());
    }

    #[test]
    fn declared_shared_reference_produces_association_not_causal_finding() {
        let mut graph = SharedResourceGraphV1 {
            schema: SHARED_RESOURCE_GRAPH_SCHEMA_V1.into(),
            graph_id: "resources-1".into(),
            inventory_complete: true,
            bindings: vec![
                SharedResourceBindingV1 {
                    context_id: "tool-1/ch-1".into(),
                    resource_id: "cal-ref-1".into(),
                    resource_kind: "calibration_reference".into(),
                },
                SharedResourceBindingV1 {
                    context_id: "tool-2/ch-2".into(),
                    resource_id: "cal-ref-1".into(),
                    resource_kind: "calibration_reference".into(),
                },
            ],
            graph_sha256: String::new(),
        };
        graph.graph_sha256 = seal_shared_resource_graph_v1(&graph).unwrap();
        let alerts = vec![
            DriftAlertContextV1 {
                alert_id: "a1".into(),
                context_id: "tool-1/ch-1".into(),
                observation_receipt_sha256: digest('a'),
            },
            DriftAlertContextV1 {
                alert_id: "a2".into(),
                context_id: "tool-2/ch-2".into(),
                observation_receipt_sha256: digest('b'),
            },
        ];
        let report = assess_declared_common_resource_association_v1(&graph, &alerts).unwrap();
        assert_eq!(
            report.state,
            CommonCauseAssessmentStateV1::SharedResourceAssociationCandidate
        );
        assert_eq!(report.shared_resource_ids, vec!["cal-ref-1"]);
        assert_eq!(
            report.causal_attribution,
            CausalAttributionStateV1::NotEstablished
        );
    }

    #[test]
    fn tampered_observation_receipt_cannot_be_sealed_into_episode() {
        let mut engine = engine();
        let observation = input(
            "x1",
            "feature",
            DriftEvidencePlaneV1::InputFeature,
            DriftDetectorKindV1::Ewma,
            DriftEvidenceKindV1::FeatureWindow,
            DriftQualificationV1::NotApplicable,
            Some(0.5),
            100,
            'c',
        );
        let mut receipt = observe_drift_channel_v1(&mut engine, &observation).unwrap();
        receipt.statistic = 999.0;
        assert!(seal_drift_episode_v1("episode-1", 100, 102, &[receipt]).is_err());
    }
}
