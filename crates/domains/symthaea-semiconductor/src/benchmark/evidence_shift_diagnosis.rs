// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-evidence sequential shift monitoring for semiconductor VM.
//!
//! This module detects channel changes and emits conservative diagnostic
//! hypotheses. It does not prove physical root cause, update a VM, or control
//! manufacturing equipment. Qualified physical residuals are required before
//! the product/VM residual channel may advance.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
};

pub const EVIDENCE_SHIFT_POLICY_SCHEMA_V1: &str =
    "luminous.semi-bench.evidence-shift-policy.v1";
pub const EVIDENCE_SHIFT_OBSERVATION_SCHEMA_V1: &str =
    "luminous.semi-bench.evidence-shift-observation.v1";
pub const EVIDENCE_SHIFT_ENGINE_SCHEMA_V1: &str =
    "luminous.semi-bench.evidence-shift-engine.v1";
pub const EVIDENCE_SHIFT_ASSESSMENT_SCHEMA_V1: &str =
    "luminous.semi-bench.evidence-shift-assessment.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShiftSignalChannelV1 {
    VmResidual,
    ReferenceStandardResidual,
    SensorObservabilityScore,
    ProcessEventMarker,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShiftDiagnosticStateV1 {
    Nominal,
    ProcessShiftSuspected,
    MeasurementSystemShiftSuspected,
    SensorObservabilityShift,
    CommonCauseOrMixedShift,
    Unresolved,
    InsufficientEvidence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShiftAdaptationRecommendationV1 {
    Continue,
    Slow,
    Hold,
    RequestMetrology,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShiftReasonCodeV1 {
    BaselineDigestMismatch,
    PolicyDigestMismatch,
    ClockDomainMismatch,
    EvidenceNotAvailableAtObservation,
    EventSequenceNotContiguous,
    EventTimeRegression,
    DuplicateObservation,
    MissingQualifiedLabelReport,
    LabelNotCurrentK1,
    LabelNotEligible,
    LabelReportDigestMismatch,
    LabelPayloadDigestInvalid,
    LabelRevisionInvalid,
    ReferenceEvidenceMissing,
    ReferenceEvidenceNotQualified,
    SensorProjectionDigestMissing,
    ProcessEventEvidenceMissing,
    NonFiniteObservation,
    MissingDetectorChannel,
    InsufficientIndependentEvidence,
    ResidualAndReferenceBothShifted,
    ResidualShiftWithoutIdentifyingEvidence,
    ProcessMarkerWithoutMeasuredShift,
    LateCalibrationRevocationNotReplayed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShiftChannelPolicyV1 {
    /// Mean of the frozen nominal reference period.
    pub baseline_mean: f64,
    /// Standard deviation of the frozen nominal reference period.
    pub baseline_standard_deviation: f64,
    /// CUSUM reference allowance in standardized units.
    pub cusum_allowance: f64,
    /// Two-sided CUSUM decision threshold.
    pub cusum_threshold: f64,
    /// EWMA smoothing factor.
    pub ewma_alpha: f64,
    /// Absolute EWMA alert threshold in standardized units.
    pub ewma_threshold: f64,
    /// A channel is considered stable only after this many observations.
    pub minimum_observations: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceShiftPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    pub clock_domain_id: String,
    pub nominal_baseline_manifest_sha256: String,
    pub process_event_context_window_seconds: u64,
    pub maximum_absolute_standardized_value: f64,
    pub channels: BTreeMap<ShiftSignalChannelV1, ShiftChannelPolicyV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShiftChannelStateV1 {
    pub observations: u64,
    pub positive_cusum: f64,
    pub negative_cusum: f64,
    pub ewma: f64,
    pub last_event_time_unix_seconds: i64,
    pub last_observation_id: String,
    pub alert: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProcessEventMarkerV1 {
    pub event_id: String,
    pub event_kind: String,
    pub event_time_unix_seconds: i64,
    pub source_evidence_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceShiftObservationV1 {
    pub schema: String,
    pub observation_id: String,
    /// Contiguous source sequence. The engine never sorts events by arrival time.
    pub stream_sequence: u64,
    pub scope_id: String,
    pub event_time_unix_seconds: i64,
    pub available_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub nominal_baseline_manifest_sha256: String,
    pub channel: ShiftSignalChannelV1,
    /// For reference-standard and sensor channels; VM residual is computed
    /// from the qualified K1 corrected value minus prediction_value.
    pub observation_value: Option<f64>,
    pub prediction_value: Option<f64>,
    pub label_report: Option<LabelConsumptionEligibilityReportV1>,
    pub reference_standard_id: Option<String>,
    pub reference_evidence_sha256: Option<String>,
    pub reference_evidence_qualified: bool,
    pub sensor_projection_sha256: Option<String>,
    pub process_event_marker: Option<ProcessEventMarkerV1>,
    pub source_evidence_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShiftChannelAssessmentV1 {
    pub channel: ShiftSignalChannelV1,
    pub observations: u64,
    pub positive_cusum: f64,
    pub negative_cusum: f64,
    pub ewma: f64,
    pub alert: bool,
    pub stable_reference_window: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceShiftAssessmentV1 {
    pub schema: String,
    pub engine_id: String,
    pub scope_id: String,
    pub observation_id: String,
    pub stream_sequence: u64,
    pub diagnostic_state: ShiftDiagnosticStateV1,
    pub recommendation: ShiftAdaptationRecommendationV1,
    pub channel_assessments: Vec<ShiftChannelAssessmentV1>,
    pub supporting_channels: Vec<ShiftSignalChannelV1>,
    pub contradicting_channels: Vec<ShiftSignalChannelV1>,
    pub process_event_id: Option<String>,
    pub qualified_source_label_ids: Vec<String>,
    pub reason_codes: Vec<ShiftReasonCodeV1>,
    pub nominal_baseline_manifest_sha256: String,
    pub policy_sha256: String,
    pub previous_assessment_sha256: Option<String>,
    pub assessment_sha256: String,
    pub diagnostic_is_hypothesis_not_root_cause_proof: bool,
    pub process_control_authority: bool,
    pub automatic_retraining_executed: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceShiftMonitorV1 {
    pub schema: String,
    pub engine_id: String,
    pub policy: EvidenceShiftPolicyV1,
    pub policy_sha256: String,
    pub last_stream_sequence: u64,
    pub last_scope_event_times: BTreeMap<String, i64>,
    pub channel_states: BTreeMap<(String, ShiftSignalChannelV1), ShiftChannelStateV1>,
    pub process_markers: BTreeMap<String, ProcessEventMarkerV1>,
    pub seen_observation_ids: BTreeSet<String>,
    pub assessments_by_observation: BTreeMap<String, EvidenceShiftAssessmentV1>,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct EvidenceShiftErrorV1(pub String);

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, EvidenceShiftErrorV1> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| EvidenceShiftErrorV1(format!("JSON digest failed: {error}")))?;
    Ok(sha256_hex(&bytes))
}

fn report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, EvidenceShiftErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn assessment_digest(
    assessment: &EvidenceShiftAssessmentV1,
) -> Result<String, EvidenceShiftErrorV1> {
    let mut canonical = assessment.clone();
    canonical.assessment_sha256.clear();
    json_sha256(&canonical)
}

fn reference_state_scope(reference_standard_id: &str) -> String {
    format!("reference-standard:{reference_standard_id}")
}

fn observation_state_scope(
    observation: &EvidenceShiftObservationV1,
) -> Result<String, EvidenceShiftErrorV1> {
    if observation.channel == ShiftSignalChannelV1::ReferenceStandardResidual {
        return observation
            .reference_standard_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .map(reference_state_scope)
            .ok_or_else(|| EvidenceShiftErrorV1(
                "reference-standard observation requires an explicit standard identity".into(),
            ));
    }
    Ok(observation.scope_id.clone())
}

fn validate_policy(policy: &EvidenceShiftPolicyV1) -> Result<(), EvidenceShiftErrorV1> {
    if policy.schema != EVIDENCE_SHIFT_POLICY_SCHEMA_V1 {
        return Err(EvidenceShiftErrorV1("unsupported shift policy schema".into()));
    }
    if policy.policy_id.trim().is_empty()
        || policy.clock_domain_id.trim().is_empty()
        || policy.revision == 0
        || !valid_sha256(&policy.nominal_baseline_manifest_sha256)
        || !policy.maximum_absolute_standardized_value.is_finite()
        || policy.maximum_absolute_standardized_value <= 0.0
        || policy.channels.len() != 3
    {
        return Err(EvidenceShiftErrorV1(
            "shift policy identity, baseline digest or channel inventory is invalid".into(),
        ));
    }
    for channel in [
        ShiftSignalChannelV1::VmResidual,
        ShiftSignalChannelV1::ReferenceStandardResidual,
        ShiftSignalChannelV1::SensorObservabilityScore,
    ] {
        let config = policy.channels.get(&channel).ok_or_else(|| {
            EvidenceShiftErrorV1(format!("shift policy is missing {channel:?}"))
        })?;
        if !config.baseline_mean.is_finite()
            || !config.baseline_standard_deviation.is_finite()
            || config.baseline_standard_deviation <= 0.0
            || !config.cusum_allowance.is_finite()
            || config.cusum_allowance < 0.0
            || !config.cusum_threshold.is_finite()
            || config.cusum_threshold <= 0.0
            || !config.ewma_alpha.is_finite()
            || !(0.0..=1.0).contains(&config.ewma_alpha)
            || !config.ewma_threshold.is_finite()
            || config.ewma_threshold <= 0.0
            || config.minimum_observations == 0
        {
            return Err(EvidenceShiftErrorV1(format!(
                "shift policy has invalid {channel:?} baseline or thresholds"
            )));
        }
    }
    Ok(())
}

fn validate_engine(engine: &EvidenceShiftMonitorV1) -> Result<(), EvidenceShiftErrorV1> {
    if engine.schema != EVIDENCE_SHIFT_ENGINE_SCHEMA_V1 {
        return Err(EvidenceShiftErrorV1("unsupported shift engine schema".into()));
    }
    validate_policy(&engine.policy)?;
    if json_sha256(&engine.policy)? != engine.policy_sha256 {
        return Err(EvidenceShiftErrorV1(
            "shift policy content does not match policy digest".into(),
        ));
    }
    Ok(())
}

pub fn create_evidence_shift_monitor_v1(
    engine_id: &str,
    policy: EvidenceShiftPolicyV1,
) -> Result<EvidenceShiftMonitorV1, EvidenceShiftErrorV1> {
    if engine_id.trim().is_empty() {
        return Err(EvidenceShiftErrorV1("engine_id is empty".into()));
    }
    validate_policy(&policy)?;
    let policy_sha256 = json_sha256(&policy)?;
    Ok(EvidenceShiftMonitorV1 {
        schema: EVIDENCE_SHIFT_ENGINE_SCHEMA_V1.into(),
        engine_id: engine_id.into(),
        policy,
        policy_sha256,
        last_stream_sequence: 0,
        last_scope_event_times: BTreeMap::new(),
        channel_states: BTreeMap::new(),
        process_markers: BTreeMap::new(),
        seen_observation_ids: BTreeSet::new(),
        assessments_by_observation: BTreeMap::new(),
    })
}

fn normalize_channel_value(
    observation: &EvidenceShiftObservationV1,
) -> Result<(Option<f64>, Option<String>), Vec<ShiftReasonCodeV1>> {
    let mut reasons = Vec::new();
    match observation.channel {
        ShiftSignalChannelV1::VmResidual => {
            let Some(report) = observation.label_report.as_ref() else {
                return Err(vec![ShiftReasonCodeV1::MissingQualifiedLabelReport]);
            };
            if report.frontier != LabelEvidenceFrontierV1::CurrentAtK1 {
                reasons.push(ShiftReasonCodeV1::LabelNotCurrentK1);
            }
            if report.eligibility != LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
                || !report.downstream_consumption_allowed
            {
                reasons.push(ShiftReasonCodeV1::LabelNotEligible);
            }
            if report.label_revision == 0 {
                reasons.push(ShiftReasonCodeV1::LabelRevisionInvalid);
            }
            if !valid_sha256(&report.label_sha256) {
                reasons.push(ShiftReasonCodeV1::LabelPayloadDigestInvalid);
            }
            if report_digest(report).map_err(|_| vec![ShiftReasonCodeV1::LabelReportDigestMismatch])?
                != report.report_sha256
            {
                reasons.push(ShiftReasonCodeV1::LabelReportDigestMismatch);
            }
            let Some(prediction) = observation.prediction_value else {
                return Err(vec![ShiftReasonCodeV1::NonFiniteObservation]);
            };
            if !prediction.is_finite() || !report.qualified_corrected_value.is_finite() {
                reasons.push(ShiftReasonCodeV1::NonFiniteObservation);
            }
            if !reasons.is_empty() {
                return Err(reasons);
            }
            Ok((
                Some(report.qualified_corrected_value - prediction),
                Some(report.label_id.clone()),
            ))
        }
        ShiftSignalChannelV1::ReferenceStandardResidual => {
            if !observation.reference_evidence_qualified {
                reasons.push(ShiftReasonCodeV1::ReferenceEvidenceNotQualified);
            }
            if observation
                .reference_standard_id
                .as_deref()
                .is_none_or(|value| value.trim().is_empty())
                || observation
                    .reference_evidence_sha256
                    .as_deref()
                    .is_none_or(|value| !valid_sha256(value))
            {
                reasons.push(ShiftReasonCodeV1::ReferenceEvidenceMissing);
            }
            let Some(value) = observation.observation_value else {
                return Err(vec![ShiftReasonCodeV1::NonFiniteObservation]);
            };
            if !value.is_finite() {
                reasons.push(ShiftReasonCodeV1::NonFiniteObservation);
            }
            if !reasons.is_empty() {
                return Err(reasons);
            }
            Ok((Some(value), None))
        }
        ShiftSignalChannelV1::SensorObservabilityScore => {
            if observation
                .sensor_projection_sha256
                .as_deref()
                .is_none_or(|value| !valid_sha256(value))
            {
                reasons.push(ShiftReasonCodeV1::SensorProjectionDigestMissing);
            }
            let Some(value) = observation.observation_value else {
                return Err(vec![ShiftReasonCodeV1::NonFiniteObservation]);
            };
            if !value.is_finite() {
                reasons.push(ShiftReasonCodeV1::NonFiniteObservation);
            }
            if !reasons.is_empty() {
                return Err(reasons);
            }
            Ok((Some(value), None))
        }
        ShiftSignalChannelV1::ProcessEventMarker => Ok((None, None)),
    }
}

fn update_channel_state(
    old: Option<&ShiftChannelStateV1>,
    channel_policy: &ShiftChannelPolicyV1,
    standardized_value: f64,
    event_time: i64,
    observation_id: &str,
) -> ShiftChannelStateV1 {
    let mut state = old.cloned().unwrap_or(ShiftChannelStateV1 {
        observations: 0,
        positive_cusum: 0.0,
        negative_cusum: 0.0,
        ewma: 0.0,
        last_event_time_unix_seconds: event_time,
        last_observation_id: observation_id.into(),
        alert: false,
    });
    state.positive_cusum =
        (state.positive_cusum + standardized_value - channel_policy.cusum_allowance).max(0.0);
    state.negative_cusum =
        (state.negative_cusum - standardized_value - channel_policy.cusum_allowance).max(0.0);
    state.ewma = channel_policy.ewma_alpha * standardized_value
        + (1.0 - channel_policy.ewma_alpha) * state.ewma;
    state.observations = state.observations.saturating_add(1);
    state.last_event_time_unix_seconds = event_time;
    state.last_observation_id = observation_id.into();
    state.alert = state.positive_cusum >= channel_policy.cusum_threshold
        || state.negative_cusum >= channel_policy.cusum_threshold
        || state.ewma.abs() >= channel_policy.ewma_threshold;
    state
}

fn recent_state(
    states: &BTreeMap<(String, ShiftSignalChannelV1), ShiftChannelStateV1>,
    scope: &str,
    channel: ShiftSignalChannelV1,
    now: i64,
    horizon: u64,
    minimum: u64,
) -> Option<ShiftChannelAssessmentV1> {
    let state = states.get(&(scope.to_string(), channel))?;
    let age = now.saturating_sub(state.last_event_time_unix_seconds);
    if age < 0 || age as u64 > horizon {
        return None;
    }
    Some(ShiftChannelAssessmentV1 {
        channel,
        observations: state.observations,
        positive_cusum: state.positive_cusum,
        negative_cusum: state.negative_cusum,
        ewma: state.ewma,
        alert: state.alert,
        stable_reference_window: state.observations >= minimum && !state.alert,
    })
}

fn classify(
    assessments: &[ShiftChannelAssessmentV1],
    process_marker_recent: bool,
) -> (
    ShiftDiagnosticStateV1,
    ShiftAdaptationRecommendationV1,
    Vec<ShiftSignalChannelV1>,
    Vec<ShiftSignalChannelV1>,
    Vec<ShiftReasonCodeV1>,
) {
    let find = |channel| assessments.iter().find(|entry| entry.channel == channel);
    let residual = find(ShiftSignalChannelV1::VmResidual);
    let reference = find(ShiftSignalChannelV1::ReferenceStandardResidual);
    let sensor = find(ShiftSignalChannelV1::SensorObservabilityScore);
    let residual_alert = residual.is_some_and(|entry| entry.alert);
    let reference_alert = reference.is_some_and(|entry| entry.alert);
    let sensor_alert = sensor.is_some_and(|entry| entry.alert);
    let reference_stable = reference.is_some_and(|entry| entry.stable_reference_window);
    let sensor_stable = sensor.is_some_and(|entry| entry.stable_reference_window);
    let mut supporting = Vec::new();
    let mut contradicting = Vec::new();
    let mut reasons = Vec::new();

    let (state, recommendation) = if (residual_alert && reference_alert)
        || (reference_alert && sensor_alert)
    {
        supporting.extend([
            ShiftSignalChannelV1::VmResidual,
            ShiftSignalChannelV1::ReferenceStandardResidual,
        ]);
        reasons.push(ShiftReasonCodeV1::ResidualAndReferenceBothShifted);
        (
            ShiftDiagnosticStateV1::CommonCauseOrMixedShift,
            ShiftAdaptationRecommendationV1::Hold,
        )
    } else if reference_alert {
        supporting.push(ShiftSignalChannelV1::ReferenceStandardResidual);
        (
            ShiftDiagnosticStateV1::MeasurementSystemShiftSuspected,
            ShiftAdaptationRecommendationV1::Hold,
        )
    } else if sensor_alert {
        supporting.push(ShiftSignalChannelV1::SensorObservabilityScore);
        if residual_alert {
            supporting.push(ShiftSignalChannelV1::VmResidual);
        }
        (
            ShiftDiagnosticStateV1::SensorObservabilityShift,
            ShiftAdaptationRecommendationV1::Hold,
        )
    } else if residual_alert {
        supporting.push(ShiftSignalChannelV1::VmResidual);
        if reference_stable {
            contradicting.push(ShiftSignalChannelV1::ReferenceStandardResidual);
        }
        if sensor_stable {
            contradicting.push(ShiftSignalChannelV1::SensorObservabilityScore);
        }
        if reference_stable && sensor_stable && process_marker_recent {
            (
                ShiftDiagnosticStateV1::ProcessShiftSuspected,
                ShiftAdaptationRecommendationV1::RequestMetrology,
            )
        } else {
            reasons.push(ShiftReasonCodeV1::ResidualShiftWithoutIdentifyingEvidence);
            (
                ShiftDiagnosticStateV1::Unresolved,
                ShiftAdaptationRecommendationV1::Hold,
            )
        }
    } else if residual.is_none() {
        reasons.push(ShiftReasonCodeV1::MissingDetectorChannel);
        (
            ShiftDiagnosticStateV1::InsufficientEvidence,
            ShiftAdaptationRecommendationV1::Hold,
        )
    } else if residual.is_some_and(|entry| entry.observations == 0) {
        reasons.push(ShiftReasonCodeV1::MissingDetectorChannel);
        (
            ShiftDiagnosticStateV1::InsufficientEvidence,
            ShiftAdaptationRecommendationV1::Hold,
        )
    } else if residual.is_none()
        || !residual.is_some_and(|entry| entry.stable_reference_window)
        || !reference_stable
        || !sensor_stable
    {
        reasons.push(ShiftReasonCodeV1::InsufficientIndependentEvidence);
        (
            ShiftDiagnosticStateV1::InsufficientEvidence,
            ShiftAdaptationRecommendationV1::Hold,
        )
    } else {
        if process_marker_recent {
            reasons.push(ShiftReasonCodeV1::ProcessMarkerWithoutMeasuredShift);
        }
        (
            ShiftDiagnosticStateV1::Nominal,
            ShiftAdaptationRecommendationV1::Continue,
        )
    };
    supporting.sort();
    supporting.dedup();
    contradicting.sort();
    contradicting.dedup();
    reasons.sort();
    reasons.dedup();
    (state, recommendation, supporting, contradicting, reasons)
}

/// Applies one ordered evidence observation and emits an immutable assessment.
///
/// VM residual observations are admitted only from current K1-qualified,
/// digest-valid labels. Invalid observations produce an INSUFFICIENT_EVIDENCE
/// assessment and do not advance any detector channel state.
pub fn observe_evidence_shift_v1(
    engine: &mut EvidenceShiftMonitorV1,
    observation: &EvidenceShiftObservationV1,
) -> Result<EvidenceShiftAssessmentV1, EvidenceShiftErrorV1> {
    validate_engine(engine)?;
    if observation.schema != EVIDENCE_SHIFT_OBSERVATION_SCHEMA_V1 {
        return Err(EvidenceShiftErrorV1(
            "unsupported shift observation schema".into(),
        ));
    }
    if observation.observation_id.trim().is_empty() || observation.scope_id.trim().is_empty() {
        return Err(EvidenceShiftErrorV1(
            "observation and scope identities must be non-empty".into(),
        ));
    }
    if let Some(previous) = engine.assessments_by_observation.get(&observation.observation_id) {
        let mut duplicate = previous.clone();
        duplicate.reason_codes.push(ShiftReasonCodeV1::DuplicateObservation);
        duplicate.reason_codes.sort();
        duplicate.reason_codes.dedup();
        duplicate.assessment_sha256 = assessment_digest(&duplicate)?;
        return Ok(duplicate);
    }

    let mut early_reasons = Vec::new();
    if observation.clock_domain_id != engine.policy.clock_domain_id {
        early_reasons.push(ShiftReasonCodeV1::ClockDomainMismatch);
    }
    if observation.nominal_baseline_manifest_sha256
        != engine.policy.nominal_baseline_manifest_sha256
    {
        early_reasons.push(ShiftReasonCodeV1::BaselineDigestMismatch);
    }
    if !valid_sha256(&observation.source_evidence_sha256) {
        early_reasons.push(ShiftReasonCodeV1::ReferenceEvidenceMissing);
    }
    if observation.event_time_unix_seconds > observation.available_at_unix_seconds {
        early_reasons.push(ShiftReasonCodeV1::EvidenceNotAvailableAtObservation);
    }
    let expected_sequence = engine.last_stream_sequence.saturating_add(1);
    if observation.stream_sequence != expected_sequence {
        early_reasons.push(ShiftReasonCodeV1::EventSequenceNotContiguous);
    }
    let state_scope = observation_state_scope(observation)?;
    for chronology_scope in [
        Some(observation.scope_id.clone()),
        observation
            .reference_standard_id
            .as_deref()
            .map(reference_state_scope),
    ]
    .into_iter()
    .flatten()
    .collect::<BTreeSet<_>>()
    {
        if engine
            .last_scope_event_times
            .get(&chronology_scope)
            .is_some_and(|previous| observation.event_time_unix_seconds < *previous)
        {
            early_reasons.push(ShiftReasonCodeV1::EventTimeRegression);
        }
    }

    let mut qualified_label_ids = Vec::new();
    if early_reasons.is_empty() && observation.channel == ShiftSignalChannelV1::ProcessEventMarker
        && observation.process_event_marker.is_none()
    {
        early_reasons.push(ShiftReasonCodeV1::ProcessEventEvidenceMissing);
    }
    if early_reasons.is_empty() {
        if let Some(marker) = observation.process_event_marker.as_ref()
            && (marker.event_id.trim().is_empty()
                || marker.event_kind.trim().is_empty()
                || !valid_sha256(&marker.source_evidence_sha256)
                || marker.event_time_unix_seconds > observation.event_time_unix_seconds)
        {
            early_reasons.push(ShiftReasonCodeV1::ProcessEventEvidenceMissing);
        }
    }

    let mut standardized_value = None;
    if early_reasons.is_empty() && observation.channel != ShiftSignalChannelV1::ProcessEventMarker {
        match normalize_channel_value(observation) {
            Ok((value, label_id)) => {
                if let Some(label_id) = label_id {
                    qualified_label_ids.push(label_id);
                }
                standardized_value = value;
            }
            Err(reasons) => early_reasons.extend(reasons),
        }
    }

    if early_reasons.is_empty() {
        if let Some(value) = standardized_value {
            if !value.is_finite() {
                early_reasons.push(ShiftReasonCodeV1::NonFiniteObservation);
            } else {
                let config = engine
                    .policy
                    .channels
                    .get(&observation.channel)
                    .expect("validated policy has all detector channels");
                let standardized =
                    (value - config.baseline_mean) / config.baseline_standard_deviation;
                if !standardized.is_finite()
                    || standardized.abs() > engine.policy.maximum_absolute_standardized_value
                {
                    early_reasons.push(ShiftReasonCodeV1::NonFiniteObservation);
                } else {
                    let key = (state_scope.clone(), observation.channel);
                    let state = update_channel_state(
                        engine.channel_states.get(&key),
                        config,
                        standardized,
                        observation.event_time_unix_seconds,
                        &observation.observation_id,
                    );
                    engine.channel_states.insert(key, state);
                }
            }
        }
    }

    if early_reasons.is_empty()
        && let Some(marker) = observation.process_event_marker.as_ref()
    {
        let current = engine.process_markers.get(&observation.scope_id);
        if current.is_none_or(|existing| {
            marker.event_time_unix_seconds >= existing.event_time_unix_seconds
        }) {
            engine
                .process_markers
                .insert(observation.scope_id.clone(), marker.clone());
        }
    }

    let assessment_time = observation.event_time_unix_seconds;
    let mut channel_assessments = Vec::new();
    for channel in [
        ShiftSignalChannelV1::VmResidual,
        ShiftSignalChannelV1::ReferenceStandardResidual,
        ShiftSignalChannelV1::SensorObservabilityScore,
    ] {
        if let Some(config) = engine.policy.channels.get(&channel) {
            let channel_scope = if channel == ShiftSignalChannelV1::ReferenceStandardResidual {
                observation
                    .reference_standard_id
                    .as_deref()
                    .map(reference_state_scope)
                    .unwrap_or_else(|| observation.scope_id.clone())
            } else {
                observation.scope_id.clone()
            };
            if let Some(state) = recent_state(
                &engine.channel_states,
                &channel_scope,
                channel,
                assessment_time,
                engine.policy.process_event_context_window_seconds,
                config.minimum_observations,
            ) {
                channel_assessments.push(state);
            }
        }
    }
    let marker = engine.process_markers.get(&observation.scope_id);
    let process_marker_recent = marker.is_some_and(|marker| {
        let age = assessment_time.saturating_sub(marker.event_time_unix_seconds);
        age >= 0 && age as u64 <= engine.policy.process_event_context_window_seconds
    });
    let (mut diagnostic_state, mut recommendation, mut supporting, mut contradicting, mut reasons) =
        classify(&channel_assessments, process_marker_recent);
    if !early_reasons.is_empty() {
        diagnostic_state = ShiftDiagnosticStateV1::InsufficientEvidence;
        recommendation = ShiftAdaptationRecommendationV1::Hold;
        reasons.extend(early_reasons);
    }
    reasons.extend(
        channel_assessments
            .iter()
            .filter(|entry| !entry.stable_reference_window && !entry.alert)
            .filter(|entry| entry.observations < 1)
            .map(|_| ShiftReasonCodeV1::InsufficientIndependentEvidence),
    );
    reasons.sort();
    reasons.dedup();
    supporting.sort();
    supporting.dedup();
    contradicting.sort();
    contradicting.dedup();

    let mut assessment = EvidenceShiftAssessmentV1 {
        schema: EVIDENCE_SHIFT_ASSESSMENT_SCHEMA_V1.into(),
        engine_id: engine.engine_id.clone(),
        scope_id: observation.scope_id.clone(),
        observation_id: observation.observation_id.clone(),
        stream_sequence: observation.stream_sequence,
        diagnostic_state,
        recommendation,
        channel_assessments,
        supporting_channels: supporting,
        contradicting_channels: contradicting,
        process_event_id: marker.map(|value| value.event_id.clone()),
        qualified_source_label_ids: qualified_label_ids,
        reason_codes: reasons,
        nominal_baseline_manifest_sha256: engine
            .policy
            .nominal_baseline_manifest_sha256
            .clone(),
        policy_sha256: engine.policy_sha256.clone(),
        previous_assessment_sha256: engine
            .assessments_by_observation
            .values()
            .max_by_key(|value| value.stream_sequence)
            .map(|value| value.assessment_sha256.clone()),
        assessment_sha256: String::new(),
        diagnostic_is_hypothesis_not_root_cause_proof: true,
        process_control_authority: false,
        automatic_retraining_executed: false,
    };
    assessment.assessment_sha256 = assessment_digest(&assessment)?;

    // Invalid or out-of-order observations are not consumed by the detector.
    // Valid process markers are retained separately as contextual evidence.
    if early_reasons.is_empty() {
        engine.last_stream_sequence = observation.stream_sequence;
        engine
            .last_scope_event_times
            .insert(observation.scope_id.clone(), observation.event_time_unix_seconds);
        if let Some(reference_id) = observation.reference_standard_id.as_deref() {
            engine.last_scope_event_times.insert(
                reference_state_scope(reference_id),
                observation.event_time_unix_seconds,
            );
        }
        engine.seen_observation_ids.insert(observation.observation_id.clone());
    }
    engine
        .assessments_by_observation
        .insert(observation.observation_id.clone(), assessment.clone());
    Ok(assessment)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn policy() -> EvidenceShiftPolicyV1 {
        let config = ShiftChannelPolicyV1 {
            baseline_mean: 0.0,
            baseline_standard_deviation: 1.0,
            cusum_allowance: 0.5,
            cusum_threshold: 4.0,
            ewma_alpha: 0.5,
            ewma_threshold: 2.5,
            minimum_observations: 2,
        };
        EvidenceShiftPolicyV1 {
            schema: EVIDENCE_SHIFT_POLICY_SCHEMA_V1.into(),
            policy_id: "semi-shift-policy-001".into(),
            revision: 1,
            clock_domain_id: "fab-clock".into(),
            nominal_baseline_manifest_sha256: digest('a'),
            process_event_context_window_seconds: 100,
            maximum_absolute_standardized_value: 20.0,
            channels: BTreeMap::from([
                (ShiftSignalChannelV1::VmResidual, config.clone()),
                (ShiftSignalChannelV1::ReferenceStandardResidual, config.clone()),
                (ShiftSignalChannelV1::SensorObservabilityScore, config),
            ]),
        }
    }

    fn engine() -> EvidenceShiftMonitorV1 {
        create_evidence_shift_monitor_v1("shift-engine-001", policy()).unwrap()
    }

    fn base_observation(
        sequence: u64,
        id: &str,
        scope: &str,
        time: i64,
        channel: ShiftSignalChannelV1,
    ) -> EvidenceShiftObservationV1 {
        EvidenceShiftObservationV1 {
            schema: EVIDENCE_SHIFT_OBSERVATION_SCHEMA_V1.into(),
            observation_id: id.into(),
            stream_sequence: sequence,
            scope_id: scope.into(),
            event_time_unix_seconds: time,
            available_at_unix_seconds: time + 1,
            clock_domain_id: "fab-clock".into(),
            nominal_baseline_manifest_sha256: digest('a'),
            channel,
            observation_value: None,
            prediction_value: None,
            label_report: None,
            reference_standard_id: None,
            reference_evidence_sha256: None,
            reference_evidence_qualified: false,
            sensor_projection_sha256: None,
            process_event_marker: None,
            source_evidence_sha256: digest('b'),
        }
    }

    fn label_report(label_id: &str, event_id: &str, value: f64) -> LabelConsumptionEligibilityReportV1 {
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label_id.into(),
            label_revision: 1,
            measurement_event_id: event_id.into(),
            calibration_certificate_id: "cert-001".into(),
            frontier: LabelEvidenceFrontierV1::CurrentAtK1,
            eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            downstream_consumption_allowed: true,
            reasons: Vec::new(),
            label_sha256: digest('c'),
            measurement_sha256: digest('d'),
            qualified_corrected_value: value,
            qualified_standard_uncertainty_upper_bound: Some(0.1),
            qualification_report_sha256: digest('e'),
            requalification_report_sha256: digest('f'),
            requalification_frontier_sha256: digest('0'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        report.report_sha256 = report_digest(&report).unwrap();
        report
    }

    fn vm_residual_observation(
        sequence: u64,
        id: &str,
        scope: &str,
        time: i64,
        residual: f64,
        label_id: &str,
    ) -> EvidenceShiftObservationV1 {
        let mut observation = base_observation(
            sequence,
            id,
            scope,
            time,
            ShiftSignalChannelV1::VmResidual,
        );
        observation.prediction_value = Some(10.0);
        observation.reference_standard_id = Some("transfer-standard-001".into());
        observation.label_report = Some(label_report(label_id, &format!("event-{id}"), 10.0 + residual));
        observation
    }

    fn reference_observation(
        sequence: u64,
        id: &str,
        scope: &str,
        time: i64,
        residual: f64,
    ) -> EvidenceShiftObservationV1 {
        let mut observation = base_observation(
            sequence,
            id,
            scope,
            time,
            ShiftSignalChannelV1::ReferenceStandardResidual,
        );
        observation.observation_value = Some(residual);
        observation.reference_standard_id = Some("transfer-standard-001".into());
        observation.reference_evidence_sha256 = Some(digest('c'));
        observation.reference_evidence_qualified = true;
        observation
    }

    fn sensor_observation(
        sequence: u64,
        id: &str,
        scope: &str,
        time: i64,
        score: f64,
    ) -> EvidenceShiftObservationV1 {
        let mut observation = base_observation(
            sequence,
            id,
            scope,
            time,
            ShiftSignalChannelV1::SensorObservabilityScore,
        );
        observation.observation_value = Some(score);
        observation.sensor_projection_sha256 = Some(digest('d'));
        observation
    }

    #[test]
    fn vm_residual_never_enters_detector_without_current_qualified_label() {
        let mut engine = engine();
        let mut observation = vm_residual_observation(1, "obs-1", "tool-a/ch-1", 1, 5.0, "label-1");
        observation.label_report.as_mut().unwrap().eligibility = LabelConsumptionEligibilityV1::Unknown;
        let report = observation.label_report.as_mut().unwrap();
        report.downstream_consumption_allowed = false;
        report.report_sha256 = report_digest(report).unwrap();
        let assessment = observe_evidence_shift_v1(&mut engine, &observation).unwrap();
        assert_eq!(assessment.diagnostic_state, ShiftDiagnosticStateV1::InsufficientEvidence);
        assert!(assessment.reason_codes.contains(&ShiftReasonCodeV1::LabelNotEligible));
        assert!(!engine.channel_states.contains_key(&("tool-a/ch-1".into(), ShiftSignalChannelV1::VmResidual)));
    }

    #[test]
    fn missing_independent_channels_cannot_be_reported_nominal() {
        let mut engine = engine();
        let observation = vm_residual_observation(1, "vm-1", "tool-a/ch-1", 10, 0.0, "label-1");
        let assessment = observe_evidence_shift_v1(&mut engine, &observation).unwrap();
        assert_eq!(assessment.diagnostic_state, ShiftDiagnosticStateV1::InsufficientEvidence);
        assert_eq!(assessment.recommendation, ShiftAdaptationRecommendationV1::Hold);
    }

    #[test]
    fn invalid_channel_observation_does_not_commit_attached_process_marker() {
        let mut engine = engine();
        let mut observation = reference_observation(1, "ref-invalid", "tool-a/ch-1", 10, 2.0);
        observation.reference_evidence_qualified = false;
        observation.process_event_marker = Some(ProcessEventMarkerV1 {
            event_id: "maintenance-1".into(),
            event_kind: "maintenance".into(),
            event_time_unix_seconds: 9,
            source_evidence_sha256: digest('e'),
        });
        let assessment = observe_evidence_shift_v1(&mut engine, &observation).unwrap();
        assert_eq!(assessment.diagnostic_state, ShiftDiagnosticStateV1::InsufficientEvidence);
        assert!(!engine.process_markers.contains_key("tool-a/ch-1"));
        assert!(engine.channel_states.is_empty());
    }

    #[test]
    fn process_marker_alone_does_not_create_a_shift_alarm() {
        let mut engine = engine();
        let mut observation = base_observation(1, "maintenance-1", "tool-a/ch-1", 10, ShiftSignalChannelV1::ProcessEventMarker);
        observation.process_event_marker = Some(ProcessEventMarkerV1 {
            event_id: "maint-1".into(),
            event_kind: "maintenance".into(),
            event_time_unix_seconds: 10,
            source_evidence_sha256: digest('c'),
        });
        let assessment = observe_evidence_shift_v1(&mut engine, &observation).unwrap();
        assert_eq!(assessment.diagnostic_state, ShiftDiagnosticStateV1::InsufficientEvidence);
        assert_eq!(assessment.recommendation, ShiftAdaptationRecommendationV1::Hold);
    }

    #[test]
    fn reference_shift_is_not_mislabeled_as_process_shift() {
        let mut engine = engine();
        let mut seq = 1;
        for i in 0..3 {
            let observation = reference_observation(seq, &format!("ref-{i}"), "tool-a/ch-1", 10 + i, 5.0);
            let assessment = observe_evidence_shift_v1(&mut engine, &observation).unwrap();
            seq += 1;
            if i == 2 {
                assert_eq!(assessment.diagnostic_state, ShiftDiagnosticStateV1::MeasurementSystemShiftSuspected);
            }
        }
    }

    #[test]
    fn simultaneous_product_and_reference_shift_is_common_cause_or_mixed() {
        let mut engine = engine();
        let mut seq = 1;
        for i in 0..3 {
            let reference = reference_observation(seq, &format!("ref-{i}"), "tool-a/ch-1", 10 + i, 5.0);
            observe_evidence_shift_v1(&mut engine, &reference).unwrap();
            seq += 1;
            let residual = vm_residual_observation(seq, &format!("vm-{i}"), "tool-a/ch-1", 10 + i, 5.0, &format!("label-{i}"));
            let assessment = observe_evidence_shift_v1(&mut engine, &residual).unwrap();
            seq += 1;
            if i == 2 {
                assert_eq!(assessment.diagnostic_state, ShiftDiagnosticStateV1::CommonCauseOrMixedShift);
            }
        }
    }

    #[test]
    fn explicitly_shared_reference_standard_alert_propagates_across_tool_scopes() {
        let mut engine = engine();
        let shared_standard = "shared-transfer-standard-001";
        for index in 0..3 {
            let mut reference = reference_observation(
                index + 1,
                &format!("shared-ref-{index}"),
                "tool-a/ch-1",
                10 + index as i64,
                5.0,
            );
            reference.reference_standard_id = Some(shared_standard.into());
            observe_evidence_shift_v1(&mut engine, &reference).unwrap();
        }

        let mut vm = vm_residual_observation(
            4,
            "tool-b-vm",
            "tool-b/ch-9",
            13,
            5.0,
            "label-tool-b",
        );
        vm.reference_standard_id = Some(shared_standard.into());
        let assessment = observe_evidence_shift_v1(&mut engine, &vm).unwrap();
        assert_eq!(
            assessment.diagnostic_state,
            ShiftDiagnosticStateV1::CommonCauseOrMixedShift
        );
        assert!(
            assessment
                .supporting_channels
                .contains(&ShiftSignalChannelV1::ReferenceStandardResidual)
        );
    }

    #[test]
    fn residual_shift_without_independent_evidence_is_unresolved() {
        let mut engine = engine();
        let mut result = None;
        for i in 0..4 {
            let observation = vm_residual_observation(i + 1, &format!("vm-{i}"), "tool-a/ch-1", i as i64 + 1, 5.0, &format!("label-{i}"));
            result = Some(observe_evidence_shift_v1(&mut engine, &observation).unwrap());
        }
        assert_eq!(result.unwrap().diagnostic_state, ShiftDiagnosticStateV1::Unresolved);
    }

    #[test]
    fn process_shift_is_only_suspected_with_residual_alert_and_independent_stable_channels() {
        let mut engine = engine();
        let mut seq = 1;
        for i in 0..3 {
            let ref_obs = reference_observation(seq, &format!("ref-{i}"), "tool-a/ch-1", 10 + i, 0.0);
            observe_evidence_shift_v1(&mut engine, &ref_obs).unwrap();
            seq += 1;
            let sensor = sensor_observation(seq, &format!("sensor-{i}"), "tool-a/ch-1", 10 + i, 0.0);
            observe_evidence_shift_v1(&mut engine, &sensor).unwrap();
            seq += 1;
            let mut vm = vm_residual_observation(seq, &format!("vm-{i}"), "tool-a/ch-1", 10 + i, 5.0, &format!("label-{i}"));
            if i == 0 {
                vm.process_event_marker = Some(ProcessEventMarkerV1 {
                    event_id: "recipe-change-1".into(),
                    event_kind: "recipe_revision".into(),
                    event_time_unix_seconds: 10,
                    source_evidence_sha256: digest('e'),
                });
            }
            let assessment = observe_evidence_shift_v1(&mut engine, &vm).unwrap();
            seq += 1;
            if i == 2 {
                assert_eq!(assessment.diagnostic_state, ShiftDiagnosticStateV1::ProcessShiftSuspected);
                assert_eq!(assessment.recommendation, ShiftAdaptationRecommendationV1::RequestMetrology);
            }
        }
    }

    #[test]
    fn sensor_shift_takes_precedence_over_unattributed_process_shift() {
        let mut engine = engine();
        let mut seq = 1;
        for i in 0..3 {
            let sensor = sensor_observation(seq, &format!("sensor-{i}"), "tool-a/ch-1", 10 + i, 5.0);
            let assessment = observe_evidence_shift_v1(&mut engine, &sensor).unwrap();
            seq += 1;
            if i == 2 {
                assert_eq!(assessment.diagnostic_state, ShiftDiagnosticStateV1::SensorObservabilityShift);
            }
        }
    }

    #[test]
    fn tool_scopes_do_not_share_detector_state() {
        let mut engine = engine();
        for i in 0..4 {
            let observation = vm_residual_observation(i + 1, &format!("a-{i}"), "tool-a/ch-1", i as i64 + 1, 5.0, &format!("label-a-{i}"));
            observe_evidence_shift_v1(&mut engine, &observation).unwrap();
        }
        let tool_b = vm_residual_observation(5, "b-1", "tool-b/ch-9", 5, 0.0, "label-b-1");
        let report = observe_evidence_shift_v1(&mut engine, &tool_b).unwrap();
        assert_eq!(report.diagnostic_state, ShiftDiagnosticStateV1::InsufficientEvidence);
        assert!(!report.channel_assessments.iter().any(|c| c.channel == ShiftSignalChannelV1::VmResidual && c.alert));
    }

    #[test]
    fn event_time_regression_across_different_channels_is_rejected() {
        let mut engine = engine();
        let reference = reference_observation(1, "ref-1", "tool-a/ch-1", 20, 0.0);
        observe_evidence_shift_v1(&mut engine, &reference).unwrap();

        let vm = vm_residual_observation(2, "vm-1", "tool-a/ch-1", 19, 0.0, "label-1");
        let assessment = observe_evidence_shift_v1(&mut engine, &vm).unwrap();
        assert_eq!(assessment.diagnostic_state, ShiftDiagnosticStateV1::InsufficientEvidence);
        assert!(assessment.reason_codes.contains(&ShiftReasonCodeV1::EventTimeRegression));
        assert_eq!(engine.last_stream_sequence, 1);
    }

    #[test]
    fn process_marker_channel_requires_explicit_marker_evidence() {
        let mut engine = engine();
        let observation = base_observation(
            1,
            "marker-missing",
            "tool-a/ch-1",
            10,
            ShiftSignalChannelV1::ProcessEventMarker,
        );
        let assessment = observe_evidence_shift_v1(&mut engine, &observation).unwrap();
        assert!(assessment.reason_codes.contains(&ShiftReasonCodeV1::ProcessEventEvidenceMissing));
        assert!(!engine.process_markers.contains_key("tool-a/ch-1"));
    }

    #[test]
    fn baseline_digest_and_clock_mismatch_fail_closed() {
        let mut engine = engine();
        let mut bad = vm_residual_observation(1, "bad", "tool-a/ch-1", 1, 5.0, "label-1");
        bad.nominal_baseline_manifest_sha256 = digest('f');
        let report = observe_evidence_shift_v1(&mut engine, &bad).unwrap();
        assert_eq!(report.diagnostic_state, ShiftDiagnosticStateV1::InsufficientEvidence);
        assert!(report.reason_codes.contains(&ShiftReasonCodeV1::BaselineDigestMismatch));

        let mut bad_clock = vm_residual_observation(1, "bad-clock", "tool-a/ch-1", 1, 5.0, "label-1");
        bad_clock.clock_domain_id = "other-clock".into();
        let report = observe_evidence_shift_v1(&mut engine, &bad_clock).unwrap();
        assert!(report.reason_codes.contains(&ShiftReasonCodeV1::ClockDomainMismatch));
    }

    #[test]
    fn duplicate_observation_returns_duplicate_receipt_without_mutating_detector() {
        let mut engine = engine();
        let observation = reference_observation(1, "ref-1", "tool-a/ch-1", 10, 2.0);
        let first = observe_evidence_shift_v1(&mut engine, &observation).unwrap();
        let before = engine.channel_states.clone();
        let duplicate = observe_evidence_shift_v1(&mut engine, &observation).unwrap();
        assert!(duplicate.reason_codes.contains(&ShiftReasonCodeV1::DuplicateObservation));
        assert_eq!(engine.channel_states, before);
        assert_ne!(first.assessment_sha256, duplicate.assessment_sha256);
    }

    #[test]
    fn identical_ordered_stream_replays_byte_identically() {
        fn run() -> String {
            let mut engine = engine();
            for i in 0..3 {
                let observation = reference_observation(i + 1, &format!("ref-{i}"), "tool-a/ch-1", i as i64 + 1, 2.0);
                observe_evidence_shift_v1(&mut engine, &observation).unwrap();
            }
            json_sha256(&engine).unwrap()
        }
        assert_eq!(run(), run());
    }

    #[test]
    fn event_sequence_gap_is_not_silently_sorted_or_consumed() {
        let mut engine = engine();
        let observation = reference_observation(2, "ref-2", "tool-a/ch-1", 10, 5.0);
        let assessment = observe_evidence_shift_v1(&mut engine, &observation).unwrap();
        assert_eq!(assessment.diagnostic_state, ShiftDiagnosticStateV1::InsufficientEvidence);
        assert_eq!(engine.last_stream_sequence, 0);
    }
}
