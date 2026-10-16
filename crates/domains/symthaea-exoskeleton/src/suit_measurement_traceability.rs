// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Policy-checkable calibration traceability and sample completeness (SX-036).
//!
//! Hashes bind declared inputs; they do not authenticate sensor output, issuers,
//! calibration certificates, timestamps, or physical measurement accuracy.
//! This module is report-only and confers no hardware or mission authority.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use crate::suit_peak_reservations::{MissionTimeWindow, PeakResourceKind};
use crate::suit_peak_usage_measurements::{
    PeakResourceUsageMeasurementPlan, PeakResourceUsageMeasurementReceipt,
};
use crate::suit_profile::SuitEvidenceLevel;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementTraceabilityPlan {
    pub schema_version: u32,
    pub mission_id: String,
    pub configuration_revision: String,
    /// Must match the digest in an independently verified SX-035 receipt.
    pub measurement_plan_digest_sha256: String,
    /// Exact SX-035 receipt digest; digest binding is not signature verification.
    pub measurement_receipt_digest_sha256: String,
    pub policy: MeasurementTraceabilityPolicy,
    pub claims: Vec<MeasurementTraceabilityClaim>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementTraceabilityPolicy {
    pub max_calibration_chain_depth: usize,
    /// Conservative sum of per-link expanded-uncertainty upper bounds, same unit.
    pub maximum_total_expanded_uncertainty: u64,
    pub maximum_inter_sample_gap_ms: u64,
    pub minimum_calibration_evidence_level: SuitEvidenceLevel,
    pub require_zero_dropped_samples: bool,
    pub reference_standards: Vec<TraceabilityReferenceStandard>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceabilityReferenceStandard {
    pub resource_kind: PeakResourceKind,
    pub artifact_digest_sha256: String,
    pub evidence_level: SuitEvidenceLevel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementTraceabilityClaim {
    pub observation_id: String,
    pub pool_id: String,
    pub resource_kind: PeakResourceKind,
    pub provider_component_id: String,
    pub provider_hardware_revision: String,
    pub window: MissionTimeWindow,
    pub instrument_id: String,
    pub instrument_artifact_digest_sha256: String,
    pub instrument_hardware_revision: String,
    pub firmware_digest_sha256: String,
    pub raw_trace_digest_sha256: String,
    /// Claimed absolute Unix milliseconds, independent of mission-relative time.
    /// These are inputs to check, not a trusted clock assertion.
    pub measurement_start_unix_ms: u64,
    pub measurement_end_unix_ms: u64,
    pub sample_sequence_first: u64,
    pub sample_sequence_last: u64,
    pub sample_count: u64,
    pub dropped_sample_count: u64,
    pub maximum_observed_inter_sample_gap_ms: u64,
    /// Ordered from the instrument toward the declared reference standard.
    pub calibration_chain: Vec<CalibrationTraceLink>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationTraceLink {
    pub calibrated_artifact_digest_sha256: String,
    pub calibrated_hardware_revision: String,
    pub reference_artifact_digest_sha256: String,
    pub certificate_digest_sha256: String,
    pub resource_kind: PeakResourceKind,
    /// Declared absolute Unix-millisecond validity interval [from, until).
    pub valid_from_unix_ms: u64,
    pub valid_until_unix_ms: u64,
    pub expanded_uncertainty_upper: u64,
    pub evidence_level: SuitEvidenceLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TraceabilityFindingCode {
    ObservationBindingMismatch,
    InvalidArtifactDigest,
    InvalidMeasurementTimestamp,
    InvalidCalibrationValidityInterval,
    CalibrationChainEmpty,
    CalibrationChainTooDeep,
    CalibrationChainDiscontinuous,
    CalibrationChainCycle,
    CalibrationRootMissing,
    CalibrationRootMismatch,
    CalibrationValidityDoesNotCoverMeasurement,
    CalibrationResourceKindMismatch,
    CalibrationEvidenceBelowPolicy,
    ExpandedUncertaintyOverflow,
    ExpandedUncertaintyAbovePolicy,
    SampleSequenceInvalid,
    SampleCountMismatch,
    DroppedSamplesObserved,
    InterSampleGapAbovePolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TraceabilityFindingSeverity {
    Degradation,
    Blocker,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceabilityFinding {
    pub observation_id: String,
    pub code: TraceabilityFindingCode,
    pub severity: TraceabilityFindingSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementTraceabilityStatus {
    Blocked,
    Degraded,
    MeetsDeclaredTraceabilityPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementTraceabilityAssessment {
    pub observation_id: String,
    pub pool_id: String,
    pub resource_kind: PeakResourceKind,
    pub calibration_chain_depth: usize,
    pub total_expanded_uncertainty_upper: Option<u64>,
    pub findings: Vec<TraceabilityFinding>,
    pub status: MeasurementTraceabilityStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementTraceabilityReport {
    pub schema_version: u32,
    pub mission_id: String,
    pub configuration_revision: String,
    pub measurement_plan_digest_sha256: String,
    pub measurement_receipt_digest_sha256: String,
    pub traceability_plan_digest_sha256: String,
    pub assessments: Vec<MeasurementTraceabilityAssessment>,
    pub findings: Vec<TraceabilityFinding>,
    pub status: MeasurementTraceabilityStatus,
    pub observation_count: usize,
    pub policy_compliant_count: usize,
    pub degraded_count: usize,
    pub blocked_count: usize,
    /// Always false: provenance bookkeeping is not physical qualification.
    pub represents_physical_qualification: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasurementTraceabilityReceipt {
    pub measurement_plan_digest_sha256: String,
    pub measurement_receipt_digest_sha256: String,
    pub traceability_plan_digest_sha256: String,
    pub report_digest_sha256: String,
    /// Deterministic integrity digest, not a digital signature.
    pub receipt_digest_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementTraceabilityError {
    UnsupportedSchemaVersion,
    InvalidPlanIdentity,
    InvalidPinnedDigest,
    InvalidPolicy,
    EmptyClaimSet,
    DuplicateObservationClaim,
    ObservationCoverageMismatch,
    MeasurementIdentityMismatch,
    MeasurementPlanDigestMismatch,
    MeasurementReceiptDigestMismatch,
    InvalidMeasurementPlanObservation,
}

pub fn evaluate_measurement_traceability(
    measurement_plan: &PeakResourceUsageMeasurementPlan,
    measurement_receipt: &PeakResourceUsageMeasurementReceipt,
    traceability_plan: &MeasurementTraceabilityPlan,
) -> Result<
    (MeasurementTraceabilityReport, MeasurementTraceabilityReceipt),
    MeasurementTraceabilityError,
> {
    validate_plan(measurement_plan, measurement_receipt, traceability_plan)?;

    let mut observation_by_id = BTreeMap::new();
    for observation in &measurement_plan.observations {
        if observation.observation_id.trim().is_empty()
            || observation.sample_count == 0
            || !observation.window.is_valid()
        {
            return Err(MeasurementTraceabilityError::InvalidMeasurementPlanObservation);
        }
        if observation_by_id
            .insert(observation.observation_id.as_str(), observation)
            .is_some()
        {
            return Err(MeasurementTraceabilityError::InvalidMeasurementPlanObservation);
        }
    }

    let mut claim_by_id = BTreeMap::new();
    for claim in &traceability_plan.claims {
        if claim_by_id.insert(claim.observation_id.as_str(), claim).is_some() {
            return Err(MeasurementTraceabilityError::DuplicateObservationClaim);
        }
    }
    if observation_by_id.len() != claim_by_id.len()
        || observation_by_id.keys().any(|id| !claim_by_id.contains_key(id))
        || claim_by_id.keys().any(|id| !observation_by_id.contains_key(id))
    {
        return Err(MeasurementTraceabilityError::ObservationCoverageMismatch);
    }

    let traceability_plan_digest_sha256 = digest_traceability_plan(traceability_plan);
    let mut assessments = Vec::with_capacity(observation_by_id.len());
    let mut findings = Vec::new();

    for (observation_id, observation) in observation_by_id {
        let claim = claim_by_id[observation_id];
        let assessment = assess_claim(
            observation,
            claim,
            &traceability_plan.policy,
            measurement_plan.maximum_sampling_interval_ms,
        );
        findings.extend(assessment.findings.iter().cloned());
        assessments.push(assessment);
    }

    findings.sort_by(|left, right| {
        left.observation_id
            .cmp(&right.observation_id)
            .then(left.code.cmp(&right.code))
            .then(left.severity.cmp(&right.severity))
    });

    let blocked_count = assessments
        .iter()
        .filter(|assessment| assessment.status == MeasurementTraceabilityStatus::Blocked)
        .count();
    let degraded_count = assessments
        .iter()
        .filter(|assessment| assessment.status == MeasurementTraceabilityStatus::Degraded)
        .count();
    let policy_compliant_count = assessments
        .iter()
        .filter(|assessment| {
            assessment.status
                == MeasurementTraceabilityStatus::MeetsDeclaredTraceabilityPolicy
        })
        .count();
    let status = if blocked_count > 0 {
        MeasurementTraceabilityStatus::Blocked
    } else if degraded_count > 0 {
        MeasurementTraceabilityStatus::Degraded
    } else {
        MeasurementTraceabilityStatus::MeetsDeclaredTraceabilityPolicy
    };

    let report = MeasurementTraceabilityReport {
        schema_version: 1,
        mission_id: traceability_plan.mission_id.clone(),
        configuration_revision: traceability_plan.configuration_revision.clone(),
        measurement_plan_digest_sha256: traceability_plan
            .measurement_plan_digest_sha256
            .clone(),
        measurement_receipt_digest_sha256: traceability_plan
            .measurement_receipt_digest_sha256
            .clone(),
        traceability_plan_digest_sha256: traceability_plan_digest_sha256.clone(),
        observation_count: assessments.len(),
        policy_compliant_count,
        degraded_count,
        blocked_count,
        assessments,
        findings,
        status,
        represents_physical_qualification: false,
    };

    let report_digest_sha256 = digest_report(&report);
    let mut receipt = MeasurementTraceabilityReceipt {
        measurement_plan_digest_sha256: traceability_plan
            .measurement_plan_digest_sha256
            .clone(),
        measurement_receipt_digest_sha256: traceability_plan
            .measurement_receipt_digest_sha256
            .clone(),
        traceability_plan_digest_sha256,
        report_digest_sha256,
        receipt_digest_sha256: String::new(),
    };
    receipt.receipt_digest_sha256 = digest_receipt(&receipt);
    Ok((report, receipt))
}

/// Re-evaluates the declared inputs and compares the complete deterministic result.
/// The caller must independently verify the underlying SX-035 receipt first.
pub fn verify_measurement_traceability_receipt(
    measurement_plan: &PeakResourceUsageMeasurementPlan,
    measurement_receipt: &PeakResourceUsageMeasurementReceipt,
    traceability_plan: &MeasurementTraceabilityPlan,
    expected_report: &MeasurementTraceabilityReport,
    expected_receipt: &MeasurementTraceabilityReceipt,
) -> bool {
    match evaluate_measurement_traceability(
        measurement_plan,
        measurement_receipt,
        traceability_plan,
    ) {
        Ok((report, receipt)) => report == *expected_report && receipt == *expected_receipt,
        Err(_) => false,
    }
}

fn validate_plan(
    measurement_plan: &PeakResourceUsageMeasurementPlan,
    measurement_receipt: &PeakResourceUsageMeasurementReceipt,
    plan: &MeasurementTraceabilityPlan,
) -> Result<(), MeasurementTraceabilityError> {
    if plan.schema_version != 1 {
        return Err(MeasurementTraceabilityError::UnsupportedSchemaVersion);
    }
    if !clean_identifier(&plan.mission_id)
        || !clean_identifier(&plan.configuration_revision)
    {
        return Err(MeasurementTraceabilityError::InvalidPlanIdentity);
    }
    if !valid_sha256(&plan.measurement_plan_digest_sha256)
        || !valid_sha256(&plan.measurement_receipt_digest_sha256)
        || !valid_sha256(&measurement_receipt.measurement_plan_digest_sha256)
        || !valid_sha256(&measurement_receipt.receipt_digest_sha256)
        || !valid_sha256(&measurement_receipt.report_digest_sha256)
    {
        return Err(MeasurementTraceabilityError::InvalidPinnedDigest);
    }
    if plan.mission_id != measurement_plan.mission_id
        || plan.configuration_revision != measurement_plan.configuration_revision
    {
        return Err(MeasurementTraceabilityError::MeasurementIdentityMismatch);
    }
    if plan.measurement_plan_digest_sha256 != measurement_receipt.measurement_plan_digest_sha256 {
        return Err(MeasurementTraceabilityError::MeasurementPlanDigestMismatch);
    }
    if plan.measurement_receipt_digest_sha256 != measurement_receipt.receipt_digest_sha256 {
        return Err(MeasurementTraceabilityError::MeasurementReceiptDigestMismatch);
    }
    if plan.policy.max_calibration_chain_depth == 0
        || plan.policy.max_calibration_chain_depth > 32
        || plan.policy.maximum_inter_sample_gap_ms == 0
        || plan.policy.reference_standards.is_empty()
        || plan.claims.iter().any(|claim| claim.calibration_chain.len() > 64)
    {
        return Err(MeasurementTraceabilityError::InvalidPolicy);
    }
    if plan.claims.is_empty() {
        return Err(MeasurementTraceabilityError::EmptyClaimSet);
    }

    let mut standards = BTreeSet::new();
    for standard in &plan.policy.reference_standards {
        if !valid_sha256(&standard.artifact_digest_sha256)
            || !standards.insert(standard.resource_kind)
        {
            return Err(MeasurementTraceabilityError::InvalidPolicy);
        }
    }
    Ok(())
}

fn assess_claim(
    observation: &crate::suit_peak_usage_measurements::PeakResourceUsageObservation,
    claim: &MeasurementTraceabilityClaim,
    policy: &MeasurementTraceabilityPolicy,
    measurement_policy_maximum_gap_ms: u64,
) -> MeasurementTraceabilityAssessment {
    let mut findings = Vec::new();
    let mut add = |code, severity| {
        findings.push(TraceabilityFinding {
            observation_id: observation.observation_id.clone(),
            code,
            severity,
        });
    };

    if claim.observation_id != observation.observation_id
        || claim.pool_id != observation.pool_id
        || claim.resource_kind != observation.resource_kind
        || claim.provider_component_id != observation.provider_component_id
        || claim.provider_hardware_revision != observation.provider_hardware_revision
        || claim.window != observation.window
    {
        add(
            TraceabilityFindingCode::ObservationBindingMismatch,
            TraceabilityFindingSeverity::Blocker,
        );
    }
    if !clean_identifier(&claim.instrument_id)
        || !clean_identifier(&claim.instrument_hardware_revision)
        || !valid_sha256(&claim.instrument_artifact_digest_sha256)
        || !valid_sha256(&claim.firmware_digest_sha256)
        || !valid_sha256(&claim.raw_trace_digest_sha256)
    {
        add(
            TraceabilityFindingCode::InvalidArtifactDigest,
            TraceabilityFindingSeverity::Blocker,
        );
    }
    if claim.measurement_start_unix_ms >= claim.measurement_end_unix_ms {
        add(
            TraceabilityFindingCode::InvalidMeasurementTimestamp,
            TraceabilityFindingSeverity::Blocker,
        );
    }
    if claim.sample_count != observation.sample_count {
        add(
            TraceabilityFindingCode::SampleCountMismatch,
            TraceabilityFindingSeverity::Blocker,
        );
    }
    let sequence_length = claim
        .sample_sequence_last
        .checked_sub(claim.sample_sequence_first)
        .and_then(|span| span.checked_add(1));
    let expected_sequence_length = claim.sample_count.checked_add(claim.dropped_sample_count);
    if claim.sample_sequence_first == 0
        || sequence_length.is_none()
        || expected_sequence_length.is_none()
        || sequence_length != expected_sequence_length
        || (claim.sample_count > 1 && claim.maximum_observed_inter_sample_gap_ms == 0)
    {
        add(
            TraceabilityFindingCode::SampleSequenceInvalid,
            TraceabilityFindingSeverity::Blocker,
        );
    }
    if claim.dropped_sample_count > 0 {
        add(
            TraceabilityFindingCode::DroppedSamplesObserved,
            if policy.require_zero_dropped_samples {
                TraceabilityFindingSeverity::Blocker
            } else {
                TraceabilityFindingSeverity::Degradation
            },
        );
    }
    if claim.maximum_observed_inter_sample_gap_ms > policy.maximum_inter_sample_gap_ms
        || claim.maximum_observed_inter_sample_gap_ms > measurement_policy_maximum_gap_ms
        || claim.maximum_observed_inter_sample_gap_ms > observation.sampling_interval_ms
    {
        add(
            TraceabilityFindingCode::InterSampleGapAbovePolicy,
            TraceabilityFindingSeverity::Blocker,
        );
    }

    let mut total_uncertainty = Some(0_u64);
    if claim.calibration_chain.is_empty() {
        add(
            TraceabilityFindingCode::CalibrationChainEmpty,
            TraceabilityFindingSeverity::Blocker,
        );
        total_uncertainty = None;
    }
    if claim.calibration_chain.len() > policy.max_calibration_chain_depth {
        add(
            TraceabilityFindingCode::CalibrationChainTooDeep,
            TraceabilityFindingSeverity::Blocker,
        );
    }

    let expected_root = policy
        .reference_standards
        .iter()
        .find(|standard| standard.resource_kind == observation.resource_kind);
    if expected_root.is_none() {
        add(
            TraceabilityFindingCode::CalibrationRootMissing,
            TraceabilityFindingSeverity::Blocker,
        );
    }

    let mut expected_artifact = claim.instrument_artifact_digest_sha256.clone();
    let mut seen_artifacts = BTreeSet::new();
    seen_artifacts.insert(expected_artifact.clone());

    for (index, link) in claim.calibration_chain.iter().enumerate() {
        if !valid_sha256(&link.calibrated_artifact_digest_sha256)
            || !valid_sha256(&link.reference_artifact_digest_sha256)
            || !valid_sha256(&link.certificate_digest_sha256)
            || !clean_identifier(&link.calibrated_hardware_revision)
        {
            add(
                TraceabilityFindingCode::InvalidArtifactDigest,
                TraceabilityFindingSeverity::Blocker,
            );
        }
        if link.calibrated_artifact_digest_sha256 != expected_artifact {
            add(
                TraceabilityFindingCode::CalibrationChainDiscontinuous,
                TraceabilityFindingSeverity::Blocker,
            );
        }
        if !seen_artifacts.insert(link.reference_artifact_digest_sha256.clone()) {
            add(
                TraceabilityFindingCode::CalibrationChainCycle,
                TraceabilityFindingSeverity::Blocker,
            );
        }
        if index == 0
            && link.calibrated_hardware_revision != claim.instrument_hardware_revision
        {
            add(
                TraceabilityFindingCode::ObservationBindingMismatch,
                TraceabilityFindingSeverity::Blocker,
            );
        }
        if link.resource_kind != observation.resource_kind {
            add(
                TraceabilityFindingCode::CalibrationResourceKindMismatch,
                TraceabilityFindingSeverity::Blocker,
            );
        }
        if link.valid_from_unix_ms >= link.valid_until_unix_ms {
            add(
                TraceabilityFindingCode::InvalidCalibrationValidityInterval,
                TraceabilityFindingSeverity::Blocker,
            );
        } else if claim.measurement_start_unix_ms < link.valid_from_unix_ms
            || claim.measurement_end_unix_ms > link.valid_until_unix_ms
        {
            add(
                TraceabilityFindingCode::CalibrationValidityDoesNotCoverMeasurement,
                TraceabilityFindingSeverity::Blocker,
            );
        }
        if link.evidence_level < policy.minimum_calibration_evidence_level {
            add(
                TraceabilityFindingCode::CalibrationEvidenceBelowPolicy,
                TraceabilityFindingSeverity::Degradation,
            );
        }
        total_uncertainty = match total_uncertainty {
            Some(total) => match total.checked_add(link.expanded_uncertainty_upper) {
                Some(sum) => Some(sum),
                None => {
                    add(
                        TraceabilityFindingCode::ExpandedUncertaintyOverflow,
                        TraceabilityFindingSeverity::Blocker,
                    );
                    None
                }
            },
            None => None,
        };
        expected_artifact = link.reference_artifact_digest_sha256.clone();
    }

    if let Some(root) = expected_root {
        if expected_artifact != root.artifact_digest_sha256 {
            add(
                TraceabilityFindingCode::CalibrationRootMismatch,
                TraceabilityFindingSeverity::Blocker,
            );
        }
        if root.evidence_level < policy.minimum_calibration_evidence_level {
            add(
                TraceabilityFindingCode::CalibrationEvidenceBelowPolicy,
                TraceabilityFindingSeverity::Degradation,
            );
        }
    }
    if let Some(total) = total_uncertainty {
        if total > policy.maximum_total_expanded_uncertainty {
            add(
                TraceabilityFindingCode::ExpandedUncertaintyAbovePolicy,
                TraceabilityFindingSeverity::Blocker,
            );
        }
    }

    findings.sort_by(|left, right| {
        left.code
            .cmp(&right.code)
            .then(left.severity.cmp(&right.severity))
    });
    let has_blocker = findings
        .iter()
        .any(|finding| finding.severity == TraceabilityFindingSeverity::Blocker);
    let status = if has_blocker {
        MeasurementTraceabilityStatus::Blocked
    } else if !findings.is_empty() {
        MeasurementTraceabilityStatus::Degraded
    } else {
        MeasurementTraceabilityStatus::MeetsDeclaredTraceabilityPolicy
    };
    MeasurementTraceabilityAssessment {
        observation_id: observation.observation_id.clone(),
        pool_id: observation.pool_id.clone(),
        resource_kind: observation.resource_kind,
        calibration_chain_depth: claim.calibration_chain.len(),
        total_expanded_uncertainty_upper: total_uncertainty,
        findings,
        status,
    }
}

fn digest_traceability_plan(plan: &MeasurementTraceabilityPlan) -> String {
    let mut writer = HashWriter::new("symthaea.sx036.traceability-plan.v1");
    writer.u64(u64::from(plan.schema_version));
    writer.string(&plan.mission_id);
    writer.string(&plan.configuration_revision);
    writer.string(&plan.measurement_plan_digest_sha256);
    writer.string(&plan.measurement_receipt_digest_sha256);
    writer.u64(plan.policy.max_calibration_chain_depth as u64);
    writer.u64(plan.policy.maximum_total_expanded_uncertainty);
    writer.u64(plan.policy.maximum_inter_sample_gap_ms);
    writer.string(evidence_tag(plan.policy.minimum_calibration_evidence_level));
    writer.boolean(plan.policy.require_zero_dropped_samples);

    let mut standards: Vec<_> = plan.policy.reference_standards.iter().collect();
    standards.sort_by(|left, right| left.resource_kind.cmp(&right.resource_kind));
    writer.usize(standards.len());
    for standard in standards {
        writer.string(resource_kind_tag(standard.resource_kind));
        writer.string(&standard.artifact_digest_sha256);
        writer.string(evidence_tag(standard.evidence_level));
    }

    let mut claims: Vec<_> = plan.claims.iter().collect();
    claims.sort_by(|left, right| left.observation_id.cmp(&right.observation_id));
    writer.usize(claims.len());
    for claim in claims {
        writer.string(&claim.observation_id);
        writer.string(&claim.pool_id);
        writer.string(resource_kind_tag(claim.resource_kind));
        writer.string(&claim.provider_component_id);
        writer.string(&claim.provider_hardware_revision);
        writer.u64(claim.window.start_ms);
        writer.u64(claim.window.end_ms);
        writer.string(&claim.instrument_id);
        writer.string(&claim.instrument_artifact_digest_sha256);
        writer.string(&claim.instrument_hardware_revision);
        writer.string(&claim.firmware_digest_sha256);
        writer.string(&claim.raw_trace_digest_sha256);
        writer.u64(claim.measurement_start_unix_ms);
        writer.u64(claim.measurement_end_unix_ms);
        writer.u64(claim.sample_sequence_first);
        writer.u64(claim.sample_sequence_last);
        writer.u64(claim.sample_count);
        writer.u64(claim.dropped_sample_count);
        writer.u64(claim.maximum_observed_inter_sample_gap_ms);
        writer.usize(claim.calibration_chain.len());
        for link in &claim.calibration_chain {
            writer.string(&link.calibrated_artifact_digest_sha256);
            writer.string(&link.calibrated_hardware_revision);
            writer.string(&link.reference_artifact_digest_sha256);
            writer.string(&link.certificate_digest_sha256);
            writer.string(resource_kind_tag(link.resource_kind));
            writer.u64(link.valid_from_unix_ms);
            writer.u64(link.valid_until_unix_ms);
            writer.u64(link.expanded_uncertainty_upper);
            writer.string(evidence_tag(link.evidence_level));
        }
    }
    writer.finish()
}

fn digest_report(report: &MeasurementTraceabilityReport) -> String {
    let mut writer = HashWriter::new("symthaea.sx036.traceability-report.v1");
    writer.u64(u64::from(report.schema_version));
    writer.string(&report.mission_id);
    writer.string(&report.configuration_revision);
    writer.string(&report.measurement_plan_digest_sha256);
    writer.string(&report.measurement_receipt_digest_sha256);
    writer.string(&report.traceability_plan_digest_sha256);
    writer.string(trace_status_tag(report.status));
    writer.boolean(report.represents_physical_qualification);
    writer.usize(report.observation_count);
    writer.usize(report.policy_compliant_count);
    writer.usize(report.degraded_count);
    writer.usize(report.blocked_count);
    writer.usize(report.assessments.len());
    for assessment in &report.assessments {
        writer.string(&assessment.observation_id);
        writer.string(&assessment.pool_id);
        writer.string(resource_kind_tag(assessment.resource_kind));
        writer.usize(assessment.calibration_chain_depth);
        writer.option_u64(assessment.total_expanded_uncertainty_upper);
        writer.string(trace_status_tag(assessment.status));
        writer.usize(assessment.findings.len());
        for finding in &assessment.findings {
            write_finding(&mut writer, finding);
        }
    }
    writer.usize(report.findings.len());
    for finding in &report.findings {
        write_finding(&mut writer, finding);
    }
    writer.finish()
}

fn write_finding(writer: &mut HashWriter, finding: &TraceabilityFinding) {
    writer.string(&finding.observation_id);
    writer.string(finding_code_tag(finding.code));
    writer.string(finding_severity_tag(finding.severity));
}

fn digest_receipt(receipt: &MeasurementTraceabilityReceipt) -> String {
    let mut writer = HashWriter::new("symthaea.sx036.traceability-receipt.v1");
    writer.string(&receipt.measurement_plan_digest_sha256);
    writer.string(&receipt.measurement_receipt_digest_sha256);
    writer.string(&receipt.traceability_plan_digest_sha256);
    writer.string(&receipt.report_digest_sha256);
    writer.finish()
}

fn clean_identifier(value: &str) -> bool {
    !value.trim().is_empty() && value.trim() == value
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn evidence_tag(level: SuitEvidenceLevel) -> &'static str {
    match level {
        SuitEvidenceLevel::Declared => "declared",
        SuitEvidenceLevel::Simulated => "simulated",
        SuitEvidenceLevel::BenchTested => "bench-tested",
        SuitEvidenceLevel::HumanInLoop => "human-in-loop",
        SuitEvidenceLevel::Qualified => "qualified",
    }
}

fn resource_kind_tag(kind: PeakResourceKind) -> &'static str {
    match kind {
        PeakResourceKind::ElectricalPowerMilliwatts => "electrical-power-milliwatts",
        PeakResourceKind::CoolantFlowMicrolitersPerSecond => "coolant-flow-microliters-per-second",
        PeakResourceKind::DataRateBytesPerSecond => "data-rate-bytes-per-second",
        PeakResourceKind::ComputeBudgetUnitsPerSecond => "compute-budget-units-per-second",
        PeakResourceKind::SharedEquipmentSlots => "shared-equipment-slots",
    }
}

fn trace_status_tag(status: MeasurementTraceabilityStatus) -> &'static str {
    match status {
        MeasurementTraceabilityStatus::Blocked => "blocked",
        MeasurementTraceabilityStatus::Degraded => "degraded",
        MeasurementTraceabilityStatus::MeetsDeclaredTraceabilityPolicy => {
            "meets-declared-traceability-policy"
        }
    }
}

fn finding_code_tag(code: TraceabilityFindingCode) -> &'static str {
    match code {
        TraceabilityFindingCode::ObservationBindingMismatch => "observation-binding-mismatch",
        TraceabilityFindingCode::InvalidArtifactDigest => "invalid-artifact-digest",
        TraceabilityFindingCode::InvalidMeasurementTimestamp => "invalid-measurement-timestamp",
        TraceabilityFindingCode::InvalidCalibrationValidityInterval => {
            "invalid-calibration-validity-interval"
        }
        TraceabilityFindingCode::CalibrationChainEmpty => "calibration-chain-empty",
        TraceabilityFindingCode::CalibrationChainTooDeep => "calibration-chain-too-deep",
        TraceabilityFindingCode::CalibrationChainDiscontinuous => {
            "calibration-chain-discontinuous"
        }
        TraceabilityFindingCode::CalibrationChainCycle => "calibration-chain-cycle",
        TraceabilityFindingCode::CalibrationRootMissing => "calibration-root-missing",
        TraceabilityFindingCode::CalibrationRootMismatch => "calibration-root-mismatch",
        TraceabilityFindingCode::CalibrationValidityDoesNotCoverMeasurement => {
            "calibration-validity-does-not-cover-measurement"
        }
        TraceabilityFindingCode::CalibrationResourceKindMismatch => {
            "calibration-resource-kind-mismatch"
        }
        TraceabilityFindingCode::CalibrationEvidenceBelowPolicy => {
            "calibration-evidence-below-policy"
        }
        TraceabilityFindingCode::ExpandedUncertaintyOverflow => "expanded-uncertainty-overflow",
        TraceabilityFindingCode::ExpandedUncertaintyAbovePolicy => {
            "expanded-uncertainty-above-policy"
        }
        TraceabilityFindingCode::SampleSequenceInvalid => "sample-sequence-invalid",
        TraceabilityFindingCode::SampleCountMismatch => "sample-count-mismatch",
        TraceabilityFindingCode::DroppedSamplesObserved => "dropped-samples-observed",
        TraceabilityFindingCode::InterSampleGapAbovePolicy => "inter-sample-gap-above-policy",
    }
}

fn finding_severity_tag(severity: TraceabilityFindingSeverity) -> &'static str {
    match severity {
        TraceabilityFindingSeverity::Degradation => "degradation",
        TraceabilityFindingSeverity::Blocker => "blocker",
    }
}

struct HashWriter(Sha256);

impl HashWriter {
    fn new(domain: &str) -> Self {
        let mut writer = Self(Sha256::new());
        writer.string(domain);
        writer
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.0.update((bytes.len() as u64).to_be_bytes());
        self.0.update(bytes);
    }

    fn string(&mut self, value: &str) {
        self.raw(value.as_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.0.update(value.to_be_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn boolean(&mut self, value: bool) {
        self.0.update([u8::from(value)]);
    }

    fn option_u64(&mut self, value: Option<u64>) {
        match value {
            Some(value) => {
                self.boolean(true);
                self.u64(value);
            }
            None => self.boolean(false),
        }
    }

    fn finish(self) -> String {
        format!("sha256:{:x}", self.0.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suit_peak_reservations::{MissionTimeWindow, PeakQuantityInterval};
    use crate::suit_peak_usage_measurements::{
        PeakResourceUsageObservation, ResourceMeasurementMethod,
    };

    fn sha(ch: char) -> String {
        format!("sha256:{}", ch.to_string().repeat(64))
    }

    fn fixture() -> (
        PeakResourceUsageMeasurementPlan,
        PeakResourceUsageMeasurementReceipt,
        MeasurementTraceabilityPlan,
    ) {
        let observation = PeakResourceUsageObservation {
            observation_id: "obs-power-01".into(),
            pool_id: "pool-power".into(),
            resource_kind: PeakResourceKind::ElectricalPowerMilliwatts,
            provider_component_id: "power-controller".into(),
            provider_hardware_revision: "power-r3".into(),
            window: MissionTimeWindow {
                start_ms: 0,
                end_ms: 1_000,
            },
            observed_peak: PeakQuantityInterval {
                lower: 100,
                nominal: 110,
                upper: 120,
            },
            sample_count: 10,
            sampling_interval_ms: 100,
            measurement_method: ResourceMeasurementMethod::BenchInstrumented,
            evidence_level: SuitEvidenceLevel::BenchTested,
            evidence_reference: "artifact:synthetic-observation".into(),
        };
        let measurement_plan = PeakResourceUsageMeasurementPlan {
            schema_version: 1,
            mission_id: "synthetic-mission".into(),
            configuration_revision: "cfg-r1".into(),
            expected_configuration_digest_sha256: sha('1'),
            expected_task_plan_digest_sha256: sha('2'),
            expected_reservation_plan_digest_sha256: sha('3'),
            expected_reservation_report_digest_sha256: sha('4'),
            expected_reservation_receipt_digest_sha256: sha('5'),
            minimum_evidence_level: SuitEvidenceLevel::BenchTested,
            minimum_sample_count: 10,
            maximum_sampling_interval_ms: 100,
            observations: vec![observation],
        };
        let measurement_receipt = PeakResourceUsageMeasurementReceipt {
            configuration_digest_sha256: sha('1'),
            task_plan_digest_sha256: sha('2'),
            task_receipt_digest_sha256: sha('6'),
            reservation_plan_digest_sha256: sha('3'),
            reservation_receipt_digest_sha256: sha('7'),
            measurement_plan_digest_sha256: sha('f'),
            report_digest_sha256: sha('8'),
            receipt_digest_sha256: sha('9'),
        };
        let claim = MeasurementTraceabilityClaim {
            observation_id: "obs-power-01".into(),
            pool_id: "pool-power".into(),
            resource_kind: PeakResourceKind::ElectricalPowerMilliwatts,
            provider_component_id: "power-controller".into(),
            provider_hardware_revision: "power-r3".into(),
            window: MissionTimeWindow {
                start_ms: 0,
                end_ms: 1_000,
            },
            instrument_id: "bench-meter-7".into(),
            instrument_artifact_digest_sha256: sha('a'),
            instrument_hardware_revision: "meter-r2".into(),
            firmware_digest_sha256: sha('b'),
            raw_trace_digest_sha256: sha('c'),
            measurement_start_unix_ms: 2_000,
            measurement_end_unix_ms: 3_000,
            sample_sequence_first: 1,
            sample_sequence_last: 10,
            sample_count: 10,
            dropped_sample_count: 0,
            maximum_observed_inter_sample_gap_ms: 100,
            calibration_chain: vec![
                CalibrationTraceLink {
                    calibrated_artifact_digest_sha256: sha('a'),
                    calibrated_hardware_revision: "meter-r2".into(),
                    reference_artifact_digest_sha256: sha('d'),
                    certificate_digest_sha256: sha('e'),
                    resource_kind: PeakResourceKind::ElectricalPowerMilliwatts,
                    valid_from_unix_ms: 1_000,
                    valid_until_unix_ms: 4_000,
                    expanded_uncertainty_upper: 3,
                    evidence_level: SuitEvidenceLevel::BenchTested,
                },
                CalibrationTraceLink {
                    calibrated_artifact_digest_sha256: sha('d'),
                    calibrated_hardware_revision: "reference-r1".into(),
                    reference_artifact_digest_sha256: sha('f'),
                    certificate_digest_sha256: sha('0'),
                    resource_kind: PeakResourceKind::ElectricalPowerMilliwatts,
                    valid_from_unix_ms: 1_000,
                    valid_until_unix_ms: 4_000,
                    expanded_uncertainty_upper: 4,
                    evidence_level: SuitEvidenceLevel::BenchTested,
                },
            ],
        };
        let traceability_plan = MeasurementTraceabilityPlan {
            schema_version: 1,
            mission_id: "synthetic-mission".into(),
            configuration_revision: "cfg-r1".into(),
            measurement_plan_digest_sha256: sha('f'),
            measurement_receipt_digest_sha256: sha('9'),
            policy: MeasurementTraceabilityPolicy {
                max_calibration_chain_depth: 4,
                maximum_total_expanded_uncertainty: 10,
                maximum_inter_sample_gap_ms: 100,
                minimum_calibration_evidence_level: SuitEvidenceLevel::BenchTested,
                require_zero_dropped_samples: true,
                reference_standards: vec![
                    TraceabilityReferenceStandard {
                        resource_kind: PeakResourceKind::ElectricalPowerMilliwatts,
                        artifact_digest_sha256: sha('f'),
                        evidence_level: SuitEvidenceLevel::BenchTested,
                    },
                    TraceabilityReferenceStandard {
                        resource_kind: PeakResourceKind::SharedEquipmentSlots,
                        artifact_digest_sha256: sha('1'),
                        evidence_level: SuitEvidenceLevel::BenchTested,
                    },
                ],
            },
            claims: vec![claim],
        };
        (measurement_plan, measurement_receipt, traceability_plan)
    }

    fn evaluate(
        fixture: &(
            PeakResourceUsageMeasurementPlan,
            PeakResourceUsageMeasurementReceipt,
            MeasurementTraceabilityPlan,
        ),
    ) -> (MeasurementTraceabilityReport, MeasurementTraceabilityReceipt) {
        evaluate_measurement_traceability(&fixture.0, &fixture.1, &fixture.2).unwrap()
    }

    fn has_code(
        assessment: &MeasurementTraceabilityAssessment,
        code: TraceabilityFindingCode,
    ) -> bool {
        assessment.findings.iter().any(|finding| finding.code == code)
    }

    #[test]
    fn valid_calibration_chain_meets_declared_policy() {
        let fixture = fixture();
        let (report, _) = evaluate(&fixture);
        assert_eq!(
            report.status,
            MeasurementTraceabilityStatus::MeetsDeclaredTraceabilityPolicy
        );
        assert_eq!(report.assessments[0].total_expanded_uncertainty_upper, Some(7));
        assert!(!report.represents_physical_qualification);
    }

    #[test]
    fn broken_chain_is_blocked() {
        let mut fixture = fixture();
        fixture.2.claims[0].calibration_chain[1].calibrated_artifact_digest_sha256 = sha('7');
        let (report, _) = evaluate(&fixture);
        assert_eq!(report.status, MeasurementTraceabilityStatus::Blocked);
        assert!(has_code(
            &report.assessments[0],
            TraceabilityFindingCode::CalibrationChainDiscontinuous
        ));
    }

    #[test]
    fn missing_resource_specific_root_is_blocked() {
        let mut fixture = fixture();
        fixture.2.policy.reference_standards = vec![TraceabilityReferenceStandard {
            resource_kind: PeakResourceKind::SharedEquipmentSlots,
            artifact_digest_sha256: sha('1'),
            evidence_level: SuitEvidenceLevel::BenchTested,
        }];
        let (report, _) = evaluate(&fixture);
        assert!(has_code(
            &report.assessments[0],
            TraceabilityFindingCode::CalibrationRootMissing
        ));
        assert_eq!(report.status, MeasurementTraceabilityStatus::Blocked);
    }

    #[test]
    fn cyclic_calibration_chain_is_blocked() {
        let mut fixture = fixture();
        fixture.2.claims[0].calibration_chain[1].reference_artifact_digest_sha256 = sha('a');
        let (report, _) = evaluate(&fixture);
        assert!(has_code(
            &report.assessments[0],
            TraceabilityFindingCode::CalibrationChainCycle
        ));
        assert_eq!(report.status, MeasurementTraceabilityStatus::Blocked);
    }

    #[test]
    fn calibration_links_must_end_at_the_declared_root() {
        let mut fixture = fixture();
        fixture.2.claims[0].calibration_chain[1].reference_artifact_digest_sha256 = sha('8');
        let (report, _) = evaluate(&fixture);
        assert!(has_code(
            &report.assessments[0],
            TraceabilityFindingCode::CalibrationRootMismatch
        ));
    }

    #[test]
    fn expired_calibration_cannot_cover_a_measurement() {
        let mut fixture = fixture();
        fixture.2.claims[0].calibration_chain[0].valid_until_unix_ms = 2_500;
        let (report, _) = evaluate(&fixture);
        assert!(has_code(
            &report.assessments[0],
            TraceabilityFindingCode::CalibrationValidityDoesNotCoverMeasurement
        ));
        assert_eq!(report.status, MeasurementTraceabilityStatus::Blocked);
    }

    #[test]
    fn uncertainty_ceiling_is_enforced() {
        let mut fixture = fixture();
        fixture.2.policy.maximum_total_expanded_uncertainty = 6;
        let (report, _) = evaluate(&fixture);
        assert!(has_code(
            &report.assessments[0],
            TraceabilityFindingCode::ExpandedUncertaintyAbovePolicy
        ));
        assert_eq!(report.status, MeasurementTraceabilityStatus::Blocked);
    }

    #[test]
    fn uncertainty_overflow_is_blocked() {
        let mut fixture = fixture();
        fixture.2.claims[0].calibration_chain[0].expanded_uncertainty_upper = u64::MAX;
        let (report, _) = evaluate(&fixture);
        assert!(has_code(
            &report.assessments[0],
            TraceabilityFindingCode::ExpandedUncertaintyOverflow
        ));
        assert_eq!(report.status, MeasurementTraceabilityStatus::Blocked);
    }

    #[test]
    fn strict_policy_blocks_dropped_samples() {
        let mut fixture = fixture();
        fixture.2.claims[0].sample_sequence_last = 11;
        fixture.2.claims[0].dropped_sample_count = 1;
        let (report, _) = evaluate(&fixture);
        assert!(has_code(
            &report.assessments[0],
            TraceabilityFindingCode::DroppedSamplesObserved
        ));
        assert_eq!(report.status, MeasurementTraceabilityStatus::Blocked);
    }

    #[test]
    fn impossible_sequence_arithmetic_is_blocked() {
        let mut fixture = fixture();
        fixture.2.claims[0].sample_sequence_last = 50;
        let (report, _) = evaluate(&fixture);
        assert!(has_code(
            &report.assessments[0],
            TraceabilityFindingCode::SampleSequenceInvalid
        ));
    }

    #[test]
    fn excessive_gap_is_blocked() {
        let mut fixture = fixture();
        fixture.2.claims[0].maximum_observed_inter_sample_gap_ms = 101;
        let (report, _) = evaluate(&fixture);
        assert!(has_code(
            &report.assessments[0],
            TraceabilityFindingCode::InterSampleGapAbovePolicy
        ));
    }

    #[test]
    fn provider_or_revision_mismatch_is_blocked() {
        let mut fixture = fixture();
        fixture.2.claims[0].provider_hardware_revision = "power-r2".into();
        let (report, _) = evaluate(&fixture);
        assert!(has_code(
            &report.assessments[0],
            TraceabilityFindingCode::ObservationBindingMismatch
        ));
    }

    #[test]
    fn sample_count_must_match_the_sx035_observation() {
        let mut fixture = fixture();
        fixture.2.claims[0].sample_count = 9;
        fixture.2.claims[0].sample_sequence_last = 9;
        let (report, _) = evaluate(&fixture);
        assert!(has_code(
            &report.assessments[0],
            TraceabilityFindingCode::SampleCountMismatch
        ));
        assert_eq!(report.status, MeasurementTraceabilityStatus::Blocked);
    }

    #[test]
    fn observation_coverage_must_be_exact() {
        let mut fixture = fixture();
        fixture.2.claims.clear();
        assert_eq!(
            evaluate_measurement_traceability(&fixture.0, &fixture.1, &fixture.2),
            Err(MeasurementTraceabilityError::EmptyClaimSet)
        );
    }

    #[test]
    fn stale_measurement_receipt_pin_is_rejected() {
        let mut fixture = fixture();
        fixture.2.measurement_receipt_digest_sha256 = sha('4');
        assert_eq!(
            evaluate_measurement_traceability(&fixture.0, &fixture.1, &fixture.2),
            Err(MeasurementTraceabilityError::MeasurementReceiptDigestMismatch)
        );
    }

    #[test]
    fn lower_calibration_evidence_degrades_without_promotion() {
        let mut fixture = fixture();
        fixture.2.claims[0].calibration_chain[0].evidence_level = SuitEvidenceLevel::Simulated;
        let (report, _) = evaluate(&fixture);
        assert!(has_code(
            &report.assessments[0],
            TraceabilityFindingCode::CalibrationEvidenceBelowPolicy
        ));
        assert_eq!(report.status, MeasurementTraceabilityStatus::Degraded);
    }

    #[test]
    fn claim_and_reference_order_do_not_change_digests_or_report() {
        let mut baseline = fixture();
        let mut second_observation = baseline.0.observations[0].clone();
        second_observation.observation_id = "obs-power-02".into();
        second_observation.pool_id = "pool-power-2".into();
        baseline.0.observations.push(second_observation);
        let mut second_claim = baseline.2.claims[0].clone();
        second_claim.observation_id = "obs-power-02".into();
        second_claim.pool_id = "pool-power-2".into();
        baseline.2.claims.push(second_claim);

        let mut permuted = baseline.clone();
        permuted.0.observations.reverse();
        permuted.2.claims.reverse();
        permuted.2.policy.reference_standards.reverse();
        assert_eq!(evaluate(&baseline), evaluate(&permuted));
    }

    #[test]
    fn receipt_verification_detects_tampering() {
        let fixture = fixture();
        let (report, receipt) = evaluate(&fixture);
        assert!(verify_measurement_traceability_receipt(
            &fixture.0, &fixture.1, &fixture.2, &report, &receipt
        ));
        let mut altered = receipt;
        altered.report_digest_sha256 = sha('a');
        assert!(!verify_measurement_traceability_receipt(
            &fixture.0, &fixture.1, &fixture.2, &report, &altered
        ));
    }

    #[test]
    fn seeded_two_thousand_case_uncertainty_boundary_corpus() {
        for seed in 0_u64..2_000 {
            let mut fixture = fixture();
            let first = (seed * 37) % 17;
            let second = (seed * 19) % 17;
            fixture.2.claims[0].calibration_chain[0].expanded_uncertainty_upper = first;
            fixture.2.claims[0].calibration_chain[1].expanded_uncertainty_upper = second;
            let expected_blocked = first.checked_add(second).is_none_or(|sum| sum > 10);
            let (report, receipt) = evaluate(&fixture);
            assert_eq!(
                report.status == MeasurementTraceabilityStatus::Blocked,
                expected_blocked,
                "unexpected status at seed {seed}"
            );
            assert_eq!(
                receipt.report_digest_sha256,
                digest_report(&report),
                "report digest mismatch at seed {seed}"
            );
            assert!(verify_measurement_traceability_receipt(
                &fixture.0, &fixture.1, &fixture.2, &report, &receipt
            ));
        }
    }
}
