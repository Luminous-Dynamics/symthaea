// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adapters from domain-awareness assurance reports into hysteretic requalification.

#![deny(unsafe_code)]

use symthaea_assurance_requalification::{AssuranceSample, AssuranceState};
use symthaea_domain_awareness_common_cause::CommonCauseQualifiedTrackAssurance;
use symthaea_domain_awareness_model_assurance::OddModelAssuranceEvidence;
use symthaea_model_assurance::{ModelAssuranceError, ModelAssuranceReport, ModelAssuranceStatus};
use symthaea_sensor_common_cause::CommonCauseDiversityReport;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RequalificationAdapterError {
    InvalidEvidenceReference,
    ModelAssurance(ModelAssuranceError),
}

impl From<ModelAssuranceError> for RequalificationAdapterError {
    fn from(value: ModelAssuranceError) -> Self {
        Self::ModelAssurance(value)
    }
}

/// Convert a generic model-assurance report into one requalification sample.
///
/// A synthetically inconsistent `Aligned` report that still contains issues is
/// conservatively mapped to `Incomplete` rather than being allowed to improve recovery.
pub fn sample_from_model_assurance(
    report: &ModelAssuranceReport,
) -> Result<AssuranceSample, RequalificationAdapterError> {
    let odd = OddModelAssuranceEvidence::from_report(report)?;
    let state = if report.status == ModelAssuranceStatus::Aligned && !report.issues.is_empty() {
        AssuranceState::Incomplete
    } else {
        match report.status {
            ModelAssuranceStatus::Aligned => AssuranceState::Nominal,
            ModelAssuranceStatus::Restricted => AssuranceState::Restricted,
            ModelAssuranceStatus::Unsafe => AssuranceState::Unsafe,
            ModelAssuranceStatus::Incomplete => AssuranceState::Incomplete,
        }
    };
    Ok(AssuranceSample {
        sample_id: format!(
            "model-assurance:{}:{}",
            report.policy_id, report.assessed_at_ms
        ),
        observed_at_ms: report.assessed_at_ms,
        state,
        evidence_refs: vec![odd.evidence_ref],
    })
}

/// Convert common-cause diversity evidence into one requalification sample.
///
/// The adapter independently checks report consistency instead of trusting only
/// `requires_fail_closed`.
pub fn sample_from_common_cause(
    report: &CommonCauseDiversityReport,
    observed_at_ms: u64,
    evidence_ref: impl Into<String>,
) -> Result<AssuranceSample, RequalificationAdapterError> {
    let evidence_ref = evidence_ref.into();
    if evidence_ref.trim().is_empty() {
        return Err(RequalificationAdapterError::InvalidEvidenceReference);
    }
    let fail_closed = report.requires_fail_closed
        || !report.issues.is_empty()
        || report.accepted_physical_sources == 0
        || report.profiled_physical_sources != report.accepted_physical_sources;

    Ok(AssuranceSample {
        sample_id: format!("common-cause:{}:{observed_at_ms}", report.policy_id),
        observed_at_ms,
        state: if fail_closed {
            AssuranceState::FailClosed
        } else {
            AssuranceState::Nominal
        },
        evidence_refs: vec![evidence_ref],
    })
}

/// Convert common-cause-qualified track assurance into one recovery sample.
///
/// A downgraded or internally inconsistent qualification is `Restricted`; a clean
/// preserved qualification is `Nominal`. This is intentionally less severe than
/// the deployment-wide common-cause report adapter above.
pub fn sample_from_qualified_track(
    report: &CommonCauseQualifiedTrackAssurance,
    observed_at_ms: u64,
    evidence_ref: impl Into<String>,
) -> Result<AssuranceSample, RequalificationAdapterError> {
    let evidence_ref = evidence_ref.into();
    if evidence_ref.trim().is_empty() {
        return Err(RequalificationAdapterError::InvalidEvidenceReference);
    }
    let restricted = report.was_downgraded() || !report.issues.is_empty();
    Ok(AssuranceSample {
        sample_id: format!(
            "track-common-cause:{:?}:{:?}:{observed_at_ms}",
            report.original_level, report.qualified_level
        ),
        observed_at_ms,
        state: if restricted {
            AssuranceState::Restricted
        } else {
            AssuranceState::Nominal
        },
        evidence_refs: vec![evidence_ref],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use symthaea_assurance_requalification::{
        RequalificationAuthorization, RequalificationGate, RequalificationPhase,
        RequalificationPolicy,
    };
    use symthaea_domain_awareness_vision::TrackAssuranceLevel;
    use symthaea_model_assurance::ModelAssuranceIssue;
    use symthaea_sensor_common_cause::{CommonCauseDiversityReport, FaultDomainKind};

    fn model_report(status: ModelAssuranceStatus, at: u64) -> ModelAssuranceReport {
        ModelAssuranceReport {
            schema_version: "1".into(),
            policy_id: "camera-model-v1".into(),
            assessed_at_ms: at,
            status,
            signals: Vec::new(),
            issues: Vec::new(),
        }
    }

    fn policy() -> RequalificationPolicy {
        RequalificationPolicy {
            schema_version: "1".into(),
            policy_id: "requal-v1".into(),
            minimum_consecutive_nominal_samples: 3,
            minimum_nominal_span_ms: 2_000,
            maximum_nominal_sample_gap_ms: 1_500,
            maximum_authorization_age_ms: 10_000,
        }
    }

    fn authorization(at: u64) -> RequalificationAuthorization {
        RequalificationAuthorization {
            authorization_id: "auth-1".into(),
            procedure_ref: "procedure:requal-v1".into(),
            reviewer_ref: "reviewer:safety".into(),
            authorized_at_ms: at,
            evidence_refs: vec!["review:auth-1".into()],
        }
    }

    #[test]
    fn unsafe_model_latches_and_one_aligned_report_cannot_recover() {
        let mut gate = RequalificationGate::new(policy()).unwrap();
        gate.observe(sample_from_model_assurance(&model_report(ModelAssuranceStatus::Unsafe, 1_000)).unwrap());
        gate.begin_requalification(authorization(1_001)).unwrap();
        let report = gate.observe(
            sample_from_model_assurance(&model_report(ModelAssuranceStatus::Aligned, 2_000)).unwrap(),
        );
        assert_eq!(report.effective_state, AssuranceState::Unsafe);
        assert_eq!(report.phase, RequalificationPhase::Requalifying);
    }

    #[test]
    fn sustained_aligned_model_evidence_can_complete_reviewed_requalification() {
        let mut gate = RequalificationGate::new(policy()).unwrap();
        gate.observe(sample_from_model_assurance(&model_report(ModelAssuranceStatus::Unsafe, 1_000)).unwrap());
        gate.begin_requalification(authorization(1_001)).unwrap();
        gate.observe(sample_from_model_assurance(&model_report(ModelAssuranceStatus::Aligned, 2_000)).unwrap());
        gate.observe(sample_from_model_assurance(&model_report(ModelAssuranceStatus::Aligned, 3_000)).unwrap());
        let report = gate.observe(
            sample_from_model_assurance(&model_report(ModelAssuranceStatus::Aligned, 4_000)).unwrap(),
        );
        assert_eq!(report.effective_state, AssuranceState::Nominal);
        assert_eq!(report.phase, RequalificationPhase::Nominal);
    }

    #[test]
    fn aligned_model_report_with_issues_is_incomplete() {
        let mut report = model_report(ModelAssuranceStatus::Aligned, 1_000);
        report.issues.push(ModelAssuranceIssue::MissingEvidence("sample-x".into()));
        let sample = sample_from_model_assurance(&report).unwrap();
        assert_eq!(sample.state, AssuranceState::Incomplete);
    }

    #[test]
    fn common_cause_failure_maps_to_fail_closed_and_clean_report_is_only_nominal_evidence() {
        let failed = CommonCauseDiversityReport {
            policy_id: "diversity-v1".into(),
            accepted_physical_sources: 2,
            profiled_physical_sources: 1,
            distinct_domains: BTreeMap::from([(FaultDomainKind::Power, 1)]),
            issues: Vec::new(),
            requires_fail_closed: false,
        };
        let sample = sample_from_common_cause(&failed, 1_000, "evidence:diversity-1").unwrap();
        assert_eq!(sample.state, AssuranceState::FailClosed);

        let clean = CommonCauseDiversityReport {
            policy_id: "diversity-v1".into(),
            accepted_physical_sources: 2,
            profiled_physical_sources: 2,
            distinct_domains: BTreeMap::from([(FaultDomainKind::Power, 2)]),
            issues: Vec::new(),
            requires_fail_closed: false,
        };
        let clean_sample = sample_from_common_cause(&clean, 2_000, "evidence:diversity-2").unwrap();
        assert_eq!(clean_sample.state, AssuranceState::Nominal);

        let mut gate = RequalificationGate::new(policy()).unwrap();
        gate.observe(sample);
        let gated = gate.observe(clean_sample);
        assert_eq!(gated.effective_state, AssuranceState::FailClosed);
    }

    #[test]
    fn downgraded_track_maps_to_restricted() {
        let report = CommonCauseQualifiedTrackAssurance {
            original_level: TrackAssuranceLevel::Corroborated,
            qualified_level: TrackAssuranceLevel::Persistent,
            physical_sources_reported_by_track: 2,
            physical_sources_checked_for_common_cause: 2,
            issues: Vec::new(),
        };
        let sample = sample_from_qualified_track(&report, 1_000, "evidence:track-1").unwrap();
        assert_eq!(sample.state, AssuranceState::Restricted);
    }

    #[test]
    fn preserved_track_maps_to_nominal() {
        let report = CommonCauseQualifiedTrackAssurance {
            original_level: TrackAssuranceLevel::Corroborated,
            qualified_level: TrackAssuranceLevel::Corroborated,
            physical_sources_reported_by_track: 2,
            physical_sources_checked_for_common_cause: 2,
            issues: Vec::new(),
        };
        let sample = sample_from_qualified_track(&report, 1_000, "evidence:track-1").unwrap();
        assert_eq!(sample.state, AssuranceState::Nominal);
    }
}
