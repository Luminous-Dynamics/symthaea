// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audit binding between Golden Incident run artifacts and aggregate qualification results.
//!
//! Aggregate metrics are useful only when they can be traced back to the exact case,
//! solver view, transcript, submission, and run context that produced them.

use crate::golden_run_protocol::{
    golden_grading_artifact_digest_v1, GoldenRunProtocolErrorV1, GoldenRunTranscriptV1,
    GoldenSolverSubmissionV1,
};
use crate::golden_solver_view_v2::GoldenSolverIncidentV2;
use crate::it_qualification::{
    ItQualificationErrorV1, ItQualificationMatrixV1, ItQualificationResultV1,
    QualificationCaseKeyV1, QualificationMetricsV1, QualificationResultIdV1,
    QualificationRunContextV1, QualificationRunIdV1,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenQualificationBindingV1 {
    pub result_id: QualificationResultIdV1,
    pub run_id: QualificationRunIdV1,
    pub case_key: QualificationCaseKeyV1,
    pub case_digest: String,
    pub grading_artifact_digest: String,
    pub run_context_digest: String,
    pub metrics_digest: String,
}

impl GoldenQualificationBindingV1 {
    pub fn digest(&self) -> Result<String, GoldenQualificationBindingErrorV1> {
        digest_serializable("symthaea-golden-qualification-binding-v1", self)
    }
}

pub fn golden_run_context_digest_v1(
    context: &QualificationRunContextV1,
) -> Result<String, GoldenQualificationBindingErrorV1> {
    context.validate()?;
    digest_serializable("symthaea-golden-run-context-v1", context)
}

pub fn golden_qualification_metrics_digest_v1(
    metrics: &QualificationMetricsV1,
) -> Result<String, GoldenQualificationBindingErrorV1> {
    metrics.validate()?;
    digest_serializable("symthaea-golden-qualification-metrics-v1", metrics)
}

pub fn bind_golden_qualification_result_v1(
    matrix: &ItQualificationMatrixV1,
    view: &GoldenSolverIncidentV2,
    transcript: &GoldenRunTranscriptV1,
    submission: &GoldenSolverSubmissionV1,
    result: &ItQualificationResultV1,
) -> Result<GoldenQualificationBindingV1, GoldenQualificationBindingErrorV1> {
    result.validate()?;
    transcript.validate_against(view)?;
    submission.validate_against(view, transcript)?;

    if result.run_id != transcript.run_id {
        return Err(GoldenQualificationBindingErrorV1::RunIdMismatch);
    }
    if result.case_key != transcript.case_key {
        return Err(GoldenQualificationBindingErrorV1::CaseKeyMismatch);
    }
    if result.observed_at_unix_ms < submission.submitted_at_unix_ms {
        return Err(GoldenQualificationBindingErrorV1::ResultPredatesSubmission);
    }

    let Some(registered_case_digest) = matrix.case_digest(&result.case_key) else {
        return Err(GoldenQualificationBindingErrorV1::UnknownRegisteredCase(
            result.case_key.clone(),
        ));
    };
    if registered_case_digest != result.case_digest {
        return Err(GoldenQualificationBindingErrorV1::RegisteredCaseDigestMismatch);
    }

    let grading_artifact_digest =
        golden_grading_artifact_digest_v1(view, transcript, submission)?;
    match result.evidence_artifact_digest.as_deref() {
        Some(digest) if digest == grading_artifact_digest => {}
        Some(_) => {
            return Err(GoldenQualificationBindingErrorV1::EvidenceArtifactDigestMismatch)
        }
        None => return Err(GoldenQualificationBindingErrorV1::MissingEvidenceArtifactDigest),
    }

    Ok(GoldenQualificationBindingV1 {
        result_id: result.id.clone(),
        run_id: result.run_id.clone(),
        case_key: result.case_key.clone(),
        case_digest: result.case_digest.clone(),
        grading_artifact_digest,
        run_context_digest: golden_run_context_digest_v1(&result.run_context)?,
        metrics_digest: golden_qualification_metrics_digest_v1(&result.metrics)?,
    })
}

fn digest_serializable<T: Serialize + ?Sized>(
    domain: &'static str,
    value: &T,
) -> Result<String, GoldenQualificationBindingErrorV1> {
    let bytes = serde_json::to_vec(&(domain, value))
        .map_err(|err| GoldenQualificationBindingErrorV1::Serialization(err.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

#[derive(Debug)]
pub enum GoldenQualificationBindingErrorV1 {
    Qualification(ItQualificationErrorV1),
    RunProtocol(GoldenRunProtocolErrorV1),
    Serialization(String),
    RunIdMismatch,
    CaseKeyMismatch,
    UnknownRegisteredCase(QualificationCaseKeyV1),
    RegisteredCaseDigestMismatch,
    MissingEvidenceArtifactDigest,
    EvidenceArtifactDigestMismatch,
    ResultPredatesSubmission,
}

impl fmt::Display for GoldenQualificationBindingErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qualification(err) => write!(f, "invalid qualification result: {err}"),
            Self::RunProtocol(err) => write!(f, "invalid golden run lineage: {err}"),
            Self::Serialization(message) => {
                write!(f, "golden qualification binding serialization failed: {message}")
            }
            Self::RunIdMismatch => write!(f, "qualification result run id does not match transcript"),
            Self::CaseKeyMismatch => write!(f, "qualification result case does not match transcript"),
            Self::UnknownRegisteredCase(key) => write!(
                f,
                "qualification result references unregistered case {} revision {}",
                key.id.0, key.revision
            ),
            Self::RegisteredCaseDigestMismatch => {
                write!(f, "qualification result case digest does not match registered case")
            }
            Self::MissingEvidenceArtifactDigest => {
                write!(f, "qualification result lacks golden grading artifact digest")
            }
            Self::EvidenceArtifactDigestMismatch => {
                write!(f, "qualification result evidence artifact digest does not match golden run")
            }
            Self::ResultPredatesSubmission => {
                write!(f, "qualification result timestamp precedes solver submission")
            }
        }
    }
}

impl Error for GoldenQualificationBindingErrorV1 {}

impl From<ItQualificationErrorV1> for GoldenQualificationBindingErrorV1 {
    fn from(value: ItQualificationErrorV1) -> Self {
        Self::Qualification(value)
    }
}

impl From<GoldenRunProtocolErrorV1> for GoldenQualificationBindingErrorV1 {
    fn from(value: GoldenRunProtocolErrorV1) -> Self {
        Self::RunProtocol(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden_incidents_v2::seed_golden_incidents_v2;
    use crate::golden_run_protocol::{
        golden_solver_view_digest_v1, GoldenFindingDispositionV1,
        GoldenPresentedEvidenceSourceV1, GoldenPresentedEvidenceV1,
        GoldenRunTranscriptV1, GoldenSolverFindingV1, GoldenSolverSubmissionV1,
        GOLDEN_RUN_TRANSCRIPT_SCHEMA_V1, GOLDEN_SOLVER_SUBMISSION_SCHEMA_V1,
    };
    use crate::it_qualification::{
        ItQualificationMatrixV1, QualificationMetricsV1, QualificationResultIdV1,
        QualificationRunContextV1, QualificationRunIdV1,
    };
    use std::collections::BTreeSet;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn fixture() -> (
        ItQualificationMatrixV1,
        GoldenSolverIncidentV2,
        GoldenRunTranscriptV1,
        GoldenSolverSubmissionV1,
        ItQualificationResultV1,
    ) {
        let incident = seed_golden_incidents_v2().unwrap().cases[0].clone();
        let view = incident.solver_view();
        let mut matrix = ItQualificationMatrixV1::new();
        let case = incident.qualification_case().unwrap();
        let case_key = case.key.clone();
        matrix.register_case(case).unwrap();
        let solver_view_digest = golden_solver_view_digest_v1(&view).unwrap();
        let run_id = QualificationRunIdV1("golden-run-1".into());
        let transcript = GoldenRunTranscriptV1 {
            schema_version: GOLDEN_RUN_TRANSCRIPT_SCHEMA_V1.into(),
            run_id: run_id.clone(),
            case_key: case_key.clone(),
            solver_view_digest: solver_view_digest.clone(),
            started_at_unix_ms: 100,
            finished_at_unix_ms: 200,
            presented_evidence: view
                .initial_evidence
                .iter()
                .map(|evidence| GoldenPresentedEvidenceV1 {
                    evidence_id: evidence.id.clone(),
                    source: GoldenPresentedEvidenceSourceV1::InitialFixture,
                    private_artifact_digest: None,
                })
                .collect(),
            diagnostic_requests: Vec::new(),
        };
        let submission = GoldenSolverSubmissionV1 {
            schema_version: GOLDEN_SOLVER_SUBMISSION_SCHEMA_V1.into(),
            run_id: run_id.clone(),
            case_key: case_key.clone(),
            solver_view_digest,
            findings: vec![GoldenSolverFindingV1 {
                finding_id: "finding-1".into(),
                statement: "filesystem namespace pressure is plausible".into(),
                disposition: GoldenFindingDispositionV1::SuspectedFault,
                confidence: 0.7,
                evidence_ids: BTreeSet::from([view.initial_evidence[0].id.clone()]),
            }],
            final_confidence: 0.7,
            abstention: None,
            submitted_at_unix_ms: 210,
        };
        let grading_digest =
            golden_grading_artifact_digest_v1(&view, &transcript, &submission).unwrap();
        let result = ItQualificationResultV1 {
            id: QualificationResultIdV1("result-1".into()),
            run_id,
            case_key: case_key.clone(),
            case_digest: matrix.case_digest(&case_key).unwrap().into(),
            run_context: QualificationRunContextV1 {
                system_revision: "system-rev-1".into(),
                corpus_revision: "golden-v2-rev-1".into(),
                environment_digest: digest('a'),
                toolchain_digest: Some(digest('b')),
                model_profile: "support-eval".into(),
                harness_version: "golden-harness-v1".into(),
            },
            observed_at_unix_ms: 220,
            metrics: QualificationMetricsV1 {
                correctness: 0.9,
                calibration_error: 0.1,
                evidence_traceability: 1.0,
                applicability_accuracy: 1.0,
                unsafe_action_rate: 0.0,
                abstention_quality: Some(0.9),
                diagnostic_efficiency: Some(0.8),
            },
            evidence_artifact_digest: Some(grading_digest),
        };
        (matrix, view, transcript, submission, result)
    }

    #[test]
    fn exact_result_lineage_binds_successfully() {
        let (matrix, view, transcript, submission, result) = fixture();
        let binding = bind_golden_qualification_result_v1(
            &matrix,
            &view,
            &transcript,
            &submission,
            &result,
        )
        .unwrap();
        assert_eq!(binding.result_id, result.id);
        assert_eq!(binding.run_id, result.run_id);
        assert_eq!(binding.case_digest, result.case_digest);
        assert_eq!(binding.digest().unwrap().len(), 64);
    }

    #[test]
    fn wrong_grading_artifact_cannot_back_metrics() {
        let (matrix, view, transcript, submission, mut result) = fixture();
        result.evidence_artifact_digest = Some(digest('f'));
        assert!(matches!(
            bind_golden_qualification_result_v1(
                &matrix,
                &view,
                &transcript,
                &submission,
                &result,
            ),
            Err(GoldenQualificationBindingErrorV1::EvidenceArtifactDigestMismatch)
        ));
    }

    #[test]
    fn result_cannot_precede_solver_submission() {
        let (matrix, view, transcript, submission, mut result) = fixture();
        result.observed_at_unix_ms = submission.submitted_at_unix_ms - 1;
        assert!(matches!(
            bind_golden_qualification_result_v1(
                &matrix,
                &view,
                &transcript,
                &submission,
                &result,
            ),
            Err(GoldenQualificationBindingErrorV1::ResultPredatesSubmission)
        ));
    }

    #[test]
    fn registered_case_digest_is_not_optional() {
        let (_matrix, view, transcript, submission, result) = fixture();
        let empty = ItQualificationMatrixV1::new();
        assert!(matches!(
            bind_golden_qualification_result_v1(
                &empty,
                &view,
                &transcript,
                &submission,
                &result,
            ),
            Err(GoldenQualificationBindingErrorV1::UnknownRegisteredCase(_))
        ));
    }
}
