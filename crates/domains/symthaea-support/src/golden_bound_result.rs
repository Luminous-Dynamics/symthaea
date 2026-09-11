// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bound recording path for Golden Incident qualification results.
//!
//! This module closes the gap between deterministic metric derivation and the
//! qualification matrix: aggregate metrics are recomputed from the exact private
//! evaluation and must match the result before the result is recorded.

use crate::golden_incidents_v2::GoldenIncidentCorpusV2;
use crate::golden_metric_derivation::{
    derive_golden_qualification_metrics_v1, GoldenMetricDerivationErrorV1,
    GoldenPrivateEvaluationV1,
};
use crate::golden_qualification_binding::{
    bind_golden_qualification_result_v1, golden_qualification_metrics_digest_v1,
    GoldenQualificationBindingErrorV1, GoldenQualificationBindingV1,
};
use crate::golden_run_protocol::{GoldenRunTranscriptV1, GoldenSolverSubmissionV1};
use crate::it_qualification::{
    ItQualificationErrorV1, ItQualificationMatrixV1, ItQualificationResultV1,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenBoundQualificationResultV1 {
    pub lineage: GoldenQualificationBindingV1,
    pub private_evaluation_digest: String,
    pub derived_metrics_digest: String,
    pub evaluated_at_unix_ms: u64,
}

impl GoldenBoundQualificationResultV1 {
    pub fn digest(&self) -> Result<String, GoldenBoundResultErrorV1> {
        digest_serializable("symthaea-golden-bound-qualification-result-v1", self)
    }
}

pub fn bind_derived_golden_qualification_result_v1(
    matrix: &ItQualificationMatrixV1,
    corpus: &GoldenIncidentCorpusV2,
    transcript: &GoldenRunTranscriptV1,
    submission: &GoldenSolverSubmissionV1,
    evaluation: &GoldenPrivateEvaluationV1,
    result: &ItQualificationResultV1,
) -> Result<GoldenBoundQualificationResultV1, GoldenBoundResultErrorV1> {
    let incident = corpus
        .cases
        .iter()
        .find(|case| {
            case.id == result.case_key.id.0 && case.revision == result.case_key.revision
        })
        .ok_or(GoldenBoundResultErrorV1::IncidentMissingFromCorpus)?;
    let view = incident.solver_view();

    let derived = derive_golden_qualification_metrics_v1(
        &view,
        transcript,
        submission,
        evaluation,
    )?;

    if result.metrics != derived.metrics {
        return Err(GoldenBoundResultErrorV1::MetricsDoNotMatchDerivation);
    }
    if result.observed_at_unix_ms < evaluation.evaluated_at_unix_ms {
        return Err(GoldenBoundResultErrorV1::ResultPredatesEvaluation);
    }

    let lineage = bind_golden_qualification_result_v1(
        matrix,
        corpus,
        transcript,
        submission,
        result,
    )?;

    Ok(GoldenBoundQualificationResultV1 {
        lineage,
        private_evaluation_digest: derived.private_evaluation_digest,
        derived_metrics_digest: golden_qualification_metrics_digest_v1(&derived.metrics)?,
        evaluated_at_unix_ms: evaluation.evaluated_at_unix_ms,
    })
}

/// Verify exact corpus/run/private-evaluation lineage and only then record the result
/// in the generic qualification matrix. The matrix remains generic for non-golden
/// evaluations; golden certification should prefer this gated path.
pub fn record_derived_golden_qualification_result_v1(
    matrix: &mut ItQualificationMatrixV1,
    corpus: &GoldenIncidentCorpusV2,
    transcript: &GoldenRunTranscriptV1,
    submission: &GoldenSolverSubmissionV1,
    evaluation: &GoldenPrivateEvaluationV1,
    result: ItQualificationResultV1,
) -> Result<(bool, GoldenBoundQualificationResultV1), GoldenBoundResultErrorV1> {
    let binding = bind_derived_golden_qualification_result_v1(
        matrix,
        corpus,
        transcript,
        submission,
        evaluation,
        &result,
    )?;
    let inserted = matrix.record_result(result)?;
    Ok((inserted, binding))
}

fn digest_serializable<T: Serialize + ?Sized>(
    domain: &'static str,
    value: &T,
) -> Result<String, GoldenBoundResultErrorV1> {
    let bytes = serde_json::to_vec(&(domain, value))
        .map_err(|err| GoldenBoundResultErrorV1::Serialization(err.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

#[derive(Debug)]
pub enum GoldenBoundResultErrorV1 {
    MetricDerivation(GoldenMetricDerivationErrorV1),
    Lineage(GoldenQualificationBindingErrorV1),
    Qualification(ItQualificationErrorV1),
    Serialization(String),
    IncidentMissingFromCorpus,
    MetricsDoNotMatchDerivation,
    ResultPredatesEvaluation,
}

impl fmt::Display for GoldenBoundResultErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MetricDerivation(err) => write!(f, "golden metric derivation failed: {err}"),
            Self::Lineage(err) => write!(f, "golden qualification lineage failed: {err}"),
            Self::Qualification(err) => write!(f, "qualification recording failed: {err}"),
            Self::Serialization(message) => {
                write!(f, "bound golden result serialization failed: {message}")
            }
            Self::IncidentMissingFromCorpus => {
                write!(f, "qualification result case is missing from golden corpus")
            }
            Self::MetricsDoNotMatchDerivation => {
                write!(f, "qualification result metrics do not match deterministic derivation")
            }
            Self::ResultPredatesEvaluation => {
                write!(f, "qualification result timestamp precedes private evaluation")
            }
        }
    }
}

impl Error for GoldenBoundResultErrorV1 {}

impl From<GoldenMetricDerivationErrorV1> for GoldenBoundResultErrorV1 {
    fn from(value: GoldenMetricDerivationErrorV1) -> Self {
        Self::MetricDerivation(value)
    }
}

impl From<GoldenQualificationBindingErrorV1> for GoldenBoundResultErrorV1 {
    fn from(value: GoldenQualificationBindingErrorV1) -> Self {
        Self::Lineage(value)
    }
}

impl From<ItQualificationErrorV1> for GoldenBoundResultErrorV1 {
    fn from(value: ItQualificationErrorV1) -> Self {
        Self::Qualification(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden_incidents_v2::seed_golden_incidents_v2;
    use crate::golden_metric_derivation::{
        GoldenApplicabilityStatusV1, GoldenApplicabilityVerdictV1,
        GoldenFindingVerdictV1, GoldenPrivateEvaluationV1, GoldenRequiredFindingV1,
        GOLDEN_PRIVATE_EVALUATION_SCHEMA_V1,
    };
    use crate::golden_qualification_binding::golden_public_corpus_digest_v2;
    use crate::golden_run_protocol::{
        golden_grading_artifact_digest_v1, golden_solver_view_digest_v1,
        GoldenFindingDispositionV1, GoldenPresentedEvidenceSourceV1,
        GoldenPresentedEvidenceV1, GoldenRunTranscriptV1, GoldenSolverFindingV1,
        GoldenSolverSubmissionV1, GOLDEN_RUN_TRANSCRIPT_SCHEMA_V1,
        GOLDEN_SOLVER_SUBMISSION_SCHEMA_V1,
    };
    use crate::it_qualification::{
        ItQualificationMatrixV1, ItQualificationResultV1, QualificationResultIdV1,
        QualificationRunContextV1, QualificationRunIdV1,
    };
    use std::collections::BTreeSet;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn fixture() -> (
        ItQualificationMatrixV1,
        GoldenIncidentCorpusV2,
        GoldenRunTranscriptV1,
        GoldenSolverSubmissionV1,
        GoldenPrivateEvaluationV1,
        ItQualificationResultV1,
    ) {
        let corpus = seed_golden_incidents_v2().unwrap();
        let incident = corpus.cases[0].clone();
        let view = incident.solver_view();
        let mut matrix = ItQualificationMatrixV1::new();
        let case = incident.qualification_case().unwrap();
        let case_key = case.key.clone();
        matrix.register_case(case).unwrap();

        let run_id = QualificationRunIdV1("bound-run-1".into());
        let view_digest = golden_solver_view_digest_v1(&view).unwrap();
        let transcript = GoldenRunTranscriptV1 {
            schema_version: GOLDEN_RUN_TRANSCRIPT_SCHEMA_V1.into(),
            run_id: run_id.clone(),
            case_key: case_key.clone(),
            solver_view_digest: view_digest.clone(),
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
        let cited = view.initial_evidence[0].id.clone();
        let submission = GoldenSolverSubmissionV1 {
            schema_version: GOLDEN_SOLVER_SUBMISSION_SCHEMA_V1.into(),
            run_id: run_id.clone(),
            case_key: case_key.clone(),
            solver_view_digest: view_digest.clone(),
            findings: vec![GoldenSolverFindingV1 {
                finding_id: "finding-1".into(),
                statement: "filesystem namespace pressure contributes".into(),
                disposition: GoldenFindingDispositionV1::SuspectedFault,
                confidence: 0.8,
                evidence_ids: BTreeSet::from([cited.clone()]),
            }],
            final_confidence: 0.8,
            abstention: None,
            submitted_at_unix_ms: 210,
        };
        let evaluation = GoldenPrivateEvaluationV1 {
            schema_version: GOLDEN_PRIVATE_EVALUATION_SCHEMA_V1.into(),
            run_id: run_id.clone(),
            case_key: case_key.clone(),
            solver_view_digest: view_digest,
            transcript_digest: transcript.digest().unwrap(),
            submission_digest: submission.digest().unwrap(),
            private_oracle_digest: digest('a'),
            evaluator_profile: "private-evaluator-v1".into(),
            evaluated_at_unix_ms: 220,
            required_findings: vec![GoldenRequiredFindingV1 {
                rubric_id: "filesystem-pressure".into(),
                satisfied_by_finding_id: Some("finding-1".into()),
            }],
            finding_verdicts: vec![GoldenFindingVerdictV1 {
                finding_id: "finding-1".into(),
                correct: true,
                required_evidence_ids: BTreeSet::from([cited]),
            }],
            action_verdicts: Vec::new(),
            abstention_expected: None,
            applicability_verdicts: vec![GoldenApplicabilityVerdictV1 {
                item_id: "scope-1".into(),
                source_finding_id: "finding-1".into(),
                solver_status: GoldenApplicabilityStatusV1::Applicable,
                expected_status: GoldenApplicabilityStatusV1::Applicable,
            }],
        };
        let derived = derive_golden_qualification_metrics_v1(
            &view,
            &transcript,
            &submission,
            &evaluation,
        )
        .unwrap();
        let grading_digest =
            golden_grading_artifact_digest_v1(&view, &transcript, &submission).unwrap();
        let result = ItQualificationResultV1 {
            id: QualificationResultIdV1("bound-result-1".into()),
            run_id,
            case_key: case_key.clone(),
            case_digest: matrix.case_digest(&case_key).unwrap().into(),
            run_context: QualificationRunContextV1 {
                system_revision: "system-rev-1".into(),
                corpus_revision: golden_public_corpus_digest_v2(&corpus).unwrap(),
                environment_digest: digest('b'),
                toolchain_digest: Some(digest('c')),
                model_profile: "support-eval".into(),
                harness_version: "golden-harness-v1".into(),
            },
            observed_at_unix_ms: 230,
            metrics: derived.metrics,
            evidence_artifact_digest: Some(grading_digest),
        };
        (matrix, corpus, transcript, submission, evaluation, result)
    }

    #[test]
    fn exact_derived_result_is_bound_and_recorded() {
        let (mut matrix, corpus, transcript, submission, evaluation, result) = fixture();
        let (inserted, binding) = record_derived_golden_qualification_result_v1(
            &mut matrix,
            &corpus,
            &transcript,
            &submission,
            &evaluation,
            result,
        )
        .unwrap();
        assert!(inserted);
        assert_eq!(matrix.results().count(), 1);
        assert_eq!(binding.private_evaluation_digest.len(), 64);
        assert_eq!(binding.derived_metrics_digest.len(), 64);
        assert_eq!(binding.digest().unwrap().len(), 64);
    }

    #[test]
    fn handwritten_metric_drift_fails_closed() {
        let (matrix, corpus, transcript, submission, evaluation, mut result) = fixture();
        result.metrics.correctness = 0.5;
        assert!(matches!(
            bind_derived_golden_qualification_result_v1(
                &matrix,
                &corpus,
                &transcript,
                &submission,
                &evaluation,
                &result,
            ),
            Err(GoldenBoundResultErrorV1::MetricsDoNotMatchDerivation)
        ));
    }

    #[test]
    fn result_cannot_precede_private_evaluation() {
        let (matrix, corpus, transcript, submission, evaluation, mut result) = fixture();
        result.observed_at_unix_ms = evaluation.evaluated_at_unix_ms - 1;
        assert!(matches!(
            bind_derived_golden_qualification_result_v1(
                &matrix,
                &corpus,
                &transcript,
                &submission,
                &evaluation,
                &result,
            ),
            Err(GoldenBoundResultErrorV1::ResultPredatesEvaluation)
        ));
    }
}
