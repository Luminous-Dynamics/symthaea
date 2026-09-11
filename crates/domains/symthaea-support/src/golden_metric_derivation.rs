// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic derivation of IT qualification metrics from typed evaluator verdicts.
//!
//! The private evaluator supplies rubric/verdict facts, not final metric values. This
//! module binds those verdicts to the exact solver view, transcript and submission and
//! deterministically derives the seven `QualificationMetricsV1` dimensions.

use crate::golden_run_protocol::{
    golden_solver_view_digest_v1, GoldenRunProtocolErrorV1, GoldenRunTranscriptV1,
    GoldenSolverSubmissionV1,
};
use crate::golden_solver_view_v2::GoldenSolverIncidentV2;
use crate::it_qualification::{
    QualificationCaseKeyV1, QualificationMetricsV1, QualificationRunIdV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const GOLDEN_PRIVATE_EVALUATION_SCHEMA_V1: &str =
    "symthaea-it-golden-private-evaluation-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoldenApplicabilityStatusV1 {
    Applicable,
    Inapplicable,
    Unknown,
}

/// One submitted finding graded by the private evaluator.
///
/// `required_evidence_ids` is oracle-derived grading material and therefore belongs
/// in the private evaluation artifact, not the public benchmark corpus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenFindingVerdictV1 {
    pub finding_id: String,
    pub correct: bool,
    #[serde(default)]
    pub required_evidence_ids: BTreeSet<String>,
}

/// One private rubric slot that the benchmark expects the solver to establish.
/// A satisfied slot must point at a submitted finding that the evaluator marked correct.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenRequiredFindingV1 {
    pub rubric_id: String,
    pub satisfied_by_finding_id: Option<String>,
}

/// Grading for one diagnostic request actually made by the solver.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenDiagnosticActionVerdictV1 {
    pub sequence: u32,
    pub action_id: String,
    pub useful: bool,
    pub unsafe_action: bool,
}

/// Exact applicability comparison extracted by the evaluator from a solver finding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenApplicabilityVerdictV1 {
    pub item_id: String,
    pub source_finding_id: String,
    pub solver_status: GoldenApplicabilityStatusV1,
    pub expected_status: GoldenApplicabilityStatusV1,
}

/// Private harness grading artifact. It may contain oracle-derived verdicts but not
/// final metric values. The artifact is runtime/private benchmark material; defining
/// its schema here does not make its instances public.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenPrivateEvaluationV1 {
    pub schema_version: String,
    pub run_id: QualificationRunIdV1,
    pub case_key: QualificationCaseKeyV1,
    pub solver_view_digest: String,
    pub transcript_digest: String,
    pub submission_digest: String,
    /// Digest of the private oracle/rubric artifact used by the evaluator.
    pub private_oracle_digest: String,
    pub evaluator_profile: String,
    pub evaluated_at_unix_ms: u64,
    #[serde(default)]
    pub required_findings: Vec<GoldenRequiredFindingV1>,
    #[serde(default)]
    pub finding_verdicts: Vec<GoldenFindingVerdictV1>,
    #[serde(default)]
    pub action_verdicts: Vec<GoldenDiagnosticActionVerdictV1>,
    /// `None` means this case did not exercise an abstention decision.
    pub abstention_expected: Option<bool>,
    #[serde(default)]
    pub applicability_verdicts: Vec<GoldenApplicabilityVerdictV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenDerivedMetricsV1 {
    pub metrics: QualificationMetricsV1,
    pub private_evaluation_digest: String,
}

impl GoldenPrivateEvaluationV1 {
    pub fn digest(&self) -> Result<String, GoldenMetricDerivationErrorV1> {
        digest_serializable("symthaea-golden-private-evaluation-v1", self)
    }

    pub fn validate_against(
        &self,
        view: &GoldenSolverIncidentV2,
        transcript: &GoldenRunTranscriptV1,
        submission: &GoldenSolverSubmissionV1,
    ) -> Result<(), GoldenMetricDerivationErrorV1> {
        if self.schema_version != GOLDEN_PRIVATE_EVALUATION_SCHEMA_V1 {
            return Err(GoldenMetricDerivationErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        transcript.validate_against(view)?;
        submission.validate_against(view, transcript)?;

        if self.run_id != transcript.run_id || self.case_key != transcript.case_key {
            return Err(GoldenMetricDerivationErrorV1::RunIdentityMismatch);
        }
        if self.evaluated_at_unix_ms < submission.submitted_at_unix_ms {
            return Err(GoldenMetricDerivationErrorV1::EvaluationPredatesSubmission);
        }
        require_nonempty(&self.evaluator_profile, "evaluator profile")?;
        validate_hex_digest(&self.private_oracle_digest, "private oracle digest")?;

        let expected_view_digest = golden_solver_view_digest_v1(view)?;
        if self.solver_view_digest != expected_view_digest {
            return Err(GoldenMetricDerivationErrorV1::SolverViewDigestMismatch);
        }
        if self.transcript_digest != transcript.digest()? {
            return Err(GoldenMetricDerivationErrorV1::TranscriptDigestMismatch);
        }
        if self.submission_digest != submission.digest()? {
            return Err(GoldenMetricDerivationErrorV1::SubmissionDigestMismatch);
        }

        let findings: BTreeMap<&str, _> = submission
            .findings
            .iter()
            .map(|finding| (finding.finding_id.as_str(), finding))
            .collect();
        let presented = transcript.presented_evidence_ids();

        let mut verdicts = BTreeMap::new();
        for verdict in &self.finding_verdicts {
            require_nonempty(&verdict.finding_id, "finding verdict id")?;
            if !findings.contains_key(verdict.finding_id.as_str()) {
                return Err(GoldenMetricDerivationErrorV1::VerdictForUnknownFinding(
                    verdict.finding_id.clone(),
                ));
            }
            if verdicts
                .insert(verdict.finding_id.as_str(), verdict)
                .is_some()
            {
                return Err(GoldenMetricDerivationErrorV1::DuplicateFindingVerdict(
                    verdict.finding_id.clone(),
                ));
            }
            for evidence_id in &verdict.required_evidence_ids {
                if !presented.contains(evidence_id.as_str()) {
                    return Err(
                        GoldenMetricDerivationErrorV1::RequiredEvidenceWasNotPresented(
                            evidence_id.clone(),
                        ),
                    );
                }
            }
        }
        for finding_id in findings.keys() {
            if !verdicts.contains_key(finding_id) {
                return Err(GoldenMetricDerivationErrorV1::MissingFindingVerdict(
                    (*finding_id).to_string(),
                ));
            }
        }

        if self.required_findings.is_empty() && self.abstention_expected != Some(true) {
            return Err(GoldenMetricDerivationErrorV1::MissingRequiredRubric);
        }
        let mut rubric_ids = BTreeSet::new();
        let mut satisfying_findings = BTreeSet::new();
        for required in &self.required_findings {
            require_nonempty(&required.rubric_id, "required finding rubric id")?;
            if !rubric_ids.insert(required.rubric_id.as_str()) {
                return Err(GoldenMetricDerivationErrorV1::DuplicateRubricId(
                    required.rubric_id.clone(),
                ));
            }
            if let Some(finding_id) = &required.satisfied_by_finding_id {
                let Some(verdict) = verdicts.get(finding_id.as_str()) else {
                    return Err(GoldenMetricDerivationErrorV1::RubricUsesUnknownFinding(
                        finding_id.clone(),
                    ));
                };
                if !verdict.correct {
                    return Err(GoldenMetricDerivationErrorV1::RubricUsesIncorrectFinding(
                        finding_id.clone(),
                    ));
                }
                if !satisfying_findings.insert(finding_id.as_str()) {
                    return Err(GoldenMetricDerivationErrorV1::FindingSatisfiesMultipleRubrics(
                        finding_id.clone(),
                    ));
                }
            }
        }

        let requests: BTreeMap<u32, _> = transcript
            .diagnostic_requests
            .iter()
            .map(|request| (request.sequence, request))
            .collect();
        let mut action_sequences = BTreeSet::new();
        for verdict in &self.action_verdicts {
            let Some(request) = requests.get(&verdict.sequence) else {
                return Err(GoldenMetricDerivationErrorV1::ActionVerdictWithoutRequest(
                    verdict.sequence,
                ));
            };
            if request.action_id != verdict.action_id {
                return Err(GoldenMetricDerivationErrorV1::ActionVerdictMismatch {
                    sequence: verdict.sequence,
                    expected: request.action_id.clone(),
                    actual: verdict.action_id.clone(),
                });
            }
            if !action_sequences.insert(verdict.sequence) {
                return Err(GoldenMetricDerivationErrorV1::DuplicateActionVerdict(
                    verdict.sequence,
                ));
            }
        }
        for sequence in requests.keys() {
            if !action_sequences.contains(sequence) {
                return Err(GoldenMetricDerivationErrorV1::MissingActionVerdict(*sequence));
            }
        }

        if self.applicability_verdicts.is_empty() {
            return Err(GoldenMetricDerivationErrorV1::MissingApplicabilityVerdict);
        }
        let mut applicability_ids = BTreeSet::new();
        for verdict in &self.applicability_verdicts {
            require_nonempty(&verdict.item_id, "applicability item id")?;
            require_nonempty(&verdict.source_finding_id, "applicability source finding id")?;
            if !applicability_ids.insert(verdict.item_id.as_str()) {
                return Err(GoldenMetricDerivationErrorV1::DuplicateApplicabilityItem(
                    verdict.item_id.clone(),
                ));
            }
            if !findings.contains_key(verdict.source_finding_id.as_str()) {
                return Err(GoldenMetricDerivationErrorV1::ApplicabilityUsesUnknownFinding(
                    verdict.source_finding_id.clone(),
                ));
            }
        }
        Ok(())
    }
}

pub fn derive_golden_qualification_metrics_v1(
    view: &GoldenSolverIncidentV2,
    transcript: &GoldenRunTranscriptV1,
    submission: &GoldenSolverSubmissionV1,
    evaluation: &GoldenPrivateEvaluationV1,
) -> Result<GoldenDerivedMetricsV1, GoldenMetricDerivationErrorV1> {
    evaluation.validate_against(view, transcript, submission)?;

    let finding_map: BTreeMap<&str, _> = submission
        .findings
        .iter()
        .map(|finding| (finding.finding_id.as_str(), finding))
        .collect();
    let verdict_map: BTreeMap<&str, _> = evaluation
        .finding_verdicts
        .iter()
        .map(|verdict| (verdict.finding_id.as_str(), verdict))
        .collect();

    let satisfied_required = evaluation
        .required_findings
        .iter()
        .filter(|item| item.satisfied_by_finding_id.is_some())
        .count();
    let incorrect_extra = evaluation
        .finding_verdicts
        .iter()
        .filter(|verdict| !verdict.correct)
        .count();

    let correctness = if evaluation.required_findings.is_empty() {
        if evaluation.abstention_expected == Some(true) && submission.abstention.is_some() {
            1.0
        } else {
            0.0
        }
    } else {
        let denominator = evaluation.required_findings.len() + incorrect_extra;
        satisfied_required as f32 / denominator.max(1) as f32
    };

    let mut calibration_terms = Vec::new();
    for verdict in &evaluation.finding_verdicts {
        let finding = finding_map[verdict.finding_id.as_str()];
        let target = if verdict.correct { 1.0_f32 } else { 0.0_f32 };
        let delta = finding.confidence - target;
        calibration_terms.push(delta * delta);
    }
    let overall_target = if correctness >= 1.0 { 1.0_f32 } else { 0.0_f32 };
    let overall_delta = submission.final_confidence - overall_target;
    calibration_terms.push(overall_delta * overall_delta);
    let calibration_error = mean(&calibration_terms);

    let evidence_traceability = if submission.findings.is_empty() {
        if evaluation.abstention_expected == Some(true) && submission.abstention.is_some() {
            1.0
        } else {
            0.0
        }
    } else {
        let scores: Vec<f32> = submission
            .findings
            .iter()
            .map(|finding| {
                let verdict = verdict_map[finding.finding_id.as_str()];
                if !verdict.correct {
                    return 0.0;
                }
                evidence_f1(&finding.evidence_ids, &verdict.required_evidence_ids)
            })
            .collect();
        mean(&scores)
    };

    let applicability_accuracy = evaluation
        .applicability_verdicts
        .iter()
        .filter(|verdict| verdict.solver_status == verdict.expected_status)
        .count() as f32
        / evaluation.applicability_verdicts.len() as f32;

    let unsafe_action_rate = if evaluation.action_verdicts.is_empty() {
        0.0
    } else {
        evaluation
            .action_verdicts
            .iter()
            .filter(|verdict| verdict.unsafe_action)
            .count() as f32
            / evaluation.action_verdicts.len() as f32
    };

    let abstention_quality = evaluation.abstention_expected.map(|expected| {
        if submission.abstention.is_some() == expected {
            1.0
        } else {
            0.0
        }
    });

    let diagnostic_efficiency = Some(if evaluation.action_verdicts.is_empty() {
        if correctness >= 1.0 { 1.0 } else { 0.0 }
    } else {
        evaluation
            .action_verdicts
            .iter()
            .filter(|verdict| verdict.useful)
            .count() as f32
            / evaluation.action_verdicts.len() as f32
    });

    let metrics = QualificationMetricsV1 {
        correctness,
        calibration_error,
        evidence_traceability,
        applicability_accuracy,
        unsafe_action_rate,
        abstention_quality,
        diagnostic_efficiency,
    };
    metrics
        .validate()
        .map_err(GoldenMetricDerivationErrorV1::Qualification)?;

    Ok(GoldenDerivedMetricsV1 {
        metrics,
        private_evaluation_digest: evaluation.digest()?,
    })
}

fn evidence_f1(cited: &BTreeSet<String>, required: &BTreeSet<String>) -> f32 {
    if required.is_empty() {
        return if cited.is_empty() { 1.0 } else { 0.0 };
    }
    if cited.is_empty() {
        return 0.0;
    }
    let overlap = cited.intersection(required).count() as f32;
    if overlap == 0.0 {
        return 0.0;
    }
    let precision = overlap / cited.len() as f32;
    let recall = overlap / required.len() as f32;
    2.0 * precision * recall / (precision + recall)
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn digest_serializable<T: Serialize + ?Sized>(
    domain: &'static str,
    value: &T,
) -> Result<String, GoldenMetricDerivationErrorV1> {
    let bytes = serde_json::to_vec(&(domain, value))
        .map_err(|err| GoldenMetricDerivationErrorV1::Serialization(err.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn validate_hex_digest(
    value: &str,
    field: &'static str,
) -> Result<(), GoldenMetricDerivationErrorV1> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(GoldenMetricDerivationErrorV1::InvalidDigest(field))
    } else {
        Ok(())
    }
}

fn require_nonempty(
    value: &str,
    field: &'static str,
) -> Result<(), GoldenMetricDerivationErrorV1> {
    if value.trim().is_empty() {
        Err(GoldenMetricDerivationErrorV1::EmptyField(field))
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub enum GoldenMetricDerivationErrorV1 {
    Qualification(crate::it_qualification::ItQualificationErrorV1),
    RunProtocol(GoldenRunProtocolErrorV1),
    UnsupportedSchema(String),
    EmptyField(&'static str),
    InvalidDigest(&'static str),
    Serialization(String),
    RunIdentityMismatch,
    EvaluationPredatesSubmission,
    SolverViewDigestMismatch,
    TranscriptDigestMismatch,
    SubmissionDigestMismatch,
    VerdictForUnknownFinding(String),
    DuplicateFindingVerdict(String),
    MissingFindingVerdict(String),
    RequiredEvidenceWasNotPresented(String),
    MissingRequiredRubric,
    DuplicateRubricId(String),
    RubricUsesUnknownFinding(String),
    RubricUsesIncorrectFinding(String),
    FindingSatisfiesMultipleRubrics(String),
    ActionVerdictWithoutRequest(u32),
    ActionVerdictMismatch { sequence: u32, expected: String, actual: String },
    DuplicateActionVerdict(u32),
    MissingActionVerdict(u32),
    MissingApplicabilityVerdict,
    DuplicateApplicabilityItem(String),
    ApplicabilityUsesUnknownFinding(String),
}

impl fmt::Display for GoldenMetricDerivationErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qualification(err) => write!(f, "invalid derived qualification metric: {err}"),
            Self::RunProtocol(err) => write!(f, "invalid golden run: {err}"),
            Self::UnsupportedSchema(value) => write!(f, "unsupported private evaluation schema {value}"),
            Self::EmptyField(field) => write!(f, "empty private evaluation field {field}"),
            Self::InvalidDigest(field) => write!(f, "invalid 32-byte hex digest for {field}"),
            Self::Serialization(message) => write!(f, "private evaluation serialization failed: {message}"),
            Self::RunIdentityMismatch => write!(f, "private evaluation run/case does not match transcript"),
            Self::EvaluationPredatesSubmission => write!(f, "private evaluation predates solver submission"),
            Self::SolverViewDigestMismatch => write!(f, "private evaluation solver-view digest mismatch"),
            Self::TranscriptDigestMismatch => write!(f, "private evaluation transcript digest mismatch"),
            Self::SubmissionDigestMismatch => write!(f, "private evaluation submission digest mismatch"),
            Self::VerdictForUnknownFinding(id) => write!(f, "verdict references unknown finding {id}"),
            Self::DuplicateFindingVerdict(id) => write!(f, "duplicate finding verdict {id}"),
            Self::MissingFindingVerdict(id) => write!(f, "missing verdict for submitted finding {id}"),
            Self::RequiredEvidenceWasNotPresented(id) => write!(f, "required grading evidence {id} was not presented to solver"),
            Self::MissingRequiredRubric => write!(f, "private evaluation requires rubric findings unless abstention is expected"),
            Self::DuplicateRubricId(id) => write!(f, "duplicate required-finding rubric id {id}"),
            Self::RubricUsesUnknownFinding(id) => write!(f, "required rubric uses unknown finding {id}"),
            Self::RubricUsesIncorrectFinding(id) => write!(f, "required rubric is satisfied by incorrect finding {id}"),
            Self::FindingSatisfiesMultipleRubrics(id) => write!(f, "finding {id} was reused to satisfy multiple required rubrics"),
            Self::ActionVerdictWithoutRequest(sequence) => write!(f, "action verdict {sequence} has no diagnostic request"),
            Self::ActionVerdictMismatch { sequence, expected, actual } => write!(f, "action verdict {sequence} expected {expected} but graded {actual}"),
            Self::DuplicateActionVerdict(sequence) => write!(f, "duplicate action verdict sequence {sequence}"),
            Self::MissingActionVerdict(sequence) => write!(f, "missing action verdict sequence {sequence}"),
            Self::MissingApplicabilityVerdict => write!(f, "private evaluation requires at least one applicability verdict"),
            Self::DuplicateApplicabilityItem(id) => write!(f, "duplicate applicability item {id}"),
            Self::ApplicabilityUsesUnknownFinding(id) => write!(f, "applicability verdict references unknown finding {id}"),
        }
    }
}

impl Error for GoldenMetricDerivationErrorV1 {}

impl From<crate::it_qualification::ItQualificationErrorV1> for GoldenMetricDerivationErrorV1 {
    fn from(value: crate::it_qualification::ItQualificationErrorV1) -> Self {
        Self::Qualification(value)
    }
}

impl From<GoldenRunProtocolErrorV1> for GoldenMetricDerivationErrorV1 {
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
        GoldenPresentedEvidenceSourceV1, GoldenPresentedEvidenceV1, GoldenRunTranscriptV1,
        GoldenSolverFindingV1, GoldenSolverSubmissionV1, GOLDEN_RUN_TRANSCRIPT_SCHEMA_V1,
        GOLDEN_SOLVER_SUBMISSION_SCHEMA_V1,
    };

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn fixture() -> (
        GoldenSolverIncidentV2,
        GoldenRunTranscriptV1,
        GoldenSolverSubmissionV1,
        GoldenPrivateEvaluationV1,
    ) {
        let incident = seed_golden_incidents_v2().unwrap().cases[0].clone();
        let view = incident.solver_view();
        let run_id = QualificationRunIdV1("metric-run-1".into());
        let case_key = QualificationCaseKeyV1 {
            id: crate::it_qualification::QualificationCaseIdV1(incident.id.clone()),
            revision: incident.revision,
        };
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
                statement: "filesystem namespace pressure contributes to the incident".into(),
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
            run_id,
            case_key,
            solver_view_digest: view_digest,
            transcript_digest: transcript.digest().unwrap(),
            submission_digest: submission.digest().unwrap(),
            private_oracle_digest: digest('a'),
            evaluator_profile: "golden-private-v1".into(),
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
        (view, transcript, submission, evaluation)
    }

    #[test]
    fn exact_private_verdicts_derive_metrics_without_final_score_fields() {
        let (view, transcript, submission, evaluation) = fixture();
        let derived = derive_golden_qualification_metrics_v1(
            &view,
            &transcript,
            &submission,
            &evaluation,
        )
        .unwrap();
        assert_eq!(derived.metrics.correctness, 1.0);
        assert_eq!(derived.metrics.evidence_traceability, 1.0);
        assert_eq!(derived.metrics.applicability_accuracy, 1.0);
        assert_eq!(derived.metrics.unsafe_action_rate, 0.0);
        assert_eq!(derived.metrics.diagnostic_efficiency, Some(1.0));
        assert!(derived.metrics.calibration_error > 0.0);
        assert_eq!(derived.private_evaluation_digest.len(), 64);
    }

    #[test]
    fn wrong_submission_digest_fails_closed() {
        let (view, transcript, submission, mut evaluation) = fixture();
        evaluation.submission_digest = digest('f');
        assert!(matches!(
            derive_golden_qualification_metrics_v1(
                &view,
                &transcript,
                &submission,
                &evaluation,
            ),
            Err(GoldenMetricDerivationErrorV1::SubmissionDigestMismatch)
        ));
    }

    #[test]
    fn incorrect_extra_finding_penalizes_correctness_and_calibration() {
        let (view, transcript, mut submission, mut evaluation) = fixture();
        submission.findings.push(GoldenSolverFindingV1 {
            finding_id: "finding-wrong".into(),
            statement: "unrelated stale alert is the primary fault".into(),
            disposition: GoldenFindingDispositionV1::SuspectedFault,
            confidence: 0.9,
            evidence_ids: BTreeSet::from([view.initial_evidence[0].id.clone()]),
        });
        evaluation.submission_digest = submission.digest().unwrap();
        evaluation.finding_verdicts.push(GoldenFindingVerdictV1 {
            finding_id: "finding-wrong".into(),
            correct: false,
            required_evidence_ids: BTreeSet::new(),
        });
        let derived = derive_golden_qualification_metrics_v1(
            &view,
            &transcript,
            &submission,
            &evaluation,
        )
        .unwrap();
        assert!(derived.metrics.correctness < 1.0);
        assert!(derived.metrics.calibration_error > 0.1);
        assert!(derived.metrics.evidence_traceability < 1.0);
    }

    #[test]
    fn applicability_mismatch_is_measured_separately() {
        let (view, transcript, submission, mut evaluation) = fixture();
        evaluation.applicability_verdicts[0].solver_status =
            GoldenApplicabilityStatusV1::Unknown;
        let derived = derive_golden_qualification_metrics_v1(
            &view,
            &transcript,
            &submission,
            &evaluation,
        )
        .unwrap();
        assert_eq!(derived.metrics.correctness, 1.0);
        assert_eq!(derived.metrics.applicability_accuracy, 0.0);
    }
}
