// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reproducible run/submission protocol for Golden Incident V2 evaluation.
//!
//! The public protocol records what the solver saw and did without embedding
//! private oracle truth or private diagnostic outcome payloads.
//!
//! ```text
//! solver view digest
//!      +
//! presented evidence identities
//!      +
//! ordered diagnostic requests
//!      +
//! opaque outcome digests
//!      +
//! evidence-cited findings / abstention
//!      =
//! replayable grading artifact
//! ```

use crate::golden_solver_view_v2::GoldenSolverIncidentV2;
use crate::it_qualification::{QualificationCaseKeyV1, QualificationRunIdV1};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const GOLDEN_RUN_TRANSCRIPT_SCHEMA_V1: &str = "symthaea-it-golden-run-transcript-v1";
pub const GOLDEN_SOLVER_SUBMISSION_SCHEMA_V1: &str = "symthaea-it-golden-solver-submission-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoldenPresentedEvidenceSourceV1 {
    InitialFixture,
    DiagnosticOutcome { action_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenPresentedEvidenceV1 {
    pub evidence_id: String,
    pub source: GoldenPresentedEvidenceSourceV1,
    /// Required for diagnostic outcomes when the payload itself is kept in the
    /// private harness artifact. This protocol stores only the opaque digest.
    pub private_artifact_digest: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenDiagnosticRequestV1 {
    /// One-based contiguous request order.
    pub sequence: u32,
    pub action_id: String,
    pub rationale: String,
    /// Solver-declared estimate. It is calibration evidence, not trusted truth.
    pub expected_information_gain_bits: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenRunTranscriptV1 {
    pub schema_version: String,
    pub run_id: QualificationRunIdV1,
    pub case_key: QualificationCaseKeyV1,
    pub solver_view_digest: String,
    pub started_at_unix_ms: u64,
    pub finished_at_unix_ms: u64,
    /// Evidence actually presented to the solver. Private payloads may live
    /// elsewhere; their identities/digests still bind the grading lineage.
    pub presented_evidence: Vec<GoldenPresentedEvidenceV1>,
    pub diagnostic_requests: Vec<GoldenDiagnosticRequestV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoldenFindingDispositionV1 {
    SuspectedFault,
    ContributingCondition,
    RuledOut,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenSolverFindingV1 {
    pub finding_id: String,
    pub statement: String,
    pub disposition: GoldenFindingDispositionV1,
    pub confidence: f32,
    #[serde(default)]
    pub evidence_ids: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoldenAbstentionReasonV1 {
    InsufficientEvidence,
    ConflictingEvidence,
    ApplicabilityUnknown,
    AuthorityUnavailable,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenSolverAbstentionV1 {
    pub reason: GoldenAbstentionReasonV1,
    pub explanation: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoldenSolverSubmissionV1 {
    pub schema_version: String,
    pub run_id: QualificationRunIdV1,
    pub case_key: QualificationCaseKeyV1,
    pub solver_view_digest: String,
    #[serde(default)]
    pub findings: Vec<GoldenSolverFindingV1>,
    pub final_confidence: f32,
    pub abstention: Option<GoldenSolverAbstentionV1>,
    pub submitted_at_unix_ms: u64,
}

pub fn golden_solver_view_digest_v1(
    view: &GoldenSolverIncidentV2,
) -> Result<String, GoldenRunProtocolErrorV1> {
    digest_serializable("symthaea-golden-solver-view-v1", view)
}

impl GoldenRunTranscriptV1 {
    pub fn validate_against(
        &self,
        view: &GoldenSolverIncidentV2,
    ) -> Result<(), GoldenRunProtocolErrorV1> {
        if self.schema_version != GOLDEN_RUN_TRANSCRIPT_SCHEMA_V1 {
            return Err(GoldenRunProtocolErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        require_nonempty(&self.run_id.0, "run id")?;
        require_nonempty(&self.case_key.id.0, "case id")?;
        if self.case_key.revision == 0 {
            return Err(GoldenRunProtocolErrorV1::InvalidField(
                "case revision must be non-zero".into(),
            ));
        }
        if self.finished_at_unix_ms < self.started_at_unix_ms {
            return Err(GoldenRunProtocolErrorV1::InvalidField(
                "run finish time precedes start time".into(),
            ));
        }
        self.require_view_identity(view)?;

        let offered_actions: BTreeSet<&str> = view
            .diagnostic_actions
            .iter()
            .map(|action| action.id.as_str())
            .collect();

        let mut requested_actions = BTreeSet::new();
        for (index, request) in self.diagnostic_requests.iter().enumerate() {
            let expected = (index + 1) as u32;
            if request.sequence != expected {
                return Err(GoldenRunProtocolErrorV1::NonContiguousDiagnosticSequence {
                    expected,
                    actual: request.sequence,
                });
            }
            require_nonempty(&request.action_id, "diagnostic action id")?;
            require_nonempty(&request.rationale, "diagnostic rationale")?;
            if !offered_actions.contains(request.action_id.as_str()) {
                return Err(GoldenRunProtocolErrorV1::UnknownAction(
                    request.action_id.clone(),
                ));
            }
            if let Some(value) = request.expected_information_gain_bits {
                if !value.is_finite() || value < 0.0 {
                    return Err(GoldenRunProtocolErrorV1::InvalidInformationGain(value));
                }
            }
            requested_actions.insert(request.action_id.as_str());
        }

        let initial_ids: BTreeSet<&str> = view
            .initial_evidence
            .iter()
            .map(|evidence| evidence.id.as_str())
            .collect();
        let mut presented = BTreeMap::new();
        for evidence in &self.presented_evidence {
            require_nonempty(&evidence.evidence_id, "presented evidence id")?;
            if presented.insert(evidence.evidence_id.as_str(), evidence).is_some() {
                return Err(GoldenRunProtocolErrorV1::DuplicateEvidenceId(
                    evidence.evidence_id.clone(),
                ));
            }
            match &evidence.source {
                GoldenPresentedEvidenceSourceV1::InitialFixture => {
                    if !initial_ids.contains(evidence.evidence_id.as_str()) {
                        return Err(GoldenRunProtocolErrorV1::UnknownInitialEvidence(
                            evidence.evidence_id.clone(),
                        ));
                    }
                    if evidence.private_artifact_digest.is_some() {
                        return Err(GoldenRunProtocolErrorV1::InvalidField(format!(
                            "initial fixture evidence {} must not claim a private outcome digest",
                            evidence.evidence_id
                        )));
                    }
                }
                GoldenPresentedEvidenceSourceV1::DiagnosticOutcome { action_id } => {
                    require_nonempty(action_id, "diagnostic outcome action id")?;
                    if !requested_actions.contains(action_id.as_str()) {
                        return Err(GoldenRunProtocolErrorV1::OutcomeWithoutRequest(
                            action_id.clone(),
                        ));
                    }
                    let Some(digest) = evidence.private_artifact_digest.as_deref() else {
                        return Err(GoldenRunProtocolErrorV1::MissingPrivateArtifactDigest(
                            evidence.evidence_id.clone(),
                        ));
                    };
                    validate_hex_digest(digest, "private diagnostic artifact digest")?;
                }
            }
        }

        for initial in initial_ids {
            if !presented.contains_key(initial) {
                return Err(GoldenRunProtocolErrorV1::MissingInitialEvidence(
                    initial.to_string(),
                ));
            }
        }
        Ok(())
    }

    fn require_view_identity(
        &self,
        view: &GoldenSolverIncidentV2,
    ) -> Result<(), GoldenRunProtocolErrorV1> {
        if self.case_key.id.0 != view.id || self.case_key.revision != view.revision {
            return Err(GoldenRunProtocolErrorV1::CaseIdentityMismatch);
        }
        let expected = golden_solver_view_digest_v1(view)?;
        if self.solver_view_digest != expected {
            return Err(GoldenRunProtocolErrorV1::SolverViewDigestMismatch);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, GoldenRunProtocolErrorV1> {
        digest_serializable("symthaea-golden-run-transcript-v1", self)
    }

    pub fn presented_evidence_ids(&self) -> BTreeSet<&str> {
        self.presented_evidence
            .iter()
            .map(|evidence| evidence.evidence_id.as_str())
            .collect()
    }
}

impl GoldenSolverSubmissionV1 {
    pub fn validate_against(
        &self,
        view: &GoldenSolverIncidentV2,
        transcript: &GoldenRunTranscriptV1,
    ) -> Result<(), GoldenRunProtocolErrorV1> {
        if self.schema_version != GOLDEN_SOLVER_SUBMISSION_SCHEMA_V1 {
            return Err(GoldenRunProtocolErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        transcript.validate_against(view)?;
        if self.run_id != transcript.run_id
            || self.case_key != transcript.case_key
            || self.solver_view_digest != transcript.solver_view_digest
        {
            return Err(GoldenRunProtocolErrorV1::SubmissionTranscriptMismatch);
        }
        if self.submitted_at_unix_ms < transcript.finished_at_unix_ms {
            return Err(GoldenRunProtocolErrorV1::InvalidField(
                "solver submission precedes transcript finish".into(),
            ));
        }
        validate_unit(self.final_confidence, "final confidence")?;
        if self.findings.is_empty() && self.abstention.is_none() {
            return Err(GoldenRunProtocolErrorV1::InvalidField(
                "submission must contain findings or an explicit abstention".into(),
            ));
        }
        if let Some(abstention) = &self.abstention {
            validate_abstention(abstention)?;
        }

        let presented = transcript.presented_evidence_ids();
        let mut finding_ids = BTreeSet::new();
        for finding in &self.findings {
            require_nonempty(&finding.finding_id, "finding id")?;
            require_nonempty(&finding.statement, "finding statement")?;
            validate_unit(finding.confidence, "finding confidence")?;
            if !finding_ids.insert(finding.finding_id.as_str()) {
                return Err(GoldenRunProtocolErrorV1::DuplicateFindingId(
                    finding.finding_id.clone(),
                ));
            }
            for evidence_id in &finding.evidence_ids {
                require_nonempty(evidence_id, "finding evidence id")?;
                if !presented.contains(evidence_id.as_str()) {
                    return Err(GoldenRunProtocolErrorV1::UnpresentedEvidenceCitation(
                        evidence_id.clone(),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, GoldenRunProtocolErrorV1> {
        digest_serializable("symthaea-golden-solver-submission-v1", self)
    }
}

/// Digest suitable for `ItQualificationResultV1::evidence_artifact_digest`.
pub fn golden_grading_artifact_digest_v1(
    view: &GoldenSolverIncidentV2,
    transcript: &GoldenRunTranscriptV1,
    submission: &GoldenSolverSubmissionV1,
) -> Result<String, GoldenRunProtocolErrorV1> {
    transcript.validate_against(view)?;
    submission.validate_against(view, transcript)?;
    digest_serializable(
        "symthaea-golden-grading-artifact-v1",
        &(
            golden_solver_view_digest_v1(view)?,
            transcript.digest()?,
            submission.digest()?,
        ),
    )
}

fn validate_abstention(value: &GoldenSolverAbstentionV1) -> Result<(), GoldenRunProtocolErrorV1> {
    require_nonempty(&value.explanation, "abstention explanation")?;
    if let GoldenAbstentionReasonV1::Other(reason) = &value.reason {
        require_nonempty(reason, "other abstention reason")?;
    }
    Ok(())
}

fn digest_serializable<T: Serialize + ?Sized>(
    domain: &'static str,
    value: &T,
) -> Result<String, GoldenRunProtocolErrorV1> {
    let bytes = serde_json::to_vec(&(domain, value))
        .map_err(|err| GoldenRunProtocolErrorV1::Serialization(err.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn validate_unit(value: f32, field: &'static str) -> Result<(), GoldenRunProtocolErrorV1> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        Err(GoldenRunProtocolErrorV1::InvalidUnit { field, value })
    } else {
        Ok(())
    }
}

fn validate_hex_digest(value: &str, field: &'static str) -> Result<(), GoldenRunProtocolErrorV1> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(GoldenRunProtocolErrorV1::InvalidDigest(field))
    } else {
        Ok(())
    }
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), GoldenRunProtocolErrorV1> {
    if value.trim().is_empty() {
        Err(GoldenRunProtocolErrorV1::EmptyField(field))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GoldenRunProtocolErrorV1 {
    UnsupportedSchema(String),
    EmptyField(&'static str),
    InvalidField(String),
    InvalidUnit { field: &'static str, value: f32 },
    InvalidInformationGain(f64),
    InvalidDigest(&'static str),
    Serialization(String),
    CaseIdentityMismatch,
    SolverViewDigestMismatch,
    SubmissionTranscriptMismatch,
    UnknownAction(String),
    NonContiguousDiagnosticSequence { expected: u32, actual: u32 },
    DuplicateEvidenceId(String),
    UnknownInitialEvidence(String),
    MissingInitialEvidence(String),
    OutcomeWithoutRequest(String),
    MissingPrivateArtifactDigest(String),
    DuplicateFindingId(String),
    UnpresentedEvidenceCitation(String),
}

impl fmt::Display for GoldenRunProtocolErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchema(value) => write!(f, "unsupported golden run schema {value}"),
            Self::EmptyField(field) => write!(f, "empty golden run field {field}"),
            Self::InvalidField(message) => write!(f, "invalid golden run: {message}"),
            Self::InvalidUnit { field, value } => write!(f, "invalid {field} {value}; expected [0,1]"),
            Self::InvalidInformationGain(value) => write!(f, "invalid expected information gain {value}"),
            Self::InvalidDigest(field) => write!(f, "invalid 32-byte hex digest for {field}"),
            Self::Serialization(message) => write!(f, "golden run serialization failed: {message}"),
            Self::CaseIdentityMismatch => write!(f, "golden run case identity does not match solver view"),
            Self::SolverViewDigestMismatch => write!(f, "golden run solver-view digest mismatch"),
            Self::SubmissionTranscriptMismatch => write!(f, "solver submission does not match run transcript"),
            Self::UnknownAction(id) => write!(f, "golden run requested unknown diagnostic action {id}"),
            Self::NonContiguousDiagnosticSequence { expected, actual } => write!(f, "diagnostic request sequence expected {expected}, got {actual}"),
            Self::DuplicateEvidenceId(id) => write!(f, "duplicate presented evidence id {id}"),
            Self::UnknownInitialEvidence(id) => write!(f, "unknown initial evidence id {id}"),
            Self::MissingInitialEvidence(id) => write!(f, "initial solver evidence {id} was not recorded as presented"),
            Self::OutcomeWithoutRequest(id) => write!(f, "diagnostic outcome references unrequested action {id}"),
            Self::MissingPrivateArtifactDigest(id) => write!(f, "diagnostic outcome evidence {id} lacks private artifact digest"),
            Self::DuplicateFindingId(id) => write!(f, "duplicate solver finding id {id}"),
            Self::UnpresentedEvidenceCitation(id) => write!(f, "solver cited evidence that was never presented: {id}"),
        }
    }
}

impl Error for GoldenRunProtocolErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden_incidents_v2::seed_golden_incidents_v2;

    fn fixture() -> GoldenSolverIncidentV2 {
        seed_golden_incidents_v2().unwrap().cases[0].solver_view()
    }

    fn transcript(view: &GoldenSolverIncidentV2) -> GoldenRunTranscriptV1 {
        let digest = golden_solver_view_digest_v1(view).unwrap();
        GoldenRunTranscriptV1 {
            schema_version: GOLDEN_RUN_TRANSCRIPT_SCHEMA_V1.into(),
            run_id: QualificationRunIdV1("run-1".into()),
            case_key: QualificationCaseKeyV1 {
                id: crate::it_qualification::QualificationCaseIdV1(view.id.clone()),
                revision: view.revision,
            },
            solver_view_digest: digest,
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
        }
    }

    #[test]
    fn transcript_binds_exact_solver_view_and_initial_evidence() {
        let view = fixture();
        transcript(&view).validate_against(&view).unwrap();
        let mut changed = view.clone();
        changed.symptom.push_str(" changed");
        assert!(matches!(
            transcript(&view).validate_against(&changed),
            Err(GoldenRunProtocolErrorV1::SolverViewDigestMismatch)
        ));
    }

    #[test]
    fn diagnostic_outcome_requires_prior_request_and_private_digest() {
        let view = fixture();
        let action = view.diagnostic_actions[0].id.clone();
        let mut run = transcript(&view);
        run.diagnostic_requests.push(GoldenDiagnosticRequestV1 {
            sequence: 1,
            action_id: action.clone(),
            rationale: "discriminate filesystem capacity dimensions".into(),
            expected_information_gain_bits: Some(0.5),
        });
        run.presented_evidence.push(GoldenPresentedEvidenceV1 {
            evidence_id: "dynamic-1".into(),
            source: GoldenPresentedEvidenceSourceV1::DiagnosticOutcome { action_id: action },
            private_artifact_digest: Some("a".repeat(64)),
        });
        run.validate_against(&view).unwrap();
    }

    #[test]
    fn solver_cannot_cite_unpresented_evidence() {
        let view = fixture();
        let run = transcript(&view);
        let submission = GoldenSolverSubmissionV1 {
            schema_version: GOLDEN_SOLVER_SUBMISSION_SCHEMA_V1.into(),
            run_id: run.run_id.clone(),
            case_key: run.case_key.clone(),
            solver_view_digest: run.solver_view_digest.clone(),
            findings: vec![GoldenSolverFindingV1 {
                finding_id: "f1".into(),
                statement: "possible namespace-capacity failure".into(),
                disposition: GoldenFindingDispositionV1::SuspectedFault,
                confidence: 0.7,
                evidence_ids: BTreeSet::from(["never-presented".into()]),
            }],
            final_confidence: 0.7,
            abstention: None,
            submitted_at_unix_ms: 200,
        };
        assert_eq!(
            submission.validate_against(&view, &run),
            Err(GoldenRunProtocolErrorV1::UnpresentedEvidenceCitation(
                "never-presented".into()
            ))
        );
    }

    #[test]
    fn explicit_abstention_is_valid_without_findings() {
        let view = fixture();
        let run = transcript(&view);
        let submission = GoldenSolverSubmissionV1 {
            schema_version: GOLDEN_SOLVER_SUBMISSION_SCHEMA_V1.into(),
            run_id: run.run_id.clone(),
            case_key: run.case_key.clone(),
            solver_view_digest: run.solver_view_digest.clone(),
            findings: Vec::new(),
            final_confidence: 0.2,
            abstention: Some(GoldenSolverAbstentionV1 {
                reason: GoldenAbstentionReasonV1::InsufficientEvidence,
                explanation: "need a current filesystem namespace measurement".into(),
            }),
            submitted_at_unix_ms: 200,
        };
        submission.validate_against(&view, &run).unwrap();
    }

    #[test]
    fn grading_artifact_digest_binds_view_transcript_and_submission() {
        let view = fixture();
        let run = transcript(&view);
        let submission = GoldenSolverSubmissionV1 {
            schema_version: GOLDEN_SOLVER_SUBMISSION_SCHEMA_V1.into(),
            run_id: run.run_id.clone(),
            case_key: run.case_key.clone(),
            solver_view_digest: run.solver_view_digest.clone(),
            findings: vec![GoldenSolverFindingV1 {
                finding_id: "f1".into(),
                statement: "more evidence required".into(),
                disposition: GoldenFindingDispositionV1::Unknown,
                confidence: 0.4,
                evidence_ids: BTreeSet::from([view.initial_evidence[0].id.clone()]),
            }],
            final_confidence: 0.4,
            abstention: None,
            submitted_at_unix_ms: 200,
        };
        let a = golden_grading_artifact_digest_v1(&view, &run, &submission).unwrap();
        let b = golden_grading_artifact_digest_v1(&view, &run, &submission).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }
}
