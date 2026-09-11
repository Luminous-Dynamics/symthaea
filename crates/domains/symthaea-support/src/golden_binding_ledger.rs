// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Append-only provenance ledger for bound Golden Incident qualification results.
//!
//! The ledger preserves the output of the Golden result-binding gate without
//! changing the generic qualification result schema. It is audit material only:
//! a ledger entry is not execution authority and in-memory append is not proof of
//! durable/atomic persistence.

use crate::golden_bound_result::{
    bind_derived_golden_qualification_result_v1, GoldenBoundQualificationResultV1,
    GoldenBoundResultErrorV1,
};
use crate::golden_incidents_v2::GoldenIncidentCorpusV2;
use crate::golden_metric_derivation::GoldenPrivateEvaluationV1;
use crate::golden_run_protocol::{GoldenRunTranscriptV1, GoldenSolverSubmissionV1};
use crate::it_qualification::{
    ItQualificationErrorV1, ItQualificationMatrixV1, ItQualificationResultV1,
    QualificationResultIdV1,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

pub const GOLDEN_QUALIFICATION_LEDGER_SCHEMA_V1: &str =
    "symthaea-golden-qualification-binding-ledger-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenQualificationLedgerEntryV1 {
    pub sequence: u64,
    pub previous_entry_digest: Option<String>,
    pub result_id: QualificationResultIdV1,
    pub binding_digest: String,
    pub binding: GoldenBoundQualificationResultV1,
    pub recorded_at_unix_ms: u64,
}

impl GoldenQualificationLedgerEntryV1 {
    pub fn digest(&self) -> Result<String, GoldenQualificationLedgerErrorV1> {
        digest_serializable("symthaea-golden-qualification-ledger-entry-v1", self)
    }

    fn validate(&self) -> Result<(), GoldenQualificationLedgerErrorV1> {
        if self.sequence == 0 {
            return Err(GoldenQualificationLedgerErrorV1::InvalidSequence(0));
        }
        if self.result_id != self.binding.lineage.result_id {
            return Err(GoldenQualificationLedgerErrorV1::ResultBindingMismatch);
        }
        validate_hex_digest(&self.binding_digest, "binding digest")?;
        if self.binding_digest != self.binding.digest()? {
            return Err(GoldenQualificationLedgerErrorV1::BindingDigestMismatch);
        }
        if let Some(previous) = &self.previous_entry_digest {
            validate_hex_digest(previous, "previous entry digest")?;
        }
        if self.recorded_at_unix_ms < self.binding.evaluated_at_unix_ms {
            return Err(GoldenQualificationLedgerErrorV1::EntryPredatesEvaluation);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenQualificationBindingLedgerV1 {
    pub schema_version: String,
    entries: Vec<GoldenQualificationLedgerEntryV1>,
}

impl Default for GoldenQualificationBindingLedgerV1 {
    fn default() -> Self {
        Self::new()
    }
}

impl GoldenQualificationBindingLedgerV1 {
    pub fn new() -> Self {
        Self {
            schema_version: GOLDEN_QUALIFICATION_LEDGER_SCHEMA_V1.into(),
            entries: Vec::new(),
        }
    }

    pub fn entries(&self) -> &[GoldenQualificationLedgerEntryV1] {
        &self.entries
    }

    pub fn entry_for_result(
        &self,
        result_id: &QualificationResultIdV1,
    ) -> Option<&GoldenQualificationLedgerEntryV1> {
        self.entries.iter().find(|entry| &entry.result_id == result_id)
    }

    pub fn head_digest(&self) -> Result<Option<String>, GoldenQualificationLedgerErrorV1> {
        self.entries.last().map(|entry| entry.digest()).transpose()
    }

    pub fn digest(&self) -> Result<String, GoldenQualificationLedgerErrorV1> {
        self.validate_chain()?;
        digest_serializable("symthaea-golden-qualification-ledger-v1", self)
    }

    pub fn validate_chain(&self) -> Result<(), GoldenQualificationLedgerErrorV1> {
        if self.schema_version != GOLDEN_QUALIFICATION_LEDGER_SCHEMA_V1 {
            return Err(GoldenQualificationLedgerErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }

        let mut result_bindings = BTreeMap::<&QualificationResultIdV1, String>::new();
        let mut prior_digest: Option<String> = None;
        for (index, entry) in self.entries.iter().enumerate() {
            entry.validate()?;
            let expected_sequence = index as u64 + 1;
            if entry.sequence != expected_sequence {
                return Err(GoldenQualificationLedgerErrorV1::InvalidSequence(
                    entry.sequence,
                ));
            }
            if entry.previous_entry_digest != prior_digest {
                return Err(GoldenQualificationLedgerErrorV1::ChainLinkMismatch {
                    sequence: entry.sequence,
                });
            }
            if let Some(existing) = result_bindings.insert(&entry.result_id, entry.binding_digest.clone()) {
                if existing != entry.binding_digest {
                    return Err(GoldenQualificationLedgerErrorV1::ResultIdentityConflict(
                        entry.result_id.clone(),
                    ));
                }
                return Err(GoldenQualificationLedgerErrorV1::DuplicateResultEntry(
                    entry.result_id.clone(),
                ));
            }
            prior_digest = Some(entry.digest()?);
        }
        Ok(())
    }

    /// Preflight an append without mutating the ledger. Exact replay is idempotent.
    pub fn can_append(
        &self,
        binding: &GoldenBoundQualificationResultV1,
        recorded_at_unix_ms: u64,
    ) -> Result<bool, GoldenQualificationLedgerErrorV1> {
        self.validate_chain()?;
        if recorded_at_unix_ms < binding.evaluated_at_unix_ms {
            return Err(GoldenQualificationLedgerErrorV1::EntryPredatesEvaluation);
        }
        let binding_digest = binding.digest()?;
        if let Some(existing) = self.entry_for_result(&binding.lineage.result_id) {
            if existing.binding_digest == binding_digest && existing.binding == *binding {
                return Ok(false);
            }
            return Err(GoldenQualificationLedgerErrorV1::ResultIdentityConflict(
                binding.lineage.result_id.clone(),
            ));
        }
        Ok(true)
    }

    /// Append one exact binding. Exact replay returns `Ok(false)` and does not create
    /// a duplicate ledger entry.
    pub fn append(
        &mut self,
        binding: GoldenBoundQualificationResultV1,
        recorded_at_unix_ms: u64,
    ) -> Result<bool, GoldenQualificationLedgerErrorV1> {
        if !self.can_append(&binding, recorded_at_unix_ms)? {
            return Ok(false);
        }
        let binding_digest = binding.digest()?;
        let previous_entry_digest = self.head_digest()?;
        let entry = GoldenQualificationLedgerEntryV1 {
            sequence: self.entries.len() as u64 + 1,
            previous_entry_digest,
            result_id: binding.lineage.result_id.clone(),
            binding_digest,
            binding,
            recorded_at_unix_ms,
        };
        entry.validate()?;
        self.entries.push(entry);
        self.validate_chain()?;
        Ok(true)
    }
}

/// Preferred in-memory Golden certification path: preflight the binding ledger,
/// record the exact qualification result, then append its exact provenance binding.
///
/// This orders validation so ledger conflicts are detected before the matrix mutates.
/// It still does not claim crash-atomic persistence across external storage systems.
pub fn record_and_ledger_derived_golden_result_v1(
    matrix: &mut ItQualificationMatrixV1,
    ledger: &mut GoldenQualificationBindingLedgerV1,
    corpus: &GoldenIncidentCorpusV2,
    transcript: &GoldenRunTranscriptV1,
    submission: &GoldenSolverSubmissionV1,
    evaluation: &GoldenPrivateEvaluationV1,
    result: ItQualificationResultV1,
    recorded_at_unix_ms: u64,
) -> Result<GoldenQualificationRecordingOutcomeV1, GoldenQualificationLedgerErrorV1> {
    let binding = bind_derived_golden_qualification_result_v1(
        matrix,
        corpus,
        transcript,
        submission,
        evaluation,
        &result,
    )?;
    let should_append = ledger.can_append(&binding, recorded_at_unix_ms)?;
    let matrix_inserted = matrix.record_result(result)?;
    let ledger_appended = if should_append {
        ledger.append(binding.clone(), recorded_at_unix_ms)?
    } else {
        false
    };
    Ok(GoldenQualificationRecordingOutcomeV1 {
        matrix_inserted,
        ledger_appended,
        binding,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoldenQualificationRecordingOutcomeV1 {
    pub matrix_inserted: bool,
    pub ledger_appended: bool,
    pub binding: GoldenBoundQualificationResultV1,
}

fn digest_serializable<T: Serialize + ?Sized>(
    domain: &'static str,
    value: &T,
) -> Result<String, GoldenQualificationLedgerErrorV1> {
    let bytes = serde_json::to_vec(&(domain, value))
        .map_err(|err| GoldenQualificationLedgerErrorV1::Serialization(err.to_string()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn validate_hex_digest(
    value: &str,
    field: &'static str,
) -> Result<(), GoldenQualificationLedgerErrorV1> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(GoldenQualificationLedgerErrorV1::InvalidDigest(field))
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub enum GoldenQualificationLedgerErrorV1 {
    BoundResult(GoldenBoundResultErrorV1),
    Qualification(ItQualificationErrorV1),
    UnsupportedSchema(String),
    Serialization(String),
    InvalidDigest(&'static str),
    InvalidSequence(u64),
    ResultBindingMismatch,
    BindingDigestMismatch,
    EntryPredatesEvaluation,
    ChainLinkMismatch { sequence: u64 },
    DuplicateResultEntry(QualificationResultIdV1),
    ResultIdentityConflict(QualificationResultIdV1),
}

impl fmt::Display for GoldenQualificationLedgerErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BoundResult(err) => write!(f, "bound golden result failed: {err}"),
            Self::Qualification(err) => write!(f, "qualification matrix failed: {err}"),
            Self::UnsupportedSchema(value) => write!(f, "unsupported golden binding ledger schema {value}"),
            Self::Serialization(message) => write!(f, "golden binding ledger serialization failed: {message}"),
            Self::InvalidDigest(field) => write!(f, "invalid 32-byte hex digest for {field}"),
            Self::InvalidSequence(value) => write!(f, "invalid golden binding ledger sequence {value}"),
            Self::ResultBindingMismatch => write!(f, "ledger result id does not match bound result lineage"),
            Self::BindingDigestMismatch => write!(f, "ledger binding digest does not match binding"),
            Self::EntryPredatesEvaluation => write!(f, "ledger entry predates private evaluation"),
            Self::ChainLinkMismatch { sequence } => write!(f, "ledger chain link mismatch at sequence {sequence}"),
            Self::DuplicateResultEntry(id) => write!(f, "duplicate ledger entry for qualification result {}", id.0),
            Self::ResultIdentityConflict(id) => write!(f, "qualification result {} was rebound to different golden provenance", id.0),
        }
    }
}

impl Error for GoldenQualificationLedgerErrorV1 {}

impl From<GoldenBoundResultErrorV1> for GoldenQualificationLedgerErrorV1 {
    fn from(value: GoldenBoundResultErrorV1) -> Self {
        Self::BoundResult(value)
    }
}

impl From<ItQualificationErrorV1> for GoldenQualificationLedgerErrorV1 {
    fn from(value: ItQualificationErrorV1) -> Self {
        Self::Qualification(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::golden_incidents_v2::seed_golden_incidents_v2;
    use crate::golden_metric_derivation::{
        derive_golden_qualification_metrics_v1, GoldenApplicabilityStatusV1,
        GoldenApplicabilityVerdictV1, GoldenFindingVerdictV1, GoldenPrivateEvaluationV1,
        GoldenRequiredFindingV1, GOLDEN_PRIVATE_EVALUATION_SCHEMA_V1,
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
        let run_id = QualificationRunIdV1("ledger-run-1".into());
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
        let result = ItQualificationResultV1 {
            id: QualificationResultIdV1("ledger-result-1".into()),
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
            evidence_artifact_digest: Some(
                golden_grading_artifact_digest_v1(&view, &transcript, &submission).unwrap(),
            ),
        };
        (matrix, corpus, transcript, submission, evaluation, result)
    }

    #[test]
    fn exact_binding_append_is_idempotent_and_chain_valid() {
        let (matrix, corpus, transcript, submission, evaluation, result) = fixture();
        let binding = bind_derived_golden_qualification_result_v1(
            &matrix,
            &corpus,
            &transcript,
            &submission,
            &evaluation,
            &result,
        )
        .unwrap();
        let mut ledger = GoldenQualificationBindingLedgerV1::new();
        assert!(ledger.append(binding.clone(), 240).unwrap());
        assert!(!ledger.append(binding, 240).unwrap());
        ledger.validate_chain().unwrap();
        assert_eq!(ledger.entries().len(), 1);
        assert_eq!(ledger.head_digest().unwrap().unwrap().len(), 64);
    }

    #[test]
    fn record_and_ledger_path_preserves_bound_provenance() {
        let (mut matrix, corpus, transcript, submission, evaluation, result) = fixture();
        let mut ledger = GoldenQualificationBindingLedgerV1::new();
        let outcome = record_and_ledger_derived_golden_result_v1(
            &mut matrix,
            &mut ledger,
            &corpus,
            &transcript,
            &submission,
            &evaluation,
            result,
            240,
        )
        .unwrap();
        assert!(outcome.matrix_inserted);
        assert!(outcome.ledger_appended);
        assert_eq!(matrix.results().count(), 1);
        assert_eq!(ledger.entries().len(), 1);
    }

    #[test]
    fn tampered_chain_link_is_detected() {
        let (matrix, corpus, transcript, submission, evaluation, result) = fixture();
        let binding = bind_derived_golden_qualification_result_v1(
            &matrix,
            &corpus,
            &transcript,
            &submission,
            &evaluation,
            &result,
        )
        .unwrap();
        let mut ledger = GoldenQualificationBindingLedgerV1::new();
        ledger.append(binding, 240).unwrap();
        ledger.entries[0].previous_entry_digest = Some(digest('f'));
        assert!(matches!(
            ledger.validate_chain(),
            Err(GoldenQualificationLedgerErrorV1::ChainLinkMismatch { sequence: 1 })
        ));
    }
}
