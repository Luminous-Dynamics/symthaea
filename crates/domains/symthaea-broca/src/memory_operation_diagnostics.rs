// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Operation-level diagnosis for long-term memory evaluation.
//!
//! End-to-end QA accuracy hides whether a failure came from encoding,
//! retrieval or generation. This module records independent examiner outcomes
//! and emits multi-label defect codes. It is designed for paired oracle
//! evaluation: the encoder examiner checks whether the gold fact exists in the
//! memory store, the retriever examiner checks whether usable gold evidence is
//! returned, and the generator examiner checks whether the answer uses that
//! evidence faithfully.
//!
//! This is diagnostic accounting, not a natural-language judge. The caller
//! must supply labels from a frozen gold protocol, source-span checks or
//! blinded human adjudication. The emitted report binds the observations and
//! protocol digest so later comparisons cannot quietly mix evaluation rubrics.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const MEMORY_DIAGNOSTICS_PROTOCOL_VERSION: &str = "BROCA-MEMORY-DIAGNOSTICS-V1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncodingExaminerState {
    GoldEvidencePresent,
    GoldEvidenceAmbiguous,
    GoldEvidenceWrong,
    GoldEvidenceMissing,
    NotEvaluated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalExaminerState {
    UsableGoldEvidenceRetrieved,
    GoldEvidenceRetrievedLate,
    GoldEvidenceRetrievedWithNoise,
    GoldEvidenceMissed,
    NegativeEvidenceIntrusion,
    NotEvaluated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationExaminerState {
    CorrectAndSupported,
    HallucinatedWithoutSupport,
    IgnoredOracleEvidence,
    ReasoningFailureWithEvidence,
    FalseAbstention,
    NotEvaluated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryOperationDefect {
    EncodingMissing,
    EncodingAmbiguous,
    EncodingWrong,
    RetrievalFailure,
    RetrievalLate,
    RetrievalNoise,
    NoiseInducedRetrieval,
    GenerationHallucination,
    GenerationFaithfulness,
    GenerationReasoningFailure,
    FalseAbstention,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryOperationObservation {
    pub question_id: String,
    pub encoding_state: EncodingExaminerState,
    pub retrieval_state: RetrievalExaminerState,
    pub generation_state: GenerationExaminerState,
}

impl MemoryOperationObservation {
    pub fn validate(&self) -> Result<(), String> {
        require_non_empty(&self.question_id, "question_id")?;
        if self.encoding_state == EncodingExaminerState::NotEvaluated
            && self.retrieval_state == RetrievalExaminerState::NotEvaluated
            && self.generation_state == GenerationExaminerState::NotEvaluated
        {
            return Err("observation has no evaluated operation layer".into());
        }
        Ok(())
    }

    pub fn defects(&self) -> BTreeSet<MemoryOperationDefect> {
        let mut defects = BTreeSet::new();
        match self.encoding_state {
            EncodingExaminerState::GoldEvidencePresent
            | EncodingExaminerState::NotEvaluated => {}
            EncodingExaminerState::GoldEvidenceAmbiguous => {
                defects.insert(MemoryOperationDefect::EncodingAmbiguous);
            }
            EncodingExaminerState::GoldEvidenceWrong => {
                defects.insert(MemoryOperationDefect::EncodingWrong);
            }
            EncodingExaminerState::GoldEvidenceMissing => {
                defects.insert(MemoryOperationDefect::EncodingMissing);
            }
        }

        match self.retrieval_state {
            RetrievalExaminerState::UsableGoldEvidenceRetrieved
            | RetrievalExaminerState::NotEvaluated => {}
            RetrievalExaminerState::GoldEvidenceMissed => {
                if self.encoding_state == EncodingExaminerState::GoldEvidencePresent {
                    defects.insert(MemoryOperationDefect::RetrievalFailure);
                }
            }
            RetrievalExaminerState::GoldEvidenceRetrievedLate => {
                if self.encoding_state == EncodingExaminerState::GoldEvidencePresent {
                    defects.insert(MemoryOperationDefect::RetrievalLate);
                }
            }
            RetrievalExaminerState::GoldEvidenceRetrievedWithNoise => {
                if self.encoding_state == EncodingExaminerState::GoldEvidencePresent {
                    defects.insert(MemoryOperationDefect::RetrievalNoise);
                }
            }
            RetrievalExaminerState::NegativeEvidenceIntrusion => {
                defects.insert(MemoryOperationDefect::NoiseInducedRetrieval);
            }
        }

        match self.generation_state {
            GenerationExaminerState::CorrectAndSupported
            | GenerationExaminerState::NotEvaluated => {}
            GenerationExaminerState::HallucinatedWithoutSupport => {
                defects.insert(MemoryOperationDefect::GenerationHallucination);
            }
            GenerationExaminerState::IgnoredOracleEvidence => {
                defects.insert(MemoryOperationDefect::GenerationFaithfulness);
            }
            GenerationExaminerState::ReasoningFailureWithEvidence => {
                defects.insert(MemoryOperationDefect::GenerationReasoningFailure);
            }
            GenerationExaminerState::FalseAbstention => {
                defects.insert(MemoryOperationDefect::FalseAbstention);
            }
        }
        defects
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryOperationDefectCount {
    pub defect: MemoryOperationDefect,
    pub count: usize,
    pub rate_basis_points: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryOperationDiagnosticReport {
    pub protocol_version: String,
    pub benchmark_id: String,
    pub benchmark_revision: String,
    pub evaluation_protocol_sha256: String,
    pub observation_count: usize,
    pub fully_correct_count: usize,
    pub per_defect: Vec<MemoryOperationDefectCount>,
    pub observations_sha256: String,
    pub report_sha256: String,
}

impl MemoryOperationDiagnosticReport {
    pub fn verify_internal_consistency(&self) -> Result<(), String> {
        if self.protocol_version != MEMORY_DIAGNOSTICS_PROTOCOL_VERSION {
            return Err("unsupported memory diagnostics protocol version".into());
        }
        require_non_empty(&self.benchmark_id, "benchmark_id")?;
        require_non_empty(&self.benchmark_revision, "benchmark_revision")?;
        validate_sha256(&self.evaluation_protocol_sha256, "evaluation_protocol_sha256")?;
        validate_sha256(&self.observations_sha256, "observations_sha256")?;
        validate_sha256(&self.report_sha256, "report_sha256")?;
        if self.observation_count == 0 || self.fully_correct_count > self.observation_count {
            return Err("invalid memory diagnostic observation counts".into());
        }
        let mut seen = BTreeSet::new();
        for entry in &self.per_defect {
            if !seen.insert(entry.defect) {
                return Err("duplicate defect code in diagnostic report".into());
            }
            if entry.count > self.observation_count || entry.rate_basis_points > 10_000 {
                return Err("invalid per-defect count or rate".into());
            }
            let expected_rate = rate_basis_points(entry.count, self.observation_count);
            if entry.rate_basis_points != expected_rate {
                return Err(format!("incorrect rate for defect {:?}", entry.defect));
            }
        }
        if self.computed_report_sha256()? != self.report_sha256 {
            return Err("memory diagnostic report digest mismatch".into());
        }
        Ok(())
    }

    fn computed_report_sha256(&self) -> Result<String, String> {
        let mut canonical = self.clone();
        canonical.report_sha256.clear();
        hash_serializable(&canonical, "memory diagnostic report")
    }
}

/// Aggregate independent examiner outputs into a reproducible multi-label
/// report. One query may carry several defects; rates are per query, not a
/// mutually-exclusive classification whose values must sum to 100%.
pub fn diagnose_memory_operations(
    benchmark_id: &str,
    benchmark_revision: &str,
    evaluation_protocol_sha256: &str,
    observations: &[MemoryOperationObservation],
) -> Result<MemoryOperationDiagnosticReport, String> {
    require_non_empty(benchmark_id, "benchmark_id")?;
    require_non_empty(benchmark_revision, "benchmark_revision")?;
    validate_sha256(evaluation_protocol_sha256, "evaluation_protocol_sha256")?;
    if observations.is_empty() {
        return Err("memory diagnostics require at least one observation".into());
    }

    let mut sorted = observations.to_vec();
    sorted.sort_by(|left, right| left.question_id.cmp(&right.question_id));
    let mut question_ids = BTreeSet::new();
    let mut defect_counts = BTreeMap::<MemoryOperationDefect, usize>::new();
    let mut fully_correct_count = 0usize;

    for observation in &sorted {
        observation.validate()?;
        if !question_ids.insert(observation.question_id.as_str()) {
            return Err(format!("duplicate memory diagnostic question ID {:?}", observation.question_id));
        }
        let defects = observation.defects();
        if defects.is_empty()
            && observation.encoding_state == EncodingExaminerState::GoldEvidencePresent
            && observation.retrieval_state == RetrievalExaminerState::UsableGoldEvidenceRetrieved
            && observation.generation_state == GenerationExaminerState::CorrectAndSupported
        {
            fully_correct_count += 1;
        }
        for defect in defects {
            *defect_counts.entry(defect).or_insert(0) += 1;
        }
    }

    let per_defect: Vec<MemoryOperationDefectCount> = defect_counts
        .into_iter()
        .map(|(defect, count)| MemoryOperationDefectCount {
            defect,
            count,
            rate_basis_points: rate_basis_points(count, sorted.len()),
        })
        .collect();
    let observations_sha256 = hash_serializable(&sorted, "memory diagnostic observations")?;
    let mut report = MemoryOperationDiagnosticReport {
        protocol_version: MEMORY_DIAGNOSTICS_PROTOCOL_VERSION.into(),
        benchmark_id: benchmark_id.into(),
        benchmark_revision: benchmark_revision.into(),
        evaluation_protocol_sha256: evaluation_protocol_sha256.into(),
        observation_count: sorted.len(),
        fully_correct_count,
        per_defect,
        observations_sha256,
        report_sha256: String::new(),
    };
    report.report_sha256 = report.computed_report_sha256()?;
    report.verify_internal_consistency()?;
    Ok(report)
}

fn rate_basis_points(count: usize, denominator: usize) -> u32 {
    if denominator == 0 {
        return 0;
    }
    ((count as u128 * 10_000) / denominator as u128) as u32
}

fn hash_serializable<T: Serialize>(value: &T, label: &str) -> Result<String, String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("cannot serialize {label}: {error}"))?;
    Ok(sha256_hex(&bytes))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_sha256(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{field} must be a 64-character lowercase SHA-256 digest"));
    }
    Ok(())
}

fn require_non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(format!("{field} must be non-empty and trimmed"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: &str) -> String {
        sha256_hex(value.as_bytes())
    }

    #[test]
    fn emits_multiple_operation_defects_for_one_query() {
        let observation = MemoryOperationObservation {
            question_id: "Q-1".into(),
            encoding_state: EncodingExaminerState::GoldEvidencePresent,
            retrieval_state: RetrievalExaminerState::GoldEvidenceMissed,
            generation_state: GenerationExaminerState::HallucinatedWithoutSupport,
        };
        let defects = observation.defects();
        assert!(defects.contains(&MemoryOperationDefect::RetrievalFailure));
        assert!(defects.contains(&MemoryOperationDefect::GenerationHallucination));
        assert_eq!(defects.len(), 2);
    }

    #[test]
    fn does_not_blame_retrieval_when_encoding_never_stored_the_fact() {
        let observation = MemoryOperationObservation {
            question_id: "Q-2".into(),
            encoding_state: EncodingExaminerState::GoldEvidenceMissing,
            retrieval_state: RetrievalExaminerState::GoldEvidenceMissed,
            generation_state: GenerationExaminerState::HallucinatedWithoutSupport,
        };
        let defects = observation.defects();
        assert!(defects.contains(&MemoryOperationDefect::EncodingMissing));
        assert!(!defects.contains(&MemoryOperationDefect::RetrievalFailure));
        assert!(defects.contains(&MemoryOperationDefect::GenerationHallucination));
    }

    #[test]
    fn reports_noise_late_retrieval_and_non_faithful_generation_as_distinct_defects() {
        let late = MemoryOperationObservation {
            question_id: "Q-late".into(),
            encoding_state: EncodingExaminerState::GoldEvidencePresent,
            retrieval_state: RetrievalExaminerState::GoldEvidenceRetrievedLate,
            generation_state: GenerationExaminerState::IgnoredOracleEvidence,
        };
        let noisy = MemoryOperationObservation {
            question_id: "Q-noisy".into(),
            encoding_state: EncodingExaminerState::GoldEvidencePresent,
            retrieval_state: RetrievalExaminerState::GoldEvidenceRetrievedWithNoise,
            generation_state: GenerationExaminerState::ReasoningFailureWithEvidence,
        };
        assert!(late.defects().contains(&MemoryOperationDefect::RetrievalLate));
        assert!(late.defects().contains(&MemoryOperationDefect::GenerationFaithfulness));
        assert!(noisy.defects().contains(&MemoryOperationDefect::RetrievalNoise));
        assert!(noisy.defects().contains(&MemoryOperationDefect::GenerationReasoningFailure));
    }

    #[test]
    fn builds_deterministic_multilabel_report_with_per_query_rates() {
        let observations = vec![
            MemoryOperationObservation {
                question_id: "Q-2".into(),
                encoding_state: EncodingExaminerState::GoldEvidencePresent,
                retrieval_state: RetrievalExaminerState::UsableGoldEvidenceRetrieved,
                generation_state: GenerationExaminerState::CorrectAndSupported,
            },
            MemoryOperationObservation {
                question_id: "Q-1".into(),
                encoding_state: EncodingExaminerState::GoldEvidencePresent,
                retrieval_state: RetrievalExaminerState::GoldEvidenceMissed,
                generation_state: GenerationExaminerState::HallucinatedWithoutSupport,
            },
        ];
        let first = diagnose_memory_operations(
            "BEAM",
            "10m-v1",
            &digest("rubric"),
            &observations,
        ).unwrap();
        let second = diagnose_memory_operations(
            "BEAM",
            "10m-v1",
            &digest("rubric"),
            &observations.iter().cloned().rev().collect::<Vec<_>>(),
        ).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.fully_correct_count, 1);
        assert_eq!(first.observation_count, 2);
        assert_eq!(first.per_defect[0].count, 1);
        first.verify_internal_consistency().unwrap();
    }

    #[test]
    fn rejects_duplicate_question_ids_and_unrated_observations() {
        let invalid = MemoryOperationObservation {
            question_id: "Q-1".into(),
            encoding_state: EncodingExaminerState::NotEvaluated,
            retrieval_state: RetrievalExaminerState::NotEvaluated,
            generation_state: GenerationExaminerState::NotEvaluated,
        };
        assert!(diagnose_memory_operations(
            "benchmark", "v1", &digest("rubric"), &[invalid.clone()]
        ).unwrap_err().contains("no evaluated operation layer"));

        let mut second = invalid;
        second.generation_state = GenerationExaminerState::CorrectAndSupported;
        let first = second.clone();
        assert!(diagnose_memory_operations(
            "benchmark", "v1", &digest("rubric"), &[first, second]
        ).unwrap_err().contains("duplicate memory diagnostic question ID"));
    }

    #[test]
    fn report_rejects_tampering() {
        let observation = MemoryOperationObservation {
            question_id: "Q-1".into(),
            encoding_state: EncodingExaminerState::GoldEvidencePresent,
            retrieval_state: RetrievalExaminerState::UsableGoldEvidenceRetrieved,
            generation_state: GenerationExaminerState::CorrectAndSupported,
        };
        let mut report = diagnose_memory_operations(
            "LoCoMo", "v1", &digest("rubric"), &[observation]
        ).unwrap();
        report.fully_correct_count = 0;
        assert!(report.verify_internal_consistency().is_err());
    }
}
