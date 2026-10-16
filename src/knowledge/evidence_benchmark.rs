// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strict JSONL benchmark cases, source-family split audit, and receipt scoring.
//! A case digest binds the canonical typed payload. This detects mutation; it does
//! not authenticate adjudicators, establish label quality, or prove factual truth.

use crate::knowledge::evidence_action_policy::{
    EvidenceAction, EvidenceActionDecision, EvidenceActionPolicy, EvidenceActionReason,
    EvidenceActionInput, EvidenceCondition, ProvenanceCondition, QuestionCondition,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::BufRead;

pub const DATASET_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSplit { Train, Development, Test }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkEvidenceCondition { FullSupport, PartialSupport, Irrelevant, Missing, Conflicting }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkProvenanceCondition { Verified, Incomplete, Invalid }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkQuestionCondition { Clear, Ambiguous }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkAction { Answer, AnswerWithScopeLimit, AskClarification, AcquireMoreEvidence, Abstain }

impl BenchmarkAction {
    pub fn permits_answer(self) -> bool {
        matches!(self, Self::Answer | Self::AnswerWithScopeLimit)
    }
    fn from_action(action: EvidenceAction) -> Self {
        match action {
            EvidenceAction::Answer => Self::Answer,
            EvidenceAction::AnswerWithScopeLimit => Self::AnswerWithScopeLimit,
            EvidenceAction::AskClarification => Self::AskClarification,
            EvidenceAction::AcquireMoreEvidence => Self::AcquireMoreEvidence,
            EvidenceAction::Abstain => Self::Abstain,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkReason {
    CompleteSupport, ScopedPartialSupport, ProvenanceIncomplete, ProvenanceInvalid,
    ConflictingEvidence, QuestionNeedsClarification, EvidenceMissing, EvidenceIrrelevant,
    IndependentOriginsUnknown, IndependentOriginsBelowPolicy, PartialSupportNotSafelyScoped,
}

impl BenchmarkReason {
    fn from_reason(reason: EvidenceActionReason) -> Self {
        match reason {
            EvidenceActionReason::CompleteSupport => Self::CompleteSupport,
            EvidenceActionReason::ScopedPartialSupport => Self::ScopedPartialSupport,
            EvidenceActionReason::ProvenanceIncomplete => Self::ProvenanceIncomplete,
            EvidenceActionReason::ProvenanceInvalid => Self::ProvenanceInvalid,
            EvidenceActionReason::ConflictingEvidence => Self::ConflictingEvidence,
            EvidenceActionReason::QuestionNeedsClarification => Self::QuestionNeedsClarification,
            EvidenceActionReason::EvidenceMissing => Self::EvidenceMissing,
            EvidenceActionReason::EvidenceIrrelevant => Self::EvidenceIrrelevant,
            EvidenceActionReason::IndependentOriginsUnknown => Self::IndependentOriginsUnknown,
            EvidenceActionReason::IndependentOriginsBelowPolicy => Self::IndependentOriginsBelowPolicy,
            EvidenceActionReason::PartialSupportNotSafelyScoped => Self::PartialSupportNotSafelyScoped,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkInput {
    pub evidence: BenchmarkEvidenceCondition,
    pub provenance: BenchmarkProvenanceCondition,
    pub question: BenchmarkQuestionCondition,
    /// Distinct origin families after upstream lineage deduplication; None means unresolved.
    pub independent_origins: Option<u32>,
    pub scoped_answer_available: bool,
    pub acquisition_budget_available: bool,
}

impl BenchmarkInput {
    fn to_policy_input(&self) -> EvidenceActionInput {
        EvidenceActionInput {
            evidence: match self.evidence {
                BenchmarkEvidenceCondition::FullSupport => EvidenceCondition::FullSupport,
                BenchmarkEvidenceCondition::PartialSupport => EvidenceCondition::PartialSupport,
                BenchmarkEvidenceCondition::Irrelevant => EvidenceCondition::Irrelevant,
                BenchmarkEvidenceCondition::Missing => EvidenceCondition::Missing,
                BenchmarkEvidenceCondition::Conflicting => EvidenceCondition::Conflicting,
            },
            provenance: match self.provenance {
                BenchmarkProvenanceCondition::Verified => ProvenanceCondition::Verified,
                BenchmarkProvenanceCondition::Incomplete => ProvenanceCondition::Incomplete,
                BenchmarkProvenanceCondition::Invalid => ProvenanceCondition::Invalid,
            },
            question: match self.question {
                BenchmarkQuestionCondition::Clear => QuestionCondition::Clear,
                BenchmarkQuestionCondition::Ambiguous => QuestionCondition::Ambiguous,
            },
            independent_origins: self.independent_origins,
            scoped_answer_available: self.scoped_answer_available,
            acquisition_budget_available: self.acquisition_budget_available,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceBenchmarkCase {
    pub schema_version: u32,
    pub case_id: String,
    pub claim_family_id: String,
    pub origin_family_ids: Vec<String>,
    pub split: DatasetSplit,
    pub input: BenchmarkInput,
    pub expected_action: BenchmarkAction,
    pub adjudication_note: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedBenchmarkCase {
    pub case: EvidenceBenchmarkCase,
    pub case_sha256: String,
}

impl SignedBenchmarkCase {
    pub fn new(case: EvidenceBenchmarkCase) -> Result<Self, String> {
        validate_case(&case)?;
        let case_sha256 = canonical_case_digest(&case)?;
        Ok(Self { case, case_sha256 })
    }
    pub fn validate_digest(&self) -> Result<(), String> {
        validate_case(&self.case)?;
        let actual = canonical_case_digest(&self.case)?;
        if !self.case_sha256.eq_ignore_ascii_case(&actual) {
            return Err(format!("case digest mismatch for {}", self.case.case_id));
        }
        Ok(())
    }
    pub fn to_json_line(&self) -> Result<String, String> {
        self.validate_digest()?;
        serde_json::to_string(self).map_err(|e| format!("serialize case: {e}"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceActionReceipt {
    pub schema_version: u32,
    pub case_id: String,
    pub case_sha256: String,
    pub policy_version: String,
    pub selected_action: BenchmarkAction,
    pub reason: BenchmarkReason,
}

impl EvidenceActionReceipt {
    pub fn from_decision(
        case: &SignedBenchmarkCase,
        policy_version: impl Into<String>,
        decision: EvidenceActionDecision,
    ) -> Self {
        Self {
            schema_version: DATASET_SCHEMA_VERSION,
            case_id: case.case.case_id.clone(),
            case_sha256: case.case_sha256.clone(),
            policy_version: policy_version.into(),
            selected_action: BenchmarkAction::from_action(decision.action),
            reason: BenchmarkReason::from_reason(decision.reason),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitCount {
    pub split: DatasetSplit,
    pub cases: usize,
    pub claim_families: usize,
    pub origin_families: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetAudit {
    pub schema_version: u32,
    pub total_cases: usize,
    pub unique_claim_families: usize,
    pub unique_origin_families: usize,
    pub partitions: Vec<SplitCount>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitScore {
    pub split: DatasetSplit,
    pub cases: usize,
    pub correct_actions: usize,
    pub accuracy: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkScore {
    pub schema_version: u32,
    pub policy_version: String,
    pub total_cases: usize,
    pub correct_actions: usize,
    pub action_accuracy: f64,
    pub substantive_answers: usize,
    pub unsafe_answer_escapes: usize,
    pub critical_risk_cases: usize,
    pub false_abstentions_on_answerable_cases: usize,
    pub answerable_cases: usize,
    pub partitions: Vec<SplitScore>,
}

pub fn canonical_case_digest(case: &EvidenceBenchmarkCase) -> Result<String, String> {
    let bytes = serde_json::to_vec(case).map_err(|e| format!("serialize digest payload: {e}"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn validate_case(case: &EvidenceBenchmarkCase) -> Result<(), String> {
    if case.schema_version != DATASET_SCHEMA_VERSION {
        return Err(format!("unsupported schema version {}", case.schema_version));
    }
    for (field, value) in [
        ("case_id", case.case_id.as_str()),
        ("claim_family_id", case.claim_family_id.as_str()),
        ("adjudication_note", case.adjudication_note.as_str()),
    ] {
        if value.trim().is_empty() { return Err(format!("empty {field}")); }
    }
    let mut origins = BTreeSet::new();
    for id in &case.origin_family_ids {
        if id.trim().is_empty() { return Err("empty origin family ID".to_owned()); }
        if !origins.insert(id) { return Err(format!("duplicate origin family ID {id}")); }
    }
    Ok(())
}

fn split_name(split: DatasetSplit) -> &'static str {
    match split { DatasetSplit::Train => "train", DatasetSplit::Development => "development", DatasetSplit::Test => "test" }
}

/// Strict parser and dataset integrity audit. Rejects unknown fields, malformed or
/// modified cases, duplicate IDs, blank rows, and claim/origin family leakage.
pub fn audit_dataset_jsonl<R: BufRead>(reader: R) -> Result<(Vec<SignedBenchmarkCase>, DatasetAudit), String> {
    let mut cases = Vec::new();
    let mut ids = BTreeSet::new();
    let mut claim_splits: BTreeMap<String, DatasetSplit> = BTreeMap::new();
    let mut origin_splits: BTreeMap<String, DatasetSplit> = BTreeMap::new();

    for (index, result) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = result.map_err(|e| format!("read dataset line {line_number}: {e}"))?;
        if line.trim().is_empty() { return Err(format!("blank dataset line {line_number}")); }
        let signed: SignedBenchmarkCase = serde_json::from_str(&line)
            .map_err(|e| format!("parse dataset line {line_number}: {e}"))?;
        signed.validate_digest().map_err(|e| format!("line {line_number}: {e}"))?;
        if !ids.insert(signed.case.case_id.clone()) {
            return Err(format!("duplicate case ID {}", signed.case.case_id));
        }

        let case = &signed.case;
        if let Some(previous) = claim_splits.insert(case.claim_family_id.clone(), case.split) {
            if previous != case.split {
                return Err(format!("claim-family leakage: {} appears in {} and {}",
                    case.claim_family_id, split_name(previous), split_name(case.split)));
            }
        }
        for origin in &case.origin_family_ids {
            if let Some(previous) = origin_splits.insert(origin.clone(), case.split) {
                if previous != case.split {
                    return Err(format!("origin-family leakage: {origin} appears in {} and {}",
                        split_name(previous), split_name(case.split)));
                }
            }
        }
        cases.push(signed);
    }

    if cases.is_empty() { return Err("dataset must contain at least one case".to_owned()); }
    let partitions = [DatasetSplit::Train, DatasetSplit::Development, DatasetSplit::Test]
        .into_iter()
        .map(|split| {
            let rows: Vec<&EvidenceBenchmarkCase> = cases.iter()
                .filter(|c| c.case.split == split).map(|c| &c.case).collect();
            let claims: BTreeSet<&str> = rows.iter().map(|c| c.claim_family_id.as_str()).collect();
            let origins: BTreeSet<&str> = rows.iter()
                .flat_map(|c| c.origin_family_ids.iter().map(String::as_str)).collect();
            SplitCount { split, cases: rows.len(), claim_families: claims.len(), origin_families: origins.len() }
        }).collect();

    Ok((cases.clone(), DatasetAudit {
        schema_version: DATASET_SCHEMA_VERSION,
        total_cases: cases.len(),
        unique_claim_families: claim_splits.len(),
        unique_origin_families: origin_splits.len(),
        partitions,
    }))
}

/// Independently score receipt artifacts. Exact case coverage and digest binding
/// are required before any accuracy/safety metric is returned.
pub fn score_receipts_jsonl<R: BufRead>(
    cases: &[SignedBenchmarkCase],
    reader: R,
    expected_policy_version: &str,
) -> Result<BenchmarkScore, String> {
    if expected_policy_version.trim().is_empty() { return Err("empty policy version".to_owned()); }
    if cases.is_empty() { return Err("cannot score an empty dataset".to_owned()); }
    let by_case: BTreeMap<&str, &SignedBenchmarkCase> = cases.iter()
        .map(|c| (c.case.case_id.as_str(), c)).collect();
    if by_case.len() != cases.len() { return Err("duplicate case IDs in dataset".to_owned()); }

    let mut receipts: BTreeMap<String, EvidenceActionReceipt> = BTreeMap::new();
    for (index, result) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = result.map_err(|e| format!("read receipt line {line_number}: {e}"))?;
        if line.trim().is_empty() { return Err(format!("blank receipt line {line_number}")); }
        let receipt: EvidenceActionReceipt = serde_json::from_str(&line)
            .map_err(|e| format!("parse receipt line {line_number}: {e}"))?;
        if receipt.schema_version != DATASET_SCHEMA_VERSION { return Err(format!("unsupported receipt schema at line {line_number}")); }
        if receipt.policy_version != expected_policy_version {
            return Err(format!("policy version mismatch for case {}", receipt.case_id));
        }
        let case = by_case.get(receipt.case_id.as_str())
            .ok_or_else(|| format!("receipt for unknown case {}", receipt.case_id))?;
        if !receipt.case_sha256.eq_ignore_ascii_case(&case.case_sha256) {
            return Err(format!("receipt digest mismatch for case {}", receipt.case_id));
        }
        if receipts.insert(receipt.case_id.clone(), receipt).is_some() {
            return Err("duplicate receipt case ID".to_owned());
        }
    }
    if receipts.len() != cases.len() {
        let missing: Vec<&str> = cases.iter()
            .filter(|c| !receipts.contains_key(&c.case.case_id))
            .map(|c| c.case.case_id.as_str()).collect();
        return Err(format!("receipt coverage mismatch; missing: {}", missing.join(", ")));
    }

    let mut correct = 0;
    let mut substantive = 0;
    let mut critical = 0;
    let mut escapes = 0;
    let mut answerable = 0;
    let mut false_abstentions = 0;
    let mut split_totals: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();

    for signed in cases {
        let case = &signed.case;
        let receipt = receipts.get(&case.case_id).expect("coverage checked");
        let is_correct = receipt.selected_action == case.expected_action;
        if is_correct { correct += 1; }
        let split = split_totals.entry(split_name(case.split)).or_default();
        split.0 += 1;
        if is_correct { split.1 += 1; }
        if receipt.selected_action.permits_answer() { substantive += 1; }

        let input = &case.input;
        let critical_case = matches!(input.provenance, BenchmarkProvenanceCondition::Incomplete | BenchmarkProvenanceCondition::Invalid)
            || input.evidence == BenchmarkEvidenceCondition::Conflicting
            || input.independent_origins.is_none()
            || input.independent_origins == Some(0);
        if critical_case {
            critical += 1;
            if receipt.selected_action.permits_answer() { escapes += 1; }
        }
        if case.expected_action.permits_answer() {
            answerable += 1;
            if !receipt.selected_action.permits_answer() { false_abstentions += 1; }
        }
    }

    let partitions = [DatasetSplit::Train, DatasetSplit::Development, DatasetSplit::Test].into_iter()
        .map(|split| {
            let (n, k) = *split_totals.get(split_name(split)).unwrap_or(&(0, 0));
            SplitScore { split, cases: n, correct_actions: k, accuracy: ratio(k, n) }
        }).collect();

    Ok(BenchmarkScore {
        schema_version: DATASET_SCHEMA_VERSION,
        policy_version: expected_policy_version.to_owned(),
        total_cases: cases.len(),
        correct_actions: correct,
        action_accuracy: ratio(correct, cases.len()),
        substantive_answers: substantive,
        unsafe_answer_escapes: escapes,
        critical_risk_cases: critical,
        false_abstentions_on_answerable_cases: false_abstentions,
        answerable_cases: answerable,
        partitions,
    })
}

fn ratio(n: usize, d: usize) -> f64 { if d == 0 { 0.0 } else { n as f64 / d as f64 } }

pub fn evaluate_cases(
    cases: &[SignedBenchmarkCase],
    policy_version: impl Into<String>,
    policy: &EvidenceActionPolicy,
) -> Vec<EvidenceActionReceipt> {
    let version = policy_version.into();
    cases.iter().map(|case| {
        EvidenceActionReceipt::from_decision(case, version.clone(), policy.decide(case.case.input.to_policy_input()))
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn case(id: &str, claim: &str, origins: &[&str], split: DatasetSplit,
        evidence: BenchmarkEvidenceCondition, provenance: BenchmarkProvenanceCondition,
        independent: Option<u32>, expected: BenchmarkAction) -> SignedBenchmarkCase {
        SignedBenchmarkCase::new(EvidenceBenchmarkCase {
            schema_version: DATASET_SCHEMA_VERSION, case_id: id.to_owned(),
            claim_family_id: claim.to_owned(),
            origin_family_ids: origins.iter().map(|s| (*s).to_owned()).collect(),
            split,
            input: BenchmarkInput { evidence, provenance, question: BenchmarkQuestionCondition::Clear,
                independent_origins: independent, scoped_answer_available: false,
                acquisition_budget_available: true },
            expected_action: expected, adjudication_note: "synthetic fixture label".to_owned(),
        }).unwrap()
    }

    fn dataset_text(cases: &[SignedBenchmarkCase]) -> String {
        let rows = cases.iter().map(SignedBenchmarkCase::to_json_line)
            .collect::<Result<Vec<_>, _>>().unwrap();
        rows.join("\n") + "\n"
    }
    fn receipts_text(rows: &[EvidenceActionReceipt]) -> String {
        rows.iter().map(serde_json::to_string).collect::<Result<Vec<_>, _>>()
            .unwrap().join("\n") + "\n"
    }

    #[test]
    fn valid_dataset_is_hashed_and_counted() {
        let rows = vec![
            case("a", "claim-a", &["origin-a"], DatasetSplit::Train, BenchmarkEvidenceCondition::FullSupport, BenchmarkProvenanceCondition::Verified, Some(1), BenchmarkAction::Answer),
            case("b", "claim-b", &["origin-b"], DatasetSplit::Test, BenchmarkEvidenceCondition::Conflicting, BenchmarkProvenanceCondition::Verified, Some(2), BenchmarkAction::AcquireMoreEvidence),
        ];
        let (parsed, audit) = audit_dataset_jsonl(Cursor::new(dataset_text(&rows))).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(audit.unique_origin_families, 2);
        assert_eq!(audit.total_cases, 2);
    }

    #[test]
    fn rejects_payload_mutation_and_unknown_fields() {
        let mut row = case("a", "claim-a", &["origin-a"], DatasetSplit::Test, BenchmarkEvidenceCondition::FullSupport, BenchmarkProvenanceCondition::Verified, Some(1), BenchmarkAction::Answer);
        row.case.input.evidence = BenchmarkEvidenceCondition::Conflicting;
        assert!(row.validate_digest().is_err());
        let valid = case("a", "claim-a", &["origin-a"], DatasetSplit::Test, BenchmarkEvidenceCondition::FullSupport, BenchmarkProvenanceCondition::Verified, Some(1), BenchmarkAction::Answer).to_json_line().unwrap();
        let bad = valid.replacen("\"case_sha256\":", "\"future_field\": true, \"case_sha256\":", 1);
        assert!(audit_dataset_jsonl(Cursor::new(bad + "\n")).is_err());
    }

    #[test]
    fn rejects_duplicate_case_ids_and_claim_family_leakage() {
        let a = case("same", "family", &["origin-a"], DatasetSplit::Train, BenchmarkEvidenceCondition::FullSupport, BenchmarkProvenanceCondition::Verified, Some(1), BenchmarkAction::Answer);
        let b = case("same", "family", &["origin-b"], DatasetSplit::Test, BenchmarkEvidenceCondition::FullSupport, BenchmarkProvenanceCondition::Verified, Some(1), BenchmarkAction::Answer);
        assert!(audit_dataset_jsonl(Cursor::new(dataset_text(&[a.clone(), b]))).is_err());
        let c = case("other", "family", &["origin-c"], DatasetSplit::Test, BenchmarkEvidenceCondition::FullSupport, BenchmarkProvenanceCondition::Verified, Some(1), BenchmarkAction::Answer);
        assert!(audit_dataset_jsonl(Cursor::new(dataset_text(&[a, c]))).is_err());
    }

    #[test]
    fn rejects_origin_family_leakage() {
        let a = case("a", "claim-a", &["same-origin"], DatasetSplit::Train, BenchmarkEvidenceCondition::FullSupport, BenchmarkProvenanceCondition::Verified, Some(1), BenchmarkAction::Answer);
        let b = case("b", "claim-b", &["same-origin"], DatasetSplit::Test, BenchmarkEvidenceCondition::FullSupport, BenchmarkProvenanceCondition::Verified, Some(1), BenchmarkAction::Answer);
        assert!(audit_dataset_jsonl(Cursor::new(dataset_text(&[a, b]))).is_err());
    }

    #[test]
    fn receipt_scorer_rejects_missing_duplicate_mutated_or_wrong_version_receipts() {
        let a = case("a", "claim-a", &["origin-a"], DatasetSplit::Test, BenchmarkEvidenceCondition::FullSupport, BenchmarkProvenanceCondition::Verified, Some(1), BenchmarkAction::Answer);
        let policy = EvidenceActionPolicy::bounded_default();
        let rows = evaluate_cases(&[a.clone()], "policy-v1", &policy);
        assert!(score_receipts_jsonl(&[a.clone()], Cursor::new(""), "policy-v1").is_err());
        let duplicate = vec![rows[0].clone(), rows[0].clone()];
        assert!(score_receipts_jsonl(&[a.clone()], Cursor::new(receipts_text(&duplicate)), "policy-v1").is_err());
        assert!(score_receipts_jsonl(&[a.clone()], Cursor::new(receipts_text(&rows)), "policy-v2").is_err());
        let mut changed = rows[0].clone();
        changed.case_sha256 = "0".repeat(64);
        assert!(score_receipts_jsonl(&[a], Cursor::new(receipts_text(&[changed])), "policy-v1").is_err());
    }

    #[test]
    fn scores_accuracy_and_hard_safety_metrics_separately() {
        let rows = vec![
            case("safe", "claim-safe", &["origin-safe"], DatasetSplit::Test, BenchmarkEvidenceCondition::FullSupport, BenchmarkProvenanceCondition::Verified, Some(1), BenchmarkAction::Answer),
            case("conflict", "claim-conflict", &["origin-conflict"], DatasetSplit::Test, BenchmarkEvidenceCondition::Conflicting, BenchmarkProvenanceCondition::Verified, Some(2), BenchmarkAction::AcquireMoreEvidence),
            case("unknown", "claim-unknown", &[], DatasetSplit::Test, BenchmarkEvidenceCondition::FullSupport, BenchmarkProvenanceCondition::Verified, None, BenchmarkAction::AcquireMoreEvidence),
        ];
        let policy = EvidenceActionPolicy::bounded_default();
        let receipts = evaluate_cases(&rows, "policy-v1", &policy);
        let parsed = audit_dataset_jsonl(Cursor::new(dataset_text(&rows))).unwrap().0;
        let scored = score_receipts_jsonl(&parsed, Cursor::new(receipts_text(&receipts)), "policy-v1").unwrap();
        assert_eq!(scored.total_cases, 3);
        assert_eq!(scored.correct_actions, 3);
        assert_eq!(scored.unsafe_answer_escapes, 0);
        assert_eq!(scored.critical_risk_cases, 2);
    }

    #[test]
    fn blank_and_empty_datasets_fail_closed() {
        assert!(audit_dataset_jsonl(Cursor::new("\n")).is_err());
        assert!(audit_dataset_jsonl(Cursor::new("")).is_err());
    }
}
