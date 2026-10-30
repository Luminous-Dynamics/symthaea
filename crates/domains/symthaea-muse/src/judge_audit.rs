// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Replayable audit of automated music-edit judges against human references.
//!
//! The official MMAE evaluator remains untouched. This module records its
//! three-attempt majority decision as one official measurement while treating
//! repeated calls to the same model as correlated retries, not independent
//! judges. Secondary model families are scored separately against the same
//! human-labelled sample/rubric subset.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const SPEC_DOMAIN: &[u8] = b"luminous-dynamics-muse-judge-spec-v1\0";
const MANIFEST_DOMAIN: &[u8] = b"luminous-dynamics-muse-judge-manifest-v1\0";
const AUDIT_DOMAIN: &[u8] = b"luminous-dynamics-muse-judge-audit-v1\0";
const AUDIT_VERSION: &str = "muse-judge-measurement-audit-v1";
const OFFICIAL_ATTEMPTS: u8 = 3;
const MAX_LABEL_BYTES: usize = 256;

/// The official evaluator and independent secondary measurement systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeRole {
    Official,
    Secondary,
}

/// Rubric decision. Secondary judges may abstain; official binary-scoring
/// attempts must not abstain and are rejected if they do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeVerdict {
    Pass,
    Fail,
    Abstain,
}

impl JudgeVerdict {
    fn is_pass(self) -> bool {
        matches!(self, Self::Pass)
    }
}

/// Immutable identity of one judging configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgeSpec {
    pub role: JudgeRole,
    pub family_id: String,
    pub model_id: String,
    pub model_revision: String,
    pub prompt_sha256: String,
    pub evaluator_sha256: String,
    pub decoding_config_sha256: String,
}

/// One preserved judge response. Raw text may be stored externally; its digest
/// and the answer-order digest are retained here for replay and audit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgeAttempt {
    pub sample_id: String,
    pub rubric_id: String,
    pub source_lineage_sha256: String,
    pub judge_spec_sha256: String,
    /// One-based attempt index. Official attempts must be exactly 1, 2 and 3.
    pub attempt_index: u8,
    pub answer_order_sha256: String,
    pub raw_response_sha256: String,
    pub verdict: JudgeVerdict,
}

/// Human-adjudicated reference for one sample/rubric pair. The source lineage
/// digest binds it to the same composition cluster used in benchmark analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanRubricReference {
    pub sample_id: String,
    pub rubric_id: String,
    pub source_lineage_sha256: String,
    pub verdict: JudgeVerdict,
    pub adjudication_record_sha256: String,
}

/// Frozen input bundle for the judge reliability audit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgeEvaluationManifest {
    pub benchmark_id: String,
    pub dataset_revision: String,
    pub judge_specs: Vec<JudgeSpec>,
    pub attempts: Vec<JudgeAttempt>,
    pub human_references: Vec<HumanRubricReference>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JudgeAuditViolationKind {
    EmptyBenchmarkId,
    EmptyDatasetRevision,
    EmptyAttempts,
    EmptyHumanReferences,
    MissingOfficialJudge,
    MultipleOfficialJudges,
    MissingSecondaryJudge,
    InvalidJudgeSpec,
    DuplicateJudgeSpec,
    SecondarySharesOfficialFamily,
    UnknownJudgeSpec,
    EmptySampleId,
    EmptyRubricId,
    InvalidLineageDigest,
    InvalidAnswerOrderDigest,
    InvalidRawResponseDigest,
    InvalidHumanAdjudicationDigest,
    DuplicateAttempt,
    InvalidAttemptIndex,
    MissingHumanReference,
    DuplicateHumanReference,
    LineageMismatch,
    IncompleteOfficialTriple,
    OfficialAbstention,
    IncompleteSecondaryCoverage,
    RepeatedSecondaryAttempt,
}

impl JudgeAuditViolationKind {
    fn code(self) -> &'static str {
        match self {
            Self::EmptyBenchmarkId => "empty_benchmark_id",
            Self::EmptyDatasetRevision => "empty_dataset_revision",
            Self::EmptyAttempts => "empty_attempts",
            Self::EmptyHumanReferences => "empty_human_references",
            Self::MissingOfficialJudge => "missing_official_judge",
            Self::MultipleOfficialJudges => "multiple_official_judges",
            Self::MissingSecondaryJudge => "missing_secondary_judge",
            Self::InvalidJudgeSpec => "invalid_judge_spec",
            Self::DuplicateJudgeSpec => "duplicate_judge_spec",
            Self::SecondarySharesOfficialFamily => "secondary_shares_official_family",
            Self::UnknownJudgeSpec => "unknown_judge_spec",
            Self::EmptySampleId => "empty_sample_id",
            Self::EmptyRubricId => "empty_rubric_id",
            Self::InvalidLineageDigest => "invalid_lineage_digest",
            Self::InvalidAnswerOrderDigest => "invalid_answer_order_digest",
            Self::InvalidRawResponseDigest => "invalid_raw_response_digest",
            Self::InvalidHumanAdjudicationDigest => "invalid_human_adjudication_digest",
            Self::DuplicateAttempt => "duplicate_attempt",
            Self::InvalidAttemptIndex => "invalid_attempt_index",
            Self::MissingHumanReference => "missing_human_reference",
            Self::DuplicateHumanReference => "duplicate_human_reference",
            Self::LineageMismatch => "lineage_mismatch",
            Self::IncompleteOfficialTriple => "incomplete_official_triple",
            Self::OfficialAbstention => "official_abstention",
            Self::IncompleteSecondaryCoverage => "incomplete_secondary_coverage",
            Self::RepeatedSecondaryAttempt => "repeated_secondary_attempt",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgeAuditViolation {
    pub kind: JudgeAuditViolationKind,
    pub sample_id: Option<String>,
    pub rubric_id: Option<String>,
    pub judge_spec_sha256: Option<String>,
}

/// Confusion counts for one judge spec's individual responses against humans.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgeConfusionSummary {
    pub judge_spec_sha256: String,
    pub role: JudgeRole,
    pub compared_attempts: u64,
    pub correct: u64,
    pub incorrect: u64,
    pub false_pass: u64,
    pub false_fail: u64,
    pub abstained: u64,
}

/// Pairwise error association between the three retries of one official judge.
/// Retry calls are not counted as independent model families.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetryErrorAssociation {
    pub pair_count: u64,
    pub both_wrong: u64,
    pub left_wrong_only: u64,
    pub right_wrong_only: u64,
    pub one_wrong: u64,
    pub neither_wrong: u64,
    /// Phi correlation in millionths; None means no variation in a margin.
    pub phi_microunits: Option<i64>,
}

/// One official three-attempt majority decision, separately compared to humans.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OfficialMajorityDecision {
    pub sample_id: String,
    pub rubric_id: String,
    pub source_lineage_sha256: String,
    pub verdict: JudgeVerdict,
    pub human_verdict: JudgeVerdict,
    pub correct: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgeEvaluationAudit {
    pub audit_version: String,
    pub benchmark_id: String,
    pub dataset_revision: String,
    pub manifest_sha256: String,
    pub attempt_count: usize,
    pub human_reference_count: usize,
    pub official_majority_decisions: Vec<OfficialMajorityDecision>,
    pub official_majority_correct: u64,
    pub official_majority_incorrect: u64,
    pub per_judge_attempt_metrics: Vec<JudgeConfusionSummary>,
    pub official_retry_error_association: RetryErrorAssociation,
    pub violations: Vec<JudgeAuditViolation>,
    pub audit_sha256: String,
}

impl JudgeEvaluationAudit {
    /// True only when the violation list is empty and the exposed audit fields
    /// still match the receipt. Use the replay verifier to bind it to inputs.
    pub fn is_clean(&self) -> bool {
        self.violations.is_empty() && self.audit_sha256 == audit_digest(self)
    }

    /// True when the official majority is evaluated on the complete human
    /// subset and the audit is internally valid.
    pub fn is_qualified(&self) -> bool {
        self.is_clean()
            && self.official_majority_decisions.len() == self.human_reference_count
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_label(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= MAX_LABEL_BYTES
        && !value.chars().any(char::is_control)
}

fn hash_component(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn judge_spec_digest(spec: &JudgeSpec) -> String {
    let bytes = serde_json::to_vec(spec).expect("serializing a fixed judge spec cannot fail");
    let mut hasher = Sha256::new();
    hasher.update(SPEC_DOMAIN);
    hash_component(&mut hasher, &bytes);
    format!("{:x}", hasher.finalize())
}

fn manifest_digest(manifest: &JudgeEvaluationManifest) -> String {
    let mut hasher = Sha256::new();
    hasher.update(MANIFEST_DOMAIN);
    hash_component(&mut hasher, manifest.benchmark_id.as_bytes());
    hash_component(&mut hasher, manifest.dataset_revision.as_bytes());

    let mut specs: Vec<(String, &JudgeSpec)> = manifest
        .judge_specs
        .iter()
        .map(|spec| (judge_spec_digest(spec), spec))
        .collect();
    specs.sort_by(|a, b| a.0.cmp(&b.0));
    hasher.update((specs.len() as u64).to_le_bytes());
    for (_, spec) in specs {
        let bytes = serde_json::to_vec(spec).expect("serializing a fixed judge spec cannot fail");
        hash_component(&mut hasher, &bytes);
    }

    let mut attempts = manifest.attempts.clone();
    attempts.sort_by(|a, b| {
        (
            &a.sample_id,
            &a.rubric_id,
            &a.judge_spec_sha256,
            a.attempt_index,
            &a.source_lineage_sha256,
            &a.answer_order_sha256,
            &a.raw_response_sha256,
            a.verdict,
        )
            .cmp(&(
                &b.sample_id,
                &b.rubric_id,
                &b.judge_spec_sha256,
                b.attempt_index,
                &b.source_lineage_sha256,
                &b.answer_order_sha256,
                &b.raw_response_sha256,
                b.verdict,
            ))
    });
    hasher.update((attempts.len() as u64).to_le_bytes());
    for attempt in attempts {
        hash_component(&mut hasher, attempt.sample_id.as_bytes());
        hash_component(&mut hasher, attempt.rubric_id.as_bytes());
        hash_component(&mut hasher, attempt.source_lineage_sha256.as_bytes());
        hash_component(&mut hasher, attempt.judge_spec_sha256.as_bytes());
        hasher.update([attempt.attempt_index]);
        hash_component(&mut hasher, attempt.answer_order_sha256.as_bytes());
        hash_component(&mut hasher, attempt.raw_response_sha256.as_bytes());
        hasher.update([u8::from(attempt.verdict.is_pass())]);
    }

    let mut references = manifest.human_references.clone();
    references.sort_by(|a, b| {
        (
            &a.sample_id,
            &a.rubric_id,
            &a.source_lineage_sha256,
            &a.adjudication_record_sha256,
            a.verdict,
        )
            .cmp(&(
                &b.sample_id,
                &b.rubric_id,
                &b.source_lineage_sha256,
                &b.adjudication_record_sha256,
                b.verdict,
            ))
    });
    hasher.update((references.len() as u64).to_le_bytes());
    for reference in references {
        hash_component(&mut hasher, reference.sample_id.as_bytes());
        hash_component(&mut hasher, reference.rubric_id.as_bytes());
        hash_component(&mut hasher, reference.source_lineage_sha256.as_bytes());
        hasher.update([u8::from(reference.verdict.is_pass())]);
        hash_component(&mut hasher, reference.adjudication_record_sha256.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn audit_digest(audit: &JudgeEvaluationAudit) -> String {
    let mut hasher = Sha256::new();
    hasher.update(AUDIT_DOMAIN);
    hash_component(&mut hasher, audit.audit_version.as_bytes());
    hash_component(&mut hasher, audit.benchmark_id.as_bytes());
    hash_component(&mut hasher, audit.dataset_revision.as_bytes());
    hash_component(&mut hasher, audit.manifest_sha256.as_bytes());
    hasher.update((audit.attempt_count as u64).to_le_bytes());
    hasher.update((audit.human_reference_count as u64).to_le_bytes());
    hasher.update(audit.official_majority_correct.to_le_bytes());
    hasher.update(audit.official_majority_incorrect.to_le_bytes());
    hasher.update((audit.official_majority_decisions.len() as u64).to_le_bytes());
    for decision in &audit.official_majority_decisions {
        hash_component(&mut hasher, decision.sample_id.as_bytes());
        hash_component(&mut hasher, decision.rubric_id.as_bytes());
        hash_component(&mut hasher, decision.source_lineage_sha256.as_bytes());
        hasher.update([u8::from(decision.verdict.is_pass())]);
        hasher.update([u8::from(decision.human_verdict.is_pass())]);
        hasher.update([u8::from(decision.correct)]);
    }
    hasher.update((audit.per_judge_attempt_metrics.len() as u64).to_le_bytes());
    for summary in &audit.per_judge_attempt_metrics {
        hash_component(&mut hasher, summary.judge_spec_sha256.as_bytes());
        hasher.update([u8::from(matches!(summary.role, JudgeRole::Official))]);
        hasher.update(summary.compared_attempts.to_le_bytes());
        hasher.update(summary.correct.to_le_bytes());
        hasher.update(summary.incorrect.to_le_bytes());
        hasher.update(summary.false_pass.to_le_bytes());
        hasher.update(summary.false_fail.to_le_bytes());
        hasher.update(summary.abstained.to_le_bytes());
    }
    let retry = &audit.official_retry_error_association;
    hasher.update(retry.pair_count.to_le_bytes());
    hasher.update(retry.both_wrong.to_le_bytes());
    hasher.update(retry.left_wrong_only.to_le_bytes());
    hasher.update(retry.right_wrong_only.to_le_bytes());
    hasher.update(retry.one_wrong.to_le_bytes());
    hasher.update(retry.neither_wrong.to_le_bytes());
    match retry.phi_microunits {
        Some(value) => {
            hasher.update([1_u8]);
            hasher.update(value.to_le_bytes());
        }
        None => hasher.update([0_u8]),
    }
    hasher.update((audit.violations.len() as u64).to_le_bytes());
    for violation in &audit.violations {
        hash_component(&mut hasher, violation.kind.code().as_bytes());
        for value in [&violation.sample_id, &violation.rubric_id, &violation.judge_spec_sha256] {
            match value {
                Some(value) => {
                    hasher.update([1_u8]);
                    hash_component(&mut hasher, value.as_bytes());
                }
                None => hasher.update([0_u8]),
            }
        }
    }
    format!("{:x}", hasher.finalize())
}

type ItemKey = (String, String);

fn push_violation(
    violations: &mut Vec<JudgeAuditViolation>,
    kind: JudgeAuditViolationKind,
    sample_id: Option<&str>,
    rubric_id: Option<&str>,
    judge_spec_sha256: Option<&str>,
) {
    violations.push(JudgeAuditViolation {
        kind,
        sample_id: sample_id.map(str::to_owned),
        rubric_id: rubric_id.map(str::to_owned),
        judge_spec_sha256: judge_spec_sha256.map(str::to_owned),
    });
}

fn retry_phi(
    both_wrong: u64,
    left_wrong_only: u64,
    right_wrong_only: u64,
    neither_wrong: u64,
) -> Option<i64> {
    let a = both_wrong as f64;
    let b = left_wrong_only as f64;
    let c = right_wrong_only as f64;
    let d = neither_wrong as f64;
    let denominator = ((a + b) * (c + d) * (a + c) * (b + d)).sqrt();
    if denominator == 0.0 {
        return None;
    }
    let numerator = a * d - b * c;
    Some((1_000_000.0 * numerator / denominator).round() as i64)
}

/// Validate the official judge's three attempts and compare both attempt-level
/// and majority decisions with a blinded human-reference subset.
pub fn audit_judge_evaluation(manifest: &JudgeEvaluationManifest) -> JudgeEvaluationAudit {
    let mut violations = Vec::new();
    if !valid_label(&manifest.benchmark_id) {
        push_violation(&mut violations, JudgeAuditViolationKind::EmptyBenchmarkId, None, None, None);
    }
    if !valid_label(&manifest.dataset_revision) {
        push_violation(&mut violations, JudgeAuditViolationKind::EmptyDatasetRevision, None, None, None);
    }
    if manifest.attempts.is_empty() {
        push_violation(&mut violations, JudgeAuditViolationKind::EmptyAttempts, None, None, None);
    }
    if manifest.human_references.is_empty() {
        push_violation(&mut violations, JudgeAuditViolationKind::EmptyHumanReferences, None, None, None);
    }

    let mut specs = BTreeMap::<String, &JudgeSpec>::new();
    let mut official_specs = Vec::new();
    let mut secondary_specs = Vec::new();
    for spec in &manifest.judge_specs {
        let valid = valid_label(&spec.family_id)
            && valid_label(&spec.model_id)
            && valid_label(&spec.model_revision)
            && is_sha256(&spec.prompt_sha256)
            && is_sha256(&spec.evaluator_sha256)
            && is_sha256(&spec.decoding_config_sha256);
        if !valid {
            push_violation(&mut violations, JudgeAuditViolationKind::InvalidJudgeSpec, None, None, None);
            continue;
        }
        let digest = judge_spec_digest(spec);
        if specs.insert(digest.clone(), spec).is_some() {
            push_violation(
                &mut violations,
                JudgeAuditViolationKind::DuplicateJudgeSpec,
                None,
                None,
                Some(&digest),
            );
        }
        match spec.role {
            JudgeRole::Official => official_specs.push((digest, spec.family_id.as_str())),
            JudgeRole::Secondary => secondary_specs.push((digest, spec.family_id.as_str())),
        }
    }

    if official_specs.is_empty() {
        push_violation(&mut violations, JudgeAuditViolationKind::MissingOfficialJudge, None, None, None);
    } else if official_specs.len() > 1 {
        push_violation(&mut violations, JudgeAuditViolationKind::MultipleOfficialJudges, None, None, None);
    }
    if secondary_specs.is_empty() {
        push_violation(&mut violations, JudgeAuditViolationKind::MissingSecondaryJudge, None, None, None);
    }
    if let Some((_, official_family)) = official_specs.first() {
        for (digest, family) in &secondary_specs {
            if family == official_family {
                push_violation(
                    &mut violations,
                    JudgeAuditViolationKind::SecondarySharesOfficialFamily,
                    None,
                    None,
                    Some(digest),
                );
            }
        }
    }

    let mut references = BTreeMap::<ItemKey, &HumanRubricReference>::new();
    for reference in &manifest.human_references {
        if !valid_label(&reference.sample_id) {
            push_violation(
                &mut violations,
                JudgeAuditViolationKind::EmptySampleId,
                Some(&reference.sample_id),
                Some(&reference.rubric_id),
                None,
            );
        }
        if !valid_label(&reference.rubric_id) {
            push_violation(
                &mut violations,
                JudgeAuditViolationKind::EmptyRubricId,
                Some(&reference.sample_id),
                Some(&reference.rubric_id),
                None,
            );
        }
        if !is_sha256(&reference.source_lineage_sha256) {
            push_violation(
                &mut violations,
                JudgeAuditViolationKind::InvalidLineageDigest,
                Some(&reference.sample_id),
                Some(&reference.rubric_id),
                None,
            );
        }
        if !is_sha256(&reference.adjudication_record_sha256) {
            push_violation(
                &mut violations,
                JudgeAuditViolationKind::InvalidHumanAdjudicationDigest,
                Some(&reference.sample_id),
                Some(&reference.rubric_id),
                None,
            );
        }
        let key = (reference.sample_id.clone(), reference.rubric_id.clone());
        if references.insert(key, reference).is_some() {
            push_violation(
                &mut violations,
                JudgeAuditViolationKind::DuplicateHumanReference,
                Some(&reference.sample_id),
                Some(&reference.rubric_id),
                None,
            );
        }
    }

    let mut groups = BTreeMap::<(String, String, String), Vec<&JudgeAttempt>>::new();
    let mut seen_attempts = BTreeSet::new();
    for attempt in &manifest.attempts {
        if !valid_label(&attempt.sample_id) {
            push_violation(&mut violations, JudgeAuditViolationKind::EmptySampleId, Some(&attempt.sample_id), Some(&attempt.rubric_id), Some(&attempt.judge_spec_sha256));
        }
        if !valid_label(&attempt.rubric_id) {
            push_violation(&mut violations, JudgeAuditViolationKind::EmptyRubricId, Some(&attempt.sample_id), Some(&attempt.rubric_id), Some(&attempt.judge_spec_sha256));
        }
        if !is_sha256(&attempt.source_lineage_sha256) {
            push_violation(&mut violations, JudgeAuditViolationKind::InvalidLineageDigest, Some(&attempt.sample_id), Some(&attempt.rubric_id), Some(&attempt.judge_spec_sha256));
        }
        if !is_sha256(&attempt.answer_order_sha256) {
            push_violation(&mut violations, JudgeAuditViolationKind::InvalidAnswerOrderDigest, Some(&attempt.sample_id), Some(&attempt.rubric_id), Some(&attempt.judge_spec_sha256));
        }
        if !is_sha256(&attempt.raw_response_sha256) {
            push_violation(&mut violations, JudgeAuditViolationKind::InvalidRawResponseDigest, Some(&attempt.sample_id), Some(&attempt.rubric_id), Some(&attempt.judge_spec_sha256));
        }
        let Some(spec) = specs.get(&attempt.judge_spec_sha256) else {
            push_violation(&mut violations, JudgeAuditViolationKind::UnknownJudgeSpec, Some(&attempt.sample_id), Some(&attempt.rubric_id), Some(&attempt.judge_spec_sha256));
            continue;
        };
        let max_attempt = if spec.role == JudgeRole::Official { OFFICIAL_ATTEMPTS } else { 1 };
        if attempt.attempt_index == 0 || attempt.attempt_index > max_attempt {
            let kind = if spec.role == JudgeRole::Secondary && attempt.attempt_index > 1 {
                JudgeAuditViolationKind::RepeatedSecondaryAttempt
            } else {
                JudgeAuditViolationKind::InvalidAttemptIndex
            };
            push_violation(&mut violations, kind, Some(&attempt.sample_id), Some(&attempt.rubric_id), Some(&attempt.judge_spec_sha256));
            continue;
        }
        let unique = (
            attempt.sample_id.clone(),
            attempt.rubric_id.clone(),
            attempt.judge_spec_sha256.clone(),
            attempt.attempt_index,
        );
        if !seen_attempts.insert(unique) {
            push_violation(&mut violations, JudgeAuditViolationKind::DuplicateAttempt, Some(&attempt.sample_id), Some(&attempt.rubric_id), Some(&attempt.judge_spec_sha256));
            continue;
        }
        let key = (attempt.sample_id.clone(), attempt.rubric_id.clone());
        match references.get(&key) {
            Some(reference) if reference.source_lineage_sha256 == attempt.source_lineage_sha256 => {}
            Some(_) => push_violation(&mut violations, JudgeAuditViolationKind::LineageMismatch, Some(&attempt.sample_id), Some(&attempt.rubric_id), Some(&attempt.judge_spec_sha256)),
            None => push_violation(&mut violations, JudgeAuditViolationKind::MissingHumanReference, Some(&attempt.sample_id), Some(&attempt.rubric_id), Some(&attempt.judge_spec_sha256)),
        }
        groups
            .entry((attempt.sample_id.clone(), attempt.rubric_id.clone(), attempt.judge_spec_sha256.clone()))
            .or_default()
            .push(attempt);
    }

    // Every registered judge scores the same human-labelled subset.
    for ((sample_id, rubric_id), _) in &references {
        for (digest, _) in &official_specs {
            if !groups.contains_key(&(sample_id.clone(), rubric_id.clone(), digest.clone())) {
                push_violation(
                    &mut violations,
                    JudgeAuditViolationKind::IncompleteOfficialTriple,
                    Some(sample_id),
                    Some(rubric_id),
                    Some(digest),
                );
            }
        }
        for (digest, _) in &secondary_specs {
            if !groups.contains_key(&(sample_id.clone(), rubric_id.clone(), digest.clone())) {
                push_violation(
                    &mut violations,
                    JudgeAuditViolationKind::IncompleteSecondaryCoverage,
                    Some(sample_id),
                    Some(rubric_id),
                    Some(digest),
                );
            }
        }
    }

    let mut summaries = Vec::new();
    let mut official_decisions = Vec::new();
    let mut official_majority_correct = 0_u64;
    let mut official_majority_incorrect = 0_u64;
    let mut retry_pair_count = 0_u64;
    let mut retry_both_wrong = 0_u64;
    let mut retry_left_wrong_only = 0_u64;
    let mut retry_right_wrong_only = 0_u64;
    let mut retry_neither_wrong = 0_u64;

    for (spec_digest, spec) in &specs {
        let mut summary = JudgeConfusionSummary {
            judge_spec_sha256: spec_digest.clone(),
            role: spec.role,
            compared_attempts: 0,
            correct: 0,
            incorrect: 0,
            false_pass: 0,
            false_fail: 0,
            abstained: 0,
        };
        for ((sample_id, rubric_id, digest), attempts) in &groups {
            if digest != spec_digest {
                continue;
            }
            let Some(reference) = references.get(&(sample_id.clone(), rubric_id.clone())) else {
                continue;
            };
            for attempt in attempts {
                if attempt.verdict == JudgeVerdict::Abstain {
                    summary.abstained += 1;
                    continue;
                }
                summary.compared_attempts += 1;
                if attempt.verdict == reference.verdict {
                    summary.correct += 1;
                } else {
                    summary.incorrect += 1;
                    match attempt.verdict {
                        JudgeVerdict::Pass => summary.false_pass += 1,
                        JudgeVerdict::Fail => summary.false_fail += 1,
                        JudgeVerdict::Abstain => unreachable!("abstention handled above"),
                    }
                }
            }
            if spec.role == JudgeRole::Official {
                let mut ordered = attempts.clone();
                ordered.sort_by_key(|attempt| attempt.attempt_index);
                if ordered.len() != OFFICIAL_ATTEMPTS as usize
                    || ordered.iter().map(|a| a.attempt_index).collect::<Vec<_>>() != vec![1, 2, 3]
                {
                    push_violation(
                        &mut violations,
                        JudgeAuditViolationKind::IncompleteOfficialTriple,
                        Some(sample_id),
                        Some(rubric_id),
                        Some(spec_digest),
                    );
                    continue;
                }
                if ordered.iter().any(|attempt| attempt.verdict == JudgeVerdict::Abstain) {
                    push_violation(
                        &mut violations,
                        JudgeAuditViolationKind::OfficialAbstention,
                        Some(sample_id),
                        Some(rubric_id),
                        Some(spec_digest),
                    );
                    continue;
                }
                let pass_count = ordered.iter().filter(|a| a.verdict.is_pass()).count();
                let verdict = if pass_count >= 2 { JudgeVerdict::Pass } else { JudgeVerdict::Fail };
                let correct = verdict == reference.verdict;
                if correct {
                    official_majority_correct += 1;
                } else {
                    official_majority_incorrect += 1;
                }
                official_decisions.push(OfficialMajorityDecision {
                    sample_id: sample_id.clone(),
                    rubric_id: rubric_id.clone(),
                    source_lineage_sha256: reference.source_lineage_sha256.clone(),
                    verdict,
                    human_verdict: reference.verdict,
                    correct,
                });

                let errors: Vec<bool> = ordered.iter().map(|a| a.verdict != reference.verdict).collect();
                for left in 0..errors.len() {
                    for right in (left + 1)..errors.len() {
                        retry_pair_count += 1;
                        match (errors[left], errors[right]) {
                            (true, true) => retry_both_wrong += 1,
                            (true, false) => retry_left_wrong_only += 1,
                            (false, true) => retry_right_wrong_only += 1,
                            (false, false) => retry_neither_wrong += 1,
                        }
                    }
                }
            }
        }
        summaries.push(summary);
    }

    official_decisions.sort_by(|a, b| (&a.sample_id, &a.rubric_id).cmp(&(&b.sample_id, &b.rubric_id)));
    summaries.sort_by(|a, b| a.judge_spec_sha256.cmp(&b.judge_spec_sha256));
    violations.sort_by(|a, b| {
        (a.kind, &a.sample_id, &a.rubric_id, &a.judge_spec_sha256)
            .cmp(&(b.kind, &b.sample_id, &b.rubric_id, &b.judge_spec_sha256))
    });

    let mut audit = JudgeEvaluationAudit {
        audit_version: AUDIT_VERSION.to_owned(),
        benchmark_id: manifest.benchmark_id.clone(),
        dataset_revision: manifest.dataset_revision.clone(),
        manifest_sha256: manifest_digest(manifest),
        attempt_count: manifest.attempts.len(),
        human_reference_count: manifest.human_references.len(),
        official_majority_decisions: official_decisions,
        official_majority_correct,
        official_majority_incorrect,
        per_judge_attempt_metrics: summaries,
        official_retry_error_association: RetryErrorAssociation {
            pair_count: retry_pair_count,
            both_wrong: retry_both_wrong,
            left_wrong_only: retry_left_wrong_only,
            right_wrong_only: retry_right_wrong_only,
            one_wrong: retry_left_wrong_only + retry_right_wrong_only,
            neither_wrong: retry_neither_wrong,
            phi_microunits: retry_phi(
                retry_both_wrong,
                retry_left_wrong_only,
                retry_right_wrong_only,
                retry_neither_wrong,
            ),
        },
        violations,
        audit_sha256: String::new(),
    };
    audit.audit_sha256 = audit_digest(&audit);
    audit
}

/// Verify both the audit's own receipt and its binding to the original inputs.
pub fn verify_judge_evaluation_audit(
    manifest: &JudgeEvaluationManifest,
    supplied: &JudgeEvaluationAudit,
) -> bool {
    audit_judge_evaluation(manifest) == *supplied
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> String {
        std::iter::repeat_n(format!("{byte:02x}"), 32).collect()
    }

    fn spec(role: JudgeRole, family: &str, model: &str) -> JudgeSpec {
        JudgeSpec {
            role,
            family_id: family.to_owned(),
            model_id: model.to_owned(),
            model_revision: "pinned-revision".to_owned(),
            prompt_sha256: digest(1),
            evaluator_sha256: digest(2),
            decoding_config_sha256: digest(3),
        }
    }

    fn manifest() -> JudgeEvaluationManifest {
        let official = spec(JudgeRole::Official, "qwen-family", "qwen3-omni-frozen");
        let secondary = spec(JudgeRole::Secondary, "independent-family", "independent-judge-v1");
        let official_hash = judge_spec_digest(&official);
        let secondary_hash = judge_spec_digest(&secondary);
        let lineage = digest(8);
        let human = HumanRubricReference {
            sample_id: "sample-1".to_owned(),
            rubric_id: "preserve-vocals".to_owned(),
            source_lineage_sha256: lineage.clone(),
            verdict: JudgeVerdict::Pass,
            adjudication_record_sha256: digest(9),
        };
        let attempts = vec![
            JudgeAttempt {
                sample_id: "sample-1".to_owned(),
                rubric_id: "preserve-vocals".to_owned(),
                source_lineage_sha256: lineage.clone(),
                judge_spec_sha256: official_hash.clone(),
                attempt_index: 1,
                answer_order_sha256: digest(10),
                raw_response_sha256: digest(20),
                verdict: JudgeVerdict::Pass,
            },
            JudgeAttempt {
                sample_id: "sample-1".to_owned(),
                rubric_id: "preserve-vocals".to_owned(),
                source_lineage_sha256: lineage.clone(),
                judge_spec_sha256: official_hash.clone(),
                attempt_index: 2,
                answer_order_sha256: digest(11),
                raw_response_sha256: digest(21),
                verdict: JudgeVerdict::Fail,
            },
            JudgeAttempt {
                sample_id: "sample-1".to_owned(),
                rubric_id: "preserve-vocals".to_owned(),
                source_lineage_sha256: lineage.clone(),
                judge_spec_sha256: official_hash,
                attempt_index: 3,
                answer_order_sha256: digest(12),
                raw_response_sha256: digest(22),
                verdict: JudgeVerdict::Pass,
            },
            JudgeAttempt {
                sample_id: "sample-1".to_owned(),
                rubric_id: "preserve-vocals".to_owned(),
                source_lineage_sha256: lineage,
                judge_spec_sha256: secondary_hash,
                attempt_index: 1,
                answer_order_sha256: digest(13),
                raw_response_sha256: digest(23),
                verdict: JudgeVerdict::Pass,
            },
        ];
        JudgeEvaluationManifest {
            benchmark_id: "mmae".to_owned(),
            dataset_revision: "public-test-2026-10".to_owned(),
            judge_specs: vec![official, secondary],
            attempts,
            human_references: vec![human],
        }
    }

    #[test]
    fn official_three_attempt_majority_and_secondary_metrics_are_separate() {
        let input = manifest();
        let audit = audit_judge_evaluation(&input);
        assert!(audit.is_qualified(), "{:?}", audit.violations);
        assert_eq!(audit.official_majority_correct, 1);
        assert_eq!(audit.official_majority_incorrect, 0);
        assert_eq!(audit.official_majority_decisions.len(), 1);
        assert_eq!(audit.official_retry_error_association.pair_count, 3);
        assert_eq!(audit.official_retry_error_association.one_wrong, 2);
        assert!(verify_judge_evaluation_audit(&input, &audit));
    }

    #[test]
    fn incomplete_official_triple_fails_closed() {
        let mut input = manifest();
        let official_hash = judge_spec_digest(&input.judge_specs[0]);
        input.attempts.retain(|attempt| {
            !(attempt.judge_spec_sha256 == official_hash && attempt.attempt_index == 3)
        });
        let audit = audit_judge_evaluation(&input);
        assert!(!audit.is_qualified());
        assert!(audit.violations.iter().any(|v| v.kind == JudgeAuditViolationKind::IncompleteOfficialTriple));
    }

    #[test]
    fn same_family_secondary_is_not_counted_as_independent() {
        let mut input = manifest();
        let old_hash = judge_spec_digest(&input.judge_specs[1]);
        input.judge_specs[1].family_id = input.judge_specs[0].family_id.clone();
        let new_hash = judge_spec_digest(&input.judge_specs[1]);
        for attempt in &mut input.attempts {
            if attempt.judge_spec_sha256 == old_hash {
                attempt.judge_spec_sha256 = new_hash.clone();
            }
        }
        let audit = audit_judge_evaluation(&input);
        assert!(audit.violations.iter().any(|v| v.kind == JudgeAuditViolationKind::SecondarySharesOfficialFamily));
    }

    #[test]
    fn response_digest_and_answer_order_are_required() {
        let mut input = manifest();
        input.attempts[0].raw_response_sha256 = "missing".to_owned();
        input.attempts[1].answer_order_sha256.clear();
        let audit = audit_judge_evaluation(&input);
        assert!(audit.violations.iter().any(|v| v.kind == JudgeAuditViolationKind::InvalidRawResponseDigest));
        assert!(audit.violations.iter().any(|v| v.kind == JudgeAuditViolationKind::InvalidAnswerOrderDigest));
    }

    #[test]
    fn human_reference_must_share_source_lineage_with_attempts() {
        let mut input = manifest();
        input.human_references[0].source_lineage_sha256 = digest(99);
        let audit = audit_judge_evaluation(&input);
        assert!(audit.violations.iter().any(|v| v.kind == JudgeAuditViolationKind::LineageMismatch));
    }

    #[test]
    fn row_order_does_not_change_manifest_or_audit_digest() {
        let mut input = manifest();
        let first = audit_judge_evaluation(&input);
        input.attempts.reverse();
        input.judge_specs.reverse();
        input.human_references.reverse();
        let second = audit_judge_evaluation(&input);
        assert_eq!(first, second);
    }


    #[test]
    fn secondary_abstention_is_counted_but_not_scored_as_failure() {
        let mut input = manifest();
        input.attempts[3].verdict = JudgeVerdict::Abstain;
        let audit = audit_judge_evaluation(&input);
        assert!(audit.is_qualified(), "{:?}", audit.violations);
        let secondary_hash = judge_spec_digest(&input.judge_specs[1]);
        let secondary = audit.per_judge_attempt_metrics.iter()
            .find(|summary| summary.judge_spec_sha256 == secondary_hash)
            .unwrap();
        assert_eq!(secondary.abstained, 1);
        assert_eq!(secondary.compared_attempts, 0);
        assert_eq!(secondary.incorrect, 0);
    }

    #[test]
    fn official_abstention_is_rejected_and_not_coerced_to_fail() {
        let mut input = manifest();
        input.attempts[0].verdict = JudgeVerdict::Abstain;
        let audit = audit_judge_evaluation(&input);
        assert!(audit.violations.iter().any(|v| v.kind == JudgeAuditViolationKind::OfficialAbstention));
        assert!(!audit.is_qualified());
    }

    #[test]
    fn secondary_retry_is_not_mistaken_for_an_independent_judge() {
        let mut input = manifest();
        let mut retry = input.attempts[3].clone();
        retry.attempt_index = 2;
        retry.answer_order_sha256 = digest(14);
        retry.raw_response_sha256 = digest(24);
        input.attempts.push(retry);
        let audit = audit_judge_evaluation(&input);
        assert!(audit.violations.iter().any(|v| v.kind == JudgeAuditViolationKind::RepeatedSecondaryAttempt));
    }

    #[test]
    fn tampered_audit_fails_receipt_and_input_replay() {
        let input = manifest();
        let mut audit = audit_judge_evaluation(&input);
        assert!(audit.is_qualified());
        audit.official_majority_correct += 1;
        assert!(!audit.is_clean());
        assert!(!verify_judge_evaluation_audit(&input, &audit));
    }
}
