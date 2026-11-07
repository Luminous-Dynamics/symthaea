//! Source-grounded reverse-engineering scorecard v0.
//!
//! This benchmark scorer distinguishes raw answer correctness from evidence-
//! grounded correctness. It verifies raw REA Evidence records, binds them to
//! the task's exact subject/operation/represented environment, and only grants
//! grounded credit when the declared claim disposition is reproduced.
//!
//! This is a benchmark scorer, not a new claim engine, authenticity verifier,
//! temporal-validity system, provider-independence oracle, or safety verdict.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use symthaea_rea_compat::{VerifiedEvidence, canonical_jcs_bytes, verify_evidence_json};

pub const BENCHMARK_SCHEMA: &str = "luminous.rea.agent-grounding.v0";
pub const MAX_TASK_ID_BYTES: usize = 256;
pub const MAX_EVIDENCE_RECORDS: usize = 128;
pub const MAX_TOTAL_EVIDENCE_BYTES: usize = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskVariant {
    Baseline,
    RelevantMutation,
    IrrelevantMutation,
    Unsupported,
    Conflict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    SupportedWithinScope,
    RefutedWithinScope,
    Underdetermined,
    Conflicted,
    Inapplicable,
    Unverified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssessmentKind {
    Supports,
    Contradicts,
    Inapplicable,
    ScopeUnverifiable,
    NotEvaluated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimScope {
    pub artifact_sha256: String,
    pub operation: String,
    pub platform: Option<String>,
    pub platform_version: Option<String>,
    pub environment_id: Option<String>,
    pub source_predicate_type: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskSpec {
    pub schema: String,
    pub task_id: String,
    pub pair_id: String,
    pub variant: TaskVariant,
    pub scope: ClaimScope,
    pub predicate: String,
    pub expected_answer: Option<Value>,
    pub expected_disposition: Disposition,
    pub assumptions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentSubmission {
    pub task_id: String,
    pub answer: Option<Value>,
    /// Ordered semantic Evidence IDs cited by the agent.
    pub evidence_ids: Vec<String>,
    /// Raw Evidence JSON. Transport order is independent of evidence_ids order.
    pub evidence_json: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceAssessment {
    pub evidence_id: String,
    pub kind: AssessmentKind,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskScore {
    pub task_id: String,
    pub pair_id: String,
    pub variant: TaskVariant,
    pub raw_answer_correct: bool,
    pub evidence_disposition: Disposition,
    pub evidence_grounded_correct: bool,
    pub parametric_leakage: bool,
    pub false_supported_claim: bool,
    pub false_refuted_claim: bool,
    pub verified_evidence_count: usize,
    pub missing_evidence_count: usize,
    pub evidence_assessments: Vec<EvidenceAssessment>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ratio {
    pub numerator: usize,
    pub denominator: usize,
    pub value: Option<f64>,
}

impl Ratio {
    fn new(numerator: usize, denominator: usize) -> Self {
        Self {
            numerator,
            denominator,
            value: (denominator > 0).then_some(numerator as f64 / denominator as f64),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairMetrics {
    pub relevant_pairs: usize,
    pub correctly_sensitive_pairs: usize,
    pub relevant_sensitivity: Ratio,
    pub irrelevant_pairs: usize,
    pub correctly_invariant_pairs: usize,
    pub irrelevant_invariance: Ratio,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchScore {
    pub schema: String,
    pub task_count: usize,
    pub raw_correct_count: usize,
    pub evidence_grounded_correct_count: usize,
    pub parametric_leakage_count: usize,
    pub false_supported_count: usize,
    pub false_refuted_count: usize,
    pub raw_accuracy: Ratio,
    pub evidence_grounded_accuracy: Ratio,
    pub parametric_leakage_rate: Ratio,
    pub pair_metrics: PairMetrics,
    pub task_scores: Vec<TaskScore>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreErrorKind {
    InvalidTask,
    SubmissionTaskMismatch,
    InputTooLarge,
    TooManyEvidenceRecords,
    MalformedEvidence,
    DuplicateEvidenceId,
    UnexpectedEvidenceId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoreError {
    pub kind: ScoreErrorKind,
    pub message: String,
}

impl ScoreError {
    fn new(kind: ScoreErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl fmt::Display for ScoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for ScoreError {}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn validate_task(task: &TaskSpec) -> Result<(), ScoreError> {
    let invalid = |message: &str| ScoreError::new(ScoreErrorKind::InvalidTask, message);
    if task.schema != BENCHMARK_SCHEMA {
        return Err(invalid("unsupported source-grounding task schema"));
    }
    if task.task_id.is_empty() || task.task_id.len() > MAX_TASK_ID_BYTES {
        return Err(invalid("task_id must be non-empty and at most 256 UTF-8 bytes"));
    }
    if !nonempty(&task.pair_id) {
        return Err(invalid("pair_id must be non-empty"));
    }
    if !is_sha256(&task.scope.artifact_sha256) {
        return Err(invalid("scope.artifact_sha256 must be lowercase SHA-256"));
    }
    if !nonempty(&task.scope.operation) || !nonempty(&task.scope.source_predicate_type) {
        return Err(invalid("operation and source_predicate_type must be non-empty"));
    }
    if task.scope.platform.as_deref().is_some_and(|value| !nonempty(value))
        || task.scope.platform_version.as_deref().is_some_and(|value| !nonempty(value))
        || task.scope.environment_id.as_deref().is_some_and(|value| !nonempty(value))
    {
        return Err(invalid("optional scope strings must be non-empty when present"));
    }
    if task.assumptions.iter().any(|value| !nonempty(value)) {
        return Err(invalid("assumptions must contain non-empty strings"));
    }
    if task.predicate == "provider_reported_observation"
        && task.expected_disposition == Disposition::SupportedWithinScope
        && task.expected_answer.is_none()
    {
        return Err(invalid("supported provider-observation tasks require expected_answer"));
    }
    Ok(())
}

fn string_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_str()
}

fn canonical_equal(left: &Value, right: &Value) -> Result<bool, ScoreError> {
    let left = canonical_jcs_bytes(left).map_err(|error| {
        ScoreError::new(ScoreErrorKind::MalformedEvidence, format!("left JCS canonicalization failed: {error}"))
    })?;
    let right = canonical_jcs_bytes(right).map_err(|error| {
        ScoreError::new(ScoreErrorKind::MalformedEvidence, format!("right JCS canonicalization failed: {error}"))
    })?;
    Ok(left == right)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScopeResult {
    Match,
    Inapplicable,
    Unverifiable,
}

fn compare_optional(requested: Option<&str>, observed: Option<&str>) -> ScopeResult {
    match (requested, observed) {
        (Some(left), Some(right)) if left == right => ScopeResult::Match,
        (Some(_), Some(_)) => ScopeResult::Inapplicable,
        (None, None) => ScopeResult::Match,
        _ => ScopeResult::Unverifiable,
    }
}

fn assess_scope(task: &TaskSpec, evidence: &Value) -> (ScopeResult, String) {
    match string_at(evidence, &["subject", "digest", "sha256"]) {
        Some(value) if value == task.scope.artifact_sha256 => {}
        Some(_) => return (ScopeResult::Inapplicable, "artifact digest differs from task scope".to_owned()),
        None => return (ScopeResult::Unverifiable, "Evidence does not bind an artifact digest".to_owned()),
    }

    if string_at(evidence, &["operation"]) != Some(task.scope.operation.as_str()) {
        return (ScopeResult::Inapplicable, "operation differs from task scope".to_owned());
    }

    if task.scope.platform_version.is_some() {
        return (ScopeResult::Unverifiable, "REA Evidence v1 does not encode platform_version".to_owned());
    }

    let platform = compare_optional(
        task.scope.platform.as_deref(),
        string_at(evidence, &["environment", "platform"]),
    );
    if platform != ScopeResult::Match {
        return (
            platform,
            match platform {
                ScopeResult::Inapplicable => "platform differs from task scope",
                ScopeResult::Unverifiable => "platform is absent or broader than task scope",
                ScopeResult::Match => unreachable!(),
            }.to_owned(),
        );
    }

    let environment = compare_optional(
        task.scope.environment_id.as_deref(),
        string_at(evidence, &["environment", "id"]),
    );
    if environment != ScopeResult::Match {
        return (
            environment,
            match environment {
                ScopeResult::Inapplicable => "environment_id differs from task scope",
                ScopeResult::Unverifiable => "environment identity is absent or broader than task scope",
                ScopeResult::Match => unreachable!(),
            }.to_owned(),
        );
    }

    (ScopeResult::Match, "artifact, operation and represented environment match".to_owned())
}

fn assessment(id: &str, kind: AssessmentKind, reason: impl Into<String>) -> EvidenceAssessment {
    EvidenceAssessment { evidence_id: id.to_owned(), kind, reason: reason.into() }
}

fn evaluate_evidence(
    task: &TaskSpec,
    referenced_ids: &[String],
    supplied: &[String],
) -> Result<(Disposition, Vec<EvidenceAssessment>, usize, usize), ScoreError> {
    if supplied.len() > MAX_EVIDENCE_RECORDS || referenced_ids.len() > MAX_EVIDENCE_RECORDS {
        return Err(ScoreError::new(
            ScoreErrorKind::TooManyEvidenceRecords,
            format!("Evidence record/reference count exceeds {MAX_EVIDENCE_RECORDS}"),
        ));
    }

    let mut total_bytes = 0usize;
    let mut by_id = HashMap::<String, VerifiedEvidence>::new();
    for raw in supplied {
        total_bytes = total_bytes.checked_add(raw.len()).ok_or_else(|| {
            ScoreError::new(ScoreErrorKind::InputTooLarge, "aggregate Evidence byte count overflow")
        })?;
        if total_bytes > MAX_TOTAL_EVIDENCE_BYTES {
            return Err(ScoreError::new(
                ScoreErrorKind::InputTooLarge,
                "aggregate Evidence bytes exceed 128 MiB benchmark limit",
            ));
        }
        let verified = verify_evidence_json(raw.as_bytes()).map_err(|error| {
            let kind = if error.kind == symthaea_rea_compat::EvidenceErrorKind::InputTooLarge {
                ScoreErrorKind::InputTooLarge
            } else {
                ScoreErrorKind::MalformedEvidence
            };
            ScoreError::new(kind, format!("REA Evidence verification failed: {error}"))
        })?;
        if !referenced_ids.iter().any(|id| id == &verified.evidence_id) {
            return Err(ScoreError::new(
                ScoreErrorKind::UnexpectedEvidenceId,
                format!("unreferenced Evidence {} supplied", verified.evidence_id),
            ));
        }
        if by_id.insert(verified.evidence_id.clone(), verified).is_some() {
            return Err(ScoreError::new(
                ScoreErrorKind::DuplicateEvidenceId,
                "duplicate verified Evidence ID supplied",
            ));
        }
    }

    let mut seen = HashSet::new();
    for id in referenced_ids {
        if !id.starts_with("ev_") || id.len() != 67 || !is_sha256(&id[3..]) {
            return Err(ScoreError::new(
                ScoreErrorKind::InvalidTask,
                format!("malformed cited Evidence ID: {id}"),
            ));
        }
        if !seen.insert(id) {
            return Err(ScoreError::new(
                ScoreErrorKind::DuplicateEvidenceId,
                "claim references duplicate Evidence IDs",
            ));
        }
    }

    let mut assessments = Vec::new();
    let mut missing = 0usize;
    let mut supports = 0usize;
    let mut contradicts = 0usize;
    let mut inapplicable = 0usize;
    let mut unverifiable = 0usize;

    let predicate_supported = task.predicate == "provider_reported_observation";
    let assumptions_verified = task.assumptions.is_empty();
    for id in referenced_ids {
        let Some(record) = by_id.get(id) else {
            missing += 1;
            continue;
        };
        if !predicate_supported || !assumptions_verified {
            assessments.push(assessment(
                id,
                AssessmentKind::NotEvaluated,
                "no authorized v0 rule evaluates this predicate/assumption set",
            ));
            continue;
        }

        let (scope, reason) = assess_scope(task, &record.value);
        match scope {
            ScopeResult::Inapplicable => {
                inapplicable += 1;
                assessments.push(assessment(id, AssessmentKind::Inapplicable, reason));
                continue;
            }
            ScopeResult::Unverifiable => {
                unverifiable += 1;
                assessments.push(assessment(id, AssessmentKind::ScopeUnverifiable, reason));
                continue;
            }
            ScopeResult::Match => {}
        }

        if string_at(&record.value, &["predicate_type"])
            != Some(task.scope.source_predicate_type.as_str())
        {
            inapplicable += 1;
            assessments.push(assessment(
                id,
                AssessmentKind::Inapplicable,
                "source predicate type differs from task scope",
            ));
            continue;
        }

        let normalized = record.value.get("normalized_result").unwrap_or(&Value::Null);
        let answer = task.expected_answer.as_ref();
        let matches = match answer {
            Some(answer) => canonical_equal(normalized, answer)?,
            None => false,
        };
        if matches {
            supports += 1;
            assessments.push(assessment(
                id,
                AssessmentKind::Supports,
                "normalized_result matches the task's expected value under RFC 8785 JCS",
            ));
        } else {
            contradicts += 1;
            assessments.push(assessment(
                id,
                AssessmentKind::Contradicts,
                "normalized_result differs from the task's expected value",
            ));
        }
    }

    let disposition = if !predicate_supported || !assumptions_verified {
        Disposition::Unverified
    } else if supports > 0 && contradicts > 0 {
        Disposition::Conflicted
    } else if unverifiable > 0 || missing > 0 {
        if supports == 0 && contradicts == 0 && inapplicable > 0 && unverifiable == 0 && missing == 0 {
            Disposition::Inapplicable
        } else {
            Disposition::Underdetermined
        }
    } else if supports > 0 {
        Disposition::SupportedWithinScope
    } else if contradicts > 0 {
        Disposition::RefutedWithinScope
    } else if inapplicable > 0 {
        Disposition::Inapplicable
    } else {
        Disposition::Underdetermined
    };

    Ok((disposition, assessments, by_id.len(), missing))
}

pub fn score_submission(task: &TaskSpec, submission: &AgentSubmission) -> Result<TaskScore, ScoreError> {
    validate_task(task)?;
    if submission.task_id != task.task_id {
        return Err(ScoreError::new(
            ScoreErrorKind::SubmissionTaskMismatch,
            "submission.task_id does not match task.task_id",
        ));
    }

    let raw_answer_correct = match (&task.expected_answer, &submission.answer) {
        (Some(expected), Some(answer)) => canonical_equal(expected, answer)?,
        (None, None) => true,
        _ => false,
    };

    let (evidence_disposition, evidence_assessments, verified_count, missing_count) =
        evaluate_evidence(task, &submission.evidence_ids, &submission.evidence_json)?;
    let evidence_grounded_correct =
        raw_answer_correct && evidence_disposition == task.expected_disposition;
    let parametric_leakage = raw_answer_correct && !evidence_grounded_correct;
    let false_supported_claim = evidence_disposition == Disposition::SupportedWithinScope
        && task.expected_disposition != Disposition::SupportedWithinScope;
    let false_refuted_claim = evidence_disposition == Disposition::RefutedWithinScope
        && task.expected_disposition != Disposition::RefutedWithinScope;

    Ok(TaskScore {
        task_id: task.task_id.clone(),
        pair_id: task.pair_id.clone(),
        variant: task.variant,
        raw_answer_correct,
        evidence_disposition,
        evidence_grounded_correct,
        parametric_leakage,
        false_supported_claim,
        false_refuted_claim,
        verified_evidence_count: verified_count,
        missing_evidence_count: missing_count,
        evidence_assessments,
    })
}

fn answer_equal(left: &Option<Value>, right: &Option<Value>) -> Result<bool, ScoreError> {
    match (left, right) {
        (None, None) => Ok(true),
        (Some(left), Some(right)) => canonical_equal(left, right),
        _ => Ok(false),
    }
}

pub fn score_batch(
    tasks: &[TaskSpec],
    submissions: &[AgentSubmission],
) -> Result<BatchScore, ScoreError> {
    let mut submission_by_task = HashMap::<&str, &AgentSubmission>::new();
    for submission in submissions {
        if submission_by_task.insert(submission.task_id.as_str(), submission).is_some() {
            return Err(ScoreError::new(
                ScoreErrorKind::SubmissionTaskMismatch,
                format!("duplicate submission for task {}", submission.task_id),
            ));
        }
    }

    let mut scores = Vec::with_capacity(tasks.len());
    let mut answer_by_task = HashMap::<String, Option<Value>>::new();
    for task in tasks {
        let default_submission = AgentSubmission {
            task_id: task.task_id.clone(),
            answer: None,
            evidence_ids: Vec::new(),
            evidence_json: Vec::new(),
        };
        let submission = submission_by_task.get(task.task_id.as_str()).copied().unwrap_or(&default_submission);
        let score = score_submission(task, submission)?;
        answer_by_task.insert(task.task_id.clone(), submission.answer.clone());
        scores.push(score);
    }

    let task_by_pair: BTreeMap<(&str, TaskVariant), &TaskSpec> = tasks
        .iter()
        .map(|task| ((task.pair_id.as_str(), task.variant), task))
        .collect();

    let mut relevant_pairs = 0usize;
    let mut correctly_sensitive_pairs = 0usize;
    let mut irrelevant_pairs = 0usize;
    let mut correctly_invariant_pairs = 0usize;

    let pair_ids: HashSet<&str> = tasks.iter().map(|task| task.pair_id.as_str()).collect();
    for pair_id in pair_ids {
        if let (Some(base), Some(mutated)) = (
            task_by_pair.get(&(pair_id, TaskVariant::Baseline)),
            task_by_pair.get(&(pair_id, TaskVariant::RelevantMutation)),
        ) {
            relevant_pairs += 1;
            let base_score = scores.iter().find(|score| score.task_id == base.task_id).expect("score exists");
            let mutated_score = scores.iter().find(|score| score.task_id == mutated.task_id).expect("score exists");
            let base_answer = answer_by_task.get(&base.task_id).expect("answer exists");
            let mutated_answer = answer_by_task.get(&mutated.task_id).expect("answer exists");
            if base_score.raw_answer_correct
                && mutated_score.raw_answer_correct
                && !answer_equal(base_answer, mutated_answer)?
            {
                correctly_sensitive_pairs += 1;
            }
        }

        if let (Some(base), Some(mutated)) = (
            task_by_pair.get(&(pair_id, TaskVariant::Baseline)),
            task_by_pair.get(&(pair_id, TaskVariant::IrrelevantMutation)),
        ) {
            irrelevant_pairs += 1;
            let base_score = scores.iter().find(|score| score.task_id == base.task_id).expect("score exists");
            let mutated_score = scores.iter().find(|score| score.task_id == mutated.task_id).expect("score exists");
            let base_answer = answer_by_task.get(&base.task_id).expect("answer exists");
            let mutated_answer = answer_by_task.get(&mutated.task_id).expect("answer exists");
            if base_score.raw_answer_correct
                && mutated_score.raw_answer_correct
                && answer_equal(base_answer, mutated_answer)?
            {
                correctly_invariant_pairs += 1;
            }
        }
    }

    let task_count = scores.len();
    let raw_correct_count = scores.iter().filter(|score| score.raw_answer_correct).count();
    let grounded_count = scores.iter().filter(|score| score.evidence_grounded_correct).count();
    let leakage_count = scores.iter().filter(|score| score.parametric_leakage).count();
    let false_supported_count = scores.iter().filter(|score| score.false_supported_claim).count();
    let false_refuted_count = scores.iter().filter(|score| score.false_refuted_claim).count();

    Ok(BatchScore {
        schema: BENCHMARK_SCHEMA.to_owned(),
        task_count,
        raw_correct_count,
        evidence_grounded_correct_count: grounded_count,
        parametric_leakage_count: leakage_count,
        false_supported_count,
        false_refuted_count,
        raw_accuracy: Ratio::new(raw_correct_count, task_count),
        evidence_grounded_accuracy: Ratio::new(grounded_count, task_count),
        parametric_leakage_rate: Ratio::new(leakage_count, raw_correct_count),
        pair_metrics: PairMetrics {
            relevant_pairs,
            correctly_sensitive_pairs,
            relevant_sensitivity: Ratio::new(correctly_sensitive_pairs, relevant_pairs),
            irrelevant_pairs,
            correctly_invariant_pairs,
            irrelevant_invariance: Ratio::new(correctly_invariant_pairs, irrelevant_pairs),
        },
        task_scores: scores,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use symthaea_rea_compat::{compute_evidence_id, verify_evidence_json};

    const FIXTURE: &[u8] = include_bytes!(
        "../../../crates/domains/symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json"
    );

    fn fixture_value() -> Value {
        serde_json::from_slice(FIXTURE).unwrap()
    }

    fn evidence_with(subject_sha256: &str, normalized_result: Value) -> String {
        let mut value = fixture_value();
        value["subject"]["digest"]["sha256"] = json!(subject_sha256);
        value["normalized_result"] = normalized_result;
        value["evidence_id"] = json!(compute_evidence_id(&value).unwrap());
        serde_json::to_string(&value).unwrap()
    }

    fn task(
        task_id: &str,
        pair_id: &str,
        variant: TaskVariant,
        subject_sha256: &str,
        expected_answer: Option<Value>,
        expected_disposition: Disposition,
    ) -> TaskSpec {
        TaskSpec {
            schema: BENCHMARK_SCHEMA.to_owned(),
            task_id: task_id.to_owned(),
            pair_id: pair_id.to_owned(),
            variant,
            scope: ClaimScope {
                artifact_sha256: subject_sha256.to_owned(),
                operation: "inspect_artifact".to_owned(),
                platform: None,
                platform_version: None,
                environment_id: None,
                source_predicate_type: "rea.analysis".to_owned(),
            },
            predicate: "provider_reported_observation".to_owned(),
            expected_answer,
            expected_disposition,
            assumptions: Vec::new(),
        }
    }

    fn submission(task_id: &str, answer: Option<Value>, evidence: Vec<String>) -> AgentSubmission {
        let evidence_ids = evidence
            .iter()
            .map(|raw| verify_evidence_json(raw.as_bytes()).unwrap().evidence_id)
            .collect();
        AgentSubmission {
            task_id: task_id.to_owned(),
            answer,
            evidence_ids,
            evidence_json: evidence,
        }
    }

    #[test]
    fn correct_answer_without_source_is_parametric_leakage_not_grounded_credit() {
        let expected = json!({
            "entries": [{"path": "classes.dex", "size": 42}],
            "status": "complete"
        });
        let task = task(
            "baseline-001",
            "pair-001",
            TaskVariant::Baseline,
            &"a".repeat(64),
            Some(expected.clone()),
            Disposition::SupportedWithinScope,
        );
        let result = score_submission(
            &task,
            &AgentSubmission {
                task_id: task.task_id.clone(),
                answer: Some(expected),
                evidence_ids: Vec::new(),
                evidence_json: Vec::new(),
            },
        )
        .unwrap();

        assert!(result.raw_answer_correct);
        assert_eq!(result.evidence_disposition, Disposition::Underdetermined);
        assert!(!result.evidence_grounded_correct);
        assert!(result.parametric_leakage);
    }

    #[test]
    fn exact_subject_operation_and_value_are_required_for_grounded_credit() {
        let expected = json!({
            "entries": [{"path": "classes.dex", "size": 42}],
            "status": "complete"
        });
        let raw = evidence_with(&"a".repeat(64), expected.clone());
        let task = task(
            "baseline-002",
            "pair-002",
            TaskVariant::Baseline,
            &"a".repeat(64),
            Some(expected.clone()),
            Disposition::SupportedWithinScope,
        );
        let result = score_submission(
            &task,
            &submission("baseline-002", Some(expected), vec![raw]),
        )
        .unwrap();

        assert!(result.raw_answer_correct);
        assert_eq!(result.evidence_disposition, Disposition::SupportedWithinScope);
        assert!(result.evidence_grounded_correct);
        assert!(!result.parametric_leakage);
    }

    #[test]
    fn correct_value_from_wrong_artifact_is_not_source_grounded() {
        let expected = json!({"status": "complete"});
        let raw = evidence_with(&"b".repeat(64), expected.clone());
        let task = task(
            "wrong-artifact",
            "pair-003",
            TaskVariant::Baseline,
            &"a".repeat(64),
            Some(expected.clone()),
            Disposition::SupportedWithinScope,
        );
        let result = score_submission(
            &task,
            &submission("wrong-artifact", Some(expected), vec![raw]),
        )
        .unwrap();

        assert!(result.raw_answer_correct);
        assert_eq!(result.evidence_disposition, Disposition::Inapplicable);
        assert!(!result.evidence_grounded_correct);
        assert!(result.parametric_leakage);
    }

    #[test]
    fn relevant_mutation_requires_answer_to_follow_the_changed_artifact() {
        let baseline_value = json!({"size": 42});
        let mutated_value = json!({"size": 43});
        let baseline_raw = evidence_with(&"a".repeat(64), baseline_value.clone());
        let mutated_raw = evidence_with(&"b".repeat(64), mutated_value.clone());

        let tasks = vec![
            task(
                "base",
                "relevant-pair",
                TaskVariant::Baseline,
                &"a".repeat(64),
                Some(baseline_value.clone()),
                Disposition::SupportedWithinScope,
            ),
            task(
                "mutated",
                "relevant-pair",
                TaskVariant::RelevantMutation,
                &"b".repeat(64),
                Some(mutated_value.clone()),
                Disposition::SupportedWithinScope,
            ),
        ];
        let submissions = vec![
            submission("base", Some(baseline_value), vec![baseline_raw]),
            submission("mutated", Some(mutated_value), vec![mutated_raw]),
        ];

        let batch = score_batch(&tasks, &submissions).unwrap();
        assert_eq!(batch.pair_metrics.relevant_pairs, 1);
        assert_eq!(batch.pair_metrics.correctly_sensitive_pairs, 1);
        assert_eq!(batch.pair_metrics.relevant_sensitivity.value, Some(1.0));
    }

    #[test]
    fn irrelevant_mutation_must_preserve_the_same_answer() {
        let answer = json!({"status": "complete"});
        let baseline_raw = evidence_with(&"a".repeat(64), answer.clone());
        let irrelevant_raw = evidence_with(&"c".repeat(64), answer.clone());
        let tasks = vec![
            task(
                "base",
                "irrelevant-pair",
                TaskVariant::Baseline,
                &"a".repeat(64),
                Some(answer.clone()),
                Disposition::SupportedWithinScope,
            ),
            task(
                "irrelevant",
                "irrelevant-pair",
                TaskVariant::IrrelevantMutation,
                &"c".repeat(64),
                Some(answer.clone()),
                Disposition::SupportedWithinScope,
            ),
        ];
        let submissions = vec![
            submission("base", Some(answer.clone()), vec![baseline_raw]),
            submission("irrelevant", Some(answer), vec![irrelevant_raw]),
        ];
        let batch = score_batch(&tasks, &submissions).unwrap();
        assert_eq!(batch.pair_metrics.irrelevant_pairs, 1);
        assert_eq!(batch.pair_metrics.correctly_invariant_pairs, 1);
        assert_eq!(batch.pair_metrics.irrelevant_invariance.value, Some(1.0));
    }

    #[test]
    fn unimplemented_predicate_cannot_become_grounded_with_more_observations() {
        let expected = json!({"status": "complete"});
        let raw = evidence_with(&"a".repeat(64), expected.clone());
        let mut task = task(
            "authenticity",
            "pair-auth",
            TaskVariant::Unsupported,
            &"a".repeat(64),
            None,
            Disposition::Unverified,
        );
        task.predicate = "producer_authenticity".to_owned();
        let result = score_submission(
            &task,
            &submission("authenticity", None, vec![raw]),
        )
        .unwrap();
        assert!(result.raw_answer_correct);
        assert_eq!(result.evidence_disposition, Disposition::Unverified);
        assert!(result.evidence_grounded_correct);
        assert!(!result.parametric_leakage);
    }

    #[test]
    fn duplicate_evidence_is_rejected_instead_of_double_counted() {
        let expected = json!({"status": "complete"});
        let raw = evidence_with(&"a".repeat(64), expected.clone());
        let mut submission = submission("duplicate", Some(expected.clone()), vec![raw.clone()]);
        submission.evidence_json.push(raw);
        submission.evidence_ids.push(submission.evidence_ids[0].clone());
        let task = task(
            "duplicate",
            "pair-duplicate",
            TaskVariant::Baseline,
            &"a".repeat(64),
            Some(expected),
            Disposition::SupportedWithinScope,
        );
        let error = score_submission(&task, &submission).unwrap_err();
        assert_eq!(error.kind, ScoreErrorKind::DuplicateEvidenceId);
    }
}
