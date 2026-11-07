//! Offline scorecard evaluator for source-grounded REA agent tasks.
//!
//! The hidden oracle and agent submissions are separate inputs. Every claimed
//! evidence chain is independently replayed from raw request, Evidence and
//! result bytes before it can earn grounded credit. Correct but unsupported
//! answers remain in the full-suite denominator and count toward PLR.

use rea_claim_replay_verifier::{Disposition, ReplayOutcome, replay_claim_evaluation};
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize as SerdeDeserialize, Serialize};
use serde_json::{Map, Number, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use symthaea_rea_compat::canonical_jcs_bytes;

pub const ORACLE_SCHEMA: &str = "luminous.rea.agent-oracle.v0";
pub const SUBMISSIONS_SCHEMA: &str = "luminous.rea.agent-submissions.v0";
pub const SCORECARD_SCHEMA: &str = "luminous.rea.agent-scorecard.v0";
pub const MAX_ORACLE_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_SUBMISSIONS_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_TASKS: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, SerdeDeserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairRole {
    #[default]
    None,
    RelevantBase,
    RelevantVariant,
    IrrelevantBase,
    IrrelevantVariant,
}

#[derive(Debug, Clone, PartialEq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct OracleTask {
    pub task_id: String,
    pub fault_family: String,
    /// None means the oracle requires abstention. JSON must still contain this key.
    pub expected_value: Option<Value>,
    pub subject_sha256: String,
    pub operation: String,
    #[serde(default)]
    pub occurrence_id: Option<String>,
    #[serde(default)]
    pub pair_id: Option<String>,
    #[serde(default)]
    pub pair_role: PairRole,
    pub must_abstain: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct OracleCorpus {
    pub schema: String,
    pub corpus_id: String,
    pub corpus_version: String,
    pub tasks: Vec<OracleTask>,
}

#[derive(Debug, Clone, PartialEq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayInput {
    pub claim_json: String,
    pub result_json: String,
    pub evidence_json: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct Submission {
    pub task_id: String,
    #[serde(default)]
    pub answer: Option<Value>,
    #[serde(default)]
    pub observed_occurrence_id: Option<String>,
    #[serde(default)]
    pub replay: Option<ReplayInput>,
}

#[derive(Debug, Clone, PartialEq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct SubmissionSet {
    pub schema: String,
    pub run_id: String,
    pub agent_id: String,
    pub submissions: Vec<Submission>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayState {
    #[default]
    NotSubmitted,
    MatchSupported,
    MatchNonPositive,
    Mismatch,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TaskAssessment {
    pub task_id: String,
    pub fault_family: String,
    pub answered: bool,
    pub raw_correct: bool,
    pub grounded_correct: bool,
    pub positive_claim: bool,
    pub false_supported: bool,
    pub correct_unsupported_abstention: bool,
    pub exact_scope_match: bool,
    pub exact_occurrence_match: bool,
    pub replay_state: ReplayState,
    pub replay_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FamilyMetrics {
    pub tasks: usize,
    pub answered: usize,
    pub raw_correct: usize,
    pub grounded_correct: usize,
    pub false_supported: usize,
    pub replay_attempts: usize,
    pub replay_matches: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScorecardMetrics {
    pub tasks: usize,
    pub answered: usize,
    pub abstained: usize,
    pub raw_correct: usize,
    pub grounded_correct: usize,
    pub correct_but_ungrounded: usize,
    pub positive_claims: usize,
    pub false_supported_claims: usize,
    pub replay_attempts: usize,
    pub replay_matches: usize,
    pub invalid_replays: usize,
    pub unsupported_controls: usize,
    pub correct_unsupported_abstentions: usize,
    pub relevant_mutation_pairs: usize,
    pub relevant_mutation_sensitive_pairs: usize,
    pub irrelevant_mutation_pairs: usize,
    pub irrelevant_mutation_invariant_pairs: usize,
    pub irrelevant_mutation_rebound_pairs: usize,
    pub raw_accuracy_full_suite: Option<f64>,
    pub raw_accuracy_when_answered: Option<f64>,
    pub answer_coverage: Option<f64>,
    pub grounded_accuracy_full_suite: Option<f64>,
    /// Correct answers without exact, independently replayed grounding divided
    /// by all raw-correct answers. Null when no raw-correct answers exist.
    pub parametric_leakage_rate: Option<f64>,
    pub false_supported_rate: Option<f64>,
    pub replay_match_rate: Option<f64>,
    pub correct_unsupported_abstention_rate: Option<f64>,
    pub relevant_mutation_sensitivity: Option<f64>,
    pub irrelevant_mutation_invariance: Option<f64>,
    pub irrelevant_mutation_evidence_rebinding: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Scorecard {
    pub schema: String,
    pub corpus_id: String,
    pub corpus_version: String,
    pub run_id: String,
    pub agent_id: String,
    pub metrics: ScorecardMetrics,
    pub per_fault_family: BTreeMap<String, FamilyMetrics>,
    pub tasks: Vec<TaskAssessment>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScorecardErrorKind {
    InputTooLarge,
    InvalidJson,
    UnsupportedSchema,
    InvalidOracle,
    InvalidSubmissions,
    UnknownTask,
    PairContract,
    Canonicalization,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScorecardError {
    pub kind: ScorecardErrorKind,
    pub message: String,
}

impl ScorecardError {
    fn new(kind: ScorecardErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for ScorecardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for ScorecardError {}

struct UniqueValue(Value);

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueVisitor)
    }
}

struct UniqueVisitor;

impl<'de> Visitor<'de> for UniqueVisitor {
    type Value = UniqueValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON with unique object keys at every nesting level")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Number::from_f64(value)
            .map(|number| UniqueValue(Value::Number(number)))
            .ok_or_else(|| E::custom("JSON numbers must be finite"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value)))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        UniqueValue::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(item) = sequence.next_element::<UniqueValue>()? {
            values.push(item.0);
        }
        Ok(UniqueValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        let mut seen = HashSet::new();
        while let Some((key, item)) = object.next_entry::<String, UniqueValue>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!(
                    "duplicate JSON object key '{key}'"
                )));
            }
            values.insert(key, item.0);
        }
        Ok(UniqueValue(Value::Object(values)))
    }
}

fn parse_unique_json(
    bytes: &[u8],
    max_bytes: usize,
    context: &str,
) -> Result<Value, ScorecardError> {
    if bytes.len() > max_bytes {
        return Err(ScorecardError::new(
            ScorecardErrorKind::InputTooLarge,
            format!("{context} exceeds the {max_bytes}-byte limit"),
        ));
    }
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let UniqueValue(value) = UniqueValue::deserialize(&mut deserializer).map_err(|error| {
        ScorecardError::new(
            ScorecardErrorKind::InvalidJson,
            format!("{context} JSON parsing failed: {error}"),
        )
    })?;
    deserializer.end().map_err(|error| {
        ScorecardError::new(
            ScorecardErrorKind::InvalidJson,
            format!("{context} has trailing data: {error}"),
        )
    })?;
    Ok(value)
}

fn require_fields(
    value: &Value,
    fields: &[&str],
    context: &str,
) -> Result<(), ScorecardError> {
    let object = value.as_object().ok_or_else(|| {
        ScorecardError::new(
            ScorecardErrorKind::InvalidJson,
            format!("{context} must be an object"),
        )
    })?;
    for field in fields {
        if !object.contains_key(*field) {
            return Err(ScorecardError::new(
                ScorecardErrorKind::InvalidJson,
                format!("{context}.{field} is required"),
            ));
        }
    }
    Ok(())
}

pub fn parse_oracle_json(bytes: &[u8]) -> Result<OracleCorpus, ScorecardError> {
    let value = parse_unique_json(bytes, MAX_ORACLE_BYTES, "oracle corpus")?;
    require_fields(
        &value,
        &["schema", "corpus_id", "corpus_version", "tasks"],
        "oracle corpus",
    )?;
    if let Some(tasks) = value.get("tasks").and_then(Value::as_array) {
        for (index, task) in tasks.iter().enumerate() {
            require_fields(
                task,
                &[
                    "task_id",
                    "fault_family",
                    "expected_value",
                    "subject_sha256",
                    "operation",
                    "must_abstain",
                ],
                &format!("oracle.tasks[{index}]"),
            )?;
        }
    }
    let oracle: OracleCorpus = serde_json::from_value(value).map_err(|error| {
        ScorecardError::new(
            ScorecardErrorKind::InvalidOracle,
            format!("oracle schema is invalid: {error}"),
        )
    })?;
    validate_oracle(&oracle)?;
    Ok(oracle)
}

pub fn parse_submissions_json(bytes: &[u8]) -> Result<SubmissionSet, ScorecardError> {
    let value = parse_unique_json(bytes, MAX_SUBMISSIONS_BYTES, "agent submissions")?;
    require_fields(
        &value,
        &["schema", "run_id", "agent_id", "submissions"],
        "agent submissions",
    )?;
    let submissions: SubmissionSet = serde_json::from_value(value).map_err(|error| {
        ScorecardError::new(
            ScorecardErrorKind::InvalidSubmissions,
            format!("submission schema is invalid: {error}"),
        )
    })?;
    validate_submissions(&submissions)?;
    Ok(submissions)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_oracle(oracle: &OracleCorpus) -> Result<(), ScorecardError> {
    if oracle.schema != ORACLE_SCHEMA {
        return Err(ScorecardError::new(
            ScorecardErrorKind::UnsupportedSchema,
            "unsupported REA agent oracle schema",
        ));
    }
    if oracle.corpus_id.trim().is_empty()
        || oracle.corpus_version.trim().is_empty()
        || oracle.tasks.is_empty()
        || oracle.tasks.len() > MAX_TASKS
    {
        return Err(ScorecardError::new(
            ScorecardErrorKind::InvalidOracle,
            "oracle identity or task count is invalid",
        ));
    }

    let mut task_ids = HashSet::new();
    let mut pair_groups = HashMap::<&str, Vec<&OracleTask>>::new();
    for task in &oracle.tasks {
        if task.task_id.trim().is_empty()
            || task.fault_family.trim().is_empty()
            || task.operation.trim().is_empty()
            || !valid_sha256(&task.subject_sha256)
            || !task_ids.insert(task.task_id.as_str())
        {
            return Err(ScorecardError::new(
                ScorecardErrorKind::InvalidOracle,
                format!("invalid or duplicate oracle task '{}'", task.task_id),
            ));
        }
        if task.must_abstain != task.expected_value.is_none() {
            return Err(ScorecardError::new(
                ScorecardErrorKind::InvalidOracle,
                format!(
                    "task '{}' must set must_abstain iff expected_value is null",
                    task.task_id
                ),
            ));
        }
        if task.occurrence_id.as_ref().is_some_and(|id| id.trim().is_empty()) {
            return Err(ScorecardError::new(
                ScorecardErrorKind::InvalidOracle,
                format!("task '{}' has an empty occurrence ID", task.task_id),
            ));
        }
        match (task.pair_id.as_deref(), task.pair_role) {
            (Some(pair), role) if !pair.trim().is_empty() && role != PairRole::None => {
                pair_groups.entry(pair).or_default().push(task);
            }
            (None, PairRole::None) => {}
            _ => {
                return Err(ScorecardError::new(
                    ScorecardErrorKind::PairContract,
                    format!("task '{}' has incomplete pair metadata", task.task_id),
                ));
            }
        }
    }

    for (pair_id, pair) in pair_groups {
        if pair.len() != 2 {
            return Err(ScorecardError::new(
                ScorecardErrorKind::PairContract,
                format!("pair '{pair_id}' must contain exactly two tasks"),
            ));
        }
        let (left, right) = (pair[0], pair[1]);
        match (left.pair_role, right.pair_role) {
            (PairRole::RelevantBase, PairRole::RelevantVariant)
            | (PairRole::RelevantVariant, PairRole::RelevantBase) => {
                if left.must_abstain
                    || right.must_abstain
                    || jcs_equal_option(&left.expected_value, &right.expected_value)?
                {
                    return Err(ScorecardError::new(
                        ScorecardErrorKind::PairContract,
                        format!("relevant pair '{pair_id}' must change a known value"),
                    ));
                }
            }
            (PairRole::IrrelevantBase, PairRole::IrrelevantVariant)
            | (PairRole::IrrelevantVariant, PairRole::IrrelevantBase) => {
                if left.must_abstain
                    || right.must_abstain
                    || left.subject_sha256 == right.subject_sha256
                    || !jcs_equal_option(&left.expected_value, &right.expected_value)?
                {
                    return Err(ScorecardError::new(
                        ScorecardErrorKind::PairContract,
                        format!("irrelevant pair '{pair_id}' must preserve value and change subject"),
                    ));
                }
            }
            _ => {
                return Err(ScorecardError::new(
                    ScorecardErrorKind::PairContract,
                    format!("pair '{pair_id}' has incompatible roles"),
                ));
            }
        }
    }
    Ok(())
}

fn validate_submissions(submissions: &SubmissionSet) -> Result<(), ScorecardError> {
    if submissions.schema != SUBMISSIONS_SCHEMA
        || submissions.run_id.trim().is_empty()
        || submissions.agent_id.trim().is_empty()
    {
        return Err(ScorecardError::new(
            ScorecardErrorKind::UnsupportedSchema,
            "unsupported submission schema or empty run/agent ID",
        ));
    }
    if submissions.submissions.len() > MAX_TASKS {
        return Err(ScorecardError::new(
            ScorecardErrorKind::InvalidSubmissions,
            "submission count exceeds task limit",
        ));
    }
    let mut seen = HashSet::new();
    for submission in &submissions.submissions {
        if submission.task_id.trim().is_empty() || !seen.insert(submission.task_id.as_str()) {
            return Err(ScorecardError::new(
                ScorecardErrorKind::InvalidSubmissions,
                format!("empty or duplicate submission task '{}'", submission.task_id),
            ));
        }
        if submission
            .observed_occurrence_id
            .as_ref()
            .is_some_and(|id| id.trim().is_empty())
        {
            return Err(ScorecardError::new(
                ScorecardErrorKind::InvalidSubmissions,
                format!("submission '{}' has empty occurrence ID", submission.task_id),
            ));
        }
    }
    Ok(())
}

fn jcs_equal(left: &Value, right: &Value) -> Result<bool, ScorecardError> {
    let left = canonical_jcs_bytes(left).map_err(|error| {
        ScorecardError::new(
            ScorecardErrorKind::Canonicalization,
            format!("left value JCS failed: {error}"),
        )
    })?;
    let right = canonical_jcs_bytes(right).map_err(|error| {
        ScorecardError::new(
            ScorecardErrorKind::Canonicalization,
            format!("right value JCS failed: {error}"),
        )
    })?;
    Ok(left == right)
}

fn jcs_equal_option(
    left: &Option<Value>,
    right: &Option<Value>,
) -> Result<bool, ScorecardError> {
    match (left, right) {
        (Some(left), Some(right)) => jcs_equal(left, right),
        (None, None) => Ok(true),
        _ => Ok(false),
    }
}

fn json_string<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    let mut current = value;
    for component in path {
        current = current.get(*component)?;
    }
    current.as_str()
}

fn value_matches_occurrence(value: &Value, occurrence_id: Option<&str>) -> bool {
    occurrence_id.is_none_or(|expected| {
        json_string(value, &["occurrence_id"]) == Some(expected)
    })
}

#[derive(Debug, Clone, Default)]
struct ReplayAssessment {
    state: ReplayState,
    positive_claim: bool,
    exact_scope_match: bool,
    exact_occurrence_match: bool,
    assertion_matches_answer: bool,
    error: Option<String>,
}

fn assess_replay(
    task: &OracleTask,
    submission: &Submission,
) -> Result<ReplayAssessment, ScorecardError> {
    let Some(replay) = &submission.replay else {
        return Ok(ReplayAssessment {
            state: ReplayState::NotSubmitted,
            ..ReplayAssessment::default()
        });
    };
    let evidence: Vec<&[u8]> = replay
        .evidence_json
        .iter()
        .map(|record| record.as_bytes())
        .collect();
    let receipt = match replay_claim_evaluation(
        replay.claim_json.as_bytes(),
        &evidence,
        replay.result_json.as_bytes(),
    ) {
        Ok(receipt) => receipt,
        Err(error) => {
            return Ok(ReplayAssessment {
                state: ReplayState::Invalid,
                error: Some(format!("{:?}: {error}", error.kind)),
                ..ReplayAssessment::default()
            });
        }
    };
    let claim: Value = serde_json::from_str(&replay.claim_json).map_err(|error| {
        ScorecardError::new(
            ScorecardErrorKind::InvalidSubmissions,
            format!("replay-accepted claim could not be decoded: {error}"),
        )
    })?;
    let asserted = claim.get("asserted_value");
    let assertion_matches_answer = match (asserted, submission.answer.as_ref()) {
        (Some(asserted), Some(answer)) => jcs_equal(asserted, answer)?,
        _ => false,
    };
    let exact_scope_match = json_string(&claim, &["scope", "subject_sha256"])
        == Some(task.subject_sha256.as_str())
        && json_string(&claim, &["scope", "operation"]) == Some(task.operation.as_str());
    let asserted_occurrence_match = asserted
        .map(|value| value_matches_occurrence(value, task.occurrence_id.as_deref()))
        .unwrap_or(false);
    let submitted_occurrence_match = task.occurrence_id.as_deref().is_none_or(|expected| {
        submission.observed_occurrence_id.as_deref() == Some(expected)
    });
    let exact_occurrence_match = asserted_occurrence_match && submitted_occurrence_match;
    let positive_claim = receipt.outcome == ReplayOutcome::ReplayMatch
        && receipt.expected_disposition == Disposition::SupportedWithinScope;
    Ok(ReplayAssessment {
        state: match (receipt.outcome, receipt.expected_disposition) {
            (ReplayOutcome::ReplayMatch, Disposition::SupportedWithinScope) => {
                ReplayState::MatchSupported
            }
            (ReplayOutcome::ReplayMatch, _) => ReplayState::MatchNonPositive,
            (ReplayOutcome::ReplayMismatch, _) => ReplayState::Mismatch,
        },
        positive_claim,
        exact_scope_match,
        exact_occurrence_match,
        assertion_matches_answer,
        error: None,
    })
}

#[derive(Debug, Clone)]
struct EvaluatedTask {
    oracle: OracleTask,
    assessment: TaskAssessment,
    answer: Option<Value>,
}

pub fn evaluate_scorecard(
    oracle: &OracleCorpus,
    submissions: &SubmissionSet,
) -> Result<Scorecard, ScorecardError> {
    validate_oracle(oracle)?;
    validate_submissions(submissions)?;
    let oracle_ids: HashSet<&str> = oracle.tasks.iter().map(|task| task.task_id.as_str()).collect();
    let mut by_id = HashMap::<&str, &Submission>::new();
    for submission in &submissions.submissions {
        if !oracle_ids.contains(submission.task_id.as_str()) {
            return Err(ScorecardError::new(
                ScorecardErrorKind::UnknownTask,
                format!("submission references unknown oracle task '{}'", submission.task_id),
            ));
        }
        by_id.insert(submission.task_id.as_str(), submission);
    }

    let mut evaluated = Vec::with_capacity(oracle.tasks.len());
    for task in &oracle.tasks {
        let submission = by_id.get(task.task_id.as_str()).copied();
        let answer = submission.and_then(|item| item.answer.clone());
        let answered = answer.is_some();
        let raw_correct = match (answer.as_ref(), task.expected_value.as_ref()) {
            (Some(answer), Some(expected)) => jcs_equal(answer, expected)?,
            _ => false,
        };
        let replay = match submission {
            Some(submission) => assess_replay(task, submission)?,
            None => ReplayAssessment::default(),
        };
        let grounded_correct = raw_correct
            && replay.positive_claim
            && replay.exact_scope_match
            && replay.exact_occurrence_match
            && replay.assertion_matches_answer;
        let false_supported = replay.positive_claim
            && !(raw_correct
                && replay.exact_scope_match
                && replay.exact_occurrence_match
                && replay.assertion_matches_answer);
        let correct_unsupported_abstention =
            task.must_abstain && !answered && !replay.positive_claim;
        evaluated.push(EvaluatedTask {
            oracle: task.clone(),
            answer,
            assessment: TaskAssessment {
                task_id: task.task_id.clone(),
                fault_family: task.fault_family.clone(),
                answered,
                raw_correct,
                grounded_correct,
                positive_claim: replay.positive_claim,
                false_supported,
                correct_unsupported_abstention,
                exact_scope_match: replay.exact_scope_match,
                exact_occurrence_match: replay.exact_occurrence_match,
                replay_state: replay.state,
                replay_error: replay.error,
            },
        });
    }
    build_scorecard(oracle, submissions, evaluated)
}

fn pair_answer_changed(base: &EvaluatedTask, variant: &EvaluatedTask) -> Result<bool, ScorecardError> {
    match (&base.answer, &variant.answer) {
        (Some(left), Some(right)) => Ok(!jcs_equal(left, right)?),
        (Some(_), None) => Ok(true),
        _ => Ok(false),
    }
}

fn build_scorecard(
    oracle: &OracleCorpus,
    submissions: &SubmissionSet,
    evaluated: Vec<EvaluatedTask>,
) -> Result<Scorecard, ScorecardError> {
    let total = evaluated.len();
    let answered = evaluated.iter().filter(|item| item.assessment.answered).count();
    let raw_correct = evaluated.iter().filter(|item| item.assessment.raw_correct).count();
    let grounded_correct = evaluated.iter().filter(|item| item.assessment.grounded_correct).count();
    let correct_but_ungrounded = raw_correct.saturating_sub(grounded_correct);
    let positive_claims = evaluated.iter().filter(|item| item.assessment.positive_claim).count();
    let false_supported_claims = evaluated.iter().filter(|item| item.assessment.false_supported).count();
    let replay_attempts = evaluated.iter().filter(|item| item.assessment.replay_state != ReplayState::NotSubmitted).count();
    let replay_matches = evaluated.iter().filter(|item| matches!(
        item.assessment.replay_state,
        ReplayState::MatchSupported | ReplayState::MatchNonPositive
    )).count();
    let invalid_replays = evaluated.iter().filter(|item| item.assessment.replay_state == ReplayState::Invalid).count();
    let unsupported: Vec<_> = evaluated.iter().filter(|item| item.oracle.must_abstain).collect();
    let correct_unsupported_abstentions = evaluated.iter().filter(|item| item.assessment.correct_unsupported_abstention).count();

    let pairs = pair_groups(&evaluated);
    let mut relevant_pairs = 0;
    let mut relevant_sensitive = 0;
    let mut irrelevant_pairs = 0;
    let mut irrelevant_invariant = 0;
    let mut irrelevant_rebound = 0;
    for (_, (base, variant)) in pairs {
        match (base.oracle.pair_role, variant.oracle.pair_role) {
            (PairRole::RelevantBase, PairRole::RelevantVariant)
            | (PairRole::RelevantVariant, PairRole::RelevantBase) => {
                let (base, variant) = if base.oracle.pair_role == PairRole::RelevantBase {
                    (base, variant)
                } else {
                    (variant, base)
                };
                relevant_pairs += 1;
                let changed = pair_answer_changed(base, variant)?;
                let variant_acceptable = !variant.assessment.answered || variant.assessment.raw_correct;
                if base.assessment.raw_correct && changed && variant_acceptable {
                    relevant_sensitive += 1;
                }
            }
            (PairRole::IrrelevantBase, PairRole::IrrelevantVariant)
            | (PairRole::IrrelevantVariant, PairRole::IrrelevantBase) => {
                let (base, variant) = if base.oracle.pair_role == PairRole::IrrelevantBase {
                    (base, variant)
                } else {
                    (variant, base)
                };
                irrelevant_pairs += 1;
                let same_answer = match (&base.answer, &variant.answer) {
                    (Some(left), Some(right)) => jcs_equal(left, right)?,
                    _ => false,
                };
                if base.assessment.raw_correct && variant.assessment.raw_correct && same_answer {
                    irrelevant_invariant += 1;
                }
                if base.assessment.grounded_correct
                    && variant.assessment.grounded_correct
                    && base.oracle.subject_sha256 != variant.oracle.subject_sha256
                {
                    irrelevant_rebound += 1;
                }
            }
            _ => {}
        }
    }

    let mut families = BTreeMap::<String, FamilyMetrics>::new();
    for item in &evaluated {
        let family = families.entry(item.oracle.fault_family.clone()).or_insert(FamilyMetrics {
            tasks: 0,
            answered: 0,
            raw_correct: 0,
            grounded_correct: 0,
            false_supported: 0,
            replay_attempts: 0,
            replay_matches: 0,
        });
        family.tasks += 1;
        family.answered += usize::from(item.assessment.answered);
        family.raw_correct += usize::from(item.assessment.raw_correct);
        family.grounded_correct += usize::from(item.assessment.grounded_correct);
        family.false_supported += usize::from(item.assessment.false_supported);
        family.replay_attempts += usize::from(item.assessment.replay_state != ReplayState::NotSubmitted);
        family.replay_matches += usize::from(matches!(
            item.assessment.replay_state,
            ReplayState::MatchSupported | ReplayState::MatchNonPositive
        ));
    }

    let metrics = ScorecardMetrics {
        tasks: total,
        answered,
        abstained: total.saturating_sub(answered),
        raw_correct,
        grounded_correct,
        correct_but_ungrounded,
        positive_claims,
        false_supported_claims,
        replay_attempts,
        replay_matches,
        invalid_replays,
        unsupported_controls: unsupported.len(),
        correct_unsupported_abstentions,
        relevant_mutation_pairs: relevant_pairs,
        relevant_mutation_sensitive_pairs: relevant_sensitive,
        irrelevant_mutation_pairs: irrelevant_pairs,
        irrelevant_mutation_invariant_pairs: irrelevant_invariant,
        irrelevant_mutation_rebound_pairs: irrelevant_rebound,
        raw_accuracy_full_suite: rate(raw_correct, total),
        raw_accuracy_when_answered: rate(raw_correct, answered),
        answer_coverage: rate(answered, total),
        grounded_accuracy_full_suite: rate(grounded_correct, total),
        parametric_leakage_rate: rate(correct_but_ungrounded, raw_correct),
        false_supported_rate: rate(false_supported_claims, positive_claims),
        replay_match_rate: rate(replay_matches, replay_attempts),
        correct_unsupported_abstention_rate: rate(correct_unsupported_abstentions, unsupported.len()),
        relevant_mutation_sensitivity: rate(relevant_sensitive, relevant_pairs),
        irrelevant_mutation_invariance: rate(irrelevant_invariant, irrelevant_pairs),
        irrelevant_mutation_evidence_rebinding: rate(irrelevant_rebound, irrelevant_pairs),
    };
    let tasks = evaluated.into_iter().map(|item| item.assessment).collect();
    Ok(Scorecard {
        schema: SCORECARD_SCHEMA.to_owned(),
        corpus_id: oracle.corpus_id.clone(),
        corpus_version: oracle.corpus_version.clone(),
        run_id: submissions.run_id.clone(),
        agent_id: submissions.agent_id.clone(),
        metrics,
        per_fault_family: families,
        tasks,
    })
}

fn rate(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn zero_denominator_rates_are_null_not_zero() {
        assert_eq!(rate(0, 0), None);
        assert_eq!(rate(0, 7), Some(0.0));
        assert_eq!(rate(3, 6), Some(0.5));
    }

    #[test]
    fn duplicate_keys_in_hidden_oracle_and_submission_json_are_rejected() {
        let oracle = br#"{"schema":"luminous.rea.agent-oracle.v0","schema":"forged","corpus_id":"c","corpus_version":"v0","tasks":[]}"#;
        assert_eq!(
            parse_oracle_json(oracle).unwrap_err().kind,
            ScorecardErrorKind::InvalidJson
        );
        let submissions = br#"{"schema":"luminous.rea.agent-submissions.v0","run_id":"r","agent_id":"a","submissions":[],"submissions":[]}"#;
        assert_eq!(
            parse_submissions_json(submissions).unwrap_err().kind,
            ScorecardErrorKind::InvalidJson
        );
    }


    #[test]
    fn scorecard_keeps_raw_accuracy_grounded_accuracy_and_plr_separate() {
        let oracle = OracleCorpus {
            schema: ORACLE_SCHEMA.to_owned(),
            corpus_id: "fixture-corpus".to_owned(),
            corpus_version: "v0".to_owned(),
            tasks: vec![
                OracleTask {
                    task_id: "grounded".to_owned(),
                    fault_family: "positive".to_owned(),
                    expected_value: Some(json!({"fact": "observed"})),
                    subject_sha256: "a".repeat(64),
                    operation: "inspect_artifact".to_owned(),
                    occurrence_id: None,
                    pair_id: None,
                    pair_role: PairRole::None,
                    must_abstain: false,
                },
                OracleTask {
                    task_id: "leaked".to_owned(),
                    fault_family: "parametric".to_owned(),
                    expected_value: Some(json!({"fact": "remembered"})),
                    subject_sha256: "b".repeat(64),
                    operation: "inspect_artifact".to_owned(),
                    occurrence_id: None,
                    pair_id: None,
                    pair_role: PairRole::None,
                    must_abstain: false,
                },
                OracleTask {
                    task_id: "false-supported".to_owned(),
                    fault_family: "false-positive".to_owned(),
                    expected_value: Some(json!({"fact": "real"})),
                    subject_sha256: "c".repeat(64),
                    operation: "inspect_artifact".to_owned(),
                    occurrence_id: None,
                    pair_id: None,
                    pair_role: PairRole::None,
                    must_abstain: false,
                },
                OracleTask {
                    task_id: "unsupported".to_owned(),
                    fault_family: "unsupported".to_owned(),
                    expected_value: None,
                    subject_sha256: "d".repeat(64),
                    operation: "inspect_artifact".to_owned(),
                    occurrence_id: None,
                    pair_id: None,
                    pair_role: PairRole::None,
                    must_abstain: true,
                },
            ],
        };
        let submissions = SubmissionSet {
            schema: SUBMISSIONS_SCHEMA.to_owned(),
            run_id: "run-1".to_owned(),
            agent_id: "agent-1".to_owned(),
            submissions: Vec::new(),
        };
        let assessment = |task_id: &str,
                          fault_family: &str,
                          answered: bool,
                          raw_correct: bool,
                          grounded_correct: bool,
                          positive_claim: bool,
                          false_supported: bool,
                          correct_unsupported_abstention: bool,
                          replay_state: ReplayState| TaskAssessment {
            task_id: task_id.to_owned(),
            fault_family: fault_family.to_owned(),
            answered,
            raw_correct,
            grounded_correct,
            positive_claim,
            false_supported,
            correct_unsupported_abstention,
            exact_scope_match: grounded_correct,
            exact_occurrence_match: grounded_correct,
            replay_state,
            replay_error: None,
        };
        let evaluated = vec![
            EvaluatedTask {
                oracle: oracle.tasks[0].clone(),
                assessment: assessment(
                    "grounded", "positive", true, true, true, true, false, false,
                    ReplayState::MatchSupported,
                ),
                answer: Some(json!({"fact": "observed"})),
            },
            EvaluatedTask {
                oracle: oracle.tasks[1].clone(),
                assessment: assessment(
                    "leaked", "parametric", true, true, false, false, false, false,
                    ReplayState::NotSubmitted,
                ),
                answer: Some(json!({"fact": "remembered"})),
            },
            EvaluatedTask {
                oracle: oracle.tasks[2].clone(),
                assessment: assessment(
                    "false-supported", "false-positive", true, false, false, true, true, false,
                    ReplayState::MatchSupported,
                ),
                answer: Some(json!({"fact": "invented"})),
            },
            EvaluatedTask {
                oracle: oracle.tasks[3].clone(),
                assessment: assessment(
                    "unsupported", "unsupported", false, false, false, false, false, true,
                    ReplayState::NotSubmitted,
                ),
                answer: None,
            },
        ];

        let result = build_scorecard(&oracle, &submissions, evaluated).unwrap();
        let metrics = result.metrics;
        assert_eq!(metrics.tasks, 4);
        assert_eq!(metrics.answered, 3);
        assert_eq!(metrics.raw_correct, 2);
        assert_eq!(metrics.grounded_correct, 1);
        assert_eq!(metrics.correct_but_ungrounded, 1);
        assert_eq!(metrics.parametric_leakage_rate, Some(0.5));
        assert_eq!(metrics.false_supported_claims, 1);
        assert_eq!(metrics.false_supported_rate, Some(0.5));
        assert_eq!(metrics.unsupported_controls, 1);
        assert_eq!(metrics.correct_unsupported_abstentions, 1);
        assert_eq!(metrics.correct_unsupported_abstention_rate, Some(1.0));
        assert_eq!(metrics.raw_accuracy_full_suite, Some(0.5));
        assert_eq!(metrics.raw_accuracy_when_answered, Some(2.0 / 3.0));
        assert_eq!(metrics.answer_coverage, Some(0.75));
        assert_eq!(metrics.grounded_accuracy_full_suite, Some(0.25));
    }

    #[test]
    fn oracle_pair_contract_rejects_identical_relevant_mutation_values() {
        let oracle = OracleCorpus {
            schema: ORACLE_SCHEMA.to_owned(),
            corpus_id: "test".to_owned(),
            corpus_version: "v0".to_owned(),
            tasks: vec![
                OracleTask {
                    task_id: "base".to_owned(),
                    fault_family: "mutation".to_owned(),
                    expected_value: Some(json!({"version": 1})),
                    subject_sha256: "a".repeat(64),
                    operation: "inspect_artifact".to_owned(),
                    occurrence_id: None,
                    pair_id: Some("pair".to_owned()),
                    pair_role: PairRole::RelevantBase,
                    must_abstain: false,
                },
                OracleTask {
                    task_id: "variant".to_owned(),
                    fault_family: "mutation".to_owned(),
                    expected_value: Some(json!({"version": 1})),
                    subject_sha256: "b".repeat(64),
                    operation: "inspect_artifact".to_owned(),
                    occurrence_id: None,
                    pair_id: Some("pair".to_owned()),
                    pair_role: PairRole::RelevantVariant,
                    must_abstain: false,
                },
            ],
        };
        assert_eq!(
            validate_oracle(&oracle).unwrap_err().kind,
            ScorecardErrorKind::PairContract
        );
    }
}
