//! Offline, oracle-separated scorecard for evidence-grounded reverse-engineering agents.
//!
//! The scorer receives hidden oracle data separately from the agent submission.
//! It never trusts an agent-supplied replay flag: raw claim/result/Evidence bytes
//! are sent to the independent REA replay implementation in this crate's
//! dependency graph. A provider-reported observation earns benchmark grounding
//! credit only when the answer is correct, the exact task scope is bound, and
//! independent replay returns SupportedWithinScope.

use rea_claim_replay_verifier::{
    Disposition, MAX_EVIDENCE_RECORDS, ReplayOutcome, replay_claim_evaluation,
};
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize as SerdeDeserialize, Serialize};
use serde_json::{Map, Number, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use symthaea_rea_compat::{canonical_jcs_bytes, sha256_hex};

pub const ORACLE_SCHEMA: &str = "luminous.rea.agent-oracle.v0";
pub const SUBMISSION_SCHEMA: &str = "luminous.rea.agent-submission.v0";
pub const SCORECARD_SCHEMA: &str = "luminous.rea.agent-scorecard.v0";
pub const MAX_ORACLE_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_SUBMISSION_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_TASKS: usize = 10_000;
pub const MAX_EVIDENCE_RECORDS_PER_TASK: usize = MAX_EVIDENCE_RECORDS;
pub const MAX_EMBEDDED_EVIDENCE_BYTES_PER_TASK: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScorecardErrorKind {
    InputTooLarge,
    InvalidJson,
    InvalidOracle,
    InvalidSubmission,
    Canonicalization,
    ResourceLimit,
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
    limit: usize,
    context: &str,
) -> Result<Value, ScorecardError> {
    if bytes.len() > limit {
        return Err(ScorecardError::new(
            ScorecardErrorKind::InputTooLarge,
            format!("{context} is {} bytes; limit is {limit}", bytes.len()),
        ));
    }
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = UniqueValue::deserialize(&mut deserializer).map_err(|error| {
        ScorecardError::new(
            ScorecardErrorKind::InvalidJson,
            format!("{context} parsing failed: {error}"),
        )
    })?;
    deserializer.end().map_err(|error| {
        ScorecardError::new(
            ScorecardErrorKind::InvalidJson,
            format!("{context} has trailing data: {error}"),
        )
    })?;
    Ok(value.0)
}

fn require_keys(value: &Value, keys: &[&str], context: &str) -> Result<(), ScorecardError> {
    let object = value.as_object().ok_or_else(|| {
        ScorecardError::new(
            ScorecardErrorKind::InvalidJson,
            format!("{context} must be a JSON object"),
        )
    })?;
    for key in keys {
        if !object.contains_key(*key) {
            return Err(ScorecardError::new(
                ScorecardErrorKind::InvalidJson,
                format!("{context}.{key} is required"),
            ));
        }
    }
    Ok(())
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn jcs_equal(left: &Value, right: &Value) -> Result<bool, ScorecardError> {
    let left = canonical_jcs_bytes(left).map_err(|error| {
        ScorecardError::new(
            ScorecardErrorKind::Canonicalization,
            format!("left JSON value could not be canonicalized: {error}"),
        )
    })?;
    let right = canonical_jcs_bytes(right).map_err(|error| {
        ScorecardError::new(
            ScorecardErrorKind::Canonicalization,
            format!("right JSON value could not be canonicalized: {error}"),
        )
    })?;
    Ok(left == right)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct OracleScope {
    pub subject_sha256: String,
    pub operation: String,
    pub platform: Option<String>,
    pub platform_version: Option<String>,
    pub environment_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, SerdeDeserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairKind {
    RelevantMutation,
    IrrelevantMutation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, SerdeDeserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairRole {
    Baseline,
    Mutated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct PairBinding {
    pub pair_id: String,
    pub kind: PairKind,
    pub role: PairRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct OccurrenceExpectation {
    pub index_json_pointer: String,
    pub occurrence_index: u64,
    pub raw_name_sha256_json_pointer: Option<String>,
    pub raw_name_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct OracleTask {
    pub task_id: String,
    pub artifact_sha256: String,
    pub expected_scope: OracleScope,
    pub expected_answer: Value,
    pub fault_family: String,
    pub pair: Option<PairBinding>,
    pub occurrence: Option<OccurrenceExpectation>,
}

#[derive(Debug, Clone, PartialEq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct OracleManifest {
    pub schema: String,
    pub suite_id: String,
    pub tasks: Vec<OracleTask>,
}

#[derive(Debug, Clone, PartialEq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentSubmission {
    pub task_id: String,
    pub abstained: bool,
    pub answer: Value,
    pub claim_json: Option<String>,
    pub result_json: Option<String>,
    pub evidence_json: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
pub struct SubmissionManifest {
    pub schema: String,
    pub suite_id: String,
    pub submissions: Vec<AgentSubmission>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayStatus {
    NotAttempted,
    Match,
    Mismatch,
    Invalid,
}

impl ReplayStatus {
    fn as_str(&self) -> &'static str {
        match self {
            Self::NotAttempted => "not_attempted",
            Self::Match => "match",
            Self::Mismatch => "mismatch",
            Self::Invalid => "invalid",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TaskScore {
    pub task_id: String,
    pub fault_family: String,
    pub submitted: bool,
    pub answered: bool,
    pub abstained: bool,
    pub raw_correct: bool,
    pub claimed_supported: bool,
    pub replay_status: String,
    pub claim_scope_exact: bool,
    pub occurrence_exact: bool,
    pub evidence_grounded: bool,
    pub false_supported: bool,
    /// SHA-256 of JCS-canonical answer, allowing pair comparisons without
    /// copying answer contents into the scorecard receipt.
    pub answer_commitment_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Counts {
    pub eligible: usize,
    pub submitted: usize,
    pub answered: usize,
    pub abstained: usize,
    pub raw_correct: usize,
    pub correct_but_ungrounded: usize,
    pub evidence_grounded_correct: usize,
    pub claimed_supported: usize,
    pub false_supported: usize,
    pub replay_attempted: usize,
    pub replay_matches: usize,
    pub replay_mismatches: usize,
    pub replay_invalid: usize,
}

impl Counts {
    fn new(eligible: usize) -> Self {
        Self {
            eligible,
            submitted: 0,
            answered: 0,
            abstained: 0,
            raw_correct: 0,
            correct_but_ungrounded: 0,
            evidence_grounded_correct: 0,
            claimed_supported: 0,
            false_supported: 0,
            replay_attempted: 0,
            replay_matches: 0,
            replay_mismatches: 0,
            replay_invalid: 0,
        }
    }

    fn add(&mut self, score: &TaskScore) {
        if score.submitted {
            self.submitted += 1;
        }
        if score.answered {
            self.answered += 1;
        }
        if score.abstained {
            self.abstained += 1;
        }
        if score.raw_correct {
            self.raw_correct += 1;
        }
        if score.claimed_supported {
            self.claimed_supported += 1;
        }
        if score.false_supported {
            self.false_supported += 1;
        }
        if score.evidence_grounded {
            self.evidence_grounded_correct += 1;
        }
        if score.raw_correct && !score.evidence_grounded {
            self.correct_but_ungrounded += 1;
        }
        match score.replay_status.as_str() {
            "match" => {
                self.replay_attempted += 1;
                self.replay_matches += 1;
            }
            "mismatch" => {
                self.replay_attempted += 1;
                self.replay_mismatches += 1;
            }
            "invalid" => {
                self.replay_attempted += 1;
                self.replay_invalid += 1;
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Ratio {
    pub numerator: usize,
    pub denominator: usize,
    pub rate: Option<f64>,
}

fn answer_commitment(value: &Value) -> Result<String, ScorecardError> {
    let canonical = canonical_jcs_bytes(value).map_err(|error| {
        ScorecardError::new(
            ScorecardErrorKind::Canonicalization,
            format!("agent answer canonicalization failed: {error}"),
        )
    })?;
    Ok(sha256_hex(&canonical))
}

fn ratio(numerator: usize, denominator: usize) -> Ratio {
    Ratio {
        numerator,
        denominator,
        rate: (denominator > 0).then_some(numerator as f64 / denominator as f64),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MetricSet {
    pub counts: Counts,
    pub answer_coverage: Ratio,
    pub abstention_rate: Ratio,
    pub raw_full_suite_accuracy: Ratio,
    pub conditional_answer_accuracy: Ratio,
    pub evidence_grounded_full_suite_accuracy: Ratio,
    pub parametric_leakage_rate: Ratio,
    pub false_supported_rate: Ratio,
    pub replay_coverage: Ratio,
    pub replay_match_rate: Ratio,
}

fn metrics(counts: Counts) -> MetricSet {
    MetricSet {
        answer_coverage: ratio(counts.answered, counts.eligible),
        abstention_rate: ratio(counts.abstained, counts.eligible),
        raw_full_suite_accuracy: ratio(counts.raw_correct, counts.eligible),
        conditional_answer_accuracy: ratio(counts.raw_correct, counts.answered),
        evidence_grounded_full_suite_accuracy: ratio(
            counts.evidence_grounded_correct,
            counts.eligible,
        ),
        parametric_leakage_rate: ratio(
            counts.correct_but_ungrounded,
            counts.raw_correct,
        ),
        false_supported_rate: ratio(counts.false_supported, counts.claimed_supported),
        replay_coverage: ratio(counts.replay_attempted, counts.eligible),
        replay_match_rate: ratio(counts.replay_matches, counts.replay_attempted),
        counts,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PairScore {
    pub pair_id: String,
    pub kind: String,
    pub baseline_task_id: String,
    pub mutated_task_id: String,
    pub answer_relation_satisfied: bool,
    pub both_evidence_grounded: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PairMetrics {
    pub relevant_mutation_pairs: usize,
    pub relevant_mutation_sensitive: usize,
    pub relevant_mutation_sensitivity: Ratio,
    pub relevant_mutation_both_grounded: usize,
    pub irrelevant_mutation_pairs: usize,
    pub irrelevant_mutation_invariant: usize,
    pub irrelevant_mutation_invariance: Ratio,
    pub irrelevant_mutation_both_grounded: usize,
    pub pair_details: Vec<PairScore>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Scorecard {
    pub schema: String,
    pub suite_id: String,
    pub task_count: usize,
    pub overall: MetricSet,
    pub by_fault_family: BTreeMap<String, MetricSet>,
    pub pairs: PairMetrics,
    pub task_results: Vec<TaskScore>,
}

#[derive(Debug, Clone, PartialEq)]
struct EvaluationState {
    raw_correct: bool,
    grounded: bool,
    replay_status: ReplayStatus,
}

fn decode_pointer(pointer: &str) -> Result<Vec<String>, ScorecardError> {
    if pointer.is_empty() {
        return Ok(Vec::new());
    }
    if !pointer.starts_with('/') {
        return Err(ScorecardError::new(
            ScorecardErrorKind::InvalidOracle,
            format!("JSON Pointer '{pointer}' must be empty or begin with '/'"),
        ));
    }
    pointer
        .split('/')
        .skip(1)
        .map(|token| {
            let mut decoded = String::new();
            let mut chars = token.chars();
            while let Some(ch) = chars.next() {
                if ch == '~' {
                    match chars.next() {
                        Some('0') => decoded.push('~'),
                        Some('1') => decoded.push('/'),
                        _ => {
                            return Err(ScorecardError::new(
                                ScorecardErrorKind::InvalidOracle,
                                format!("JSON Pointer '{pointer}' has an invalid escape"),
                            ));
                        }
                    }
                } else {
                    decoded.push(ch);
                }
            }
            Ok(decoded)
        })
        .collect()
}

fn pointer_value<'a>(value: &'a Value, pointer: &str) -> Option<&'a Value> {
    let tokens = decode_pointer(pointer).ok()?;
    let mut current = value;
    for token in tokens {
        current = match current {
            Value::Object(object) => object.get(&token)?,
            Value::Array(array) => {
                if token.is_empty() || (token.len() > 1 && token.starts_with('0')) {
                    return None;
                }
                let index = token.parse::<usize>().ok()?;
                array.get(index)?
            }
            _ => return None,
        };
    }
    Some(current)
}

fn validate_occurrence(task: &OracleTask) -> Result<(), ScorecardError> {
    let Some(occurrence) = &task.occurrence else {
        return Ok(());
    };
    let index = pointer_value(&task.expected_answer, &occurrence.index_json_pointer);
    if index != Some(&Value::from(occurrence.occurrence_index)) {
        return Err(ScorecardError::new(
            ScorecardErrorKind::InvalidOracle,
            format!(
                "task '{}' occurrence index pointer does not match occurrence_index",
                task.task_id
            ),
        ));
    }
    match (
        occurrence.raw_name_sha256_json_pointer.as_deref(),
        occurrence.raw_name_sha256.as_deref(),
    ) {
        (Some(pointer), Some(expected)) if valid_sha256(expected) => {
            if pointer_value(&task.expected_answer, pointer)
                != Some(&Value::String(expected.to_owned()))
            {
                return Err(ScorecardError::new(
                    ScorecardErrorKind::InvalidOracle,
                    format!(
                        "task '{}' raw-name digest pointer does not match oracle",
                        task.task_id
                    ),
                ));
            }
        }
        (None, None) => {}
        _ => {
            return Err(ScorecardError::new(
                ScorecardErrorKind::InvalidOracle,
                format!(
                    "task '{}' raw-name digest pointer/value must be both present or both null",
                    task.task_id
                ),
            ));
        }
    }
    Ok(())
}

fn validate_oracle(oracle: &OracleManifest) -> Result<(), ScorecardError> {
    let invalid = |message: String| ScorecardError::new(ScorecardErrorKind::InvalidOracle, message);
    if oracle.schema != ORACLE_SCHEMA || !nonempty(&oracle.suite_id) {
        return Err(invalid("unsupported oracle schema or empty suite_id".to_owned()));
    }
    if oracle.tasks.is_empty() || oracle.tasks.len() > MAX_TASKS {
        return Err(ScorecardError::new(
            ScorecardErrorKind::ResourceLimit,
            format!("oracle task count must be in 1..={MAX_TASKS}"),
        ));
    }
    let mut task_ids = HashSet::new();
    let mut pairs = BTreeMap::<String, Vec<&OracleTask>>::new();

    for task in &oracle.tasks {
        if !nonempty(&task.task_id) || !task_ids.insert(task.task_id.as_str()) {
            return Err(invalid("task IDs must be non-empty and unique".to_owned()));
        }
        if !valid_sha256(&task.artifact_sha256)
            || task.expected_scope.subject_sha256 != task.artifact_sha256
            || !nonempty(&task.expected_scope.operation)
            || !nonempty(&task.fault_family)
        {
            return Err(invalid(format!(
                "task '{}' has invalid artifact/scope/fault-family metadata",
                task.task_id
            )));
        }
        for (label, value) in [
            ("platform", task.expected_scope.platform.as_deref()),
            (
                "platform_version",
                task.expected_scope.platform_version.as_deref(),
            ),
            (
                "environment_id",
                task.expected_scope.environment_id.as_deref(),
            ),
        ] {
            if value.is_some_and(|item| !nonempty(item)) {
                return Err(invalid(format!(
                    "task '{}' has empty {label} value",
                    task.task_id
                )));
            }
        }
        validate_occurrence(task)?;
        if let Some(pair) = &task.pair {
            if !nonempty(&pair.pair_id) {
                return Err(invalid(format!("task '{}' has empty pair_id", task.task_id)));
            }
            pairs.entry(pair.pair_id.clone()).or_default().push(task);
        }
    }

    for (pair_id, tasks) in pairs {
        if tasks.len() != 2 {
            return Err(invalid(format!(
                "pair '{pair_id}' must contain exactly two tasks"
            )));
        }
        let baseline = tasks
            .iter()
            .find(|task| task.pair.as_ref().is_some_and(|pair| pair.role == PairRole::Baseline))
            .copied()
            .ok_or_else(|| invalid(format!("pair '{pair_id}' lacks baseline task")))?;
        let mutated = tasks
            .iter()
            .find(|task| task.pair.as_ref().is_some_and(|pair| pair.role == PairRole::Mutated))
            .copied()
            .ok_or_else(|| invalid(format!("pair '{pair_id}' lacks mutated task")))?;
        let baseline_pair = baseline.pair.as_ref().expect("pair was grouped");
        let mutated_pair = mutated.pair.as_ref().expect("pair was grouped");
        if baseline_pair.kind != mutated_pair.kind
            || baseline.artifact_sha256 == mutated.artifact_sha256
        {
            return Err(invalid(format!(
                "pair '{pair_id}' must have one kind and distinct artifact digests"
            )));
        }
        let same_answer = jcs_equal(&baseline.expected_answer, &mutated.expected_answer)
            .map_err(|error| invalid(error.message))?;
        match baseline_pair.kind {
            PairKind::RelevantMutation if same_answer => {
                return Err(invalid(format!(
                    "relevant-mutation pair '{pair_id}' must change expected answer"
                )));
            }
            PairKind::IrrelevantMutation if !same_answer => {
                return Err(invalid(format!(
                    "irrelevant-mutation pair '{pair_id}' must preserve expected answer"
                )));
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_submission_manifest(
    submission: &SubmissionManifest,
    oracle: &OracleManifest,
) -> Result<(), ScorecardError> {
    let invalid = |message: String| {
        ScorecardError::new(ScorecardErrorKind::InvalidSubmission, message)
    };
    if submission.schema != SUBMISSION_SCHEMA || submission.suite_id != oracle.suite_id {
        return Err(invalid("submission schema or suite_id does not match oracle".to_owned()));
    }
    if submission.submissions.len() > oracle.tasks.len() {
        return Err(ScorecardError::new(
            ScorecardErrorKind::ResourceLimit,
            "submission has more task records than the oracle",
        ));
    }
    let known: HashSet<&str> = oracle.tasks.iter().map(|task| task.task_id.as_str()).collect();
    let mut seen = HashSet::new();
    for item in &submission.submissions {
        if !known.contains(item.task_id.as_str()) {
            return Err(invalid(format!("unknown task_id '{}'", item.task_id)));
        }
        if !seen.insert(item.task_id.as_str()) {
            return Err(invalid(format!("duplicate submission for '{}'", item.task_id)));
        }
        if item.abstained && item.answer != Value::Null {
            return Err(invalid(format!(
                "abstaining submission '{}' must set answer to null",
                item.task_id
            )));
        }
        if item.evidence_json.len() > MAX_EVIDENCE_RECORDS_PER_TASK {
            return Err(ScorecardError::new(
                ScorecardErrorKind::ResourceLimit,
                format!("task '{}' exceeds Evidence record count limit", item.task_id),
            ));
        }
        let evidence_bytes = item.evidence_json.iter().map(String::len).sum::<usize>();
        if evidence_bytes > MAX_EMBEDDED_EVIDENCE_BYTES_PER_TASK {
            return Err(ScorecardError::new(
                ScorecardErrorKind::ResourceLimit,
                format!("task '{}' exceeds embedded Evidence byte limit", item.task_id),
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
struct ClaimScopeView {
    subject_sha256: String,
    operation: String,
    platform: Option<String>,
    platform_version: Option<String>,
    environment_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, SerdeDeserialize)]
#[serde(deny_unknown_fields)]
struct ClaimView {
    schema: String,
    claim_id: String,
    predicate: String,
    scope: ClaimScopeView,
    asserted_value: Value,
    source_predicate_type: Option<String>,
    evidence_ids: Vec<String>,
    assumptions: Vec<String>,
    falsification_condition: String,
}

fn exact_claim_binding(
    claim_bytes: &[u8],
    task: &OracleTask,
) -> Result<(bool, bool), ScorecardError> {
    let value = parse_unique_json(claim_bytes, 1024 * 1024, "agent claim")?;
    require_keys(
        &value,
        &[
            "schema",
            "claim_id",
            "predicate",
            "scope",
            "asserted_value",
            "source_predicate_type",
            "evidence_ids",
            "assumptions",
            "falsification_condition",
        ],
        "claim",
    )?;
    require_keys(
        value.get("scope").unwrap_or(&Value::Null),
        &[
            "subject_sha256",
            "operation",
            "platform",
            "platform_version",
            "environment_id",
        ],
        "claim.scope",
    )?;
    let claim: ClaimView = serde_json::from_value(value).map_err(|error| {
        ScorecardError::new(
            ScorecardErrorKind::InvalidSubmission,
            format!("claim schema mismatch: {error}"),
        )
    })?;
    let scope_exact = claim.schema == "luminous.rea.claim.v0"
        && claim.predicate == "provider_reported_observation"
        && claim.scope.subject_sha256 == task.expected_scope.subject_sha256
        && claim.scope.operation == task.expected_scope.operation
        && claim.scope.platform == task.expected_scope.platform
        && claim.scope.platform_version == task.expected_scope.platform_version
        && claim.scope.environment_id == task.expected_scope.environment_id;
    let asserted_exact = jcs_equal(&claim.asserted_value, &task.expected_answer)
        .map_err(|error| {
            ScorecardError::new(ScorecardErrorKind::Canonicalization, error.message)
        })?;
    let occurrence_exact = task
        .occurrence
        .as_ref()
        .is_none_or(|occurrence| {
            pointer_value(&claim.asserted_value, &occurrence.index_json_pointer)
                == Some(&Value::from(occurrence.occurrence_index))
                && match (
                    occurrence.raw_name_sha256_json_pointer.as_deref(),
                    occurrence.raw_name_sha256.as_deref(),
                ) {
                    (Some(pointer), Some(expected)) => {
                        pointer_value(&claim.asserted_value, pointer)
                            == Some(&Value::String(expected.to_owned()))
                    }
                    (None, None) => true,
                    _ => false,
                }
        });
    Ok((scope_exact && asserted_exact, occurrence_exact))
}

fn reported_supported(result_bytes: &[u8]) -> bool {
    parse_unique_json(result_bytes, 4 * 1024 * 1024, "agent result")
        .ok()
        .and_then(|value| value.get("disposition").and_then(Value::as_str).map(str::to_owned))
        .is_some_and(|value| value == "supported_within_scope")
}

fn evaluate_submission(
    task: &OracleTask,
    submission: Option<&AgentSubmission>,
) -> Result<TaskScore, ScorecardError> {
    let Some(submission) = submission else {
        return Ok(TaskScore {
            task_id: task.task_id.clone(),
            fault_family: task.fault_family.clone(),
            submitted: false,
            answered: false,
            abstained: false,
            raw_correct: false,
            claimed_supported: false,
            replay_status: ReplayStatus::NotAttempted.as_str().to_owned(),
            claim_scope_exact: false,
            occurrence_exact: false,
            evidence_grounded: false,
            false_supported: false,
            answer_commitment_sha256: None,
        });
    };

    let answered = !submission.abstained;
    let raw_correct = answered
        && jcs_equal(&submission.answer, &task.expected_answer).map_err(|error| {
            ScorecardError::new(ScorecardErrorKind::Canonicalization, error.message)
        })?;
    let answer_commitment_sha256 = if answered {
        Some(answer_commitment(&submission.answer)?)
    } else {
        None
    };
    let mut claimed_supported = false;
    let mut replay_status = ReplayStatus::NotAttempted;
    let mut claim_scope_exact = false;
    let mut occurrence_exact = false;
    let mut evidence_grounded = false;

    let has_replay_material = submission.claim_json.is_some()
        || submission.result_json.is_some()
        || !submission.evidence_json.is_empty();
    if has_replay_material {
        claimed_supported = submission
            .result_json
            .as_deref()
            .is_some_and(|bytes| reported_supported(bytes.as_bytes()));

        let (Some(claim_json), Some(result_json)) = (
            submission.claim_json.as_deref(),
            submission.result_json.as_deref(),
        ) else {
            replay_status = ReplayStatus::Invalid;
            return Ok(TaskScore {
                task_id: task.task_id.clone(),
                fault_family: task.fault_family.clone(),
                submitted: true,
                answered,
                abstained: submission.abstained,
                raw_correct,
                claimed_supported,
                replay_status: replay_status.as_str().to_owned(),
                claim_scope_exact,
                occurrence_exact,
                evidence_grounded,
                false_supported: claimed_supported,
                answer_commitment_sha256,
            });
        };

        let (scope_and_assertion_exact, occurrence_matches) =
            match exact_claim_binding(claim_json.as_bytes(), task) {
                Ok(result) => result,
                Err(_) => {
                    replay_status = ReplayStatus::Invalid;
                    return Ok(TaskScore {
                        task_id: task.task_id.clone(),
                        fault_family: task.fault_family.clone(),
                        submitted: true,
                        answered,
                        abstained: submission.abstained,
                        raw_correct,
                        claimed_supported,
                        replay_status: replay_status.as_str().to_owned(),
                        claim_scope_exact: false,
                        occurrence_exact: false,
                        evidence_grounded: false,
                        false_supported: claimed_supported,
                        answer_commitment_sha256,
                    });
                }
            };
        claim_scope_exact = scope_and_assertion_exact;
        occurrence_exact = occurrence_matches;

        let evidence_refs: Vec<&[u8]> = submission
            .evidence_json
            .iter()
            .map(|value| value.as_bytes())
            .collect();
        match replay_claim_evaluation(
            claim_json.as_bytes(),
            &evidence_refs,
            result_json.as_bytes(),
        ) {
            Ok(receipt) if receipt.outcome == ReplayOutcome::ReplayMatch => {
                replay_status = ReplayStatus::Match;
                evidence_grounded = raw_correct
                    && claim_scope_exact
                    && occurrence_exact
                    && receipt.expected_disposition == Disposition::SupportedWithinScope;
            }
            Ok(receipt) if receipt.outcome == ReplayOutcome::ReplayMismatch => {
                replay_status = ReplayStatus::Mismatch;
            }
            Ok(_) => {
                replay_status = ReplayStatus::Invalid;
            }
            Err(_) => {
                replay_status = ReplayStatus::Invalid;
            }
        }
    }

    Ok(TaskScore {
        task_id: task.task_id.clone(),
        fault_family: task.fault_family.clone(),
        submitted: true,
        answered,
        abstained: submission.abstained,
        raw_correct,
        claimed_supported,
        replay_status: replay_status.as_str().to_owned(),
        claim_scope_exact,
        occurrence_exact,
        evidence_grounded,
        false_supported: claimed_supported && !evidence_grounded,
        answer_commitment_sha256,
    })
}

fn build_pair_metrics(
    oracle: &OracleManifest,
    results: &HashMap<String, TaskScore>,
) -> Result<PairMetrics, ScorecardError> {
    let mut grouped = BTreeMap::<String, Vec<&OracleTask>>::new();
    for task in &oracle.tasks {
        if let Some(pair) = &task.pair {
            grouped.entry(pair.pair_id.clone()).or_default().push(task);
        }
    }

    let mut scores = Vec::new();
    let mut relevant_total = 0usize;
    let mut relevant_sensitive = 0usize;
    let mut relevant_grounded = 0usize;
    let mut irrelevant_total = 0usize;
    let mut irrelevant_invariant = 0usize;
    let mut irrelevant_grounded = 0usize;

    for (pair_id, tasks) in grouped {
        let baseline = tasks
            .iter()
            .find(|task| task.pair.as_ref().is_some_and(|pair| pair.role == PairRole::Baseline))
            .copied()
            .expect("validated pair has baseline");
        let mutated = tasks
            .iter()
            .find(|task| task.pair.as_ref().is_some_and(|pair| pair.role == PairRole::Mutated))
            .copied()
            .expect("validated pair has mutated");
        let baseline_score = results.get(&baseline.task_id).expect("task score exists");
        let mutated_score = results.get(&mutated.task_id).expect("task score exists");
        let baseline_submission_answered = baseline_score.answered;
        let mutated_submission_answered = mutated_score.answered;

        let relation_satisfied = match baseline.pair.as_ref().expect("pair exists").kind {
            PairKind::RelevantMutation => {
                baseline_submission_answered
                    && mutated_submission_answered
                    && baseline_score.raw_correct
                    && mutated_score.raw_correct
                    && !jcs_equal(&baseline.expected_answer, &mutated.expected_answer)?
                    && baseline_score.answer_commitment_sha256
                        != mutated_score.answer_commitment_sha256
            }
            PairKind::IrrelevantMutation => {
                baseline_submission_answered
                    && mutated_submission_answered
                    && baseline_score.raw_correct
                    && mutated_score.raw_correct
                    && jcs_equal(&baseline.expected_answer, &mutated.expected_answer)?
                    && baseline_score.answer_commitment_sha256
                        == mutated_score.answer_commitment_sha256
            }
        };
        let both_grounded =
            baseline_score.evidence_grounded && mutated_score.evidence_grounded;
        let kind = baseline.pair.as_ref().expect("pair exists").kind;
        match kind {
            PairKind::RelevantMutation => {
                relevant_total += 1;
                if relation_satisfied {
                    relevant_sensitive += 1;
                }
                if both_grounded {
                    relevant_grounded += 1;
                }
            }
            PairKind::IrrelevantMutation => {
                irrelevant_total += 1;
                if relation_satisfied {
                    irrelevant_invariant += 1;
                }
                if both_grounded {
                    irrelevant_grounded += 1;
                }
            }
        }
        scores.push(PairScore {
            pair_id,
            kind: match kind {
                PairKind::RelevantMutation => "relevant_mutation".to_owned(),
                PairKind::IrrelevantMutation => "irrelevant_mutation".to_owned(),
            },
            baseline_task_id: baseline.task_id.clone(),
            mutated_task_id: mutated.task_id.clone(),
            answer_relation_satisfied: relation_satisfied,
            both_evidence_grounded: both_grounded,
        });
    }

    Ok(PairMetrics {
        relevant_mutation_pairs: relevant_total,
        relevant_mutation_sensitive: relevant_sensitive,
        relevant_mutation_sensitivity: ratio(relevant_sensitive, relevant_total),
        relevant_mutation_both_grounded: relevant_grounded,
        irrelevant_mutation_pairs: irrelevant_total,
        irrelevant_mutation_invariant: irrelevant_invariant,
        irrelevant_mutation_invariance: ratio(irrelevant_invariant, irrelevant_total),
        irrelevant_mutation_both_grounded: irrelevant_grounded,
        pair_details: scores,
    })
}

pub fn evaluate_scorecard(
    oracle_json: &[u8],
    submission_json: &[u8],
) -> Result<Scorecard, ScorecardError> {
    let oracle_value = parse_unique_json(oracle_json, MAX_ORACLE_BYTES, "oracle manifest")?;
    let submission_value =
        parse_unique_json(submission_json, MAX_SUBMISSION_BYTES, "submission manifest")?;

    require_keys(&oracle_value, &["schema", "suite_id", "tasks"], "oracle")?;
    require_keys(
        &submission_value,
        &["schema", "suite_id", "submissions"],
        "submission",
    )?;

    let oracle_tasks = oracle_value
        .get("tasks")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ScorecardError::new(
                ScorecardErrorKind::InvalidOracle,
                "oracle.tasks must be an array",
            )
        })?;
    for (index, task) in oracle_tasks.iter().enumerate() {
        let context = format!("oracle.tasks[{index}]");
        require_keys(
            task,
            &[
                "task_id",
                "artifact_sha256",
                "expected_scope",
                "expected_answer",
                "fault_family",
                "pair",
                "occurrence",
            ],
            &context,
        )?;
        require_keys(
            task.get("expected_scope").unwrap_or(&Value::Null),
            &[
                "subject_sha256",
                "operation",
                "platform",
                "platform_version",
                "environment_id",
            ],
            &format!("{context}.expected_scope"),
        )?;
        if task.get("pair").is_some_and(|value| !value.is_null()) {
            require_keys(
                &task["pair"],
                &["pair_id", "kind", "role"],
                &format!("{context}.pair"),
            )?;
        }
        if task.get("occurrence").is_some_and(|value| !value.is_null()) {
            require_keys(
                &task["occurrence"],
                &[
                    "index_json_pointer",
                    "occurrence_index",
                    "raw_name_sha256_json_pointer",
                    "raw_name_sha256",
                ],
                &format!("{context}.occurrence"),
            )?;
        }
    }

    let agent_submissions = submission_value
        .get("submissions")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ScorecardError::new(
                ScorecardErrorKind::InvalidSubmission,
                "submission.submissions must be an array",
            )
        })?;
    for (index, item) in agent_submissions.iter().enumerate() {
        require_keys(
            item,
            &[
                "task_id",
                "abstained",
                "answer",
                "claim_json",
                "result_json",
                "evidence_json",
            ],
            &format!("submission.submissions[{index}]"),
        )?;
    }

    let oracle: OracleManifest = serde_json::from_value(oracle_value).map_err(|error| {
        ScorecardError::new(
            ScorecardErrorKind::InvalidOracle,
            format!("oracle schema validation failed: {error}"),
        )
    })?;
    let submission: SubmissionManifest =
        serde_json::from_value(submission_value).map_err(|error| {
            ScorecardError::new(
                ScorecardErrorKind::InvalidSubmission,
                format!("submission schema validation failed: {error}"),
            )
        })?;
    validate_oracle(&oracle)?;
    validate_submission_manifest(&submission, &oracle)?;

    let submissions: HashMap<&str, &AgentSubmission> = submission
        .submissions
        .iter()
        .map(|item| (item.task_id.as_str(), item))
        .collect();

    let mut overall = Counts::new(oracle.tasks.len());
    let mut by_fault_family = BTreeMap::<String, Counts>::new();
    let mut task_results = Vec::with_capacity(oracle.tasks.len());
    let mut result_map = HashMap::<String, TaskScore>::new();

    for task in &oracle.tasks {
        let score = evaluate_submission(task, submissions.get(task.task_id.as_str()).copied())?;
        overall.add(&score);
        let family_counts = by_fault_family
            .entry(task.fault_family.clone())
            .or_insert_with(|| Counts::new(0));
        family_counts.eligible += 1;
        family_counts.add(&score);
        result_map.insert(task.task_id.clone(), score.clone());
        task_results.push(score);
    }

    let pairs = build_pair_metrics(&oracle, &result_map)?;
    Ok(Scorecard {
        schema: SCORECARD_SCHEMA.to_owned(),
        suite_id: oracle.suite_id,
        task_count: oracle.tasks.len(),
        overall: metrics(overall),
        by_fault_family: by_fault_family
            .into_iter()
            .map(|(family, counts)| (family, metrics(counts)))
            .collect(),
        pairs,
        task_results,
    })
}
