#![forbid(unsafe_code)]

//! Semantic validator for benchmark run receipts.
//! Structural Draft 2020-12 validation is a separate CI step. A valid receipt
//! establishes accounting and integrity consistency, not answer correctness
//! or producer authenticity.

use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fmt;

pub const MAX_RECEIPT_JSON_BYTES: usize = 64 * 1024 * 1024;
pub const RECEIPT_SCHEMA_ID: &str = "luminous.benchmark-run-receipt.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptErrorKind {
    InvalidJson,
    InputTooLarge,
    SchemaMarkerMismatch,
    MissingField,
    InvalidType,
    InvalidDigest,
    AccountingMismatch,
    DuplicateTask,
    BudgetExceeded,
    InvalidOfficialScore,
    InvalidDerivedMetric,
    InvalidComparison,
    IntegrityMismatch,
    InvalidExclusion,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptValidationError {
    pub kind: ReceiptErrorKind,
    pub message: String,
}

impl ReceiptValidationError {
    fn new(kind: ReceiptErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl fmt::Display for ReceiptValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}
impl std::error::Error for ReceiptValidationError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedBenchmarkReceipt {
    pub run_id: String,
    pub raw_file_sha256: String,
    pub canonical_receipt_sha256: String,
    pub eligible: u64,
    pub attempted: u64,
    pub not_attempted: u64,
    pub graded: u64,
    pub passed: u64,
    pub authenticity_verified: bool,
}

fn err(kind: ReceiptErrorKind, message: impl Into<String>) -> ReceiptValidationError {
    ReceiptValidationError::new(kind, message)
}

fn get<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Value, ReceiptValidationError> {
    value.get(key).ok_or_else(|| err(
        ReceiptErrorKind::MissingField,
        format!("{context}.{key} is required"),
    ))
}

fn text<'a>(value: &'a Value, context: &str) -> Result<&'a str, ReceiptValidationError> {
    let s = value.as_str().ok_or_else(|| err(
        ReceiptErrorKind::InvalidType, format!("{context} must be a string"),
    ))?;
    if s.trim().is_empty() {
        return Err(err(ReceiptErrorKind::InvalidType, format!("{context} must not be empty")));
    }
    Ok(s)
}

fn integer(value: &Value, context: &str) -> Result<u64, ReceiptValidationError> {
    let n = value.as_u64().ok_or_else(|| err(
        ReceiptErrorKind::InvalidType, format!("{context} must be a non-negative integer"),
    ))?;
    if n > 9_007_199_254_740_991 {
        return Err(err(ReceiptErrorKind::InvalidType, format!("{context} exceeds the JavaScript safe-integer range")));
    }
    Ok(n)
}

fn number(value: &Value, context: &str) -> Result<f64, ReceiptValidationError> {
    let n = value.as_f64().ok_or_else(|| err(
        ReceiptErrorKind::InvalidType, format!("{context} must be numeric"),
    ))?;
    if !n.is_finite() {
        return Err(err(ReceiptErrorKind::InvalidType, format!("{context} must be finite")));
    }
    Ok(n)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn digest_field<'a>(value: &'a Value, context: &str) -> Result<&'a str, ReceiptValidationError> {
    let digest = text(value, context)?;
    if !is_sha256(digest) {
        return Err(err(ReceiptErrorKind::InvalidDigest, format!("{context} must be 64 lowercase hex characters")));
    }
    Ok(digest)
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(64);
    for b in Sha256::digest(bytes) {
        use fmt::Write as _;
        let _ = write!(output, "{b:02x}");
    }
    output
}

pub fn canonical_sha256(value: &Value) -> Result<String, ReceiptValidationError> {
    let bytes = serde_json_canonicalizer::to_vec(value)
        .map_err(|e| err(ReceiptErrorKind::InvalidJson, format!("JCS serialization failed: {e}")))?;
    Ok(sha256_hex(&bytes))
}

fn ratio_matches(value: f64, numerator: f64, denominator: f64) -> bool {
    denominator > 0.0
        && (value - numerator / denominator).abs() <= 1e-12 * (value.abs().max(1.0))
}

fn validate_accounting(receipt: &Value) -> Result<(u64, u64, u64, u64, u64), ReceiptValidationError> {
    let a = get(receipt, "task_accounting", "receipt")?;
    let eligible = integer(get(a, "eligible", "task_accounting")?, "eligible")?;
    let attempted = integer(get(a, "attempted", "task_accounting")?, "attempted")?;
    let not_attempted = integer(get(a, "not_attempted", "task_accounting")?, "not_attempted")?;
    let graded = integer(get(a, "graded", "task_accounting")?, "graded")?;
    if attempted.checked_add(not_attempted) != Some(eligible) {
        return Err(err(ReceiptErrorKind::AccountingMismatch, "eligible must equal attempted + not_attempted"));
    }

    let rows = get(receipt, "task_outcomes", "receipt")?.as_array()
        .ok_or_else(|| err(ReceiptErrorKind::InvalidType, "task_outcomes must be an array"))?;
    if rows.len() as u64 != eligible {
        return Err(err(ReceiptErrorKind::AccountingMismatch, "eligible count does not equal task row count"));
    }

    let disposition_names = ["passed", "failed", "refused", "abstained", "unsupported", "timed_out", "tool_failed", "ungraded"];
    let declared = get(a, "disposition_counts", "task_accounting")?;
    let mut declared_total = 0u64;
    for name in disposition_names {
        declared_total = declared_total.checked_add(integer(get(declared, name, "disposition_counts")?, name)?)
            .ok_or_else(|| err(ReceiptErrorKind::AccountingMismatch, "disposition sum overflow"))?;
    }
    if declared_total != attempted {
        return Err(err(ReceiptErrorKind::AccountingMismatch, "attempted count does not equal terminal disposition sum"));
    }

    let mut seen = HashSet::new();
    let mut actual: HashMap<&str, u64> = HashMap::new();
    let mut actual_attempted = 0u64;
    let mut actual_not_attempted = 0u64;
    let mut actual_graded = 0u64;
    let mut passed = 0u64;

    for (i, row) in rows.iter().enumerate() {
        let cx = format!("task_outcomes[{i}]");
        let id = text(get(row, "task_id", &cx)?, "task_id")?;
        if !seen.insert(id.to_owned()) {
            return Err(err(ReceiptErrorKind::DuplicateTask, format!("duplicate task id {id}")));
        }
        digest_field(get(row, "input_sha256", &cx)?, "task input_sha256")?;
        let attempted_flag = get(row, "attempted", &cx)?.as_bool()
            .ok_or_else(|| err(ReceiptErrorKind::InvalidType, "task attempted must be boolean"))?;
        let graded_flag = get(row, "graded", &cx)?.as_bool()
            .ok_or_else(|| err(ReceiptErrorKind::InvalidType, "task graded must be boolean"))?;
        let disposition = text(get(row, "disposition", &cx)?, "task disposition")?;
        let grader_result = get(row, "grader_result", &cx)?;

        if disposition == "not_attempted" {
            if attempted_flag || graded_flag || !grader_result.is_null() {
                return Err(err(ReceiptErrorKind::AccountingMismatch, format!("{cx}: not_attempted row must not be attempted or graded")));
            }
            actual_not_attempted += 1;
            continue;
        }
        if !attempted_flag {
            return Err(err(ReceiptErrorKind::AccountingMismatch, format!("{cx}: only not_attempted rows may have attempted=false")));
        }
        if !disposition_names.contains(&disposition) {
            return Err(err(ReceiptErrorKind::InvalidType, format!("{cx}: unknown terminal disposition")));
        }
        actual_attempted += 1;
        *actual.entry(disposition).or_insert(0) += 1;

        let should_be_graded = disposition == "passed" || disposition == "failed";
        if graded_flag != should_be_graded {
            return Err(err(ReceiptErrorKind::AccountingMismatch, format!("{cx}: only passed/failed rows are graded in receipt v1")));
        }
        if graded_flag {
            actual_graded += 1;
            if !grader_result.is_object() {
                return Err(err(ReceiptErrorKind::InvalidType, format!("{cx}: graded task requires a grader_result object")));
            }
            digest_field(get(grader_result, "result_sha256", "grader_result")?, "grader_result.result_sha256")?;
            number(get(grader_result, "grade", "grader_result")?, "grader_result.grade")?;
        } else if !grader_result.is_null() {
            return Err(err(ReceiptErrorKind::AccountingMismatch, format!("{cx}: ungraded row must have null grader_result")));
        }
        if disposition == "passed" {
            passed += 1;
        }
    }

    if actual_attempted != attempted || actual_not_attempted != not_attempted || actual_graded != graded {
        return Err(err(ReceiptErrorKind::AccountingMismatch, "task rows do not reconcile with accounting totals"));
    }
    for name in disposition_names {
        let expected = integer(get(declared, name, "disposition_counts")?, name)?;
        if actual.get(name).copied().unwrap_or(0) != expected {
            return Err(err(ReceiptErrorKind::AccountingMismatch, format!("disposition count {name} does not match task rows")));
        }
    }
    Ok((eligible, attempted, not_attempted, graded, passed))
}

fn validate_budgets(receipt: &Value) -> Result<(), ReceiptValidationError> {
    let caps = get(receipt, "budget_caps", "receipt")?;
    let usage = get(receipt, "actual_usage", "receipt")?;
    for (cap_key, usage_key) in [
        ("wall_time_ms", "wall_time_ms"),
        ("cpu_time_ms", "cpu_time_ms"),
        ("memory_bytes", "peak_memory_bytes"),
        ("tool_calls", "tool_calls"),
        ("model_tokens", "model_tokens"),
        ("cost_microusd", "cost_microusd"),
        ("energy_mwh", "energy_mwh"),
    ] {
        let cap = number(get(caps, cap_key, "budget_caps")?, cap_key)?;
        let m = get(usage, usage_key, "actual_usage")?;
        match text(get(m, "status", usage_key)?, "measurement.status")? {
            "measured" => {
                let actual = number(get(m, "value", usage_key)?, "measurement.value")?;
                if actual < 0.0 || actual > cap {
                    return Err(err(ReceiptErrorKind::BudgetExceeded, format!("{usage_key} exceeded declared budget")));
                }
            }
            "unavailable" | "not_measured" => {
                text(get(m, "reason", usage_key)?, "measurement.reason")?;
            }
            _ => return Err(err(ReceiptErrorKind::InvalidType, format!("unknown measurement state for {usage_key}"))),
        }
    }
    Ok(())
}

fn validate_scores(receipt: &Value) -> Result<(), ReceiptValidationError> {
    let grader = get(receipt, "grader", "receipt")?;
    let grader_official = get(grader, "official_grader", "grader")?.as_bool()
        .ok_or_else(|| err(ReceiptErrorKind::InvalidType, "official_grader must be boolean"))?;
    let grader_revision = text(get(grader, "revision", "grader")?, "grader.revision")?;
    let artifacts = get(receipt, "artifacts", "receipt")?.as_array()
        .ok_or_else(|| err(ReceiptErrorKind::InvalidType, "artifacts must be array"))?;
    let official_outputs: HashSet<String> = artifacts.iter()
        .filter(|x| x.get("role").and_then(Value::as_str) == Some("official-grader-output"))
        .filter_map(|x| x.get("sha256").and_then(Value::as_str).map(str::to_owned))
        .collect();

    for (i, score) in get(receipt, "official_scores", "receipt")?.as_array()
        .ok_or_else(|| err(ReceiptErrorKind::InvalidType, "official_scores must be array"))?.iter().enumerate()
    {
        let cx = format!("official_scores[{i}]");
        if get(score, "is_official", &cx)?.as_bool() != Some(true) || !grader_official {
            return Err(err(ReceiptErrorKind::InvalidOfficialScore, "official score requires an official grader and is_official=true"));
        }
        let value = number(get(score, "value", &cx)?, "score.value")?;
        let numerator = number(get(score, "numerator", &cx)?, "score.numerator")?;
        let denominator = number(get(score, "denominator", &cx)?, "score.denominator")?;
        if !ratio_matches(value, numerator, denominator) {
            return Err(err(ReceiptErrorKind::InvalidOfficialScore, "official score does not equal numerator / denominator"));
        }
        if text(get(score, "formula", &cx)?, "score.formula")? != "numerator_divided_by_denominator"
            || text(get(score, "scoring_revision", &cx)?, "score.scoring_revision")? != grader_revision
        {
            return Err(err(ReceiptErrorKind::InvalidOfficialScore, "official formula or grader revision mismatch"));
        }
        let output_hash = digest_field(get(score, "grader_output_sha256", &cx)?, "grader_output_sha256")?;
        if !official_outputs.contains(output_hash) {
            return Err(err(ReceiptErrorKind::InvalidOfficialScore, "official grader output is not present in artifacts"));
        }
    }

    let accounting = get(receipt, "task_accounting", "receipt")?;
    let eligible = integer(get(accounting, "eligible", "task_accounting")?, "eligible")?;
    let attempted = integer(get(accounting, "attempted", "task_accounting")?, "attempted")?;
    let graded = integer(get(accounting, "graded", "task_accounting")?, "graded")?;
    let passed = integer(get(get(accounting, "disposition_counts", "task_accounting")?, "passed", "disposition_counts")?, "passed")?;

    for (i, metric) in get(receipt, "derived_metrics", "receipt")?.as_array()
        .ok_or_else(|| err(ReceiptErrorKind::InvalidType, "derived_metrics must be array"))?.iter().enumerate()
    {
        let cx = format!("derived_metrics[{i}]");
        if get(metric, "is_official", &cx)?.as_bool() != Some(false) {
            return Err(err(ReceiptErrorKind::InvalidDerivedMetric, "derived metrics must be marked non-official"));
        }
        let metric_id = text(get(metric, "metric_id", &cx)?, "derived metric_id")?;
        let value = get(metric, "value", &cx)?;
        let denominator = eligible as f64;
        let expected = match metric_id {
            "eligible_coverage" => Some(attempted as f64 / denominator),
            "graded_coverage" => Some(graded as f64 / denominator),
            "coverage_adjusted_task_yield" => Some(passed as f64 / denominator),
            _ => None,
        };
        if let Some(expected) = expected {
            if denominator <= 0.0 || (number(value, "derived metric value")? - expected).abs() > 1e-12 {
                return Err(err(ReceiptErrorKind::InvalidDerivedMetric, format!("{metric_id} does not match task accounting")));
            }
        } else if value.is_null() {
            text(get(metric, "null_reason", &cx)?, "derived metric null_reason")?;
        } else {
            number(value, "derived metric value")?;
        }
    }
    Ok(())
}

fn validate_comparison(receipt: &Value) -> Result<(), ReceiptValidationError> {
    let m = get(receipt, "matched_comparison", "receipt")?;
    let matched = get(m, "matched", "matched_comparison")?.as_bool()
        .ok_or_else(|| err(ReceiptErrorKind::InvalidType, "matched must be boolean"))?;
    let keys = ["comparison_group_id", "eligible_task_set_sha256", "budget_profile_sha256", "cohort_sha256", "scoring_revision"];
    if !matched {
        for key in keys {
            if !get(m, key, "matched_comparison")?.is_null() {
                return Err(err(ReceiptErrorKind::InvalidComparison, "unmatched receipt must not carry matched-group claims"));
            }
        }
        return Ok(());
    }

    text(get(m, "comparison_group_id", "matched_comparison")?, "comparison_group_id")?;
    let dataset_digest = text(get(get(receipt, "dataset", "receipt")?, "eligible_task_set_sha256", "dataset")?, "dataset.eligible_task_set_sha256")?;
    if text(get(m, "eligible_task_set_sha256", "matched_comparison")?, "eligible_task_set_sha256")? != dataset_digest {
        return Err(err(ReceiptErrorKind::InvalidComparison, "matched eligible-task digest differs from dataset"));
    }
    let budget_hash = canonical_sha256(get(receipt, "budget_caps", "receipt")?)?;
    if text(get(m, "budget_profile_sha256", "matched_comparison")?, "budget_profile_sha256")? != budget_hash {
        return Err(err(ReceiptErrorKind::InvalidComparison, "matched budget digest differs from budget_caps"));
    }
    let cohort_hash = canonical_sha256(get(receipt, "environment", "receipt")?)?;
    if text(get(m, "cohort_sha256", "matched_comparison")?, "cohort_sha256")? != cohort_hash {
        return Err(err(ReceiptErrorKind::InvalidComparison, "matched cohort digest differs from environment"));
    }
    if text(get(m, "scoring_revision", "matched_comparison")?, "scoring_revision")?
        != text(get(get(receipt, "grader", "receipt")?, "revision", "grader")?, "grader.revision")?
    {
        return Err(err(ReceiptErrorKind::InvalidComparison, "matched scoring revision differs from grader"));
    }
    Ok(())
}

pub fn validate_receipt_value(receipt: &Value) -> Result<VerifiedBenchmarkReceipt, ReceiptValidationError> {
    if text(get(receipt, "schema", "receipt")?, "schema")? != RECEIPT_SCHEMA_ID {
        return Err(err(ReceiptErrorKind::SchemaMarkerMismatch, "unsupported receipt schema"));
    }
    let run_id = text(get(receipt, "run_id", "receipt")?, "run_id")?;
    if !run_id.strip_prefix("run_").is_some_and(is_sha256) {
        return Err(err(ReceiptErrorKind::InvalidDigest, "run_id must be run_ followed by 64 lowercase hex characters"));
    }
    let dataset = get(receipt, "dataset", "receipt")?;
    let synthetic = get(dataset, "synthetic", "dataset")?.as_bool()
        .ok_or_else(|| err(ReceiptErrorKind::InvalidType, "dataset.synthetic must be boolean"))?;
    let access_class = text(get(get(receipt, "benchmark", "receipt")?, "access_class", "benchmark")?, "benchmark.access_class")?;
    if synthetic != (access_class == "synthetic") {
        return Err(err(ReceiptErrorKind::InvalidComparison, "synthetic flag and benchmark access class disagree"));
    }

    let (eligible, attempted, not_attempted, graded, passed) = validate_accounting(receipt)?;
    validate_budgets(receipt)?;
    validate_scores(receipt)?;
    validate_comparison(receipt)?;

    for (i, exclusion) in get(receipt, "exclusions", "receipt")?.as_array()
        .ok_or_else(|| err(ReceiptErrorKind::InvalidType, "exclusions must be array"))?.iter().enumerate()
    {
        let cx = format!("exclusions[{i}]");
        if get(exclusion, "preregistered", &cx)?.as_bool() != Some(true) {
            return Err(err(ReceiptErrorKind::InvalidExclusion, "post-hoc exclusion rejected"));
        }
        text(get(exclusion, "reason", &cx)?, "exclusion.reason")?;
        text(get(exclusion, "approval_reference", &cx)?, "exclusion.approval_reference")?;
    }

    let integrity = get(receipt, "integrity", "receipt")?;
    let expected = digest_field(get(integrity, "receipt_sha256", "integrity")?, "integrity.receipt_sha256")?;
    let mut semantic = receipt.clone();
    semantic.get_mut("integrity").and_then(Value::as_object_mut)
        .ok_or_else(|| err(ReceiptErrorKind::InvalidType, "integrity must be object"))?
        .remove("receipt_sha256");
    let computed = canonical_sha256(&semantic)?;
    if computed != expected {
        return Err(err(ReceiptErrorKind::IntegrityMismatch, "canonical receipt hash mismatch"));
    }

    Ok(VerifiedBenchmarkReceipt {
        run_id: run_id.to_owned(),
        raw_file_sha256: String::new(),
        canonical_receipt_sha256: computed,
        eligible,
        attempted,
        not_attempted,
        graded,
        passed,
        authenticity_verified: false,
    })
}

pub fn validate_receipt_json(input: &[u8]) -> Result<VerifiedBenchmarkReceipt, ReceiptValidationError> {
    if input.len() > MAX_RECEIPT_JSON_BYTES {
        return Err(err(ReceiptErrorKind::InputTooLarge, "receipt exceeds maximum input size"));
    }
    let mut deserializer = serde_json::Deserializer::from_slice(input);
    let parsed = StrictValue::deserialize(&mut deserializer)
        .map_err(|e| err(ReceiptErrorKind::InvalidJson, format!("invalid receipt JSON: {e}")))?;
    deserializer.end().map_err(|e| err(ReceiptErrorKind::InvalidJson, format!("trailing JSON data: {e}")))?;
    let mut verified = validate_receipt_value(&parsed.0)?;
    verified.raw_file_sha256 = sha256_hex(input);
    Ok(verified)
}

pub fn validate_matched_receipts(receipts: &[Value]) -> Result<(), ReceiptValidationError> {
    if receipts.len() < 2 {
        return Err(err(ReceiptErrorKind::InvalidComparison, "at least two receipts are required"));
    }
    let mut ids = HashSet::new();
    let mut arms = HashSet::new();
    let mut group: Option<String> = None;
    let mut task_set: Option<String> = None;
    let mut budget: Option<String> = None;
    let mut cohort: Option<String> = None;
    let mut scoring: Option<String> = None;
    let mut benchmark_id: Option<String> = None;

    for receipt in receipts {
        validate_receipt_value(receipt)?;
        let id = text(get(receipt, "run_id", "receipt")?, "run_id")?.to_owned();
        let arm = text(get(receipt, "arm", "receipt")?, "arm")?.to_owned();
        if !ids.insert(id) || !arms.insert(arm) {
            return Err(err(ReceiptErrorKind::InvalidComparison, "duplicate run id or arm in matched comparison"));
        }
        let m = get(receipt, "matched_comparison", "receipt")?;
        if get(m, "matched", "matched_comparison")?.as_bool() != Some(true) {
            return Err(err(ReceiptErrorKind::InvalidComparison, "every matched arm must declare matched=true"));
        }
        let current_group = text(get(m, "comparison_group_id", "matched_comparison")?, "comparison_group_id")?.to_owned();
        let current_tasks = text(get(m, "eligible_task_set_sha256", "matched_comparison")?, "eligible_task_set_sha256")?.to_owned();
        let current_budget = text(get(m, "budget_profile_sha256", "matched_comparison")?, "budget_profile_sha256")?.to_owned();
        let current_cohort = text(get(m, "cohort_sha256", "matched_comparison")?, "cohort_sha256")?.to_owned();
        let current_scoring = text(get(m, "scoring_revision", "matched_comparison")?, "scoring_revision")?.to_owned();
        let current_benchmark = text(get(get(receipt, "benchmark", "receipt")?, "benchmark_id", "benchmark")?, "benchmark_id")?.to_owned();

        if group.as_ref().is_some_and(|v| v != &current_group)
            || task_set.as_ref().is_some_and(|v| v != &current_tasks)
            || budget.as_ref().is_some_and(|v| v != &current_budget)
            || cohort.as_ref().is_some_and(|v| v != &current_cohort)
            || scoring.as_ref().is_some_and(|v| v != &current_scoring)
            || benchmark_id.as_ref().is_some_and(|v| v != &current_benchmark)
        {
            return Err(err(ReceiptErrorKind::InvalidComparison, "matched arms differ in benchmark/taskset/budget/cohort/scoring scope"));
        }
        group = Some(current_group);
        task_set = Some(current_tasks);
        budget = Some(current_budget);
        cohort = Some(current_cohort);
        scoring = Some(current_scoring);
        benchmark_id = Some(current_benchmark);
    }
    Ok(())
}

struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictVisitor)
    }
}

struct StrictVisitor;

impl<'de> Visitor<'de> for StrictVisitor {
    type Value = StrictValue;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("JSON with unique object keys")
    }
    fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> { Ok(StrictValue(Value::Null)) }
    fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Self::Value, E> { Ok(StrictValue(Value::Bool(v))) }
    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> { Ok(StrictValue(Value::Number(Number::from(v)))) }
    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> { Ok(StrictValue(Value::Number(Number::from(v)))) }
    fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
        let n = Number::from_f64(v).ok_or_else(|| E::custom("non-finite JSON number"))?;
        Ok(StrictValue(Value::Number(n)))
    }
    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> { Ok(StrictValue(Value::String(v.to_owned()))) }
    fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> { Ok(StrictValue(Value::String(v))) }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut values = Vec::new();
        while let Some(v) = seq.next_element::<StrictValue>()? { values.push(v.0); }
        Ok(StrictValue(Value::Array(values)))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut values = Map::new();
        let mut keys = HashSet::new();
        while let Some((key, value)) = map.next_entry::<String, StrictValue>()? {
            if !keys.insert(key.clone()) {
                return Err(serde::de::Error::custom(format!("duplicate object key {key}")));
            }
            values.insert(key, value.0);
        }
        Ok(StrictValue(Value::Object(values)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<u8> {
        include_bytes!("../../../examples/benchmark-run-receipt-v1.synthetic.json").to_vec()
    }

    #[test]
    fn accepts_synthetic_receipt_and_separates_raw_and_canonical_digests() {
        let bytes = fixture();
        let v = validate_receipt_json(&bytes).unwrap();
        assert_eq!((v.eligible, v.attempted, v.not_attempted, v.graded, v.passed), (5, 4, 1, 2, 2));
        assert!(!v.authenticity_verified);
        assert_eq!(v.raw_file_sha256, sha256_hex(&bytes));
        assert_ne!(v.raw_file_sha256, v.canonical_receipt_sha256);
    }

    #[test]
    fn rejects_duplicate_task_ids() {
        let mut v: Value = serde_json::from_slice(&fixture()).unwrap();
        v["task_outcomes"][1] = v["task_outcomes"][0].clone();
        assert_eq!(validate_receipt_value(&v).unwrap_err().kind, ReceiptErrorKind::DuplicateTask);
    }

    #[test]
    fn rejects_hidden_refusal_and_denominator_laundering() {
        let mut v: Value = serde_json::from_slice(&fixture()).unwrap();
        v["task_accounting"]["attempted"] = Value::from(3);
        v["task_accounting"]["not_attempted"] = Value::from(2);
        v["task_accounting"]["disposition_counts"]["refused"] = Value::from(0);
        assert_eq!(validate_receipt_value(&v).unwrap_err().kind, ReceiptErrorKind::AccountingMismatch);
    }

    #[test]
    fn rejects_dropped_task_rows() {
        let mut v: Value = serde_json::from_slice(&fixture()).unwrap();
        v["task_outcomes"].as_array_mut().unwrap().pop();
        assert_eq!(validate_receipt_value(&v).unwrap_err().kind, ReceiptErrorKind::AccountingMismatch);
    }

    #[test]
    fn rejects_missing_grader_result_for_graded_task() {
        let mut v: Value = serde_json::from_slice(&fixture()).unwrap();
        v["task_outcomes"][0]["grader_result"] = Value::Null;
        assert_eq!(validate_receipt_value(&v).unwrap_err().kind, ReceiptErrorKind::InvalidType);
    }

    #[test]
    fn rejects_budget_overrun() {
        let mut v: Value = serde_json::from_slice(&fixture()).unwrap();
        v["actual_usage"]["tool_calls"]["value"] = Value::from(26);
        assert_eq!(validate_receipt_value(&v).unwrap_err().kind, ReceiptErrorKind::BudgetExceeded);
    }

    #[test]
    fn rejects_official_score_without_official_grader_and_output_artifact() {
        let mut v: Value = serde_json::from_slice(&fixture()).unwrap();
        v["official_scores"] = serde_json::json!([{
            "is_official": true,
            "value": 1.0,
            "numerator": 1.0,
            "denominator": 1.0,
            "formula": "numerator_divided_by_denominator",
            "scoring_revision": "fixture-v1",
            "grader_output_sha256": "e".repeat(64)
        }]);
        assert_eq!(validate_receipt_value(&v).unwrap_err().kind, ReceiptErrorKind::InvalidOfficialScore);
    }

    #[test]
    fn rejects_falsified_coverage_adjusted_metrics() {
        let mut v: Value = serde_json::from_slice(&fixture()).unwrap();
        v["derived_metrics"][2]["value"] = Value::from(0.9);
        assert_eq!(validate_receipt_value(&v).unwrap_err().kind, ReceiptErrorKind::InvalidDerivedMetric);
    }

    #[test]
    fn rejects_unmatched_receipt_carrying_a_comparison_claim() {
        let mut v: Value = serde_json::from_slice(&fixture()).unwrap();
        v["matched_comparison"]["comparison_group_id"] = Value::String("group-1".to_owned());
        assert_eq!(validate_receipt_value(&v).unwrap_err().kind, ReceiptErrorKind::InvalidComparison);
    }

    fn rehash_receipt(value: &mut Value) {
        let mut semantic = value.clone();
        semantic["integrity"].as_object_mut().unwrap().remove("receipt_sha256");
        value["integrity"]["receipt_sha256"] = Value::String(canonical_sha256(&semantic).unwrap());
    }

    fn make_matched_receipt(run_char: char, arm: &str) -> Value {
        let mut value: Value = serde_json::from_slice(&fixture()).unwrap();
        value["run_id"] = Value::String(format!("run_{}", run_char.to_string().repeat(64)));
        value["arm"] = Value::String(arm.to_owned());
        let budget_hash = canonical_sha256(&value["budget_caps"]).unwrap();
        let cohort_hash = canonical_sha256(&value["environment"]).unwrap();
        let task_hash = value["dataset"]["eligible_task_set_sha256"].clone();
        value["matched_comparison"] = serde_json::json!({
            "matched": true,
            "comparison_group_id": "ablation-group-1",
            "eligible_task_set_sha256": task_hash,
            "budget_profile_sha256": budget_hash,
            "cohort_sha256": cohort_hash,
            "scoring_revision": "fixture-v1"
        });
        rehash_receipt(&mut value);
        value
    }

    #[test]
    fn accepts_budget_and_taskset_matched_distinct_arms() {
        let rea = make_matched_receipt('b', "rea_only");
        let combined = make_matched_receipt('c', "rea_plus_symthaea");
        assert!(validate_matched_receipts(&[rea, combined]).is_ok());
    }

    #[test]
    fn rejects_matched_arms_with_different_eligible_task_sets() {
        let rea = make_matched_receipt('b', "rea_only");
        let mut combined = make_matched_receipt('c', "rea_plus_symthaea");
        combined["dataset"]["eligible_task_set_sha256"] = Value::String("d".repeat(64));
        combined["matched_comparison"]["eligible_task_set_sha256"] = Value::String("d".repeat(64));
        rehash_receipt(&mut combined);
        assert_eq!(
            validate_matched_receipts(&[rea, combined]).unwrap_err().kind,
            ReceiptErrorKind::InvalidComparison
        );
    }

    #[test]
    fn rejects_posthoc_exclusions() {
        let mut v: Value = serde_json::from_slice(&fixture()).unwrap();
        v["exclusions"] = serde_json::json!([{
            "task_id": "task-003",
            "reason": "hard task",
            "preregistered": false,
            "approval_reference": "none"
        }]);
        assert_eq!(validate_receipt_value(&v).unwrap_err().kind, ReceiptErrorKind::InvalidExclusion);
    }

    #[test]
    fn rejects_tampered_semantic_receipt_hash() {
        let mut v: Value = serde_json::from_slice(&fixture()).unwrap();
        v["integrity"]["receipt_sha256"] = Value::String("0".repeat(64));
        assert_eq!(validate_receipt_value(&v).unwrap_err().kind, ReceiptErrorKind::IntegrityMismatch);
    }

    #[test]
    fn rejects_duplicate_json_keys_in_raw_receipt() {
        let bytes = br#"{"schema":"luminous.benchmark-run-receipt.v1","schema":"other"}"#;
        assert_eq!(validate_receipt_json(bytes).unwrap_err().kind, ReceiptErrorKind::InvalidJson);
    }
}
