#![forbid(unsafe_code)]

//! Semantic validation for benchmark run receipts.
//!
//! A valid receipt establishes accounting, scope and digest consistency. It does
//! not establish answer truth: the pinned independent grader remains the
//! correctness oracle, and a self-hash is not a signature.

use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fmt;

pub const MAX_RECEIPT_BYTES: usize = 16 * 1024 * 1024;

const TOP_LEVEL_KEYS: &[&str] = &[
    "schema_version",
    "run_id",
    "generated_at",
    "evaluation_started_at",
    "evaluation_completed_at",
    "benchmark",
    "system",
    "target_scope",
    "runner",
    "comparison",
    "task_universe",
    "eligible_task_set_sha256",
    "task_universe_sha256",
    "task_results",
    "grading_mode",
    "accounting",
    "budget_caps",
    "actual_usage",
    "unavailable_usage",
    "official_score",
    "derived_metrics",
    "provenance",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptErrorKind {
    InvalidJson,
    InputTooLarge,
    SchemaViolation,
    AccountingMismatch,
    DigestMismatch,
    BudgetExceeded,
    NonComparable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptError {
    pub kind: ReceiptErrorKind,
    pub message: String,
}

impl ReceiptError {
    fn new(kind: ReceiptErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for ReceiptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ReceiptError {}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedReceipt {
    pub run_id: String,
    /// SHA-256 of the exact input JSON bytes.
    pub raw_file_sha256: String,
    /// SHA-256 of the canonical semantic JSON representation.
    pub canonical_receipt_sha256: String,
    pub value: Value,
    pub accounting: BTreeMap<String, u64>,
}

fn error(kind: ReceiptErrorKind, message: impl Into<String>) -> ReceiptError {
    ReceiptError::new(kind, message)
}

fn obj<'a>(value: &'a Value, context: &str) -> Result<&'a Map<String, Value>, ReceiptError> {
    value.as_object().ok_or_else(|| {
        error(
            ReceiptErrorKind::SchemaViolation,
            format!("{context} must be an object"),
        )
    })
}

fn arr<'a>(value: &'a Value, context: &str) -> Result<&'a Vec<Value>, ReceiptError> {
    value.as_array().ok_or_else(|| {
        error(
            ReceiptErrorKind::SchemaViolation,
            format!("{context} must be an array"),
        )
    })
}

fn required<'a>(
    map: &'a Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<&'a Value, ReceiptError> {
    map.get(key).ok_or_else(|| {
        error(
            ReceiptErrorKind::SchemaViolation,
            format!("{context}.{key} is required"),
        )
    })
}

fn strict_keys(
    map: &Map<String, Value>,
    required_keys: &[&str],
    optional_keys: &[&str],
    context: &str,
) -> Result<(), ReceiptError> {
    for key in required_keys {
        if !map.contains_key(*key) {
            return Err(error(
                ReceiptErrorKind::SchemaViolation,
                format!("{context}.{key} is required"),
            ));
        }
    }
    for key in map.keys() {
        if !required_keys.contains(&key.as_str()) && !optional_keys.contains(&key.as_str()) {
            return Err(error(
                ReceiptErrorKind::SchemaViolation,
                format!("{context} contains unknown field '{key}'"),
            ));
        }
    }
    Ok(())
}

fn string<'a>(value: &'a Value, context: &str, allow_empty: bool) -> Result<&'a str, ReceiptError> {
    let value = value.as_str().ok_or_else(|| {
        error(
            ReceiptErrorKind::SchemaViolation,
            format!("{context} must be a string"),
        )
    })?;
    if !allow_empty && value.trim().is_empty() {
        return Err(error(
            ReceiptErrorKind::SchemaViolation,
            format!("{context} must not be empty"),
        ));
    }
    if value.contains('\0') {
        return Err(error(
            ReceiptErrorKind::SchemaViolation,
            format!("{context} must not contain NUL"),
        ));
    }
    Ok(value)
}

fn bool_value(value: &Value, context: &str) -> Result<bool, ReceiptError> {
    value.as_bool().ok_or_else(|| {
        error(
            ReceiptErrorKind::SchemaViolation,
            format!("{context} must be boolean"),
        )
    })
}

fn nonnegative_u64(value: &Value, context: &str) -> Result<u64, ReceiptError> {
    let n = value.as_u64().ok_or_else(|| {
        error(
            ReceiptErrorKind::SchemaViolation,
            format!("{context} must be a non-negative integer"),
        )
    })?;
    const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
    if n > MAX_SAFE_INTEGER {
        return Err(error(
            ReceiptErrorKind::SchemaViolation,
            format!("{context} exceeds the JavaScript safe-integer range"),
        ));
    }
    Ok(n)
}

fn validate_utc_timestamp<'a>(value: &'a Value, context: &str) -> Result<&'a str, ReceiptError> {
    let timestamp = string(value, context, false)?;
    let bytes = timestamp.as_bytes();
    let separators_ok = bytes.len() == 20
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes[10] == b'T'
        && bytes[13] == b':'
        && bytes[16] == b':'
        && bytes[19] == b'Z';
    let digit_positions = [0usize, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18];
    if !separators_ok
        || digit_positions
            .iter()
            .any(|index| !bytes[*index].is_ascii_digit())
    {
        return Err(error(
            ReceiptErrorKind::SchemaViolation,
            format!("{context} must be a UTC timestamp in YYYY-MM-DDTHH:MM:SSZ form"),
        ));
    }
    let year = timestamp[0..4].parse::<u32>().unwrap_or(0);
    let month = timestamp[5..7].parse::<u32>().unwrap_or(0);
    let day = timestamp[8..10].parse::<u32>().unwrap_or(0);
    let hour = timestamp[11..13].parse::<u32>().unwrap_or(99);
    let minute = timestamp[14..16].parse::<u32>().unwrap_or(99);
    let second = timestamp[17..19].parse::<u32>().unwrap_or(99);
    let leap = year % 400 == 0 || (year % 4 == 0 && year % 100 != 0);
    let month_days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => 0,
    };
    if year == 0 || day == 0 || day > month_days || hour > 23 || minute > 59 || second > 59 {
        return Err(error(
            ReceiptErrorKind::SchemaViolation,
            format!("{context} is not a valid UTC calendar timestamp"),
        ));
    }
    Ok(timestamp)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn sha256_field<'a>(value: &'a Value, context: &str) -> Result<&'a str, ReceiptError> {
    let value = string(value, context, false)?;
    if !is_sha256(value) {
        return Err(error(
            ReceiptErrorKind::SchemaViolation,
            format!("{context} must be 64 lowercase hexadecimal characters"),
        ));
    }
    Ok(value)
}

/// Emit every parsed JSON number as IEEE-754 binary64 before RFC 8785 JCS.
struct EcmaJsonValue<'a>(&'a Value);

impl Serialize for EcmaJsonValue<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self.0 {
            Value::Null => serializer.serialize_unit(),
            Value::Bool(value) => serializer.serialize_bool(*value),
            Value::Number(value) => {
                let number = value.as_f64().ok_or_else(|| {
                    <S::Error as serde::ser::Error>::custom(
                        "JSON number cannot be represented as IEEE-754 binary64",
                    )
                })?;
                if !number.is_finite() {
                    return Err(<S::Error as serde::ser::Error>::custom(
                        "JSON numbers must be finite",
                    ));
                }
                serializer.serialize_f64(number)
            }
            Value::String(value) => serializer.serialize_str(value),
            Value::Array(values) => {
                let mut seq = serializer.serialize_seq(Some(values.len()))?;
                for value in values {
                    seq.serialize_element(&EcmaJsonValue(value))?;
                }
                seq.end()
            }
            Value::Object(values) => {
                let mut map = serializer.serialize_map(Some(values.len()))?;
                for (key, value) in values {
                    map.serialize_entry(key, &EcmaJsonValue(value))?;
                }
                map.end()
            }
        }
    }
}

pub fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>, ReceiptError> {
    serde_json_canonicalizer::to_vec(&EcmaJsonValue(value)).map_err(|e| {
        error(
            ReceiptErrorKind::InvalidJson,
            format!("canonical JSON serialization failed: {e}"),
        )
    })
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn canonical_digest(value: &Value) -> Result<String, ReceiptError> {
    Ok(sha256_hex(&canonical_json_bytes(value)?))
}

fn validate_string_field(
    map: &Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<(), ReceiptError> {
    string(
        required(map, key, context)?,
        &format!("{context}.{key}"),
        false,
    )?;
    Ok(())
}

fn validate_hash_field(
    map: &Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<(), ReceiptError> {
    sha256_field(required(map, key, context)?, &format!("{context}.{key}"))?;
    Ok(())
}

fn validate_target_scope(value: &Value) -> Result<(), ReceiptError> {
    let map = obj(value, "target_scope")?;
    strict_keys(
        map,
        &["platform", "os", "architecture", "cohort"],
        &[],
        "target_scope",
    )?;
    for key in ["platform", "os", "architecture", "cohort"] {
        validate_string_field(map, key, "target_scope")?;
    }
    Ok(())
}

fn validate_task_universe(
    value: &Value,
    evaluation_started_at: &str,
) -> Result<(Vec<String>, Vec<Value>, u64, u64), ReceiptError> {
    let tasks = arr(value, "task_universe")?;
    if tasks.is_empty() {
        return Err(error(
            ReceiptErrorKind::SchemaViolation,
            "task_universe must not be empty",
        ));
    }

    let mut ids = HashSet::new();
    let mut eligible = Vec::new();
    let mut canonical_rows = Vec::new();
    let mut excluded_count = 0u64;

    for (index, task) in tasks.iter().enumerate() {
        let context = format!("task_universe[{index}]");
        let map = obj(task, &context)?;
        strict_keys(map, &["task_id", "eligibility", "exclusion"], &[], &context)?;
        let id = string(
            required(map, "task_id", &context)?,
            &format!("{context}.task_id"),
            false,
        )?;
        if !ids.insert(id.to_owned()) {
            return Err(error(
                ReceiptErrorKind::AccountingMismatch,
                format!("duplicate task ID '{id}' in task_universe"),
            ));
        }

        let eligibility = string(
            required(map, "eligibility", &context)?,
            &format!("{context}.eligibility"),
            false,
        )?;
        let exclusion = required(map, "exclusion", &context)?;
        match eligibility {
            "eligible" => {
                if !exclusion.is_null() {
                    return Err(error(
                        ReceiptErrorKind::SchemaViolation,
                        format!("{context}.exclusion must be null for eligible tasks"),
                    ));
                }
                eligible.push(id.to_owned());
            }
            "excluded" => {
                excluded_count += 1;
                let ex = obj(exclusion, &format!("{context}.exclusion"))?;
                strict_keys(
                    ex,
                    &["predeclared", "reason", "declared_at"],
                    &[],
                    &format!("{context}.exclusion"),
                )?;
                if !bool_value(
                    required(ex, "predeclared", &context)?,
                    &format!("{context}.exclusion.predeclared"),
                )? {
                    return Err(error(
                        ReceiptErrorKind::AccountingMismatch,
                        format!("{context} was excluded without a predeclared exclusion"),
                    ));
                }
                string(
                    required(ex, "reason", &context)?,
                    &format!("{context}.exclusion.reason"),
                    false,
                )?;
                let declared_at = validate_utc_timestamp(
                    required(ex, "declared_at", &context)?,
                    &format!("{context}.exclusion.declared_at"),
                )?;
                if declared_at > evaluation_started_at {
                    return Err(error(
                        ReceiptErrorKind::AccountingMismatch,
                        format!("{context} exclusion was declared after evaluation started"),
                    ));
                }
            }
            other => {
                return Err(error(
                    ReceiptErrorKind::SchemaViolation,
                    format!("{context}.eligibility '{other}' is unsupported"),
                ));
            }
        }
        canonical_rows.push(task.clone());
    }

    eligible.sort();
    canonical_rows.sort_by(|left, right| {
        left.get("task_id")
            .and_then(Value::as_str)
            .cmp(&right.get("task_id").and_then(Value::as_str))
    });
    Ok((eligible, canonical_rows, tasks.len() as u64, excluded_count))
}

#[derive(Default, Clone, Copy)]
struct Counts {
    universe: u64,
    eligible: u64,
    excluded: u64,
    attempted: u64,
    not_attempted: u64,
    in_progress: u64,
    completed: u64,
    graded: u64,
    passed: u64,
    failed: u64,
    refused: u64,
    abstained: u64,
    unsupported: u64,
    timed_out: u64,
    tool_failed: u64,
    ungraded: u64,
}

impl Counts {
    fn as_map(self) -> BTreeMap<String, u64> {
        [
            ("universe_count", self.universe),
            ("eligible_count", self.eligible),
            ("excluded_count", self.excluded),
            ("attempted_count", self.attempted),
            ("not_attempted_count", self.not_attempted),
            ("in_progress_count", self.in_progress),
            ("completed_count", self.completed),
            ("graded_count", self.graded),
            ("passed_count", self.passed),
            ("failed_count", self.failed),
            ("refused_count", self.refused),
            ("abstained_count", self.abstained),
            ("unsupported_count", self.unsupported),
            ("timed_out_count", self.timed_out),
            ("tool_failed_count", self.tool_failed),
            ("ungraded_count", self.ungraded),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect()
    }
}

fn validate_grade(
    value: &Value,
    grading_mode: &str,
    expected_grader_version: &str,
    context: &str,
) -> Result<(), ReceiptError> {
    let map = obj(value, context)?;
    strict_keys(
        map,
        &["kind", "value", "grader_version", "grader_output_sha256"],
        &[],
        context,
    )?;
    let kind = string(
        required(map, "kind", context)?,
        &format!("{context}.kind"),
        false,
    )?;
    if kind != grading_mode {
        return Err(error(
            ReceiptErrorKind::SchemaViolation,
            format!("{context}.kind '{kind}' does not match grading_mode '{grading_mode}'"),
        ));
    }

    let grade_value = required(map, "value", context)?;
    let valid = match grading_mode {
        "binary" => grade_value.is_boolean(),
        "numeric" => grade_value.as_f64().is_some_and(f64::is_finite),
        "categorical" => grade_value.as_str().is_some(),
        "custom" => {
            grade_value.is_boolean()
                || grade_value.as_str().is_some()
                || grade_value.as_f64().is_some_and(f64::is_finite)
        }
        other => {
            return Err(error(
                ReceiptErrorKind::SchemaViolation,
                format!("grading_mode '{other}' is unsupported"),
            ));
        }
    };
    if !valid {
        return Err(error(
            ReceiptErrorKind::SchemaViolation,
            format!("{context}.value has the wrong type for grading_mode"),
        ));
    }
    let grade_grader_version = string(
        required(map, "grader_version", context)?,
        &format!("{context}.grader_version"),
        false,
    )?;
    if grade_grader_version != expected_grader_version {
        return Err(error(
            ReceiptErrorKind::NonComparable,
            format!("{context}.grader_version does not match the benchmark grader version"),
        ));
    }
    sha256_field(
        required(map, "grader_output_sha256", context)?,
        &format!("{context}.grader_output_sha256"),
    )?;
    Ok(())
}

#[derive(Default, Clone, Copy)]
struct TaskUsage {
    tool_calls: u64,
    model_calls: u64,
}

fn validate_task_results(
    value: &Value,
    eligible_ids: &[String],
    grading_mode: &str,
    grader_version: &str,
) -> Result<(Counts, TaskUsage), ReceiptError> {
    let rows = arr(value, "task_results")?;
    let eligible: HashSet<&str> = eligible_ids.iter().map(String::as_str).collect();
    let mut seen = HashSet::new();
    let mut counts = Counts {
        eligible: eligible_ids.len() as u64,
        ..Counts::default()
    };
    let mut usage = TaskUsage::default();

    for (index, row) in rows.iter().enumerate() {
        let context = format!("task_results[{index}]");
        let map = obj(row, &context)?;
        strict_keys(
            map,
            &[
                "task_id",
                "input_sha256",
                "attempt_state",
                "attempts",
                "terminal_disposition",
                "grade",
                "tool_calls",
                "model_calls",
                "wall_time_ms",
                "output_sha256",
            ],
            &[],
            &context,
        )?;

        let id = string(
            required(map, "task_id", &context)?,
            &format!("{context}.task_id"),
            false,
        )?;
        if !seen.insert(id.to_owned()) {
            return Err(error(
                ReceiptErrorKind::AccountingMismatch,
                format!("duplicate task result for '{id}'"),
            ));
        }
        if !eligible.contains(id) {
            return Err(error(
                ReceiptErrorKind::AccountingMismatch,
                format!("task result '{id}' is not eligible in task_universe"),
            ));
        }

        sha256_field(
            required(map, "input_sha256", &context)?,
            &format!("{context}.input_sha256"),
        )?;
        let state = string(
            required(map, "attempt_state", &context)?,
            &format!("{context}.attempt_state"),
            false,
        )?;
        let attempts = nonnegative_u64(
            required(map, "attempts", &context)?,
            &format!("{context}.attempts"),
        )?;
        let disposition = required(map, "terminal_disposition", &context)?;
        let grade = required(map, "grade", &context)?;
        let tool_calls = nonnegative_u64(
            required(map, "tool_calls", &context)?,
            &format!("{context}.tool_calls"),
        )?;
        let model_calls = nonnegative_u64(
            required(map, "model_calls", &context)?,
            &format!("{context}.model_calls"),
        )?;
        let _wall_time = nonnegative_u64(
            required(map, "wall_time_ms", &context)?,
            &format!("{context}.wall_time_ms"),
        )?;
        usage.tool_calls = usage.tool_calls.checked_add(tool_calls).ok_or_else(|| {
            error(
                ReceiptErrorKind::AccountingMismatch,
                "per-task tool-call count overflow",
            )
        })?;
        usage.model_calls = usage.model_calls.checked_add(model_calls).ok_or_else(|| {
            error(
                ReceiptErrorKind::AccountingMismatch,
                "per-task model-call count overflow",
            )
        })?;
        for (output_index, output) in arr(
            required(map, "output_sha256", &context)?,
            &format!("{context}.output_sha256"),
        )?
        .iter()
        .enumerate()
        {
            sha256_field(output, &format!("{context}.output_sha256[{output_index}]"))?;
        }

        match state {
            "not_attempted" => {
                counts.not_attempted += 1;
                if attempts != 0
                    || !disposition.is_null()
                    || !grade.is_null()
                    || tool_calls != 0
                    || model_calls != 0
                {
                    return Err(error(
                        ReceiptErrorKind::AccountingMismatch,
                        format!("{context} not_attempted row contains attempt activity"),
                    ));
                }
            }
            "in_progress" => {
                counts.in_progress += 1;
                counts.attempted += 1;
                if attempts == 0 || !disposition.is_null() || !grade.is_null() {
                    return Err(error(
                        ReceiptErrorKind::AccountingMismatch,
                        format!("{context} in_progress row needs attempts and no terminal result"),
                    ));
                }
            }
            "terminal" => {
                counts.attempted += 1;
                counts.completed += 1;
                if attempts == 0 {
                    return Err(error(
                        ReceiptErrorKind::AccountingMismatch,
                        format!("{context} terminal row must have at least one attempt"),
                    ));
                }
                let outcome = string(
                    disposition,
                    &format!("{context}.terminal_disposition"),
                    false,
                )?;
                match outcome {
                    "passed" => counts.passed += 1,
                    "failed" => counts.failed += 1,
                    "refused" => counts.refused += 1,
                    "abstained" => counts.abstained += 1,
                    "unsupported" => counts.unsupported += 1,
                    "timed_out" => counts.timed_out += 1,
                    "tool_failed" => counts.tool_failed += 1,
                    "ungraded" => counts.ungraded += 1,
                    other => {
                        return Err(error(
                            ReceiptErrorKind::SchemaViolation,
                            format!("{context} has unknown terminal disposition '{other}'"),
                        ));
                    }
                }

                match outcome {
                    "passed" | "failed" => {
                        if grade.is_null() {
                            return Err(error(
                                ReceiptErrorKind::AccountingMismatch,
                                format!("{context} graded terminal disposition requires a grade"),
                            ));
                        }
                        validate_grade(
                            grade,
                            grading_mode,
                            grader_version,
                            &format!("{context}.grade"),
                        )?;
                        counts.graded += 1;
                        if grading_mode == "binary" {
                            let grade_map = obj(grade, &format!("{context}.grade"))?;
                            let value = bool_value(
                                required(grade_map, "value", &format!("{context}.grade"))?,
                                &format!("{context}.grade.value"),
                            )?;
                            if (outcome == "passed") != value {
                                return Err(error(
                                    ReceiptErrorKind::AccountingMismatch,
                                    format!(
                                        "{context} binary grade contradicts terminal disposition"
                                    ),
                                ));
                            }
                        }
                    }
                    _ if !grade.is_null() => {
                        return Err(error(
                            ReceiptErrorKind::AccountingMismatch,
                            format!("{context} non-pass/fail disposition must not carry a grade"),
                        ));
                    }
                    _ => {}
                }
            }
            other => {
                return Err(error(
                    ReceiptErrorKind::SchemaViolation,
                    format!("{context}.attempt_state '{other}' is unsupported"),
                ));
            }
        }
    }

    if seen.len() != eligible.len() {
        let missing: Vec<_> = eligible
            .iter()
            .filter(|id| !seen.contains(**id))
            .copied()
            .collect();
        return Err(error(
            ReceiptErrorKind::AccountingMismatch,
            format!(
                "task_results omit {} eligible tasks: {}",
                missing.len(),
                missing.join(", ")
            ),
        ));
    }
    Ok((counts, usage))
}

fn validate_accounting(
    value: &Value,
    actual: Counts,
) -> Result<BTreeMap<String, u64>, ReceiptError> {
    let map = obj(value, "accounting")?;
    let keys = [
        "universe_count",
        "eligible_count",
        "excluded_count",
        "attempted_count",
        "not_attempted_count",
        "in_progress_count",
        "completed_count",
        "graded_count",
        "passed_count",
        "failed_count",
        "refused_count",
        "abstained_count",
        "unsupported_count",
        "timed_out_count",
        "tool_failed_count",
        "ungraded_count",
    ];
    strict_keys(map, &keys, &[], "accounting")?;
    for (key, expected) in actual.as_map() {
        let supplied = nonnegative_u64(
            required(map, &key, "accounting")?,
            &format!("accounting.{key}"),
        )?;
        if supplied != expected {
            return Err(error(
                ReceiptErrorKind::AccountingMismatch,
                format!("accounting.{key}={supplied}, computed {expected}"),
            ));
        }
    }
    if actual.universe != actual.eligible + actual.excluded {
        return Err(error(
            ReceiptErrorKind::AccountingMismatch,
            "universe_count != eligible_count + excluded_count",
        ));
    }
    if actual.eligible != actual.attempted + actual.not_attempted {
        return Err(error(
            ReceiptErrorKind::AccountingMismatch,
            "eligible_count != attempted_count + not_attempted_count",
        ));
    }
    if actual.attempted != actual.in_progress + actual.completed {
        return Err(error(
            ReceiptErrorKind::AccountingMismatch,
            "attempted_count != in_progress_count + completed_count",
        ));
    }
    if actual.completed
        != actual.passed
            + actual.failed
            + actual.refused
            + actual.abstained
            + actual.unsupported
            + actual.timed_out
            + actual.tool_failed
            + actual.ungraded
    {
        return Err(error(
            ReceiptErrorKind::AccountingMismatch,
            "completed_count != sum of terminal dispositions",
        ));
    }
    if actual.graded != actual.passed + actual.failed {
        return Err(error(
            ReceiptErrorKind::AccountingMismatch,
            "graded_count != passed_count + failed_count",
        ));
    }
    Ok(actual.as_map())
}

fn validate_resource_objects(
    caps_value: &Value,
    actual_value: &Value,
    unavailable_value: &Value,
    task_usage: TaskUsage,
) -> Result<(), ReceiptError> {
    let fields = [
        "wall_time_ms",
        "cpu_time_ms",
        "peak_memory_bytes",
        "model_calls",
        "input_tokens",
        "output_tokens",
        "tool_calls",
    ];
    let caps = obj(caps_value, "budget_caps")?;
    let actual = obj(actual_value, "actual_usage")?;
    let unavailable = obj(unavailable_value, "unavailable_usage")?;
    strict_keys(caps, &fields, &[], "budget_caps")?;
    strict_keys(actual, &fields, &[], "actual_usage")?;

    for field in fields {
        let cap = required(caps, field, "budget_caps")?;
        let actual = required(actual, field, "actual_usage")?;
        let reason = unavailable.get(field);
        let actual_value = if actual.is_null() {
            if reason.is_none() {
                return Err(error(
                    ReceiptErrorKind::SchemaViolation,
                    format!("actual_usage.{field} is unavailable without a reason"),
                ));
            }
            None
        } else {
            if reason.is_some() {
                return Err(error(
                    ReceiptErrorKind::SchemaViolation,
                    format!("unavailable_usage.{field} conflicts with measured usage"),
                ));
            }
            Some(nonnegative_u64(actual, &format!("actual_usage.{field}"))?)
        };
        if let (Some(actual), false) = (actual_value, cap.is_null()) {
            let cap = nonnegative_u64(cap, &format!("budget_caps.{field}"))?;
            if actual > cap {
                return Err(error(
                    ReceiptErrorKind::BudgetExceeded,
                    format!("actual_usage.{field}={actual} exceeds budget_caps.{field}={cap}"),
                ));
            }
        }
        if let Some(reason) = reason {
            string(reason, &format!("unavailable_usage.{field}"), false)?;
        }
    }
    let measured_tool_calls = required(actual, "tool_calls", "actual_usage")?;
    if !measured_tool_calls.is_null()
        && nonnegative_u64(measured_tool_calls, "actual_usage.tool_calls")? < task_usage.tool_calls
    {
        return Err(error(
            ReceiptErrorKind::AccountingMismatch,
            "run-level tool_calls is lower than the sum of task-level tool_calls",
        ));
    }
    let measured_model_calls = required(actual, "model_calls", "actual_usage")?;
    if !measured_model_calls.is_null()
        && nonnegative_u64(measured_model_calls, "actual_usage.model_calls")?
            < task_usage.model_calls
    {
        return Err(error(
            ReceiptErrorKind::AccountingMismatch,
            "run-level model_calls is lower than the sum of task-level model_calls",
        ));
    }
    for key in unavailable.keys() {
        if !fields.contains(&key.as_str()) {
            return Err(error(
                ReceiptErrorKind::SchemaViolation,
                format!("unavailable_usage contains unknown metric '{key}'"),
            ));
        }
    }
    Ok(())
}

fn validate_official_and_derived(
    value: &Value,
    derived: &Value,
    access_class: &str,
    benchmark_grader_output_sha256: &str,
    counts: Counts,
) -> Result<(), ReceiptError> {
    let score = obj(value, "official_score")?;
    strict_keys(
        score,
        &[
            "metric_id",
            "formula_version",
            "value",
            "unit",
            "numerator",
            "denominator",
            "is_official",
            "source_output_sha256",
            "unavailable_reason",
        ],
        &[],
        "official_score",
    )?;
    string(
        required(score, "metric_id", "official_score")?,
        "official_score.metric_id",
        false,
    )?;
    string(
        required(score, "formula_version", "official_score")?,
        "official_score.formula_version",
        false,
    )?;
    string(
        required(score, "unit", "official_score")?,
        "official_score.unit",
        false,
    )?;

    let is_official = bool_value(
        required(score, "is_official", "official_score")?,
        "official_score.is_official",
    )?;
    let score_value = required(score, "value", "official_score")?;
    let numerator = required(score, "numerator", "official_score")?;
    let denominator = required(score, "denominator", "official_score")?;
    let source_output = required(score, "source_output_sha256", "official_score")?;
    let unavailable_reason = required(score, "unavailable_reason", "official_score")?;

    if is_official {
        if access_class == "synthetic" {
            return Err(error(
                ReceiptErrorKind::SchemaViolation,
                "synthetic receipts cannot claim a benchmark-owner official score",
            ));
        }
        let official_source_sha256 =
            sha256_field(source_output, "official_score.source_output_sha256")?;
        if official_source_sha256 != benchmark_grader_output_sha256 {
            return Err(error(
                ReceiptErrorKind::DigestMismatch,
                "official_score.source_output_sha256 must match benchmark.grader_output_sha256",
            ));
        }
        if score_value.is_null() {
            if !numerator.is_null() || !denominator.is_null() {
                return Err(error(
                    ReceiptErrorKind::SchemaViolation,
                    "unavailable official score requires null numerator and denominator",
                ));
            }
            string(
                unavailable_reason,
                "official_score.unavailable_reason",
                false,
            )?;
        } else {
            if score_value
                .as_f64()
                .is_none_or(|number| !number.is_finite())
            {
                return Err(error(
                    ReceiptErrorKind::SchemaViolation,
                    "official_score.value must be finite or null",
                ));
            }
            if denominator.is_null()
                || denominator
                    .as_f64()
                    .is_none_or(|number| !number.is_finite() || number <= 0.0)
            {
                return Err(error(
                    ReceiptErrorKind::SchemaViolation,
                    "official_score.denominator must be positive and finite for a reported score",
                ));
            }
            if !numerator.is_null() && numerator.as_f64().is_none_or(|number| !number.is_finite()) {
                return Err(error(
                    ReceiptErrorKind::SchemaViolation,
                    "official_score.numerator must be finite or null",
                ));
            }
            if !unavailable_reason.is_null() {
                return Err(error(
                    ReceiptErrorKind::SchemaViolation,
                    "official_score.unavailable_reason must be null for a reported score",
                ));
            }
        }
    } else {
        if !score_value.is_null()
            || !numerator.is_null()
            || !denominator.is_null()
            || !source_output.is_null()
        {
            return Err(error(
                ReceiptErrorKind::SchemaViolation,
                "a non-official score record must not contain an official value, numerator, denominator, or grader-output digest",
            ));
        }
        string(
            unavailable_reason,
            "official_score.unavailable_reason",
            false,
        )?;
    }

    let mut seen_metric_ids = HashSet::new();
    for (index, item) in arr(derived, "derived_metrics")?.iter().enumerate() {
        let context = format!("derived_metrics[{index}]");
        let map = obj(item, &context)?;
        strict_keys(
            map,
            &[
                "metric_id",
                "value",
                "unit",
                "formula",
                "is_official",
                "unavailable_reason",
            ],
            &[],
            &context,
        )?;
        let metric_id = string(
            required(map, "metric_id", &context)?,
            &format!("{context}.metric_id"),
            false,
        )?;
        if !seen_metric_ids.insert(metric_id.to_owned()) {
            return Err(error(
                ReceiptErrorKind::SchemaViolation,
                format!("duplicate derived metric id '{metric_id}'"),
            ));
        }
        let unit = string(
            required(map, "unit", &context)?,
            &format!("{context}.unit"),
            false,
        )?;
        let formula = string(
            required(map, "formula", &context)?,
            &format!("{context}.formula"),
            false,
        )?;
        if bool_value(
            required(map, "is_official", &context)?,
            &format!("{context}.is_official"),
        )? {
            return Err(error(
                ReceiptErrorKind::SchemaViolation,
                format!("{context}.is_official must be false"),
            ));
        }
        let metric_value = required(map, "value", &context)?;
        let reason = required(map, "unavailable_reason", &context)?;
        if metric_value.is_null() {
            string(reason, &format!("{context}.unavailable_reason"), false)?;
        } else {
            if metric_value
                .as_f64()
                .is_none_or(|number| !number.is_finite())
            {
                return Err(error(
                    ReceiptErrorKind::SchemaViolation,
                    format!("{context}.value must be finite or null"),
                ));
            }
            if !reason.is_null() {
                return Err(error(
                    ReceiptErrorKind::SchemaViolation,
                    format!("{context}.unavailable_reason must be null for measured values"),
                ));
            }
        }

        // Recompute common coverage/yield metrics from the validated denominator.
        // Caller-supplied formulas and values are not accepted as their own oracle.
        let known = match metric_id {
            "eligible-coverage" => {
                Some((counts.attempted, counts.eligible, "attempted / eligible"))
            }
            "graded-coverage" => Some((counts.graded, counts.eligible, "graded / eligible")),
            "coverage-adjusted-task-yield" => {
                Some((counts.passed, counts.eligible, "passed / eligible"))
            }
            "conditional-success" => Some((counts.passed, counts.graded, "passed / graded")),
            _ => None,
        };
        if let Some((numerator, denominator, expected_formula)) = known {
            if formula != expected_formula || unit != "fraction" {
                return Err(error(
                    ReceiptErrorKind::SchemaViolation,
                    format!("{context} does not use the canonical formula/unit for '{metric_id}'"),
                ));
            }
            if denominator == 0 {
                if !metric_value.is_null() || reason.is_null() {
                    return Err(error(
                        ReceiptErrorKind::AccountingMismatch,
                        format!(
                            "{context} must be null with a reason when its denominator is zero"
                        ),
                    ));
                }
            } else {
                let expected = numerator as f64 / denominator as f64;
                let actual = metric_value.as_f64().ok_or_else(|| {
                    error(
                        ReceiptErrorKind::AccountingMismatch,
                        format!("{context}.value does not match task accounting"),
                    )
                })?;
                if (actual - expected).abs() > 1e-12 || !reason.is_null() {
                    return Err(error(
                        ReceiptErrorKind::AccountingMismatch,
                        format!("{context}.value does not match recomputed task accounting"),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Validate a parsed receipt and all internal count/digest/effect invariants.
pub fn verify_receipt_value(value: &Value) -> Result<VerifiedReceipt, ReceiptError> {
    let top = obj(value, "receipt")?;
    strict_keys(top, TOP_LEVEL_KEYS, &[], "receipt")?;
    if string(
        required(top, "schema_version", "receipt")?,
        "schema_version",
        false,
    )? != "luminous.benchmark-run-receipt.v1"
    {
        return Err(error(
            ReceiptErrorKind::SchemaViolation,
            "unsupported schema_version",
        ));
    }
    let run_id = string(required(top, "run_id", "receipt")?, "run_id", false)?.to_owned();
    let evaluation_started_at = validate_utc_timestamp(
        required(top, "evaluation_started_at", "receipt")?,
        "evaluation_started_at",
    )?;
    let evaluation_completed_at = validate_utc_timestamp(
        required(top, "evaluation_completed_at", "receipt")?,
        "evaluation_completed_at",
    )?;
    let generated_at =
        validate_utc_timestamp(required(top, "generated_at", "receipt")?, "generated_at")?;
    if evaluation_started_at > evaluation_completed_at || evaluation_completed_at > generated_at {
        return Err(error(
            ReceiptErrorKind::AccountingMismatch,
            "timestamps must satisfy evaluation_started_at <= evaluation_completed_at <= generated_at",
        ));
    }

    let benchmark = obj(required(top, "benchmark", "receipt")?, "benchmark")?;
    strict_keys(
        benchmark,
        &[
            "id",
            "version",
            "revision",
            "dataset_manifest_sha256",
            "access_class",
            "scoring_formula_id",
            "grader_version",
            "grader_output_sha256",
        ],
        &[],
        "benchmark",
    )?;
    for key in [
        "id",
        "version",
        "revision",
        "scoring_formula_id",
        "grader_version",
    ] {
        validate_string_field(benchmark, key, "benchmark")?;
    }
    for key in ["dataset_manifest_sha256", "grader_output_sha256"] {
        validate_hash_field(benchmark, key, "benchmark")?;
    }
    let access = string(
        required(benchmark, "access_class", "benchmark")?,
        "benchmark.access_class",
        false,
    )?;
    let benchmark_grader_output_sha256 = sha256_field(
        required(benchmark, "grader_output_sha256", "benchmark")?,
        "benchmark.grader_output_sha256",
    )?;
    if !["public", "authorized_private", "synthetic", "internal"].contains(&access) {
        return Err(error(
            ReceiptErrorKind::SchemaViolation,
            "benchmark.access_class is unsupported",
        ));
    }

    let system = obj(required(top, "system", "receipt")?, "system")?;
    strict_keys(
        system,
        &[
            "arm",
            "name",
            "version",
            "source_revision",
            "executable_sha256",
            "model_id",
            "config_sha256",
            "prompt_sha256",
        ],
        &[],
        "system",
    )?;
    for key in ["name", "version", "source_revision"] {
        validate_string_field(system, key, "system")?;
    }
    for key in ["executable_sha256", "config_sha256", "prompt_sha256"] {
        validate_hash_field(system, key, "system")?;
    }
    let arm = string(required(system, "arm", "system")?, "system.arm", false)?;
    if ![
        "REA_only",
        "Symthaea_only",
        "REA_plus_Symthaea",
        "specialist_tools",
        "combined_stack",
        "human",
    ]
    .contains(&arm)
    {
        return Err(error(
            ReceiptErrorKind::SchemaViolation,
            "system.arm is unsupported",
        ));
    }
    let model_id = required(system, "model_id", "system")?;
    if !model_id.is_null() {
        string(model_id, "system.model_id", false)?;
    }

    validate_target_scope(required(top, "target_scope", "receipt")?)?;
    let runner = obj(required(top, "runner", "receipt")?, "runner")?;
    strict_keys(
        runner,
        &[
            "image_sha256",
            "host_os",
            "host_architecture",
            "network_policy",
        ],
        &[],
        "runner",
    )?;
    validate_hash_field(runner, "image_sha256", "runner")?;
    validate_string_field(runner, "host_os", "runner")?;
    validate_string_field(runner, "host_architecture", "runner")?;
    let network = string(
        required(runner, "network_policy", "runner")?,
        "runner.network_policy",
        false,
    )?;
    if ![
        "disabled",
        "allowlisted",
        "unrestricted",
        "benchmark_managed",
    ]
    .contains(&network)
    {
        return Err(error(
            ReceiptErrorKind::SchemaViolation,
            "runner.network_policy is unsupported",
        ));
    }

    let eligible_task_set_sha = sha256_field(
        required(top, "eligible_task_set_sha256", "receipt")?,
        "eligible_task_set_sha256",
    )?;
    let task_universe_sha = sha256_field(
        required(top, "task_universe_sha256", "receipt")?,
        "task_universe_sha256",
    )?;
    let (eligible_ids, canonical_rows, universe_count, excluded_count) = validate_task_universe(
        required(top, "task_universe", "receipt")?,
        evaluation_started_at,
    )?;
    if eligible_ids.is_empty() {
        return Err(error(
            ReceiptErrorKind::SchemaViolation,
            "receipt must contain at least one eligible task",
        ));
    }
    let eligible_value = Value::Array(eligible_ids.iter().cloned().map(Value::String).collect());
    if canonical_digest(&eligible_value)? != eligible_task_set_sha {
        return Err(error(
            ReceiptErrorKind::DigestMismatch,
            "eligible_task_set_sha256 does not match sorted eligible task IDs",
        ));
    }
    if canonical_digest(&Value::Array(canonical_rows))? != task_universe_sha {
        return Err(error(
            ReceiptErrorKind::DigestMismatch,
            "task_universe_sha256 does not match sorted task-universe rows",
        ));
    }

    let grading_mode = string(
        required(top, "grading_mode", "receipt")?,
        "grading_mode",
        false,
    )?;
    if !["binary", "numeric", "categorical", "custom"].contains(&grading_mode) {
        return Err(error(
            ReceiptErrorKind::SchemaViolation,
            "grading_mode is unsupported",
        ));
    }
    let grader_version = string(
        required(benchmark, "grader_version", "benchmark")?,
        "benchmark.grader_version",
        false,
    )?;
    let (mut computed, task_usage) = validate_task_results(
        required(top, "task_results", "receipt")?,
        &eligible_ids,
        grading_mode,
        grader_version,
    )?;
    computed.universe = universe_count;
    computed.excluded = excluded_count;
    validate_accounting(required(top, "accounting", "receipt")?, computed)?;

    let comparison = required(top, "comparison", "receipt")?;
    if !comparison.is_null() {
        let map = obj(comparison, "comparison")?;
        strict_keys(
            map,
            &["group_id", "protocol_sha256", "eligible_task_set_sha256"],
            &[],
            "comparison",
        )?;
        string(
            required(map, "group_id", "comparison")?,
            "comparison.group_id",
            false,
        )?;
        sha256_field(
            required(map, "protocol_sha256", "comparison")?,
            "comparison.protocol_sha256",
        )?;
        let linked = sha256_field(
            required(map, "eligible_task_set_sha256", "comparison")?,
            "comparison.eligible_task_set_sha256",
        )?;
        if linked != eligible_task_set_sha {
            return Err(error(
                ReceiptErrorKind::NonComparable,
                "comparison eligible-task digest does not match this receipt",
            ));
        }
    }

    validate_resource_objects(
        required(top, "budget_caps", "receipt")?,
        required(top, "actual_usage", "receipt")?,
        required(top, "unavailable_usage", "receipt")?,
        task_usage,
    )?;
    validate_official_and_derived(
        required(top, "official_score", "receipt")?,
        required(top, "derived_metrics", "receipt")?,
        access,
        benchmark_grader_output_sha256,
        computed,
    )?;

    let provenance = obj(required(top, "provenance", "receipt")?, "provenance")?;
    strict_keys(
        provenance,
        &[
            "input_manifest_sha256",
            "output_manifest_sha256",
            "transcript_sha256",
        ],
        &[],
        "provenance",
    )?;
    for key in [
        "input_manifest_sha256",
        "output_manifest_sha256",
        "transcript_sha256",
    ] {
        validate_hash_field(provenance, key, "provenance")?;
    }

    Ok(VerifiedReceipt {
        run_id,
        raw_file_sha256: String::new(),
        canonical_receipt_sha256: canonical_digest(value)?,
        value: value.clone(),
        accounting: computed.as_map(),
    })
}

/// Require matching benchmark, dataset, target, runner, budget, protocol and
/// eligible-task universe across two or more comparison-arm receipts.
pub fn validate_comparable_arms(receipts: &[VerifiedReceipt]) -> Result<(), ReceiptError> {
    if receipts.len() < 2 {
        return Err(error(
            ReceiptErrorKind::NonComparable,
            "at least two receipts are required",
        ));
    }
    let first = &receipts[0].value;
    let first_top = obj(first, "receipt")?;
    let first_comparison = required(first_top, "comparison", "receipt")?;
    if first_comparison.is_null() {
        return Err(error(
            ReceiptErrorKind::NonComparable,
            "comparison metadata is null",
        ));
    }
    let first_benchmark = obj(required(first_top, "benchmark", "receipt")?, "benchmark")?;
    let first_grading_mode = required(first_top, "grading_mode", "receipt")?;
    let first_task_inputs = task_input_digests(first_top)?;
    for receipt in receipts.iter().skip(1) {
        let top = obj(&receipt.value, "receipt")?;
        let benchmark = obj(required(top, "benchmark", "receipt")?, "benchmark")?;
        for field in [
            "id",
            "version",
            "revision",
            "dataset_manifest_sha256",
            "access_class",
            "scoring_formula_id",
            "grader_version",
        ] {
            if required(benchmark, field, "benchmark")?
                != required(first_benchmark, field, "benchmark")?
            {
                return Err(error(
                    ReceiptErrorKind::NonComparable,
                    format!(
                        "run '{}' differs in benchmark field '{field}'",
                        receipt.run_id
                    ),
                ));
            }
        }
        for field in [
            "target_scope",
            "runner",
            "budget_caps",
            "eligible_task_set_sha256",
            "task_universe_sha256",
            "comparison",
            "grading_mode",
        ] {
            if required(top, field, "receipt")? != required(first_top, field, "receipt")? {
                return Err(error(
                    ReceiptErrorKind::NonComparable,
                    format!(
                        "run '{}' differs in comparison field '{field}'",
                        receipt.run_id
                    ),
                ));
            }
        }
        if required(top, "grading_mode", "receipt")? != first_grading_mode {
            return Err(error(
                ReceiptErrorKind::NonComparable,
                format!("run '{}' differs in grading_mode", receipt.run_id),
            ));
        }
        if task_input_digests(top)? != first_task_inputs {
            return Err(error(
                ReceiptErrorKind::NonComparable,
                format!(
                    "run '{}' does not use the same task input digests",
                    receipt.run_id
                ),
            ));
        }
    }
    Ok(())
}

fn task_input_digests(top: &Map<String, Value>) -> Result<BTreeMap<String, String>, ReceiptError> {
    let rows = arr(required(top, "task_results", "receipt")?, "task_results")?;
    let mut inputs = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        let context = format!("task_results[{index}]");
        let map = obj(row, &context)?;
        let id = string(
            required(map, "task_id", &context)?,
            &format!("{context}.task_id"),
            false,
        )?
        .to_owned();
        let digest = sha256_field(
            required(map, "input_sha256", &context)?,
            &format!("{context}.input_sha256"),
        )?
        .to_owned();
        if inputs.insert(id.clone(), digest).is_some() {
            return Err(error(
                ReceiptErrorKind::AccountingMismatch,
                format!("duplicate task input digest row for '{id}'"),
            ));
        }
    }
    Ok(inputs)
}

/// Parse raw JSON with duplicate-key rejection and return separate raw and
/// canonical hashes.
pub fn verify_receipt_json(input: &[u8]) -> Result<VerifiedReceipt, ReceiptError> {
    if input.len() > MAX_RECEIPT_BYTES {
        return Err(error(
            ReceiptErrorKind::InputTooLarge,
            format!(
                "receipt is {} bytes; maximum is {}",
                input.len(),
                MAX_RECEIPT_BYTES
            ),
        ));
    }
    let mut de = serde_json::Deserializer::from_slice(input);
    let parsed = StrictJsonValue::deserialize(&mut de).map_err(|e| {
        error(
            ReceiptErrorKind::InvalidJson,
            format!("receipt JSON parse failed: {e}"),
        )
    })?;
    de.end().map_err(|e| {
        error(
            ReceiptErrorKind::InvalidJson,
            format!("receipt JSON has trailing data: {e}"),
        )
    })?;
    let raw_hash = sha256_hex(input);
    let mut verified = verify_receipt_value(&parsed.0)?;
    verified.raw_file_sha256 = raw_hash;
    Ok(verified)
}

struct StrictJsonValue(Value);

impl<'de> Deserialize<'de> for StrictJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = StrictJsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value with unique object keys")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(StrictJsonValue(Value::Null))
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(StrictJsonValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(StrictJsonValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(StrictJsonValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        let number =
            Number::from_f64(value).ok_or_else(|| E::custom("JSON numbers must be finite"))?;
        Ok(StrictJsonValue(Value::Number(number)))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(StrictJsonValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(StrictJsonValue(Value::String(value)))
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = seq.next_element::<StrictJsonValue>()? {
            values.push(value.0);
        }
        Ok(StrictJsonValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        let mut seen = HashSet::new();
        while let Some((key, value)) = map.next_entry::<String, StrictJsonValue>()? {
            if !seen.insert(key.clone()) {
                return Err(serde::de::Error::custom(format!(
                    "duplicate JSON object key '{key}'"
                )));
            }
            values.insert(key, value.0);
        }
        Ok(StrictJsonValue(Value::Object(values)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn base_receipt() -> Value {
        let task_universe = json!([
            {"task_id":"task-a","eligibility":"eligible","exclusion":null},
            {"task_id":"task-b","eligibility":"eligible","exclusion":null},
            {"task_id":"task-excluded","eligibility":"excluded","exclusion":{
                "predeclared":true,"reason":"not a mobile cohort fixture","declared_at":"2026-11-01T00:00:00Z"
            }}
        ]);
        let eligible = json!(["task-a", "task-b"]);
        let universe_digest = canonical_digest(&json!([
            {"task_id":"task-a","eligibility":"eligible","exclusion":null},
            {"task_id":"task-b","eligibility":"eligible","exclusion":null},
            {"task_id":"task-excluded","eligibility":"excluded","exclusion":{
                "predeclared":true,"reason":"not a mobile cohort fixture","declared_at":"2026-11-01T00:00:00Z"
            }}
        ])).unwrap();
        let eligible_digest = canonical_digest(&eligible).unwrap();
        json!({
            "schema_version":"luminous.benchmark-run-receipt.v1",
            "run_id":"synthetic-run-001",
            "generated_at":"2026-11-03T00:01:01Z",
            "evaluation_started_at":"2026-11-03T00:00:00Z",
            "evaluation_completed_at":"2026-11-03T00:01:00Z",
            "benchmark":{
                "id":"synthetic-re-bench","version":"0.1","revision":"fixture-1",
                "dataset_manifest_sha256":A,"access_class":"synthetic",
                "scoring_formula_id":"binary-pass-rate-v1","grader_version":"fixture-grader-1",
                "grader_output_sha256":B
            },
            "system":{
                "arm":"REA_plus_Symthaea","name":"synthetic-system","version":"0.1",
                "source_revision":"0123456789abcdef","executable_sha256":A,"model_id":"synthetic-model",
                "config_sha256":B,"prompt_sha256":C
            },
            "target_scope":{"platform":"android","os":"synthetic","architecture":"x86_64","cohort":"emulator"},
            "runner":{"image_sha256":A,"host_os":"linux","host_architecture":"x86_64","network_policy":"disabled"},
            "comparison":{"group_id":"synthetic-ablation-1","protocol_sha256":C,"eligible_task_set_sha256":eligible_digest},
            "task_universe":task_universe,
            "eligible_task_set_sha256":eligible_digest,"task_universe_sha256":universe_digest,
            "task_results":[
                {"task_id":"task-a","input_sha256":A,"attempt_state":"terminal","attempts":1,"terminal_disposition":"passed",
                 "grade":{"kind":"binary","value":true,"grader_version":"fixture-grader-1","grader_output_sha256":B},
                 "tool_calls":2,"model_calls":1,"wall_time_ms":100,"output_sha256":[C]},
                {"task_id":"task-b","input_sha256":B,"attempt_state":"not_attempted","attempts":0,"terminal_disposition":null,
                 "grade":null,"tool_calls":0,"model_calls":0,"wall_time_ms":0,"output_sha256":[]}
            ],
            "grading_mode":"binary",
            "accounting":{
                "universe_count":3,"eligible_count":2,"excluded_count":1,
                "attempted_count":1,"not_attempted_count":1,"in_progress_count":0,"completed_count":1,
                "graded_count":1,"passed_count":1,"failed_count":0,"refused_count":0,"abstained_count":0,
                "unsupported_count":0,"timed_out_count":0,"tool_failed_count":0,"ungraded_count":0
            },
            "budget_caps":{"wall_time_ms":1000,"cpu_time_ms":1000,"peak_memory_bytes":1000000,"model_calls":10,"input_tokens":10000,"output_tokens":10000,"tool_calls":20},
            "actual_usage":{"wall_time_ms":100,"cpu_time_ms":50,"peak_memory_bytes":10000,"model_calls":1,"input_tokens":100,"output_tokens":50,"tool_calls":2},
            "unavailable_usage":{},
            "official_score":{"metric_id":"benchmark-owner-official-score","formula_version":"not-applicable-synthetic","value":null,"unit":"fraction","numerator":null,"denominator":null,"is_official":false,"source_output_sha256":null,"unavailable_reason":"Synthetic fixture has no benchmark-owner official score."},
            "derived_metrics":[
                {"metric_id":"coverage-adjusted-task-yield","value":0.5,"unit":"fraction","formula":"passed / eligible","is_official":false,"unavailable_reason":null}
            ],
            "provenance":{"input_manifest_sha256":A,"output_manifest_sha256":B,"transcript_sha256":C}
        })
    }

    fn refresh_task_digests(value: &mut Value) {
        // Own the task universe before mutating sibling receipt fields. This
        // keeps the digest inputs stable and avoids aliasing the mutable root.
        let tasks = value["task_universe"].as_array().unwrap().clone();
        let mut eligible: Vec<String> = tasks
            .iter()
            .filter(|task| task["eligibility"] == "eligible")
            .map(|task| task["task_id"].as_str().unwrap().to_owned())
            .collect();
        eligible.sort();
        let eligible_digest = canonical_digest(&json!(eligible)).unwrap();

        let mut sorted = tasks;
        sorted.sort_by(|left, right| left["task_id"].as_str().cmp(&right["task_id"].as_str()));
        let universe_digest = canonical_digest(&Value::Array(sorted)).unwrap();

        value["eligible_task_set_sha256"] = json!(eligible_digest);
        value["task_universe_sha256"] = json!(universe_digest);
        value["comparison"]["eligible_task_set_sha256"] = json!(eligible_digest);
    }

    #[test]
    fn accepts_valid_synthetic_receipt_and_separates_raw_from_canonical_digest() {
        let bytes = serde_json::to_vec_pretty(&base_receipt()).unwrap();
        let verified = verify_receipt_json(&bytes).unwrap();
        assert!(!verified.raw_file_sha256.is_empty());
        assert!(!verified.canonical_receipt_sha256.is_empty());
        assert_ne!(verified.raw_file_sha256, verified.canonical_receipt_sha256);
        assert_eq!(verified.accounting["eligible_count"], 2);
    }

    #[test]
    fn rejects_hidden_eligible_task_without_result_row() {
        let mut receipt = base_receipt();
        receipt["task_results"].as_array_mut().unwrap().pop();
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::AccountingMismatch
        );
    }

    #[test]
    fn rejects_duplicate_task_universe_ids() {
        let mut receipt = base_receipt();
        let duplicate = receipt["task_universe"][0].clone();
        receipt["task_universe"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::AccountingMismatch
        );
    }

    #[test]
    fn rejects_exclusion_not_predeclared() {
        let mut receipt = base_receipt();
        receipt["task_universe"][2]["exclusion"]["predeclared"] = json!(false);
        refresh_task_digests(&mut receipt);
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::AccountingMismatch
        );
    }

    #[test]
    fn rejects_accounting_denominator_laundering() {
        let mut receipt = base_receipt();
        receipt["accounting"]["eligible_count"] = json!(1);
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::AccountingMismatch
        );
    }

    #[test]
    fn rejects_task_result_for_excluded_id() {
        let mut receipt = base_receipt();
        let mut wrong = receipt["task_results"][0].clone();
        wrong["task_id"] = json!("task-excluded");
        receipt["task_results"][0] = wrong;
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::AccountingMismatch
        );
    }

    #[test]
    fn rejects_binary_grade_conflicting_with_pass_label() {
        let mut receipt = base_receipt();
        receipt["task_results"][0]["grade"]["value"] = json!(false);
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::AccountingMismatch
        );
    }

    #[test]
    fn rejects_actual_usage_over_budget() {
        let mut receipt = base_receipt();
        receipt["actual_usage"]["tool_calls"] = json!(21);
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::BudgetExceeded
        );
    }

    #[test]
    fn requires_reason_for_unavailable_usage() {
        let mut receipt = base_receipt();
        receipt["actual_usage"]["cpu_time_ms"] = Value::Null;
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::SchemaViolation
        );
    }

    #[test]
    fn accepts_reason_for_unavailable_usage() {
        let mut receipt = base_receipt();
        receipt["actual_usage"]["cpu_time_ms"] = Value::Null;
        receipt["unavailable_usage"]["cpu_time_ms"] = json!("runner did not expose CPU accounting");
        assert!(verify_receipt_value(&receipt).is_ok());
    }

    #[test]
    fn rejects_synthetic_receipts_claiming_an_official_score() {
        let mut receipt = base_receipt();
        receipt["official_score"]["is_official"] = json!(true);
        receipt["official_score"]["value"] = json!(1.0);
        receipt["official_score"]["numerator"] = json!(1);
        receipt["official_score"]["denominator"] = json!(1);
        receipt["official_score"]["source_output_sha256"] = json!(C);
        receipt["official_score"]["unavailable_reason"] = Value::Null;
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::SchemaViolation
        );
    }

    #[test]
    fn accepts_official_score_only_for_non_synthetic_receipt() {
        let mut receipt = base_receipt();
        receipt["benchmark"]["access_class"] = json!("public");
        receipt["official_score"]["is_official"] = json!(true);
        receipt["official_score"]["value"] = json!(1.0);
        receipt["official_score"]["numerator"] = json!(1);
        receipt["official_score"]["denominator"] = json!(1);
        receipt["official_score"]["source_output_sha256"] = json!(B);
        receipt["official_score"]["unavailable_reason"] = Value::Null;
        assert!(verify_receipt_value(&receipt).is_ok());
    }

    #[test]
    fn rejects_official_score_from_a_different_grader_output_artifact() {
        let mut receipt = base_receipt();
        receipt["benchmark"]["access_class"] = json!("public");
        receipt["official_score"]["is_official"] = json!(true);
        receipt["official_score"]["value"] = json!(1.0);
        receipt["official_score"]["numerator"] = json!(1);
        receipt["official_score"]["denominator"] = json!(1);
        receipt["official_score"]["source_output_sha256"] = json!(C);
        receipt["official_score"]["unavailable_reason"] = Value::Null;
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::DigestMismatch
        );
    }

    #[test]
    fn permits_official_score_unavailable_without_fabricating_zero_percent() {
        let mut receipt = base_receipt();
        receipt["official_score"]["value"] = Value::Null;
        receipt["official_score"]["numerator"] = Value::Null;
        receipt["official_score"]["denominator"] = Value::Null;
        receipt["official_score"]["unavailable_reason"] =
            json!("no tasks reached the benchmark grader");
        assert!(verify_receipt_value(&receipt).is_ok());
    }

    #[test]
    fn rejects_official_score_flag_on_derived_metric() {
        let mut receipt = base_receipt();
        receipt["derived_metrics"][0]["is_official"] = json!(true);
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::SchemaViolation
        );
    }

    #[test]
    fn rejects_falsified_known_coverage_metric() {
        let mut receipt = base_receipt();
        receipt["derived_metrics"] = json!([
            {
                "metric_id":"coverage-adjusted-task-yield",
                "value":0.6,
                "unit":"fraction",
                "formula":"passed / eligible",
                "is_official":false,
                "unavailable_reason":null
            }
        ]);
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::AccountingMismatch
        );
    }

    #[test]
    fn rejects_unknown_top_level_fields() {
        let mut receipt = base_receipt();
        receipt["unreviewed"] = json!(true);
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::SchemaViolation
        );
    }

    #[test]
    fn rejects_duplicate_json_object_keys() {
        let err = verify_receipt_json(br#"{"run_id":"a","run_id":"b"}"#).unwrap_err();
        assert_eq!(err.kind, ReceiptErrorKind::InvalidJson);
    }

    #[test]
    fn rejects_comparison_protocol_with_different_task_set() {
        let mut receipt = base_receipt();
        receipt["comparison"]["eligible_task_set_sha256"] = json!(A);
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::NonComparable
        );
    }

    #[test]
    fn rejects_invalid_canonical_task_set_digest() {
        let mut receipt = base_receipt();
        receipt["eligible_task_set_sha256"] = json!(A);
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::DigestMismatch
        );
    }

    #[test]
    fn comparison_arms_must_match_dataset_scope_protocol_and_budgets() {
        let first = verify_receipt_value(&base_receipt()).unwrap();
        let mut second_value = base_receipt();
        second_value["run_id"] = json!("synthetic-run-002");
        second_value["system"]["arm"] = json!("REA_only");
        let second = verify_receipt_value(&second_value).unwrap();
        assert!(validate_comparable_arms(&[first.clone(), second.clone()]).is_ok());

        second_value["budget_caps"]["tool_calls"] = json!(21);
        let third = verify_receipt_value(&second_value).unwrap();
        assert_eq!(
            validate_comparable_arms(&[first, third]).unwrap_err().kind,
            ReceiptErrorKind::NonComparable
        );
    }

    #[test]
    fn rejects_task_grade_with_wrong_pinned_grader_version() {
        let mut receipt = base_receipt();
        receipt["task_results"][0]["grade"]["grader_version"] = json!("different-grader");
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::NonComparable
        );
    }

    #[test]
    fn rejects_run_level_tool_usage_below_task_level_sum() {
        let mut receipt = base_receipt();
        receipt["actual_usage"]["tool_calls"] = json!(1);
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::AccountingMismatch
        );
    }

    #[test]
    fn rejects_matched_arms_with_same_ids_but_different_input_bytes() {
        let first = verify_receipt_value(&base_receipt()).unwrap();
        let mut second_value = base_receipt();
        second_value["run_id"] = json!("synthetic-run-003");
        second_value["system"]["arm"] = json!("REA_only");
        second_value["task_results"][0]["input_sha256"] = json!(C);
        let second = verify_receipt_value(&second_value).unwrap();
        assert_eq!(
            validate_comparable_arms(&[first, second]).unwrap_err().kind,
            ReceiptErrorKind::NonComparable
        );
    }

    #[test]
    fn rejects_excluded_task_without_reason() {
        let mut receipt = base_receipt();
        receipt["task_universe"][2]["exclusion"]["reason"] = Value::Null;
        refresh_task_digests(&mut receipt);
        assert_eq!(
            verify_receipt_value(&receipt).unwrap_err().kind,
            ReceiptErrorKind::SchemaViolation
        );
    }
}
