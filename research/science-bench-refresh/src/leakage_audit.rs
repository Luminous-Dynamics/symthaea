//! Conservative cross-task-set overlap audit for scientific benchmark inputs.
//!
//! Exact policy-input collisions are blocking. Three-significant-digit numeric
//! fingerprints are triage-only candidates: they must never be reported as
//! confirmed leakage without lineage or domain-aware equivalence evidence.

use crate::{sha256_hex, TaskSetArtifactV1, TASK_SET_SCHEMA};
use serde::Serialize;
use serde_json::{Map, Number, Value};
use std::collections::{BTreeMap, BTreeSet};

const REPORT_SCHEMA: &str = "symthaea.science_bench_refresh_overlap_report.v1";
const MAX_TASKS_PER_SET: usize = 100_000;
const MAX_IDS_PER_GROUP: usize = 20;

#[derive(Debug, Clone, Serialize)]
pub struct OverlapGroupV1 {
    pub fingerprint_sha256: String,
    pub left_task_count: u64,
    pub right_task_count: u64,
    pub cross_set_pair_count: u64,
    pub left_task_ids_sample: Vec<String>,
    pub right_task_ids_sample: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OverlapAuditReportV1 {
    pub schema: String,
    pub valid: bool,
    pub left_split_label: String,
    pub right_split_label: String,
    pub left_generation_id: String,
    pub right_generation_id: String,
    pub left_task_set_sha256: String,
    pub right_task_set_sha256: String,
    pub left_task_count: u64,
    pub right_task_count: u64,
    pub exact_overlap_pair_count: u64,
    pub near_duplicate_candidate_pair_count: u64,
    pub exact_overlaps: Vec<OverlapGroupV1>,
    pub near_duplicate_candidates: Vec<OverlapGroupV1>,
    pub admission: String,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone)]
struct IndexedTask {
    task_id: String,
    exact_fingerprint: String,
    tolerant_fingerprint: String,
}

fn validate_split_label(label: &str) -> Result<(), String> {
    if label.is_empty()
        || label.len() > 128
        || !label.is_ascii()
        || !label.chars().all(|character| {
            character.is_ascii_alphanumeric() || character == '_' || character == '-' || character == ':'
        })
    {
        return Err(format!("invalid split label: {label}"));
    }

    let mut parts = label.split(':');
    let role = parts.next().unwrap_or_default();
    let suffix = parts.next();
    if parts.next().is_some() {
        return Err(format!("split label has too many separators: {label}"));
    }
    if !matches!(role, "development" | "anchor" | "refresh") {
        return Err(format!("unknown split role in label: {label}"));
    }
    if suffix.is_some_and(str::is_empty) {
        return Err(format!("split label suffix must not be empty: {label}"));
    }
    Ok(())
}

fn canonicalize(value: &Value, numeric_tolerant: bool) -> Result<Value, String> {
    match value {
        Value::Null => Ok(Value::Null),
        Value::Bool(value) => Ok(Value::Bool(*value)),
        Value::String(value) => Ok(Value::String(value.clone())),
        Value::Number(number) if !numeric_tolerant => Ok(Value::Number(canonical_exact_number(number)?)),
        Value::Number(number) => {
            let value = number.as_f64().ok_or_else(|| {
                "numeric fingerprint cannot represent a JSON number as f64".to_owned()
            })?;
            if !value.is_finite() {
                return Err("numeric fingerprint encountered a non-finite value".to_owned());
            }
            // Store the rounded number as a tagged string so subnormal values
            // and values near f64::MAX remain representable in the fingerprint.
            Ok(Value::String(bucket_three_significant_digits(value)?))
        }
        Value::Array(items) => items
            .iter()
            .map(|item| canonicalize(item, numeric_tolerant))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        Value::Object(fields) => {
            let mut keys: Vec<&String> = fields.keys().collect();
            keys.sort();
            let mut canonical = Map::new();
            for key in keys {
                canonical.insert(key.clone(), canonicalize(&fields[key], numeric_tolerant)?);
            }
            Ok(Value::Object(canonical))
        }
    }
}

fn canonical_exact_number(number: &Number) -> Result<Number, String> {
    if let Some(value) = number.as_i64() {
        return Ok(Number::from(value));
    }
    if let Some(value) = number.as_u64() {
        return Ok(Number::from(value));
    }
    let value = number
        .as_f64()
        .ok_or_else(|| "exact canonicalization cannot represent a JSON number".to_owned())?;
    // Normalize integral decimal spellings such as 1 and 1.0 without losing
    // precision for integers beyond the exactly representable f64 range.
    if value.is_finite()
        && value.fract() == 0.0
        && value.abs() <= 9_007_199_254_740_991.0
    {
        return Ok(Number::from(value as i64));
    }
    Ok(number.clone())
}

fn bucket_three_significant_digits(value: f64) -> Result<String, String> {
    if value == 0.0 {
        return Ok("number:0e0".to_owned());
    }
    let rounded = format!("number:{value:.2e}");
    if rounded.contains("NaN") || rounded.contains("inf") {
        return Err("numeric fingerprint produced a non-finite bucket".to_owned());
    }
    Ok(rounded)
}

fn fingerprint(value: &Value, numeric_tolerant: bool) -> Result<String, String> {
    let canonical = canonicalize(value, numeric_tolerant)?;
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| format!("cannot serialize canonical policy input: {error}"))?;
    Ok(sha256_hex(&bytes))
}

fn load_task_set(bytes: &[u8], label: &str) -> Result<TaskSetArtifactV1, String> {
    let task_set: TaskSetArtifactV1 = serde_json::from_slice(bytes)
        .map_err(|error| format!("{label} task set does not match strict v1 shape: {error}"))?;
    if task_set.schema != TASK_SET_SCHEMA {
        return Err(format!("{label} task set schema must be {TASK_SET_SCHEMA}"));
    }
    if task_set.generation_id.trim().is_empty() {
        return Err(format!("{label} generation_id must not be empty"));
    }
    if task_set.tasks.is_empty() {
        return Err(format!("{label} task set must not be empty"));
    }
    if task_set.tasks.len() > MAX_TASKS_PER_SET {
        return Err(format!(
            "{label} task set exceeds safety limit of {MAX_TASKS_PER_SET} tasks"
        ));
    }

    let mut seen_ids = BTreeSet::new();
    for task in &task_set.tasks {
        if task.task_id.trim().is_empty() {
            return Err(format!("{label} contains an empty task_id"));
        }
        if !seen_ids.insert(task.task_id.as_str()) {
            return Err(format!("{label} contains duplicate task_id {}", task.task_id));
        }
        if task.family_id.trim().is_empty() {
            return Err(format!("{label} contains an empty family_id"));
        }
        if !task.policy_input.is_object() {
            return Err(format!("{label} task {} policy_input must be an object", task.task_id));
        }
        if !task.evaluator_only.is_object() {
            return Err(format!("{label} task {} evaluator_only must be an object", task.task_id));
        }
    }
    Ok(task_set)
}

fn index_tasks(task_set: &TaskSetArtifactV1) -> Result<Vec<IndexedTask>, String> {
    task_set
        .tasks
        .iter()
        .map(|task| {
            Ok(IndexedTask {
                task_id: task.task_id.clone(),
                exact_fingerprint: fingerprint(&task.policy_input, false)?,
                tolerant_fingerprint: fingerprint(&task.policy_input, true)?,
            })
        })
        .collect()
}

fn grouped<'a>(
    tasks: &'a [IndexedTask],
    select: impl Fn(&'a IndexedTask) -> &'a str,
) -> BTreeMap<String, Vec<&'a IndexedTask>> {
    let mut result = BTreeMap::<String, Vec<&IndexedTask>>::new();
    for task in tasks {
        result.entry(select(task).to_owned()).or_default().push(task);
    }
    for group in result.values_mut() {
        group.sort_by(|left, right| left.task_id.cmp(&right.task_id));
    }
    result
}

fn checked_pair_count(left: usize, right: usize) -> Result<u64, String> {
    let pairs = left
        .checked_mul(right)
        .ok_or_else(|| "overlap pair count overflows usize".to_owned())?;
    u64::try_from(pairs).map_err(|_| "overlap pair count exceeds u64".to_owned())
}

fn sample_ids<'a>(tasks: impl Iterator<Item = &'a IndexedTask>) -> Vec<String> {
    tasks
        .take(MAX_IDS_PER_GROUP)
        .map(|task| task.task_id.clone())
        .collect()
}

fn build_exact_groups(
    left: &[IndexedTask],
    right: &[IndexedTask],
) -> Result<(u64, Vec<OverlapGroupV1>), String> {
    let left_groups = grouped(left, |task| &task.exact_fingerprint);
    let right_groups = grouped(right, |task| &task.exact_fingerprint);
    let mut total = 0_u64;
    let mut groups = Vec::new();

    for (digest, left_tasks) in &left_groups {
        let Some(right_tasks) = right_groups.get(digest) else {
            continue;
        };
        let pairs = checked_pair_count(left_tasks.len(), right_tasks.len())?;
        total = total
            .checked_add(pairs)
            .ok_or_else(|| "total exact overlap count overflows u64".to_owned())?;
        groups.push(OverlapGroupV1 {
            fingerprint_sha256: digest.clone(),
            left_task_count: u64::try_from(left_tasks.len()).map_err(|_| "left count overflow")?,
            right_task_count: u64::try_from(right_tasks.len()).map_err(|_| "right count overflow")?,
            cross_set_pair_count: pairs,
            left_task_ids_sample: sample_ids(left_tasks.iter().copied()),
            right_task_ids_sample: sample_ids(right_tasks.iter().copied()),
        });
    }

    Ok((total, groups))
}

fn build_near_groups(
    left: &[IndexedTask],
    right: &[IndexedTask],
) -> Result<(u64, Vec<OverlapGroupV1>), String> {
    let left_groups = grouped(left, |task| &task.tolerant_fingerprint);
    let right_groups = grouped(right, |task| &task.tolerant_fingerprint);
    let mut total = 0_u64;
    let mut groups = Vec::new();

    for (digest, left_tasks) in &left_groups {
        let Some(right_tasks) = right_groups.get(digest) else {
            continue;
        };
        let all_pairs = checked_pair_count(left_tasks.len(), right_tasks.len())?;

        let mut left_exact_counts = BTreeMap::<&str, usize>::new();
        let mut right_exact_counts = BTreeMap::<&str, usize>::new();
        for task in left_tasks {
            *left_exact_counts.entry(&task.exact_fingerprint).or_default() += 1;
        }
        for task in right_tasks {
            *right_exact_counts.entry(&task.exact_fingerprint).or_default() += 1;
        }
        let mut exact_pairs = 0_u64;
        for (exact, left_count) in &left_exact_counts {
            if let Some(right_count) = right_exact_counts.get(exact) {
                exact_pairs = exact_pairs
                    .checked_add(checked_pair_count(*left_count, *right_count)?)
                    .ok_or_else(|| "exact pair count overflows u64".to_owned())?;
            }
        }
        let candidates = all_pairs
            .checked_sub(exact_pairs)
            .ok_or_else(|| "exact pairs exceeded tolerant pairs".to_owned())?;
        if candidates == 0 {
            continue;
        }
        total = total
            .checked_add(candidates)
            .ok_or_else(|| "total near-duplicate candidate count overflows u64".to_owned())?;

        let right_exact_set: BTreeSet<&str> = right_exact_counts.keys().copied().collect();
        let left_exact_set: BTreeSet<&str> = left_exact_counts.keys().copied().collect();
        let left_candidates: Vec<&IndexedTask> = left_tasks
            .iter()
            .copied()
            .filter(|task| right_exact_set.iter().any(|exact| *exact != task.exact_fingerprint.as_str()))
            .collect();
        let right_candidates: Vec<&IndexedTask> = right_tasks
            .iter()
            .copied()
            .filter(|task| left_exact_set.iter().any(|exact| *exact != task.exact_fingerprint))
            .collect();

        groups.push(OverlapGroupV1 {
            fingerprint_sha256: digest.clone(),
            left_task_count: u64::try_from(left_candidates.len()).map_err(|_| "left candidate count overflow")?,
            right_task_count: u64::try_from(right_candidates.len()).map_err(|_| "right candidate count overflow")?,
            cross_set_pair_count: candidates,
            left_task_ids_sample: sample_ids(left_candidates.into_iter()),
            right_task_ids_sample: sample_ids(right_candidates.into_iter()),
        });
    }

    Ok((total, groups))
}

/// Compare the policy-visible portions of two committed task-set artifacts.
///
/// Exact canonical policy-input matches block admission. Numeric-tolerant matches
/// are conservative triage candidates only. The split labels are caller-provided
/// report labels, not cryptographic evidence of a runner's actual isolation.
pub fn audit_policy_input_overlap(
    left_split_label: &str,
    left_bytes: &[u8],
    right_split_label: &str,
    right_bytes: &[u8],
) -> Result<OverlapAuditReportV1, String> {
    validate_split_label(left_split_label)?;
    validate_split_label(right_split_label)?;
    if left_split_label == right_split_label {
        return Err("left and right split labels must differ".to_owned());
    }

    let left_set = load_task_set(left_bytes, "left")?;
    let right_set = load_task_set(right_bytes, "right")?;
    let left_tasks = index_tasks(&left_set)?;
    let right_tasks = index_tasks(&right_set)?;
    let (exact_overlap_pair_count, exact_overlaps) =
        build_exact_groups(&left_tasks, &right_tasks)?;
    let (near_duplicate_candidate_pair_count, near_duplicate_candidates) =
        build_near_groups(&left_tasks, &right_tasks)?;

    let admission = if exact_overlap_pair_count > 0 {
        "blocked_exact_policy_input_overlap"
    } else if near_duplicate_candidate_pair_count > 0 {
        "manual_review_required_for_numeric_near_duplicates"
    } else {
        "no_detected_exact_or_numeric_tolerant_overlap"
    };

    Ok(OverlapAuditReportV1 {
        schema: REPORT_SCHEMA.to_owned(),
        valid: true,
        left_split_label: left_split_label.to_owned(),
        right_split_label: right_split_label.to_owned(),
        left_generation_id: left_set.generation_id,
        right_generation_id: right_set.generation_id,
        left_task_set_sha256: sha256_hex(left_bytes),
        right_task_set_sha256: sha256_hex(right_bytes),
        left_task_count: u64::try_from(left_tasks.len()).map_err(|_| "left task count overflow")?,
        right_task_count: u64::try_from(right_tasks.len()).map_err(|_| "right task count overflow")?,
        exact_overlap_pair_count,
        near_duplicate_candidate_pair_count,
        exact_overlaps,
        near_duplicate_candidates,
        admission: admission.to_owned(),
        issues: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn task_set(generation: &str, tasks: Value) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema": TASK_SET_SCHEMA,
            "generation_id": generation,
            "tasks": tasks
        }))
        .expect("fixture serializes")
    }

    fn task(id: &str, input: Value) -> Value {
        json!({
            "task_id": id,
            "family_id": "fixture-family",
            "policy_input": input,
            "evaluator_only": {"expected_law": "not emitted by overlap auditor"}
        })
    }

    fn input(x: f64, y: f64) -> Value {
        json!({
            "instruction": "Recover the relationship.",
            "observations": [{"x": x, "y": y}, {"x": x + 1.0, "y": y + 2.0}]
        })
    }

    #[test]
    fn exact_duplicate_is_detected_even_when_task_ids_and_generation_change() {
        let left = task_set("development-1", json!([task("dev-1", input(1.0, 2.0))]));
        let right = task_set("refresh-2", json!([task("new-id", input(1.0, 2.0))]));
        let report = audit_policy_input_overlap("development", &left, "refresh:g2", &right).unwrap();
        assert_eq!(report.exact_overlap_pair_count, 1);
        assert_eq!(report.near_duplicate_candidate_pair_count, 0);
        assert_eq!(report.admission, "blocked_exact_policy_input_overlap");
    }

    #[test]
    fn json_object_key_order_and_whitespace_do_not_change_exact_identity() {
        let left = br#"{"schema":"symthaea.science_bench_refresh_task_set.v1","generation_id":"d","tasks":[{"task_id":"1","family_id":"f","policy_input":{"instruction":"Recover","observations":[{"x":1,"y":2}]},"evaluator_only":{}}]}"#;
        let right = br#"{
          "schema": "symthaea.science_bench_refresh_task_set.v1",
          "generation_id": "r",
          "tasks": [{
            "task_id": "different-id",
            "family_id": "different-family",
            "policy_input": {"observations":[{"y":2.0,"x":1.0}],"instruction":"Recover"},
            "evaluator_only": {"expected_law":"hidden"}
          }]
        }"#;
        let report = audit_policy_input_overlap("development", left, "refresh:g2", right).unwrap();
        assert_eq!(report.exact_overlap_pair_count, 1);
    }

    #[test]
    fn close_numeric_inputs_are_review_candidates_not_confirmed_leaks() {
        let left = task_set("d", json!([task("a", input(1.23451, 2.34561))]));
        let right = task_set("r", json!([task("b", input(1.23452, 2.34562))]));
        let report = audit_policy_input_overlap("development", &left, "refresh:g2", &right).unwrap();
        assert_eq!(report.exact_overlap_pair_count, 0);
        assert_eq!(report.near_duplicate_candidate_pair_count, 1);
        assert_eq!(report.admission, "manual_review_required_for_numeric_near_duplicates");
    }

    #[test]
    fn distant_numeric_worlds_with_same_schema_are_not_flagged() {
        let left = task_set("d", json!([task("a", input(1.0, 2.0))]));
        let right = task_set("r", json!([task("b", input(100.0, 200.0))]));
        let report = audit_policy_input_overlap("development", &left, "refresh:g2", &right).unwrap();
        assert_eq!(report.exact_overlap_pair_count, 0);
        assert_eq!(report.near_duplicate_candidate_pair_count, 0);
        assert_eq!(report.admission, "no_detected_exact_or_numeric_tolerant_overlap");
    }

    #[test]
    fn reports_are_deterministic_and_do_not_disclose_evaluator_targets() {
        let left = task_set("d", json!([task("a", input(1.0, 2.0))]));
        let right = task_set("r", json!([task("b", input(1.0001, 2.0001))]));
        let first = audit_policy_input_overlap("development", &left, "refresh:g2", &right).unwrap();
        let second = audit_policy_input_overlap("development", &left, "refresh:g2", &right).unwrap();
        let first_bytes = serde_json::to_vec(&first).unwrap();
        let second_bytes = serde_json::to_vec(&second).unwrap();
        assert_eq!(first_bytes, second_bytes);
        assert!(!String::from_utf8(first_bytes).unwrap().contains("not emitted by overlap auditor"));
    }

    #[test]
    fn duplicate_task_ids_and_unknown_split_roles_fail_closed() {
        let duplicate_ids = task_set("d", json!([
            task("same", input(1.0, 2.0)),
            task("same", input(3.0, 4.0))
        ]));
        assert!(audit_policy_input_overlap("development", &duplicate_ids, "refresh:g2", &task_set("r", json!([task("x", input(9.0, 10.0))]))).is_err());
        assert!(audit_policy_input_overlap("dev", &task_set("d", json!([task("a", input(1.0, 2.0))])), "refresh:g2", &task_set("r", json!([task("b", input(9.0, 10.0))]))).is_err());
    }
}
