//! Data-only DynaMapBench corpus-manifest validation.
//!
//! This validator checks immutable identities, split isolation, evidence
//! cutoffs, no-gold semantics, gold span anchors and evaluation budgets. It
//! does not fetch repositories, authenticate Git objects, run agents or
//! execute repository-controlled code.

use std::collections::{BTreeMap, BTreeSet};

use chrono::DateTime;
use serde_json::{Map, Value};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestDiagnostic {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DynaMapBenchManifestReport {
    pub diagnostics: Vec<ManifestDiagnostic>,
}

impl DynaMapBenchManifestReport {
    pub fn is_valid(&self) -> bool {
        self.diagnostics.is_empty()
    }

    fn error(&mut self, code: &'static str, path: impl Into<String>, message: impl Into<String>) {
        self.diagnostics.push(ManifestDiagnostic {
            code,
            path: path.into(),
            message: message.into(),
        });
    }

    fn sort(&mut self) {
        self.diagnostics.sort_by(|a, b| {
            (&a.path, a.code, &a.message).cmp(&(&b.path, b.code, &b.message))
        });
    }
}

fn object<'a>(
    value: &'a Value,
    path: &str,
    report: &mut DynaMapBenchManifestReport,
) -> Option<&'a Map<String, Value>> {
    match value.as_object() {
        Some(value) => Some(value),
        None => {
            report.error("DMB-TYPE-001", path, "expected an object");
            None
        }
    }
}

fn required_string<'a>(
    value: &'a Map<String, Value>,
    key: &str,
    path: &str,
    report: &mut DynaMapBenchManifestReport,
) -> Option<&'a str> {
    match value.get(key).and_then(Value::as_str) {
        Some(text) if !text.trim().is_empty() => Some(text),
        Some(_) => {
            report.error("DMB-STRING-001", format!("{path}.{key}"), "must not be empty");
            None
        }
        None => {
            report.error("DMB-REQUIRED-001", format!("{path}.{key}"), "required non-empty string is missing");
            None
        }
    }
}

fn required_value<'a>(
    value: &'a Map<String, Value>,
    key: &str,
    path: &str,
    report: &mut DynaMapBenchManifestReport,
) -> Option<&'a Value> {
    match value.get(key) {
        Some(value) => Some(value),
        None => {
            report.error("DMB-REQUIRED-001", format!("{path}.{key}"), "required field is missing");
            None
        }
    }
}

fn allowed_keys(
    value: &Map<String, Value>,
    allowed: &[&str],
    path: &str,
    report: &mut DynaMapBenchManifestReport,
) {
    for key in value.keys() {
        if !allowed.contains(&key.as_str()) {
            report.error("DMB-UNKNOWN-FIELD-001", format!("{path}.{key}"), "field is not part of the DynaMapBench v0.1 contract");
        }
    }
}

fn valid_oid(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64)
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_path(value: &str) -> bool {
    !value.trim().is_empty()
        && !value.starts_with('/')
        && !value.ends_with('/')
        && !value.contains('\\')
        && !value.contains('\0')
        && !value.split('/').any(|part| part.is_empty() || part == "." || part == "..")
}

fn parse_time(value: &str) -> Option<DateTime<chrono::FixedOffset>> {
    DateTime::parse_from_rfc3339(value).ok()
}

fn check_digest(
    value: Option<&Value>,
    path: &str,
    report: &mut DynaMapBenchManifestReport,
) {
    match value.and_then(Value::as_str) {
        Some(digest) if valid_sha256(digest) => {}
        Some(_) => report.error("DMB-DIGEST-001", path, "expected a 64-character SHA-256 hexadecimal digest"),
        None => report.error("DMB-REQUIRED-001", path, "required digest is missing or not a string"),
    }
}

fn check_budget(
    case: &Map<String, Value>,
    path: &str,
    report: &mut DynaMapBenchManifestReport,
) {
    let Some(budget_value) = required_value(case, "budget", path, report) else {
        return;
    };
    let budget_path = format!("{path}.budget");
    let Some(budget) = object(budget_value, &budget_path, report) else {
        return;
    };
    allowed_keys(budget, &["tokenizer_id", "token_limit", "line_limit", "tool_call_limit", "time_limit_ms"], &budget_path, report);
    required_string(budget, "tokenizer_id", &budget_path, report);
    for key in ["token_limit", "line_limit", "tool_call_limit", "time_limit_ms"] {
        match budget.get(key).and_then(Value::as_u64) {
            Some(value) if value > 0 => {}
            _ => report.error("DMB-BUDGET-001", format!("{budget_path}.{key}"), "budget must be a positive integer"),
        }
    }
}

fn check_harness(
    case: &Map<String, Value>,
    path: &str,
    report: &mut DynaMapBenchManifestReport,
) {
    let Some(harness_value) = required_value(case, "harness", path, report) else {
        return;
    };
    let harness_path = format!("{path}.harness");
    let Some(harness) = object(harness_value, &harness_path, report) else {
        return;
    };
    allowed_keys(harness, &["agent", "model", "harness", "prompt_sha256", "evaluator_sha256"], &harness_path, report);
    for key in ["agent", "model", "harness"] {
        required_string(harness, key, &harness_path, report);
    }
    check_digest(harness.get("prompt_sha256"), &format!("{harness_path}.prompt_sha256"), report);
    check_digest(harness.get("evaluator_sha256"), &format!("{harness_path}.evaluator_sha256"), report);
}

fn check_repository(
    case: &Map<String, Value>,
    path: &str,
    report: &mut DynaMapBenchManifestReport,
) -> Option<(String, String)> {
    let repo_value = required_value(case, "repository", path, report)?;
    let repo_path = format!("{path}.repository");
    let repository = object(repo_value, &repo_path, report)?;
    allowed_keys(repository, &["full_name", "family_id", "commit_oid", "tree_oid"], &repo_path, report);
    let full_name = required_string(repository, "full_name", &repo_path, report)?.to_owned();
    let family = required_string(repository, "family_id", &repo_path, report)?.to_owned();
    for key in ["commit_oid", "tree_oid"] {
        match required_string(repository, key, &repo_path, report) {
            Some(oid) if valid_oid(oid) => {}
            Some(_) => report.error("DMB-OID-001", format!("{repo_path}.{key}"), "expected a 40- or 64-character Git object ID"),
            None => {}
        }
    }
    Some((full_name, family))
}

fn check_source_objects(
    case: &Map<String, Value>,
    path: &str,
    report: &mut DynaMapBenchManifestReport,
) -> BTreeSet<String> {
    let mut paths = BTreeSet::new();
    let Some(value) = required_value(case, "source_objects", path, report) else {
        return paths;
    };
    let Some(items) = value.as_array() else {
        report.error("DMB-TYPE-002", format!("{path}.source_objects"), "expected an array");
        return paths;
    };
    if items.is_empty() {
        report.error("DMB-SOURCE-001", format!("{path}.source_objects"), "at least one source object is required");
    }
    for (index, item) in items.iter().enumerate() {
        let item_path = format!("{path}.source_objects[{index}]");
        let Some(source) = object(item, &item_path, report) else {
            continue;
        };
        allowed_keys(source, &["path", "blob_oid"], &item_path, report);
        if let Some(source_path) = required_string(source, "path", &item_path, report) {
            if !valid_path(source_path) {
                report.error("DMB-PATH-001", format!("{item_path}.path"), "path must be canonical and repository-relative");
            } else if !paths.insert(source_path.to_owned()) {
                report.error("DMB-PATH-002", format!("{item_path}.path"), "duplicate source path");
            }
        }
        match required_string(source, "blob_oid", &item_path, report) {
            Some(oid) if valid_oid(oid) => {}
            Some(_) => report.error("DMB-OID-002", format!("{item_path}.blob_oid"), "expected a 40- or 64-character Git blob OID"),
            None => {}
        }
    }
    paths
}

fn check_temporal_inputs(
    case: &Map<String, Value>,
    path: &str,
    report: &mut DynaMapBenchManifestReport,
) {
    let cutoff_value = required_value(case, "decision_cutoff", path, report);
    let cutoff = match cutoff_value {
        Some(Value::Null) => None,
        Some(Value::String(value)) => match parse_time(value) {
            Some(time) => Some(time),
            None => {
                report.error("DMB-TIME-001", format!("{path}.decision_cutoff"), "expected an RFC3339 timestamp or null");
                None
            }
        },
        Some(_) => {
            report.error("DMB-TIME-001", format!("{path}.decision_cutoff"), "expected an RFC3339 timestamp or null");
            None
        }
        None => None,
    };

    let available = case.get("available_inputs");
    let available_items = match available {
        Some(Value::Array(items)) => items,
        Some(_) => {
            report.error("DMB-TYPE-002", format!("{path}.available_inputs"), "expected an array");
            return;
        }
        None => {
            report.error("DMB-REQUIRED-001", format!("{path}.available_inputs"), "required array is missing");
            return;
        }
    };
    if !available_items.is_empty() && cutoff.is_none() {
        report.error("DMB-TIME-002", format!("{path}.decision_cutoff"), "available evidence requires a decision cutoff");
    }
    let mut source_ids = BTreeSet::new();
    for (index, item) in available_items.iter().enumerate() {
        let item_path = format!("{path}.available_inputs[{index}]");
        let Some(source) = object(item, &item_path, report) else {
            continue;
        };
        allowed_keys(source, &["source_id", "available_at", "object_oid"], &item_path, report);
        if let Some(id) = required_string(source, "source_id", &item_path, report) {
            if !source_ids.insert(id.to_owned()) {
                report.error("DMB-EVIDENCE-001", format!("{item_path}.source_id"), "source ID is duplicated across available/forbidden inputs");
            }
        }
        match required_string(source, "object_oid", &item_path, report) {
            Some(oid) if valid_oid(oid) => {}
            Some(_) => report.error("DMB-OID-003", format!("{item_path}.object_oid"), "expected a Git object ID"),
            None => {}
        }
        if let Some(available_at) = required_string(source, "available_at", &item_path, report) {
            match parse_time(available_at) {
                Some(time) if cutoff.as_ref().is_some_and(|cutoff| time <= *cutoff) => {}
                Some(_) if cutoff.is_none() => {}
                Some(_) => report.error("DMB-TIME-003", format!("{item_path}.available_at"), "available input is later than the decision cutoff"),
                None => report.error("DMB-TIME-001", format!("{item_path}.available_at"), "expected an RFC3339 timestamp"),
            }
        }
    }

    let forbidden = case.get("forbidden_inputs");
    let forbidden_items = match forbidden {
        Some(Value::Array(items)) => items,
        Some(_) => {
            report.error("DMB-TYPE-002", format!("{path}.forbidden_inputs"), "expected an array");
            return;
        }
        None => {
            report.error("DMB-REQUIRED-001", format!("{path}.forbidden_inputs"), "required array is missing");
            return;
        }
    };
    for (index, item) in forbidden_items.iter().enumerate() {
        let item_path = format!("{path}.forbidden_inputs[{index}]");
        let Some(source) = object(item, &item_path, report) else {
            continue;
        };
        allowed_keys(source, &["source_id", "reason", "available_at", "object_oid"], &item_path, report);
        if let Some(id) = required_string(source, "source_id", &item_path, report) {
            if !source_ids.insert(id.to_owned()) {
                report.error("DMB-EVIDENCE-001", format!("{item_path}.source_id"), "source ID is duplicated across available/forbidden inputs");
            }
        }
        let reason = required_string(source, "reason", &item_path, report);
        if let Some(reason) = reason {
            if !["future", "unauthorized", "wrong_revision", "wrong_world", "superseded", "expired", "incomplete"].contains(&reason) {
                report.error("DMB-EVIDENCE-002", format!("{item_path}.reason"), "unsupported forbidden-evidence reason");
            }
            if reason == "future" {
                match (
                    source.get("available_at").and_then(Value::as_str).and_then(parse_time),
                    cutoff.as_ref(),
                ) {
                    (Some(available_at), Some(cutoff)) if available_at > *cutoff => {}
                    _ => report.error("DMB-TIME-004", format!("{item_path}.available_at"), "future evidence must have a valid availability time after the decision cutoff"),
                }
            }
        }
        if let Some(oid) = source.get("object_oid").and_then(Value::as_str) {
            if !valid_oid(oid) {
                report.error("DMB-OID-003", format!("{item_path}.object_oid"), "expected a Git object ID");
            }
        }
    }
}

fn validate_spans(
    gold: &Map<String, Value>,
    key: &str,
    path: &str,
    source_paths: &BTreeSet<String>,
    report: &mut DynaMapBenchManifestReport,
) -> usize {
    let field_path = format!("{path}.{key}");
    let Some(value) = required_value(gold, key, path, report) else {
        return 0;
    };
    let Some(spans) = value.as_array() else {
        report.error("DMB-TYPE-002", &field_path, "expected an array");
        return 0;
    };
    let mut count = 0;
    for (index, span_value) in spans.iter().enumerate() {
        let span_path = format!("{field_path}[{index}]");
        let Some(span) = object(span_value, &span_path, report) else {
            continue;
        };
        allowed_keys(span, &["path", "start_line", "end_line"], &span_path, report);
        let source_path = required_string(span, "path", &span_path, report);
        if let Some(source_path) = source_path {
            if !source_paths.contains(source_path) {
                report.error("DMB-SPAN-001", format!("{span_path}.path"), "gold span path is not one of the case's exact source objects");
            }
        }
        let start = span.get("start_line").and_then(Value::as_u64);
        let end = span.get("end_line").and_then(Value::as_u64);
        match (start, end) {
            (Some(start), Some(end)) if start > 0 && start <= end => {}
            _ => report.error("DMB-SPAN-002", &span_path, "line range must satisfy 1 <= start_line <= end_line"),
        }
        count += 1;
    }
    count
}

fn check_gold(
    case: &Map<String, Value>,
    path: &str,
    source_paths: &BTreeSet<String>,
    report: &mut DynaMapBenchManifestReport,
) {
    let Some(gold_value) = required_value(case, "gold", path, report) else {
        return;
    };
    let gold_path = format!("{path}.gold");
    let Some(gold) = object(gold_value, &gold_path, report) else {
        return;
    };
    allowed_keys(gold, &["disposition", "authority", "review_count", "adjudicated", "relevant_spans", "forbidden_spans"], &gold_path, report);
    let disposition = required_string(gold, "disposition", &gold_path, report);
    if let Some(disposition) = disposition {
        if !["positive", "natural_no_gold", "counterfactual_no_gold"].contains(&disposition) {
            report.error("DMB-GOLD-001", format!("{gold_path}.disposition"), "unsupported gold disposition");
        }
    }
    let authority = required_string(gold, "authority", &gold_path, report);
    if let Some(authority) = authority {
        if !["independently_adjudicated", "synthetic_exact"].contains(&authority) {
            report.error("DMB-GOLD-002", format!("{gold_path}.authority"), "unsupported gold authority");
        }
    }
    let review_count = gold.get("review_count").and_then(Value::as_u64);
    let adjudicated = gold.get("adjudicated").and_then(Value::as_bool);
    if review_count.is_none() {
        report.error("DMB-GOLD-003", format!("{gold_path}.review_count"), "review_count must be a non-negative integer");
    }
    if adjudicated.is_none() {
        report.error("DMB-GOLD-004", format!("{gold_path}.adjudicated"), "adjudicated must be boolean");
    }
    if authority == Some("independently_adjudicated") {
        if review_count.unwrap_or(0) < 2 || adjudicated != Some(true) {
            report.error("DMB-GOLD-005", &gold_path, "independent gold requires at least two reviews and adjudicated=true");
        }
    }
    let relevant = validate_spans(gold, "relevant_spans", &gold_path, source_paths, report);
    validate_spans(gold, "forbidden_spans", &gold_path, source_paths, report);
    match disposition {
        Some("positive") if relevant == 0 => report.error("DMB-GOLD-006", format!("{gold_path}.relevant_spans"), "positive case requires at least one relevant span"),
        Some("natural_no_gold" | "counterfactual_no_gold") if relevant != 0 => report.error("DMB-GOLD-007", format!("{gold_path}.relevant_spans"), "no-gold case must not carry positive relevant spans"),
        _ => {}
    }
}


fn check_split_policy(
    root: &Map<String, Value>,
    report: &mut DynaMapBenchManifestReport,
) -> Option<DateTime<chrono::FixedOffset>> {
    let value = required_value(root, "split_policy", "$", report)?;
    let path = "$.split_policy";
    let policy = object(value, path, report)?;
    allowed_keys(policy, &["repository_family_holdout", "temporal_holdout", "test_after"], path, report);
    for key in ["repository_family_holdout", "temporal_holdout"] {
        match policy.get(key).and_then(Value::as_bool) {
            Some(true) => {}
            Some(false) => report.error("DMB-SPLIT-002", format!("{path}.{key}"), "DynaMapBench v0.1 requires this split control"),
            None => report.error("DMB-SPLIT-003", format!("{path}.{key}"), "required boolean true is missing"),
        }
    }
    match required_string(policy, "test_after", path, report) {
        Some(value) => match parse_time(value) {
            Some(value) => Some(value),
            None => {
                report.error("DMB-TIME-005", format!("{path}.test_after"), "expected RFC3339 timestamp");
                None
            }
        },
        None => None,
    }
}

fn validate_case(
    case_value: &Value,
    index: usize,
    ids: &mut BTreeSet<String>,
    families: &mut BTreeMap<String, String>,
    temporal_split_after: Option<&DateTime<chrono::FixedOffset>>,
    report: &mut DynaMapBenchManifestReport,
) {
    let path = format!("cases[{index}]");
    let Some(case) = object(case_value, &path, report) else {
        return;
    };
    allowed_keys(
        case,
        &[
            "case_id", "lane", "task_family", "split", "case_time", "repository", "source_objects",
            "decision_cutoff", "available_inputs", "forbidden_inputs", "budget", "harness", "gold",
        ],
        &path,
        report,
    );
    if let Some(id) = required_string(case, "case_id", &path, report) {
        if !ids.insert(id.to_owned()) {
            report.error("DMB-CASE-001", format!("{path}.case_id"), "case_id must be globally unique within the manifest");
        }
    }
    if let Some(lane) = required_string(case, "lane", &path, report) {
        if !["D1", "D2", "D3", "D4", "D5", "D6", "F"].contains(&lane) {
            report.error("DMB-LANE-001", format!("{path}.lane"), "unsupported benchmark lane");
        }
    }
    required_string(case, "task_family", &path, report);
    let split = required_string(case, "split", &path, report);
    let case_time = required_string(case, "case_time", &path, report).and_then(|value| {
        match parse_time(value) {
            Some(time) => Some(time),
            None => {
                report.error("DMB-TIME-005", format!("{path}.case_time"), "expected RFC3339 timestamp");
                None
            }
        }
    });
    if let (Some(split), Some(case_time), Some(test_after)) = (split, case_time.as_ref(), temporal_split_after) {
        if ["train", "dev"].contains(&split) && case_time > test_after {
            report.error("DMB-LEAKAGE-002", format!("{path}.case_time"), "train/dev case time is after the temporal test boundary");
        }
        if split == "test" && case_time <= test_after {
            report.error("DMB-LEAKAGE-003", format!("{path}.case_time"), "test case time is not after the temporal test boundary");
        }
    }
    if let Some(split) = split {
        if !["train", "dev", "test"].contains(&split) {
            report.error("DMB-SPLIT-001", format!("{path}.split"), "split must be train, dev or test");
        }
    }
    if let Some((_, family)) = check_repository(case, &path, report) {
        if let Some(split) = split {
            match families.get(&family) {
                Some(previous) if previous != split => report.error(
                    "DMB-LEAKAGE-001",
                    format!("{path}.repository.family_id"),
                    format!("repository family appears in both {previous} and {split} splits"),
                ),
                None => { families.insert(family, split.to_owned()); }
                _ => {}
            }
        }
    }
    let source_paths = check_source_objects(case, &path, report);
    check_temporal_inputs(case, &path, report);
    check_budget(case, &path, report);
    check_harness(case, &path, report);
    check_gold(case, &path, &source_paths, report);
}

/// Validate a DynaMapBench v0.1 JSON corpus manifest without network or code execution.
pub fn validate_dynamapbench_manifest_json(
    raw_json: &str,
) -> Result<DynaMapBenchManifestReport, String> {
    let document: Value = serde_json::from_str(raw_json)
        .map_err(|error| format!("DynaMapBench manifest JSON parse error: {error}"))?;
    let mut report = DynaMapBenchManifestReport::default();
    let Some(root) = object(&document, "$", &mut report) else {
        return Ok(report);
    };
    allowed_keys(root, &["schema_version", "benchmark_id", "release", "split_policy", "corpus_artifact", "evaluator", "cases"], "$", &mut report);
    if required_string(root, "schema_version", "$", &mut report) != Some("0.1.0") {
        report.error("DMB-SCHEMA-001", "$.schema_version", "expected schema_version 0.1.0");
    }
    if required_string(root, "benchmark_id", "$", &mut report) != Some("DynaMapBench") {
        report.error("DMB-IDENTITY-001", "$.benchmark_id", "expected benchmark_id DynaMapBench");
    }
    required_string(root, "release", "$", &mut report);
    let temporal_split_after = check_split_policy(root, &mut report);

    for key in ["corpus_artifact", "evaluator"] {
        if let Some(value) = required_value(root, key, "$", &mut report) {
            let path = format!("$.{key}");
            if let Some(object_value) = object(value, &path, &mut report) {
                let allowed = if key == "corpus_artifact" {
                    &["uri", "sha256"][..]
                } else {
                    &["name", "version", "sha256"][..]
                };
                allowed_keys(object_value, allowed, &path, &mut report);
                if key == "corpus_artifact" {
                    required_string(object_value, "uri", &path, &mut report);
                } else {
                    required_string(object_value, "name", &path, &mut report);
                    required_string(object_value, "version", &path, &mut report);
                }
                check_digest(object_value.get("sha256"), &format!("{path}.sha256"), &mut report);
            }
        }
    }

    let Some(cases_value) = required_value(root, "cases", "$", &mut report) else {
        report.sort();
        return Ok(report);
    };
    let Some(cases) = cases_value.as_array() else {
        report.error("DMB-TYPE-002", "$.cases", "expected an array");
        report.sort();
        return Ok(report);
    };
    if cases.is_empty() {
        report.error("DMB-CASE-002", "$.cases", "manifest must contain at least one case");
    }
    let mut ids = BTreeSet::new();
    let mut families = BTreeMap::new();
    for (index, case) in cases.iter().enumerate() {
        validate_case(case, index, &mut ids, &mut families, temporal_split_after.as_ref(), &mut report);
    }
    report.sort();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn case(case_id: &str, family: &str, split: &str) -> Value {
        json!({
            "case_id": case_id,
            "case_time": if split == "test" { "2026-02-02T00:00:00Z" } else { "2026-01-01T00:00:00Z" },
            "lane": "D3",
            "task_family": "trace2code",
            "split": split,
            "repository": {
                "full_name": "example/project",
                "family_id": family,
                "commit_oid": "a".repeat(40),
                "tree_oid": "b".repeat(40)
            },
            "source_objects": [
                {"path":"src/lib.rs","blob_oid":"c".repeat(40)}
            ],
            "decision_cutoff": "2026-01-02T00:00:00Z",
            "available_inputs": [
                {"source_id":"issue-1","available_at":"2026-01-01T00:00:00Z","object_oid":"d".repeat(40)}
            ],
            "forbidden_inputs": [
                {"source_id":"fix-1","reason":"future","available_at":"2026-01-03T00:00:00Z","object_oid":"e".repeat(40)}
            ],
            "budget": {
                "tokenizer_id":"test-tokenizer@1",
                "token_limit":8192,
                "line_limit":500,
                "tool_call_limit":20,
                "time_limit_ms":60000
            },
            "harness": {
                "agent":"baseline-agent",
                "model":"frozen-model",
                "harness":"harness@1",
                "prompt_sha256":"1".repeat(64),
                "evaluator_sha256":"2".repeat(64)
            },
            "gold": {
                "disposition":"positive",
                "authority":"independently_adjudicated",
                "review_count":2,
                "adjudicated":true,
                "relevant_spans":[{"path":"src/lib.rs","start_line":1,"end_line":4}],
                "forbidden_spans":[]
            }
        })
    }

    fn manifest(cases: Vec<Value>) -> String {
        json!({
            "schema_version":"0.1.0",
            "benchmark_id":"DynaMapBench",
            "release":"v0.1-test",
            "split_policy":{"repository_family_holdout":true,"temporal_holdout":true,"test_after":"2026-01-31T00:00:00Z"},
            "corpus_artifact":{"uri":"https://example.invalid/corpus.tar","sha256":"3".repeat(64)},
            "evaluator":{"name":"dynamapbench-evaluator","version":"0.1.0","sha256":"4".repeat(64)},
            "cases":cases
        }).to_string()
    }


    #[test]
    fn rejects_test_case_not_after_temporal_holdout_boundary() {
        let mut c = case("case-1","family-a","test");
        c["case_time"] = json!("2026-01-15T00:00:00Z");
        let report = validate_dynamapbench_manifest_json(&manifest(vec![c])).expect("report");
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-LEAKAGE-003"));
    }

    #[test]
    fn rejects_train_case_after_temporal_holdout_boundary() {
        let mut c = case("case-1","family-a","train");
        c["case_time"] = json!("2026-02-15T00:00:00Z");
        let report = validate_dynamapbench_manifest_json(&manifest(vec![c])).expect("report");
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-LEAKAGE-002"));
    }

    #[test]
    fn requires_repository_and_temporal_holdout_policies() {
        let mut doc: Value = serde_json::from_str(&manifest(vec![case("case-1","family-a","test")])).expect("fixture");
        doc["split_policy"]["temporal_holdout"] = json!(false);
        let report = validate_dynamapbench_manifest_json(&doc.to_string()).expect("report");
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-SPLIT-002"));
    }

    #[test]
    fn accepts_valid_independently_adjudicated_case() {
        let report = validate_dynamapbench_manifest_json(&manifest(vec![case("case-1","family-a","test")])).expect("valid JSON");
        assert!(report.is_valid(), "{:?}", report.diagnostics);
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(validate_dynamapbench_manifest_json("{").is_err());
    }

    #[test]
    fn rejects_wrong_benchmark_identity() {
        let mut doc: Value = serde_json::from_str(&manifest(vec![case("case-1","family-a","test")])).expect("fixture");
        doc["benchmark_id"] = json!("OtherBench");
        let report = validate_dynamapbench_manifest_json(&doc.to_string()).expect("parsed report");
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-IDENTITY-001"));
    }

    #[test]
    fn rejects_duplicate_case_ids() {
        let report = validate_dynamapbench_manifest_json(&manifest(vec![
            case("case-1","family-a","train"), case("case-1","family-b","test")
        ])).expect("report");
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-CASE-001"));
    }

    #[test]
    fn rejects_repository_family_cross_split_leakage() {
        let report = validate_dynamapbench_manifest_json(&manifest(vec![
            case("case-1","family-a","train"), case("case-2","family-a","test")
        ])).expect("report");
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-LEAKAGE-001"));
    }

    #[test]
    fn allows_multiple_cases_from_one_family_in_one_split() {
        let report = validate_dynamapbench_manifest_json(&manifest(vec![
            case("case-1","family-a","test"), case("case-2","family-a","test")
        ])).expect("report");
        assert!(report.is_valid(), "{:?}", report.diagnostics);
    }

    #[test]
    fn rejects_invalid_repository_commit_oid() {
        let mut doc: Value = serde_json::from_str(&manifest(vec![case("case-1","family-a","test")])).expect("fixture");
        doc["cases"][0]["repository"]["commit_oid"] = json!("main");
        let report = validate_dynamapbench_manifest_json(&doc.to_string()).expect("report");
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-OID-001"));
    }

    #[test]
    fn rejects_duplicate_source_paths() {
        let mut c = case("case-1","family-a","test");
        c["source_objects"].as_array_mut().expect("array").push(json!({"path":"src/lib.rs","blob_oid":"f".repeat(40)}));
        let report = validate_dynamapbench_manifest_json(&manifest(vec![c])).expect("report");
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-PATH-002"));
    }

    #[test]
    fn rejects_parent_traversal_source_paths() {
        let mut c = case("case-1","family-a","test");
        c["source_objects"][0]["path"] = json!("../lib.rs");
        let report = validate_dynamapbench_manifest_json(&manifest(vec![c])).expect("report");
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-PATH-001"));
    }

    #[test]
    fn rejects_input_available_after_decision_cutoff() {
        let mut c = case("case-1","family-a","test");
        c["available_inputs"][0]["available_at"] = json!("2026-01-03T00:00:00Z");
        let report = validate_dynamapbench_manifest_json(&manifest(vec![c])).expect("report");
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-TIME-003"));
    }

    #[test]
    fn rejects_future_forbidden_evidence_not_after_cutoff() {
        let mut c = case("case-1","family-a","test");
        c["forbidden_inputs"][0]["available_at"] = json!("2026-01-01T12:00:00Z");
        let report = validate_dynamapbench_manifest_json(&manifest(vec![c])).expect("report");
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-TIME-004"));
    }

    #[test]
    fn requires_cutoff_when_available_inputs_exist() {
        let mut c = case("case-1","family-a","test");
        c["decision_cutoff"] = Value::Null;
        let report = validate_dynamapbench_manifest_json(&manifest(vec![c])).expect("report");
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-TIME-002"));
    }

    #[test]
    fn rejects_positive_case_without_relevant_gold_span() {
        let mut c = case("case-1","family-a","test");
        c["gold"]["relevant_spans"] = json!([]);
        let report = validate_dynamapbench_manifest_json(&manifest(vec![c])).expect("report");
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-GOLD-006"));
    }

    #[test]
    fn rejects_no_gold_case_with_positive_spans() {
        let mut c = case("case-1","family-a","test");
        c["gold"]["disposition"] = json!("natural_no_gold");
        let report = validate_dynamapbench_manifest_json(&manifest(vec![c])).expect("report");
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-GOLD-007"));
    }

    #[test]
    fn rejects_gold_span_not_anchored_to_source_objects() {
        let mut c = case("case-1","family-a","test");
        c["gold"]["relevant_spans"][0]["path"] = json!("src/other.rs");
        let report = validate_dynamapbench_manifest_json(&manifest(vec![c])).expect("report");
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-SPAN-001"));
    }

    #[test]
    fn rejects_invalid_harness_digest_and_zero_budget() {
        let mut c = case("case-1","family-a","test");
        c["harness"]["prompt_sha256"] = json!("bad");
        c["budget"]["token_limit"] = json!(0);
        let report = validate_dynamapbench_manifest_json(&manifest(vec![c])).expect("report");
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-DIGEST-001"));
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-BUDGET-001"));
    }

    #[test]
    fn requires_two_reviews_for_independent_gold() {
        let mut c = case("case-1","family-a","test");
        c["gold"]["review_count"] = json!(1);
        let report = validate_dynamapbench_manifest_json(&manifest(vec![c])).expect("report");
        assert!(report.diagnostics.iter().any(|d| d.code == "DMB-GOLD-005"));
    }
}
