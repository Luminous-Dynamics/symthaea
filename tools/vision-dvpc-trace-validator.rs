// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-005: strict paired-trace admission validator.
//!
//! This validates a dataset-neutral TSV contract; it does not run a vision model,
//! validate a dataset license, recompute source-data digests, or establish SOTA.
//!
//! Usage:
//!   rustc --edition=2024 --test tools/vision-dvpc-trace-validator.rs -o /tmp/vis-dvpc-005-tests
//!   /tmp/vis-dvpc-005-tests
//!   rustc --edition=2024 -O tools/vision-dvpc-trace-validator.rs -o /tmp/vis-dvpc-005
//!   /tmp/vis-dvpc-005 fixtures/vision-dvpc-observations.tsv fixtures/vision-dvpc-evaluations.tsv

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;

const FORMAT_VERSION: &str = "VIS-DVPC-TRACE-001-v1";
const OBS_HEADER: &[&str] = &[
    "OBS_HEADER",
    "trace_id",
    "dataset_id",
    "dataset_revision",
    "split",
    "group_id",
    "step_index",
    "sample_sha256",
    "cheap_output_sha256",
    "pre_features_sha256",
    "utility_spec_sha256",
    "declared_extra_latency_us",
    "declared_extra_energy_uj",
    "estimated_dvpc",
    "reported_uncertainty",
    "shift_signal_sha256",
];
const EVAL_HEADER: &[&str] = &[
    "EVAL_HEADER",
    "trace_id",
    "sample_sha256",
    "cheap_utility_01",
    "full_utility_01",
    "cheap_safety_violations",
    "full_safety_violations",
    "actual_extra_latency_us",
    "actual_extra_energy_uj",
    "shift_label",
    "shift_mechanism",
];
const REQUIRED_META: &[&str] = &[
    "format",
    "dataset_id",
    "dataset_revision",
    "dataset_snapshot_sha256",
    "source_manifest_sha256",
    "split_manifest_sha256",
    "utility_spec_sha256",
    "license_id",
    "license_scope",
    "higher_is_better",
];

#[derive(Debug, Clone)]
struct Table {
    meta: BTreeMap<String, String>,
    rows: Vec<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AdmissionSummary {
    observation_rows: usize,
    evaluation_rows: usize,
    groups: usize,
    split_rows: BTreeMap<String, usize>,
}

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

fn parse_table(
    input: &str,
    expected_header: &[&str],
    record_tag: &str,
) -> Result<Table, String> {
    let mut meta = BTreeMap::new();
    let mut rows = Vec::new();
    let mut found_header = false;

    for (line_index, line) in input.lines().enumerate() {
        let line_number = line_index + 1;
        if line.is_empty() {
            return Err(invalid(format!("line {line_number}: blank lines are forbidden")));
        }
        let fields: Vec<String> = line.split('\t').map(str::to_owned).collect();

        if fields.first().map(String::as_str) == Some("META") {
            if found_header {
                return Err(invalid(format!(
                    "line {line_number}: metadata must precede the data header"
                )));
            }
            if fields.len() != 3 || fields[1].is_empty() || fields[2].is_empty() {
                return Err(invalid(format!(
                    "line {line_number}: META requires exactly key and value"
                )));
            }
            if meta.insert(fields[1].clone(), fields[2].clone()).is_some() {
                return Err(invalid(format!(
                    "line {line_number}: duplicate metadata key {}",
                    fields[1]
                )));
            }
            continue;
        }

        if !found_header {
            let actual: Vec<&str> = fields.iter().map(String::as_str).collect();
            if actual.as_slice() != expected_header {
                return Err(invalid(format!(
                    "line {line_number}: header mismatch; unknown or reordered columns fail closed"
                )));
            }
            found_header = true;
            continue;
        }

        if fields.first().map(String::as_str) != Some(record_tag) {
            return Err(invalid(format!(
                "line {line_number}: expected {record_tag} data row"
            )));
        }
        if fields.len() != expected_header.len() {
            return Err(invalid(format!(
                "line {line_number}: expected {} fields, got {}",
                expected_header.len(),
                fields.len()
            )));
        }
        if fields.iter().any(String::is_empty) {
            return Err(invalid(format!(
                "line {line_number}: empty fields are forbidden; use '-' only for optional digests"
            )));
        }
        rows.push(fields);
    }

    if !found_header {
        return Err(invalid("missing data header"));
    }
    if rows.is_empty() {
        return Err(invalid("at least one data row is required"));
    }
    validate_metadata(&meta)?;
    Ok(Table { meta, rows })
}

fn validate_metadata(meta: &BTreeMap<String, String>) -> Result<(), String> {
    let expected: BTreeSet<&str> = REQUIRED_META.iter().copied().collect();
    let actual: BTreeSet<&str> = meta.keys().map(String::as_str).collect();
    if actual != expected {
        let missing: Vec<&str> = expected.difference(&actual).copied().collect();
        let unknown: Vec<&str> = actual.difference(&expected).copied().collect();
        return Err(invalid(format!(
            "metadata key set mismatch; missing={missing:?}, unknown={unknown:?}"
        )));
    }
    if meta.get("format").map(String::as_str) != Some(FORMAT_VERSION) {
        return Err(invalid("unsupported format version"));
    }
    if meta.get("higher_is_better").map(String::as_str) != Some("true") {
        return Err(invalid("utility direction must be explicitly higher-is-better"));
    }

    for key in [
        "dataset_snapshot_sha256",
        "source_manifest_sha256",
        "split_manifest_sha256",
        "utility_spec_sha256",
    ] {
        validate_sha256(meta.get(key).expect("required metadata checked"), key)?;
    }

    let scope = meta.get("license_scope").expect("required metadata checked");
    if !matches!(
        scope.as_str(),
        "synthetic_public_fixture" | "noncommercial_research_only" | "commercial_use_authorized"
    ) {
        return Err(invalid(
            "license_scope must be explicit; review_required/unknown is not admissible",
        ));
    }
    for key in ["dataset_id", "dataset_revision", "license_id"] {
        validate_token(meta.get(key).expect("required metadata checked"), key)?;
    }
    Ok(())
}

fn validate_token(value: &str, field: &str) -> Result<(), String> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b':'))
    {
        return Err(invalid(format!(
            "{field} must be a non-empty identifier token without paths or whitespace"
        )));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(invalid(format!("{field} must be a 64-hex-character SHA-256 digest")));
    }
    Ok(())
}

fn parse_u64(value: &str, field: &str, row: usize) -> Result<u64, String> {
    value.parse::<u64>().map_err(|_| {
        invalid(format!(
            "data row {row}: {field} must be a non-negative integer"
        ))
    })
}

fn parse_f64(value: &str, field: &str, row: usize) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| invalid(format!("data row {row}: {field} must be numeric")))?;
    if !parsed.is_finite() {
        return Err(invalid(format!(
            "data row {row}: {field} must be finite"
        )));
    }
    Ok(parsed)
}

fn same_meta(left: &Table, right: &Table) -> Result<(), String> {
    if left.meta != right.meta {
        return Err(invalid(
            "observation/evaluation metadata differ; paired files must bind the same snapshot and utility profile",
        ));
    }
    Ok(())
}

fn validate_observations(
    table: &Table,
) -> Result<(BTreeMap<String, String>, BTreeMap<String, usize>, usize), String> {
    let mut trace_to_sample = BTreeMap::new();
    let mut group_to_split: BTreeMap<String, String> = BTreeMap::new();
    let mut group_steps: BTreeMap<String, BTreeSet<u64>> = BTreeMap::new();
    let mut split_rows: BTreeMap<String, usize> = BTreeMap::new();

    let dataset_id = table.meta.get("dataset_id").expect("metadata validated");
    let revision = table.meta.get("dataset_revision").expect("metadata validated");
    let utility_spec = table.meta.get("utility_spec_sha256").expect("metadata validated");

    for (index, row) in table.rows.iter().enumerate() {
        let row_number = index + 1;
        validate_token(&row[1], "trace_id")?;
        validate_token(&row[2], "dataset_id")?;
        validate_token(&row[3], "dataset_revision")?;
        validate_token(&row[5], "group_id")?;

        if &row[2] != dataset_id || &row[3] != revision {
            return Err(invalid(format!(
                "data row {row_number}: dataset identity/revision differs from metadata"
            )));
        }
        let split = row[4].as_str();
        if !matches!(split, "train" | "validation" | "test" | "private_test") {
            return Err(invalid(format!(
                "data row {row_number}: unsupported split label {split}"
            )));
        }
        *split_rows.entry(split.to_owned()).or_default() += 1;

        validate_sha256(&row[7], "sample_sha256")?;
        validate_sha256(&row[8], "cheap_output_sha256")?;
        validate_sha256(&row[9], "pre_features_sha256")?;
        validate_sha256(&row[10], "utility_spec_sha256")?;
        if &row[10] != utility_spec {
            return Err(invalid(format!(
                "data row {row_number}: utility specification digest mismatch"
            )));
        }
        if row[15] != "-" {
            validate_sha256(&row[15], "shift_signal_sha256")?;
        }

        let declared_latency = parse_u64(&row[11], "declared_extra_latency_us", row_number)?;
        let declared_energy = parse_u64(&row[12], "declared_extra_energy_uj", row_number)?;
        if declared_latency == 0 || declared_energy == 0 {
            return Err(invalid(format!(
                "data row {row_number}: declared escalation latency and energy must be positive"
            )));
        }
        let estimate = parse_f64(&row[13], "estimated_dvpc", row_number)?;
        let uncertainty = parse_f64(&row[14], "reported_uncertainty", row_number)?;
        if uncertainty < 0.0 {
            return Err(invalid(format!(
                "data row {row_number}: reported_uncertainty must be non-negative"
            )));
        }
        // Finite estimate is checked above; value can be positive or negative.
        let _ = estimate;

        if trace_to_sample
            .insert(row[1].clone(), row[7].clone())
            .is_some()
        {
            return Err(invalid(format!(
                "duplicate observation trace_id {}",
                row[1]
            )));
        }

        let group_key = format!("{}:{}:{}", row[2], row[3], row[5]);
        match group_to_split.get(&group_key) {
            Some(previous) if previous != split => {
                return Err(invalid(format!(
                    "source group {} crosses splits: {previous} and {split}",
                    row[5]
                )));
            }
            None => {
                group_to_split.insert(group_key.clone(), split.to_owned());
            }
            _ => {}
        }

        let step = parse_u64(&row[6], "step_index", row_number)?;
        if !group_steps.entry(group_key.clone()).or_default().insert(step) {
            return Err(invalid(format!(
                "duplicate step_index {step} in group {}",
                row[5]
            )));
        }
    }

    for (group, steps) in &group_steps {
        for (expected, observed) in steps.iter().enumerate() {
            if *observed != expected as u64 {
                return Err(invalid(format!(
                    "group {group} has non-contiguous step index: expected {expected}, got {observed}"
                )));
            }
        }
    }

    let group_count = group_steps.len();
    Ok((trace_to_sample, split_rows, group_count))
}

fn validate_evaluations(
    table: &Table,
    observation_sample_by_trace: &BTreeMap<String, String>,
) -> Result<(), String> {
    let mut seen = BTreeSet::new();

    for (index, row) in table.rows.iter().enumerate() {
        let row_number = index + 1;
        validate_token(&row[1], "trace_id")?;
        validate_sha256(&row[2], "sample_sha256")?;

        if !seen.insert(row[1].clone()) {
            return Err(invalid(format!("duplicate evaluation trace_id {}", row[1])));
        }
        let expected_sample = observation_sample_by_trace
            .get(&row[1])
            .ok_or_else(|| invalid(format!("evaluation has unmatched trace_id {}", row[1])))?;
        if expected_sample != &row[2] {
            return Err(invalid(format!(
                "sample digest mismatch for trace_id {}",
                row[1]
            )));
        }

        for (position, field) in [(3, "cheap_utility_01"), (4, "full_utility_01")] {
            let value = parse_f64(&row[position], field, row_number)?;
            if !(0.0..=1.0).contains(&value) {
                return Err(invalid(format!(
                    "data row {row_number}: {field} must lie in [0,1]"
                )));
            }
        }
        parse_u64(&row[5], "cheap_safety_violations", row_number)?;
        parse_u64(&row[6], "full_safety_violations", row_number)?;
        let latency = parse_u64(&row[7], "actual_extra_latency_us", row_number)?;
        let energy = parse_u64(&row[8], "actual_extra_energy_uj", row_number)?;
        if latency == 0 || energy == 0 {
            return Err(invalid(format!(
                "data row {row_number}: realized escalation latency and energy must be positive"
            )));
        }
        if !matches!(row[9].as_str(), "in_domain" | "shifted" | "unknown") {
            return Err(invalid(format!(
                "data row {row_number}: invalid shift_label"
            )));
        }
        if !matches!(
            row[10].as_str(),
            "none"
                | "illumination"
                | "weather"
                | "occlusion"
                | "sensor"
                | "density"
                | "other"
                | "unknown"
        ) {
            return Err(invalid(format!(
                "data row {row_number}: invalid shift_mechanism"
            )));
        }
        if row[9] == "in_domain" && row[10] != "none" {
            return Err(invalid(format!(
                "data row {row_number}: in_domain label requires shift_mechanism=none"
            )));
        }
        if row[9] == "shifted" && matches!(row[10].as_str(), "none" | "unknown") {
            return Err(invalid(format!(
                "data row {row_number}: shifted label requires a specific mechanism"
            )));
        }
    }

    if seen.len() != observation_sample_by_trace.len() {
        let missing: Vec<&String> = observation_sample_by_trace
            .keys()
            .filter(|key| !seen.contains(*key))
            .collect();
        return Err(invalid(format!(
            "evaluation file is missing {} paired trace(s): {:?}",
            missing.len(),
            missing
        )));
    }
    Ok(())
}

fn admit(observations: &str, evaluations: &str) -> Result<AdmissionSummary, String> {
    let obs = parse_table(observations, OBS_HEADER, "OBS")?;
    let eval = parse_table(evaluations, EVAL_HEADER, "EVAL")?;
    same_meta(&obs, &eval)?;

    let (sample_by_trace, split_rows, groups) = validate_observations(&obs)?;
    validate_evaluations(&eval, &sample_by_trace)?;

    Ok(AdmissionSummary {
        observation_rows: obs.rows.len(),
        evaluation_rows: eval.rows.len(),
        groups,
        split_rows,
    })
}

fn render_admission(summary: &AdmissionSummary) -> String {
    let mut output = String::new();
    output.push_str("ADMISSION\tstatus\tPASS_SCHEMA_ONLY\n");
    output.push_str(&format!(
        "ADMISSION\tobservation_rows\t{}\n",
        summary.observation_rows
    ));
    output.push_str(&format!(
        "ADMISSION\tevaluation_rows\t{}\n",
        summary.evaluation_rows
    ));
    output.push_str(&format!("ADMISSION\tgroups\t{}\n", summary.groups));
    for split in ["train", "validation", "test", "private_test"] {
        output.push_str(&format!(
            "ADMISSION\t{split}_rows\t{}\n",
            summary.split_rows.get(split).copied().unwrap_or(0)
        ));
    }
    output.push_str("ADMISSION\tperformance_claim\tNONE\n");
    output
}

fn run_cli(args: &[String]) -> Result<String, String> {
    if args.len() != 3 {
        return Err(
            "usage: vision-dvpc-trace-validator OBSERVATIONS.tsv EVALUATIONS.tsv".to_owned(),
        );
    }
    let observations = fs::read_to_string(&args[1])
        .map_err(|e| invalid(format!("cannot read observations file: {e}")))?;
    let evaluations = fs::read_to_string(&args[2])
        .map_err(|e| invalid(format!("cannot read evaluations file: {e}")))?;
    let summary = admit(&observations, &evaluations)?;
    Ok(render_admission(&summary))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match run_cli(&args) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("REJECT\treason\t{}", error.replace('\t', " "));
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OBS: &str = include_str!("../fixtures/vision-dvpc-observations.tsv");
    const EVAL: &str = include_str!("../fixtures/vision-dvpc-evaluations.tsv");

    #[test]
    fn admits_valid_paired_trace_fixture() {
        let summary = admit(OBS, EVAL).expect("valid fixtures should be admitted");
        assert_eq!(summary.observation_rows, 2);
        assert_eq!(summary.evaluation_rows, 2);
        assert_eq!(summary.groups, 2);
        assert_eq!(summary.split_rows.get("train"), Some(&1));
        assert_eq!(summary.split_rows.get("test"), Some(&1));
        assert!(render_admission(&summary).contains("PASS_SCHEMA_ONLY"));
    }

    #[test]
    fn duplicate_observation_trace_ids_fail_closed() {
        let row = OBS.lines().last().expect("fixture has rows");
        let broken = format!("{}{}\n", OBS, row);
        assert!(admit(&broken, EVAL).unwrap_err().contains("duplicate observation trace_id"));
    }

    #[test]
    fn producer_schema_rejects_evaluator_label_columns() {
        let broken = OBS.replace(
            "OBS_HEADER\ttrace_id",
            "OBS_HEADER\tshift_label\ttrace_id",
        );
        assert!(admit(&broken, EVAL).unwrap_err().contains("header mismatch"));
    }

    #[test]
    fn source_group_cannot_cross_splits() {
        let broken = OBS.replace("seq-test-000", "seq-train-000");
        assert!(admit(&broken, EVAL).unwrap_err().contains("crosses splits"));
    }

    #[test]
    fn sample_digest_mismatch_fails_closed() {
        let original_prefix = format!("EVAL\ttrace-train-000\t{}\t", "e".repeat(64));
        let changed_prefix = format!("EVAL\ttrace-train-000\t{}\t", "f".repeat(64));
        let broken = EVAL.replace(&original_prefix, &changed_prefix);
        assert!(admit(OBS, &broken).unwrap_err().contains("sample digest mismatch"));
    }

    #[test]
    fn out_of_range_utility_fails_closed() {
        let broken = EVAL.replace("\t0.400000\t0.550000\t", "\t1.200000\t0.550000\t");
        assert!(admit(OBS, &broken).unwrap_err().contains("must lie in [0,1]"));
    }

    #[test]
    fn missing_evaluation_pair_fails_closed() {
        let kept: String = EVAL.lines()
            .filter(|line| !line.starts_with("EVAL\ttrace-test-000\t"))
            .map(|line| format!("{line}\n"))
            .collect();
        assert!(admit(OBS, &kept).unwrap_err().contains("missing 1 paired trace"));
    }

    #[test]
    fn unknown_license_scope_fails_closed() {
        let broken = OBS.replace(
            "license_scope\tsynthetic_public_fixture",
            "license_scope\treview_required",
        );
        assert!(admit(&broken, EVAL).unwrap_err().contains("license_scope must be explicit"));
    }

    #[test]
    fn non_finite_uncertainty_fails_closed() {
        let broken = OBS.replace("\t0.100000\t", "\tNaN\t");
        assert!(admit(&broken, EVAL).unwrap_err().contains("must be finite"));
    }
}
