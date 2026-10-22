// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reproducible paired-seed analysis for the episode CSV schema used by the
//! allostatic-control benchmark. One invocation compares one predeclared pair of
//! policies; each scenario × shielded condition is reported separately.

use allostatic_control_bench::paired_statistics::{PairedEpisode, PairedSummary};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

type Condition = (String, bool);
type PolicyValues = BTreeMap<String, (f64, bool)>;
type IndexedRows = BTreeMap<Condition, BTreeMap<u64, PolicyValues>>;

fn main() {
    if let Err(error) = run() {
        eprintln!("FAIL: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 8 {
        return Err(format!(
            "usage: {} INPUT.csv OUTPUT.csv METRIC_COLUMN VIOLATION_COLUMN BASELINE_POLICY CANDIDATE_POLICY BOOTSTRAP_REPLICATES",
            args.first().map_or("allostatic-paired-eval-bench", String::as_str)
        ));
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let metric_name = &args[3];
    let violation_name = &args[4];
    let baseline_policy = &args[5];
    let candidate_policy = &args[6];
    let bootstrap_replicates = args[7]
        .parse::<usize>()
        .map_err(|_| "bootstrap replicate count must be an integer".to_owned())?;

    if baseline_policy == candidate_policy {
        return Err("baseline and candidate policy must differ".to_owned());
    }
    if !is_slug(baseline_policy) || !is_slug(candidate_policy) {
        return Err("policy identifiers must be simple unquoted slugs".to_owned());
    }
    if bootstrap_replicates < 1_000 {
        return Err("use at least 1000 paired bootstrap replicates".to_owned());
    }
    if !is_slug(metric_name) || !is_slug(violation_name) {
        return Err("metric and violation column names must be simple unquoted slugs".to_owned());
    }
    if !Path::new(input_path).is_file() {
        return Err(format!("input CSV does not exist: {input_path}"));
    }
    let canonical_input = fs::canonicalize(input_path)
        .map_err(|error| format!("cannot resolve input CSV {input_path}: {error}"))?;
    let canonical_output = canonical_output_path(Path::new(output_path))?;
    if canonical_input == canonical_output {
        return Err("input and output resolve to the same file; refusing to overwrite source data".to_owned());
    }

    let content = fs::read_to_string(input_path)
        .map_err(|error| format!("cannot read input CSV {input_path}: {error}"))?;
    let mut lines = content.lines();
    let header_line = lines.next().ok_or("input CSV is empty")?;
    let headers = csv_fields(header_line, 1)?;
    let mut unique_headers = BTreeSet::new();
    if headers.iter().any(|header| !unique_headers.insert(*header)) {
        return Err("input CSV contains duplicate header names".to_owned());
    }

    let scenario_col = column_index(&headers, "scenario")?;
    let seed_col = column_index(&headers, "seed")?;
    let policy_col = column_index(&headers, "policy")?;
    let shielded_col = column_index(&headers, "shielded")?;
    let metric_col = column_index(&headers, metric_name)?;
    let violation_col = column_index(&headers, violation_name)?;
    let mut rows: IndexedRows = BTreeMap::new();

    for (zero_index, line) in lines.enumerate() {
        let line_number = zero_index + 2;
        if line.trim().is_empty() {
            continue;
        }
        let fields = csv_fields(line, line_number)?;
        if fields.len() != headers.len() {
            return Err(format!(
                "line {line_number} has {} fields; header has {}",
                fields.len(),
                headers.len()
            ));
        }

        let policy = fields[policy_col];
        if policy != baseline_policy && policy != candidate_policy {
            continue;
        }

        let scenario = fields[scenario_col];
        if !is_slug(scenario) {
            return Err(format!(
                "line {line_number} has invalid scenario key: {scenario:?}"
            ));
        }
        let seed = fields[seed_col]
            .parse::<u64>()
            .map_err(|_| format!("line {line_number} has invalid numeric seed"))?;
        let shielded = parse_bool(fields[shielded_col])
            .map_err(|_| format!("line {line_number} has invalid shielded flag"))?;
        let metric = fields[metric_col]
            .parse::<f64>()
            .map_err(|_| format!("line {line_number} has invalid metric value"))?;
        if !metric.is_finite() {
            return Err(format!("line {line_number} has a non-finite metric"));
        }
        let violation = parse_bool(fields[violation_col])
            .map_err(|_| format!("line {line_number} has invalid episode-violation flag"))?;

        let policy_values = rows
            .entry((scenario.to_owned(), shielded))
            .or_default()
            .entry(seed)
            .or_default();
        if policy_values
            .insert(policy.to_owned(), (metric, violation))
            .is_some()
        {
            return Err(format!(
                "duplicate row for scenario={scenario}, seed={seed}, shielded={shielded}, policy={policy}"
            ));
        }
    }

    if rows.is_empty() {
        return Err("no rows found for the selected baseline/candidate policies".to_owned());
    }

    let mut summaries: Vec<(String, bool, u64, PairedSummary)> = Vec::new();
    for ((scenario, shielded), by_seed) in rows {
        let mut paired = Vec::with_capacity(by_seed.len());
        for (seed, policies) in by_seed {
            let baseline = policies.get(baseline_policy).ok_or_else(|| {
                format!(
                    "missing baseline row for scenario={scenario}, seed={seed}, shielded={shielded}"
                )
            })?;
            let candidate = policies.get(candidate_policy).ok_or_else(|| {
                format!(
                    "missing candidate row for scenario={scenario}, seed={seed}, shielded={shielded}"
                )
            })?;
            if policies.len() != 2 {
                return Err(format!(
                    "unexpected duplicate/extra selected policy rows for scenario={scenario}, seed={seed}"
                ));
            }
            paired.push(PairedEpisode {
                seed,
                baseline_metric: baseline.0,
                candidate_metric: candidate.0,
                baseline_violation: baseline.1,
                candidate_violation: candidate.1,
            });
        }

        let random_seed = 0xA110_57A7_u64
            .wrapping_add(scenario.bytes().map(u64::from).sum::<u64>())
            .wrapping_add(if shielded { 1 } else { 0 });
        let summary = PairedSummary::from_samples(&paired, bootstrap_replicates, random_seed)
            .map_err(|error| format!("{scenario}/{shielded}: {error}"))?;
        summaries.push((scenario, shielded, random_seed, summary));
    }

    // Write to a sibling temporary file and rename only after all rows are complete.
    // This prevents a failed write from leaving a truncated "successful" report.
    let target = Path::new(output_path);
    let temporary = temporary_sibling_path(target)?;
    let write_result = (|| -> Result<(), String> {
        let mut report = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| format!("cannot create temporary report {}: {error}", temporary.display()))?;
        writeln!(
            report,
            "scenario,shielded,metric,baseline_policy,candidate_policy,n_pairs,baseline_mean,candidate_mean,candidate_minus_baseline_mean,median_paired_delta,paired_bootstrap_ci95_low,paired_bootstrap_ci95_high,bootstrap_replicates,bootstrap_seed,baseline_violations,candidate_violations,candidate_only_violations,baseline_only_violations,mcnemar_exact_two_sided_p,baseline_violation_rate,candidate_violation_rate,baseline_violation_upper95_exact,candidate_violation_upper95_exact"
        )
        .map_err(|error| format!("cannot write report header: {error}"))?;
        for (scenario, shielded, random_seed, summary) in summaries {
            write_summary_row(
                &mut report,
                &scenario,
                shielded,
                metric_name,
                baseline_policy,
                candidate_policy,
                random_seed,
                &summary,
            )?;
        }
        report
            .flush()
            .map_err(|error| format!("cannot flush report: {error}"))?;
        report
            .sync_all()
            .map_err(|error| format!("cannot sync report: {error}"))?;
        drop(report);
        fs::rename(&temporary, target)
            .map_err(|error| format!("cannot atomically publish report {}: {error}", target.display()))?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
}

fn canonical_output_path(path: &Path) -> Result<std::path::PathBuf, String> {
    if path.exists() {
        return fs::canonicalize(path)
            .map_err(|error| format!("cannot resolve output path {}: {error}", path.display()));
    }
    let filename = path
        .file_name()
        .ok_or_else(|| format!("output path has no filename: {}", path.display()))?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let canonical_parent = fs::canonicalize(parent)
        .map_err(|error| format!("cannot resolve output directory {}: {error}", parent.display()))?;
    Ok(canonical_parent.join(filename))
}

fn temporary_sibling_path(target: &Path) -> Result<std::path::PathBuf, String> {
    let filename = target
        .file_name()
        .ok_or_else(|| format!("output path has no filename: {}", target.display()))?;
    let parent = target
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(parent.join(format!(
        ".{}.tmp-{}",
        filename.to_string_lossy(),
        std::process::id()
    )))
}

fn csv_fields(line: &str, line_number: usize) -> Result<Vec<&str>, String> {
    if line.contains('"') {
        return Err(format!(
            "line {line_number} contains CSV quoting; this strict reader accepts only unquoted fields"
        ));
    }
    Ok(line.trim_end_matches('\r').split(',').collect())
}

fn column_index(headers: &[&str], name: &str) -> Result<usize, String> {
    headers
        .iter()
        .position(|header| *header == name)
        .ok_or_else(|| format!("required CSV column is missing: {name}"))
}

fn parse_bool(value: &str) -> Result<bool, ()> {
    match value {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        _ => Err(()),
    }
}

fn is_slug(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-' || byte == b'.')
}

fn write_summary_row(
    output: &mut fs::File,
    scenario: &str,
    shielded: bool,
    metric_name: &str,
    baseline_policy: &str,
    candidate_policy: &str,
    random_seed: u64,
    summary: &PairedSummary,
) -> Result<(), String> {
    let baseline_rate = summary.baseline_violations as f64 / summary.n_pairs as f64;
    let candidate_rate = summary.candidate_violations as f64 / summary.n_pairs as f64;
    writeln!(
        output,
        "{scenario},{shielded},{metric_name},{baseline_policy},{candidate_policy},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9}",
        summary.n_pairs,
        summary.baseline_mean,
        summary.candidate_mean,
        summary.mean_delta,
        summary.median_delta,
        summary.bootstrap_ci95_low,
        summary.bootstrap_ci95_high,
        summary.bootstrap_replicates,
        random_seed,
        summary.baseline_violations,
        summary.candidate_violations,
        summary.candidate_only_violations,
        summary.baseline_only_violations,
        summary.mcnemar_exact_two_sided_p,
        baseline_rate,
        candidate_rate,
        summary.baseline_violation_upper95_exact,
        summary.candidate_violation_upper95_exact,
    )
    .map_err(|error| format!("cannot write summary row: {error}"))
}

#[cfg(test)]
mod tests {
    use super::{csv_fields, is_slug, parse_bool};

    #[test]
    fn metric_columns_must_be_safe_unquoted_slugs() {
        assert!(is_slug("task_reward"));
        assert!(!is_slug("reward,unsafe"));
        assert!(!is_slug("reward name"));
    }

    #[test]
    fn csv_reader_rejects_quotes_instead_of_silently_misparsing() {
        assert!(csv_fields("a,\"b,c\"", 7).is_err());
    }

    #[test]
    fn identifiers_are_restricted_to_unquoted_safe_slugs() {
        assert!(is_slug("safe_control_gym-cartpole.v1"));
        assert!(!is_slug("unsafe,other"));
        assert!(!is_slug(""));
    }

    #[test]
    fn boolean_parser_is_fail_closed() {
        assert_eq!(parse_bool("true"), Ok(true));
        assert_eq!(parse_bool("false"), Ok(false));
        assert!(parse_bool("unknown").is_err());
    }
}
