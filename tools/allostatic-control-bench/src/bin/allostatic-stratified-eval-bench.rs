// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Task-balanced aggregation of paired normalized scores.
//!
//! Input contract: task,seed,baseline_score,candidate_score; one pair per seed/task.
//! Scores must already be normalized to [0, 1] using a predeclared task-specific
//! mapping where higher is better. The CLI does not normalize or choose directions.

use allostatic_control_bench::stratified_metrics::{StratifiedSummary, TaskEpisode};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as FmtWrite;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const REPORT_HEADER: &str = "row_type,task,n_pairs,baseline_mean,candidate_mean,mean_delta,iqm_delta,paired_win_rate,macro_baseline_mean,macro_candidate_mean,macro_mean_delta,macro_mean_delta_ci95_low,macro_mean_delta_ci95_high,macro_iqm_delta,macro_iqm_delta_ci95_low,macro_iqm_delta_ci95_high,macro_paired_win_rate,macro_paired_win_rate_ci95_low,macro_paired_win_rate_ci95_high,fraction_tasks_positive_mean_delta,task_count,bootstrap_replicates,bootstrap_seed";

fn main() {
    if let Err(error) = run() {
        eprintln!("FAIL: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        return Err(format!(
            "usage: {} INPUT.csv OUTPUT.csv BOOTSTRAP_REPLICATES BOOTSTRAP_SEED",
            args.first().map_or("allostatic-stratified-eval-bench", String::as_str)
        ));
    }
    let input_path = Path::new(&args[1]);
    let output_path = Path::new(&args[2]);
    let bootstrap_replicates = args[3]
        .parse::<usize>()
        .map_err(|_| "bootstrap replicate count must be an integer".to_owned())?;
    let bootstrap_seed = args[4]
        .parse::<u64>()
        .map_err(|_| "bootstrap seed must be a nonnegative integer".to_owned())?;
    if bootstrap_replicates < 1_000 {
        return Err("use at least 1000 stratified bootstrap replicates".to_owned());
    }
    if !input_path.is_file() {
        return Err(format!("input CSV does not exist: {}", input_path.display()));
    }
    ensure_distinct_paths(input_path, output_path)?;

    let content = fs::read_to_string(input_path)
        .map_err(|error| format!("cannot read input CSV {}: {error}", input_path.display()))?;
    let mut lines = content.lines();
    let header = lines.next().ok_or("input CSV is empty")?;
    let headers = csv_fields(header, 1)?;
    let mut unique_headers = BTreeSet::new();
    if headers.iter().any(|name| !unique_headers.insert(*name)) {
        return Err("input CSV contains duplicate header names".to_owned());
    }
    let expected = ["task", "seed", "baseline_score", "candidate_score"];
    if headers.as_slice() != &expected[..] {
        return Err(format!(
            "expected exact input header {}, found {}",
            expected.join(","),
            headers.join(",")
        ));
    }

    let mut tasks: BTreeMap<String, BTreeMap<u64, TaskEpisode>> = BTreeMap::new();
    for (zero_index, line) in lines.enumerate() {
        let line_number = zero_index + 2;
        if line.trim().is_empty() {
            continue;
        }
        let fields = csv_fields(line, line_number)?;
        if fields.len() != expected.len() {
            return Err(format!(
                "line {line_number} has {} fields; expected {}",
                fields.len(),
                expected.len()
            ));
        }
        let task = fields[0];
        if !is_slug(task) {
            return Err(format!("line {line_number} has invalid task identifier: {task:?}"));
        }
        let seed = fields[1]
            .parse::<u64>()
            .map_err(|_| format!("line {line_number} has invalid numeric seed"))?;
        let baseline_score = fields[2]
            .parse::<f64>()
            .map_err(|_| format!("line {line_number} has invalid baseline score"))?;
        let candidate_score = fields[3]
            .parse::<f64>()
            .map_err(|_| format!("line {line_number} has invalid candidate score"))?;
        if !baseline_score.is_finite()
            || !candidate_score.is_finite()
            || !(0.0..=1.0).contains(&baseline_score)
            || !(0.0..=1.0).contains(&candidate_score)
        {
            return Err(format!(
                "line {line_number} scores must be finite normalized values in [0, 1]"
            ));
        }

        let per_task = tasks.entry(task.to_owned()).or_default();
        if per_task
            .insert(seed, TaskEpisode { seed, baseline_score, candidate_score })
            .is_some()
        {
            return Err(format!("duplicate task/seed pair: {task}/{seed}"));
        }
    }
    if tasks.is_empty() {
        return Err("input has no episode rows".to_owned());
    }

    let episode_sets: BTreeMap<String, Vec<TaskEpisode>> = tasks
        .into_iter()
        .map(|(task, episodes)| (task, episodes.into_values().collect()))
        .collect();
    let (task_summaries, aggregate) = StratifiedSummary::from_tasks(
        &episode_sets,
        bootstrap_replicates,
        bootstrap_seed,
    )?;

    let mut report = String::new();
    writeln!(&mut report, "{REPORT_HEADER}")
        .map_err(|error| format!("cannot write report header: {error}"))?;
    for summary in task_summaries {
        let mut row = vec![
            "task".to_owned(),
            summary.task,
            summary.n_pairs.to_string(),
            format!("{:.9}", summary.baseline_mean),
            format!("{:.9}", summary.candidate_mean),
            format!("{:.9}", summary.mean_delta),
            format!("{:.9}", summary.iqm_delta),
            format!("{:.9}", summary.paired_win_rate),
        ];
        row.resize(23, String::new());
        writeln!(&mut report, "{}", row.join(","))
            .map_err(|error| format!("cannot write task row: {error}"))?;
    }

    let aggregate_row = vec![
        "aggregate".to_owned(),
        "ALL".to_owned(),
        aggregate.total_pairs.to_string(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        format!("{:.9}", aggregate.macro_baseline_mean),
        format!("{:.9}", aggregate.macro_candidate_mean),
        format!("{:.9}", aggregate.macro_mean_delta),
        format!("{:.9}", aggregate.macro_mean_delta_ci95_low),
        format!("{:.9}", aggregate.macro_mean_delta_ci95_high),
        format!("{:.9}", aggregate.macro_iqm_delta),
        format!("{:.9}", aggregate.macro_iqm_delta_ci95_low),
        format!("{:.9}", aggregate.macro_iqm_delta_ci95_high),
        format!("{:.9}", aggregate.macro_paired_win_rate),
        format!("{:.9}", aggregate.macro_paired_win_rate_ci95_low),
        format!("{:.9}", aggregate.macro_paired_win_rate_ci95_high),
        format!("{:.9}", aggregate.fraction_tasks_positive_mean_delta),
        aggregate.task_count.to_string(),
        aggregate.bootstrap_replicates.to_string(),
        aggregate.bootstrap_seed.to_string(),
    ];
    writeln!(&mut report, "{}", aggregate_row.join(","))
        .map_err(|error| format!("cannot write aggregate row: {error}"))?;

    atomic_write(output_path, report.as_bytes())
}

fn csv_fields(line: &str, line_number: usize) -> Result<Vec<&str>, String> {
    if line.contains('"') {
        return Err(format!(
            "line {line_number} contains CSV quoting; use the strict unquoted schema"
        ));
    }
    Ok(line.trim_end_matches('\r').split(',').collect())
}

fn is_slug(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| {
                byte.is_ascii_alphanumeric()
                    || byte == b'_'
                    || byte == b'-'
                    || byte == b'.'
            })
}

fn ensure_distinct_paths(input: &Path, output: &Path) -> Result<(), String> {
    let canonical_input = fs::canonicalize(input)
        .map_err(|error| format!("cannot resolve input CSV {}: {error}", input.display()))?;
    let canonical_output = canonical_output_path(output)?;
    if canonical_input == canonical_output {
        return Err(
            "input and output resolve to the same file; refusing to overwrite source data".to_owned(),
        );
    }
    Ok(())
}

fn canonical_output_path(path: &Path) -> Result<PathBuf, String> {
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

fn atomic_write(target: &Path, content: &[u8]) -> Result<(), String> {
    let filename = target
        .file_name()
        .ok_or_else(|| format!("output path has no filename: {}", target.display()))?;
    let parent = target
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let temporary = parent.join(format!(
        ".{}.tmp-{}",
        filename.to_string_lossy(),
        std::process::id()
    ));
    // Open outside the cleanup closure: if create_new fails because another
    // writer owns this temporary path, do not remove that writer's file.
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| {
            format!("cannot create temporary report {}: {error}", temporary.display())
        })?;
    let write_result = (|| -> Result<(), String> {
        file.write_all(content)
            .map_err(|error| format!("cannot write temporary report: {error}"))?;
        file.flush()
            .map_err(|error| format!("cannot flush temporary report: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("cannot sync temporary report: {error}"))?;
        drop(file);
        fs::rename(&temporary, target)
            .map_err(|error| {
                format!("cannot atomically publish report {}: {error}", target.display())
            })?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{csv_fields, ensure_distinct_paths, is_slug};
    use std::fs;

    #[test]
    fn strict_csv_rejects_quoted_fields() {
        assert!(csv_fields("task,1,0.5,\"0.6,0.7\"", 2).is_err());
    }

    #[test]
    fn task_names_must_be_slugs() {
        assert!(is_slug("safe-control_gym.cartpole"));
        assert!(!is_slug("cartpole,unsafe"));
    }

    #[test]
    fn input_output_path_alias_is_rejected() {
        let source = std::env::temp_dir().join(format!(
            "allostatic-stratified-same-path-{}.csv",
            std::process::id()
        ));
        fs::write(&source, "task,seed,baseline_score,candidate_score\n")
            .expect("write temporary source fixture");
        let alias = source
            .parent()
            .expect("temp file has a parent")
            .join(".")
            .join(source.file_name().expect("temp file has a name"));
        let result = ensure_distinct_paths(&source, &alias);
        fs::remove_file(&source).expect("remove temporary source fixture");
        assert!(result.expect_err("same path must be rejected").contains("same file"));
    }
}
