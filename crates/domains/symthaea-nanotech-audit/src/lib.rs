//! Read-only artifact integrity checks for exported MADE benchmark runs.
//!
//! This verifier checks whether exported trajectories, per-episode summaries,
//! CSV records and aggregate metric summaries agree with one another. It does
//! not verify source/data provenance, oracle call costs, scientific validity,
//! or experimental confirmation.

use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const ABS_TOLERANCE: f64 = 1.0e-10;
const REL_TOLERANCE: f64 = 1.0e-9;

#[derive(Debug, Clone, Serialize)]
pub struct AuditFinding {
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditReport {
    pub status: String,
    pub run_dir: String,
    pub trajectory_files: usize,
    pub episode_rows: usize,
    pub findings: Vec<AuditFinding>,
    pub checks_completed: Vec<String>,
    pub scope_ceiling: Vec<String>,
}

impl AuditReport {
    fn new(run_dir: &Path) -> Self {
        Self {
            status: "fail".to_owned(),
            run_dir: run_dir.display().to_string(),
            trajectory_files: 0,
            episode_rows: 0,
            findings: Vec::new(),
            checks_completed: Vec::new(),
            scope_ceiling: vec![
                "An artifact-integrity pass is not a scientific-performance pass.".to_owned(),
                "Source commit, data rights and dataset identity/hash are not verified.".to_owned(),
                "Oracle setup calls, caching, retries and total compute cost are not verified.".to_owned(),
                "No physical experiment or independent replication is verified.".to_owned(),
            ],
        }
    }

    fn fail(&mut self, code: impl Into<String>, detail: impl Into<String>) {
        self.findings.push(AuditFinding {
            code: code.into(),
            detail: detail.into(),
        });
    }

    fn checked(&mut self, check: impl Into<String>) {
        self.checks_completed.push(check.into());
    }
}

/// Audits a standard exported MADE run directory.
pub fn audit_run(run_dir: &Path) -> AuditReport {
    let mut report = AuditReport::new(run_dir);
    let config_path = run_dir.join(".hydra").join("config.yaml");
    match fs::read_to_string(&config_path) {
        Ok(config) if !config.trim().is_empty() => {
            report.checked("resolved_config_present_and_nonempty");
        }
        Ok(_) => report.fail(
            "empty_config",
            format!("resolved config is empty: {}", config_path.display()),
        ),
        Err(error) => report.fail(
            "missing_or_unreadable_config",
            format!("{}: {error}", config_path.display()),
        ),
    }

    let episodes_path = run_dir.join("summary").join("episodes.json");
    let aggregate_path = run_dir.join("summary").join("summary.json");
    let csv_path = run_dir.join("summary").join("episodes.csv");
    let episodes_value = read_json(&episodes_path, &mut report);
    let aggregate_value = read_json(&aggregate_path, &mut report);
    let episode_rows = episodes_value.as_ref().and_then(Value::as_array).cloned();
    let aggregate = aggregate_value.as_ref().and_then(Value::as_object);

    if episodes_value.is_some() && episode_rows.is_none() {
        report.fail(
            "episodes_json_not_array",
            format!("{} must contain a JSON array", episodes_path.display()),
        );
    }
    if aggregate_value.is_some() && aggregate.is_none() {
        report.fail(
            "summary_json_not_object",
            format!("{} must contain a JSON object", aggregate_path.display()),
        );
    }

    let mut trajectory_files = match discover_trajectories(&run_dir.join("trajectories")) {
        Ok(files) => files,
        Err(error) => {
            report.fail("trajectory_directory_unreadable", error);
            Vec::new()
        }
    };
    trajectory_files.sort_by_key(|(index, _)| *index);
    report.trajectory_files = trajectory_files.len();
    report.episode_rows = episode_rows.as_ref().map_or(0, Vec::len);
    if episode_rows.as_ref().is_some_and(|rows| rows.is_empty()) {
        report.fail("no_episode_rows", "an audit run must contain at least one completed episode");
    }

    if let Some(rows) = episode_rows.as_ref() {
        if let Some(aggregate) = aggregate {
            match aggregate.get("episodes").and_then(Value::as_u64) {
                Some(count) if count as usize == rows.len() => {
                    report.checked("aggregate_episode_count_matches_rows");
                }
                Some(count) => report.fail(
                    "aggregate_episode_count_mismatch",
                    format!(
                        "summary.json declares {count} episodes, episodes.json has {}",
                        rows.len()
                    ),
                ),
                None => report.fail(
                    "aggregate_episode_count_missing",
                    "summary.json must contain a non-negative integer field named episodes",
                ),
            }
        }

        let trajectory_findings_before = report.findings.len();
        if trajectory_files.len() == rows.len() {
            report.checked("trajectory_count_matches_episode_rows");
        } else {
            report.fail(
                "trajectory_count_mismatch",
                format!(
                    "found {} episode trajectory files but {} rows in episodes.json",
                    trajectory_files.len(),
                    rows.len()
                ),
            );
        }

        for (expected_index, (actual_index, _)) in trajectory_files.iter().enumerate() {
            if expected_index != *actual_index {
                report.fail(
                    "trajectory_sequence_gap",
                    format!(
                        "expected episode index {expected_index} but found {actual_index}; indices must start at zero and be contiguous"
                    ),
                );
                break;
            }
        }

        for (episode_index, ((file_index, path), row)) in
            trajectory_files.iter().zip(rows.iter()).enumerate()
        {
            if *file_index != episode_index {
                continue;
            }
            let Some(trajectory_doc) = read_json(path, &mut report) else {
                continue;
            };
            let Some(metrics) = trajectory_doc.get("metrics").and_then(Value::as_object) else {
                report.fail(
                    "trajectory_metrics_missing",
                    format!("{} has no metrics object", path.display()),
                );
                continue;
            };

            if !metrics
                .get("episode_id")
                .and_then(Value::as_u64)
                .is_some_and(|id| id == episode_index as u64)
            {
                report.fail(
                    "trajectory_episode_id_mismatch",
                    format!(
                        "{} metrics.episode_id does not equal {episode_index}",
                        path.display()
                    ),
                );
            }
            if !values_equal(&Value::Object(metrics.clone()), row) {
                report.fail(
                    "episode_metrics_mismatch",
                    format!(
                        "{} metrics object differs from its corresponding summary/episodes.json row",
                        path.display()
                    ),
                );
            }

            let Some(raw_trajectory) = trajectory_doc.get("trajectory").and_then(Value::as_array) else {
                report.fail(
                    "raw_trajectory_missing",
                    format!("{} has no trajectory array", path.display()),
                );
                continue;
            };
            match metrics.get("final/queries_used").and_then(Value::as_u64) {
                Some(query_count) if query_count as usize == raw_trajectory.len() => {}
                Some(query_count) => report.fail(
                    "query_count_mismatch",
                    format!(
                        "{} declares {query_count} queries but contains {} trajectory observations",
                        path.display(),
                        raw_trajectory.len()
                    ),
                ),
                None => report.fail(
                    "query_count_missing",
                    format!(
                        "{} has no non-negative integer final/queries_used metric",
                        path.display()
                    ),
                ),
            }

            if trajectory_doc
                .get("final_env_state")
                .and_then(Value::as_object)
                .and_then(|state| state.get("phase_diagram_all_entries"))
                .and_then(Value::as_array)
                .is_none()
            {
                report.fail(
                    "final_phase_diagram_missing",
                    format!(
                        "{} lacks final_env_state.phase_diagram_all_entries",
                        path.display()
                    ),
                );
            }
            if trajectory_doc.get("phase_diagram_gt").and_then(Value::as_array).is_none() {
                report.fail(
                    "ground_truth_phase_diagram_missing",
                    format!("{} lacks phase_diagram_gt array", path.display()),
                );
            }
        }

        if !rows.is_empty()
            && trajectory_files.len() == rows.len()
            && report.findings.len() == trajectory_findings_before
        {
            report.checked("trajectory_and_per_episode_records_reconcile");
        }

        audit_csv(&csv_path, rows, &mut report);
        audit_aggregate(rows, aggregate_value.as_ref(), &mut report);
    }

    report.status = if report.findings.is_empty() {
        "artifact_integrity_pass".to_owned()
    } else {
        "fail".to_owned()
    };
    report
}

fn read_json(path: &Path, report: &mut AuditReport) -> Option<Value> {
    match fs::read_to_string(path) {
        Ok(content) => match serde_json::from_str::<Value>(&content) {
            Ok(value) => Some(value),
            Err(error) => {
                report.fail("invalid_json", format!("{}: {error}", path.display()));
                None
            }
        },
        Err(error) => {
            report.fail(
                "missing_or_unreadable_json",
                format!("{}: {error}", path.display()),
            );
            None
        }
    }
}

fn discover_trajectories(directory: &Path) -> Result<Vec<(usize, PathBuf)>, String> {
    let entries =
        fs::read_dir(directory).map_err(|error| format!("{}: {error}", directory.display()))?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("{}: {error}", directory.display()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.starts_with("episode_") || !name.ends_with(".json") {
            continue;
        }
        let digits = &name["episode_".len()..name.len() - ".json".len()];
        if digits.is_empty() || !digits.chars().all(|character| character.is_ascii_digit()) {
            return Err(format!("malformed episode filename: {}", path.display()));
        }
        let index = digits
            .parse::<usize>()
            .map_err(|error| format!("episode index out of range in {}: {error}", path.display()))?;
        files.push((index, path));
    }
    Ok(files)
}

fn audit_csv(path: &Path, rows: &[Value], report: &mut AuditReport) {
    let mut reader = match csv::ReaderBuilder::new().has_headers(true).from_path(path) {
        Ok(reader) => reader,
        Err(error) => {
            report.fail("csv_unreadable", format!("{}: {error}", path.display()));
            return;
        }
    };
    let headers = match reader.headers() {
        Ok(headers) => headers.iter().map(str::to_owned).collect::<Vec<_>>(),
        Err(error) => {
            report.fail("csv_header_invalid", format!("{}: {error}", path.display()));
            return;
        }
    };

    let expected_headers = rows
        .iter()
        .filter_map(Value::as_object)
        .flat_map(|object| object.keys().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    if headers != expected_headers {
        report.fail(
            "csv_header_mismatch",
            format!(
                "{} header differs from the sorted union of episode metric keys",
                path.display()
            ),
        );
        return;
    }

    let mut row_count = 0usize;
    for (row_index, csv_row) in reader.records().enumerate() {
        let csv_row = match csv_row {
            Ok(row) => row,
            Err(error) => {
                report.fail(
                    "csv_row_invalid",
                    format!("{} row {}: {error}", path.display(), row_index + 2),
                );
                continue;
            }
        };
        row_count += 1;
        let Some(json_row) = rows.get(row_index).and_then(Value::as_object) else {
            report.fail(
                "csv_has_extra_rows",
                format!(
                    "{} has a row without a matching episodes.json object",
                    path.display()
                ),
            );
            continue;
        };

        for (column, key) in headers.iter().enumerate() {
            let actual = csv_row.get(column).unwrap_or_default();
            let expected = json_row.get(key);
            let matches = match expected {
                None | Some(Value::Null) => actual.is_empty(),
                Some(Value::Number(number)) => number
                    .as_f64()
                    .zip(actual.parse::<f64>().ok())
                    .is_some_and(|(left, right)| floats_equal(left, right)),
                Some(Value::Bool(value)) => actual == if *value { "True" } else { "False" },
                Some(Value::String(value)) => actual == value,
                Some(_) => false,
            };
            if !matches {
                report.fail(
                    "csv_cell_mismatch",
                    format!(
                        "{} row {}, column {} does not match episodes.json",
                        path.display(),
                        row_index + 2,
                        key
                    ),
                );
            }
        }
    }

    if row_count != rows.len() {
        report.fail(
            "csv_row_count_mismatch",
            format!(
                "{} has {row_count} data rows but episodes.json has {}",
                path.display(),
                rows.len()
            ),
        );
    } else if !report.findings.iter().any(|finding| finding.code.starts_with("csv_")) {
        report.checked("episodes_csv_reconciles");
    }
}

fn audit_aggregate(rows: &[Value], aggregate_value: Option<&Value>, report: &mut AuditReport) {
    let Some(aggregate_object) = aggregate_value.and_then(Value::as_object) else {
        return;
    };
    let Some(aggregate) = aggregate_object.get("summary").and_then(Value::as_object) else {
        report.fail(
            "aggregate_summary_missing",
            "summary.json must contain a summary object",
        );
        return;
    };
    if rows.is_empty() {
        return;
    }
    let findings_before = report.findings.len();

    let mut values_by_metric: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for row in rows {
        let Some(object) = row.as_object() else {
            report.fail("episode_row_not_object", "each episodes.json row must be an object");
            continue;
        };
        for (key, value) in object {
            if !key.starts_with("final/") {
                continue;
            }
            if let Some(number) = value.as_f64() {
                if number.is_finite() {
                    values_by_metric.entry(key.clone()).or_default().push(number);
                } else {
                    report.fail(
                        "non_finite_metric",
                        format!("episodes.json contains non-finite metric {key}"),
                    );
                }
            }
        }
    }

    let expected_keys = values_by_metric.keys().cloned().collect::<BTreeSet<_>>();
    let actual_keys = aggregate.keys().cloned().collect::<BTreeSet<_>>();
    if expected_keys != actual_keys {
        report.fail(
            "aggregate_metric_keys_mismatch",
            format!(
                "summary metric keys differ; expected {expected_keys:?}, got {actual_keys:?}"
            ),
        );
        return;
    }

    for (key, values) in values_by_metric {
        if values.is_empty() {
            report.fail("empty_aggregate_metric", format!("no numeric values for {key}"));
            continue;
        }
        let Some(metric_summary) = aggregate.get(&key).and_then(Value::as_object) else {
            report.fail(
                "aggregate_metric_invalid",
                format!("summary for {key} must be an object"),
            );
            continue;
        };

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / n;
        let std = variance.sqrt();
        let sem = std / n.sqrt();

        for (name, expected) in [("mean", mean), ("std", std), ("sem", sem)] {
            match metric_summary.get(name).and_then(Value::as_f64) {
                Some(actual) if floats_equal(actual, expected) => {}
                Some(actual) => report.fail(
                    "aggregate_statistic_mismatch",
                    format!(
                        "summary {key}.{name} is {actual} but independently reconstructed value is {expected}"
                    ),
                ),
                None => report.fail(
                    "aggregate_statistic_missing",
                    format!("summary {key} lacks numeric field {name}"),
                ),
            }
        }
    }

    if report.findings.len() == findings_before {
        report.checked("aggregate_mean_std_sem_reconstructed");
    }
}

fn values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => left
            .as_f64()
            .zip(right.as_f64())
            .is_some_and(|(left, right)| floats_equal(left, right)),
        (Value::Array(left), Value::Array(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right.iter())
                    .all(|(left, right)| values_equal(left, right))
        }
        (Value::Object(left), Value::Object(right)) => {
            left.len() == right.len()
                && left.iter().all(|(key, value)| {
                    right.get(key).is_some_and(|other| values_equal(value, other))
                })
        }
        _ => left == right,
    }
}

fn floats_equal(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs()
            <= ABS_TOLERANCE.max(REL_TOLERANCE * left.abs().max(right.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "symthaea-nanotech-audit-{}-{id}",
                std::process::id()
            ));
            fs::create_dir_all(path.join(".hydra")).expect("create config dir");
            fs::create_dir_all(path.join("trajectories")).expect("create trajectories dir");
            fs::create_dir_all(path.join("summary")).expect("create summary dir");
            fs::write(
                path.join(".hydra/config.yaml"),
                "experiment:\n  num_episodes: 2\n",
            )
            .expect("write config");

            let metrics0 = json!({
                "episode_id": 0,
                "final/queries_used": 2,
                "final/score": 1.0
            });
            let metrics1 = json!({
                "episode_id": 1,
                "final/queries_used": 2,
                "final/score": 3.0
            });

            for (index, metrics) in [(0, &metrics0), (1, &metrics1)] {
                let result = json!({
                    "metrics": metrics,
                    "trajectory": [{ "step": 1 }, { "step": 2 }],
                    "final_env_state": { "phase_diagram_all_entries": [] },
                    "phase_diagram_gt": []
                });
                fs::write(
                    path.join("trajectories").join(format!("episode_{index:03}.json")),
                    serde_json::to_vec_pretty(&result).expect("serialize episode"),
                )
                .expect("write episode");
            }

            fs::write(
                path.join("summary/episodes.json"),
                serde_json::to_vec_pretty(&json!([metrics0, metrics1])).expect("serialize rows"),
            )
            .expect("write episode rows");
            fs::write(
                path.join("summary/summary.json"),
                serde_json::to_vec_pretty(&json!({
                    "episodes": 2,
                    "summary": {
                        "final/queries_used": { "mean": 2.0, "std": 0.0, "sem": 0.0 },
                        "final/score": {
                            "mean": 2.0,
                            "std": 1.0,
                            "sem": 0.7071067811865475
                        }
                    }
                }))
                .expect("serialize aggregate"),
            )
            .expect("write aggregate");
            fs::write(
                path.join("summary/episodes.csv"),
                "episode_id,final/queries_used,final/score\n0,2,1.0\n1,2,3.0\n",
            )
            .expect("write csv");
            Self(path)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn accepts_consistent_episode_trajectory_csv_and_aggregate() {
        let fixture = Fixture::new();
        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "artifact_integrity_pass", "{:?}", report.findings);
        assert!(report.findings.is_empty());
        assert!(report
            .checks_completed
            .contains(&"aggregate_mean_std_sem_reconstructed".to_owned()));
    }

    #[test]
    fn rejects_a_mismatched_aggregate_mean() {
        let fixture = Fixture::new();
        let path = fixture.0.join("summary/summary.json");
        let mut aggregate: Value =
            serde_json::from_str(&fs::read_to_string(&path).expect("read summary")).unwrap();
        aggregate["summary"]["final/score"]["mean"] = json!(99.0);
        fs::write(&path, serde_json::to_vec_pretty(&aggregate).unwrap()).unwrap();

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "aggregate_statistic_mismatch"));
    }

    #[test]
    fn rejects_trajectory_query_count_mismatch() {
        let fixture = Fixture::new();
        let path = fixture.0.join("trajectories/episode_000.json");
        let mut episode: Value =
            serde_json::from_str(&fs::read_to_string(&path).expect("read episode")).unwrap();
        episode["trajectory"].as_array_mut().unwrap().pop();
        fs::write(&path, serde_json::to_vec_pretty(&episode).unwrap()).unwrap();

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "query_count_mismatch"));
    }

    #[test]
    fn rejects_an_empty_episode_set() {
        let fixture = Fixture::new();
        fs::write(
            fixture.0.join("summary/episodes.json"),
            serde_json::to_vec_pretty(&json!([])).unwrap(),
        )
        .unwrap();

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert!(report.findings.iter().any(|finding| finding.code == "no_episode_rows"));
    }

    #[test]
    fn rejects_a_gap_in_episode_indices() {
        let fixture = Fixture::new();
        fs::rename(
            fixture.0.join("trajectories/episode_001.json"),
            fixture.0.join("trajectories/episode_002.json"),
        )
        .unwrap();

        let report = audit_run(&fixture.0);
        assert_eq!(report.status, "fail");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.code == "trajectory_sequence_gap"));
    }
}
