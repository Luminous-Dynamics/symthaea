//! Rust scorer for upstream HiddenBench result files.
//!
//! Reproduces the official per-run average-vote and strict-majority metrics,
//! then adds task-clustered summaries and paired task-bootstrap intervals.
//! This scorer is evaluator-side only; do not expose answer keys to agents.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;

#[derive(Debug, Deserialize)]
struct Task {
    id: Value,
    name: String,
    correct_answer: String,
}

#[derive(Debug, Clone, Copy)]
struct RunScore {
    pre_average: f64,
    post_average: f64,
    pre_majority: f64,
    post_majority: f64,
}

#[derive(Debug, Default)]
struct TaskAccumulator {
    runs: Vec<RunScore>,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct MetricSummary {
    pre: f64,
    post: f64,
    improvement: f64,
    pre_sem: f64,
    post_sem: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct ClusterSummary {
    task_count: usize,
    pre_task_mean: f64,
    post_task_mean: f64,
    paired_improvement: f64,
    pre_task_cluster_se: f64,
    post_task_cluster_se: f64,
    improvement_cluster_se: f64,
    improvement_ci_95_low: f64,
    improvement_ci_95_high: f64,
}

#[derive(Debug, Serialize)]
struct ScoreReport {
    protocol_id: &'static str,
    model: Option<String>,
    profile: Option<String>,
    run_count: usize,
    task_count: usize,
    average_accuracy: MetricSummary,
    majority_accuracy: MetricSummary,
    task_clustered_average_accuracy: ClusterSummary,
    task_clustered_majority_accuracy: ClusterSummary,
    full_profile_pre_average_accuracy: Option<f64>,
    official_gap_to_full_profile_pre: Option<f64>,
    task_clustered_hidden_post_minus_full_pre_gap: Option<f64>,
    bootstrap_replicates: usize,
    bootstrap_seed: u64,
    claim_ceiling: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct TaskMean {
    pre_average: f64,
    post_average: f64,
    pre_majority: f64,
    post_majority: f64,
}

fn id_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) if !value.trim().is_empty() => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

fn load_tasks(path: &str) -> Result<(Vec<Task>, HashMap<String, usize>), String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("cannot read benchmark {path}: {error}"))?;
    let tasks: Vec<Task> = serde_json::from_str(&raw)
        .map_err(|error| format!("invalid benchmark task JSON: {error}"))?;
    if tasks.is_empty() {
        return Err("benchmark task file is empty".to_string());
    }

    let mut lookup = HashMap::new();
    for (index, task) in tasks.iter().enumerate() {
        let Some(id) = id_string(&task.id) else {
            return Err(format!("task at index {index} has invalid id"));
        };
        if lookup.insert(id.clone(), index).is_some() {
            return Err(format!("duplicate task id {id}"));
        }
        if task.name.trim().is_empty() || task.correct_answer.trim().is_empty() {
            return Err(format!("task {id} has empty name or correct answer"));
        }
        if lookup.insert(task.name.clone(), index).is_some() {
            return Err(format!("task name/id collision: {}", task.name));
        }
    }
    Ok((tasks, lookup))
}

fn load_run_records(path: &str) -> Result<(Value, Vec<Value>), String> {
    let raw = fs::read_to_string(path)
        .map_err(|error| format!("cannot read result file {path}: {error}"))?;
    let results: Value = serde_json::from_str(&raw)
        .map_err(|error| format!("invalid result JSON: {error}"))?;

    if let Some(runs) = results.get("runs").and_then(Value::as_array) {
        return Ok((results, runs.clone()));
    }

    let mut flattened = Vec::new();
    if let Some(conditions) = results.get("conditions").and_then(Value::as_array) {
        for condition in conditions {
            if let Some(runs) = condition.get("runs").and_then(Value::as_array) {
                let condition_metadata = serde_json::json!({
                    "condition_id": condition.get("condition_id"),
                    "model": condition.get("model"),
                    "percentage_special": condition.get("percentage_special"),
                    "discussion_mode": condition.get("discussion_mode")
                });
                for run in runs {
                    let mut enriched = run.clone();
                    if let Some(object) = enriched.as_object_mut() {
                        object.entry("_condition").or_insert(condition_metadata.clone());
                    }
                    flattened.push(enriched);
                }
            }
        }
    }
    if flattened.is_empty() {
        return Err("result file contains neither a non-empty runs array nor legacy conditions[].runs".to_string());
    }
    Ok((results, flattened))
}

fn result_model_name(results: &Value) -> Option<String> {
    if let Some(model) = metadata_string(results, "model") {
        if !model.trim().is_empty() {
            return Some(model);
        }
    }

    let mut models = std::collections::BTreeSet::<String>::new();
    if let Some(conditions) = results.get("conditions").and_then(Value::as_array) {
        for condition in conditions {
            if let Some(model) = condition.get("model").and_then(Value::as_str) {
                if !model.trim().is_empty() {
                    models.insert(model.to_string());
                }
            }
        }
    }
    if let Some(runs) = results.get("runs").and_then(Value::as_array) {
        for run in runs {
            if let Some(model) = run.get("model").and_then(Value::as_str) {
                if !model.trim().is_empty() {
                    models.insert(model.to_string());
                }
            }
        }
    }
    if models.len() == 1 {
        models.into_iter().next()
    } else if models.is_empty() {
        None
    } else {
        Some(models.into_iter().collect::<Vec<_>>().join(","))
    }
}

fn result_profile_name(results: &Value) -> Option<String> {
    if let Some(profile) = metadata_string(results, "profile") {
        if !profile.trim().is_empty() {
            return Some(profile);
        }
    }

    let mut profiles = std::collections::BTreeSet::<String>::new();
    if let Some(runs) = results.get("runs").and_then(Value::as_array) {
        for run in runs.iter().take(10) {
            if let Some(profile) = run.get("profile").and_then(Value::as_str) {
                if !profile.trim().is_empty() {
                    profiles.insert(profile.to_string());
                }
            }
        }
    }
    if let Some(conditions) = results.get("conditions").and_then(Value::as_array) {
        for condition in conditions {
            if let Some(runs) = condition.get("runs").and_then(Value::as_array) {
                for run in runs.iter().take(10) {
                    if let Some(profile) = run.get("profile").and_then(Value::as_str) {
                        if !profile.trim().is_empty() {
                            profiles.insert(profile.to_string());
                        }
                    }
                }
            }
        }
    }

    if profiles.len() == 1 {
        return profiles.into_iter().next();
    }
    if profiles.len() > 1 {
        return Some(profiles.into_iter().collect::<Vec<_>>().join(","));
    }

    if let Some(conditions) = results.get("conditions").and_then(Value::as_array) {
        if conditions.iter().any(|condition| {
            condition.get("percentage_special").is_some_and(|value| !value.is_null())
        }) {
            return Some("legacy".to_string());
        }
    }
    None
}

fn require_single_identity(value: Option<String>, field: &str, path: &str) -> Result<Option<String>, String> {
    if let Some(value) = &value {
        if value.contains(',') {
            return Err(format!("{field} is ambiguous across multiple values in {path}: {value}"));
        }
    }
    Ok(value)
}

fn load_multiple_run_records(
    paths: &[String],
    expected_profile: &str,
) -> Result<(Value, Vec<Value>), String> {
    if paths.is_empty() {
        return Err("at least one result file is required".to_string());
    }

    let mut combined_runs = Vec::new();
    let mut expected_model: Option<String> = None;
    let mut observed_profiles = std::collections::BTreeSet::<String>::new();

    for path in paths {
        let (results, runs) = load_run_records(path)?;
        let model = require_single_identity(result_model_name(&results), "model", path)?
            .ok_or_else(|| format!("result file has no model identity: {path}"))?;
        let profile = require_single_identity(result_profile_name(&results), "profile", path)?
            .ok_or_else(|| format!("result file has no profile identity: {path}"))?;

        if let Some(profile) = &Some(profile.clone()) {
            if profile != "legacy" && profile != expected_profile {
                return Err(format!(
                    "expected {expected_profile} profile input, found {profile} ({path})"
                ));
            }
            observed_profiles.insert(profile.clone());
        }

        if let Some(expected) = &expected_model {
            if expected != &model {
                return Err(format!(
                    "result files use different models: {expected} vs {model} ({path})"
                ));
            }
        } else {
            expected_model = Some(model.clone());
        }
        combined_runs.extend(runs);
    }

    if observed_profiles.len() > 1 {
        return Err(format!(
            "result files mix incompatible profile metadata: {}",
            observed_profiles.into_iter().collect::<Vec<_>>().join(",")
        ));
    }

    let normalized_metadata = serde_json::json!({
        "metadata": {
            "model": expected_model,
            "profile": expected_profile
        }
    });
    Ok((normalized_metadata, combined_runs))
}

fn vote_values(run: &Value, field: &str) -> Vec<String> {
    run.get(field)
        .and_then(Value::as_array)
        .map(|votes| {
            votes.iter()
                .map(|vote| {
                    vote.get("vote")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string()
                })
                .collect()
        })
        .unwrap_or_default()
}

fn individual_accuracy(votes: &[String], correct_answer: &str) -> f64 {
    if votes.is_empty() {
        return 0.0;
    }
    votes.iter().filter(|vote| vote.as_str() == correct_answer).count() as f64
        / votes.len() as f64
}

fn majority_accuracy(votes: &[String], correct_answer: &str) -> f64 {
    if votes.is_empty() {
        return 0.0;
    }
    let correct = votes.iter().filter(|vote| vote.as_str() == correct_answer).count();
    if correct * 2 > votes.len() { 1.0 } else { 0.0 }
}

fn sample_sem(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt() / (values.len() as f64).sqrt()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn task_means(accumulators: &BTreeMap<String, TaskAccumulator>) -> Vec<TaskMean> {
    accumulators.values().map(|accumulator| {
        let runs = &accumulator.runs;
        TaskMean {
            pre_average: mean(&runs.iter().map(|run| run.pre_average).collect::<Vec<_>>()),
            post_average: mean(&runs.iter().map(|run| run.post_average).collect::<Vec<_>>()),
            pre_majority: mean(&runs.iter().map(|run| run.pre_majority).collect::<Vec<_>>()),
            post_majority: mean(&runs.iter().map(|run| run.post_majority).collect::<Vec<_>>()),
        }
    }).collect()
}

fn task_cluster_se(pre: &[f64], post: &[f64]) -> (f64, f64, f64) {
    let differences: Vec<f64> = post.iter().zip(pre.iter()).map(|(after, before)| after - before).collect();
    (sample_sem(pre), sample_sem(post), sample_sem(&differences))
}

#[derive(Clone, Copy)]
struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next() as usize) % upper
    }
}

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let fraction = position - lower as f64;
        sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
    }
}

fn paired_cluster_bootstrap_ci(
    pre: &[f64],
    post: &[f64],
    replicates: usize,
    seed: u64,
) -> (f64, f64) {
    if pre.is_empty() || pre.len() != post.len() {
        return (0.0, 0.0);
    }
    if pre.len() == 1 {
        let difference = post[0] - pre[0];
        return (difference, difference);
    }

    let mut rng = XorShift64(seed.max(1));
    let mut estimates = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut delta = 0.0;
        for _ in 0..pre.len() {
            let index = rng.index(pre.len());
            delta += post[index] - pre[index];
        }
        estimates.push(delta / pre.len() as f64);
    }
    estimates.sort_by(f64::total_cmp);
    (quantile(&estimates, 0.025), quantile(&estimates, 0.975))
}

fn metadata_string(results: &Value, key: &str) -> Option<String> {
    results.get("metadata")?.get(key)?.as_str().map(str::to_string)
}

fn score(
    benchmark_path: &str,
    hidden_results_paths: &[String],
    full_results_path: Option<&str>,
) -> Result<ScoreReport, String> {
    const BOOTSTRAP_REPLICATES: usize = 10_000;
    const BOOTSTRAP_SEED: u64 = 20_261_118;

    let (tasks, lookup) = load_tasks(benchmark_path)?;
    let (results_json, runs) = load_multiple_run_records(hidden_results_paths, "hidden")?;
    let mut per_run = Vec::with_capacity(runs.len());
    let mut by_task = BTreeMap::<String, TaskAccumulator>::new();

    for (run_index, run) in runs.iter().enumerate() {
        let id_match = run.get("task_id").and_then(id_string)
            .and_then(|id| lookup.get(&id).copied());
        let name_match = run.get("scenario").and_then(Value::as_str)
            .and_then(|name| lookup.get(name).copied());
        let task_index = match (id_match, name_match) {
            (Some(a), Some(b)) if a != b => {
                return Err(format!("run {run_index} task_id and scenario identify different tasks"));
            }
            (Some(index), _) | (_, Some(index)) => index,
            (None, None) => {
                return Err(format!("run {run_index} cannot be matched to a benchmark task"));
            }
        };
        let task = &tasks[task_index];
        let initial_votes = vote_values(run, "initial_votes");
        let final_votes = vote_values(run, "final_votes");
        let run_score = RunScore {
            pre_average: individual_accuracy(&initial_votes, &task.correct_answer),
            post_average: individual_accuracy(&final_votes, &task.correct_answer),
            pre_majority: majority_accuracy(&initial_votes, &task.correct_answer),
            post_majority: majority_accuracy(&final_votes, &task.correct_answer),
        };
        per_run.push(run_score);
        let task_id = id_string(&task.id).expect("task ids validated on load");
        by_task.entry(task_id).or_default().runs.push(run_score);
    }

    if per_run.is_empty() {
        return Err("no scored result runs".to_string());
    }

    let pre_average: Vec<f64> = per_run.iter().map(|run| run.pre_average).collect();
    let post_average: Vec<f64> = per_run.iter().map(|run| run.post_average).collect();
    let pre_majority: Vec<f64> = per_run.iter().map(|run| run.pre_majority).collect();
    let post_majority: Vec<f64> = per_run.iter().map(|run| run.post_majority).collect();

    let average_accuracy = MetricSummary {
        pre: mean(&pre_average),
        post: mean(&post_average),
        improvement: mean(&post_average) - mean(&pre_average),
        pre_sem: sample_sem(&pre_average),
        post_sem: sample_sem(&post_average),
    };
    let majority_accuracy = MetricSummary {
        pre: mean(&pre_majority),
        post: mean(&post_majority),
        improvement: mean(&post_majority) - mean(&pre_majority),
        pre_sem: sample_sem(&pre_majority),
        post_sem: sample_sem(&post_majority),
    };

    let by_task_means = task_means(&by_task);
    let cluster_pre_average: Vec<f64> = by_task_means.iter().map(|task| task.pre_average).collect();
    let cluster_post_average: Vec<f64> = by_task_means.iter().map(|task| task.post_average).collect();
    let cluster_pre_majority: Vec<f64> = by_task_means.iter().map(|task| task.pre_majority).collect();
    let cluster_post_majority: Vec<f64> = by_task_means.iter().map(|task| task.post_majority).collect();

    let (avg_pre_se, avg_post_se, avg_delta_se) =
        task_cluster_se(&cluster_pre_average, &cluster_post_average);
    let (maj_pre_se, maj_post_se, maj_delta_se) =
        task_cluster_se(&cluster_pre_majority, &cluster_post_majority);
    let avg_ci = paired_cluster_bootstrap_ci(
        &cluster_pre_average, &cluster_post_average, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED,
    );
    let maj_ci = paired_cluster_bootstrap_ci(
        &cluster_pre_majority, &cluster_post_majority, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED ^ 0x9E37_79B9,
    );

    let task_clustered_average_accuracy = ClusterSummary {
        task_count: by_task_means.len(),
        pre_task_mean: mean(&cluster_pre_average),
        post_task_mean: mean(&cluster_post_average),
        paired_improvement: mean(&cluster_post_average) - mean(&cluster_pre_average),
        pre_task_cluster_se: avg_pre_se,
        post_task_cluster_se: avg_post_se,
        improvement_cluster_se: avg_delta_se,
        improvement_ci_95_low: avg_ci.0,
        improvement_ci_95_high: avg_ci.1,
    };
    let task_clustered_majority_accuracy = ClusterSummary {
        task_count: by_task_means.len(),
        pre_task_mean: mean(&cluster_pre_majority),
        post_task_mean: mean(&cluster_post_majority),
        paired_improvement: mean(&cluster_post_majority) - mean(&cluster_pre_majority),
        pre_task_cluster_se: maj_pre_se,
        post_task_cluster_se: maj_post_se,
        improvement_cluster_se: maj_delta_se,
        improvement_ci_95_low: maj_ci.0,
        improvement_ci_95_high: maj_ci.1,
    };

    let mut full_profile_pre_average_accuracy = None;
    let mut official_gap_to_full_profile_pre = None;
    let mut task_clustered_hidden_post_minus_full_pre_gap = None;
    if let Some(full_path) = full_results_path {
        let (full_json, full_runs) = load_run_records(full_path)?;
        let full_profile = require_single_identity(result_profile_name(&full_json), "profile", full_path)?;
        if full_profile.as_deref().is_some_and(|value| value != "full" && value != "legacy") {
            return Err(format!("expected full-profile input, found {} ({full_path})", full_profile.unwrap_or_default()));
        }
        match (
            result_model_name(&results_json),
            result_model_name(&full_json),
        ) {
            (Some(hidden_model), Some(full_model)) if hidden_model == full_model => {}
            (Some(hidden_model), Some(full_model)) => {
                return Err(format!(
                    "hidden and full-profile inputs use different models: {hidden_model} vs {full_model}"
                ));
            }
            _ => return Err("cannot verify that hidden and full-profile results use the same model".to_string()),
        }
        let mut full_by_task = BTreeMap::<String, Vec<f64>>::new();
        for (run_index, run) in full_runs.iter().enumerate() {
            let id_match = run.get("task_id").and_then(id_string)
                .and_then(|id| lookup.get(&id).copied());
            let name_match = run.get("scenario").and_then(Value::as_str)
                .and_then(|name| lookup.get(name).copied());
            let task_index = match (id_match, name_match) {
                (Some(a), Some(b)) if a != b => {
                    return Err(format!("full-profile run {run_index} task_id and scenario mismatch"));
                }
                (Some(index), _) | (_, Some(index)) => index,
                (None, None) => return Err(format!("full-profile run {run_index} cannot be matched")),
            };
            let task = &tasks[task_index];
            let votes = vote_values(run, "initial_votes");
            let acc = individual_accuracy(&votes, &task.correct_answer);
            let task_id = id_string(&task.id).expect("task ids validated on load");
            full_by_task.entry(task_id).or_default().push(acc);
        }
        let full_all: Vec<f64> = full_by_task.values().flatten().copied().collect();
        if !full_all.is_empty() {
            let full_pre_mean = mean(&full_all);
            full_profile_pre_average_accuracy = Some(full_pre_mean);
            official_gap_to_full_profile_pre = Some(average_accuracy.post - full_pre_mean);
        }

        let mut paired_gaps = Vec::new();
        for (index, task) in by_task_means.iter().enumerate() {
            let task_id = by_task.keys().nth(index).expect("index is within map");
            if let Some(full_values) = full_by_task.get(task_id) {
                paired_gaps.push(task.post_average - mean(full_values));
            }
        }
        if !paired_gaps.is_empty() {
            task_clustered_hidden_post_minus_full_pre_gap = Some(mean(&paired_gaps));
        }
    }

    Ok(ScoreReport {
        protocol_id: "IG-HIDDEN-INFO-001",
        model: metadata_string(&results_json, "model"),
        profile: result_profile_name(&results_json),
        run_count: per_run.len(),
        task_count: by_task.len(),
        average_accuracy,
        majority_accuracy,
        task_clustered_average_accuracy,
        task_clustered_majority_accuracy,
        full_profile_pre_average_accuracy,
        official_gap_to_full_profile_pre,
        task_clustered_hidden_post_minus_full_pre_gap,
        bootstrap_replicates: BOOTSTRAP_REPLICATES,
        bootstrap_seed: BOOTSTRAP_SEED,
        claim_ceiling: "Scored result-file metrics only. Interpret as model performance only when result provenance, prompts, task split, provider/model revision, budget, and run validity are independently qualified.",
    })
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(benchmark_path) = args.next() else {
        eprintln!("usage: score_hiddenbench <benchmark.json> <hidden_results_part1.json> [hidden_results_part2.json ...] [--full <full_profile_results.json>]");
        std::process::exit(2);
    };
    let remaining: Vec<String> = args.collect();
    let full_index = remaining.iter().position(|argument| argument == "--full");
    let (hidden_paths, full_path): (Vec<String>, Option<&str>) = match full_index {
        Some(index) => {
            if index == 0 || index + 2 != remaining.len() {
                eprintln!("use one or more hidden result files followed by --full <full_profile_results.json>");
                std::process::exit(2);
            }
            (remaining[..index].to_vec(), Some(remaining[index + 1].as_str()))
        }
        None => (remaining, None),
    };
    if hidden_paths.is_empty() {
        eprintln!("at least one hidden-profile result file is required");
        std::process::exit(2);
    }

    match score(&benchmark_path, &hidden_paths, full_path) {
        Ok(report) => println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("score report serialization should succeed")
        ),
        Err(error) => {
            eprintln!("HiddenBench scoring failed: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BENCHMARK: &str = include_str!("../../../external/hiddenbench/benchmark.json");

    fn synthetic_results() -> Value {
        serde_json::json!({
            "metadata": {"model":"test-model","profile":"hidden"},
            "runs": [
                {
                    "task_id": 1,
                    "scenario": "evacuation_west_city",
                    "initial_votes": [{"vote":"West City"},{"vote":"East Town"}],
                    "final_votes": [{"vote":"West City"},{"vote":"West City"}]
                },
                {
                    "task_id": 2,
                    "scenario": "evacuation_north_hill",
                    "initial_votes": [{"vote":"North Hill"},{"vote":"North Hill"}],
                    "final_votes": [{"vote":"North Hill"},{"vote":"West City"}]
                }
            ]
        })
    }

    #[test]
    fn reproduces_official_average_and_strict_majority_rules() {
        let data: Vec<Task> = serde_json::from_str(BENCHMARK).expect("benchmark parses");
        let lookup: HashMap<String, usize> = data.iter().enumerate().flat_map(|(index, task)| {
            vec![
                (id_string(&task.id).expect("valid task id"), index),
                (task.name.clone(), index)
            ]
        }).collect();
        let results = synthetic_results();
        let runs = results.get("runs").and_then(Value::as_array).expect("runs array");
        let mut scores = Vec::new();
        for run in runs {
            let key = run.get("scenario").and_then(Value::as_str).expect("scenario");
            let task = &data[*lookup.get(key).expect("task lookup")];
            let pre = vote_values(run, "initial_votes");
            let post = vote_values(run, "final_votes");
            scores.push(RunScore {
                pre_average: individual_accuracy(&pre, &task.correct_answer),
                post_average: individual_accuracy(&post, &task.correct_answer),
                pre_majority: majority_accuracy(&pre, &task.correct_answer),
                post_majority: majority_accuracy(&post, &task.correct_answer),
            });
        }
        assert!((mean(&scores.iter().map(|s| s.pre_average).collect::<Vec<_>>()) - 0.75).abs() < 1e-12);
        assert!((mean(&scores.iter().map(|s| s.post_average).collect::<Vec<_>>()) - 0.75).abs() < 1e-12);
        assert!((mean(&scores.iter().map(|s| s.pre_majority).collect::<Vec<_>>()) - 0.5).abs() < 1e-12);
        assert!((mean(&scores.iter().map(|s| s.post_majority).collect::<Vec<_>>()) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn combines_manual_and_generated_hidden_profile_splits() {
        let results = synthetic_results();
        let runs = results.get("runs").and_then(Value::as_array).expect("runs array");
        let first = serde_json::json!({
            "metadata": {"model":"test-model","profile":"hidden"},
            "runs": [runs[0].clone()]
        });
        let second = serde_json::json!({
            "metadata": {"model":"test-model","profile":"hidden"},
            "runs": [runs[1].clone()]
        });
        let first_path = std::env::temp_dir().join("hiddenbench-split-a.json");
        let second_path = std::env::temp_dir().join("hiddenbench-split-b.json");
        std::fs::write(&first_path, first.to_string()).expect("write first split");
        std::fs::write(&second_path, second.to_string()).expect("write second split");
        let paths = vec![
            first_path.to_string_lossy().to_string(),
            second_path.to_string_lossy().to_string(),
        ];
        let report = score(
            "research/ig-hidden-info-001/external/hiddenbench/benchmark.json",
            &paths,
            None,
        ).expect("score concatenated hidden splits");
        let _ = std::fs::remove_file(first_path);
        let _ = std::fs::remove_file(second_path);
        assert_eq!(report.run_count, 2);
        assert_eq!(report.task_count, 2);
        assert_eq!(report.model.as_deref(), Some("test-model"));
        assert_eq!(report.profile.as_deref(), Some("hidden"));
    }

    #[test]
    fn mixed_model_hidden_splits_are_rejected() {
        let first = serde_json::json!({"metadata":{"model":"model-a","profile":"hidden"},"runs":[]});
        let second = serde_json::json!({"metadata":{"model":"model-b","profile":"hidden"},"runs":[]});
        let first_path = std::env::temp_dir().join("hiddenbench-model-a.json");
        let second_path = std::env::temp_dir().join("hiddenbench-model-b.json");
        std::fs::write(&first_path, first.to_string()).expect("write first split");
        std::fs::write(&second_path, second.to_string()).expect("write second split");
        let paths = vec![
            first_path.to_string_lossy().to_string(),
            second_path.to_string_lossy().to_string(),
        ];
        let error = load_multiple_run_records(&paths, "hidden").expect_err("different models must not combine");
        let _ = std::fs::remove_file(first_path);
        let _ = std::fs::remove_file(second_path);
        assert!(error.contains("different models"));
    }

    #[test]
    fn strict_majority_does_not_count_a_tie() {
        let votes = vec!["West City".to_string(), "East Town".to_string()];
        assert_eq!(majority_accuracy(&votes, "West City"), 0.0);
    }

    #[test]
    fn deterministic_task_cluster_bootstrap_is_paired() {
        let pre = [0.0, 0.0, 1.0, 1.0];
        let post = [1.0, 1.0, 0.0, 0.0];
        let first = paired_cluster_bootstrap_ci(&pre, &post, 2_000, 42);
        let second = paired_cluster_bootstrap_ci(&pre, &post, 2_000, 42);
        assert_eq!(first, second);
        assert!(first.0 <= 0.0 && first.1 >= 0.0);
    }

    #[test]
    fn one_task_cluster_returns_exact_paired_delta() {
        let ci = paired_cluster_bootstrap_ci(&[0.25], &[0.75], 1_000, 7);
        assert_eq!(ci, (0.5, 0.5));
    }

    #[test]
    fn result_schema_supports_legacy_conditions() {
        let legacy = serde_json::json!({
            "conditions": [
                {"condition_id":"c1","runs":[{"task_id":1}]},
                {"condition_id":"c2","runs":[{"task_id":2}]}
            ]
        });
        let raw = legacy.to_string();
        let path = std::env::temp_dir().join("hiddenbench-legacy-results-test.json");
        std::fs::write(&path, raw).expect("write temp result");
        let (_value, runs) = load_run_records(path.to_str().expect("utf8 path")).expect("legacy schema parses");
        let _ = std::fs::remove_file(path);
        assert_eq!(runs.len(), 2);
    }
}
