//! Audits preregistered paired GCS-on/off repository-repair experiments.
//!
//! This tool is an analysis gate, not a benchmark runner. It binds outcomes to the
//! independently audited task-score reports, preserves assigned tasks with missing
//! trials as unresolved, and avoids treating incomplete pairs as causal wins/losses.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};

const PROTOCOL_ID: &str = "symthaea-rust-repair-ablation-v1";
const SCORE_PROTOCOL_ID: &str = "symthaea-rust-repair";
const FORMAT_VERSION: u32 = 1;
const MIN_BOOTSTRAP_RESAMPLES: usize = 1_000;
const MAX_BOOTSTRAP_RESAMPLES: usize = 1_000_000;
const MIN_REPOSITORIES_FOR_CI: usize = 10;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum Arm {
    Control,
    Gcs,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Outcome {
    Pass,
    Fail,
    Unresolved,
}

impl Outcome {
    fn is_resolved(self) -> bool {
        self != Self::Unresolved
    }

    fn passed(self) -> bool {
        self == Self::Pass
    }

    fn value_if_resolved(self) -> Option<f64> {
        match self {
            Self::Pass => Some(1.0),
            Self::Fail => Some(0.0),
            Self::Unresolved => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Assignment {
    instance_id: String,
    repository_id: String,
    first_arm: Arm,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Preregistration {
    format_version: u32,
    protocol_id: String,
    benchmark_manifest_sha256: String,
    model_id: String,
    agent_harness_sha256: String,
    prompt_sha256: String,
    scaffold_sha256: String,
    budget_id: String,
    container_image_id: String,
    network_policy: String,
    randomization_seed: String,
    bootstrap_seed: u64,
    bootstrap_resamples: usize,
    assignment: Vec<Assignment>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TrialRecord {
    run_id: String,
    instance_id: String,
    arm: Arm,
    outcome: Outcome,
    gcs_enabled: bool,
    cost_usd: f64,
    latency_ms: u64,
    started_at_ms: u64,
    model_id: String,
    agent_harness_sha256: String,
    prompt_sha256: String,
    scaffold_sha256: String,
    budget_id: String,
    container_image_id: String,
    network_policy: String,
    score_report_path: String,
    score_report_sha256: String,
}

#[derive(Clone, Debug)]
struct ObservedTrial {
    record: TrialRecord,
    jsonl_row_sha256: String,
}

#[derive(Clone, Debug)]
struct AuditPair {
    assignment: Assignment,
    gcs: Option<ObservedTrial>,
    control: Option<ObservedTrial>,
}

impl AuditPair {
    fn outcome(&self, arm: Arm) -> Outcome {
        let trial = match arm {
            Arm::Control => self.control.as_ref(),
            Arm::Gcs => self.gcs.as_ref(),
        };
        trial.map_or(Outcome::Unresolved, |observed| observed.record.outcome)
    }

    fn trial(&self, arm: Arm) -> Option<&ObservedTrial> {
        match arm {
            Arm::Control => self.control.as_ref(),
            Arm::Gcs => self.gcs.as_ref(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct ArmMetrics {
    task_count: usize,
    observed_record_count: usize,
    verified_pass_count: usize,
    failure_count: usize,
    unresolved_count: usize,
    verified_pass_rate: f64,
    resolved_pass_rate: Option<f64>,
    unresolved_rate: f64,
    observed_cost_usd: f64,
    cost_data_complete: bool,
    cost_per_verified_fix_usd: Option<f64>,
    observed_latency_count: usize,
    median_latency_ms: Option<f64>,
    p95_latency_ms: Option<f64>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct ConfidenceInterval {
    lower_percentage_points: f64,
    upper_percentage_points: f64,
}

#[derive(Clone, Debug, Serialize)]
struct PairedMetrics {
    assignment_count: usize,
    fully_resolved_pair_count: usize,
    unresolved_pair_count: usize,
    gcs_wins: usize,
    control_wins: usize,
    ties: usize,
    resolved_effect_percentage_points: Option<f64>,
    resolved_repository_count: usize,
    cluster_bootstrap_ci95: Option<ConfidenceInterval>,
    conservative_verified_pass_delta_percentage_points: f64,
    unresolved_sensitivity_lower_percentage_points: f64,
    unresolved_sensitivity_upper_percentage_points: f64,
}

#[derive(Clone, Debug, Serialize)]
struct DecisionGates {
    overall_unresolved_rate_below_2_percent: bool,
    arm_unresolved_rate_difference_below_1_percentage_point: bool,
    at_least_10_resolved_repositories: bool,
    full_assignment_sensitivity_lower_bound_above_zero: bool,
    gcs_superiority_gate_met: bool,
    cost_efficiency_noninferiority_gate_met: bool,
    gcs_superiority_target_percentage_points: f64,
    cost_reduction_target_fraction: f64,
    correctness_noninferiority_margin_percentage_points: f64,
}

#[derive(Clone, Debug, Serialize)]
struct TaskAuditRecord {
    instance_id: String,
    repository_id: String,
    preregistered_first_arm: Arm,
    gcs_outcome: Outcome,
    control_outcome: Outcome,
    gcs_jsonl_row_sha256: Option<String>,
    control_jsonl_row_sha256: Option<String>,
    gcs_score_report_sha256: Option<String>,
    control_score_report_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct AuditReport {
    format_version: u32,
    protocol_id: String,
    preregistration_sha256: String,
    benchmark_manifest_sha256: String,
    trial_results_sha256: String,
    assignment_count: usize,
    input_trial_record_count: usize,
    missing_trial_record_count: usize,
    repository_count: usize,
    resolved_repository_count: usize,
    randomization_seed: String,
    bootstrap_seed: u64,
    bootstrap_resamples: usize,
    control: ArmMetrics,
    gcs: ArmMetrics,
    paired: PairedMetrics,
    decision_gates: DecisionGates,
    tasks: Vec<TaskAuditRecord>,
    interpretation: String,
}

#[derive(Clone, Debug)]
struct Cli {
    preregistration: PathBuf,
    expected_preregistration_sha256: String,
    trial_results: PathBuf,
    score_reports_dir: PathBuf,
    report_out: PathBuf,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut text = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(text, "{byte:02x}");
    }
    text
}

fn is_hex(value: &str, expected_len: usize) -> bool {
    value.len() == expected_len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_container_image_id(value: &str) -> bool {
    value
        .strip_prefix("sha256:")
        .is_some_and(|digest| is_hex(digest, 64))
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn randomized_first_arm(seed: &str, instance_id: &str) -> Arm {
    let mut material = Vec::with_capacity(seed.len() + instance_id.len() + 1);
    material.extend_from_slice(seed.as_bytes());
    material.push(0);
    material.extend_from_slice(instance_id.as_bytes());
    let digest = Sha256::digest(material);
    if digest[0] & 1 == 0 {
        Arm::Control
    } else {
        Arm::Gcs
    }
}

fn validate_preregistration(prereg: &Preregistration) -> Result<(), String> {
    if prereg.format_version != FORMAT_VERSION || prereg.protocol_id != PROTOCOL_ID {
        return Err("unsupported preregistration format_version or protocol_id".to_string());
    }
    for (field, value) in [
        (
            "benchmark_manifest_sha256",
            prereg.benchmark_manifest_sha256.as_str(),
        ),
        ("agent_harness_sha256", prereg.agent_harness_sha256.as_str()),
        ("prompt_sha256", prereg.prompt_sha256.as_str()),
        ("scaffold_sha256", prereg.scaffold_sha256.as_str()),
    ] {
        if !is_hex(value, 64) {
            return Err(format!("{field} must be a 64-character SHA-256 hex digest"));
        }
    }
    for (field, value) in [
        ("model_id", prereg.model_id.as_str()),
        ("budget_id", prereg.budget_id.as_str()),
        ("randomization_seed", prereg.randomization_seed.as_str()),
    ] {
        if !non_empty(value) {
            return Err(format!("{field} must not be empty"));
        }
    }
    if !valid_container_image_id(&prereg.container_image_id) {
        return Err("container_image_id must be sha256:<64-hex>".to_string());
    }
    if prereg.network_policy != "none" {
        return Err("strict paired comparison requires network_policy=\"none\"".to_string());
    }
    if !(MIN_BOOTSTRAP_RESAMPLES..=MAX_BOOTSTRAP_RESAMPLES).contains(&prereg.bootstrap_resamples) {
        return Err(format!(
            "bootstrap_resamples must be in {MIN_BOOTSTRAP_RESAMPLES}..={MAX_BOOTSTRAP_RESAMPLES}"
        ));
    }
    if prereg.assignment.is_empty() {
        return Err("preregistration assignment must not be empty".to_string());
    }
    let mut task_ids = BTreeSet::new();
    for item in &prereg.assignment {
        if !non_empty(&item.instance_id) || !non_empty(&item.repository_id) {
            return Err("assignment instance_id and repository_id must not be empty".to_string());
        }
        if !task_ids.insert(item.instance_id.clone()) {
            return Err(format!(
                "duplicate preregistered instance_id {:?}",
                item.instance_id
            ));
        }
        if item.first_arm != randomized_first_arm(&prereg.randomization_seed, &item.instance_id) {
            return Err(format!(
                "first_arm for task {:?} does not match the preregistered hash-randomization seed",
                item.instance_id
            ));
        }
    }
    Ok(())
}

fn validate_trial_static(trial: &TrialRecord, prereg: &Preregistration) -> Result<(), String> {
    if !non_empty(&trial.run_id) || !non_empty(&trial.instance_id) {
        return Err("run_id and instance_id must not be empty".to_string());
    }
    if !trial.cost_usd.is_finite() || trial.cost_usd < 0.0 {
        return Err(format!("run {:?} has invalid cost_usd", trial.run_id));
    }
    if trial.started_at_ms == 0 {
        return Err(format!("run {:?} has started_at_ms=0", trial.run_id));
    }
    if !valid_container_image_id(&trial.container_image_id) {
        return Err(format!(
            "run {:?} container_image_id must be sha256:<64-hex>",
            trial.run_id
        ));
    }
    if !is_hex(&trial.agent_harness_sha256, 64)
        || !is_hex(&trial.prompt_sha256, 64)
        || !is_hex(&trial.scaffold_sha256, 64)
        || !is_hex(&trial.score_report_sha256, 64)
    {
        return Err(format!(
            "run {:?} contains a malformed SHA-256 digest",
            trial.run_id
        ));
    }
    let expected_gcs_enabled = trial.arm == Arm::Gcs;
    if trial.gcs_enabled != expected_gcs_enabled {
        return Err(format!(
            "run {:?} arm and gcs_enabled disagree",
            trial.run_id
        ));
    }
    for (field, observed, expected) in [
        (
            "model_id",
            trial.model_id.as_str(),
            prereg.model_id.as_str(),
        ),
        (
            "agent_harness_sha256",
            trial.agent_harness_sha256.as_str(),
            prereg.agent_harness_sha256.as_str(),
        ),
        (
            "prompt_sha256",
            trial.prompt_sha256.as_str(),
            prereg.prompt_sha256.as_str(),
        ),
        (
            "scaffold_sha256",
            trial.scaffold_sha256.as_str(),
            prereg.scaffold_sha256.as_str(),
        ),
        (
            "budget_id",
            trial.budget_id.as_str(),
            prereg.budget_id.as_str(),
        ),
        (
            "container_image_id",
            trial.container_image_id.as_str(),
            prereg.container_image_id.as_str(),
        ),
        (
            "network_policy",
            trial.network_policy.as_str(),
            prereg.network_policy.as_str(),
        ),
    ] {
        if observed != expected {
            return Err(format!(
                "run {:?} {field} differs from preregistration",
                trial.run_id
            ));
        }
    }
    validate_relative_report_path(&trial.score_report_path)?;
    Ok(())
}

fn validate_relative_report_path(value: &str) -> Result<(), String> {
    let path = Path::new(value);
    if value.contains('\\')
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
        || !path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "score_report_path must be a safe relative path, got {value:?}"
        ));
    }
    Ok(())
}

fn load_score_report(
    reports_dir: &Path,
    relative_path: &str,
    expected_sha256: &str,
    expected_task_manifest_sha256: &str,
) -> Result<BTreeMap<String, Outcome>, String> {
    validate_relative_report_path(relative_path)?;
    let path = reports_dir.join(relative_path);
    let bytes = fs::read(&path)
        .map_err(|error| format!("cannot read score report {}: {error}", path.display()))?;
    let observed_sha256 = sha256_bytes(&bytes);
    if observed_sha256 != expected_sha256 {
        return Err(format!(
            "score report digest mismatch for {}: expected {expected_sha256}, observed {observed_sha256}",
            path.display()
        ));
    }
    let report: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("score report {} is malformed JSON: {error}", path.display()))?;
    if report.get("format_version").and_then(Value::as_u64) != Some(1)
        || report.get("protocol_id").and_then(Value::as_str) != Some(SCORE_PROTOCOL_ID)
        || report.get("task_manifest_sha256").and_then(Value::as_str)
            != Some(expected_task_manifest_sha256)
    {
        return Err(format!(
            "score report {} has wrong format, protocol, or task-manifest pin",
            path.display()
        ));
    }
    let task_rows = report
        .get("tasks")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("score report {} has no tasks array", path.display()))?;
    if report.get("task_count").and_then(Value::as_u64) != Some(task_rows.len() as u64) {
        return Err(format!(
            "score report {} task_count differs from its task rows",
            path.display()
        ));
    }
    let mut outcomes = BTreeMap::new();
    for row in task_rows {
        let instance_id = row
            .get("instance_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                format!(
                    "score report {} has task without instance_id",
                    path.display()
                )
            })?;
        let outcome_text = row
            .get("outcome")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("score report {} has task without outcome", path.display()))?;
        let outcome = match outcome_text {
            "pass" => Outcome::Pass,
            "fail" => Outcome::Fail,
            "unresolved" => Outcome::Unresolved,
            _ => {
                return Err(format!(
                    "score report {} has invalid outcome {outcome_text:?}",
                    path.display()
                ));
            }
        };
        if outcomes.insert(instance_id.to_string(), outcome).is_some() {
            return Err(format!(
                "score report {} repeats instance_id {instance_id:?}",
                path.display()
            ));
        }
    }
    Ok(outcomes)
}

fn parse_trial_results(
    bytes: &[u8],
    prereg: &Preregistration,
    reports_dir: &Path,
) -> Result<(BTreeMap<String, BTreeMap<Arm, ObservedTrial>>, usize), String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("trial-results file is not UTF-8: {error}"))?;
    let assignments = prereg
        .assignment
        .iter()
        .map(|item| (item.instance_id.as_str(), item))
        .collect::<BTreeMap<_, _>>();
    let mut grouped = BTreeMap::<String, BTreeMap<Arm, ObservedTrial>>::new();
    let mut run_ids = BTreeSet::new();
    let mut score_cache = BTreeMap::<String, (String, BTreeMap<String, Outcome>)>::new();
    let mut record_count = 0usize;

    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            return Err(format!(
                "trial-results JSONL has blank line at {}",
                line_index + 1
            ));
        }
        let trial: TrialRecord = serde_json::from_str(line).map_err(|error| {
            format!(
                "invalid trial-results JSONL line {}: {error}",
                line_index + 1
            )
        })?;
        validate_trial_static(&trial, prereg)?;
        if !run_ids.insert(trial.run_id.clone()) {
            return Err(format!("duplicate run_id {:?}", trial.run_id));
        }
        let assignment = assignments.get(trial.instance_id.as_str()).ok_or_else(|| {
            format!(
                "trial {:?} has instance_id outside preregistered assignment",
                trial.run_id
            )
        })?;
        if let Some((cached_sha, _)) = score_cache.get(&trial.score_report_path) {
            if cached_sha != &trial.score_report_sha256 {
                return Err(format!(
                    "score report path {:?} was paired with multiple SHA-256 digests",
                    trial.score_report_path
                ));
            }
        } else {
            let outcomes = load_score_report(
                reports_dir,
                &trial.score_report_path,
                &trial.score_report_sha256,
                &prereg.benchmark_manifest_sha256,
            )?;
            score_cache.insert(
                trial.score_report_path.clone(),
                (trial.score_report_sha256.clone(), outcomes),
            );
        }
        let score_map = &score_cache
            .get(&trial.score_report_path)
            .expect("score report inserted or validated above")
            .1;
        let scored_outcome = score_map.get(&trial.instance_id).ok_or_else(|| {
            format!(
                "score report {:?} has no outcome for task {:?}",
                trial.score_report_path, trial.instance_id
            )
        })?;
        if *scored_outcome != trial.outcome {
            return Err(format!(
                "trial {:?} outcome disagrees with its independently scored task receipt",
                trial.run_id
            ));
        }
        let observed = ObservedTrial {
            record: trial.clone(),
            jsonl_row_sha256: sha256_bytes(line.as_bytes()),
        };
        let arm_rows = grouped.entry(trial.instance_id.clone()).or_default();
        if arm_rows.insert(trial.arm, observed).is_some() {
            return Err(format!(
                "duplicate {:?} trial for task {:?}",
                trial.arm, trial.instance_id
            ));
        }
        if assignment.instance_id != trial.instance_id {
            return Err("internal assignment lookup mismatch".to_string());
        }
        record_count += 1;
    }
    Ok((grouped, record_count))
}

fn join_assignments(
    prereg: &Preregistration,
    grouped: BTreeMap<String, BTreeMap<Arm, ObservedTrial>>,
) -> Result<Vec<AuditPair>, String> {
    let mut pairs = Vec::with_capacity(prereg.assignment.len());
    for assignment in &prereg.assignment {
        let arm_rows = grouped.get(&assignment.instance_id);
        let gcs = arm_rows.and_then(|rows| rows.get(&Arm::Gcs)).cloned();
        let control = arm_rows.and_then(|rows| rows.get(&Arm::Control)).cloned();

        if let (Some(gcs_trial), Some(control_trial)) = (&gcs, &control) {
            if gcs_trial.record.started_at_ms == control_trial.record.started_at_ms {
                return Err(format!(
                    "task {:?} GCS/control start timestamps must differ",
                    assignment.instance_id
                ));
            }
            let earliest_arm =
                if gcs_trial.record.started_at_ms < control_trial.record.started_at_ms {
                    Arm::Gcs
                } else {
                    Arm::Control
                };
            if earliest_arm != assignment.first_arm {
                return Err(format!(
                    "task {:?} observed execution order differs from preregistered first_arm",
                    assignment.instance_id
                ));
            }
        }
        pairs.push(AuditPair {
            assignment: assignment.clone(),
            gcs,
            control,
        });
    }
    Ok(pairs)
}

fn percentile(sorted_values: &[f64], quantile: f64) -> Option<f64> {
    if sorted_values.is_empty() {
        return None;
    }
    let index = ((sorted_values.len() - 1) as f64 * quantile).floor() as usize;
    sorted_values.get(index).copied()
}

fn latency_quantiles(values: &mut [u64]) -> (Option<f64>, Option<f64>) {
    if values.is_empty() {
        return (None, None);
    }
    values.sort_unstable();
    let median = if values.len() % 2 == 0 {
        let upper = values.len() / 2;
        Some((values[upper - 1] as f64 + values[upper] as f64) / 2.0)
    } else {
        Some(values[values.len() / 2] as f64)
    };
    let p95_index = ((values.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    (median, Some(values[p95_index] as f64))
}

fn arm_metrics(pairs: &[AuditPair], arm: Arm) -> ArmMetrics {
    let mut observed_record_count = 0usize;
    let mut verified_pass_count = 0usize;
    let mut failure_count = 0usize;
    let mut unresolved_count = 0usize;
    let mut observed_cost_usd = 0.0;
    let mut latencies = Vec::new();
    for pair in pairs {
        if let Some(trial) = pair.trial(arm) {
            observed_record_count += 1;
            match trial.record.outcome {
                Outcome::Pass => verified_pass_count += 1,
                Outcome::Fail => failure_count += 1,
                Outcome::Unresolved => unresolved_count += 1,
            }
            observed_cost_usd += trial.record.cost_usd;
            latencies.push(trial.record.latency_ms);
        } else {
            unresolved_count += 1;
        }
    }
    let task_count = pairs.len();
    let resolved_count = verified_pass_count + failure_count;
    let unresolved_rate = unresolved_count as f64 / task_count as f64;
    let (median_latency_ms, p95_latency_ms) = latency_quantiles(&mut latencies);
    let cost_data_complete = observed_record_count == task_count;
    ArmMetrics {
        task_count,
        observed_record_count,
        verified_pass_count,
        failure_count,
        unresolved_count,
        verified_pass_rate: verified_pass_count as f64 / task_count as f64,
        resolved_pass_rate: (resolved_count > 0)
            .then(|| verified_pass_count as f64 / resolved_count as f64),
        unresolved_rate,
        observed_cost_usd,
        cost_data_complete,
        cost_per_verified_fix_usd: (cost_data_complete && verified_pass_count > 0)
            .then(|| observed_cost_usd / verified_pass_count as f64),
        observed_latency_count: latencies.len(),
        median_latency_ms,
        p95_latency_ms,
    }
}

fn resolved_effect(pairs: &[AuditPair]) -> (usize, usize, usize, usize, Option<f64>) {
    let mut resolved_count = 0usize;
    let mut gcs_wins = 0usize;
    let mut control_wins = 0usize;
    let mut gcs_passes = 0usize;
    let mut control_passes = 0usize;
    for pair in pairs {
        let gcs = pair.outcome(Arm::Gcs);
        let control = pair.outcome(Arm::Control);
        if !gcs.is_resolved() || !control.is_resolved() {
            continue;
        }
        resolved_count += 1;
        let gcs_pass = gcs.passed();
        let control_pass = control.passed();
        gcs_passes += usize::from(gcs_pass);
        control_passes += usize::from(control_pass);
        match (gcs_pass, control_pass) {
            (true, false) => gcs_wins += 1,
            (false, true) => control_wins += 1,
            _ => {}
        }
    }
    let ties = resolved_count.saturating_sub(gcs_wins + control_wins);
    let effect = (resolved_count > 0).then_some(
        (gcs_passes as f64 / resolved_count as f64 - control_passes as f64 / resolved_count as f64)
            * 100.0,
    );
    (resolved_count, gcs_wins, control_wins, ties, effect)
}

fn unresolved_sensitivity_bounds(pairs: &[AuditPair]) -> (f64, f64) {
    let mut lower_total = 0.0;
    let mut upper_total = 0.0;
    for pair in pairs {
        let gcs = pair.outcome(Arm::Gcs);
        let control = pair.outcome(Arm::Control);
        let gcs_low = if gcs == Outcome::Pass { 1.0 } else { 0.0 };
        let gcs_high = if gcs == Outcome::Unresolved {
            1.0
        } else {
            gcs_low
        };
        let control_low = if control == Outcome::Pass { 1.0 } else { 0.0 };
        let control_high = if control == Outcome::Unresolved {
            1.0
        } else {
            control_low
        };
        lower_total += gcs_low - control_high;
        upper_total += gcs_high - control_low;
    }
    let denominator = pairs.len() as f64;
    (
        lower_total / denominator * 100.0,
        upper_total / denominator * 100.0,
    )
}

#[derive(Clone, Copy)]
struct CompleteBinaryPair {
    gcs_pass: bool,
    control_pass: bool,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^ (value >> 31)
}

fn cluster_bootstrap_ci(
    pairs: &[AuditPair],
    seed: u64,
    resamples: usize,
) -> (usize, Option<ConfidenceInterval>) {
    let mut clusters = BTreeMap::<String, Vec<CompleteBinaryPair>>::new();
    for pair in pairs {
        let gcs = pair.outcome(Arm::Gcs);
        let control = pair.outcome(Arm::Control);
        if let (Some(gcs_pass), Some(control_pass)) =
            (gcs.value_if_resolved(), control.value_if_resolved())
        {
            clusters
                .entry(pair.assignment.repository_id.clone())
                .or_default()
                .push(CompleteBinaryPair {
                    gcs_pass: gcs_pass == 1.0,
                    control_pass: control_pass == 1.0,
                });
        }
    }
    let cluster_count = clusters.len();
    if cluster_count < MIN_REPOSITORIES_FOR_CI {
        return (cluster_count, None);
    }
    let cluster_values = clusters.into_values().collect::<Vec<_>>();
    let mut rng_state = seed;
    let mut estimates = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut gcs_successes = 0usize;
        let mut control_successes = 0usize;
        let mut pair_count = 0usize;
        for _ in 0..cluster_count {
            let index = (splitmix64(&mut rng_state) % cluster_count as u64) as usize;
            for pair in &cluster_values[index] {
                gcs_successes += usize::from(pair.gcs_pass);
                control_successes += usize::from(pair.control_pass);
                pair_count += 1;
            }
        }
        if pair_count > 0 {
            estimates.push(
                (gcs_successes as f64 / pair_count as f64
                    - control_successes as f64 / pair_count as f64)
                    * 100.0,
            );
        }
    }
    estimates.sort_by(f64::total_cmp);
    let interval = match (percentile(&estimates, 0.025), percentile(&estimates, 0.975)) {
        (Some(lower_percentage_points), Some(upper_percentage_points)) => {
            Some(ConfidenceInterval {
                lower_percentage_points,
                upper_percentage_points,
            })
        }
        _ => None,
    };
    (cluster_count, interval)
}

fn make_report(
    prereg: &Preregistration,
    preregistration_sha256: String,
    trial_results_sha256: String,
    input_trial_record_count: usize,
    pairs: Vec<AuditPair>,
) -> AuditReport {
    let control = arm_metrics(&pairs, Arm::Control);
    let gcs = arm_metrics(&pairs, Arm::Gcs);
    let (
        fully_resolved_pair_count,
        gcs_wins,
        control_wins,
        ties,
        resolved_effect_percentage_points,
    ) = resolved_effect(&pairs);
    let (resolved_repository_count, cluster_bootstrap_ci95) =
        cluster_bootstrap_ci(&pairs, prereg.bootstrap_seed, prereg.bootstrap_resamples);
    let denominator = pairs.len() as f64;
    let conservative_delta = (gcs.verified_pass_count as f64 / denominator
        - control.verified_pass_count as f64 / denominator)
        * 100.0;
    let (
        unresolved_sensitivity_lower_percentage_points,
        unresolved_sensitivity_upper_percentage_points,
    ) = unresolved_sensitivity_bounds(&pairs);
    let unresolved_pair_count = pairs.len().saturating_sub(fully_resolved_pair_count);
    let overall_unresolved_rate =
        (gcs.unresolved_count + control.unresolved_count) as f64 / (2.0 * pairs.len() as f64);
    let arm_unresolved_rate_difference = (gcs.unresolved_rate - control.unresolved_rate).abs();
    let overall_unresolved_rate_below_2_percent = overall_unresolved_rate < 0.02;
    let arm_unresolved_rate_difference_below_1_percentage_point =
        arm_unresolved_rate_difference < 0.01;
    let at_least_10_resolved_repositories = resolved_repository_count >= MIN_REPOSITORIES_FOR_CI;
    let superiority_effect_satisfied =
        resolved_effect_percentage_points.is_some_and(|effect| effect >= 5.0);
    let superiority_ci_satisfied =
        cluster_bootstrap_ci95.is_some_and(|interval| interval.lower_percentage_points > 0.0);
    let full_assignment_sensitivity_lower_bound_above_zero =
        unresolved_sensitivity_lower_percentage_points > 0.0;
    let unresolved_gates_satisfied = overall_unresolved_rate_below_2_percent
        && arm_unresolved_rate_difference_below_1_percentage_point;
    let gcs_superiority_gate_met = superiority_effect_satisfied
        && superiority_ci_satisfied
        && full_assignment_sensitivity_lower_bound_above_zero
        && at_least_10_resolved_repositories
        && unresolved_gates_satisfied;
    let cost_reduction_satisfied = match (
        gcs.cost_per_verified_fix_usd,
        control.cost_per_verified_fix_usd,
    ) {
        (Some(gcs_cost), Some(control_cost)) if control_cost > 0.0 => {
            gcs_cost <= control_cost * 0.8
        }
        _ => false,
    };
    let noninferiority_satisfied =
        cluster_bootstrap_ci95.is_some_and(|interval| interval.lower_percentage_points >= -1.5)
            && unresolved_sensitivity_lower_percentage_points >= -1.5;
    let cost_efficiency_noninferiority_gate_met = cost_reduction_satisfied
        && noninferiority_satisfied
        && at_least_10_resolved_repositories
        && unresolved_gates_satisfied;
    let repository_count = pairs
        .iter()
        .map(|pair| pair.assignment.repository_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let missing_trial_record_count = pairs
        .iter()
        .map(|pair| usize::from(pair.gcs.is_none()) + usize::from(pair.control.is_none()))
        .sum();
    let mut tasks = pairs
        .iter()
        .map(|pair| TaskAuditRecord {
            instance_id: pair.assignment.instance_id.clone(),
            repository_id: pair.assignment.repository_id.clone(),
            preregistered_first_arm: pair.assignment.first_arm,
            gcs_outcome: pair.outcome(Arm::Gcs),
            control_outcome: pair.outcome(Arm::Control),
            gcs_jsonl_row_sha256: pair
                .gcs
                .as_ref()
                .map(|trial| trial.jsonl_row_sha256.clone()),
            control_jsonl_row_sha256: pair
                .control
                .as_ref()
                .map(|trial| trial.jsonl_row_sha256.clone()),
            gcs_score_report_sha256: pair
                .gcs
                .as_ref()
                .map(|trial| trial.record.score_report_sha256.clone()),
            control_score_report_sha256: pair
                .control
                .as_ref()
                .map(|trial| trial.record.score_report_sha256.clone()),
        })
        .collect::<Vec<_>>();
    tasks.sort_by(|left, right| left.instance_id.cmp(&right.instance_id));
    let all_observed = missing_trial_record_count == 0;
    let interpretation = if !all_observed {
        "INCOMPLETE: assigned task/arm records are missing. They remain unresolved in the full denominator; do not claim a benchmark or superiority result.".to_string()
    } else if gcs_superiority_gate_met {
        "The preregistered internal GCS superiority gates are met for this dataset. This is not, by itself, a state-of-the-art or generalization claim; require an independent replication and the same-protocol external comparison.".to_string()
    } else if cost_efficiency_noninferiority_gate_met {
        "The preregistered internal cost-efficiency/non-inferiority gates are met for this dataset. This is not, by itself, a state-of-the-art or generalization claim; require an independent replication.".to_string()
    } else {
        "Audit completed, but neither preregistered mechanism gate is met. Report the point estimates, uncertainty, unresolved sensitivity bounds, and cost/latency diagnostics; do not reinterpret an inconclusive result as a win.".to_string()
    };
    AuditReport {
        format_version: FORMAT_VERSION,
        protocol_id: PROTOCOL_ID.to_string(),
        preregistration_sha256,
        benchmark_manifest_sha256: prereg.benchmark_manifest_sha256.clone(),
        trial_results_sha256,
        assignment_count: pairs.len(),
        input_trial_record_count,
        missing_trial_record_count,
        repository_count,
        resolved_repository_count,
        randomization_seed: prereg.randomization_seed.clone(),
        bootstrap_seed: prereg.bootstrap_seed,
        bootstrap_resamples: prereg.bootstrap_resamples,
        control,
        gcs,
        paired: PairedMetrics {
            assignment_count: pairs.len(),
            fully_resolved_pair_count,
            unresolved_pair_count,
            gcs_wins,
            control_wins,
            ties,
            resolved_effect_percentage_points,
            resolved_repository_count,
            cluster_bootstrap_ci95,
            conservative_verified_pass_delta_percentage_points: conservative_delta,
            unresolved_sensitivity_lower_percentage_points,
            unresolved_sensitivity_upper_percentage_points,
        },
        decision_gates: DecisionGates {
            overall_unresolved_rate_below_2_percent,
            arm_unresolved_rate_difference_below_1_percentage_point,
            at_least_10_resolved_repositories,
            full_assignment_sensitivity_lower_bound_above_zero,
            gcs_superiority_gate_met,
            cost_efficiency_noninferiority_gate_met,
            gcs_superiority_target_percentage_points: 5.0,
            cost_reduction_target_fraction: 0.2,
            correctness_noninferiority_margin_percentage_points: 1.5,
        },
        tasks,
        interpretation,
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot serialize audit report: {error}"))?;
    bytes.push(b'\n');
    fs::write(path, bytes).map_err(|error| format!("cannot write {}: {error}", path.display()))
}

fn parse_cli(args: &[String]) -> Result<Option<Cli>, String> {
    if args
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!(concat!(
            "Usage: rust-repair-ablation-audit ",
            "--preregistration PATH --expected-preregistration-sha256 HEX ",
            "--trial-results JSONL --score-reports-dir PATH --report-out PATH"
        ));
        return Ok(None);
    }
    let mut values = BTreeMap::<String, String>::new();
    let mut index = 0usize;
    while index < args.len() {
        let key = &args[index];
        if !key.starts_with("--") {
            return Err(format!("unexpected positional argument {key:?}"));
        }
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        if value.starts_with("--") {
            return Err(format!("missing value for {key}"));
        }
        if values.insert(key.clone(), value.clone()).is_some() {
            return Err(format!("duplicate argument {key}"));
        }
        index += 2;
    }
    let required = [
        "--preregistration",
        "--expected-preregistration-sha256",
        "--trial-results",
        "--score-reports-dir",
        "--report-out",
    ];
    for key in required {
        if !values.contains_key(key) {
            return Err(format!("missing required argument {key}"));
        }
    }
    if values.keys().any(|key| !required.contains(&key.as_str())) {
        return Err("unknown command-line argument".to_string());
    }
    let expected_preregistration_sha256 = values
        .get("--expected-preregistration-sha256")
        .expect("required arg checked")
        .to_ascii_lowercase();
    if !is_hex(&expected_preregistration_sha256, 64) {
        return Err("--expected-preregistration-sha256 must be 64 hex characters".to_string());
    }
    Ok(Some(Cli {
        preregistration: PathBuf::from(values.get("--preregistration").expect("required arg")),
        expected_preregistration_sha256,
        trial_results: PathBuf::from(values.get("--trial-results").expect("required arg")),
        score_reports_dir: PathBuf::from(values.get("--score-reports-dir").expect("required arg")),
        report_out: PathBuf::from(values.get("--report-out").expect("required arg")),
    }))
}

fn run(cli: Cli) -> Result<(), String> {
    let preregistration_bytes = fs::read(&cli.preregistration)
        .map_err(|error| format!("cannot read preregistration: {error}"))?;
    let preregistration_sha256 = sha256_bytes(&preregistration_bytes);
    if preregistration_sha256 != cli.expected_preregistration_sha256 {
        return Err(format!(
            "preregistration digest mismatch: expected {}, observed {}",
            cli.expected_preregistration_sha256, preregistration_sha256
        ));
    }
    let prereg: Preregistration = serde_json::from_slice(&preregistration_bytes)
        .map_err(|error| format!("preregistration JSON is malformed: {error}"))?;
    validate_preregistration(&prereg)?;

    let trial_bytes = fs::read(&cli.trial_results)
        .map_err(|error| format!("cannot read trial results: {error}"))?;
    let trial_results_sha256 = sha256_bytes(&trial_bytes);
    let (grouped, input_trial_record_count) =
        parse_trial_results(&trial_bytes, &prereg, &cli.score_reports_dir)?;
    let pairs = join_assignments(&prereg, grouped)?;
    let report = make_report(
        &prereg,
        preregistration_sha256,
        trial_results_sha256,
        input_trial_record_count,
        pairs,
    );
    write_json(&cli.report_out, &report)?;
    let report_bytes = fs::read(&cli.report_out).map_err(|error| {
        format!(
            "cannot re-read report {}: {error}",
            cli.report_out.display()
        )
    })?;
    println!(
        "ABLATION AUDIT: {} assigned tasks, {} fully resolved pairs, GCS wins {}, control wins {}; report sha256:{} ({})",
        report.assignment_count,
        report.paired.fully_resolved_pair_count,
        report.paired.gcs_wins,
        report.paired.control_wins,
        sha256_bytes(&report_bytes),
        cli.report_out.display()
    );
    Ok(())
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let cli = match parse_cli(&args) {
        Ok(Some(cli)) => cli,
        Ok(None) => return,
        Err(error) => {
            eprintln!("ERROR: {error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = run(cli) {
        eprintln!("ERROR: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_pair(repository_id: &str, gcs: Outcome, control: Outcome) -> AuditPair {
        AuditPair {
            assignment: Assignment {
                instance_id: format!("task-{repository_id}"),
                repository_id: repository_id.to_string(),
                first_arm: Arm::Gcs,
            },
            gcs: Some(ObservedTrial {
                record: test_trial("gcs", Arm::Gcs, gcs, 1),
                jsonl_row_sha256: "a".repeat(64),
            }),
            control: Some(ObservedTrial {
                record: test_trial("control", Arm::Control, control, 2),
                jsonl_row_sha256: "b".repeat(64),
            }),
        }
    }

    fn test_trial(run_id: &str, arm: Arm, outcome: Outcome, started_at_ms: u64) -> TrialRecord {
        TrialRecord {
            run_id: format!("{run_id}-run"),
            instance_id: "task".to_string(),
            arm,
            outcome,
            gcs_enabled: arm == Arm::Gcs,
            cost_usd: 1.0,
            latency_ms: 1_000,
            started_at_ms,
            model_id: "test-model".to_string(),
            agent_harness_sha256: "a".repeat(64),
            prompt_sha256: "b".repeat(64),
            scaffold_sha256: "c".repeat(64),
            budget_id: "test-budget".to_string(),
            container_image_id: format!("sha256:{}", "d".repeat(64)),
            network_policy: "none".to_string(),
            score_report_path: "report.json".to_string(),
            score_report_sha256: "e".repeat(64),
        }
    }

    #[test]
    fn sha256_standard_vector() {
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn preregistration_rejects_networked_trials_and_short_bootstrap() {
        let mut prereg = test_preregistration(10);
        prereg.network_policy = "bridge".to_string();
        assert!(
            validate_preregistration(&prereg)
                .unwrap_err()
                .contains("network_policy")
        );
        prereg.network_policy = "none".to_string();
        prereg.bootstrap_resamples = 999;
        assert!(
            validate_preregistration(&prereg)
                .unwrap_err()
                .contains("bootstrap_resamples")
        );
    }

    #[test]
    fn preregistration_rejects_hand_edited_randomized_order() {
        let mut prereg = test_preregistration(1);
        prereg.assignment[0].first_arm = match prereg.assignment[0].first_arm {
            Arm::Control => Arm::Gcs,
            Arm::Gcs => Arm::Control,
        };
        assert!(
            validate_preregistration(&prereg)
                .unwrap_err()
                .contains("hash-randomization seed")
        );
    }

    #[test]
    fn observed_run_order_must_match_preregistered_order() {
        let prereg = test_preregistration(1);
        let assignment = prereg.assignment[0].clone();
        let (gcs_time, control_time) = if assignment.first_arm == Arm::Gcs {
            (2, 1)
        } else {
            (1, 2)
        };
        let grouped = BTreeMap::from([(
            assignment.instance_id.clone(),
            BTreeMap::from([
                (
                    Arm::Gcs,
                    ObservedTrial {
                        record: test_trial("gcs", Arm::Gcs, Outcome::Pass, gcs_time),
                        jsonl_row_sha256: "a".repeat(64),
                    },
                ),
                (
                    Arm::Control,
                    ObservedTrial {
                        record: test_trial("control", Arm::Control, Outcome::Pass, control_time),
                        jsonl_row_sha256: "b".repeat(64),
                    },
                ),
            ]),
        )]);
        assert!(
            join_assignments(&prereg, grouped)
                .unwrap_err()
                .contains("execution order")
        );
    }

    #[test]
    fn duplicate_preregistered_tasks_are_rejected() {
        let mut prereg = test_preregistration(10);
        prereg.assignment.push(prereg.assignment[0].clone());
        assert!(
            validate_preregistration(&prereg)
                .unwrap_err()
                .contains("duplicate")
        );
    }

    #[test]
    fn preregistration_requires_immutable_digests() {
        let mut prereg = test_preregistration(10);
        prereg.agent_harness_sha256 = "not-a-hash".to_string();
        assert!(
            validate_preregistration(&prereg)
                .unwrap_err()
                .contains("agent_harness_sha256")
        );
    }

    #[test]
    fn gcs_flag_must_match_registered_arm() {
        let prereg = test_preregistration(10);
        let mut trial = test_trial("gcs", Arm::Gcs, Outcome::Pass, 1);
        trial.gcs_enabled = false;
        assert!(
            validate_trial_static(&trial, &prereg)
                .unwrap_err()
                .contains("gcs_enabled")
        );
    }

    #[test]
    fn common_configuration_must_match_preregistration() {
        let prereg = test_preregistration(10);
        let mut trial = test_trial("gcs", Arm::Gcs, Outcome::Pass, 1);
        trial.model_id = "different-model".to_string();
        assert!(
            validate_trial_static(&trial, &prereg)
                .unwrap_err()
                .contains("model_id")
        );
    }

    #[test]
    fn path_traversal_in_score_report_path_is_rejected() {
        assert!(validate_relative_report_path("../other/report.json").is_err());
        assert!(validate_relative_report_path("/tmp/report.json").is_err());
        assert!(validate_relative_report_path("nested/report.json").is_ok());
    }

    #[test]
    fn unresolved_results_expand_sensitivity_bounds_not_successes() {
        let pairs = vec![
            test_pair("repo-a", Outcome::Pass, Outcome::Fail),
            test_pair("repo-b", Outcome::Unresolved, Outcome::Pass),
        ];
        let (lower, upper) = unresolved_sensitivity_bounds(&pairs);
        assert_eq!(lower, 0.0);
        assert_eq!(upper, 50.0);
        let gcs = arm_metrics(&pairs, Arm::Gcs);
        assert_eq!(gcs.verified_pass_count, 1);
        assert_eq!(gcs.unresolved_count, 1);
        assert_eq!(gcs.verified_pass_rate, 0.5);
    }

    #[test]
    fn resolved_effect_ignores_incomplete_pairs_and_keeps_win_loss_tie_counts() {
        let pairs = vec![
            test_pair("repo-a", Outcome::Pass, Outcome::Fail),
            test_pair("repo-b", Outcome::Fail, Outcome::Pass),
            test_pair("repo-c", Outcome::Pass, Outcome::Pass),
            test_pair("repo-d", Outcome::Unresolved, Outcome::Pass),
        ];
        let (resolved, gcs_wins, control_wins, ties, effect) = resolved_effect(&pairs);
        assert_eq!(resolved, 3);
        assert_eq!(gcs_wins, 1);
        assert_eq!(control_wins, 1);
        assert_eq!(ties, 1);
        assert_eq!(effect, Some(0.0));
    }

    #[test]
    fn cluster_bootstrap_is_deterministic_and_withholds_ci_for_too_few_repositories() {
        let small = vec![
            test_pair("repo-a", Outcome::Pass, Outcome::Fail),
            test_pair("repo-b", Outcome::Pass, Outcome::Fail),
        ];
        let (small_clusters, interval) = cluster_bootstrap_ci(&small, 7, 1_000);
        assert_eq!(small_clusters, 2);
        assert!(interval.is_none());

        let large = (0..10)
            .map(|index| test_pair(&format!("repo-{index}"), Outcome::Pass, Outcome::Fail))
            .collect::<Vec<_>>();
        let first = cluster_bootstrap_ci(&large, 7, 1_000);
        let second = cluster_bootstrap_ci(&large, 7, 1_000);
        assert_eq!(first.0, 10);
        assert_eq!(
            first
                .1
                .map(|ci| (ci.lower_percentage_points, ci.upper_percentage_points)),
            second
                .1
                .map(|ci| (ci.lower_percentage_points, ci.upper_percentage_points))
        );
        assert_eq!(first.1.unwrap().lower_percentage_points, 100.0);
    }

    #[test]
    fn missing_arm_records_are_preserved_as_unresolved() {
        let pairs = vec![AuditPair {
            assignment: Assignment {
                instance_id: "task-missing".to_string(),
                repository_id: "repo-a".to_string(),
                first_arm: Arm::Gcs,
            },
            gcs: None,
            control: None,
        }];
        assert_eq!(pairs[0].outcome(Arm::Gcs), Outcome::Unresolved);
        assert_eq!(pairs[0].outcome(Arm::Control), Outcome::Unresolved);
        let (lower, upper) = unresolved_sensitivity_bounds(&pairs);
        assert_eq!(lower, -100.0);
        assert_eq!(upper, 100.0);
    }

    #[test]
    fn score_receipt_requires_the_pinned_manifest_and_task_outcomes() {
        let temp = tempfile::tempdir().unwrap();
        let task_manifest = "a".repeat(64);
        let report = serde_json::json!({
            "format_version": 1,
            "protocol_id": SCORE_PROTOCOL_ID,
            "task_manifest_sha256": task_manifest,
            "task_count": 1,
            "tasks": [{"instance_id":"task-a","outcome":"pass"}]
        });
        let bytes = serde_json::to_vec(&report).unwrap();
        fs::write(temp.path().join("score.json"), &bytes).unwrap();
        let digest = sha256_bytes(&bytes);
        let outcomes =
            load_score_report(temp.path(), "score.json", &digest, &task_manifest).unwrap();
        assert_eq!(outcomes.get("task-a"), Some(&Outcome::Pass));
        assert!(load_score_report(temp.path(), "score.json", &digest, &"b".repeat(64)).is_err());
        assert!(
            load_score_report(temp.path(), "score.json", &"0".repeat(64), &task_manifest).is_err()
        );
    }

    fn test_preregistration(repository_count: usize) -> Preregistration {
        Preregistration {
            format_version: FORMAT_VERSION,
            protocol_id: PROTOCOL_ID.to_string(),
            benchmark_manifest_sha256: "a".repeat(64),
            model_id: "test-model".to_string(),
            agent_harness_sha256: "a".repeat(64),
            prompt_sha256: "b".repeat(64),
            scaffold_sha256: "c".repeat(64),
            budget_id: "test-budget".to_string(),
            container_image_id: format!("sha256:{}", "d".repeat(64)),
            network_policy: "none".to_string(),
            randomization_seed: "seed-locked-before-results".to_string(),
            bootstrap_seed: 7,
            bootstrap_resamples: 1_000,
            assignment: (0..repository_count)
                .map(|index| Assignment {
                    instance_id: format!("task-{index}"),
                    repository_id: format!("repo-{index}"),
                    first_arm: randomized_first_arm(
                        "seed-locked-before-results",
                        &format!("task-{index}"),
                    ),
                })
                .collect(),
        }
    }
}
