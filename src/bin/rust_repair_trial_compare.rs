//! Paired, repository-clustered comparison of two audited Rust repair runs.
//!
//! This binary consumes score reports from rust-repair-score-audit; it does not
//! execute tasks, attest provider identities, or turn missing evidence into failures.
//! Unresolved outcomes stay in the denominator and widen the primary effect bounds.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::PathBuf;

const PROTOCOL_ID: &str = "symthaea-rust-repair-paired-comparison";
const SCORE_PROTOCOL_ID: &str = "symthaea-rust-repair";
const REPORT_VERSION: u32 = 1;

#[derive(Debug)]
struct Cli {
    preregistration: PathBuf,
    expected_preregistration_sha256: String,
    control_report: PathBuf,
    treatment_report: PathBuf,
    cost_ledger: PathBuf,
    report_out: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    Pass,
    Fail,
    Unresolved,
}

impl Outcome {
    fn parse(value: &Value, context: &str) -> Result<Self, String> {
        match value.as_str() {
            Some("pass") => Ok(Self::Pass),
            Some("fail") => Ok(Self::Fail),
            Some("unresolved") => Ok(Self::Unresolved),
            Some(other) => Err(format!("{context}: invalid outcome {other:?}")),
            None => Err(format!("{context}: outcome must be a string")),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::Unresolved => "unresolved",
        }
    }

    fn known_success(self) -> Option<bool> {
        match self {
            Self::Pass => Some(true),
            Self::Fail => Some(false),
            Self::Unresolved => None,
        }
    }

    fn lower_success(self) -> f64 {
        f64::from(matches!(self, Self::Pass))
    }

    fn upper_success(self) -> f64 {
        f64::from(!matches!(self, Self::Fail))
    }
}

#[derive(Clone, Debug)]
struct TaskSpec {
    instance_id: String,
    repo: String,
    stratum: String,
    evaluation_order: String,
}

#[derive(Clone, Debug)]
struct CostRow {
    instance_id: String,
    control_cost_usd: f64,
    treatment_cost_usd: f64,
    control_receipt_sha256: String,
    treatment_receipt_sha256: String,
}

#[derive(Clone, Debug)]
struct PairedTask {
    spec: TaskSpec,
    control: Outcome,
    treatment: Outcome,
    costs: CostRow,
}

fn usage() -> &'static str {
    "Usage:\n  cargo run --locked --bin rust-repair-trial-compare --features rust-repair-manifest -- \\\n    --preregistration <locked-preregistration.json> \\\n    --expected-preregistration-sha256 <64-hex-pinned-externally> \\\n    --control-report <control-score-report.json> \\\n    --treatment-report <treatment-score-report.json> \\\n    --cost-ledger <per-task-cost-and-receipt-ledger.json> \\\n    --report-out <comparison-report.json>"
}

fn parse_cli(args: &[String]) -> Result<Option<Cli>, String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{}", usage());
        return Ok(None);
    }
    let mut values = BTreeMap::<String, String>::new();
    let mut index = 0;
    while index < args.len() {
        let key = &args[index];
        if !key.starts_with("--") {
            return Err(format!("unexpected positional argument {key:?}"));
        }
        if index + 1 >= args.len() || args[index + 1].starts_with("--") {
            return Err(format!("missing value for {key}"));
        }
        if values.insert(key.clone(), args[index + 1].clone()).is_some() {
            return Err(format!("duplicate argument {key}"));
        }
        index += 2;
    }

    let mut take = |key: &str| {
        values
            .remove(key)
            .ok_or_else(|| format!("required argument {key} missing"))
    };
    let cli = Cli {
        preregistration: PathBuf::from(take("--preregistration")?),
        expected_preregistration_sha256: take("--expected-preregistration-sha256")?,
        control_report: PathBuf::from(take("--control-report")?),
        treatment_report: PathBuf::from(take("--treatment-report")?),
        cost_ledger: PathBuf::from(take("--cost-ledger")?),
        report_out: PathBuf::from(take("--report-out")?),
    };
    if !values.is_empty() {
        return Err(format!("unknown arguments: {:?}", values.keys().collect::<Vec<_>>()));
    }
    if !is_hex(&cli.expected_preregistration_sha256, 64) {
        return Err("--expected-preregistration-sha256 must be 64 hex characters".into());
    }
    Ok(Some(cli))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn read_json(path: &PathBuf, label: &str) -> Result<(Vec<u8>, Value), String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read {label}: {error}"))?;
    let value = serde_json::from_slice::<Value>(&bytes)
        .map_err(|error| format!("{label} is invalid JSON: {error}"))?;
    Ok((bytes, value))
}

fn required_string<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| format!("{context}: {key} missing or invalid"))
}

fn required_count(value: &Value, key: &str, context: &str) -> Result<usize, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| format!("{context}: {key} missing or invalid"))
}

fn required_fraction(value: &Value, key: &str, context: &str) -> Result<f64, String> {
    let number = value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{context}: {key} missing or invalid"))?;
    if !number.is_finite() || !(0.0..=1.0).contains(&number) {
        return Err(format!("{context}: {key} must be finite and in [0, 1]"));
    }
    Ok(number)
}

fn parse_task_specs(prereg: &Value) -> Result<Vec<TaskSpec>, String> {
    let rows = prereg
        .get("tasks")
        .and_then(Value::as_array)
        .ok_or_else(|| "preregistration: tasks must be an array".to_string())?;
    if rows.is_empty() {
        return Err("preregistration: tasks cannot be empty".into());
    }
    let mut seen = BTreeSet::new();
    let mut specs = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let context = format!("preregistration task row {index}");
        let instance_id = required_string(row, "instance_id", &context)?.to_string();
        if !seen.insert(instance_id.clone()) {
            return Err(format!("preregistration duplicates task {instance_id:?}"));
        }
        let repo = required_string(row, "repo", &context)?.to_string();
        let stratum = required_string(row, "stratum", &context)?.to_string();
        let evaluation_order = required_string(row, "evaluation_order", &context)?.to_string();
        if !matches!(evaluation_order.as_str(), "control_first" | "treatment_first") {
            return Err(format!(
                "{context}: evaluation_order must be control_first or treatment_first"
            ));
        }
        specs.push(TaskSpec { instance_id, repo, stratum, evaluation_order });
    }
    Ok(specs)
}

fn validate_preregistration(prereg: &Value) -> Result<Vec<TaskSpec>, String> {
    if prereg.get("format_version").and_then(Value::as_u64) != Some(1)
        || prereg.get("protocol_id").and_then(Value::as_str) != Some(PROTOCOL_ID)
    {
        return Err("unsupported preregistration format/protocol".into());
    }
    for field in [
        "study_id",
        "control_arm",
        "treatment_arm",
        "control_model_scaffold",
        "treatment_model_scaffold",
        "task_manifest_sha256",
        "candidate_jsonl_sha256",
        "bootstrap_seed",
    ] {
        required_string(prereg, field, "preregistration")?;
    }
    if prereg["control_arm"] == prereg["treatment_arm"] {
        return Err("control_arm and treatment_arm must differ".into());
    }
    for key in ["task_manifest_sha256", "candidate_jsonl_sha256"] {
        if !is_hex(required_string(prereg, key, "preregistration")?, 64) {
            return Err(format!("preregistration: {key} must be a 64-character SHA-256"));
        }
    }
    let min_tasks = required_count(prereg, "min_tasks", "preregistration")?;
    let min_repositories = required_count(prereg, "min_repositories", "preregistration")?;
    let bootstrap_replicates = required_count(prereg, "bootstrap_replicates", "preregistration")?;
    if min_tasks == 0 || min_repositories == 0 {
        return Err("preregistration minimum tasks/repositories must be positive".into());
    }
    if !(1_000..=100_000).contains(&bootstrap_replicates) {
        return Err("bootstrap_replicates must be between 1000 and 100000".into());
    }
    for key in [
        "max_unresolved_rate",
        "max_unresolved_rate_difference",
        "target_absolute_effect",
        "cost_reduction_target",
        "noninferiority_margin",
    ] {
        let value = prereg.get(key).and_then(Value::as_f64)
            .ok_or_else(|| format!("preregistration: {key} missing or invalid"))?;
        if !value.is_finite() || value < 0.0 || value > 1.0 {
            return Err(format!("preregistration: {key} must be finite and in [0, 1]"));
        }
    }
    parse_task_specs(prereg)
}

fn outcome_map(report: &Value, label: &str, manifest_sha: &str, candidate_sha: &str)
    -> Result<BTreeMap<String, Outcome>, String>
{
    if report.get("format_version").and_then(Value::as_u64) != Some(1)
        || report.get("protocol_id").and_then(Value::as_str) != Some(SCORE_PROTOCOL_ID)
    {
        return Err(format!("{label}: unsupported scored-run report format/protocol"));
    }
    if required_string(report, "task_manifest_sha256", label)? != manifest_sha {
        return Err(format!("{label}: task-manifest digest differs from preregistration"));
    }
    if required_string(report, "candidate_jsonl_sha256", label)? != candidate_sha {
        return Err(format!("{label}: candidate split digest differs from preregistration"));
    }
    let rows = report.get("tasks").and_then(Value::as_array)
        .ok_or_else(|| format!("{label}: tasks missing or not an array"))?;
    let task_count = required_count(report, "task_count", label)?;
    if task_count != rows.len() {
        return Err(format!("{label}: task_count does not equal number of task rows"));
    }
    let mut outcomes = BTreeMap::new();
    let mut counts = BTreeMap::from([
        ("pass", 0usize),
        ("fail", 0usize),
        ("unresolved", 0usize),
    ]);
    for (index, row) in rows.iter().enumerate() {
        let context = format!("{label} task row {index}");
        let id = required_string(row, "instance_id", &context)?.to_string();
        let outcome = Outcome::parse(
            row.get("outcome").ok_or_else(|| format!("{context}: outcome missing"))?,
            &context,
        )?;
        if outcomes.insert(id.clone(), outcome).is_some() {
            return Err(format!("{label}: duplicate task id {id:?}"));
        }
        *counts.get_mut(outcome.label()).expect("known outcome label") += 1;
    }
    for (key, expected) in [
        ("success_count", counts["pass"]),
        ("failure_count", counts["fail"]),
        ("unresolved_count", counts["unresolved"]),
    ] {
        if required_count(report, key, label)? != expected {
            return Err(format!("{label}: {key} disagrees with independently re-counted task rows"));
        }
    }
    let pass_at_1 = report.get("pass_at_1").filter(|value| !value.is_null()).and_then(Value::as_f64);
    if counts["unresolved"] == 0 {
        let expected = counts["pass"] as f64 / rows.len() as f64;
        if pass_at_1.is_none_or(|observed| (observed - expected).abs() > 1e-12) {
            return Err(format!("{label}: pass_at_1 disagrees with task rows"));
        }
    } else if !report.get("pass_at_1").is_none_or(Value::is_null) {
        return Err(format!("{label}: pass_at_1 must be null when any task is unresolved"));
    }
    Ok(outcomes)
}

fn parse_cost_ledger(
    ledger: &Value,
    study_id: &str,
    expected_ids: &BTreeSet<String>,
) -> Result<BTreeMap<String, CostRow>, String> {
    if ledger.get("format_version").and_then(Value::as_u64) != Some(1)
        || required_string(ledger, "study_id", "cost ledger")? != study_id
    {
        return Err("cost ledger format or study_id does not match preregistration".into());
    }
    let rows = ledger.get("tasks").and_then(Value::as_array)
        .ok_or_else(|| "cost ledger tasks missing or not an array".to_string())?;
    let mut parsed = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        let context = format!("cost ledger row {index}");
        let instance_id = required_string(row, "instance_id", &context)?.to_string();
        let control_cost_usd = row.get("control_cost_usd").and_then(Value::as_f64)
            .ok_or_else(|| format!("{context}: control_cost_usd missing or invalid"))?;
        let treatment_cost_usd = row.get("treatment_cost_usd").and_then(Value::as_f64)
            .ok_or_else(|| format!("{context}: treatment_cost_usd missing or invalid"))?;
        if !control_cost_usd.is_finite() || control_cost_usd < 0.0
            || !treatment_cost_usd.is_finite() || treatment_cost_usd < 0.0
        {
            return Err(format!("{context}: observed cost must be finite and non-negative"));
        }
        let control_receipt_sha256 = required_string(row, "control_receipt_sha256", &context)?.to_string();
        let treatment_receipt_sha256 = required_string(row, "treatment_receipt_sha256", &context)?.to_string();
        if !is_hex(&control_receipt_sha256, 64) || !is_hex(&treatment_receipt_sha256, 64) {
            return Err(format!("{context}: receipt hashes must be 64-character SHA-256 values"));
        }
        let cost_row = CostRow {
            instance_id: instance_id.clone(),
            control_cost_usd,
            treatment_cost_usd,
            control_receipt_sha256,
            treatment_receipt_sha256,
        };
        if parsed.insert(instance_id.clone(), cost_row).is_some() {
            return Err(format!("cost ledger duplicates task {instance_id:?}"));
        }
    }
    let actual_ids = parsed.keys().cloned().collect::<BTreeSet<_>>();
    if &actual_ids != expected_ids {
        return Err("cost ledger task IDs must exactly equal preregistered task IDs".into());
    }
    Ok(parsed)
}

fn quantile(values: &mut [f64], probability: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let position = ((values.len() - 1) as f64 * probability).round() as usize;
    values[position]
}

#[derive(Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }
}

fn conservative_effects(control: Outcome, treatment: Outcome) -> (f64, f64) {
    (
        treatment.lower_success() - control.upper_success(),
        treatment.upper_success() - control.lower_success(),
    )
}

fn resolved_effects(tasks: &[PairedTask]) -> (Option<f64>, usize) {
    let mut control_passes = 0usize;
    let mut treatment_passes = 0usize;
    let mut resolved = 0usize;
    for task in tasks {
        if let (Some(control), Some(treatment)) =
            (task.control.known_success(), task.treatment.known_success())
        {
            resolved += 1;
            control_passes += usize::from(control);
            treatment_passes += usize::from(treatment);
        }
    }
    let effect = (resolved > 0).then(|| {
        (treatment_passes as f64 - control_passes as f64) / resolved as f64
    });
    (effect, resolved)
}

fn repo_cluster_bootstrap(
    tasks: &[PairedTask],
    reps: usize,
    seed: u64,
) -> Result<(f64, f64), String> {
    let mut clusters = BTreeMap::<String, Vec<usize>>::new();
    for (index, task) in tasks.iter().enumerate() {
        clusters.entry(task.spec.repo.clone()).or_default().push(index);
    }
    let cluster_indices = clusters.values().collect::<Vec<_>>();
    if cluster_indices.is_empty() {
        return Err("cannot bootstrap an empty task set".into());
    }
    let mut rng = SplitMix64 { state: seed };
    let mut lower_draws = Vec::with_capacity(reps);
    let mut upper_draws = Vec::with_capacity(reps);
    for _ in 0..reps {
        let mut lower_sum = 0.0;
        let mut upper_sum = 0.0;
        let mut denominator = 0usize;
        for _ in 0..cluster_indices.len() {
            let selected = (rng.next() as usize) % cluster_indices.len();
            for index in cluster_indices[selected] {
                let task = &tasks[*index];
                let (lower, upper) = conservative_effects(task.control, task.treatment);
                lower_sum += lower;
                upper_sum += upper;
                denominator += 1;
            }
        }
        if denominator == 0 {
            return Err("bootstrap replicate contained no task rows".into());
        }
        lower_draws.push(lower_sum / denominator as f64);
        upper_draws.push(upper_sum / denominator as f64);
    }
    let ci_low = quantile(&mut lower_draws, 0.025);
    let ci_high = quantile(&mut upper_draws, 0.975);
    Ok((ci_low, ci_high))
}

fn run(cli: Cli) -> Result<(), String> {
    let (prereg_bytes, prereg) = read_json(&cli.preregistration, "preregistration")?;
    let prereg_sha256 = sha256_bytes(&prereg_bytes);
    if prereg_sha256 != cli.expected_preregistration_sha256 {
        return Err(format!(
            "preregistration SHA-256 mismatch: expected {}, observed {}",
            cli.expected_preregistration_sha256, prereg_sha256
        ));
    }
    let task_specs = validate_preregistration(&prereg)?;
    let control_label = required_string(&prereg, "control_arm", "preregistration")?;
    let treatment_label = required_string(&prereg, "treatment_arm", "preregistration")?;
    let study_id = required_string(&prereg, "study_id", "preregistration")?;
    let manifest_sha = required_string(&prereg, "task_manifest_sha256", "preregistration")?;
    let candidate_sha = required_string(&prereg, "candidate_jsonl_sha256", "preregistration")?;

    let (control_bytes, control_report) = read_json(&cli.control_report, "control score report")?;
    let (treatment_bytes, treatment_report) = read_json(&cli.treatment_report, "treatment score report")?;
    let (cost_bytes, cost_ledger) = read_json(&cli.cost_ledger, "cost ledger")?;
    let control_outcomes = outcome_map(&control_report, "control report", manifest_sha, candidate_sha)?;
    let treatment_outcomes = outcome_map(&treatment_report, "treatment report", manifest_sha, candidate_sha)?;
    let expected_ids = task_specs.iter().map(|spec| spec.instance_id.clone()).collect::<BTreeSet<_>>();
    if control_outcomes.keys().cloned().collect::<BTreeSet<_>>() != expected_ids {
        return Err("control report task IDs do not exactly equal preregistered task IDs".into());
    }
    if treatment_outcomes.keys().cloned().collect::<BTreeSet<_>>() != expected_ids {
        return Err("treatment report task IDs do not exactly equal preregistered task IDs".into());
    }
    let costs = parse_cost_ledger(&cost_ledger, study_id, &expected_ids)?;

    let mut tasks = Vec::with_capacity(task_specs.len());
    for spec in task_specs {
        let id = spec.instance_id.clone();
        tasks.push(PairedTask {
            control: control_outcomes[&id],
            treatment: treatment_outcomes[&id],
            costs: costs[&id].clone(),
            spec,
        });
    }
    let n = tasks.len();
    let repos = tasks.iter().map(|task| task.spec.repo.clone()).collect::<BTreeSet<_>>();
    let strata = tasks.iter().map(|task| task.spec.stratum.clone()).collect::<BTreeSet<_>>();
    let control_successes = tasks.iter().filter(|task| task.control == Outcome::Pass).count();
    let treatment_successes = tasks.iter().filter(|task| task.treatment == Outcome::Pass).count();
    let control_unresolved = tasks.iter().filter(|task| task.control == Outcome::Unresolved).count();
    let treatment_unresolved = tasks.iter().filter(|task| task.treatment == Outcome::Unresolved).count();
    let control_unresolved_rate = control_unresolved as f64 / n as f64;
    let treatment_unresolved_rate = treatment_unresolved as f64 / n as f64;
    let unresolved_rate_difference = (treatment_unresolved_rate - control_unresolved_rate).abs();
    let control_resolved_rate = (n > control_unresolved).then(|| {
        control_successes as f64 / (n - control_unresolved) as f64
    });
    let treatment_resolved_rate = (n > treatment_unresolved).then(|| {
        treatment_successes as f64 / (n - treatment_unresolved) as f64
    });
    let lower_point = tasks.iter().map(|task| conservative_effects(task.control, task.treatment).0).sum::<f64>() / n as f64;
    let upper_point = tasks.iter().map(|task| conservative_effects(task.control, task.treatment).1).sum::<f64>() / n as f64;
    let (resolved_effect, resolved_pair_count) = resolved_effects(&tasks);

    let seed_material = format!(
        "symthaea-rust-repair-bootstrap-v1:{}:{}",
        prereg_sha256,
        required_string(&prereg, "bootstrap_seed", "preregistration")?
    );
    let seed_digest = sha256_bytes(seed_material.as_bytes());
    let seed = u64::from_le_bytes(
        hex_to_bytes_prefix(&seed_digest).try_into().map_err(|_| "invalid internal bootstrap seed".to_string())?
    );
    let bootstrap_replicates = required_count(&prereg, "bootstrap_replicates", "preregistration")?;
    let (ci_lower, ci_upper) = repo_cluster_bootstrap(&tasks, bootstrap_replicates, seed)?;

    let control_cost_total = tasks.iter().map(|task| task.costs.control_cost_usd).sum::<f64>();
    let treatment_cost_total = tasks.iter().map(|task| task.costs.treatment_cost_usd).sum::<f64>();
    let control_cost_per_fix = (control_successes > 0)
        .then(|| control_cost_total / control_successes as f64);
    let treatment_cost_per_fix = (treatment_successes > 0)
        .then(|| treatment_cost_total / treatment_successes as f64);
    let max_unresolved = required_fraction(&prereg, "max_unresolved_rate", "preregistration")?;
    let max_unresolved_diff = required_fraction(&prereg, "max_unresolved_rate_difference", "preregistration")?;
    let target_effect = required_fraction(&prereg, "target_absolute_effect", "preregistration")?;
    let cost_reduction_target = required_fraction(&prereg, "cost_reduction_target", "preregistration")?;
    let noninferiority_margin = required_fraction(&prereg, "noninferiority_margin", "preregistration")?;
    let min_tasks = required_count(&prereg, "min_tasks", "preregistration")?;
    let min_repos = required_count(&prereg, "min_repositories", "preregistration")?;
    let quality_gates_pass = n >= min_tasks
        && repos.len() >= min_repos
        && control_unresolved_rate <= max_unresolved
        && treatment_unresolved_rate <= max_unresolved
        && unresolved_rate_difference <= max_unresolved_diff;

    let superiority_target_met = quality_gates_pass
        && resolved_effect.is_some_and(|effect| effect >= target_effect)
        && ci_lower > 0.0;
    let cost_target_met = match (control_cost_per_fix, treatment_cost_per_fix) {
        (Some(control), Some(treatment)) => {
            quality_gates_pass
                && treatment <= control * (1.0 - cost_reduction_target)
                && ci_lower >= -noninferiority_margin
        }
        _ => false,
    };

    let output_tasks = tasks.iter().map(|task| json!({
        "instance_id": task.spec.instance_id,
        "repo": task.spec.repo,
        "stratum": task.spec.stratum,
        "evaluation_order": task.spec.evaluation_order,
        "control_outcome": task.control.label(),
        "treatment_outcome": task.treatment.label(),
        "control_cost_usd": task.costs.control_cost_usd,
        "treatment_cost_usd": task.costs.treatment_cost_usd,
        "control_receipt_sha256": task.costs.control_receipt_sha256,
        "treatment_receipt_sha256": task.costs.treatment_receipt_sha256,
        "conservative_effect_lower": conservative_effects(task.control, task.treatment).0,
        "conservative_effect_upper": conservative_effects(task.control, task.treatment).1,
    })).collect::<Vec<_>>();

    let output = json!({
        "format_version": REPORT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "study_id": study_id,
        "control_arm": control_label,
        "treatment_arm": treatment_label,
        "preregistration_sha256": prereg_sha256,
        "task_manifest_sha256": manifest_sha,
        "candidate_jsonl_sha256": candidate_sha,
        "control_score_report_sha256": sha256_bytes(&control_bytes),
        "treatment_score_report_sha256": sha256_bytes(&treatment_bytes),
        "cost_ledger_sha256": sha256_bytes(&cost_bytes),
        "task_count": n,
        "repository_count": repos.len(),
        "stratum_count": strata.len(),
        "control": {
            "successes": control_successes,
            "unresolved": control_unresolved,
            "unresolved_rate": control_unresolved_rate,
            "resolved_success_rate": control_resolved_rate,
            "total_cost_usd": control_cost_total,
            "cost_per_verified_fix_usd": control_cost_per_fix,
        },
        "treatment": {
            "successes": treatment_successes,
            "unresolved": treatment_unresolved,
            "unresolved_rate": treatment_unresolved_rate,
            "resolved_success_rate": treatment_resolved_rate,
            "total_cost_usd": treatment_cost_total,
            "cost_per_verified_fix_usd": treatment_cost_per_fix,
        },
        "paired": {
            "resolved_pair_count": resolved_pair_count,
            "resolved_complete_case_effect": resolved_effect,
            "conservative_effect_lower": lower_point,
            "conservative_effect_upper": upper_point,
            "repository_cluster_bootstrap_replicates": bootstrap_replicates,
            "repository_cluster_bootstrap_95pct_conservative_interval": {
                "lower": ci_lower,
                "upper": ci_upper,
            },
            "unresolved_rate_difference_absolute": unresolved_rate_difference,
        },
        "gates": {
            "quality_gates_pass": quality_gates_pass,
            "minimum_tasks": min_tasks,
            "minimum_repositories": min_repos,
            "max_unresolved_rate": max_unresolved,
            "max_unresolved_rate_difference": max_unresolved_diff,
            "target_absolute_effect": target_effect,
            "cost_reduction_target": cost_reduction_target,
            "noninferiority_margin": noninferiority_margin,
            "superiority_target_met": superiority_target_met,
            "cost_target_met": cost_target_met,
            "decision_grade": quality_gates_pass,
        },
        "interpretation": "This report compares two scored-run reports on the exact same preregistered task set. Unresolved outcomes remain in the denominator and widen the conservative effect interval. The bootstrap resamples repositories, not individual tasks. The target flags are internal preregistered decision gates, not a universal industry standard or an SOTA claim. Receipt hashes are recorded, not independently authenticated; preserve and externally audit the raw receipts.",
        "tasks": output_tasks,
    });
    let serialized = serde_json::to_vec_pretty(&output)
        .map_err(|error| format!("cannot serialize comparison report: {error}"))?;
    if let Some(parent) = cli.report_out.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|error| format!("cannot create report directory: {error}"))?;
    }
    fs::write(&cli.report_out, serialized).map_err(|error| format!("cannot write report: {error}"))?;
    Ok(())
}

fn hex_to_bytes_prefix(hex: &str) -> Vec<u8> {
    hex.as_bytes()
        .chunks_exact(2)
        .take(8)
        .map(|pair| {
            let high = (pair[0] as char).to_digit(16).unwrap_or(0);
            let low = (pair[1] as char).to_digit(16).unwrap_or(0);
            ((high << 4) | low) as u8
        })
        .collect()
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let cli = match parse_cli(&args) {
        Ok(Some(cli)) => cli,
        Ok(None) => return,
        Err(error) => {
            eprintln!("ERROR: {error}\n{}", usage());
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

    #[test]
    fn unresolved_outcomes_expand_effect_bounds_in_the_conservative_direction() {
        assert_eq!(conservative_effects(Outcome::Unresolved, Outcome::Pass), (0.0, 1.0));
        assert_eq!(conservative_effects(Outcome::Pass, Outcome::Unresolved), (-1.0, 0.0));
        assert_eq!(conservative_effects(Outcome::Fail, Outcome::Pass), (1.0, 1.0));
        assert_eq!(conservative_effects(Outcome::Pass, Outcome::Fail), (-1.0, -1.0));
    }

    #[test]
    fn splitmix_is_deterministic() {
        let mut left = SplitMix64 { state: 7 };
        let mut right = SplitMix64 { state: 7 };
        for _ in 0..50 {
            assert_eq!(left.next(), right.next());
        }
    }

    #[test]
    fn quantile_is_nearest_rank_with_rounding() {
        let mut values = vec![10.0, 0.0, 5.0, 15.0, 20.0];
        assert_eq!(quantile(&mut values, 0.0), 0.0);
        assert_eq!(quantile(&mut values, 1.0), 20.0);
        assert_eq!(quantile(&mut values, 0.5), 10.0);
    }

    #[test]
    fn resolved_effect_excludes_unresolved_pairs_but_counts_them_elsewhere() {
        let spec = TaskSpec {
            instance_id: "a".into(), repo: "owner/repo".into(),
            stratum: "bug".into(), evaluation_order: "control_first".into(),
        };
        let costs = CostRow {
            instance_id: "a".into(), control_cost_usd: 1.0, treatment_cost_usd: 1.0,
            control_receipt_sha256: "a".repeat(64), treatment_receipt_sha256: "b".repeat(64),
        };
        let tasks = vec![
            PairedTask { spec: spec.clone(), control: Outcome::Fail, treatment: Outcome::Pass, costs: costs.clone() },
            PairedTask { spec: TaskSpec { instance_id: "b".into(), ..spec.clone() }, control: Outcome::Pass, treatment: Outcome::Pass, costs: CostRow { instance_id: "b".into(), ..costs.clone() } },
            PairedTask { spec: TaskSpec { instance_id: "c".into(), ..spec }, control: Outcome::Unresolved, treatment: Outcome::Pass, costs: CostRow { instance_id: "c".into(), ..costs } },
        ];
        let (effect, resolved) = resolved_effects(&tasks);
        assert_eq!(resolved, 2);
        assert_eq!(effect, Some(0.5));
    }

    #[test]
    fn score_parser_rejects_duplicate_task_ids() {
        let report = json!({
            "format_version": 1, "protocol_id": SCORE_PROTOCOL_ID,
            "task_manifest_sha256": "a".repeat(64),
            "candidate_jsonl_sha256": "b".repeat(64),
            "task_count": 2, "success_count": 2, "failure_count": 0,
            "unresolved_count": 0, "pass_at_1": 1.0,
            "tasks": [
                {"instance_id":"x", "outcome":"pass"},
                {"instance_id":"x", "outcome":"pass"}
            ]
        });
        assert!(outcome_map(&report, "test", &"a".repeat(64), &"b".repeat(64)).is_err());
    }

    #[test]
    fn score_parser_rejects_a_false_headline_counter() {
        let report = json!({
            "format_version": 1, "protocol_id": SCORE_PROTOCOL_ID,
            "task_manifest_sha256": "a".repeat(64),
            "candidate_jsonl_sha256": "b".repeat(64),
            "task_count": 1, "success_count": 0, "failure_count": 0,
            "unresolved_count": 1, "pass_at_1": null,
            "tasks": [{"instance_id":"x", "outcome":"unresolved"}]
        });
        let mut inconsistent = report.clone();
        inconsistent["success_count"] = json!(1);
        assert!(outcome_map(&inconsistent, "test", &"a".repeat(64), &"b".repeat(64)).is_err());
        assert!(outcome_map(&report, "test", &"a".repeat(64), &"b".repeat(64)).is_ok());
    }

    #[test]
    fn score_parser_rejects_headline_with_unresolved_rows() {
        let report = json!({
            "format_version": 1, "protocol_id": SCORE_PROTOCOL_ID,
            "task_manifest_sha256": "a".repeat(64),
            "candidate_jsonl_sha256": "b".repeat(64),
            "task_count": 1, "success_count": 0, "failure_count": 0,
            "unresolved_count": 1, "pass_at_1": 0.0,
            "tasks": [{"instance_id":"x", "outcome":"unresolved"}]
        });
        assert!(outcome_map(&report, "test", &"a".repeat(64), &"b".repeat(64)).is_err());
    }

    #[test]
    fn task_specs_reject_duplicate_ids_and_unknown_order() {
        let duplicate = json!({"tasks":[
            {"instance_id":"x","repo":"a/b","stratum":"bug","evaluation_order":"control_first"},
            {"instance_id":"x","repo":"a/c","stratum":"bug","evaluation_order":"treatment_first"}
        ]});
        assert!(parse_task_specs(&duplicate).is_err());
        let bad_order = json!({"tasks":[
            {"instance_id":"x","repo":"a/b","stratum":"bug","evaluation_order":"random"}
        ]});
        assert!(parse_task_specs(&bad_order).is_err());
    }

    #[test]
    fn cost_ledger_requires_exact_task_coverage() {
        let ledger = json!({
            "format_version":1,"study_id":"s","tasks":[
                {"instance_id":"x","control_cost_usd":1.0,"treatment_cost_usd":2.0,
                 "control_receipt_sha256":"a".repeat(64),"treatment_receipt_sha256":"b".repeat(64)}
            ]
        });
        assert!(parse_cost_ledger(&ledger,"s",&BTreeSet::from(["x".to_string(),"y".to_string()])).is_err());
        assert!(parse_cost_ledger(&ledger,"wrong",&BTreeSet::from(["x".to_string()])).is_err());
        assert!(parse_cost_ledger(&ledger,"s",&BTreeSet::from(["x".to_string()])).is_ok());
    }

    #[test]
    fn repo_cluster_bootstrap_is_reproducible() {
        let mut tasks = Vec::new();
        for (id, repo, control, treatment) in [
            ("a", "org/one", Outcome::Fail, Outcome::Pass),
            ("b", "org/one", Outcome::Fail, Outcome::Pass),
            ("c", "org/two", Outcome::Pass, Outcome::Fail),
            ("d", "org/two", Outcome::Fail, Outcome::Pass),
        ] {
            let spec = TaskSpec {
                instance_id: id.into(), repo: repo.into(), stratum: "bugfix".into(),
                evaluation_order: "control_first".into(),
            };
            let costs = CostRow {
                instance_id: id.into(), control_cost_usd: 1.0, treatment_cost_usd: 1.0,
                control_receipt_sha256: "a".repeat(64), treatment_receipt_sha256: "b".repeat(64),
            };
            tasks.push(PairedTask { spec, control, treatment, costs });
        }
        assert_eq!(repo_cluster_bootstrap(&tasks, 1000, 42).unwrap(), repo_cluster_bootstrap(&tasks, 1000, 42).unwrap());
    }

    #[test]
    fn hash_validation_rejects_wrong_length_and_non_hex() {
        assert!(is_hex(&"a".repeat(64), 64));
        assert!(!is_hex(&"a".repeat(63), 64));
        assert!(!is_hex(&format!("{}z", "a".repeat(63)), 64));
    }
}
