use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};

const EXPECTED_EVALUATION_BLOB: &str = "af3b663429dc4689ee728d2fb5c7d8d5034476f7";
const RUN_SCHEMA: &str = "pitb.sorrydb-proof-run.v0";
const REPORT_SCHEMA: &str = "pitb.clustered-paired-comparison.v0";
const DEFAULT_REPLICATES: usize = 10_000;
const HARD_MAX_REPLICATES: usize = 200_000;

#[derive(Debug, Clone, Deserialize)]
struct Dataset {
    sorries: Vec<Task>,
}

#[derive(Debug, Clone, Deserialize)]
struct Task {
    id: String,
    repo: Repo,
    location: Location,
    debug_info: DebugInfo,
}

#[derive(Debug, Clone, Deserialize)]
struct Repo {
    remote: String,
    commit: String,
    lean_version: String,
}

#[derive(Debug, Clone, Deserialize)]
struct Location {
    path: String,
    start_line: u64,
    start_column: u64,
    end_line: u64,
    end_column: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct DebugInfo {
    goal: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct Strategy {
    name: String,
    version: String,
    implementation_sha256: String,
    model_id: Option<String>,
    model_version: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ProofRun {
    schema_version: String,
    run_id: String,
    task_id: String,
    repo_commit: String,
    lean_version: String,
    source_path: String,
    start_line: u64,
    start_column: u64,
    end_line: u64,
    end_column: u64,
    goal_sha256: String,
    declaration_signature_before_sha256: String,
    declaration_signature_after_sha256: String,
    target_is_prop: bool,
    source_file_sha256: String,
    replacement_patch_sha256: String,
    proof_artifact_sha256: String,
    checker_receipt_sha256: String,
    semantic_gate_receipt_sha256: String,
    source_binding_receipt_sha256: String,
    strategy: Strategy,
    outcome: String,
    proof_inserted: bool,
    exact_sorry_replaced: bool,
    sorry_absent_after_replacement: bool,
    target_sorry_ax_absent: bool,
    build_passed: bool,
    kernel_passed: bool,
    semantic_gate_passed: bool,
    source_binding_passed: bool,
    vacuity_gate_passed: bool,
    axiom_delta_zero: bool,
    new_sorry_delta_zero: bool,
    wall_time_ms: u64,
    input_tokens: u64,
    output_tokens: u64,
    cost_microusd: u64,
    failure_type: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct StrategyIdentity {
    strategy_id: String,
    strategy: Strategy,
    attempted_tasks: usize,
    retries: usize,
    unique_verified_tasks: usize,
    assurance_adjusted_pass_rate: f64,
    total_wall_time_ms: u64,
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_cost_microusd: u64,
    cost_per_qualified_task_microusd: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct Comparison {
    left_strategy_id: String,
    right_strategy_id: String,
    common_task_count: usize,
    left_unique_qualified_tasks: usize,
    right_unique_qualified_tasks: usize,
    left_assurance_adjusted_pass_rate: f64,
    right_assurance_adjusted_pass_rate: f64,
    paired_pass_rate_difference: f64,
    paired_pass_rate_difference_percentage_points: f64,
    repository_cluster_count: usize,
    source_file_cluster_count: usize,
    bootstrap_replicates: usize,
    bootstrap_seed: u64,
    bootstrap_ci_95_low: f64,
    bootstrap_ci_95_high: f64,
    bootstrap_probability_left_better: f64,
    left_cost_per_qualified_task_microusd: Option<f64>,
    right_cost_per_qualified_task_microusd: Option<f64>,
    interpretation: String,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    schema_version: String,
    evaluation_split_git_blob_sha1: String,
    evaluation_split_raw_sha256: String,
    proof_runs_jsonl_sha256: String,
    evaluation_task_count: usize,
    strategy_summaries: Vec<StrategyIdentity>,
    comparison: Comparison,
    sampling_design: String,
    statistical_boundary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct FileKey {
    repository: String,
    path: String,
}

#[derive(Debug, Clone)]
struct TaskOutcome {
    solved: bool,
    wall_time_ms: u64,
    input_tokens: u64,
    output_tokens: u64,
    cost_microusd: u64,
}

#[derive(Debug, Clone)]
struct AggregatedStrategy {
    id: String,
    strategy: Strategy,
    outcomes: BTreeMap<String, TaskOutcome>,
    attempt_count: usize,
    total_wall_time_ms: u64,
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_cost_microusd: u64,
}

#[derive(Debug, Clone)]
struct Cluster {
    repository: String,
    files: Vec<FileCluster>,
}

#[derive(Debug, Clone)]
struct FileCluster {
    task_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
struct Difference {
    left_solved: f64,
    right_solved: f64,
    task_weight: f64,
}

#[derive(Debug, Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed } }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next() % upper as u64) as usize
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn git_blob_sha1(bytes: &[u8]) -> Result<String, String> {
    let mut child = Command::new("git")
        .args(["hash-object", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("start git hash-object: {e}"))?;
    child.stdin.take().ok_or("git hash-object stdin missing")?
        .write_all(bytes).map_err(|e| format!("write Git blob bytes: {e}"))?;
    let output = child.wait_with_output().map_err(|e| format!("wait for git hash-object: {e}"))?;
    if !output.status.success() {
        return Err(format!("git hash-object failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    let value = String::from_utf8(output.stdout).map_err(|e| e.to_string())?.trim().to_owned();
    if value.len() != 40 || !value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Err("git hash-object returned malformed lowercase SHA-1".into());
    }
    Ok(value)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn parse_split(bytes: &[u8]) -> Result<BTreeMap<String, Task>, String> {
    let blob = git_blob_sha1(bytes)?;
    if blob != EXPECTED_EVALUATION_BLOB {
        return Err(format!("evaluation split Git blob mismatch: expected {EXPECTED_EVALUATION_BLOB}, got {blob}"));
    }
    let dataset: Dataset = serde_json::from_slice(bytes)
        .map_err(|e| format!("parse SorryDB evaluation split: {e}"))?;
    if dataset.sorries.len() != 1000 {
        return Err(format!("frozen SorryDB evaluation split must have 1,000 tasks, got {}", dataset.sorries.len()));
    }
    let mut tasks = BTreeMap::new();
    for task in dataset.sorries {
        if task.id.trim().is_empty() || task.repo.remote.trim().is_empty()
            || task.location.path.trim().is_empty()
            || task.repo.commit.len() != 40
            || !task.repo.commit.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            || task.location.start_line == 0
            || task.location.end_line == 0
            || (task.location.end_line, task.location.end_column) < (task.location.start_line, task.location.start_column)
        {
            return Err(format!("invalid source identity or coordinates for task {}", task.id));
        }
        if tasks.insert(task.id.clone(), task).is_some() {
            return Err("duplicate task ID in evaluation split".into());
        }
    }
    Ok(tasks)
}

fn strategy_id(strategy: &Strategy) -> Result<String, String> {
    let bytes = serde_json::to_vec(strategy).map_err(|e| format!("encode strategy identity: {e}"))?;
    Ok(sha256(&bytes))
}

fn proof_qualified(run: &ProofRun) -> bool {
    run.outcome == "verified_success"
        && run.proof_inserted
        && run.exact_sorry_replaced
        && run.sorry_absent_after_replacement
        && run.target_sorry_ax_absent
        && run.target_is_prop
        && run.build_passed
        && run.kernel_passed
        && run.semantic_gate_passed
        && run.source_binding_passed
        && run.vacuity_gate_passed
        && run.axiom_delta_zero
        && run.new_sorry_delta_zero
}

fn validate_run(run: &ProofRun, task: &Task) -> Result<(), String> {
    if run.schema_version != RUN_SCHEMA {
        return Err(format!("run {} has wrong schema version", run.run_id));
    }
    if run.run_id.trim().is_empty() || run.task_id != task.id
        || run.repo_commit != task.repo.commit
        || run.lean_version != task.repo.lean_version
        || run.source_path != task.location.path
        || run.start_line != task.location.start_line
        || run.start_column != task.location.start_column
        || run.end_line != task.location.end_line
        || run.end_column != task.location.end_column
    {
        return Err(format!("run {} does not bind the exact SorryDB source task", run.run_id));
    }
    for (name, value) in [
        ("goal_sha256", &run.goal_sha256),
        ("declaration_signature_before_sha256", &run.declaration_signature_before_sha256),
        ("declaration_signature_after_sha256", &run.declaration_signature_after_sha256),
        ("source_file_sha256", &run.source_file_sha256),
        ("replacement_patch_sha256", &run.replacement_patch_sha256),
        ("proof_artifact_sha256", &run.proof_artifact_sha256),
        ("checker_receipt_sha256", &run.checker_receipt_sha256),
        ("semantic_gate_receipt_sha256", &run.semantic_gate_receipt_sha256),
        ("source_binding_receipt_sha256", &run.source_binding_receipt_sha256),
        ("implementation_sha256", &run.strategy.implementation_sha256),
    ] {
        if !valid_sha256(value) {
            return Err(format!("run {} has malformed {name}", run.run_id));
        }
    }
    if run.goal_sha256 != sha256(task.debug_info.goal.as_bytes()) {
        return Err(format!("run {} goal digest differs from the exact SorryDB task", run.run_id));
    }
    if run.declaration_signature_before_sha256 != run.declaration_signature_after_sha256 {
        return Err(format!("run {} changes the exact target declaration signature", run.run_id));
    }
    for (label, value) in [
        ("strategy.name", &run.strategy.name),
        ("strategy.version", &run.strategy.version),
    ] {
        if value.trim().is_empty() {
            return Err(format!("run {} has empty {label}", run.run_id));
        }
    }
    if run.strategy.model_id.as_ref().is_some_and(|x| x.trim().is_empty())
        || run.strategy.model_version.as_ref().is_some_and(|x| x.trim().is_empty())
    {
        return Err(format!("run {} has an empty optional model identity", run.run_id));
    }
    if !matches!(run.outcome.as_str(), "verified_success" | "non_solve" | "infrastructure_failure" | "invalid_target") {
        return Err(format!("run {} has unknown outcome", run.run_id));
    }
    let all_gates = [
        run.proof_inserted,
        run.exact_sorry_replaced,
        run.sorry_absent_after_replacement,
        run.target_sorry_ax_absent,
        run.target_is_prop,
        run.build_passed,
        run.kernel_passed,
        run.semantic_gate_passed,
        run.source_binding_passed,
        run.vacuity_gate_passed,
        run.axiom_delta_zero,
        run.new_sorry_delta_zero,
    ];
    if run.outcome == "verified_success" && all_gates.iter().any(|passed| !passed) {
        return Err(format!("run {} claims verified_success without all assurance gates", run.run_id));
    }
    if run.outcome != "verified_success" && all_gates.iter().all(|passed| *passed) {
        return Err(format!("run {} passes every success gate but is labeled non-success", run.run_id));
    }
    if run.outcome == "infrastructure_failure"
        && run.failure_type.as_deref().unwrap_or("").trim().is_empty()
    {
        return Err(format!("run {} infrastructure failure requires failure_type", run.run_id));
    }
    Ok(())
}

fn validate_strategy_coverage(
    tasks: &BTreeMap<String, Task>,
    strategies: &BTreeMap<String, AggregatedStrategy>,
) -> Result<(), String> {
    for (id, strategy) in strategies {
        let expected: BTreeSet<&str> = tasks.keys().map(String::as_str).collect();
        let observed: BTreeSet<&str> = strategy.outcomes.keys().map(String::as_str).collect();
        if observed != expected {
            let missing = expected.difference(&observed).take(5).copied().collect::<Vec<_>>();
            let extra = observed.difference(&expected).take(5).copied().collect::<Vec<_>>();
            return Err(format!(
                "strategy {id} does not cover the exact frozen task set; missing={missing:?}, extra={extra:?}"
            ));
        }
    }
    Ok(())
}

fn validate_pair_coverage(
    tasks: &BTreeMap<String, Task>,
    strategies: &BTreeMap<String, AggregatedStrategy>,
    left_id: &str,
    right_id: &str,
) -> Result<(), String> {
    let left = strategies.get(left_id).ok_or("left strategy ID was not found")?;
    let right = strategies.get(right_id).ok_or("right strategy ID was not found")?;
    let pair = BTreeMap::from([
        (left_id.to_owned(), left.clone()),
        (right_id.to_owned(), right.clone()),
    ]);
    validate_strategy_coverage(tasks, &pair)
}

fn load_runs(bytes: &[u8], tasks: &BTreeMap<String, Task>) -> Result<BTreeMap<String, AggregatedStrategy>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("proof-runs JSONL is not UTF-8: {e}"))?;
    let mut strategies = BTreeMap::<String, AggregatedStrategy>::new();
    let mut seen_run_ids = BTreeSet::new();
    let mut task_contexts = BTreeMap::<String, (String, String, String)>::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() { continue; }
        let run: ProofRun = serde_json::from_str(line)
            .map_err(|e| format!("proof-runs line {}: {e}", index + 1))?;
        let task = tasks.get(&run.task_id)
            .ok_or_else(|| format!("proof-run {} references task outside frozen evaluation split", run.run_id))?;
        validate_run(&run, task)?;
        let context = (
            run.source_file_sha256.clone(),
            run.goal_sha256.clone(),
            run.declaration_signature_before_sha256.clone(),
        );
        if let Some(previous) = task_contexts.get(&run.task_id) {
            if previous != &context {
                return Err(format!(
                    "task {} source file, goal or declaration-signature identity differs across strategy runs",
                    run.task_id
                ));
            }
        } else {
            task_contexts.insert(run.task_id.clone(), context);
        }
        if !seen_run_ids.insert(run.run_id.clone()) {
            return Err(format!("duplicate proof run ID {}", run.run_id));
        }
        let id = strategy_id(&run.strategy)?;
        let solved = proof_qualified(&run);
        let entry = strategies.entry(id.clone()).or_insert_with(|| AggregatedStrategy {
            id,
            strategy: run.strategy.clone(),
            outcomes: BTreeMap::new(),
            attempt_count: 0,
            total_wall_time_ms: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_cost_microusd: 0,
        });
        if entry.strategy != run.strategy {
            return Err("strategy identity hash collision".into());
        }
        entry.attempt_count += 1;
        entry.total_wall_time_ms = entry.total_wall_time_ms.saturating_add(run.wall_time_ms);
        entry.total_input_tokens = entry.total_input_tokens.saturating_add(run.input_tokens);
        entry.total_output_tokens = entry.total_output_tokens.saturating_add(run.output_tokens);
        entry.total_cost_microusd = entry.total_cost_microusd.saturating_add(run.cost_microusd);
        let outcome = entry.outcomes.entry(run.task_id.clone()).or_insert(TaskOutcome {
            solved: false,
            wall_time_ms: 0,
            input_tokens: 0,
            output_tokens: 0,
            cost_microusd: 0,
        });
        outcome.solved |= solved;
        outcome.wall_time_ms = outcome.wall_time_ms.saturating_add(run.wall_time_ms);
        outcome.input_tokens = outcome.input_tokens.saturating_add(run.input_tokens);
        outcome.output_tokens = outcome.output_tokens.saturating_add(run.output_tokens);
        outcome.cost_microusd = outcome.cost_microusd.saturating_add(run.cost_microusd);
    }
    if strategies.len() < 2 {
        return Err("at least two distinct strategy identities are required for a paired comparison".into());
    }
    validate_strategy_coverage(tasks, &strategies)?;
    Ok(strategies)
}

fn build_clusters(tasks: &BTreeMap<String, Task>) -> Vec<Cluster> {
    let mut repos = BTreeMap::<String, BTreeMap<String, Vec<String>>>::new();
    for task in tasks.values() {
        repos.entry(task.repo.remote.clone())
            .or_default()
            .entry(task.location.path.clone())
            .or_default()
            .push(task.id.clone());
    }
    repos.into_iter().map(|(repository, files)| Cluster {
        repository,
        files: files.into_iter().map(|(_, mut ids)| {
            ids.sort();
            FileCluster { task_ids: ids }
        }).collect(),
    }).collect()
}

fn strategy_summary(strategy: &AggregatedStrategy, task_count: usize) -> StrategyIdentity {
    let unique = strategy.outcomes.values().filter(|x| x.solved).count();
    StrategyIdentity {
        strategy_id: strategy.id.clone(),
        strategy: strategy.strategy.clone(),
        attempted_tasks: strategy.outcomes.len(),
        retries: strategy.attempt_count.saturating_sub(strategy.outcomes.len()),
        unique_verified_tasks: unique,
        assurance_adjusted_pass_rate: unique as f64 / task_count as f64,
        total_wall_time_ms: strategy.total_wall_time_ms,
        total_input_tokens: strategy.total_input_tokens,
        total_output_tokens: strategy.total_output_tokens,
        total_cost_microusd: strategy.total_cost_microusd,
        cost_per_qualified_task_microusd: if unique > 0 {
            Some(strategy.total_cost_microusd as f64 / unique as f64)
        } else {
            None
        },
    }
}

fn paired_difference(
    clusters: &[Cluster],
    left: &AggregatedStrategy,
    right: &AggregatedStrategy,
    rng: &mut XorShift64,
) -> Result<f64, String> {
    if clusters.is_empty() {
        return Err("no repository clusters".into());
    }
    let mut left_solved = 0.0;
    let mut right_solved = 0.0;
    let mut weight = 0.0;
    // Two-stage paired cluster bootstrap:
    // 1) sample repositories with replacement;
    // 2) within each sampled repository, sample source files with replacement;
    // 3) retain all task rows inside each selected source-file cluster.
    for _ in 0..clusters.len() {
        let repo = &clusters[rng.index(clusters.len())];
        if repo.files.is_empty() {
            return Err(format!("repository cluster {} has no source files", repo.repository));
        }
        for _ in 0..repo.files.len() {
            let file = &repo.files[rng.index(repo.files.len())];
            for task_id in &file.task_ids {
                let l = left.outcomes.get(task_id)
                    .ok_or_else(|| format!("left strategy missing task {task_id}"))?;
                let r = right.outcomes.get(task_id)
                    .ok_or_else(|| format!("right strategy missing task {task_id}"))?;
                left_solved += f64::from(l.solved as u8);
                right_solved += f64::from(r.solved as u8);
                weight += 1.0;
            }
        }
    }
    if weight == 0.0 {
        return Err("bootstrap draw contains no tasks".into());
    }
    Ok((left_solved - right_solved) / weight)
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let position = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let alpha = position - lower as f64;
        sorted[lower] * (1.0 - alpha) + sorted[upper] * alpha
    }
}

fn compare(
    split_bytes: &[u8],
    runs_bytes: &[u8],
    tasks: &BTreeMap<String, Task>,
    strategies: &BTreeMap<String, AggregatedStrategy>,
    left_id: &str,
    right_id: &str,
    replicates: usize,
    seed: u64,
) -> Result<Report, String> {
    if replicates == 0 || replicates > HARD_MAX_REPLICATES {
        return Err(format!("bootstrap replicate count must be 1..={HARD_MAX_REPLICATES}"));
    }
    if left_id == right_id {
        return Err("left and right strategy IDs must differ".into());
    }
    let left = strategies.get(left_id).ok_or("left strategy ID was not found in proof-run JSONL")?;
    let right = strategies.get(right_id).ok_or("right strategy ID was not found in proof-run JSONL")?;
    validate_pair_coverage(tasks, strategies, left_id, right_id)?;
    let task_count = tasks.len();
    let left_unique = left.outcomes.values().filter(|x| x.solved).count();
    let right_unique = right.outcomes.values().filter(|x| x.solved).count();
    let left_rate = left_unique as f64 / task_count as f64;
    let right_rate = right_unique as f64 / task_count as f64;
    let clusters = build_clusters(tasks);
    let file_count: usize = clusters.iter().map(|r| r.files.len()).sum();

    let mut rng = XorShift64::new(seed);
    let mut samples = Vec::with_capacity(replicates);
    let mut wins = 0usize;
    for _ in 0..replicates {
        let delta = paired_difference(&clusters, left, right, &mut rng)?;
        if delta > 0.0 { wins += 1; }
        samples.push(delta);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let raw_delta = left_rate - right_rate;
    let comparison = Comparison {
        left_strategy_id: left.id.clone(),
        right_strategy_id: right.id.clone(),
        common_task_count: task_count,
        left_unique_qualified_tasks: left_unique,
        right_unique_qualified_tasks: right_unique,
        left_assurance_adjusted_pass_rate: left_rate,
        right_assurance_adjusted_pass_rate: right_rate,
        paired_pass_rate_difference: raw_delta,
        paired_pass_rate_difference_percentage_points: raw_delta * 100.0,
        repository_cluster_count: clusters.len(),
        source_file_cluster_count: file_count,
        bootstrap_replicates: replicates,
        bootstrap_seed: seed,
        bootstrap_ci_95_low: percentile(&samples, 0.025),
        bootstrap_ci_95_high: percentile(&samples, 0.975),
        bootstrap_probability_left_better: wins as f64 / replicates as f64,
        left_cost_per_qualified_task_microusd: if left_unique > 0 {
            Some(left.total_cost_microusd as f64 / left_unique as f64)
        } else { None },
        right_cost_per_qualified_task_microusd: if right_unique > 0 {
            Some(right.total_cost_microusd as f64 / right_unique as f64)
        } else { None },
        interpretation: "Paired two-stage cluster bootstrap: resample repositories, then source files within repositories, retaining all task outcomes within selected files. This addresses repository/file clustering but does not remove model/scaffold/budget confounding; those must be matched by experimental design.".into(),
    };

    let summaries = strategies.values().map(|s| strategy_summary(s, task_count)).collect();
    Ok(Report {
        schema_version: REPORT_SCHEMA.into(),
        evaluation_split_git_blob_sha1: EXPECTED_EVALUATION_BLOB.into(),
        evaluation_split_raw_sha256: sha256(split_bytes),
        proof_runs_jsonl_sha256: sha256(runs_bytes),
        evaluation_task_count: task_count,
        strategy_summaries: summaries,
        comparison,
        sampling_design: "paired hierarchical bootstrap over repository clusters and source-file clusters nested within each selected repository; all tasks in a sampled file are retained together".into(),
        statistical_boundary: "Descriptive paired estimate under the supplied fixed task snapshot and receipts. Use matched model, agent scaffold, source access, Lean/tool versions, wall/CPU/GPU/token budgets and stopping policy before interpreting the difference causally. The hierarchical bootstrap is not a substitute for repeated independent model runs.".into(),
    })
}

fn list_strategies(strategies: &BTreeMap<String, AggregatedStrategy>) -> Result<String, String> {
    let rows: Vec<_> = strategies.values().map(|s| {
        serde_json::json!({
            "strategy_id": s.id,
            "name": s.strategy.name,
            "version": s.strategy.version,
            "model_id": s.strategy.model_id,
            "model_version": s.strategy.model_version,
            "implementation_sha256": s.strategy.implementation_sha256,
            "unique_task_count": s.outcomes.len(),
            "attempt_count": s.attempt_count
        })
    }).collect();
    serde_json::to_string_pretty(&rows).map_err(|e| format!("serialize strategy list: {e}"))
}

fn fake_hash(label: &str) -> String {
    sha256(label.as_bytes())
}

fn synthetic_task(id: &str, repository: &str, path: &str, commit: &str) -> Task {
    Task {
        id: id.into(),
        repo: Repo { remote: repository.into(), commit: commit.into(), lean_version: "v4.26.0".into() },
        location: Location { path: path.into(), start_line: 10, start_column: 0, end_line: 10, end_column: 5 },
        debug_info: DebugInfo { goal: "⊢ True".into() },
    }
}

fn synthetic_run(task: &Task, strategy: &Strategy, run_id: &str, solved: bool) -> ProofRun {
    let h = fake_hash(&format!("{}:{}", strategy.name, task.id));
    ProofRun {
        schema_version: RUN_SCHEMA.into(),
        run_id: run_id.into(),
        task_id: task.id.clone(),
        repo_commit: task.repo.commit.clone(),
        lean_version: task.repo.lean_version.clone(),
        source_path: task.location.path.clone(),
        start_line: task.location.start_line,
        start_column: task.location.start_column,
        end_line: task.location.end_line,
        end_column: task.location.end_column,
        goal_sha256: sha256(task.debug_info.goal.as_bytes()),
        declaration_signature_before_sha256: fake_hash(&format!("before:{}", task.id)),
        declaration_signature_after_sha256: fake_hash(&format!("after:{}", task.id)),
        target_is_prop: true,
        source_file_sha256: fake_hash(&format!("source:{}", task.id)),
        replacement_patch_sha256: fake_hash(&format!("patch:{h}")),
        proof_artifact_sha256: fake_hash(&format!("proof:{h}")),
        checker_receipt_sha256: fake_hash(&format!("checker:{h}")),
        semantic_gate_receipt_sha256: fake_hash(&format!("semantic:{h}")),
        source_binding_receipt_sha256: fake_hash(&format!("source-binding:{h}")),
        strategy: strategy.clone(),
        outcome: if solved { "verified_success" } else { "non_solve" }.into(),
        proof_inserted: solved,
        exact_sorry_replaced: solved,
        sorry_absent_after_replacement: solved,
        target_sorry_ax_absent: solved,
        build_passed: solved,
        kernel_passed: solved,
        semantic_gate_passed: solved,
        source_binding_passed: solved,
        vacuity_gate_passed: solved,
        axiom_delta_zero: solved,
        new_sorry_delta_zero: solved,
        wall_time_ms: 100,
        input_tokens: 200,
        output_tokens: 30,
        cost_microusd: 100,
        failure_type: None,
    }
}

fn run_self_tests() -> Result<(), String> {
    let tasks = vec![
        synthetic_task("t1", "https://github.com/a/repo1", "A.lean", &"1".repeat(40)),
        synthetic_task("t2", "https://github.com/a/repo1", "A.lean", &"1".repeat(40)),
        synthetic_task("t3", "https://github.com/a/repo1", "B.lean", &"1".repeat(40)),
        synthetic_task("t4", "https://github.com/b/repo2", "C.lean", &"2".repeat(40)),
        synthetic_task("t5", "https://github.com/b/repo2", "C.lean", &"2".repeat(40)),
    ];
    let task_map: BTreeMap<String, Task> = tasks.iter().cloned().map(|t| (t.id.clone(), t)).collect();
    let left = Strategy {
        name: "left".into(), version: "v1".into(), implementation_sha256: fake_hash("left"),
        model_id: Some("model-left".into()), model_version: Some("2026-11".into()),
    };
    let right = Strategy {
        name: "right".into(), version: "v1".into(), implementation_sha256: fake_hash("right"),
        model_id: Some("model-right".into()), model_version: Some("2026-11".into()),
    };
    let mut runs = Vec::new();
    for (i, task) in tasks.iter().enumerate() {
        runs.push(synthetic_run(task, &left, &format!("l-{i}"), i < 3));
        runs.push(synthetic_run(task, &right, &format!("r-{i}"), i < 2));
    }
    // Add a retry for an already-solved task: task-level unique count must not change.
    runs.push(synthetic_run(&tasks[0], &left, "l-retry", true));
    let mut strategies = BTreeMap::new();
    for run in &runs {
        let id = strategy_id(&run.strategy)?;
        let entry = strategies.entry(id.clone()).or_insert_with(|| AggregatedStrategy {
            id, strategy: run.strategy.clone(), outcomes: BTreeMap::new(),
            attempt_count: 0, total_wall_time_ms: 0, total_input_tokens: 0,
            total_output_tokens: 0, total_cost_microusd: 0,
        });
        entry.attempt_count += 1;
        entry.total_wall_time_ms = entry.total_wall_time_ms.saturating_add(run.wall_time_ms);
        entry.total_input_tokens = entry.total_input_tokens.saturating_add(run.input_tokens);
        entry.total_output_tokens = entry.total_output_tokens.saturating_add(run.output_tokens);
        entry.total_cost_microusd = entry.total_cost_microusd.saturating_add(run.cost_microusd);
        let outcome = entry.outcomes.entry(run.task_id.clone()).or_insert(TaskOutcome {
            solved: false, wall_time_ms: 0, input_tokens: 0, output_tokens: 0, cost_microusd: 0,
        });
        outcome.solved |= proof_qualified(run);
        outcome.wall_time_ms += run.wall_time_ms;
        outcome.input_tokens += run.input_tokens;
        outcome.output_tokens += run.output_tokens;
        outcome.cost_microusd += run.cost_microusd;
    }
    let left_id = strategy_id(&left)?;
    let right_id = strategy_id(&right)?;
    let left_summary = strategy_summary(&strategies[&left_id], tasks.len());
    if left_summary.unique_verified_tasks != 3 || left_summary.retries != 1 {
        return Err("retry aggregation or unique-task denominator control failed".into());
    }
    println!("SELF_TEST_PASS retries collapse to unique task outcomes");

    let split_bytes = serde_json::to_vec(&serde_json::json!({"sorries": tasks})).map_err(|e| e.to_string())?;
    let runs_bytes = runs.iter().map(serde_json::to_string).collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?.join("\n");
    let clusters = build_clusters(&task_map);
    if clusters.len() != 2 || clusters.iter().map(|r| r.files.len()).sum::<usize>() != 3 {
        return Err("repository/file nested cluster construction failed".into());
    }
    println!("SELF_TEST_PASS nested repository/file clusters preserve source grouping");

    let report = compare(&split_bytes, runs_bytes.as_bytes(), &task_map, &strategies, &left_id, &right_id, 500, 12345)?;
    if report.comparison.left_unique_qualified_tasks != 3
        || report.comparison.right_unique_qualified_tasks != 2
        || (report.comparison.paired_pass_rate_difference - 0.2).abs() > 1e-12
        || report.comparison.bootstrap_replicates != 500
    {
        return Err("paired assurance-adjusted comparison control failed".into());
    }
    let again = compare(&split_bytes, runs_bytes.as_bytes(), &task_map, &strategies, &left_id, &right_id, 500, 12345)?;
    if report.comparison.bootstrap_ci_95_low != again.comparison.bootstrap_ci_95_low
        || report.comparison.bootstrap_ci_95_high != again.comparison.bootstrap_ci_95_high
    {
        return Err("fixed-seed bootstrap is not deterministic".into());
    }
    println!("SELF_TEST_PASS paired assurance-adjusted delta and deterministic cluster bootstrap");

    let mut bad = runs.clone();
    bad[0].source_path = "different.lean".into();
    if validate_run(&bad[0], &tasks[0]).is_ok() {
        return Err("mismatched task source identity was accepted".into());
    }
    println!("SELF_TEST_PASS all paired runs bind exact source task identity");

    let mut bad = runs.clone();
    bad[0].kernel_passed = false;
    if proof_qualified(&bad[0]) {
        return Err("kernel-failing run counted as assurance-qualified".into());
    }
    println!("SELF_TEST_PASS all proof and semantic gates are required");

    Ok(())
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        run_self_tests()?;
        println!("PITB_CLUSTERED_STATS_SELF_TEST_PASS");
        return Ok(());
    }
    if args.len() == 2 && args[1] == "--self-test-report" {
        return Err("--self-test-report requires an output path".into());
    }
    if args.len() == 3 && args[1] == "--self-test-report" {
        run_self_tests()?;
        let (tasks, strategies, runs_bytes, left_id, right_id) = synthetic_inputs()?;
        let split_bytes = serde_json::to_vec(&serde_json::json!({"sorries": tasks.values().cloned().collect::<Vec<_>>()}))
            .map_err(|e| e.to_string())?;
        let report = compare(&split_bytes, runs_bytes.as_bytes(), &tasks, &strategies, &left_id, &right_id, 500, 12345)?;
        let bytes = serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?;
        fs::write(&args[2], bytes).map_err(|e| format!("write report: {e}"))?;
        println!("PITB_CLUSTERED_STATS_SYNTHETIC_REPORT_OK");
        return Ok(());
    }
    if args.len() == 2 && args[1] == "--list-strategies" {
        return Err("--list-strategies requires <evaluation-split.json> <proof-runs.jsonl>".into());
    }
    if args.len() == 4 && args[1] == "--list-strategies" {
        let split_bytes = fs::read(&args[2]).map_err(|e| format!("read split: {e}"))?;
        let tasks = parse_split(&split_bytes)?;
        let runs_bytes = fs::read(&args[3]).map_err(|e| format!("read proof runs: {e}"))?;
        let strategies = load_runs(&runs_bytes, &tasks)?;
        println!("{}", list_strategies(&strategies)?);
        return Ok(());
    }
    if args.len() != 9 || args[1] != "--compare" {
        return Err(format!(
            "usage: {} --compare <evaluation-split.json> <proof-runs.jsonl> <left-strategy-id> <right-strategy-id> <report.json> <replicates> <seed>; OR --list-strategies <split> <runs>; OR --self-test",
            args.first().map(String::as_str).unwrap_or("pitb-clustered-benchmark-stats")
        ));
    }
    let split_path = PathBuf::from(&args[2]);
    let runs_path = PathBuf::from(&args[3]);
    let report_path = PathBuf::from(&args[6]);
    let replicates: usize = args[7].parse().map_err(|_| "replicates must be an integer")?;
    let seed: u64 = args[8].parse().map_err(|_| "seed must be an integer")?;
    let split_bytes = fs::read(&split_path).map_err(|e| format!("read evaluation split: {e}"))?;
    let tasks = parse_split(&split_bytes)?;
    let runs_bytes = fs::read(&runs_path).map_err(|e| format!("read proof runs: {e}"))?;
    let strategies = load_runs(&runs_bytes, &tasks)?;
    let report = compare(&split_bytes, &runs_bytes, &tasks, &strategies, &args[4], &args[5], replicates, seed)?;
    let bytes = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize comparison report: {e}"))?;
    fs::write(report_path, bytes).map_err(|e| format!("write comparison report: {e}"))?;
    println!("PITB_CLUSTERED_PAIRED_COMPARISON_OK");
    Ok(())
}

fn synthetic_inputs() -> Result<(
    BTreeMap<String, Task>,
    BTreeMap<String, AggregatedStrategy>,
    String,
    String,
    String,
), String> {
    let tasks_vec = vec![
        synthetic_task("t1", "https://github.com/a/repo1", "A.lean", &"1".repeat(40)),
        synthetic_task("t2", "https://github.com/a/repo1", "A.lean", &"1".repeat(40)),
        synthetic_task("t3", "https://github.com/a/repo1", "B.lean", &"1".repeat(40)),
        synthetic_task("t4", "https://github.com/b/repo2", "C.lean", &"2".repeat(40)),
        synthetic_task("t5", "https://github.com/b/repo2", "C.lean", &"2".repeat(40)),
    ];
    let tasks: BTreeMap<String, Task> = tasks_vec.iter().cloned().map(|t| (t.id.clone(), t)).collect();
    let left = Strategy {
        name: "left".into(), version: "v1".into(), implementation_sha256: fake_hash("left"),
        model_id: Some("model-left".into()), model_version: Some("2026-11".into()),
    };
    let right = Strategy {
        name: "right".into(), version: "v1".into(), implementation_sha256: fake_hash("right"),
        model_id: Some("model-right".into()), model_version: Some("2026-11".into()),
    };
    let mut runs = Vec::new();
    for (i, task) in tasks_vec.iter().enumerate() {
        runs.push(synthetic_run(task, &left, &format!("l-{i}"), i < 3));
        runs.push(synthetic_run(task, &right, &format!("r-{i}"), i < 2));
    }
    runs.push(synthetic_run(&tasks_vec[0], &left, "l-retry", true));
    let mut strategies = BTreeMap::new();
    for run in &runs {
        let id = strategy_id(&run.strategy)?;
        let entry = strategies.entry(id.clone()).or_insert_with(|| AggregatedStrategy {
            id, strategy: run.strategy.clone(), outcomes: BTreeMap::new(),
            attempt_count: 0, total_wall_time_ms: 0, total_input_tokens: 0,
            total_output_tokens: 0, total_cost_microusd: 0,
        });
        entry.attempt_count += 1;
        let outcome = entry.outcomes.entry(run.task_id.clone()).or_insert(TaskOutcome {
            solved: false, wall_time_ms: 0, input_tokens: 0, output_tokens: 0, cost_microusd: 0,
        });
        outcome.solved |= proof_qualified(run);
        outcome.wall_time_ms += run.wall_time_ms;
        outcome.input_tokens += run.input_tokens;
        outcome.output_tokens += run.output_tokens;
        outcome.cost_microusd += run.cost_microusd;
    }
    let runs_bytes = runs.iter().map(serde_json::to_string).collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?.join("\n");
    Ok((tasks, strategies, runs_bytes, strategy_id(&left)?, strategy_id(&right)?))
}

fn main() -> std::process::ExitCode {
    match execute() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("INVALID: {message}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_clustered_statistics_controls_pass() {
        run_self_tests().expect("all clustered statistics controls must pass");
    }

    #[test]
    fn incomplete_strategy_task_coverage_is_rejected() {
        let tasks = vec![
            synthetic_task("t1", "https://github.com/a/r", "A.lean", &"1".repeat(40)),
            synthetic_task("t2", "https://github.com/a/r", "B.lean", &"1".repeat(40)),
        ];
        let left = Strategy { name:"left".into(),version:"v1".into(),implementation_sha256:fake_hash("l"),model_id:None,model_version:None };
        let right = Strategy { name:"right".into(),version:"v1".into(),implementation_sha256:fake_hash("r"),model_id:None,model_version:None };
        let mut strategies = BTreeMap::new();
        for (strategy, count) in [(left, 2usize), (right, 1usize)] {
            let id = strategy_id(&strategy).unwrap();
            let mut outcomes = BTreeMap::new();
            for task in tasks.iter().take(count) {
                outcomes.insert(task.id.clone(), TaskOutcome { solved:false,wall_time_ms:0,input_tokens:0,output_tokens:0,cost_microusd:0 });
            }
            strategies.insert(id.clone(), AggregatedStrategy {
                id,strategy,outcomes,attempt_count:count,total_wall_time_ms:0,
                total_input_tokens:0,total_output_tokens:0,total_cost_microusd:0
            });
        }
        let map: BTreeMap<String,Task> = tasks.into_iter().map(|t|(t.id.clone(),t)).collect();
        assert!(validate_strategy_coverage(&map, &strategies).is_err());
    }
}
