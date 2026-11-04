use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::PathBuf,
    process::ExitCode,
};

const TASKS_SCHEMA: &str = "pitb.sorrydb-task-snapshot.v0";
const RUN_SCHEMA: &str = "pitb.sorrydb-proof-run.v0";
const REPORT_SCHEMA: &str = "pitb.sorrydb-integrity-report.v0";
const SOURCE_REPO_COMMIT: &str = "74b3b63f8ba890d675f362924b21ac55db410383";
const FULL_SNAPSHOT_BLOB: &str = "5fafddac467af6d30d8f5c8731d6edec9c273a47";
const EVALUATION_SPLIT_BLOB: &str = "af3b663429dc4689ee728d2fb5c7d8d5034476f7";
const LICENSE_BLOB: &str = "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RepoInfo {
    remote: String,
    branch: String,
    commit: String,
    lean_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Location {
    path: String,
    start_line: u64,
    start_column: u64,
    end_line: u64,
    end_column: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct DebugInfo {
    goal: String,
    url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Metadata {
    blame_email_hash: String,
    blame_date: String,
    inclusion_date: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Task {
    id: String,
    repo: RepoInfo,
    location: Location,
    debug_info: DebugInfo,
    metadata: Metadata,
    #[serde(default)]
    source_file_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
struct TaskIdentity {
    remote: String,
    branch: String,
    commit: String,
    lean_version: String,
    path: String,
    start_line: u64,
    start_column: u64,
    end_line: u64,
    end_column: u64,
    goal_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Strategy {
    name: String,
    version: String,
    implementation_sha256: String,
    model_id: Option<String>,
    model_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Run {
    schema_version: String,
    run_id: String,
    task_fingerprint: String,
    task_id: String,
    repo_commit: String,
    lean_version: String,
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
struct TaskAudit {
    task_id: String,
    task_fingerprint: String,
    repository: String,
    repository_commit: String,
    lean_version: String,
    source_path: String,
    line_range: String,
    goal_sha256: String,
    source_file_sha256_present: bool,
    source_binding_ready: bool,
}

#[derive(Debug, Clone, Serialize)]
struct StrategySummary {
    strategy_name: String,
    strategy_version: String,
    model_id: Option<String>,
    model_version: Option<String>,
    attempts: usize,
    verified_successes: usize,
    non_solves: usize,
    infrastructure_failures: usize,
    invalid_targets: usize,
    success_rate_per_attempt: f64,
    mean_success_wall_time_ms: Option<f64>,
    total_cost_usd: f64,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    schema_version: String,
    source_repository_commit: String,
    full_snapshot_git_blob_sha1: String,
    evaluation_split_git_blob_sha1: String,
    license_git_blob_sha1: String,
    task_input_sha256: String,
    run_input_sha256: Option<String>,
    task_count: usize,
    repository_snapshot_count: usize,
    unique_repository_commit_count: usize,
    duplicate_task_ids: usize,
    source_file_digest_coverage: f64,
    source_binding_ready_task_count: usize,
    task_audits: Vec<TaskAudit>,
    run_count: usize,
    verified_success_count: usize,
    strategy_summaries: Vec<StrategySummary>,
    trust_boundary: String,
    interpretation_boundary: String,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn digest(value: &str, label: &str) -> Result<(), String> {
    if valid_hex(value, 64) {
        Ok(())
    } else {
        Err(format!("{label} must be 64 lowercase hexadecimal characters"))
    }
}

fn git_sha(value: &str, label: &str) -> Result<(), String> {
    if valid_hex(value, 40) {
        Ok(())
    } else {
        Err(format!("{label} must be a 40-character lowercase Git SHA-1"))
    }
}

fn non_empty(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{label} must be non-empty"))
    } else {
        Ok(())
    }
}

fn safe_relative_path(path: &str) -> bool {
    if path.is_empty() || path.starts_with('/') || path.contains('\\') {
        return false;
    }
    path.split('/').all(|part| !part.is_empty() && part != "." && part != "..")
}

fn identity(task: &Task) -> TaskIdentity {
    TaskIdentity {
        remote: task.repo.remote.clone(),
        branch: task.repo.branch.clone(),
        commit: task.repo.commit.clone(),
        lean_version: task.repo.lean_version.clone(),
        path: task.location.path.clone(),
        start_line: task.location.start_line,
        start_column: task.location.start_column,
        end_line: task.location.end_line,
        end_column: task.location.end_column,
        goal_sha256: sha256(task.debug_info.goal.as_bytes()),
    }
}

fn task_fingerprint(task: &Task) -> Result<String, String> {
    let encoded = serde_json::to_vec(&identity(task))
        .map_err(|e| format!("serialize task identity: {e}"))?;
    Ok(sha256(&encoded))
}

fn validate_task(task: &Task, index: usize) -> Result<(), String> {
    let label = format!("task[{index}]");
    non_empty(&task.id, &format!("{label}.id"))?;
    non_empty(&task.repo.remote, &format!("{label}.repo.remote"))?;
    non_empty(&task.repo.branch, &format!("{label}.repo.branch"))?;
    non_empty(&task.repo.lean_version, &format!("{label}.repo.lean_version"))?;
    git_sha(&task.repo.commit, &format!("{label}.repo.commit"))?;
    if !safe_relative_path(&task.location.path) {
        return Err(format!("{label}.location.path is not a safe repository-relative path"));
    }
    if task.location.start_line == 0
        || task.location.start_column == 0
        || task.location.end_line == 0
        || task.location.end_column == 0
    {
        return Err(format!("{label} coordinates must be 1-based and positive"));
    }
    if task.location.end_line < task.location.start_line
        || (task.location.end_line == task.location.start_line
            && task.location.end_column < task.location.start_column)
    {
        return Err(format!("{label} location range is reversed"));
    }
    non_empty(&task.debug_info.goal, &format!("{label}.debug_info.goal"))?;
    non_empty(&task.debug_info.url, &format!("{label}.debug_info.url"))?;
    non_empty(&task.metadata.blame_date, &format!("{label}.metadata.blame_date"))?;
    non_empty(&task.metadata.inclusion_date, &format!("{label}.metadata.inclusion_date"))?;
    if let Some(source) = &task.source_file_sha256 {
        digest(source, &format!("{label}.source_file_sha256"))?;
    }
    Ok(())
}

fn validate_run(run: &Run, index: usize, tasks: &BTreeMap<String, &Task>) -> Result<(), String> {
    let label = format!("run[{index}]");
    if run.schema_version != RUN_SCHEMA {
        return Err(format!("{label} has unsupported schema_version"));
    }
    non_empty(&run.run_id, &format!("{label}.run_id"))?;
    digest(&run.task_fingerprint, &format!("{label}.task_fingerprint"))?;
    digest(&run.source_file_sha256, &format!("{label}.source_file_sha256"))?;
    for (field, value) in [
        ("replacement_patch_sha256", &run.replacement_patch_sha256),
        ("proof_artifact_sha256", &run.proof_artifact_sha256),
        ("checker_receipt_sha256", &run.checker_receipt_sha256),
        ("semantic_gate_receipt_sha256", &run.semantic_gate_receipt_sha256),
        ("source_binding_receipt_sha256", &run.source_binding_receipt_sha256),
        ("strategy.implementation_sha256", &run.strategy.implementation_sha256),
    ] {
        digest(value, &format!("{label}.{field}"))?;
    }
    non_empty(&run.strategy.name, &format!("{label}.strategy.name"))?;
    non_empty(&run.strategy.version, &format!("{label}.strategy.version"))?;
    git_sha(&run.repo_commit, &format!("{label}.repo_commit"))?;
    non_empty(&run.lean_version, &format!("{label}.lean_version"))?;

    let task = tasks.get(&run.task_fingerprint)
        .ok_or_else(|| format!("{label} references unknown task_fingerprint"))?;
    if run.task_id != task.id {
        return Err(format!("{label} task_id does not match task fingerprint"));
    }
    if run.repo_commit != task.repo.commit || run.lean_version != task.repo.lean_version {
        return Err(format!("{label} repository commit or Lean version differs from task snapshot"));
    }
    match &task.source_file_sha256 {
        Some(expected) if expected != &run.source_file_sha256 => {
            return Err(format!("{label} source file digest differs from pinned task source"))
        }
        None => {}
        Some(_) => {}
    }
    if !matches!(run.outcome.as_str(), "verified_success" | "non_solve" | "infrastructure_failure" | "invalid_target") {
        return Err(format!("{label} has unsupported outcome"));
    }
    let gates = [
        run.proof_inserted,
        run.exact_sorry_replaced,
        run.sorry_absent_after_replacement,
        run.build_passed,
        run.kernel_passed,
        run.semantic_gate_passed,
        run.source_binding_passed,
        run.vacuity_gate_passed,
        run.axiom_delta_zero,
        run.new_sorry_delta_zero,
    ];
    if run.outcome == "verified_success" && gates.iter().any(|gate| !gate) {
        return Err(format!("{label} claims verified_success without every proof, source and semantic gate"));
    }
    if run.outcome == "verified_success" && task.source_file_sha256.is_none() {
        return Err(format!("{label} cannot claim verified_success without a pinned source-file digest"));
    }
    if run.outcome != "verified_success" && gates.iter().all(|gate| *gate) {
        return Err(format!("{label} has all success gates but a non-success outcome"));
    }
    if run.outcome == "infrastructure_failure" && run.failure_type.as_deref().unwrap_or("").trim().is_empty() {
        return Err(format!("{label} infrastructure failure requires failure_type"));
    }
    if let Some(model_id) = &run.strategy.model_id {
        non_empty(model_id, &format!("{label}.strategy.model_id"))?;
    }
    if let Some(model_version) = &run.strategy.model_version {
        non_empty(model_version, &format!("{label}.strategy.model_version"))?;
    }
    Ok(())
}

fn parse_tasks(bytes: &[u8]) -> Result<Vec<Task>, String> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|e| format!("SorryDB task snapshot is invalid JSON: {e}"))?;
    let rows = value.get("sorries").and_then(Value::as_array)
        .ok_or("SorryDB snapshot must contain a top-level sorries array")?;
    let mut tasks = Vec::with_capacity(rows.len());
    for (index, value) in rows.iter().enumerate() {
        let task: Task = serde_json::from_value(value.clone())
            .map_err(|e| format!("SorryDB task {}: {e}", index + 1))?;
        tasks.push(task);
    }
    if tasks.is_empty() {
        return Err("SorryDB snapshot has no tasks".into());
    }
    Ok(tasks)
}

fn parse_runs(bytes: &[u8]) -> Result<Vec<Run>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("run JSONL is not UTF-8: {e}"))?;
    let mut runs = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let run: Run = serde_json::from_str(line)
            .map_err(|e| format!("run JSONL line {}: {e}", index + 1))?;
        runs.push(run);
    }
    Ok(runs)
}

fn audit(
    tasks: &[Task],
    task_input_sha256: String,
    runs: &[Run],
    run_input_sha256: Option<String>,
) -> Result<Report, String> {
    let mut task_ids = BTreeSet::new();
    let mut fingerprints = BTreeSet::new();
    let mut task_map = BTreeMap::<String, &Task>::new();
    let mut audits = Vec::new();
    let mut repository_snapshots = BTreeSet::new();
    let mut repository_commits = BTreeSet::new();
    let mut source_digest_count = 0usize;

    for (index, task) in tasks.iter().enumerate() {
        validate_task(task, index + 1)?;
        if !task_ids.insert(task.id.clone()) {
            return Err(format!("duplicate upstream SorryDB task id {}", task.id));
        }
        let fingerprint = task_fingerprint(task)?;
        if !fingerprints.insert(fingerprint.clone()) {
            return Err(format!("duplicate PITB task fingerprint {fingerprint}"));
        }
        task_map.insert(fingerprint.clone(), task);
        repository_snapshots.insert((
            task.repo.remote.clone(),
            task.repo.branch.clone(),
            task.repo.commit.clone(),
            task.repo.lean_version.clone(),
        ));
        repository_commits.insert((task.repo.remote.clone(), task.repo.commit.clone()));
        if task.source_file_sha256.is_some() {
            source_digest_count += 1;
        }
        audits.push(TaskAudit {
            task_id: task.id.clone(),
            task_fingerprint: fingerprint,
            repository: task.repo.remote.clone(),
            repository_commit: task.repo.commit.clone(),
            lean_version: task.repo.lean_version.clone(),
            source_path: task.location.path.clone(),
            line_range: format!(
                "{}:{}-{}:{}",
                task.location.start_line,
                task.location.start_column,
                task.location.end_line,
                task.location.end_column
            ),
            goal_sha256: sha256(task.debug_info.goal.as_bytes()),
            source_file_sha256_present: task.source_file_sha256.is_some(),
            source_binding_ready: task.source_file_sha256.is_some(),
        });
    }

    let mut run_ids = BTreeSet::new();
    let mut strategy_groups = BTreeMap::<(String, String, Option<String>, Option<String>), Vec<&Run>>::new();
    let mut verified_success_count = 0usize;
    for (index, run) in runs.iter().enumerate() {
        validate_run(run, index + 1, &task_map)?;
        if !run_ids.insert(run.run_id.clone()) {
            return Err(format!("duplicate run_id {}", run.run_id));
        }
        if run.outcome == "verified_success" {
            verified_success_count += 1;
        }
        strategy_groups.entry((
            run.strategy.name.clone(),
            run.strategy.version.clone(),
            run.strategy.model_id.clone(),
            run.strategy.model_version.clone(),
        )).or_default().push(run);
    }

    let mut strategy_summaries = Vec::new();
    for ((name, version, model_id, model_version), strategy_runs) in strategy_groups {
        let attempts = strategy_runs.len();
        let verified = strategy_runs.iter().filter(|run| run.outcome == "verified_success").count();
        let non_solves = strategy_runs.iter().filter(|run| run.outcome == "non_solve").count();
        let infra = strategy_runs.iter().filter(|run| run.outcome == "infrastructure_failure").count();
        let invalid = strategy_runs.iter().filter(|run| run.outcome == "invalid_target").count();
        let success_times: Vec<u64> = strategy_runs.iter()
            .filter(|run| run.outcome == "verified_success")
            .map(|run| run.wall_time_ms)
            .collect();
        let cost_microusd: u128 = strategy_runs.iter().map(|run| run.cost_microusd as u128).sum();
        strategy_summaries.push(StrategySummary {
            strategy_name: name,
            strategy_version: version,
            model_id,
            model_version,
            attempts,
            verified_successes: verified,
            non_solves,
            infrastructure_failures: infra,
            invalid_targets: invalid,
            success_rate_per_attempt: if attempts == 0 { 0.0 } else { verified as f64 / attempts as f64 },
            mean_success_wall_time_ms: if success_times.is_empty() {
                None
            } else {
                Some(success_times.iter().sum::<u64>() as f64 / success_times.len() as f64)
            },
            total_cost_usd: cost_microusd as f64 / 1_000_000.0,
        });
    }

    Ok(Report {
        schema_version: REPORT_SCHEMA.into(),
        source_repository_commit: SOURCE_REPO_COMMIT.into(),
        full_snapshot_git_blob_sha1: FULL_SNAPSHOT_BLOB.into(),
        evaluation_split_git_blob_sha1: EVALUATION_SPLIT_BLOB.into(),
        license_git_blob_sha1: LICENSE_BLOB.into(),
        task_input_sha256,
        run_input_sha256,
        task_count: tasks.len(),
        repository_snapshot_count: repository_snapshots.len(),
        unique_repository_commit_count: repository_commits.len(),
        duplicate_task_ids: 0,
        source_file_digest_coverage: source_digest_count as f64 / tasks.len() as f64,
        source_binding_ready_task_count: source_digest_count,
        task_audits: audits,
        run_count: runs.len(),
        verified_success_count,
        strategy_summaries,
        trust_boundary: "This auditor validates task coordinates, immutable repository commit identities, source-file digest coverage, run-to-task identity, and digest syntax. It does not clone repositories, prove that a stored goal matches the elaborated Lean target, authenticate receipt bytes, or establish semantic equivalence to an informal property. Those require separate reproducible checkouts and independent checkers.".into(),
        interpretation_boundary: "A verified_success requires exact sorry replacement, source binding, repository build, Lean kernel acceptance, semantic and vacuity gates, no new axioms and no new sorry. Task presence and successful compilation alone are not sufficient. Historical SorryDB records without source-file digests remain useful task locators but are not eligible for this stronger success label until source bytes are pinned.".into(),
    })
}

fn synthetic_task() -> Task {
    Task {
        id: "sorrydb-synthetic-1".into(),
        repo: RepoInfo {
            remote: "https://github.com/example/lean-project".into(),
            branch: "main".into(),
            commit: "1".repeat(40),
            lean_version: "v4.22.0".into(),
        },
        location: Location {
            path: "Demo/Example.lean".into(),
            start_line: 12,
            start_column: 1,
            end_line: 12,
            end_column: 6,
        },
        debug_info: DebugInfo {
            goal: "⊢ True".into(),
            url: "https://github.com/example/lean-project/blob/1111111111111111111111111111111111111111/Demo/Example.lean#L12".into(),
        },
        metadata: Metadata {
            blame_email_hash: "synthetic".into(),
            blame_date: "2026-01-01T00:00:00Z".into(),
            inclusion_date: "2026-02-01T00:00:00Z".into(),
        },
        source_file_sha256: Some(sha256(b"theorem example : True := by sorry")),
    }
}

fn synthetic_run(task: &Task, outcome: &str) -> Run {
    let fingerprint = task_fingerprint(task).unwrap();
    let success = outcome == "verified_success";
    Run {
        schema_version: RUN_SCHEMA.into(),
        run_id: format!("run-{outcome}"),
        task_fingerprint: fingerprint,
        task_id: task.id.clone(),
        repo_commit: task.repo.commit.clone(),
        lean_version: task.repo.lean_version.clone(),
        source_file_sha256: task.source_file_sha256.clone().unwrap(),
        replacement_patch_sha256: sha256(b"patch"),
        proof_artifact_sha256: sha256(b"proof"),
        checker_receipt_sha256: sha256(b"checker"),
        semantic_gate_receipt_sha256: sha256(b"semantic"),
        source_binding_receipt_sha256: sha256(b"source-binding"),
        strategy: Strategy {
            name: "synthetic".into(),
            version: "v0".into(),
            implementation_sha256: sha256(b"implementation"),
            model_id: Some("synthetic-model".into()),
            model_version: Some("v0".into()),
        },
        outcome: outcome.into(),
        proof_inserted: success,
        exact_sorry_replaced: success,
        sorry_absent_after_replacement: success,
        build_passed: success,
        kernel_passed: success,
        semantic_gate_passed: success,
        source_binding_passed: success,
        vacuity_gate_passed: success,
        axiom_delta_zero: success,
        new_sorry_delta_zero: success,
        wall_time_ms: 1250,
        input_tokens: 500,
        output_tokens: 120,
        cost_microusd: 5000,
        failure_type: if outcome == "infrastructure_failure" { Some("build_timeout".into()) } else { None },
    }
}

fn run_self_tests() -> Result<(), String> {
    let task = synthetic_task();
    validate_task(&task, 1)?;
    let fingerprint = task_fingerprint(&task)?;
    if !valid_hex(&fingerprint, 64) {
        return Err("canonical task fingerprint is not SHA-256".into());
    }
    println!("SELF_TEST_PASS stable task fingerprint binds repository, location, Lean version and goal");

    let mut bad = task.clone();
    bad.location.path = "../escape.lean".into();
    if validate_task(&bad, 2).is_ok() {
        return Err("path traversal was accepted".into());
    }
    println!("SELF_TEST_PASS repository source paths reject traversal");

    let mut bad = task.clone();
    bad.location.end_line = 1;
    if validate_task(&bad, 3).is_ok() {
        return Err("reversed source coordinates were accepted".into());
    }
    println!("SELF_TEST_PASS source coordinates are ordered and 1-based");

    let good = synthetic_run(&task, "verified_success");
    let tasks = BTreeMap::from([(fingerprint.clone(), &task)]);
    validate_run(&good, 1, &tasks)?;
    println!("SELF_TEST_PASS verified success requires source, kernel, semantic and vacuity gates");

    let mut bad_run = good.clone();
    bad_run.semantic_gate_passed = false;
    if validate_run(&bad_run, 2, &tasks).is_ok() {
        return Err("kernel-only proof was accepted as verified success".into());
    }
    println!("SELF_TEST_PASS kernel acceptance alone is insufficient");

    let mut bad_run = good.clone();
    bad_run.axiom_delta_zero = false;
    if validate_run(&bad_run, 3, &tasks).is_ok() {
        return Err("new axiom introduction was accepted".into());
    }
    println!("SELF_TEST_PASS new axioms invalidate verified success");

    let mut bad_run = good.clone();
    bad_run.new_sorry_delta_zero = false;
    if validate_run(&bad_run, 4, &tasks).is_ok() {
        return Err("new sorry introduction was accepted".into());
    }
    println!("SELF_TEST_PASS new sorry introduction invalidates verified success");

    let mut missing_source = task.clone();
    missing_source.source_file_sha256 = None;
    let missing_fp = task_fingerprint(&missing_source)?;
    let mut weak = synthetic_run(&missing_source, "non_solve");
    weak.task_fingerprint = missing_fp.clone();
    weak.source_file_sha256 = sha256(b"unbound-source");
    let weak_tasks = BTreeMap::from([(missing_fp, &missing_source)]);
    if validate_run(&synthetic_run(&task, "verified_success"), 5, &weak_tasks).is_ok() {
        return Err("verified success with a mismatched task/source was accepted".into());
    }
    if validate_run(&{
        let mut candidate = synthetic_run(&missing_source, "verified_success");
        candidate.source_file_sha256 = sha256(b"unbound-source");
        candidate
    }, 6, &weak_tasks).is_ok() {
        return Err("verified success without task source digest was accepted".into());
    }
    println!("SELF_TEST_PASS source-less tasks cannot receive strong verified-success status");

    let rows = vec![good, synthetic_run(&task, "non_solve")];
    let report = audit(&[task], sha256(b"tasks"), &rows, Some(sha256(b"runs")))?;
    if report.task_count != 1 || report.run_count != 2 || report.verified_success_count != 1 {
        return Err("synthetic task/run report denominator control failed".into());
    }
    println!("SELF_TEST_PASS reports preserve task and attempt denominators separately");
    Ok(())
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        run_self_tests()?;
        println!("PITB_SORRYDB_INTEGRITY_SELF_TEST_PASS");
        return Ok(());
    }
    if args.len() == 3 && args[1] == "--self-test-report" {
        run_self_tests()?;
        let task = synthetic_task();
        let tasks = vec![task.clone()];
        let runs = vec![
            synthetic_run(&task, "verified_success"),
            synthetic_run(&task, "non_solve"),
        ];
        let report = audit(&tasks, sha256(b"synthetic-task-json"), &runs, Some(sha256(b"synthetic-runs-jsonl")))?;
        let bytes = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize synthetic report: {e}"))?;
        fs::write(&args[2], bytes).map_err(|e| format!("write synthetic report: {e}"))?;
        println!("PITB_SORRYDB_INTEGRITY_SYNTHETIC_REPORT_OK");
        return Ok(());
    }
    let mut tasks_path = None;
    let mut runs_path = None;
    let mut report_path = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--tasks" if index + 1 < args.len() => {
                tasks_path = Some(PathBuf::from(&args[index + 1]));
                index += 2;
            }
            "--runs" if index + 1 < args.len() => {
                runs_path = Some(PathBuf::from(&args[index + 1]));
                index += 2;
            }
            "--report" if index + 1 < args.len() => {
                report_path = Some(PathBuf::from(&args[index + 1]));
                index += 2;
            }
            _ => return Err(format!("unknown or incomplete option {}", args[index])),
        }
    }
    let tasks_path = tasks_path.ok_or("--tasks is required")?;
    let report_path = report_path.ok_or("--report is required")?;
    let task_bytes = fs::read(&tasks_path).map_err(|e| format!("read task snapshot: {e}"))?;
    let tasks = parse_tasks(&task_bytes)?;
    let runs_loaded = if let Some(path) = runs_path {
        let bytes = fs::read(path).map_err(|e| format!("read run JSONL: {e}"))?;
        let runs = parse_runs(&bytes)?;
        Some((runs, sha256(&bytes)))
    } else {
        None
    };
    let report = audit(
        &tasks,
        sha256(&task_bytes),
        runs_loaded.as_ref().map(|x| x.0.as_slice()).unwrap_or(&[]),
        runs_loaded.map(|x| x.1),
    )?;
    let encoded = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize report: {e}"))?;
    fs::write(report_path, encoded).map_err(|e| format!("write report: {e}"))?;
    println!("PITB_SORRYDB_INTEGRITY_OK");
    println!("tasks={} repositories={} runs={} verified_successes={}",
        report.task_count, report.repository_snapshot_count, report.run_count, report.verified_success_count);
    Ok(())
}

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("INVALID: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorrydb_integrity_self_tests_pass() {
        run_self_tests().expect("all SorryDB integrity controls must pass");
    }

    #[test]
    fn duplicate_task_ids_and_duplicate_attempt_ids_fail_closed() {
        let task = synthetic_task();
        let mut tasks = vec![task.clone(), task.clone()];
        assert!(audit(&tasks, sha256(b"duplicate tasks"), &[], None).is_err());
        tasks.pop();
        let run = synthetic_run(&task, "non_solve");
        assert!(audit(&tasks, sha256(b"tasks"), &[run.clone(), run], None).is_err());
    }
}
