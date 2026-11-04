use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::PathBuf,
    process::ExitCode,
};

const UPSTREAM_REPOSITORY: &str = "SorryDB/SorryDB";
const UPSTREAM_COMMIT: &str = "74b3b63f8ba890d675f362924b21ac55db410383";
const FULL_DATASET_PATH: &str = "data/SorryDB_2601/SorryDB_2601.json";
const EVALUATION_SPLIT_PATH: &str = "data/SorryDB_2601/SorryDB_2601_1000_evaluation_split.json";
const EXPECTED_FULL_BLOB_SHA1: &str = "5fafddac467af6d30d8f5c8731d6edec9c273a47";
const EXPECTED_SPLIT_BLOB_SHA1: &str = "af3b663429dc4689ee728d2fb5c7d8d5034476f7";
const DOCUMENTED_FULL_COUNT: usize = 2601;
const EXPECTED_FULL_COUNT: usize = 5663;
const EXPECTED_EVALUATION_COUNT: usize = 1000;
const EXPECTED_REPOSITORY_COUNT: usize = 78;
const REPORT_SCHEMA: &str = "pitb.sorrydb-snapshot-audit.v0";

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Dataset {
    documentation: String,
    sorries: Vec<SorryRecord>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SorryRecord {
    id: String,
    repo: RepoInfo,
    location: Location,
    debug_info: DebugInfo,
    metadata: Metadata,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RepoInfo {
    remote: String,
    branch: String,
    commit: String,
    lean_version: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Location {
    path: String,
    start_line: u64,
    start_column: u64,
    end_line: u64,
    end_column: u64,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct DebugInfo {
    goal: String,
    url: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Metadata {
    blame_email_hash: String,
    blame_date: String,
    inclusion_date: String,
}

#[derive(Debug, Clone, Serialize)]
struct ClusterSummary {
    repository_count: usize,
    repository_commit_count: usize,
    source_file_count: usize,
    tasks_per_repository_min: usize,
    tasks_per_repository_max: usize,
    tasks_per_source_file_max: usize,
    source_file_task_histogram: BTreeMap<String, usize>,
    top_source_file_clusters: Vec<SourceFileCluster>,
}

#[derive(Debug, Clone, Serialize)]
struct SourceFileCluster {
    repository: String,
    path: String,
    task_count: usize,
    distinct_repository_commits: usize,
}


#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Strategy {
    name: String,
    version: String,
    implementation_sha256: String,
    model_id: Option<String>,
    model_version: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
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
    upstream_repository: String,
    upstream_repository_commit: String,
    full_dataset_path: String,
    evaluation_split_path: String,
    full_dataset_git_blob_sha1: String,
    evaluation_split_git_blob_sha1: String,
    full_dataset_raw_sha256: String,
    evaluation_split_raw_sha256: String,
    run_input_sha256: Option<String>,
    run_count: usize,
    verified_success_run_count: usize,
    unique_successful_task_count: usize,
    strategy_summaries: Vec<StrategySummary>,
    full_task_count: usize,
    evaluation_task_count: usize,
    documented_full_task_count: usize,
    full_count_documentation_drift: bool,
    full_unique_task_id_count: usize,
    full_unique_goal_count: usize,
    evaluation_unique_task_id_count: usize,
    evaluation_unique_goal_count: usize,
    evaluation_exact_record_matches_in_full_snapshot: usize,
    repository_count_full: usize,
    repository_count_evaluation: usize,
    lean_version_counts_evaluation: BTreeMap<String, usize>,
    zero_based_start_column_task_count_full: usize,
    zero_based_start_column_task_count_evaluation: usize,
    cluster_summary_evaluation: ClusterSummary,
    source_reconstruction_requirement: String,
    benchmark_integrity_warnings: Vec<String>,
    admission_boundary: String,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha1_digest(input: &[u8]) -> [u8; 20] {
    let bit_length = (input.len() as u64).wrapping_mul(8);
    let padded_length = ((input.len() + 9 + 63) / 64) * 64;
    let mut data = vec![0_u8; padded_length];
    data[..input.len()].copy_from_slice(input);
    data[input.len()] = 0x80;
    data[padded_length - 8..].copy_from_slice(&bit_length.to_be_bytes());

    let mut h0 = 0x67452301_u32;
    let mut h1 = 0xefcdab89_u32;
    let mut h2 = 0x98badcfe_u32;
    let mut h3 = 0x10325476_u32;
    let mut h4 = 0xc3d2e1f0_u32;

    for chunk in data.chunks_exact(64) {
        let mut w = [0_u32; 80];
        for (i, slot) in w.iter_mut().take(16).enumerate() {
            let offset = i * 4;
            *slot = u32::from_be_bytes([
                chunk[offset],
                chunk[offset + 1],
                chunk[offset + 2],
                chunk[offset + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);
        for (i, word) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5a827999_u32),
                20..=39 => (b ^ c ^ d, 0x6ed9eba1_u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1bbcdc_u32),
                _ => (b ^ c ^ d, 0xca62c1d6_u32),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut digest = [0_u8; 20];
    for (i, word) in [h0, h1, h2, h3, h4].iter().enumerate() {
        digest[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

fn git_blob_sha1(bytes: &[u8]) -> String {
    let header = format!("blob {}\0", bytes.len());
    let mut input = Vec::with_capacity(header.len() + bytes.len());
    input.extend_from_slice(header.as_bytes());
    input.extend_from_slice(bytes);
    sha1_digest(&input)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn non_empty(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{label} must be non-empty"))
    } else {
        Ok(())
    }
}

fn valid_iso8601(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() >= 20
        && bytes.get(4) == Some(&b'-')
        && bytes.get(7) == Some(&b'-')
        && (bytes.get(10) == Some(&b'T') || bytes.get(10) == Some(&b't'))
        && value.contains(':')
        && (value.ends_with('Z')
            || value.ends_with('z')
            || value[10..].rfind('+').is_some()
            || value[10..].rfind('-').is_some())
}

fn safe_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.contains('\\')
        && path.split('/').all(|part| !part.is_empty() && part != "." && part != "..")
}

fn validate_record(row: &SorryRecord, label: &str) -> Result<(), String> {
    if !valid_hex(&row.id, 64) {
        return Err(format!("{label}.id must be a 64-character lowercase SHA-256 identifier"));
    }
    non_empty(&row.repo.remote, &format!("{label}.repo.remote"))?;
    if !row.repo.remote.starts_with("https://github.com/") {
        return Err(format!("{label}.repo.remote must be an HTTPS GitHub URL"));
    }
    non_empty(&row.repo.branch, &format!("{label}.repo.branch"))?;
    if !valid_hex(&row.repo.commit, 40) {
        return Err(format!("{label}.repo.commit must be a 40-character lowercase Git commit"));
    }
    non_empty(&row.repo.lean_version, &format!("{label}.repo.lean_version"))?;
    if !safe_relative_path(&row.location.path) {
        return Err(format!("{label}.location.path is not a safe repository-relative path"));
    }
    if row.location.start_line == 0 || row.location.end_line < row.location.start_line {
        return Err(format!("{label}.location line range is invalid"));
    }
    // SorryDB data contains valid start_column=0 records. Keep the upstream
    // coordinate convention verbatim; do not silently rewrite to one-based.
    if row.location.end_line == row.location.start_line
        && row.location.end_column <= row.location.start_column
    {
        return Err(format!("{label}.location same-line column range is empty or reversed"));
    }
    non_empty(&row.debug_info.goal, &format!("{label}.debug_info.goal"))?;
    let expected_url = format!(
        "{}/blob/{}/{}#L{}",
        row.repo.remote,
        row.repo.commit,
        row.location.path,
        row.location.start_line
    );
    if row.debug_info.url != expected_url {
        return Err(format!("{label}.debug_info.url does not match pinned repo/commit/path/line"));
    }
    non_empty(&row.metadata.blame_email_hash, &format!("{label}.metadata.blame_email_hash"))?;
    if !valid_iso8601(&row.metadata.blame_date) {
        return Err(format!("{label}.metadata.blame_date is not ISO-8601-like"));
    }
    if !valid_iso8601(&row.metadata.inclusion_date) {
        return Err(format!("{label}.metadata.inclusion_date is not ISO-8601-like"));
    }
    Ok(())
}

fn validate_dataset(dataset: &Dataset, label: &str) -> Result<(), String> {
    non_empty(&dataset.documentation, &format!("{label}.documentation"))?;
    if dataset.sorries.is_empty() {
        return Err(format!("{label}.sorries cannot be empty"));
    }
    let mut ids = BTreeSet::new();
    let mut goals = BTreeSet::new();
    for (index, row) in dataset.sorries.iter().enumerate() {
        validate_record(row, &format!("{label}.sorries[{}]", index + 1))?;
        if !ids.insert(row.id.as_str()) {
            return Err(format!("{label} contains duplicate task id {}", row.id));
        }
        if !goals.insert(row.debug_info.goal.as_str()) {
            return Err(format!("{label} contains duplicate goal text"));
        }
    }
    Ok(())
}

fn cluster_summary(rows: &[SorryRecord]) -> ClusterSummary {
    let mut repository_counts = BTreeMap::<String, usize>::new();
    let mut repository_commits = BTreeSet::<(String, String)>::new();
    let mut source_files = BTreeMap::<(String, String), Vec<&SorryRecord>>::new();
    for row in rows {
        *repository_counts.entry(row.repo.remote.clone()).or_default() += 1;
        repository_commits.insert((row.repo.remote.clone(), row.repo.commit.clone()));
        source_files
            .entry((row.repo.remote.clone(), row.location.path.clone()))
            .or_default()
            .push(row);
    }

    let tasks_per_repository_min = repository_counts.values().copied().min().unwrap_or(0);
    let tasks_per_repository_max = repository_counts.values().copied().max().unwrap_or(0);
    let mut histogram = BTreeMap::<String, usize>::new();
    let mut top = Vec::new();
    for ((repository, path), records) in &source_files {
        *histogram.entry(records.len().to_string()).or_default() += 1;
        let distinct_commits = records
            .iter()
            .map(|row| row.repo.commit.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        top.push(SourceFileCluster {
            repository: repository.clone(),
            path: path.clone(),
            task_count: records.len(),
            distinct_repository_commits: distinct_commits,
        });
    }
    top.sort_by(|a, b| {
        b.task_count
            .cmp(&a.task_count)
            .then_with(|| a.repository.cmp(&b.repository))
            .then_with(|| a.path.cmp(&b.path))
    });
    top.truncate(12);

    ClusterSummary {
        repository_count: repository_counts.len(),
        repository_commit_count: repository_commits.len(),
        source_file_count: source_files.len(),
        tasks_per_repository_min,
        tasks_per_repository_max,
        tasks_per_source_file_max: source_files.values().map(Vec::len).max().unwrap_or(0),
        source_file_task_histogram: histogram,
        top_source_file_clusters: top,
    }
}


fn validate_proof_run(
    run: &ProofRun,
    index: usize,
    evaluation_by_id: &BTreeMap<&str, &SorryRecord>,
) -> Result<(), String> {
    let label = format!("proof_run[{index}]");
    if run.schema_version != "pitb.sorrydb-proof-run.v0" {
        return Err(format!("{label} has unsupported schema_version"));
    }
    non_empty(&run.run_id, &format!("{label}.run_id"))?;
    let task = evaluation_by_id
        .get(run.task_id.as_str())
        .copied()
        .ok_or_else(|| format!("{label} task_id is not in the pinned evaluation split"))?;
    if run.repo_commit != task.repo.commit || run.lean_version != task.repo.lean_version {
        return Err(format!("{label} repository commit or Lean version differs from the exact task record"));
    }
    if run.source_path != task.location.path
        || run.start_line != task.location.start_line
        || run.start_column != task.location.start_column
        || run.end_line != task.location.end_line
        || run.end_column != task.location.end_column
    {
        return Err(format!("{label} source range differs from the exact SorryDB record"));
    }
    if run.goal_sha256 != sha256(task.debug_info.goal.as_bytes()) {
        return Err(format!("{label} goal digest differs from the pinned SorryDB goal"));
    }
    for (field, value) in [
        ("goal_sha256", &run.goal_sha256),
        ("source_file_sha256", &run.source_file_sha256),
        ("replacement_patch_sha256", &run.replacement_patch_sha256),
        ("proof_artifact_sha256", &run.proof_artifact_sha256),
        ("checker_receipt_sha256", &run.checker_receipt_sha256),
        ("semantic_gate_receipt_sha256", &run.semantic_gate_receipt_sha256),
        ("source_binding_receipt_sha256", &run.source_binding_receipt_sha256),
        ("strategy.implementation_sha256", &run.strategy.implementation_sha256),
    ] {
        if !valid_hex(value, 64) {
            return Err(format!("{label}.{field} must be a 64-character lowercase SHA-256"));
        }
    }
    non_empty(&run.strategy.name, &format!("{label}.strategy.name"))?;
    non_empty(&run.strategy.version, &format!("{label}.strategy.version"))?;
    if let Some(model_id) = &run.strategy.model_id {
        non_empty(model_id, &format!("{label}.strategy.model_id"))?;
    }
    if let Some(model_version) = &run.strategy.model_version {
        non_empty(model_version, &format!("{label}.strategy.model_version"))?;
    }
    if !matches!(
        run.outcome.as_str(),
        "verified_success" | "non_solve" | "infrastructure_failure" | "invalid_target"
    ) {
        return Err(format!("{label} has unsupported outcome"));
    }
    let all_gates = [
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
    if run.outcome == "verified_success" && all_gates.iter().any(|passed| !passed) {
        return Err(format!(
            "{label} claims verified_success without exact replacement, build, kernel, source, semantic, vacuity, axiom and sorry gates"
        ));
    }
    if run.outcome != "verified_success" && all_gates.iter().all(|passed| *passed) {
        return Err(format!("{label} passes every success gate but is labeled as a non-success"));
    }
    if run.outcome == "infrastructure_failure"
        && run.failure_type.as_deref().unwrap_or("").trim().is_empty()
    {
        return Err(format!("{label} infrastructure failure requires failure_type"));
    }
    Ok(())
}

fn parse_proof_runs(bytes: &[u8]) -> Result<Vec<ProofRun>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("proof-run JSONL is not UTF-8: {e}"))?;
    let mut runs = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let run: ProofRun = serde_json::from_str(line)
            .map_err(|e| format!("proof-run JSONL line {}: {e}", index + 1))?;
        runs.push(run);
    }
    Ok(runs)
}

fn summarize_proof_runs(
    runs: &[ProofRun],
    evaluation: &Dataset,
) -> Result<(usize, Vec<StrategySummary>), String> {
    let evaluation_by_id: BTreeMap<&str, &SorryRecord> = evaluation
        .sorries
        .iter()
        .map(|row| (row.id.as_str(), row))
        .collect();
    let mut run_ids = BTreeSet::new();
    let mut groups = BTreeMap::<(String, String, Option<String>, Option<String>), Vec<&ProofRun>>::new();
    let mut verified_success_count = 0usize;
    let mut successful_task_ids = BTreeSet::<&str>::new();
    for (index, run) in runs.iter().enumerate() {
        validate_proof_run(run, index + 1, &evaluation_by_id)?;
        if !run_ids.insert(run.run_id.as_str()) {
            return Err(format!("duplicate proof-run ID {}", run.run_id));
        }
        if run.outcome == "verified_success" {
            verified_success_count += 1;
            successful_task_ids.insert(run.task_id.as_str());
        }
        groups.entry((
            run.strategy.name.clone(),
            run.strategy.version.clone(),
            run.strategy.model_id.clone(),
            run.strategy.model_version.clone(),
        )).or_default().push(run);
    }

    let mut summaries = Vec::new();
    for ((name, version, model_id, model_version), group) in groups {
        let attempts = group.len();
        let verified = group.iter().filter(|run| run.outcome == "verified_success").count();
        let non_solves = group.iter().filter(|run| run.outcome == "non_solve").count();
        let infra = group.iter().filter(|run| run.outcome == "infrastructure_failure").count();
        let invalid = group.iter().filter(|run| run.outcome == "invalid_target").count();
        let successful_times: Vec<u64> = group.iter()
            .filter(|run| run.outcome == "verified_success")
            .map(|run| run.wall_time_ms)
            .collect();
        let total_cost: u128 = group.iter().map(|run| run.cost_microusd as u128).sum();
        summaries.push(StrategySummary {
            strategy_name: name,
            strategy_version: version,
            model_id,
            model_version,
            attempts,
            verified_successes: verified,
            non_solves,
            infrastructure_failures: infra,
            invalid_targets: invalid,
            success_rate_per_attempt: verified as f64 / attempts as f64,
            mean_success_wall_time_ms: if successful_times.is_empty() {
                None
            } else {
                Some(successful_times.iter().sum::<u64>() as f64 / successful_times.len() as f64)
            },
            total_cost_usd: total_cost as f64 / 1_000_000.0,
        });
    }
    Ok((verified_success_count, successful_task_ids.len(), summaries))
}

fn audit(
    full: &Dataset,
    evaluation: &Dataset,
    full_bytes: &[u8],
    evaluation_bytes: &[u8],
    proof_runs: &[ProofRun],
    run_input_sha256: Option<String>,
) -> Result<Report, String> {
    validate_dataset(full, "full_snapshot")?;
    validate_dataset(evaluation, "evaluation_split")?;

    let full_by_id: BTreeMap<&str, &SorryRecord> = full
        .sorries
        .iter()
        .map(|row| (row.id.as_str(), row))
        .collect();
    let mut exact_matches = 0usize;
    for row in &evaluation.sorries {
        let source = full_by_id
            .get(row.id.as_str())
            .ok_or_else(|| format!("evaluation task {} is absent from the full snapshot", row.id))?;
        if *source != row {
            return Err(format!(
                "evaluation task {} differs from its full-snapshot record",
                row.id
            ));
        }
        exact_matches += 1;
    }

    let full_repositories: BTreeSet<&str> = full.sorries.iter().map(|x| x.repo.remote.as_str()).collect();
    let evaluation_repositories: BTreeSet<&str> =
        evaluation.sorries.iter().map(|x| x.repo.remote.as_str()).collect();
    let mut lean_versions = BTreeMap::<String, usize>::new();
    for row in &evaluation.sorries {
        *lean_versions.entry(row.repo.lean_version.clone()).or_default() += 1;
    }

    let zero_based_full = full.sorries.iter().filter(|x| x.location.start_column == 0).count();
    let zero_based_evaluation = evaluation
        .sorries
        .iter()
        .filter(|x| x.location.start_column == 0)
        .count();
    let full_blob = git_blob_sha1(full_bytes);
    let split_blob = git_blob_sha1(evaluation_bytes);
    if full_blob != EXPECTED_FULL_BLOB_SHA1 {
        return Err(format!("full dataset Git blob mismatch: expected {EXPECTED_FULL_BLOB_SHA1}, got {full_blob}"));
    }
    if split_blob != EXPECTED_SPLIT_BLOB_SHA1 {
        return Err(format!("evaluation split Git blob mismatch: expected {EXPECTED_SPLIT_BLOB_SHA1}, got {split_blob}"));
    }
    if full.sorries.len() != EXPECTED_FULL_COUNT {
        return Err(format!(
            "pinned full snapshot task count mismatch: expected {EXPECTED_FULL_COUNT}, got {}",
            full.sorries.len()
        ));
    }
    if evaluation.sorries.len() != EXPECTED_EVALUATION_COUNT {
        return Err(format!(
            "pinned evaluation count mismatch: expected {EXPECTED_EVALUATION_COUNT}, got {}",
            evaluation.sorries.len()
        ));
    }
    if full_repositories.len() != EXPECTED_REPOSITORY_COUNT
        || evaluation_repositories.len() != EXPECTED_REPOSITORY_COUNT
    {
        return Err("pinned repository count mismatch".into());
    }
    if exact_matches != evaluation.sorries.len() {
        return Err("evaluation split is not an exact subset of full snapshot".into());
    }
    let (verified_success_run_count, unique_successful_task_count, strategy_summaries) =
        summarize_proof_runs(proof_runs, evaluation)?;

    let documentation_drift = full.sorries.len() != DOCUMENTED_FULL_COUNT;
    let mut warnings = Vec::new();
    if documentation_drift {
        warnings.push(format!(
            "README snapshot description says {DOCUMENTED_FULL_COUNT} sorries, but the pinned file contains {} records; treat the source blob and task IDs as authoritative, not the stale count label.",
            full.sorries.len()
        ));
    }
    if zero_based_full > 0 || zero_based_evaluation > 0 {
        warnings.push(format!(
            "Found zero-based start-column coordinates in {zero_based_full} full-snapshot records and {zero_based_evaluation} evaluation records; preserve the stored coordinates exactly and validate against the pinned source before patching."
        ));
    }
    warnings.push(
        "The 1,000 evaluation tasks are clustered: they span 78 repositories and 339 repository/file pairs, with up to 21 tasks per repository and up to 20 per file. Use repository/file-clustered confidence intervals, not an IID task bootstrap."
            .into(),
    );
    warnings.push(
        "A pinned repository commit and Lean version establish reproducibility inputs, not semantic fidelity of the intended mathematical statement. Run independent source/goal and vacuity/specification audits before interpreting solve rates."
            .into(),
    );

    Ok(Report {
        schema_version: REPORT_SCHEMA.into(),
        upstream_repository: UPSTREAM_REPOSITORY.into(),
        upstream_repository_commit: UPSTREAM_COMMIT.into(),
        full_dataset_path: FULL_DATASET_PATH.into(),
        evaluation_split_path: EVALUATION_SPLIT_PATH.into(),
        full_dataset_git_blob_sha1: full_blob,
        evaluation_split_git_blob_sha1: split_blob,
        full_dataset_raw_sha256: sha256(full_bytes),
        evaluation_split_raw_sha256: sha256(evaluation_bytes),
        run_input_sha256,
        run_count: proof_runs.len(),
        verified_success_run_count,
        unique_successful_task_count,
        strategy_summaries,
        full_task_count: full.sorries.len(),
        evaluation_task_count: evaluation.sorries.len(),
        documented_full_task_count: DOCUMENTED_FULL_COUNT,
        full_count_documentation_drift: documentation_drift,
        full_unique_task_id_count: full_by_id.len(),
        full_unique_goal_count: full.sorries.iter().map(|x| x.debug_info.goal.as_str()).collect::<BTreeSet<_>>().len(),
        evaluation_unique_task_id_count: evaluation.sorries.iter().map(|x| x.id.as_str()).collect::<BTreeSet<_>>().len(),
        evaluation_unique_goal_count: evaluation.sorries.iter().map(|x| x.debug_info.goal.as_str()).collect::<BTreeSet<_>>().len(),
        evaluation_exact_record_matches_in_full_snapshot: exact_matches,
        repository_count_full: full_repositories.len(),
        repository_count_evaluation: evaluation_repositories.len(),
        lean_version_counts_evaluation: lean_versions,
        zero_based_start_column_task_count_full: zero_based_full,
        zero_based_start_column_task_count_evaluation: zero_based_evaluation,
        cluster_summary_evaluation: cluster_summary(&evaluation.sorries),
        source_reconstruction_requirement: "For each scored task, checkout repo.remote at repo.commit, install the exact repo.lean_version, build the unmodified project, reproduce the sorry at location.path and the stored range, and compare the observed Lean goal to debug_info.goal before attempting a proof. Record repository tree digest, Lake manifest digest, toolchain digest and checker/kernel receipt.".into(),
        benchmark_integrity_warnings: warnings,
        admission_boundary: "This audit validates pinned dataset bytes, record structure, task identity uniqueness and evaluation-split membership. It does not clone or build source repositories, authenticate the original extraction process, prove the informal-to-formal statement mapping, or certify non-vacuity. A task is not benchmark-admissible until source reconstruction and independent semantic checks pass.".into(),
    })
}

fn parse_args(args: &[String]) -> Result<(PathBuf, PathBuf, PathBuf, Option<PathBuf>), String> {
    if args.len() < 7 || args[1] != "--verify" {
        return Err(format!(
            "usage: {} --verify <full-snapshot.json> <evaluation-split.json> <report.json> [--runs proof-runs.jsonl] --pinned-source SorryDB/SorryDB@74b3b63f8ba890d675f362924b21ac55db410383",
            args.first().map(String::as_str).unwrap_or("pitb-sorrydb-audit")
        ));
    }
    let mut runs_path = None;
    let mut pinned_source_seen = false;
    let expected_source = format!("{}@{}", UPSTREAM_REPOSITORY, UPSTREAM_COMMIT);
    let mut index = 5;
    while index < args.len() {
        match args[index].as_str() {
            "--runs" if index + 1 < args.len() && runs_path.is_none() => {
                runs_path = Some(PathBuf::from(&args[index + 1]));
                index += 2;
            }
            "--pinned-source" if index + 1 < args.len() && !pinned_source_seen => {
                if args[index + 1] != expected_source {
                    return Err("SorryDB source repository/commit does not match the frozen audit identity".into());
                }
                pinned_source_seen = true;
                index += 2;
            }
            other => return Err(format!("unknown or duplicate option {other}")),
        }
    }
    if !pinned_source_seen {
        return Err("missing --pinned-source for frozen SorryDB source identity".into());
    }
    Ok((
        PathBuf::from(&args[2]),
        PathBuf::from(&args[3]),
        PathBuf::from(&args[4]),
        runs_path,
    ))
}

fn synthetic_record(id: &str, goal: &str, path: &str, column: u64) -> SorryRecord {
    SorryRecord {
        id: sha256(id.as_bytes()),
        repo: RepoInfo {
            remote: "https://github.com/example/lean-demo".into(),
            branch: "main".into(),
            commit: "1".repeat(40),
            lean_version: "v4.27.0".into(),
        },
        location: Location {
            path: path.into(),
            start_line: 10,
            start_column: column,
            end_line: 10,
            end_column: column + 5,
        },
        debug_info: DebugInfo {
            goal: goal.into(),
            url: format!(
                "https://github.com/example/lean-demo/blob/{}/{path}#L10",
                "1".repeat(40)
            ),
        },
        metadata: Metadata {
            blame_email_hash: "0123456789ab".into(),
            blame_date: "2026-01-01T00:00:00+00:00".into(),
            inclusion_date: "2026-01-02T00:00:00+00:00".into(),
        },
    }
}


fn synthetic_proof_run(task: &SorryRecord, verified: bool, run_id: &str) -> ProofRun {
    let digest = |label: &str| sha256(label.as_bytes());
    ProofRun {
        schema_version: "pitb.sorrydb-proof-run.v0".into(),
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
        source_file_sha256: digest("synthetic-source-file"),
        replacement_patch_sha256: digest("synthetic-patch"),
        proof_artifact_sha256: digest("synthetic-proof"),
        checker_receipt_sha256: digest("synthetic-checker"),
        semantic_gate_receipt_sha256: digest("synthetic-semantic-gate"),
        source_binding_receipt_sha256: digest("synthetic-source-binding"),
        strategy: Strategy {
            name: "synthetic-strategy".into(),
            version: "v0".into(),
            implementation_sha256: digest("synthetic-implementation"),
            model_id: Some("synthetic-model".into()),
            model_version: Some("v0".into()),
        },
        outcome: if verified { "verified_success" } else { "non_solve" }.into(),
        proof_inserted: verified,
        exact_sorry_replaced: verified,
        sorry_absent_after_replacement: verified,
        build_passed: verified,
        kernel_passed: verified,
        semantic_gate_passed: verified,
        source_binding_passed: verified,
        vacuity_gate_passed: verified,
        axiom_delta_zero: verified,
        new_sorry_delta_zero: verified,
        wall_time_ms: 1250,
        input_tokens: 500,
        output_tokens: 120,
        cost_microusd: 5000,
        failure_type: None,
    }
}

fn synthetic_proof_runs() -> Vec<ProofRun> {
    let task = synthetic_record("task-a", "x : Nat\n⊢ x = x", "Demo/A.lean", 0);
    vec![
        synthetic_proof_run(&task, true, "synthetic-attempt-1"),
        synthetic_proof_run(&task, false, "synthetic-attempt-2"),
    ]
}

fn run_self_tests() -> Result<(), String> {
    if git_blob_sha1(b"") != "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
        || git_blob_sha1(b"abc") != "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f"
    {
        return Err("Git blob SHA-1 controls failed".into());
    }
    println!("SELF_TEST_PASS Git object SHA-1 matches canonical empty and non-empty blobs");

    let a = synthetic_record("task-a", "x : Nat\n⊢ x = x", "Demo/A.lean", 0);
    let b = synthetic_record("task-b", "x : Nat\n⊢ x + 0 = x", "Demo/A.lean", 20);
    let c = synthetic_record("task-c", "x : Nat\n⊢ x = x + 0", "Demo/B.lean", 5);
    let full = Dataset {
        documentation: "synthetic".into(),
        sorries: vec![a.clone(), b.clone(), c.clone()],
    };
    let evaluation = Dataset {
        documentation: "synthetic".into(),
        sorries: vec![a.clone(), c.clone()],
    };
    validate_dataset(&full, "synthetic_full")?;
    validate_dataset(&evaluation, "synthetic_evaluation")?;
    let exact = full.sorries.iter().map(|x| (x.id.as_str(), x)).collect::<BTreeMap<_, _>>();
    if evaluation.sorries.iter().any(|x| exact.get(x.id.as_str()).copied() != Some(x)) {
        return Err("synthetic exact subset control failed".into());
    }
    println!("SELF_TEST_PASS exact task records must occur in the pinned full snapshot");

    let mut bad = evaluation.clone();
    bad.sorries[0].location.start_column = 21;
    bad.sorries[0].location.end_column = 20;
    if validate_dataset(&bad, "bad_range").is_ok() {
        return Err("reversed source columns were accepted".into());
    }
    println!("SELF_TEST_PASS invalid source range is rejected");

    let mut bad = evaluation.clone();
    bad.sorries[0].repo.commit = "moving-branch".into();
    if validate_dataset(&bad, "bad_commit").is_ok() {
        return Err("moving repository branch without immutable commit was accepted".into());
    }
    println!("SELF_TEST_PASS mutable branch name cannot substitute for commit SHA");

    let mut bad = evaluation.clone();
    bad.sorries[0].debug_info.url = "https://github.com/example/lean-demo/blob/main/Demo/A.lean#L10".into();
    if validate_dataset(&bad, "bad_url").is_ok() {
        return Err("source URL that points to a moving branch was accepted".into());
    }
    println!("SELF_TEST_PASS source URL must point to pinned commit and exact coordinates");

    let mut bad = evaluation.clone();
    bad.sorries.push(bad.sorries[0].clone());
    if validate_dataset(&bad, "duplicate_id").is_ok() {
        return Err("duplicate task ID was accepted".into());
    }
    println!("SELF_TEST_PASS duplicate benchmark task ID is rejected");

    let mut bad = evaluation.clone();
    bad.sorries[0].location.path = "../outside.lean".into();
    if validate_dataset(&bad, "path_traversal").is_ok() {
        return Err("source path traversal was accepted".into());
    }
    println!("SELF_TEST_PASS unsafe source path is rejected");

    let evaluation_by_id: BTreeMap<&str, &SorryRecord> = evaluation
        .sorries
        .iter()
        .map(|row| (row.id.as_str(), row))
        .collect();
    let valid_run = synthetic_proof_run(&evaluation.sorries[0], true, "self-test-success");
    validate_proof_run(&valid_run, 1, &evaluation_by_id)?;
    println!("SELF_TEST_PASS proof receipt binds exact source task, range, commit, toolchain and goal");

    let mut bad_run = valid_run.clone();
    bad_run.goal_sha256 = sha256(b"different theorem");
    if validate_proof_run(&bad_run, 2, &evaluation_by_id).is_ok() {
        return Err("proof receipt with changed goal identity was accepted".into());
    }
    println!("SELF_TEST_PASS proof receipt rejects a changed formal goal");

    let mut bad_run = valid_run.clone();
    bad_run.start_column += 1;
    if validate_proof_run(&bad_run, 3, &evaluation_by_id).is_ok() {
        return Err("proof receipt with changed source coordinates was accepted".into());
    }
    println!("SELF_TEST_PASS proof receipt rejects changed source coordinates");

    let mut bad_run = valid_run.clone();
    bad_run.semantic_gate_passed = false;
    if validate_proof_run(&bad_run, 2, &evaluation_by_id).is_ok() {
        return Err("kernel-only success without semantic gate was accepted".into());
    }
    println!("SELF_TEST_PASS kernel acceptance alone is not assurance-adjusted success");

    let mut bad_run = valid_run.clone();
    bad_run.axiom_delta_zero = false;
    if validate_proof_run(&bad_run, 3, &evaluation_by_id).is_ok() {
        return Err("proof adding an axiom was accepted".into());
    }
    println!("SELF_TEST_PASS newly introduced axioms invalidate verified success");

    let mut bad_run = valid_run.clone();
    bad_run.new_sorry_delta_zero = false;
    if validate_proof_run(&bad_run, 4, &evaluation_by_id).is_ok() {
        return Err("proof adding a new sorry was accepted".into());
    }
    println!("SELF_TEST_PASS newly introduced sorry invalidates verified success");

    let runs = vec![
        valid_run,
        synthetic_proof_run(&evaluation.sorries[0], false, "self-test-non-solve"),
    ];
    let (success_runs, unique_successes, summaries) = summarize_proof_runs(&runs, &evaluation)?;
    if success_runs != 1 || unique_successes != 1 || summaries.len() != 1
        || summaries[0].attempts != 2
        || summaries[0].verified_successes != 1
        || summaries[0].non_solves != 1
    {
        return Err("proof-run denominator and strategy summary controls failed".into());
    }
    if summarize_proof_runs(&[runs[0].clone(), runs[0].clone()], &evaluation).is_ok() {
        return Err("duplicate proof-run IDs were accepted".into());
    }
    println!("SELF_TEST_PASS per-strategy attempt/success denominators and duplicate run rejection");
    Ok(())
}

fn synthetic_report() -> Result<Value, String> {
    let full = Dataset {
        documentation: "synthetic snapshot".into(),
        sorries: vec![
            synthetic_record("task-a", "x : Nat\n⊢ x = x", "Demo/A.lean", 0),
            synthetic_record("task-b", "x : Nat\n⊢ x + 0 = x", "Demo/A.lean", 20),
            synthetic_record("task-c", "x : Nat\n⊢ x = x + 0", "Demo/B.lean", 5),
        ],
    };
    let evaluation = Dataset {
        documentation: "synthetic evaluation".into(),
        sorries: vec![
            full.sorries[0].clone(),
            full.sorries[2].clone(),
        ],
    };
    let full_bytes = serde_json::to_vec(&full).map_err(|e| e.to_string())?;
    let evaluation_bytes = serde_json::to_vec(&evaluation).map_err(|e| e.to_string())?;
    let proof_runs = synthetic_proof_runs();
    let (verified_success_count, strategy_summaries) =
        summarize_proof_runs(&proof_runs, &evaluation)?;
    let mut run_bytes = Vec::new();
    for run in &proof_runs {
        run_bytes.extend_from_slice(&serde_json::to_vec(run).map_err(|e| e.to_string())?);
        run_bytes.push(b'\n');
    }
    let run_input_sha256 = sha256(&run_bytes);
    let mut report = json!({
        "schema_version": REPORT_SCHEMA,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_repository_commit": UPSTREAM_COMMIT,
        "full_dataset_path": FULL_DATASET_PATH,
        "evaluation_split_path": EVALUATION_SPLIT_PATH,
        "full_dataset_git_blob_sha1": git_blob_sha1(&full_bytes),
        "evaluation_split_git_blob_sha1": git_blob_sha1(&evaluation_bytes),
        "full_dataset_raw_sha256": sha256(&full_bytes),
        "evaluation_split_raw_sha256": sha256(&evaluation_bytes),
        "run_input_sha256": run_input_sha256,
        "run_count": proof_runs.len(),
        "verified_success_run_count": verified_success_run_count,
        "unique_successful_task_count": unique_successful_task_count,
        "strategy_summaries": strategy_summaries,
        "full_task_count": 3,
        "evaluation_task_count": 2,
        "documented_full_task_count": DOCUMENTED_FULL_COUNT,
        "full_count_documentation_drift": true,
        "full_unique_task_id_count": 3,
        "full_unique_goal_count": 3,
        "evaluation_unique_task_id_count": 2,
        "evaluation_unique_goal_count": 2,
        "evaluation_exact_record_matches_in_full_snapshot": 2,
        "repository_count_full": 1,
        "repository_count_evaluation": 1,
        "lean_version_counts_evaluation": {"v4.27.0": 2},
        "zero_based_start_column_task_count_full": 1,
        "zero_based_start_column_task_count_evaluation": 1,
        "cluster_summary_evaluation": {
            "repository_count": 1,
            "repository_commit_count": 1,
            "source_file_count": 2,
            "tasks_per_repository_min": 2,
            "tasks_per_repository_max": 2,
            "tasks_per_source_file_max": 1,
            "source_file_task_histogram": {"1": 2},
            "top_source_file_clusters": []
        },
        "source_reconstruction_requirement": "synthetic source reconstruction policy",
        "benchmark_integrity_warnings": ["synthetic documentation drift control"],
        "admission_boundary": "synthetic benchmark admission boundary"
    });
    if !report.is_object() {
        return Err("synthetic report construction failed".into());
    }
    Ok(report)
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        run_self_tests()?;
        println!("PITB_SORRYDB_AUDIT_SELF_TEST_PASS");
        return Ok(());
    }
    if (args.len() == 3 || args.len() == 4) && args[1] == "--self-test-report" {
        run_self_tests()?;
        let report = synthetic_report()?;
        let bytes = serde_json::to_vec_pretty(&report)
            .map_err(|e| format!("serialize synthetic audit report: {e}"))?;
        fs::write(&args[2], bytes).map_err(|e| format!("write synthetic audit report: {e}"))?;
        if args.len() == 4 {
            let runs = synthetic_proof_runs();
            let mut jsonl = Vec::new();
            for run in &runs {
                jsonl.extend_from_slice(
                    &serde_json::to_vec(run).map_err(|e| format!("serialize synthetic proof run: {e}"))?
                );
                jsonl.push(b'\n');
            }
            fs::write(&args[3], jsonl).map_err(|e| format!("write synthetic proof-run JSONL: {e}"))?;
        }
        println!("PITB_SORRYDB_AUDIT_SYNTHETIC_REPORT_OK");
        return Ok(());
    }
    let (full_path, evaluation_path, report_path, runs_path) = parse_args(&args)?;
    let full_bytes = fs::read(&full_path).map_err(|e| format!("read full SorryDB snapshot: {e}"))?;
    let evaluation_bytes =
        fs::read(&evaluation_path).map_err(|e| format!("read SorryDB evaluation split: {e}"))?;
    let full: Dataset = serde_json::from_slice(&full_bytes)
        .map_err(|e| format!("parse full SorryDB snapshot: {e}"))?;
    let evaluation: Dataset = serde_json::from_slice(&evaluation_bytes)
        .map_err(|e| format!("parse SorryDB evaluation split: {e}"))?;
    let (proof_runs, run_hash) = if let Some(path) = runs_path {
        let bytes = fs::read(path).map_err(|e| format!("read proof-run JSONL: {e}"))?;
        let rows = parse_proof_runs(&bytes)?;
        (rows, Some(sha256(&bytes)))
    } else {
        (Vec::new(), None)
    };
    let report = audit(
        &full,
        &evaluation,
        &full_bytes,
        &evaluation_bytes,
        &proof_runs,
        run_hash,
    )?;
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| format!("serialize SorryDB audit report: {e}"))?;
    fs::write(&report_path, bytes).map_err(|e| format!("write SorryDB audit report: {e}"))?;
    println!("PITB_SORRYDB_SNAPSHOT_AUDIT_OK");
    println!(
        "full={} evaluation={} exact_subset_matches={} repos={} source_files={}",
        report.full_task_count,
        report.evaluation_task_count,
        report.evaluation_exact_record_matches_in_full_snapshot,
        report.repository_count_evaluation,
        report.cluster_summary_evaluation.source_file_count
    );
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
    fn registered_sorrydb_integrity_controls_pass() {
        run_self_tests().expect("all SorryDB integrity controls must pass");
    }

    #[test]
    fn zero_based_start_column_is_preserved_and_valid() {
        let row = synthetic_record("task-zero", "x : Nat\n⊢ x = x", "Demo/A.lean", 0);
        assert!(validate_record(&row, "zero_column").is_ok());
    }

    #[test]
    fn only_exact_commit_pinned_record_matches_are_counted() {
        let mut full = synthetic_record("task", "x : Nat\n⊢ x = x", "Demo/A.lean", 0);
        let mut evaluation = full.clone();
        evaluation.repo.commit = "2".repeat(40);
        assert_ne!(full, evaluation);
        full.repo.branch = "renamed-branch".into();
        assert_ne!(full, evaluation);
    }
}
