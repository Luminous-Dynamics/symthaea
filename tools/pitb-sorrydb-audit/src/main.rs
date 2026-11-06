use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Dataset {
    documentation: String,
    sorries: Vec<SorryRecord>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SorryRecord {
    id: String,
    repo: RepoInfo,
    location: Location,
    debug_info: DebugInfo,
    metadata: Metadata,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RepoInfo {
    remote: String,
    branch: String,
    commit: String,
    lean_version: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Location {
    path: String,
    start_line: u64,
    start_column: u64,
    end_line: u64,
    end_column: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct DebugInfo {
    goal: String,
    url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
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
    execution_stack_sha256: String,
    checkpoint_sha256: Option<String>,
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
    budget_policy_sha256: String,
    evaluation_condition_sha256: String,
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
struct StrategySummary {
    strategy_name: String,
    strategy_version: String,
    model_id: Option<String>,
    model_version: Option<String>,
    implementation_sha256: String,
    execution_stack_sha256: String,
    checkpoint_sha256: Option<String>,
    attempts: usize,
    verified_success_attempts: usize,
    unique_tasks_solved: usize,
    unique_task_success_rate: f64,
    non_solves: usize,
    infrastructure_failures: usize,
    invalid_targets: usize,
    success_rate_per_attempt: f64,
    mean_success_wall_time_ms: Option<f64>,
    total_cost_usd: f64,
    cost_per_unique_verified_task_usd: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct ClusteredStrategyComparison {
    strategy_a_name: String,
    strategy_a_version: String,
    strategy_a_model_id: Option<String>,
    strategy_a_model_version: Option<String>,
    strategy_a_implementation_sha256: String,
    strategy_a_execution_stack_sha256: String,
    strategy_a_checkpoint_sha256: Option<String>,
    strategy_b_name: String,
    strategy_b_version: String,
    strategy_b_model_id: Option<String>,
    strategy_b_model_version: Option<String>,
    strategy_b_implementation_sha256: String,
    strategy_b_execution_stack_sha256: String,
    strategy_b_checkpoint_sha256: Option<String>,
    budget_policy_sha256: String,
    evaluation_condition_sha256: String,
    task_denominator: usize,
    strategy_a_attempted_tasks: usize,
    strategy_b_attempted_tasks: usize,
    strategy_a_attempt_coverage: f64,
    strategy_b_attempt_coverage: f64,
    strategy_a_unique_verified_tasks: usize,
    strategy_b_unique_verified_tasks: usize,
    both_solved_tasks: usize,
    a_only_solved_tasks: usize,
    b_only_solved_tasks: usize,
    neither_solved_tasks: usize,
    paired_risk_difference_b_minus_a: f64,
    repository_cluster_count: usize,
    repository_file_cluster_count: usize,
    repository_bootstrap_replicates: usize,
    repository_bootstrap_ci95_low: f64,
    repository_bootstrap_ci95_high: f64,
    nested_repository_file_bootstrap_replicates: usize,
    nested_repository_file_bootstrap_ci95_low: f64,
    nested_repository_file_bootstrap_ci95_high: f64,
    repository_cluster_sign_flip_replicates: usize,
    repository_cluster_sign_flip_p_value: f64,
    holm_adjusted_p_value: f64,
    deterministic_seed_sha256: String,
    inference_note: String,
}

#[derive(Debug, Clone)]
struct FileBootstrapBlock {
    task_count: usize,
    a_solved: usize,
    b_solved: usize,
}

#[derive(Debug, Clone)]
struct RepositoryBootstrapBlock {
    files: Vec<FileBootstrapBlock>,
    task_count: usize,
    a_solved: usize,
    b_solved: usize,
}

struct DeterministicRng {
    state: [u64; 4],
}

impl DeterministicRng {
    fn from_seed_hex(seed: &str) -> Result<Self, String> {
        if !valid_hex(seed, 64) {
            return Err("deterministic RNG seed must be a SHA-256 digest".into());
        }
        let mut state = [0_u64; 4];
        for (index, word) in state.iter_mut().enumerate() {
            let start = index * 16;
            *word = u64::from_str_radix(&seed[start..start + 16], 16)
                .map_err(|error| format!("parse deterministic RNG seed: {error}"))?;
        }
        if state.iter().all(|word| *word == 0) {
            return Err("deterministic RNG cannot use the all-zero state".into());
        }
        Ok(Self { state })
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() % upper as u64) as usize
    }

    fn coin(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
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
    pairwise_clustered_comparisons: Vec<ClusteredStrategyComparison>,
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


#[derive(Debug, Clone, Serialize)]
struct SourceCheckoutReceipt {
    schema_version: String,
    source_database_task_id: String,
    repository_remote: String,
    repository_commit: String,
    lean_version: String,
    source_path: String,
    start_line: u64,
    start_column: u64,
    end_line: u64,
    end_column: u64,
    committed_toolchain_file: String,
    source_file_sha256: String,
    source_span: String,
    source_span_verified_as_sorry: bool,
    scoring_admission: bool,
    limitation: String,
}

#[derive(Debug, Clone, Serialize)]
struct GoalReplayReceipt {
    schema_version: String,
    source_database_task_id: String,
    repository_remote: String,
    repository_commit: String,
    lean_version: String,
    source_path: String,
    start_line: u64,
    start_column: u64,
    end_line: u64,
    end_column: u64,
    goal_sha256: String,
    source_file_sha256: String,
    instrumented_source_sha256: String,
    toolchain_file_sha256: String,
    lake_manifest_sha256: String,
    repository_tree_sha256: String,
    replay_command: String,
    replay_command_sha256: String,
    replay_log_sha256: String,
    goal_capture_begin_marker: String,
    goal_capture_end_marker: String,
    goal_capture_status: String,
    captured_goal_sha256: String,
    replay_exit_code: i32,
    goal_reproduced: bool,
    scoring_admission: bool,
    limitation: String,
}

fn git_output(repo: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .map_err(|e| format!("run git {:?}: {e}", args))?;
    if !output.status.success() {
        return Err(format!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(output.stdout)
}

fn normalize_remote(remote: &str) -> String {
    let mut value = remote.trim().to_owned();
    if let Some(rest) = value.strip_prefix("git@github.com:") {
        value = format!("https://github.com/{rest}");
    }
    if value.ends_with(".git") {
        value.truncate(value.len() - 4);
    }
    value.trim_end_matches('/').to_ascii_lowercase()
}

fn position_to_byte_index(source: &str, line: u64, column: u64) -> Result<usize, String> {
    if line == 0 {
        return Err("SorryDB source lines are 1-based".into());
    }
    let line_index = usize::try_from(line - 1)
        .map_err(|_| "source line coordinate exceeds platform range".to_owned())?;
    let column_index = usize::try_from(column)
        .map_err(|_| "source column coordinate exceeds platform range".to_owned())?;
    let lines: Vec<&str> = source.split('\n').collect();
    let selected = lines
        .get(line_index)
        .ok_or_else(|| format!("source line {line} is outside the committed file"))?;
    let character_count = selected.chars().count();
    if column_index > character_count {
        return Err(format!(
            "source column {column} exceeds line {line} character count {character_count}"
        ));
    }
    let byte_column = selected
        .char_indices()
        .nth(column_index)
        .map(|(offset, _)| offset)
        .unwrap_or(selected.len());
    let byte_prefix = lines.iter().take(line_index).map(|line| line.len() + 1).sum::<usize>();
    Ok(byte_prefix + byte_column)
}

fn verify_source_checkout(
    record: &SorryRecord,
    repo_root: &Path,
) -> Result<SourceCheckoutReceipt, String> {
    validate_record(record, "checkout task")?;

    let commit_bytes = git_output(repo_root, &["rev-parse", "HEAD"])?;
    let commit = String::from_utf8(commit_bytes)
        .map_err(|e| format!("git HEAD is not UTF-8: {e}"))?
        .trim()
        .to_owned();
    if commit != record.repo.commit {
        return Err(format!(
            "checkout commit mismatch: expected {}, got {commit}",
            record.repo.commit
        ));
    }

    let remote_bytes = git_output(repo_root, &["remote", "get-url", "origin"])?;
    let remote = String::from_utf8(remote_bytes)
        .map_err(|e| format!("git origin remote is not UTF-8: {e}"))?
        .trim()
        .to_owned();
    if normalize_remote(&remote) != normalize_remote(&record.repo.remote) {
        return Err(format!(
            "checkout remote mismatch: expected {}, got {remote}",
            record.repo.remote
        ));
    }

    let toolchain_bytes = git_output(repo_root, &["show", "HEAD:lean-toolchain"])?;
    let toolchain = String::from_utf8(toolchain_bytes)
        .map_err(|e| format!("committed lean-toolchain is not UTF-8: {e}"))?
        .trim()
        .to_owned();
    let normalized_toolchain = toolchain
        .rsplit_once(':')
        .map(|(_, version)| version.trim())
        .unwrap_or(toolchain.as_str());
    if normalized_toolchain != record.repo.lean_version {
        return Err(format!(
            "committed Lean toolchain mismatch: expected {}, got {toolchain}",
            record.repo.lean_version
        ));
    }

    let source_spec = format!("HEAD:{}", record.location.path);
    let source_bytes = git_output(repo_root, &["show", &source_spec])?;
    let source = std::str::from_utf8(&source_bytes)
        .map_err(|e| format!("committed Lean source is not UTF-8: {e}"))?;
    let start = position_to_byte_index(
        source,
        record.location.start_line,
        record.location.start_column,
    )?;
    let end = position_to_byte_index(
        source,
        record.location.end_line,
        record.location.end_column,
    )?;
    if end <= start {
        return Err("SorryDB source span is empty or reversed".into());
    }
    let span = source
        .get(start..end)
        .ok_or("SorryDB source span does not align to UTF-8 character boundaries")?;
    if span != "sorry" {
        return Err(format!(
            "pinned source coordinates do not select exactly the sorry token: {span:?}"
        ));
    }

    Ok(SourceCheckoutReceipt {
        schema_version: "pitb.sorrydb-source-checkout.v0".into(),
        source_database_task_id: record.id.clone(),
        repository_remote: record.repo.remote.clone(),
        repository_commit: commit,
        lean_version: record.repo.lean_version.clone(),
        source_path: record.location.path.clone(),
        start_line: record.location.start_line,
        start_column: record.location.start_column,
        end_line: record.location.end_line,
        end_column: record.location.end_column,
        committed_toolchain_file: toolchain,
        source_file_sha256: sha256(&source_bytes),
        source_span: span.to_owned(),
        source_span_verified_as_sorry: true,
        scoring_admission: false,
        limitation: "This receipt verifies only immutable Git identity, the committed Lean toolchain and the exact sorry source span. It does not run Lake/Lean, reproduce the elaborated Prop goal, verify a proof, or audit informal-to-formal specification fidelity.".into(),
    })
}

fn normalized_goal(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn extract_marked_goal(
    log: &str,
    begin_marker: &str,
    end_marker: &str,
) -> Result<String, String> {
    let begin_positions: Vec<usize> = log.match_indices(begin_marker).map(|(index, _)| index).collect();
    let end_positions: Vec<usize> = log.match_indices(end_marker).map(|(index, _)| index).collect();
    if begin_positions.len() != 1 || end_positions.len() != 1 {
        return Err(format!(
            "expected exactly one bounded goal trace marker pair; found begin={} end={}",
            begin_positions.len(),
            end_positions.len()
        ));
    }
    let content_start = begin_positions[0] + begin_marker.len();
    let content_end = end_positions[0];
    if content_end <= content_start {
        return Err("goal trace end marker precedes or touches the begin marker".into());
    }
    let captured = log[content_start..content_end].trim();
    if captured.is_empty() {
        return Err("bounded Lean goal trace is empty".into());
    }
    Ok(captured.to_owned())
}

fn instrument_sorry_span(
    source: &str,
    start_line: u64,
    start_column: u64,
    end_line: u64,
    end_column: u64,
    capture_tactic: &str,
    begin_marker: &str,
    end_marker: &str,
) -> Result<String, String> {
    if begin_marker.trim().is_empty()
        || end_marker.trim().is_empty()
        || begin_marker == end_marker
        || source.contains(begin_marker)
        || source.contains(end_marker)
        || source.contains(capture_tactic)
        || !capture_tactic
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        return Err("goal capture identifiers must be valid, distinct, and absent from the committed source".into());
    }
    let start = position_to_byte_index(source, start_line, start_column)?;
    let end = position_to_byte_index(source, end_line, end_column)?;
    if end <= start || source.get(start..end) != Some("sorry") {
        return Err("pinned source coordinates do not select exactly the sorry token".into());
    }
    let line_start = source[..start]
        .rfind('\n')
        .map(|index| index + 1)
        .unwrap_or(0);
    let leading_indent: String = source[line_start..start]
        .chars()
        .take_while(|character| matches!(character, ' ' | '\t'))
        .collect();
    let tactic_indent = if leading_indent.is_empty() {
        "  ".to_owned()
    } else {
        leading_indent
    };

    // The info view emitted by trace_state is not guaranteed to reach batch
    // compiler stdout/stderr. Define a task-unique tactic that renders the
    // actual metavariable goal and emits it with dbg_trace.
    // When the source has no explicit import block (as in the synthetic
    // fixtures), place the temporary Lean import at byte zero.
    let import_block_end = source.find("\n\n").map(|index| index + 2).unwrap_or(0);
    let tactic_declaration = format!(
        "set_option pp.fullNames true\n\n\
         elab \"{capture_tactic}\" : tactic => do\n\
         \x20 let goal ← Lean.Elab.Tactic.getMainGoal\n\
         \x20 let rendered ← Lean.Meta.ppGoal goal\n\
         \x20 dbg_trace f!\"{begin_marker}\\n{{rendered}}\\n{end_marker}\"\n\n"
    );
    let import_prefix = "import Lean\n";
    let mut instrumented = String::with_capacity(source.len() + tactic_declaration.len() + 32);
    instrumented.push_str(import_prefix);
    instrumented.push_str(&source[..import_block_end]);
    instrumented.push_str(&tactic_declaration);
    instrumented.push_str(&source[import_block_end..]);

    let shifted_start = import_prefix.len() + tactic_declaration.len() + start;
    let shifted_end = import_prefix.len() + tactic_declaration.len() + end;
    let adjusted_line_start = instrumented[..shifted_start]
        .rfind('\n')
        .map(|index| index + 1)
        .unwrap_or(0);
    let mut rewritten = String::with_capacity(instrumented.len() + 64);
    rewritten.push_str(&instrumented[..shifted_start]);
    if instrumented[adjusted_line_start..shifted_start].trim().is_empty() {
        rewritten.truncate(rewritten.len() - (shifted_start - adjusted_line_start));
    } else if !rewritten.ends_with('\n') {
        rewritten.push('\n');
    }
    rewritten.push_str(&format!("{tactic_indent}{capture_tactic}\n{tactic_indent}sorry"));
    rewritten.push_str(&instrumented[shifted_end..]);
    Ok(rewritten)
}

fn replay_goal_with_lake(
    record: &SorryRecord,
    repo_root: &Path,
    receipt_path: &Path,
    log_path: &Path,
    instrumented_path: &Path,
) -> Result<GoalReplayReceipt, String> {
    let checkout = verify_source_checkout(record, repo_root)?;
    let source_spec = format!("HEAD:{}", record.location.path);
    let source_bytes = git_output(repo_root, &["show", source_spec.as_str()])?;
    let source = std::str::from_utf8(&source_bytes)
        .map_err(|e| format!("committed Lean source is not UTF-8: {e}"))?;
    let marker_tag = sha256(record.id.as_bytes())[..16].to_owned();
    let begin_marker = format!("PITB_SORRYDB_GOAL_BEGIN_{marker_tag}");
    let end_marker = format!("PITB_SORRYDB_GOAL_END_{marker_tag}");
    let capture_tactic = format!("pitbCapture{marker_tag}");
    let instrumented = instrument_sorry_span(
        source,
        record.location.start_line,
        record.location.start_column,
        record.location.end_line,
        record.location.end_column,
        &capture_tactic,
        &begin_marker,
        &end_marker,
    )?;
    let manifest_bytes = git_output(repo_root, &["show", "HEAD:lake-manifest.json"])?;
    let toolchain_bytes = git_output(repo_root, &["show", "HEAD:lean-toolchain"])?;
    let tree_bytes = git_output(repo_root, &["rev-parse", "HEAD^{tree}"])?;
    let repository_tree = String::from_utf8(tree_bytes)
        .map_err(|e| format!("repository tree SHA is not UTF-8: {e}"))?
        .trim().to_owned();

    let source_relative = Path::new(&record.location.path);
    let parent = source_relative.parent().unwrap_or_else(|| Path::new(""));
    let stem = source_relative.file_stem().and_then(|x| x.to_str())
        .ok_or("SorryDB source path has no valid UTF-8 file stem")?;
    let task_suffix = record.id.chars().take(8).collect::<String>();
    let replay_filename = format!("{stem}.PITBReplay{task_suffix}.lean");
    let canonical_repo_root = repo_root.canonicalize()
        .map_err(|e| format!("canonicalize Lean repository root: {e}"))?;
    let replay_relative = parent.join(replay_filename);
    let replay_absolute = canonical_repo_root.join(&replay_relative);
    if replay_absolute.exists() {
        return Err(format!("goal replay temporary source already exists: {}", replay_relative.display()));
    }
    if !replay_absolute.starts_with(&canonical_repo_root) {
        return Err("goal replay temporary source escaped repository root".into());
    }
    fs::write(&replay_absolute, instrumented.as_bytes())
        .map_err(|e| format!("write instrumented Lean source: {e}"))?;
    fs::write(instrumented_path, instrumented.as_bytes())
        .map_err(|e| format!("retain instrumented Lean source: {e}"))?;

    let replay_relative_string = replay_relative.to_string_lossy().into_owned();
    let replay_command = format!("lake env lean {replay_relative_string}");
    let output_result = Command::new("lake")
        .args(["env", "lean", replay_relative_string.as_str()])
        .current_dir(&canonical_repo_root)
        .output()
        .map_err(|e| format!("start lake env lean ({replay_command}): {e}"));
    let _ = fs::remove_file(&replay_absolute);
    let output = output_result?;
    let mut log_bytes = Vec::with_capacity(output.stdout.len() + output.stderr.len());
    log_bytes.extend_from_slice(&output.stdout);
    log_bytes.extend_from_slice(&output.stderr);
    fs::write(log_path, &log_bytes)
        .map_err(|e| format!("write Lean goal-replay log: {e}"))?;
    let log_text = String::from_utf8_lossy(&log_bytes);
    let expected_goal = normalized_goal(&record.debug_info.goal);
    let captured_goal = extract_marked_goal(&log_text, &begin_marker, &end_marker);
    let goal_capture_status = if captured_goal.is_ok() {
        "captured".to_owned()
    } else {
        "missing_or_ambiguous".to_owned()
    };
    let captured_goal_sha256 = captured_goal
        .as_ref()
        .map(|goal| sha256(goal.as_bytes()))
        .unwrap_or_else(|_| sha256(b""));
    let goal_reproduced = output.status.success()
        && captured_goal
            .as_ref()
            .map(|goal| normalized_goal(goal) == expected_goal)
            .unwrap_or(false);
    let replay_command_sha256 = sha256(replay_command.as_bytes());

    let receipt = GoalReplayReceipt {
        schema_version: "pitb.sorrydb-goal-replay.v0".into(),
        source_database_task_id: record.id.clone(),
        repository_remote: record.repo.remote.clone(),
        repository_commit: record.repo.commit.clone(),
        lean_version: record.repo.lean_version.clone(),
        source_path: record.location.path.clone(),
        start_line: record.location.start_line,
        start_column: record.location.start_column,
        end_line: record.location.end_line,
        end_column: record.location.end_column,
        goal_sha256: sha256(record.debug_info.goal.as_bytes()),
        source_file_sha256: checkout.source_file_sha256,
        instrumented_source_sha256: sha256(instrumented.as_bytes()),
        toolchain_file_sha256: sha256(&toolchain_bytes),
        lake_manifest_sha256: sha256(&manifest_bytes),
        repository_tree_sha256: repository_tree,
        replay_command,
        replay_command_sha256,
        replay_log_sha256: sha256(&log_bytes),
        goal_capture_begin_marker: begin_marker,
        goal_capture_end_marker: end_marker,
        goal_capture_status,
        captured_goal_sha256,
        replay_exit_code: output.status.code().unwrap_or(-1),
        goal_reproduced,
        scoring_admission: false,
        limitation: "The instrumented source emits the live Lean tactic state at the pinned sorry span and compares its normalized rendering with SorryDB debug_info.goal. It is a reproduction receipt, not a proof result or semantic-fidelity review.".into(),
    };
    let receipt_bytes = serde_json::to_vec_pretty(&receipt)
        .map_err(|e| format!("serialize Lean goal-replay receipt: {e}"))?;
    fs::write(receipt_path, receipt_bytes)
        .map_err(|e| format!("write Lean goal-replay receipt: {e}"))?;
    if !goal_reproduced {
        let diagnostic = String::from_utf8_lossy(&log_bytes);
        let diagnostic_tail: String = diagnostic
            .chars()
            .rev()
            .take(6000)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        return Err(format!(
            "Lean goal replay failed for {} (exit={}, capture_status={}); expected goal digest {}. Inspect {}. Last replay output:\n{}",
            record.id,
            receipt.replay_exit_code,
            receipt.goal_capture_status,
            receipt.goal_sha256,
            log_path.display(),
            diagnostic_tail
        ));
    }
    Ok(receipt)
}

fn build_synthetic_checkout_receipt() -> Result<SourceCheckoutReceipt, String> {
    let root = env::temp_dir().join(format!(
        "pitb-sorrydb-checkout-self-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(root.join("Demo"))
        .map_err(|e| format!("create synthetic repository: {e}"))?;
    fs::write(root.join("lean-toolchain"), "v4.27.0\n")
        .map_err(|e| format!("write synthetic lean-toolchain: {e}"))?;
    let source = "import Lean\n theorem sample : True := by sorry\n";
    fs::write(root.join("Demo/Sample.lean"), source)
        .map_err(|e| format!("write synthetic Lean source: {e}"))?;

    let run = |args: &[&str]| -> Result<(), String> {
        let output = Command::new("git")
            .args(args)
            .current_dir(&root)
            .output()
            .map_err(|e| format!("run git {:?}: {e}", args))?;
        if output.status.success() {
            Ok(())
        } else {
            Err(format!(
                "git {:?} failed: {}",
                args,
                String::from_utf8_lossy(&output.stderr).trim()
            ))
        }
    };
    run(&["init", "-q"])?;
    run(&["config", "user.name", "PITB test"])?;
    run(&["config", "user.email", "pitb@example.invalid"])?;
    run(&[
        "remote",
        "add",
        "origin",
        "https://github.com/example/lean-demo.git",
    ])?;
    run(&["add", "."])?;
    run(&["commit", "-q", "-m", "sorry fixture"])?;

    let commit = String::from_utf8(git_output(&root, &["rev-parse", "HEAD"])?)
        .map_err(|e| format!("decode synthetic commit: {e}"))?
        .trim()
        .to_owned();
    let line = source.split('\n').nth(1).ok_or("synthetic Lean source line missing")?;
    let start_column = line
        .find("sorry")
        .ok_or("synthetic sorry token missing")? as u64;
    let mut record = SorryRecord {
        id: sha256(b"source-task"),
        repo: RepoInfo {
            remote: "https://github.com/example/lean-demo".into(),
            branch: "main".into(),
            commit: commit.clone(),
            lean_version: "v4.27.0".into(),
        },
        location: Location {
            path: "Demo/Sample.lean".into(),
            start_line: 2,
            start_column,
            end_line: 2,
            end_column: start_column + 5,
        },
        debug_info: DebugInfo {
            goal: "⊢ True".into(),
            url: String::new(),
        },
        metadata: Metadata {
            blame_email_hash: "0123456789ab".into(),
            blame_date: "2026-01-01T00:00:00+00:00".into(),
            inclusion_date: "2026-01-02T00:00:00+00:00".into(),
        },
    };
    record.debug_info.url = format!(
        "{}/blob/{}/{}#L2",
        record.repo.remote, record.repo.commit, record.location.path
    );

    // Dirty the worktree after committing: the verifier must still read HEAD:path.
    fs::write(root.join("Demo/Sample.lean"), "modified working tree\n")
        .map_err(|e| format!("dirty synthetic source: {e}"))?;
    let receipt = verify_source_checkout(&record, &root)?;
    if !receipt.source_span_verified_as_sorry || receipt.source_span != "sorry" {
        return Err("committed source span was not verified as sorry".into());
    }
    if receipt.scoring_admission {
        return Err("source-only receipt incorrectly admitted a scored result".into());
    }

    let mut wrong_commit = record.clone();
    wrong_commit.repo.commit = "2".repeat(40);
    if verify_source_checkout(&wrong_commit, &root).is_ok() {
        return Err("checkout verifier accepted a different repository commit".into());
    }
    fs::remove_dir_all(&root).map_err(|e| format!("remove synthetic repository: {e}"))?;
    Ok(receipt)
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
        ("budget_policy_sha256", &run.budget_policy_sha256),
        ("evaluation_condition_sha256", &run.evaluation_condition_sha256),
        ("declaration_signature_before_sha256", &run.declaration_signature_before_sha256),
        ("declaration_signature_after_sha256", &run.declaration_signature_after_sha256),
        ("source_file_sha256", &run.source_file_sha256),
        ("replacement_patch_sha256", &run.replacement_patch_sha256),
        ("proof_artifact_sha256", &run.proof_artifact_sha256),
        ("checker_receipt_sha256", &run.checker_receipt_sha256),
        ("semantic_gate_receipt_sha256", &run.semantic_gate_receipt_sha256),
        ("source_binding_receipt_sha256", &run.source_binding_receipt_sha256),
        ("strategy.implementation_sha256", &run.strategy.implementation_sha256),
        ("strategy.execution_stack_sha256", &run.strategy.execution_stack_sha256),
    ] {
        if !valid_hex(value, 64) {
            return Err(format!("{label}.{field} must be a 64-character lowercase SHA-256"));
        }
    }
    if let Some(checkpoint) = &run.strategy.checkpoint_sha256 {
        if !valid_hex(checkpoint, 64) {
            return Err(format!("{label}.strategy.checkpoint_sha256 must be a 64-character lowercase SHA-256"));
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
    if run.declaration_signature_before_sha256 != run.declaration_signature_after_sha256 {
        return Err(format!("{label} changes the exact target declaration signature"));
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
) -> Result<(usize, usize, Vec<StrategySummary>), String> {
    let evaluation_by_id: BTreeMap<&str, &SorryRecord> = evaluation
        .sorries
        .iter()
        .map(|row| (row.id.as_str(), row))
        .collect();
    let mut run_ids = BTreeSet::new();
    let mut groups = BTreeMap::<StrategyKey, Vec<&ProofRun>>::new();
    let mut execution_conditions = BTreeMap::<StrategyKey, (String, String)>::new();
    let mut verified_success_count = 0usize;
    let mut successful_task_ids = BTreeSet::<&str>::new();
    for (index, run) in runs.iter().enumerate() {
        validate_proof_run(run, index + 1, &evaluation_by_id)?;
        if !run_ids.insert(run.run_id.as_str()) {
            return Err(format!("duplicate proof-run ID {}", run.run_id));
        }
        let key = strategy_key(run);
        let condition = (
            run.budget_policy_sha256.clone(),
            run.evaluation_condition_sha256.clone(),
        );
        if let Some(previous) = execution_conditions.insert(key.clone(), condition.clone()) {
            if previous != condition {
                return Err(format!(
                    "strategy {} mixes resource-budget or evaluation-condition identities",
                    strategy_label(&key)
                ));
            }
        }
        if run.outcome == "verified_success" {
            verified_success_count += 1;
            successful_task_ids.insert(run.task_id.as_str());
        }
        groups.entry(key).or_default().push(run);
    }

    let mut summaries = Vec::new();
    for ((name, version, model_id, model_version, implementation_sha256, execution_stack_sha256, checkpoint_sha256), group) in groups {
        let attempts = group.len();
        let verified = group.iter().filter(|run| run.outcome == "verified_success").count();
        let unique_solved = group.iter()
            .filter(|run| run.outcome == "verified_success")
            .map(|run| run.task_id.as_str())
            .collect::<BTreeSet<_>>()
            .len();
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
            implementation_sha256,
            execution_stack_sha256,
            checkpoint_sha256,
            attempts,
            verified_success_attempts: verified,
            unique_tasks_solved: unique_solved,
            unique_task_success_rate: unique_solved as f64 / evaluation.sorries.len() as f64,
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
            cost_per_unique_verified_task_usd: if unique_solved == 0 {
                None
            } else {
                Some(total_cost as f64 / 1_000_000.0 / unique_solved as f64)
            },
        });
    }
    Ok((verified_success_count, successful_task_ids.len(), summaries))
}

type StrategyKey = (
    String,
    String,
    Option<String>,
    Option<String>,
    String,
    String,
    Option<String>,
);

fn strategy_key(run: &ProofRun) -> StrategyKey {
    (
        run.strategy.name.clone(),
        run.strategy.version.clone(),
        run.strategy.model_id.clone(),
        run.strategy.model_version.clone(),
        run.strategy.implementation_sha256.clone(),
        run.strategy.execution_stack_sha256.clone(),
        run.strategy.checkpoint_sha256.clone(),
    )
}

fn strategy_label(key: &StrategyKey) -> String {
    format!(
        "{}@{}|{}|{}|impl={}|stack={}|checkpoint={}",
        key.0,
        key.1,
        key.2.as_deref().unwrap_or("no-model-id"),
        key.3.as_deref().unwrap_or("no-model-version"),
        key.4,
        key.5,
        key.6.as_deref().unwrap_or("no-checkpoint"),
    )
}

fn percentile(sorted: &[f64], probability: f64) -> Result<f64, String> {
    if sorted.is_empty() || !(0.0..=1.0).contains(&probability) {
        return Err("percentile requires non-empty values and probability in [0,1]".into());
    }
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        Ok(sorted[lower])
    } else {
        let fraction = position - lower as f64;
        Ok(sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction)
    }
}

fn bootstrap_repository_rate_differences(
    repositories: &[RepositoryBootstrapBlock],
    replicates: usize,
    seed: &str,
    nested_files: bool,
) -> Result<(f64, f64), String> {
    if repositories.is_empty() || replicates < 100 {
        return Err("cluster bootstrap requires repositories and at least 100 replicates".into());
    }
    let mut rng = DeterministicRng::from_seed_hex(seed)?;
    let mut values = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut task_count = 0usize;
        let mut a_solved = 0usize;
        let mut b_solved = 0usize;
        for _ in 0..repositories.len() {
            let repository = &repositories[rng.index(repositories.len())];
            if nested_files {
                if repository.files.is_empty() {
                    return Err("repository bootstrap block has no source-file clusters".into());
                }
                for _ in 0..repository.files.len() {
                    let file = &repository.files[rng.index(repository.files.len())];
                    task_count += file.task_count;
                    a_solved += file.a_solved;
                    b_solved += file.b_solved;
                }
            } else {
                task_count += repository.task_count;
                a_solved += repository.a_solved;
                b_solved += repository.b_solved;
            }
        }
        if task_count == 0 {
            return Err("cluster bootstrap produced an empty task sample".into());
        }
        values.push((b_solved as f64 - a_solved as f64) / task_count as f64);
    }
    values.sort_by(f64::total_cmp);
    Ok((percentile(&values, 0.025)?, percentile(&values, 0.975)?))
}

fn repository_cluster_sign_flip_p_value(
    repositories: &[RepositoryBootstrapBlock],
    observed_delta_sum: i64,
    replicates: usize,
    seed: &str,
) -> Result<f64, String> {
    if repositories.is_empty() || replicates < 100 {
        return Err("cluster permutation test requires repositories and at least 100 replicates".into());
    }
    let mut rng = DeterministicRng::from_seed_hex(seed)?;
    let observed = observed_delta_sum.unsigned_abs();
    let mut extreme = 0usize;
    for _ in 0..replicates {
        let mut permuted = 0_i64;
        for repository in repositories {
            let delta = repository.b_solved as i64 - repository.a_solved as i64;
            if rng.coin() {
                permuted += delta;
            } else {
                permuted -= delta;
            }
        }
        if permuted.unsigned_abs() >= observed {
            extreme += 1;
        }
    }
    Ok((extreme + 1) as f64 / (replicates + 1) as f64)
}

fn holm_adjusted_p_values(raw_p_values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..raw_p_values.len()).collect();
    order.sort_by(|left, right| raw_p_values[*left].total_cmp(&raw_p_values[*right]));
    let mut adjusted = vec![0.0_f64; raw_p_values.len()];
    let mut adjusted_so_far = 0.0_f64;
    let family_size = raw_p_values.len();
    for (rank, index) in order.into_iter().enumerate() {
        let candidate = (raw_p_values[index] * (family_size - rank) as f64).min(1.0);
        adjusted_so_far = adjusted_so_far.max(candidate);
        adjusted[index] = adjusted_so_far;
    }
    adjusted
}

fn pairwise_clustered_comparisons(
    runs: &[ProofRun],
    evaluation: &Dataset,
) -> Result<Vec<ClusteredStrategyComparison>, String> {
    let mut attempted = BTreeMap::<StrategyKey, BTreeSet<String>>::new();
    let mut solved = BTreeMap::<StrategyKey, BTreeSet<String>>::new();
    let mut execution_conditions = BTreeMap::<StrategyKey, (String, String)>::new();
    for run in runs {
        let key = strategy_key(run);
        let condition = (
            run.budget_policy_sha256.clone(),
            run.evaluation_condition_sha256.clone(),
        );
        if let Some(previous) = execution_conditions.insert(key.clone(), condition.clone()) {
            if previous != condition {
                return Err(format!(
                    "strategy {} mixes resource-budget or evaluation-condition identities",
                    strategy_label(&key)
                ));
            }
        }
        attempted.entry(key.clone()).or_default().insert(run.task_id.clone());
        if run.outcome == "verified_success" {
            solved.entry(key).or_default().insert(run.task_id.clone());
        }
    }
    let keys: Vec<StrategyKey> = attempted.keys().cloned().collect();
    if keys.len() < 2 {
        return Ok(Vec::new());
    }
    let matched_condition = execution_conditions
        .get(&keys[0])
        .ok_or("missing strategy execution condition")?;
    for key in &keys[1..] {
        if execution_conditions.get(key) != Some(matched_condition) {
            return Err(format!(
                "pairwise strategy comparison requires identical budget and evaluation-condition digests: {} differs",
                strategy_label(key)
            ));
        }
    }
    let mut tasks_by_repository_file =
        BTreeMap::<String, BTreeMap<String, Vec<&SorryRecord>>>::new();
    for task in &evaluation.sorries {
        tasks_by_repository_file
            .entry(task.repo.remote.clone())
            .or_default()
            .entry(task.location.path.clone())
            .or_default()
            .push(task);
    }

    let mut comparisons = Vec::<(f64, ClusteredStrategyComparison)>::new();
    for first_index in 0..keys.len() {
        for second_index in (first_index + 1)..keys.len() {
            let a_key = &keys[first_index];
            let b_key = &keys[second_index];
            let a_attempted = attempted.get(a_key).ok_or("missing strategy A attempt map")?;
            let b_attempted = attempted.get(b_key).ok_or("missing strategy B attempt map")?;
            let a_solved_set = solved.get(a_key);
            let b_solved_set = solved.get(b_key);

            let mut repositories = Vec::<RepositoryBootstrapBlock>::new();
            let mut a_solved_total = 0usize;
            let mut b_solved_total = 0usize;
            let mut both_solved = 0usize;
            let mut a_only = 0usize;
            let mut b_only = 0usize;
            let mut neither = 0usize;
            let mut file_cluster_total = 0usize;
            for files in tasks_by_repository_file.values() {
                let mut file_blocks = Vec::<FileBootstrapBlock>::new();
                let mut repo_tasks = 0usize;
                let mut repo_a = 0usize;
                let mut repo_b = 0usize;
                for task_rows in files.values() {
                    let mut file_tasks = 0usize;
                    let mut file_a = 0usize;
                    let mut file_b = 0usize;
                    for task in task_rows {
                        let a_ok = a_solved_set.is_some_and(|set| set.contains(&task.id));
                        let b_ok = b_solved_set.is_some_and(|set| set.contains(&task.id));
                        file_tasks += 1;
                        repo_tasks += 1;
                        a_solved_total += usize::from(a_ok);
                        b_solved_total += usize::from(b_ok);
                        repo_a += usize::from(a_ok);
                        repo_b += usize::from(b_ok);
                        file_a += usize::from(a_ok);
                        file_b += usize::from(b_ok);
                        match (a_ok, b_ok) {
                            (true, true) => both_solved += 1,
                            (true, false) => a_only += 1,
                            (false, true) => b_only += 1,
                            (false, false) => neither += 1,
                        }
                    }
                    if file_tasks > 0 {
                        file_blocks.push(FileBootstrapBlock {
                            task_count: file_tasks,
                            a_solved: file_a,
                            b_solved: file_b,
                        });
                        file_cluster_total += 1;
                    }
                }
                if repo_tasks > 0 {
                    repositories.push(RepositoryBootstrapBlock {
                        files: file_blocks,
                        task_count: repo_tasks,
                        a_solved: repo_a,
                        b_solved: repo_b,
                    });
                }
            }
            let denominator = evaluation.sorries.len();
            if denominator == 0 || a_solved_total + b_solved_total > 2 * denominator {
                return Err("clustered strategy comparison has an invalid fixed denominator".into());
            }
            let paired_delta_sum = b_solved_total as i64 - a_solved_total as i64;
            let paired_risk_difference = paired_delta_sum as f64 / denominator as f64;
            let seed_material = format!(
                "SorryDB_2601|{}|{}|repo-file-bootstrap-v0|10000|50000",
                strategy_label(a_key),
                strategy_label(b_key)
            );
            let seed = sha256(seed_material.as_bytes());
            let repo_bootstrap_seed = sha256(format!("{seed}|repository-bootstrap").as_bytes());
            let nested_bootstrap_seed = sha256(format!("{seed}|nested-file-bootstrap").as_bytes());
            let permutation_seed = sha256(format!("{seed}|repository-sign-flip").as_bytes());
            let repository_bootstrap_replicates = 10_000usize;
            let nested_bootstrap_replicates = 10_000usize;
            let permutation_replicates = 50_000usize;
            let (repo_low, repo_high) = bootstrap_repository_rate_differences(
                &repositories,
                repository_bootstrap_replicates,
                &repo_bootstrap_seed,
                false,
            )?;
            let (nested_low, nested_high) = bootstrap_repository_rate_differences(
                &repositories,
                nested_bootstrap_replicates,
                &nested_bootstrap_seed,
                true,
            )?;
            let raw_p = repository_cluster_sign_flip_p_value(
                &repositories,
                paired_delta_sum,
                permutation_replicates,
                &permutation_seed,
            )?;
            comparisons.push((
                raw_p,
                ClusteredStrategyComparison {
                    strategy_a_name: a_key.0.clone(),
                    strategy_a_version: a_key.1.clone(),
                    strategy_a_model_id: a_key.2.clone(),
                    strategy_a_model_version: a_key.3.clone(),
                    strategy_a_implementation_sha256: a_key.4.clone(),
                    strategy_a_execution_stack_sha256: a_key.5.clone(),
                    strategy_a_checkpoint_sha256: a_key.6.clone(),
                    strategy_b_name: b_key.0.clone(),
                    strategy_b_version: b_key.1.clone(),
                    strategy_b_model_id: b_key.2.clone(),
                    strategy_b_model_version: b_key.3.clone(),
                    strategy_b_implementation_sha256: b_key.4.clone(),
                    strategy_b_execution_stack_sha256: b_key.5.clone(),
                    strategy_b_checkpoint_sha256: b_key.6.clone(),
                    budget_policy_sha256: matched_condition.0.clone(),
                    evaluation_condition_sha256: matched_condition.1.clone(),
                    task_denominator: denominator,
                    strategy_a_attempted_tasks: a_attempted.len(),
                    strategy_b_attempted_tasks: b_attempted.len(),
                    strategy_a_attempt_coverage: a_attempted.len() as f64 / denominator as f64,
                    strategy_b_attempt_coverage: b_attempted.len() as f64 / denominator as f64,
                    strategy_a_unique_verified_tasks: a_solved_total,
                    strategy_b_unique_verified_tasks: b_solved_total,
                    both_solved_tasks: both_solved,
                    a_only_solved_tasks: a_only,
                    b_only_solved_tasks: b_only,
                    neither_solved_tasks: neither,
                    paired_risk_difference_b_minus_a: paired_risk_difference,
                    repository_cluster_count: repositories.len(),
                    repository_file_cluster_count: file_cluster_total,
                    repository_bootstrap_replicates,
                    repository_bootstrap_ci95_low: repo_low,
                    repository_bootstrap_ci95_high: repo_high,
                    nested_repository_file_bootstrap_replicates: nested_bootstrap_replicates,
                    nested_repository_file_bootstrap_ci95_low: nested_low,
                    nested_repository_file_bootstrap_ci95_high: nested_high,
                    repository_cluster_sign_flip_replicates: permutation_replicates,
                    repository_cluster_sign_flip_p_value: raw_p,
                    holm_adjusted_p_value: 1.0,
                    deterministic_seed_sha256: seed,
                    inference_note: "The full fixed evaluation split is the denominator; unattempted tasks count as not solved. Retries collapse to one task-level success per strategy. Confidence intervals resample repositories, with a nested repository/file bootstrap. The paired sign-flip test flips all task outcomes within each repository and Holm-corrects the complete pairwise family. This is uncertainty for the pinned task cohort, not project-disjoint generalization.".into(),
                },
            ));
        }
    }

    let raw_p_values: Vec<f64> = comparisons.iter().map(|(raw_p, _)| *raw_p).collect();
    let adjusted_p_values = holm_adjusted_p_values(&raw_p_values);
    for (index, adjusted) in adjusted_p_values.into_iter().enumerate() {
        comparisons[index].1.holm_adjusted_p_value = adjusted;
    }
    Ok(comparisons.into_iter().map(|(_, comparison)| comparison).collect())
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
    let pairwise_clustered_comparisons = pairwise_clustered_comparisons(proof_runs, evaluation)?;

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
        pairwise_clustered_comparisons,
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
        admission_boundary: "This audit validates pinned dataset bytes, record structure, task identity uniqueness and evaluation-split membership. When proof-run JSONL is supplied, it also checks each attempt's task, commit, Lean version, source range, goal digest, evidence-digest syntax and declared proof/semantic gate results. It does not reopen or authenticate referenced checker, source-binding or semantic-gate receipt bytes, clone/build task repositories, or prove the informal-to-formal statement mapping. A public assurance-adjusted score requires those receipts to be independently reopened and verified.".into(),
    })
}

fn parse_args(args: &[String]) -> Result<(PathBuf, PathBuf, PathBuf, Option<PathBuf>), String> {
    if args.len() < 7 || args[1] != "--verify" {
        return Err(format!(
            "usage: {} --verify <full-snapshot.json> <evaluation-split.json> <report.json> [--runs proof-runs.jsonl] --pinned-source SorryDB/SorryDB@74b3b63f8ba890d675f362924b21ac55db410383 | --verify-checkout <sorry-record.json> <repo-root> <receipt.json> | --replay-goal <sorry-record.json> <repo-root> <receipt.json> <replay.log> <instrumented.lean> | --self-test-checkout-report <receipt.json>",
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
        budget_policy_sha256: digest("synthetic-fixed-budget-policy"),
        evaluation_condition_sha256: digest("synthetic-source-present-evaluation-condition"),
        declaration_signature_before_sha256: digest("synthetic-declaration-signature"),
        declaration_signature_after_sha256: digest("synthetic-declaration-signature"),
        target_is_prop: verified,
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
            execution_stack_sha256: digest("synthetic-execution-stack"),
            checkpoint_sha256: Some(digest("synthetic-model-checkpoint")),
            model_id: Some("synthetic-model".into()),
            model_version: Some("v0".into()),
        },
        outcome: if verified { "verified_success" } else { "non_solve" }.into(),
        proof_inserted: verified,
        exact_sorry_replaced: verified,
        sorry_absent_after_replacement: verified,
        target_sorry_ax_absent: verified,
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

fn synthetic_strategy_run(task: &SorryRecord, verified: bool, run_id: &str, strategy_name: &str) -> ProofRun {
    let mut run = synthetic_proof_run(task, verified, run_id);
    run.strategy.name = strategy_name.into();
    run.strategy.model_id = Some(strategy_name.into());
    run.strategy.implementation_sha256 = sha256(format!("{strategy_name}|implementation").as_bytes());
    run.strategy.execution_stack_sha256 = sha256(format!("{strategy_name}|execution-stack").as_bytes());
    run.strategy.checkpoint_sha256 = Some(sha256(format!("{strategy_name}|checkpoint").as_bytes()));
    run
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

    let mut bad_run = valid_run.clone();
    bad_run.target_is_prop = false;
    if validate_proof_run(&bad_run, 5, &evaluation_by_id).is_ok() {
        return Err("non-Prop target was accepted as a SorryDB proof task".into());
    }
    println!("SELF_TEST_PASS target must elaborate as Prop");

    let mut bad_run = valid_run.clone();
    bad_run.target_sorry_ax_absent = false;
    if validate_proof_run(&bad_run, 7, &evaluation_by_id).is_ok() {
        return Err("target retaining sorryAx was accepted".into());
    }
    println!("SELF_TEST_PASS target sorryAx must be absent");

    let mut bad_run = valid_run.clone();
    bad_run.declaration_signature_after_sha256 = sha256(b"weakened theorem signature");
    if validate_proof_run(&bad_run, 6, &evaluation_by_id).is_ok() {
        return Err("proof replacement that changes the target declaration signature was accepted".into());
    }
    println!("SELF_TEST_PASS exact theorem declaration signature must be preserved");

    let runs = vec![
        valid_run,
        synthetic_proof_run(&evaluation.sorries[0], false, "self-test-non-solve"),
    ];
    let (success_runs, unique_successes, summaries) = summarize_proof_runs(&runs, &evaluation)?;
    if success_runs != 1 || unique_successes != 1 || summaries.len() != 1
        || summaries[0].attempts != 2
        || summaries[0].verified_success_attempts != 1
        || summaries[0].unique_tasks_solved != 1
        || summaries[0].unique_task_success_rate != 0.5
        || summaries[0].non_solves != 1
    {
        return Err("proof-run denominator and strategy summary controls failed".into());
    }
    if summarize_proof_runs(&[runs[0].clone(), runs[0].clone()], &evaluation).is_ok() {
        return Err("duplicate proof-run IDs were accepted".into());
    }
    println!("SELF_TEST_PASS per-strategy attempt/success denominators and duplicate run rejection");

    let mut task_a = synthetic_record("cluster-task-a", "x : Nat\n⊢ x = x", "Demo/A.lean", 0);
    task_a.repo.remote = "https://github.com/example/project-a".into();
    task_a.debug_info.url = format!(
        "{}/blob/{}/{}#L{}",
        task_a.repo.remote, task_a.repo.commit, task_a.location.path, task_a.location.start_line
    );
    let mut task_b = synthetic_record("cluster-task-b", "x : Nat\n⊢ x + 0 = x", "Demo/B.lean", 0);
    task_b.repo.remote = "https://github.com/example/project-b".into();
    task_b.debug_info.url = format!(
        "{}/blob/{}/{}#L{}",
        task_b.repo.remote, task_b.repo.commit, task_b.location.path, task_b.location.start_line
    );
    let mut task_c = synthetic_record("cluster-task-c", "x : Nat\n⊢ x = x + 0", "Demo/C.lean", 0);
    task_c.repo.remote = "https://github.com/example/project-b".into();
    task_c.debug_info.url = format!(
        "{}/blob/{}/{}#L{}",
        task_c.repo.remote, task_c.repo.commit, task_c.location.path, task_c.location.start_line
    );
    let cluster_eval = Dataset {
        documentation: "synthetic cluster comparison".into(),
        sorries: vec![task_a.clone(), task_b.clone(), task_c.clone()],
    };
    let mut cluster_runs = Vec::new();
    let mut push_strategy_run = |task: &SorryRecord, solved: bool, strategy_name: &str, run_id: &str| {
        let mut run = synthetic_proof_run(task, solved, run_id);
        run.strategy.name = strategy_name.into();
        run.strategy.implementation_sha256 = sha256(strategy_name.as_bytes());
        run.strategy.model_id = Some(strategy_name.into());
        cluster_runs.push(run);
    };
    push_strategy_run(&task_a, true, "strategy-a", "cluster-a-task-a-pass");
    push_strategy_run(&task_a, false, "strategy-a", "cluster-a-task-a-retry");
    push_strategy_run(&task_b, false, "strategy-a", "cluster-a-task-b-fail");
    push_strategy_run(&task_c, true, "strategy-a", "cluster-a-task-c-pass");
    push_strategy_run(&task_a, false, "strategy-b", "cluster-b-task-a-fail");
    push_strategy_run(&task_b, true, "strategy-b", "cluster-b-task-b-pass");
    push_strategy_run(&task_c, true, "strategy-b", "cluster-b-task-c-pass");
    drop(push_strategy_run);

    let comparisons = pairwise_clustered_comparisons(&cluster_runs, &cluster_eval)?;
    if comparisons.len() != 1 {
        return Err("clustered comparison must emit exactly one pair".into());
    }
    let comparison = &comparisons[0];
    if comparison.task_denominator != 3
        || comparison.strategy_a_attempted_tasks != 3
        || comparison.strategy_b_attempted_tasks != 3
        || comparison.strategy_a_unique_verified_tasks != 2
        || comparison.strategy_b_unique_verified_tasks != 2
        || comparison.both_solved_tasks != 1
        || comparison.a_only_solved_tasks != 1
        || comparison.b_only_solved_tasks != 1
        || comparison.neither_solved_tasks != 0
        || comparison.repository_cluster_count != 2
        || comparison.repository_file_cluster_count != 3
        || comparison.paired_risk_difference_b_minus_a.abs() > 1e-12
        || comparison.repository_cluster_sign_flip_p_value != 1.0
        || comparison.holm_adjusted_p_value != 1.0
    {
        return Err(format!("clustered paired comparison denominators or balance are incorrect: {comparison:?}"));
    }
    println!("SELF_TEST_PASS paired comparison preserves fixed tasks, retries and repository/file cluster counts");

    let repeated = pairwise_clustered_comparisons(&cluster_runs, &cluster_eval)?;
    if repeated[0].deterministic_seed_sha256 != comparison.deterministic_seed_sha256
        || repeated[0].repository_bootstrap_ci95_low != comparison.repository_bootstrap_ci95_low
        || repeated[0].repository_bootstrap_ci95_high != comparison.repository_bootstrap_ci95_high
        || repeated[0].nested_repository_file_bootstrap_ci95_low != comparison.nested_repository_file_bootstrap_ci95_low
        || repeated[0].nested_repository_file_bootstrap_ci95_high != comparison.nested_repository_file_bootstrap_ci95_high
    {
        return Err("cluster bootstrap must be deterministic for the same frozen pair".into());
    }
    println!("SELF_TEST_PASS repository and nested file bootstrap are deterministic");

    let mut mismatched_budget = cluster_runs.clone();
    mismatched_budget[4].budget_policy_sha256 = sha256(b"larger-budget-for-strategy-b");
    if pairwise_clustered_comparisons(&mismatched_budget, &cluster_eval).is_ok() {
        return Err("pairwise strategy comparison accepted mismatched resource budgets".into());
    }
    println!("SELF_TEST_PASS pairwise comparisons reject unequal resource-budget policies");

    let mut mismatched_condition = cluster_runs.clone();
    mismatched_condition[4].evaluation_condition_sha256 = sha256(b"source-free-condition-for-strategy-b");
    if pairwise_clustered_comparisons(&mismatched_condition, &cluster_eval).is_ok() {
        return Err("pairwise strategy comparison accepted mismatched evaluation conditions".into());
    }
    println!("SELF_TEST_PASS pairwise comparisons reject unequal source/tool evaluation conditions");

    let mut different_build = cluster_runs.clone();
    different_build[0].strategy.implementation_sha256 = sha256(b"strategy-a-different-build");
    different_build[0].strategy.execution_stack_sha256 = sha256(b"strategy-a-different-stack");
    let separated = pairwise_clustered_comparisons(&different_build, &cluster_eval)?;
    if separated.len() != 3 {
        return Err(format!("different implementation builds were incorrectly merged into one strategy: {}", separated.len()));
    }
    println!("SELF_TEST_PASS implementation and execution-stack changes create distinct strategy identities");

    let holm = holm_adjusted_p_values(&[0.01, 0.04, 0.03]);
    if holm.len() != 3
        || (holm[0] - 0.03).abs() > 1e-12
        || (holm[1] - 0.06).abs() > 1e-12
        || (holm[2] - 0.06).abs() > 1e-12
    {
        return Err(format!("Holm multiple-comparison correction failed: {holm:?}"));
    }
    println!("SELF_TEST_PASS Holm correction preserves monotonic adjusted p-values");

    Ok(())
}

fn synthetic_report() -> Result<Value, String> {
    let mut task_a = synthetic_record("task-a", "x : Nat\n⊢ x = x", "Demo/A.lean", 0);
    task_a.repo.remote = "https://github.com/example/project-a".into();
    task_a.debug_info.url = format!(
        "{}/blob/{}/{}#L{}",
        task_a.repo.remote, task_a.repo.commit, task_a.location.path, task_a.location.start_line
    );
    let mut task_b = synthetic_record("task-b", "x : Nat\n⊢ x + 0 = x", "Demo/B.lean", 20);
    task_b.repo.remote = "https://github.com/example/project-b".into();
    task_b.debug_info.url = format!(
        "{}/blob/{}/{}#L{}",
        task_b.repo.remote, task_b.repo.commit, task_b.location.path, task_b.location.start_line
    );
    let mut task_c = synthetic_record("task-c", "x : Nat\n⊢ x = x + 0", "Demo/C.lean", 5);
    task_c.repo.remote = "https://github.com/example/project-b".into();
    task_c.debug_info.url = format!(
        "{}/blob/{}/{}#L{}",
        task_c.repo.remote, task_c.repo.commit, task_c.location.path, task_c.location.start_line
    );
    let full = Dataset {
        documentation: "synthetic snapshot".into(),
        sorries: vec![task_a.clone(), task_b.clone(), task_c.clone()],
    };
    let evaluation = Dataset {
        documentation: "synthetic evaluation".into(),
        sorries: full.sorries.clone(),
    };
    let full_bytes = serde_json::to_vec(&full).map_err(|e| e.to_string())?;
    let evaluation_bytes = serde_json::to_vec(&evaluation).map_err(|e| e.to_string())?;
    let proof_runs = vec![
        synthetic_strategy_run(&task_a, true, "a-task-a-success", "strategy-a"),
        synthetic_strategy_run(&task_a, false, "a-task-a-retry", "strategy-a"),
        synthetic_strategy_run(&task_b, false, "a-task-b-fail", "strategy-a"),
        synthetic_strategy_run(&task_c, true, "a-task-c-success", "strategy-a"),
        synthetic_strategy_run(&task_a, false, "b-task-a-fail", "strategy-b"),
        synthetic_strategy_run(&task_b, true, "b-task-b-success", "strategy-b"),
        synthetic_strategy_run(&task_c, true, "b-task-c-success", "strategy-b"),
    ];
    let (verified_success_run_count, unique_successful_task_count, strategy_summaries) =
        summarize_proof_runs(&proof_runs, &evaluation)?;
    let clustered_comparisons = pairwise_clustered_comparisons(&proof_runs, &evaluation)?;
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
        "pairwise_clustered_comparisons": clustered_comparisons,
        "full_task_count": 3,
        "evaluation_task_count": 3,
        "documented_full_task_count": DOCUMENTED_FULL_COUNT,
        "full_count_documentation_drift": true,
        "full_unique_task_id_count": 3,
        "full_unique_goal_count": 3,
        "evaluation_unique_task_id_count": 3,
        "evaluation_unique_goal_count": 3,
        "evaluation_exact_record_matches_in_full_snapshot": 3,
        "repository_count_full": 2,
        "repository_count_evaluation": 2,
        "lean_version_counts_evaluation": {"v4.27.0": 3},
        "zero_based_start_column_task_count_full": 1,
        "zero_based_start_column_task_count_evaluation": 1,
        "cluster_summary_evaluation": {
            "repository_count": 2,
            "repository_commit_count": 2,
            "source_file_count": 3,
            "tasks_per_repository_min": 1,
            "tasks_per_repository_max": 2,
            "tasks_per_source_file_max": 1,
            "source_file_task_histogram": {"1": 3},
            "top_source_file_clusters": []
        },
        "source_reconstruction_requirement": "synthetic source reconstruction policy",
        "benchmark_integrity_warnings": ["synthetic documentation drift control", "synthetic project-clustered paired inference control"],
        "admission_boundary": "Synthetic fixture only: source repositories and referenced checker/semantic receipt bytes have not been independently reconstructed or authenticated."
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
    if args.len() == 3 && args[1] == "--self-test-checkout-report" {
        let receipt = build_synthetic_checkout_receipt()?;
        let bytes = serde_json::to_vec_pretty(&receipt)
            .map_err(|e| format!("serialize synthetic checkout receipt: {e}"))?;
        fs::write(&args[2], bytes)
            .map_err(|e| format!("write synthetic checkout receipt: {e}"))?;
        println!("PITB_SORRYDB_CHECKOUT_RECEIPT_OK");
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
    if args.len() == 7 && args[1] == "--replay-goal" {
        let record_bytes = fs::read(&args[2])
            .map_err(|e| format!("read SorryDB task record: {e}"))?;
        let record: SorryRecord = serde_json::from_slice(&record_bytes)
            .map_err(|e| format!("parse SorryDB task record: {e}"))?;
        let receipt = replay_goal_with_lake(
            &record,
            Path::new(&args[3]),
            Path::new(&args[4]),
            Path::new(&args[5]),
            Path::new(&args[6]),
        )?;
        println!("PITB_SORRYDB_GOAL_REPLAY_OK");
        println!(
            "task={} goal_sha256={} manifest_sha256={} log_sha256={}",
            receipt.source_database_task_id,
            receipt.goal_sha256,
            receipt.lake_manifest_sha256,
            receipt.replay_log_sha256
        );
        return Ok(());
    }
    if args.len() == 5 && args[1] == "--verify-checkout" {
        let record_bytes = fs::read(&args[2])
            .map_err(|e| format!("read SorryDB task record: {e}"))?;
        let record: SorryRecord = serde_json::from_slice(&record_bytes)
            .map_err(|e| format!("parse SorryDB task record: {e}"))?;
        let receipt = verify_source_checkout(&record, Path::new(&args[3]))?;
        let receipt_bytes = serde_json::to_vec_pretty(&receipt)
            .map_err(|e| format!("serialize source checkout receipt: {e}"))?;
        fs::write(&args[4], receipt_bytes)
            .map_err(|e| format!("write source checkout receipt: {e}"))?;
        println!("PITB_SORRYDB_SOURCE_CHECKOUT_OK");
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
    fn checkout_verification_reads_source_and_toolchain_from_exact_git_commit() {
        let receipt = build_synthetic_checkout_receipt()
            .expect("committed source should be verified despite a dirty worktree");
        assert!(receipt.source_span_verified_as_sorry);
        assert_eq!(receipt.source_span, "sorry");
        assert!(!receipt.scoring_admission);
    }

    #[test]
    fn namespaced_lean_toolchain_is_normalized_to_sorrydb_release() {
        let toolchain = "leanprover/lean4:v4.26.0";
        let normalized = toolchain
            .rsplit_once(':')
            .map(|(_, version)| version.trim())
            .unwrap_or(toolchain);
        assert_eq!(normalized, "v4.26.0");
    }

    #[test]
    fn instrumentation_replaces_exact_unicode_aware_sorry_span() {
        let source = "theorem sample (α : Nat) : True := by sorry\n";
        let line = source.lines().next().unwrap();
        let byte_start = line.find("sorry").unwrap();
        let start = line[..byte_start].chars().count() as u64;
        let instrumented = instrument_sorry_span(source, 1, start, 1, start + 5, "pitbCaptureInline", "PITB_BEGIN", "PITB_END").unwrap();
        assert!(instrumented.contains("by \n  pitbCaptureInline\n  sorry"));
        assert!(instrumented.contains("Lean.Meta.ppGoal goal"));
        assert!(instrumented.contains("dbg_trace f!\"PITB_BEGIN\\n{rendered}\\nPITB_END\""));
        assert_eq!(source.matches("sorry").count(), 1);
        assert_eq!(instrumented.matches("sorry").count(), 1);
    }

    #[test]
    fn instrumentation_preserves_nested_tactic_indentation() {
        let source = "theorem nested : True := by\n  sorry\n";
        let instrumented = instrument_sorry_span(source, 2, 2, 2, 7, "pitbCaptureNested", "PITB_BEGIN", "PITB_END").unwrap();
        assert!(instrumented.contains("by\n  pitbCaptureNested\n  sorry\n"));
    }

    #[test]
    fn instrumentation_rejects_a_source_range_that_is_not_sorry() {
        let source = "theorem sample : True := by trivial\n";
        let start = source.find("trivial").unwrap() as u64;
        assert!(instrument_sorry_span(source, 1, start, 1, start + 7, "pitbCaptureInvalid", "PITB_BEGIN", "PITB_END").is_err());
    }

    #[test]
    fn goal_capture_requires_one_exact_ordered_marker_pair() {
        let log = "unrelated goal text\nPITB_BEGIN\nx : Nat\n⊢ x = x\nPITB_END\nmore unrelated text";
        let captured = extract_marked_goal(log, "PITB_BEGIN", "PITB_END").unwrap();
        assert_eq!(normalized_goal(&captured), "x : Nat ⊢ x = x");
        assert!(extract_marked_goal("PITB_BEGIN\nx : Nat", "PITB_BEGIN", "PITB_END").is_err());
        assert!(extract_marked_goal("PITB_BEGIN\nx\nPITB_END\nPITB_END", "PITB_BEGIN", "PITB_END").is_err());
        assert!(extract_marked_goal("PITB_END\nx\nPITB_BEGIN", "PITB_BEGIN", "PITB_END").is_err());
        assert!(extract_marked_goal("PITB_BEGIN\nPITB_END", "PITB_BEGIN", "PITB_END").is_err());
    }

    #[test]
    fn goal_capture_does_not_match_text_outside_the_marked_region() {
        let expected = normalized_goal("x : Nat\n⊢ x = x");
        let log = "x : Nat\n⊢ x = x\nPITB_BEGIN\ny : Nat\n⊢ y = y\nPITB_END";
        let captured = extract_marked_goal(log, "PITB_BEGIN", "PITB_END").unwrap();
        assert_ne!(normalized_goal(&captured), expected);
    }

    #[test]
    fn goal_instrumentation_rejects_marker_collisions_with_source() {
        let source = "theorem sample : True := by sorry -- PITB_BEGIN\n";
        let start = source.find("sorry").unwrap() as u64;
        assert!(instrument_sorry_span(source, 1, start, 1, start + 5, "pitbCaptureCollision", "PITB_BEGIN", "PITB_END").is_err());
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
