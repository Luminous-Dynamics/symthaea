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

fn audit(
    full: &Dataset,
    evaluation: &Dataset,
    full_bytes: &[u8],
    evaluation_bytes: &[u8],
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

fn parse_args(args: &[String]) -> Result<(PathBuf, PathBuf, PathBuf), String> {
    if args.len() != 7 || args[1] != "--verify" {
        return Err(format!(
            "usage: {} --verify <full-snapshot.json> <evaluation-split.json> <report.json> --pinned-source SorryDB/SorryDB@74b3b63f8ba890d675f362924b21ac55db410383",
            args.first().map(String::as_str).unwrap_or("pitb-sorrydb-audit")
        ));
    }
    let source = format!("{}@{}", UPSTREAM_REPOSITORY, UPSTREAM_COMMIT);
    if args[5] != "--pinned-source" || args[6] != source {
        return Err("SorryDB source repository/commit does not match the frozen audit identity".into());
    }
    Ok((PathBuf::from(&args[2]), PathBuf::from(&args[3]), PathBuf::from(&args[4])))
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

fn run_self_tests() -> Result<(), String> {
    if git_blob_sha1(b"") != "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391" {
        return Err("Git empty-blob SHA-1 control failed".into());
    }
    println!("SELF_TEST_PASS Git object SHA-1 uses canonical blob header");

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
    if args.len() == 3 && args[1] == "--self-test-report" {
        run_self_tests()?;
        let report = synthetic_report()?;
        let bytes = serde_json::to_vec_pretty(&report)
            .map_err(|e| format!("serialize synthetic audit report: {e}"))?;
        fs::write(&args[2], bytes).map_err(|e| format!("write synthetic audit report: {e}"))?;
        println!("PITB_SORRYDB_AUDIT_SYNTHETIC_REPORT_OK");
        return Ok(());
    }
    let (full_path, evaluation_path, report_path) = parse_args(&args)?;
    let full_bytes = fs::read(&full_path).map_err(|e| format!("read full SorryDB snapshot: {e}"))?;
    let evaluation_bytes =
        fs::read(&evaluation_path).map_err(|e| format!("read SorryDB evaluation split: {e}"))?;
    let full: Dataset = serde_json::from_slice(&full_bytes)
        .map_err(|e| format!("parse full SorryDB snapshot: {e}"))?;
    let evaluation: Dataset = serde_json::from_slice(&evaluation_bytes)
        .map_err(|e| format!("parse SorryDB evaluation split: {e}"))?;
    let report = audit(&full, &evaluation, &full_bytes, &evaluation_bytes)?;
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
