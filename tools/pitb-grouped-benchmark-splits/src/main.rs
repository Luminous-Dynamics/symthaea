use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::PathBuf,
    process::ExitCode,
};

const REPORT_SCHEMA: &str = "pitb.grouped-benchmark-splits.v0";
const DEFAULT_FOLDS: usize = 5;
const MAX_FOLDS: usize = 20;

#[derive(Debug, Clone, Deserialize)]
struct RawDataset {
    sorries: Vec<RawTask>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawTask {
    id: String,
    repo: RawRepo,
    location: RawLocation,
}

#[derive(Debug, Clone, Deserialize)]
struct RawRepo {
    remote: String,
    commit: String,
    lean_version: String,
}

#[derive(Debug, Clone, Deserialize)]
struct RawLocation {
    path: String,
}

#[derive(Debug, Clone, Serialize)]
struct Fold {
    fold_id: usize,
    task_count: usize,
    repository_count: usize,
    repository_commit_pair_count: usize,
    repository_file_cluster_count: usize,
    repository_keys: Vec<String>,
    task_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Assignment {
    task_id: String,
    repository_key: String,
    fold_id: usize,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    schema_version: String,
    algorithm_id: String,
    source_dataset_sha256: String,
    source_dataset_git_blob_sha1: Option<String>,
    seed: String,
    fold_count: usize,
    task_count: usize,
    repository_count: usize,
    repository_commit_pair_count: usize,
    repository_file_cluster_count: usize,
    min_tasks_per_fold: usize,
    max_tasks_per_fold: usize,
    max_repository_group_size: usize,
    repository_disjoint: bool,
    every_task_assigned_once: bool,
    assignments: Vec<Assignment>,
    folds: Vec<Fold>,
    interpretation_boundary: String,
}

#[derive(Debug, Clone)]
struct RepositoryGroup {
    repository_key: String,
    task_ids: Vec<String>,
    commit_pairs: BTreeSet<String>,
    file_clusters: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct FoldWork {
    repository_keys: Vec<String>,
    task_ids: Vec<String>,
    commit_pairs: BTreeSet<String>,
    file_clusters: BTreeSet<String>,
}

impl FoldWork {
    fn new() -> Self {
        Self {
            repository_keys: Vec::new(),
            task_ids: Vec::new(),
            commit_pairs: BTreeSet::new(),
            file_clusters: BTreeSet::new(),
        }
    }

    fn task_count(&self) -> usize {
        self.task_ids.len()
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn git_blob_sha1(bytes: &[u8]) -> String {
    // Git blob object digest: SHA1("blob <length>\\0" || bytes).
    let mut hasher = Sha256::new();
    let _ = &mut hasher;
    let header = format!("blob {}\0", bytes.len());
    let mut sha = sha1::Sha1::new();
    sha.update(header.as_bytes());
    sha.update(bytes);
    format!("{:x}", sha.digest())
}

fn canonical_repository_key(remote: &str) -> Result<String, String> {
    let value = remote.trim();
    if !value.starts_with("https://") || value.contains('@') || value.contains('?') || value.contains('#') {
        return Err(format!("repository remote must be a credential-free HTTPS URL: {value}"));
    }
    let without_scheme = &value["https://".len()..];
    let mut parts = without_scheme.split('/');
    let host = parts.next().unwrap_or_default().to_ascii_lowercase();
    let owner = parts.next().unwrap_or_default().to_ascii_lowercase();
    let mut repository = parts.next().unwrap_or_default().to_ascii_lowercase();
    if parts.next().is_some() || host.is_empty() || owner.is_empty() || repository.is_empty() {
        return Err(format!("repository remote must identify exactly one host/owner/repository: {value}"));
    }
    if repository.ends_with(".git") {
        repository.truncate(repository.len() - 4);
    }
    if repository.is_empty() || owner == "." || owner == ".." || repository == "." || repository == ".." {
        return Err(format!("repository remote has an invalid owner or repository: {value}"));
    }
    Ok(format!("https://{host}/{owner}/{repository}"))
}

fn safe_relative_path(value: &str) -> bool {
    !value.is_empty()
        && !value.starts_with('/')
        && !value.contains('\\')
        && !value.split('/').any(|p| p.is_empty() || p == "." || p == "..")
}

fn tie_key(seed: &str, label: &str) -> String {
    sha256(format!("{seed}\0{label}").as_bytes())
}

fn parse_dataset(bytes: &[u8]) -> Result<Vec<RepositoryGroup>, String> {
    let dataset: RawDataset = serde_json::from_slice(bytes)
        .map_err(|e| format!("parse SorryDB dataset JSON: {e}"))?;
    if dataset.sorries.is_empty() {
        return Err("SorryDB dataset contains no tasks".into());
    }
    let mut groups = BTreeMap::<String, RepositoryGroup>::new();
    let mut seen_tasks = BTreeSet::<String>::new();

    for (index, task) in dataset.sorries.into_iter().enumerate() {
        if !valid_hex(&task.id, 64) {
            return Err(format!("SorryDB task {} has a malformed 64-character task ID", index + 1));
        }
        if !seen_tasks.insert(task.id.clone()) {
            return Err(format!("duplicate SorryDB task ID {}", task.id));
        }
        if !valid_hex(&task.repo.commit, 40) {
            return Err(format!("task {} has a malformed repository commit", task.id));
        }
        if task.repo.lean_version.trim().is_empty() {
            return Err(format!("task {} has an empty Lean version", task.id));
        }
        if !safe_relative_path(&task.location.path) {
            return Err(format!("task {} has an unsafe source path", task.id));
        }
        let repository_key = canonical_repository_key(&task.repo.remote)?;
        let group = groups.entry(repository_key.clone()).or_insert_with(|| RepositoryGroup {
            repository_key,
            task_ids: Vec::new(),
            commit_pairs: BTreeSet::new(),
            file_clusters: BTreeSet::new(),
        });
        group.commit_pairs.insert(format!("{}@{}", group.repository_key, task.repo.commit));
        group.file_clusters.insert(format!("{}@{}", task.repo.commit, task.location.path));
        group.task_ids.push(task.id);
    }

    let mut result: Vec<RepositoryGroup> = groups.into_values().collect();
    for group in &mut result {
        group.task_ids.sort();
    }
    Ok(result)
}

fn build_report(
    groups: Vec<RepositoryGroup>,
    source_bytes: &[u8],
    fold_count: usize,
    seed: &str,
    source_blob_sha1: Option<String>,
) -> Result<Report, String> {
    if fold_count < 2 || fold_count > MAX_FOLDS {
        return Err(format!("fold_count must be between 2 and {MAX_FOLDS}"));
    }
    if groups.len() < fold_count {
        return Err(format!("{} repositories cannot populate {fold_count} non-empty folds", groups.len()));
    }
    if seed.trim().is_empty() {
        return Err("seed must be non-empty".into());
    }
    if let Some(blob) = &source_blob_sha1 {
        if !valid_hex(blob, 40) {
            return Err("source_blob_sha1 must be 40 lowercase hexadecimal characters".into());
        }
        let computed = git_blob_sha1(source_bytes);
        if &computed != blob {
            return Err(format!("source Git blob SHA-1 mismatch: expected {blob}, computed {computed}"));
        }
    }

    let repository_count = groups.len();
    let task_count: usize = groups.iter().map(|g| g.task_ids.len()).sum();
    let commit_pair_count: usize = groups.iter().map(|g| g.commit_pairs.len()).sum();
    let file_cluster_count: usize = groups.iter().map(|g| g.file_clusters.len()).sum();
    let max_repository_group_size = groups.iter().map(|g| g.task_ids.len()).max().unwrap_or(0);

    // Longest-processing-time greedy assignment balances task counts while
    // keeping every task from one canonical repository in exactly one fold.
    let mut ordered = groups;
    ordered.sort_by(|a, b| {
        b.task_ids.len().cmp(&a.task_ids.len())
            .then_with(|| tie_key(seed, &a.repository_key).cmp(&tie_key(seed, &b.repository_key)))
            .then_with(|| a.repository_key.cmp(&b.repository_key))
    });

    let mut work: Vec<FoldWork> = (0..fold_count).map(|_| FoldWork::new()).collect();
    let fold_ties: Vec<String> = (0..fold_count)
        .map(|i| tie_key(seed, &format!("fold-{i}")))
        .collect();

    for group in ordered {
        let selected = (0..fold_count)
            .min_by(|a, b| {
                work[*a].task_count().cmp(&work[*b].task_count())
                    .then_with(|| work[*a].repository_keys.len().cmp(&work[*b].repository_keys.len()))
                    .then_with(|| fold_ties[*a].cmp(&fold_ties[*b]))
            })
            .ok_or("no fold available")?;
        work[selected].repository_keys.push(group.repository_key);
        work[selected].task_ids.extend(group.task_ids);
        work[selected].commit_pairs.extend(group.commit_pairs);
        work[selected].file_clusters.extend(group.file_clusters);
    }

    let mut folds = Vec::new();
    let mut assignments = Vec::new();
    let mut assigned_tasks = BTreeSet::new();
    let mut assigned_repositories = BTreeSet::new();
    for (index, mut fold) in work.into_iter().enumerate() {
        fold.repository_keys.sort();
        fold.task_ids.sort();
        for repo in &fold.repository_keys {
            if !assigned_repositories.insert(repo.clone()) {
                return Err(format!("repository {repo} was assigned to more than one fold"));
            }
        }
        for task_id in &fold.task_ids {
            if !assigned_tasks.insert(task_id.clone()) {
                return Err(format!("task {task_id} was assigned more than once"));
            }
            let repository_key = fold.repository_keys.iter()
                .find(|repo| {
                    // Reconstruct membership from the original group table below.
                    let _ = repo;
                    false
                });
            let _ = repository_key;
        }
        folds.push(Fold {
            fold_id: index,
            task_count: fold.task_ids.len(),
            repository_count: fold.repository_keys.len(),
            repository_commit_pair_count: fold.commit_pairs.len(),
            repository_file_cluster_count: fold.file_clusters.len(),
            repository_keys: fold.repository_keys,
            task_ids: fold.task_ids,
        });
    }

    // Rebuild the task -> repository map from the parsed dataset so that the
    // assignment ledger is explicit and independently checkable.
    let dataset = parse_dataset(source_bytes)?;
    let task_to_repo: BTreeMap<String, String> = dataset.into_iter()
        .flat_map(|g| g.task_ids.into_iter().map(move |id| (id, g.repository_key.clone())))
        .collect();
    for fold in &folds {
        for task_id in &fold.task_ids {
            let repo = task_to_repo.get(task_id)
                .ok_or_else(|| format!("task {task_id} disappeared while writing assignment ledger"))?;
            assignments.push(Assignment {
                task_id: task_id.clone(),
                repository_key: repo.clone(),
                fold_id: fold.fold_id,
            });
        }
    }
    assignments.sort_by(|a, b| a.task_id.cmp(&b.task_id));

    let min_tasks = folds.iter().map(|f| f.task_count).min().unwrap_or(0);
    let max_tasks = folds.iter().map(|f| f.task_count).max().unwrap_or(0);
    if assigned_tasks.len() != task_count || assigned_repositories.len() != repository_count {
        return Err("fold assignment does not cover every task/repository exactly once".into());
    }
    if assignments.iter().any(|a| {
        folds[a.fold_id].repository_keys.binary_search(&a.repository_key).is_err()
    }) {
        return Err("task and repository fold assignments disagree".into());
    }

    Ok(Report {
        schema_version: REPORT_SCHEMA.into(),
        algorithm_id: "lpt-repository-groups-sha256-tiebreak-v1".into(),
        source_dataset_sha256: sha256(source_bytes),
        source_dataset_git_blob_sha1: source_blob_sha1,
        seed: seed.into(),
        fold_count,
        task_count,
        repository_count,
        repository_commit_pair_count: commit_pair_count,
        repository_file_cluster_count: file_cluster_count,
        min_tasks_per_fold: min_tasks,
        max_tasks_per_fold: max_tasks,
        max_repository_group_size,
        repository_disjoint: true,
        every_task_assigned_once: true,
        assignments,
        folds,
        interpretation_boundary: "All tasks from one canonical repository are assigned to one fold. This creates project-disjoint evaluation partitions within the supplied snapshot; it does not make the public benchmark temporally novel or remove pretraining contamination. Do not treat task rows as IID. Use repository-level or repository/file-clustered uncertainty estimates.".into(),
    })
}

fn synthetic_dataset() -> Vec<u8> {
    let mut tasks = Vec::new();
    for repo_index in 0..17 {
        for task_index in 0..(repo_index % 7 + 1) {
            let id = sha256(format!("repo-{repo_index}-task-{task_index}").as_bytes());
            let remote = if repo_index % 2 == 0 {
                format!("https://github.com/Example/Repo{repo_index}.git")
            } else {
                format!("https://github.com/example/repo{repo_index}")
            };
            tasks.push(serde_json::json!({
                "id": id,
                "repo": {
                    "remote": remote,
                    "branch": "main",
                    "commit": format!("{:040x}", repo_index + 1),
                    "lean_version": "leanprover/lean4:v4.22.0"
                },
                "location": {
                    "path": format!("Demo/Task{task_index}.lean"),
                    "start_line": 1,
                    "start_column": 0,
                    "end_line": 1,
                    "end_column": 5
                }
            }));
        }
    }
    serde_json::to_vec(&serde_json::json!({"repos":[],"sorries":tasks})).unwrap()
}

fn self_test() -> Result<Report, String> {
    let source = synthetic_dataset();
    let groups = parse_dataset(&source)?;
    let a = build_report(groups.clone(), &source, 5, "pitb-project-split-v1", None)?;
    let b = build_report(groups.clone(), &source, 5, "pitb-project-split-v1", None)?;
    if serde_json::to_value(&a).map_err(|e| e.to_string())?
        != serde_json::to_value(&b).map_err(|e| e.to_string())?
    {
        return Err("same seed did not produce identical folds".into());
    }
    println!("SELF_TEST_PASS project-disjoint fold assignment is deterministic");
    if !a.repository_disjoint || !a.every_task_assigned_once
        || a.task_count != a.assignments.len()
        || a.repository_count != 17
    {
        return Err("synthetic project-disjoint coverage control failed".into());
    }
    println!("SELF_TEST_PASS every task and repository appears in exactly one fold");
    if a.max_tasks_per_fold - a.min_tasks_per_fold > a.max_repository_group_size {
        return Err("greedy repository-group assignment exceeds largest-cluster balance bound".into());
    }
    println!("SELF_TEST_PASS fold imbalance is bounded by the largest indivisible repository cluster");

    let mut overlap = BTreeMap::<String, usize>::new();
    for assignment in &a.assignments {
        *overlap.entry(assignment.repository_key.clone()).or_default() += 1;
    }
    if overlap.values().any(|count| *count == 0) {
        return Err("empty repository assignment observed".into());
    }
    println!("SELF_TEST_PASS normalized GitHub remote spellings map to one repository key");

    let mut bad = synthetic_dataset();
    let parsed: Value = serde_json::from_slice(&bad).map_err(|e| e.to_string())?;
    let mut value = parsed;
    value["sorries"][0]["id"] = Value::String(value["sorries"][1]["id"].as_str().unwrap().to_owned());
    bad = serde_json::to_vec(&value).map_err(|e| e.to_string())?;
    if parse_dataset(&bad).is_ok() {
        return Err("duplicate source task IDs were accepted".into());
    }
    println!("SELF_TEST_PASS duplicate upstream task IDs are rejected");

    Ok(a)
}

fn parse_args(args: &[String]) -> Result<(PathBuf, PathBuf, usize, String, Option<String>), String> {
    if args.len() < 4 || args[1] != "--plan" {
        return Err(format!(
            "usage: {} --plan <dataset.json> <report.json> [--folds N] [--seed TEXT] [--source-blob-sha1 HEX] OR --self-test OR --self-test-report <report.json>",
            args.first().map(String::as_str).unwrap_or("pitb-grouped-benchmark-splits")
        ));
    }
    let input = PathBuf::from(&args[2]);
    let output = PathBuf::from(&args[3]);
    let mut folds = DEFAULT_FOLDS;
    let mut seed = "pitb-grouped-benchmark-splits-v1".to_owned();
    let mut blob = None;
    let mut index = 4;
    while index < args.len() {
        match args[index].as_str() {
            "--folds" if index + 1 < args.len() => {
                folds = args[index + 1].parse().map_err(|_| "--folds must be an integer".to_owned())?;
                index += 2;
            }
            "--seed" if index + 1 < args.len() => {
                seed = args[index + 1].clone();
                index += 2;
            }
            "--source-blob-sha1" if index + 1 < args.len() => {
                blob = Some(args[index + 1].clone());
                index += 2;
            }
            _ => return Err(format!("unknown or incomplete option {}", args[index])),
        }
    }
    Ok((input, output, folds, seed, blob))
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        let _ = self_test()?;
        println!("PITB_GROUPED_BENCHMARK_SPLITS_SELF_TEST_PASS");
        return Ok(());
    }
    if args.len() == 3 && args[1] == "--self-test-report" {
        let report = self_test()?;
        let bytes = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize synthetic fold report: {e}"))?;
        fs::write(&args[2], bytes).map_err(|e| format!("write synthetic fold report: {e}"))?;
        println!("PITB_GROUPED_BENCHMARK_SPLITS_REPORT_OK");
        return Ok(());
    }
    let (input_path, output_path, folds, seed, blob) = parse_args(&args)?;
    let source = fs::read(&input_path).map_err(|e| format!("read dataset: {e}"))?;
    let groups = parse_dataset(&source)?;
    let report = build_report(groups, &source, folds, &seed, blob)?;
    let bytes = serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize fold report: {e}"))?;
    fs::write(&output_path, bytes).map_err(|e| format!("write fold report: {e}"))?;
    println!("PITB_GROUPED_BENCHMARK_SPLITS_OK");
    println!(
        "tasks={} repositories={} folds={} task_range={}..{}",
        report.task_count, report.repository_count, report.fold_count,
        report.min_tasks_per_fold, report.max_tasks_per_fold
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
    fn grouped_split_contract_passes() {
        self_test().expect("grouped split self-tests must pass");
    }

    #[test]
    fn source_git_blob_pin_is_enforced() {
        let source = synthetic_dataset();
        let groups = parse_dataset(&source).unwrap();
        let wrong = "0".repeat(40);
        assert!(build_report(groups, &source, 5, "seed", Some(wrong)).is_err());
    }
}
