//! Deterministic counterfactual fixture runner for a narrow BASALT subset.
//!
//! This runner creates ground truth outside the observer under test. It launches
//! a known task worker in separate child processes, mutates one fixture resource,
//! and compares task outputs and bytes-read counters. It deliberately supports
//! only paired filesystem controls currently implemented below; unsupported
//! catalog scenarios are rejected rather than synthesized.

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

const FIXTURE_CASES: &[&str] = &[
    "hidden-untracked-read",
    "unused-neighbor-change",
    "irrelevant-docs-change",
    "docs-consumed-positive-control",
];
const INDEPENDENT_TASK_OUTPUT: &[u8] = b"basalt-fixture-task-output:v1:input-independent";

#[derive(Debug, Serialize)]
pub struct FixtureRunReport {
    pub report_version: u32,
    pub run_id: String,
    pub selected_case_ids: Vec<String>,
    pub oracle_path: String,
    pub evidence_path: String,
    pub workspace_path: String,
    pub oracle_digest: String,
    pub harness_binary_digest: String,
    pub fixture_digest: String,
    pub worker_binary_digest: String,
    pub qualification_eligible: bool,
}

#[derive(Debug, Serialize)]
struct FixtureEvidence {
    format_version: u32,
    suite_id: &'static str,
    run_id: String,
    catalog_digest: String,
    selected_case_ids: Vec<String>,
    harness_binary_digest: String,
    worker_binary_digest: String,
    worker_binary_name: &'static str,
    workspace_relative_path: &'static str,
    gitignore_content_digest: String,
    git_baseline_commit: String,
    executions: Vec<ExecutionEvidence>,
    qualification_eligible: bool,
}

#[derive(Debug, Serialize)]
struct ExecutionEvidence {
    case_id: String,
    resource_id: String,
    relative_path: String,
    task_mode: String,
    git_tracking_state: String,
    git_baseline_commit: String,
    before_input_digest: String,
    after_input_digest: String,
    baseline_output_digest: String,
    mutated_output_digest: String,
    baseline_bytes_read: u64,
    mutated_bytes_read: u64,
    baseline_process_id: u32,
    mutated_process_id: u32,
    counterfactual_check: String,
}

#[derive(Debug, Deserialize)]
pub struct WorkerOutput {
    format_version: u32,
    task_mode: String,
    output_digest: String,
    bytes_read: u64,
}

#[derive(Debug, Deserialize)]
struct CatalogCase {
    case_id: String,
    channel: String,
    expected_disposition: String,
    ground_truth_resource_id: Option<String>,
    paired_case_ids: Vec<String>,
}

struct FixturePlan {
    case_id: String,
    resource_id: String,
    relative_path: PathBuf,
    mode: &'static str,
    before_bytes: Vec<u8>,
    after_bytes: Vec<u8>,
}

pub fn run(
    catalog_path: &Path,
    out_dir: &Path,
    requested_case_ids: &[String],
) -> anyhow::Result<FixtureRunReport> {
    if requested_case_ids.is_empty() {
        bail!("at least one --case-id is required");
    }

    let catalog_metadata = fs::metadata(catalog_path)
        .with_context(|| format!("inspecting challenge catalog {}", catalog_path.display()))?;
    if catalog_metadata.len() > 16 * 1024 * 1024 {
        bail!("challenge catalog exceeds the 16 MiB fixture-run limit");
    }
    let catalog_bytes = fs::read(catalog_path)
        .with_context(|| format!("reading challenge catalog {}", catalog_path.display()))?;
    let catalog_audit = crate::basalt_observer::audit_bytes(&catalog_bytes);
    if !catalog_audit.structurally_valid {
        bail!("challenge catalog failed structural validation");
    }
    let catalog: Value = serde_json::from_slice(&catalog_bytes)?;
    let profile_id = catalog
        .get("profile_id")
        .and_then(Value::as_str)
        .context("catalog.profile_id must be a string")?
        .to_string();
    let cases_value = catalog
        .get("cases")
        .and_then(Value::as_array)
        .context("catalog.cases must be an array")?;
    let cases: Vec<CatalogCase> = serde_json::from_value(Value::Array(cases_value.clone()))?;
    let cases_by_id: BTreeMap<&str, &CatalogCase> =
        cases.iter().map(|case| (case.case_id.as_str(), case)).collect();

    let mut selected = BTreeSet::<String>::new();
    for case_id in requested_case_ids {
        if !cases_by_id.contains_key(case_id.as_str()) {
            bail!("requested case ID is absent from catalog: {case_id}");
        }
        selected.insert(case_id.clone());
    }

    // Pair closure is automatic: a paired control is one indivisible experiment.
    let initially_selected: Vec<String> = selected.iter().cloned().collect();
    for case_id in initially_selected {
        let case = cases_by_id
            .get(case_id.as_str())
            .copied()
            .context("selected case disappeared from catalog index")?;
        for paired_id in &case.paired_case_ids {
            selected.insert(paired_id.clone());
        }
    }

    let ordered_ids: Vec<String> = cases
        .iter()
        .filter(|case| selected.contains(&case.case_id))
        .map(|case| case.case_id.clone())
        .collect();

    if ordered_ids.is_empty() {
        bail!("fixture selection is empty");
    }
    for case_id in &ordered_ids {
        if !FIXTURE_CASES.contains(&case_id.as_str()) {
            bail!(
                "fixture runner does not yet implement {case_id}; it will not manufacture ground truth for unsupported scenarios"
            );
        }
    }

    let harness_binary_path = std::env::current_exe()
        .context("resolving current xtask executable for oracle harness identity")?;
    let harness_binary_digest = sha256_file(&harness_binary_path)?;
    let worker_binary_name = format!(
        "basalt_observer_fixture_worker{}",
        std::env::consts::EXE_SUFFIX
    );
    let worker_binary_path = harness_binary_path.with_file_name(&worker_binary_name);
    if !worker_binary_path.is_file() {
        bail!(
            "separate fixture worker is missing at {}; build both xtask binaries with cargo build -p xtask --bins",
            worker_binary_path.display()
        );
    }
    let worker_binary_digest = sha256_file(&worker_binary_path)?;
    let now_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX epoch")?
        .as_nanos();
    let run_id = format!("basalt-{}-{now_nanos}", std::process::id());
    let run_dir = out_dir.join(&run_id);
    let workspace = run_dir.join("workspace");
    fs::create_dir_all(out_dir)
        .with_context(|| format!("creating fixture output directory {}", out_dir.display()))?;
    // Refuse collisions instead of reusing or overwriting any prior run evidence.
    fs::create_dir(&run_dir)
        .with_context(|| format!("creating unique fixture run directory {}", run_dir.display()))?;
    fs::create_dir(&workspace)
        .with_context(|| format!("creating fixture workspace {}", workspace.display()))?;

    // The fixture is a real, isolated Git repository. The hidden resource is
    // ignored/untracked; the docs control is committed/tracked at baseline.
    let gitignore = b"unlisted-secret.txt\n";
    write_new_synced(&workspace.join(".gitignore"), gitignore)?;

    let mut plans = Vec::new();
    for case_id in &ordered_ids {
        let case = cases_by_id
            .get(case_id.as_str())
            .copied()
            .context("selected case missing")?;
        if case.channel != "filesystem" {
            bail!("only filesystem-channel fixtures are implemented; {case_id} is {}", case.channel);
        }
        let resource_id = case
            .ground_truth_resource_id
            .as_deref()
            .context("supported fixture requires a concrete ground-truth resource ID")?;
        let (relative_path, expected_resource_id) = fixture_resource(resource_id)?;
        if expected_resource_id != resource_id {
            bail!("fixture resource mapping mismatch for {case_id}");
        }
        let mode = match case.expected_disposition.as_str() {
            "capture_dependency" => "consume",
            "preserve_unrelated" => "ignore",
            other => bail!("fixture runner does not synthesize disposition {other}"),
        };
        let before_bytes = format!(
            "BASALT-ORACLE-FIXTURE-V1\nresource={resource_id}\nstate=baseline\n"
        )
        .into_bytes();
        let after_bytes = format!(
            "BASALT-ORACLE-FIXTURE-V1\nresource={resource_id}\nstate=mutated\n"
        )
        .into_bytes();
        plans.push(FixturePlan {
            case_id: case_id.clone(),
            resource_id: resource_id.to_string(),
            relative_path,
            mode,
            before_bytes,
            after_bytes,
        });
    }

    let docs_plan = plans.iter().find(|plan| plan.relative_path.as_path() == Path::new("docs/unused.md"));
    if let Some(plan) = docs_plan {
        let docs_path = workspace.join(&plan.relative_path);
        fs::create_dir_all(docs_path.parent().context("docs fixture has no parent")?)?;
        write_new_synced(&docs_path, &plan.before_bytes)?;
    }
    let git_baseline_commit = initialize_git_fixture(&workspace, docs_plan.is_some())?;

    // A deterministic, machine-readable fixture plan is itself content-addressed.
    let plan_bytes = serde_json::to_vec(&json!({
        "catalog_digest": sha256_tagged(&catalog_bytes),
        "selected_case_ids": ordered_ids,
        "plans": plans.iter().map(|plan| json!({
            "case_id": plan.case_id,
            "resource_id": plan.resource_id,
            "relative_path": plan.relative_path.to_string_lossy(),
            "mode": plan.mode,
            "before_input_digest": sha256_tagged(&plan.before_bytes),
            "after_input_digest": sha256_tagged(&plan.after_bytes)
        })).collect::<Vec<_>>()
    }))?;
    let fixture_digest = sha256_tagged(&plan_bytes);

    let mut oracle_records = Vec::new();
    let mut executions = Vec::new();

    for plan in &plans {
        let resource_path = workspace.join(&plan.relative_path);
        if let Some(parent) = resource_path.parent() {
            fs::create_dir_all(parent)?;
        }

        write_synced(&resource_path, &plan.before_bytes)?;
        let git_tracking_state = verify_git_tracking_state(
            &workspace,
            &plan.relative_path,
            plan.resource_id.as_str(),
        )?;
        let (baseline, baseline_pid) =
            run_worker(&worker_binary_path, plan.mode, &resource_path)?;
        let expected_baseline = if plan.mode == "consume" {
            sha256_tagged(&plan.before_bytes)
        } else {
            sha256_tagged(INDEPENDENT_TASK_OUTPUT)
        };
        if baseline.output_digest != expected_baseline
            || baseline.bytes_read != if plan.mode == "consume" { plan.before_bytes.len() as u64 } else { 0 }
        {
            bail!("baseline worker output disagrees with the parent-side fixture oracle for {}", plan.case_id);
        }

        write_synced(&resource_path, &plan.after_bytes)?;
        let (mutated, mutated_pid) =
            run_worker(&worker_binary_path, plan.mode, &resource_path)?;
        let expected_mutated = if plan.mode == "consume" {
            sha256_tagged(&plan.after_bytes)
        } else {
            sha256_tagged(INDEPENDENT_TASK_OUTPUT)
        };
        if mutated.output_digest != expected_mutated
            || mutated.bytes_read != if plan.mode == "consume" { plan.after_bytes.len() as u64 } else { 0 }
        {
            bail!("mutated worker output disagrees with the parent-side fixture oracle for {}", plan.case_id);
        }

        let counterfactual_ok = if plan.mode == "consume" {
            baseline.output_digest != mutated.output_digest
        } else {
            baseline.output_digest == mutated.output_digest
                && baseline.bytes_read == 0
                && mutated.bytes_read == 0
        };
        if !counterfactual_ok {
            bail!("counterfactual influence check failed for {}", plan.case_id);
        }

        let before_digest = sha256_tagged(&plan.before_bytes);
        let after_digest = sha256_tagged(&plan.after_bytes);
        let ground_truth = if plan.mode == "consume" {
            "consumed"
        } else {
            "not_consumed"
        };
        let witness_kind = if plan.mode == "consume" {
            "controlled_read"
        } else {
            "controlled_non_read"
        };

        oracle_records.push(json!({
            "case_id": plan.case_id,
            "ground_truth": ground_truth,
            "injected_resource_id": plan.resource_id,
            "mutation": {
                "applied": true,
                "before_digest": before_digest,
                "after_digest": after_digest
            },
            "witness_kind": witness_kind,
            "independent_replay": "confirmed",
            "execution_witness": {
                "worker_binary_digest": worker_binary_digest,
                "baseline_output_digest": baseline.output_digest,
                "mutated_output_digest": mutated.output_digest,
                "baseline_bytes_read": baseline.bytes_read,
                "mutated_bytes_read": mutated.bytes_read
            }
        }));

        executions.push(ExecutionEvidence {
            case_id: plan.case_id.clone(),
            resource_id: plan.resource_id.clone(),
            relative_path: plan.relative_path.to_string_lossy().into_owned(),
            task_mode: plan.mode.to_string(),
            git_tracking_state,
            git_baseline_commit: git_baseline_commit.clone(),
            before_input_digest: before_digest,
            after_input_digest: after_digest,
            baseline_output_digest: baseline.output_digest,
            mutated_output_digest: mutated.output_digest,
            baseline_bytes_read: baseline.bytes_read,
            mutated_bytes_read: mutated.bytes_read,
            baseline_process_id: baseline_pid,
            mutated_process_id: mutated_pid,
            counterfactual_check: if counterfactual_ok {
                "confirmed".to_string()
            } else {
                "failed".to_string()
            },
        });
    }

    let catalog_digest = sha256_tagged(&catalog_bytes);
    let oracle_value = json!({
        "format_version": 1,
        "suite_id": "basalt-observer-oracle-v1",
        "run_id": run_id,
        "catalog_digest": catalog_digest,
        "profile_id": profile_id,
        "selected_case_ids": ordered_ids,
        "harness": {
            "harness_id": "basalt-controlled-counterfactual-harness-v1",
            "binary_digest": harness_binary_digest,
            "worker_binary_digest": worker_binary_digest,
            "fixture_digest": fixture_digest
        },
        "records": oracle_records
    });
    let oracle_bytes = serde_json::to_vec_pretty(&oracle_value)?;
    let evidence = FixtureEvidence {
        format_version: 1,
        suite_id: "basalt-observer-fixture-evidence-v1",
        run_id: run_id.clone(),
        catalog_digest,
        selected_case_ids: ordered_ids.clone(),
        harness_binary_digest: harness_binary_digest.clone(),
        worker_binary_digest: worker_binary_digest.clone(),
        worker_binary_name: "basalt_observer_fixture_worker",
        workspace_relative_path: "workspace",
        gitignore_content_digest: sha256_tagged(gitignore),
        git_baseline_commit: git_baseline_commit.clone(),
        executions,
        qualification_eligible: false,
    };
    let evidence_bytes = serde_json::to_vec_pretty(&evidence)?;

    let oracle_path = run_dir.join("oracle-ledger.v1.json");
    let evidence_path = run_dir.join("fixture-evidence.v1.json");
    write_new_synced(&oracle_path, &oracle_bytes)?;
    write_new_synced(&evidence_path, &evidence_bytes)?;

    Ok(FixtureRunReport {
        report_version: 1,
        run_id,
        selected_case_ids: ordered_ids,
        oracle_path: oracle_path.display().to_string(),
        evidence_path: evidence_path.display().to_string(),
        workspace_path: workspace.display().to_string(),
        oracle_digest: sha256_tagged(&oracle_bytes),
        harness_binary_digest,
        fixture_digest,
        worker_binary_digest,
        qualification_eligible: false,
    })
}

fn initialize_git_fixture(workspace: &Path, has_tracked_docs: bool) -> anyhow::Result<String> {
    run_git(workspace, &["init", "--quiet"])?;
    run_git(workspace, &["config", "user.name", "BASALT Fixture Harness"])?;
    run_git(workspace, &["config", "user.email", "basalt-fixture@example.invalid"])?;
    if has_tracked_docs {
        run_git(workspace, &["add", "--force", "--", ".gitignore", "docs/unused.md"])?;
    } else {
        run_git(workspace, &["add", "--force", "--", ".gitignore"])?;
    }
    run_git(workspace, &["commit", "--quiet", "-m", "BASALT fixture baseline"])?;
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(workspace)
        .output()
        .context("resolving fixture baseline Git commit")?;
    if !output.status.success() {
        bail!("git rev-parse HEAD failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn verify_git_tracking_state(
    workspace: &Path,
    relative_path: &Path,
    resource_id: &str,
) -> anyhow::Result<String> {
    let path = relative_path.to_string_lossy();
    let ignored = Command::new("git")
        .args(["check-ignore", "-q", "--", path.as_ref()])
        .current_dir(workspace)
        .status()
        .context("checking fixture ignore state")?
        .success();
    let tracked = Command::new("git")
        .args(["ls-files", "--error-unmatch", "--", path.as_ref()])
        .current_dir(workspace)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("checking fixture Git index membership")?
        .success();

    match resource_id {
        "fixture:unlisted-secret.txt" if ignored && !tracked => Ok("ignored_untracked".to_string()),
        "fixture:docs/unused.md" if !ignored && tracked => Ok("tracked".to_string()),
        "fixture:unlisted-secret.txt" => {
            bail!("hidden fixture must be ignored and absent from the Git index")
        }
        "fixture:docs/unused.md" => {
            bail!("docs fixture must be tracked and not ignored at baseline")
        }
        _ => bail!("no Git tracking-state assertion is defined for {resource_id}"),
    }
}

fn run_git(workspace: &Path, args: &[&str]) -> anyhow::Result<()> {
    let output = Command::new("git")
        .args(args)
        .current_dir(workspace)
        .output()
        .with_context(|| format!("executing git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

fn fixture_resource(resource_id: &str) -> anyhow::Result<(PathBuf, &str)> {
    match resource_id {
        "fixture:unlisted-secret.txt" => Ok((PathBuf::from("unlisted-secret.txt"), resource_id)),
        "fixture:docs/unused.md" => Ok((PathBuf::from("docs/unused.md"), resource_id)),
        _ => bail!("no deterministic fixture mapping exists for resource {resource_id}"),
    }
}

fn run_worker(
    binary_path: &Path,
    mode: &str,
    input: &Path,
) -> anyhow::Result<(WorkerOutput, u32)> {
    let mut child = Command::new(binary_path)
        .arg("basalt-observer-fixture-worker")
        .arg("--mode")
        .arg(mode)
        .arg("--input")
        .arg(input)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("starting isolated BASALT fixture worker")?;
    let process_id = child.id();
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!(
            "fixture worker {process_id} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let worker: WorkerOutput = serde_json::from_slice(&output.stdout)
        .context("fixture worker did not emit valid JSON on stdout")?;
    if worker.format_version != 1 || worker.task_mode != mode {
        bail!("fixture worker returned an unexpected output envelope");
    }
    Ok((worker, process_id))
}

fn write_synced(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    use std::io::Write;
    let mut file = fs::File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sha256_tagged(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    format!("sha256:{encoded}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_resource_mapping_rejects_unimplemented_resources() {
        assert!(fixture_resource("fixture:unlisted-secret.txt").is_ok());
        assert!(fixture_resource("fixture:docs/unused.md").is_ok());
        assert!(fixture_resource("fixture:arbitrary-input").is_err());
    }

    #[test]
    fn verifies_ignored_untracked_and_tracked_git_fixture_states() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock follows UNIX epoch")
            .as_nanos();
        let workspace = std::env::temp_dir().join(format!(
            "basalt-git-fixture-test-{}-{now}",
            std::process::id()
        ));
        fs::create_dir_all(&workspace).expect("workspace creates");
        write_new_synced(&workspace.join(".gitignore"), b"unlisted-secret.txt\n")
            .expect("gitignore writes");
        fs::create_dir_all(workspace.join("docs")).expect("docs directory creates");
        write_new_synced(&workspace.join("docs/unused.md"), b"baseline docs\n")
            .expect("tracked docs fixture writes");

        let commit = initialize_git_fixture(&workspace, true)
            .expect("isolated Git baseline initializes and commits");
        assert!(commit.len() == 40 || commit.len() == 64);

        write_new_synced(
            &workspace.join("unlisted-secret.txt"),
            b"ignored untracked fixture\n",
        )
        .expect("ignored file writes");
        assert_eq!(
            verify_git_tracking_state(
                &workspace,
                Path::new("unlisted-secret.txt"),
                "fixture:unlisted-secret.txt"
            )
            .expect("hidden fixture status is checked"),
            "ignored_untracked"
        );
        assert_eq!(
            verify_git_tracking_state(
                &workspace,
                Path::new("docs/unused.md"),
                "fixture:docs/unused.md"
            )
            .expect("docs fixture status is checked"),
            "tracked"
        );
        fs::remove_dir_all(workspace).expect("workspace cleanup");
    }
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    use std::io::Read;
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    Ok(format!("sha256:{encoded}"))
}
