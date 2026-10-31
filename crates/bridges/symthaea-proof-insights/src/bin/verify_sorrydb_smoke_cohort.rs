// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Verifies the three-project SorryDB smoke cohort against materialized checkouts.
//! This checks source identity only; it does not build Lean or attempt proofs.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;

const SOURCE_COMMIT: &str = "74b3b63f8ba890d675f362924b21ac55db410383";
const SOURCE_BLOB: &str = "af3b663429dc4689ee728d2fb5c7d8d5034476f7";
const SOURCE_SHA256: &str =
    "26bfd11099f6b8db7f80d49bdea5d9f8ea0d23ba65d44d93f67ed42ba1530062";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: String,
    dataset: Dataset,
    cohort_status: String,
    policy: Policy,
    toolchain_stratification: Stratification,
    tasks: Vec<Task>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Dataset {
    snapshot_id: String,
    source_repository: String,
    source_repository_commit: String,
    source_split_path: String,
    source_git_blob_sha1: String,
    source_sha256: String,
    source_bytes: usize,
    task_count: usize,
    project_count: usize,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Policy {
    candidate_selection: String,
    proof_state_source: String,
    proof_result_claimed: bool,
    build_or_import_run_claimed: bool,
    attempt_budget: String,
    next_action: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Stratification {
    distinct_toolchain_tags: usize,
    task_counts: Vec<ToolchainCount>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolchainCount {
    lean_version: String,
    tasks: usize,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Task {
    ordinal: usize,
    task_id: String,
    selection_role: String,
    checkout_directory: String,
    repository: Repository,
    location: Location,
    project_pins: ProjectPins,
    target_hint: String,
    debug_context_signal: String,
    source_inspected: bool,
    environment_reconstructed: bool,
    project_build_status: String,
    lean_evidence_status: String,
    proof_attempt_status: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Repository {
    remote: String,
    branch: String,
    commit: String,
    lean_version: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Location {
    path: String,
    start_line: usize,
    start_column: usize,
    end_line: usize,
    end_column: usize,
    column_indexing: String,
    verified_source_slice: String,
    source_file_git_blob_sha1: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectPins {
    lean_toolchain_file_git_blob_sha1: String,
    lakefile_toml_git_blob_sha1: String,
    lake_manifest_json_git_blob_sha1: String,
    mathlib_revision: String,
}

struct Expected {
    id: &'static str,
    remote: &'static str,
    commit: &'static str,
    lean: &'static str,
    checkout: &'static str,
    branch: &'static str,
    role: &'static str,
    path: &'static str,
    start_line: usize,
    start_column: usize,
    end_line: usize,
    end_column: usize,
    source_blob: &'static str,
    toolchain_blob: &'static str,
    lakefile_blob: &'static str,
    lake_manifest_blob: &'static str,
    mathlib_revision: &'static str,
    target_hint: &'static str,
    context_signal: &'static str,
}
const TASKS: [Expected; 3] = [
    Expected {
        id: "0ffe8d507806bde948dfed1d59a579a2650452acaa68992cca839c14cf2f365f",
        remote: "https://github.com/fpvandoorn/LeanCourse25",
        commit: "cd0f32f5b07311e08a9e89733a1375f54eb790c0",
        lean: "v4.24.0",
        checkout: "lean-course25",
        branch: "master",
        role: "stable_toolchain_baseline",
        path: "LeanCourse25/Assignments/Assignment11.lean",
        start_line: 39,
        start_column: 78,
        end_line: 39,
        end_column: 83,
        source_blob: "d553bc3bda3d14af1bc90373b7d0c5fc81bfba0a",
        toolchain_blob: "c00a53503cc3cc776c290056ed21ce7e5d80b083",
        lakefile_blob: "fcaf0ab48e8cd8ed6a1af0acf355e0430ed11bfd",
        lake_manifest_blob: "6dce7dda77492aaef6d63db207f10f364e1bae4b",
        mathlib_revision: "f897ebcf72cd16f89ab4577d0c826cd14afaafc7",
        target_hint: "nullmeasurable_measurable_null",
        context_signal: "ordinary proposition/measure-theory context; not an authoritative reconstructed proof state",
    },
    Expected {
        id: "7956fe8e4946a9ddf7098ee84c69669db5bc902715dad6e7cdf0db121aa5bfce",
        remote: "https://github.com/AlexKontorovich/PrimeNumberTheoremAnd",
        commit: "ddcb6542f2b62883c8642d3db4ecc115afcce33e",
        lean: "v4.26.0",
        checkout: "prime-number-theorem",
        branch: "More-FKS",
        role: "stable_toolchain_and_dependency_context",
        path: "PrimeNumberTheoremAnd/FioriKadiriSwidinsky.lean",
        start_line: 44,
        start_column: 137,
        end_line: 44,
        end_column: 142,
        source_blob: "6214f90ce5f8c61aca4a4d696ff5687d872dfc0e",
        toolchain_blob: "2654c20bb16bfd3a12f77a6cdeb73d48bf3b8359",
        lakefile_blob: "17c674bdc92fe6e333c839adc27ff7b8948be42c",
        lake_manifest_blob: "c0405e3c9643b00b1ce6a8120516ca482e6fe4ec",
        mathlib_revision: "2df2f0150c275ad53cb3c90f7c98ec15a56a1a67",
        target_hint: "table_1_prop",
        context_signal: "numerical prime-number-theorem estimate; no instance-registry signal inferred from debug goal",
    },
    Expected {
        id: "19f8c8d2aa81dccb1ea0da25a7f9271c9330e18eb3ec4aa453917a3c5d29cbed",
        remote: "https://github.com/Clap-lang/clap-lean",
        commit: "326f55435f91295d95a6166b060cace7ef92e6e4",
        lean: "v4.26.0-rc2",
        checkout: "clap-lean",
        branch: "assert_range2",
        role: "explicit_local_instance_registry_challenge",
        path: "Clap/Compilation.lean",
        start_line: 278,
        start_column: 2,
        end_line: 278,
        end_column: 7,
        source_blob: "ec8e38968c481b479cd6588756feaeebd2164286",
        toolchain_blob: "7035713c5522d1113de6e26251f8791eba689308",
        lakefile_blob: "0ca8d438bf079e52a9fff208ba57dfb4bd7e266e",
        lake_manifest_blob: "001d03474b6ea317800a505fe646c39da0acddaf",
        mathlib_revision: "d5c9558e75342a10d6321e6a8c798a14f68ae23c",
        target_hint: "fail₁",
        context_signal: "inst✝ : Fact (Nat.Prime p); selection hint only, actual LocalInstances registry must be captured from Lean",
    },
];
const TOOLCHAINS: [(&str, usize); 14] = [
    ("v4.27.0-rc1", 318), ("v4.26.0", 218), ("v4.24.0", 145), ("v4.26.0-rc2", 97),
    ("v4.22.0", 72), ("v4.23.0-rc2", 37), ("v4.24.0-rc1", 36), ("v4.25.0-rc2", 22),
    ("v4.18.0-rc1", 14), ("v4.25.0", 12), ("v4.18.0", 11), ("v4.26.0-rc1", 10),
    ("v4.21.0", 7), ("v4.22.0-rc4", 1),
];

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn validate_relative_path(value: &str) -> Result<()> {
    if value.is_empty()
        || value.starts_with('/')
        || value.chars().any(|c| matches!(c, '\\' | '\0' | ':'))
        || value.split('/').any(|p| p.is_empty() || p == "." || p == "..")
    {
        bail!("path is absolute, empty, or contains an unsafe component");
    }
    Ok(())
}
fn validate_checkout_name(value: &str) -> Result<()> {
    if value.is_empty()
        || !value.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    {
        bail!("checkout directory must be lowercase kebab-case");
    }
    Ok(())
}
fn git(checkout: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C").arg(checkout).args(args).output()
        .with_context(|| format!("run git {:?} in {}", args, checkout.display()))?;
    if !output.status.success() {
        bail!("git {:?} failed in {}: {}", args, checkout.display(),
            String::from_utf8_lossy(&output.stderr));
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}
fn normalize_remote(value: &str) -> String {
    value.trim_end_matches('/').trim_end_matches(".git").to_owned()
}
fn blob_sha1(checkout: &Path, path: &str) -> Result<String> {
    let digest = git(checkout, &["hash-object", "--", path])?;
    if !valid_hex(&digest, 40) {
        bail!("git hash-object returned an invalid SHA-1 for {path}");
    }
    Ok(digest)
}
fn char_column_to_byte(line: &str, column: usize) -> Result<usize> {
    let count = line.chars().count();
    if column > count {
        bail!("column {column} exceeds line character count {count}");
    }
    if column == count {
        return Ok(line.len());
    }
    line.char_indices().nth(column).map(|(i, _)| i)
        .context("map Unicode codepoint column to UTF-8 byte offset")
}
fn extract_span(
    source: &str, start_line: usize, start_column: usize,
    end_line: usize, end_column: usize,
) -> Result<String> {
    if start_line == 0 || start_line != end_line {
        bail!("smoke task requires one-based lines and a same-line span");
    }
    let line = source.split('\n').nth(start_line - 1)
        .context("source line is outside the file")?;
    let start = char_column_to_byte(line, start_column)?;
    let end = char_column_to_byte(line, end_column)?;
    if end <= start {
        bail!("source span is empty or reversed");
    }
    Ok(line.get(start..end).context("span is not on UTF-8 boundaries")?.to_owned())
}
fn validate_mathlib_revision(checkout: &Path, expected: &str) -> Result<()> {
    let manifest: Value = serde_json::from_slice(&fs::read(checkout.join("lake-manifest.json"))?)?;
    let packages = manifest.get("packages").and_then(Value::as_array)
        .context("Lake manifest has no packages array")?;
    let matches: Vec<&Value> = packages.iter()
        .filter(|p| p.get("name").and_then(Value::as_str) == Some("mathlib")).collect();
    if matches.len() != 1 {
        bail!("expected exactly one Mathlib package pin in Lake manifest");
    }
    let actual = matches[0].get("rev").and_then(Value::as_str)
        .context("Mathlib pin has no revision")?;
    if actual != expected {
        bail!("Mathlib revision mismatch: expected {expected}, received {actual}");
    }
    Ok(())
}
fn validate_task(task: &Task, root: &Path, index: usize) -> Result<()> {
    let expected = TASKS.get(index).context("too many tasks in smoke cohort")?;
    if task.ordinal != index || task.task_id != expected.id
        || task.repository.remote != expected.remote || task.repository.commit != expected.commit
        || task.repository.branch != expected.branch
        || task.repository.lean_version != expected.lean
        || task.checkout_directory != expected.checkout
        || task.selection_role != expected.role
        || task.location.path != expected.path
        || task.location.start_line != expected.start_line
        || task.location.start_column != expected.start_column
        || task.location.end_line != expected.end_line
        || task.location.end_column != expected.end_column
        || task.location.source_file_git_blob_sha1 != expected.source_blob
        || task.project_pins.lean_toolchain_file_git_blob_sha1 != expected.toolchain_blob
        || task.project_pins.lakefile_toml_git_blob_sha1 != expected.lakefile_blob
        || task.project_pins.lake_manifest_json_git_blob_sha1 != expected.lake_manifest_blob
        || task.project_pins.mathlib_revision != expected.mathlib_revision
        || task.target_hint != expected.target_hint
        || task.debug_context_signal != expected.context_signal
    {
        bail!("task {index} differs from its fixed SorryDB identity, source span, or project pins");
    }
    if task.repository.branch.is_empty() || task.target_hint.is_empty()
        || !task.source_inspected || task.environment_reconstructed
        || task.project_build_status != "not_run" || task.lean_evidence_status != "not_run"
        || task.proof_attempt_status != "not_run"
    {
        bail!("task {index} has a contradictory or prematurely advanced state");
    }
    if task.location.column_indexing != "zero_based"
        || task.location.verified_source_slice != "sorry"
        || task.location.start_line == 0 || task.location.start_line != task.location.end_line
    {
        bail!("task {index} has unsupported source coordinates");
    }
    validate_relative_path(&task.location.path)?;
    validate_checkout_name(&task.checkout_directory)?;

    let canonical_root = root.canonicalize().context("canonicalize checkout root")?;
    let checkout = canonical_root.join(&task.checkout_directory).canonicalize()
        .with_context(|| format!("resolve checkout {}", task.checkout_directory))?;
    if !checkout.starts_with(&canonical_root) {
        bail!("task {index} checkout escaped the declared checkout root");
    }
    let head = git(&checkout, &["rev-parse", "HEAD"])?;
    if head != task.repository.commit {
        bail!("task {index} HEAD mismatch: expected {}, received {head}", task.repository.commit);
    }
    let remote = git(&checkout, &["remote", "get-url", "origin"])?;
    if normalize_remote(&remote) != normalize_remote(&task.repository.remote) {
        bail!(
            "task {index} remote mismatch: expected {}, received {remote}",
            task.repository.remote
        );
    }
    if !git(&checkout, &["status", "--porcelain", "--untracked-files=all"])?.is_empty() {
        bail!("task {index} checkout has tracked modifications or untracked files");
    }

    let pins = [
        ("lean-toolchain", task.project_pins.lean_toolchain_file_git_blob_sha1.as_str()),
        ("lakefile.toml", task.project_pins.lakefile_toml_git_blob_sha1.as_str()),
        ("lake-manifest.json", task.project_pins.lake_manifest_json_git_blob_sha1.as_str()),
        (task.location.path.as_str(), task.location.source_file_git_blob_sha1.as_str()),
    ];
    for (pin_path, expected_sha) in pins {
        validate_relative_path(pin_path)?;
        if !valid_hex(expected_sha, 40) {
            bail!("task {index} has invalid blob digest for {pin_path}");
        }
        let file_path = checkout.join(pin_path);
        let metadata = fs::symlink_metadata(&file_path)
            .with_context(|| format!("inspect pinned file {pin_path}"))?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            bail!("task {index} pinned path {pin_path} is not a regular file");
        }
        let canonical_file = file_path.canonicalize()
            .with_context(|| format!("canonicalize pinned file {pin_path}"))?;
        if !canonical_file.starts_with(&checkout) {
            bail!("task {index} pinned file {pin_path} escapes the checkout");
        }
        let actual = blob_sha1(&checkout, pin_path)?;
        if actual != expected_sha {
            bail!(
                "task {index} blob mismatch for {pin_path}: expected {expected_sha}, received {actual}"
            );
        }
    }
    let toolchain = fs::read_to_string(checkout.join("lean-toolchain"))?;
    let expected_toolchain = format!("leanprover/lean4:{}", task.repository.lean_version);
    if toolchain.trim() != expected_toolchain {
        bail!("task {index} lean-toolchain does not match {}", task.repository.lean_version);
    }
    let lakefile = fs::read_to_string(checkout.join("lakefile.toml"))?;
    if !lakefile.contains(task.repository.lean_version.as_str()) {
        bail!("task {index} lakefile does not mention its Lean toolchain");
    }
    validate_mathlib_revision(&checkout, &task.project_pins.mathlib_revision)?;

    let source_path = checkout.join(&task.location.path);
    let source_metadata = fs::symlink_metadata(&source_path)?;
    if !source_metadata.is_file() || source_metadata.file_type().is_symlink() {
        bail!("task {index} sorry source is not a regular file");
    }
    let source = fs::read_to_string(&source_path)?;
    let selected = extract_span(&source, task.location.start_line, task.location.start_column,
        task.location.end_line, task.location.end_column)?;
    if selected != "sorry" {
        bail!("task {index} coordinates select {selected:?}, not the sorry token");
    }
    println!("PITB_SORRYDB_SMOKE_SOURCE_TASK_PASS: ordinal={} id={} lean={} columns={}..{}",
        task.ordinal, task.task_id, task.repository.lean_version,
        task.location.start_column, task.location.end_column);
    Ok(())
}
fn validate_manifest(manifest: &Manifest, root: &Path) -> Result<()> {
    if manifest.schema_version != "pitb.sorrydb.smoke-candidates.v0"
        || manifest.cohort_status != "selected_candidate_not_reconstructed"
    {
        bail!("unsupported smoke-cohort schema version or status");
    }
    let d = &manifest.dataset;
    if d.snapshot_id != "SorryDB_2601" || d.source_repository != "SorryDB/SorryDB"
        || d.source_repository_commit != SOURCE_COMMIT
        || d.source_split_path != "data/SorryDB_2601/SorryDB_2601_1000_evaluation_split.json"
        || d.source_git_blob_sha1 != SOURCE_BLOB || d.source_sha256 != SOURCE_SHA256
        || d.source_bytes != 1_599_271 || d.task_count != 1_000 || d.project_count != 78
    {
        bail!("SorryDB source identity differs from the frozen evaluation snapshot");
    }
    if manifest.policy.proof_result_claimed || manifest.policy.build_or_import_run_claimed
        || manifest.policy.attempt_budget != "not_registered"
        || !manifest.policy.proof_state_source.contains("must be reconstructed")
        || !manifest.policy.next_action.contains("export real Lean proof state")
        || manifest.policy.candidate_selection.is_empty()
    {
        bail!("smoke policy contains an unsupported execution or claim state");
    }
    if manifest.toolchain_stratification.distinct_toolchain_tags != TOOLCHAINS.len()
        || manifest.toolchain_stratification.task_counts.len() != TOOLCHAINS.len()
    {
        bail!("toolchain distribution does not contain all 14 fixed strata");
    }
    let mut total = 0;
    for (i, (version, count)) in TOOLCHAINS.iter().enumerate() {
        let row = &manifest.toolchain_stratification.task_counts[i];
        if row.lean_version != *version || row.tasks != *count {
            bail!("toolchain stratum {i} differs from the pinned cohort");
        }
        total += row.tasks;
    }
    if total != d.task_count || manifest.tasks.len() != TASKS.len() {
        bail!("task/toolchain counts do not reconcile with the frozen snapshot");
    }
    for (i, task) in manifest.tasks.iter().enumerate() {
        validate_task(task, root, i).with_context(|| format!("verify smoke task {i}"))?;
    }
    if !manifest.tasks[2].debug_context_signal.contains("Fact (Nat.Prime p)") {
        bail!("local-instance smoke candidate lost its recorded selection signal");
    }
    Ok(())
}
fn self_test() -> Result<()> {
    validate_relative_path("Mathlib/Algebra/Example.lean")?;
    for bad in ["", "/etc/passwd", "../escape", "A/../B", "A//B", r"A\B"] {
        if validate_relative_path(bad).is_ok() {
            bail!("self-test accepted unsafe path {bad:?}");
        }
    }
    validate_checkout_name("lean-course25")?;
    for bad in ["../escape", "/absolute", "Bad_Name", ""] {
        if validate_checkout_name(bad).is_ok() {
            bail!("self-test accepted unsafe checkout name {bad:?}");
        }
    }
    if extract_span("a ∃ β sorry", 1, 6, 1, 11)? != "sorry" {
        bail!("Unicode codepoint column self-test failed");
    }
    if extract_span("hello", 1, 4, 1, 2).is_ok() {
        bail!("self-test accepted reversed source span");
    }
    println!("PITB_SORRYDB_SMOKE_VERIFIER_SELF_TEST_PASS");
    Ok(())
}
enum Mode {
    Help,
    SelfTest,
    Verify { manifest: PathBuf, root: PathBuf },
}
fn parse_args() -> Result<Mode> {
    let mut args = env::args().skip(1);
    let mut manifest = None;
    let mut root = None;
    let mut self_test = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--manifest" => manifest = args.next().map(PathBuf::from),
            "--checkouts-root" => root = args.next().map(PathBuf::from),
            "--self-test" => self_test = true,
            "--help" | "-h" => return Ok(Mode::Help),
            _ => bail!("unknown argument: {arg}"),
        }
    }
    if self_test {
        if manifest.is_some() || root.is_some() {
            bail!("--self-test cannot be combined with manifest arguments");
        }
        return Ok(Mode::SelfTest);
    }
    match (manifest, root) {
        (Some(manifest), Some(root)) => Ok(Mode::Verify { manifest, root }),
        _ => bail!("--manifest and --checkouts-root are required"),
    }
}
fn run() -> Result<()> {
    match parse_args()? {
        Mode::Help => {
            println!(
                concat!(
                    "Usage: verify_sorrydb_smoke_cohort --self-test | ",
                    "--manifest FILE --checkouts-root DIR"
                )
            );
            Ok(())
        }
        Mode::SelfTest => self_test(),
        Mode::Verify { manifest, root } => {
            let bytes = fs::read(&manifest)
                .with_context(|| format!("read manifest {}", manifest.display()))?;
            let parsed: Manifest = serde_json::from_slice(&bytes)?;
            validate_manifest(&parsed, &root)?;
            println!(
                concat!(
                    "PITB_SORRYDB_SMOKE_SOURCE_VALIDATION_PASS: tasks=3 ",
                    "projects=3 toolchains=3; no build/proof result"
                )
            );
            Ok(())
        }
    }
}
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("PITB_SORRYDB_SMOKE_SOURCE_VALIDATION_FAIL: {error:#}");
            ExitCode::FAILURE
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zero_based_unicode_columns_are_codepoint_columns() {
        assert_eq!(extract_span("a ∃ β sorry", 1, 6, 1, 11).unwrap(), "sorry");
    }
    #[test]
    fn unsafe_paths_and_reversed_spans_are_rejected() {
        assert!(validate_relative_path("../escape").is_err());
        assert!(extract_span("hello", 1, 4, 1, 2).is_err());
    }
    #[test]
    fn smoke_cohort_covers_three_lean_toolchains() {
        assert_ne!(TASKS[0].lean, TASKS[1].lean);
        assert_ne!(TASKS[1].lean, TASKS[2].lean);
    }
}
