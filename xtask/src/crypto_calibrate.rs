//! Reproducible execution for Symthaea's known-break cryptography calibration suite.
//!
//! This runner intentionally executes only a strict subset of Cargo test commands.
//! Suite data is reviewed input, never a shell program. A passing test means the
//! repository's known-break witness reproduced; it is not a novel discovery or a
//! statement about production cryptography.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Component, Path, PathBuf},
    process::{Command, Output},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const SUITE_SCHEMA: &str = "symthaea.crypto-discovery-calibration/v1";
const ALLOWED_PACKAGES: &[&str] = &["symthaea-hdc-crypto", "symthaea-core", "symthaea-soma"];
const MAX_EXCERPT_CHARS: usize = 4_000;

#[derive(Debug, Deserialize)]
struct Suite {
    schema_version: String,
    suite_id: String,
    status: String,
    tasks: Vec<Task>,
}

#[derive(Debug, Deserialize)]
struct Task {
    task_id: String,
    category: String,
    security_notion: String,
    quarantined: bool,
    production_use_allowed: bool,
    novel_discovery: bool,
    source_regressions: Vec<Regression>,
    positive_controls: Vec<Regression>,
}

#[derive(Debug, Deserialize)]
struct Regression {
    path: String,
    test: String,
    command: String,
}

#[derive(Debug)]
struct ParsedCommand {
    package: String,
    test_filter: String,
    cargo_args: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Receipt {
    schema_version: &'static str,
    suite_id: String,
    suite_sha256: String,
    source_revision: Option<String>,
    working_tree_clean: bool,
    working_tree_status_sha256: String,
    rustc_version: Option<String>,
    cargo_version: Option<String>,
    mode: &'static str,
    started_at_unix_s: u64,
    finished_at_unix_s: u64,
    task_results: Vec<TaskResult>,
    summary: Summary,
}

#[derive(Debug, Serialize)]
struct TaskResult {
    task_id: String,
    role: &'static str,
    category: String,
    security_notion: String,
    source_path: String,
    regression_test: String,
    command_argv: Vec<String>,
    outcome: &'static str,
    exit_code: Option<i32>,
    duration_ms: u128,
    stdout_sha256: String,
    stderr_sha256: String,
    stdout_excerpt: String,
    stderr_excerpt: String,
    note: String,
}

#[derive(Debug, Serialize)]
struct Summary {
    selected_tasks: usize,
    regression_commands: usize,
    planned_commands: usize,
    reproduced_known_breaks: usize,
    positive_control_commands: usize,
    positive_controls_passed: usize,
    positive_control_failures: usize,
    command_failures: usize,
    launch_errors: usize,
    novel_discoveries: usize,
}

pub(super) fn run(
    root: &Path,
    suite_arg: &Path,
    task_id: Option<&str>,
    all: bool,
    list: bool,
    dry_run: bool,
    out_arg: Option<&Path>,
) -> Result<()> {
    let selected_modes = task_id.is_some() as usize + all as usize + list as usize;
    if selected_modes != 1 {
        bail!("choose exactly one of --list, --all, or --task-id <ID>");
    }

    let suite_path = resolve_in_root(root, suite_arg)?;
    let suite_bytes = fs::read(&suite_path)
        .with_context(|| format!("read calibration suite {}", suite_path.display()))?;
    let suite_sha256 = sha256_hex(&suite_bytes);
    let suite: Suite = serde_json::from_slice(&suite_bytes)
        .with_context(|| format!("parse calibration suite {}", suite_path.display()))?;
    validate_suite(&suite)?;

    if list {
        // Listing is also a manifest integrity check: every declared regression
        // must still map to a test function and an allowlisted Cargo invocation.
        for task in &suite.tasks {
            for regression in task.source_regressions.iter().chain(&task.positive_controls) {
                validate_regression(root, regression)
                    .with_context(|| format!("validate {} / {}", task.task_id, regression.test))?;
            }
        }
        println!(
            "suite_id={} schema={} sha256={}",
            suite.suite_id, suite.schema_version, suite_sha256
        );
        for task in &suite.tasks {
            println!(
                "{}\tcategory={}\tregressions={}\tsecurity_notion={}",
                task.task_id,
                task.category,
                task.source_regressions.len(),
                task.security_notion
            );
        }
        return Ok(());
    }

    let selected: Vec<&Task> = if let Some(id) = task_id {
        let task = suite
            .tasks
            .iter()
            .find(|task| task.task_id == id)
            .with_context(|| format!("unknown calibration task ID: {id}"))?;
        vec![task]
    } else {
        suite.tasks.iter().collect()
    };

    let started_at = unix_seconds();
    let source_revision = command_text(root, "git", &["rev-parse", "HEAD"]);
    let status_output = command_output(root, "git", &["status", "--porcelain"]);
    let tree_status = status_output.unwrap_or_else(|| "git status unavailable".to_owned());
    let working_tree_clean = tree_status.is_empty();
    if !dry_run && !working_tree_clean {
        bail!(
            "refusing to execute calibration from a dirty/unverifiable working tree (status digest {}); commit or clean the tree first",
            sha256_hex(tree_status.as_bytes())
        );
    }
    let rustc_version = command_text(root, "rustc", &["--version"]);
    let cargo_version = command_text(root, "cargo", &["--version"]);

    // Preflight every selected task before running any command. A stale source
    // path or disallowed invocation must not leave a partially executed suite.
    let mut validated = Vec::new();
    for task in &selected {
        for (role, regressions) in [
            ("known_break", &task.source_regressions),
            ("positive_control", &task.positive_controls),
        ] {
            for regression in regressions {
                let parsed = validate_regression(root, regression)
                    .with_context(|| format!("validate {} / {}", task.task_id, regression.test))?;
                validated.push((*task, regression, parsed, role));
            }
        }
    }

    let mut task_results = Vec::new();
    for (task, regression, parsed, role) in validated {
        if dry_run {
            task_results.push(TaskResult {
                task_id: task.task_id.clone(),
                role,
                category: task.category.clone(),
                security_notion: task.security_notion.clone(),
                source_path: regression.path.clone(),
                regression_test: regression.test.clone(),
                command_argv: parsed.cargo_args,
                outcome: "planned",
                exit_code: None,
                duration_ms: 0,
                stdout_sha256: sha256_hex(&[]),
                stderr_sha256: sha256_hex(&[]),
                stdout_excerpt: String::new(),
                stderr_excerpt: String::new(),
                note: "not executed; no attack result inferred".to_owned(),
            });
        } else {
            task_results.push(execute_regression(root, task, regression, parsed, role));
        }
    }

    let reproduced_known_breaks = task_results
        .iter()
        .filter(|r| r.outcome == "reproduced_known_break")
        .count();
    let positive_control_commands = task_results
        .iter()
        .filter(|r| r.role == "positive_control")
        .count();
    let positive_controls_passed = task_results
        .iter()
        .filter(|r| r.outcome == "positive_control_passed")
        .count();
    let positive_control_failures = task_results
        .iter()
        .filter(|r| {
            r.role == "positive_control"
                && (r.outcome == "command_failed" || r.outcome == "launch_error")
        })
        .count();
    let command_failures = task_results.iter().filter(|r| r.outcome == "command_failed").count();
    let launch_errors = task_results.iter().filter(|r| r.outcome == "launch_error").count();
    let planned_commands = task_results.iter().filter(|r| r.outcome == "planned").count();

    let receipt = Receipt {
        schema_version: "symthaea.crypto-discovery-receipt/v1",
        suite_id: suite.suite_id,
        suite_sha256,
        source_revision,
        working_tree_clean,
        working_tree_status_sha256: sha256_hex(tree_status.as_bytes()),
        rustc_version,
        cargo_version,
        mode: if dry_run { "dry_run" } else { "executed" },
        started_at_unix_s: started_at,
        finished_at_unix_s: unix_seconds(),
        summary: Summary {
            selected_tasks: selected.len(),
            regression_commands: task_results.len(),
            planned_commands,
            reproduced_known_breaks,
            positive_control_commands,
            positive_controls_passed,
            positive_control_failures,
            command_failures,
            launch_errors,
            novel_discoveries: 0,
        },
        task_results,
    };

    let receipt_bytes = serde_json::to_vec_pretty(&receipt)?;
    if let Some(out_arg) = out_arg {
        let out_path = resolve_output_path(root, out_arg)?;
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create receipt directory {}", parent.display()))?;
        }
        fs::write(&out_path, &receipt_bytes)
            .with_context(|| format!("write evidence receipt {}", out_path.display()))?;
        eprintln!("wrote evidence receipt: {}", out_path.display());
    }
    println!("{}", String::from_utf8(receipt_bytes)?);

    // Fail closed on execution errors. A command failure may be a compile/infra
    // failure rather than a disproved witness; the receipt retains that distinction.
    if !dry_run && (command_failures > 0 || launch_errors > 0) {
        bail!(
            "calibration incomplete: {} command failure(s), {} launch error(s); inspect the receipt",
            command_failures,
            launch_errors
        );
    }
    Ok(())
}

fn validate_suite(suite: &Suite) -> Result<()> {
    if suite.schema_version != SUITE_SCHEMA {
        bail!("unsupported suite schema: {}", suite.schema_version);
    }
    if suite.status != "calibration-only" {
        bail!("refusing non-calibration suite status: {}", suite.status);
    }
    if suite.tasks.is_empty() {
        bail!("calibration suite must contain at least one task");
    }

    let mut ids = std::collections::HashSet::new();
    for task in &suite.tasks {
        if task.task_id.trim().is_empty() || !ids.insert(task.task_id.as_str()) {
            bail!("empty or duplicate task ID: {:?}", task.task_id);
        }
        if !task.quarantined || task.production_use_allowed || task.novel_discovery {
            bail!(
                "task {} violates calibration safety flags; expected quarantined=true, production_use_allowed=false, novel_discovery=false",
                task.task_id
            );
        }
        if task.source_regressions.is_empty() {
            bail!("task {} has no source regression commands", task.task_id);
        }
        if task.positive_controls.is_empty() {
            bail!("task {} has no positive controls", task.task_id);
        }
    }
    Ok(())
}

fn validate_regression(root: &Path, regression: &Regression) -> Result<ParsedCommand> {
    let relative = Path::new(&regression.path);
    if relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        bail!(
            "regression source path must be repository-relative and cannot traverse: {}",
            regression.path
        );
    }
    let source_path = root.join(relative);
    let source = fs::read_to_string(&source_path)
        .with_context(|| format!("read regression source {}", source_path.display()))?;
    let signature = format!("fn {}(", regression.test);
    if !source.contains(&signature) {
        bail!(
            "manifest references missing test function {} in {}",
            regression.test,
            regression.path
        );
    }

    parse_command_for_test(&regression.command, &regression.test)
}

fn parse_command_for_test(command: &str, expected_test: &str) -> Result<ParsedCommand> {
    let tokens: Vec<&str> = command.split_ascii_whitespace().collect();
    if tokens.len() < 5 || tokens[0] != "cargo" || tokens[1] != "test" || tokens[2] != "-p" {
        bail!("only 'cargo test -p <approved-package> [--lib] <test-filter>' is allowed");
    }
    let package = tokens[3];
    if !ALLOWED_PACKAGES.contains(&package) {
        bail!("unapproved calibration package: {package}");
    }

    let (flags, filter) = if tokens.len() == 5 {
        (Vec::<String>::new(), tokens[4])
    } else if tokens.len() == 6 && tokens[4] == "--lib" {
        (vec!["--lib".to_owned()], tokens[5])
    } else {
        bail!("unapproved cargo test arguments for {}", expected_test);
    };
    if filter != expected_test {
        bail!(
            "command filter '{}' does not match manifest test '{}'",
            filter,
            expected_test
        );
    }

    let mut cargo_args = vec!["test".to_owned(), "-p".to_owned(), package.to_owned()];
    cargo_args.extend(flags);
    cargo_args.push(filter.to_owned());
    Ok(ParsedCommand {
        package: package.to_owned(),
        test_filter: filter.to_owned(),
        cargo_args,
    })
}

fn execute_regression(
    root: &Path,
    task: &Task,
    regression: &Regression,
    parsed: ParsedCommand,
    role: &'static str,
) -> TaskResult {
    let started = Instant::now();
    let output = Command::new("cargo")
        .args(&parsed.cargo_args)
        .current_dir(root)
        .output();

    let elapsed = started.elapsed().as_millis();
    match output {
        Ok(output) => task_result_from_output(task, regression, parsed, output, elapsed, role),
        Err(error) => TaskResult {
            task_id: task.task_id.clone(),
            role,
            category: task.category.clone(),
            security_notion: task.security_notion.clone(),
            source_path: regression.path.clone(),
            regression_test: regression.test.clone(),
            command_argv: parsed.cargo_args,
            outcome: "launch_error",
            exit_code: None,
            duration_ms: elapsed,
            stdout_sha256: sha256_hex(&[]),
            stderr_sha256: sha256_hex(error.to_string().as_bytes()),
            stdout_excerpt: String::new(),
            stderr_excerpt: clip(&error.to_string()),
            note: format!(
                "Cargo could not be started ({}); no attack result inferred",
                regression.command
            ),
        },
    }
}

fn task_result_from_output(
    task: &Task,
    regression: &Regression,
    parsed: ParsedCommand,
    output: Output,
    elapsed_ms: u128,
    role: &'static str,
) -> TaskResult {
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let exit_code = output.status.code();
    let outcome = if output.status.success() && role == "positive_control" {
        "positive_control_passed"
    } else if output.status.success() {
        "reproduced_known_break"
    } else {
        "command_failed"
    };
    TaskResult {
        task_id: task.task_id.clone(),
        role,
        category: task.category.clone(),
        security_notion: task.security_notion.clone(),
        source_path: regression.path.clone(),
        regression_test: parsed.test_filter.clone(),
        command_argv: parsed.cargo_args.clone(),
        outcome,
        exit_code,
        duration_ms: elapsed_ms,
        stdout_sha256: sha256_hex(stdout.as_bytes()),
        stderr_sha256: sha256_hex(stderr.as_bytes()),
        stdout_excerpt: clip(&stdout),
        stderr_excerpt: clip(&stderr),
        note: if output.status.success() && role == "positive_control" {
            "functional positive control passed; this does not establish a security property"
                .to_owned()
        } else if output.status.success() {
            format!(
                "known-break regression passed for package {}; this is calibration, not novelty",
                parsed.package
            )
        } else {
            "command exited nonzero; this may be a compile, environment, or witness failure "
                .to_owned()
                + "and requires triage"
        },
    }
}

fn resolve_in_root(root: &Path, path: &Path) -> Result<PathBuf> {
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let canonical_root = root
        .canonicalize()
        .with_context(|| format!("canonicalize {}", root.display()))?;
    let canonical_path = resolved
        .canonicalize()
        .with_context(|| format!("resolve {}", resolved.display()))?;
    if !canonical_path.starts_with(&canonical_root) {
        bail!("path must remain inside repository root: {}", path.display());
    }
    Ok(canonical_path)
}

fn resolve_output_path(root: &Path, path: &Path) -> Result<PathBuf> {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    let canonical_root = root
        .canonicalize()
        .with_context(|| format!("canonicalize {}", root.display()))?;
    // Output directories may not exist yet. Canonicalize the closest existing parent,
    // then append the remaining normal components to ensure the final path stays in-root.
    let mut ancestor = path.as_path();
    let mut suffix = Vec::new();
    while !ancestor.exists() {
        let name = ancestor.file_name().context("output path has no filename")?;
        suffix.push(name.to_os_string());
        ancestor = ancestor.parent().context("output path has no existing parent")?;
    }
    let canonical_ancestor = ancestor.canonicalize()?;
    if !canonical_ancestor.starts_with(&canonical_root) {
        bail!("output path must remain inside repository root: {}", path.display());
    }
    let mut resolved = canonical_ancestor;
    for component in suffix.iter().rev() {
        resolved.push(component);
    }
    Ok(resolved)
}

fn command_text(root: &Path, program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).current_dir(root).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn command_output(root: &Path, program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).current_dir(root).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn clip(text: &str) -> String {
    let count = text.chars().count();
    if count <= MAX_EXCERPT_CHARS {
        return text.to_owned();
    }
    let suffix: String = text
        .chars()
        .skip(count.saturating_sub(MAX_EXCERPT_CHARS))
        .collect();
    format!("[excerpt begins near end; full output digest retained]…{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_only_approved_cargo_test_shapes() {
        let parsed = parse_command_for_test(
            "cargo test -p symthaea-core --lib legacy_attack_known_pair_forges_hdc_mac",
            "legacy_attack_known_pair_forges_hdc_mac",
        )
        .unwrap();
        assert_eq!(parsed.package, "symthaea-core");
        assert_eq!(parsed.test_filter, "legacy_attack_known_pair_forges_hdc_mac");
        assert_eq!(
            parsed.cargo_args,
            vec![
                "test",
                "-p",
                "symthaea-core",
                "--lib",
                "legacy_attack_known_pair_forges_hdc_mac"
            ]
        );
    }

    #[test]
    fn rejects_shell_or_arbitrary_arguments() {
        assert!(
            parse_command_for_test("cargo test -p symthaea-core test; rm -rf /", "test").is_err()
        );
        assert!(parse_command_for_test("sh -c cargo test", "test").is_err());
        assert!(parse_command_for_test("cargo test -p unknown-package test", "test").is_err());
    }
}
