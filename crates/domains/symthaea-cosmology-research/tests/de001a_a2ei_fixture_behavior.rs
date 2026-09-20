// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Behavioral fixture tests for the DE-001A2EI executor core.
//!
//! These tests run only inert local binaries built by the same Cargo invocation.
//! They perform no cosmology, likelihood evaluation, sampling, minimization,
//! optimization, networking, or scientific classification.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TempRoot {
    path: PathBuf,
}

impl TempRoot {
    fn new(label: &str) -> Self {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock after epoch")
            .as_nanos();
        let candidate = std::env::temp_dir().join(format!(
            "de001a-a2ei-{label}-{}-{nanos}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&candidate).expect("create temporary root");
        let path = fs::canonicalize(&candidate).expect("canonicalize temporary root");
        Self { path }
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

struct Case {
    _temp: TempRoot,
    root: PathBuf,
    request: PathBuf,
    result_root: PathBuf,
    evidence_root: PathBuf,
    counter: PathBuf,
    result_file: PathBuf,
    executor: PathBuf,
    fixture: PathBuf,
}

impl Case {
    fn new(label: &str) -> Self {
        let temp = TempRoot::new(label);
        let root = temp.path.clone();
        let request = root.join("request.json");
        let result_root = root.join("results");
        let evidence_root = root.join("evidence");
        let counter = root.join("invocation-count.txt");
        fs::create_dir(&result_root).expect("create result root");
        fs::create_dir(&evidence_root).expect("create evidence root");

        Self {
            _temp: temp,
            root,
            request,
            result_file: result_root.join("fixture-result.txt"),
            result_root,
            evidence_root,
            counter,
            executor: canonical_binary(
                option_env!("CARGO_BIN_EXE_de001a-a2ei-executor")
                    .expect("Cargo must provide de001a-a2ei-executor binary"),
            ),
            fixture: canonical_binary(
                option_env!("CARGO_BIN_EXE_de001a-a2ei-fixture")
                    .expect("Cargo must provide de001a-a2ei-fixture binary"),
            ),
        }
    }

    fn write_request(&self, mode: &str, result_files: Value, expected_sha256: &str) {
        let request = json!({
            "schema_version": 1,
            "protocol": "DE-001A2EI-FIXTURE-EXECUTION-REQUEST-v1",
            "scientific_claim": "NONE",
            "authority": "fixture-qualification-request-only",
            "execution_mode": "fixture-qualification-only",
            "executable": path_text(&self.fixture),
            "argv": [
                mode,
                path_text(&self.counter),
                path_text(&self.result_file)
            ],
            "expected_executable_sha256": expected_sha256,
            "result_root": path_text(&self.result_root),
            "result_files": result_files
        });
        fs::write(
            &self.request,
            serde_json::to_vec(&request).expect("serialize fixture request"),
        )
        .expect("write fixture request");
    }

    fn write_default_request(&self, mode: &str) {
        self.write_request(
            mode,
            json!([{
                "role": "fixture-result",
                "relative_path": "fixture-result.txt"
            }]),
            &sha256_file(&self.fixture),
        );
    }

    fn run(&self) -> Output {
        Command::new(&self.executor)
            .arg(&self.request)
            .arg(&self.evidence_root)
            .output()
            .expect("run fixture executor")
    }

    fn receipt(&self) -> Value {
        let bytes = fs::read(self.evidence_root.join("receipt.json"))
            .expect("read executor receipt");
        serde_json::from_slice(&bytes).expect("parse executor receipt")
    }
}

fn canonical_binary(path: &str) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|error| panic!("canonicalize {path}: {error}"))
}

fn path_text(path: &Path) -> String {
    path.to_str()
        .unwrap_or_else(|| panic!("path is not UTF-8: {}", path.display()))
        .to_owned()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn sha256_file(path: &Path) -> String {
    sha256_bytes(&fs::read(path).unwrap_or_else(|error| {
        panic!("read {} for SHA-256: {error}", path.display())
    }))
}

fn assert_non_authoritative(receipt: &Value) {
    assert_eq!(receipt["scientific_claim"].as_str(), Some("NONE"));
    assert_eq!(
        receipt["reproduction_verdict"].as_str(),
        Some("UNASSESSED")
    );
    assert_eq!(receipt["executor_implementation_qualified"].as_bool(), Some(false));
    assert_eq!(receipt["optimizer_execution_authorized"].as_bool(), Some(false));
    assert_eq!(receipt["a2_execution_authorized"].as_bool(), Some(false));
    assert_eq!(
        receipt["real_optimizer_execution_authorized"].as_bool(),
        Some(false)
    );
    assert_eq!(receipt["a2q_execution_authorized"].as_bool(), Some(false));
    assert_eq!(receipt["network_isolation_proven"].as_bool(), Some(false));
    assert_eq!(receipt["automatic_retry_allowed"].as_bool(), Some(false));
}

fn assert_counter_once(case: &Case) {
    assert_eq!(
        fs::read_to_string(&case.counter).expect("read invocation counter"),
        "1\n"
    );
}

fn assert_persisted_evidence(case: &Case, receipt: &Value, mode: &str) {
    let argv = fs::read(case.evidence_root.join("argv.json")).expect("read argv evidence");
    let stdout = fs::read(case.evidence_root.join("stdout.bin")).expect("read stdout evidence");
    let stderr = fs::read(case.evidence_root.join("stderr.bin")).expect("read stderr evidence");
    let manifest = fs::read(case.evidence_root.join("result-manifest.json"))
        .expect("read manifest evidence");

    assert_eq!(receipt["command_argv_sha256"].as_str(), Some(sha256_bytes(&argv).as_str()));
    assert_eq!(receipt["stdout_sha256"].as_str(), Some(sha256_bytes(&stdout).as_str()));
    assert_eq!(receipt["stderr_sha256"].as_str(), Some(sha256_bytes(&stderr).as_str()));
    assert_eq!(
        receipt["result_manifest_sha256"].as_str(),
        Some(sha256_bytes(&manifest).as_str())
    );

    assert_eq!(stdout, b"fixture-stdout-v1 count=1\n");
    assert_eq!(stderr, format!("fixture-stderr-v1 mode={mode}\n").as_bytes());

    let argv_value: Value = serde_json::from_slice(&argv).expect("parse argv evidence");
    let argv_array = argv_value.as_array().expect("argv evidence array");
    assert_eq!(argv_array.len(), 4);
    assert_eq!(argv_array[0].as_str(), Some(path_text(&case.fixture).as_str()));
    assert_eq!(argv_array[1].as_str(), Some(mode));
    assert_eq!(argv_array[2].as_str(), Some(path_text(&case.counter).as_str()));
    assert_eq!(argv_array[3].as_str(), Some(path_text(&case.result_file).as_str()));

    let manifest_value: Value = serde_json::from_slice(&manifest).expect("parse result manifest");
    let entries = manifest_value.as_array().expect("manifest array");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["role"].as_str(), Some("fixture-result"));
    assert_eq!(
        entries[0]["relative_path"].as_str(),
        Some("fixture-result.txt")
    );
    assert_eq!(
        entries[0]["bytes"].as_u64(),
        Some(fs::metadata(&case.result_file).expect("result metadata").len())
    );
    assert_eq!(
        entries[0]["sha256"].as_str(),
        Some(sha256_file(&case.result_file).as_str())
    );

    assert_eq!(
        receipt["result_files"],
        manifest_value,
        "receipt result_files must equal persisted manifest bytes"
    );
}

fn assert_invalid_without_spawn(case: &Case, output: &Output) {
    assert_eq!(output.status.code(), Some(2));
    let receipt = case.receipt();
    assert_eq!(receipt["execution_state"].as_str(), Some("INVALID"));
    assert_eq!(receipt["actual_process_invocation_count"].as_u64(), Some(0));
    assert_non_authoritative(&receipt);
    assert!(!case.counter.exists(), "invalid preflight must not invoke child");
}

#[test]
fn success_fixture_executes_once_and_binds_exact_evidence() {
    let case = Case::new("success");
    case.write_default_request("success");

    let output = case.run();
    assert_eq!(output.status.code(), Some(0));
    assert_counter_once(&case);

    let receipt = case.receipt();
    assert_eq!(receipt["protocol"].as_str(), Some("DE-001A2EI-FIXTURE-EXECUTION-v1"));
    assert_eq!(receipt["execution_state"].as_str(), Some("EXECUTED"));
    assert_eq!(receipt["actual_process_invocation_count"].as_u64(), Some(1));
    assert_eq!(receipt["process_exit_code"].as_i64(), Some(0));
    assert_eq!(
        receipt["request_sha256_preflight"],
        receipt["request_sha256_postflight"]
    );
    assert_eq!(
        receipt["executor_executable_sha256_preflight"],
        receipt["executor_executable_sha256_postflight"]
    );
    assert_eq!(
        receipt["child_executable_sha256_preflight"],
        receipt["child_executable_sha256_postflight"]
    );
    assert_non_authoritative(&receipt);
    assert_persisted_evidence(&case, &receipt, "success");
}

#[test]
fn nonzero_fixture_is_execution_error_without_retry() {
    let case = Case::new("nonzero");
    case.write_default_request("nonzero");

    let output = case.run();
    assert_eq!(output.status.code(), Some(1));
    assert_counter_once(&case);

    let receipt = case.receipt();
    assert_eq!(receipt["execution_state"].as_str(), Some("EXECUTION_ERROR"));
    assert_eq!(receipt["actual_process_invocation_count"].as_u64(), Some(1));
    assert_eq!(receipt["process_exit_code"].as_i64(), Some(23));
    assert_non_authoritative(&receipt);
    assert_persisted_evidence(&case, &receipt, "nonzero");
}

#[test]
fn duplicate_result_role_is_invalid_before_spawn() {
    let case = Case::new("duplicate-role");
    case.write_request(
        "success",
        json!([
            {"role": "same", "relative_path": "fixture-result.txt"},
            {"role": "same", "relative_path": "other.txt"}
        ]),
        &sha256_file(&case.fixture),
    );

    let output = case.run();
    assert_invalid_without_spawn(&case, &output);
}

#[test]
fn executable_hash_mismatch_is_invalid_before_spawn() {
    let case = Case::new("hash-mismatch");
    case.write_request(
        "success",
        json!([{"role": "fixture-result", "relative_path": "fixture-result.txt"}]),
        &"0".repeat(64),
    );

    let output = case.run();
    assert_invalid_without_spawn(&case, &output);
}

#[test]
fn dot_path_alias_is_invalid_before_spawn() {
    let case = Case::new("dot-path");
    case.write_request(
        "success",
        json!([{"role": "fixture-result", "relative_path": "./fixture-result.txt"}]),
        &sha256_file(&case.fixture),
    );

    let output = case.run();
    assert_invalid_without_spawn(&case, &output);
}

#[cfg(unix)]
#[test]
fn intermediate_directory_symlink_is_invalid_before_spawn() {
    use std::os::unix::fs::symlink;

    let mut case = Case::new("intermediate-symlink");
    let real_dir = case.root.join("real-result-parent");
    fs::create_dir(&real_dir).expect("create real parent");
    symlink(&real_dir, case.result_root.join("nested")).expect("create intermediate symlink");
    case.result_file = case.result_root.join("nested/fixture-result.txt");
    case.write_request(
        "success",
        json!([{"role": "fixture-result", "relative_path": "nested/fixture-result.txt"}]),
        &sha256_file(&case.fixture),
    );

    let output = case.run();
    assert_invalid_without_spawn(&case, &output);
}

#[cfg(unix)]
#[test]
fn result_root_symlink_is_invalid_before_spawn() {
    use std::os::unix::fs::symlink;

    let mut case = Case::new("root-symlink");
    let real_root = case.result_root.clone();
    let linked_root = case.root.join("linked-results");
    symlink(&real_root, &linked_root).expect("create result-root symlink");
    case.result_root = linked_root;
    case.result_file = case.result_root.join("fixture-result.txt");
    case.write_request(
        "success",
        json!([{"role": "fixture-result", "relative_path": "fixture-result.txt"}]),
        &sha256_file(&case.fixture),
    );

    let output = case.run();
    assert_invalid_without_spawn(&case, &output);
}
