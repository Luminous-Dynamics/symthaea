// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! DE-001A2EI production-shaped fixture executor.
//!
//! This binary is fixture-qualification-only. It performs no cosmology and has
//! no real optimizer authority. It launches exactly one locally identified child
//! process, captures bounded evidence, validates result-tree confinement, and
//! emits only `UNASSESSED` reproduction semantics.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;

const REQUEST_PROTOCOL: &str = "DE-001A2EI-FIXTURE-EXECUTION-REQUEST-v1";
const REQUEST_AUTHORITY: &str = "fixture-qualification-request-only";
const RECEIPT_PROTOCOL: &str = "DE-001A2EI-FIXTURE-EXECUTION-v1";
const RECEIPT_AUTHORITY: &str = "fixture-execution-evidence-only";
const MODE: &str = "fixture-qualification-only";
const MAX_REQUEST_BYTES: u64 = 1024 * 1024;
const MAX_EXECUTABLE_BYTES: u64 = 256 * 1024 * 1024;
const MAX_CAPTURE_BYTES: usize = 1024 * 1024;
const MAX_RESULT_FILES: usize = 64;
const MAX_RESULT_FILE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_RESULT_AGGREGATE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ARGV: usize = 64;
const MAX_ARG_BYTES: usize = 4096;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema_version: u64,
    protocol: String,
    scientific_claim: String,
    authority: String,
    execution_mode: String,
    executable: String,
    argv: Vec<String>,
    expected_executable_sha256: String,
    result_root: String,
    result_files: Vec<ResultSpec>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultSpec {
    role: String,
    relative_path: String,
}

#[derive(Debug, Clone, Serialize)]
struct ManifestEntry {
    role: String,
    relative_path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct Receipt {
    protocol: &'static str,
    execution_state: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    execution_mode: &'static str,
    reproduction_verdict: &'static str,
    executor_implementation_qualified: bool,
    optimizer_execution_authorized: bool,
    a2_execution_authorized: bool,
    real_optimizer_execution_authorized: bool,
    a2q_execution_authorized: bool,
    actual_process_invocation_count: u64,
    process_exit_code: Option<i32>,
    request_sha256_preflight: String,
    request_sha256_postflight: String,
    executor_executable_sha256_preflight: String,
    executor_executable_sha256_postflight: String,
    executor_executable_bytes: u64,
    child_executable_sha256_preflight: String,
    child_executable_sha256_postflight: String,
    child_executable_bytes: u64,
    command_argv_sha256: String,
    stdout_sha256: String,
    stderr_sha256: String,
    result_manifest_sha256: String,
    result_files: Vec<ManifestEntry>,
    environment_cleared: bool,
    network_isolation_proven: bool,
    automatic_retry_allowed: bool,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct InvalidReceipt {
    protocol: &'static str,
    execution_state: &'static str,
    scientific_claim: &'static str,
    authority: &'static str,
    execution_mode: &'static str,
    reproduction_verdict: &'static str,
    executor_implementation_qualified: bool,
    optimizer_execution_authorized: bool,
    a2_execution_authorized: bool,
    real_optimizer_execution_authorized: bool,
    a2q_execution_authorized: bool,
    actual_process_invocation_count: u64,
    network_isolation_proven: bool,
    automatic_retry_allowed: bool,
    error: String,
}

#[derive(Debug)]
struct FileIdentity {
    bytes: u64,
    sha256: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_absolute_normal(path: &Path, label: &str) -> Result<(), String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute: {}", path.display()));
    }
    for component in path.components() {
        match component {
            Component::RootDir | Component::Normal(_) => {}
            Component::CurDir | Component::ParentDir | Component::Prefix(_) => {
                return Err(format!("{label} contains non-normal component: {}", path.display()));
            }
        }
    }
    Ok(())
}

fn require_real_dir(path: &Path, label: &str) -> Result<PathBuf, String> {
    require_absolute_normal(path, label)?;
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!("{label} must be a real directory: {}", path.display()));
    }
    let canonical = fs::canonicalize(path)
        .map_err(|error| format!("canonicalize {label} {}: {error}", path.display()))?;
    if canonical != path {
        return Err(format!("{label} must already be canonical: {} -> {}", path.display(), canonical.display()));
    }
    Ok(canonical)
}

fn file_identity(path: &Path, label: &str, max_bytes: u64) -> Result<FileIdentity, String> {
    require_absolute_normal(path, label)?;
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a real regular file: {}", path.display()));
    }
    if metadata.len() > max_bytes {
        return Err(format!("{label} exceeds byte limit: {} > {max_bytes}", metadata.len()));
    }
    let canonical = fs::canonicalize(path)
        .map_err(|error| format!("canonicalize {label} {}: {error}", path.display()))?;
    if canonical != path {
        return Err(format!("{label} must already be canonical: {} -> {}", path.display(), canonical.display()));
    }
    let bytes = fs::read(path)
        .map_err(|error| format!("read {label} {}: {error}", path.display()))?;
    Ok(FileIdentity {
        bytes: metadata.len(),
        sha256: sha256_hex(&bytes),
    })
}

fn validate_relative_path(raw: &str) -> Result<(), String> {
    if raw.is_empty() || raw.contains('\\') {
        return Err(format!("invalid relative result path: {raw:?}"));
    }
    let segments: Vec<&str> = raw.split('/').collect();
    if segments
        .iter()
        .any(|segment| segment.is_empty() || *segment == "." || *segment == "..")
    {
        return Err(format!("result path contains empty/dot/parent segment: {raw:?}"));
    }
    let path = Path::new(raw);
    if path.is_absolute()
        || !path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(format!("result path is not normal relative path: {raw:?}"));
    }
    Ok(())
}

fn validate_role(role: &str) -> Result<(), String> {
    if role.is_empty()
        || role.len() > 64
        || !role
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
    {
        return Err(format!("invalid result role: {role:?}"));
    }
    Ok(())
}

fn validate_result_specs(root: &Path, specs: &[ResultSpec]) -> Result<(), String> {
    if specs.len() > MAX_RESULT_FILES {
        return Err(format!("too many result files: {} > {MAX_RESULT_FILES}", specs.len()));
    }
    let mut roles = BTreeSet::new();
    let mut paths = BTreeSet::new();
    for spec in specs {
        validate_role(&spec.role)?;
        validate_relative_path(&spec.relative_path)?;
        if !roles.insert(spec.role.as_str()) {
            return Err(format!("duplicate result role: {}", spec.role));
        }
        if !paths.insert(spec.relative_path.as_str()) {
            return Err(format!("duplicate result path: {}", spec.relative_path));
        }

        let mut current = root.to_path_buf();
        let segments: Vec<&str> = spec.relative_path.split('/').collect();
        for segment in &segments[..segments.len().saturating_sub(1)] {
            current.push(segment);
            let metadata = fs::symlink_metadata(&current).map_err(|error| {
                format!("result parent must exist before execution {}: {error}", current.display())
            })?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(format!("result parent must be real directory: {}", current.display()));
            }
        }
        let target = root.join(&spec.relative_path);
        match fs::symlink_metadata(&target) {
            Ok(_) => return Err(format!("result target must not preexist: {}", target.display())),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("stat result target {}: {error}", target.display())),
        }
    }
    Ok(())
}

fn collect_results(root: &Path, specs: &[ResultSpec]) -> Result<Vec<ManifestEntry>, String> {
    let mut aggregate = 0_u64;
    let mut manifest = Vec::with_capacity(specs.len());
    for spec in specs {
        let mut current = root.to_path_buf();
        let segments: Vec<&str> = spec.relative_path.split('/').collect();
        for (index, segment) in segments.iter().enumerate() {
            current.push(segment);
            let metadata = fs::symlink_metadata(&current)
                .map_err(|error| format!("stat result {}: {error}", current.display()))?;
            if metadata.file_type().is_symlink() {
                return Err(format!("result path contains symlink: {}", current.display()));
            }
            let terminal = index + 1 == segments.len();
            if terminal {
                if !metadata.is_file() {
                    return Err(format!("result terminal is not regular file: {}", current.display()));
                }
                if metadata.len() > MAX_RESULT_FILE_BYTES {
                    return Err(format!("result file exceeds byte limit: {}", current.display()));
                }
                aggregate = aggregate
                    .checked_add(metadata.len())
                    .ok_or_else(|| "result aggregate byte count overflow".to_owned())?;
                if aggregate > MAX_RESULT_AGGREGATE_BYTES {
                    return Err("result aggregate exceeds byte limit".to_owned());
                }
                let bytes = fs::read(&current)
                    .map_err(|error| format!("read result {}: {error}", current.display()))?;
                manifest.push(ManifestEntry {
                    role: spec.role.clone(),
                    relative_path: spec.relative_path.clone(),
                    bytes: metadata.len(),
                    sha256: sha256_hex(&bytes),
                });
            } else if !metadata.is_dir() {
                return Err(format!("result intermediate component is not directory: {}", current.display()));
            }
        }
    }
    Ok(manifest)
}

fn capture_bounded<R: Read>(mut reader: R) -> io::Result<(Vec<u8>, bool)> {
    let mut kept = Vec::new();
    let mut overflow = false;
    let mut buffer = [0_u8; 8192];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        if kept.len() < MAX_CAPTURE_BYTES {
            let available = MAX_CAPTURE_BYTES - kept.len();
            let take = available.min(count);
            kept.extend_from_slice(&buffer[..take]);
            if take < count {
                overflow = true;
            }
        } else {
            overflow = true;
        }
    }
    Ok((kept, overflow))
}

fn validate_request(request: &Request) -> Result<(), String> {
    if request.schema_version != 1 {
        return Err("schema_version must equal 1".to_owned());
    }
    if request.protocol != REQUEST_PROTOCOL
        || request.scientific_claim != "NONE"
        || request.authority != REQUEST_AUTHORITY
        || request.execution_mode != MODE
    {
        return Err("fixture request identity/authority drifted".to_owned());
    }
    if !is_sha256(&request.expected_executable_sha256) {
        return Err("expected_executable_sha256 must be lowercase SHA-256".to_owned());
    }
    if request.argv.len() > MAX_ARGV {
        return Err(format!("too many argv entries: {} > {MAX_ARGV}", request.argv.len()));
    }
    if request.argv.iter().any(|arg| arg.len() > MAX_ARG_BYTES) {
        return Err("argv entry exceeds byte limit".to_owned());
    }
    Ok(())
}

fn write_evidence(path: &Path, bytes: &[u8]) -> Result<(), String> {
    fs::write(path, bytes).map_err(|error| format!("write evidence {}: {error}", path.display()))
}

fn execute(
    request_path: &Path,
    evidence_root: &Path,
    invocation_count: &mut u64,
) -> Result<Receipt, String> {
    let request_pre = file_identity(request_path, "request", MAX_REQUEST_BYTES)?;
    let request_bytes = fs::read(request_path)
        .map_err(|error| format!("read request {}: {error}", request_path.display()))?;
    let request: Request = serde_json::from_slice(&request_bytes)
        .map_err(|error| format!("parse strict request JSON: {error}"))?;
    validate_request(&request)?;

    let result_root = require_real_dir(Path::new(&request.result_root), "result_root")?;
    if result_root.starts_with(evidence_root) || evidence_root.starts_with(&result_root) {
        return Err("result_root and evidence_root must be disjoint".to_owned());
    }
    if request_path.starts_with(&result_root) || request_path.starts_with(evidence_root) {
        return Err("request file must be outside writable result/evidence roots".to_owned());
    }
    validate_result_specs(&result_root, &request.result_files)?;

    let child_path = Path::new(&request.executable);
    let child_pre = file_identity(child_path, "child executable", MAX_EXECUTABLE_BYTES)?;
    if child_pre.sha256 != request.expected_executable_sha256 {
        return Err("child executable SHA-256 does not match request".to_owned());
    }
    if child_path.starts_with(&result_root) || child_path.starts_with(evidence_root) {
        return Err("child executable must be outside writable roots".to_owned());
    }

    let self_path = fs::canonicalize(env::current_exe().map_err(|error| format!("current_exe: {error}"))?)
        .map_err(|error| format!("canonicalize current executor: {error}"))?;
    let self_pre = file_identity(&self_path, "executor executable", MAX_EXECUTABLE_BYTES)?;

    let mut launched_argv = Vec::with_capacity(request.argv.len() + 1);
    launched_argv.push(request.executable.clone());
    launched_argv.extend(request.argv.iter().cloned());
    let argv_bytes = serde_json::to_vec(&launched_argv)
        .map_err(|error| format!("serialize exact argv: {error}"))?;

    let mut command = Command::new(child_path);
    command
        .args(&request.argv)
        .current_dir(&result_root)
        .env_clear()
        .env("LANG", "C")
        .env("LC_ALL", "C")
        .env("TZ", "UTC")
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .env("RAYON_NUM_THREADS", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| format!("spawn fixture process: {error}"))?;
    *invocation_count = 1;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "child stdout pipe missing".to_owned())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "child stderr pipe missing".to_owned())?;
    let stdout_thread = thread::spawn(move || capture_bounded(stdout));
    let stderr_thread = thread::spawn(move || capture_bounded(stderr));
    let status = child
        .wait()
        .map_err(|error| format!("wait for fixture process: {error}"))?;
    let (stdout_bytes, stdout_overflow) = stdout_thread
        .join()
        .map_err(|_| "stdout capture thread panicked".to_owned())?
        .map_err(|error| format!("capture stdout: {error}"))?;
    let (stderr_bytes, stderr_overflow) = stderr_thread
        .join()
        .map_err(|_| "stderr capture thread panicked".to_owned())?
        .map_err(|error| format!("capture stderr: {error}"))?;
    if stdout_overflow || stderr_overflow {
        return Err("captured stdout/stderr exceeded byte limit".to_owned());
    }

    let request_post = file_identity(request_path, "request", MAX_REQUEST_BYTES)?;
    let self_post = file_identity(&self_path, "executor executable", MAX_EXECUTABLE_BYTES)?;
    let child_post = file_identity(child_path, "child executable", MAX_EXECUTABLE_BYTES)?;
    if request_pre.sha256 != request_post.sha256
        || self_pre.sha256 != self_post.sha256
        || child_pre.sha256 != child_post.sha256
    {
        return Err("pre/postflight implementation identity changed".to_owned());
    }

    let result_files = collect_results(&result_root, &request.result_files)?;
    let manifest_bytes = serde_json::to_vec(&result_files)
        .map_err(|error| format!("serialize result manifest: {error}"))?;

    write_evidence(&evidence_root.join("argv.json"), &argv_bytes)?;
    write_evidence(&evidence_root.join("stdout.bin"), &stdout_bytes)?;
    write_evidence(&evidence_root.join("stderr.bin"), &stderr_bytes)?;
    write_evidence(&evidence_root.join("result-manifest.json"), &manifest_bytes)?;

    Ok(Receipt {
        protocol: RECEIPT_PROTOCOL,
        execution_state: if status.success() { "EXECUTED" } else { "EXECUTION_ERROR" },
        scientific_claim: "NONE",
        authority: RECEIPT_AUTHORITY,
        execution_mode: MODE,
        reproduction_verdict: "UNASSESSED",
        executor_implementation_qualified: false,
        optimizer_execution_authorized: false,
        a2_execution_authorized: false,
        real_optimizer_execution_authorized: false,
        a2q_execution_authorized: false,
        actual_process_invocation_count: 1,
        process_exit_code: status.code(),
        request_sha256_preflight: request_pre.sha256,
        request_sha256_postflight: request_post.sha256,
        executor_executable_sha256_preflight: self_pre.sha256,
        executor_executable_sha256_postflight: self_post.sha256,
        executor_executable_bytes: self_pre.bytes,
        child_executable_sha256_preflight: child_pre.sha256,
        child_executable_sha256_postflight: child_post.sha256,
        child_executable_bytes: child_pre.bytes,
        command_argv_sha256: sha256_hex(&argv_bytes),
        stdout_sha256: sha256_hex(&stdout_bytes),
        stderr_sha256: sha256_hex(&stderr_bytes),
        result_manifest_sha256: sha256_hex(&manifest_bytes),
        result_files,
        environment_cleared: true,
        network_isolation_proven: false,
        automatic_retry_allowed: false,
        error: None,
    })
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: de001a-a2ei-executor REQUEST_JSON EVIDENCE_ROOT");
        std::process::exit(2);
    }

    let request_path = PathBuf::from(&args[1]);
    let evidence_root = match require_real_dir(Path::new(&args[2]), "evidence_root") {
        Ok(path) => path,
        Err(error) => {
            eprintln!("de001a-a2ei-executor: INVALID: {error}");
            std::process::exit(2);
        }
    };
    match fs::read_dir(&evidence_root) {
        Ok(mut entries) if entries.next().is_none() => {}
        Ok(_) => {
            eprintln!("de001a-a2ei-executor: INVALID: evidence_root must be empty");
            std::process::exit(2);
        }
        Err(error) => {
            eprintln!("de001a-a2ei-executor: INVALID: read evidence_root: {error}");
            std::process::exit(2);
        }
    }

    let mut invocation_count = 0_u64;
    let (bytes, exit_code) = match execute(&request_path, &evidence_root, &mut invocation_count) {
        Ok(receipt) => {
            let code = if receipt.execution_state == "EXECUTED" { 0 } else { 1 };
            (
                serde_json::to_vec(&receipt)
                    .unwrap_or_else(|error| panic!("serialize fixture receipt: {error}")),
                code,
            )
        }
        Err(error) => {
            let receipt = InvalidReceipt {
                protocol: RECEIPT_PROTOCOL,
                execution_state: "INVALID",
                scientific_claim: "NONE",
                authority: RECEIPT_AUTHORITY,
                execution_mode: MODE,
                reproduction_verdict: "UNASSESSED",
                executor_implementation_qualified: false,
                optimizer_execution_authorized: false,
                a2_execution_authorized: false,
                real_optimizer_execution_authorized: false,
                a2q_execution_authorized: false,
                actual_process_invocation_count: invocation_count,
                network_isolation_proven: false,
                automatic_retry_allowed: false,
                error,
            };
            (
                serde_json::to_vec(&receipt)
                    .unwrap_or_else(|error| panic!("serialize invalid fixture receipt: {error}")),
                2,
            )
        }
    };

    let receipt_path = evidence_root.join("receipt.json");
    if let Err(error) = fs::write(&receipt_path, &bytes) {
        eprintln!("de001a-a2ei-executor: cannot write receipt {}: {error}", receipt_path.display());
        std::process::exit(2);
    }
    println!("{}", String::from_utf8_lossy(&bytes));
    std::process::exit(exit_code);
}

#[cfg(test)]
mod tests {
    use super::validate_relative_path;

    #[test]
    fn relative_path_grammar_is_fail_closed() {
        for invalid in ["", "/abs", "./x", "a/./b", "a//b", "../x", "a/../b", "a\\b"] {
            assert!(validate_relative_path(invalid).is_err(), "accepted {invalid:?}");
        }
        for valid in ["result.txt", "nested/result.json", ".hidden"] {
            assert!(validate_relative_path(valid).is_ok(), "rejected {valid:?}");
        }
    }
}
