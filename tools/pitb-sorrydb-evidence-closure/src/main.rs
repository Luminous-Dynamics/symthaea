use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
};

const EXPECTED_EVALUATION_BLOB: &str = "af3b663429dc4689ee728d2fb5c7d8d5034476f7";
const EXPECTED_EVALUATION_COUNT: usize = 1000;
const RUN_SCHEMA: &str = "pitb.sorrydb-proof-run.v0";
const REPORT_SCHEMA: &str = "pitb.sorrydb-evidence-closure-report.v0";

#[derive(Debug, Clone, Deserialize)]
struct Dataset {
    sorries: Vec<SorryRecord>,
}

#[derive(Debug, Clone, Deserialize)]
struct SorryRecord {
    id: String,
    repo: RepoInfo,
    location: Location,
    debug_info: DebugInfo,
}

#[derive(Debug, Clone, Deserialize)]
struct RepoInfo {
    commit: String,
    lean_version: String,
}

#[derive(Debug, Clone, Deserialize)]
struct Location {
    path: String,
    start_line: u64,
    start_column: u64,
    end_line: u64,
    end_column: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct DebugInfo {
    goal: String,
}

#[derive(Debug, Clone, Deserialize)]
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
    declaration_signature_before_sha256: String,
    declaration_signature_after_sha256: String,
    source_file_sha256: String,
    replacement_patch_sha256: String,
    proof_artifact_sha256: String,
    checker_receipt_sha256: String,
    semantic_gate_receipt_sha256: String,
    source_binding_receipt_sha256: String,
    outcome: String,
    proof_inserted: bool,
    exact_sorry_replaced: bool,
    sorry_absent_after_replacement: bool,
    target_sorry_ax_absent: bool,
    target_is_prop: bool,
    build_passed: bool,
    kernel_passed: bool,
    semantic_gate_passed: bool,
    source_binding_passed: bool,
    vacuity_gate_passed: bool,
    axiom_delta_zero: bool,
    new_sorry_delta_zero: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceBindingReceipt {
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
    source_file_sha256: String,
    goal_sha256: String,
    source_span_sha256: String,
    source_span: String,
    source_span_verified_as_sorry: bool,
    target_goal_reproduced: bool,
    environment_sha256: String,
    lake_manifest_sha256: String,
    repository_tree_sha256: String,
    reproduction_log_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckerReceipt {
    schema_version: String,
    run_id: String,
    task_id: String,
    repo_commit: String,
    lean_version: String,
    source_file_sha256: String,
    modified_source_sha256: String,
    replacement_patch_sha256: String,
    proof_artifact_sha256: String,
    declaration_signature_before_sha256: String,
    declaration_signature_after_sha256: String,
    target_is_prop: bool,
    build_passed: bool,
    kernel_passed: bool,
    target_sorry_ax_absent: bool,
    axiom_delta_zero: bool,
    new_sorry_delta_zero: bool,
    checker_binary_sha256: String,
    checker_command_sha256: String,
    environment_sha256: String,
    checker_log_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SemanticGateReceipt {
    schema_version: String,
    run_id: String,
    task_id: String,
    goal_sha256: String,
    declaration_signature_sha256: String,
    source_binding_receipt_sha256: String,
    source_correspondence_status: String,
    hypothesis_audit_status: String,
    vacuity_audit_status: String,
    reviewer_decision: String,
    reviewer_id: String,
    reviewer_key_id: String,
    reviewer_attestation_sha256: String,
    semantic_log_sha256: String,
    semantic_gate_passed: bool,
    vacuity_gate_passed: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReviewerAttestation {
    schema_version: String,
    canonicalization: String,
    key_id: String,
    reviewer_id: String,
    run_id: String,
    task_id: String,
    goal_sha256: String,
    declaration_signature_sha256: String,
    source_binding_receipt_sha256: String,
    source_correspondence_status: String,
    hypothesis_audit_status: String,
    vacuity_audit_status: String,
    reviewer_decision: String,
    semantic_log_sha256: String,
    signature_ed25519_hex: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReviewerKey {
    key_id: String,
    reviewer_id: String,
    public_key_pem: String,
    status: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReviewerKeyRegistry {
    schema_version: String,
    registry_id: String,
    keys: Vec<ReviewerKey>,
}

#[derive(Debug, Clone, Serialize)]
struct RunClosure {
    run_id: String,
    task_id: String,
    outcome: String,
    closure_status: String,
    source_file_sha256: Option<String>,
    modified_source_sha256: Option<String>,
    proof_artifact_sha256: Option<String>,
    checker_receipt_sha256: Option<String>,
    semantic_gate_receipt_sha256: Option<String>,
    source_binding_receipt_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Report {
    schema_version: String,
    evaluation_split_git_blob_sha1: String,
    evaluation_split_raw_sha256: String,
    proof_runs_jsonl_sha256: String,
    reviewer_key_registry_sha256: String,
    attempt_count: usize,
    verified_success_run_count: usize,
    closed_verified_success_run_count: usize,
    closed_unique_task_count: usize,
    non_success_run_count: usize,
    run_closures: Vec<RunClosure>,
    admission_boundary: String,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn digest(value: &str, label: &str) -> Result<(), String> {
    if valid_hex(value, 64) {
        Ok(())
    } else {
        Err(format!("{label} must be a 64-character lowercase SHA-256"))
    }
}

fn non_empty(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{label} must be non-empty"))
    } else {
        Ok(())
    }
}

fn git_blob_sha1(bytes: &[u8]) -> Result<String, String> {
    let mut child = Command::new("git")
        .args(["hash-object", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("start git hash-object: {e}"))?;
    child.stdin.as_mut().ok_or("git hash-object stdin missing")?
        .write_all(bytes).map_err(|e| format!("write Git blob bytes: {e}"))?;
    let output = child.wait_with_output().map_err(|e| format!("wait for git hash-object: {e}"))?;
    if !output.status.success() {
        return Err(format!("git hash-object failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    let value = String::from_utf8(output.stdout)
        .map_err(|e| format!("Git blob digest is not UTF-8: {e}"))?
        .trim().to_owned();
    if !valid_hex(&value, 40) {
        return Err("git hash-object returned a malformed SHA-1".into());
    }
    Ok(value)
}

fn position_to_byte_index(source: &str, line: u64, column: u64) -> Result<usize, String> {
    if line == 0 {
        return Err("SorryDB lines are 1-based".into());
    }
    let line_index = usize::try_from(line - 1).map_err(|_| "line coordinate overflows usize")?;
    let column_index = usize::try_from(column).map_err(|_| "column coordinate overflows usize")?;
    let lines: Vec<&str> = source.split('\n').collect();
    let selected = lines.get(line_index)
        .ok_or_else(|| format!("source line {line} is outside the committed file"))?;
    let char_count = selected.chars().count();
    if column_index > char_count {
        return Err(format!("source column {column} exceeds line {line} character count {char_count}"));
    }
    let byte_column = selected.char_indices().nth(column_index)
        .map(|(offset, _)| offset).unwrap_or(selected.len());
    let byte_prefix = lines.iter().take(line_index).map(|line| line.len() + 1).sum::<usize>();
    Ok(byte_prefix + byte_column)
}

fn safe_run_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

fn read_required(root: &Path, run_id: &str, name: &str) -> Result<Vec<u8>, String> {
    fs::read(root.join(run_id).join(name))
        .map_err(|e| format!("{run_id}/{name} missing or unreadable: {e}"))
}

fn parse_runs(bytes: &[u8]) -> Result<Vec<ProofRun>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("proof-run JSONL is not UTF-8: {e}"))?;
    let mut runs = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() { continue; }
        let run: ProofRun = serde_json::from_str(line)
            .map_err(|e| format!("proof-run JSONL line {}: {e}", index + 1))?;
        runs.push(run);
    }
    if runs.is_empty() {
        return Err("proof-run JSONL contains no rows".into());
    }
    Ok(runs)
}

fn load_task_map(bytes: &[u8], enforce_pin: bool) -> Result<BTreeMap<String, SorryRecord>, String> {
    let blob = git_blob_sha1(bytes)?;
    if enforce_pin && blob != EXPECTED_EVALUATION_BLOB {
        return Err(format!("SorryDB evaluation split blob mismatch: expected {EXPECTED_EVALUATION_BLOB}, got {blob}"));
    }
    let dataset: Dataset = serde_json::from_slice(bytes)
        .map_err(|e| format!("parse SorryDB evaluation split: {e}"))?;
    if enforce_pin && dataset.sorries.len() != EXPECTED_EVALUATION_COUNT {
        return Err(format!("SorryDB evaluation split must contain {EXPECTED_EVALUATION_COUNT} tasks"));
    }
    let mut map = BTreeMap::new();
    for task in dataset.sorries {
        if map.insert(task.id.clone(), task).is_some() {
            return Err("duplicate SorryDB evaluation task ID".into());
        }
    }
    if enforce_pin && map.len() != EXPECTED_EVALUATION_COUNT {
        return Err("SorryDB evaluation task IDs are not unique".into());
    }
    Ok(map)
}

fn check_run_task(run: &ProofRun, task: &SorryRecord) -> Result<(), String> {
    if run.schema_version != RUN_SCHEMA {
        return Err(format!("run {} has unsupported proof-run schema", run.run_id));
    }
    non_empty(&run.run_id, "proof-run run_id")?;
    if !safe_run_id(&run.run_id) {
        return Err(format!("run ID {} is not a safe evidence directory name", run.run_id));
    }
    if run.task_id != task.id
        || run.repo_commit != task.repo.commit
        || run.lean_version != task.repo.lean_version
        || run.source_path != task.location.path
        || run.start_line != task.location.start_line
        || run.start_column != task.location.start_column
        || run.end_line != task.location.end_line
        || run.end_column != task.location.end_column
        || run.goal_sha256 != sha256(task.debug_info.goal.as_bytes())
    {
        return Err(format!("run {} does not bind to the exact SorryDB evaluation task", run.run_id));
    }
    for (label, value) in [
        ("goal_sha256", run.goal_sha256.as_str()),
        ("declaration_signature_before_sha256", run.declaration_signature_before_sha256.as_str()),
        ("declaration_signature_after_sha256", run.declaration_signature_after_sha256.as_str()),
        ("source_file_sha256", run.source_file_sha256.as_str()),
        ("replacement_patch_sha256", run.replacement_patch_sha256.as_str()),
        ("proof_artifact_sha256", run.proof_artifact_sha256.as_str()),
        ("checker_receipt_sha256", run.checker_receipt_sha256.as_str()),
        ("semantic_gate_receipt_sha256", run.semantic_gate_receipt_sha256.as_str()),
        ("source_binding_receipt_sha256", run.source_binding_receipt_sha256.as_str()),
    ] {
        digest(value, &format!("run {} {label}", run.run_id))?;
    }
    if !matches!(run.outcome.as_str(), "verified_success" | "non_solve" | "infrastructure_failure" | "invalid_target") {
        return Err(format!("run {} has unsupported outcome {}", run.run_id, run.outcome));
    }
    if run.outcome == "verified_success" {
        let gates = [
            run.proof_inserted, run.exact_sorry_replaced, run.sorry_absent_after_replacement,
            run.target_sorry_ax_absent, run.target_is_prop, run.build_passed, run.kernel_passed,
            run.semantic_gate_passed, run.source_binding_passed, run.vacuity_gate_passed,
            run.axiom_delta_zero, run.new_sorry_delta_zero,
        ];
        if gates.iter().any(|v| !v) {
            return Err(format!("run {} claims success without all assurance gates", run.run_id));
        }
        if run.declaration_signature_before_sha256 != run.declaration_signature_after_sha256 {
            return Err(format!("run {} changes the target declaration signature", run.run_id));
        }
    }
    Ok(())
}


fn decode_hex(value: &str) -> Result<Vec<u8>, String> {
    if value.len() % 2 != 0 || !value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("signature must be even-length lowercase hexadecimal".into());
    }
    (0..value.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&value[i..i + 2], 16).map_err(|e| format!("decode signature: {e}")))
        .collect()
}

fn openssl_verify_ed25519(public_key_pem: &str, payload: &[u8], signature: &[u8], temp_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(temp_dir).map_err(|e| format!("create reviewer signature temp directory: {e}"))?;
    let public_key_path = temp_dir.join("reviewer-public-key.pem");
    let payload_path = temp_dir.join("reviewer-attestation-payload.json");
    let signature_path = temp_dir.join("reviewer-attestation-signature.bin");
    fs::write(&public_key_path, public_key_pem.as_bytes()).map_err(|e| format!("write reviewer public key: {e}"))?;
    fs::write(&payload_path, payload).map_err(|e| format!("write reviewer signed payload: {e}"))?;
    fs::write(&signature_path, signature).map_err(|e| format!("write reviewer signature: {e}"))?;
    let output = Command::new("openssl")
        .args([
            "pkeyutl", "-verify", "-pubin", "-inkey",
            public_key_path.to_str().ok_or("reviewer public key path is not UTF-8")?,
            "-rawin", "-in",
            payload_path.to_str().ok_or("reviewer payload path is not UTF-8")?,
            "-sigfile",
            signature_path.to_str().ok_or("reviewer signature path is not UTF-8")?,
        ])
        .output()
        .map_err(|e| format!("run OpenSSL Ed25519 verifier: {e}"))?;
    let _ = fs::remove_dir_all(temp_dir);
    if !output.status.success() {
        return Err(format!(
            "reviewer Ed25519 signature verification failed: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

fn verify_reviewer_attestation(
    semantic: &SemanticGateReceipt,
    attestation_bytes: &[u8],
    registry_bytes: &[u8],
    expected_registry_sha256: &str,
    temp_root: &Path,
    run_id: &str,
) -> Result<(), String> {
    digest(expected_registry_sha256, "out-of-band reviewer key-registry SHA-256")?;
    let actual_registry_sha256 = sha256(registry_bytes);
    if actual_registry_sha256 != expected_registry_sha256 {
        return Err("reviewer key registry differs from the out-of-band pinned SHA-256".into());
    }
    let registry: ReviewerKeyRegistry = serde_json::from_slice(registry_bytes)
        .map_err(|e| format!("parse reviewer key registry: {e}"))?;
    if registry.schema_version != "pitb.sorrydb-reviewer-key-registry.v1" || registry.registry_id.trim().is_empty() {
        return Err("unsupported or unnamed reviewer key registry".into());
    }
    if registry.keys.is_empty() {
        return Err("reviewer key registry cannot be empty".into());
    }
    let mut previous_key_id: Option<&str> = None;
    let mut matching_key: Option<&ReviewerKey> = None;
    for key in &registry.keys {
        if key.key_id.trim().is_empty() || key.reviewer_id.trim().is_empty()
            || !key.public_key_pem.contains("-----BEGIN PUBLIC KEY-----")
            || !key.public_key_pem.contains("-----END PUBLIC KEY-----")
        {
            return Err("reviewer key registry contains an invalid key entry".into());
        }
        if previous_key_id.is_some_and(|previous| previous >= key.key_id.as_str()) {
            return Err("reviewer key registry must be sorted and unique by key_id".into());
        }
        previous_key_id = Some(&key.key_id);
        if !matches!(key.status.as_str(), "active" | "revoked") {
            return Err(format!("reviewer key {} has unknown status", key.key_id));
        }
    }

    let attestation: ReviewerAttestation = serde_json::from_slice(attestation_bytes)
        .map_err(|e| format!("parse reviewer attestation: {e}"))?;
    if attestation.schema_version != "pitb.sorrydb-reviewer-attestation.v1"
        || attestation.canonicalization != "serde-json-map-sorted-compact-v0"
        || attestation.key_id.trim().is_empty()
        || attestation.reviewer_id.trim().is_empty()
        || attestation.run_id != semantic.run_id
        || attestation.task_id != semantic.task_id
        || attestation.goal_sha256 != semantic.goal_sha256
        || attestation.declaration_signature_sha256 != semantic.declaration_signature_sha256
        || attestation.source_binding_receipt_sha256 != semantic.source_binding_receipt_sha256
        || attestation.source_correspondence_status != semantic.source_correspondence_status
        || attestation.hypothesis_audit_status != semantic.hypothesis_audit_status
        || attestation.vacuity_audit_status != semantic.vacuity_audit_status
        || attestation.reviewer_decision != semantic.reviewer_decision
        || attestation.reviewer_id != semantic.reviewer_id
        || attestation.key_id != semantic.reviewer_key_id
        || attestation.semantic_log_sha256 != semantic.semantic_log_sha256
    {
        return Err(format!("run {} reviewer attestation does not bind the semantic-gate receipt", run_id));
    }

    for key in &registry.keys {
        if key.key_id == attestation.key_id {
            matching_key = Some(key);
            break;
        }
    }
    let key = matching_key.ok_or("reviewer attestation key ID is absent from pinned registry")?;
    if key.status != "active" {
        return Err(format!("reviewer key {} is revoked", key.key_id));
    }
    if key.reviewer_id != attestation.reviewer_id {
        return Err("reviewer identity does not match pinned key registry".into());
    }

    let mut payload_value: Value = serde_json::from_slice(attestation_bytes)
        .map_err(|e| format!("parse reviewer attestation for canonicalization: {e}"))?;
    payload_value.as_object_mut()
        .ok_or("reviewer attestation must be a JSON object")?
        .remove("signature_ed25519_hex");
    let payload = serde_json::to_vec(&payload_value)
        .map_err(|e| format!("canonicalize reviewer attestation: {e}"))?;
    let signature = decode_hex(&attestation.signature_ed25519_hex)?;
    if signature.len() != 64 {
        return Err("Ed25519 signature must contain exactly 64 bytes".into());
    }
    let temp_dir = temp_root.join(format!("reviewer-verify-{}", sha256(run_id.as_bytes())));
    openssl_verify_ed25519(&key.public_key_pem, &payload, &signature, &temp_dir)?;
    Ok(())
}

fn verify_bundle(
    run: &ProofRun,
    task: &SorryRecord,
    evidence_root: &Path,
    reviewer_registry_bytes: &[u8],
    reviewer_registry_sha256: &str,
) -> Result<RunClosure, String> {
    check_run_task(run, task)?;
    if run.outcome != "verified_success" {
        return Ok(RunClosure {
            run_id: run.run_id.clone(), task_id: run.task_id.clone(), outcome: run.outcome.clone(),
            closure_status: "not_required_non_success".into(),
            source_file_sha256: None, modified_source_sha256: None,
            proof_artifact_sha256: None, checker_receipt_sha256: None,
            semantic_gate_receipt_sha256: None, source_binding_receipt_sha256: None,
        });
    }

    let source_before = read_required(evidence_root, &run.run_id, "source-before.lean")?;
    let source_after = read_required(evidence_root, &run.run_id, "source-after.lean")?;
    let proof = read_required(evidence_root, &run.run_id, "proof.txt")?;
    let patch = read_required(evidence_root, &run.run_id, "replacement.patch")?;
    let checker_log = read_required(evidence_root, &run.run_id, "checker.log")?;
    let semantic_log = read_required(evidence_root, &run.run_id, "semantic-audit.log")?;
    let reproduction_log = read_required(evidence_root, &run.run_id, "source-reproduction.log")?;
    let lake_manifest = read_required(evidence_root, &run.run_id, "lake-manifest.json")?;
    let repository_tree = read_required(evidence_root, &run.run_id, "repository-tree.json")?;
    let lean_environment = read_required(evidence_root, &run.run_id, "lean-environment.json")?;
    let checker_command = read_required(evidence_root, &run.run_id, "checker-command.json")?;

    if proof.is_empty() {
        return Err(format!("run {} has an empty proof artifact", run.run_id));
    }
    for (label, bytes, expected) in [
        ("source-before.lean", &source_before, &run.source_file_sha256),
        ("proof.txt", &proof, &run.proof_artifact_sha256),
        ("replacement.patch", &patch, &run.replacement_patch_sha256),
    ] {
        if sha256(bytes) != *expected {
            return Err(format!("run {} artifact {label} SHA-256 differs from proof-run receipt", run.run_id));
        }
    }

    let source_text = std::str::from_utf8(&source_before)
        .map_err(|e| format!("run {} source-before.lean is not UTF-8: {e}", run.run_id))?;
    let start = position_to_byte_index(source_text, run.start_line, run.start_column)?;
    let end = position_to_byte_index(source_text, run.end_line, run.end_column)?;
    if end <= start || source_before.get(start..end) != Some(b"sorry".as_slice()) {
        return Err(format!("run {} pinned source range does not select exactly sorry", run.run_id));
    }
    let mut expected_after = Vec::with_capacity(source_before.len() + proof.len());
    expected_after.extend_from_slice(&source_before[..start]);
    expected_after.extend_from_slice(&proof);
    expected_after.extend_from_slice(&source_before[end..]);
    if expected_after != source_after {
        return Err(format!("run {} source-after.lean is not the exact single-span replacement", run.run_id));
    }

    let source_receipt_bytes = read_required(evidence_root, &run.run_id, "source-binding-receipt.json")?;
    let checker_receipt_bytes = read_required(evidence_root, &run.run_id, "checker-receipt.json")?;
    let semantic_receipt_bytes = read_required(evidence_root, &run.run_id, "semantic-gate-receipt.json")?;
    let reviewer_attestation_bytes = read_required(evidence_root, &run.run_id, "reviewer-attestation.json")?;
    if sha256(&source_receipt_bytes) != run.source_binding_receipt_sha256 {
        return Err(format!("run {} source-binding receipt digest mismatch", run.run_id));
    }
    if sha256(&checker_receipt_bytes) != run.checker_receipt_sha256 {
        return Err(format!("run {} checker receipt digest mismatch", run.run_id));
    }
    if sha256(&semantic_receipt_bytes) != run.semantic_gate_receipt_sha256 {
        return Err(format!("run {} semantic-gate receipt digest mismatch", run.run_id));
    }
    let reviewer_attestation_digest = sha256(&reviewer_attestation_bytes);

    let source_receipt: SourceBindingReceipt = serde_json::from_slice(&source_receipt_bytes)
        .map_err(|e| format!("run {} source-binding receipt invalid: {e}", run.run_id))?;
    let checker: CheckerReceipt = serde_json::from_slice(&checker_receipt_bytes)
        .map_err(|e| format!("run {} checker receipt invalid: {e}", run.run_id))?;
    let semantic: SemanticGateReceipt = serde_json::from_slice(&semantic_receipt_bytes)
        .map_err(|e| format!("run {} semantic-gate receipt invalid: {e}", run.run_id))?;

    if source_receipt.schema_version != "pitb.sorrydb-source-binding.v1"
        || source_receipt.run_id != run.run_id
        || source_receipt.task_id != run.task_id
        || source_receipt.repo_commit != run.repo_commit
        || source_receipt.lean_version != run.lean_version
        || source_receipt.source_path != run.source_path
        || source_receipt.start_line != run.start_line
        || source_receipt.start_column != run.start_column
        || source_receipt.end_line != run.end_line
        || source_receipt.end_column != run.end_column
        || source_receipt.source_file_sha256 != run.source_file_sha256
        || source_receipt.goal_sha256 != run.goal_sha256
        || source_receipt.source_span != "sorry"
        || source_receipt.source_span_sha256 != sha256(b"sorry")
        || !source_receipt.source_span_verified_as_sorry
        || !source_receipt.target_goal_reproduced
    {
        return Err(format!("run {} source-binding receipt does not close over exact task/source/goal identity", run.run_id));
    }
    for (label, value) in [
        ("environment_sha256", &source_receipt.environment_sha256),
        ("lake_manifest_sha256", &source_receipt.lake_manifest_sha256),
        ("repository_tree_sha256", &source_receipt.repository_tree_sha256),
        ("reproduction_log_sha256", &source_receipt.reproduction_log_sha256),
    ] {
        digest(value, &format!("run {} source {label}", run.run_id))?;
    }
    if sha256(&lake_manifest) != source_receipt.lake_manifest_sha256
        || sha256(&reproduction_log) != source_receipt.reproduction_log_sha256
        || sha256(&repository_tree) != source_receipt.repository_tree_sha256
        || sha256(&lean_environment) != source_receipt.environment_sha256
    {
        return Err(format!("run {} source environment/repository/reproduction artifacts do not match source-binding receipt", run.run_id));
    }
    let tree_manifest: Value = serde_json::from_slice(&repository_tree)
        .map_err(|e| format!("run {} repository-tree.json is invalid: {e}", run.run_id))?;
    if tree_manifest["schema_version"] != "pitb.sorrydb.repository-tree-manifest.v0"
        || tree_manifest["repository_commit"] != run.repo_commit
        || tree_manifest["source_path"] != run.source_path
        || tree_manifest["source_file_sha256"] != run.source_file_sha256
        || tree_manifest["lean_version"] != run.lean_version
        || tree_manifest["lake_manifest_sha256"] != source_receipt.lake_manifest_sha256
    {
        return Err(format!("run {} repository-tree manifest does not bind exact source and environment", run.run_id));
    }
    let environment_manifest: Value = serde_json::from_slice(&lean_environment)
        .map_err(|e| format!("run {} lean-environment.json is invalid: {e}", run.run_id))?;
    if environment_manifest["schema_version"] != "pitb.sorrydb.lean-environment-manifest.v0"
        || environment_manifest["repository_commit"] != run.repo_commit
        || environment_manifest["lean_version"] != run.lean_version
        || environment_manifest["lake_manifest_sha256"] != source_receipt.lake_manifest_sha256
        || environment_manifest["repository_tree_sha256"] != source_receipt.repository_tree_sha256
    {
        return Err(format!("run {} Lean environment manifest does not bind exact repository/toolchain", run.run_id));
    }

    if checker.schema_version != "pitb.sorrydb-checker-receipt.v1"
        || checker.run_id != run.run_id
        || checker.task_id != run.task_id
        || checker.repo_commit != run.repo_commit
        || checker.lean_version != run.lean_version
        || checker.source_file_sha256 != run.source_file_sha256
        || checker.replacement_patch_sha256 != run.replacement_patch_sha256
        || checker.proof_artifact_sha256 != run.proof_artifact_sha256
        || checker.declaration_signature_before_sha256 != run.declaration_signature_before_sha256
        || checker.declaration_signature_after_sha256 != run.declaration_signature_after_sha256
        || checker.target_is_prop != run.target_is_prop
        || checker.build_passed != run.build_passed
        || checker.kernel_passed != run.kernel_passed
        || checker.target_sorry_ax_absent != run.target_sorry_ax_absent
        || checker.axiom_delta_zero != run.axiom_delta_zero
        || checker.new_sorry_delta_zero != run.new_sorry_delta_zero
        || checker.modified_source_sha256 != sha256(&source_after)
        || checker.environment_sha256 != source_receipt.environment_sha256
    {
        return Err(format!("run {} checker receipt contradicts its proof-run record or source artifacts", run.run_id));
    }
    for (label, value) in [
        ("checker_binary_sha256", &checker.checker_binary_sha256),
        ("checker_command_sha256", &checker.checker_command_sha256),
        ("environment_sha256", &checker.environment_sha256),
        ("checker_log_sha256", &checker.checker_log_sha256),
    ] {
        digest(value, &format!("run {} checker {label}", run.run_id))?;
    }
    if sha256(&checker_log) != checker.checker_log_sha256
        || sha256(&checker_command) != checker.checker_command_sha256
    {
        return Err(format!("run {} checker log/command bytes do not match checker receipt", run.run_id));
    }
    let command_manifest: Value = serde_json::from_slice(&checker_command)
        .map_err(|e| format!("run {} checker-command.json is invalid: {e}", run.run_id))?;
    if command_manifest["schema_version"] != "pitb.sorrydb.checker-command.v0"
        || command_manifest["run_id"] != run.run_id
        || command_manifest["repo_commit"] != run.repo_commit
        || command_manifest["lean_version"] != run.lean_version
        || command_manifest["environment_sha256"] != source_receipt.environment_sha256
    {
        return Err(format!("run {} checker command manifest does not bind exact run/environment", run.run_id));
    }

    if semantic.schema_version != "pitb.sorrydb-semantic-gate-receipt.v1"
        || semantic.run_id != run.run_id
        || semantic.task_id != run.task_id
        || semantic.goal_sha256 != run.goal_sha256
        || semantic.declaration_signature_sha256 != run.declaration_signature_before_sha256
        || semantic.source_binding_receipt_sha256 != run.source_binding_receipt_sha256
        || semantic.source_correspondence_status != "pass"
        || semantic.hypothesis_audit_status != "pass"
        || semantic.vacuity_audit_status != "pass"
        || semantic.reviewer_decision != "pass"
        || semantic.reviewer_id.trim().is_empty()
        || !valid_hex(&semantic.reviewer_attestation_sha256, 64)
        || semantic.reviewer_attestation_sha256 != reviewer_attestation_digest
        || !semantic.semantic_gate_passed
        || !semantic.vacuity_gate_passed
        || semantic.semantic_gate_passed != run.semantic_gate_passed
        || semantic.vacuity_gate_passed != run.vacuity_gate_passed
    {
        return Err(format!("run {} semantic-gate receipt is unresolved, failed, or contradicts proof-run record", run.run_id));
    }
    for (label, value) in [
        ("reviewer_attestation_sha256", &semantic.reviewer_attestation_sha256),
        ("semantic_log_sha256", &semantic.semantic_log_sha256),
    ] {
        digest(value, &format!("run {} semantic {label}", run.run_id))?;
    }
    if sha256(&semantic_log) != semantic.semantic_log_sha256 {
        return Err(format!("run {} semantic audit log digest mismatch", run.run_id));
    }
    verify_reviewer_attestation(
        &semantic,
        &reviewer_attestation_bytes,
        reviewer_registry_bytes,
        reviewer_registry_sha256,
        evidence_root,
        &run.run_id,
    )?;

    Ok(RunClosure {
        run_id: run.run_id.clone(), task_id: run.task_id.clone(), outcome: run.outcome.clone(),
        closure_status: "closed_verified_success".into(),
        source_file_sha256: Some(run.source_file_sha256.clone()),
        modified_source_sha256: Some(sha256(&source_after)),
        proof_artifact_sha256: Some(run.proof_artifact_sha256.clone()),
        checker_receipt_sha256: Some(run.checker_receipt_sha256.clone()),
        semantic_gate_receipt_sha256: Some(run.semantic_gate_receipt_sha256.clone()),
        source_binding_receipt_sha256: Some(run.source_binding_receipt_sha256.clone()),
    })
}

fn verify_all(
    evaluation_bytes: &[u8],
    runs_bytes: &[u8],
    evidence_root: &Path,
    reviewer_registry_bytes: &[u8],
    reviewer_registry_sha256: &str,
    enforce_pin: bool,
) -> Result<Report, String> {
    let evaluation_blob = git_blob_sha1(evaluation_bytes)?;
    if enforce_pin && evaluation_blob != EXPECTED_EVALUATION_BLOB {
        return Err(format!("SorryDB evaluation split blob mismatch: expected {EXPECTED_EVALUATION_BLOB}, got {evaluation_blob}"));
    }
    let evaluation: Dataset = serde_json::from_slice(evaluation_bytes)
        .map_err(|e| format!("parse SorryDB evaluation split: {e}"))?;
    if enforce_pin && evaluation.sorries.len() != EXPECTED_EVALUATION_COUNT {
        return Err(format!("SorryDB evaluation split must contain {EXPECTED_EVALUATION_COUNT} tasks"));
    }
    let mut task_map = BTreeMap::<String, SorryRecord>::new();
    for task in evaluation.sorries {
        if task_map.insert(task.id.clone(), task).is_some() {
            return Err("duplicate SorryDB evaluation task ID".into());
        }
    }
    if enforce_pin && task_map.len() != EXPECTED_EVALUATION_COUNT {
        return Err("SorryDB evaluation task IDs are not unique".into());
    }
    let runs = parse_runs(runs_bytes)?;
    let mut seen_run_ids = BTreeSet::new();
    let mut closures = Vec::new();
    let mut verified_successes = 0usize;
    let mut closed_unique_tasks = BTreeSet::new();
    for run in &runs {
        if !seen_run_ids.insert(run.run_id.clone()) {
            return Err(format!("duplicate run_id {}", run.run_id));
        }
        let task = task_map.get(&run.task_id)
            .ok_or_else(|| format!("run {} task is outside the pinned evaluation split", run.run_id))?;
        check_run_task(run, task)?;
        if run.outcome == "verified_success" {
            verified_successes += 1;
            closures.push(verify_bundle(
                run, task, evidence_root, reviewer_registry_bytes, reviewer_registry_sha256,
            )?);
            closed_unique_tasks.insert(run.task_id.clone());
        } else {
            closures.push(RunClosure {
                run_id: run.run_id.clone(), task_id: run.task_id.clone(), outcome: run.outcome.clone(),
                closure_status: "not_required_non_success".into(),
                source_file_sha256: None, modified_source_sha256: None,
                proof_artifact_sha256: None, checker_receipt_sha256: None,
                semantic_gate_receipt_sha256: None, source_binding_receipt_sha256: None,
            });
        }
    }
    let closed = closures.iter().filter(|x| x.closure_status == "closed_verified_success").count();
    if closed != verified_successes {
        return Err("not every claimed verified success has a closed evidence bundle".into());
    }
    Ok(Report {
        schema_version: REPORT_SCHEMA.into(),
        evaluation_split_git_blob_sha1: evaluation_blob,
        evaluation_split_raw_sha256: sha256(evaluation_bytes),
        proof_runs_jsonl_sha256: sha256(runs_bytes),
        reviewer_key_registry_sha256: reviewer_registry_sha256.to_owned(),
        attempt_count: runs.len(),
        verified_success_run_count: verified_successes,
        closed_verified_success_run_count: closed,
        closed_unique_task_count: closed_unique_tasks.len(),
        non_success_run_count: runs.len() - verified_successes,
        run_closures: closures,
        admission_boundary: "This report verifies byte-level evidence closure and cross-binding among the pinned SorryDB task, exact source-before/source-after replacement, proof/patch artifacts, source-binding receipt, checker receipt, semantic-gate receipt, reviewer-attestation bytes, reviewer Ed25519 signature and externally pinned reviewer key registry. It does not execute Lean or re-run the checker. Public scoring still requires independent replay of the retained commands and source/specification review.".into(),
    })
}


fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}


fn create_synthetic_reviewer_identity(root: &Path) -> Result<(Vec<u8>, String, PathBuf), String> {
    let key_dir = root.join("synthetic-reviewer-keygen");
    fs::create_dir_all(&key_dir).map_err(|e| format!("create synthetic reviewer key directory: {e}"))?;
    let private_key_path = key_dir.join("private-key.pem");
    let public_key_path = key_dir.join("public-key.pem");
    let private_key = private_key_path.to_str().ok_or("synthetic private-key path is not UTF-8")?;
    let public_key = public_key_path.to_str().ok_or("synthetic public-key path is not UTF-8")?;

    let output = Command::new("openssl")
        .args(["genpkey", "-algorithm", "ED25519", "-out", private_key])
        .output()
        .map_err(|e| format!("generate synthetic Ed25519 test key: {e}"))?;
    if !output.status.success() {
        return Err(format!("generate synthetic Ed25519 key failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    let output = Command::new("openssl")
        .args(["pkey", "-in", private_key, "-pubout", "-out", public_key])
        .output()
        .map_err(|e| format!("derive synthetic reviewer public key: {e}"))?;
    if !output.status.success() {
        return Err(format!("derive synthetic reviewer public key failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    let public_pem = fs::read(&public_key_path).map_err(|e| format!("read synthetic public key: {e}"))?;
    let public_pem_text = String::from_utf8(public_pem)
        .map_err(|e| format!("synthetic reviewer public key is not UTF-8: {e}"))?;
    let registry = json!({
        "schema_version": "pitb.sorrydb-reviewer-key-registry.v1",
        "registry_id": "synthetic-registry-test-only",
        "keys": [{
            "key_id": "synthetic-reviewer-key-v1",
            "reviewer_id": "synthetic-reviewer",
            "public_key_pem": public_pem_text,
            "status": "active"
        }]
    });
    let registry_bytes = serde_json::to_vec(&registry)
        .map_err(|e| format!("serialize synthetic reviewer registry: {e}"))?;
    let registry_sha256 = sha256(&registry_bytes);
    fs::write(root.join("reviewer-key-registry.json"), &registry_bytes)
        .map_err(|e| format!("write synthetic reviewer registry: {e}"))?;
    Ok((registry_bytes, registry_sha256, private_key_path))
}

fn sign_synthetic_attestation(
    attestation_without_signature: &Value,
    private_key_path: &Path,
    temp_dir: &Path,
) -> Result<String, String> {
    fs::create_dir_all(temp_dir).map_err(|e| format!("create synthetic signing directory: {e}"))?;
    let payload_path = temp_dir.join("reviewer-attestation-payload.json");
    let signature_path = temp_dir.join("reviewer-attestation-signature.bin");
    let mut payload_value = attestation_without_signature.clone();
    payload_value.as_object_mut()
        .ok_or("synthetic reviewer attestation must be a JSON object")?
        .remove("signature_ed25519_hex");
    let payload = serde_json::to_vec(&payload_value)
        .map_err(|e| format!("serialize synthetic signed reviewer payload: {e}"))?;
    fs::write(&payload_path, &payload).map_err(|e| format!("write synthetic reviewer payload: {e}"))?;
    let private_key = private_key_path.to_str().ok_or("synthetic private-key path is not UTF-8")?;
    let payload_file = payload_path.to_str().ok_or("synthetic reviewer payload path is not UTF-8")?;
    let signature_file = signature_path.to_str().ok_or("synthetic reviewer signature path is not UTF-8")?;
    let output = Command::new("openssl")
        .args(["pkeyutl", "-sign", "-inkey", private_key, "-rawin", "-in", payload_file, "-out", signature_file])
        .output()
        .map_err(|e| format!("sign synthetic reviewer attestation: {e}"))?;
    if !output.status.success() {
        return Err(format!("sign synthetic reviewer attestation failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    let signature = fs::read(&signature_path).map_err(|e| format!("read synthetic reviewer signature: {e}"))?;
    if signature.len() != 64 {
        return Err("synthetic Ed25519 signature is not 64 bytes".into());
    }
    Ok(encode_hex(&signature))
}

fn make_synthetic_bundle(root: &Path) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, String), String> {
    let (reviewer_registry_bytes, reviewer_registry_sha256, reviewer_private_key_path) =
        create_synthetic_reviewer_identity(root)?;
    let run_id = "synthetic-run";
    let dir = root.join(run_id);
    fs::create_dir_all(&dir).map_err(|e| format!("create synthetic bundle: {e}"))?;
    let source_before = b"import Lean\ntheorem sample : True := by sorry\n".to_vec();
    let source_text = std::str::from_utf8(&source_before).map_err(|e| e.to_string())?;
    let offset = source_text.find("sorry").ok_or("synthetic source missing sorry")?;
    let prefix = &source_text[..offset];
    let line = prefix.bytes().filter(|b| *b == b'\n').count() as u64 + 1;
    let column = prefix.rsplit('\n').next().unwrap().chars().count() as u64;
    let proof = b"trivial".to_vec();
    let patch = b"synthetic exact replacement patch\n".to_vec();
    let mut source_after = Vec::new();
    source_after.extend_from_slice(&source_before[..offset]);
    source_after.extend_from_slice(&proof);
    source_after.extend_from_slice(&source_before[offset + 5..]);
    let checker_log = b"synthetic checker replay pass\n".to_vec();
    let semantic_log = b"synthetic semantic review pass\n".to_vec();
    let reproduction_log = b"synthetic Lean goal reproduced\n".to_vec();
    let lake_manifest = b"{\"version\":1}\n".to_vec();
    let repository_tree = json!({
        "schema_version":"pitb.sorrydb.repository-tree-manifest.v0",
        "repository_commit":"1".repeat(40),
        "source_path":"Demo/Sample.lean",
        "source_file_sha256":sha256(&source_before),
        "lean_version":"v4.27.0",
        "lake_manifest_sha256":sha256(&lake_manifest)
    });
    let repository_tree_bytes = serde_json::to_vec(&repository_tree).map_err(|e| e.to_string())?;
    let repository_tree_sha256 = sha256(&repository_tree_bytes);
    let lean_environment = json!({
        "schema_version":"pitb.sorrydb.lean-environment-manifest.v0",
        "repository_commit":"1".repeat(40),
        "lean_version":"v4.27.0",
        "lake_manifest_sha256":sha256(&lake_manifest),
        "repository_tree_sha256":repository_tree_sha256
    });
    let lean_environment_bytes = serde_json::to_vec(&lean_environment).map_err(|e| e.to_string())?;
    let environment_sha256 = sha256(&lean_environment_bytes);
    let checker_command = json!({
        "schema_version":"pitb.sorrydb.checker-command.v0",
        "run_id":run_id,
        "repo_commit":"1".repeat(40),
        "lean_version":"v4.27.0",
        "environment_sha256":environment_sha256
    });
    let checker_command_bytes = serde_json::to_vec(&checker_command).map_err(|e| e.to_string())?;
    let checker_command_sha256 = sha256(&checker_command_bytes);

    let task = json!({
        "id": "a".repeat(64),
        "repo": {"commit": "1".repeat(40), "lean_version": "v4.27.0"},
        "location": {"path":"Demo/Sample.lean","start_line":line,"start_column":column,"end_line":line,"end_column":column+5},
        "debug_info": {"goal":"⊢ True"}
    });
    let goal_sha = sha256("⊢ True".as_bytes());
    let sig = sha256(b"theorem sample : True");
    let source_file_sha = sha256(&source_before);
    let modified_source_sha = sha256(&source_after);
    let proof_sha = sha256(&proof);
    let patch_sha = sha256(&patch);

    let source_receipt = json!({
        "schema_version":"pitb.sorrydb-source-binding.v1","run_id":run_id,"task_id":"a".repeat(64),
        "repo_commit":"1".repeat(40),"lean_version":"v4.27.0","source_path":"Demo/Sample.lean",
        "start_line":line,"start_column":column,"end_line":line,"end_column":column+5,
        "source_file_sha256":source_file_sha,"goal_sha256":goal_sha,
        "source_span_sha256":sha256(b"sorry"),"source_span":"sorry",
        "source_span_verified_as_sorry":true,"target_goal_reproduced":true,
        "environment_sha256":environment_sha256,
        "lake_manifest_sha256":sha256(&lake_manifest),"repository_tree_sha256":repository_tree_sha256,
        "reproduction_log_sha256":sha256(&reproduction_log)
    });
    let source_receipt_bytes = serde_json::to_vec(&source_receipt).map_err(|e| e.to_string())?;
    let checker_receipt = json!({
        "schema_version":"pitb.sorrydb-checker-receipt.v1","run_id":run_id,"task_id":"a".repeat(64),
        "repo_commit":"1".repeat(40),"lean_version":"v4.27.0","source_file_sha256":source_file_sha,
        "modified_source_sha256":modified_source_sha,"replacement_patch_sha256":patch_sha,
        "proof_artifact_sha256":proof_sha,"declaration_signature_before_sha256":sig,
        "declaration_signature_after_sha256":sig,"target_is_prop":true,"build_passed":true,
        "kernel_passed":true,"target_sorry_ax_absent":true,"axiom_delta_zero":true,"new_sorry_delta_zero":true,
        "checker_binary_sha256":sha256(b"checker binary"),"checker_command_sha256":checker_command_sha256,
        "environment_sha256":environment_sha256,"checker_log_sha256":sha256(&checker_log)
    });
    let checker_receipt_bytes = serde_json::to_vec(&checker_receipt).map_err(|e| e.to_string())?;
    let mut reviewer_attestation = json!({
        "schema_version":"pitb.sorrydb-reviewer-attestation.v1",
        "canonicalization":"serde-json-map-sorted-compact-v0",
        "key_id":"synthetic-reviewer-key-v1",
        "reviewer_id":"synthetic-reviewer",
        "run_id":run_id,
        "task_id":"a".repeat(64),
        "goal_sha256":goal_sha,
        "declaration_signature_sha256":sig,
        "source_binding_receipt_sha256":sha256(&source_receipt_bytes),
        "source_correspondence_status":"pass",
        "hypothesis_audit_status":"pass",
        "vacuity_audit_status":"pass",
        "reviewer_decision":"pass",
        "semantic_log_sha256":sha256(&semantic_log),
        "signature_ed25519_hex":""
    });
    let signature = sign_synthetic_attestation(
        &reviewer_attestation,
        &reviewer_private_key_path,
        &root.join("synthetic-reviewer-signing-temp"),
    )?;
    reviewer_attestation["signature_ed25519_hex"] = Value::String(signature);
    let reviewer_attestation_bytes = serde_json::to_vec(&reviewer_attestation)
        .map_err(|e| format!("serialize synthetic reviewer attestation: {e}"))?;
    let reviewer_attestation_sha256 = sha256(&reviewer_attestation_bytes);

    let semantic_receipt = json!({
        "schema_version":"pitb.sorrydb-semantic-gate-receipt.v1","run_id":run_id,"task_id":"a".repeat(64),
        "goal_sha256":goal_sha,"declaration_signature_sha256":sig,
        "source_binding_receipt_sha256":sha256(&source_receipt_bytes),
        "source_correspondence_status":"pass","hypothesis_audit_status":"pass",
        "vacuity_audit_status":"pass","reviewer_decision":"pass","reviewer_id":"synthetic-reviewer",
        "reviewer_key_id":"synthetic-reviewer-key-v1",
        "reviewer_attestation_sha256":reviewer_attestation_sha256,"semantic_log_sha256":sha256(&semantic_log),
        "semantic_gate_passed":true,"vacuity_gate_passed":true
    });
    let semantic_receipt_bytes = serde_json::to_vec(&semantic_receipt).map_err(|e| e.to_string())?;

    for (name, bytes) in [
        ("source-before.lean", &source_before),("source-after.lean", &source_after),
        ("proof.txt", &proof),("replacement.patch", &patch),("checker.log", &checker_log),
        ("semantic-audit.log", &semantic_log),("source-reproduction.log", &reproduction_log),
        ("lake-manifest.json", &lake_manifest),("repository-tree.json", &repository_tree_bytes),
        ("lean-environment.json", &lean_environment_bytes),("checker-command.json", &checker_command_bytes),
        ("source-binding-receipt.json", &source_receipt_bytes),
        ("checker-receipt.json", &checker_receipt_bytes),("semantic-gate-receipt.json", &semantic_receipt_bytes),
        ("reviewer-attestation.json", &reviewer_attestation_bytes),
    ] {
        fs::write(dir.join(name), bytes).map_err(|e| format!("write synthetic {name}: {e}"))?;
    }

    let run = json!({
        "schema_version":RUN_SCHEMA,"run_id":run_id,"task_id":"a".repeat(64),
        "repo_commit":"1".repeat(40),"lean_version":"v4.27.0","source_path":"Demo/Sample.lean",
        "start_line":line,"start_column":column,"end_line":line,"end_column":column+5,"goal_sha256":goal_sha,
        "declaration_signature_before_sha256":sig,"declaration_signature_after_sha256":sig,
        "source_file_sha256":source_file_sha,"replacement_patch_sha256":patch_sha,"proof_artifact_sha256":proof_sha,
        "checker_receipt_sha256":sha256(&checker_receipt_bytes),"semantic_gate_receipt_sha256":sha256(&semantic_receipt_bytes),
        "source_binding_receipt_sha256":sha256(&source_receipt_bytes),"outcome":"verified_success",
        "proof_inserted":true,"exact_sorry_replaced":true,"sorry_absent_after_replacement":true,
        "target_sorry_ax_absent":true,"target_is_prop":true,"build_passed":true,"kernel_passed":true,
        "semantic_gate_passed":true,"source_binding_passed":true,"vacuity_gate_passed":true,
        "axiom_delta_zero":true,"new_sorry_delta_zero":true,
        "strategy":{"name":"synthetic-strategy","version":"v0","implementation_sha256":sha256(b"synthetic-strategy"),"model_id":null,"model_version":null},
        "wall_time_ms":1,"input_tokens":1,"output_tokens":1,"cost_microusd":1,"failure_type":null
    });
    let run_bytes = [serde_json::to_vec(&run).map_err(|e| e.to_string())?, b"\n".to_vec()].concat();
    let evaluation = json!({"sorries":[task]});
    let evaluation_bytes = serde_json::to_vec(&evaluation).map_err(|e| e.to_string())?;
    let _ = fs::remove_dir_all(root.join("synthetic-reviewer-keygen"));
    let _ = fs::remove_dir_all(root.join("synthetic-reviewer-signing-temp"));
    Ok((evaluation_bytes, run_bytes, reviewer_registry_bytes, reviewer_registry_sha256))
}

fn run_self_tests() -> Result<(), String> {
    let root = env::temp_dir().join(format!("pitb-sorrydb-evidence-closure-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    let (evaluation_bytes, runs_bytes, reviewer_registry_bytes, reviewer_registry_sha256) =
        make_synthetic_bundle(&root)?;
    let report = verify_all(
        &evaluation_bytes, &runs_bytes, &root,
        &reviewer_registry_bytes, &reviewer_registry_sha256, false,
    )?;
    if report.closed_verified_success_run_count != 1 || report.closed_unique_task_count != 1 {
        return Err("synthetic evidence closure denominator control failed".into());
    }
    println!("SELF_TEST_PASS source-before/source-after are exact single-span sorry replacement");

    let run_dir = root.join("synthetic-run");
    let semantic_receipt_bytes = fs::read(run_dir.join("semantic-gate-receipt.json"))
        .map_err(|e| format!("read synthetic semantic receipt: {e}"))?;
    let semantic: SemanticGateReceipt = serde_json::from_slice(&semantic_receipt_bytes)
        .map_err(|e| format!("parse synthetic semantic receipt: {e}"))?;
    let attestation_bytes = fs::read(run_dir.join("reviewer-attestation.json"))
        .map_err(|e| format!("read synthetic reviewer attestation: {e}"))?;

    let wrong_registry_pin = "0".repeat(64);
    if verify_reviewer_attestation(
        &semantic, &attestation_bytes, &reviewer_registry_bytes,
        &wrong_registry_pin, &root, "synthetic-run",
    ).is_ok() {
        return Err("wrong out-of-band reviewer key-registry digest was accepted".into());
    }
    println!("SELF_TEST_PASS out-of-band reviewer registry digest is mandatory");

    let mut tampered_attestation: Value = serde_json::from_slice(&attestation_bytes)
        .map_err(|e| format!("parse synthetic reviewer attestation: {e}"))?;
    let signature = tampered_attestation["signature_ed25519_hex"].as_str()
        .ok_or("synthetic reviewer signature missing")?;
    let replacement = if signature.starts_with('0') { '1' } else { '0' };
    let mut changed_signature = signature.to_owned();
    changed_signature.replace_range(0..1, &replacement.to_string());
    tampered_attestation["signature_ed25519_hex"] = Value::String(changed_signature);
    let tampered_attestation_bytes = serde_json::to_vec(&tampered_attestation)
        .map_err(|e| format!("serialize tampered reviewer attestation: {e}"))?;
    if verify_reviewer_attestation(
        &semantic, &tampered_attestation_bytes, &reviewer_registry_bytes,
        &reviewer_registry_sha256, &root, "synthetic-run",
    ).is_ok() {
        return Err("tampered reviewer signature was accepted".into());
    }
    println!("SELF_TEST_PASS tampered reviewer signature is rejected");

    let mut revoked_registry: Value = serde_json::from_slice(&reviewer_registry_bytes)
        .map_err(|e| format!("parse synthetic reviewer registry: {e}"))?;
    revoked_registry["keys"][0]["status"] = Value::String("revoked".into());
    let revoked_registry_bytes = serde_json::to_vec(&revoked_registry)
        .map_err(|e| format!("serialize revoked reviewer registry: {e}"))?;
    let revoked_registry_sha256 = sha256(&revoked_registry_bytes);
    if verify_reviewer_attestation(
        &semantic, &attestation_bytes, &revoked_registry_bytes,
        &revoked_registry_sha256, &root, "synthetic-run",
    ).is_ok() {
        return Err("revoked reviewer key was accepted".into());
    }
    println!("SELF_TEST_PASS revoked reviewer key is rejected");

    let source_after_path = run_dir.join("source-after.lean");
    let valid_after = fs::read(&source_after_path).map_err(|e| e.to_string())?;
    fs::write(&source_after_path, b"unrelated source edit\n").map_err(|e| e.to_string())?;
    if verify_all(&evaluation_bytes, &runs_bytes, &root, &reviewer_registry_bytes, &reviewer_registry_sha256, false).is_ok() {
        return Err("unrelated source edit was accepted".into());
    }
    fs::write(&source_after_path, &valid_after).map_err(|e| e.to_string())?;
    println!("SELF_TEST_PASS unrelated source mutations are rejected");

    let semantic_path = run_dir.join("semantic-gate-receipt.json");
    let semantic_bytes = fs::read(&semantic_path).map_err(|e| e.to_string())?;
    let mut semantic: Value = serde_json::from_slice(&semantic_bytes).map_err(|e| e.to_string())?;
    semantic["source_correspondence_status"] = Value::String("unresolved".into());
    let semantic_mutated_bytes = serde_json::to_vec(&semantic).map_err(|e| e.to_string())?;
    fs::write(&semantic_path, &semantic_mutated_bytes).map_err(|e| e.to_string())?;
    let run_line = runs_bytes.strip_suffix(b"\\n").unwrap_or(&runs_bytes);
    let mut run_value: Value = serde_json::from_slice(run_line).map_err(|e| e.to_string())?;
    run_value["semantic_gate_receipt_sha256"] = Value::String(sha256(&semantic_mutated_bytes));
    let rebound_runs = [serde_json::to_vec(&run_value).map_err(|e| e.to_string())?, b"\\n".to_vec()].concat();
    if verify_all(&evaluation_bytes, &rebound_runs, &root, &reviewer_registry_bytes, &reviewer_registry_sha256, false).is_ok() {
        return Err("unresolved semantic review was accepted after valid digest rebinding".into());
    }
    println!("SELF_TEST_PASS unresolved semantic status is rejected after digest rebinding");
    fs::write(&semantic_path, &semantic_bytes).map_err(|e| e.to_string())?;

    let checker_log_path = run_dir.join("checker.log");
    let checker_log = fs::read(&checker_log_path).map_err(|e| e.to_string())?;
    fs::write(&checker_log_path, b"tampered checker log").map_err(|e| e.to_string())?;
    if verify_all(&evaluation_bytes, &runs_bytes, &root, &reviewer_registry_bytes, &reviewer_registry_sha256, false).is_ok() {
        return Err("tampered checker log was accepted".into());
    }
    fs::write(&checker_log_path, checker_log).map_err(|e| e.to_string())?;
    println!("SELF_TEST_PASS checker and semantic logs are digest-closed");

    let proof_path = run_dir.join("proof.txt");
    let proof = fs::read(&proof_path).map_err(|e| e.to_string())?;
    fs::remove_file(&proof_path).map_err(|e| e.to_string())?;
    if verify_all(&evaluation_bytes, &runs_bytes, &root, &reviewer_registry_bytes, &reviewer_registry_sha256, false).is_ok() {
        return Err("missing proof artifact was accepted".into());
    }
    fs::write(&proof_path, proof).map_err(|e| e.to_string())?;
    println!("SELF_TEST_PASS every claimed success requires exact proof artifact bytes");
    let _ = fs::remove_dir_all(&root);
    Ok(())
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        run_self_tests()?;
        println!("PITB_SORRYDB_EVIDENCE_CLOSURE_SELF_TEST_PASS");
        return Ok(());
    }
    if args.len() == 4 && args[1] == "--self-test-report" {
        run_self_tests()?;
        let root = PathBuf::from(&args[3]);
        fs::create_dir_all(&root).map_err(|e| format!("create evidence bundle root: {e}"))?;
        let (evaluation, runs, reviewer_registry, reviewer_registry_sha256) = make_synthetic_bundle(&root)?;
        fs::write(root.join("reviewer-key-registry.json"), &reviewer_registry)
            .map_err(|e| format!("write synthetic reviewer registry: {e}"))?;
        fs::write(root.join("reviewer-key-registry.sha256"), &reviewer_registry_sha256)
            .map_err(|e| format!("write synthetic reviewer registry pin: {e}"))?;
        let report = verify_all(
            &evaluation, &runs, &root, &reviewer_registry, &reviewer_registry_sha256, false,
        )?;
        fs::write(root.join("evaluation-split.json"), &evaluation)
            .map_err(|e| format!("write synthetic evaluation split: {e}"))?;
        fs::write(root.join("proof-runs.jsonl"), &runs)
            .map_err(|e| format!("write synthetic proof-run JSONL: {e}"))?;
        fs::write(&args[2], serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)
            .map_err(|e| format!("write evidence closure report: {e}"))?;
        println!("PITB_SORRYDB_EVIDENCE_CLOSURE_SYNTHETIC_REPORT_OK");
        return Ok(());
    }
    if args.len() != 8 || args[1] != "--verify" {
        return Err(format!(
            "usage: {} --verify <evaluation-split.json> <proof-runs.jsonl> <evidence-root> <reviewer-key-registry.json> <pinned-registry-sha256> <report.json> OR --self-test OR --self-test-report <report.json> <evidence-bundle-dir>",
            args.first().map(String::as_str).unwrap_or("pitb-sorrydb-evidence-closure")
        ));
    }
    let evaluation = fs::read(&args[2]).map_err(|e| format!("read evaluation split: {e}"))?;
    let runs = fs::read(&args[3]).map_err(|e| format!("read proof runs JSONL: {e}"))?;
    let reviewer_registry = fs::read(&args[5]).map_err(|e| format!("read reviewer key registry: {e}"))?;
    let report = verify_all(
        &evaluation, &runs, Path::new(&args[4]), &reviewer_registry, &args[6], true,
    )?;
    fs::write(&args[7], serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)
        .map_err(|e| format!("write closure report: {e}"))?;
    println!("PITB_SORRYDB_EVIDENCE_CLOSURE_OK");
    println!(
        "attempts={} verified_successes={} closed={} unique_closed_tasks={}",
        report.attempt_count,
        report.verified_success_run_count,
        report.closed_verified_success_run_count,
        report.closed_unique_task_count
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
    fn registered_evidence_closure_controls_pass() {
        run_self_tests().expect("all SorryDB evidence-closure controls must pass");
    }

    #[test]
    fn zero_based_unicode_column_maps_to_source_bytes() {
        let source = "α := by sorry\n";
        let char_column = source.find("sorry").unwrap();
        let column = source[..char_column].chars().count() as u64;
        let start = position_to_byte_index(source, 1, column).unwrap();
        assert_eq!(&source[start..start + 5], "sorry");
    }
}
