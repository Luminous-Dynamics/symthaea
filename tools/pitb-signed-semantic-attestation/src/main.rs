use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env, fs,
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

const RECEIPT_SCHEMA: &str = "pitb.sorrydb-semantic-attestation.v2";
const REGISTRY_SCHEMA: &str = "pitb.reviewer-key-registry.v0";
const ROOT_SCHEMA: &str = "pitb.reviewer-trust-root.v0";
const REPORT_SCHEMA: &str = "pitb.signed-semantic-attestation-verification.v0";
const RECEIPT_DOMAIN: &[u8] = b"PITB-SORRYDB-SEMANTIC-GATE-V2\0";
const REGISTRY_DOMAIN: &[u8] = b"PITB-REVIEWER-KEY-REGISTRY-V0\0";
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustRoot {
    schema_version: String,
    root_key_id: String,
    registry_id: String,
    minimum_registry_version: u64,
    public_key_der_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReviewerKey {
    key_id: String,
    reviewer_id: String,
    public_key_der_hex: String,
    status: String,
    valid_from_unix: u64,
    valid_until_unix: u64,
    scopes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReviewerKeyRegistry {
    schema_version: String,
    registry_id: String,
    registry_version: u64,
    generated_at_unix: u64,
    authority_key_id: String,
    keys: Vec<ReviewerKey>,
    signature_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SemanticAttestation {
    schema_version: String,
    attestation_id: String,
    run_id: String,
    task_id: String,
    repository_remote: String,
    repository_commit: String,
    lean_version: String,
    source_path: String,
    start_line: u64,
    start_column: u64,
    end_line: u64,
    end_column: u64,
    source_file_sha256: String,
    goal_sha256: String,
    declaration_signature_sha256: String,
    source_binding_receipt_sha256: String,
    source_correspondence_status: String,
    hypothesis_audit_status: String,
    vacuity_audit_status: String,
    reviewer_decision: String,
    reviewer_id: String,
    reviewer_key_id: String,
    issued_at_unix: u64,
    semantic_log_sha256: String,
    semantic_gate_passed: bool,
    vacuity_gate_passed: bool,
    signature_algorithm: String,
    signature_hex: String,
}

#[derive(Debug, Clone, Serialize)]
struct VerificationReport {
    schema_version: String,
    receipt_sha256: String,
    registry_sha256: String,
    trust_root_sha256: String,
    registry_id: String,
    registry_version: u64,
    root_key_id: String,
    authority_key_id: String,
    reviewer_id: String,
    reviewer_key_id: String,
    signer_status: String,
    signer_scopes: Vec<String>,
    signed_at_unix: u64,
    registry_payload_sha256: String,
    attestation_payload_sha256: String,
    registry_signature_sha256: String,
    attestation_signature_sha256: String,
    registry_signature_verified: bool,
    reviewer_signature_verified: bool,
    semantic_review_approved: bool,
    openssl_version: String,
    openssl_binary_sha256: String,
    minimum_registry_version: u64,
    authenticated: bool,
    scoring_admission: bool,
    trust_boundary: String,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn non_empty(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{label} must be non-empty"))
    } else {
        Ok(())
    }
}

fn decode_hex(value: &str, label: &str) -> Result<Vec<u8>, String> {
    if value.len() % 2 != 0 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("{label} must be even-length hexadecimal"));
    }
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let high = (pair[0] as char).to_digit(16).ok_or_else(|| format!("{label} has invalid hex"))?;
        let low = (pair[1] as char).to_digit(16).ok_or_else(|| format!("{label} has invalid hex"))?;
        output.push(((high << 4) | low) as u8);
    }
    Ok(output)
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn validate_ed25519_public_der(value: &str, label: &str) -> Result<Vec<u8>, String> {
    if !valid_hex(value, 88) {
        return Err(format!("{label} must be 88 lowercase hexadecimal characters"));
    }
    let bytes = decode_hex(value, label)?;
    // RFC 8410 SubjectPublicKeyInfo prefix for an Ed25519 public key.
    const PREFIX: &[u8] = &[
        0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00,
    ];
    if bytes.len() != 44 || &bytes[..12] != PREFIX {
        return Err(format!("{label} must be 44-byte Ed25519 SubjectPublicKeyInfo DER"));
    }
    Ok(bytes)
}

fn append_field(out: &mut Vec<u8>, tag: u16, value: &[u8]) -> Result<(), String> {
    let length = u32::try_from(value.len()).map_err(|_| "canonical field exceeds u32 length")?;
    out.extend_from_slice(&tag.to_be_bytes());
    out.extend_from_slice(&length.to_be_bytes());
    out.extend_from_slice(value);
    Ok(())
}

fn append_text(out: &mut Vec<u8>, tag: u16, value: &str) -> Result<(), String> {
    append_field(out, tag, value.as_bytes())
}

fn append_u64(out: &mut Vec<u8>, tag: u16, value: u64) -> Result<(), String> {
    append_field(out, tag, &value.to_be_bytes())
}

fn append_bool(out: &mut Vec<u8>, tag: u16, value: bool) -> Result<(), String> {
    append_field(out, tag, &[u8::from(value)])
}

// This versioned, tag-ordered, length-prefixed encoding is the signing wire format.
// It is independent of JSON whitespace, object member order and serializer defaults.
fn registry_signing_payload(registry: &ReviewerKeyRegistry) -> Result<Vec<u8>, String> {
    let mut out = REGISTRY_DOMAIN.to_vec();
    append_text(&mut out, 1, &registry.schema_version)?;
    append_text(&mut out, 2, &registry.registry_id)?;
    append_u64(&mut out, 3, registry.registry_version)?;
    append_u64(&mut out, 4, registry.generated_at_unix)?;
    append_text(&mut out, 5, &registry.authority_key_id)?;
    append_u64(&mut out, 6, registry.keys.len() as u64)?;
    for key in &registry.keys {
        let mut encoded_key = Vec::new();
        append_text(&mut encoded_key, 1, &key.key_id)?;
        append_text(&mut encoded_key, 2, &key.reviewer_id)?;
        append_text(&mut encoded_key, 3, &key.public_key_der_hex)?;
        append_text(&mut encoded_key, 4, &key.status)?;
        append_u64(&mut encoded_key, 5, key.valid_from_unix)?;
        append_u64(&mut encoded_key, 6, key.valid_until_unix)?;
        append_u64(&mut encoded_key, 7, key.scopes.len() as u64)?;
        for scope in &key.scopes {
            append_text(&mut encoded_key, 8, scope)?;
        }
        append_field(&mut out, 7, &encoded_key)?;
    }
    Ok(out)
}

fn attestation_payload(receipt: &SemanticAttestation) -> Result<Vec<u8>, String> {
    let mut out = RECEIPT_DOMAIN.to_vec();
    append_text(&mut out, 1, &receipt.schema_version)?;
    append_text(&mut out, 2, &receipt.attestation_id)?;
    append_text(&mut out, 3, &receipt.run_id)?;
    append_text(&mut out, 4, &receipt.task_id)?;
    append_text(&mut out, 5, &receipt.repository_remote)?;
    append_text(&mut out, 6, &receipt.repository_commit)?;
    append_text(&mut out, 7, &receipt.lean_version)?;
    append_text(&mut out, 8, &receipt.source_path)?;
    append_u64(&mut out, 9, receipt.start_line)?;
    append_u64(&mut out, 10, receipt.start_column)?;
    append_u64(&mut out, 11, receipt.end_line)?;
    append_u64(&mut out, 12, receipt.end_column)?;
    append_text(&mut out, 13, &receipt.source_file_sha256)?;
    append_text(&mut out, 14, &receipt.goal_sha256)?;
    append_text(&mut out, 15, &receipt.declaration_signature_sha256)?;
    append_text(&mut out, 16, &receipt.source_binding_receipt_sha256)?;
    append_text(&mut out, 17, &receipt.source_correspondence_status)?;
    append_text(&mut out, 18, &receipt.hypothesis_audit_status)?;
    append_text(&mut out, 19, &receipt.vacuity_audit_status)?;
    append_text(&mut out, 20, &receipt.reviewer_decision)?;
    append_text(&mut out, 21, &receipt.reviewer_id)?;
    append_text(&mut out, 22, &receipt.reviewer_key_id)?;
    append_u64(&mut out, 23, receipt.issued_at_unix)?;
    append_text(&mut out, 24, &receipt.semantic_log_sha256)?;
    append_bool(&mut out, 25, receipt.semantic_gate_passed)?;
    append_bool(&mut out, 26, receipt.vacuity_gate_passed)?;
    append_text(&mut out, 27, &receipt.signature_algorithm)?;
    Ok(out)
}

fn find_executable(name: &str) -> Result<PathBuf, String> {
    let path = env::var_os("PATH").ok_or("PATH is not set")?;
    for dir in env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return candidate.canonicalize().map_err(|e| format!("canonicalize {name}: {e}"));
        }
    }
    Err(format!("required executable {name} was not found on PATH"))
}

fn openssl_engine() -> Result<OpenSslEngine, String> {
    let binary = find_executable("openssl")?;
    let output = Command::new(&binary).arg("version").output()
        .map_err(|e| format!("run openssl version: {e}"))?;
    if !output.status.success() {
        return Err(format!("openssl version failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    let version = String::from_utf8(output.stdout)
        .map_err(|e| format!("openssl version output is not UTF-8: {e}"))?
        .trim().to_owned();
    let version_number = version.split_whitespace().nth(1).ok_or("cannot parse OpenSSL version")?;
    let major = version_number.split('.').next().and_then(|x| x.parse::<u32>().ok())
        .ok_or("cannot parse OpenSSL major version")?;
    if major < 3 {
        return Err(format!("OpenSSL 3 or newer is required; found {version}"));
    }
    let bytes = fs::read(&binary).map_err(|e| format!("read OpenSSL binary: {e}"))?;
    Ok(OpenSslEngine { binary, version, binary_sha256: sha256(&bytes) })
}

fn create_temp_dir(prefix: &str) -> Result<PathBuf, String> {
    let now = SystemTime::now().duration_since(UNIX_EPOCH)
        .map_err(|e| format!("system clock before UNIX epoch: {e}"))?
        .as_nanos();
    for _ in 0..32 {
        let serial = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!("{prefix}-{}-{now}-{serial}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    fs::set_permissions(&path, fs::Permissions::from_mode(0o700))
                        .map_err(|e| format!("set temporary directory permissions: {e}"))?;
                }
                return Ok(path);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("create temporary directory: {error}")),
        }
    }
    Err("could not allocate a unique verifier temporary directory".into())
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path).map_err(|e| format!("create verifier input {}: {e}", path.display()))?;
    file.write_all(bytes).map_err(|e| format!("write verifier input {}: {e}", path.display()))?;
    file.sync_all().map_err(|e| format!("sync verifier input {}: {e}", path.display()))?;
    Ok(())
}

fn verify_ed25519(
    engine: &OpenSslEngine,
    public_key_der: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<(), String> {
    if signature.len() != 64 {
        return Err("Ed25519 signature must be exactly 64 bytes".into());
    }
    let temp = create_temp_dir("pitb-ed25519-verify")?;
    let result = (|| {
        let key_path = temp.join("public-key.der");
        let message_path = temp.join("signed-payload.bin");
        let signature_path = temp.join("signature.bin");
        write_private(&key_path, public_key_der)?;
        write_private(&message_path, message)?;
        write_private(&signature_path, signature)?;
        let output = Command::new(&engine.binary)
            .args([
                "pkeyutl", "-verify", "-pubin", "-keyform", "DER",
                "-inkey", key_path.to_str().ok_or("temporary key path is not UTF-8")?,
                "-sigfile", signature_path.to_str().ok_or("temporary signature path is not UTF-8")?,
                "-rawin", "-in", message_path.to_str().ok_or("temporary payload path is not UTF-8")?,
            ])
            .output()
            .map_err(|e| format!("run OpenSSL Ed25519 verification: {e}"))?;
        if output.status.success() {
            Ok(())
        } else {
            Err(format!(
                "Ed25519 signature verification failed: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    })();
    let _ = fs::remove_dir_all(&temp);
    result
}

fn validate_root(root: &TrustRoot) -> Result<Vec<u8>, String> {
    if root.schema_version != ROOT_SCHEMA {
        return Err("unsupported trust-root schema".into());
    }
    if root.minimum_registry_version == 0 {
        return Err("trust-root minimum_registry_version must be positive".into());
    }
    non_empty(&root.root_key_id, "trust root key_id")?;
    non_empty(&root.registry_id, "trust root registry_id")?;
    validate_ed25519_public_der(&root.public_key_der_hex, "trust root public key")
}

fn validate_registry(
    registry: &ReviewerKeyRegistry,
    root: &TrustRoot,
    root_public_der: &[u8],
    engine: &OpenSslEngine,
) -> Result<(String, Vec<u8>), String> {
    if registry.schema_version != REGISTRY_SCHEMA {
        return Err("unsupported reviewer registry schema".into());
    }
    if registry.registry_id != root.registry_id {
        return Err("reviewer registry ID is not authorized by the pinned trust root".into());
    }
    if registry.authority_key_id != root.root_key_id {
        return Err("reviewer registry authority key ID differs from pinned trust root".into());
    }
    non_empty(&registry.registry_id, "registry_id")?;
    if registry.registry_version == 0 || registry.generated_at_unix == 0 {
        return Err("registry version and generated_at_unix must be positive".into());
    }
    if registry.registry_version < root.minimum_registry_version {
        return Err(format!(
            "reviewer registry version {} is below trust-root minimum {}",
            registry.registry_version, root.minimum_registry_version
        ));
    }
    if registry.keys.is_empty() {
        return Err("reviewer registry cannot be empty".into());
    }
    let mut previous = None::<&str>;
    let mut seen_reviewers = BTreeSet::new();
    for key in &registry.keys {
        non_empty(&key.key_id, "reviewer key_id")?;
        non_empty(&key.reviewer_id, "reviewer_id")?;
        if previous.is_some_and(|p| p >= key.key_id.as_str()) {
            return Err("reviewer registry keys must be sorted by unique key_id".into());
        }
        previous = Some(&key.key_id);
        if !seen_reviewers.insert((key.reviewer_id.clone(), key.key_id.clone())) {
            return Err("duplicate reviewer/key identity".into());
        }
        if !matches!(key.status.as_str(), "active" | "revoked") {
            return Err(format!("unsupported key status {}", key.status));
        }
        if key.valid_from_unix == 0 || key.valid_until_unix < key.valid_from_unix {
            return Err(format!("reviewer key {} has invalid validity interval", key.key_id));
        }
        if key.scopes.is_empty() || !key.scopes.windows(2).all(|w| w[0] < w[1]) {
            return Err(format!("reviewer key {} scopes must be sorted and unique", key.key_id));
        }
        validate_ed25519_public_der(&key.public_key_der_hex, &format!("reviewer key {}", key.key_id))?;
    }
    let payload = registry_signing_payload(registry)?;
    let payload_sha256 = sha256(&payload);
    if !valid_hex(&registry.signature_hex, 128) {
        return Err("registry signature must be 128 lowercase hexadecimal characters".into());
    }
    let signature = decode_hex(&registry.signature_hex, "registry signature")?;
    verify_ed25519(engine, root_public_der, &payload, &signature)
        .map_err(|e| format!("reviewer registry signature invalid: {e}"))?;
    Ok((payload_sha256, signature))
}

fn valid_github_remote(value: &str) -> bool {
    let Some(rest) = value.strip_prefix("https://github.com/") else {
        return false;
    };
    if rest.contains('?') || rest.contains('#') || rest.ends_with('/') {
        return false;
    }
    let clean = rest.strip_suffix(".git").unwrap_or(rest);
    let parts: Vec<&str> = clean.split('/').collect();
    parts.len() == 2
        && parts.iter().all(|part| {
            !part.is_empty()
                && *part != "."
                && *part != ".."
                && part.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        })
}

fn validate_attestation(receipt: &SemanticAttestation) -> Result<(), String> {
    if receipt.schema_version != RECEIPT_SCHEMA {
        return Err("unsupported semantic attestation schema".into());
    }
    for (label, value) in [
        ("attestation_id", receipt.attestation_id.as_str()),
        ("run_id", receipt.run_id.as_str()),
        ("task_id", receipt.task_id.as_str()),
        ("lean_version", receipt.lean_version.as_str()),
        ("source_path", receipt.source_path.as_str()),
        ("reviewer_id", receipt.reviewer_id.as_str()),
        ("reviewer_key_id", receipt.reviewer_key_id.as_str()),
    ] {
        non_empty(value, label)?;
    }
    if !valid_github_remote(&receipt.repository_remote) {
        return Err("repository_remote must be a canonical GitHub HTTPS owner/repository URL".into());
    }
    let source = Path::new(&receipt.source_path);
    if source.is_absolute()
        || receipt.source_path.contains('\\')
        || source.components().any(|part| matches!(part, std::path::Component::ParentDir | std::path::Component::CurDir))
        || !receipt.source_path.ends_with(".lean")
    {
        return Err("source_path must be a normalized repository-relative Lean path".into());
    }
    if receipt.start_line == 0
        || receipt.end_line < receipt.start_line
        || (receipt.end_line == receipt.start_line && receipt.end_column <= receipt.start_column)
    {
        return Err("source coordinates are invalid".into());
    }
    if !valid_hex(&receipt.task_id, 64) || !valid_hex(&receipt.repository_commit, 40) {
        return Err("task_id or repository commit has invalid digest syntax".into());
    }
    for (label, value) in [
        ("source_file_sha256", &receipt.source_file_sha256),
        ("goal_sha256", &receipt.goal_sha256),
        ("declaration_signature_sha256", &receipt.declaration_signature_sha256),
        ("source_binding_receipt_sha256", &receipt.source_binding_receipt_sha256),
        ("semantic_log_sha256", &receipt.semantic_log_sha256),
    ] {
        if !valid_hex(value, 64) {
            return Err(format!("{label} must be lowercase SHA-256 hex"));
        }
    }
    if receipt.issued_at_unix == 0 {
        return Err("issued_at_unix must be positive".into());
    }
    if !matches!(
        receipt.source_correspondence_status.as_str(),
        "pass" | "fail" | "unresolved"
    ) || !matches!(
        receipt.hypothesis_audit_status.as_str(),
        "pass" | "fail" | "unresolved"
    ) || !matches!(
        receipt.vacuity_audit_status.as_str(),
        "pass" | "fail" | "unresolved"
    ) || !matches!(receipt.reviewer_decision.as_str(), "pass" | "fail" | "abstain")
    {
        return Err("semantic statuses or reviewer decision are unsupported".into());
    }
    if receipt.signature_algorithm != "Ed25519" {
        return Err("semantic attestation signature_algorithm must be Ed25519".into());
    }
    if !valid_hex(&receipt.signature_hex, 128) {
        return Err("reviewer signature must be 128 lowercase hexadecimal characters".into());
    }
    if receipt.reviewer_decision == "pass"
        && (!receipt.semantic_gate_passed
            || !receipt.vacuity_gate_passed
            || receipt.source_correspondence_status != "pass"
            || receipt.hypothesis_audit_status != "pass"
            || receipt.vacuity_audit_status != "pass")
    {
        return Err("reviewer pass contradicts semantic/hypothesis/vacuity gate results".into());
    }
    Ok(())
}

fn verify_all(
    receipt: &SemanticAttestation,
    registry: &ReviewerKeyRegistry,
    root: &TrustRoot,
    engine: &OpenSslEngine,
    receipt_bytes: &[u8],
    registry_bytes: &[u8],
    root_bytes: &[u8],
    expected_root_sha256: &str,
    expected_registry_sha256: &str,
    expected_openssl_sha256: &str,
) -> Result<VerificationReport, String> {
    validate_attestation(receipt)?;
    for (label, value) in [
        ("expected trust-root SHA-256", expected_root_sha256),
        ("expected reviewer-registry SHA-256", expected_registry_sha256),
        ("expected OpenSSL-binary SHA-256", expected_openssl_sha256),
    ] {
        if !valid_hex(value, 64) {
            return Err(format!("{label} must be 64 lowercase hexadecimal characters"));
        }
    }
    if sha256(root_bytes) != expected_root_sha256 {
        return Err("trusted reviewer root SHA-256 differs from the out-of-band pin".into());
    }
    if sha256(registry_bytes) != expected_registry_sha256 {
        return Err("reviewer registry SHA-256 differs from the out-of-band pin".into());
    }
    if engine.binary_sha256 != expected_openssl_sha256 {
        return Err("OpenSSL binary SHA-256 differs from the out-of-band pin".into());
    }
    let root_public_der = validate_root(root)?;
    let (registry_payload_sha256, _) = validate_registry(registry, root, &root_public_der, engine)?;
    if registry.generated_at_unix > receipt.issued_at_unix {
        return Err("attestation predates the reviewer registry that claims to authorize it".into());
    }
    let key = registry.keys.iter().find(|key| key.key_id == receipt.reviewer_key_id)
        .ok_or_else(|| format!("reviewer key {} is not present in the signed registry", receipt.reviewer_key_id))?;
    if key.reviewer_id != receipt.reviewer_id {
        return Err("attestation reviewer_id differs from registry key owner".into());
    }
    if key.status != "active" {
        return Err(format!("reviewer key {} is not active", key.key_id));
    }
    if receipt.issued_at_unix < key.valid_from_unix || receipt.issued_at_unix > key.valid_until_unix {
        return Err(format!("attestation time falls outside reviewer key {} validity interval", key.key_id));
    }
    for required_scope in ["semantic_gate", "hypothesis_audit", "vacuity_audit"] {
        if key.scopes.binary_search_by(|scope| scope.as_str().cmp(required_scope)).is_err() {
            return Err(format!("reviewer key {} lacks required scope {required_scope}", key.key_id));
        }
    }
    let payload = attestation_payload(receipt)?;
    let attestation_payload_sha256 = sha256(&payload);
    let signature = decode_hex(&receipt.signature_hex, "reviewer signature")?;
    verify_ed25519(engine, &validate_ed25519_public_der(&key.public_key_der_hex, "reviewer key")?, &payload, &signature)
        .map_err(|e| format!("reviewer semantic attestation signature invalid: {e}"))?;
    let reviewer_registry_sha256 = sha256(registry_bytes);
    let trust_root_sha256 = sha256(root_bytes);
    Ok(VerificationReport {
        schema_version: REPORT_SCHEMA.into(),
        receipt_sha256: sha256(receipt_bytes),
        registry_sha256: reviewer_registry_sha256,
        trust_root_sha256,
        registry_id: registry.registry_id.clone(),
        registry_version: registry.registry_version,
        root_key_id: root.root_key_id.clone(),
        authority_key_id: registry.authority_key_id.clone(),
        reviewer_id: receipt.reviewer_id.clone(),
        reviewer_key_id: receipt.reviewer_key_id.clone(),
        signer_status: key.status.clone(),
        signer_scopes: key.scopes.clone(),
        signed_at_unix: receipt.issued_at_unix,
        registry_payload_sha256,
        attestation_payload_sha256,
        registry_signature_sha256: sha256(&decode_hex(&registry.signature_hex, "registry signature")?),
        attestation_signature_sha256: sha256(&signature),
        registry_signature_verified: true,
        reviewer_signature_verified: true,
        semantic_review_approved: receipt.reviewer_decision == "pass"
            && receipt.semantic_gate_passed
            && receipt.vacuity_gate_passed,
        openssl_version: engine.version.clone(),
        openssl_binary_sha256: engine.binary_sha256.clone(),
        minimum_registry_version: root.minimum_registry_version,
        authenticated: true,
        scoring_admission: false,
        trust_boundary: "Both the reviewer registry and semantic-gate attestation are Ed25519-verified. The trust-root JSON is an out-of-band public-key pin and its SHA-256 must be independently pinned by the benchmark or deployment policy. Authentication does not establish the truth of the review, kernel correctness, source fidelity, or proof admissibility.".into(),
    })
}

fn unique_temp(prefix: &str) -> Result<PathBuf, String> {
    create_temp_dir(prefix)
}

fn run_openssl(engine: &OpenSslEngine, args: &[&str], context: &str) -> Result<(), String> {
    let output = Command::new(&engine.binary)
        .args(args)
        .output()
        .map_err(|e| format!("{context}: {e}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "{context}: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

fn generate_test_key(engine: &OpenSslEngine, directory: &Path, name: &str) -> Result<(PathBuf, String), String> {
    let private_path = directory.join(format!("{name}.private.pem"));
    let public_path = directory.join(format!("{name}.public.der"));
    run_openssl(
        engine,
        &["genpkey", "-algorithm", "ED25519", "-out", private_path.to_str().ok_or("private path UTF-8")?],
        "generate synthetic Ed25519 key",
    )?;
    run_openssl(
        engine,
        &[
            "pkey", "-in", private_path.to_str().ok_or("private path UTF-8")?,
            "-pubout", "-outform", "DER", "-out", public_path.to_str().ok_or("public path UTF-8")?,
        ],
        "extract synthetic Ed25519 public key",
    )?;
    let public_der = fs::read(&public_path).map_err(|e| format!("read synthetic public key: {e}"))?;
    if public_der.len() != 44 || public_der[..12] != [0x30,0x2a,0x30,0x05,0x06,0x03,0x2b,0x65,0x70,0x03,0x21,0x00] {
        return Err("OpenSSL returned unexpected Ed25519 public-key DER".into());
    }
    Ok((private_path, encode_hex(&public_der)))
}

fn sign_test_payload(
    engine: &OpenSslEngine,
    private_path: &Path,
    payload: &[u8],
    output_dir: &Path,
    name: &str,
) -> Result<String, String> {
    let input = output_dir.join(format!("{name}.payload.bin"));
    let signature_path = output_dir.join(format!("{name}.signature.bin"));
    fs::write(&input, payload).map_err(|e| format!("write synthetic signed payload: {e}"))?;
    run_openssl(
        engine,
        &[
            "pkeyutl", "-sign",
            "-inkey", private_path.to_str().ok_or("private key path UTF-8")?,
            "-rawin", "-in", input.to_str().ok_or("payload path UTF-8")?,
            "-out", signature_path.to_str().ok_or("signature path UTF-8")?,
        ],
        "sign synthetic Ed25519 payload",
    )?;
    let signature = fs::read(signature_path).map_err(|e| format!("read synthetic signature: {e}"))?;
    if signature.len() != 64 {
        return Err("OpenSSL generated non-64-byte Ed25519 signature".into());
    }
    Ok(encode_hex(&signature))
}

fn sign_registry(
    registry: &mut ReviewerKeyRegistry,
    engine: &OpenSslEngine,
    private_path: &Path,
    output_dir: &Path,
) -> Result<(), String> {
    registry.signature_hex.clear();
    let payload = registry_signing_payload(registry)?;
    registry.signature_hex = sign_test_payload(engine, private_path, &payload, output_dir, "registry")?;
    Ok(())
}

fn sign_attestation(
    receipt: &mut SemanticAttestation,
    engine: &OpenSslEngine,
    private_path: &Path,
    output_dir: &Path,
) -> Result<(), String> {
    receipt.signature_hex.clear();
    let payload = attestation_payload(receipt)?;
    receipt.signature_hex = sign_test_payload(engine, private_path, &payload, output_dir, "attestation")?;
    Ok(())
}

fn synthetic_receipt() -> SemanticAttestation {
    SemanticAttestation {
        schema_version: RECEIPT_SCHEMA.into(),
        attestation_id: "synthetic-attestation-001".into(),
        run_id: "sorrydb-run-001".into(),
        task_id: "a".repeat(64),
        repository_remote: "https://github.com/AlexKontorovich/PrimeNumberTheoremAnd".into(),
        repository_commit: "b".repeat(40),
        lean_version: "v4.26.0".into(),
        source_path: "PrimeNumberTheoremAnd/FioriKadiriSwidinsky.lean".into(),
        start_line: 44,
        start_column: 137,
        end_line: 44,
        end_column: 142,
        source_file_sha256: sha256(b"source file"),
        goal_sha256: sha256(b"exact target goal"),
        declaration_signature_sha256: sha256(b"target signature"),
        source_binding_receipt_sha256: sha256(b"source binding receipt"),
        source_correspondence_status: "pass".into(),
        hypothesis_audit_status: "pass".into(),
        vacuity_audit_status: "pass".into(),
        reviewer_decision: "pass".into(),
        reviewer_id: "reviewer-synthetic".into(),
        reviewer_key_id: "reviewer-synthetic-key-01".into(),
        issued_at_unix: 1_800_000_000,
        semantic_log_sha256: sha256(b"review log bytes"),
        semantic_gate_passed: true,
        vacuity_gate_passed: true,
        signature_algorithm: "Ed25519".into(),
        signature_hex: String::new(),
    }
}

fn create_synthetic_attestation_bundle(root: &Path) -> Result<VerificationReport, String> {
    let engine = openssl_engine()?;
    let signer_dir = unique_temp("pitb-synthetic-ed25519-keys")?;
    let result = (|| {
        let (root_private, root_public_hex) = generate_test_key(&engine, &signer_dir, "root")?;
        let (reviewer_private, reviewer_public_hex) = generate_test_key(&engine, &signer_dir, "reviewer")?;
        let trust_root = TrustRoot {
            schema_version: ROOT_SCHEMA.into(),
            root_key_id: "pitb-test-root-001".into(),
            registry_id: "pitb.synthetic-reviewer-registry".into(),
            minimum_registry_version: 1,
            public_key_der_hex: root_public_hex,
        };
        let reviewer_key = ReviewerKey {
            key_id: "reviewer-synthetic-key-01".into(),
            reviewer_id: "reviewer-synthetic".into(),
            public_key_der_hex: reviewer_public_hex,
            status: "active".into(),
            valid_from_unix: 1_700_000_000,
            valid_until_unix: 1_900_000_000,
            scopes: vec![
                "hypothesis_audit".into(),
                "semantic_gate".into(),
                "vacuity_audit".into(),
            ],
        };
        let mut registry = ReviewerKeyRegistry {
            schema_version: REGISTRY_SCHEMA.into(),
            registry_id: trust_root.registry_id.clone(),
            registry_version: 1,
            generated_at_unix: 1_799_999_900,
            authority_key_id: trust_root.root_key_id.clone(),
            keys: vec![reviewer_key],
            signature_hex: String::new(),
        };
        sign_registry(&mut registry, &engine, &root_private, &signer_dir)?;

        let mut receipt = synthetic_receipt();
        sign_attestation(&mut receipt, &engine, &reviewer_private, &signer_dir)?;
        let receipt_bytes = serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?;
        let registry_bytes = serde_json::to_vec_pretty(&registry).map_err(|e| e.to_string())?;
        let root_bytes = serde_json::to_vec_pretty(&trust_root).map_err(|e| e.to_string())?;
        let report = verify_all(
            &receipt, &registry, &trust_root, &engine,
            &receipt_bytes, &registry_bytes, &root_bytes,
            &sha256(&root_bytes), &sha256(&registry_bytes), &engine.binary_sha256,
        )?;

        fs::create_dir_all(root).map_err(|e| format!("create synthetic signed-attestation bundle: {e}"))?;
        fs::write(root.join("semantic-gate-attestation.json"), &receipt_bytes)
            .map_err(|e| format!("write synthetic semantic attestation: {e}"))?;
        fs::write(root.join("reviewer-key-registry.json"), &registry_bytes)
            .map_err(|e| format!("write synthetic reviewer registry: {e}"))?;
        fs::write(root.join("reviewer-trust-root.synthetic.json"), &root_bytes)
            .map_err(|e| format!("write synthetic trust root: {e}"))?;
        fs::write(root.join("verification-report.json"), serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)
            .map_err(|e| format!("write synthetic verification report: {e}"))?;

        // Negative controls: tampered payload, revoked key, and wrong root must fail.
        let mut tampered = receipt.clone();
        tampered.semantic_log_sha256 = sha256(b"changed review log");
        if verify_all(&tampered, &registry, &trust_root, &engine, &receipt_bytes, &registry_bytes, &root_bytes,
            &sha256(&root_bytes), &sha256(&registry_bytes), &engine.binary_sha256).is_ok() {
            return Err("tampered reviewer payload was accepted".into());
        }

        let mut revoked_registry = registry.clone();
        revoked_registry.keys[0].status = "revoked".into();
        sign_registry(&mut revoked_registry, &engine, &root_private, &signer_dir)?;
        let revoked_registry_bytes = serde_json::to_vec(&revoked_registry).unwrap();
        if verify_all(&receipt, &revoked_registry, &trust_root, &engine, &receipt_bytes, &revoked_registry_bytes, &root_bytes,
            &sha256(&root_bytes), &sha256(&revoked_registry_bytes), &engine.binary_sha256).is_ok() {
            return Err("revoked reviewer key was accepted".into());
        }

        let (_, wrong_root_public_hex) = generate_test_key(&engine, &signer_dir, "wrong-root")?;
        let wrong_root = TrustRoot { public_key_der_hex: wrong_root_public_hex, ..trust_root.clone() };
        let wrong_root_bytes = serde_json::to_vec(&wrong_root).unwrap();
        if verify_all(&receipt, &registry, &wrong_root, &engine, &receipt_bytes, &registry_bytes, &wrong_root_bytes,
            &sha256(&wrong_root_bytes), &sha256(&registry_bytes), &engine.binary_sha256).is_ok() {
            return Err("registry signed by another root was accepted".into());
        }
        Ok(report)
    })();
    let _ = fs::remove_dir_all(&signer_dir);
    result
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        let root = unique_temp("pitb-signed-attestation-selftest")?;
        let report = create_synthetic_attestation_bundle(&root)?;
        println!("SELF_TEST_PASS Ed25519 reviewer signature and root-signed registry");
        println!("SELF_TEST_PASS modified semantic payload rejected");
        println!("SELF_TEST_PASS revoked reviewer key rejected");
        println!("SELF_TEST_PASS registry signed by untrusted root rejected");
        println!("OPENSSL_VERSION {}", report.openssl_version);
        println!("PITB_SIGNED_SEMANTIC_ATTESTATION_SELF_TEST_PASS");
        let _ = fs::remove_dir_all(root);
        return Ok(());
    }
    if args.len() == 4 && args[1] == "--self-test-report" {
        let root = PathBuf::from(&args[3]);
        let report = create_synthetic_attestation_bundle(&root)?;
        fs::write(&args[2], serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)
            .map_err(|e| format!("write verification report: {e}"))?;
        println!("PITB_SIGNED_SEMANTIC_ATTESTATION_SYNTHETIC_REPORT_OK");
        return Ok(());
    }
    if args.len() != 12 || args[1] != "--verify"
        || args[6] != "--expected-trust-root-sha256"
        || args[8] != "--expected-registry-sha256"
        || args[10] != "--expected-openssl-sha256" {
        return Err(format!(
            "usage: {} --verify <attestation.json> <reviewer-registry.json> <trusted-root.json> <report.json> --expected-trust-root-sha256 <sha256> --expected-registry-sha256 <sha256> --expected-openssl-sha256 <sha256> OR --self-test OR --self-test-report <report.json> <bundle-dir>",
            args.first().map(String::as_str).unwrap_or("pitb-signed-semantic-attestation")
        ));
    }
    let receipt_bytes = fs::read(&args[2]).map_err(|e| format!("read semantic attestation: {e}"))?;
    let registry_bytes = fs::read(&args[3]).map_err(|e| format!("read reviewer registry: {e}"))?;
    let root_bytes = fs::read(&args[4]).map_err(|e| format!("read trusted reviewer root: {e}"))?;
    let receipt: SemanticAttestation = serde_json::from_slice(&receipt_bytes)
        .map_err(|e| format!("parse semantic attestation: {e}"))?;
    let registry: ReviewerKeyRegistry = serde_json::from_slice(&registry_bytes)
        .map_err(|e| format!("parse reviewer registry: {e}"))?;
    let root: TrustRoot = serde_json::from_slice(&root_bytes)
        .map_err(|e| format!("parse trusted reviewer root: {e}"))?;
    let engine = openssl_engine()?;
    let report = verify_all(
        &receipt, &registry, &root, &engine, &receipt_bytes, &registry_bytes, &root_bytes,
        &args[7], &args[9], &args[11],
    )?;
    fs::write(&args[5], serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?)
        .map_err(|e| format!("write signature verification report: {e}"))?;
    println!("PITB_SIGNED_SEMANTIC_ATTESTATION_OK");
    println!("reviewer={} key={} registry={}", report.reviewer_id, report.reviewer_key_id, report.registry_id);
    println!("authenticated={} semantic_review_approved={} scoring_admission={}", report.authenticated, report.semantic_review_approved, report.scoring_admission);
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
    fn registered_ed25519_signature_and_key_lifecycle_controls_pass() {
        let root = unique_temp("pitb-signed-attestation-tests").unwrap();
        let report = create_synthetic_attestation_bundle(&root).unwrap();
        assert!(report.registry_signature_verified);
        assert!(report.reviewer_signature_verified);
        assert!(report.authenticated);
        assert!(!report.scoring_admission);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn repository_identity_requires_canonical_github_owner_and_repo() {
        assert!(valid_github_remote("https://github.com/owner/repository"));
        assert!(valid_github_remote("https://github.com/owner/repo.git"));
        assert!(!valid_github_remote("https://github.com/owner/repo/extra"));
        assert!(!valid_github_remote("https://github.com/owner/../repo"));
        assert!(!valid_github_remote("https://github.com/owner/repo?ref=main"));
        assert!(!valid_github_remote("git@github.com:owner/repo.git"));
    }

    #[test]
    fn canonical_payload_is_stable_under_json_whitespace_and_member_order() {
        let payload = synthetic_receipt();
        let expected = attestation_payload(&payload).unwrap();
        let value = serde_json::to_value(&payload).unwrap();
        let object = value.as_object().unwrap();
        let mut fields: Vec<_> = object.iter().collect();
        fields.reverse();
        let reordered_fields = fields.iter().map(|(key, value)| {
            format!(
                "{}: {}",
                serde_json::to_string(key).unwrap(),
                serde_json::to_string(value).unwrap()
            )
        }).collect::<Vec<_>>().join(",\n  ");
        let reordered_json = format!("{{\n  {reordered_fields}\n}}");
        let decoded: SemanticAttestation = serde_json::from_str(&reordered_json).unwrap();
        assert_eq!(expected, attestation_payload(&decoded).unwrap());
    }
}
