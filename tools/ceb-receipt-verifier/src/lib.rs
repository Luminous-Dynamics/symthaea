//! Offline, fail-closed verifier for a CEB primary/pristine-regrade pair.
//!
//! The verifier never executes candidate code and has no network or shell path.
//! Trust roots are supplied out of band; JSON Schema files are embedded at build time.

use base64::{engine::general_purpose::STANDARD, Engine as _};
use chrono::{DateTime, FixedOffset};
use ed25519_dalek::{Signature, VerifyingKey};
use serde::de::{Deserializer, Error as DeError, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

const STATEMENT_TYPE: &str = "https://in-toto.io/Statement/v1";
const PREDICATE_TYPE: &str = "https://luminousdynamics.io/attestations/ceb-run/v0";
const PAYLOAD_TYPE: &str = "application/vnd.in-toto+json";
const MAX_JSON_BYTES: usize = 8 * 1024 * 1024;
const MAX_PATCH_BYTES: usize = 32 * 1024 * 1024;
const MAX_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TOTAL_ARTIFACT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_ARTIFACTS: usize = 256;
const MAX_SIGNATURES: usize = 32;

const TASK_SCHEMA: &str = include_str!("../../../benchmarks/ceb/task.schema.json");
const RECEIPT_SCHEMA: &str = include_str!("../../../benchmarks/ceb/run-receipt.schema.json");
const ENVELOPE_SCHEMA: &str = include_str!("../../../benchmarks/ceb/dsse-envelope.schema.json");

#[derive(Clone, Debug, Serialize)]
pub struct VerificationReport {
    pub report_version: u32,
    pub verifier_version: &'static str,
    pub status: String,
    pub accepted: bool,
    pub task_id: Option<String>,
    pub task_contract_sha256: Option<String>,
    pub source_tree_sha256: Option<String>,
    pub patch_sha256: Option<String>,
    pub primary_envelope_sha256: Option<String>,
    pub regrade_envelope_sha256: Option<String>,
    pub primary_run_id: Option<String>,
    pub regrade_run_id: Option<String>,
    pub primary_signer_ids: Vec<String>,
    pub regrade_signer_ids: Vec<String>,
    pub primary_signer_domains: Vec<String>,
    pub regrade_signer_domains: Vec<String>,
    pub primary_check_evidence_sha256: Vec<String>,
    pub regrade_check_evidence_sha256: Vec<String>,
    pub primary_artifact_evidence_sha256: Vec<String>,
    pub regrade_artifact_evidence_sha256: Vec<String>,
    pub trust_policy_sha256: Option<String>,
    pub schema_sha256: SchemaDigests,
    pub reasons: Vec<String>,
    pub claim_ceiling: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct SchemaDigests {
    pub task: String,
    pub receipt: String,
    pub envelope: String,
}

pub struct PairInputs {
    pub task_contract: Vec<u8>,
    pub task_id: String,
    pub source_tree_sha256: String,
    pub patch: Vec<u8>,
    pub primary_envelope: Vec<u8>,
    pub primary_artifacts_dir: PathBuf,
    pub regrade_envelope: Vec<u8>,
    pub regrade_artifacts_dir: PathBuf,
    pub trust_policy: Vec<u8>,
    /// Out-of-band pin; lowercase SHA-256 hex without a prefix.
    pub expected_trust_policy_sha256: String,
}

#[derive(Debug)]
struct Failure {
    status: &'static str,
    reason: String,
}

impl Failure {
    fn protocol(reason: impl Into<String>) -> Self {
        Self { status: "protocol_violation", reason: reason.into() }
    }
    fn rejected(reason: impl Into<String>) -> Self {
        Self { status: "rejected", reason: reason.into() }
    }
    fn unresolved(reason: impl Into<String>) -> Self {
        Self { status: "unresolved", reason: reason.into() }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustPolicy {
    schema_version: u32,
    policy_id: String,
    minimum_signers: usize,
    keys: Vec<TrustedKey>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustedKey {
    key_id: String,
    algorithm: String,
    public_key_base64: String,
    role: String,
    trust_domain: String,
    status: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictEnvelope {
    #[serde(rename = "payloadType")]
    payload_type: String,
    payload: String,
    signatures: Vec<StrictSignature>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictSignature {
    keyid: String,
    sig: String,
}

/// Serde's ordinary Value map accepts duplicate object members by retaining the last
/// value. Attestations must not have parser-dependent meaning, so reject duplicates at
/// every nesting level before schema or semantic validation.
struct UniqueValue(Value);

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueVisitor)
    }
}

struct UniqueVisitor;

impl<'de> Visitor<'de> for UniqueVisitor {
    type Value = UniqueValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("strict JSON without duplicate object keys")
    }

    fn visit_bool<E: DeError>(self, value: bool) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Bool(value)))
    }

    fn visit_i64<E: DeError>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E: DeError>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: DeError,
    {
        Number::from_f64(value)
            .map(|number| UniqueValue(Value::Number(number)))
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E: DeError>(self, value: &str) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value.to_owned())))
    }

    fn visit_string<E: DeError>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value)))
    }

    fn visit_none<E: DeError>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_unit<E: DeError>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        UniqueValue::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(UniqueValue(value)) = sequence.next_element::<UniqueValue>()? {
            values.push(value);
        }
        Ok(UniqueValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut object: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some(key) = object.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(A::Error::custom(format!("duplicate JSON object key: {key}")));
            }
            let UniqueValue(value) = object.next_value::<UniqueValue>()?;
            values.insert(key, value);
        }
        Ok(UniqueValue(Value::Object(values)))
    }
}

fn parse_strict_json(bytes: &[u8], label: &str) -> Result<Value, Failure> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let UniqueValue(value) = UniqueValue::deserialize(&mut deserializer)
        .map_err(|error| Failure::protocol(format!("{label}: invalid strict JSON: {error}")))?;
    deserializer.end()
        .map_err(|error| Failure::protocol(format!("{label}: trailing JSON data: {error}")))?;
    Ok(value)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn schema_digests() -> SchemaDigests {
    SchemaDigests {
        task: sha256_hex(TASK_SCHEMA.as_bytes()),
        receipt: sha256_hex(RECEIPT_SCHEMA.as_bytes()),
        envelope: sha256_hex(ENVELOPE_SCHEMA.as_bytes()),
    }
}

fn validate_against_schema(
    schema_source: &str,
    instance: &Value,
    label: &str,
) -> Result<(), Failure> {
    let schema: Value = serde_json::from_str(schema_source)
        .map_err(|error| Failure::protocol(format!("embedded {label} schema is invalid JSON: {error}")))?;
    let validator = jsonschema::draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|error| Failure::protocol(format!("cannot compile embedded {label} schema: {error}")))?;
    let errors: Vec<String> = validator
        .iter_errors(instance)
        .take(12)
        .map(|error| format!("{} at {}", error, error.instance_path))
        .collect();
    if !errors.is_empty() {
        return Err(Failure::protocol(format!(
            "{label} schema validation failed: {}",
            errors.join("; ")
        )));
    }
    Ok(())
}

fn field<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Value, Failure> {
    value.get(key).ok_or_else(|| Failure::protocol(format!("{context}.{key} is missing")))
}

fn string_field(value: &Value, key: &str, context: &str) -> Result<String, Failure> {
    field(value, key, context)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| Failure::protocol(format!("{context}.{key} must be a string")))
}

fn optional_string_field(value: &Value, key: &str, context: &str) -> Result<Option<String>, Failure> {
    match field(value, key, context)? {
        Value::Null => Ok(None),
        Value::String(text) => Ok(Some(text.clone())),
        _ => Err(Failure::protocol(format!("{context}.{key} must be a string or null"))),
    }
}

fn bool_field(value: &Value, key: &str, context: &str) -> Result<bool, Failure> {
    field(value, key, context)?
        .as_bool()
        .ok_or_else(|| Failure::protocol(format!("{context}.{key} must be a boolean")))
}

fn u64_field(value: &Value, key: &str, context: &str) -> Result<u64, Failure> {
    field(value, key, context)?
        .as_u64()
        .ok_or_else(|| Failure::protocol(format!("{context}.{key} must be a non-negative integer")))
}

fn canonical_base64(encoded: &str, label: &str) -> Result<Vec<u8>, Failure> {
    let decoded = STANDARD
        .decode(encoded)
        .map_err(|error| Failure::protocol(format!("{label} is not valid standard Base64: {error}")))?;
    if STANDARD.encode(&decoded) != encoded {
        return Err(Failure::protocol(format!("{label} is not canonical padded Base64")));
    }
    Ok(decoded)
}

#[derive(Clone, Debug)]
struct PreparedKey {
    key_id: String,
    role: String,
    trust_domain: String,
    verifying_key: VerifyingKey,
}

fn prepare_trust_policy(policy_bytes: &[u8], expected_digest: &str) -> Result<(TrustPolicy, Vec<PreparedKey>), Failure> {
    if !is_sha256(expected_digest) {
        return Err(Failure::protocol("expected trust-policy SHA-256 pin must be 64 lowercase hexadecimal characters"));
    }
    let actual_digest = sha256_hex(policy_bytes);
    if actual_digest != expected_digest {
        return Err(Failure::protocol(format!(
            "trust-policy digest mismatch: expected {expected_digest}, got {actual_digest}"
        )));
    }
    let policy_value = parse_strict_json(policy_bytes, "trust policy")?;
    let policy: TrustPolicy = serde_json::from_value(policy_value)
        .map_err(|error| Failure::protocol(format!("trust policy has unknown, duplicate, or invalid fields: {error}")))?;
    if policy.schema_version != 0 || policy.policy_id.trim().is_empty() || policy.minimum_signers == 0 {
        return Err(Failure::protocol("trust-policy identity, schema_version, or minimum_signers is invalid"));
    }
    if policy.keys.is_empty() || policy.keys.len() > 128 {
        return Err(Failure::protocol("trust policy must contain between 1 and 128 keys"));
    }

    let mut seen_ids = HashSet::new();
    let mut seen_public_keys = HashSet::new();
    let mut prepared = Vec::new();
    for entry in &policy.keys {
        if entry.key_id.trim().is_empty() || !seen_ids.insert(entry.key_id.as_str()) {
            return Err(Failure::protocol(format!("empty or duplicate trust key_id: {}", entry.key_id)));
        }
        if entry.algorithm != "ed25519" {
            return Err(Failure::protocol(format!("unsupported trust-key algorithm for {}", entry.key_id)));
        }
        if !matches!(entry.role.as_str(), "primary_runner" | "regrade_runner") {
            return Err(Failure::protocol(format!("unsupported trust-key role for {}", entry.key_id)));
        }
        if entry.trust_domain.trim().is_empty() {
            return Err(Failure::protocol(format!("empty trust domain for {}", entry.key_id)));
        }
        if !matches!(entry.status.as_str(), "active" | "revoked") {
            return Err(Failure::protocol(format!("unsupported trust-key status for {}", entry.key_id)));
        }
        let bytes = canonical_base64(&entry.public_key_base64, "trust public key")?;
        let public_key: [u8; 32] = bytes
            .as_slice()
            .try_into()
            .map_err(|_| Failure::protocol(format!("Ed25519 key {} must be exactly 32 bytes", entry.key_id)))?;
        if !seen_public_keys.insert(public_key) {
            return Err(Failure::protocol("the same Ed25519 public key is registered more than once"));
        }
        if entry.status == "active" {
            let verifying_key = VerifyingKey::from_bytes(&public_key)
                .map_err(|error| Failure::protocol(format!("invalid Ed25519 key {}: {error}", entry.key_id)))?;
            prepared.push(PreparedKey {
                key_id: entry.key_id.clone(),
                role: entry.role.clone(),
                trust_domain: entry.trust_domain.clone(),
                verifying_key,
            });
        }
    }
    for role in ["primary_runner", "regrade_runner"] {
        let distinct_domains: BTreeSet<&str> = prepared
            .iter()
            .filter(|key| key.role == role)
            .map(|key| key.trust_domain.as_str())
            .collect();
        if distinct_domains.len() < policy.minimum_signers {
            return Err(Failure::protocol(format!(
                "trust policy has fewer than minimum_signers distinct active trust domains for {role}"
            )));
        }
    }
    Ok((policy, prepared))
}

fn dsse_pae(payload_type: &str, payload: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(32 + payload_type.len() + payload.len());
    result.extend_from_slice(b"DSSEv1 ");
    result.extend_from_slice(payload_type.len().to_string().as_bytes());
    result.push(b' ');
    result.extend_from_slice(payload_type.as_bytes());
    result.push(b' ');
    result.extend_from_slice(payload.len().to_string().as_bytes());
    result.push(b' ');
    result.extend_from_slice(payload);
    result
}

#[derive(Clone, Debug)]
struct ArtifactSets {
    /// All non-patch evidence, including the stage report, available to bind checks.
    evidence_digests: BTreeSet<String>,
    /// Run-generated evidence excluding the stage report and submitted patch.
    execution_digests: BTreeSet<String>,
}

#[derive(Clone, Debug)]
struct VerifiedReceipt {
    predicate: Value,
    envelope_sha256: String,
    run_id: String,
    receipt_id: String,
    started_at: DateTime<FixedOffset>,
    finished_at: DateTime<FixedOffset>,
    signer_ids: BTreeSet<String>,
    signer_domains: BTreeSet<String>,
    check_evidence: BTreeSet<String>,
    artifact_evidence: BTreeSet<String>,
    execution_artifact_digests: BTreeSet<String>,
}

fn verify_signatures(
    signatures: &[StrictSignature],
    payload_type: &str,
    payload: &[u8],
    policy: &TrustPolicy,
    keys: &[PreparedKey],
    required_role: &str,
) -> Result<(BTreeSet<String>, BTreeSet<String>), Failure> {
    if signatures.is_empty() || signatures.len() > MAX_SIGNATURES {
        return Err(Failure::protocol(format!("DSSE signature count must be in 1..={MAX_SIGNATURES}")));
    }
    let pae = dsse_pae(payload_type, payload);
    let mut accepted_ids = BTreeSet::new();
    let mut accepted_domains = BTreeSet::new();
    for entry in signatures {
        if !accepted_ids.insert(entry.keyid.clone()) {
            return Err(Failure::protocol("duplicate DSSE keyid in one envelope"));
        }
        let trusted = keys.iter().find(|key| key.key_id == entry.keyid)
            .ok_or_else(|| Failure::protocol(format!("DSSE keyid is absent from pinned trust policy: {}", entry.keyid)))?;
        if trusted.role != required_role {
            return Err(Failure::protocol(format!("DSSE signer {} has role {}, expected {required_role}", entry.keyid, trusted.role)));
        }
        let policy_entry = policy.keys.iter().find(|key| key.key_id == entry.keyid)
            .ok_or_else(|| Failure::protocol("internal trust-policy key lookup failed"))?;
        if policy_entry.status != "active" {
            return Err(Failure::protocol(format!("revoked or inactive DSSE signer: {}", entry.keyid)));
        }
        let signature_bytes = canonical_base64(&entry.sig, "DSSE signature")?;
        let signature = Signature::from_slice(&signature_bytes)
            .map_err(|error| Failure::protocol(format!("malformed Ed25519 signature: {error}")))?;
        trusted.verifying_key.verify_strict(&pae, &signature)
            .map_err(|_| Failure::protocol(format!("DSSE signature verification failed for key {}", entry.keyid)))?;
        if !accepted_domains.insert(trusted.trust_domain.clone()) {
            return Err(Failure::protocol("multiple signatures from one trust domain cannot increase the signer threshold"));
        }
    }
    if accepted_ids.len() < policy.minimum_signers {
        return Err(Failure::protocol(format!(
            "DSSE signature threshold not met: {} valid signers, {} required",
            accepted_ids.len(),
            policy.minimum_signers
        )));
    }
    Ok((accepted_ids, accepted_domains))
}

fn parse_timestamp(value: &Value, key: &str) -> Result<DateTime<FixedOffset>, Failure> {
    let text = value.as_str().ok_or_else(|| Failure::protocol(format!("execution.{key} must be an RFC 3339 date-time")))?;
    DateTime::parse_from_rfc3339(text)
        .map_err(|error| Failure::protocol(format!("execution.{key} is not a valid RFC 3339 date-time: {error}")))
}

fn safe_relative_path(value: &str) -> bool {
    if value.is_empty()
        || value.starts_with('/')
        || value.contains('\\')
        || value.contains(':')
        || value.contains('\0')
        || value.starts_with('~')
    {
        return false;
    }
    value.split('/').all(|part| !part.is_empty() && part != "." && part != "..")
}

fn hash_artifact(root: &Path, relative: &str, expected_size: u64, expected_digest: &str) -> Result<(), Failure> {
    if !safe_relative_path(relative) {
        return Err(Failure::protocol(format!("unsafe artifact path: {relative}")));
    }
    let canonical_root = root.canonicalize()
        .map_err(|error| Failure::protocol(format!("cannot resolve artifact root {}: {error}", root.display())))?;
    if !canonical_root.is_dir() {
        return Err(Failure::protocol("artifact root is not a directory"));
    }

    // Reject every symlink component, then verify the canonical target remains under
    // the selected root. The artifact store must be immutable/read-only during checking.
    let mut candidate = canonical_root.clone();
    for component in relative.split('/') {
        candidate.push(component);
        let metadata = fs::symlink_metadata(&candidate)
            .map_err(|error| Failure::protocol(format!("artifact {relative} is missing or unreadable: {error}")))?;
        if metadata.file_type().is_symlink() {
            return Err(Failure::protocol(format!("artifact path traverses a symlink: {relative}")));
        }
    }
    let canonical_file = candidate.canonicalize()
        .map_err(|error| Failure::protocol(format!("cannot canonicalize artifact {relative}: {error}")))?;
    if !canonical_file.starts_with(&canonical_root) {
        return Err(Failure::protocol(format!("artifact escapes its root: {relative}")));
    }
    let metadata = fs::metadata(&canonical_file)
        .map_err(|error| Failure::protocol(format!("cannot stat artifact {relative}: {error}")))?;
    if !metadata.is_file() || metadata.len() > MAX_ARTIFACT_BYTES {
        return Err(Failure::protocol(format!("artifact {relative} is not a regular file within the size limit")));
    }
    if metadata.len() != expected_size {
        return Err(Failure::protocol(format!("artifact {relative} size mismatch")));
    }
    let mut file = File::open(&canonical_file)
        .map_err(|error| Failure::protocol(format!("cannot open artifact {relative}: {error}")))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let read = file.read(&mut buffer)
            .map_err(|error| Failure::protocol(format!("cannot read artifact {relative}: {error}")))?;
        if read == 0 { break; }
        total = total.checked_add(read as u64)
            .ok_or_else(|| Failure::protocol("artifact length arithmetic overflow"))?;
        if total > MAX_ARTIFACT_BYTES {
            return Err(Failure::protocol(format!("artifact {relative} exceeds the size limit while reading")));
        }
        hasher.update(&buffer[..read]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if total != expected_size || actual != expected_digest {
        return Err(Failure::protocol(format!("artifact {relative} digest or final size mismatch")));
    }
    Ok(())
}

fn validate_artifacts(
    predicate: &Value,
    root: &Path,
    expected_patch_sha256: &str,
    expected_stage: &str,
) -> Result<ArtifactSets, Failure> {
    let artifacts = field(predicate, "artifacts", "predicate")?
        .as_array()
        .ok_or_else(|| Failure::protocol("predicate.artifacts must be an array"))?;
    if artifacts.len() < 2 || artifacts.len() > MAX_ARTIFACTS {
        return Err(Failure::unresolved(format!(
            "receipt must reference a patch plus at least one retained execution artifact (maximum {MAX_ARTIFACTS})"
        )));
    }
    let mut ids = HashSet::new();
    let mut paths = HashSet::new();
    let mut patch_count = 0usize;
    let mut stage_report_count = 0usize;
    let mut evidence_digests = BTreeSet::new();
    let mut execution_digests = BTreeSet::new();
    let mut all_digests = BTreeSet::new();
    let mut total_artifact_bytes = 0u64;
    let required_report_role = match expected_stage {
        "primary" => "primary_report",
        "pristine_regrade" => "regrade_report",
        _ => return Err(Failure::protocol("unsupported receipt stage for artifact verification")),
    };
    for artifact in artifacts {
        let id = string_field(artifact, "artifact_id", "artifact")?;
        let path = string_field(artifact, "path", "artifact")?;
        let role = string_field(artifact, "role", "artifact")?;
        let digest = string_field(artifact, "sha256", "artifact")?;
        let size = u64_field(artifact, "size_bytes", "artifact")?;
        total_artifact_bytes = total_artifact_bytes.checked_add(size)
            .ok_or_else(|| Failure::protocol("artifact aggregate size overflow"))?;
        if total_artifact_bytes > MAX_TOTAL_ARTIFACT_BYTES {
            return Err(Failure::unresolved("retained artifact set exceeds the aggregate 512 MiB verifier limit"));
        }
        if id.trim().is_empty() || id.contains("..") || id.contains('/') || id.contains('\\') || id.contains('\0') {
            return Err(Failure::protocol(format!("unsafe artifact_id: {id}")));
        }
        if !ids.insert(id.clone()) || !paths.insert(path.clone()) {
            return Err(Failure::protocol("duplicate artifact_id or artifact path"));
        }
        if !is_sha256(&digest) || !all_digests.insert(digest.clone()) {
            return Err(Failure::protocol(format!("artifact {id} has malformed or duplicate SHA-256")));
        }
        hash_artifact(root, &path, size, &digest)?;
        if role == "patch" {
            patch_count += 1;
            if digest != expected_patch_sha256 {
                return Err(Failure::protocol("retained patch artifact digest differs from the candidate patch"));
            }
        } else {
            evidence_digests.insert(digest.clone());
            if role == "primary_report" || role == "regrade_report" {
                if role != required_report_role {
                    return Err(Failure::protocol("receipt contains a report for the wrong execution stage"));
                }
            } else {
                execution_digests.insert(digest.clone());
            }
        }
        if role == required_report_role {
            stage_report_count += 1;
        }
    }
    if patch_count != 1 {
        return Err(Failure::protocol("each receipt must contain exactly one patch artifact"));
    }
    if stage_report_count != 1 {
        return Err(Failure::unresolved(format!("receipt must retain exactly one {required_report_role} artifact")));
    }
    if execution_digests.is_empty() {
        return Err(Failure::unresolved("receipt contains no retained execution evidence artifact"));
    }
    Ok(ArtifactSets {
        evidence_digests,
        execution_digests,
    })
}
fn validate_changed_paths(candidate: &Value) -> Result<(), Failure> {
    let paths = field(candidate, "changed_paths", "candidate")?
        .as_array()
        .ok_or_else(|| Failure::protocol("candidate.changed_paths must be an array"))?;
    if paths.is_empty() {
        return Err(Failure::protocol("non-empty candidate patch must declare at least one changed path"));
    }
    let mut seen = HashSet::new();
    for path in paths {
        let path = path.as_str().ok_or_else(|| Failure::protocol("changed_paths entries must be strings"))?;
        if !safe_relative_path(path) {
            return Err(Failure::protocol(format!("unsafe changed path: {path}")));
        }
        if !seen.insert(path) {
            return Err(Failure::protocol(format!("duplicate changed path: {path}")));
        }
    }
    Ok(())
}

fn validate_check_aggregates(predicate: &Value) -> Result<BTreeSet<String>, Failure> {
    let checks = field(predicate, "checks", "predicate")?
        .as_array()
        .ok_or_else(|| Failure::protocol("predicate.checks must be an array"))?;
    let mut ids = HashSet::new();
    let mut evidence = BTreeSet::new();
    let mut count = 0u64;
    let mut passed = 0u64;
    let mut failed = 0u64;
    let mut errored = 0u64;
    let mut skipped = 0u64;
    for check in checks {
        let id = string_field(check, "check_id", "check")?;
        if !ids.insert(id.clone()) {
            return Err(Failure::protocol(format!("duplicate check_id: {id}")));
        }
        let required = bool_field(check, "required", "check")?;
        let status = string_field(check, "status", "check")?;
        let digest = optional_string_field(check, "evidence_sha256", "check")?;
        if let Some(digest) = digest {
            if !is_sha256(&digest) {
                return Err(Failure::protocol(format!("check {id} has malformed evidence SHA-256")));
            }
            evidence.insert(digest);
        }
        if !required { continue; }
        count += 1;
        match status.as_str() {
            "pass" => {
                passed += 1;
                if field(check, "evidence_sha256", "check")? == &Value::Null {
                    return Err(Failure::protocol(format!("passing required check {id} has no evidence digest")));
                }
            }
            "fail" => failed += 1,
            "error" => errored += 1,
            "skipped" => skipped += 1,
            _ => return Err(Failure::protocol(format!("unsupported check status for {id}"))),
        }
    }

    let result = field(predicate, "result", "predicate")?;
    let declared = [
        ("required_check_count", count),
        ("required_pass_count", passed),
        ("required_fail_count", failed),
        ("required_error_count", errored),
        ("required_skipped_count", skipped),
    ];
    for (key, computed) in declared {
        if u64_field(result, key, "result")? != computed {
            return Err(Failure::protocol(format!("result.{key} does not match required check rows")));
        }
    }
    let verdict = string_field(result, "verdict", "result")?;
    let safety = bool_field(result, "safety_violation", "result")?;
    let protocol = bool_field(result, "protocol_violation", "result")?;
    match verdict.as_str() {
        "pass" => {
            if count == 0 || passed != count || failed != 0 || errored != 0 || skipped != 0 || safety || protocol {
                return Err(Failure::protocol("claimed pass conflicts with required checks or violation flags"));
            }
        }
        "fail" => {
            if failed == 0 && errored == 0 && !safety && !protocol {
                return Err(Failure::protocol("claimed fail has no failed/error check or declared violation"));
            }
        }
        "unresolved" => {}
        _ => return Err(Failure::protocol("unsupported result verdict")),
    }
    Ok(evidence)
}

fn validate_task_check_binding(task: &Value, predicate: &Value) -> Result<(), Failure> {
    let declared = field(field(task, "acceptance", "task")?, "checks", "acceptance")?
        .as_array()
        .ok_or_else(|| Failure::protocol("task acceptance checks must be an array"))?;
    let actual = field(predicate, "checks", "predicate")?
        .as_array()
        .ok_or_else(|| Failure::protocol("receipt checks must be an array"))?;
    if declared.is_empty() {
        return Err(Failure::unresolved("qualified task contract declares no acceptance checks"));
    }
    if declared.len() != actual.len() {
        return Err(Failure::protocol("receipt check set does not exactly match task acceptance check set"));
    }
    let mut declared_by_id = BTreeMap::new();
    for check in declared {
        let id = string_field(check, "check_id", "task acceptance check")?;
        if declared_by_id.insert(id.clone(), check).is_some() {
            return Err(Failure::protocol(format!("duplicate task acceptance check_id: {id}")));
        }
    }
    let mut seen = HashSet::new();
    for check in actual {
        let id = string_field(check, "check_id", "receipt check")?;
        if !seen.insert(id.clone()) {
            return Err(Failure::protocol(format!("duplicate receipt check_id: {id}")));
        }
        let expected = declared_by_id.get(&id)
            .ok_or_else(|| Failure::protocol(format!("receipt contains undeclared acceptance check: {id}")))?;
        if string_field(check, "kind", "receipt check")?
            != string_field(expected, "kind", "task acceptance check")?
            || bool_field(check, "required", "receipt check")?
                != bool_field(expected, "required", "task acceptance check")?
        {
            return Err(Failure::protocol(format!("receipt check definition differs from task contract: {id}")));
        }
        let declared_authority = string_field(expected, "authority", "task acceptance check")?;
        let receipt_authority = string_field(check, "authority", "receipt check")?;
        if declared_authority != "task_spec" && declared_authority != receipt_authority {
            return Err(Failure::protocol(format!("receipt check authority differs from task contract: {id}")));
        }
    }
    if seen.len() != declared_by_id.len() {
        return Err(Failure::protocol("receipt omits one or more task acceptance checks"));
    }
    Ok(())
}

fn check_definitions(predicate: &Value) -> Result<BTreeMap<String, (String, bool, String, String)>, Failure> {
    let checks = field(predicate, "checks", "predicate")?
        .as_array()
        .ok_or_else(|| Failure::protocol("predicate.checks must be an array"))?;
    let mut output = BTreeMap::new();
    for check in checks {
        let id = string_field(check, "check_id", "check")?;
        let definition = (
            string_field(check, "kind", "check")?,
            bool_field(check, "required", "check")?,
            string_field(check, "authority", "check")?,
            string_field(check, "status", "check")?,
        );
        if output.insert(id.clone(), definition).is_some() {
            return Err(Failure::protocol(format!("duplicate check_id: {id}")));
        }
    }
    Ok(output)
}

fn validate_execution_environment(task: &Value, execution: &Value) -> Result<(), Failure> {
    let runner_profile = string_field(execution, "runner_profile", "execution")?;
    let network_policy = string_field(execution, "network_policy", "execution")?;
    let image_digest = optional_string_field(execution, "image_digest", "execution")?;
    if !matches!(runner_profile.as_str(), "isolated_container" | "isolated_microvm")
        || !matches!(network_policy.as_str(), "disabled" | "allowlisted")
        || image_digest.as_deref().is_none_or(|digest| !digest.starts_with("sha256:") || !is_sha256(&digest[7..]))
    {
        return Err(Failure::unresolved("runner profile, image digest, or network policy is not qualified"));
    }
    let task_environment = field(task, "environment", "task contract")?;
    let expected_profile = string_field(task_environment, "runner_profile", "environment")?;
    let expected_network = string_field(task_environment, "network_policy", "environment")?;
    let expected_runner_policy = string_field(task_environment, "runner_policy_sha256", "environment")?;
    let expected_toolchain = string_field(task_environment, "toolchain_lock_sha256", "environment")?;
    let expected_image = string_field(task_environment, "image", "environment")?;
    let expected_image_digest = expected_image
        .rsplit_once("@sha256:")
        .map(|(_, digest)| format!("sha256:{digest}"))
        .ok_or_else(|| Failure::unresolved("task environment image is not pinned by an OCI SHA-256 digest"))?;
    if runner_profile != expected_profile
        || network_policy != expected_network
        || image_digest.as_deref() != Some(expected_image_digest.as_str())
        || string_field(execution, "runner_policy_sha256", "execution")? != expected_runner_policy
        || string_field(execution, "toolchain_lock_sha256", "execution")? != expected_toolchain
    {
        return Err(Failure::protocol("receipt runtime environment differs from the immutable task contract"));
    }
    Ok(())
}

fn check_budget(task: &Value, predicate: &Value) -> Result<(), Failure> {
    let budget = field(task, "budget", "task")?;
    let resources = field(predicate, "resources", "predicate")?;
    let checks: [(&str, &str, u64); 4] = [
        ("wall_clock_seconds", "wall_clock_ms", 1_000),
        ("max_turns", "turns", 1),
        ("max_tool_calls", "tool_calls", 1),
        ("input_token_budget", "input_tokens", 1),
    ];
    for (limit_key, usage_key, multiplier) in checks {
        let limit = field(budget, limit_key, "budget")?;
        if limit.is_null() { continue; }
        let maximum = limit.as_u64()
            .ok_or_else(|| Failure::protocol(format!("budget.{limit_key} must be an integer or null")))?
            .checked_mul(multiplier)
            .ok_or_else(|| Failure::protocol(format!("budget.{limit_key} overflow")))?;
        let usage_value = field(resources, usage_key, "resources")?;
        let usage = usage_value.as_u64().ok_or_else(|| {
            Failure::unresolved(format!("resources.{usage_key} is required to enforce the declared task budget"))
        })?;
        if usage > maximum {
            return Err(Failure::rejected(format!("resources.{usage_key} exceeds budget.{limit_key}")));
        }
    }
    let output_limit = field(budget, "output_token_budget", "budget")?;
    if let Some(maximum) = output_limit.as_u64() {
        let usage = field(resources, "output_tokens", "resources")?.as_u64()
            .ok_or_else(|| Failure::unresolved("resources.output_tokens is required to enforce output_token_budget"))?;
        if usage > maximum {
            return Err(Failure::rejected("resources.output_tokens exceeds budget.output_token_budget"));
        }
    }
    let cost_limit = field(budget, "cost_budget_usd", "budget")?;
    if let Some(maximum) = cost_limit.as_f64() {
        let usage = field(resources, "cost_usd", "resources")?.as_f64()
            .ok_or_else(|| Failure::unresolved("resources.cost_usd is required to enforce cost_budget_usd"))?;
        if !usage.is_finite() || usage > maximum {
            return Err(Failure::rejected("resources.cost_usd exceeds budget.cost_budget_usd"));
        }
    }
    Ok(())
}

fn verify_one_receipt(
    envelope_bytes: &[u8],
    artifact_root: &Path,
    trust_policy: &TrustPolicy,
    trusted_keys: &[PreparedKey],
    expected_role: &str,
    expected_stage: &str,
    task_contract_sha256: &str,
    task: &Value,
    source_tree_sha256: &str,
    patch_sha256: &str,
    requested_task_id: &str,
) -> Result<VerifiedReceipt, Failure> {
    if envelope_bytes.len() > MAX_JSON_BYTES {
        return Err(Failure::protocol("DSSE envelope exceeds the maximum JSON size"));
    }
    let envelope_value = parse_strict_json(envelope_bytes, "DSSE envelope")?;
    validate_against_schema(ENVELOPE_SCHEMA, &envelope_value, "DSSE envelope")?;
    let envelope: StrictEnvelope = serde_json::from_value(envelope_value)
        .map_err(|error| Failure::protocol(format!("invalid DSSE envelope fields: {error}")))?;
    if envelope.payload_type != PAYLOAD_TYPE {
        return Err(Failure::protocol("unexpected DSSE payloadType"));
    }
    let payload = canonical_base64(&envelope.payload, "DSSE payload")?;
    if payload.len() > MAX_JSON_BYTES {
        return Err(Failure::protocol("DSSE payload exceeds the maximum JSON size"));
    }
    let statement = parse_strict_json(&payload, "in-toto statement")?;
    validate_against_schema(RECEIPT_SCHEMA, &statement, "CEB run receipt")?;
    let (signer_ids, signer_domains) = verify_signatures(
        &envelope.signatures,
        &envelope.payload_type,
        &payload,
        trust_policy,
        trusted_keys,
        expected_role,
    )?;

    if string_field(&statement, "_type", "statement")? != STATEMENT_TYPE
        || string_field(&statement, "predicateType", "statement")? != PREDICATE_TYPE
    {
        return Err(Failure::protocol("in-toto statement type or predicate type mismatch"));
    }
    let predicate = field(&statement, "predicate", "statement")?;
    let task_part = field(predicate, "task", "predicate")?;
    if string_field(task_part, "task_id", "task")? != requested_task_id {
        return Err(Failure::protocol("receipt task_id does not match requested task"));
    }
    if string_field(task_part, "task_contract_sha256", "task")? != task_contract_sha256 {
        return Err(Failure::protocol("receipt task contract digest mismatch"));
    }
    if string_field(task_part, "source_tree_sha256", "task")? != source_tree_sha256 {
        return Err(Failure::protocol("receipt source tree digest mismatch"));
    }
    let source_snapshot = field(task, "source_snapshot", "task contract")?;
    if string_field(task_part, "fixture_revision", "task")?
        != string_field(source_snapshot, "revision", "source_snapshot")?
    {
        return Err(Failure::protocol("receipt fixture_revision does not match the task contract"));
    }

    let candidate = field(predicate, "candidate", "predicate")?;
    if string_field(candidate, "base_tree_sha256", "candidate")? != source_tree_sha256
        || string_field(candidate, "patch_sha256", "candidate")? != patch_sha256
    {
        return Err(Failure::protocol("candidate base-tree or patch digest mismatch"));
    }
    validate_changed_paths(candidate)?;

    let subjects = field(&statement, "subject", "statement")?
        .as_array()
        .ok_or_else(|| Failure::protocol("statement.subject must be an array"))?;
    if subjects.len() != 2 {
        return Err(Failure::protocol("CEB statement must contain exactly the patch and source-tree subjects"));
    }
    let mut subject_names = HashSet::new();
    for subject in subjects {
        let name = string_field(subject, "name", "subject")?;
        let digest = field(subject, "digest", "subject")?;
        let digest = string_field(digest, "sha256", "subject.digest")?;
        if !subject_names.insert(name.clone()) {
            return Err(Failure::protocol("duplicate in-toto subject name"));
        }
        match name.as_str() {
            "ceb/candidate.patch" if digest == patch_sha256 => {}
            "ceb/source.tree" if digest == source_tree_sha256 => {}
            "ceb/candidate.patch" | "ceb/source.tree" => {
                return Err(Failure::protocol(format!("in-toto subject digest mismatch for {name}")));
            }
            _ => return Err(Failure::protocol(format!("unexpected in-toto subject: {name}"))),
        }
    }
    if !subject_names.contains("ceb/candidate.patch") || !subject_names.contains("ceb/source.tree") {
        return Err(Failure::protocol("required in-toto subjects are absent"));
    }

    let agent = field(predicate, "agent", "predicate")?;
    if bool_field(agent, "secrets_exposed", "agent")? {
        return Err(Failure::protocol("agent was exposed to secrets"));
    }
    let execution = field(predicate, "execution", "predicate")?;
    if string_field(execution, "stage", "execution")? != expected_stage {
        return Err(Failure::protocol(format!("receipt execution stage must be {expected_stage}")));
    }
    let run_id = string_field(execution, "run_id", "execution")?;
    let receipt_id = string_field(predicate, "receipt_id", "predicate")?;
    if run_id.trim().is_empty() || receipt_id.trim().is_empty() {
        return Err(Failure::protocol("run_id and receipt_id must be non-empty"));
    }
    let started_at = parse_timestamp(field(execution, "started_at_utc", "execution")?, "started_at_utc")?;
    let finished_at = parse_timestamp(field(execution, "finished_at_utc", "execution")?, "finished_at_utc")?;
    if started_at.offset().local_minus_utc() != 0 || finished_at.offset().local_minus_utc() != 0 {
        return Err(Failure::protocol("execution timestamps suffixed _utc must use a zero UTC offset"));
    }
    if finished_at < started_at {
        return Err(Failure::protocol("execution finished_at_utc precedes started_at_utc"));
    }

    validate_execution_environment(task, execution)?;

    let grader = field(predicate, "grader", "predicate")?;
    if !matches!(string_field(grader, "trust_domain", "grader")?.as_str(), "separate_host_control_plane" | "separate_vm")
        || string_field(grader, "candidate_access", "grader")? != "read_only"
        || bool_field(grader, "hidden_tests_visible_to_agent", "grader")?
        || bool_field(grader, "grader_writable_by_agent", "grader")?
    {
        return Err(Failure::unresolved("grader is not isolated from candidate writes or hidden-test disclosure"));
    }

    validate_task_check_binding(task, predicate)?;
    let check_evidence = validate_check_aggregates(predicate)?;
    let artifact_sets = validate_artifacts(predicate, artifact_root, patch_sha256, expected_stage)?;
    if !check_evidence.is_subset(&artifact_sets.evidence_digests) {
        return Err(Failure::protocol("one or more check evidence digests are not present in the retained artifacts"));
    }
    check_budget(task, predicate)?;

    let verdict = string_field(field(predicate, "result", "predicate")?, "verdict", "result")?;
    match verdict.as_str() {
        "pass" => {}
        "fail" => return Err(Failure::rejected(format!("{expected_stage} receipt reports a failed task"))),
        "unresolved" => return Err(Failure::unresolved(format!("{expected_stage} receipt is unresolved"))),
        _ => return Err(Failure::protocol("unknown receipt verdict")),
    }

    let regrade_of = optional_string_field(execution, "regrade_of_receipt_sha256", "execution")?;
    if expected_stage == "primary" && regrade_of.is_some() {
        return Err(Failure::protocol("primary receipt must not reference a prior regrade"));
    }
    let envelope_sha256 = sha256_hex(envelope_bytes);
    Ok(VerifiedReceipt {
        predicate: predicate.clone(),
        envelope_sha256,
        run_id,
        receipt_id,
        started_at,
        finished_at,
        signer_ids,
        signer_domains,
        check_evidence,
        artifact_evidence: artifact_sets.evidence_digests,
        execution_artifact_digests: artifact_sets.execution_digests,
    })
}

fn check_pair_consistency(primary: &VerifiedReceipt, regrade: &VerifiedReceipt) -> Result<(), Failure> {
    if primary.run_id == regrade.run_id || primary.receipt_id == regrade.receipt_id {
        return Err(Failure::protocol("primary and regrade must have distinct run_id and receipt_id values"));
    }
    if primary.envelope_sha256 == regrade.envelope_sha256 {
        return Err(Failure::protocol("primary and regrade envelope digests are identical"));
    }
    if regrade.started_at < primary.finished_at {
        return Err(Failure::protocol("pristine regrade started before the primary execution finished"));
    }
    let primary_predicate = &primary.predicate;
    let regrade_predicate = &regrade.predicate;
    if check_definitions(primary_predicate)? != check_definitions(regrade_predicate)? {
        return Err(Failure::protocol("primary and regrade check definitions or verdicts differ"));
    }
    for key in ["task", "candidate", "agent", "grader"] {
        if field(primary_predicate, key, "primary predicate")?
            != field(regrade_predicate, key, "regrade predicate")?
        {
            return Err(Failure::protocol(format!("primary/regrade {key} binding differs")));
        }
    }
    let primary_execution = field(primary_predicate, "execution", "primary predicate")?;
    let regrade_execution = field(regrade_predicate, "execution", "regrade predicate")?;
    for key in ["runner_profile", "image_digest", "runner_policy_sha256", "toolchain_lock_sha256", "network_policy"] {
        if field(primary_execution, key, "primary execution")?
            != field(regrade_execution, key, "regrade execution")?
        {
            return Err(Failure::protocol(format!("primary/regrade execution.{key} binding differs")));
        }
    }
    let linked = optional_string_field(regrade_execution, "regrade_of_receipt_sha256", "regrade execution")?;
    if linked.as_deref() != Some(primary.envelope_sha256.as_str()) {
        return Err(Failure::protocol("pristine regrade does not link the exact primary DSSE envelope digest"));
    }
    if !primary.signer_ids.is_disjoint(&regrade.signer_ids)
        || !primary.signer_domains.is_disjoint(&regrade.signer_domains)
    {
        return Err(Failure::protocol("primary and regrade signer identities/trust domains are not independent"));
    }
    if primary.check_evidence.is_empty()
        || regrade.check_evidence.is_empty()
        || !primary.check_evidence.is_disjoint(&regrade.check_evidence)
    {
        return Err(Failure::protocol("primary and regrade check-evidence digests are not independent"));
    }
    if primary.execution_artifact_digests.is_empty()
        || regrade.execution_artifact_digests.is_empty()
        || !primary.execution_artifact_digests.is_disjoint(&regrade.execution_artifact_digests)
    {
        return Err(Failure::protocol("primary and regrade run-generated execution artifact digests are not independent"));
    }
    Ok(())
}

fn base_report(status: &str, reason: String) -> VerificationReport {
    VerificationReport {
        report_version: 1,
        verifier_version: env!("CARGO_PKG_VERSION"),
        status: status.to_owned(),
        accepted: status == "accepted",
        task_id: None,
        task_contract_sha256: None,
        source_tree_sha256: None,
        patch_sha256: None,
        primary_envelope_sha256: None,
        regrade_envelope_sha256: None,
        primary_run_id: None,
        regrade_run_id: None,
        primary_signer_ids: Vec::new(),
        regrade_signer_ids: Vec::new(),
        primary_signer_domains: Vec::new(),
        regrade_signer_domains: Vec::new(),
        primary_check_evidence_sha256: Vec::new(),
        regrade_check_evidence_sha256: Vec::new(),
        primary_artifact_evidence_sha256: Vec::new(),
        regrade_artifact_evidence_sha256: Vec::new(),
        trust_policy_sha256: None,
        schema_sha256: schema_digests(),
        reasons: vec![reason],
        claim_ceiling: "Authenticated, schema-valid, task-bound evidence pair under the supplied trust policy; this does not prove runner isolation, independently adjudicate test semantics, recompute the resulting source tree, or authenticate the claimed model provider.",
    }
}

/// Verify one primary receipt and one independently signed pristine regrade.
///
/// All byte inputs are treated as untrusted. The function performs no shell calls,
/// process creation, network access, candidate execution, or filesystem writes.
pub fn verify_pair(inputs: PairInputs) -> VerificationReport {
    match verify_pair_inner(&inputs) {
        Ok(report) => report,
        Err(failure) => base_report(failure.status, failure.reason),
    }
}

fn verify_pair_inner(inputs: &PairInputs) -> Result<VerificationReport, Failure> {
    if inputs.task_contract.len() > MAX_JSON_BYTES || inputs.trust_policy.len() > MAX_JSON_BYTES {
        return Err(Failure::protocol("task contract or trust policy exceeds the maximum JSON size"));
    }
    if inputs.patch.len() > MAX_PATCH_BYTES {
        return Err(Failure::protocol("candidate patch exceeds the maximum size"));
    }
    if inputs.patch.is_empty() {
        return Err(Failure::rejected("empty candidate patch cannot satisfy the CEB positive control"));
    }
    if inputs.primary_envelope.len() > MAX_JSON_BYTES || inputs.regrade_envelope.len() > MAX_JSON_BYTES {
        return Err(Failure::protocol("a DSSE envelope exceeds the maximum JSON size"));
    }
    let task_contract_sha256 = sha256_hex(&inputs.task_contract);
    let patch_sha256 = sha256_hex(&inputs.patch);
    if !is_sha256(&inputs.source_tree_sha256) {
        return Err(Failure::protocol("source-tree pin must be 64 lowercase hexadecimal characters"));
    }

    let (trust_policy, trusted_keys) = prepare_trust_policy(
        &inputs.trust_policy,
        &inputs.expected_trust_policy_sha256,
    )?;
    let task = parse_strict_json(&inputs.task_contract, "task contract")?;
    validate_against_schema(TASK_SCHEMA, &task, "CEB task")?;
    if string_field(&task, "task_id", "task contract")? != inputs.task_id {
        return Err(Failure::protocol("task contract task_id does not match CLI task_id"));
    }
    let source_snapshot = field(&task, "source_snapshot", "task contract")?;
    let task_source_tree = optional_string_field(source_snapshot, "tree_sha256", "source_snapshot")?;
    let execution_status = string_field(&task, "execution_status", "task contract")?;
    if execution_status != "qualified" {
        return Err(Failure::unresolved("task contract is not qualified; design-only and fixture-ready tasks cannot be scored"));
    }
    if task_source_tree.as_deref() != Some(inputs.source_tree_sha256.as_str()) {
        return Err(Failure::protocol("source-tree pin does not match the task contract"));
    }

    let primary = verify_one_receipt(
        &inputs.primary_envelope,
        &inputs.primary_artifacts_dir,
        &trust_policy,
        &trusted_keys,
        "primary_runner",
        "primary",
        &task_contract_sha256,
        &task,
        &inputs.source_tree_sha256,
        &patch_sha256,
        &inputs.task_id,
    )?;
    let regrade = verify_one_receipt(
        &inputs.regrade_envelope,
        &inputs.regrade_artifacts_dir,
        &trust_policy,
        &trusted_keys,
        "regrade_runner",
        "pristine_regrade",
        &task_contract_sha256,
        &task,
        &inputs.source_tree_sha256,
        &patch_sha256,
        &inputs.task_id,
    )?;
    check_pair_consistency(&primary, &regrade)?;

    Ok(VerificationReport {
        report_version: 1,
        verifier_version: env!("CARGO_PKG_VERSION"),
        status: "accepted".to_owned(),
        accepted: true,
        task_id: Some(inputs.task_id.clone()),
        task_contract_sha256: Some(task_contract_sha256),
        source_tree_sha256: Some(inputs.source_tree_sha256.clone()),
        patch_sha256: Some(patch_sha256),
        primary_envelope_sha256: Some(primary.envelope_sha256),
        regrade_envelope_sha256: Some(regrade.envelope_sha256),
        primary_run_id: Some(primary.run_id),
        regrade_run_id: Some(regrade.run_id),
        primary_signer_ids: primary.signer_ids.into_iter().collect(),
        regrade_signer_ids: regrade.signer_ids.into_iter().collect(),
        primary_signer_domains: primary.signer_domains.into_iter().collect(),
        regrade_signer_domains: regrade.signer_domains.into_iter().collect(),
        primary_check_evidence_sha256: primary.check_evidence.into_iter().collect(),
        regrade_check_evidence_sha256: regrade.check_evidence.into_iter().collect(),
        primary_artifact_evidence_sha256: primary.artifact_evidence.into_iter().collect(),
        regrade_artifact_evidence_sha256: regrade.artifact_evidence.into_iter().collect(),
        trust_policy_sha256: Some(sha256_hex(&inputs.trust_policy)),
        schema_sha256: schema_digests(),
        reasons: Vec::new(),
        claim_ceiling: "Authenticated, schema-valid, task-bound evidence pair under the supplied trust policy; this does not prove runner isolation, independently adjudicate test semantics, recompute the resulting source tree, or authenticate the claimed model provider.",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn signing_key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    #[test]
    fn dsse_pae_counts_utf8_bytes_and_binds_payload_type() {
        let bytes = dsse_pae("application/vnd.in-toto+json", br#"{"ok":true}"#);
        assert_eq!(
            bytes,
            b"DSSEv1 28 application/vnd.in-toto+json 11 {\"ok\":true}"
        );
        assert_ne!(
            dsse_pae("application/vnd.in-toto+json", b"payload"),
            dsse_pae("application/json", b"payload")
        );
    }

    #[test]
    fn duplicate_object_members_are_rejected_recursively() {
        assert!(parse_strict_json(br#"{"a":1,"a":2}"#, "test").is_err());
        assert!(parse_strict_json(br#"{"a":{"b":1,"b":2}}"#, "test").is_err());
        assert!(parse_strict_json(br#"{"a":[1,2]}"#, "test").is_ok());
    }

    #[test]
    fn base64_must_be_canonical_padded_standard_encoding() {
        assert_eq!(canonical_base64("YWJj", "test").unwrap(), b"abc");
        assert!(canonical_base64("YWJj\n", "test").is_err());
        assert!(canonical_base64("YWJj==", "test").is_err());
    }

    #[test]
    fn safe_paths_reject_absolute_traversal_windows_and_empty_segments() {
        for unsafe_path in ["/etc/passwd", "../secret", "a/../secret", "a//b", "C:/secret", "a\\b", "a\0b", "~/secret"] {
            assert!(!safe_relative_path(unsafe_path), "accepted unsafe path {unsafe_path:?}");
        }
        assert!(safe_relative_path("logs/run-01/test.log"));
    }

    #[test]
    fn signature_verification_binds_pae_and_trust_role() {
        let key = signing_key(7);
        let verifying = key.verifying_key();
        let public_key_base64 = STANDARD.encode(verifying.to_bytes());
        let policy = TrustPolicy {
            schema_version: 0,
            policy_id: "unit-test".to_string(),
            minimum_signers: 1,
            keys: vec![
                TrustedKey {
                    key_id: "primary".to_string(),
                    algorithm: "ed25519".to_string(),
                    public_key_base64,
                    role: "primary_runner".to_string(),
                    trust_domain: "primary-domain".to_string(),
                    status: "active".to_string(),
                },
                {
                    let second = signing_key(8);
                    TrustedKey {
                        key_id: "regrade".to_string(),
                        algorithm: "ed25519".to_string(),
                        public_key_base64: STANDARD.encode(second.verifying_key().to_bytes()),
                        role: "regrade_runner".to_string(),
                        trust_domain: "regrade-domain".to_string(),
                        status: "active".to_string(),
                    }
                },
            ],
        };
        let prepared = prepare_trust_policy_keys_for_test(&policy);
        let payload = br#"{"_type":"test"}"#;
        let pae = dsse_pae(PAYLOAD_TYPE, payload);
        let signature = key.sign(&pae);
        let signatures = vec![StrictSignature {
            keyid: "primary".to_string(),
            sig: STANDARD.encode(signature.to_bytes()),
        }];
        let result = verify_signatures(&signatures, PAYLOAD_TYPE, payload, &policy, &prepared, "primary_runner");
        assert!(result.is_ok());
        assert!(verify_signatures(&signatures, "application/json", payload, &policy, &prepared, "primary_runner").is_err());
        assert!(verify_signatures(&signatures, PAYLOAD_TYPE, payload, &policy, &prepared, "regrade_runner").is_err());
    }

    fn prepare_trust_policy_keys_for_test(policy: &TrustPolicy) -> Vec<PreparedKey> {
        policy.keys.iter().map(|entry| {
            let decoded = STANDARD.decode(&entry.public_key_base64).unwrap();
            let bytes: [u8; 32] = decoded.try_into().unwrap();
            PreparedKey {
                key_id: entry.key_id.clone(),
                role: entry.role.clone(),
                trust_domain: entry.trust_domain.clone(),
                verifying_key: VerifyingKey::from_bytes(&bytes).unwrap(),
            }
        }).collect()
    }

    #[test]
    fn check_aggregate_counter_mismatch_is_protocol_violation() {
        let predicate = serde_json::json!({
            "checks": [{
                "check_id":"unit.test",
                "required":true,
                "status":"pass",
                "evidence_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            }],
            "result": {
                "verdict":"pass",
                "required_check_count":2,
                "required_pass_count":1,
                "required_fail_count":0,
                "required_error_count":0,
                "required_skipped_count":0,
                "safety_violation":false,
                "protocol_violation":false
            }
        });
        assert!(validate_check_aggregates(&predicate).is_err());
    }

    #[test]
    fn a_pass_cannot_hide_a_skipped_required_check() {
        let predicate = serde_json::json!({
            "checks": [
                {"check_id":"unit.one","required":true,"status":"pass","evidence_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
                {"check_id":"unit.two","required":true,"status":"skipped","evidence_sha256":null}
            ],
            "result": {
                "verdict":"pass",
                "required_check_count":2,
                "required_pass_count":1,
                "required_fail_count":0,
                "required_error_count":0,
                "required_skipped_count":1,
                "safety_violation":false,
                "protocol_violation":false
            }
        });
        assert!(validate_check_aggregates(&predicate).is_err());
    }

    #[test]
    fn schema_digests_are_stable_lowercase_sha256() {
        let digests = schema_digests();
        assert!(is_sha256(&digests.task));
        assert!(is_sha256(&digests.receipt));
        assert!(is_sha256(&digests.envelope));
    }
    #[test]
    fn receipt_check_set_must_exactly_match_task_contract() {
        let task = serde_json::json!({
            "acceptance": {
                "checks": [
                    {"check_id":"compile.clean","kind":"compile","authority":"task_spec","required":true},
                    {"check_id":"tests.behavior","kind":"unit_test","authority":"independent_grader","required":true}
                ]
            }
        });
        let matching = serde_json::json!({
            "checks": [
                {"check_id":"compile.clean","kind":"compile","authority":"external_tool","required":true},
                {"check_id":"tests.behavior","kind":"unit_test","authority":"independent_grader","required":true}
            ]
        });
        assert!(validate_task_check_binding(&task, &matching).is_ok());

        let omitted = serde_json::json!({
            "checks": [
                {"check_id":"compile.clean","kind":"compile","authority":"external_tool","required":true}
            ]
        });
        assert!(validate_task_check_binding(&task, &omitted).is_err());

        let undeclared = serde_json::json!({
            "checks": [
                {"check_id":"compile.clean","kind":"compile","authority":"external_tool","required":true},
                {"check_id":"tests.behavior","kind":"unit_test","authority":"independent_grader","required":true},
                {"check_id":"bonus.fake","kind":"unit_test","authority":"independent_grader","required":false}
            ]
        });
        assert!(validate_task_check_binding(&task, &undeclared).is_err());

        let wrong_kind = serde_json::json!({
            "checks": [
                {"check_id":"compile.clean","kind":"unit_test","authority":"external_tool","required":true},
                {"check_id":"tests.behavior","kind":"unit_test","authority":"independent_grader","required":true}
            ]
        });
        assert!(validate_task_check_binding(&task, &wrong_kind).is_err());

        let wrong_authority = serde_json::json!({
            "checks": [
                {"check_id":"compile.clean","kind":"compile","authority":"external_tool","required":true},
                {"check_id":"tests.behavior","kind":"unit_test","authority":"qualified_reviewer","required":true}
            ]
        });
        assert!(validate_task_check_binding(&task, &wrong_authority).is_err());
    }

    #[test]
    fn artifact_paths_are_confined_and_content_digests_are_recomputed() {
        use std::fs;
        let root = std::env::temp_dir().join(format!(
            "ceb-receipt-verifier-{}-artifact-test",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("logs")).unwrap();
        let patch = b"diff --git a/src/lib.rs b/src/lib.rs";
        let log = b"independent test report with unique run evidence";
        let report = b"primary stage report with a unique receipt digest";
        fs::write(root.join("candidate.patch"), patch).unwrap();
        fs::write(root.join("logs/test.log"), log).unwrap();
        fs::write(root.join("primary-report.json"), report).unwrap();
        let predicate = serde_json::json!({
            "artifacts": [
                {"artifact_id":"patch","path":"candidate.patch","role":"patch","sha256":sha256_hex(patch),"size_bytes":patch.len()},
                {"artifact_id":"test-log","path":"logs/test.log","role":"test_log","sha256":sha256_hex(log),"size_bytes":log.len()},
                {"artifact_id":"primary-report","path":"primary-report.json","role":"primary_report","sha256":sha256_hex(report),"size_bytes":report.len()}
            ]
        });
        let evidence = validate_artifacts(&predicate, &root, &sha256_hex(patch), "primary").unwrap();
        assert!(evidence.evidence_digests.contains(&sha256_hex(log)));
        assert!(evidence.execution_digests.contains(&sha256_hex(log)));

        let report_only = serde_json::json!({
            "artifacts": [
                {"artifact_id":"patch","path":"candidate.patch","role":"patch","sha256":sha256_hex(patch),"size_bytes":patch.len()},
                {"artifact_id":"primary-report","path":"primary-report.json","role":"primary_report","sha256":sha256_hex(report),"size_bytes":report.len()}
            ]
        });
        assert!(validate_artifacts(&report_only, &root, &sha256_hex(patch), "primary").is_err());

        let wrong_patch_digest = "a".repeat(64);
        assert!(validate_artifacts(&predicate, &root, &wrong_patch_digest, "primary").is_err());

        let traversal = serde_json::json!({
            "artifacts": [
                {"artifact_id":"patch","path":"../candidate.patch","role":"patch","sha256":sha256_hex(patch),"size_bytes":patch.len()},
                {"artifact_id":"test-log","path":"logs/test.log","role":"test_log","sha256":sha256_hex(log),"size_bytes":log.len()},
                {"artifact_id":"primary-report","path":"primary-report.json","role":"primary_report","sha256":sha256_hex(report),"size_bytes":report.len()}
            ]
        });
        assert!(validate_artifacts(&traversal, &root, &sha256_hex(patch), "primary").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn budget_is_enforced_against_measured_usage() {
        let task = serde_json::json!({
            "budget": {
                "wall_clock_seconds":10,
                "max_turns":20,
                "max_tool_calls":30,
                "input_token_budget":1000,
                "output_token_budget":500,
                "cost_budget_usd":0.25
            }
        });
        let within = serde_json::json!({
            "resources": {
                "wall_clock_ms":9000,
                "turns":20,
                "tool_calls":30,
                "input_tokens":1000,
                "output_tokens":500,
                "cost_usd":0.25
            }
        });
        assert!(check_budget(&task, &within).is_ok());
        let over = serde_json::json!({
            "resources": {
                "wall_clock_ms":9000,
                "turns":21,
                "tool_calls":30,
                "input_tokens":1000,
                "output_tokens":500,
                "cost_usd":0.25
            }
        });
        assert!(check_budget(&task, &over).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn artifact_symlinks_are_rejected() {
        use std::fs;
        use std::os::unix::fs::symlink;
        let root = std::env::temp_dir().join(format!(
            "ceb-receipt-verifier-{}-symlink-test",
            std::process::id()
        ));
        let outside = root.with_extension("outside");
        let _ = fs::remove_dir_all(&root);
        let _ = fs::remove_file(&outside);
        fs::create_dir_all(&root).unwrap();
        let patch = b"patch";
        let log = b"evidence";
        let report = b"primary stage report";
        fs::write(&outside, log).unwrap();
        fs::write(root.join("candidate.patch"), patch).unwrap();
        fs::write(root.join("primary-report.json"), report).unwrap();
        symlink(&outside, root.join("outside.log")).unwrap();
        let predicate = serde_json::json!({
            "artifacts": [
                {"artifact_id":"patch","path":"candidate.patch","role":"patch","sha256":sha256_hex(patch),"size_bytes":patch.len()},
                {"artifact_id":"log","path":"outside.log","role":"test_log","sha256":sha256_hex(log),"size_bytes":log.len()},
                {"artifact_id":"primary-report","path":"primary-report.json","role":"primary_report","sha256":sha256_hex(report),"size_bytes":report.len()}
            ]
        });
        assert!(validate_artifacts(&predicate, &root, &sha256_hex(patch), "primary").is_err());
        fs::remove_dir_all(&root).unwrap();
        fs::remove_file(outside).unwrap();
    }

    #[test]
    fn execution_environment_must_match_the_task_contract() {
        let task = serde_json::json!({
            "environment": {
                "image": "oci://registry.example/ceb@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "toolchain_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "runner_profile": "isolated_microvm",
                "network_policy": "disabled",
                "runner_policy_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
            }
        });
        let execution = serde_json::json!({
            "runner_profile": "isolated_microvm",
            "image_digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "runner_policy_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "toolchain_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "network_policy": "disabled"
        });
        assert!(validate_execution_environment(&task, &execution).is_ok());

        let mut changed_image = execution.clone();
        changed_image["image_digest"] = serde_json::json!("sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd");
        assert!(validate_execution_environment(&task, &changed_image).is_err());

        let mut changed_network = execution.clone();
        changed_network["network_policy"] = serde_json::json!("allowlisted");
        assert!(validate_execution_environment(&task, &changed_network).is_err());

        let mut changed_toolchain = execution.clone();
        changed_toolchain["toolchain_lock_sha256"] = serde_json::json!("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee");
        assert!(validate_execution_environment(&task, &changed_toolchain).is_err());

        let mut mutable_image = task.clone();
        mutable_image["environment"]["image"] = serde_json::json!("oci://registry.example/ceb:latest");
        assert!(validate_execution_environment(&mutable_image, &execution).is_err());
    }

    fn write_test_artifact(root: &std::path::Path, path: &str, bytes: &[u8]) -> Value {
        let target = root.join(path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&target, bytes).unwrap();
        serde_json::json!({
            "artifact_id": path.replace('/', "-"),
            "role": if path == "candidate.patch" { "patch" } else if path == "primary-report.json" { "primary_report" } else if path == "regrade-report.json" { "regrade_report" } else { "test_log" },
            "sha256": sha256_hex(bytes),
            "size_bytes": bytes.len(),
            "path": path
        })
    }

    fn signed_test_envelope(
        statement: &Value,
        payload_type: &str,
        key_id: &str,
        signing_key: &SigningKey,
    ) -> Vec<u8> {
        let payload = serde_json::to_vec(statement).unwrap();
        let pae = dsse_pae(payload_type, &payload);
        let signature = signing_key.sign(&pae);
        serde_json::to_vec(&serde_json::json!({
            "payloadType": payload_type,
            "payload": STANDARD.encode(payload),
            "signatures": [{
                "keyid": key_id,
                "sig": STANDARD.encode(signature.to_bytes())
            }]
        })).unwrap()
    }

    fn build_test_receipt(
        stage: &str,
        run_id: &str,
        receipt_id: &str,
        started_at: &str,
        finished_at: &str,
        parent_envelope_sha256: Option<&str>,
        task_contract_sha256: &str,
        source_tree_sha256: &str,
        patch_sha256: &str,
        artifact_root: &std::path::Path,
    ) -> Value {
        let patch_bytes = b"diff --git a/src/lib.rs b/src/lib.rs\n+pub fn verified() -> bool { true }\n";
        let log_bytes = format!("independent {stage} test evidence").into_bytes();
        let report_bytes = format!("signed {stage} run report").into_bytes();
        let patch_artifact = write_test_artifact(artifact_root, "candidate.patch", patch_bytes);
        let compile_log = format!("{stage} compiler success").into_bytes();
        let compile_artifact = write_test_artifact(artifact_root, "logs/compile.log", &compile_log);
        let log_artifact = write_test_artifact(artifact_root, "logs/test.log", &log_bytes);
        let report_name = if stage == "primary" { "primary-report.json" } else { "regrade-report.json" };
        let report_artifact = write_test_artifact(artifact_root, report_name, &report_bytes);
        let compile_digest = sha256_hex(&compile_log);
        let check_digest = sha256_hex(&log_bytes);
        serde_json::json!({
            "_type": STATEMENT_TYPE,
            "subject": [
                {"name":"ceb/candidate.patch","digest":{"sha256":patch_sha256}},
                {"name":"ceb/source.tree","digest":{"sha256":source_tree_sha256}}
            ],
            "predicateType": PREDICATE_TYPE,
            "predicate": {
                "schema_version": 0,
                "receipt_kind": "ceb_run_receipt",
                "receipt_id": receipt_id,
                "task": {
                    "task_id": "CEB-RUST-001",
                    "task_contract_sha256": task_contract_sha256,
                    "source_tree_sha256": source_tree_sha256,
                    "fixture_revision": "synthetic-rust-fixture-v1"
                },
                "candidate": {
                    "base_tree_sha256": source_tree_sha256,
                    "patch_sha256": patch_sha256,
                    "result_tree_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                    "changed_paths": ["src/lib.rs"]
                },
                "agent": {
                    "agent_id": "unit-test-agent",
                    "model_id": "unit-test-model",
                    "harness_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                    "prompt_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                    "configuration_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
                    "secrets_exposed": false
                },
                "execution": {
                    "run_id": run_id,
                    "stage": stage,
                    "runner_profile": "isolated_microvm",
                    "image_digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "runner_policy_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                    "toolchain_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "network_policy": "disabled",
                    "started_at_utc": started_at,
                    "finished_at_utc": finished_at,
                    "regrade_of_receipt_sha256": parent_envelope_sha256
                },
                "grader": {
                    "grader_id": "unit-independent-grader",
                    "grader_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
                    "trust_domain": "separate_host_control_plane",
                    "candidate_access": "read_only",
                    "hidden_tests_visible_to_agent": false,
                    "grader_writable_by_agent": false
                },
                "checks": [
                    {
                        "check_id": "compile.clean",
                        "kind": "compile",
                        "required": true,
                        "authority": "external_tool",
                        "status": "pass",
                        "evidence_sha256": compile_digest
                    },
                    {
                        "check_id": "tests.correctness",
                        "kind": "hidden_test",
                        "required": true,
                        "authority": "independent_grader",
                        "status": "pass",
                        "evidence_sha256": check_digest
                    }
                ],
                "result": {
                    "verdict": "pass",
                    "required_check_count": 2,
                    "required_pass_count": 2,
                    "required_fail_count": 0,
                    "required_error_count": 0,
                    "required_skipped_count": 0,
                    "safety_violation": false,
                    "protocol_violation": false
                },
                "resources": {
                    "wall_clock_ms": 5000,
                    "cpu_time_ms": 4000,
                    "peak_memory_bytes": 1048576,
                    "peak_disk_bytes": 2097152,
                    "output_bytes": 1024,
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "turns": 3,
                    "tool_calls": 4,
                    "cost_usd": 0.01
                },
                "artifacts": [patch_artifact, compile_artifact, log_artifact, report_artifact]
            }
        })
    }

    fn build_test_task() -> Value {
        serde_json::json!({
            "schema_version": 0,
            "benchmark_id": "CEB",
            "task_id": "CEB-RUST-001",
            "execution_status": "qualified",
            "lane": "rust_engineering",
            "title": "Qualify a small Rust API change",
            "objective": "Implement a deterministic API behavior and verify all required contracts.",
            "agent_prompt": "Implement the described API change without modifying the grader or hidden tests.",
            "source_snapshot": {
                "kind": "synthetic_fixture",
                "uri": "fixture://ceb-e2e/rust",
                "revision": "synthetic-rust-fixture-v1",
                "tree_sha256": "3333333333333333333333333333333333333333333333333333333333333333"
            },
            "environment": {
                "image": "oci://registry.example/ceb@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "toolchain_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "runner_profile": "isolated_microvm",
                "network_policy": "disabled",
                "runner_policy_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
            },
            "agent_policy": {
                "writable_paths": ["src"],
                "read_only_paths": ["/grader", "/hidden-tests"],
                "secrets_exposed": false,
                "grader_visible": false,
                "human_approval_required": []
            },
            "acceptance": {
                "reference_patch_sha256": "4444444444444444444444444444444444444444444444444444444444444444",
                "empty_patch_must_fail": true,
                "independent_grader": "unit-independent-grader",
                "grader_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
                "checks": [
                    {
                        "check_id": "compile.clean",
                        "kind": "compile",
                        "authority": "external_tool",
                        "required": true,
                        "description": "The exact task source must compile under the pinned toolchain."
                    },
                    {
                        "check_id": "tests.correctness",
                        "kind": "hidden_test",
                        "authority": "independent_grader",
                        "required": true,
                        "description": "The hidden behavioral contract must pass in the isolated grader."
                    }
                ]
            },
            "budget": {
                "wall_clock_seconds": 10,
                "max_turns": 10,
                "max_tool_calls": 10,
                "input_token_budget": 1000,
                "output_token_budget": 500,
                "cost_budget_usd": 0.25
            },
            "split": {
                "partition": "private_holdout",
                "sealed": true,
                "leakage_groups": ["synthetic-rust-api"],
                "freshness_policy": "Never expose this task or its reference patch during training."
            }
        })
    }

    fn build_test_trust_policy(
        primary: &SigningKey,
        regrade: &SigningKey,
    ) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 0,
            "policy_id": "ceb-e2e-unit-test-policy-v1",
            "minimum_signers": 1,
            "keys": [
                {
                    "key_id": "primary-test-key",
                    "algorithm": "ed25519",
                    "public_key_base64": STANDARD.encode(primary.verifying_key().to_bytes()),
                    "role": "primary_runner",
                    "trust_domain": "primary-control-plane",
                    "status": "active"
                },
                {
                    "key_id": "regrade-test-key",
                    "algorithm": "ed25519",
                    "public_key_base64": STANDARD.encode(regrade.verifying_key().to_bytes()),
                    "role": "regrade_runner",
                    "trust_domain": "regrade-control-plane",
                    "status": "active"
                }
            ]
        })).unwrap()
    }

    #[test]
    fn complete_independently_signed_primary_regrade_pair_is_accepted() {
        use std::fs;
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let root = std::env::temp_dir().join(format!(
            "ceb-e2e-pair-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let primary_root = root.join("primary");
        let regrade_root = root.join("regrade");
        fs::create_dir_all(&primary_root).unwrap();
        fs::create_dir_all(&regrade_root).unwrap();

        let primary_key = signing_key(31);
        let regrade_key = signing_key(32);
        let policy_bytes = build_test_trust_policy(&primary_key, &regrade_key);
        let policy_digest = sha256_hex(&policy_bytes);
        let task_bytes = serde_json::to_vec(&build_test_task()).unwrap();
        let task_digest = sha256_hex(&task_bytes);
        let source_digest = "3333333333333333333333333333333333333333333333333333333333333333".to_string();
        let patch = b"diff --git a/src/lib.rs b/src/lib.rs\n+pub fn verified() -> bool { true }\n".to_vec();
        let patch_digest = sha256_hex(&patch);

        let primary_statement = build_test_receipt(
            "primary",
            "primary-run-0001",
            "primary-receipt-0001",
            "2026-11-03T00:00:00Z",
            "2026-11-03T00:00:05Z",
            None,
            &task_digest,
            &source_digest,
            &patch_digest,
            &primary_root
        );
        let primary_envelope = signed_test_envelope(
            &primary_statement,
            PAYLOAD_TYPE,
            "primary-test-key",
            &primary_key
        );
        let primary_digest = sha256_hex(&primary_envelope);

        let regrade_statement = build_test_receipt(
            "pristine_regrade",
            "regrade-run-0002",
            "regrade-receipt-0002",
            "2026-11-03T00:00:06Z",
            "2026-11-03T00:00:11Z",
            Some(&primary_digest),
            &task_digest,
            &source_digest,
            &patch_digest,
            &regrade_root
        );
        let regrade_envelope = signed_test_envelope(
            &regrade_statement,
            PAYLOAD_TYPE,
            "regrade-test-key",
            &regrade_key
        );

        let report = verify_pair(PairInputs {
            task_contract: task_bytes.clone(),
            task_id: "CEB-RUST-001".to_string(),
            source_tree_sha256: source_digest,
            patch,
            primary_envelope: primary_envelope.clone(),
            primary_artifacts_dir: primary_root.clone(),
            regrade_envelope: regrade_envelope.clone(),
            regrade_artifacts_dir: regrade_root.clone(),
            trust_policy: policy_bytes.clone(),
            expected_trust_policy_sha256: policy_digest.clone(),
        });
        assert!(report.accepted, "unexpected verifier rejection: {:?}", report.reasons);
        assert_eq!(report.status, "accepted");
        assert_eq!(report.primary_run_id.as_deref(), Some("primary-run-0001"));
        assert_eq!(report.regrade_run_id.as_deref(), Some("regrade-run-0002"));
        assert_ne!(report.primary_envelope_sha256, report.regrade_envelope_sha256);
        assert_ne!(report.primary_artifact_evidence_sha256, report.regrade_artifact_evidence_sha256);
        assert_eq!(report.primary_signer_ids, vec!["primary-test-key".to_string()]);
        assert_eq!(report.regrade_signer_ids, vec!["regrade-test-key".to_string()]);
        assert_eq!(report.primary_check_evidence_sha256.len(), 2);
        assert_eq!(report.regrade_check_evidence_sha256.len(), 2);

        // Re-signing a regrade that does not link the exact primary envelope must fail.
        let mut detached = regrade_statement.clone();
        detached["predicate"]["execution"]["regrade_of_receipt_sha256"] =
            serde_json::json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        let detached_envelope = signed_test_envelope(
            &detached, PAYLOAD_TYPE, "regrade-test-key", &regrade_key
        );
        let detached_report = verify_pair(PairInputs {
            task_contract: task_bytes.clone(),
            task_id: "CEB-RUST-001".to_string(),
            source_tree_sha256: "3333333333333333333333333333333333333333333333333333333333333333".to_string(),
            patch: b"diff --git a/src/lib.rs b/src/lib.rs\n+pub fn verified() -> bool { true }\n".to_vec(),
            primary_envelope: primary_envelope.clone(),
            primary_artifacts_dir: primary_root.clone(),
            regrade_envelope: detached_envelope,
            regrade_artifacts_dir: regrade_root.clone(),
            trust_policy: build_test_trust_policy(&primary_key, &regrade_key),
            expected_trust_policy_sha256: policy_digest.clone(),
        });
        assert!(!detached_report.accepted);
        assert_eq!(detached_report.status, "protocol_violation");

        // A second run cannot reuse primary execution evidence.
        let mut reused = regrade_statement;
        reused["predicate"]["checks"][0]["evidence_sha256"] =
            serde_json::json!(primary_statement["predicate"]["checks"][0]["evidence_sha256"].clone());
        reused["predicate"]["artifacts"][1]["sha256"] =
            primary_statement["predicate"]["artifacts"][1]["sha256"].clone();
        let reused_envelope = signed_test_envelope(
            &reused, PAYLOAD_TYPE, "regrade-test-key", &regrade_key
        );
        let reused_report = verify_pair(PairInputs {
            task_contract: task_bytes,
            task_id: "CEB-RUST-001".to_string(),
            source_tree_sha256: "3333333333333333333333333333333333333333333333333333333333333333".to_string(),
            patch: b"diff --git a/src/lib.rs b/src/lib.rs\n+pub fn verified() -> bool { true }\n".to_vec(),
            primary_envelope,
            primary_artifacts_dir: primary_root,
            regrade_envelope: reused_envelope,
            regrade_artifacts_dir: regrade_root,
            trust_policy: build_test_trust_policy(&primary_key, &regrade_key),
            expected_trust_policy_sha256: policy_digest,
        });
        assert!(!reused_report.accepted);
        assert!(matches!(reused_report.status.as_str(), "protocol_violation" | "rejected"));

        fs::remove_dir_all(root).unwrap();
    }

}
