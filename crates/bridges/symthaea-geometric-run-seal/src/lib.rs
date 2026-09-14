// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! GEOM run-environment sealing and evidence inventory.
//!
//! This crate keeps two surfaces deliberately separate:
//!
//! - an immutable scientific environment snapshot, captured before and after
//!   execution and required to have the same domain-separated commitment;
//! - a content-addressed evidence inventory, which is allowed to contain the
//!   outputs produced by the experiment harness.
//!
//! The first GEOM D0/D1 campaign additionally requires the agent persistence
//! root to remain empty throughout execution. Experiment outputs therefore
//! cannot be mistaken for agent state.

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use symthaea_geometric_environment_preflight::GeomEnvironmentPreflightReport;

pub const RUN_ENVIRONMENT_SCHEMA: &str = "symthaea.geom.run-environment.v1";
pub const EVIDENCE_INVENTORY_SCHEMA: &str = "symthaea.geom.evidence-inventory.v1";

const RUN_ENVIRONMENT_DOMAIN: &str = "symthaea:geom:run-environment:v1";
const EVIDENCE_INVENTORY_DOMAIN: &str = "symthaea:geom:evidence-inventory:v1";
const CONFIG_JSON_DOMAIN: &str = "symthaea:geom:cognitive-loop-config-json:v1";
const PREFLIGHT_JSON_DOMAIN: &str = "symthaea:geom:environment-preflight-json:v1";
const PROCESS_ENV_DOMAIN: &str = "symthaea:geom:process-environment:v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceIdentity {
    pub subject_sha: String,
    pub tree_sha: String,
    pub clean_tree: bool,
    pub cargo_lock_blake3: String,
    pub flake_lock_blake3: Option<String>,
    pub rust_toolchain_blake3: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeIdentity {
    pub rustc_version: String,
    pub cargo_version: String,
    pub host_triple: String,
    pub target_triple: String,
    pub os: String,
    pub arch: String,
    pub hardware_identity: String,
    pub thread_policy: String,
    pub cargo_features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignIdentity {
    pub campaign_id: String,
    pub arm_id: String,
    pub analysis_authority_revision: String,
    pub input_schedule_commitment: String,
    pub arm_order_commitment: String,
    pub fixed_utc_hour: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunEnvironmentSnapshot {
    pub schema: String,
    pub source: SourceIdentity,
    pub runtime: RuntimeIdentity,
    pub campaign: CampaignIdentity,
    pub config_commitment: String,
    pub preflight_commitment: String,
    pub process_environment_commitment: String,
    pub canonical_persistence_root: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SealedRunEnvironment {
    /// Non-authoritative metadata. This value is explicitly excluded from the
    /// scientific environment commitment and pre/post equality test.
    pub captured_at_utc: Option<String>,
    pub snapshot: RunEnvironmentSnapshot,
    pub commitment: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentStabilityReceipt {
    pub schema: String,
    pub stable_commitment: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceArtifact {
    pub relative_path: String,
    pub bytes: u64,
    pub blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceInventory {
    pub schema: String,
    pub artifacts: Vec<EvidenceArtifact>,
    pub commitment: String,
}

#[derive(Debug)]
pub enum RunSealError {
    InvalidGitObjectId { field: &'static str },
    DirtySourceTree,
    InvalidDigest { field: &'static str },
    EmptyField { field: &'static str },
    SchemaMismatch { expected: &'static str, observed: String },
    MissingRequiredFeature { feature: &'static str },
    InvalidFixedUtcHour,
    PreflightClockMismatch,
    Serialization(String),
    NonUtf8Environment,
    EnvironmentDrift { before: String, after: String },
    RootMissing { kind: &'static str },
    RootNotDirectory { kind: &'static str },
    RootSymlink { kind: &'static str },
    RootUnreadable { kind: &'static str },
    PersistenceRootNotEmpty,
    NonUtf8Path,
    SymlinkRejected { path: String },
    SpecialFileRejected { path: String },
    FileRead { path: String },
}

impl fmt::Display for RunSealError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGitObjectId { field } => {
                write!(f, "{field} must be a 40- or 64-hex Git object id")
            }
            Self::DirtySourceTree => write!(f, "source tree must be clean"),
            Self::InvalidDigest { field } => write!(f, "{field} must be a 64-hex BLAKE3 digest"),
            Self::EmptyField { field } => write!(f, "{field} must be non-empty"),
            Self::SchemaMismatch { expected, observed } => {
                write!(f, "schema mismatch: expected {expected}, observed {observed}")
            }
            Self::MissingRequiredFeature { feature } => {
                write!(f, "required Cargo feature is missing: {feature}")
            }
            Self::InvalidFixedUtcHour => {
                write!(f, "fixed UTC hour must be finite and in [0, 24)")
            }
            Self::PreflightClockMismatch => {
                write!(f, "campaign fixed UTC hour does not match the accepted preflight")
            }
            Self::Serialization(message) => write!(f, "canonical serialization failed: {message}"),
            Self::NonUtf8Environment => {
                write!(f, "process environment contains a non-UTF-8 key or value")
            }
            Self::EnvironmentDrift { before, after } => write!(
                f,
                "run environment drifted: before commitment {before}, after commitment {after}"
            ),
            Self::RootMissing { kind } => write!(f, "{kind} root does not exist"),
            Self::RootNotDirectory { kind } => write!(f, "{kind} root is not a directory"),
            Self::RootSymlink { kind } => write!(f, "{kind} root must not be a symlink"),
            Self::RootUnreadable { kind } => write!(f, "{kind} root cannot be read/canonicalized"),
            Self::PersistenceRootNotEmpty => {
                write!(f, "agent persistence root must remain empty for the primary campaign")
            }
            Self::NonUtf8Path => write!(f, "evidence path is not valid UTF-8"),
            Self::SymlinkRejected { path } => write!(f, "symlink rejected from evidence tree: {path}"),
            Self::SpecialFileRejected { path } => {
                write!(f, "non-regular evidence filesystem entry rejected: {path}")
            }
            Self::FileRead { path } => write!(f, "failed to read evidence file: {path}"),
        }
    }
}

impl std::error::Error for RunSealError {}

/// Construct a normalized run snapshot from explicit source/runtime/campaign
/// identity plus the already-accepted D0A2 preflight.
///
/// The full config is committed via canonical JSON (sorted object keys) so
/// HashMap insertion order cannot alter the commitment.
pub fn build_run_environment_snapshot<T: Serialize>(
    mut source: SourceIdentity,
    mut runtime: RuntimeIdentity,
    mut campaign: CampaignIdentity,
    config: &T,
    preflight: &GeomEnvironmentPreflightReport,
    process_environment_commitment: String,
) -> Result<RunEnvironmentSnapshot, RunSealError> {
    runtime.cargo_features = normalized_features(runtime.cargo_features);
    validate_source(&source)?;
    validate_required_runtime(&runtime)?;
    validate_campaign(&campaign)?;

    campaign.fixed_utc_hour = normalize_hour(campaign.fixed_utc_hour)?;

    let preflight_hour = normalize_hour(preflight.fixed_utc_hour)?;
    if campaign.fixed_utc_hour.to_bits() != preflight_hour.to_bits() {
        return Err(RunSealError::PreflightClockMismatch);
    }

    validate_blake3_hex(
        "process_environment_commitment",
        &process_environment_commitment,
    )?;

    // Normalize Git ids to lowercase so case-only textual differences cannot
    // create artificial scientific identities.
    source.subject_sha.make_ascii_lowercase();
    source.tree_sha.make_ascii_lowercase();

    let config_commitment = canonical_json_commitment(CONFIG_JSON_DOMAIN, config)?;
    let preflight_commitment = canonical_json_commitment(PREFLIGHT_JSON_DOMAIN, preflight)?;

    Ok(RunEnvironmentSnapshot {
        schema: RUN_ENVIRONMENT_SCHEMA.to_string(),
        source,
        runtime,
        campaign,
        config_commitment,
        preflight_commitment,
        process_environment_commitment: process_environment_commitment.to_ascii_lowercase(),
        canonical_persistence_root: preflight.canonical_persistence_root.clone(),
    })
}

/// Capture the full process environment as one aggregate commitment.
///
/// The serialized run report never exposes environment values. The aggregate
/// commitment nevertheless changes if any UTF-8 key/value pair changes.
pub fn capture_process_environment_commitment() -> Result<String, RunSealError> {
    let mut entries = Vec::new();
    for (key, value) in std::env::vars_os() {
        let key = key.into_string().map_err(|_| RunSealError::NonUtf8Environment)?;
        let value = value
            .into_string()
            .map_err(|_| RunSealError::NonUtf8Environment)?;
        entries.push((key, value));
    }
    Ok(process_environment_commitment_from_entries(entries))
}

/// Deterministic helper used by independent tests and external harnesses.
pub fn process_environment_commitment_from_entries<I, K, V>(entries: I) -> String
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<String>,
    V: Into<String>,
{
    let mut entries: Vec<(String, String)> = entries
        .into_iter()
        .map(|(key, value)| (key.into(), value.into()))
        .collect();
    entries.sort();

    let mut writer = CommitmentWriter::new(PROCESS_ENV_DOMAIN);
    writer.u64(entries.len() as u64);
    for (key, value) in entries {
        writer.str(&key);
        writer.str(&value);
    }
    writer.finish()
}

/// Seal a snapshot. `captured_at_utc` is metadata only and is excluded from
/// the authoritative commitment.
pub fn seal_run_environment(
    snapshot: RunEnvironmentSnapshot,
    captured_at_utc: Option<String>,
) -> Result<SealedRunEnvironment, RunSealError> {
    let commitment = snapshot_commitment(&snapshot)?;
    Ok(SealedRunEnvironment {
        captured_at_utc,
        snapshot,
        commitment,
    })
}

/// Require exact scientific-environment identity before and after execution.
///
/// Capture timestamps are intentionally ignored. The snapshot commitments are
/// authoritative and include all scientific fields.
pub fn verify_environment_stable(
    before: &SealedRunEnvironment,
    after: &SealedRunEnvironment,
) -> Result<EnvironmentStabilityReceipt, RunSealError> {
    let before_recomputed = snapshot_commitment(&before.snapshot)?;
    let after_recomputed = snapshot_commitment(&after.snapshot)?;

    // Reject stale/tampered serialized commitment fields before comparing.
    if before.commitment != before_recomputed {
        return Err(RunSealError::EnvironmentDrift {
            before: before.commitment.clone(),
            after: before_recomputed,
        });
    }
    if after.commitment != after_recomputed {
        return Err(RunSealError::EnvironmentDrift {
            before: after.commitment.clone(),
            after: after_recomputed,
        });
    }

    if before_recomputed != after_recomputed {
        return Err(RunSealError::EnvironmentDrift {
            before: before_recomputed,
            after: after_recomputed,
        });
    }

    Ok(EnvironmentStabilityReceipt {
        schema: RUN_ENVIRONMENT_SCHEMA.to_string(),
        stable_commitment: before_recomputed,
    })
}

/// Re-check the primary campaign's agent persistence root after execution.
///
/// The root itself may exist, but it must be a real directory (not a symlink)
/// and contain no entries.
pub fn verify_primary_persistence_root_empty(root: &Path) -> Result<PathBuf, RunSealError> {
    let metadata = fs::symlink_metadata(root).map_err(|_| RunSealError::RootMissing {
        kind: "persistence",
    })?;
    if metadata.file_type().is_symlink() {
        return Err(RunSealError::RootSymlink {
            kind: "persistence",
        });
    }
    if !metadata.is_dir() {
        return Err(RunSealError::RootNotDirectory {
            kind: "persistence",
        });
    }
    let canonical = root
        .canonicalize()
        .map_err(|_| RunSealError::RootUnreadable {
            kind: "persistence",
        })?;
    let mut entries = fs::read_dir(&canonical).map_err(|_| RunSealError::RootUnreadable {
        kind: "persistence",
    })?;
    if entries.next().is_some() {
        return Err(RunSealError::PersistenceRootNotEmpty);
    }
    Ok(canonical)
}

/// Build a sorted, content-addressed inventory of experiment evidence.
///
/// Symlinks and special files are rejected. Only regular files and directories
/// are admitted.
pub fn inventory_evidence_directory(root: &Path) -> Result<EvidenceInventory, RunSealError> {
    let metadata = fs::symlink_metadata(root).map_err(|_| RunSealError::RootMissing {
        kind: "evidence",
    })?;
    if metadata.file_type().is_symlink() {
        return Err(RunSealError::RootSymlink { kind: "evidence" });
    }
    if !metadata.is_dir() {
        return Err(RunSealError::RootNotDirectory { kind: "evidence" });
    }
    let canonical = root
        .canonicalize()
        .map_err(|_| RunSealError::RootUnreadable { kind: "evidence" })?;

    let mut artifacts = Vec::new();
    inventory_recursive(&canonical, &canonical, &mut artifacts)?;
    artifacts.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));

    let commitment = evidence_inventory_commitment(&artifacts);
    Ok(EvidenceInventory {
        schema: EVIDENCE_INVENTORY_SCHEMA.to_string(),
        artifacts,
        commitment,
    })
}

/// BLAKE3 digest a regular file using streaming I/O.
pub fn blake3_file(path: &Path) -> Result<String, RunSealError> {
    let mut file = fs::File::open(path).map_err(|_| RunSealError::FileRead {
        path: path.to_string_lossy().into_owned(),
    })?;
    let mut hasher = Hasher::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|_| RunSealError::FileRead {
            path: path.to_string_lossy().into_owned(),
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

/// Canonical JSON commitment with lexicographically sorted object keys.
///
/// This is a Symthaea-local canonicalization profile, not a claim of RFC 8785
/// compliance. The domain tag is part of the commitment definition.
pub fn canonical_json_commitment<T: Serialize>(
    domain: &str,
    value: &T,
) -> Result<String, RunSealError> {
    if domain.trim().is_empty() {
        return Err(RunSealError::EmptyField { field: "domain" });
    }
    let value =
        serde_json::to_value(value).map_err(|err| RunSealError::Serialization(err.to_string()))?;
    let mut bytes = Vec::new();
    write_canonical_json(&value, &mut bytes)?;

    let mut writer = CommitmentWriter::new(domain);
    writer.bytes(&bytes);
    Ok(writer.finish())
}

fn snapshot_commitment(snapshot: &RunEnvironmentSnapshot) -> Result<String, RunSealError> {
    if snapshot.schema != RUN_ENVIRONMENT_SCHEMA {
        return Err(RunSealError::SchemaMismatch {
            expected: RUN_ENVIRONMENT_SCHEMA,
            observed: snapshot.schema.clone(),
        });
    }
    validate_source(&snapshot.source)?;
    validate_required_runtime(&snapshot.runtime)?;
    validate_campaign(&snapshot.campaign)?;
    validate_blake3_hex("config_commitment", &snapshot.config_commitment)?;
    validate_blake3_hex("preflight_commitment", &snapshot.preflight_commitment)?;
    validate_blake3_hex(
        "process_environment_commitment",
        &snapshot.process_environment_commitment,
    )?;
    require_nonempty("canonical_persistence_root", &snapshot.canonical_persistence_root)?;

    let mut writer = CommitmentWriter::new(RUN_ENVIRONMENT_DOMAIN);
    writer.str("schema");
    writer.str(&snapshot.schema);

    writer.str("subject_sha");
    writer.str(&snapshot.source.subject_sha.to_ascii_lowercase());
    writer.str("tree_sha");
    writer.str(&snapshot.source.tree_sha.to_ascii_lowercase());
    writer.str("clean_tree");
    writer.bool(snapshot.source.clean_tree);
    writer.str("cargo_lock_blake3");
    writer.str(&snapshot.source.cargo_lock_blake3.to_ascii_lowercase());
    writer.str("flake_lock_blake3");
    writer.opt_str(snapshot.source.flake_lock_blake3.as_deref());
    writer.str("rust_toolchain_blake3");
    writer.opt_str(snapshot.source.rust_toolchain_blake3.as_deref());

    writer.str("rustc_version");
    writer.str(&snapshot.runtime.rustc_version);
    writer.str("cargo_version");
    writer.str(&snapshot.runtime.cargo_version);
    writer.str("host_triple");
    writer.str(&snapshot.runtime.host_triple);
    writer.str("target_triple");
    writer.str(&snapshot.runtime.target_triple);
    writer.str("os");
    writer.str(&snapshot.runtime.os);
    writer.str("arch");
    writer.str(&snapshot.runtime.arch);
    writer.str("hardware_identity");
    writer.str(&snapshot.runtime.hardware_identity);
    writer.str("thread_policy");
    writer.str(&snapshot.runtime.thread_policy);
    writer.str("cargo_features");
    let features = normalized_features(snapshot.runtime.cargo_features.clone());
    writer.str_vec(&features);

    writer.str("campaign_id");
    writer.str(&snapshot.campaign.campaign_id);
    writer.str("arm_id");
    writer.str(&snapshot.campaign.arm_id);
    writer.str("analysis_authority_revision");
    writer.str(&snapshot.campaign.analysis_authority_revision);
    writer.str("input_schedule_commitment");
    writer.str(&snapshot.campaign.input_schedule_commitment);
    writer.str("arm_order_commitment");
    writer.str(&snapshot.campaign.arm_order_commitment);
    writer.str("fixed_utc_hour_bits");
    writer.u64(normalize_hour(snapshot.campaign.fixed_utc_hour)?.to_bits());

    writer.str("config_commitment");
    writer.str(&snapshot.config_commitment.to_ascii_lowercase());
    writer.str("preflight_commitment");
    writer.str(&snapshot.preflight_commitment.to_ascii_lowercase());
    writer.str("process_environment_commitment");
    writer.str(&snapshot.process_environment_commitment.to_ascii_lowercase());
    writer.str("canonical_persistence_root");
    writer.str(&snapshot.canonical_persistence_root);

    Ok(writer.finish())
}

fn validate_source(source: &SourceIdentity) -> Result<(), RunSealError> {
    if !is_git_object_id(&source.subject_sha) {
        return Err(RunSealError::InvalidGitObjectId {
            field: "subject_sha",
        });
    }
    if !is_git_object_id(&source.tree_sha) {
        return Err(RunSealError::InvalidGitObjectId { field: "tree_sha" });
    }
    if !source.clean_tree {
        return Err(RunSealError::DirtySourceTree);
    }
    validate_blake3_hex("cargo_lock_blake3", &source.cargo_lock_blake3)?;
    if let Some(digest) = &source.flake_lock_blake3 {
        validate_blake3_hex("flake_lock_blake3", digest)?;
    }
    if let Some(digest) = &source.rust_toolchain_blake3 {
        validate_blake3_hex("rust_toolchain_blake3", digest)?;
    }
    Ok(())
}

fn validate_required_runtime(runtime: &RuntimeIdentity) -> Result<(), RunSealError> {
    for (field, value) in [
        ("rustc_version", runtime.rustc_version.as_str()),
        ("cargo_version", runtime.cargo_version.as_str()),
        ("host_triple", runtime.host_triple.as_str()),
        ("target_triple", runtime.target_triple.as_str()),
        ("os", runtime.os.as_str()),
        ("arch", runtime.arch.as_str()),
        ("hardware_identity", runtime.hardware_identity.as_str()),
        ("thread_policy", runtime.thread_policy.as_str()),
    ] {
        require_nonempty(field, value)?;
    }
    let features = normalized_features(runtime.cargo_features.clone());
    if !features.iter().any(|feature| feature == "scientific_method") {
        return Err(RunSealError::MissingRequiredFeature {
            feature: "scientific_method",
        });
    }
    Ok(())
}

fn validate_campaign(campaign: &CampaignIdentity) -> Result<(), RunSealError> {
    for (field, value) in [
        ("campaign_id", campaign.campaign_id.as_str()),
        ("arm_id", campaign.arm_id.as_str()),
        (
            "analysis_authority_revision",
            campaign.analysis_authority_revision.as_str(),
        ),
        (
            "input_schedule_commitment",
            campaign.input_schedule_commitment.as_str(),
        ),
        (
            "arm_order_commitment",
            campaign.arm_order_commitment.as_str(),
        ),
    ] {
        require_nonempty(field, value)?;
    }
    validate_blake3_hex(
        "input_schedule_commitment",
        &campaign.input_schedule_commitment,
    )?;
    validate_blake3_hex("arm_order_commitment", &campaign.arm_order_commitment)?;
    normalize_hour(campaign.fixed_utc_hour)?;
    Ok(())
}

fn normalize_hour(hour: f64) -> Result<f64, RunSealError> {
    if !hour.is_finite() || !(0.0..24.0).contains(&hour) {
        return Err(RunSealError::InvalidFixedUtcHour);
    }
    if hour == 0.0 {
        Ok(0.0)
    } else {
        Ok(hour)
    }
}

fn normalized_features(features: Vec<String>) -> Vec<String> {
    let mut features: Vec<String> = features
        .into_iter()
        .map(|feature| feature.trim().to_string())
        .filter(|feature| !feature.is_empty())
        .collect();
    features.sort();
    features.dedup();
    features
}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), RunSealError> {
    if value.trim().is_empty() {
        Err(RunSealError::EmptyField { field })
    } else {
        Ok(())
    }
}

fn validate_blake3_hex(field: &'static str, digest: &str) -> Result<(), RunSealError> {
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(RunSealError::InvalidDigest { field });
    }
    Ok(())
}

fn is_git_object_id(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn inventory_recursive(
    root: &Path,
    current: &Path,
    artifacts: &mut Vec<EvidenceArtifact>,
) -> Result<(), RunSealError> {
    let entries = fs::read_dir(current).map_err(|_| RunSealError::RootUnreadable {
        kind: "evidence",
    })?;

    for entry in entries {
        let entry = entry.map_err(|_| RunSealError::RootUnreadable { kind: "evidence" })?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).map_err(|_| RunSealError::FileRead {
            path: path.to_string_lossy().into_owned(),
        })?;
        let relative = path
            .strip_prefix(root)
            .map_err(|_| RunSealError::NonUtf8Path)?;
        let relative = relative
            .to_str()
            .ok_or(RunSealError::NonUtf8Path)?
            .replace('\\', "/");

        if metadata.file_type().is_symlink() {
            return Err(RunSealError::SymlinkRejected { path: relative });
        }
        if metadata.is_dir() {
            inventory_recursive(root, &path, artifacts)?;
            continue;
        }
        if !metadata.is_file() {
            return Err(RunSealError::SpecialFileRejected { path: relative });
        }

        let digest = blake3_file(&path)?;
        artifacts.push(EvidenceArtifact {
            relative_path: relative,
            bytes: metadata.len(),
            blake3: digest,
        });
    }
    Ok(())
}

fn evidence_inventory_commitment(artifacts: &[EvidenceArtifact]) -> String {
    let mut writer = CommitmentWriter::new(EVIDENCE_INVENTORY_DOMAIN);
    writer.str(EVIDENCE_INVENTORY_SCHEMA);
    writer.u64(artifacts.len() as u64);
    for artifact in artifacts {
        writer.str(&artifact.relative_path);
        writer.u64(artifact.bytes);
        writer.str(&artifact.blake3);
    }
    writer.finish()
}

fn write_canonical_json(value: &Value, output: &mut Vec<u8>) -> Result<(), RunSealError> {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(true) => output.extend_from_slice(b"true"),
        Value::Bool(false) => output.extend_from_slice(b"false"),
        Value::Number(number) => output.extend_from_slice(number.to_string().as_bytes()),
        Value::String(string) => {
            let encoded = serde_json::to_string(string)
                .map_err(|err| RunSealError::Serialization(err.to_string()))?;
            output.extend_from_slice(encoded.as_bytes());
        }
        Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                write_canonical_json(value, output)?;
            }
            output.push(b']');
        }
        Value::Object(map) => {
            output.push(b'{');
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                let encoded_key = serde_json::to_string(key)
                    .map_err(|err| RunSealError::Serialization(err.to_string()))?;
                output.extend_from_slice(encoded_key.as_bytes());
                output.push(b':');
                let item = map.get(key).ok_or_else(|| {
                    RunSealError::Serialization("canonical object key disappeared".to_string())
                })?;
                write_canonical_json(item, output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

struct CommitmentWriter {
    hasher: Hasher,
}

impl CommitmentWriter {
    fn new(domain: &str) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(b"SYMTHEAEA-COMMITMENT\0");
        let mut writer = Self { hasher };
        writer.str(domain);
        writer
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.u64(bytes.len() as u64);
        self.hasher.update(bytes);
    }

    fn str(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }

    fn bool(&mut self, value: bool) {
        self.hasher.update(&[u8::from(value)]);
    }

    fn u64(&mut self, value: u64) {
        self.hasher.update(&value.to_le_bytes());
    }

    fn opt_str(&mut self, value: Option<&str>) {
        self.bool(value.is_some());
        if let Some(value) = value {
            self.str(&value.to_ascii_lowercase());
        }
    }

    fn str_vec(&mut self, values: &[String]) {
        self.u64(values.len() as u64);
        for value in values {
            self.str(value);
        }
    }

    fn finish(self) -> String {
        self.hasher.finalize().to_hex().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU64, Ordering};
    use symthaea_geometric_environment_preflight::{
        EnvironmentSnapshot, GeomEnvironmentPreflightReport,
    };

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "symthaea_geom_run_seal_{label}_{}_{}",
            std::process::id(),
            NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).expect("create temp root");
        root.canonicalize().expect("canonical temp root")
    }

    fn hex64(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn fake_preflight(root: &Path, hour: f64) -> GeomEnvironmentPreflightReport {
        GeomEnvironmentPreflightReport {
            fixed_utc_hour: hour,
            genesis_phrase_present: true,
            async_training: false,
            online_learning: false,
            canonical_persistence_root: root.to_string_lossy().into_owned(),
            persistence_root_empty: true,
            aesthetic_memory_path: root.join("aesthetic.json").to_string_lossy().into_owned(),
            memory_db_path: None,
            epistemic_auditor_db_path: None,
            environment: EnvironmentSnapshot::from_present_names(&[]),
        }
    }

    fn source() -> SourceIdentity {
        SourceIdentity {
            subject_sha: "a".repeat(40),
            tree_sha: "b".repeat(40),
            clean_tree: true,
            cargo_lock_blake3: hex64('c'),
            flake_lock_blake3: Some(hex64('d')),
            rust_toolchain_blake3: Some(hex64('e')),
        }
    }

    fn runtime(features: Vec<&str>) -> RuntimeIdentity {
        RuntimeIdentity {
            rustc_version: "rustc 1.96.0".to_string(),
            cargo_version: "cargo 1.96.0".to_string(),
            host_triple: "x86_64-unknown-linux-gnu".to_string(),
            target_triple: "x86_64-unknown-linux-gnu".to_string(),
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            hardware_identity: "test-cpu".to_string(),
            thread_policy: "single-thread".to_string(),
            cargo_features: features.into_iter().map(str::to_string).collect(),
        }
    }

    fn campaign(hour: f64) -> CampaignIdentity {
        CampaignIdentity {
            campaign_id: "GEOM-003D0".to_string(),
            arm_id: "A-intact".to_string(),
            analysis_authority_revision: "GEOM-003D0B-v1".to_string(),
            input_schedule_commitment: hex64('2'),
            arm_order_commitment: hex64('3'),
            fixed_utc_hour: hour,
        }
    }

    #[derive(Serialize)]
    struct MapConfig {
        label: String,
        values: HashMap<String, usize>,
    }

    fn snapshot_with_map(
        root: &Path,
        values: HashMap<String, usize>,
        env_commitment: String,
    ) -> RunEnvironmentSnapshot {
        let config = MapConfig {
            label: "config".to_string(),
            values,
        };
        build_run_environment_snapshot(
            source(),
            runtime(vec!["scientific_method", "mathematics"]),
            campaign(12.5),
            &config,
            &fake_preflight(root, 12.5),
            env_commitment,
        )
        .expect("valid snapshot")
    }

    #[test]
    fn canonical_config_commitment_ignores_map_insertion_order() {
        let root = temp_root("map_order");
        let mut left = HashMap::new();
        left.insert("alpha".to_string(), 1);
        left.insert("beta".to_string(), 2);

        let mut right = HashMap::new();
        right.insert("beta".to_string(), 2);
        right.insert("alpha".to_string(), 1);

        let env = process_environment_commitment_from_entries([("A", "1"), ("B", "2")]);
        let a = snapshot_with_map(&root, left, env.clone());
        let b = snapshot_with_map(&root, right, env);

        assert_eq!(a.config_commitment, b.config_commitment);
        assert_eq!(
            snapshot_commitment(&a).unwrap(),
            snapshot_commitment(&b).unwrap()
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn process_environment_commitment_is_order_independent_but_value_sensitive() {
        let a = process_environment_commitment_from_entries([("B", "2"), ("A", "1")]);
        let b = process_environment_commitment_from_entries([("A", "1"), ("B", "2")]);
        let c = process_environment_commitment_from_entries([("A", "1"), ("B", "3")]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn cargo_features_are_sorted_and_deduplicated_before_sealing() {
        let root = temp_root("features");
        let config = MapConfig {
            label: "config".to_string(),
            values: HashMap::new(),
        };
        let snapshot = build_run_environment_snapshot(
            source(),
            runtime(vec![
                "mathematics",
                "scientific_method",
                "mathematics",
                "  scientific_method  ",
            ]),
            campaign(12.5),
            &config,
            &fake_preflight(&root, 12.5),
            hex64('f'),
        )
        .unwrap();
        assert_eq!(
            snapshot.runtime.cargo_features,
            vec!["mathematics".to_string(), "scientific_method".to_string()]
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn preflight_clock_mismatch_fails() {
        let root = temp_root("clock");
        let config = MapConfig {
            label: "config".to_string(),
            values: HashMap::new(),
        };
        let result = build_run_environment_snapshot(
            source(),
            runtime(vec!["scientific_method"]),
            campaign(11.0),
            &config,
            &fake_preflight(&root, 12.5),
            hex64('f'),
        );
        assert!(matches!(result, Err(RunSealError::PreflightClockMismatch)));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn capture_timestamp_is_not_part_of_environment_authority() {
        let root = temp_root("timestamps");
        let env = hex64('f');
        let snapshot = snapshot_with_map(&root, HashMap::new(), env);
        let before =
            seal_run_environment(snapshot.clone(), Some("2026-09-14T20:00:00Z".to_string()))
                .unwrap();
        let after =
            seal_run_environment(snapshot, Some("2026-09-14T20:01:00Z".to_string())).unwrap();
        let receipt = verify_environment_stable(&before, &after).unwrap();
        assert_eq!(receipt.stable_commitment, before.commitment);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn scientific_field_changes_trigger_environment_drift() {
        let root = temp_root("drift");
        let base = snapshot_with_map(&root, HashMap::new(), hex64('f'));
        let before = seal_run_environment(base.clone(), None).unwrap();

        let mut variants = Vec::new();

        let mut changed = base.clone();
        changed.source.subject_sha = "9".repeat(40);
        variants.push(changed);

        let mut changed = base.clone();
        changed.source.tree_sha = "8".repeat(40);
        variants.push(changed);

        let mut changed = base.clone();
        changed.runtime.rustc_version.push_str("-different");
        variants.push(changed);

        let mut changed = base.clone();
        changed.runtime.hardware_identity.push_str("-different");
        variants.push(changed);

        let mut changed = base.clone();
        changed.config_commitment = hex64('6');
        variants.push(changed);

        let mut changed = base.clone();
        changed.campaign.input_schedule_commitment = hex64('4');
        variants.push(changed);

        let mut changed = base.clone();
        changed.campaign.arm_order_commitment = hex64('5');
        variants.push(changed);

        let mut changed = base.clone();
        changed.campaign.arm_id.push_str("-different");
        variants.push(changed);

        let mut changed = base.clone();
        changed.campaign.fixed_utc_hour = 13.0;
        variants.push(changed);

        let mut changed = base.clone();
        changed.process_environment_commitment = hex64('1');
        variants.push(changed);

        for changed in variants {
            let after = seal_run_environment(changed, None).unwrap();
            assert!(matches!(
                verify_environment_stable(&before, &after),
                Err(RunSealError::EnvironmentDrift { .. })
            ));
        }
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn primary_persistence_root_must_remain_empty() {
        let root = temp_root("persistence");
        assert_eq!(
            verify_primary_persistence_root_empty(&root).unwrap(),
            root
        );
        fs::write(root.join("unexpected-state"), b"state").unwrap();
        assert!(matches!(
            verify_primary_persistence_root_empty(&root),
            Err(RunSealError::PersistenceRootNotEmpty)
        ));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn evidence_inventory_is_path_and_content_sensitive() {
        let root = temp_root("evidence");
        fs::write(root.join("a.json"), b"alpha").unwrap();
        fs::create_dir(root.join("nested")).unwrap();
        fs::write(root.join("nested").join("b.bin"), b"beta").unwrap();

        let first = inventory_evidence_directory(&root).unwrap();
        assert_eq!(first.artifacts.len(), 2);

        fs::write(root.join("a.json"), b"alpha-changed").unwrap();
        let changed_content = inventory_evidence_directory(&root).unwrap();
        assert_ne!(first.commitment, changed_content.commitment);

        fs::rename(root.join("a.json"), root.join("renamed.json")).unwrap();
        let changed_path = inventory_evidence_directory(&root).unwrap();
        assert_ne!(changed_content.commitment, changed_path.commitment);

        fs::write(root.join("extra.txt"), b"extra").unwrap();
        let added = inventory_evidence_directory(&root).unwrap();
        assert_ne!(changed_path.commitment, added.commitment);

        fs::remove_file(root.join("extra.txt")).unwrap();
        let removed = inventory_evidence_directory(&root).unwrap();
        assert_eq!(removed.commitment, changed_path.commitment);

        fs::remove_dir_all(root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn evidence_inventory_rejects_symlinks_and_special_files() {
        use std::os::unix::fs::symlink;
        use std::os::unix::net::UnixListener;

        let root = temp_root("special");
        let target = root.join("target.txt");
        fs::write(&target, b"x").unwrap();
        symlink(&target, root.join("link.txt")).unwrap();

        assert!(matches!(
            inventory_evidence_directory(&root),
            Err(RunSealError::SymlinkRejected { .. })
        ));
        fs::remove_file(root.join("link.txt")).unwrap();

        let socket = root.join("socket");
        let _listener = UnixListener::bind(&socket).unwrap();
        assert!(matches!(
            inventory_evidence_directory(&root),
            Err(RunSealError::SpecialFileRejected { .. })
        ));

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn stale_serialized_commitment_is_rejected() {
        let root = temp_root("stale");
        let snapshot = snapshot_with_map(&root, HashMap::new(), hex64('f'));
        let mut before = seal_run_environment(snapshot.clone(), None).unwrap();
        let after = seal_run_environment(snapshot, None).unwrap();
        before.commitment = hex64('0');
        assert!(matches!(
            verify_environment_stable(&before, &after),
            Err(RunSealError::EnvironmentDrift { .. })
        ));
        fs::remove_dir_all(root).ok();
    }
}
