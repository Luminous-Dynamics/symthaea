// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded, checksummed envelopes for serialized vision checkpoints.
//!
//! The checksum is an integrity checksum, not a cryptographic signature. It
//! detects accidental truncation, corruption, and metadata/payload mismatches
//! before a checkpoint is deserialized into its concrete state type.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{Error as IoError, ErrorKind, Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Current compact outer-envelope schema.
pub const CHECKPOINT_ENVELOPE_SCHEMA_VERSION: u32 = 2;
/// Legacy JSON envelope schema accepted for backwards compatibility.
const LEGACY_JSON_ENVELOPE_SCHEMA_VERSION: u32 = 1;
/// Binary marker for compact schema-2 envelopes.
const CHECKPOINT_BINARY_MAGIC: &[u8; 8] = b"SVMCKPT2";
/// Fixed bytes before the kind and payload in a compact envelope.
const CHECKPOINT_BINARY_HEADER_BYTES: usize = 8 + 4 + 2 + 4 + 8 + 8;
/// Binary marker for caller-authenticated checkpoint wrappers.
const CHECKPOINT_AUTH_MAGIC: &[u8; 8] = b"SVMAUTH1";
/// Fixed bytes before the inner envelope and authentication tag.
const CHECKPOINT_AUTH_HEADER_BYTES: usize = 8 + 8 + 4;
/// Conservative maximum authentication tag size.
pub const DEFAULT_MAX_CHECKPOINT_AUTH_TAG_BYTES: usize = 16 * 1024;
/// Conservative default maximum for the serialized state payload.
pub const DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;
/// Accepted envelope bound includes legacy JSON byte-array amplification.
pub const DEFAULT_MAX_CHECKPOINT_ENVELOPE_BYTES: usize =
    DEFAULT_MAX_CHECKPOINT_PAYLOAD_BYTES * 4 + 4096;
/// Bytes written between lease-refresh opportunities during atomic saves.
const CHECKPOINT_WRITE_CHUNK_BYTES: usize = 1024 * 1024;

static CHECKPOINT_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
static CHECKPOINT_LOCK_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Maximum lock acquisition attempts accepted by the writer policy.
pub const MAX_CHECKPOINT_LOCK_ATTEMPTS: usize = 10_000;
/// Maximum size accepted for a checkpoint writer-lock token.
const MAX_CHECKPOINT_LOCK_TOKEN_BYTES: usize = 512;

/// Writer that refuses to grow beyond a caller-supplied payload ceiling.
///
/// `serde_json::to_vec` only reports the final size after allocating the full
/// serialized state. Keeping the limit in the `Write` implementation makes the
/// allocation contract effective during serialization.
#[derive(Debug)]
struct BoundedPayloadWriter {
    bytes: Vec<u8>,
    limit: usize,
    overflowed: bool,
}

impl BoundedPayloadWriter {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(limit.min(64 * 1024)),
            limit,
            overflowed: false,
        }
    }

    fn into_inner(self) -> Vec<u8> {
        self.bytes
    }
}

impl Write for BoundedPayloadWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let remaining = self.limit.saturating_sub(self.bytes.len());
        if buf.len() > remaining {
            self.overflowed = true;
            return Err(IoError::new(
                ErrorKind::WriteZero,
                "checkpoint payload exceeds configured limit",
            ));
        }
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Age after which abandoned checkpoint temporary files are eligible for cleanup.
pub const DEFAULT_STALE_CHECKPOINT_TEMP_AGE: Duration = Duration::from_secs(24 * 60 * 60);
/// Maximum abandoned temporary files removed during one checkpoint write.
pub const DEFAULT_STALE_CHECKPOINT_TEMP_LIMIT: usize = 32;
/// Maximum previous generations retained by bounded recovery policies.
pub const MAX_CHECKPOINT_PREVIOUS_GENERATIONS: usize = 8;

/// Self-describing serialized checkpoint envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointEnvelope {
    pub schema_version: u32,
    pub kind: String,
    pub payload_schema_version: u32,
    pub payload_len: usize,
    pub checksum_fnv1a64: u64,
    pub payload: Vec<u8>,
}

/// Which generation supplied a successfully recovered checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointRecoverySource {
    Primary,
    Previous,
}

/// Verified outer-envelope metadata without deserializing the concrete state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointMetadata {
    pub envelope_schema_version: u32,
    pub kind: String,
    pub payload_schema_version: u32,
    pub payload_len: usize,
    pub checksum_fnv1a64: u64,
    pub compact: bool,
}

/// Observable result of one atomic checkpoint replacement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointWriteReport {
    pub encoded_bytes: usize,
    pub stale_temps_removed: usize,
    pub stale_temp_cleanup_error: Option<String>,
}

/// Observable result of a checkpoint save, including fallback rotation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointSaveReport {
    pub primary_write: CheckpointWriteReport,
    pub previous_write: Option<CheckpointWriteReport>,
}

/// Observable result of checkpoint loading and recovery selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointLoadReport {
    pub source: CheckpointRecoverySource,
    pub metadata: CheckpointMetadata,
    pub encoded_bytes: usize,
    pub previous_generation: Option<usize>,
    pub promotion_write: Option<CheckpointWriteReport>,
}

/// Cross-process coordination policy for checkpoint writers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointWriterLockPolicy {
    /// Number of create-new attempts before contention is reported.
    pub attempts: usize,
    /// Delay between contended attempts.
    pub retry_delay: Duration,
    /// Optional age after which an abandoned regular lock file may be removed.
    pub stale_after: Option<Duration>,
}

impl Default for CheckpointWriterLockPolicy {
    fn default() -> Self {
        Self {
            attempts: 50,
            retry_delay: Duration::from_millis(20),
            stale_after: Some(Duration::from_secs(10 * 60)),
        }
    }
}

impl CheckpointWriterLockPolicy {
    pub fn validate(self) -> Result<(), String> {
        if self.attempts == 0 || self.attempts > MAX_CHECKPOINT_LOCK_ATTEMPTS {
            return Err(format!(
                "checkpoint lock attempts must be in 1..={MAX_CHECKPOINT_LOCK_ATTEMPTS}, got {}",
                self.attempts
            ));
        }
        if self.retry_delay > Duration::from_secs(60) {
            return Err("checkpoint lock retry delay must not exceed 60 seconds".to_string());
        }
        Ok(())
    }
}

/// Structured ownership and liveness evidence stored in a writer lock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointWriterLockEvidence {
    pub pid: u32,
    pub nonce: u64,
    pub acquired_unix_secs: u64,
    pub heartbeat_unix_secs: u64,
    /// Linux boot identity prevents a pre-reboot PID from appearing live.
    #[serde(default)]
    pub boot_id: Option<String>,
    /// Linux `/proc/<pid>/stat` start tick prevents PID-reuse false positives.
    #[serde(default)]
    pub process_start_ticks: Option<u64>,
}

impl CheckpointWriterLockEvidence {
    /// Best-effort owner liveness using boot and process-start identity.
    ///
    /// `None` means the current platform cannot establish liveness safely.
    pub fn owner_is_alive(&self) -> Option<bool> {
        checkpoint_writer_owner_is_alive(self)
    }
}

/// Held checkpoint writer lease. The lock file is removed on drop only when
/// its complete ownership token still matches this guard.
#[derive(Debug)]
pub struct CheckpointWriterLock {
    path: PathBuf,
    token: String,
    evidence: CheckpointWriterLockEvidence,
}

impl CheckpointWriterLock {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn evidence(&self) -> &CheckpointWriterLockEvidence {
        &self.evidence
    }

    /// Refresh the lease heartbeat without changing ownership.
    ///
    /// The existing token is verified on the same opened file before writing.
    /// A post-write path verification detects unlink-and-replace races.
    pub fn refresh(&mut self) -> Result<(), String> {
        let mut options = OpenOptions::new();
        options.read(true).write(true);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        options.custom_flags(0o400000 | 0o2000000);
        let mut file = options.open(&self.path).map_err(|error| {
            format!(
                "failed to open checkpoint writer lock {}: {error}",
                self.path.display()
            )
        })?;
        let metadata = file.metadata().map_err(|error| {
            format!(
                "failed to inspect checkpoint writer lock {}: {error}",
                self.path.display()
            )
        })?;
        if !metadata.file_type().is_file()
            || metadata.len() > MAX_CHECKPOINT_LOCK_TOKEN_BYTES as u64
        {
            return Err(format!(
                "checkpoint writer lock is not a bounded regular file: {}",
                self.path.display()
            ));
        }
        let mut current = Vec::with_capacity(metadata.len() as usize);
        file.read_to_end(&mut current).map_err(|error| {
            format!(
                "failed to read checkpoint writer lock {}: {error}",
                self.path.display()
            )
        })?;
        if current != self.token.as_bytes() {
            return Err(format!(
                "checkpoint writer lock ownership changed before refresh: {}",
                self.path.display()
            ));
        }

        let now = unix_time_secs()?;
        let mut evidence = self.evidence.clone();
        evidence.heartbeat_unix_secs = now.max(evidence.heartbeat_unix_secs);
        let token = encode_writer_lock_evidence(&evidence)?;
        file.seek(SeekFrom::Start(0))
            .and_then(|_| file.set_len(0))
            .and_then(|_| file.write_all(token.as_bytes()))
            .and_then(|_| file.sync_all())
            .map_err(|error| {
                format!(
                    "failed to refresh checkpoint writer lock {}: {error}",
                    self.path.display()
                )
            })?;

        let visible = read_small_regular_file(&self.path, MAX_CHECKPOINT_LOCK_TOKEN_BYTES)?;
        if visible != token.as_bytes() {
            return Err(format!(
                "checkpoint writer lock ownership changed during refresh: {}",
                self.path.display()
            ));
        }
        self.token = token;
        self.evidence = evidence;
        Ok(())
    }
}

impl Drop for CheckpointWriterLock {
    fn drop(&mut self) {
        let current = read_small_regular_file(&self.path, MAX_CHECKPOINT_LOCK_TOKEN_BYTES).ok();
        if current.as_deref() == Some(self.token.as_bytes()) {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Bounded previous-generation retention policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointRetentionPolicy {
    pub previous_generations: usize,
}

impl CheckpointRetentionPolicy {
    pub fn validate(self) -> Result<(), String> {
        if self.previous_generations > MAX_CHECKPOINT_PREVIOUS_GENERATIONS {
            return Err(format!(
                "checkpoint retention exceeds supported bound: {} > {MAX_CHECKPOINT_PREVIOUS_GENERATIONS}",
                self.previous_generations
            ));
        }
        Ok(())
    }
}

/// Observable result of a bounded multi-generation save.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointRetentionSaveReport {
    pub primary_write: CheckpointWriteReport,
    pub generation_writes: Vec<(usize, CheckpointWriteReport)>,
}

/// Location represented by a generation inspection record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointGenerationLocation {
    Primary,
    Previous(usize),
}

/// Verified operational inventory entry for one checkpoint generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointGenerationInspection {
    pub location: CheckpointGenerationLocation,
    pub path: PathBuf,
    pub exists: bool,
    pub encoded_bytes: Option<usize>,
    pub metadata: Option<CheckpointMetadata>,
    pub error: Option<String>,
}

impl CheckpointGenerationInspection {
    pub fn is_valid(&self) -> bool {
        self.exists && self.metadata.is_some() && self.error.is_none()
    }
}

/// Outcome recorded while searching retained checkpoint history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointRecoveryAttemptOutcome {
    /// Envelope/file admission or decoding failed.
    StructuralFailure(String),
    /// The decoded payload failed the caller's semantic validator.
    SemanticFailure(String),
    /// This generation was selected for restoration.
    Selected,
}

/// One auditable checkpoint-generation recovery attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointRecoveryAttempt {
    pub location: CheckpointGenerationLocation,
    pub path: PathBuf,
    pub outcome: CheckpointRecoveryAttemptOutcome,
}

/// Selected load report plus every newer generation that was considered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointSemanticRecoveryReport {
    pub selected: CheckpointLoadReport,
    pub attempts: Vec<CheckpointRecoveryAttempt>,
}

/// Structured failure returned when no retained generation is semantically usable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointSemanticRecoveryFailure {
    /// Every generation that was actually inspected, in newest-to-oldest order.
    pub attempts: Vec<CheckpointRecoveryAttempt>,
    /// Setup failure that prevented a complete retained-history search.
    pub setup_error: Option<String>,
}

impl std::fmt::Display for CheckpointSemanticRecoveryFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(error) = &self.setup_error {
            return write!(formatter, "checkpoint recovery setup failed: {error}");
        }
        if self.attempts.is_empty() {
            return formatter.write_str("no checkpoint generations were inspected");
        }
        formatter
            .write_str("no structurally and semantically valid checkpoint generation found")?;
        for attempt in &self.attempts {
            match &attempt.outcome {
                CheckpointRecoveryAttemptOutcome::StructuralFailure(error) => {
                    write!(
                        formatter,
                        "; {:?}: structural failure: {error}",
                        attempt.location
                    )?;
                }
                CheckpointRecoveryAttemptOutcome::SemanticFailure(error) => {
                    write!(
                        formatter,
                        "; {:?}: semantic failure: {error}",
                        attempt.location
                    )?;
                }
                CheckpointRecoveryAttemptOutcome::Selected => {
                    write!(formatter, "; {:?}: unexpectedly selected", attempt.location)?;
                }
            }
        }
        Ok(())
    }
}

impl std::error::Error for CheckpointSemanticRecoveryFailure {}

/// Result of removing generations beyond a reduced retention policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointPruneReport {
    pub kept_previous_generations: usize,
    pub removed_generations: Vec<usize>,
    pub absent_generations: Vec<usize>,
}

fn validate_kind(kind: &str) -> Result<(), String> {
    if kind.is_empty() || kind.len() > 64 {
        return Err("checkpoint kind must contain 1..=64 bytes".to_string());
    }
    if !kind
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(format!(
            "checkpoint kind contains unsupported characters: {kind}"
        ));
    }
    Ok(())
}

fn fnv1a64_update(mut hash: u64, bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

#[cfg(test)]
pub(crate) fn fnv1a64_for_testing(seed: u64, bytes: &[u8]) -> u64 {
    fnv1a64_update(seed, bytes)
}

fn envelope_checksum(
    schema_version: u32,
    kind: &str,
    payload_schema_version: u32,
    payload: &[u8],
) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    hash = fnv1a64_update(hash, &schema_version.to_le_bytes());
    hash = fnv1a64_update(hash, &(kind.len() as u64).to_le_bytes());
    hash = fnv1a64_update(hash, kind.as_bytes());
    hash = fnv1a64_update(hash, &payload_schema_version.to_le_bytes());
    hash = fnv1a64_update(hash, &(payload.len() as u64).to_le_bytes());
    fnv1a64_update(hash, payload)
}

/// Maximum accepted envelope size.
///
/// The wider legacy bound remains intentional so existing schema-1 JSON
/// envelopes can still be loaded. Newly encoded schema-2 envelopes use
/// [`max_compact_envelope_bytes`] and do not expand payload bytes into a JSON
/// integer array.
pub fn max_envelope_bytes(max_payload_bytes: usize) -> usize {
    max_payload_bytes.saturating_mul(4).saturating_add(4096)
}

/// Maximum size of a newly written compact binary envelope.
pub fn max_compact_envelope_bytes(max_payload_bytes: usize) -> usize {
    max_payload_bytes
        .saturating_add(CHECKPOINT_BINARY_HEADER_BYTES)
        .saturating_add(64)
}

fn checkpoint_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn temporary_checkpoint_path(path: &Path) -> Result<PathBuf, String> {
    let parent = checkpoint_parent(path);
    let file_name = path
        .file_name()
        .ok_or_else(|| "checkpoint path must name a file".to_string())?
        .to_string_lossy();
    let sequence = CHECKPOINT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(parent.join(format!(
        ".{file_name}.tmp-{}-{sequence}",
        std::process::id()
    )))
}

fn checkpoint_temp_prefix(path: &Path) -> Result<String, String> {
    let file_name = path
        .file_name()
        .ok_or_else(|| "checkpoint path must name a file".to_string())?
        .to_string_lossy();
    Ok(format!(".{file_name}.tmp-"))
}

/// Deterministic writer-lock path for a checkpoint destination.
pub fn checkpoint_writer_lock_path(path: impl AsRef<Path>) -> Result<PathBuf, String> {
    let path = path.as_ref();
    let parent = checkpoint_parent(path);
    let file_name = path
        .file_name()
        .ok_or_else(|| "checkpoint path must name a file".to_string())?
        .to_string_lossy();
    Ok(parent.join(format!(".{file_name}.lock")))
}

fn read_small_regular_file(path: &Path, limit: usize) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("failed to inspect lock file {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(format!(
            "checkpoint writer lock must be a regular non-symlink file: {}",
            path.display()
        ));
    }
    if metadata.len() > limit as u64 {
        return Err(format!("checkpoint writer lock exceeds {limit} bytes"));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    options.custom_flags(0o400000 | 0o2000000);
    let mut file = options
        .open(path)
        .map_err(|error| format!("failed to open lock file {}: {error}", path.display()))?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    Read::by_ref(&mut file)
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read lock file {}: {error}", path.display()))?;
    if bytes.len() > limit {
        return Err(format!("checkpoint writer lock exceeds {limit} bytes"));
    }
    Ok(bytes)
}

fn unix_time_secs() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| format!("system clock precedes Unix epoch: {error}"))
}

fn encode_writer_lock_evidence(evidence: &CheckpointWriterLockEvidence) -> Result<String, String> {
    let mut token = serde_json::to_string(evidence)
        .map_err(|error| format!("failed to encode checkpoint writer lock evidence: {error}"))?;
    token.push('\n');
    if token.len() > MAX_CHECKPOINT_LOCK_TOKEN_BYTES {
        return Err("checkpoint writer lock evidence exceeds bounded token size".to_string());
    }
    Ok(token)
}

fn linux_boot_id() -> Option<String> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let value = fs::read_to_string("/proc/sys/kernel/random/boot_id").ok()?;
        let value = value.trim();
        if value.is_empty() || value.len() > 128 {
            None
        } else {
            Some(value.to_string())
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProcessStartLookup {
    Found(u64),
    Missing,
    Unknown,
}

fn linux_process_start_lookup(pid: u32) -> ProcessStartLookup {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let stat = match fs::read_to_string(format!("/proc/{pid}/stat")) {
            Ok(stat) => stat,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return ProcessStartLookup::Missing;
            }
            Err(_) => return ProcessStartLookup::Unknown,
        };
        let Some(command_end) = stat.rfind(')') else {
            return ProcessStartLookup::Unknown;
        };
        let Some(start_ticks) = stat
            .get(command_end + 1..)
            .and_then(|suffix| suffix.split_whitespace().nth(19))
            .and_then(|value| value.parse().ok())
        else {
            return ProcessStartLookup::Unknown;
        };
        ProcessStartLookup::Found(start_ticks)
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        let _ = pid;
        ProcessStartLookup::Unknown
    }
}

fn linux_process_start_ticks(pid: u32) -> Option<u64> {
    match linux_process_start_lookup(pid) {
        ProcessStartLookup::Found(ticks) => Some(ticks),
        ProcessStartLookup::Missing | ProcessStartLookup::Unknown => None,
    }
}

fn checkpoint_writer_owner_is_alive(evidence: &CheckpointWriterLockEvidence) -> Option<bool> {
    if let Some(saved_boot_id) = evidence.boot_id.as_deref() {
        let current_boot_id = linux_boot_id()?;
        if current_boot_id != saved_boot_id {
            return Some(false);
        }
    }
    if let Some(saved_start_ticks) = evidence.process_start_ticks {
        return match linux_process_start_lookup(evidence.pid) {
            ProcessStartLookup::Found(current_start_ticks) => {
                Some(current_start_ticks == saved_start_ticks)
            }
            ProcessStartLookup::Missing => Some(false),
            ProcessStartLookup::Unknown => None,
        };
    }
    None
}

fn writer_lock_stale_age(
    evidence: Option<&CheckpointWriterLockEvidence>,
    metadata: &fs::Metadata,
) -> Option<Duration> {
    if let Some(evidence) = evidence {
        let now = unix_time_secs().ok()?;
        if evidence.heartbeat_unix_secs > now {
            return None;
        }
        return Some(Duration::from_secs(
            now.saturating_sub(evidence.heartbeat_unix_secs),
        ));
    }
    metadata
        .modified()
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
}

/// Inspect structured ownership and heartbeat evidence for a writer lock.
pub fn inspect_checkpoint_writer_lock(
    path: impl AsRef<Path>,
) -> Result<CheckpointWriterLockEvidence, String> {
    let lock_path = checkpoint_writer_lock_path(path)?;
    let bytes = read_small_regular_file(&lock_path, MAX_CHECKPOINT_LOCK_TOKEN_BYTES)?;
    serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "invalid checkpoint writer lock evidence {}: {error}",
            lock_path.display()
        )
    })
}

/// Acquire an exclusive writer lease for a checkpoint destination.
///
/// The lease uses create-new semantics, so only one cooperating process can
/// rotate and replace a checkpoint generation set at a time.
pub fn acquire_checkpoint_writer_lock(
    path: impl AsRef<Path>,
    policy: CheckpointWriterLockPolicy,
) -> Result<CheckpointWriterLock, String> {
    policy.validate()?;
    let path = path.as_ref();
    let parent = checkpoint_parent(path);
    if !parent.is_dir() {
        return Err(format!(
            "checkpoint parent directory does not exist: {}",
            parent.display()
        ));
    }
    let lock_path = checkpoint_writer_lock_path(path)?;
    let now = unix_time_secs()?;
    let evidence = CheckpointWriterLockEvidence {
        pid: std::process::id(),
        nonce: CHECKPOINT_LOCK_COUNTER.fetch_add(1, Ordering::Relaxed),
        acquired_unix_secs: now,
        heartbeat_unix_secs: now,
        boot_id: linux_boot_id(),
        process_start_ticks: linux_process_start_ticks(std::process::id()),
    };
    let token = encode_writer_lock_evidence(&evidence)?;

    for attempt in 0..policy.attempts {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&lock_path) {
            Ok(mut file) => {
                if let Err(error) = file
                    .write_all(token.as_bytes())
                    .and_then(|_| file.sync_all())
                {
                    let _ = fs::remove_file(&lock_path);
                    return Err(format!(
                        "failed to initialize checkpoint writer lock {}: {error}",
                        lock_path.display()
                    ));
                }
                return Ok(CheckpointWriterLock {
                    path: lock_path,
                    token,
                    evidence,
                });
            }
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {
                let metadata = fs::symlink_metadata(&lock_path).map_err(|inspect_error| {
                    format!(
                        "checkpoint writer lock is contended and could not be inspected {}: {inspect_error}",
                        lock_path.display()
                    )
                })?;
                if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
                    return Err(format!(
                        "checkpoint writer lock is not a regular file: {}",
                        lock_path.display()
                    ));
                }
                let existing_evidence =
                    read_small_regular_file(&lock_path, MAX_CHECKPOINT_LOCK_TOKEN_BYTES)
                        .ok()
                        .and_then(|bytes| {
                            serde_json::from_slice::<CheckpointWriterLockEvidence>(&bytes).ok()
                        });
                let stale = policy.stale_after.is_some_and(|maximum_age| {
                    writer_lock_stale_age(existing_evidence.as_ref(), &metadata)
                        .is_some_and(|age| age >= maximum_age)
                });
                let owner_alive = existing_evidence
                    .as_ref()
                    .and_then(CheckpointWriterLockEvidence::owner_is_alive);
                let has_process_identity = existing_evidence.as_ref().is_some_and(|evidence| {
                    evidence.boot_id.is_some() || evidence.process_start_ticks.is_some()
                });
                let reclaimable = match owner_alive {
                    Some(true) => false,
                    Some(false) => true,
                    None => !has_process_identity,
                };
                if stale && reclaimable {
                    fs::remove_file(&lock_path).map_err(|remove_error| {
                        format!(
                            "failed to remove stale checkpoint writer lock {}: {remove_error}",
                            lock_path.display()
                        )
                    })?;
                    continue;
                }
                if attempt + 1 < policy.attempts && !policy.retry_delay.is_zero() {
                    std::thread::sleep(policy.retry_delay);
                }
            }
            Err(error) => {
                return Err(format!(
                    "failed to create checkpoint writer lock {}: {error}",
                    lock_path.display()
                ));
            }
        }
    }
    Err(format!(
        "checkpoint writer lock remained contended after {} attempts: {}",
        policy.attempts,
        lock_path.display()
    ))
}

/// Run a complete checkpoint transaction while holding its writer lease.
pub fn with_checkpoint_writer_lock<T, F>(
    path: impl AsRef<Path>,
    policy: CheckpointWriterLockPolicy,
    operation: F,
) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String>,
{
    let _guard = acquire_checkpoint_writer_lock(path, policy)?;
    operation()
}

/// Deterministic previous-generation path used by recoverable checkpoint APIs.
pub fn checkpoint_previous_path(path: impl AsRef<Path>) -> Result<PathBuf, String> {
    let path = path.as_ref();
    let parent = checkpoint_parent(path);
    let file_name = path
        .file_name()
        .ok_or_else(|| "checkpoint path must name a file".to_string())?
        .to_string_lossy();
    Ok(parent.join(format!(".{file_name}.previous")))
}

/// Deterministic path for a bounded previous generation.
///
/// Generation 1 preserves the historical `.previous` name. Higher generations
/// use `.previous-N` suffixes.
pub fn checkpoint_generation_path(
    path: impl AsRef<Path>,
    generation: usize,
) -> Result<PathBuf, String> {
    if generation == 0 || generation > MAX_CHECKPOINT_PREVIOUS_GENERATIONS {
        return Err(format!(
            "checkpoint generation must be in 1..={MAX_CHECKPOINT_PREVIOUS_GENERATIONS}, got {generation}"
        ));
    }
    if generation == 1 {
        return checkpoint_previous_path(path);
    }
    let path = path.as_ref();
    let parent = checkpoint_parent(path);
    let file_name = path
        .file_name()
        .ok_or_else(|| "checkpoint path must name a file".to_string())?
        .to_string_lossy();
    Ok(parent.join(format!(".{file_name}.previous-{generation}")))
}

/// Remove bounded, stale same-destination temporary files left by interrupted writes.
///
/// Only regular files whose names use this crate's exact destination-specific
/// prefix are eligible. Symlinks, directories, future-dated files, unrelated
/// names, and files younger than `minimum_age` are left untouched.
pub fn cleanup_checkpoint_temps(
    path: impl AsRef<Path>,
    minimum_age: Duration,
    max_remove: usize,
) -> Result<usize, String> {
    if max_remove == 0 {
        return Ok(0);
    }
    let path = path.as_ref();
    let parent = checkpoint_parent(path);
    if !parent.is_dir() {
        return Err(format!(
            "checkpoint parent directory does not exist: {}",
            parent.display()
        ));
    }
    let prefix = checkpoint_temp_prefix(path)?;
    let now = SystemTime::now();
    let entries = fs::read_dir(parent).map_err(|error| {
        format!(
            "failed to scan checkpoint directory {}: {error}",
            parent.display()
        )
    })?;
    let mut removed = 0usize;
    for entry in entries {
        if removed >= max_remove {
            break;
        }
        let entry = entry.map_err(|error| {
            format!(
                "failed to inspect checkpoint directory {}: {error}",
                parent.display()
            )
        })?;
        if !entry.file_name().to_string_lossy().starts_with(&prefix) {
            continue;
        }
        let metadata = fs::symlink_metadata(entry.path()).map_err(|error| {
            format!(
                "failed to stat checkpoint temporary file {}: {error}",
                entry.path().display()
            )
        })?;
        if !metadata.file_type().is_file() {
            continue;
        }
        let modified = match metadata.modified() {
            Ok(modified) => modified,
            Err(_) => continue,
        };
        let age = match now.duration_since(modified) {
            Ok(age) => age,
            Err(_) => continue,
        };
        if age < minimum_age {
            continue;
        }
        fs::remove_file(entry.path()).map_err(|error| {
            format!(
                "failed to remove stale checkpoint temporary file {}: {error}",
                entry.path().display()
            )
        })?;
        removed += 1;
    }
    Ok(removed)
}

fn validate_existing_checkpoint_destination(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            let file_type = metadata.file_type();
            if file_type.is_symlink() {
                return Err(format!(
                    "checkpoint destination must not be a symlink: {}",
                    path.display()
                ));
            }
            if !file_type.is_file() {
                return Err(format!(
                    "checkpoint destination must be a regular file: {}",
                    path.display()
                ));
            }
            Ok(())
        }
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "failed to inspect checkpoint destination {}: {error}",
            path.display()
        )),
    }
}

fn open_checkpoint_readonly(path: &Path) -> Result<File, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("failed to inspect checkpoint {}: {error}", path.display()))?;
    let file_type = metadata.file_type();
    if file_type.is_symlink() {
        return Err(format!(
            "checkpoint input must not be a symlink: {}",
            path.display()
        ));
    }
    if !file_type.is_file() {
        return Err(format!(
            "checkpoint input must be a regular file: {}",
            path.display()
        ));
    }

    let mut options = OpenOptions::new();
    options.read(true);
    // Linux/NixOS: prevent a symlink swap between the metadata check and open.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    options.custom_flags(0o400000 | 0o2000000); // O_NOFOLLOW | O_CLOEXEC
    let file = options
        .open(path)
        .map_err(|error| format!("failed to open checkpoint {}: {error}", path.display()))?;
    let opened_type = file
        .metadata()
        .map_err(|error| format!("failed to stat checkpoint {}: {error}", path.display()))?
        .file_type();
    if !opened_type.is_file() {
        return Err(format!(
            "opened checkpoint is not a regular file: {}",
            path.display()
        ));
    }
    Ok(file)
}

/// Atomically replace a checkpoint file with already encoded bytes.
///
/// The temporary file is created in the destination directory, flushed, and
/// renamed over the destination. Callers must still use the envelope checksum
/// to detect storage corruption after the rename.
fn write_checkpoint_atomic_report_with_heartbeat(
    path: impl AsRef<Path>,
    encoded: &[u8],
    max_envelope_bytes: usize,
    heartbeat: &mut dyn FnMut() -> Result<(), String>,
) -> Result<CheckpointWriteReport, String> {
    let path = path.as_ref();
    if max_envelope_bytes == 0 {
        return Err("checkpoint file limit must be non-zero".to_string());
    }
    if encoded.len() > max_envelope_bytes {
        return Err(format!(
            "checkpoint envelope exceeds file limit: {} > {max_envelope_bytes} bytes",
            encoded.len()
        ));
    }
    let parent = checkpoint_parent(path);
    if !parent.is_dir() {
        return Err(format!(
            "checkpoint parent directory does not exist: {}",
            parent.display()
        ));
    }
    validate_existing_checkpoint_destination(path)?;

    // Cleanup is deliberately bounded and best-effort. A directory permission
    // issue must not prevent a new unique temporary file from being attempted,
    // but the outcome is returned for operator telemetry.
    let (stale_temps_removed, stale_temp_cleanup_error) = match cleanup_checkpoint_temps(
        path,
        DEFAULT_STALE_CHECKPOINT_TEMP_AGE,
        DEFAULT_STALE_CHECKPOINT_TEMP_LIMIT,
    ) {
        Ok(removed) => (removed, None),
        Err(error) => (0, Some(error)),
    };

    let mut created = None;
    for _ in 0..128 {
        let temporary = temporary_checkpoint_path(path)?;
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&temporary) {
            Ok(file) => {
                created = Some((temporary, file));
                break;
            }
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(format!(
                    "failed to create checkpoint temporary file {}: {error}",
                    temporary.display()
                ));
            }
        }
    }
    let (temporary, mut file) = created.ok_or_else(|| {
        "failed to allocate a unique checkpoint temporary file after 128 attempts".to_string()
    })?;

    let write_result = (|| -> Result<(), String> {
        for chunk in encoded.chunks(CHECKPOINT_WRITE_CHUNK_BYTES) {
            heartbeat()?;
            file.write_all(chunk).map_err(|error| {
                format!(
                    "failed to write checkpoint temporary file {}: {error}",
                    temporary.display()
                )
            })?;
        }
        heartbeat()?;
        file.sync_all().map_err(|error| {
            format!(
                "failed to flush checkpoint temporary file {}: {error}",
                temporary.display()
            )
        })?;
        drop(file);
        heartbeat()?;
        fs::rename(&temporary, path).map_err(|error| {
            format!(
                "failed to atomically replace checkpoint {}: {error}",
                path.display()
            )
        })?;
        heartbeat()?;
        #[cfg(unix)]
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| {
                format!(
                    "checkpoint was replaced but parent directory sync failed for {}: {error}",
                    parent.display()
                )
            })?;
        Ok(())
    })();

    if write_result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    write_result?;
    Ok(CheckpointWriteReport {
        encoded_bytes: encoded.len(),
        stale_temps_removed,
        stale_temp_cleanup_error,
    })
}

/// Atomically replace a checkpoint file with already encoded bytes.
///
/// Unlocked callers use a no-op progress callback. Retention writers use the
/// internal heartbeat-aware path so a multi-megabyte write cannot let an
/// otherwise healthy lease appear abandoned.
pub fn write_checkpoint_atomic_report(
    path: impl AsRef<Path>,
    encoded: &[u8],
    max_envelope_bytes: usize,
) -> Result<CheckpointWriteReport, String> {
    let mut heartbeat = || Ok(());
    write_checkpoint_atomic_report_with_heartbeat(path, encoded, max_envelope_bytes, &mut heartbeat)
}

/// Atomically replace a checkpoint file with already encoded bytes.
pub fn write_checkpoint_atomic(
    path: impl AsRef<Path>,
    encoded: &[u8],
    max_envelope_bytes: usize,
) -> Result<(), String> {
    write_checkpoint_atomic_report(path, encoded, max_envelope_bytes).map(|_| ())
}

/// Read an encoded checkpoint file without allowing unbounded allocation.
pub fn read_checkpoint_bounded(
    path: impl AsRef<Path>,
    max_envelope_bytes: usize,
) -> Result<Vec<u8>, String> {
    let path = path.as_ref();
    if max_envelope_bytes == 0 {
        return Err("checkpoint file limit must be non-zero".to_string());
    }
    let mut file = open_checkpoint_readonly(path)?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("failed to stat checkpoint {}: {error}", path.display()))?;
    if metadata.len() > max_envelope_bytes as u64 {
        return Err(format!(
            "checkpoint file exceeds limit: {} > {max_envelope_bytes} bytes",
            metadata.len()
        ));
    }
    let mut encoded = Vec::with_capacity(metadata.len() as usize);
    Read::by_ref(&mut file)
        .take(max_envelope_bytes.saturating_add(1) as u64)
        .read_to_end(&mut encoded)
        .map_err(|error| format!("failed to read checkpoint {}: {error}", path.display()))?;
    if encoded.len() > max_envelope_bytes {
        return Err(format!(
            "checkpoint file grew beyond limit while reading: {} bytes",
            encoded.len()
        ));
    }
    if encoded.len() as u64 != metadata.len() {
        return Err(format!(
            "checkpoint file changed while reading: initial={} bytes, read={} bytes",
            metadata.len(),
            encoded.len()
        ));
    }
    Ok(encoded)
}

/// Encode and atomically persist a checkpoint state with an operation report.
pub fn save_checkpoint_file_report<T: Serialize>(
    path: impl AsRef<Path>,
    kind: &str,
    payload_schema_version: u32,
    state: &T,
    max_payload_bytes: usize,
) -> Result<CheckpointWriteReport, String> {
    let encoded = encode_checkpoint(kind, payload_schema_version, state, max_payload_bytes)?;
    write_checkpoint_atomic_report(path, &encoded, max_envelope_bytes(max_payload_bytes))
}

/// Encode and atomically persist a checkpoint state.
pub fn save_checkpoint_file<T: Serialize>(
    path: impl AsRef<Path>,
    kind: &str,
    payload_schema_version: u32,
    state: &T,
    max_payload_bytes: usize,
) -> Result<(), String> {
    save_checkpoint_file_report(path, kind, payload_schema_version, state, max_payload_bytes)
        .map(|_| ())
}

/// Read, verify, and decode a checkpoint state with an operation report.
pub fn load_checkpoint_file_report<T: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_kind: &str,
    max_payload_bytes: usize,
) -> Result<(u32, T, CheckpointLoadReport), String> {
    let encoded = read_checkpoint_bounded(path, max_envelope_bytes(max_payload_bytes))?;
    let metadata = inspect_checkpoint(&encoded, max_payload_bytes)?;
    if metadata.kind != expected_kind {
        return Err(format!(
            "checkpoint kind mismatch: saved={}, expected={expected_kind}",
            metadata.kind
        ));
    }
    let (schema, state) = decode_checkpoint(&encoded, expected_kind, max_payload_bytes)?;
    Ok((
        schema,
        state,
        CheckpointLoadReport {
            source: CheckpointRecoverySource::Primary,
            metadata,
            encoded_bytes: encoded.len(),
            previous_generation: None,
            promotion_write: None,
        },
    ))
}

/// Read, verify, and decode a checkpoint state from disk.
pub fn load_checkpoint_file<T: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_kind: &str,
    max_payload_bytes: usize,
) -> Result<(u32, T), String> {
    load_checkpoint_file_report(path, expected_kind, max_payload_bytes)
        .map(|(schema, state, _)| (schema, state))
}

/// Persist a checkpoint while retaining the last verified primary generation.
///
/// The existing primary is copied to the deterministic previous-generation
/// path only when its envelope, kind, checksum, and JSON payload all validate.
/// A corrupt primary therefore cannot overwrite a known-good fallback.
pub fn save_checkpoint_file_recoverable_report<T: Serialize>(
    path: impl AsRef<Path>,
    kind: &str,
    payload_schema_version: u32,
    state: &T,
    max_payload_bytes: usize,
) -> Result<CheckpointSaveReport, String> {
    let path = path.as_ref();
    let encoded = encode_checkpoint(kind, payload_schema_version, state, max_payload_bytes)?;
    let envelope_limit = max_envelope_bytes(max_payload_bytes);
    let previous = checkpoint_previous_path(path)?;
    let mut previous_write = None;

    match fs::symlink_metadata(path) {
        Ok(_) => {
            if let Ok(existing) = read_checkpoint_bounded(path, envelope_limit) {
                let verified =
                    decode_checkpoint::<serde_json::Value>(&existing, kind, max_payload_bytes)
                        .is_ok();
                if verified {
                    previous_write = Some(write_checkpoint_atomic_report(
                        &previous,
                        &existing,
                        envelope_limit,
                    )?);
                }
            }
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!(
                "failed to inspect checkpoint {} before rotation: {error}",
                path.display()
            ));
        }
    }

    let primary_write = write_checkpoint_atomic_report(path, &encoded, envelope_limit)?;
    Ok(CheckpointSaveReport {
        primary_write,
        previous_write,
    })
}

pub fn save_checkpoint_file_recoverable<T: Serialize>(
    path: impl AsRef<Path>,
    kind: &str,
    payload_schema_version: u32,
    state: &T,
    max_payload_bytes: usize,
) -> Result<(), String> {
    save_checkpoint_file_recoverable_report(
        path,
        kind,
        payload_schema_version,
        state,
        max_payload_bytes,
    )
    .map(|_| ())
}

/// Load a primary checkpoint, falling back to its last verified generation.
pub fn load_checkpoint_file_recoverable_report<T: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_kind: &str,
    max_payload_bytes: usize,
) -> Result<(u32, T, CheckpointLoadReport), String> {
    let path = path.as_ref();
    match load_checkpoint_file_report(path, expected_kind, max_payload_bytes) {
        Ok(result) => Ok(result),
        Err(primary_error) => {
            let previous = checkpoint_previous_path(path)?;
            match load_checkpoint_file_report(&previous, expected_kind, max_payload_bytes) {
                Ok((schema, state, mut report)) => {
                    report.source = CheckpointRecoverySource::Previous;
                    report.previous_generation = Some(1);
                    Ok((schema, state, report))
                }
                Err(previous_error) => Err(format!(
                    "primary checkpoint failed: {primary_error}; previous generation failed: {previous_error}"
                )),
            }
        }
    }
}

pub fn load_checkpoint_file_recoverable<T: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_kind: &str,
    max_payload_bytes: usize,
) -> Result<(u32, T, CheckpointRecoverySource), String> {
    load_checkpoint_file_recoverable_report(path, expected_kind, max_payload_bytes)
        .map(|(schema, state, report)| (schema, state, report.source))
}

/// Recover from the previous verified generation and atomically promote it back
/// to the primary path before returning the decoded state.
pub fn load_checkpoint_file_recoverable_promote<T, F>(
    path: impl AsRef<Path>,
    expected_kind: &str,
    max_payload_bytes: usize,
    validate: F,
) -> Result<(u32, T, CheckpointLoadReport), String>
where
    T: DeserializeOwned,
    F: FnOnce(u32, &T) -> Result<(), String>,
{
    let path = path.as_ref();
    match load_checkpoint_file_report(path, expected_kind, max_payload_bytes) {
        Ok((schema, state, report)) => {
            validate(schema, &state)?;
            Ok((schema, state, report))
        }
        Err(primary_error) => {
            let previous = checkpoint_previous_path(path)?;
            let envelope_limit = max_envelope_bytes(max_payload_bytes);
            let encoded = read_checkpoint_bounded(&previous, envelope_limit).map_err(|error| {
                format!(
                    "primary checkpoint failed: {primary_error}; previous generation failed: {error}"
                )
            })?;
            let metadata = inspect_checkpoint(&encoded, max_payload_bytes).map_err(|error| {
                format!(
                    "primary checkpoint failed: {primary_error}; previous generation failed: {error}"
                )
            })?;
            if metadata.kind != expected_kind {
                return Err(format!(
                    "primary checkpoint failed: {primary_error}; previous generation kind mismatch: saved={}, expected={expected_kind}",
                    metadata.kind
                ));
            }
            let (schema, state) = decode_checkpoint(&encoded, expected_kind, max_payload_bytes)
                .map_err(|error| {
                    format!(
                        "primary checkpoint failed: {primary_error}; previous generation failed: {error}"
                    )
                })?;
            validate(schema, &state).map_err(|error| {
                format!(
                    "primary checkpoint failed: {primary_error}; previous generation state validation failed: {error}"
                )
            })?;
            let promotion_write = write_checkpoint_atomic_report(path, &encoded, envelope_limit)?;
            Ok((
                schema,
                state,
                CheckpointLoadReport {
                    source: CheckpointRecoverySource::Previous,
                    metadata,
                    encoded_bytes: encoded.len(),
                    previous_generation: Some(1),
                    promotion_write: Some(promotion_write),
                },
            ))
        }
    }
}

fn verified_checkpoint_bytes(path: &Path, kind: &str, max_payload_bytes: usize) -> Option<Vec<u8>> {
    let envelope_limit = max_envelope_bytes(max_payload_bytes);
    let encoded = read_checkpoint_bounded(path, envelope_limit).ok()?;
    decode_checkpoint::<serde_json::Value>(&encoded, kind, max_payload_bytes).ok()?;
    Some(encoded)
}

/// Inspect the primary and configured previous generations without decoding
/// their concrete payload type. Checksums, kind tags, size bounds, and file
/// admission rules are still enforced for every existing generation.
pub fn inspect_checkpoint_generations(
    path: impl AsRef<Path>,
    expected_kind: &str,
    max_payload_bytes: usize,
    policy: CheckpointRetentionPolicy,
) -> Result<Vec<CheckpointGenerationInspection>, String> {
    policy.validate()?;
    validate_kind(expected_kind)?;
    let path = path.as_ref();
    let mut entries = Vec::with_capacity(policy.previous_generations.saturating_add(1));
    for generation in 0..=policy.previous_generations {
        let (location, generation_path) = if generation == 0 {
            (CheckpointGenerationLocation::Primary, path.to_path_buf())
        } else {
            (
                CheckpointGenerationLocation::Previous(generation),
                checkpoint_generation_path(path, generation)?,
            )
        };
        match fs::symlink_metadata(&generation_path) {
            Err(error) if error.kind() == ErrorKind::NotFound => {
                entries.push(CheckpointGenerationInspection {
                    location,
                    path: generation_path,
                    exists: false,
                    encoded_bytes: None,
                    metadata: None,
                    error: None,
                });
            }
            Err(error) => {
                entries.push(CheckpointGenerationInspection {
                    location,
                    path: generation_path,
                    exists: true,
                    encoded_bytes: None,
                    metadata: None,
                    error: Some(format!("failed to inspect checkpoint generation: {error}")),
                });
            }
            Ok(_) => match read_checkpoint_bounded(
                &generation_path,
                max_envelope_bytes(max_payload_bytes),
            ) {
                Ok(encoded) => match inspect_checkpoint(&encoded, max_payload_bytes) {
                    Ok(metadata) if metadata.kind == expected_kind => {
                        entries.push(CheckpointGenerationInspection {
                            location,
                            path: generation_path,
                            exists: true,
                            encoded_bytes: Some(encoded.len()),
                            metadata: Some(metadata),
                            error: None,
                        });
                    }
                    Ok(metadata) => {
                        entries.push(CheckpointGenerationInspection {
                            location,
                            path: generation_path,
                            exists: true,
                            encoded_bytes: Some(encoded.len()),
                            metadata: None,
                            error: Some(format!(
                                "checkpoint kind mismatch: saved={}, expected={expected_kind}",
                                metadata.kind
                            )),
                        });
                    }
                    Err(error) => {
                        entries.push(CheckpointGenerationInspection {
                            location,
                            path: generation_path,
                            exists: true,
                            encoded_bytes: Some(encoded.len()),
                            metadata: None,
                            error: Some(error),
                        });
                    }
                },
                Err(error) => {
                    entries.push(CheckpointGenerationInspection {
                        location,
                        path: generation_path,
                        exists: true,
                        encoded_bytes: None,
                        metadata: None,
                        error: Some(error),
                    });
                }
            },
        }
    }
    Ok(entries)
}

/// Remove previous-generation files beyond a newly reduced retention bound.
///
/// All candidate paths are preflighted before the first removal. Symlinks,
/// directories, and other special files fail closed rather than being followed
/// or deleted. The primary checkpoint and retained generations are untouched.
pub fn prune_checkpoint_generations(
    path: impl AsRef<Path>,
    keep_previous_generations: usize,
) -> Result<CheckpointPruneReport, String> {
    CheckpointRetentionPolicy {
        previous_generations: keep_previous_generations,
    }
    .validate()?;
    let path = path.as_ref();
    let mut removable = Vec::new();
    let mut absent_generations = Vec::new();

    for generation in
        keep_previous_generations.saturating_add(1)..=MAX_CHECKPOINT_PREVIOUS_GENERATIONS
    {
        let generation_path = checkpoint_generation_path(path, generation)?;
        match fs::symlink_metadata(&generation_path) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
                    return Err(format!(
                        "checkpoint generation {generation} is not a regular non-symlink file: {}",
                        generation_path.display()
                    ));
                }
                removable.push((generation, generation_path));
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {
                absent_generations.push(generation);
            }
            Err(error) => {
                return Err(format!(
                    "failed to inspect checkpoint generation {generation} {}: {error}",
                    generation_path.display()
                ));
            }
        }
    }

    let mut removed_generations = Vec::with_capacity(removable.len());
    for (generation, generation_path) in removable {
        fs::remove_file(&generation_path).map_err(|error| {
            format!(
                "failed to prune checkpoint generation {generation} {}: {error}",
                generation_path.display()
            )
        })?;
        removed_generations.push(generation);
    }

    Ok(CheckpointPruneReport {
        kept_previous_generations: keep_previous_generations,
        removed_generations,
        absent_generations,
    })
}

/// Prune old generations while holding the same writer lease used by retained
/// saves, preventing cooperating writers from rotating files concurrently.
pub fn prune_checkpoint_generations_locked(
    path: impl AsRef<Path>,
    keep_previous_generations: usize,
    lock_policy: CheckpointWriterLockPolicy,
) -> Result<CheckpointPruneReport, String> {
    let path = path.as_ref();
    with_checkpoint_writer_lock(path, lock_policy, || {
        prune_checkpoint_generations(path, keep_previous_generations)
    })
}

/// Write an already encoded checkpoint while rotating verified generations.
///
/// `heartbeat` is called before and after every potentially slow generation
/// read/write boundary. Locked callers use it to refresh their writer lease;
/// unlocked callers provide a no-op closure.
fn write_checkpoint_file_with_retention_report<H>(
    path: &Path,
    kind: &str,
    encoded: &[u8],
    max_payload_bytes: usize,
    policy: CheckpointRetentionPolicy,
    mut heartbeat: H,
) -> Result<CheckpointRetentionSaveReport, String>
where
    H: FnMut() -> Result<(), String>,
{
    policy.validate()?;
    let envelope_limit = max_envelope_bytes(max_payload_bytes);
    let mut generation_writes = Vec::new();

    heartbeat()?;
    for generation in (1..=policy.previous_generations).rev() {
        let source = if generation == 1 {
            path.to_path_buf()
        } else {
            checkpoint_generation_path(path, generation - 1)?
        };
        let destination = checkpoint_generation_path(path, generation)?;
        heartbeat()?;
        if let Some(previous_bytes) = verified_checkpoint_bytes(&source, kind, max_payload_bytes) {
            let report = write_checkpoint_atomic_report_with_heartbeat(
                &destination,
                &previous_bytes,
                envelope_limit,
                &mut heartbeat,
            )?;
            generation_writes.push((generation, report));
        }
        heartbeat()?;
    }
    generation_writes.sort_by_key(|(generation, _)| *generation);
    heartbeat()?;
    let primary_write = write_checkpoint_atomic_report_with_heartbeat(
        path,
        encoded,
        envelope_limit,
        &mut heartbeat,
    )?;
    heartbeat()?;
    Ok(CheckpointRetentionSaveReport {
        primary_write,
        generation_writes,
    })
}

/// Save with a bounded number of verified previous generations.
pub fn save_checkpoint_file_with_retention_report<T: Serialize>(
    path: impl AsRef<Path>,
    kind: &str,
    payload_schema_version: u32,
    state: &T,
    max_payload_bytes: usize,
    policy: CheckpointRetentionPolicy,
) -> Result<CheckpointRetentionSaveReport, String> {
    policy.validate()?;
    let path = path.as_ref();
    let encoded = encode_checkpoint(kind, payload_schema_version, state, max_payload_bytes)?;
    write_checkpoint_file_with_retention_report(
        path,
        kind,
        &encoded,
        max_payload_bytes,
        policy,
        || Ok(()),
    )
}

pub fn save_checkpoint_file_with_retention<T: Serialize>(
    path: impl AsRef<Path>,
    kind: &str,
    payload_schema_version: u32,
    state: &T,
    max_payload_bytes: usize,
    policy: CheckpointRetentionPolicy,
) -> Result<(), String> {
    save_checkpoint_file_with_retention_report(
        path,
        kind,
        payload_schema_version,
        state,
        max_payload_bytes,
        policy,
    )
    .map(|_| ())
}

/// Save a complete retained generation set under one cross-process writer lease.
///
/// Serialization happens before acquiring the filesystem lease. Once the lease
/// is held, every generation boundary refreshes the heartbeat automatically.
pub fn save_checkpoint_file_with_retention_locked_report<T: Serialize>(
    path: impl AsRef<Path>,
    kind: &str,
    payload_schema_version: u32,
    state: &T,
    max_payload_bytes: usize,
    retention: CheckpointRetentionPolicy,
    lock_policy: CheckpointWriterLockPolicy,
) -> Result<CheckpointRetentionSaveReport, String> {
    retention.validate()?;
    let path = path.as_ref();
    let encoded = encode_checkpoint(kind, payload_schema_version, state, max_payload_bytes)?;
    let mut lock = acquire_checkpoint_writer_lock(path, lock_policy)?;
    write_checkpoint_file_with_retention_report(
        path,
        kind,
        &encoded,
        max_payload_bytes,
        retention,
        || lock.refresh(),
    )
}

/// Load the primary checkpoint or the first valid bounded previous generation.
pub fn load_checkpoint_file_with_retention_report<T: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_kind: &str,
    max_payload_bytes: usize,
    policy: CheckpointRetentionPolicy,
) -> Result<(u32, T, CheckpointLoadReport), String> {
    policy.validate()?;
    let path = path.as_ref();
    let mut failures = Vec::new();
    match load_checkpoint_file_report(path, expected_kind, max_payload_bytes) {
        Ok(result) => return Ok(result),
        Err(error) => failures.push(format!("primary: {error}")),
    }
    for generation in 1..=policy.previous_generations {
        let generation_path = checkpoint_generation_path(path, generation)?;
        match load_checkpoint_file_report(&generation_path, expected_kind, max_payload_bytes) {
            Ok((schema, state, mut report)) => {
                report.source = CheckpointRecoverySource::Previous;
                report.previous_generation = Some(generation);
                return Ok((schema, state, report));
            }
            Err(error) => failures.push(format!("previous generation {generation}: {error}")),
        }
    }
    Err(format!(
        "no valid checkpoint generation found: {}",
        failures.join("; ")
    ))
}

pub fn load_checkpoint_file_with_retention<T: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_kind: &str,
    max_payload_bytes: usize,
    policy: CheckpointRetentionPolicy,
) -> Result<(u32, T, CheckpointRecoverySource, Option<usize>), String> {
    load_checkpoint_file_with_retention_report(path, expected_kind, max_payload_bytes, policy)
        .map(|(schema, state, report)| (schema, state, report.source, report.previous_generation))
}

/// Load the first generation that passes both envelope validation and a
/// caller-supplied semantic validator.
///
/// This is intentionally different from validating only after recovery: a
/// semantically incompatible primary or recent fallback is skipped, allowing
/// older retained history to remain useful.
pub fn load_checkpoint_file_with_retention_validated_report<T, F>(
    path: impl AsRef<Path>,
    expected_kind: &str,
    max_payload_bytes: usize,
    policy: CheckpointRetentionPolicy,
    validate: F,
) -> Result<(u32, T, CheckpointLoadReport), String>
where
    T: DeserializeOwned,
    F: FnMut(u32, &T) -> Result<(), String>,
{
    load_checkpoint_file_with_retention_audited(
        path,
        expected_kind,
        max_payload_bytes,
        policy,
        validate,
    )
    .map(|(schema, state, report)| (schema, state, report.selected))
}

/// Recover the first semantically usable generation while preserving a
/// structured audit trail for every newer generation considered.
pub fn load_checkpoint_file_with_retention_audited<T, F>(
    path: impl AsRef<Path>,
    expected_kind: &str,
    max_payload_bytes: usize,
    policy: CheckpointRetentionPolicy,
    validate: F,
) -> Result<(u32, T, CheckpointSemanticRecoveryReport), String>
where
    T: DeserializeOwned,
    F: FnMut(u32, &T) -> Result<(), String>,
{
    load_checkpoint_file_with_retention_audited_detailed(
        path,
        expected_kind,
        max_payload_bytes,
        policy,
        validate,
    )
    .map_err(|error| error.to_string())
}

/// Detailed semantic recovery that preserves the full failure audit even when
/// no generation can be selected.
pub fn load_checkpoint_file_with_retention_audited_detailed<T, F>(
    path: impl AsRef<Path>,
    expected_kind: &str,
    max_payload_bytes: usize,
    policy: CheckpointRetentionPolicy,
    mut validate: F,
) -> Result<(u32, T, CheckpointSemanticRecoveryReport), CheckpointSemanticRecoveryFailure>
where
    T: DeserializeOwned,
    F: FnMut(u32, &T) -> Result<(), String>,
{
    if let Err(error) = policy.validate() {
        return Err(CheckpointSemanticRecoveryFailure {
            attempts: Vec::new(),
            setup_error: Some(error),
        });
    }
    let path = path.as_ref();
    let mut attempts = Vec::new();

    for generation in 0..=policy.previous_generations {
        let (location, generation_path) = if generation == 0 {
            (CheckpointGenerationLocation::Primary, path.to_path_buf())
        } else {
            let generation_path =
                checkpoint_generation_path(path, generation).map_err(|error| {
                    CheckpointSemanticRecoveryFailure {
                        attempts: attempts.clone(),
                        setup_error: Some(error),
                    }
                })?;
            (
                CheckpointGenerationLocation::Previous(generation),
                generation_path,
            )
        };
        match load_checkpoint_file_report::<T>(&generation_path, expected_kind, max_payload_bytes) {
            Ok((schema, state, mut report)) => match validate(schema, &state) {
                Ok(()) => {
                    if generation > 0 {
                        report.source = CheckpointRecoverySource::Previous;
                        report.previous_generation = Some(generation);
                    }
                    attempts.push(CheckpointRecoveryAttempt {
                        location,
                        path: generation_path,
                        outcome: CheckpointRecoveryAttemptOutcome::Selected,
                    });
                    return Ok((
                        schema,
                        state,
                        CheckpointSemanticRecoveryReport {
                            selected: report,
                            attempts,
                        },
                    ));
                }
                Err(error) => attempts.push(CheckpointRecoveryAttempt {
                    location,
                    path: generation_path,
                    outcome: CheckpointRecoveryAttemptOutcome::SemanticFailure(error),
                }),
            },
            Err(error) => attempts.push(CheckpointRecoveryAttempt {
                location,
                path: generation_path,
                outcome: CheckpointRecoveryAttemptOutcome::StructuralFailure(error),
            }),
        }
    }

    Err(CheckpointSemanticRecoveryFailure {
        attempts,
        setup_error: None,
    })
}

/// Convenience wrapper returning the selected recovery source and generation.
pub fn load_checkpoint_file_with_retention_validated<T, F>(
    path: impl AsRef<Path>,
    expected_kind: &str,
    max_payload_bytes: usize,
    policy: CheckpointRetentionPolicy,
    validate: F,
) -> Result<(u32, T, CheckpointRecoverySource, Option<usize>), String>
where
    T: DeserializeOwned,
    F: FnMut(u32, &T) -> Result<(), String>,
{
    load_checkpoint_file_with_retention_validated_report(
        path,
        expected_kind,
        max_payload_bytes,
        policy,
        validate,
    )
    .map(|(schema, state, report)| (schema, state, report.source, report.previous_generation))
}

/// Serialize a state into a bounded, checksummed compact envelope.
pub fn encode_checkpoint<T: Serialize>(
    kind: &str,
    payload_schema_version: u32,
    state: &T,
    max_payload_bytes: usize,
) -> Result<Vec<u8>, String> {
    validate_kind(kind)?;
    if payload_schema_version == 0 {
        return Err("checkpoint payload schema must be non-zero".to_string());
    }
    if max_payload_bytes == 0 {
        return Err("checkpoint payload limit must be non-zero".to_string());
    }
    let mut payload_writer = BoundedPayloadWriter::new(max_payload_bytes);
    if let Err(error) = serde_json::to_writer(&mut payload_writer, state) {
        if payload_writer.overflowed {
            return Err(format!(
                "checkpoint payload exceeds limit during serialization: > {max_payload_bytes} bytes"
            ));
        }
        return Err(format!("failed to serialize checkpoint payload: {error}"));
    }
    let payload = payload_writer.into_inner();

    let kind_len = u16::try_from(kind.len())
        .map_err(|_| "checkpoint kind length exceeds binary envelope range".to_string())?;
    let payload_len = u64::try_from(payload.len())
        .map_err(|_| "checkpoint payload length exceeds binary envelope range".to_string())?;
    let checksum = envelope_checksum(
        CHECKPOINT_ENVELOPE_SCHEMA_VERSION,
        kind,
        payload_schema_version,
        &payload,
    );
    let capacity = CHECKPOINT_BINARY_HEADER_BYTES
        .checked_add(kind.len())
        .and_then(|size| size.checked_add(payload.len()))
        .ok_or_else(|| "checkpoint envelope size overflow".to_string())?;
    if capacity > max_compact_envelope_bytes(max_payload_bytes) {
        return Err("compact checkpoint envelope exceeds its derived size bound".to_string());
    }

    let mut encoded = Vec::with_capacity(capacity);
    encoded.extend_from_slice(CHECKPOINT_BINARY_MAGIC);
    encoded.extend_from_slice(&CHECKPOINT_ENVELOPE_SCHEMA_VERSION.to_le_bytes());
    encoded.extend_from_slice(&kind_len.to_le_bytes());
    encoded.extend_from_slice(&payload_schema_version.to_le_bytes());
    encoded.extend_from_slice(&payload_len.to_le_bytes());
    encoded.extend_from_slice(&checksum.to_le_bytes());
    encoded.extend_from_slice(kind.as_bytes());
    encoded.extend_from_slice(&payload);
    Ok(encoded)
}

fn read_u16(encoded: &[u8], offset: &mut usize) -> Result<u16, String> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| "checkpoint header offset overflow".to_string())?;
    let bytes: [u8; 2] = encoded
        .get(*offset..end)
        .ok_or_else(|| "truncated checkpoint binary header".to_string())?
        .try_into()
        .map_err(|_| "invalid checkpoint u16 field".to_string())?;
    *offset = end;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(encoded: &[u8], offset: &mut usize) -> Result<u32, String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| "checkpoint header offset overflow".to_string())?;
    let bytes: [u8; 4] = encoded
        .get(*offset..end)
        .ok_or_else(|| "truncated checkpoint binary header".to_string())?
        .try_into()
        .map_err(|_| "invalid checkpoint u32 field".to_string())?;
    *offset = end;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(encoded: &[u8], offset: &mut usize) -> Result<u64, String> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| "checkpoint header offset overflow".to_string())?;
    let bytes: [u8; 8] = encoded
        .get(*offset..end)
        .ok_or_else(|| "truncated checkpoint binary header".to_string())?
        .try_into()
        .map_err(|_| "invalid checkpoint u64 field".to_string())?;
    *offset = end;
    Ok(u64::from_le_bytes(bytes))
}

/// Wrap a normal checkpoint envelope in a caller-supplied authentication tag.
///
/// The crate intentionally does not choose a cryptographic primitive. Production
/// callers can provide HMAC, a detached signature, or a hardware-backed signing
/// operation. The callback receives the complete compact checkpoint envelope so
/// kind, schema, payload length, checksum, and payload are all authenticated.
pub fn encode_authenticated_checkpoint<T, S>(
    kind: &str,
    payload_schema_version: u32,
    state: &T,
    max_payload_bytes: usize,
    max_tag_bytes: usize,
    sign: S,
) -> Result<Vec<u8>, String>
where
    T: Serialize,
    S: FnOnce(&[u8]) -> Result<Vec<u8>, String>,
{
    if max_tag_bytes == 0 || max_tag_bytes > DEFAULT_MAX_CHECKPOINT_AUTH_TAG_BYTES {
        return Err(format!(
            "checkpoint authentication tag limit must be in 1..={DEFAULT_MAX_CHECKPOINT_AUTH_TAG_BYTES}, got {max_tag_bytes}"
        ));
    }
    let inner = encode_checkpoint(kind, payload_schema_version, state, max_payload_bytes)?;
    let tag = sign(&inner).map_err(|error| format!("checkpoint signer failed: {error}"))?;
    if tag.is_empty() || tag.len() > max_tag_bytes {
        return Err(format!(
            "checkpoint authentication tag length must be in 1..={max_tag_bytes}, got {}",
            tag.len()
        ));
    }
    let inner_len = u64::try_from(inner.len())
        .map_err(|_| "authenticated checkpoint length exceeds binary range".to_string())?;
    let tag_len = u32::try_from(tag.len())
        .map_err(|_| "authentication tag length exceeds binary range".to_string())?;
    let capacity = CHECKPOINT_AUTH_HEADER_BYTES
        .checked_add(inner.len())
        .and_then(|size| size.checked_add(tag.len()))
        .ok_or_else(|| "authenticated checkpoint size overflow".to_string())?;
    let max_capacity = max_envelope_bytes(max_payload_bytes)
        .checked_add(CHECKPOINT_AUTH_HEADER_BYTES)
        .and_then(|size| size.checked_add(max_tag_bytes))
        .ok_or_else(|| "authenticated checkpoint limit overflow".to_string())?;
    if capacity > max_capacity {
        return Err("authenticated checkpoint exceeds derived size bound".to_string());
    }

    let mut encoded = Vec::with_capacity(capacity);
    encoded.extend_from_slice(CHECKPOINT_AUTH_MAGIC);
    encoded.extend_from_slice(&inner_len.to_le_bytes());
    encoded.extend_from_slice(&tag_len.to_le_bytes());
    encoded.extend_from_slice(&inner);
    encoded.extend_from_slice(&tag);
    Ok(encoded)
}

fn authenticated_envelope_limit(
    max_payload_bytes: usize,
    max_tag_bytes: usize,
) -> Result<usize, String> {
    max_envelope_bytes(max_payload_bytes)
        .checked_add(CHECKPOINT_AUTH_HEADER_BYTES)
        .and_then(|size| size.checked_add(max_tag_bytes))
        .ok_or_else(|| "authenticated checkpoint limit overflow".to_string())
}

fn authenticated_checkpoint_parts(
    encoded: &[u8],
    max_payload_bytes: usize,
    max_tag_bytes: usize,
) -> Result<(&[u8], &[u8]), String> {
    if max_tag_bytes == 0 || max_tag_bytes > DEFAULT_MAX_CHECKPOINT_AUTH_TAG_BYTES {
        return Err(format!(
            "checkpoint authentication tag limit must be in 1..={DEFAULT_MAX_CHECKPOINT_AUTH_TAG_BYTES}, got {max_tag_bytes}"
        ));
    }
    if !encoded.starts_with(CHECKPOINT_AUTH_MAGIC) {
        return Err("checkpoint is not an authenticated wrapper".to_string());
    }
    if encoded.len() < CHECKPOINT_AUTH_HEADER_BYTES {
        return Err("truncated authenticated checkpoint header".to_string());
    }
    if encoded.len() > authenticated_envelope_limit(max_payload_bytes, max_tag_bytes)? {
        return Err(format!(
            "authenticated checkpoint exceeds size bound: {} bytes",
            encoded.len()
        ));
    }
    let mut offset = CHECKPOINT_AUTH_MAGIC.len();
    let inner_len = usize::try_from(read_u64(encoded, &mut offset)?)
        .map_err(|_| "authenticated checkpoint inner length exceeds this platform".to_string())?;
    let tag_len = read_u32(encoded, &mut offset)? as usize;
    if tag_len == 0 || tag_len > max_tag_bytes {
        return Err(format!(
            "authenticated checkpoint tag length is invalid: {tag_len}"
        ));
    }
    let inner_end = offset
        .checked_add(inner_len)
        .ok_or_else(|| "authenticated checkpoint inner offset overflow".to_string())?;
    let tag_end = inner_end
        .checked_add(tag_len)
        .ok_or_else(|| "authenticated checkpoint tag offset overflow".to_string())?;
    if tag_end != encoded.len() {
        return Err(format!(
            "authenticated checkpoint length mismatch: declared={tag_end}, actual={}",
            encoded.len()
        ));
    }
    if inner_len > max_envelope_bytes(max_payload_bytes) {
        return Err(format!(
            "authenticated inner checkpoint exceeds size bound: {inner_len} bytes"
        ));
    }
    let inner = encoded
        .get(offset..inner_end)
        .ok_or_else(|| "truncated authenticated inner checkpoint".to_string())?;
    let tag = encoded
        .get(inner_end..tag_end)
        .ok_or_else(|| "truncated checkpoint authentication tag".to_string())?;
    Ok((inner, tag))
}

/// Verify a caller-authenticated wrapper before deserializing its inner state.
///
/// Authentication is checked before `decode_checkpoint`, so an invalid tag
/// cannot trigger concrete state deserialization.
pub fn decode_authenticated_checkpoint<T, V>(
    encoded: &[u8],
    expected_kind: &str,
    max_payload_bytes: usize,
    max_tag_bytes: usize,
    verify: V,
) -> Result<(u32, T), String>
where
    T: DeserializeOwned,
    V: FnOnce(&[u8], &[u8]) -> Result<bool, String>,
{
    let (inner, tag) = authenticated_checkpoint_parts(encoded, max_payload_bytes, max_tag_bytes)?;
    let valid =
        verify(inner, tag).map_err(|error| format!("checkpoint authenticator failed: {error}"))?;
    if !valid {
        return Err("checkpoint authentication failed".to_string());
    }
    decode_checkpoint(inner, expected_kind, max_payload_bytes)
}

/// Atomically persist a caller-authenticated checkpoint wrapper.
pub fn save_authenticated_checkpoint_file<T, S>(
    path: impl AsRef<Path>,
    kind: &str,
    payload_schema_version: u32,
    state: &T,
    max_payload_bytes: usize,
    max_tag_bytes: usize,
    sign: S,
) -> Result<CheckpointWriteReport, String>
where
    T: Serialize,
    S: FnOnce(&[u8]) -> Result<Vec<u8>, String>,
{
    let encoded = encode_authenticated_checkpoint(
        kind,
        payload_schema_version,
        state,
        max_payload_bytes,
        max_tag_bytes,
        sign,
    )?;
    let limit = authenticated_envelope_limit(max_payload_bytes, max_tag_bytes)?;
    write_checkpoint_atomic_report(path, &encoded, limit)
}

/// Read and authenticate a checkpoint file before concrete deserialization.
pub fn load_authenticated_checkpoint_file<T, V>(
    path: impl AsRef<Path>,
    expected_kind: &str,
    max_payload_bytes: usize,
    max_tag_bytes: usize,
    verify: V,
) -> Result<(u32, T), String>
where
    T: DeserializeOwned,
    V: FnOnce(&[u8], &[u8]) -> Result<bool, String>,
{
    let limit = authenticated_envelope_limit(max_payload_bytes, max_tag_bytes)?;
    let encoded = read_checkpoint_bounded(path, limit)?;
    decode_authenticated_checkpoint(
        &encoded,
        expected_kind,
        max_payload_bytes,
        max_tag_bytes,
        verify,
    )
}

/// Read, authenticate, inspect, and decode a checkpoint with an operation report.
pub fn load_authenticated_checkpoint_file_report<T, V>(
    path: impl AsRef<Path>,
    expected_kind: &str,
    max_payload_bytes: usize,
    max_tag_bytes: usize,
    verify: V,
) -> Result<(u32, T, CheckpointLoadReport), String>
where
    T: DeserializeOwned,
    V: FnOnce(&[u8], &[u8]) -> Result<bool, String>,
{
    let limit = authenticated_envelope_limit(max_payload_bytes, max_tag_bytes)?;
    let encoded = read_checkpoint_bounded(path, limit)?;
    let (inner, tag) = authenticated_checkpoint_parts(&encoded, max_payload_bytes, max_tag_bytes)?;
    let valid =
        verify(inner, tag).map_err(|error| format!("checkpoint authenticator failed: {error}"))?;
    if !valid {
        return Err("checkpoint authentication failed".to_string());
    }
    let metadata = inspect_checkpoint(inner, max_payload_bytes)?;
    if metadata.kind != expected_kind {
        return Err(format!(
            "checkpoint kind mismatch: saved={}, expected={expected_kind}",
            metadata.kind
        ));
    }
    let (schema, state) = decode_checkpoint(inner, expected_kind, max_payload_bytes)?;
    Ok((
        schema,
        state,
        CheckpointLoadReport {
            source: CheckpointRecoverySource::Primary,
            metadata,
            encoded_bytes: encoded.len(),
            previous_generation: None,
            promotion_write: None,
        },
    ))
}

fn verified_authenticated_checkpoint_bytes<V>(
    path: &Path,
    expected_kind: &str,
    max_payload_bytes: usize,
    max_tag_bytes: usize,
    verify: &V,
) -> Option<Vec<u8>>
where
    V: Fn(&[u8], &[u8]) -> Result<bool, String>,
{
    let limit = authenticated_envelope_limit(max_payload_bytes, max_tag_bytes).ok()?;
    let encoded = read_checkpoint_bounded(path, limit).ok()?;
    let (inner, tag) =
        authenticated_checkpoint_parts(&encoded, max_payload_bytes, max_tag_bytes).ok()?;
    if !verify(inner, tag).ok()? {
        return None;
    }
    decode_checkpoint::<serde_json::Value>(inner, expected_kind, max_payload_bytes).ok()?;
    Some(encoded)
}

// Genuinely needs this many parameters for its authenticated+versioned+retention-report API
// surface (path, kind, encoded payload, two size limits, retention policy, verify closure,
// heartbeat closure) -- bundling them into a struct would just move the same complexity, not
// remove it, for a private helper with a small number of call sites.
#[allow(clippy::too_many_arguments)]
fn write_authenticated_checkpoint_file_with_retention_report<V, H>(
    path: &Path,
    kind: &str,
    encoded: &[u8],
    max_payload_bytes: usize,
    max_tag_bytes: usize,
    policy: CheckpointRetentionPolicy,
    verify: &V,
    mut heartbeat: H,
) -> Result<CheckpointRetentionSaveReport, String>
where
    V: Fn(&[u8], &[u8]) -> Result<bool, String>,
    H: FnMut() -> Result<(), String>,
{
    policy.validate()?;
    let envelope_limit = authenticated_envelope_limit(max_payload_bytes, max_tag_bytes)?;
    let mut generation_writes = Vec::new();

    heartbeat()?;
    for generation in (1..=policy.previous_generations).rev() {
        let source = if generation == 1 {
            path.to_path_buf()
        } else {
            checkpoint_generation_path(path, generation - 1)?
        };
        let destination = checkpoint_generation_path(path, generation)?;
        heartbeat()?;
        if let Some(previous_bytes) = verified_authenticated_checkpoint_bytes(
            &source,
            kind,
            max_payload_bytes,
            max_tag_bytes,
            verify,
        ) {
            let report = write_checkpoint_atomic_report_with_heartbeat(
                &destination,
                &previous_bytes,
                envelope_limit,
                &mut heartbeat,
            )?;
            generation_writes.push((generation, report));
        }
        heartbeat()?;
    }
    generation_writes.sort_by_key(|(generation, _)| *generation);
    let primary_write = write_checkpoint_atomic_report_with_heartbeat(
        path,
        encoded,
        envelope_limit,
        &mut heartbeat,
    )?;
    Ok(CheckpointRetentionSaveReport {
        primary_write,
        generation_writes,
    })
}

/// Save a caller-authenticated checkpoint with bounded verified history.
///
/// Existing generations are rotated only after their authentication tag,
/// integrity envelope, kind, and JSON payload all validate.
// Genuinely needs this many parameters -- same rationale as
// write_authenticated_checkpoint_file_with_retention_report above.
#[allow(clippy::too_many_arguments)]
pub fn save_authenticated_checkpoint_file_with_retention_report<T, S, V>(
    path: impl AsRef<Path>,
    kind: &str,
    payload_schema_version: u32,
    state: &T,
    max_payload_bytes: usize,
    max_tag_bytes: usize,
    policy: CheckpointRetentionPolicy,
    sign: S,
    verify: V,
) -> Result<CheckpointRetentionSaveReport, String>
where
    T: Serialize,
    S: FnOnce(&[u8]) -> Result<Vec<u8>, String>,
    V: Fn(&[u8], &[u8]) -> Result<bool, String>,
{
    policy.validate()?;
    let encoded = encode_authenticated_checkpoint(
        kind,
        payload_schema_version,
        state,
        max_payload_bytes,
        max_tag_bytes,
        sign,
    )?;
    write_authenticated_checkpoint_file_with_retention_report(
        path.as_ref(),
        kind,
        &encoded,
        max_payload_bytes,
        max_tag_bytes,
        policy,
        &verify,
        || Ok(()),
    )
}

/// Save authenticated retained history under one cross-process writer lease.
// Genuinely needs this many parameters -- adds a writer-lock policy on top of the same
// authenticated+versioned+retention-report surface as the sibling functions above.
#[allow(clippy::too_many_arguments)]
pub fn save_authenticated_checkpoint_file_with_retention_locked_report<T, S, V>(
    path: impl AsRef<Path>,
    kind: &str,
    payload_schema_version: u32,
    state: &T,
    max_payload_bytes: usize,
    max_tag_bytes: usize,
    retention: CheckpointRetentionPolicy,
    lock_policy: CheckpointWriterLockPolicy,
    sign: S,
    verify: V,
) -> Result<CheckpointRetentionSaveReport, String>
where
    T: Serialize,
    S: FnOnce(&[u8]) -> Result<Vec<u8>, String>,
    V: Fn(&[u8], &[u8]) -> Result<bool, String>,
{
    retention.validate()?;
    let path = path.as_ref();
    let encoded = encode_authenticated_checkpoint(
        kind,
        payload_schema_version,
        state,
        max_payload_bytes,
        max_tag_bytes,
        sign,
    )?;
    let mut lock = acquire_checkpoint_writer_lock(path, lock_policy)?;
    write_authenticated_checkpoint_file_with_retention_report(
        path,
        kind,
        &encoded,
        max_payload_bytes,
        max_tag_bytes,
        retention,
        &verify,
        || lock.refresh(),
    )
}

/// Recover the newest authenticated generation that also passes semantic validation.
pub fn load_authenticated_checkpoint_file_with_retention_audited_detailed<T, V, F>(
    path: impl AsRef<Path>,
    expected_kind: &str,
    max_payload_bytes: usize,
    max_tag_bytes: usize,
    policy: CheckpointRetentionPolicy,
    verify: V,
    mut validate: F,
) -> Result<(u32, T, CheckpointSemanticRecoveryReport), CheckpointSemanticRecoveryFailure>
where
    T: DeserializeOwned,
    V: Fn(&[u8], &[u8]) -> Result<bool, String>,
    F: FnMut(u32, &T) -> Result<(), String>,
{
    if let Err(error) = policy.validate() {
        return Err(CheckpointSemanticRecoveryFailure {
            attempts: Vec::new(),
            setup_error: Some(error),
        });
    }
    let path = path.as_ref();
    let mut attempts = Vec::new();
    for generation in 0..=policy.previous_generations {
        let (location, generation_path) = if generation == 0 {
            (CheckpointGenerationLocation::Primary, path.to_path_buf())
        } else {
            let generation_path =
                checkpoint_generation_path(path, generation).map_err(|error| {
                    CheckpointSemanticRecoveryFailure {
                        attempts: attempts.clone(),
                        setup_error: Some(error),
                    }
                })?;
            (
                CheckpointGenerationLocation::Previous(generation),
                generation_path,
            )
        };
        match load_authenticated_checkpoint_file_report::<T, _>(
            &generation_path,
            expected_kind,
            max_payload_bytes,
            max_tag_bytes,
            |inner, tag| verify(inner, tag),
        ) {
            Ok((schema, state, mut report)) => match validate(schema, &state) {
                Ok(()) => {
                    if generation > 0 {
                        report.source = CheckpointRecoverySource::Previous;
                        report.previous_generation = Some(generation);
                    }
                    attempts.push(CheckpointRecoveryAttempt {
                        location,
                        path: generation_path,
                        outcome: CheckpointRecoveryAttemptOutcome::Selected,
                    });
                    return Ok((
                        schema,
                        state,
                        CheckpointSemanticRecoveryReport {
                            selected: report,
                            attempts,
                        },
                    ));
                }
                Err(error) => attempts.push(CheckpointRecoveryAttempt {
                    location,
                    path: generation_path,
                    outcome: CheckpointRecoveryAttemptOutcome::SemanticFailure(error),
                }),
            },
            Err(error) => attempts.push(CheckpointRecoveryAttempt {
                location,
                path: generation_path,
                outcome: CheckpointRecoveryAttemptOutcome::StructuralFailure(error),
            }),
        }
    }
    Err(CheckpointSemanticRecoveryFailure {
        attempts,
        setup_error: None,
    })
}

fn inspect_compact_checkpoint(
    encoded: &[u8],
    max_payload_bytes: usize,
) -> Result<CheckpointMetadata, String> {
    if encoded.len() < CHECKPOINT_BINARY_HEADER_BYTES {
        return Err("truncated compact checkpoint envelope".to_string());
    }
    let mut offset = CHECKPOINT_BINARY_MAGIC.len();
    let schema_version = read_u32(encoded, &mut offset)?;
    if schema_version != CHECKPOINT_ENVELOPE_SCHEMA_VERSION {
        return Err(format!(
            "unsupported checkpoint envelope schema: saved={schema_version}, supported={CHECKPOINT_ENVELOPE_SCHEMA_VERSION}"
        ));
    }
    let kind_len = read_u16(encoded, &mut offset)? as usize;
    let payload_schema_version = read_u32(encoded, &mut offset)?;
    let payload_len_u64 = read_u64(encoded, &mut offset)?;
    let saved_checksum = read_u64(encoded, &mut offset)?;
    if kind_len == 0 || kind_len > 64 {
        return Err(format!(
            "invalid compact checkpoint kind length: {kind_len}"
        ));
    }
    if payload_schema_version == 0 {
        return Err("checkpoint payload schema must be non-zero".to_string());
    }
    let payload_len = usize::try_from(payload_len_u64)
        .map_err(|_| "checkpoint payload length exceeds this platform".to_string())?;
    if payload_len > max_payload_bytes {
        return Err(format!(
            "checkpoint payload exceeds limit: {payload_len} > {max_payload_bytes} bytes"
        ));
    }
    let kind_end = offset
        .checked_add(kind_len)
        .ok_or_else(|| "checkpoint kind offset overflow".to_string())?;
    let payload_end = kind_end
        .checked_add(payload_len)
        .ok_or_else(|| "checkpoint payload offset overflow".to_string())?;
    if payload_end != encoded.len() {
        return Err(format!(
            "compact checkpoint length mismatch: declared={payload_end}, actual={}",
            encoded.len()
        ));
    }
    let kind = std::str::from_utf8(
        encoded
            .get(offset..kind_end)
            .ok_or_else(|| "truncated checkpoint kind".to_string())?,
    )
    .map_err(|error| format!("checkpoint kind is not UTF-8: {error}"))?;
    validate_kind(kind)?;
    let payload = encoded
        .get(kind_end..payload_end)
        .ok_or_else(|| "truncated checkpoint payload".to_string())?;
    let expected_checksum =
        envelope_checksum(schema_version, kind, payload_schema_version, payload);
    if saved_checksum != expected_checksum {
        return Err("checkpoint checksum mismatch".to_string());
    }
    Ok(CheckpointMetadata {
        envelope_schema_version: schema_version,
        kind: kind.to_string(),
        payload_schema_version,
        payload_len,
        checksum_fnv1a64: saved_checksum,
        compact: true,
    })
}

fn inspect_legacy_json_checkpoint(
    encoded: &[u8],
    max_payload_bytes: usize,
) -> Result<CheckpointMetadata, String> {
    let envelope: CheckpointEnvelope = serde_json::from_slice(encoded)
        .map_err(|error| format!("failed to deserialize checkpoint envelope: {error}"))?;
    if envelope.schema_version != LEGACY_JSON_ENVELOPE_SCHEMA_VERSION {
        return Err(format!(
            "unsupported legacy checkpoint envelope schema: saved={}, supported={LEGACY_JSON_ENVELOPE_SCHEMA_VERSION}",
            envelope.schema_version
        ));
    }
    validate_kind(&envelope.kind)?;
    if envelope.payload_schema_version == 0 {
        return Err("checkpoint payload schema must be non-zero".to_string());
    }
    if envelope.payload_len != envelope.payload.len() {
        return Err(format!(
            "checkpoint payload length mismatch: declared={}, actual={}",
            envelope.payload_len,
            envelope.payload.len()
        ));
    }
    if envelope.payload.len() > max_payload_bytes {
        return Err(format!(
            "checkpoint payload exceeds limit: {} > {max_payload_bytes} bytes",
            envelope.payload.len()
        ));
    }
    let expected_checksum = envelope_checksum(
        envelope.schema_version,
        &envelope.kind,
        envelope.payload_schema_version,
        &envelope.payload,
    );
    if envelope.checksum_fnv1a64 != expected_checksum {
        return Err("checkpoint checksum mismatch".to_string());
    }
    Ok(CheckpointMetadata {
        envelope_schema_version: envelope.schema_version,
        kind: envelope.kind,
        payload_schema_version: envelope.payload_schema_version,
        payload_len: envelope.payload_len,
        checksum_fnv1a64: envelope.checksum_fnv1a64,
        compact: false,
    })
}

/// Inspect and integrity-check a bounded checkpoint envelope without decoding
/// its concrete state payload.
pub fn inspect_checkpoint(
    encoded: &[u8],
    max_payload_bytes: usize,
) -> Result<CheckpointMetadata, String> {
    if max_payload_bytes == 0 {
        return Err("checkpoint payload limit must be non-zero".to_string());
    }
    if encoded.len() > max_envelope_bytes(max_payload_bytes) {
        return Err(format!(
            "checkpoint envelope exceeds size bound: {} bytes",
            encoded.len()
        ));
    }
    if encoded.starts_with(CHECKPOINT_BINARY_MAGIC) {
        inspect_compact_checkpoint(encoded, max_payload_bytes)
    } else {
        inspect_legacy_json_checkpoint(encoded, max_payload_bytes)
    }
}

/// Inspect a checkpoint file with the same bounded, no-symlink admission used
/// by concrete checkpoint loading.
pub fn inspect_checkpoint_file(
    path: impl AsRef<Path>,
    max_payload_bytes: usize,
) -> Result<CheckpointMetadata, String> {
    let encoded = read_checkpoint_bounded(path, max_envelope_bytes(max_payload_bytes))?;
    inspect_checkpoint(&encoded, max_payload_bytes)
}

fn decode_compact_checkpoint<T: DeserializeOwned>(
    encoded: &[u8],
    expected_kind: &str,
    max_payload_bytes: usize,
) -> Result<(u32, T), String> {
    if encoded.len() < CHECKPOINT_BINARY_HEADER_BYTES {
        return Err("truncated compact checkpoint envelope".to_string());
    }
    let mut offset = CHECKPOINT_BINARY_MAGIC.len();
    let schema_version = read_u32(encoded, &mut offset)?;
    if schema_version != CHECKPOINT_ENVELOPE_SCHEMA_VERSION {
        return Err(format!(
            "unsupported checkpoint envelope schema: saved={schema_version}, supported={CHECKPOINT_ENVELOPE_SCHEMA_VERSION}"
        ));
    }
    let kind_len = read_u16(encoded, &mut offset)? as usize;
    let payload_schema_version = read_u32(encoded, &mut offset)?;
    let payload_len_u64 = read_u64(encoded, &mut offset)?;
    let saved_checksum = read_u64(encoded, &mut offset)?;
    if kind_len == 0 || kind_len > 64 {
        return Err(format!(
            "invalid compact checkpoint kind length: {kind_len}"
        ));
    }
    if payload_schema_version == 0 {
        return Err("checkpoint payload schema must be non-zero".to_string());
    }
    let payload_len = usize::try_from(payload_len_u64)
        .map_err(|_| "checkpoint payload length exceeds this platform".to_string())?;
    if payload_len > max_payload_bytes {
        return Err(format!(
            "checkpoint payload exceeds limit: {payload_len} > {max_payload_bytes} bytes"
        ));
    }
    let kind_end = offset
        .checked_add(kind_len)
        .ok_or_else(|| "checkpoint kind offset overflow".to_string())?;
    let payload_end = kind_end
        .checked_add(payload_len)
        .ok_or_else(|| "checkpoint payload offset overflow".to_string())?;
    if payload_end != encoded.len() {
        return Err(format!(
            "compact checkpoint length mismatch: declared={payload_end}, actual={}",
            encoded.len()
        ));
    }
    let kind = std::str::from_utf8(
        encoded
            .get(offset..kind_end)
            .ok_or_else(|| "truncated checkpoint kind".to_string())?,
    )
    .map_err(|error| format!("checkpoint kind is not UTF-8: {error}"))?;
    validate_kind(kind)?;
    if kind != expected_kind {
        return Err(format!(
            "checkpoint kind mismatch: saved={kind}, expected={expected_kind}"
        ));
    }
    let payload = encoded
        .get(kind_end..payload_end)
        .ok_or_else(|| "truncated checkpoint payload".to_string())?;
    let expected_checksum =
        envelope_checksum(schema_version, kind, payload_schema_version, payload);
    if saved_checksum != expected_checksum {
        return Err("checkpoint checksum mismatch".to_string());
    }
    let state = serde_json::from_slice(payload)
        .map_err(|error| format!("failed to deserialize checkpoint payload: {error}"))?;
    Ok((payload_schema_version, state))
}

fn decode_legacy_json_checkpoint<T: DeserializeOwned>(
    encoded: &[u8],
    expected_kind: &str,
    max_payload_bytes: usize,
) -> Result<(u32, T), String> {
    let envelope: CheckpointEnvelope = serde_json::from_slice(encoded)
        .map_err(|error| format!("failed to deserialize checkpoint envelope: {error}"))?;
    if envelope.schema_version != LEGACY_JSON_ENVELOPE_SCHEMA_VERSION {
        return Err(format!(
            "unsupported legacy checkpoint envelope schema: saved={}, supported={LEGACY_JSON_ENVELOPE_SCHEMA_VERSION}",
            envelope.schema_version
        ));
    }
    if envelope.kind != expected_kind {
        return Err(format!(
            "checkpoint kind mismatch: saved={}, expected={expected_kind}",
            envelope.kind
        ));
    }
    if envelope.payload_schema_version == 0 {
        return Err("checkpoint payload schema must be non-zero".to_string());
    }
    if envelope.payload_len != envelope.payload.len() {
        return Err(format!(
            "checkpoint payload length mismatch: declared={}, actual={}",
            envelope.payload_len,
            envelope.payload.len()
        ));
    }
    if envelope.payload.len() > max_payload_bytes {
        return Err(format!(
            "checkpoint payload exceeds limit: {} > {max_payload_bytes} bytes",
            envelope.payload.len()
        ));
    }
    let expected_checksum = envelope_checksum(
        envelope.schema_version,
        &envelope.kind,
        envelope.payload_schema_version,
        &envelope.payload,
    );
    if envelope.checksum_fnv1a64 != expected_checksum {
        return Err("checkpoint checksum mismatch".to_string());
    }
    let state = serde_json::from_slice(&envelope.payload)
        .map_err(|error| format!("failed to deserialize checkpoint payload: {error}"))?;
    Ok((envelope.payload_schema_version, state))
}

/// Validate and deserialize a bounded checkpoint envelope.
///
/// Compact schema-2 envelopes are preferred. Legacy schema-1 JSON envelopes
/// remain accepted so existing recovery files do not become unreadable.
pub fn decode_checkpoint<T: DeserializeOwned>(
    encoded: &[u8],
    expected_kind: &str,
    max_payload_bytes: usize,
) -> Result<(u32, T), String> {
    validate_kind(expected_kind)?;
    if max_payload_bytes == 0 {
        return Err("checkpoint payload limit must be non-zero".to_string());
    }
    if encoded.len() > max_envelope_bytes(max_payload_bytes) {
        return Err(format!(
            "checkpoint envelope exceeds size bound: {} bytes",
            encoded.len()
        ));
    }
    if encoded.starts_with(CHECKPOINT_BINARY_MAGIC) {
        decode_compact_checkpoint(encoded, expected_kind, max_payload_bytes)
    } else {
        decode_legacy_json_checkpoint(encoded, expected_kind, max_payload_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns a fresh, collision-free scratch directory path under the system temp dir,
    /// following this module's existing `symthaea-vision-checkpoint-*` temp-path convention.
    fn unique_test_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "symthaea-vision-checkpoint-{label}-{}-{}",
            std::process::id(),
            CHECKPOINT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    struct ExampleState {
        schema_version: u32,
        value: String,
    }

    #[test]
    fn bounded_payload_writer_rejects_growth_before_exceeding_limit() {
        let mut writer = BoundedPayloadWriter::new(4);
        assert_eq!(writer.write(b"1234").unwrap(), 4);
        assert!(writer.write(b"5").is_err());
        assert!(writer.overflowed);
        assert_eq!(writer.bytes, b"1234");
    }

    #[test]
    fn checkpoint_serialization_is_bounded_during_encoding() {
        let state = ExampleState {
            schema_version: 1,
            value: "x".repeat(16 * 1024),
        };
        let error = encode_checkpoint("example", 1, &state, 128).unwrap_err();
        assert!(error.contains("exceeds limit during serialization"));
    }

    #[test]
    fn checkpoint_metadata_is_available_without_state_decoding() {
        let state = ExampleState {
            schema_version: 7,
            value: "metadata".to_string(),
        };
        let encoded = encode_checkpoint("example", 7, &state, 4096).unwrap();
        let metadata = inspect_checkpoint(&encoded, 4096).unwrap();
        assert_eq!(
            metadata.envelope_schema_version,
            CHECKPOINT_ENVELOPE_SCHEMA_VERSION
        );
        assert_eq!(metadata.kind, "example");
        assert_eq!(metadata.payload_schema_version, 7);
        assert!(metadata.payload_len > 0);
        assert!(metadata.compact);
    }

    #[test]
    fn checkpoint_metadata_inspection_rejects_corruption() {
        let state = ExampleState {
            schema_version: 1,
            value: "integrity".to_string(),
        };
        let mut encoded = encode_checkpoint("example", 1, &state, 4096).unwrap();
        *encoded.last_mut().unwrap() ^= 0x01;
        assert!(inspect_checkpoint(&encoded, 4096).is_err());
    }

    #[test]
    fn checkpoint_envelope_roundtrips_and_checks_kind() {
        let state = ExampleState {
            schema_version: 3,
            value: "vision".to_string(),
        };
        let encoded = encode_checkpoint("example", 3, &state, 4096).unwrap();
        let (schema, decoded): (u32, ExampleState) =
            decode_checkpoint(&encoded, "example", 4096).unwrap();
        assert_eq!(schema, 3);
        assert_eq!(decoded, state);
        assert!(decode_checkpoint::<ExampleState>(&encoded, "other", 4096).is_err());
    }

    #[test]
    fn checkpoint_envelope_rejects_corruption_and_bounds() {
        let state = ExampleState {
            schema_version: 1,
            value: "bounded".repeat(16),
        };
        assert!(encode_checkpoint("example", 1, &state, 8).is_err());
        let mut corrupted = encode_checkpoint("example", 1, &state, 4096).unwrap();
        let last = corrupted.len() - 1;
        corrupted[last] ^= 0x01;
        assert!(decode_checkpoint::<ExampleState>(&corrupted, "example", 4096).is_err());
    }

    #[test]
    fn compact_envelope_avoids_json_byte_array_amplification() {
        let state = ExampleState {
            schema_version: 2,
            value: "compact".repeat(256),
        };
        let payload = serde_json::to_vec(&state).unwrap();
        let encoded = encode_checkpoint("example", 2, &state, 4096).unwrap();
        assert!(encoded.starts_with(CHECKPOINT_BINARY_MAGIC));
        assert!(encoded.len() <= payload.len() + CHECKPOINT_BINARY_HEADER_BYTES + 64);
    }

    #[test]
    fn legacy_json_envelopes_remain_readable() {
        let state = ExampleState {
            schema_version: 1,
            value: "legacy".to_string(),
        };
        let payload = serde_json::to_vec(&state).unwrap();
        let envelope = CheckpointEnvelope {
            schema_version: LEGACY_JSON_ENVELOPE_SCHEMA_VERSION,
            kind: "example".to_string(),
            payload_schema_version: 1,
            payload_len: payload.len(),
            checksum_fnv1a64: envelope_checksum(
                LEGACY_JSON_ENVELOPE_SCHEMA_VERSION,
                "example",
                1,
                &payload,
            ),
            payload,
        };
        let encoded = serde_json::to_vec(&envelope).unwrap();
        let (_, decoded): (u32, ExampleState) =
            decode_checkpoint(&encoded, "example", 4096).unwrap();
        assert_eq!(decoded, state);
    }

    #[test]
    fn checkpoint_file_roundtrips_and_replaces_atomically() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-vision-checkpoint-{}-{}.json",
            std::process::id(),
            CHECKPOINT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let first = ExampleState {
            schema_version: 1,
            value: "first".to_string(),
        };
        save_checkpoint_file(&path, "example", 1, &first, 4096).unwrap();
        let (_, decoded): (u32, ExampleState) =
            load_checkpoint_file(&path, "example", 4096).unwrap();
        assert_eq!(decoded, first);

        let second = ExampleState {
            schema_version: 2,
            value: "second".to_string(),
        };
        save_checkpoint_file(&path, "example", 2, &second, 4096).unwrap();
        let (schema, decoded): (u32, ExampleState) =
            load_checkpoint_file(&path, "example", 4096).unwrap();
        assert_eq!(schema, 2);
        assert_eq!(decoded, second);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn checkpoint_file_size_rejection_preserves_existing_bytes() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-vision-checkpoint-bound-{}-{}.bin",
            std::process::id(),
            CHECKPOINT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, b"old").unwrap();
        assert!(write_checkpoint_atomic(&path, b"too-large", 3).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"old");
        assert!(read_checkpoint_bounded(&path, 2).is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn checkpoint_file_accepts_a_relative_destination() {
        let name = format!(
            "symthaea-relative-checkpoint-{}-{}.json",
            std::process::id(),
            CHECKPOINT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let state = ExampleState {
            schema_version: 1,
            value: "relative".to_string(),
        };
        save_checkpoint_file(&name, "example", 1, &state, 4096).unwrap();
        let (_, decoded): (u32, ExampleState) =
            load_checkpoint_file(&name, "example", 4096).unwrap();
        assert_eq!(decoded, state);
        let _ = std::fs::remove_file(name);
    }

    #[test]
    fn stale_checkpoint_temp_cleanup_is_destination_scoped_and_bounded() {
        let directory = std::env::temp_dir().join(format!(
            "symthaea-checkpoint-cleanup-{}-{}",
            std::process::id(),
            CHECKPOINT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let destination = directory.join("vision.chk");
        let matching_a = directory.join(".vision.chk.tmp-111-1");
        let matching_b = directory.join(".vision.chk.tmp-222-2");
        let unrelated = directory.join(".other.chk.tmp-111-1");
        std::fs::write(&matching_a, b"a").unwrap();
        std::fs::write(&matching_b, b"b").unwrap();
        std::fs::write(&unrelated, b"keep").unwrap();

        assert_eq!(
            cleanup_checkpoint_temps(&destination, Duration::ZERO, 1).unwrap(),
            1
        );
        assert_eq!(
            cleanup_checkpoint_temps(&destination, Duration::ZERO, 8).unwrap(),
            1
        );
        assert!(unrelated.exists());
        assert_eq!(
            cleanup_checkpoint_temps(&destination, Duration::ZERO, 8).unwrap(),
            0
        );

        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn checkpoint_io_rejects_non_regular_files() {
        let directory = std::env::temp_dir().join(format!(
            "symthaea-checkpoint-file-type-{}-{}",
            std::process::id(),
            CHECKPOINT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        assert!(read_checkpoint_bounded(&directory, 4096).is_err());
        assert!(write_checkpoint_atomic(&directory, b"data", 4096).is_err());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[cfg(unix)]
    #[test]
    fn checkpoint_io_rejects_symlink_substitution() {
        use std::os::unix::fs::symlink;

        let directory = std::env::temp_dir().join(format!(
            "symthaea-checkpoint-symlink-{}-{}",
            std::process::id(),
            CHECKPOINT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let target = directory.join("target.chk");
        let link = directory.join("link.chk");
        std::fs::write(&target, b"original").unwrap();
        symlink(&target, &link).unwrap();

        assert!(read_checkpoint_bounded(&link, 4096).is_err());
        assert!(write_checkpoint_atomic(&link, b"replacement", 4096).is_err());
        assert_eq!(std::fs::read(&target).unwrap(), b"original");

        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn bounded_retention_recovers_older_verified_generation() {
        let directory = unique_test_path("retention");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let policy = CheckpointRetentionPolicy {
            previous_generations: 2,
        };
        let states: Vec<ExampleState> = (1..=3)
            .map(|schema_version| ExampleState {
                schema_version,
                value: format!("generation-{schema_version}"),
            })
            .collect();
        for state in &states {
            save_checkpoint_file_with_retention(
                &path,
                "example",
                state.schema_version,
                state,
                4096,
                policy,
            )
            .unwrap();
        }

        std::fs::write(&path, b"bad primary").unwrap();
        std::fs::write(
            checkpoint_generation_path(&path, 1).unwrap(),
            b"bad previous",
        )
        .unwrap();
        let (schema, restored, report): (u32, ExampleState, CheckpointLoadReport) =
            load_checkpoint_file_with_retention_report(&path, "example", 4096, policy).unwrap();
        assert_eq!(schema, 1);
        assert_eq!(restored, states[0]);
        assert_eq!(report.source, CheckpointRecoverySource::Previous);
        assert_eq!(report.previous_generation, Some(2));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn corrupt_generation_does_not_overwrite_known_good_older_history() {
        let directory = unique_test_path("retention-corrupt");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let policy = CheckpointRetentionPolicy {
            previous_generations: 2,
        };
        let first = ExampleState {
            schema_version: 1,
            value: "first".to_string(),
        };
        let second = ExampleState {
            schema_version: 2,
            value: "second".to_string(),
        };
        let third = ExampleState {
            schema_version: 3,
            value: "third".to_string(),
        };
        save_checkpoint_file_with_retention(&path, "example", 1, &first, 4096, policy).unwrap();
        save_checkpoint_file_with_retention(&path, "example", 2, &second, 4096, policy).unwrap();
        save_checkpoint_file_with_retention(&path, "example", 3, &third, 4096, policy).unwrap();
        let second_generation = checkpoint_generation_path(&path, 2).unwrap();
        let known_good = std::fs::read(&second_generation).unwrap();
        std::fs::write(checkpoint_generation_path(&path, 1).unwrap(), b"corrupt").unwrap();

        let fourth = ExampleState {
            schema_version: 4,
            value: "fourth".to_string(),
        };
        save_checkpoint_file_with_retention(&path, "example", 4, &fourth, 4096, policy).unwrap();
        assert_eq!(std::fs::read(second_generation).unwrap(), known_good);
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn recovery_promotion_waits_for_payload_validation() {
        let directory = unique_test_path("promote-validation");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let first = ExampleState {
            schema_version: 1,
            value: "first".to_string(),
        };
        let second = ExampleState {
            schema_version: 2,
            value: "second".to_string(),
        };
        save_checkpoint_file_recoverable(&path, "example", 1, &first, 4096).unwrap();
        save_checkpoint_file_recoverable(&path, "example", 2, &second, 4096).unwrap();
        std::fs::write(&path, b"corrupt primary").unwrap();

        let error = load_checkpoint_file_recoverable_promote::<ExampleState, _>(
            &path,
            "example",
            4096,
            |_schema, _state| Err("semantic rejection".to_string()),
        )
        .unwrap_err();
        assert!(error.contains("semantic rejection"));
        assert_eq!(std::fs::read(&path).unwrap(), b"corrupt primary");
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn recovered_previous_generation_can_be_promoted_to_primary() {
        let directory = unique_test_path("promote");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let first = ExampleState {
            schema_version: 1,
            value: "first".to_string(),
        };
        let second = ExampleState {
            schema_version: 2,
            value: "second".to_string(),
        };
        save_checkpoint_file_recoverable(&path, "example", 1, &first, 4096).unwrap();
        save_checkpoint_file_recoverable(&path, "example", 2, &second, 4096).unwrap();
        std::fs::write(&path, b"corrupt").unwrap();

        let (schema, restored, report): (u32, ExampleState, CheckpointLoadReport) =
            load_checkpoint_file_recoverable_promote(
                &path,
                "example",
                4096,
                |schema, state: &ExampleState| {
                    if schema == state.schema_version {
                        Ok(())
                    } else {
                        Err("schema mismatch".to_string())
                    }
                },
            )
            .unwrap();
        assert_eq!(schema, 1);
        assert_eq!(restored, first);
        assert_eq!(report.source, CheckpointRecoverySource::Previous);
        assert!(report.promotion_write.is_some());

        let (primary_schema, primary): (u32, ExampleState) =
            load_checkpoint_file(&path, "example", 4096).unwrap();
        assert_eq!(primary_schema, 1);
        assert_eq!(primary, first);
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn checkpoint_reports_surface_write_and_recovery_evidence() {
        let directory = unique_test_path("reports");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let first = ExampleState {
            schema_version: 1,
            value: "first".to_string(),
        };
        let second = ExampleState {
            schema_version: 2,
            value: "second".to_string(),
        };

        let first_report =
            save_checkpoint_file_recoverable_report(&path, "example", 1, &first, 4096).unwrap();
        assert!(first_report.primary_write.encoded_bytes > 0);
        assert!(first_report.previous_write.is_none());

        let second_report =
            save_checkpoint_file_recoverable_report(&path, "example", 2, &second, 4096).unwrap();
        assert!(second_report.previous_write.is_some());

        std::fs::write(&path, b"corrupt").unwrap();
        let (schema, restored, load_report): (u32, ExampleState, CheckpointLoadReport) =
            load_checkpoint_file_recoverable_report(&path, "example", 4096).unwrap();
        assert_eq!(schema, 1);
        assert_eq!(restored, first);
        assert_eq!(load_report.source, CheckpointRecoverySource::Previous);
        assert_eq!(load_report.metadata.kind, "example");
        assert!(load_report.encoded_bytes > 0);

        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn recoverable_checkpoint_falls_back_to_last_verified_generation() {
        let directory = std::env::temp_dir().join(format!(
            "symthaea-checkpoint-recovery-{}-{}",
            std::process::id(),
            CHECKPOINT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("vision.chk");
        let first = ExampleState {
            schema_version: 1,
            value: "first".to_string(),
        };
        let second = ExampleState {
            schema_version: 2,
            value: "second".to_string(),
        };

        save_checkpoint_file_recoverable(&path, "example", 1, &first, 4096).unwrap();
        save_checkpoint_file_recoverable(&path, "example", 2, &second, 4096).unwrap();
        let (_, current, source): (u32, ExampleState, CheckpointRecoverySource) =
            load_checkpoint_file_recoverable(&path, "example", 4096).unwrap();
        assert_eq!(source, CheckpointRecoverySource::Primary);
        assert_eq!(current, second);

        std::fs::write(&path, b"corrupted primary").unwrap();
        let (schema, recovered, source): (u32, ExampleState, CheckpointRecoverySource) =
            load_checkpoint_file_recoverable(&path, "example", 4096).unwrap();
        assert_eq!(schema, 1);
        assert_eq!(source, CheckpointRecoverySource::Previous);
        assert_eq!(recovered, first);

        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn corrupt_primary_does_not_replace_a_verified_previous_generation() {
        let directory = std::env::temp_dir().join(format!(
            "symthaea-checkpoint-recovery-preserve-{}-{}",
            std::process::id(),
            CHECKPOINT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("vision.chk");
        let first = ExampleState {
            schema_version: 1,
            value: "first".to_string(),
        };
        let second = ExampleState {
            schema_version: 2,
            value: "second".to_string(),
        };
        let third = ExampleState {
            schema_version: 3,
            value: "third".to_string(),
        };

        save_checkpoint_file_recoverable(&path, "example", 1, &first, 4096).unwrap();
        save_checkpoint_file_recoverable(&path, "example", 2, &second, 4096).unwrap();
        std::fs::write(&path, b"bad").unwrap();
        save_checkpoint_file_recoverable(&path, "example", 3, &third, 4096).unwrap();
        std::fs::write(&path, b"bad again").unwrap();

        let (_, recovered, source): (u32, ExampleState, CheckpointRecoverySource) =
            load_checkpoint_file_recoverable(&path, "example", 4096).unwrap();
        assert_eq!(source, CheckpointRecoverySource::Previous);
        assert_eq!(recovered, first);
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn writer_lock_exposes_and_refreshes_structured_heartbeat() {
        let directory = unique_test_path("writer-heartbeat");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let mut lock = acquire_checkpoint_writer_lock(
            &path,
            CheckpointWriterLockPolicy {
                attempts: 1,
                retry_delay: Duration::ZERO,
                stale_after: None,
            },
        )
        .unwrap();
        let before = inspect_checkpoint_writer_lock(&path).unwrap();
        assert_eq!(before.pid, std::process::id());
        lock.refresh().unwrap();
        let after = inspect_checkpoint_writer_lock(&path).unwrap();
        assert_eq!(after.nonce, before.nonce);
        assert!(after.heartbeat_unix_secs >= before.heartbeat_unix_secs);
        drop(lock);
        assert!(!checkpoint_writer_lock_path(&path).unwrap().exists());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn writer_lock_refresh_detects_ownership_replacement() {
        let directory = unique_test_path("writer-heartbeat-replaced");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let mut lock = acquire_checkpoint_writer_lock(
            &path,
            CheckpointWriterLockPolicy {
                attempts: 1,
                retry_delay: Duration::ZERO,
                stale_after: None,
            },
        )
        .unwrap();
        let lock_path = checkpoint_writer_lock_path(&path).unwrap();
        std::fs::write(&lock_path, b"replacement").unwrap();
        assert!(lock.refresh().is_err());
        drop(lock);
        assert_eq!(std::fs::read(&lock_path).unwrap(), b"replacement");
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn checkpoint_writer_lock_rejects_contention_and_releases_by_token() {
        let directory = unique_test_path("writer-lock");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let policy = CheckpointWriterLockPolicy {
            attempts: 1,
            retry_delay: Duration::ZERO,
            stale_after: None,
        };
        let first = acquire_checkpoint_writer_lock(&path, policy).unwrap();
        assert!(acquire_checkpoint_writer_lock(&path, policy).is_err());
        let lock_path = first.path().to_path_buf();
        assert!(lock_path.exists());
        drop(first);
        assert!(!lock_path.exists());
        assert!(acquire_checkpoint_writer_lock(&path, policy).is_ok());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn retained_save_can_be_coordinated_under_one_writer_lock() {
        let directory = unique_test_path("writer-lock-retention");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let state = ExampleState {
            schema_version: 1,
            value: "locked".to_string(),
        };
        let report = save_checkpoint_file_with_retention_locked_report(
            &path,
            "example",
            1,
            &state,
            4096,
            CheckpointRetentionPolicy {
                previous_generations: 2,
            },
            CheckpointWriterLockPolicy {
                attempts: 1,
                retry_delay: Duration::ZERO,
                stale_after: None,
            },
        )
        .unwrap();
        assert!(report.primary_write.encoded_bytes > 0);
        assert!(!checkpoint_writer_lock_path(&path).unwrap().exists());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn generation_inventory_distinguishes_valid_corrupt_and_missing_entries() {
        let directory = unique_test_path("generation-inventory");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let first = ExampleState {
            schema_version: 1,
            value: "first".to_string(),
        };
        let second = ExampleState {
            schema_version: 2,
            value: "second".to_string(),
        };
        let policy = CheckpointRetentionPolicy {
            previous_generations: 3,
        };
        save_checkpoint_file_with_retention(&path, "example", 1, &first, 4096, policy).unwrap();
        save_checkpoint_file_with_retention(&path, "example", 2, &second, 4096, policy).unwrap();
        std::fs::write(checkpoint_generation_path(&path, 2).unwrap(), b"corrupt").unwrap();

        let inventory = inspect_checkpoint_generations(&path, "example", 4096, policy).unwrap();
        assert_eq!(inventory.len(), 4);
        assert!(inventory[0].is_valid());
        assert!(inventory[1].is_valid());
        assert!(inventory[2].exists);
        assert!(!inventory[2].is_valid());
        assert!(!inventory[3].exists);
        assert!(inventory[3].error.is_none());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn generation_pruning_removes_only_files_beyond_the_new_policy() {
        let directory = unique_test_path("generation-prune");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let policy = CheckpointRetentionPolicy {
            previous_generations: 3,
        };
        for version in 1..=4 {
            let state = ExampleState {
                schema_version: version,
                value: format!("state-{version}"),
            };
            save_checkpoint_file_with_retention(&path, "example", version, &state, 4096, policy)
                .unwrap();
        }
        let retained = checkpoint_generation_path(&path, 1).unwrap();
        let retained_bytes = std::fs::read(&retained).unwrap();
        let report = prune_checkpoint_generations(&path, 1).unwrap();
        assert_eq!(report.removed_generations, vec![2, 3]);
        assert_eq!(std::fs::read(retained).unwrap(), retained_bytes);
        assert!(path.exists());
        assert!(!checkpoint_generation_path(&path, 2).unwrap().exists());
        assert!(!checkpoint_generation_path(&path, 3).unwrap().exists());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn audited_semantic_recovery_preserves_attempt_outcomes() {
        let directory = unique_test_path("semantic-audit");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let policy = CheckpointRetentionPolicy {
            previous_generations: 2,
        };
        for schema_version in 1..=3 {
            let state = ExampleState {
                schema_version,
                value: format!("generation-{schema_version}"),
            };
            save_checkpoint_file_with_retention(
                &path,
                "example",
                schema_version,
                &state,
                4096,
                policy,
            )
            .unwrap();
        }
        std::fs::write(&path, b"corrupt primary").unwrap();

        let (_, restored, report) = load_checkpoint_file_with_retention_audited(
            &path,
            "example",
            4096,
            policy,
            |schema, _state: &ExampleState| {
                if schema == 2 {
                    Err("schema two rejected".to_string())
                } else {
                    Ok(())
                }
            },
        )
        .unwrap();

        assert_eq!(restored.schema_version, 1);
        assert_eq!(report.selected.previous_generation, Some(2));
        assert_eq!(report.attempts.len(), 3);
        assert!(matches!(
            &report.attempts[0].outcome,
            CheckpointRecoveryAttemptOutcome::StructuralFailure(_)
        ));
        assert!(matches!(
            &report.attempts[1].outcome,
            CheckpointRecoveryAttemptOutcome::SemanticFailure(_)
        ));
        assert_eq!(
            report.attempts[2].outcome,
            CheckpointRecoveryAttemptOutcome::Selected
        );
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn semantic_recovery_skips_newer_incompatible_generations() {
        let directory = unique_test_path("semantic-retention");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let policy = CheckpointRetentionPolicy {
            previous_generations: 2,
        };
        let states = [
            ExampleState {
                schema_version: 1,
                value: "compatible".to_string(),
            },
            ExampleState {
                schema_version: 2,
                value: "incompatible-middle".to_string(),
            },
            ExampleState {
                schema_version: 3,
                value: "incompatible-newest".to_string(),
            },
        ];
        for state in &states {
            save_checkpoint_file_with_retention(
                &path,
                "example",
                state.schema_version,
                state,
                4096,
                policy,
            )
            .unwrap();
        }

        let (schema, restored, source, generation): (
            u32,
            ExampleState,
            CheckpointRecoverySource,
            Option<usize>,
        ) = load_checkpoint_file_with_retention_validated(
            &path,
            "example",
            4096,
            policy,
            |_schema, state: &ExampleState| {
                if state.value == "compatible" {
                    Ok(())
                } else {
                    Err("unsupported logical configuration".to_string())
                }
            },
        )
        .unwrap();
        assert_eq!(schema, 1);
        assert_eq!(restored, states[0]);
        assert_eq!(source, CheckpointRecoverySource::Previous);
        assert_eq!(generation, Some(2));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn audited_detailed_failure_preserves_all_attempts() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea-vision-semantic-detailed-failure-{}-{}",
            std::process::id(),
            CHECKPOINT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("state.chk");
        save_checkpoint_file_with_retention(
            &path,
            "test-kind",
            1,
            &vec![1u32, 2, 3],
            1024,
            CheckpointRetentionPolicy {
                previous_generations: 1,
            },
        )
        .unwrap();

        let error = load_checkpoint_file_with_retention_audited_detailed::<Vec<u32>, _>(
            &path,
            "test-kind",
            1024,
            CheckpointRetentionPolicy {
                previous_generations: 1,
            },
            |_schema, _state| Err("semantic mismatch".to_string()),
        )
        .unwrap_err();
        assert_eq!(error.attempts.len(), 2);
        assert!(matches!(
            &error.attempts[0].outcome,
            CheckpointRecoveryAttemptOutcome::SemanticFailure(_)
        ));
        assert!(matches!(
            &error.attempts[1].outcome,
            CheckpointRecoveryAttemptOutcome::StructuralFailure(_)
        ));
        assert!(error.to_string().contains("semantic failure"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn audited_detailed_reports_policy_setup_failure() {
        let error = load_checkpoint_file_with_retention_audited_detailed::<Vec<u32>, _>(
            "state.chk",
            "test-kind",
            1024,
            CheckpointRetentionPolicy {
                previous_generations: MAX_CHECKPOINT_PREVIOUS_GENERATIONS + 1,
            },
            |_schema, _state| Ok(()),
        )
        .unwrap_err();
        assert!(error.attempts.is_empty());
        assert!(error.setup_error.is_some());
    }

    #[test]
    fn retained_writer_heartbeats_across_generation_boundaries() {
        let directory = std::env::temp_dir().join(format!(
            "symthaea-vision-retention-heartbeat-{}-{}",
            std::process::id(),
            CHECKPOINT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let encoded = encode_checkpoint("test-kind", 1, &vec![1u32], 1024).unwrap();
        write_checkpoint_atomic(&path, &encoded, max_envelope_bytes(1024)).unwrap();
        let mut heartbeats = 0usize;
        write_checkpoint_file_with_retention_report(
            &path,
            "test-kind",
            &encoded,
            1024,
            CheckpointRetentionPolicy {
                previous_generations: 2,
            },
            || {
                heartbeats += 1;
                Ok(())
            },
        )
        .unwrap();
        assert!(
            heartbeats >= 7,
            "expected boundary heartbeats, got {heartbeats}"
        );
        assert!(checkpoint_generation_path(&path, 1).unwrap().exists());
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn writer_lock_evidence_detects_live_owner_identity() {
        let evidence = CheckpointWriterLockEvidence {
            pid: std::process::id(),
            nonce: 1,
            acquired_unix_secs: unix_time_secs().unwrap(),
            heartbeat_unix_secs: unix_time_secs().unwrap(),
            boot_id: linux_boot_id(),
            process_start_ticks: linux_process_start_ticks(std::process::id()),
        };
        assert_eq!(evidence.owner_is_alive(), Some(true));
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn stale_lock_with_live_process_identity_is_not_stolen() {
        let directory = std::env::temp_dir().join(format!(
            "symthaea-vision-live-writer-lock-{}-{}",
            std::process::id(),
            CHECKPOINT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let lock_path = checkpoint_writer_lock_path(&path).unwrap();
        let evidence = CheckpointWriterLockEvidence {
            pid: std::process::id(),
            nonce: 99,
            acquired_unix_secs: 0,
            heartbeat_unix_secs: 0,
            boot_id: linux_boot_id(),
            process_start_ticks: linux_process_start_ticks(std::process::id()),
        };
        fs::write(&lock_path, encode_writer_lock_evidence(&evidence).unwrap()).unwrap();
        let error = acquire_checkpoint_writer_lock(
            &path,
            CheckpointWriterLockPolicy {
                attempts: 1,
                retry_delay: Duration::ZERO,
                stale_after: Some(Duration::from_secs(1)),
            },
        )
        .unwrap_err();
        assert!(error.contains("remained contended"));
        assert!(lock_path.exists());
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn stale_lock_with_reused_pid_identity_can_be_reclaimed() {
        let directory = std::env::temp_dir().join(format!(
            "symthaea-vision-reused-writer-lock-{}-{}",
            std::process::id(),
            CHECKPOINT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let lock_path = checkpoint_writer_lock_path(&path).unwrap();
        let evidence = CheckpointWriterLockEvidence {
            pid: std::process::id(),
            nonce: 100,
            acquired_unix_secs: 0,
            heartbeat_unix_secs: 0,
            boot_id: linux_boot_id(),
            process_start_ticks: linux_process_start_ticks(std::process::id())
                .map(|ticks| ticks.saturating_add(1)),
        };
        fs::write(&lock_path, encode_writer_lock_evidence(&evidence).unwrap()).unwrap();
        let lock = acquire_checkpoint_writer_lock(
            &path,
            CheckpointWriterLockPolicy {
                attempts: 2,
                retry_delay: Duration::ZERO,
                stale_after: Some(Duration::from_secs(1)),
            },
        )
        .unwrap();
        assert_eq!(lock.evidence().owner_is_alive(), Some(true));
        drop(lock);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn legacy_writer_lock_evidence_defaults_process_identity() {
        let evidence: CheckpointWriterLockEvidence = serde_json::from_str(
            r#"{"pid":1,"nonce":2,"acquired_unix_secs":3,"heartbeat_unix_secs":4}"#,
        )
        .unwrap();
        assert!(evidence.boot_id.is_none());
        assert!(evidence.process_start_ticks.is_none());
    }

    #[test]
    fn atomic_write_refreshes_progress_for_large_envelopes() {
        let directory = std::env::temp_dir().join(format!(
            "symthaea-vision-progress-heartbeat-{}-{}",
            std::process::id(),
            CHECKPOINT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let encoded = vec![7u8; CHECKPOINT_WRITE_CHUNK_BYTES * 2 + 17];
        let mut heartbeats = 0usize;
        let mut heartbeat = || {
            heartbeats += 1;
            Ok(())
        };
        write_checkpoint_atomic_report_with_heartbeat(
            &path,
            &encoded,
            encoded.len(),
            &mut heartbeat,
        )
        .unwrap();
        assert!(heartbeats >= 6, "expected chunk and durability heartbeats");
        assert_eq!(fs::read(&path).unwrap(), encoded);
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn authenticated_checkpoint_verifies_before_deserialization() {
        let state = ExampleState {
            schema_version: 4,
            value: "authenticated".to_string(),
        };
        let key = 0x5a5a_1234_9876_fedcu64;
        let sign = |bytes: &[u8]| Ok((fnv1a64_update(key, bytes)).to_le_bytes().to_vec());
        let encoded =
            encode_authenticated_checkpoint("example", 4, &state, 4096, 64, sign).unwrap();
        let verify = |bytes: &[u8], tag: &[u8]| Ok(tag == fnv1a64_update(key, bytes).to_le_bytes());
        let (schema, restored): (u32, ExampleState) =
            decode_authenticated_checkpoint(&encoded, "example", 4096, 64, verify).unwrap();
        assert_eq!(schema, 4);
        assert_eq!(restored, state);

        let mut tampered = encoded.clone();
        let index = CHECKPOINT_AUTH_HEADER_BYTES + 3;
        tampered[index] ^= 0x01;
        let error = decode_authenticated_checkpoint::<ExampleState, _>(
            &tampered,
            "example",
            4096,
            64,
            |bytes, tag| Ok(tag == fnv1a64_update(key, bytes).to_le_bytes()),
        )
        .unwrap_err();
        assert!(error.contains("authentication failed"));
    }

    #[test]
    fn authenticated_retention_skips_corrupt_primary_and_recovers_history() {
        let directory = std::env::temp_dir().join(format!(
            "symthaea-vision-auth-retention-{}-{}",
            std::process::id(),
            CHECKPOINT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("state.chk");
        let key = 0x5a5a_1234_9876_fedcu64;
        let policy = CheckpointRetentionPolicy {
            previous_generations: 1,
        };
        let first = ExampleState {
            schema_version: 1,
            value: "first".to_string(),
        };
        let second = ExampleState {
            schema_version: 2,
            value: "second".to_string(),
        };
        let signer = |bytes: &[u8]| Ok(fnv1a64_update(key, bytes).to_le_bytes().to_vec());
        let verifier =
            |bytes: &[u8], tag: &[u8]| Ok(tag == fnv1a64_update(key, bytes).to_le_bytes());
        save_authenticated_checkpoint_file_with_retention_report(
            &path, "example", 1, &first, 4096, 64, policy, signer, verifier,
        )
        .unwrap();
        save_authenticated_checkpoint_file_with_retention_report(
            &path,
            "example",
            2,
            &second,
            4096,
            64,
            policy,
            |bytes| Ok(fnv1a64_update(key, bytes).to_le_bytes().to_vec()),
            |bytes, tag| Ok(tag == fnv1a64_update(key, bytes).to_le_bytes()),
        )
        .unwrap();

        let mut primary = fs::read(&path).unwrap();
        let index = CHECKPOINT_AUTH_HEADER_BYTES + 5;
        primary[index] ^= 0x40;
        fs::write(&path, primary).unwrap();
        let (schema, recovered, report) =
            load_authenticated_checkpoint_file_with_retention_audited_detailed::<ExampleState, _, _>(
                &path,
                "example",
                4096,
                64,
                policy,
                |bytes, tag| Ok(tag == fnv1a64_update(key, bytes).to_le_bytes()),
                |schema, state| {
                    if schema == state.schema_version {
                        Ok(())
                    } else {
                        Err("schema mismatch".to_string())
                    }
                },
            )
            .unwrap();
        assert_eq!(schema, 1);
        assert_eq!(recovered, first);
        assert_eq!(report.selected.previous_generation, Some(1));
        assert!(matches!(
            report.attempts[0].outcome,
            CheckpointRecoveryAttemptOutcome::StructuralFailure(_)
        ));
        let _ = fs::remove_dir_all(directory);
    }
}
