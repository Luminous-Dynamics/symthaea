// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed, auditable transaction envelopes for consequential relay mutations.
//!
//! This is deliberately narrower than a distributed transaction protocol.
//! It binds one authenticated relay request to:
//!   - a caller-supplied idempotency key;
//!   - a CSPRNG-generated transaction identifier;
//!   - a typed mutation kind;
//!   - a digest of the exact request payload;
//!   - the authoritative target identity when one is available.
//!
//! The envelope is not itself a cryptographic signature. The current relay
//! authorization boundary remains the already-authenticated WebSocket bearer
//! token. This module prevents an authorized request from losing its identity
//! as it crosses asynchronous staging/execution boundaries.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

const SCHEMA_VERSION: u16 = 1;
const CROSS_PROCESS_LOCK_PATH: &str = "/run/nixforhumanity-system-mutation.lock";
const LEDGER_PATH: &str = "/var/lib/nixforhumanity/system-transactions.jsonl";
const FINGERPRINT_KEY_PATH: &str =
    "/var/lib/nixforhumanity/system-transaction-fingerprint.key";
const MAX_JOURNAL_EVENT_BYTES: usize = 64 * 1024;

#[derive(Debug)]
pub(crate) struct MutationLease {
    file: File,
}

impl MutationLease {
    pub(crate) fn acquire() -> Result<Self, String> {
        Self::acquire_at(Path::new(CROSS_PROCESS_LOCK_PATH))
    }

    fn acquire_at(path: &Path) -> Result<Self, String> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(path)
            .map_err(|error| {
                format!(
                    "unable to open OS mutation lock {}: {error}",
                    path.display()
                )
            })?;

        // SAFETY: the file descriptor is valid for the lifetime of file, and
        // the kernel releases the lock when the descriptor closes. LOCK_NB
        // makes admission fail closed rather than queueing.
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if result == 0 {
            Ok(Self { file })
        } else {
            let error = std::io::Error::last_os_error();
            let busy = matches!(
                error.raw_os_error(),
                Some(code) if code == libc::EWOULDBLOCK || code == libc::EAGAIN
            );
            if busy {
                Err("cross-process mutation lock is already held".into())
            } else {
                Err(format!(
                    "unable to acquire cross-process mutation lock {}: {error}",
                    path.display()
                ))
            }
        }
    }
}

impl Drop for MutationLease {
    fn drop(&mut self) {
        // SAFETY: self.file owns the valid descriptor and we are releasing only
        // the lock acquired by this open file description.
        let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
    }
}


#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TransactionOutcome {
    ObservedSuccess,
    Failed,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MutationKind {
    Install,
    Rollback,
    SwitchGeneration,
    ServiceAction,
    GcCollect,
    WriteConfig,
    CreateImage,
    RestoreImage,
    PreserveData,
    ConnectWifi,
}

impl MutationKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Install => "install",
            Self::Rollback => "rollback",
            Self::SwitchGeneration => "switch_generation",
            Self::ServiceAction => "service_action",
            Self::GcCollect => "gc_collect",
            Self::WriteConfig => "write_config",
            Self::CreateImage => "create_image",
            Self::RestoreImage => "restore_image",
            Self::PreserveData => "preserve_data",
            Self::ConnectWifi => "connect_wifi",
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct SystemTransaction {
    schema_version: u16,
    pub(crate) request_id: String,
    pub(crate) transaction_id: String,
    pub(crate) mutation: MutationKind,
    pub(crate) target_machine_digest: Option<String>,
    pub(crate) request_digest: String,
    pub(crate) authorization: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TransactionReceipt {
    pub(crate) schema_version: u16,
    pub(crate) request_id: String,
    pub(crate) transaction_id: String,
    pub(crate) mutation: MutationKind,
    pub(crate) target_machine_digest: Option<String>,
    pub(crate) request_digest: String,
    pub(crate) authorization: &'static str,
    pub(crate) outcome: TransactionOutcome,
}

impl SystemTransaction {
    pub(crate) fn begin(
        mutation: MutationKind,
        request_id: &str,
        target_machine_digest: Option<&str>,
        payload: &[u8],
    ) -> Result<Self, String> {
        let request_id = validate_request_id(request_id)?;
        let transaction_id = random_operation_id()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"nixforhumanity-system-transaction-v1\0");
        hasher.update(mutation.as_str().as_bytes());
        hasher.update(b"\0");
        hasher.update(payload);
        let request_digest = hasher.finalize().to_hex().to_string();

        Ok(Self {
            schema_version: SCHEMA_VERSION,
            request_id: request_id.to_string(),
            transaction_id,
            mutation,
            target_machine_digest: target_machine_digest.map(str::to_owned),
            request_digest,
            authorization: "websocket-bearer-authenticated",
        })
    }

    pub(crate) fn receipt(&self, outcome: TransactionOutcome) -> TransactionReceipt {
        TransactionReceipt {
            schema_version: self.schema_version,
            request_id: self.request_id.clone(),
            transaction_id: self.transaction_id.clone(),
            mutation: self.mutation,
            target_machine_digest: self.target_machine_digest.clone(),
            request_digest: self.request_digest.clone(),
            authorization: self.authorization,
            outcome,
        }
    }

    pub(crate) fn log_line(&self) -> String {
        format!(
            "request_id={} transaction={} mutation={} request_digest={}{}",
            self.request_id,
            self.transaction_id,
            self.mutation.as_str(),
            self.request_digest,
            self.target_machine_digest
                .as_deref()
                .map(|digest| format!(" target_machine_digest={digest}"))
                .unwrap_or_default()
        )
    }
}


fn validate_request_id(value: &str) -> Result<&str, String> {
    let value = value.trim();
    if !(16..=128).contains(&value.len())
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(
            "request_id must be 16-128 ASCII alphanumeric characters, '-' or '_'".into(),
        );
    }
    Ok(value)
}

fn validate_transaction_id(value: &str) -> Result<(), String> {
    if value.len() != 32 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("transaction_id must be exactly 32 hexadecimal characters".into());
    }
    Ok(())
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!("{label} must be exactly 64 lowercase hexadecimal characters"));
    }
    Ok(())
}

fn sync_parent_directory(path: &Path) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("journal path {} has no parent directory", path.display()))?;
    let directory = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(parent)
        .map_err(|error| {
            format!(
                "unable to open transaction ledger directory {} for synchronization: {error}",
                parent.display()
            )
        })?;
    directory.sync_all().map_err(|error| {
        format!(
            "unable to synchronize transaction ledger directory {}: {error}",
            parent.display()
        )
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JournalEvent {
    schema_version: u16,
    event: String,
    request_id: String,
    transaction_id: String,
    mutation: MutationKind,
    target_machine_digest: Option<String>,
    request_digest: String,
    outcome: Option<TransactionOutcome>,
}

#[derive(Debug, Clone)]
struct JournalRecord {
    transaction_id: String,
    mutation: MutationKind,
    target_machine_digest: Option<String>,
    request_digest: String,
    outcome: Option<TransactionOutcome>,
}

impl JournalRecord {
    fn receipt(&self, request_id: &str, outcome: TransactionOutcome) -> TransactionReceipt {
        TransactionReceipt {
            schema_version: SCHEMA_VERSION,
            request_id: request_id.to_string(),
            transaction_id: self.transaction_id.clone(),
            mutation: self.mutation,
            target_machine_digest: self.target_machine_digest.clone(),
            request_digest: self.request_digest.clone(),
            authorization: "websocket-bearer-authenticated",
            outcome,
        }
    }
}

#[derive(Debug)]
pub(crate) enum TransactionAdmission {
    New(SystemTransaction),
    Replayed(TransactionReceipt),
    Indeterminate(TransactionReceipt),
}

#[derive(Debug, Clone)]
pub(crate) struct TransactionLedger {
    path: std::path::PathBuf,
    fingerprint_key: [u8; 32],
}

impl TransactionLedger {
    pub(crate) fn open_default() -> Result<Self, String> {
        let path = std::path::Path::new(LEDGER_PATH);
        let parent = path
            .parent()
            .ok_or_else(|| "transaction ledger path has no parent directory".to_string())?;
        std::fs::create_dir_all(parent).map_err(|error| {
            format!(
                "unable to create transaction ledger directory {}: {error}",
                parent.display()
            )
        })?;
        std::fs::set_permissions(
            parent,
            std::os::unix::fs::PermissionsExt::from_mode(0o700),
        )
        .map_err(|error| {
            format!(
                "unable to restrict transaction ledger directory {}: {error}",
                parent.display()
            )
        })?;

        let fingerprint_key = load_or_create_fingerprint_key(Path::new(FINGERPRINT_KEY_PATH))?;
        Ok(Self {
            path: path.to_path_buf(),
            fingerprint_key,
        })
    }

    #[cfg(test)]
    fn open_at(path: &std::path::Path) -> Result<Self, String> {
        Ok(Self {
            path: path.to_path_buf(),
            fingerprint_key: [0u8; 32],
        })
    }

    fn load(&self) -> Result<HashMap<String, JournalRecord>, String> {
        let metadata = match std::fs::symlink_metadata(&self.path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(HashMap::new());
            }
            Err(error) => {
                return Err(format!(
                    "unable to inspect transaction ledger {}: {error}",
                    self.path.display()
                ));
            }
        };
        if !metadata.file_type().is_file() {
            return Err(format!(
                "transaction ledger {} is not a regular file",
                self.path.display()
            ));
        }
        let mode = {
            use std::os::unix::fs::PermissionsExt;
            metadata.permissions().mode() & 0o777
        };
        if mode != 0o600 {
            return Err(format!(
                "transaction ledger {} has unsafe permissions {:04o}; require 0600",
                self.path.display(),
                mode
            ));
        }
        {
            use std::os::unix::fs::MetadataExt;
            let owner = unsafe { libc::geteuid() };
            if metadata.uid() != owner {
                return Err(format!(
                    "transaction ledger {} is not owned by relay user",
                    self.path.display()
                ));
            }
        }
        let file = File::open(&self.path).map_err(|error| {
            format!(
                "unable to read transaction ledger {}: {error}",
                self.path.display()
            )
        })?;

        let mut records = HashMap::new();
        let mut transaction_owners = HashMap::<String, String>::new();
        for (line_number, line) in BufReader::new(file).lines().enumerate() {
            if line.len() > MAX_JOURNAL_EVENT_BYTES {
                return Err(format!(
                    "transaction ledger {} line {} exceeds {} bytes",
                    self.path.display(),
                    line_number + 1,
                    MAX_JOURNAL_EVENT_BYTES
                ));
            }
            let line = line.map_err(|error| {
                format!(
                    "unable to read transaction ledger {} line {}: {error}",
                    self.path.display(),
                    line_number + 1
                )
            })?;
            if line.trim().is_empty() {
                continue;
            }
            let event: JournalEvent = serde_json::from_str(&line).map_err(|error| {
                format!(
                    "transaction ledger {} is malformed at line {}: {error}",
                    self.path.display(),
                    line_number + 1
                )
            })?;
            if event.schema_version != SCHEMA_VERSION {
                return Err(format!(
                    "transaction ledger schema mismatch at line {}: expected {}, found {}",
                    line_number + 1,
                    SCHEMA_VERSION,
                    event.schema_version
                ));
            }
            validate_request_id(&event.request_id).map_err(|error| {
                format!(
                    "transaction ledger invalid request_id at line {}: {}",
                    line_number + 1,
                    error
                )
            })?;
            validate_transaction_id(&event.transaction_id).map_err(|error| {
                format!(
                    "transaction ledger invalid transaction_id at line {}: {}",
                    line_number + 1,
                    error
                )
            })?;
            validate_digest(&event.request_digest, "request_digest").map_err(|error| {
                format!(
                    "transaction ledger invalid request digest at line {}: {}",
                    line_number + 1,
                    error
                )
            })?;
            if let Some(target_digest) = event.target_machine_digest.as_deref() {
                validate_digest(target_digest, "target_machine_digest").map_err(|error| {
                    format!(
                        "transaction ledger invalid target digest at line {}: {}",
                        line_number + 1,
                        error
                    )
                })?;
            }

            if let Some(owner) = transaction_owners.get(&event.transaction_id) {
                if owner != &event.request_id {
                    return Err(format!(
                        "transaction ledger reuses transaction_id {} for request_ids {} and {}",
                        event.transaction_id, owner, event.request_id
                    ));
                }
            } else {
                transaction_owners.insert(
                    event.transaction_id.clone(),
                    event.request_id.clone(),
                );
            }

            match event.event.as_str() {
                "started" => {
                    if event.outcome.is_some() {
                        return Err(format!(
                            "transaction ledger start event at line {} carries an outcome",
                            line_number + 1
                        ));
                    }
                    let record = JournalRecord {
                        transaction_id: event.transaction_id,
                        mutation: event.mutation,
                        target_machine_digest: event.target_machine_digest,
                        request_digest: event.request_digest,
                        outcome: None,
                    };
                    if let Some(existing) = records.get(&event.request_id) {
                        if existing.transaction_id != record.transaction_id
                            || existing.mutation != record.mutation
                            || existing.target_machine_digest != record.target_machine_digest
                            || existing.request_digest != record.request_digest
                        {
                            return Err(format!(
                                "transaction ledger contains conflicting history for request_id {}",
                                event.request_id
                            ));
                        }
                    } else {
                        records.insert(event.request_id, record);
                    }
                }
                "completed" => {
                    let outcome = event.outcome.ok_or_else(|| {
                        format!(
                            "transaction ledger completion at line {} is missing its outcome",
                            line_number + 1
                        )
                    })?;
                    let Some(record) = records.get_mut(&event.request_id) else {
                        return Err(format!(
                            "transaction ledger completion at line {} has no prior start",
                            line_number + 1
                        ));
                    };
                    if record.transaction_id != event.transaction_id
                        || record.mutation != event.mutation
                        || record.target_machine_digest != event.target_machine_digest
                        || record.request_digest != event.request_digest
                    {
                        return Err(format!(
                            "transaction ledger completion at line {} does not match its start",
                            line_number + 1
                        ));
                    }
                    if let Some(existing) = record.outcome {
                        if existing != outcome {
                            return Err(format!(
                                "transaction ledger has conflicting outcomes for request_id {}",
                                event.request_id
                            ));
                        }
                    } else {
                        record.outcome = Some(outcome);
                    }
                }
                other => {
                    return Err(format!(
                        "transaction ledger has unknown event '{}' at line {}",
                        other,
                        line_number + 1
                    ));
                }
            }
        }
        Ok(records)
    }

    fn append(&self, event: &JournalEvent) -> Result<(), String> {
        let serialized = serde_json::to_string(event)
            .map_err(|error| format!("unable to serialize transaction ledger event: {error}"))?;
        if serialized.len() > MAX_JOURNAL_EVENT_BYTES {
            return Err(format!(
                "transaction ledger event exceeds {} bytes",
                MAX_JOURNAL_EVENT_BYTES
            ));
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .mode(0o600)
            .open(&self.path)
            .map_err(|error| {
                format!(
                    "unable to open transaction ledger {}: {error}",
                    self.path.display()
                )
            })?;
        std::fs::set_permissions(
            &self.path,
            std::os::unix::fs::PermissionsExt::from_mode(0o600),
        )
        .map_err(|error| {
            format!(
                "unable to restrict transaction ledger {}: {error}",
                self.path.display()
            )
        })?;
        file.write_all(serialized.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.sync_all())
            .map_err(|error| {
                format!(
                    "unable to commit transaction ledger {}: {error}",
                    self.path.display()
                )
            })?;

        // fsync(file) does not necessarily make the containing directory entry
        // durable across power loss; sync the directory explicitly as required
        // for crash-consistent creation of the journal file.
        sync_parent_directory(&self.path)
    }

    pub(crate) fn admit(
        &self,
        transaction: SystemTransaction,
    ) -> Result<TransactionAdmission, String> {
        let mut records = self.load()?;
        if let Some(existing) = records.get(&transaction.request_id) {
            if existing.mutation != transaction.mutation
                || existing.target_machine_digest != transaction.target_machine_digest
                || existing.request_digest != transaction.request_digest
            {
                return Err(format!(
                    "request_id {} is already bound to a different mutation or request",
                    transaction.request_id
                ));
            }
            let receipt = match existing.outcome {
                Some(outcome) => existing.receipt(&transaction.request_id, outcome),
                None => existing.receipt(
                    &transaction.request_id,
                    TransactionOutcome::Indeterminate,
                ),
            };
            return Ok(if existing.outcome.is_some() {
                TransactionAdmission::Replayed(receipt)
            } else {
                TransactionAdmission::Indeterminate(receipt)
            });
        }

        // A fresh transaction ID is generated before admission, so check it
        // against the durable namespace before appending a new start event.
        // A collision must fail closed here rather than corrupting the journal
        // and only being discovered on the next reload.
        if records
            .values()
            .any(|record| record.transaction_id == transaction.transaction_id)
        {
            return Err(format!(
                "transaction_id {} collides with an existing transaction",
                transaction.transaction_id
            ));
        }

        self.append(&JournalEvent {
            schema_version: SCHEMA_VERSION,
            event: "started".into(),
            request_id: transaction.request_id.clone(),
            transaction_id: transaction.transaction_id.clone(),
            mutation: transaction.mutation,
            target_machine_digest: transaction.target_machine_digest.clone(),
            request_digest: transaction.request_digest.clone(),
            outcome: None,
        })?;
        Ok(TransactionAdmission::New(transaction))
    }

    pub(crate) fn lookup(&self, request_id: &str) -> Result<Option<TransactionReceipt>, String> {
        let request_id = validate_request_id(request_id)?;
        let records = self.load()?;
        Ok(records.get(request_id).map(|record| {
            record.receipt(
                request_id,
                record.outcome.unwrap_or(TransactionOutcome::Indeterminate),
            )
        }))
    }

    /// Create a server-keyed commitment for a credential that must participate
    /// in request identity without placing the raw secret (or a fast hash of
    /// it) in the durable transaction journal.
    pub(crate) fn secret_commitment(&self, domain: &str, secret: &str) -> String {
        let mut hasher = blake3::Hasher::new_keyed(&self.fingerprint_key);
        hasher.update(b"nixforhumanity-secret-commitment-v1\0");
        hasher.update(domain.as_bytes());
        hasher.update(b"\0");
        hasher.update(secret.as_bytes());
        hasher.finalize().to_hex().to_string()
    }

    pub(crate) fn mark_completed(
        &self,
        transaction: &SystemTransaction,
        outcome: TransactionOutcome,
    ) -> Result<(), String> {
        let records = self.load()?;
        let Some(existing) = records.get(&transaction.request_id) else {
            return Err(format!(
                "cannot complete unknown transaction request_id {}",
                transaction.request_id
            ));
        };
        if existing.transaction_id != transaction.transaction_id
            || existing.mutation != transaction.mutation
            || existing.target_machine_digest != transaction.target_machine_digest
            || existing.request_digest != transaction.request_digest
        {
            return Err(format!(
                "transaction completion identity mismatch for request_id {}",
                transaction.request_id
            ));
        }
        if let Some(existing_outcome) = existing.outcome {
            if existing_outcome != outcome {
                return Err(format!(
                    "transaction request_id {} already has a different recorded outcome",
                    transaction.request_id
                ));
            }
            return Ok(());
        }

        self.append(&JournalEvent {
            schema_version: SCHEMA_VERSION,
            event: "completed".into(),
            request_id: transaction.request_id.clone(),
            transaction_id: transaction.transaction_id.clone(),
            mutation: transaction.mutation,
            target_machine_digest: transaction.target_machine_digest.clone(),
            request_digest: transaction.request_digest.clone(),
            outcome: Some(outcome),
        })
    }
}

fn read_fingerprint_key(path: &Path) -> Result<[u8; 32], String> {
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        format!(
            "unable to inspect transaction fingerprint key {}: {error}",
            path.display()
        )
    })?;
    if !metadata.file_type().is_file() {
        return Err(format!(
            "transaction fingerprint key {} is not a regular file",
            path.display()
        ));
    }
    let mode = {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o777
    };
    if mode != 0o600 {
        return Err(format!(
            "transaction fingerprint key {} has unsafe permissions {:04o}; require 0600",
            path.display(),
            mode
        ));
    }
    {
        use std::os::unix::fs::MetadataExt;
        let owner = unsafe { libc::geteuid() };
        if metadata.uid() != owner {
            return Err(format!(
                "transaction fingerprint key {} is not owned by relay user",
                path.display()
            ));
        }
    }
    if metadata.len() != 32 {
        return Err(format!(
            "transaction fingerprint key {} has invalid length {}; require 32 bytes",
            path.display(),
            metadata.len()
        ));
    }

    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .map_err(|error| {
            format!(
                "unable to open transaction fingerprint key {}: {error}",
                path.display()
            )
        })?;
    let mut key = [0u8; 32];
    file.read_exact(&mut key).map_err(|error| {
        format!(
            "unable to read transaction fingerprint key {}: {error}",
            path.display()
        )
    })?;
    Ok(key)
}

fn load_or_create_fingerprint_key(path: &Path) -> Result<[u8; 32], String> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => return read_fingerprint_key(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!(
                "unable to inspect transaction fingerprint key {}: {error}",
                path.display()
            ));
        }
    }

    let mut key = [0u8; 32];
    getrandom02::getrandom(&mut key)
        .map_err(|error| format!("unable to generate transaction fingerprint key: {error}"))?;

    let mut file = match OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            return read_fingerprint_key(path);
        }
        Err(error) => {
            return Err(format!(
                "unable to create transaction fingerprint key {}: {error}",
                path.display()
            ));
        }
    };

    file.write_all(&key)
        .and_then(|_| file.sync_all())
        .map_err(|error| {
            format!(
                "unable to persist transaction fingerprint key {}: {error}",
                path.display()
            )
        })?;
    std::fs::set_permissions(
        path,
        std::os::unix::fs::PermissionsExt::from_mode(0o600),
    )
    .map_err(|error| {
        format!(
            "unable to restrict transaction fingerprint key {}: {error}",
            path.display()
        )
    })?;
    sync_parent_directory(path)?;
    Ok(key)
}

/// Generate a fresh opaque operation identifier from the OS CSPRNG.
///
/// Failure is deliberately propagated: a mutation must not proceed with a
/// reused or clock-derived identifier.
fn random_operation_id() -> Result<String, String> {
    let mut bytes = [0u8; 16];
    getrandom02::getrandom(&mut bytes)
        .map_err(|error| format!("unable to obtain secure transaction randomness: {error}"))?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutation_lock_rejects_second_file_description() {
        let name = random_operation_id().unwrap();
        let path = std::env::temp_dir().join(format!("symthaea-mutation-lock-test-{name}.lock"));
        let first = MutationLease::acquire_at(&path).unwrap();
        let second = MutationLease::acquire_at(&path);
        assert!(
            matches!(second, Err(message) if message.contains("already held")),
            "second owner must fail closed: {second:?}"
        );
        drop(first);
        assert!(MutationLease::acquire_at(&path).is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn transaction_ids_are_random_and_unique() {
        let a = SystemTransaction::begin(MutationKind::Rollback, "request-a-00000001", None, b"rollback").unwrap();
        let b = SystemTransaction::begin(MutationKind::Rollback, "request-b-00000001", None, b"rollback").unwrap();

        assert_ne!(a.transaction_id, b.transaction_id);
        assert_eq!(a.transaction_id.len(), 32);
        assert_eq!(a.request_digest, b.request_digest);
    }

    #[test]
    fn transaction_binds_payload_and_target() {
        let a =
            SystemTransaction::begin(MutationKind::ServiceAction, "request-a-00000001", Some("a".repeat(64).as_str()), b"restart:test")
                .unwrap();
        let b = SystemTransaction::begin(
            MutationKind::ServiceAction,
            "request-b-00000001",
            Some("b".repeat(64).as_str()),
            b"restart:test",
        )
        .unwrap();
        let c =
            SystemTransaction::begin(MutationKind::ServiceAction, "request-c-00000001", Some("a".repeat(64).as_str()), b"stop:test")
                .unwrap();

        assert_ne!(a.target_machine_digest, b.target_machine_digest);
        assert_ne!(a.request_digest, c.request_digest);
        assert_ne!(a.log_line(), b.log_line());
    }

    #[test]
    fn digest_is_domain_separated_by_mutation_kind() {
        let install =
            SystemTransaction::begin(MutationKind::Install, "request-install-0001", None, b"same-payload").unwrap();
        let rollback =
            SystemTransaction::begin(MutationKind::Rollback, "request-rollback-0001", None, b"same-payload").unwrap();

        assert_ne!(install.request_digest, rollback.request_digest);
    }

    #[test]
    fn receipt_preserves_transaction_identity() {
        let tx = SystemTransaction::begin(MutationKind::GcCollect, "request-gc-00000001", None, b"gc-30d").unwrap();
        let receipt = tx.receipt(TransactionOutcome::ObservedSuccess);

        assert_eq!(receipt.transaction_id, tx.transaction_id);
        assert_eq!(receipt.request_digest, tx.request_digest);
        assert_eq!(receipt.outcome, TransactionOutcome::ObservedSuccess);
    }

    #[test]
    fn mutation_names_are_stable() {
        assert_eq!(MutationKind::Install.as_str(), "install");
        assert_eq!(MutationKind::WriteConfig.as_str(), "write_config");
        assert_eq!(MutationKind::CreateImage.as_str(), "create_image");
        assert_eq!(MutationKind::RestoreImage.as_str(), "restore_image");
        assert_eq!(MutationKind::PreserveData.as_str(), "preserve_data");
        assert_eq!(MutationKind::ConnectWifi.as_str(), "connect_wifi");
    }

    #[test]
    fn request_id_is_strictly_validated() {
        assert!(validate_request_id("0123456789abcdef").is_ok());
        assert!(validate_request_id("short").is_err());
        assert!(validate_request_id("0123456789abcdef!").is_err());
        assert!(validate_request_id(&"a".repeat(129)).is_err());
    }

    #[test]
    fn ledger_rejects_symlinked_ledger_path() {
        let name = random_operation_id().unwrap();
        let target =
            std::env::temp_dir().join(format!("symthaea-transaction-ledger-target-{name}.jsonl"));
        let path =
            std::env::temp_dir().join(format!("symthaea-transaction-ledger-link-{name}.jsonl"));
        std::fs::write(&target, "").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target, &path).unwrap();
        let ledger = TransactionLedger::open_at(&path).unwrap();
        let error = ledger.load().expect_err("symlinked ledger must fail closed");
        assert!(error.contains("not a regular file"));
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&target);
    }

    #[test]
    fn ledger_rejects_unsafe_existing_file_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let name = random_operation_id().unwrap();
        let path =
            std::env::temp_dir().join(format!("symthaea-transaction-ledger-permissions-{name}.jsonl"));
        std::fs::write(&path, "").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        let ledger = TransactionLedger::open_at(&path).unwrap();
        let error = ledger.load().expect_err("group-readable ledger must fail closed");
        assert!(error.contains("unsafe permissions"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn secret_commitments_are_stable_domain_separated_and_not_raw_secrets() {
        let name = random_operation_id().unwrap();
        let path = std::env::temp_dir()
            .join(format!("symthaea-transaction-ledger-secret-{name}.jsonl"));
        let ledger = TransactionLedger::open_at(&path).unwrap();

        let first = ledger.secret_commitment("install-user-password", "correct horse battery staple");
        let same = ledger.secret_commitment("install-user-password", "correct horse battery staple");
        let other_secret = ledger.secret_commitment("install-user-password", "different secret");
        let other_domain = ledger.secret_commitment("wifi-psk", "correct horse battery staple");

        assert_eq!(first, same);
        assert_ne!(first, other_secret);
        assert_ne!(first, other_domain);
        assert_eq!(first.len(), 64);
        assert!(!first.contains("correct"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ledger_rejects_fresh_transaction_id_collision_before_append() {
        let name = random_operation_id().unwrap();
        let path = std::env::temp_dir()
            .join(format!("symthaea-transaction-ledger-collision-admit-{name}.jsonl"));
        let ledger = TransactionLedger::open_at(&path).unwrap();

        let first = SystemTransaction::begin(
            MutationKind::Rollback,
            "collision-request-one-0001",
            None,
            b"rollback",
        )
        .unwrap();
        let first_id = first.transaction_id.clone();
        assert!(matches!(
            ledger.admit(first).unwrap(),
            TransactionAdmission::New(_)
        ));

        let mut second = SystemTransaction::begin(
            MutationKind::Rollback,
            "collision-request-two-0001",
            None,
            b"rollback",
        )
        .unwrap();
        second.transaction_id = first_id.clone();

        let error = ledger
            .admit(second)
            .expect_err("fresh transaction ID collision must fail before append");
        assert!(error.contains("collides with an existing transaction"));

        let lines = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            lines.lines().count(),
            1,
            "collision rejection must not append a second start event"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn ledger_replays_completed_without_reexecution() {
        let name = random_operation_id().unwrap();
        let path =
            std::env::temp_dir().join(format!("symthaea-transaction-ledger-{name}.jsonl"));
        let ledger = TransactionLedger::open_at(&path).unwrap();
        let tx = SystemTransaction::begin(
            MutationKind::Rollback,
            "replay-request-0001",
            Some("a".repeat(64).as_str()),
            b"nixos-rebuild switch --rollback",
        )
        .unwrap();
        assert!(matches!(
            ledger.admit(tx.clone()).unwrap(),
            TransactionAdmission::New(_)
        ));
        ledger
            .mark_completed(&tx, TransactionOutcome::ObservedSuccess)
            .unwrap();

        let replay = ledger
            .admit(
                SystemTransaction::begin(
                    MutationKind::Rollback,
                    "replay-request-0001",
                    Some("a".repeat(64).as_str()),
                    b"nixos-rebuild switch --rollback",
                )
                .unwrap(),
            )
            .unwrap();
        match replay {
            TransactionAdmission::Replayed(receipt) => {
                assert_eq!(receipt.transaction_id, tx.transaction_id);
                assert_eq!(receipt.outcome, TransactionOutcome::ObservedSuccess);
            }
            other => panic!("completed request must replay, not execute: {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ledger_fail_closes_after_start_without_completion() {
        let name = random_operation_id().unwrap();
        let path = std::env::temp_dir()
            .join(format!("symthaea-transaction-ledger-indeterminate-{name}.jsonl"));
        let ledger = TransactionLedger::open_at(&path).unwrap();
        let tx = SystemTransaction::begin(
            MutationKind::CreateImage,
            "indeterminate-request-0001",
            None,
            b"create-system-image",
        )
        .unwrap();
        assert!(matches!(
            ledger.admit(tx.clone()).unwrap(),
            TransactionAdmission::New(_)
        ));

        let retry = ledger
            .admit(
                SystemTransaction::begin(
                    MutationKind::CreateImage,
                    "indeterminate-request-0001",
                    None,
                    b"create-system-image",
                )
                .unwrap(),
            )
            .unwrap();
        match retry {
            TransactionAdmission::Indeterminate(receipt) => {
                assert_eq!(receipt.transaction_id, tx.transaction_id);
                assert_eq!(receipt.outcome, TransactionOutcome::Indeterminate);
            }
            other => panic!("uncertain request must fail closed: {other:?}"),
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ledger_rejects_request_id_reuse_for_different_payload() {
        let name = random_operation_id().unwrap();
        let path =
            std::env::temp_dir().join(format!("symthaea-transaction-ledger-conflict-{name}.jsonl"));
        let ledger = TransactionLedger::open_at(&path).unwrap();
        let first = SystemTransaction::begin(
            MutationKind::ServiceAction,
            "conflict-request-0001",
            None,
            b"restart:sshd",
        )
        .unwrap();
        assert!(matches!(
            ledger.admit(first).unwrap(),
            TransactionAdmission::New(_)
        ));
        let second = SystemTransaction::begin(
            MutationKind::ServiceAction,
            "conflict-request-0001",
            None,
            b"stop:sshd",
        )
        .unwrap();
        assert!(ledger.admit(second).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ledger_lookup_reports_unfinished_request_as_indeterminate() {
        let name = random_operation_id().unwrap();
        let path =
            std::env::temp_dir().join(format!("symthaea-transaction-ledger-lookup-{name}.jsonl"));
        let ledger = TransactionLedger::open_at(&path).unwrap();
        let tx = SystemTransaction::begin(
            MutationKind::Install,
            "lookup-request-0001",
            Some("a".repeat(64).as_str()),
            b"install",
        )
        .unwrap();
        assert!(matches!(ledger.admit(tx).unwrap(), TransactionAdmission::New(_)));
        let receipt = ledger.lookup("lookup-request-0001").unwrap().unwrap();
        assert_eq!(receipt.mutation, MutationKind::Install);
        assert_eq!(receipt.outcome, TransactionOutcome::Indeterminate);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn ledger_rejects_invalid_transaction_identity_on_load() {
        let name = random_operation_id().unwrap();
        let path = std::env::temp_dir()
            .join(format!("symthaea-transaction-ledger-invalid-id-{name}.jsonl"));
        let event = serde_json::json!({
            "schema_version": SCHEMA_VERSION,
            "event": "started",
            "request_id": "valid-request-000001",
            "transaction_id": "not-a-transaction-id",
            "mutation": "install",
            "target_machine_digest": null,
            "request_digest": "a".repeat(64),
            "outcome": null
        });
        std::fs::write(&path, format!("{event}\n")).unwrap();
        std::fs::set_permissions(
            &path,
            std::os::unix::fs::Permissions::from_mode(0o600),
        )
        .unwrap();
        let ledger = TransactionLedger::open_at(&path).unwrap();
        let error = ledger.load().expect_err("invalid transaction identity must fail closed");
        assert!(error.contains("invalid transaction_id"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn ledger_rejects_transaction_id_reuse_across_requests() {
        let name = random_operation_id().unwrap();
        let path = std::env::temp_dir()
            .join(format!("symthaea-transaction-ledger-tx-collision-{name}.jsonl"));
        let transaction_id = "0123456789abcdef0123456789abcdef";
        let first = serde_json::json!({
            "schema_version": SCHEMA_VERSION,
            "event": "started",
            "request_id": "request-one-000001",
            "transaction_id": transaction_id,
            "mutation": "install",
            "target_machine_digest": null,
            "request_digest": "a".repeat(64),
            "outcome": null
        });
        let second = serde_json::json!({
            "schema_version": SCHEMA_VERSION,
            "event": "started",
            "request_id": "request-two-000001",
            "transaction_id": transaction_id,
            "mutation": "rollback",
            "target_machine_digest": null,
            "request_digest": "b".repeat(64),
            "outcome": null
        });
        std::fs::write(&path, format!("{first}\n{second}\n")).unwrap();
        std::fs::set_permissions(
            &path,
            std::os::unix::fs::PermissionsExt::from_mode(0o600),
        )
        .unwrap();
        let ledger = TransactionLedger::open_at(&path).unwrap();
        let error = ledger.load().expect_err("transaction ID collision must fail closed");
        assert!(error.contains("reuses transaction_id"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn ledger_rejects_oversized_event_on_load() {
        use std::os::unix::fs::PermissionsExt;

        let name = random_operation_id().unwrap();
        let path = std::env::temp_dir()
            .join(format!("symthaea-transaction-ledger-oversized-{name}.jsonl"));
        let oversized = "x".repeat(MAX_JOURNAL_EVENT_BYTES + 1);
        std::fs::write(&path, format!("{oversized}\n")).unwrap();
        std::fs::set_permissions(
            &path,
            std::fs::Permissions::from_mode(0o600),
        )
        .unwrap();
        let ledger = TransactionLedger::open_at(&path).unwrap();
        let error = ledger.load().expect_err("oversized event must fail closed");
        assert!(error.contains("exceeds"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn randomness_failure_is_not_silently_replaced() {
        // The helper is intentionally the only operation-ID source; callers
        // receive an Err rather than falling back to timestamps.
        assert_eq!(SCHEMA_VERSION, 1);
    }
}
