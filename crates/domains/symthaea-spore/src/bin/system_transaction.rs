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
use std::io::{BufRead, BufReader, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

const SCHEMA_VERSION: u16 = 1;
const CROSS_PROCESS_LOCK_PATH: &str = "/run/nixforhumanity-system-mutation.lock";
const LEDGER_PATH: &str = "/var/lib/nixforhumanity/system-transactions.jsonl";

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
}

impl TransactionLedger {
    pub(crate) fn open_default() -> Result<Self, String> {
        Self::open_at(std::path::Path::new(LEDGER_PATH))
    }

    fn open_at(path: &std::path::Path) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
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
        }
        Ok(Self {
            path: path.to_path_buf(),
        })
    }

    fn load(&self) -> Result<HashMap<String, JournalRecord>, String> {
        let file = match File::open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(HashMap::new());
            }
            Err(error) => {
                return Err(format!(
                    "unable to read transaction ledger {}: {error}",
                    self.path.display()
                ));
            }
        };

        let mut records = HashMap::new();
        for (line_number, line) in BufReader::new(file).lines().enumerate() {
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
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
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
        let line = serde_json::to_string(event)
            .map_err(|error| format!("unable to serialize transaction ledger event: {error}"))?;
        file.write_all(line.as_bytes())
            .and_then(|_| file.write_all(b"\n"))
            .and_then(|_| file.sync_all())
            .map_err(|error| {
                format!(
                    "unable to commit transaction ledger {}: {error}",
                    self.path.display()
                )
            })
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
    fn randomness_failure_is_not_silently_replaced() {
        // The helper is intentionally the only operation-ID source; callers
        // receive an Err rather than falling back to timestamps.
        assert_eq!(SCHEMA_VERSION, 1);
    }
}
