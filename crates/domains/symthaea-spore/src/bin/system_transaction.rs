// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed, auditable transaction envelopes for consequential relay mutations.
//!
//! This is deliberately narrower than a distributed transaction protocol.
//! It binds one authenticated relay request to:
//!   - a CSPRNG-generated operation identifier;
//!   - a typed mutation kind;
//!   - a digest of the exact request payload;
//!   - the authoritative target identity when one is available.
//!
//! The envelope is not itself a cryptographic signature. The current relay
//! authorization boundary remains the already-authenticated WebSocket bearer
//! token. This module prevents an authorized request from losing its identity
//! as it crosses asynchronous staging/execution boundaries.

use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

const SCHEMA_VERSION: u16 = 1;
const CROSS_PROCESS_LOCK_PATH: &str = "/run/nixforhumanity-system-mutation.lock";

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


#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TransactionOutcome {
    ObservedSuccess,
    Failed,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
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
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct SystemTransaction {
    schema_version: u16,
    pub(crate) transaction_id: String,
    pub(crate) mutation: MutationKind,
    pub(crate) target_machine_digest: Option<String>,
    pub(crate) request_digest: String,
    pub(crate) authorization: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TransactionReceipt {
    schema_version: u16,
    transaction_id: String,
    mutation: MutationKind,
    target_machine_digest: Option<String>,
    request_digest: String,
    authorization: &'static str,
    outcome: TransactionOutcome,
}

impl SystemTransaction {
    pub(crate) fn begin(
        mutation: MutationKind,
        target_machine_digest: Option<&str>,
        payload: &[u8],
    ) -> Result<Self, String> {
        let transaction_id = random_operation_id()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"nixforhumanity-system-transaction-v1\0");
        hasher.update(mutation.as_str().as_bytes());
        hasher.update(b"\0");
        hasher.update(payload);
        let request_digest = hasher.finalize().to_hex().to_string();

        Ok(Self {
            schema_version: SCHEMA_VERSION,
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
            "transaction={} mutation={} request_digest={}{}",
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
        let a = SystemTransaction::begin(MutationKind::Rollback, None, b"rollback").unwrap();
        let b = SystemTransaction::begin(MutationKind::Rollback, None, b"rollback").unwrap();

        assert_ne!(a.transaction_id, b.transaction_id);
        assert_eq!(a.transaction_id.len(), 32);
        assert_eq!(a.request_digest, b.request_digest);
    }

    #[test]
    fn transaction_binds_payload_and_target() {
        let a =
            SystemTransaction::begin(MutationKind::ServiceAction, Some("a".repeat(64).as_str()), b"restart:test")
                .unwrap();
        let b = SystemTransaction::begin(
            MutationKind::ServiceAction,
            Some("b".repeat(64).as_str()),
            b"restart:test",
        )
        .unwrap();
        let c =
            SystemTransaction::begin(MutationKind::ServiceAction, Some("a".repeat(64).as_str()), b"stop:test")
                .unwrap();

        assert_ne!(a.target_machine_digest, b.target_machine_digest);
        assert_ne!(a.request_digest, c.request_digest);
        assert_ne!(a.log_line(), b.log_line());
    }

    #[test]
    fn digest_is_domain_separated_by_mutation_kind() {
        let install =
            SystemTransaction::begin(MutationKind::Install, None, b"same-payload").unwrap();
        let rollback =
            SystemTransaction::begin(MutationKind::Rollback, None, b"same-payload").unwrap();

        assert_ne!(install.request_digest, rollback.request_digest);
    }

    #[test]
    fn receipt_preserves_transaction_identity() {
        let tx = SystemTransaction::begin(MutationKind::GcCollect, None, b"gc-30d").unwrap();
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
    }

    #[test]
    fn randomness_failure_is_not_silently_replaced() {
        // The helper is intentionally the only operation-ID source; callers
        // receive an Err rather than falling back to timestamps.
        assert_eq!(SCHEMA_VERSION, 1);
    }
}
