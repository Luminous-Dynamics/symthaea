// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Same-directory atomic persistence for gateway state envelopes.

use crate::crypto_digest::Sha256Digest;
use crate::gateway_state::{GatewayStateEnvelope, GatewayStateError, MAX_GATEWAY_STATE_BYTES};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum GatewayStoreError {
    InvalidPath,
    LockBusy(PathBuf),
    PendingStateExists(PathBuf),
    StateMissing(PathBuf),
    StateTooLarge {
        actual: u64,
        maximum: usize,
    },
    StaleWriter {
        expected: Option<Sha256Digest>,
        actual: Option<Sha256Digest>,
    },
    State(GatewayStateError),
    Io(std::io::Error),
}

impl From<GatewayStateError> for GatewayStoreError {
    fn from(error: GatewayStateError) -> Self {
        Self::State(error)
    }
}

impl From<std::io::Error> for GatewayStoreError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub struct GatewayStateStore;

impl GatewayStateStore {
    pub fn load(path: impl AsRef<Path>) -> Result<GatewayStateEnvelope, GatewayStoreError> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(GatewayStoreError::StateMissing(path.to_path_buf()));
        }
        let metadata = fs::metadata(path)?;
        if metadata.len() > MAX_GATEWAY_STATE_BYTES as u64 {
            return Err(GatewayStoreError::StateTooLarge {
                actual: metadata.len(),
                maximum: MAX_GATEWAY_STATE_BYTES,
            });
        }
        let capacity = usize::try_from(metadata.len())
            .unwrap_or(MAX_GATEWAY_STATE_BYTES)
            .min(MAX_GATEWAY_STATE_BYTES);
        let mut bytes = Vec::with_capacity(capacity);
        File::open(path)?
            .take(MAX_GATEWAY_STATE_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > MAX_GATEWAY_STATE_BYTES {
            return Err(GatewayStoreError::StateTooLarge {
                actual: bytes.len() as u64,
                maximum: MAX_GATEWAY_STATE_BYTES,
            });
        }
        GatewayStateEnvelope::from_bytes(&bytes).map_err(GatewayStoreError::State)
    }

    /// Commit one envelope with a same-directory lock and temporary file.
    ///
    /// `expected_current` is an optimistic-concurrency guard. `None` requires
    /// that no current state exists. A digest requires an exact current state.
    pub fn commit(
        path: impl AsRef<Path>,
        envelope: &GatewayStateEnvelope,
        expected_current: Option<Sha256Digest>,
    ) -> Result<(), GatewayStoreError> {
        let path = path.as_ref();
        let file_name = path.file_name().ok_or(GatewayStoreError::InvalidPath)?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;

        let lock_path = sibling_path(parent, file_name, "lock")?;
        let pending_path = sibling_path(parent, file_name, "pending")?;
        let _lock = StoreLock::acquire(lock_path)?;

        let actual = match Self::load(path) {
            Ok(current) => Some(current.state_digest),
            Err(GatewayStoreError::StateMissing(_)) => None,
            Err(error) => return Err(error),
        };
        if actual != expected_current {
            return Err(GatewayStoreError::StaleWriter {
                expected: expected_current,
                actual,
            });
        }
        if pending_path.exists() {
            return Err(GatewayStoreError::PendingStateExists(pending_path));
        }

        let bytes = envelope.to_bytes()?;
        let mut pending = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&pending_path)?;
        if let Err(error) = (|| -> Result<(), std::io::Error> {
            pending.write_all(&bytes)?;
            pending.sync_all()?;
            Ok(())
        })() {
            let _ = fs::remove_file(&pending_path);
            return Err(GatewayStoreError::Io(error));
        }
        drop(pending);

        if let Err(error) = fs::rename(&pending_path, path) {
            let _ = fs::remove_file(&pending_path);
            return Err(GatewayStoreError::Io(error));
        }
        sync_directory(parent)?;
        Ok(())
    }

    /// Remove a stale pending file after an operator has independently verified
    /// that the committed state is intact and the pending state was never made
    /// authoritative.
    pub fn discard_pending(path: impl AsRef<Path>) -> Result<bool, GatewayStoreError> {
        let path = path.as_ref();
        let file_name = path.file_name().ok_or(GatewayStoreError::InvalidPath)?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let lock_path = sibling_path(parent, file_name, "lock")?;
        let pending_path = sibling_path(parent, file_name, "pending")?;
        let _lock = StoreLock::acquire(lock_path)?;
        if pending_path.exists() {
            fs::remove_file(pending_path)?;
            sync_directory(parent)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

struct StoreLock {
    path: PathBuf,
    file: File,
}

impl StoreLock {
    fn acquire(path: PathBuf) -> Result<Self, GatewayStoreError> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| {
                if error.kind() == std::io::ErrorKind::AlreadyExists {
                    GatewayStoreError::LockBusy(path.clone())
                } else {
                    GatewayStoreError::Io(error)
                }
            })?;
        Ok(Self { path, file })
    }
}

impl Drop for StoreLock {
    fn drop(&mut self) {
        let _ = self.file.sync_all();
        let _ = fs::remove_file(&self.path);
    }
}

fn sibling_path(
    parent: &Path,
    file_name: &std::ffi::OsStr,
    suffix: &str,
) -> Result<PathBuf, GatewayStoreError> {
    let name = file_name.to_str().ok_or(GatewayStoreError::InvalidPath)?;
    Ok(parent.join(format!(".{name}.{suffix}")))
}

fn sync_directory(path: &Path) -> Result<(), GatewayStoreError> {
    #[cfg(unix)]
    {
        File::open(path)?.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::SignatureAlgorithm;
    use crate::audit::AuditJournal;
    use crate::gateway_consensus_tracker::GatewayConsensusTracker;
    use crate::gateway_state::FabricationGatewayState;
    use crate::incident_ledger::IncidentLedger;
    use crate::operator_command_tracker::OperatorCommandTracker;
    use crate::session::MachineSessionTracker;
    use crate::submission_ledger::SubmissionLedger;
    use crate::telemetry_tracker::MachineTelemetryTracker;
    use crate::trust::{KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot};
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_ID: AtomicU64 = AtomicU64::new(1);

    fn envelope() -> GatewayStateEnvelope {
        let trust = TrustSnapshot::new(
            1,
            100,
            1_000,
            vec![KeyTrustRecord {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "root".into(),
                not_before_unix_s: 100,
                not_after_unix_s: None,
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::FabricationManifest]),
            }],
        )
        .unwrap();
        GatewayStateEnvelope::seal(
            FabricationGatewayState::genesis(
                500_000,
                trust,
                AuditJournal::default(),
                MachineSessionTracker::default(),
                MachineTelemetryTracker::default(),
                SubmissionLedger::default(),
                OperatorCommandTracker::default(),
                GatewayConsensusTracker::default(),
                IncidentLedger::default(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn test_path(name: &str) -> PathBuf {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "symthaea-fabrication-{name}-{}-{id}.json",
            std::process::id()
        ))
    }

    #[test]
    fn commit_load_and_stale_writer_gate() {
        let path = test_path("gateway-store");
        let envelope = envelope();
        GatewayStateStore::commit(&path, &envelope, None).unwrap();
        assert_eq!(GatewayStateStore::load(&path).unwrap(), envelope);
        assert!(matches!(
            GatewayStateStore::commit(&path, &envelope, None),
            Err(GatewayStoreError::StaleWriter { .. })
        ));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn tampered_state_fails_to_load() {
        let path = test_path("gateway-tamper");
        let envelope = envelope();
        GatewayStateStore::commit(&path, &envelope, None).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        let index = bytes.len() / 2;
        bytes[index] ^= 1;
        fs::write(&path, bytes).unwrap();
        assert!(GatewayStateStore::load(&path).is_err());
        let _ = fs::remove_file(path);
    }
}
