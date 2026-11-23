// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Append-only filesystem persistence for label-blind retrieval trace receipts.
//!
//! This is a local durability adapter for restart/replay and idempotent
//! invocation binding. It is NOT an independent trust anchor: an attacker who
//! can restore/replace the directory can roll it back, and a privileged actor
//! can rewrite a receipt and its hash consistently. Production needs separate
//! witness custody and authenticated trace provenance.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::retrieval_policy_memory::{
    RetrievalPolicyMemoryV1, RetrievalTraceReceiptV1,
};

const TRACE_SUFFIX: &str = ".trace.json";
const MAX_TRACE_FILE_BYTES: u64 = 1024 * 1024;
const MAX_STORED_TRACE_FILES: usize = 100_000;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

#[derive(Clone, Debug)]
pub struct RetrievalPolicyTraceFileStoreV1 {
    directory: PathBuf,
}

impl RetrievalPolicyTraceFileStoreV1 {
    /// Open/create a private trace directory. The directory is local persistence,
    /// not a remotely witnessed rollback-resistant store.
    pub fn open(directory: impl AsRef<Path>) -> Result<Self, String> {
        fs::create_dir_all(directory.as_ref())
            .map_err(|error| format!("create retrieval trace directory: {error}"))?;
        let directory = fs::canonicalize(directory.as_ref())
            .map_err(|error| format!("canonicalize retrieval trace directory: {error}"))?;
        if !directory.is_dir() {
            return Err("retrieval trace path must be a directory".to_owned());
        }
        #[cfg(unix)]
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))
            .map_err(|error| format!("restrict retrieval trace directory: {error}"))?;
        Ok(Self { directory })
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    fn trace_path(&self, trace: &RetrievalTraceReceiptV1) -> PathBuf {
        let invocation_digest = hex_digest(
            &Sha256::digest(trace.query_invocation_id.as_bytes())
        );
        self.directory.join(format!("{invocation_digest}{TRACE_SUFFIX}"))
    }

    /// Persist a trace create-once and bind it into the in-memory policy store.
    ///
    /// A repeated invocation is accepted only if its canonical JSON bytes are
    /// identical. The final file is published atomically through a same-directory
    /// temporary file and hard-link create-if-absent operation.
    pub fn record_trace(
        &self,
        memory: &mut RetrievalPolicyMemoryV1,
        trace: &RetrievalTraceReceiptV1,
    ) -> Result<String, String> {
        let value = trace.to_json_value();
        let decoded = RetrievalTraceReceiptV1::from_json_value(&value)
            .map_err(|error| format!("validate retrieval trace before persistence: {error:?}"))?;
        if decoded != *trace {
            return Err("retrieval trace does not round-trip through its JSON schema".to_owned());
        }
        let encoded = serde_json::to_vec(&value)
            .map_err(|error| format!("serialize retrieval trace: {error}"))?;
        if encoded.len() as u64 > MAX_TRACE_FILE_BYTES {
            return Err("retrieval trace exceeds maximum file size".to_owned());
        }

        let path = self.trace_path(trace);
        self.publish_create_once(&path, &encoded).map_err(|error| {
            format!(
                "retrieval trace persistence outcome may be unknown; reopen and reconcile before retry: {error}"
            )
        })?;
        memory
            .restore_trace_json_value(&value)
            .map_err(|error| format!("bind persisted retrieval trace: {error:?}"))
    }

    /// Replay every persisted receipt into a fresh in-memory policy store.
    /// Unknown files, symlinks, malformed receipts, bad filenames, oversized
    /// entries and duplicate invocation rebinding fail closed.
    pub fn load_into(
        &self,
        memory: &mut RetrievalPolicyMemoryV1,
    ) -> Result<usize, String> {
        let entries = fs::read_dir(&self.directory)
            .map_err(|error| format!("read retrieval trace directory: {error}"))?;
        let mut paths = Vec::new();

        for entry in entries {
            let entry = entry.map_err(|error| format!("read retrieval trace entry: {error}"))?;
            let name = entry.file_name();
            let name = name.to_str().ok_or_else(|| {
                "retrieval trace directory contains a non-UTF-8 filename".to_owned()
            })?;
            if name.starts_with(".pending-") {
                // An interrupted temporary write is not a published receipt.
                continue;
            }
            if !name.ends_with(TRACE_SUFFIX) {
                return Err(format!("unexpected file in retrieval trace directory: {name}"));
            }
            let metadata = entry
                .file_type()
                .map_err(|error| format!("inspect retrieval trace entry: {error}"))?;
            if !metadata.is_file() {
                return Err(format!("retrieval trace entry is not a regular file: {name}"));
            }
            paths.push(entry.path());
        }

        if paths.len() > MAX_STORED_TRACE_FILES {
            return Err("retrieval trace store exceeds maximum receipt count".to_owned());
        }
        paths.sort();

        let mut loaded = 0_usize;
        for path in paths {
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| format!("inspect retrieval trace file: {error}"))?;
            if !metadata.file_type().is_file() || metadata.len() > MAX_TRACE_FILE_BYTES {
                return Err(format!("invalid retrieval trace file: {}", path.display()));
            }
            let bytes = read_bounded(&path)?;
            let value: Value = serde_json::from_slice(&bytes)
                .map_err(|error| format!("parse retrieval trace {}: {error}", path.display()))?;
            let trace = RetrievalTraceReceiptV1::from_json_value(&value)
                .map_err(|error| format!("validate retrieval trace {}: {error:?}", path.display()))?;
            let expected_path = self.trace_path(&trace);
            if expected_path != path {
                return Err(format!(
                    "retrieval trace filename does not match its invocation ID: {}",
                    path.display()
                ));
            }
            let canonical = serde_json::to_vec(&trace.to_json_value())
                .map_err(|error| format!("canonicalize retrieval trace: {error}"))?;
            if canonical != bytes {
                return Err(format!(
                    "retrieval trace JSON is not canonical: {}",
                    path.display()
                ));
            }
            memory
                .restore_trace_json_value(&value)
                .map_err(|error| format!("restore retrieval trace into policy memory: {error:?}"))?;
            loaded += 1;
        }
        Ok(loaded)
    }

    fn publish_create_once(&self, final_path: &Path, encoded: &[u8]) -> Result<(), String> {
        if final_path.exists() {
            return compare_existing(final_path, encoded);
        }

        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temporary_path = self.directory.join(format!(
            ".pending-{}-{counter:016x}",
            std::process::id()
        ));

        let result = (|| {
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            options.mode(0o600);
            let mut file = options
                .open(&temporary_path)
                .map_err(|error| format!("create temporary retrieval trace: {error}"))?;
            file.write_all(encoded)
                .map_err(|error| format!("write temporary retrieval trace: {error}"))?;
            file.sync_all()
                .map_err(|error| format!("sync temporary retrieval trace: {error}"))?;
            drop(file);

            match fs::hard_link(&temporary_path, final_path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    compare_existing(final_path, encoded)?;
                }
                Err(error) => return Err(format!("publish retrieval trace: {error}")),
            }
            fs::remove_file(&temporary_path)
                .map_err(|error| format!("remove temporary retrieval trace: {error}"))?;
            sync_directory(&self.directory)
                .map_err(|error| format!("sync retrieval trace directory: {error}"))?;
            compare_existing(final_path, encoded)
        })();

        let _ = fs::remove_file(&temporary_path);
        result
    }
}

fn compare_existing(path: &Path, expected: &[u8]) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect existing retrieval trace: {error}"))?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_TRACE_FILE_BYTES {
        return Err("existing retrieval trace is not a bounded regular file".to_owned());
    }
    let existing = read_bounded(path)?;
    if existing != expected {
        return Err("opaque query invocation is already bound to different persisted trace bytes".to_owned());
    }
    Ok(())
}

fn read_bounded(path: &Path) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect retrieval trace: {error}"))?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_TRACE_FILE_BYTES {
        return Err("retrieval trace is not a bounded regular file".to_owned());
    }
    let file = File::open(path)
        .map_err(|error| format!("open retrieval trace: {error}"))?;
    let mut limited = file.take(MAX_TRACE_FILE_BYTES + 1);
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    limited.read_to_end(&mut bytes)
        .map_err(|error| format!("read retrieval trace: {error}"))?;
    if bytes.len() as u64 > MAX_TRACE_FILE_BYTES {
        return Err("retrieval trace exceeds maximum file size".to_owned());
    }
    Ok(bytes)
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> std::io::Result<()> {
    File::open(path)?.sync_all()
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> std::io::Result<()> {
    // Platforms without directory fsync do not receive a crash-durability claim.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval_policy_memory::{
        RetrievalOperationOutcomeV1, RetrievalOperationV1, RetrievalStopReasonV1,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_directory(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "mace-policy-trace-store-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).expect("create fixture directory");
        root
    }

    fn trace(invocation: &str) -> RetrievalTraceReceiptV1 {
        RetrievalTraceReceiptV1 {
            run_id: "run-1".to_owned(),
            query_invocation_id: invocation.to_owned(),
            domain_id: "web".to_owned(),
            environment_revision_sha256: "a".repeat(64),
            tool_profile_sha256: "b".repeat(64),
            query_sha256: "c".repeat(64),
            candidate_set_sha256: None,
            operations: vec![RetrievalOperationV1 {
                tool_id: "trajectory.bm25.search".to_owned(),
                tool_revision_sha256: "b".repeat(64),
                action_template_sha256: "d".repeat(64),
                observed_evidence_ids: Vec::new(),
                outcome: RetrievalOperationOutcomeV1::NoEvidence,
                elapsed_ns: 5,
                inspected_bytes: 0,
                inspected_bytes_measured: false,
            }],
            stop_reason: RetrievalStopReasonV1::NoCandidateEvidence,
            tool_call_budget: 1,
            tool_calls_used: 1,
            elapsed_ns: 5,
            inspected_bytes: 0,
            inspected_bytes_complete: false,
            created_at_unix: 100,
        }
    }

    #[test]
    fn receipt_survives_restart_and_restores_policy_trace_memory() {
        let root = temporary_directory("restart");
        let store = RetrievalPolicyTraceFileStoreV1::open(&root).expect("open store");
        let mut first_memory = RetrievalPolicyMemoryV1::default();
        let receipt = trace("invocation-1");
        let digest = store.record_trace(&mut first_memory, &receipt).expect("persist trace");
        assert_eq!(first_memory.trace_count(), 1);

        let reopened_store = RetrievalPolicyTraceFileStoreV1::open(&root).expect("reopen store");
        let mut restored_memory = RetrievalPolicyMemoryV1::default();
        assert_eq!(reopened_store.load_into(&mut restored_memory).expect("restore"), 1);
        assert_eq!(restored_memory.trace_count(), 1);
        assert_eq!(
            restored_memory.trace_receipt_sha256("run-1", "invocation-1"),
            Some(digest.as_str())
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn repeated_receipt_is_idempotent_but_different_invocation_binding_is_rejected() {
        let root = temporary_directory("idempotent");
        let store = RetrievalPolicyTraceFileStoreV1::open(&root).expect("open store");
        let mut memory = RetrievalPolicyMemoryV1::default();
        let original = trace("invocation-2");
        let first = store.record_trace(&mut memory, &original).expect("persist");
        assert_eq!(store.record_trace(&mut memory, &original).expect("repeat"), first);

        let mut changed = original;
        changed.query_sha256 = "e".repeat(64);
        changed.candidate_set_sha256 = None;
        assert!(store.record_trace(&mut memory, &changed).is_err());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn post_publication_memory_bind_conflict_is_reported_as_unknown_and_reconcilable() {
        let root = temporary_directory("unknown-bind");
        let store = RetrievalPolicyTraceFileStoreV1::open(&root).expect("open store");
        let mut memory = RetrievalPolicyMemoryV1::default();

        let mut conflicting = trace("invocation-bind-conflict");
        conflicting.query_sha256 = "e".repeat(64);
        memory.record_trace(conflicting).expect("seed conflicting in-memory binding");

        let persisted = trace("invocation-bind-conflict");
        let error = store.record_trace(&mut memory, &persisted)
            .expect_err("same invocation cannot rebind the live memory map");
        assert!(error.contains("outcome may be unknown"));
        assert_eq!(fs::read_dir(&root).expect("trace directory").count(), 1);

        let mut reconciled = RetrievalPolicyMemoryV1::default();
        assert_eq!(store.load_into(&mut reconciled).expect("reconcile"), 1);
        assert_eq!(
            reconciled.trace_receipt_sha256("run-1", "invocation-bind-conflict"),
            Some(persisted.receipt_sha256().as_str())
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn changed_or_tampered_receipt_fails_restore() {
        let root = temporary_directory("tamper");
        let store = RetrievalPolicyTraceFileStoreV1::open(&root).expect("open store");
        let mut memory = RetrievalPolicyMemoryV1::default();
        store.record_trace(&mut memory, &trace("invocation-3")).expect("persist");
        let entry = fs::read_dir(&root).expect("read dir").next().expect("entry").expect("file");
        let mut bytes = fs::read(entry.path()).expect("read receipt");
        let index = bytes.iter().position(|byte| *byte == b'5').expect("trace elapsed digit");
        bytes[index] = b'6';
        fs::write(entry.path(), bytes).expect("tamper");
        let mut restored = RetrievalPolicyMemoryV1::default();
        assert!(store.load_into(&mut restored).is_err());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn pending_files_are_not_replayed_as_published_receipts() {
        let root = temporary_directory("pending");
        let store = RetrievalPolicyTraceFileStoreV1::open(&root).expect("open");
        fs::write(root.join(".pending-interrupted"), b"partial").expect("write pending");
        let mut memory = RetrievalPolicyMemoryV1::default();
        assert_eq!(store.load_into(&mut memory).expect("load"), 0);
        fs::remove_dir_all(root).expect("cleanup");
    }
}
