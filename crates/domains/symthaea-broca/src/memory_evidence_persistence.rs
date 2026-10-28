//! Crash-conscious, generation-based persistence for Broca evidence memory.
//!
//! The store publishes immutable HDC-index/evidence-snapshot generations and
//! atomically replaces a small CURRENT pointer last. Recovery is fail-closed:
//! an absent CURRENT means no committed generation; a corrupt CURRENT or any
//! mismatched/missing artifact is an integrity error, never a reason to fall
//! back to an older snapshot. This module assumes a Unix filesystem with
//! same-filesystem rename and directory fsync semantics, and serializes writers
//! with an exclusive lock file. A lock left by a crashed writer requires
//! operator recovery after verifying that no writer remains alive.
//!
//! This is a filesystem protocol, not a guarantee against every hardware,
//! controller, filesystem, or power-loss failure. The crash-injection tests
//! simulate interruption boundaries in-process; true power-loss qualification
//! must be performed on each claimed filesystem/device configuration.

use super::memory_evidence::{EvidenceCommitManifest, EvidenceIndex};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tempfile::{Builder, NamedTempFile};

const CURRENT_SCHEMA_VERSION: &str = "BROCA-EVIDENCE-CURRENT-V1";
const INDEX_FILE: &str = "hdc-index.bin";
const EVIDENCE_FILE: &str = "evidence-snapshot.json";
const MANIFEST_FILE: &str = "commit-manifest.json";
const CURRENT_FILE: &str = "CURRENT";
const LOCK_FILE: &str = ".writer.lock";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CurrentPointer {
    schema_version: String,
    generation: u64,
    manifest_sha256: String,
}

/// A successfully recovered, digest-verified generation.
#[derive(Debug)]
pub struct RecoveredEvidenceGeneration {
    pub generation: u64,
    pub hdc_index_bytes: Vec<u8>,
    pub evidence_index: EvidenceIndex,
    pub manifest: EvidenceCommitManifest,
}

/// A single-writer, fail-closed persistence store.
///
/// Multiple threads using the same instance are serialized. Cross-process
/// writers are serialized by `.writer.lock`. If a process dies, the lock may
/// remain; remove it only after externally establishing that the writer is
/// dead. The store never auto-clears a possibly live writer's lock.
pub struct EvidenceDiskStore {
    root: PathBuf,
    local_writer: Mutex<()>,
}

struct WriterLock {
    path: PathBuf,
    _file: File,
}

impl Drop for WriterLock {
    fn drop(&mut self) {
        // Best-effort cleanup; if it fails, the next writer fails closed.
        let _ = fs::remove_file(&self.path);
        let _ = sync_directory(self.path.parent().unwrap_or_else(|| Path::new(".")));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitFaultPoint {
    AfterIndexWrite,
    AfterEvidenceWrite,
    AfterManifestWrite,
    AfterGenerationPublish,
    BeforeCurrentPublish,
    AfterCurrentPublish,
}

impl EvidenceDiskStore {
    /// Open or initialize a store root. This does not silently repair CURRENT.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, String> {
        fs::create_dir_all(root.as_ref())
            .map_err(|error| format!("failed to create evidence store root: {error}"))?;
        let root = fs::canonicalize(root.as_ref())
            .map_err(|error| format!("failed to canonicalize evidence store root: {error}"))?;
        let generations = root.join("generations");
        fs::create_dir_all(&generations)
            .map_err(|error| format!("failed to create generation directory: {error}"))?;
        let generations_canonical = fs::canonicalize(&generations)
            .map_err(|error| format!("failed to resolve generation directory: {error}"))?;
        if !generations_canonical.starts_with(&root) {
            return Err("generation directory resolves outside store root".into());
        }
        sync_directory(&generations_canonical)?;
        sync_directory(&root)?;
        Ok(Self { root, local_writer: Mutex::new(()) })
    }

    /// Commit one logical generation. The caller must supply the exact serialized
    /// HDC index bytes corresponding to the supplied EvidenceIndex state.
    ///
    /// Generation is assigned from the fully verified current commit. If CURRENT
    /// is corrupt, commit fails rather than guessing a generation or restoring an
    /// older state. An error after CURRENT publication has an indeterminate
    /// durability outcome; call recover() and inspect the committed generation.
    pub fn commit(
        &self,
        hdc_index_bytes: &[u8],
        evidence_index: &EvidenceIndex,
    ) -> Result<u64, String> {
        self.commit_with_fault(hdc_index_bytes, evidence_index, None)
    }

    /// Recover only the CURRENT generation. No fallback is attempted on error.
    pub fn recover(&self) -> Result<Option<RecoveredEvidenceGeneration>, String> {
        let current_path = self.root.join(CURRENT_FILE);
        let pointer_bytes = match read_optional_regular_file(&current_path)? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        let pointer: CurrentPointer = serde_json::from_slice(&pointer_bytes)
            .map_err(|error| format!("invalid CURRENT pointer: {error}"))?;
        if pointer.schema_version != CURRENT_SCHEMA_VERSION {
            return Err("unsupported CURRENT pointer schema version".into());
        }
        if pointer.generation == 0 || !is_sha256(&pointer.manifest_sha256) {
            return Err("CURRENT pointer has invalid generation or manifest digest".into());
        }
        self.read_generation(&pointer).map(Some)
    }

    fn commit_with_fault(
        &self,
        hdc_index_bytes: &[u8],
        evidence_index: &EvidenceIndex,
        fault: Option<CommitFaultPoint>,
    ) -> Result<u64, String> {
        let _local = self.local_writer.lock()
            .map_err(|_| "evidence store writer mutex is poisoned".to_string())?;
        let _process = self.acquire_process_lock()?;

        if hdc_index_bytes.is_empty() {
            return Err("serialized HDC index bytes must be non-empty".into());
        }
        let current = self.recover()?;
        let generation = match current {
            Some(committed) => committed.generation.checked_add(1)
                .ok_or_else(|| "evidence generation counter overflow".to_string())?,
            None => 1,
        };

        let snapshot_bytes = evidence_index.snapshot_bytes(generation)?;
        let manifest = EvidenceCommitManifest::new(generation, hdc_index_bytes, &snapshot_bytes)?;
        let manifest_bytes = manifest.to_json_bytes()?;
        let manifest_sha256 = sha256_hex(&manifest_bytes);
        let pointer = CurrentPointer {
            schema_version: CURRENT_SCHEMA_VERSION.to_string(),
            generation,
            manifest_sha256: manifest_sha256.clone(),
        };
        let pointer_bytes = serde_json::to_vec(&pointer)
            .map_err(|error| format!("failed to encode CURRENT pointer: {error}"))?;
        let generations = self.root.join("generations");
        let stage = Builder::new().prefix(".staging-").tempdir_in(&generations)
            .map_err(|error| format!("failed to create generation staging directory: {error}"))?;

        write_new_synced(&stage.path().join(INDEX_FILE), hdc_index_bytes)?;
        if fault == Some(CommitFaultPoint::AfterIndexWrite) {
            let path = stage.keep();
            return Err(format!("simulated interruption after HDC index write; orphan staging at {}", path.display()));
        }

        write_new_synced(&stage.path().join(EVIDENCE_FILE), &snapshot_bytes)?;
        if fault == Some(CommitFaultPoint::AfterEvidenceWrite) {
            let path = stage.keep();
            return Err(format!("simulated interruption after evidence snapshot write; orphan staging at {}", path.display()));
        }

        write_new_synced(&stage.path().join(MANIFEST_FILE), &manifest_bytes)?;
        sync_directory(stage.path())?;
        if fault == Some(CommitFaultPoint::AfterManifestWrite) {
            let path = stage.keep();
            return Err(format!("simulated interruption after manifest write; orphan staging at {}", path.display()));
        }

        let final_name = generation_directory_name(generation, &manifest_sha256);
        let final_dir = generations.join(final_name);
        let stage_path = stage.keep();
        match fs::symlink_metadata(&final_dir) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    return Err("generation path exists but is not a real directory".into());
                }
                // Idempotent retry after a crash between generation rename and CURRENT
                // publication: reuse only byte-identical, internally valid artifacts.
                let same = read_regular_file(&final_dir.join(INDEX_FILE)).ok().as_deref() == Some(hdc_index_bytes)
                    && read_regular_file(&final_dir.join(EVIDENCE_FILE)).ok().as_deref() == Some(snapshot_bytes.as_slice())
                    && read_regular_file(&final_dir.join(MANIFEST_FILE)).ok().as_deref() == Some(manifest_bytes.as_slice());
                if !same {
                    return Err("generation directory collision with different artifact bytes".into());
                }
                let existing_pointer = CurrentPointer {
                    schema_version: CURRENT_SCHEMA_VERSION.to_string(),
                    generation,
                    manifest_sha256: manifest_sha256.clone(),
                };
                self.read_generation(&existing_pointer)?;
                fs::remove_dir_all(&stage_path)
                    .map_err(|error| format!("failed to remove redundant staging directory: {error}"))?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::rename(&stage_path, &final_dir)
                    .map_err(|error| format!("failed to publish immutable generation directory: {error}"))?;
            }
            Err(error) => return Err(format!("failed to inspect final generation path: {error}")),
        }
        sync_directory(&generations)?;
        // Read back and independently verify staged bytes before making them current.
        self.read_generation(&pointer)?;
        if fault == Some(CommitFaultPoint::AfterGenerationPublish) {
            return Err("simulated interruption after generation publication; CURRENT remains unchanged".into());
        }

        let mut pointer_file = NamedTempFile::new_in(&self.root)
            .map_err(|error| format!("failed to stage CURRENT pointer: {error}"))?;
        pointer_file.write_all(&pointer_bytes)
            .map_err(|error| format!("failed to write CURRENT pointer: {error}"))?;
        pointer_file.as_file().sync_all()
            .map_err(|error| format!("failed to sync staged CURRENT pointer: {error}"))?;
        if fault == Some(CommitFaultPoint::BeforeCurrentPublish) {
            return Err("simulated interruption before CURRENT publication; previous commit remains current".into());
        }

        pointer_file.persist(self.root.join(CURRENT_FILE))
            .map_err(|error| format!("failed to atomically publish CURRENT pointer: {error.error}"))?;
        if fault == Some(CommitFaultPoint::AfterCurrentPublish) {
            // Preserve the lock file to model process death: normal Drop cleanup
            // would itself fsync the root and invalidate this test boundary.
            std::mem::forget(_process);
            return Err("simulated interruption after CURRENT rename but before directory sync; lock remains and durability is ambiguous".into());
        }
        sync_directory(&self.root)?;
        Ok(generation)
    }

    fn acquire_process_lock(&self) -> Result<WriterLock, String> {
        let path = self.root.join(LOCK_FILE);
        let file = OpenOptions::new().write(true).create_new(true).open(&path)
            .map_err(|error| format!("cannot acquire evidence store writer lock (possibly held or stale): {error}"))?;
        let mut lock = WriterLock { path, _file: file };
        writeln!(&mut lock._file, "pid={}", std::process::id())
            .map_err(|error| format!("failed to write writer lock: {error}"))?;
        lock._file.sync_all()
            .map_err(|error| format!("failed to sync writer lock: {error}"))?;
        sync_directory(&self.root)?;
        Ok(lock)
    }

    fn read_generation(&self, pointer: &CurrentPointer) -> Result<RecoveredEvidenceGeneration, String> {
        let generations = self.root.join("generations");
        let dir = generations.join(generation_directory_name(pointer.generation, &pointer.manifest_sha256));
        let metadata = fs::symlink_metadata(&dir)
            .map_err(|error| format!("committed generation directory is missing: {error}"))?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err("committed generation path is not a real directory".into());
        }
        let index_bytes = read_regular_file(&dir.join(INDEX_FILE))?;
        let snapshot_bytes = read_regular_file(&dir.join(EVIDENCE_FILE))?;
        let manifest_bytes = read_regular_file(&dir.join(MANIFEST_FILE))?;
        if sha256_hex(&manifest_bytes) != pointer.manifest_sha256 {
            return Err("committed manifest digest does not match CURRENT".into());
        }
        let manifest = EvidenceCommitManifest::from_json_bytes(&manifest_bytes)?;
        if manifest.generation() != pointer.generation {
            return Err("manifest generation does not match CURRENT".into());
        }
        manifest.verify(&index_bytes, &snapshot_bytes)?;
        let evidence_index = EvidenceIndex::restore_snapshot(&snapshot_bytes)?;
        Ok(RecoveredEvidenceGeneration {
            generation: pointer.generation,
            hdc_index_bytes: index_bytes,
            evidence_index,
            manifest,
        })
    }
}

fn generation_directory_name(generation: u64, manifest_sha256: &str) -> String {
    format!("generation-{generation:020}-{manifest_sha256}")
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)
        .map_err(|error| format!("failed to create generation artifact: {error}"))?;
    file.write_all(bytes).map_err(|error| format!("failed to write generation artifact: {error}"))?;
    file.sync_all().map_err(|error| format!("failed to sync generation artifact: {error}"))
}

fn read_optional_regular_file(path: &Path) -> Result<Option<Vec<u8>>, String> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("failed to inspect {}: {error}", path.display())),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("expected a regular non-symlink file: {}", path.display()));
    }
    fs::read(path)
        .map(Some)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))
}

fn read_regular_file(path: &Path) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                format!("file not found: {}", path.display())
            } else {
                format!("failed to inspect {}: {error}", path.display())
            }
        })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("expected a regular non-symlink file: {}", path.display()));
    }
    fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("failed to sync directory {}: {error}", path.display()))
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<(), String> {
    Err("evidence disk store currently requires Unix directory-fsync semantics".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_evidence::{
        EvidenceLifecycle, EvidencePayload, EvidenceRecord, EvidenceSource, EVIDENCE_SCHEMA_VERSION,
    };
    use tempfile::tempdir;

    fn sha(bytes: &[u8]) -> String {
        sha256_hex(bytes)
    }

    fn sample_index(text: &str) -> EvidenceIndex {
        let mut index = EvidenceIndex::new();
        let source = b"canonical trajectory bytes";
        index.insert(EvidenceRecord {
            schema_version: EVIDENCE_SCHEMA_VERSION.to_string(),
            evidence_id: "episode-7-step-2".into(),
            hdc_id: 7,
            scope_id: "web".into(),
            source: EvidenceSource {
                trajectory_id: "episode-7".into(),
                step_id: Some("step-2".into()),
                step_index: Some(2),
                source_sha256: sha(source),
            },
            payload: EvidencePayload::Text { value: text.into() },
            content_sha256: sha(text.as_bytes()),
            lifecycle: EvidenceLifecycle::Active,
            occurred_at_unix_ms: Some(100),
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            superseded_by: None,
        }, source).unwrap();
        index
    }

    fn generation_path(root: &Path, generation: u64) -> PathBuf {
        let pointer: CurrentPointer = serde_json::from_slice(&fs::read(root.join(CURRENT_FILE)).unwrap()).unwrap();
        assert_eq!(pointer.generation, generation);
        root.join("generations").join(generation_directory_name(generation, &pointer.manifest_sha256))
    }

    #[test]
    fn commit_and_recover_verifies_matching_generation() {
        let root = tempdir().unwrap();
        let store = EvidenceDiskStore::open(root.path()).unwrap();
        let evidence = sample_index("current route");
        assert_eq!(store.commit(b"index-generation-one", &evidence).unwrap(), 1);
        let recovered = store.recover().unwrap().unwrap();
        assert_eq!(recovered.generation, 1);
        assert_eq!(recovered.hdc_index_bytes, b"index-generation-one");
        assert_eq!(recovered.evidence_index.resolve(&[7], "web", 150, Path::new("."), 5).unwrap()[0].value, "current route");

        assert_eq!(store.commit(b"index-generation-two", &evidence).unwrap(), 2);
        assert_eq!(store.recover().unwrap().unwrap().generation, 2);
    }

    #[test]
    fn deleted_tombstone_survives_disk_restart_and_old_hdc_hit() {
        let root = tempdir().unwrap();
        let store = EvidenceDiskStore::open(root.path()).unwrap();
        let mut evidence = sample_index("must not return after deletion");
        evidence.mark_deleted(7).unwrap();
        store.commit(b"index-with-deleted-id", &evidence).unwrap();

        // Reopening constructs fresh in-memory lock and ID maps, as after restart.
        let restarted = EvidenceDiskStore::open(root.path()).unwrap();
        let recovered = restarted.recover().unwrap().unwrap();
        assert!(recovered.evidence_index.resolve(&[7], "web", 150, Path::new("."), 5).unwrap().is_empty());
        assert!(recovered.evidence_index.insert(
            EvidenceRecord {
                schema_version: EVIDENCE_SCHEMA_VERSION.to_string(),
                evidence_id: "new-id-must-not-overwrite-tombstone".into(),
                hdc_id: 7,
                scope_id: "web".into(),
                source: EvidenceSource {
                    trajectory_id: "episode-new".into(),
                    step_id: None,
                    step_index: None,
                    source_sha256: sha(b"canonical trajectory bytes"),
                },
                payload: EvidencePayload::Text { value: "resurrection".into() },
                content_sha256: sha(b"resurrection"),
                lifecycle: EvidenceLifecycle::Active,
                occurred_at_unix_ms: None,
                valid_from_unix_ms: None,
                valid_until_unix_ms: None,
                superseded_by: None,
            },
            b"canonical trajectory bytes"
        ).is_err());
    }

    #[test]
    fn failures_before_current_publication_never_replace_previous_generation() {
        let points = [
            CommitFaultPoint::AfterIndexWrite,
            CommitFaultPoint::AfterEvidenceWrite,
            CommitFaultPoint::AfterManifestWrite,
            CommitFaultPoint::AfterGenerationPublish,
            CommitFaultPoint::BeforeCurrentPublish,
        ];
        for point in points {
            let root = tempdir().unwrap();
            let store = EvidenceDiskStore::open(root.path()).unwrap();
            store.commit(b"index-v1", &sample_index("old evidence")).unwrap();
            let result = store.commit_with_fault(b"index-v2", &sample_index("new evidence"), Some(point));
            assert!(result.is_err(), "failpoint {point:?} unexpectedly succeeded");
            let recovered = store.recover().unwrap().unwrap();
            assert_eq!(recovered.generation, 1, "failpoint {point:?} changed CURRENT");
            assert_eq!(recovered.hdc_index_bytes, b"index-v1");
            assert_eq!(recovered.evidence_index.resolve(&[7], "web", 150, Path::new("."), 5).unwrap()[0].value, "old evidence");
        }
    }

    #[test]
    fn interruption_after_current_rename_has_explicit_ambiguous_commit_outcome() {
        let root = tempdir().unwrap();
        let store = EvidenceDiskStore::open(root.path()).unwrap();
        store.commit(b"index-v1", &sample_index("old evidence")).unwrap();
        assert!(store.commit_with_fault(b"index-v2", &sample_index("new evidence"), Some(CommitFaultPoint::AfterCurrentPublish)).is_err());
        // The pointer is visible in this process, though power-loss durability has not been tested.
        let recovered = store.recover().unwrap().unwrap();
        assert_eq!(recovered.generation, 2);
        assert_eq!(recovered.hdc_index_bytes, b"index-v2");
    }

    #[test]
    fn corrupt_current_or_committed_artifact_fails_closed_without_fallback() {
        let root = tempdir().unwrap();
        let store = EvidenceDiskStore::open(root.path()).unwrap();
        store.commit(b"index-v1", &sample_index("old evidence")).unwrap();
        store.commit(b"index-v2", &sample_index("new evidence")).unwrap();
        let dir = generation_path(root.path(), 2);
        fs::write(dir.join(INDEX_FILE), b"tampered index").unwrap();
        assert!(store.recover().is_err(), "corrupt newest generation must not fall back to generation 1");

        fs::write(root.path().join(CURRENT_FILE), b"{not json").unwrap();
        assert!(store.recover().is_err(), "malformed CURRENT must not fall back");
    }

    #[test]
    fn lock_file_blocks_concurrent_or_unreconciled_writer() {
        let root = tempdir().unwrap();
        let store = EvidenceDiskStore::open(root.path()).unwrap();
        let lock = store.acquire_process_lock().unwrap();
        assert!(store.commit(b"index-v1", &sample_index("state"),).is_err());
        drop(lock);
        assert_eq!(store.commit(b"index-v1", &sample_index("state")).unwrap(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn dangling_current_symlink_is_corruption_not_empty_store() {
        use std::os::unix::fs::symlink;
        let root = tempdir().unwrap();
        let store = EvidenceDiskStore::open(root.path()).unwrap();
        symlink(root.path().join("missing-target"), root.path().join(CURRENT_FILE)).unwrap();
        assert!(store.recover().is_err());
    }

    #[test]
    fn empty_index_bytes_and_corrupt_snapshot_are_rejected() {
        let root = tempdir().unwrap();
        let store = EvidenceDiskStore::open(root.path()).unwrap();
        assert!(store.commit(b"", &sample_index("state")).is_err());
        store.commit(b"index-v1", &sample_index("state")).unwrap();
        let dir = generation_path(root.path(), 1);
        fs::write(dir.join(EVIDENCE_FILE), b"{}").unwrap();
        assert!(store.recover().is_err());
    }
}
