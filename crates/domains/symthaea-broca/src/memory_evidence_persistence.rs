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

use super::memory_evidence::{EvidenceCommitManifest, EvidenceIndex, HdcStoreIdentity, LexicalIndex, MemoryContextItem, ResolvedMemoryContextItem};
use std::collections::BTreeSet;
use symthaea_core::hdc::BinaryHV;
use symthaea_hdc_store::{HdcStore, HdcStoreReader};
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
const LEXICAL_FILE: &str = "lexical-index.json";
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
    /// Present for typed production commits whose manifest binds the lexical artifact.
    pub lexical_index: Option<LexicalIndex>,
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

/// Explicit visible-file retention result. This is not a secure-erasure receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidencePruneReport {
    pub current_generation: u64,
    pub retained_generation_window_start: u64,
    pub removed_generation_directories: Vec<String>,
    pub removed_staging_directories: usize,
}

/// Result of a query against one verified, immutable evidence/HDC generation.
///
/// Candidate IDs and scores are retained for audit/benchmark diagnostics; only
/// the scope- and lifecycle-eligible payloads are emitted as context.
#[derive(Debug, Clone, PartialEq)]
pub struct CommittedEvidenceQuery {
    pub generation: u64,
    pub candidates: Vec<(u64, f32)>,
    pub resolved_context: Vec<ResolvedMemoryContextItem>,
}

impl CommittedEvidenceQuery {
    /// Project the provenance-bearing result to the official benchmark ABI.
    pub fn context_items(&self) -> Vec<MemoryContextItem> {
        self.resolved_context.iter().map(|resolved| resolved.item.clone()).collect()
    }
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
    AfterLexicalIndexWrite,
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

    /// Low-level opaque-byte fixture used only by module tests. Production
    /// callers must use commit_hdc_store so the actual HDC generation, checksum
    /// and live-ID set are independently bound and revalidated.
    #[cfg(test)]
    fn commit(
        &self,
        hdc_index_bytes: &[u8],
        evidence_index: &EvidenceIndex,
    ) -> Result<u64, String> {
        self.commit_with_fault(hdc_index_bytes, evidence_index, None)
    }

    /// Production commit path for a real HdcStore. The exact live IDs and logical
    /// checksum are captured from one generation-pinned read view after syncing.
    /// Active evidence IDs must match the live vector IDs exactly.
    pub fn commit_hdc_store(
        &self,
        hdc_store: &mut HdcStore,
        evidence_index: &EvidenceIndex,
    ) -> Result<u64, String> {
        hdc_store.sync_all()
            .map_err(|error| format!("failed to sync canonical HDC store: {error}"))?;
        let (identity, hdc_live_ids) = {
            let view = hdc_store.read_view();
            let checksum = view.content_checksum();
            let live_ids = view.ids().collect::<BTreeSet<_>>();
            (
                HdcStoreIdentity {
                    store_generation: view.generation(),
                    live_count: checksum.live_count,
                    checksum_version: checksum.version,
                    checksum_hex: checksum.hex(),
                },
                live_ids,
            )
        };
        let evidence_live_ids = evidence_index.active_hdc_ids();
        if hdc_live_ids != evidence_live_ids {
            let missing_payloads: Vec<u64> = hdc_live_ids.difference(&evidence_live_ids).copied().collect();
            let missing_vectors: Vec<u64> = evidence_live_ids.difference(&hdc_live_ids).copied().collect();
            return Err(format!(
                "HDC/evidence live-ID mismatch; vectors_without_active_evidence={missing_payloads:?}; active_evidence_without_vector={missing_vectors:?}"
            ));
        }
        let hdc_index_bytes = fs::read(hdc_store.path())
            .map_err(|error| format!("failed to read synced canonical HDC store: {error}"))?;
        if hdc_index_bytes.is_empty() {
            return Err("canonical HDC store file is empty".into());
        }
        self.commit_with_identity_and_fault(&hdc_index_bytes, evidence_index, Some(identity), None)
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

    /// Query the exact HDC index and evidence snapshot named by one CURRENT pointer.
    ///
    /// This is the first query-serving boundary for the disk store: it reads one
    /// pointer, verifies that generation, opens the HDC bytes from that same
    /// immutable generation, rechecks the typed HDC identity, performs exact
    /// nearest-neighbor search, and only then resolves scope/lifecycle/time-
    /// eligible payloads. It intentionally refuses legacy opaque-byte fixture
    /// commits because those do not bind a real HDC identity.
    ///
    /// `Ok(None)` means no generation has ever been committed. Any malformed
    /// or mismatched committed state is an error; callers must not turn it into
    /// a successful answer or fall back to a separate mutable HDC store.
    /// Query committed text evidence with a deterministic BM25 lexical baseline.
    ///
    /// Unlike `query_committed_context`, this path does not require a trained
    /// text-to-HDC encoder. It is intentionally lexical and should be evaluated
    /// as its own benchmark arm. The committed HDC index is still independently
    /// validated as part of the generation, while candidate ranking is computed
    /// from eligible, digest-verified text payloads in that same evidence snapshot.
    ///
    /// Image evidence is not ranked by this baseline. An empty lexical match
    /// returns a successful generation with no context; integrity failures remain
    /// errors. A store without a typed HDC commit is not allowed to serve context.
    pub fn query_committed_text_context(
        &self,
        query_text: &str,
        candidate_k: usize,
        scope_id: &str,
        now_ms: u64,
        payload_root: impl AsRef<Path>,
        max_items: usize,
    ) -> Result<Option<CommittedEvidenceQuery>, String> {
        if scope_id.trim().is_empty() {
            return Err("scope_id must be non-empty".into());
        }

        let pointer_bytes = match read_optional_regular_file(&self.root.join(CURRENT_FILE))? {
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

        let recovered = self.read_generation(&pointer)?;
        if recovered.manifest.hdc_store_identity().is_none() {
            return Err("committed text query requires a typed HDC identity; opaque-byte fixtures cannot serve context".into());
        }
        let lexical_index = recovered.lexical_index.as_ref()
            .ok_or_else(|| "typed committed generation has no manifest-bound lexical index".to_string())?;
        let candidates = recovered.evidence_index.rank_text_indexed(
            lexical_index, recovered.generation, query_text, scope_id, now_ms, candidate_k,
        )?;
        let candidate_ids: Vec<u64> = candidates.iter().map(|(id, _score)| *id).collect();
        let resolved_context = recovered.evidence_index.resolve_with_provenance(
            &candidate_ids,
            scope_id,
            now_ms,
            payload_root.as_ref(),
            max_items,
        )?;

        Ok(Some(CommittedEvidenceQuery {
            generation: recovered.generation,
            candidates,
            resolved_context,
        }))
    }

    pub fn query_committed_context(
        &self,
        query: &BinaryHV,
        candidate_k: usize,
        scope_id: &str,
        now_ms: u64,
        payload_root: impl AsRef<Path>,
        max_items: usize,
    ) -> Result<Option<CommittedEvidenceQuery>, String> {
        if scope_id.trim().is_empty() {
            return Err("scope_id must be non-empty".into());
        }

        // Read CURRENT exactly once so candidate search and payload resolution
        // cannot intentionally select different generations across a commit.
        let pointer_bytes = match read_optional_regular_file(&self.root.join(CURRENT_FILE))? {
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

        let recovered = self.read_generation(&pointer)?;
        let identity = recovered.manifest.hdc_store_identity()
            .ok_or_else(|| "committed query requires a typed HDC identity; opaque-byte fixtures cannot serve context".to_string())?;
        identity.validate_for_recovery()?;

        let index_path = self.root
            .join("generations")
            .join(generation_directory_name(pointer.generation, &pointer.manifest_sha256))
            .join(INDEX_FILE);
        let reader = HdcStoreReader::open(&index_path)
            .map_err(|error| format!("committed query could not open its HDC generation: {error}"))?;

        // Revalidate the reader that will actually answer the query. The earlier
        // recovery pass validates the manifest's exact file bytes; this second
        // check ensures the opened mapping still matches that committed identity.
        let checksum = reader.content_checksum();
        if reader.generation() != identity.store_generation
            || reader.live_count() != identity.live_count
            || checksum.version != identity.checksum_version
            || checksum.live_count != identity.live_count
            || checksum.hex() != identity.checksum_hex
        {
            return Err("opened query HDC generation no longer matches committed identity".into());
        }
        if reader.ids().collect::<BTreeSet<_>>() != recovered.evidence_index.active_hdc_ids() {
            return Err("opened query HDC live IDs do not match committed active evidence IDs".into());
        }

        let candidates = if candidate_k == 0 {
            Vec::new()
        } else {
            reader.scan_similar(query, candidate_k)
        };
        let candidate_ids: Vec<u64> = candidates.iter().map(|(id, _score)| *id).collect();
        let resolved_context = recovered.evidence_index.resolve_with_provenance(
            &candidate_ids,
            scope_id,
            now_ms,
            payload_root.as_ref(),
            max_items,
        )?;

        Ok(Some(CommittedEvidenceQuery {
            generation: recovered.generation,
            candidates,
            resolved_context,
        }))
    }

    /// Remove old visible generation directories under the exclusive writer lock.
    ///
    /// retain_generations is a window of generation numbers, including CURRENT.
    /// CURRENT is always retained. Published-but-never-current future generations
    /// are discarded, along with stale staging directories. If multiple immutable
    /// artifacts exist for a retained historical generation, all are preserved
    /// because this store has no authenticated predecessor chain to distinguish
    /// a former commit from a failed alternate commit at that generation.
    ///
    /// This removes visible files only. It is NOT secure erasure: filesystem
    /// journals, snapshots, backups, SSD remapping, or external copies may retain
    /// the bytes. Use application-level encryption/crypto-erasure where required.
    pub fn prune_history(&self, retain_generations: usize) -> Result<EvidencePruneReport, String> {
        if retain_generations == 0 {
            return Err("retain_generations must be at least 1".into());
        }
        let _local = self.local_writer.lock()
            .map_err(|_| "evidence store writer mutex is poisoned".to_string())?;
        let _process = self.acquire_process_lock()?;
        let current = self.recover()?
            .ok_or_else(|| "cannot prune evidence history without a committed CURRENT generation".to_string())?;

        let current_path = self.root.join(CURRENT_FILE);
        let pointer_bytes = read_regular_file(&current_path)?;
        let pointer: CurrentPointer = serde_json::from_slice(&pointer_bytes)
            .map_err(|error| format!("invalid CURRENT pointer during prune: {error}"))?;
        if pointer.generation != current.generation {
            return Err("CURRENT changed while acquiring the retention lock".into());
        }
        let current_dir_name = generation_directory_name(pointer.generation, &pointer.manifest_sha256);
        let generations = self.root.join("generations");
        let retain_u64 = u64::try_from(retain_generations).unwrap_or(u64::MAX);
        let width = retain_u64.saturating_sub(1);
        let window_start = current.generation.saturating_sub(width);

        let mut generation_dirs: Vec<(u64, String, PathBuf)> = Vec::new();
        let mut staging_dirs: Vec<PathBuf> = Vec::new();
        for entry in fs::read_dir(&generations)
            .map_err(|error| format!("failed to enumerate evidence generations: {error}"))?
        {
            let entry = entry.map_err(|error| format!("failed to read generation entry: {error}"))?;
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| format!("failed to inspect generation entry: {error}"))?;
            if metadata.file_type().is_symlink() {
                return Err(format!("refusing to prune symlinked generation entry {name}"));
            }
            if name.starts_with(".staging-") {
                if !metadata.is_dir() {
                    return Err(format!("staging entry is not a directory: {name}"));
                }
                staging_dirs.push(path);
                continue;
            }
            if !name.starts_with("generation-") || !metadata.is_dir() {
                return Err(format!("unrecognized evidence generation entry: {name}"));
            }
            let rest = name.strip_prefix("generation-").expect("prefix checked");
            let (number, digest) = rest.split_once('-')
                .ok_or_else(|| format!("malformed generation directory name: {name}"))?;
            if number.len() != 20 || digest.len() != 64 || !is_sha256(digest) {
                return Err(format!("malformed generation directory name: {name}"));
            }
            let generation = number.parse::<u64>()
                .map_err(|_| format!("invalid generation number in {name}"))?;
            if generation == 0 {
                return Err(format!("zero generation number in {name}"));
            }
            generation_dirs.push((generation, name, path));
        }

        if !generation_dirs.iter().any(|(_, name, _)| name == &current_dir_name) {
            return Err("CURRENT generation directory is missing from retention scan".into());
        }

        // Never delete while the directory inventory contains malformed entries;
        // after the full scan succeeds, apply the already-decided deletion set.
        generation_dirs.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        let mut removed_generation_directories = Vec::new();
        for (generation, name, path) in generation_dirs {
            let is_current = name == current_dir_name;
            let outside_window = generation < window_start || generation > current.generation;
            let alternate_at_current_generation = generation == current.generation && !is_current;
            if !is_current && (outside_window || alternate_at_current_generation) {
                fs::remove_dir_all(&path)
                    .map_err(|error| format!("failed to prune generation {name}: {error}"))?;
                removed_generation_directories.push(name);
            }
        }

        let mut removed_staging_directories = 0usize;
        for path in staging_dirs {
            fs::remove_dir_all(&path)
                .map_err(|error| format!("failed to remove stale staging directory {}: {error}", path.display()))?;
            removed_staging_directories += 1;
        }
        sync_directory(&generations)?;
        Ok(EvidencePruneReport {
            current_generation: current.generation,
            retained_generation_window_start: window_start,
            removed_generation_directories,
            removed_staging_directories,
        })
    }

    #[cfg(test)]
    fn commit_with_fault(
        &self,
        hdc_index_bytes: &[u8],
        evidence_index: &EvidenceIndex,
        fault: Option<CommitFaultPoint>,
    ) -> Result<u64, String> {
        self.commit_with_identity_and_fault(hdc_index_bytes, evidence_index, None, fault)
    }

    fn commit_with_identity_and_fault(
        &self,
        hdc_index_bytes: &[u8],
        evidence_index: &EvidenceIndex,
        hdc_identity: Option<HdcStoreIdentity>,
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
        let lexical_index_bytes = if hdc_identity.is_some() {
            Some(LexicalIndex::build(evidence_index, generation)?.to_json_bytes()?)
        } else {
            None
        };
        let manifest = match hdc_identity {
            Some(identity) => {
                let lexical_bytes = lexical_index_bytes.as_deref()
                    .ok_or_else(|| "typed HDC commit is missing its lexical index".to_string())?;
                EvidenceCommitManifest::new_with_hdc_identity_and_lexical_index(
                    generation, hdc_index_bytes, &snapshot_bytes, identity, lexical_bytes,
                )?
            }
            None => EvidenceCommitManifest::new(generation, hdc_index_bytes, &snapshot_bytes)?,
        };
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
        if let Some(lexical_bytes) = lexical_index_bytes.as_deref() {
            write_new_synced(&stage.path().join(LEXICAL_FILE), lexical_bytes)?;
            if fault == Some(CommitFaultPoint::AfterLexicalIndexWrite) {
                let path = stage.keep();
                return Err(format!("simulated interruption after lexical index write; orphan staging at {}", path.display()));
            }
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
                let existing_lexical = read_optional_regular_file(&final_dir.join(LEXICAL_FILE)).ok().flatten();
                let same = read_regular_file(&final_dir.join(INDEX_FILE)).ok().as_deref() == Some(hdc_index_bytes)
                    && read_regular_file(&final_dir.join(EVIDENCE_FILE)).ok().as_deref() == Some(snapshot_bytes.as_slice())
                    && read_regular_file(&final_dir.join(MANIFEST_FILE)).ok().as_deref() == Some(manifest_bytes.as_slice())
                    && existing_lexical.as_deref() == lexical_index_bytes.as_deref();
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
        let lexical_bytes = read_optional_regular_file(&dir.join(LEXICAL_FILE))?;
        manifest.verify_lexical_index(lexical_bytes.as_deref())?;
        let evidence_index = EvidenceIndex::restore_snapshot(&snapshot_bytes)?;
        let lexical_index = match lexical_bytes.as_deref() {
            Some(bytes) => {
                let index = LexicalIndex::from_json_bytes(bytes)?;
                index.validate_for_evidence(&evidence_index, pointer.generation)?;
                Some(index)
            }
            None if manifest.hdc_store_identity().is_some() => {
                // Legacy V2 typed generations predate the manifest-bound lexical
                // artifact. Their verified evidence snapshot remains authoritative,
                // so rebuild a deterministic in-memory index for reads. The next
                // typed commit publishes a V3 generation with the index persisted
                // and digest-bound; never write this fallback into the old generation.
                Some(LexicalIndex::build(&evidence_index, pointer.generation)?)
            }
            None => None,
        };
        if let Some(identity) = manifest.hdc_store_identity() {
            identity.validate_for_recovery()?;
            // Integrity validation needs the canonical data reader, not an ANN
            // sidecar rebuild. ANN materialization is a separate query policy;
            // rebuilding it here would add cost to every exact/lexical query.
            let reader = HdcStoreReader::open(&dir.join(INDEX_FILE))
                .map_err(|error| format!("committed HDC store failed independent reopen: {error}"))?;
            if reader.generation() != identity.store_generation {
                return Err("committed HDC internal generation does not match manifest identity".into());
            }
            let checksum = reader.content_checksum();
            if reader.live_count() != identity.live_count
                || checksum.version != identity.checksum_version
                || checksum.live_count != identity.live_count
                || checksum.hex() != identity.checksum_hex
            {
                return Err("committed HDC logical content checksum does not match manifest identity".into());
            }
            let hdc_live_ids = reader.ids().collect::<BTreeSet<_>>();
            if hdc_live_ids != evidence_index.active_hdc_ids() {
                return Err("committed HDC live IDs do not match active evidence IDs".into());
            }
        }
        Ok(RecoveredEvidenceGeneration {
            generation: pointer.generation,
            hdc_index_bytes: index_bytes,
            evidence_index,
            lexical_index,
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
    use symthaea_core::hdc::BinaryHV;
    use symthaea_hdc_store::StoreConfig;
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

    fn active_evidence(id: u64, text: &str) -> EvidenceRecord {
        let source = b"canonical trajectory bytes";
        EvidenceRecord {
            schema_version: EVIDENCE_SCHEMA_VERSION.to_string(),
            evidence_id: format!("evidence-{id}"),
            hdc_id: id,
            scope_id: "web".into(),
            source: EvidenceSource {
                trajectory_id: "trajectory-typed-hdc".into(),
                step_id: Some(format!("step-{id}")),
                step_index: Some(id),
                source_sha256: sha(source),
            },
            payload: EvidencePayload::Text { value: text.into() },
            content_sha256: sha(text.as_bytes()),
            lifecycle: EvidenceLifecycle::Active,
            occurred_at_unix_ms: Some(100),
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            superseded_by: None,
        }
    }

    fn generation_path(root: &Path, generation: u64) -> PathBuf {
        let pointer: CurrentPointer = serde_json::from_slice(&fs::read(root.join(CURRENT_FILE)).unwrap()).unwrap();
        assert_eq!(pointer.generation, generation);
        root.join("generations").join(generation_directory_name(generation, &pointer.manifest_sha256))
    }

    #[test]
    fn typed_hdc_commit_binds_real_generation_checksum_and_live_ids() {
        let root = tempdir().unwrap();
        let hdc_path = root.path().join("canonical.hdc");
        let mut hdc = HdcStore::create(&hdc_path, StoreConfig::default()).unwrap();
        hdc.append(7, &BinaryHV::random(7)).unwrap();
        let mut evidence = EvidenceIndex::new();
        evidence.insert(active_evidence(7, "correct route"), b"canonical trajectory bytes").unwrap();

        let disk = EvidenceDiskStore::open(root.path().join("evidence-generations")).unwrap();
        assert_eq!(disk.commit_hdc_store(&mut hdc, &evidence).unwrap(), 1);
        let recovered = disk.recover().unwrap().unwrap();
        assert_eq!(recovered.generation, 1);
        assert_eq!(recovered.evidence_index.active_hdc_ids(), [7_u64].into_iter().collect());
        let identity = recovered.manifest.hdc_store_identity().expect("typed commit must persist HDC identity");
        assert_eq!(identity.live_count, 1);
        assert_eq!(identity.checksum_hex.len(), 16);
    }

    #[test]
    fn typed_hdc_commit_rejects_missing_or_extra_active_evidence() {
        let root = tempdir().unwrap();
        let hdc_path = root.path().join("canonical.hdc");
        let mut hdc = HdcStore::create(&hdc_path, StoreConfig::default()).unwrap();
        hdc.append(7, &BinaryHV::random(7)).unwrap();
        let disk = EvidenceDiskStore::open(root.path().join("evidence-generations")).unwrap();

        let empty = EvidenceIndex::new();
        assert!(disk.commit_hdc_store(&mut hdc, &empty).is_err());
        assert!(disk.recover().unwrap().is_none());

        let mut evidence = EvidenceIndex::new();
        evidence.insert(active_evidence(7, "bound route"), b"canonical trajectory bytes").unwrap();
        evidence.insert(active_evidence(8, "extra route"), b"canonical trajectory bytes").unwrap();
        assert!(disk.commit_hdc_store(&mut hdc, &evidence).is_err());
        assert!(disk.recover().unwrap().is_none());
    }

    #[test]
    fn typed_hdc_commit_requires_vector_deletion_before_evidence_tombstone_commit() {
        let root = tempdir().unwrap();
        let hdc_path = root.path().join("canonical.hdc");
        let mut hdc = HdcStore::create(&hdc_path, StoreConfig::default()).unwrap();
        hdc.append(7, &BinaryHV::random(7)).unwrap();
        let mut evidence = EvidenceIndex::new();
        evidence.insert(active_evidence(7, "to be deleted"), b"canonical trajectory bytes").unwrap();
        let disk = EvidenceDiskStore::open(root.path().join("evidence-generations")).unwrap();
        disk.commit_hdc_store(&mut hdc, &evidence).unwrap();

        evidence.mark_deleted(7).unwrap();
        // A tombstone cannot be committed while the actual vector remains live.
        assert!(disk.commit_hdc_store(&mut hdc, &evidence).is_err());
        assert_eq!(disk.recover().unwrap().unwrap().generation, 1);

        assert!(hdc.delete(7).unwrap());
        assert_eq!(disk.commit_hdc_store(&mut hdc, &evidence).unwrap(), 2);
        let recovered = disk.recover().unwrap().unwrap();
        assert!(recovered.evidence_index.resolve(&[7], "web", 150, Path::new("."), 5).unwrap().is_empty());
        assert!(recovered.evidence_index.insert(
            active_evidence(7, "resurrected"),
            b"canonical trajectory bytes",
        ).is_err());
    }

    #[test]
    fn typed_hdc_recovery_rejects_replaced_index_even_if_file_is_present() {
        let root = tempdir().unwrap();
        let hdc_path = root.path().join("canonical.hdc");
        let mut hdc = HdcStore::create(&hdc_path, StoreConfig::default()).unwrap();
        hdc.append(7, &BinaryHV::random(7)).unwrap();
        let mut evidence = EvidenceIndex::new();
        evidence.insert(active_evidence(7, "bound route"), b"canonical trajectory bytes").unwrap();
        let disk = EvidenceDiskStore::open(root.path().join("evidence-generations")).unwrap();
        disk.commit_hdc_store(&mut hdc, &evidence).unwrap();

        let dir = generation_path(&root.path().join("evidence-generations"), 1);
        fs::write(dir.join(INDEX_FILE), b"replacement bytes").unwrap();
        assert!(disk.recover().is_err());
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

    #[test]
    fn retention_keeps_current_and_prunes_old_and_unpublished_future_generations() {
        let root = tempdir().unwrap();
        let store = EvidenceDiskStore::open(root.path()).unwrap();
        store.commit(b"index-v1", &sample_index("old state")).unwrap();
        store.commit(b"index-v2", &sample_index("current state")).unwrap();

        // Publish a complete generation directory but stop before CURRENT.
        assert!(store.commit_with_fault(
            b"index-v3-orphan",
            &sample_index("unpublished state"),
            Some(CommitFaultPoint::AfterGenerationPublish),
        ).is_err());
        let report = store.prune_history(1).unwrap();
        assert_eq!(report.current_generation, 2);
        assert_eq!(report.retained_generation_window_start, 2);
        assert!(report.removed_generation_directories.iter().any(|name| name.contains("generation-00000000000000000001-")));
        assert!(report.removed_generation_directories.iter().any(|name| name.contains("generation-00000000000000000003-")));
        assert_eq!(store.recover().unwrap().unwrap().generation, 2);

        let entries = fs::read_dir(root.path().join("generations")).unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(entries.iter().filter(|name| name.starts_with("generation-")).count(), 1);
    }

    #[test]
    fn retention_removes_stale_staging_but_never_prunes_current() {
        let root = tempdir().unwrap();
        let store = EvidenceDiskStore::open(root.path()).unwrap();
        assert!(store.commit_with_fault(
            b"index-first-attempt",
            &sample_index("staging payload"),
            Some(CommitFaultPoint::AfterIndexWrite),
        ).is_err());
        store.commit(b"index-v1", &sample_index("current state")).unwrap();

        let report = store.prune_history(1).unwrap();
        assert_eq!(report.removed_staging_directories, 1);
        assert_eq!(store.recover().unwrap().unwrap().generation, 1);
        let entries = fs::read_dir(root.path().join("generations")).unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(entries.iter().any(|name| name.starts_with("generation-")));
        assert!(!entries.iter().any(|name| name.starts_with(".staging-")));
    }

    #[test]
    fn retention_rejects_zero_and_corrupt_current_without_deleting_history() {
        let root = tempdir().unwrap();
        let store = EvidenceDiskStore::open(root.path()).unwrap();
        assert!(store.prune_history(0).is_err());
        store.commit(b"index-v1", &sample_index("current state")).unwrap();
        let dir = generation_path(root.path(), 1);
        fs::write(root.path().join(CURRENT_FILE), b"not a current pointer").unwrap();
        assert!(store.prune_history(1).is_err());
        assert!(dir.exists(), "corrupt CURRENT must prevent retention deletion");
    }


    #[test]
    fn typed_generation_rejects_missing_or_tampered_lexical_index() {
        let root = tempdir().unwrap();
        let hdc_path = root.path().join("canonical.hdc");
        let mut hdc = HdcStore::create(&hdc_path, StoreConfig::default()).unwrap();
        hdc.append(7, &BinaryHV::random(7)).unwrap();
        let mut evidence = EvidenceIndex::new();
        evidence.insert(active_evidence(7, "cart checkout evidence"), b"canonical trajectory bytes").unwrap();
        let disk = EvidenceDiskStore::open(root.path().join("evidence-generations")).unwrap();
        disk.commit_hdc_store(&mut hdc, &evidence).unwrap();

        let dir = generation_path(&root.path().join("evidence-generations"), 1);
        let lexical_path = dir.join(LEXICAL_FILE);
        let original = fs::read(&lexical_path).unwrap();
        fs::write(&lexical_path, b"tampered lexical index").unwrap();
        assert!(disk.recover().is_err(), "tampered lexical index must fail manifest verification");

        fs::write(&lexical_path, &original).unwrap();
        fs::remove_file(&lexical_path).unwrap();
        assert!(disk.recover().is_err(), "missing manifest-bound lexical index must fail closed");
    }

    #[test]
    fn interruption_after_lexical_index_write_keeps_previous_generation_current() {
        let root = tempdir().unwrap();
        let hdc_path = root.path().join("canonical.hdc");
        let mut hdc = HdcStore::create(&hdc_path, StoreConfig::default()).unwrap();
        hdc.append(7, &BinaryHV::random(7)).unwrap();
        let mut evidence = EvidenceIndex::new();
        evidence.insert(active_evidence(7, "first generation"), b"canonical trajectory bytes").unwrap();
        let disk = EvidenceDiskStore::open(root.path().join("evidence-generations")).unwrap();
        disk.commit_hdc_store(&mut hdc, &evidence).unwrap();

        hdc.append(8, &BinaryHV::random(8)).unwrap();
        evidence.insert(active_evidence(8, "second generation"), b"canonical trajectory bytes").unwrap();
        hdc.sync_all().unwrap();
        let identity = {
            let view = hdc.read_view();
            let checksum = view.content_checksum();
            HdcStoreIdentity {
                store_generation: view.generation(),
                live_count: checksum.live_count,
                checksum_version: checksum.version,
                checksum_hex: checksum.hex(),
            }
        };
        let canonical_bytes = fs::read(hdc.path()).unwrap();
        assert!(disk.commit_with_identity_and_fault(
            &canonical_bytes,
            &evidence,
            Some(identity),
            Some(CommitFaultPoint::AfterLexicalIndexWrite),
        ).is_err());

        let recovered = disk.recover().unwrap().unwrap();
        assert_eq!(recovered.generation, 1, "unpublished lexical artifact must not advance CURRENT");
        assert_eq!(recovered.evidence_index.active_hdc_ids(), [7_u64].into_iter().collect());
        assert!(recovered.lexical_index.is_some());
    }

    #[test]
    fn legacy_v2_typed_generation_recovers_and_rebuilds_lexical_index_in_memory() {
        let root = tempdir().unwrap();
        let hdc_path = root.path().join("canonical.hdc");
        let mut hdc = HdcStore::create(&hdc_path, StoreConfig::default()).unwrap();
        hdc.append(7, &BinaryHV::random(7)).unwrap();
        let mut evidence = EvidenceIndex::new();
        evidence.insert(
            active_evidence(7, "legacy cart checkout evidence"),
            b"canonical trajectory bytes",
        ).unwrap();

        let store_root = root.path().join("evidence-generations");
        let disk = EvidenceDiskStore::open(&store_root).unwrap();
        disk.commit_hdc_store(&mut hdc, &evidence).unwrap();

        // Convert the typed V3 fixture into the exact legacy V2 shape: the
        // HDC/evidence manifest fields remain intact, but no lexical digest or
        // lexical artifact exists in the old generation.
        let old_dir = generation_path(&store_root, 1);
        let old_pointer: CurrentPointer =
            serde_json::from_slice(&fs::read(store_root.join(CURRENT_FILE)).unwrap()).unwrap();
        let mut manifest_value: serde_json::Value =
            serde_json::from_slice(&fs::read(old_dir.join(MANIFEST_FILE)).unwrap()).unwrap();
        manifest_value["schema_version"] = serde_json::Value::String(
            "BROCA-EVIDENCE-COMMIT-MANIFEST-V2".into(),
        );
        manifest_value.as_object_mut().unwrap().remove("lexical_index_sha256");
        let legacy_manifest = serde_json::to_vec(&manifest_value).unwrap();
        let legacy_digest = sha256_hex(&legacy_manifest);
        fs::write(old_dir.join(MANIFEST_FILE), &legacy_manifest).unwrap();
        fs::remove_file(old_dir.join(LEXICAL_FILE)).unwrap();

        let new_dir = store_root.join("generations").join(generation_directory_name(1, &legacy_digest));
        fs::rename(&old_dir, &new_dir).unwrap();
        let legacy_pointer = CurrentPointer {
            schema_version: CURRENT_SCHEMA_VERSION.into(),
            generation: old_pointer.generation,
            manifest_sha256: legacy_digest,
        };
        fs::write(store_root.join(CURRENT_FILE), serde_json::to_vec(&legacy_pointer).unwrap()).unwrap();

        let recovered = disk.recover().unwrap().unwrap();
        assert_eq!(recovered.generation, 1);
        assert!(recovered.lexical_index.is_some(), "legacy lexical index should be rebuilt from verified snapshot");
        assert_eq!(recovered.manifest.lexical_index_sha256(), None);

        let result = disk.query_committed_text_context(
            "cart checkout", 8, "web", 150, Path::new("."), 4,
        ).unwrap().unwrap();
        assert_eq!(result.generation, 1);
        assert_eq!(result.resolved_context[0].item.value, "legacy cart checkout evidence");
    }

    #[test]
    fn committed_text_query_returns_grounded_lexical_context_with_provenance() {
        let root = tempdir().unwrap();
        let hdc_path = root.path().join("canonical.hdc");
        let mut hdc = HdcStore::create(&hdc_path, StoreConfig::default()).unwrap();
        hdc.append(7, &BinaryHV::random(7)).unwrap();
        hdc.append(8, &BinaryHV::random(8)).unwrap();

        let mut evidence = EvidenceIndex::new();
        evidence.insert(active_evidence(7, "cart checkout button opens the shopping cart"), b"canonical trajectory bytes").unwrap();
        let mut unrelated = active_evidence(8, "mountain trails and hiking boots");
        unrelated.scope_id = "enterprise".into();
        evidence.insert(unrelated, b"canonical trajectory bytes").unwrap();

        let disk = EvidenceDiskStore::open(root.path().join("evidence-generations")).unwrap();
        disk.commit_hdc_store(&mut hdc, &evidence).unwrap();
        let result = disk.query_committed_text_context(
            "Where is the cart checkout button?", 8, "web", 150, Path::new("."), 4,
        ).unwrap().expect("typed committed generation should be queryable");

        assert_eq!(result.generation, 1);
        assert_eq!(result.resolved_context.len(), 1);
        assert_eq!(result.resolved_context[0].hdc_id, 7);
        assert_eq!(result.resolved_context[0].item.value, "cart checkout button opens the shopping cart");
        assert_eq!(result.context_items()[0].kind, super::super::memory_evidence::MemoryContextType::Text);
    }

    #[test]
    fn committed_text_query_refuses_opaque_fixture_generation() {
        let root = tempdir().unwrap();
        let disk = EvidenceDiskStore::open(root.path()).unwrap();
        let evidence = sample_index("fixture-only text");
        disk.commit(b"not a real HDC index", &evidence).unwrap();
        let error = disk.query_committed_text_context(
            "fixture", 4, "web", 150, Path::new("."), 5,
        ).unwrap_err();
        assert!(error.contains("typed HDC identity"));
    }

    #[test]
    fn committed_query_uses_one_typed_generation_and_filters_scope() {
        let root = tempdir().unwrap();
        let hdc_path = root.path().join("canonical.hdc");
        let mut hdc = HdcStore::create(&hdc_path, StoreConfig::default()).unwrap();
        let query_hv = BinaryHV::random(7);
        hdc.append(7, &query_hv).unwrap();
        hdc.append(8, &BinaryHV::random(8)).unwrap();

        let mut evidence = EvidenceIndex::new();
        evidence.insert(active_evidence(7, "correct route"), b"canonical trajectory bytes").unwrap();
        let mut enterprise_only = active_evidence(8, "enterprise-only route");
        enterprise_only.scope_id = "enterprise".into();
        evidence.insert(enterprise_only, b"canonical trajectory bytes").unwrap();

        let disk = EvidenceDiskStore::open(root.path().join("evidence-generations")).unwrap();
        disk.commit_hdc_store(&mut hdc, &evidence).unwrap();
        let result = disk.query_committed_context(
            &query_hv, 2, "web", 150, Path::new("."), 5,
        ).unwrap().expect("typed committed generation should be queryable");

        assert_eq!(result.generation, 1);
        assert_eq!(result.candidates.len(), 2);
        assert_eq!(result.candidates[0].0, 7);
        assert_eq!(result.resolved_context.len(), 1);
        assert_eq!(result.resolved_context[0].hdc_id, 7);
        assert_eq!(result.resolved_context[0].evidence_id, "evidence-7");
        assert_eq!(result.resolved_context[0].source.trajectory_id, "trajectory-typed-hdc");
        assert_eq!(result.context_items()[0].value, "correct route");
    }

    #[test]
    fn committed_query_refuses_opaque_fixture_generation() {
        let root = tempdir().unwrap();
        let disk = EvidenceDiskStore::open(root.path()).unwrap();
        let evidence = sample_index("fixture-only text");
        disk.commit(b"not a real HDC index", &evidence).unwrap();

        let error = disk.query_committed_context(
            &BinaryHV::random(7), 4, "web", 150, Path::new("."), 5,
        ).unwrap_err();
        assert!(error.contains("typed HDC identity"));
    }

    #[test]
    fn committed_query_distinguishes_uninitialized_store_from_corruption() {
        let root = tempdir().unwrap();
        let disk = EvidenceDiskStore::open(root.path()).unwrap();
        assert!(disk.query_committed_context(
            &BinaryHV::random(7), 4, "web", 150, Path::new("."), 5,
        ).unwrap().is_none());
        fs::write(root.path().join(CURRENT_FILE), b"{broken").unwrap();
        assert!(disk.query_committed_context(
            &BinaryHV::random(7), 4, "web", 150, Path::new("."), 5,
        ).is_err());
    }

}
