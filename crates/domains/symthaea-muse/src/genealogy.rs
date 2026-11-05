// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! SQLite-backed genealogy ledger for kept Muse pieces.
//!
//! V1 scope (deliberately minimal): genealogy manifests are allocated only
//! at *keep* time (`muse_studio`'s `/api/keeper/{id}` handler), never for
//! ephemeral in-session candidates -- most candidates in a batch are never
//! heard again, and `Studio.candidates`'s ids reset on every server
//! restart, so addressing every candidate would flood the ledger with
//! addresses for music nobody kept. Every manifest allocated today is a
//! [`symthaea_muse_protocol::GenealogyRelation::Root`]: nothing in the
//! compose/keep flow yet records "this kept piece was derived from that
//! kept piece," so there is no real parent to report. The schema and the
//! `GenealogyRelation`/`GenealogyOrigin` enums already support non-root
//! relations for when that derivation signal exists.
//!
//! Storage split follows the existing `data/taste/audio/<key>/...` layout
//! this server already uses for kept audio/MIDI: this ledger stores only
//! structured metadata and sha256 pointers, never the audio bytes
//! themselves. Correction: that existing layout is NOT content-addressed
//! today -- `keeper_artifact_key()` derives its key from
//! `(unix_nanos, pid, candidate_id, sequence)`, not from a content hash.
//! It's collision-resistant and unique, but two identical pieces kept
//! twice get two different keys. This module verifies and stores the real
//! hashes (`recipe_sha256`/`score_sha256`/`audio_sha256`) without assuming
//! the storage layout is CAS; introducing a true `objects/sha256/ab/cdef…`
//! object store is a real, separate future change, not bundled here.
//!
//! Namespace scope: every manifest here uses `"C"` (compositional
//! lineage) -- the only lineage tier this codebase actually persists
//! independently today. `"R"` (rendition lineage) is reserved for if/when
//! a rendition ever gets its own manifest independent of its composition's
//! -- it is not promised or written by any code path yet, so it isn't
//! named here as an active namespace.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Mutex;

use rusqlite::{Connection, OptionalExtension, Row, params};
use sha2::{Digest, Sha256};
use symthaea_muse_protocol::{
    GenealogyEdgeKindV1, GenealogyLineageVerificationV1, GenealogyManifest, GenealogyOrigin,
    GenealogyParentEdgeV1, GenealogyParentInputV1, GenealogyRelation, RmiArtifactBindingV1,
};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS genealogy_manifests (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id          INTEGER NOT NULL,
    parent_id          INTEGER REFERENCES genealogy_manifests(id),
    namespace          TEXT NOT NULL,
    relation_json      TEXT NOT NULL,
    origin_json        TEXT NOT NULL,
    audio_key          TEXT NOT NULL UNIQUE,
    recipe_sha256      TEXT NOT NULL,
    score_sha256       TEXT,
    audio_sha256       TEXT NOT NULL,
    manifest_sha256    TEXT NOT NULL UNIQUE,
    created_at_unix_ms INTEGER NOT NULL,
    lineage_sha256     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_genealogy_parent ON genealogy_manifests(parent_id);
CREATE INDEX IF NOT EXISTS idx_genealogy_family ON genealogy_manifests(family_id);
";

const EDGE_SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS genealogy_parent_edges (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    child_manifest_id   INTEGER NOT NULL REFERENCES genealogy_manifests(id) ON DELETE CASCADE,
    parent_manifest_id  INTEGER NOT NULL REFERENCES genealogy_manifests(id) ON DELETE RESTRICT,
    kind_json           TEXT NOT NULL,
    source_event_id     TEXT,
    contributor_ref     TEXT,
    authority_receipt_ref TEXT,
    idempotency_key     TEXT NOT NULL UNIQUE,
    edge_sha256         TEXT NOT NULL UNIQUE,
    created_at_unix_ms  INTEGER NOT NULL,
    CHECK (child_manifest_id <> parent_manifest_id)
);
CREATE INDEX IF NOT EXISTS idx_genealogy_edge_parent ON genealogy_parent_edges(parent_manifest_id);
CREATE INDEX IF NOT EXISTS idx_genealogy_edge_child ON genealogy_parent_edges(child_manifest_id);
CREATE TABLE IF NOT EXISTS genealogy_lineage_operations (
    idempotency_key   TEXT PRIMARY KEY,
    child_manifest_id INTEGER NOT NULL UNIQUE REFERENCES genealogy_manifests(id) ON DELETE CASCADE,
    operation_sha256  TEXT NOT NULL
);
";

/// Apply the current schema and additive migrations while preserving existing
/// single-parent manifests. A blank lineage digest marks a legacy/v1 manifest.
fn initialize_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(SCHEMA)?;
    let mut has_lineage_column = false;
    {
        let mut statement = conn.prepare("PRAGMA table_info(genealogy_manifests)")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let column_name: String = row.get(1)?;
            if column_name == "lineage_sha256" {
                has_lineage_column = true;
                break;
            }
        }
    }
    if !has_lineage_column {
        conn.execute_batch(
            "ALTER TABLE genealogy_manifests
             ADD COLUMN lineage_sha256 TEXT NOT NULL DEFAULT '';",
        )?;
    }
    conn.execute_batch(EDGE_SCHEMA)?;
    Ok(())
}

const SELECT_COLUMNS: &str = "id, family_id, parent_id, namespace, relation_json, origin_json,
     audio_key, recipe_sha256, score_sha256, audio_sha256, manifest_sha256, created_at_unix_ms";

/// Failure while re-reading or verifying exact keeper bundle bytes before
/// writing their genealogy manifest.
#[derive(Debug)]
pub enum GenealogyBundleError {
    /// A required artifact could not be read from the committed bundle.
    Io(std::io::Error),
    /// Persisted bytes differ from the exact payload that Studio committed.
    ByteMismatch { artifact: &'static str },
    /// A persisted artifact no longer matches the hash in its manifest.
    ManifestMismatch { artifact: &'static str },
    /// The keeper has no genealogy manifest.
    MissingManifest { audio_key: String },
    /// RMI event IDs or the reference shape are invalid.
    InvalidRmiBinding(&'static str),
    /// The verified content hashes could not be committed to SQLite.
    Database(rusqlite::Error),
}

impl std::fmt::Display for GenealogyBundleError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "keeper bundle read failed: {error}"),
            Self::ByteMismatch { artifact } => {
                write!(formatter, "persisted keeper artifact differs: {artifact}")
            }
            Self::ManifestMismatch { artifact } => {
                write!(
                    formatter,
                    "keeper artifact no longer matches genealogy manifest: {artifact}"
                )
            }
            Self::MissingManifest { audio_key } => {
                write!(formatter, "keeper has no genealogy manifest: {audio_key}")
            }
            Self::InvalidRmiBinding(reason) => {
                write!(formatter, "invalid RMI artifact binding: {reason}")
            }
            Self::Database(error) => {
                write!(formatter, "genealogy database operation failed: {error}")
            }
        }
    }
}

impl std::error::Error for GenealogyBundleError {}

/// Failure while allocating or independently verifying an immutable
/// multi-parent genealogy subgraph.
#[derive(Debug)]
pub enum GenealogyLineageError {
    /// SQLite persistence or query failure.
    Database(rusqlite::Error),
    /// A parent, relation, idempotency key, or input edge is invalid.
    InvalidInput(&'static str),
    /// An idempotency key was reused with a different operation payload.
    IdempotencyConflict,
    /// Stored lineage edges or their manifest commitment no longer agree.
    CorruptLineage,
}

impl std::fmt::Display for GenealogyLineageError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Database(error) => write!(formatter, "genealogy lineage database error: {error}"),
            Self::InvalidInput(reason) => write!(formatter, "invalid genealogy lineage input: {reason}"),
            Self::IdempotencyConflict => write!(formatter, "genealogy lineage idempotency key conflicts with a prior operation"),
            Self::CorruptLineage => write!(formatter, "genealogy lineage edge set or manifest commitment is corrupt"),
        }
    }
}

impl std::error::Error for GenealogyLineageError {}

/// Read-only result after resolving a keeper's genealogy manifest and
/// independently re-hashing every artifact currently present in its bundle.
///
/// This is suitable as a provenance reference for downstream RMI/evaluation
/// records. It does not authenticate a musician, grant consent, or establish
/// authorship or musical quality.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedArtifactBundleV1 {
    /// Existing durable genealogy record.
    pub manifest: GenealogyManifest,
    /// Exact recipe bytes re-read from recipe.json.
    pub recipe_sha256: String,
    /// Exact score bytes re-read from score.json, when the manifest binds one.
    pub score_sha256: Option<String>,
    /// Exact audio bytes re-read from audio.wav.
    pub audio_sha256: String,
    /// Number of bytes in recipe.json.
    pub recipe_bytes_len: u64,
    /// Number of bytes in score.json, when present and bound.
    pub score_bytes_len: Option<u64>,
    /// Number of bytes in audio.wav.
    pub audio_bytes_len: u64,
}

impl VerifiedArtifactBundleV1 {
    /// Bind this independently verified keeper bundle to one RMI source event
    /// and one action event. The returned DTO carries the genealogy manifest
    /// commitment and exact byte lengths; it remains a reference, not a
    /// signature or a consent/authorship assertion.
    pub fn bind_rmi_events(
        &self,
        source_event_id: &str,
        action_event_id: &str,
    ) -> Result<RmiArtifactBindingV1, GenealogyBundleError> {
        let binding = RmiArtifactBindingV1 {
            schema_version: 1,
            source_event_id: source_event_id.to_owned(),
            action_event_id: action_event_id.to_owned(),
            genealogy: self.manifest.clone(),
            recipe_bytes_len: self.recipe_bytes_len,
            score_bytes_len: self.score_bytes_len,
            audio_bytes_len: self.audio_bytes_len,
        };
        binding
            .validate_reference()
            .map_err(GenealogyBundleError::InvalidRmiBinding)?;
        Ok(binding)
    }
}

pub struct GenealogyStore {
    conn: Mutex<Connection>,
}

impl GenealogyStore {
    /// Allocate a root manifest from the exact bytes of each persisted artifact.
    ///
    /// Unlike allocate_root, this entry point does not accept caller-supplied
    /// digest strings. It recomputes SHA-256 from the recipe, optional score,
    /// and audio byte slices immediately before persisting the manifest.
    /// The storage locator remains separate from content identity.
    /// Resolve the existing manifest for a keeper and independently verify
    /// its current recipe, optional score, and audio files against the stored
    /// hashes. Call this before attaching a keeper to a downstream RMI record.
    ///
    /// Unlike allocate_root_from_bundle, this read-only operation needs no
    /// expected caller buffers: the existing manifest is the reference. A
    /// changed, missing, or substituted file fails closed.
    pub fn verify_bundle(
        &self,
        audio_key: &str,
        bundle_directory: &Path,
    ) -> Result<VerifiedArtifactBundleV1, GenealogyBundleError> {
        let manifest = self
            .manifest_by_audio_key(audio_key)
            .map_err(GenealogyBundleError::Database)?
            .ok_or_else(|| GenealogyBundleError::MissingManifest {
                audio_key: audio_key.to_owned(),
            })?;

        let recipe_bytes = std::fs::read(bundle_directory.join("recipe.json"))
            .map_err(GenealogyBundleError::Io)?;
        let audio_bytes =
            std::fs::read(bundle_directory.join("audio.wav")).map_err(GenealogyBundleError::Io)?;
        let score_bytes = match manifest.score_sha256.as_ref() {
            Some(_) => Some(
                std::fs::read(bundle_directory.join("score.json"))
                    .map_err(GenealogyBundleError::Io)?,
            ),
            None => None,
        };

        let recipe_sha256 = artifact_sha256_hex(&recipe_bytes);
        if recipe_sha256 != manifest.recipe_sha256 {
            return Err(GenealogyBundleError::ManifestMismatch {
                artifact: "recipe.json",
            });
        }
        let audio_sha256 = artifact_sha256_hex(&audio_bytes);
        if audio_sha256 != manifest.audio_sha256 {
            return Err(GenealogyBundleError::ManifestMismatch {
                artifact: "audio.wav",
            });
        }
        let score_sha256 = score_bytes.as_deref().map(artifact_sha256_hex);
        if score_sha256 != manifest.score_sha256 {
            return Err(GenealogyBundleError::ManifestMismatch {
                artifact: "score.json",
            });
        }

        Ok(VerifiedArtifactBundleV1 {
            manifest,
            recipe_sha256,
            score_sha256,
            audio_sha256,
            recipe_bytes_len: recipe_bytes.len() as u64,
            score_bytes_len: score_bytes.as_ref().map(|bytes| bytes.len() as u64),
            audio_bytes_len: audio_bytes.len() as u64,
        })
    }

    /// Re-read the published keeper bundle, compare every byte vector with
    /// the payload passed to Studio's atomic bundle writer, and only then
    /// commit a genealogy manifest from the re-read bytes.
    ///
    /// Hashes identify the bytes actually present after the staging-directory
    /// rename, not just buffers that were intended to be written.
    pub fn allocate_root_from_bundle(
        &self,
        origin: &GenealogyOrigin,
        audio_key: &str,
        bundle_directory: &Path,
        expected_recipe_bytes: &[u8],
        expected_score_bytes: &[u8],
        expected_audio_bytes: &[u8],
        created_at_unix_ms: u64,
    ) -> Result<GenealogyManifest, GenealogyBundleError> {
        let recipe_bytes = std::fs::read(bundle_directory.join("recipe.json"))
            .map_err(GenealogyBundleError::Io)?;
        let score_bytes =
            std::fs::read(bundle_directory.join("score.json")).map_err(GenealogyBundleError::Io)?;
        let audio_bytes =
            std::fs::read(bundle_directory.join("audio.wav")).map_err(GenealogyBundleError::Io)?;

        if recipe_bytes != expected_recipe_bytes {
            return Err(GenealogyBundleError::ByteMismatch {
                artifact: "recipe.json",
            });
        }
        if score_bytes != expected_score_bytes {
            return Err(GenealogyBundleError::ByteMismatch {
                artifact: "score.json",
            });
        }
        if audio_bytes != expected_audio_bytes {
            return Err(GenealogyBundleError::ByteMismatch {
                artifact: "audio.wav",
            });
        }

        self.allocate_root_from_bytes(
            origin,
            audio_key,
            &recipe_bytes,
            Some(&score_bytes),
            &audio_bytes,
            created_at_unix_ms,
        )
        .map_err(GenealogyBundleError::Database)
    }

    pub fn allocate_root_from_bytes(
        &self,
        origin: &GenealogyOrigin,
        audio_key: &str,
        recipe_bytes: &[u8],
        score_bytes: Option<&[u8]>,
        audio_bytes: &[u8],
        created_at_unix_ms: u64,
    ) -> rusqlite::Result<GenealogyManifest> {
        let recipe_sha256 = artifact_sha256_hex(recipe_bytes);
        let score_sha256 = score_bytes.map(artifact_sha256_hex);
        let audio_sha256 = artifact_sha256_hex(audio_bytes);
        self.allocate_root(
            origin,
            audio_key,
            &recipe_sha256,
            score_sha256.as_deref(),
            &audio_sha256,
            created_at_unix_ms,
        )
    }

    /// Allocate one derived artifact with an immutable set of source parents.
    ///
    /// A single parent keeps the legacy `parent_id`/`family_id` projection.
    /// Multiple parents produce a merge node with `parent_id = NULL` and
    /// `family_id = child.id`; the edge table is then authoritative. The
    /// operation is atomic across the manifest, all edges, and idempotency
    /// record. Exact retries return the same manifest; a changed payload under
    /// the same idempotency key fails closed.
    pub fn allocate_derived_from_bytes(
        &self,
        relation: &GenealogyRelation,
        origin: &GenealogyOrigin,
        audio_key: &str,
        recipe_bytes: &[u8],
        score_bytes: Option<&[u8]>,
        audio_bytes: &[u8],
        parents: &[GenealogyParentInputV1],
        allocation_idempotency_key: &str,
        created_at_unix_ms: u64,
    ) -> Result<GenealogyManifest, GenealogyLineageError> {
        if matches!(relation, GenealogyRelation::Root) {
            return Err(GenealogyLineageError::InvalidInput(
                "derived allocations cannot use the Root relation",
            ));
        }
        if audio_key.trim().is_empty()
            || allocation_idempotency_key.trim().is_empty()
            || recipe_bytes.is_empty()
            || audio_bytes.is_empty()
            || parents.is_empty()
        {
            return Err(GenealogyLineageError::InvalidInput(
                "audio key, idempotency key, recipe/audio bytes, and parents are required",
            ));
        }

        let mut parent_ids = BTreeSet::new();
        let mut edge_keys = BTreeSet::new();
        for parent in parents {
            if parent.parent_manifest_id <= 0
                || parent.idempotency_key.trim().is_empty()
                || !edge_keys.insert(parent.idempotency_key.as_str())
                || !parent_ids.insert(parent.parent_manifest_id)
                || parent.source_event_id.as_deref().is_some_and(|value| value.trim().is_empty())
                || parent.contributor_ref.as_deref().is_some_and(|value| value.trim().is_empty())
                || parent.authority_receipt_ref.as_deref().is_some_and(|value| value.trim().is_empty())
            {
                return Err(GenealogyLineageError::InvalidInput(
                    "parent IDs and edge keys must be unique and all optional references non-empty",
                ));
            }
        }

        let recipe_sha256 = artifact_sha256_hex(recipe_bytes);
        let score_sha256 = score_bytes.map(artifact_sha256_hex);
        let audio_sha256 = artifact_sha256_hex(audio_bytes);
        let namespace = "C".to_owned();
        let relation_json = serde_json::to_string(relation)
            .map_err(|_| GenealogyLineageError::InvalidInput("relation serialization failed"))?;
        let origin_json = serde_json::to_string(origin)
            .map_err(|_| GenealogyLineageError::InvalidInput("origin serialization failed"))?;

        let mut conn = self.conn.lock().map_err(|_| {
            GenealogyLineageError::InvalidInput("genealogy database mutex is poisoned")
        })?;
        let tx = conn.transaction().map_err(GenealogyLineageError::Database)?;

        let mut resolved: Vec<(GenealogyParentInputV1, GenealogyManifest)> =
            Vec::with_capacity(parents.len());
        for parent in parents {
            let manifest = tx
                .query_row(
                    &format!("SELECT {SELECT_COLUMNS} FROM genealogy_manifests WHERE id = ?1"),
                    params![parent.parent_manifest_id],
                    row_to_manifest,
                )
                .optional()
                .map_err(GenealogyLineageError::Database)?
                .ok_or(GenealogyLineageError::InvalidInput(
                    "a referenced parent manifest does not exist",
                ))?;
            resolved.push((parent.clone(), manifest));
        }
        resolved.sort_by(|left, right| {
            (
                left.1.id,
                &left.0.idempotency_key,
                &left.0.source_event_id,
            )
                .cmp(&(
                    right.1.id,
                    &right.0.idempotency_key,
                    &right.0.source_event_id,
                ))
        });

        let lineage_sha256 = lineage_content_hash(&resolved);
        let manifest_sha256 = derived_manifest_content_hash(
            &namespace,
            &relation_json,
            &origin_json,
            audio_key,
            &recipe_sha256,
            score_sha256.as_deref(),
            &audio_sha256,
            &lineage_sha256,
        );
        let operation_sha256 = manifest_sha256.clone();

        if let Some((existing_child_id, existing_operation_sha256)) = tx
            .query_row(
                "SELECT child_manifest_id, operation_sha256
                 FROM genealogy_lineage_operations WHERE idempotency_key = ?1",
                params![allocation_idempotency_key],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()
            .map_err(GenealogyLineageError::Database)?
        {
            if existing_operation_sha256 != operation_sha256 {
                return Err(GenealogyLineageError::IdempotencyConflict);
            }
            let existing = tx
                .query_row(
                    &format!("SELECT {SELECT_COLUMNS} FROM genealogy_manifests WHERE id = ?1"),
                    params![existing_child_id],
                    row_to_manifest,
                )
                .optional()
                .map_err(GenealogyLineageError::Database)?
                .ok_or(GenealogyLineageError::CorruptLineage)?;
            tx.commit().map_err(GenealogyLineageError::Database)?;
            return Ok(existing);
        }

        let parent_projection = (resolved.len() == 1).then_some(resolved[0].1.id);
        let legacy_family_projection = if resolved.len() == 1 {
            resolved[0].1.family_id
        } else {
            0
        };
        tx.execute(
            "INSERT INTO genealogy_manifests
             (family_id, parent_id, namespace, relation_json, origin_json, audio_key,
              recipe_sha256, score_sha256, audio_sha256, manifest_sha256,
              created_at_unix_ms, lineage_sha256)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                legacy_family_projection,
                parent_projection,
                namespace,
                relation_json,
                origin_json,
                audio_key,
                recipe_sha256,
                score_sha256,
                audio_sha256,
                manifest_sha256,
                created_at_unix_ms as i64,
                lineage_sha256,
            ],
        )
        .map_err(GenealogyLineageError::Database)?;
        let child_id = tx.last_insert_rowid();
        let family_id = if resolved.len() == 1 {
            legacy_family_projection
        } else {
            child_id
        };
        tx.execute(
            "UPDATE genealogy_manifests SET family_id = ?1 WHERE id = ?1",
            params![family_id],
        )
        .map_err(GenealogyLineageError::Database)?;

        for (parent, manifest) in &resolved {
            let kind_json = serde_json::to_string(&parent.kind)
                .map_err(|_| GenealogyLineageError::InvalidInput("edge kind serialization failed"))?;
            let edge_sha256 = genealogy_edge_content_hash(
                manifest.id,
                child_id,
                &manifest.manifest_sha256,
                &manifest_sha256,
                &kind_json,
                parent.source_event_id.as_deref(),
                parent.contributor_ref.as_deref(),
                parent.authority_receipt_ref.as_deref(),
                &parent.idempotency_key,
            );
            tx.execute(
                "INSERT INTO genealogy_parent_edges
                 (child_manifest_id, parent_manifest_id, kind_json, source_event_id,
                  contributor_ref, authority_receipt_ref, idempotency_key, edge_sha256,
                  created_at_unix_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    child_id,
                    manifest.id,
                    kind_json,
                    parent.source_event_id,
                    parent.contributor_ref,
                    parent.authority_receipt_ref,
                    parent.idempotency_key,
                    edge_sha256,
                    created_at_unix_ms as i64,
                ],
            )
            .map_err(GenealogyLineageError::Database)?;
        }
        tx.execute(
            "INSERT INTO genealogy_lineage_operations
             (idempotency_key, child_manifest_id, operation_sha256)
             VALUES (?1, ?2, ?3)",
            params![allocation_idempotency_key, child_id, operation_sha256],
        )
        .map_err(GenealogyLineageError::Database)?;

        tx.commit().map_err(GenealogyLineageError::Database)?;
        Ok(GenealogyManifest {
            id: child_id,
            family_id,
            parent_id: parent_projection,
            namespace,
            relation: relation.clone(),
            origin: origin.clone(),
            audio_key: audio_key.to_owned(),
            recipe_sha256,
            score_sha256,
            audio_sha256,
            manifest_sha256,
            created_at_unix_ms,
        })
    }

    /// Opens (creating if absent) the genealogy ledger at `path`, applying
    /// the schema. A fresh file gets the schema for free; re-opening an
    /// existing one is idempotent (`CREATE ... IF NOT EXISTS`).
    pub fn open(path: &Path) -> rusqlite::Result<Self> {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let conn = Connection::open(path)?;
        // WAL + NORMAL is the standard durability/throughput tradeoff for
        // WAL mode (SQLite docs); `FULL` sync is stronger but adds an
        // fsync this single-local-user workload doesn't need. busy_timeout
        // avoids spurious SQLITE_BUSY under this process's own concurrent
        // handlers; foreign_keys enforces the parent_id reference.
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA busy_timeout = 5000;",
        )?;
        initialize_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Allocate a new root-family manifest for a just-kept piece.
    /// `audio_key` must be unique (the caller's existing keeper artifact
    /// key already is, by construction) -- a repeat key is a caller bug,
    /// not a legitimate "kept twice," and surfaces as a rusqlite constraint
    /// error rather than silently succeeding.
    pub fn allocate_root(
        &self,
        origin: &GenealogyOrigin,
        audio_key: &str,
        recipe_sha256: &str,
        score_sha256: Option<&str>,
        audio_sha256: &str,
        created_at_unix_ms: u64,
    ) -> rusqlite::Result<GenealogyManifest> {
        let relation = GenealogyRelation::Root;
        let namespace = "C";
        let relation_json =
            serde_json::to_string(&relation).expect("GenealogyRelation always serializes");
        let origin_json = serde_json::to_string(origin).expect("GenealogyOrigin always serializes");
        let manifest_sha256 = manifest_content_hash(
            namespace,
            &relation_json,
            &origin_json,
            audio_key,
            recipe_sha256,
            score_sha256,
            audio_sha256,
        );

        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO genealogy_manifests
             (family_id, parent_id, namespace, relation_json, origin_json, audio_key,
              recipe_sha256, score_sha256, audio_sha256, manifest_sha256, created_at_unix_ms)
             VALUES (0, NULL, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                namespace,
                relation_json,
                origin_json,
                audio_key,
                recipe_sha256,
                score_sha256,
                audio_sha256,
                manifest_sha256,
                created_at_unix_ms as i64,
            ],
        )?;
        let id = tx.last_insert_rowid();
        // A root's family_id is its own id -- only knowable after insert,
        // hence the follow-up UPDATE inside the same transaction.
        tx.execute(
            "UPDATE genealogy_manifests SET family_id = ?1 WHERE id = ?1",
            params![id],
        )?;
        tx.commit()?;

        Ok(GenealogyManifest {
            id,
            family_id: id,
            parent_id: None,
            namespace: namespace.to_string(),
            relation,
            origin: origin.clone(),
            audio_key: audio_key.to_string(),
            recipe_sha256: recipe_sha256.to_string(),
            score_sha256: score_sha256.map(str::to_string),
            audio_sha256: audio_sha256.to_string(),
            manifest_sha256,
            created_at_unix_ms,
        })
    }

    pub fn manifest(&self, id: i64) -> rusqlite::Result<Option<GenealogyManifest>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            &format!("SELECT {SELECT_COLUMNS} FROM genealogy_manifests WHERE id = ?1"),
            params![id],
            row_to_manifest,
        )
        .optional()
    }

    pub fn manifest_by_audio_key(
        &self,
        audio_key: &str,
    ) -> rusqlite::Result<Option<GenealogyManifest>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            &format!("SELECT {SELECT_COLUMNS} FROM genealogy_manifests WHERE audio_key = ?1"),
            params![audio_key],
            row_to_manifest,
        )
        .optional()
    }

    /// Read the distinct source manifests for this result. This combines the
    /// legacy single-parent projection with all immutable parent-edge rows.
    pub fn parents(&self, child_id: i64) -> rusqlite::Result<Vec<GenealogyManifest>> {
        let conn = self.conn.lock().unwrap();
        let mut parent_ids = BTreeSet::new();
        if let Some(parent_id) = conn
            .query_row(
                "SELECT parent_id FROM genealogy_manifests WHERE id = ?1",
                params![child_id],
                |row| row.get::<_, Option<i64>>(0),
            )
            .optional()?
            .flatten()
        {
            parent_ids.insert(parent_id);
        }
        {
            let mut statement = conn.prepare(
                "SELECT parent_manifest_id FROM genealogy_parent_edges
                 WHERE child_manifest_id = ?1 ORDER BY parent_manifest_id",
            )?;
            let rows = statement.query_map(params![child_id], |row| row.get::<_, i64>(0))?;
            for row in rows {
                parent_ids.insert(row?);
            }
        }
        parent_ids
            .into_iter()
            .map(|parent_id| {
                conn.query_row(
                    &format!("SELECT {SELECT_COLUMNS} FROM genealogy_manifests WHERE id = ?1"),
                    params![parent_id],
                    row_to_manifest,
                )
            })
            .collect()
    }

    /// Return all explicitly persisted parent edges for a derived manifest.
    /// Legacy single-parent manifests have no edge rows and remain readable
    /// through `parents()` and the `parent_id` projection.
    pub fn parent_edges(
        &self,
        child_id: i64,
    ) -> rusqlite::Result<Vec<GenealogyParentEdgeV1>> {
        let conn = self.conn.lock().unwrap();
        let mut statement = conn.prepare(&format!(
            "SELECT {EDGE_SELECT_COLUMNS} FROM genealogy_parent_edges
             WHERE child_manifest_id = ?1 ORDER BY parent_manifest_id, idempotency_key"
        ))?;
        statement
            .query_map(params![child_id], stored_edge_from_row)?
            .collect()
    }

    /// Independently verify the stored parent graph and its content
    /// commitments. Legacy manifests retain their v1 hash and single-parent
    /// projection. Derived v2 manifests must have an exact, non-empty edge set
    /// whose digest is committed by the child manifest.
    pub fn verify_lineage(
        &self,
        child_id: i64,
    ) -> Result<GenealogyLineageVerificationV1, GenealogyLineageError> {
        let conn = self.conn.lock().map_err(|_| {
            GenealogyLineageError::InvalidInput("genealogy database mutex is poisoned")
        })?;
        let manifest = conn
            .query_row(
                &format!("SELECT {SELECT_COLUMNS} FROM genealogy_manifests WHERE id = ?1"),
                params![child_id],
                row_to_manifest,
            )
            .optional()
            .map_err(GenealogyLineageError::Database)?
            .ok_or(GenealogyLineageError::InvalidInput(
                "child manifest does not exist",
            ))?;
        let (relation_json, origin_json, stored_lineage_sha256): (String, String, String) = conn
            .query_row(
                "SELECT relation_json, origin_json, lineage_sha256
                 FROM genealogy_manifests WHERE id = ?1",
                params![child_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .map_err(GenealogyLineageError::Database)?;

        let mut statement = conn
            .prepare(&format!(
                "SELECT {EDGE_SELECT_COLUMNS} FROM genealogy_parent_edges
                 WHERE child_manifest_id = ?1 ORDER BY parent_manifest_id, idempotency_key"
            ))
            .map_err(GenealogyLineageError::Database)?;
        let edges: Vec<GenealogyParentEdgeV1> = statement
            .query_map(params![child_id], stored_edge_from_row)
            .map_err(GenealogyLineageError::Database)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(GenealogyLineageError::Database)?;

        if stored_lineage_sha256.is_empty() {
            if !edges.is_empty() {
                return Err(GenealogyLineageError::CorruptLineage);
            }
            let expected_manifest_sha256 = manifest_content_hash(
                &manifest.namespace,
                &relation_json,
                &origin_json,
                &manifest.audio_key,
                &manifest.recipe_sha256,
                manifest.score_sha256.as_deref(),
                &manifest.audio_sha256,
            );
            if expected_manifest_sha256 != manifest.manifest_sha256 {
                return Err(GenealogyLineageError::CorruptLineage);
            }
            if let Some(parent_id) = manifest.parent_id {
                if parent_id >= manifest.id {
                    return Err(GenealogyLineageError::CorruptLineage);
                }
                let parent_family_id: i64 = conn
                    .query_row(
                        "SELECT family_id FROM genealogy_manifests WHERE id = ?1",
                        params![parent_id],
                        |row| row.get(0),
                    )
                    .map_err(GenealogyLineageError::Database)?;
                if parent_family_id != manifest.family_id {
                    return Err(GenealogyLineageError::CorruptLineage);
                }
            } else if manifest.family_id != manifest.id {
                return Err(GenealogyLineageError::CorruptLineage);
            }
            return Ok(GenealogyLineageVerificationV1 {
                manifest,
                parent_edges: Vec::new(),
                lineage_sha256: String::new(),
            });
        }

        if edges.is_empty() {
            return Err(GenealogyLineageError::CorruptLineage);
        }
        let mut resolved: Vec<(GenealogyParentInputV1, GenealogyManifest)> =
            Vec::with_capacity(edges.len());
        for edge in &edges {
            if edge.child_manifest_id != manifest.id
                || edge.parent_manifest_id >= manifest.id
                || edge.idempotency_key.trim().is_empty()
            {
                return Err(GenealogyLineageError::CorruptLineage);
            }
            let parent = conn
                .query_row(
                    &format!(
                        "SELECT {SELECT_COLUMNS} FROM genealogy_manifests WHERE id = ?1"
                    ),
                    params![edge.parent_manifest_id],
                    row_to_manifest,
                )
                .optional()
                .map_err(GenealogyLineageError::Database)?
                .ok_or(GenealogyLineageError::CorruptLineage)?;
            let kind_json = serde_json::to_string(&edge.kind)
                .map_err(|_| GenealogyLineageError::CorruptLineage)?;
            let expected_edge_sha256 = genealogy_edge_content_hash(
                edge.parent_manifest_id,
                edge.child_manifest_id,
                &parent.manifest_sha256,
                &manifest.manifest_sha256,
                &kind_json,
                edge.source_event_id.as_deref(),
                edge.contributor_ref.as_deref(),
                edge.authority_receipt_ref.as_deref(),
                &edge.idempotency_key,
            );
            if expected_edge_sha256 != edge.edge_sha256 {
                return Err(GenealogyLineageError::CorruptLineage);
            }
            resolved.push((
                GenealogyParentInputV1 {
                    parent_manifest_id: edge.parent_manifest_id,
                    kind: edge.kind.clone(),
                    source_event_id: edge.source_event_id.clone(),
                    contributor_ref: edge.contributor_ref.clone(),
                    authority_receipt_ref: edge.authority_receipt_ref.clone(),
                    idempotency_key: edge.idempotency_key.clone(),
                },
                parent,
            ));
        }
        resolved.sort_by(|left, right| {
            (
                left.1.id,
                &left.0.idempotency_key,
                &left.0.source_event_id,
            )
                .cmp(&(
                    right.1.id,
                    &right.0.idempotency_key,
                    &right.0.source_event_id,
                ))
        });
        let computed_lineage_sha256 = lineage_content_hash(&resolved);
        let expected_manifest_sha256 = derived_manifest_content_hash(
            &manifest.namespace,
            &relation_json,
            &origin_json,
            &manifest.audio_key,
            &manifest.recipe_sha256,
            manifest.score_sha256.as_deref(),
            &manifest.audio_sha256,
            &computed_lineage_sha256,
        );
        if computed_lineage_sha256 != stored_lineage_sha256
            || expected_manifest_sha256 != manifest.manifest_sha256
        {
            return Err(GenealogyLineageError::CorruptLineage);
        }

        if resolved.len() == 1 {
            if manifest.parent_id != Some(resolved[0].1.id)
                || manifest.family_id != resolved[0].1.family_id
            {
                return Err(GenealogyLineageError::CorruptLineage);
            }
        } else if manifest.parent_id.is_some() || manifest.family_id != manifest.id {
            return Err(GenealogyLineageError::CorruptLineage);
        }

        Ok(GenealogyLineageVerificationV1 {
            manifest,
            parent_edges: edges,
            lineage_sha256: computed_lineage_sha256,
        })
    }

    /// Roll back a manifest allocated by a keep operation. Parent edges and
    /// operation idempotency entries cascade only when the derived child is
    /// deleted; a manifest referenced as a parent is protected by RESTRICT.
    pub fn delete(&self, id: i64) -> rusqlite::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM genealogy_manifests WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Direct children, including children connected through the plural edge
    /// table. Results are unique and ordered by the result manifest ID.
    pub fn children(&self, id: i64) -> rusqlite::Result<Vec<GenealogyManifest>> {
        let conn = self.conn.lock().unwrap();
        let mut statement = conn.prepare(&format!(
            "SELECT {SELECT_COLUMNS} FROM genealogy_manifests m
             WHERE m.parent_id = ?1
                OR EXISTS (
                    SELECT 1 FROM genealogy_parent_edges e
                    WHERE e.child_manifest_id = m.id AND e.parent_manifest_id = ?1
                )
             ORDER BY m.id"
        ))?;
        statement
            .query_map(params![id], row_to_manifest)?
            .collect()
    }

    /// This manifest plus every ancestor, closest-first, across both legacy
    /// single-parent projections and plural source edges. Shared ancestors in
    /// a diamond are returned once; the visited set also prevents traversal
    /// from looping if an externally corrupted database contains a cycle.
    pub fn ancestry(&self, id: i64) -> rusqlite::Result<Vec<GenealogyManifest>> {
        let conn = self.conn.lock().unwrap();
        let mut pending = std::collections::VecDeque::from([(id, 0usize)]);
        let mut seen = BTreeSet::new();
        let mut found: Vec<(usize, GenealogyManifest)> = Vec::new();

        while let Some((current_id, depth)) = pending.pop_front() {
            if !seen.insert(current_id) {
                continue;
            }
            let manifest = match conn
                .query_row(
                    &format!("SELECT {SELECT_COLUMNS} FROM genealogy_manifests WHERE id = ?1"),
                    params![current_id],
                    row_to_manifest,
                )
                .optional()?
            {
                Some(manifest) => manifest,
                None if depth == 0 => return Ok(Vec::new()),
                None => continue,
            };
            found.push((depth, manifest));

            if let Some(parent_id) = conn
                .query_row(
                    "SELECT parent_id FROM genealogy_manifests WHERE id = ?1",
                    params![current_id],
                    |row| row.get::<_, Option<i64>>(0),
                )
                .optional()?
                .flatten()
            {
                pending.push_back((parent_id, depth + 1));
            }
            let mut statement = conn.prepare(
                "SELECT parent_manifest_id FROM genealogy_parent_edges
                 WHERE child_manifest_id = ?1 ORDER BY parent_manifest_id",
            )?;
            let rows = statement.query_map(params![current_id], |row| row.get::<_, i64>(0))?;
            for row in rows {
                pending.push_back((row?, depth + 1));
            }
        }

        found.sort_by_key(|(depth, manifest)| (*depth, manifest.id));
        Ok(found.into_iter().map(|(_, manifest)| manifest).collect())
    }
}

fn row_to_manifest(row: &Row) -> rusqlite::Result<GenealogyManifest> {
    let relation_json: String = row.get(4)?;
    let origin_json: String = row.get(5)?;
    let created_at_unix_ms: i64 = row.get(11)?;
    Ok(GenealogyManifest {
        id: row.get(0)?,
        family_id: row.get(1)?,
        parent_id: row.get(2)?,
        namespace: row.get(3)?,
        relation: serde_json::from_str(&relation_json).unwrap_or(GenealogyRelation::Root),
        origin: serde_json::from_str(&origin_json).unwrap_or(GenealogyOrigin::ManuallyAuthored {
            initial_recipe_sha256: String::new(),
        }),
        audio_key: row.get(6)?,
        recipe_sha256: row.get(7)?,
        score_sha256: row.get(8)?,
        audio_sha256: row.get(9)?,
        manifest_sha256: row.get(10)?,
        created_at_unix_ms: created_at_unix_ms as u64,
    })
}

/// Domain separator for `manifest_sha256`, versioned so a future field-set
/// change gets a new tag rather than silently reinterpreting old manifests
/// under a new byte layout.
const MANIFEST_HASH_DOMAIN: &str = "symthaea-muse-genealogy-manifest-v1";

/// The exact, explicitly ordered byte layout a manifest's content hash
/// covers. Deliberately excludes:
/// - `id`/`family_id` (ledger-assigned sequence numbers, not content --
///   the hash must not depend on allocation order), and
/// - `created_at_unix_ms` (operational metadata, not musical identity --
///   two byte-identical manifests kept a second apart should hash equal).
///
/// Each field is length-prefixed (`len:value`) rather than delimiter-
/// joined, so no field's content (which may itself contain `|` from JSON)
/// can shift a hash boundary.
/// Compute the canonical lowercase SHA-256 content identity for exact bytes.
///
/// This identifies bytes, not authorship, rights, authenticity, or musical value.
pub fn artifact_sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn manifest_content_hash(
    namespace: &str,
    relation_json: &str,
    origin_json: &str,
    audio_key: &str,
    recipe_sha256: &str,
    score_sha256: Option<&str>,
    audio_sha256: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(MANIFEST_HASH_DOMAIN.as_bytes());
    hasher.update([0u8]);
    for field in [
        namespace,
        relation_json,
        origin_json,
        audio_key,
        recipe_sha256,
        score_sha256.unwrap_or(""),
        audio_sha256,
    ] {
        hasher.update(field.len().to_le_bytes());
        hasher.update(field.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

const LINEAGE_HASH_DOMAIN: &str = "symthaea-muse-genealogy-lineage-v1";
const DERIVED_MANIFEST_HASH_DOMAIN: &str = "symthaea-muse-genealogy-manifest-v2";
const EDGE_HASH_DOMAIN: &str = "symthaea-muse-genealogy-parent-edge-v1";

fn hash_field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn hash_optional_field(hasher: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update([1u8]);
            hash_field(hasher, value.as_bytes());
        }
        None => hasher.update([0u8]),
    }
}

fn hash_finish(hasher: Sha256) -> String {
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn lineage_content_hash(
    resolved: &[(GenealogyParentInputV1, GenealogyManifest)],
) -> String {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, LINEAGE_HASH_DOMAIN.as_bytes());
    for (input, parent) in resolved {
        hash_field(&mut hasher, parent.manifest_sha256.as_bytes());
        let kind_json = serde_json::to_string(&input.kind).unwrap_or_default();
        hash_field(&mut hasher, kind_json.as_bytes());
        hash_optional_field(&mut hasher, input.source_event_id.as_deref());
        hash_optional_field(&mut hasher, input.contributor_ref.as_deref());
        hash_optional_field(&mut hasher, input.authority_receipt_ref.as_deref());
        hash_field(&mut hasher, input.idempotency_key.as_bytes());
    }
    hash_finish(hasher)
}

fn derived_manifest_content_hash(
    namespace: &str,
    relation_json: &str,
    origin_json: &str,
    audio_key: &str,
    recipe_sha256: &str,
    score_sha256: Option<&str>,
    audio_sha256: &str,
    lineage_sha256: &str,
) -> String {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, DERIVED_MANIFEST_HASH_DOMAIN.as_bytes());
    for field in [
        namespace,
        relation_json,
        origin_json,
        audio_key,
        recipe_sha256,
        score_sha256.unwrap_or(""),
        audio_sha256,
        lineage_sha256,
    ] {
        hash_field(&mut hasher, field.as_bytes());
    }
    hash_finish(hasher)
}

#[allow(clippy::too_many_arguments)]
fn genealogy_edge_content_hash(
    parent_manifest_id: i64,
    child_manifest_id: i64,
    parent_manifest_sha256: &str,
    child_manifest_sha256: &str,
    kind_json: &str,
    source_event_id: Option<&str>,
    contributor_ref: Option<&str>,
    authority_receipt_ref: Option<&str>,
    idempotency_key: &str,
) -> String {
    let mut hasher = Sha256::new();
    hash_field(&mut hasher, EDGE_HASH_DOMAIN.as_bytes());
    hash_field(&mut hasher, &parent_manifest_id.to_le_bytes());
    hash_field(&mut hasher, &child_manifest_id.to_le_bytes());
    hash_field(&mut hasher, parent_manifest_sha256.as_bytes());
    hash_field(&mut hasher, child_manifest_sha256.as_bytes());
    hash_field(&mut hasher, kind_json.as_bytes());
    hash_optional_field(&mut hasher, source_event_id);
    hash_optional_field(&mut hasher, contributor_ref);
    hash_optional_field(&mut hasher, authority_receipt_ref);
    hash_field(&mut hasher, idempotency_key.as_bytes());
    hash_finish(hasher)
}

#[derive(Clone)]
struct ResolvedStoredEdge {
    edge: GenealogyParentEdgeV1,
    parent_manifest: GenealogyManifest,
}

fn stored_edge_from_row(row: &Row) -> rusqlite::Result<GenealogyParentEdgeV1> {
    let kind_json: String = row.get(3)?;
    let created_at_unix_ms: i64 = row.get(9)?;
    Ok(GenealogyParentEdgeV1 {
        id: row.get(0)?,
        child_manifest_id: row.get(1)?,
        parent_manifest_id: row.get(2)?,
        kind: serde_json::from_str(&kind_json).unwrap_or(GenealogyEdgeKindV1::ReferenceOnly),
        source_event_id: row.get(4)?,
        contributor_ref: row.get(5)?,
        authority_receipt_ref: row.get(6)?,
        idempotency_key: row.get(7)?,
        edge_sha256: row.get(8)?,
        created_at_unix_ms: created_at_unix_ms as u64,
    })
}

const EDGE_SELECT_COLUMNS: &str = "id, child_manifest_id, parent_manifest_id, kind_json,
    source_event_id, contributor_ref, authority_receipt_ref, idempotency_key, edge_sha256,
    created_at_unix_ms";

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> GenealogyStore {
        // In-memory sqlite: `Connection::open` on a real path is exercised
        // via `open()`'s own file-creation logic in `muse_studio`'s
        // integration; unit tests here only need schema + query behavior.
        let conn = Connection::open_in_memory().unwrap();
        initialize_schema(&conn).unwrap();
        GenealogyStore {
            conn: Mutex::new(conn),
        }
    }

    fn origin() -> GenealogyOrigin {
        GenealogyOrigin::MuseGenerated {
            seed: 42,
            style_name: "Minimalism".into(),
        }
    }

    #[test]
    fn artifact_sha256_matches_standard_known_vectors() {
        assert_eq!(
            artifact_sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            artifact_sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn allocate_root_from_bytes_hashes_the_exact_recipe_score_and_audio_bytes() {
        let store = store();
        let recipe_bytes = br#"{
  "name": "exact recipe"
}
"#;
        let score_bytes = br#"{"notes":[60,64,67]}"#;
        let audio_bytes = b"RIFF\x04\x00\x00\x00WAVE";

        let manifest = store
            .allocate_root_from_bytes(
                &origin(),
                "key-exact-bytes",
                recipe_bytes,
                Some(score_bytes),
                audio_bytes,
                1001,
            )
            .unwrap();

        assert_eq!(manifest.recipe_sha256, artifact_sha256_hex(recipe_bytes));
        assert_eq!(
            manifest.score_sha256,
            Some(artifact_sha256_hex(score_bytes))
        );
        assert_eq!(manifest.audio_sha256, artifact_sha256_hex(audio_bytes));
        assert_ne!(
            manifest.recipe_sha256,
            artifact_sha256_hex(br#"{"name":"exact recipe"}"#),
            "pretty-printed persisted bytes must not be hashed as compact re-serialization"
        );
    }

    #[test]
    fn allocate_root_from_bundle_re_reads_and_commits_exact_published_bytes() {
        let store = store();
        let directory = std::env::temp_dir().join(format!(
            "muse-genealogy-byte-bundle-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();

        let recipe = br#"{
  "name": "persisted recipe"
}
"#;
        let score = br#"{"notes":[60,64,67]}"#;
        let audio = b"RIFF\x04\x00\x00\x00WAVE";
        std::fs::write(directory.join("recipe.json"), recipe).unwrap();
        std::fs::write(directory.join("score.json"), score).unwrap();
        std::fs::write(directory.join("audio.wav"), audio).unwrap();

        let manifest = store
            .allocate_root_from_bundle(
                &origin(),
                "bundle-exact",
                &directory,
                recipe,
                score,
                audio,
                1003,
            )
            .unwrap();
        assert_eq!(manifest.recipe_sha256, artifact_sha256_hex(recipe));
        assert_eq!(manifest.score_sha256, Some(artifact_sha256_hex(score)));
        assert_eq!(manifest.audio_sha256, artifact_sha256_hex(audio));

        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn allocate_root_from_bundle_rejects_byte_substitution_before_manifest_commit() {
        let store = store();
        let directory = std::env::temp_dir().join(format!(
            "muse-genealogy-byte-substitution-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();

        let recipe = b"recipe-original";
        let score = b"score-original";
        let audio = b"audio-original";
        std::fs::write(directory.join("recipe.json"), recipe).unwrap();
        std::fs::write(directory.join("score.json"), b"score-substituted").unwrap();
        std::fs::write(directory.join("audio.wav"), audio).unwrap();

        assert!(matches!(
            store.allocate_root_from_bundle(
                &origin(),
                "bundle-substituted",
                &directory,
                recipe,
                score,
                audio,
                1004,
            ),
            Err(GenealogyBundleError::ByteMismatch {
                artifact: "score.json"
            })
        ));
        assert!(
            store
                .manifest_by_audio_key("bundle-substituted")
                .unwrap()
                .is_none()
        );

        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn verify_bundle_resolves_manifest_and_rehashes_current_artifact_bytes() {
        let store = store();
        let directory = std::env::temp_dir().join(format!(
            "muse-genealogy-readback-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();

        let recipe = br#"{
  "name": "verified recipe"
}
"#;
        let score = br#"{"notes":[60,64,67]}"#;
        let audio = b"RIFF\x04\x00\x00\x00WAVE";
        std::fs::write(directory.join("recipe.json"), recipe).unwrap();
        std::fs::write(directory.join("score.json"), score).unwrap();
        std::fs::write(directory.join("audio.wav"), audio).unwrap();

        let manifest = store
            .allocate_root_from_bundle(
                &origin(),
                "verify-after-keep",
                &directory,
                recipe,
                score,
                audio,
                1005,
            )
            .unwrap();
        let verified = store
            .verify_bundle("verify-after-keep", &directory)
            .unwrap();

        assert_eq!(verified.manifest.id, manifest.id);
        assert_eq!(verified.manifest.manifest_sha256, manifest.manifest_sha256);
        assert_eq!(verified.recipe_sha256, artifact_sha256_hex(recipe));
        assert_eq!(verified.score_sha256, Some(artifact_sha256_hex(score)));
        assert_eq!(verified.audio_sha256, artifact_sha256_hex(audio));
        assert_eq!(verified.recipe_bytes_len, recipe.len() as u64);
        assert_eq!(verified.score_bytes_len, Some(score.len() as u64));
        assert_eq!(verified.audio_bytes_len, audio.len() as u64);

        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn verify_bundle_rejects_post_commit_file_substitution() {
        let store = store();
        let directory = std::env::temp_dir().join(format!(
            "muse-genealogy-post-commit-substitution-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();

        let recipe = b"recipe-original";
        let score = b"score-original";
        let audio = b"audio-original";
        std::fs::write(directory.join("recipe.json"), recipe).unwrap();
        std::fs::write(directory.join("score.json"), score).unwrap();
        std::fs::write(directory.join("audio.wav"), audio).unwrap();

        store
            .allocate_root_from_bundle(
                &origin(),
                "verify-substitution",
                &directory,
                recipe,
                score,
                audio,
                1006,
            )
            .unwrap();
        std::fs::write(directory.join("audio.wav"), b"audio-substituted").unwrap();

        assert!(matches!(
            store.verify_bundle("verify-substitution", &directory),
            Err(GenealogyBundleError::ManifestMismatch {
                artifact: "audio.wav"
            })
        ));

        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn verified_bundle_binds_exact_manifest_to_distinct_rmi_events() {
        let store = store();
        let directory = std::env::temp_dir().join(format!(
            "muse-genealogy-rmi-binding-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();

        let recipe = b"rmi-bound-recipe";
        let score = b"rmi-bound-score";
        let audio = b"rmi-bound-audio";
        std::fs::write(directory.join("recipe.json"), recipe).unwrap();
        std::fs::write(directory.join("score.json"), score).unwrap();
        std::fs::write(directory.join("audio.wav"), audio).unwrap();
        store
            .allocate_root_from_bundle(
                &origin(),
                "rmi-binding-key",
                &directory,
                recipe,
                score,
                audio,
                1008,
            )
            .unwrap();

        let verified = store.verify_bundle("rmi-binding-key", &directory).unwrap();
        let binding = verified
            .bind_rmi_events("partner-input-17", "agent-action-18")
            .unwrap();
        assert_eq!(binding.genealogy.audio_key, "rmi-binding-key");
        assert_eq!(binding.genealogy.recipe_sha256, artifact_sha256_hex(recipe));
        assert_eq!(
            binding.genealogy.score_sha256,
            Some(artifact_sha256_hex(score))
        );
        assert_eq!(binding.genealogy.audio_sha256, artifact_sha256_hex(audio));
        assert_eq!(binding.source_event_id, "partner-input-17");
        assert_eq!(binding.action_event_id, "agent-action-18");
        binding.validate_reference().unwrap();

        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn verified_bundle_refuses_ambiguous_rmi_event_binding() {
        let store = store();
        let directory = std::env::temp_dir().join(format!(
            "muse-genealogy-rmi-invalid-binding-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();

        let recipe = b"recipe";
        let score = b"score";
        let audio = b"audio";
        std::fs::write(directory.join("recipe.json"), recipe).unwrap();
        std::fs::write(directory.join("score.json"), score).unwrap();
        std::fs::write(directory.join("audio.wav"), audio).unwrap();
        store
            .allocate_root_from_bundle(
                &origin(),
                "rmi-invalid-key",
                &directory,
                recipe,
                score,
                audio,
                1009,
            )
            .unwrap();

        let verified = store.verify_bundle("rmi-invalid-key", &directory).unwrap();
        assert!(matches!(
            verified.bind_rmi_events("same-event", "same-event"),
            Err(GenealogyBundleError::InvalidRmiBinding(
                "RMI source/action event IDs must be distinct and non-empty"
            ))
        ));

        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn verify_bundle_preserves_manifest_with_no_score_artifact() {
        let store = store();
        let directory = std::env::temp_dir().join(format!(
            "muse-genealogy-optional-score-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();

        let recipe = b"recipe-only";
        let audio = b"audio-only";
        std::fs::write(directory.join("recipe.json"), recipe).unwrap();
        std::fs::write(directory.join("audio.wav"), audio).unwrap();
        store
            .allocate_root_from_bytes(&origin(), "verify-no-score", recipe, None, audio, 1007)
            .unwrap();
        let verified = store.verify_bundle("verify-no-score", &directory).unwrap();
        assert_eq!(verified.score_sha256, None);
        assert_eq!(verified.score_bytes_len, None);

        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn allocate_root_from_bytes_preserves_absent_score_as_none() {
        let store = store();
        let manifest = store
            .allocate_root_from_bytes(&origin(), "key-no-score", b"recipe", None, b"audio", 1002)
            .unwrap();
        assert_eq!(manifest.score_sha256, None);
    }

    #[test]
    fn allocate_root_is_self_family() {
        let store = store();
        let manifest = store
            .allocate_root(
                &origin(),
                "key-1",
                "recipe-hash",
                Some("score-hash"),
                "audio-hash",
                1000,
            )
            .unwrap();
        assert_eq!(manifest.family_id, manifest.id);
        assert_eq!(manifest.parent_id, None);
        assert_eq!(manifest.relation, GenealogyRelation::Root);
    }

    #[test]
    fn manifest_round_trips_by_id_and_audio_key() {
        let store = store();
        let allocated = store
            .allocate_root(&origin(), "key-2", "recipe-hash", None, "audio-hash", 2000)
            .unwrap();
        let by_id = store.manifest(allocated.id).unwrap().unwrap();
        let by_key = store.manifest_by_audio_key("key-2").unwrap().unwrap();
        assert_eq!(by_id, allocated);
        assert_eq!(by_key, allocated);
        assert!(store.manifest(999_999).unwrap().is_none());
    }

    #[test]
    fn duplicate_audio_key_is_rejected() {
        let store = store();
        store
            .allocate_root(&origin(), "dup", "r1", None, "a1", 1)
            .unwrap();
        let second = store.allocate_root(&origin(), "dup", "r2", None, "a2", 2);
        assert!(second.is_err());
    }

    #[test]
    fn ancestry_of_a_root_is_itself_only() {
        let store = store();
        let manifest = store
            .allocate_root(&origin(), "key-3", "recipe-hash", None, "audio-hash", 3000)
            .unwrap();
        let ancestry = store.ancestry(manifest.id).unwrap();
        assert_eq!(ancestry, vec![manifest]);
    }

    #[test]
    fn children_of_a_root_are_empty_in_v1() {
        let store = store();
        let manifest = store
            .allocate_root(&origin(), "key-4", "recipe-hash", None, "audio-hash", 4000)
            .unwrap();
        assert!(store.children(manifest.id).unwrap().is_empty());
    }

    #[test]
    fn delete_rolls_back_an_allocated_manifest() {
        let store = store();
        let manifest = store
            .allocate_root(&origin(), "key-5", "recipe-hash", None, "audio-hash", 5000)
            .unwrap();
        store.delete(manifest.id).unwrap();
        assert!(store.manifest(manifest.id).unwrap().is_none());
        // Deleting an id that was never allocated (or already deleted) is
        // a no-op, not an error -- the caller doesn't need to know
        // whether genealogy allocation happened before rolling it back.
        assert!(store.delete(999_999).is_ok());
    }

    #[test]
    fn manifest_hash_is_deterministic_and_has_no_created_at_input() {
        // `manifest_content_hash`'s signature has no `created_at_unix_ms`
        // parameter at all -- this is the actual guarantee (operational
        // metadata cannot influence the content hash by construction, not
        // by convention). What's left to verify is that it's otherwise a
        // pure function of its real inputs.
        let a = manifest_content_hash("C", "rel", "orig", "key", "recipe", Some("score"), "audio");
        let b = manifest_content_hash("C", "rel", "orig", "key", "recipe", Some("score"), "audio");
        assert_eq!(a, b);
        let different_audio_key = manifest_content_hash(
            "C",
            "rel",
            "orig",
            "other-key",
            "recipe",
            Some("score"),
            "audio",
        );
        assert_ne!(a, different_audio_key);
    }

    #[test]
    fn store_persists_across_reopen_at_the_same_path() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-muse-genealogy-test-{}-{}.sqlite3",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let allocated = {
            let opened = GenealogyStore::open(&path).unwrap();
            opened
                .allocate_root(
                    &origin(),
                    "restart-key",
                    "recipe-hash",
                    None,
                    "audio-hash",
                    7000,
                )
                .unwrap()
            // `opened` (and its Connection) drops here, closing the file.
        };
        let reopened = GenealogyStore::open(&path).unwrap();
        let found = reopened.manifest(allocated.id).unwrap().unwrap();
        assert_eq!(found, allocated);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
    }

    #[test]
    fn concurrent_allocate_root_from_multiple_threads_does_not_lose_rows() {
        let store = std::sync::Arc::new(store());
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let store = std::sync::Arc::clone(&store);
                std::thread::spawn(move || {
                    store
                        .allocate_root(
                            &origin(),
                            &format!("concurrent-{i}"),
                            "recipe-hash",
                            None,
                            "audio-hash",
                            6000 + i,
                        )
                        .unwrap()
                })
            })
            .collect();
        let mut ids: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap().id)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            ids.len(),
            8,
            "every concurrent allocation got a distinct id"
        );
    }
}
