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

    /// Roll back a manifest allocated by [`Self::allocate_root`] whose
    /// *keep* ultimately failed downstream (e.g. the JSONL index append
    /// failed after genealogy allocation already committed). Genealogy
    /// allocation is an enrichment that must not block a keep on failure;
    /// symmetrically, a keep that itself ultimately fails must not leave a
    /// dangling manifest referencing an audio_key whose bundle directory
    /// the caller is about to delete. Safe to call on an id that was never
    /// allocated (e.g. genealogy was already `None`) -- deletes zero rows.
    pub fn delete(&self, id: i64) -> rusqlite::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM genealogy_manifests WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Direct children only (not full descendants) -- V1 has no non-root
    /// relation live yet, so this is always empty today; kept as a real
    /// query rather than a stub so it needs no changes once one exists.
    pub fn children(&self, id: i64) -> rusqlite::Result<Vec<GenealogyManifest>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!(
            "SELECT {SELECT_COLUMNS} FROM genealogy_manifests WHERE parent_id = ?1 ORDER BY id"
        ))?;
        stmt.query_map(params![id], row_to_manifest)?.collect()
    }

    /// This manifest plus every ancestor, closest-first, via a recursive
    /// walk up `parent_id`.
    pub fn ancestry(&self, id: i64) -> rusqlite::Result<Vec<GenealogyManifest>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&format!(
            "WITH RECURSIVE ancestors({SELECT_COLUMNS}) AS (
                SELECT {SELECT_COLUMNS} FROM genealogy_manifests WHERE id = ?1
                UNION ALL
                SELECT m.id, m.family_id, m.parent_id, m.namespace, m.relation_json,
                       m.origin_json, m.audio_key, m.recipe_sha256, m.score_sha256,
                       m.audio_sha256, m.manifest_sha256, m.created_at_unix_ms
                FROM genealogy_manifests m JOIN ancestors a ON m.id = a.parent_id
             )
             SELECT {SELECT_COLUMNS} FROM ancestors"
        ))?;
        stmt.query_map(params![id], row_to_manifest)?.collect()
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
