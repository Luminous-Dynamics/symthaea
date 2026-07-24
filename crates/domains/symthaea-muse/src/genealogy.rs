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

use std::path::Path;
use std::sync::Mutex;

use rusqlite::{Connection, OptionalExtension, Row, params};
use sha2::{Digest, Sha256};
use symthaea_muse_protocol::{GenealogyManifest, GenealogyOrigin, GenealogyRelation};

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
    created_at_unix_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_genealogy_parent ON genealogy_manifests(parent_id);
CREATE INDEX IF NOT EXISTS idx_genealogy_family ON genealogy_manifests(family_id);
";

const SELECT_COLUMNS: &str = "id, family_id, parent_id, namespace, relation_json, origin_json,
     audio_key, recipe_sha256, score_sha256, audio_sha256, manifest_sha256, created_at_unix_ms";

pub struct GenealogyStore {
    conn: Mutex<Connection>,
}

impl GenealogyStore {
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
        conn.execute_batch(SCHEMA)?;
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
        conn.execute_batch(SCHEMA).unwrap();
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
