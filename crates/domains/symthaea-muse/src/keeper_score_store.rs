// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Durable exact symbolic-score storage for kept Melothaea pieces.
//!
//! `score_sha256` has existed as a durable keeper/genealogy commitment, but the
//! exact serialized `Score` bytes were not persisted with the keeper. That left
//! restart-time consumers tempted to recompose a score from the stored recipe,
//! which can silently bind an old keeper to a newer engine's realization.
//!
//! This module closes that gap for new-format keepers. `score.json` is written as
//! the exact compact `serde_json::to_vec(score)` byte sequence whose SHA-256 is
//! already carried by `KeeperSemanticBundleV1::score_sha256`. Reads are allowed
//! only when a valid persisted semantic sidecar exists and the score bytes match
//! its commitment. There is no recipe-recomposition fallback.

use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};
use symthaea_muse_protocol::keeper_semantic::KeeperSemanticBundleV1;

use crate::keeper_semantic_store::{
    KeeperSemanticStoreError, read_persisted_semantic_bundle, semantic_bundle_path,
};

pub const KEEPER_SCORE_FILENAME: &str = "score.json";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeeperScoreStoreError {
    AudioKeyMismatch,
    CommitmentMismatch,
    AlreadyExists,
    MissingScore,
    SymlinkNotAllowed(&'static str),
    Io(String),
    Serialize(String),
    Parse(String),
    Semantic(KeeperSemanticStoreError),
}

impl fmt::Display for KeeperScoreStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AudioKeyMismatch => {
                write!(f, "keeper score audio_key does not match semantic evidence")
            }
            Self::CommitmentMismatch => {
                write!(f, "keeper score bytes do not match score_sha256 commitment")
            }
            Self::AlreadyExists => write!(f, "keeper score sidecar already exists"),
            Self::MissingScore => write!(
                f,
                "keeper has semantic evidence but its committed score.json is missing"
            ),
            Self::SymlinkNotAllowed(label) => {
                write!(f, "keeper score {label} must not be a symbolic link")
            }
            Self::Io(error) => write!(f, "keeper score storage I/O failed: {error}"),
            Self::Serialize(error) => write!(f, "keeper score serialization failed: {error}"),
            Self::Parse(error) => write!(f, "keeper score parse failed: {error}"),
            Self::Semantic(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for KeeperScoreStoreError {}

impl From<KeeperSemanticStoreError> for KeeperScoreStoreError {
    fn from(value: KeeperSemanticStoreError) -> Self {
        Self::Semantic(value)
    }
}

/// Serialize `score` exactly once, verify that those exact bytes satisfy the
/// semantic bundle's score commitment, then create `score.json` in the keeper
/// staging directory.
///
/// The file is create-once and both the file and staging directory are synced.
/// The caller must still publish the whole keeper directory atomically only
/// after every mandatory artifact has staged successfully.
pub fn write_score_to_staging<S: Serialize>(
    staging_dir: &Path,
    expected_audio_key: &str,
    semantic_bundle: &KeeperSemanticBundleV1,
    score: &S,
) -> Result<(), KeeperScoreStoreError> {
    if semantic_bundle.audio_key != expected_audio_key {
        return Err(KeeperScoreStoreError::AudioKeyMismatch);
    }
    require_real_directory(staging_dir, "staging directory")?;

    let bytes = serde_json::to_vec(score)
        .map_err(|error| KeeperScoreStoreError::Serialize(error.to_string()))?;
    if sha256_hex(&bytes) != semantic_bundle.score_sha256 {
        return Err(KeeperScoreStoreError::CommitmentMismatch);
    }

    let path = staging_dir.join(KEEPER_SCORE_FILENAME);
    match std::fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(KeeperScoreStoreError::SymlinkNotAllowed("sidecar"));
        }
        Ok(_) => return Err(KeeperScoreStoreError::AlreadyExists),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(KeeperScoreStoreError::Io(error.to_string())),
    }

    let mut file = match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(KeeperScoreStoreError::AlreadyExists);
        }
        Err(error) => return Err(KeeperScoreStoreError::Io(error.to_string())),
    };

    let result = file
        .write_all(&bytes)
        .and_then(|()| file.sync_all())
        .and_then(|()| sync_directory(staging_dir));
    if let Err(error) = result {
        drop(file);
        let _ = std::fs::remove_file(&path);
        return Err(KeeperScoreStoreError::Io(error.to_string()));
    }
    Ok(())
}

/// Read the exact durable score for a keeper and verify its bytes against the
/// persisted semantic commitment before deserializing it.
///
/// `Ok(None)` is reserved for a legacy keeper with no semantic sidecar at all.
/// Once semantic evidence exists, missing/tampered/unparseable `score.json` is a
/// contradiction and therefore an error, never a reason to recompose a score.
pub fn read_verified_score<T: DeserializeOwned>(
    root: &Path,
    audio_key: &str,
) -> Result<Option<T>, KeeperScoreStoreError> {
    let Some(semantic_bundle) = read_persisted_semantic_bundle(root, audio_key)? else {
        return Ok(None);
    };

    let path = score_path(root, audio_key);
    let metadata = match std::fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(KeeperScoreStoreError::MissingScore);
        }
        Err(error) => return Err(KeeperScoreStoreError::Io(error.to_string())),
    };
    if metadata.file_type().is_symlink() {
        return Err(KeeperScoreStoreError::SymlinkNotAllowed("sidecar"));
    }
    if !metadata.is_file() {
        return Err(KeeperScoreStoreError::Io(
            "keeper score sidecar is not a regular file".to_string(),
        ));
    }

    let bytes = std::fs::read(&path)
        .map_err(|error| KeeperScoreStoreError::Io(error.to_string()))?;
    if sha256_hex(&bytes) != semantic_bundle.score_sha256 {
        return Err(KeeperScoreStoreError::CommitmentMismatch);
    }
    let score = serde_json::from_slice::<T>(&bytes)
        .map_err(|error| KeeperScoreStoreError::Parse(error.to_string()))?;
    Ok(Some(score))
}

pub fn score_path(root: &Path, audio_key: &str) -> PathBuf {
    semantic_bundle_path(root, audio_key)
        .parent()
        .expect("semantic bundle path always has a keeper parent")
        .join(KEEPER_SCORE_FILENAME)
}

fn require_real_directory(
    path: &Path,
    label: &'static str,
) -> Result<(), KeeperScoreStoreError> {
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            KeeperScoreStoreError::Io(format!("keeper score {label} does not exist"))
        } else {
            KeeperScoreStoreError::Io(error.to_string())
        }
    })?;
    if metadata.file_type().is_symlink() {
        return Err(KeeperScoreStoreError::SymlinkNotAllowed(label));
    }
    if !metadata.is_dir() {
        return Err(KeeperScoreStoreError::Io(format!(
            "keeper score {label} is not a directory"
        )));
    }
    Ok(())
}

fn sync_directory(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        std::fs::File::open(path)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde::{Deserialize, Serialize};
    use symthaea_muse_protocol::{
        LISTEN_COMPOSITION_BUNDLE_VERSION, ListenCompositionBundle, MeterPoint, MusicalTime,
        TempoPoint,
        keeper_semantic::{KEEPER_SEMANTIC_BUNDLE_SCHEMA_VERSION, KeeperSemanticBundleV1},
    };

    use crate::keeper_semantic_store::write_semantic_bundle_to_staging;

    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct ScoreFixture {
        notes: Vec<u8>,
        meter: u8,
    }

    fn test_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "melothaea-keeper-score-store-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn zero() -> MusicalTime {
        MusicalTime {
            tick: 0,
            beats: 0.0,
            seconds: 0.0,
        }
    }

    fn score() -> ScoreFixture {
        ScoreFixture {
            notes: vec![60, 64, 67, 72],
            meter: 4,
        }
    }

    fn semantic_bundle(audio_key: &str, score: &ScoreFixture) -> KeeperSemanticBundleV1 {
        KeeperSemanticBundleV1 {
            schema_version: KEEPER_SEMANTIC_BUNDLE_SCHEMA_VERSION,
            audio_key: audio_key.to_string(),
            listen_bundle_version: LISTEN_COMPOSITION_BUNDLE_VERSION,
            score_sha256: sha256_hex(&serde_json::to_vec(score).unwrap()),
            recipe_sha256: "b".repeat(64),
            audio_sha256: "c".repeat(64),
            warnings: Vec::new(),
            payload: ListenCompositionBundle {
                ticks_per_beat: 960,
                duration_ticks: 3840,
                duration_beats: 4.0,
                duration_seconds: 2.0,
                form_kind: "Ternary".to_string(),
                tempo_map: vec![TempoPoint {
                    at: zero(),
                    bpm: 120.0,
                }],
                meter_map: vec![MeterPoint {
                    at: zero(),
                    numerator: 4,
                    denominator: 4,
                }],
                sections: Vec::new(),
                phrases: Vec::new(),
                notes: Vec::new(),
                motif_definitions: Vec::new(),
                motif_occurrences: Vec::new(),
                cadences: Vec::new(),
                sonorities: Vec::new(),
                orchestration: Vec::new(),
                resonance: None,
            },
        }
    }

    #[test]
    fn persisted_score_bytes_are_exactly_the_committed_serialization() {
        let root = test_root("exact-bytes");
        let staging = root.join("staging");
        std::fs::create_dir_all(&staging).unwrap();
        let score = score();
        let semantic = semantic_bundle("keeper-a", &score);

        write_score_to_staging(&staging, "keeper-a", &semantic, &score).unwrap();
        let bytes = std::fs::read(staging.join(KEEPER_SCORE_FILENAME)).unwrap();
        assert_eq!(bytes, serde_json::to_vec(&score).unwrap());
        assert_eq!(sha256_hex(&bytes), semantic.score_sha256);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn published_score_round_trips_from_disk_without_recomposition() {
        let root = test_root("roundtrip");
        std::fs::create_dir_all(&root).unwrap();
        let staging = root.join("staging");
        std::fs::create_dir(&staging).unwrap();
        let score = score();
        let semantic = semantic_bundle("keeper-a", &score);

        write_score_to_staging(&staging, "keeper-a", &semantic, &score).unwrap();
        write_semantic_bundle_to_staging(&staging, "keeper-a", &semantic).unwrap();
        std::fs::rename(&staging, root.join("keeper-a")).unwrap();

        let loaded = read_verified_score::<ScoreFixture>(&root, "keeper-a")
            .unwrap()
            .expect("new-format keeper has exact score");
        assert_eq!(loaded, score);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn legacy_absence_is_distinct_from_new_format_missing_score() {
        let root = test_root("absence");
        std::fs::create_dir_all(root.join("legacy")).unwrap();
        assert_eq!(
            read_verified_score::<ScoreFixture>(&root, "legacy").unwrap(),
            None
        );

        let score = score();
        let semantic = semantic_bundle("new", &score);
        std::fs::create_dir_all(root.join("new")).unwrap();
        write_semantic_bundle_to_staging(root.join("new").as_path(), "new", &semantic).unwrap();
        assert_eq!(
            read_verified_score::<ScoreFixture>(&root, "new"),
            Err(KeeperScoreStoreError::MissingScore)
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn tampered_score_never_downgrades_to_recomposition() {
        let root = test_root("tamper");
        let directory = root.join("keeper-a");
        std::fs::create_dir_all(&directory).unwrap();
        let score = score();
        let semantic = semantic_bundle("keeper-a", &score);
        write_semantic_bundle_to_staging(&directory, "keeper-a", &semantic).unwrap();
        std::fs::write(directory.join(KEEPER_SCORE_FILENAME), b"{\"notes\":[]}").unwrap();

        assert_eq!(
            read_verified_score::<ScoreFixture>(&root, "keeper-a"),
            Err(KeeperScoreStoreError::CommitmentMismatch)
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_score_is_rejected() {
        use std::os::unix::fs::symlink;

        let root = test_root("symlink");
        let directory = root.join("keeper-a");
        std::fs::create_dir_all(&directory).unwrap();
        let score = score();
        let semantic = semantic_bundle("keeper-a", &score);
        write_semantic_bundle_to_staging(&directory, "keeper-a", &semantic).unwrap();
        let target = root.join("elsewhere.json");
        std::fs::write(&target, serde_json::to_vec(&score).unwrap()).unwrap();
        symlink(&target, directory.join(KEEPER_SCORE_FILENAME)).unwrap();

        assert_eq!(
            read_verified_score::<ScoreFixture>(&root, "keeper-a"),
            Err(KeeperScoreStoreError::SymlinkNotAllowed("sidecar"))
        );
        let _ = std::fs::remove_dir_all(root);
    }
}