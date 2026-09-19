// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Durable filesystem storage for keeper semantic evidence.
//!
//! This module owns storage mechanics only. It never derives, reconstructs, or
//! upgrades musical evidence: callers provide an already-built shared
//! [`KeeperSemanticBundleV1`], and reads return only bytes that were actually
//! persisted for that keeper. A missing sidecar therefore remains missing.
//!
//! New keepers are written into the caller's existing staging directory. The
//! caller remains responsible for atomically publishing that whole directory;
//! this module intentionally performs no publication rename of its own.

use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};

use symthaea_muse_protocol::keeper_semantic::KeeperSemanticBundleV1;

pub const KEEPER_SEMANTIC_FILENAME: &str = "semantic-bundle.json";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeeperSemanticStoreError {
    InvalidArtifactKey,
    AudioKeyMismatch,
    AlreadyExists,
    Io(String),
    Serialize(String),
    Parse(String),
}

impl fmt::Display for KeeperSemanticStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArtifactKey => write!(f, "keeper audio key is not path-safe"),
            Self::AudioKeyMismatch => write!(
                f,
                "keeper semantic bundle audio_key does not match the requested keeper"
            ),
            Self::AlreadyExists => write!(f, "keeper semantic sidecar already exists"),
            Self::Io(error) => write!(f, "keeper semantic storage I/O failed: {error}"),
            Self::Serialize(error) => {
                write!(f, "keeper semantic bundle serialization failed: {error}")
            }
            Self::Parse(error) => write!(f, "keeper semantic bundle parse failed: {error}"),
        }
    }
}

impl std::error::Error for KeeperSemanticStoreError {}

/// Write a semantic sidecar into an already-created keeper staging directory.
///
/// The sidecar is create-once: an existing path is never truncated or replaced.
/// A partial new file is removed if writing or syncing fails. The caller should
/// publish the entire staging directory only after this succeeds.
pub fn write_semantic_bundle_to_staging(
    staging_dir: &Path,
    expected_audio_key: &str,
    bundle: &KeeperSemanticBundleV1,
) -> Result<(), KeeperSemanticStoreError> {
    validate_key(expected_audio_key)?;
    if bundle.audio_key != expected_audio_key {
        return Err(KeeperSemanticStoreError::AudioKeyMismatch);
    }
    if !staging_dir.is_dir() {
        return Err(KeeperSemanticStoreError::Io(
            "keeper staging directory does not exist".to_string(),
        ));
    }

    let bytes = serde_json::to_vec_pretty(bundle)
        .map_err(|error| KeeperSemanticStoreError::Serialize(error.to_string()))?;
    let path = staging_dir.join(KEEPER_SEMANTIC_FILENAME);
    let mut file = match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(KeeperSemanticStoreError::AlreadyExists);
        }
        Err(error) => return Err(KeeperSemanticStoreError::Io(error.to_string())),
    };

    let result = file.write_all(&bytes).and_then(|()| file.sync_all());
    if let Err(error) = result {
        drop(file);
        let _ = std::fs::remove_file(&path);
        return Err(KeeperSemanticStoreError::Io(error.to_string()));
    }
    Ok(())
}

/// Read only the semantic sidecar actually persisted under `root/<key>/`.
///
/// `Ok(None)` means there is no stored sidecar (for example, a legacy keeper).
/// There is deliberately no reconstruction fallback from recipe, score, audio,
/// or the current engine version.
pub fn read_persisted_semantic_bundle(
    root: &Path,
    audio_key: &str,
) -> Result<Option<KeeperSemanticBundleV1>, KeeperSemanticStoreError> {
    validate_key(audio_key)?;
    let path = semantic_bundle_path(root, audio_key);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(KeeperSemanticStoreError::Io(error.to_string())),
    };
    let bundle = serde_json::from_slice::<KeeperSemanticBundleV1>(&bytes)
        .map_err(|error| KeeperSemanticStoreError::Parse(error.to_string()))?;
    if bundle.audio_key != audio_key {
        return Err(KeeperSemanticStoreError::AudioKeyMismatch);
    }
    Ok(Some(bundle))
}

pub fn semantic_bundle_path(root: &Path, audio_key: &str) -> PathBuf {
    root.join(audio_key).join(KEEPER_SEMANTIC_FILENAME)
}

fn validate_key(key: &str) -> Result<(), KeeperSemanticStoreError> {
    if !key.is_empty()
        && key.len() <= 80
        && key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    {
        Ok(())
    } else {
        Err(KeeperSemanticStoreError::InvalidArtifactKey)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    use symthaea_muse_protocol::{
        LISTEN_COMPOSITION_BUNDLE_VERSION, ListenCompositionBundle, MeterPoint, MusicalTime,
        TempoPoint,
        keeper_semantic::{KEEPER_SEMANTIC_BUNDLE_SCHEMA_VERSION, KeeperSemanticBundleV1},
    };

    fn test_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "melothaea-keeper-semantic-store-{label}-{}-{nonce}",
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

    fn bundle(audio_key: &str) -> KeeperSemanticBundleV1 {
        KeeperSemanticBundleV1 {
            schema_version: KEEPER_SEMANTIC_BUNDLE_SCHEMA_VERSION,
            audio_key: audio_key.to_string(),
            listen_bundle_version: LISTEN_COMPOSITION_BUNDLE_VERSION,
            score_sha256: "a".repeat(64),
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
    fn staging_write_becomes_readable_only_after_directory_publication() {
        let root = test_root("publish");
        std::fs::create_dir_all(&root).unwrap();
        let key = "keeper-a";
        let staging = root.join(".tmp-keeper-a");
        std::fs::create_dir(&staging).unwrap();
        let expected = bundle(key);

        write_semantic_bundle_to_staging(&staging, key, &expected).unwrap();
        assert_eq!(read_persisted_semantic_bundle(&root, key).unwrap(), None);

        std::fs::rename(&staging, root.join(key)).unwrap();
        assert_eq!(
            read_persisted_semantic_bundle(&root, key).unwrap(),
            Some(expected)
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn legacy_keeper_without_sidecar_remains_explicitly_unavailable() {
        let root = test_root("legacy");
        let key = "keeper-legacy";
        std::fs::create_dir_all(root.join(key)).unwrap();
        std::fs::write(root.join(key).join("audio.wav"), b"legacy").unwrap();

        assert_eq!(read_persisted_semantic_bundle(&root, key).unwrap(), None);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn write_rejects_key_mismatch_and_path_traversal() {
        let root = test_root("write-reject");
        let staging = root.join("staging");
        std::fs::create_dir_all(&staging).unwrap();

        assert_eq!(
            write_semantic_bundle_to_staging(&staging, "keeper-a", &bundle("keeper-b")),
            Err(KeeperSemanticStoreError::AudioKeyMismatch)
        );
        assert_eq!(
            write_semantic_bundle_to_staging(&staging, "../keeper", &bundle("../keeper")),
            Err(KeeperSemanticStoreError::InvalidArtifactKey)
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn sidecar_is_create_once_and_second_write_preserves_original_bytes() {
        let root = test_root("create-once");
        let staging = root.join("staging");
        std::fs::create_dir_all(&staging).unwrap();
        let first = bundle("keeper-a");
        write_semantic_bundle_to_staging(&staging, "keeper-a", &first).unwrap();
        let path = staging.join(KEEPER_SEMANTIC_FILENAME);
        let original = std::fs::read(&path).unwrap();

        let mut second = first.clone();
        second.score_sha256 = "d".repeat(64);
        assert_eq!(
            write_semantic_bundle_to_staging(&staging, "keeper-a", &second),
            Err(KeeperSemanticStoreError::AlreadyExists)
        );
        assert_eq!(std::fs::read(&path).unwrap(), original);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn read_rejects_malformed_or_misbinding_sidecars_instead_of_hiding_them() {
        let root = test_root("read-reject");
        let key = "keeper-a";
        let directory = root.join(key);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join(KEEPER_SEMANTIC_FILENAME);
        std::fs::write(&path, b"not json").unwrap();
        assert!(matches!(
            read_persisted_semantic_bundle(&root, key),
            Err(KeeperSemanticStoreError::Parse(_))
        ));

        std::fs::write(&path, serde_json::to_vec(&bundle("keeper-b")).unwrap()).unwrap();
        assert_eq!(
            read_persisted_semantic_bundle(&root, key),
            Err(KeeperSemanticStoreError::AudioKeyMismatch)
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn read_rejects_invalid_keys_before_touching_storage() {
        let root = test_root("bad-key");
        assert_eq!(
            read_persisted_semantic_bundle(&root, "keeper/child"),
            Err(KeeperSemanticStoreError::InvalidArtifactKey)
        );
        assert_eq!(
            read_persisted_semantic_bundle(&root, ""),
            Err(KeeperSemanticStoreError::InvalidArtifactKey)
        );
    }
}
