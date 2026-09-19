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
    SymlinkNotAllowed(&'static str),
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
            Self::SymlinkNotAllowed(label) => {
                write!(f, "keeper semantic {label} must not be a symbolic link")
            }
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
///
/// On Unix, syncing the staging directory after the sidecar also persists the
/// directory entries for audio/recipe/MIDI files the caller wrote before this
/// function. That makes this helper suitable as the final write step before the
/// existing atomic keeper-directory rename.
pub fn write_semantic_bundle_to_staging(
    staging_dir: &Path,
    expected_audio_key: &str,
    bundle: &KeeperSemanticBundleV1,
) -> Result<(), KeeperSemanticStoreError> {
    validate_key(expected_audio_key)?;
    if bundle.audio_key != expected_audio_key {
        return Err(KeeperSemanticStoreError::AudioKeyMismatch);
    }
    require_real_directory(staging_dir, "staging directory")?;

    let bytes = serde_json::to_vec_pretty(bundle)
        .map_err(|error| KeeperSemanticStoreError::Serialize(error.to_string()))?;
    let path = staging_dir.join(KEEPER_SEMANTIC_FILENAME);
    match std::fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(KeeperSemanticStoreError::SymlinkNotAllowed("sidecar"));
        }
        Ok(_) => return Err(KeeperSemanticStoreError::AlreadyExists),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(KeeperSemanticStoreError::Io(error.to_string())),
    }

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

    let result = file
        .write_all(&bytes)
        .and_then(|()| file.sync_all())
        .and_then(|()| sync_directory(staging_dir));
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
/// or the current engine version. The trusted `root` itself may be a deployment
/// symlink, but the untrusted key-selected keeper directory and sidecar may not.
pub fn read_persisted_semantic_bundle(
    root: &Path,
    audio_key: &str,
) -> Result<Option<KeeperSemanticBundleV1>, KeeperSemanticStoreError> {
    validate_key(audio_key)?;
    let keeper_dir = root.join(audio_key);
    let keeper_metadata = match std::fs::symlink_metadata(&keeper_dir) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(KeeperSemanticStoreError::Io(error.to_string())),
    };
    if keeper_metadata.file_type().is_symlink() {
        return Err(KeeperSemanticStoreError::SymlinkNotAllowed(
            "keeper directory",
        ));
    }
    if !keeper_metadata.is_dir() {
        return Err(KeeperSemanticStoreError::Io(
            "keeper artifact path is not a directory".to_string(),
        ));
    }

    let path = keeper_dir.join(KEEPER_SEMANTIC_FILENAME);
    let metadata = match std::fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(KeeperSemanticStoreError::Io(error.to_string())),
    };
    if metadata.file_type().is_symlink() {
        return Err(KeeperSemanticStoreError::SymlinkNotAllowed("sidecar"));
    }
    if !metadata.is_file() {
        return Err(KeeperSemanticStoreError::Io(
            "keeper semantic sidecar is not a regular file".to_string(),
        ));
    }

    let bytes = std::fs::read(&path)
        .map_err(|error| KeeperSemanticStoreError::Io(error.to_string()))?;
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

fn require_real_directory(
    path: &Path,
    label: &'static str,
) -> Result<(), KeeperSemanticStoreError> {
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            KeeperSemanticStoreError::Io(format!("keeper {label} does not exist"))
        } else {
            KeeperSemanticStoreError::Io(error.to_string())
        }
    })?;
    if metadata.file_type().is_symlink() {
        return Err(KeeperSemanticStoreError::SymlinkNotAllowed(label));
    }
    if !metadata.is_dir() {
        return Err(KeeperSemanticStoreError::Io(format!(
            "keeper {label} is not a directory"
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

    #[cfg(unix)]
    #[test]
    fn symlinked_staging_or_persisted_paths_are_rejected() {
        use std::os::unix::fs::symlink;

        let root = test_root("symlink");
        let real_staging = root.join("real-staging");
        let linked_staging = root.join("linked-staging");
        std::fs::create_dir_all(&real_staging).unwrap();
        symlink(&real_staging, &linked_staging).unwrap();
        assert_eq!(
            write_semantic_bundle_to_staging(&linked_staging, "keeper-a", &bundle("keeper-a")),
            Err(KeeperSemanticStoreError::SymlinkNotAllowed(
                "staging directory"
            ))
        );

        let keeper = root.join("keeper-a");
        std::fs::create_dir_all(&keeper).unwrap();
        let external = root.join("external.json");
        std::fs::write(&external, serde_json::to_vec(&bundle("keeper-a")).unwrap()).unwrap();
        symlink(&external, keeper.join(KEEPER_SEMANTIC_FILENAME)).unwrap();
        assert_eq!(
            read_persisted_semantic_bundle(&root, "keeper-a"),
            Err(KeeperSemanticStoreError::SymlinkNotAllowed("sidecar"))
        );

        let real_keeper = root.join("real-keeper");
        std::fs::create_dir_all(&real_keeper).unwrap();
        write_semantic_bundle_to_staging(&real_keeper, "keeper-linked", &bundle("keeper-linked"))
            .unwrap();
        symlink(&real_keeper, root.join("keeper-linked")).unwrap();
        assert_eq!(
            read_persisted_semantic_bundle(&root, "keeper-linked"),
            Err(KeeperSemanticStoreError::SymlinkNotAllowed(
                "keeper directory"
            ))
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
