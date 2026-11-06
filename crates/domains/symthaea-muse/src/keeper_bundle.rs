// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-back verification for durable Muse Studio keeper bundles.
//!
//! This module verifies exact bytes reopened from the published keeper
//! directory. It is separate from identity, consent, signatures, rights,
//! authorship, and musical-quality verification.

use std::{
    collections::BTreeSet,
    error::Error,
    fmt,
    fs,
    io,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Current directory layout for newly published, fully verified keepers.
pub const KEEPER_BUNDLE_LAYOUT_V2: &str = "keeper-directory-v2";
/// Version of the serialized read-back receipt.
pub const KEEPER_BUNDLE_VERIFICATION_SCHEMA_V1: u16 = 1;
/// Stable verifier identifier included in each read-back receipt.
pub const KEEPER_BUNDLE_VERIFIER_ID: &str = "symthaea-muse-keeper-readback";
/// Version of the byte-level read-back verifier.
pub const KEEPER_BUNDLE_VERIFIER_VERSION: &str = "1";

/// Hashes captured from intended payloads before publication.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KeeperBundleExpectedDigestsV1 {
    /// Exact intended bytes for audio.wav.
    pub audio_sha256: String,
    /// Exact intended bytes for recipe.json.
    pub recipe_file_sha256: String,
    /// Canonical serialized score bytes written to score.json.
    pub score_sha256: String,
    /// Exact bytes of performance.mid, absent only when MIDI export failed.
    pub midi_sha256: Option<String>,
    /// Existing logical hash of the typed recipe serialization.
    pub recipe_object_sha256: String,
}

/// One file whose bytes were independently reopened from the published
/// keeper directory.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct KeeperArtifactDigestV1 {
    /// Stable artifact kind.
    pub kind: String,
    /// Canonical path relative to the keeper directory.
    pub file_name: String,
    /// Declared media type.
    pub media_type: String,
    /// SHA-256 of the exact reopened file bytes.
    pub sha256: String,
    /// Exact number of reopened bytes.
    pub byte_len: u64,
}

/// Durable receipt of the post-publication keeper byte read-back.
///
/// The receipt is integrity evidence over the listed file bytes. It is not
/// signed and does not authenticate actors, consent, copyright, or authorship.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct KeeperBundleVerificationV1 {
    /// Receipt schema version.
    pub schema_version: u16,
    /// Collision-resistant storage key for the published directory.
    pub artifact_key: String,
    /// Directory layout used by this keeper.
    pub artifact_layout: String,
    /// Existing logical recipe identity from Studio's typed recipe.
    pub recipe_object_sha256: String,
    /// Independent verifier identifier.
    pub verifier_id: String,
    /// Independent verifier version.
    pub verifier_version: String,
    /// Operational read-back time; not part of musical content identity.
    pub verified_at_unix_ms: u64,
    /// Exact files found and re-hashed in canonical order.
    pub artifacts: Vec<KeeperArtifactDigestV1>,
}

impl KeeperBundleVerificationV1 {
    /// SHA-256 of the canonical JSON serialization of this receipt.
    ///
    /// Store this digest beside the receipt in the atomic keeper index. This
    /// digest is not a signature or proof of signer identity.
    pub fn receipt_sha256(&self) -> Result<String, KeeperBundleVerificationError> {
        let bytes = serde_json::to_vec(self)
            .map_err(KeeperBundleVerificationError::SerializeReceipt)?;
        Ok(sha256_hex(&bytes))
    }
}

/// Errors from exact keeper-directory read-back.
#[derive(Debug)]
pub enum KeeperBundleVerificationError {
    /// Artifact keys must be a single safe path component.
    InvalidArtifactKey,
    /// The supplied directory must be the exact key-named directory.
    DirectoryKeyMismatch,
    /// The bundle directory is absent, not a directory, or a symlink.
    InvalidBundleDirectory,
    /// A directory entry is unexpected, duplicated, or not a regular file.
    InvalidDirectoryEntry(String),
    /// A required artifact file is missing.
    MissingArtifact(String),
    /// An expected digest is not lowercase, 64-character hexadecimal.
    InvalidExpectedDigest(String),
    /// A file could not be reopened.
    ReadArtifact {
        /// Artifact filename.
        file_name: String,
        /// Underlying filesystem error.
        source: io::Error,
    },
    /// Reopened bytes do not match the intended or receipt digest.
    HashMismatch {
        /// Artifact filename.
        file_name: String,
        /// Expected digest.
        expected: String,
        /// Digest computed over reopened bytes.
        actual: String,
    },
    /// Reopened bytes differ in length from the receipt.
    LengthMismatch {
        /// Artifact filename.
        file_name: String,
        /// Expected number of bytes.
        expected: u64,
        /// Actual number of bytes.
        actual: u64,
    },
    /// Receipt metadata does not describe the canonical layout.
    InvalidReceipt,
    /// Receipt serialization failed.
    SerializeReceipt(serde_json::Error),
}

impl fmt::Display for KeeperBundleVerificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArtifactKey => write!(formatter, "invalid keeper artifact key"),
            Self::DirectoryKeyMismatch => write!(formatter, "keeper directory/key mismatch"),
            Self::InvalidBundleDirectory => write!(formatter, "invalid keeper bundle directory"),
            Self::InvalidDirectoryEntry(name) => {
                write!(formatter, "invalid or unexpected keeper directory entry: {name}")
            }
            Self::MissingArtifact(name) => write!(formatter, "missing keeper artifact: {name}"),
            Self::InvalidExpectedDigest(name) => {
                write!(formatter, "invalid expected SHA-256 for {name}")
            }
            Self::ReadArtifact { file_name, source } => {
                write!(formatter, "cannot reopen keeper artifact {file_name}: {source}")
            }
            Self::HashMismatch { file_name, expected, actual } => write!(
                formatter,
                "keeper artifact hash mismatch for {file_name}: expected {expected}, read {actual}"
            ),
            Self::LengthMismatch { file_name, expected, actual } => write!(
                formatter,
                "keeper artifact length mismatch for {file_name}: expected {expected}, read {actual}"
            ),
            Self::InvalidReceipt => write!(formatter, "invalid keeper verification receipt"),
            Self::SerializeReceipt(error) => {
                write!(formatter, "cannot serialize keeper verification receipt: {error}")
            }
        }
    }
}

impl Error for KeeperBundleVerificationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ReadArtifact { source, .. } => Some(source),
            Self::SerializeReceipt(error) => Some(error),
            _ => None,
        }
    }
}

/// Verify a published keeper directory against hashes captured before
/// publication. The function reopens the final path, rejects extra files and
/// symlinks, and derives the receipt only from bytes read after publication.
///
/// Required files are audio.wav, recipe.json, and score.json. performance.mid
/// must exist exactly when expected.midi_sha256 is Some.
pub fn verify_published_keeper_bundle(
    bundle_dir: &Path,
    artifact_key: &str,
    expected: &KeeperBundleExpectedDigestsV1,
    verified_at_unix_ms: u64,
) -> Result<KeeperBundleVerificationV1, KeeperBundleVerificationError> {
    validate_artifact_key(artifact_key)?;
    if bundle_dir.file_name().and_then(|name| name.to_str()) != Some(artifact_key) {
        return Err(KeeperBundleVerificationError::DirectoryKeyMismatch);
    }
    validate_expected_digests(expected)?;

    let expected_files = expected_file_names(expected.midi_sha256.is_some());
    validate_directory_entries(bundle_dir, &expected_files)?;

    let mut artifacts = Vec::with_capacity(expected_files.len());
    for (file_name, kind, media_type, expected_sha256) in [
        ("audio.wav", "rendered_audio", "audio/wav", Some(expected.audio_sha256.as_str())),
        ("recipe.json", "composition_recipe", "application/json", Some(expected.recipe_file_sha256.as_str())),
        ("score.json", "symbolic_score", "application/json", Some(expected.score_sha256.as_str())),
        ("performance.mid", "performed_midi", "audio/midi", expected.midi_sha256.as_deref()),
    ] {
        let Some(expected_sha256) = expected_sha256 else {
            continue;
        };
        let bytes = read_regular_file(bundle_dir, file_name)?;
        let actual_sha256 = sha256_hex(&bytes);
        if actual_sha256 != expected_sha256 {
            return Err(KeeperBundleVerificationError::HashMismatch {
                file_name: file_name.to_owned(),
                expected: expected_sha256.to_owned(),
                actual: actual_sha256,
            });
        }
        artifacts.push(KeeperArtifactDigestV1 {
            kind: kind.to_owned(),
            file_name: file_name.to_owned(),
            media_type: media_type.to_owned(),
            sha256: actual_sha256,
            byte_len: bytes.len() as u64,
        });
    }

    let receipt = KeeperBundleVerificationV1 {
        schema_version: KEEPER_BUNDLE_VERIFICATION_SCHEMA_V1,
        artifact_key: artifact_key.to_owned(),
        artifact_layout: KEEPER_BUNDLE_LAYOUT_V2.to_owned(),
        recipe_object_sha256: expected.recipe_object_sha256.clone(),
        verifier_id: KEEPER_BUNDLE_VERIFIER_ID.to_owned(),
        verifier_version: KEEPER_BUNDLE_VERIFIER_VERSION.to_owned(),
        verified_at_unix_ms,
        artifacts,
    };

    // Route the new receipt through the same checks used by later readers.
    verify_keeper_bundle_against_receipt(bundle_dir, &receipt)?;
    Ok(receipt)
}

/// Reopen a keeper directory against a previously persisted receipt.
/// This is the recovery/read path for future Studio/API integrations.
pub fn verify_keeper_bundle_against_receipt(
    bundle_dir: &Path,
    receipt: &KeeperBundleVerificationV1,
) -> Result<(), KeeperBundleVerificationError> {
    validate_artifact_key(&receipt.artifact_key)?;
    if bundle_dir.file_name().and_then(|name| name.to_str())
        != Some(receipt.artifact_key.as_str())
    {
        return Err(KeeperBundleVerificationError::DirectoryKeyMismatch);
    }
    if receipt.schema_version != KEEPER_BUNDLE_VERIFICATION_SCHEMA_V1
        || receipt.artifact_layout != KEEPER_BUNDLE_LAYOUT_V2
        || receipt.verifier_id != KEEPER_BUNDLE_VERIFIER_ID
        || receipt.verifier_version != KEEPER_BUNDLE_VERIFIER_VERSION
        || !is_sha256_hex(&receipt.recipe_object_sha256)
    {
        return Err(KeeperBundleVerificationError::InvalidReceipt);
    }

    let has_midi = receipt.artifacts.iter().any(|artifact| artifact.file_name == "performance.mid");
    let expected_names = expected_file_names(has_midi);
    let supplied_names: BTreeSet<&str> = receipt
        .artifacts
        .iter()
        .map(|artifact| artifact.file_name.as_str())
        .collect();
    if supplied_names.len() != receipt.artifacts.len()
        || supplied_names != expected_names.iter().map(String::as_str).collect()
    {
        return Err(KeeperBundleVerificationError::InvalidReceipt);
    }
    validate_directory_entries(bundle_dir, &expected_names)?;

    for artifact in &receipt.artifacts {
        if !is_canonical_artifact_metadata(artifact) || !is_sha256_hex(&artifact.sha256) {
            return Err(KeeperBundleVerificationError::InvalidReceipt);
        }
        let bytes = read_regular_file(bundle_dir, &artifact.file_name)?;
        if bytes.len() as u64 != artifact.byte_len {
            return Err(KeeperBundleVerificationError::LengthMismatch {
                file_name: artifact.file_name.clone(),
                expected: artifact.byte_len,
                actual: bytes.len() as u64,
            });
        }
        let actual = sha256_hex(&bytes);
        if actual != artifact.sha256 {
            return Err(KeeperBundleVerificationError::HashMismatch {
                file_name: artifact.file_name.clone(),
                expected: artifact.sha256.clone(),
                actual,
            });
        }
    }
    Ok(())
}

/// Return the SHA-256 of exact file bytes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_artifact_key(artifact_key: &str) -> Result<(), KeeperBundleVerificationError> {
    if artifact_key.is_empty()
        || artifact_key.len() > 128
        || !artifact_key.bytes().all(|byte| byte.is_ascii_hexdigit() || byte == b'_')
    {
        return Err(KeeperBundleVerificationError::InvalidArtifactKey);
    }
    Ok(())
}

fn validate_expected_digests(
    expected: &KeeperBundleExpectedDigestsV1,
) -> Result<(), KeeperBundleVerificationError> {
    for (name, digest) in [
        ("audio.wav", Some(expected.audio_sha256.as_str())),
        ("recipe.json", Some(expected.recipe_file_sha256.as_str())),
        ("score.json", Some(expected.score_sha256.as_str())),
        ("recipe object", Some(expected.recipe_object_sha256.as_str())),
        ("performance.mid", expected.midi_sha256.as_deref()),
    ] {
        if digest.is_some_and(|value| !is_sha256_hex(value)) {
            return Err(KeeperBundleVerificationError::InvalidExpectedDigest(name.to_owned()));
        }
    }
    Ok(())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn expected_file_names(has_midi: bool) -> BTreeSet<String> {
    let mut names = BTreeSet::from([
        "audio.wav".to_owned(),
        "recipe.json".to_owned(),
        "score.json".to_owned(),
    ]);
    if has_midi {
        names.insert("performance.mid".to_owned());
    }
    names
}

fn validate_directory_entries(
    bundle_dir: &Path,
    expected_names: &BTreeSet<String>,
) -> Result<(), KeeperBundleVerificationError> {
    let metadata = fs::symlink_metadata(bundle_dir)
        .map_err(|_| KeeperBundleVerificationError::InvalidBundleDirectory)?;
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        return Err(KeeperBundleVerificationError::InvalidBundleDirectory);
    }

    let mut actual_names = BTreeSet::new();
    for entry in fs::read_dir(bundle_dir)
        .map_err(|_| KeeperBundleVerificationError::InvalidBundleDirectory)?
    {
        let entry = entry.map_err(|_| KeeperBundleVerificationError::InvalidBundleDirectory)?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| KeeperBundleVerificationError::InvalidDirectoryEntry("<non-utf8>".to_owned()))?;
        let metadata = fs::symlink_metadata(entry.path())
            .map_err(|_| KeeperBundleVerificationError::InvalidDirectoryEntry(name.clone()))?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Err(KeeperBundleVerificationError::InvalidDirectoryEntry(name));
        }
        if !actual_names.insert(name.clone()) || !expected_names.contains(&name) {
            return Err(KeeperBundleVerificationError::InvalidDirectoryEntry(name));
        }
    }

    if actual_names != *expected_names {
        let missing = expected_names
            .difference(&actual_names)
            .next()
            .cloned()
            .unwrap_or_else(|| "<unknown>".to_owned());
        return Err(KeeperBundleVerificationError::MissingArtifact(missing));
    }
    Ok(())
}

fn read_regular_file(
    bundle_dir: &Path,
    file_name: &str,
) -> Result<Vec<u8>, KeeperBundleVerificationError> {
    let path: PathBuf = bundle_dir.join(file_name);
    let metadata = fs::symlink_metadata(&path).map_err(|source| {
        if source.kind() == io::ErrorKind::NotFound {
            KeeperBundleVerificationError::MissingArtifact(file_name.to_owned())
        } else {
            KeeperBundleVerificationError::ReadArtifact {
                file_name: file_name.to_owned(),
                source,
            }
        }
    })?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(KeeperBundleVerificationError::InvalidDirectoryEntry(file_name.to_owned()));
    }
    fs::read(path).map_err(|source| KeeperBundleVerificationError::ReadArtifact {
        file_name: file_name.to_owned(),
        source,
    })
}

fn is_canonical_artifact_metadata(artifact: &KeeperArtifactDigestV1) -> bool {
    match artifact.file_name.as_str() {
        "audio.wav" => artifact.kind == "rendered_audio" && artifact.media_type == "audio/wav",
        "recipe.json" => artifact.kind == "composition_recipe" && artifact.media_type == "application/json",
        "score.json" => artifact.kind == "symbolic_score" && artifact.media_type == "application/json",
        "performance.mid" => artifact.kind == "performed_midi" && artifact.media_type == "audio/midi",
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        root: PathBuf,
        bundle: PathBuf,
        expected: KeeperBundleExpectedDigestsV1,
    }

    impl Fixture {
        fn new(with_midi: bool) -> Self {
            let suffix = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "mel_keeper_readback_{}_{}",
                std::process::id(),
                suffix
            ));
            let bundle = root.join("0001_deadbeef");
            fs::create_dir_all(&bundle).unwrap();

            let audio = b"RIFF\x04\x00\x00\x00WAVEfmt ".to_vec();
            let recipe = br#"{ "seed": 41, "style": "test" }"#.to_vec();
            let score = br#"{"notes":[],"tempo_bpm":120}"#.to_vec();
            fs::write(bundle.join("audio.wav"), &audio).unwrap();
            fs::write(bundle.join("recipe.json"), &recipe).unwrap();
            fs::write(bundle.join("score.json"), &score).unwrap();

            let midi = if with_midi {
                let bytes = b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x00\x60".to_vec();
                fs::write(bundle.join("performance.mid"), &bytes).unwrap();
                Some(sha256_hex(&bytes))
            } else {
                None
            };

            let expected = KeeperBundleExpectedDigestsV1 {
                audio_sha256: sha256_hex(&audio),
                recipe_file_sha256: sha256_hex(&recipe),
                score_sha256: sha256_hex(&score),
                midi_sha256: midi,
                recipe_object_sha256: "a".repeat(64),
            };
            Self { root, bundle, expected }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn readback_receipt_binds_all_exact_files_in_published_directory() {
        let fixture = Fixture::new(true);
        let receipt = verify_published_keeper_bundle(
            &fixture.bundle,
            "0001_deadbeef",
            &fixture.expected,
            1234,
        )
        .unwrap();
        assert_eq!(receipt.schema_version, 1);
        assert_eq!(receipt.artifact_layout, KEEPER_BUNDLE_LAYOUT_V2);
        assert_eq!(receipt.artifacts.len(), 4);
        assert_eq!(
            receipt.artifacts.iter().map(|item| item.file_name.as_str()).collect::<Vec<_>>(),
            vec!["audio.wav", "recipe.json", "score.json", "performance.mid"]
        );
        assert!(is_sha256_hex(&receipt.receipt_sha256().unwrap()));
        verify_keeper_bundle_against_receipt(&fixture.bundle, &receipt).unwrap();
    }

    #[test]
    fn readback_receipt_supports_declared_no_midi_bundle() {
        let fixture = Fixture::new(false);
        let receipt = verify_published_keeper_bundle(
            &fixture.bundle,
            "0001_deadbeef",
            &fixture.expected,
            1234,
        )
        .unwrap();
        assert_eq!(receipt.artifacts.len(), 3);
        verify_keeper_bundle_against_receipt(&fixture.bundle, &receipt).unwrap();
    }

    #[test]
    fn changed_bytes_after_publication_are_rejected() {
        let fixture = Fixture::new(true);
        let receipt = verify_published_keeper_bundle(
            &fixture.bundle,
            "0001_deadbeef",
            &fixture.expected,
            1234,
        )
        .unwrap();
        fs::write(fixture.bundle.join("audio.wav"), b"substituted audio").unwrap();
        assert!(matches!(
            verify_keeper_bundle_against_receipt(&fixture.bundle, &receipt),
            Err(KeeperBundleVerificationError::LengthMismatch { .. })
                | Err(KeeperBundleVerificationError::HashMismatch { .. })
        ));
    }

    #[test]
    fn unexpected_files_are_rejected() {
        let fixture = Fixture::new(false);
        fs::write(fixture.bundle.join("unexpected.bin"), b"x").unwrap();
        assert!(matches!(
            verify_published_keeper_bundle(
                &fixture.bundle,
                "0001_deadbeef",
                &fixture.expected,
                1234
            ),
            Err(KeeperBundleVerificationError::InvalidDirectoryEntry(_))
        ));
    }

    #[test]
    fn missing_declared_midi_is_rejected() {
        let fixture = Fixture::new(true);
        fs::remove_file(fixture.bundle.join("performance.mid")).unwrap();
        assert!(matches!(
            verify_published_keeper_bundle(
                &fixture.bundle,
                "0001_deadbeef",
                &fixture.expected,
                1234
            ),
            Err(KeeperBundleVerificationError::MissingArtifact(_))
        ));
    }

    #[test]
    fn symlinked_artifacts_are_rejected() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let fixture = Fixture::new(false);
            let external = fixture.root.join("outside.wav");
            fs::write(&external, b"outside").unwrap();
            fs::remove_file(fixture.bundle.join("audio.wav")).unwrap();
            symlink(external, fixture.bundle.join("audio.wav")).unwrap();
            assert!(matches!(
                verify_published_keeper_bundle(
                    &fixture.bundle,
                    "0001_deadbeef",
                    &fixture.expected,
                    1234
                ),
                Err(KeeperBundleVerificationError::InvalidDirectoryEntry(_))
            ));
        }
    }

    #[test]
    fn traversal_keys_and_mismatched_directories_are_rejected() {
        let fixture = Fixture::new(false);
        assert!(matches!(
            verify_published_keeper_bundle(
                &fixture.bundle,
                "../0001_deadbeef",
                &fixture.expected,
                1234
            ),
            Err(KeeperBundleVerificationError::InvalidArtifactKey)
        ));
        assert!(matches!(
            verify_published_keeper_bundle(
                &fixture.bundle,
                "0002_other",
                &fixture.expected,
                1234
            ),
            Err(KeeperBundleVerificationError::DirectoryKeyMismatch)
        ));
    }

    #[test]
    fn malformed_expected_hashes_are_rejected_before_reading() {
        let fixture = Fixture::new(false);
        let mut expected = fixture.expected.clone();
        expected.score_sha256 = "main".to_owned();
        assert!(matches!(
            verify_published_keeper_bundle(
                &fixture.bundle,
                "0001_deadbeef",
                &expected,
                1234
            ),
            Err(KeeperBundleVerificationError::InvalidExpectedDigest(_))
        ));
    }

    #[test]
    fn receipt_rejects_noncanonical_file_entries() {
        let fixture = Fixture::new(false);
        let mut receipt = verify_published_keeper_bundle(
            &fixture.bundle,
            "0001_deadbeef",
            &fixture.expected,
            1234,
        )
        .unwrap();
        receipt.artifacts[0].file_name = "anything.wav".to_owned();
        assert!(matches!(
            verify_keeper_bundle_against_receipt(&fixture.bundle, &receipt),
            Err(KeeperBundleVerificationError::InvalidReceipt)
        ));
    }
}
