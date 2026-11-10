//! Content-addressed export of the official MMAE submission.jsonl.
//!
//! This crate verifies the exact expected ID set, unique canonical relative
//! paths, regular-file safety and SHA-256 content identities. It writes only
//! the official two-field manifest. WAV parsing, USTAR creation, archive size
//! checks and uploads are intentionally delegated to the organizer's SDK.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path};
use std::sync::atomic::{AtomicU64, Ordering};

pub const REQUIRED_SAMPLE_COUNT: usize = 500;
pub const OFFICIAL_MANIFEST_NAME: &str = "submission.jsonl";
pub const OFFICIAL_MANIFEST_MAX_BYTES: usize = 1024 * 1024;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportError {
    pub message: String,
}

impl ExportError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for ExportError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ExportError {}

/// Local producer-side row. These three fields are not copied into the
/// official submission.jsonl; only sample ID and relative WAV path are.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OutputMappingV1 {
    pub sample_id: String,
    pub audio_path: String,
    pub audio_sha256: String,
}

/// Exact official output row. Keep this schema to the two fields required by
/// the challenge SDK; do not add local provenance fields to submission.jsonl.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OfficialManifestRowV1 {
    pub id: String,
    pub audio_path: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportReportV1 {
    pub sample_count: usize,
    pub manifest_bytes: u64,
    pub total_audio_bytes: u64,
    pub manifest_sha256: String,
    pub already_present: bool,
}

/// Validate the ID map and publish a deterministic official submission.jsonl.
///
/// The expected ID slice must come from the exact track-specific expected-ID
/// list shipped with or obtained through the official SDK. The organizer SDK
/// fetches current server policy separately and remains authoritative at upload
/// time. This function does not validate WAV encoding or create a tar archive.
pub fn export_submission_manifest(
    expected_ids: &[String],
    mappings: &[OutputMappingV1],
    submission_root: &Path,
) -> Result<ExportReportV1, ExportError> {
    let root_meta = fs::symlink_metadata(submission_root).map_err(|error| {
        ExportError::new(format!(
            "cannot inspect submission root {}: {error}",
            submission_root.display()
        ))
    })?;
    if root_meta.file_type().is_symlink() || !root_meta.is_dir() {
        return Err(ExportError::new(
            "submission root must be an existing, non-symlink directory",
        ));
    }
    let root = submission_root
        .canonicalize()
        .map_err(|error| ExportError::new(format!("cannot resolve submission root: {error}")))?;

    let expected = validate_expected_ids(expected_ids)?;
    if mappings.len() != REQUIRED_SAMPLE_COUNT {
        return Err(ExportError::new(format!(
            "expected exactly {REQUIRED_SAMPLE_COUNT} output mappings, received {}",
            mappings.len()
        )));
    }

    let mut by_id = BTreeMap::new();
    let mut paths = BTreeSet::new();
    for mapping in mappings {
        require_non_empty(&mapping.sample_id, "sample_id")?;
        validate_sha256(&mapping.audio_sha256, "audio_sha256")?;
        validate_audio_path(&mapping.audio_path)?;

        if !expected.contains(&mapping.sample_id) {
            return Err(ExportError::new(format!(
                "output map contains an unknown sample ID: {}",
                mapping.sample_id
            )));
        }
        if by_id
            .insert(mapping.sample_id.as_str(), mapping)
            .is_some()
        {
            return Err(ExportError::new(format!(
                "duplicate output mapping for sample ID {}",
                mapping.sample_id
            )));
        }
        if !paths.insert(mapping.audio_path.as_str()) {
            return Err(ExportError::new(format!(
                "audio path is reused by multiple sample IDs: {}",
                mapping.audio_path
            )));
        }
    }

    let missing: Vec<&str> = expected
        .iter()
        .filter(|sample_id| !by_id.contains_key(sample_id.as_str()))
        .map(String::as_str)
        .collect();
    if !missing.is_empty() {
        return Err(ExportError::new(format!(
            "output map is missing {} expected sample IDs",
            missing.len()
        )));
    }

    let mut manifest = Vec::new();
    let mut total_audio_bytes = 0u64;

    for sample_id in &expected {
        let mapping = by_id
            .get(sample_id.as_str())
            .copied()
            .ok_or_else(|| ExportError::new("validated output mapping disappeared"))?;
        let audio_path = resolve_regular_audio(&root, &mapping.audio_path)?;
        let (actual_digest, byte_count) = hash_file(&audio_path)?;
        if actual_digest != mapping.audio_sha256 {
            return Err(ExportError::new(format!(
                "SHA-256 mismatch for sample {} at {}",
                sample_id, mapping.audio_path
            )));
        }
        total_audio_bytes = total_audio_bytes
            .checked_add(byte_count)
            .ok_or_else(|| ExportError::new("total audio byte count overflow"))?;

        let row = OfficialManifestRowV1 {
            id: sample_id.clone(),
            audio_path: mapping.audio_path.clone(),
        };
        let json = serde_json::to_string(&row)
            .map_err(|error| ExportError::new(format!("cannot serialize manifest row: {error}")))?;
        manifest.extend_from_slice(json.as_bytes());
        manifest.push(b'\n');
    }

    if manifest.len() > OFFICIAL_MANIFEST_MAX_BYTES {
        return Err(ExportError::new(format!(
            "submission.jsonl exceeds the official {}-byte manifest limit",
            OFFICIAL_MANIFEST_MAX_BYTES
        )));
    }
    let manifest_sha256 = sha256_bytes(&manifest);
    let manifest_bytes = u64::try_from(manifest.len())
        .map_err(|_| ExportError::new("manifest size does not fit in u64"))?;
    let destination = root.join(OFFICIAL_MANIFEST_NAME);

    match fs::symlink_metadata(&destination) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(ExportError::new(
                    "existing submission.jsonl is not a regular file",
                ));
            }
            let existing = fs::read(&destination).map_err(|error| {
                ExportError::new(format!("cannot read existing submission.jsonl: {error}"))
            })?;
            if existing != manifest {
                return Err(ExportError::new(
                    "submission.jsonl already exists with different content; remove it deliberately before replacing",
                ));
            }
            return Ok(ExportReportV1 {
                sample_count: expected.len(),
                manifest_bytes,
                total_audio_bytes,
                manifest_sha256,
                already_present: true,
            });
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(ExportError::new(format!(
                "cannot inspect destination submission.jsonl: {error}"
            )));
        }
    }

    write_manifest_atomically(&root, &destination, &manifest)?;

    Ok(ExportReportV1 {
        sample_count: expected.len(),
        manifest_bytes,
        total_audio_bytes,
        manifest_sha256,
        already_present: false,
    })
}

fn validate_expected_ids(expected_ids: &[String]) -> Result<Vec<String>, ExportError> {
    if expected_ids.len() != REQUIRED_SAMPLE_COUNT {
        return Err(ExportError::new(format!(
            "expected-ID artifact must contain exactly {REQUIRED_SAMPLE_COUNT} IDs, received {}",
            expected_ids.len()
        )));
    }
    let mut unique = BTreeSet::new();
    for sample_id in expected_ids {
        require_non_empty(sample_id, "expected sample ID")?;
        if !unique.insert(sample_id.as_str()) {
            return Err(ExportError::new(format!(
                "expected-ID artifact contains a duplicate ID: {sample_id}"
            )));
        }
    }
    let mut ordered = expected_ids.to_vec();
    ordered.sort();
    Ok(ordered)
}

fn validate_audio_path(value: &str) -> Result<(), ExportError> {
    require_non_empty(value, "audio_path")?;
    if value.contains('\\')
        || value.contains(':')
        || value.contains('\0')
        || value.chars().any(char::is_control)
    {
        return Err(ExportError::new(format!(
            "audio_path contains a forbidden character: {value:?}"
        )));
    }
    if !value.to_ascii_lowercase().ends_with(".wav") {
        return Err(ExportError::new(format!(
            "audio_path must end in .wav: {value}"
        )));
    }

    let path = Path::new(value);
    if path.is_absolute() {
        return Err(ExportError::new(format!(
            "audio_path must be relative: {value}"
        )));
    }
    let mut normalized = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => {
                let part = part.to_str().ok_or_else(|| {
                    ExportError::new(format!("audio_path must be UTF-8: {value:?}"))
                })?;
                if part.is_empty() {
                    return Err(ExportError::new(format!(
                        "audio_path contains an empty component: {value}"
                    )));
                }
                normalized.push(part);
            }
            _ => {
                return Err(ExportError::new(format!(
                    "audio_path is not a canonical relative path: {value}"
                )));
            }
        }
    }
    if normalized.join("/") != value {
        return Err(ExportError::new(format!(
            "audio_path must use canonical POSIX separators: {value}"
        )));
    }
    Ok(())
}

fn resolve_regular_audio(root: &Path, relative: &str) -> Result<std::path::PathBuf, ExportError> {
    validate_audio_path(relative)?;
    let mut current = root.to_path_buf();
    let parts: Vec<_> = Path::new(relative).components().collect();
    for (index, component) in parts.iter().enumerate() {
        let Component::Normal(part) = component else {
            return Err(ExportError::new(format!(
                "audio_path has an invalid component: {relative}"
            )));
        };
        current.push(part);
        let metadata = fs::symlink_metadata(&current).map_err(|error| {
            ExportError::new(format!(
                "cannot inspect audio path {relative}: {error}"
            ))
        })?;
        if metadata.file_type().is_symlink() {
            return Err(ExportError::new(format!(
                "symlinks are not allowed in audio_path: {relative}"
            )));
        }
        if index + 1 == parts.len() {
            if !metadata.is_file() {
                return Err(ExportError::new(format!(
                    "audio_path must reference a regular file: {relative}"
                )));
            }
        } else if !metadata.is_dir() {
            return Err(ExportError::new(format!(
                "audio_path parent component is not a directory: {relative}"
            )));
        }
    }

    let resolved = current
        .canonicalize()
        .map_err(|error| ExportError::new(format!("cannot resolve {relative}: {error}")))?;
    if !resolved.starts_with(root) {
        return Err(ExportError::new(format!(
            "audio_path escapes the submission directory: {relative}"
        )));
    }
    Ok(resolved)
}

fn hash_file(path: &Path) -> Result<(String, u64), ExportError> {
    let mut file = File::open(path)
        .map_err(|error| ExportError::new(format!("cannot open {}: {error}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| ExportError::new(format!("cannot read {}: {error}", path.display())))?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .ok_or_else(|| ExportError::new("audio file byte count overflow"))?;
        hasher.update(&buffer[..count]);
    }
    Ok((format!("sha256:{:x}", hasher.finalize()), total))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn validate_sha256(value: &str, field: &str) -> Result<(), ExportError> {
    let valid = value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex.bytes().all(|byte| {
                byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
            })
    });
    if !valid {
        return Err(ExportError::new(format!(
            "{field} must be sha256: followed by 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn require_non_empty(value: &str, field: &str) -> Result<(), ExportError> {
    if value.trim().is_empty() {
        Err(ExportError::new(format!("{field} must not be empty")))
    } else {
        Ok(())
    }
}

fn write_manifest_atomically(
    root: &Path,
    destination: &Path,
    bytes: &[u8],
) -> Result<(), ExportError> {
    let mut temp_path = None;
    let mut temp_file = None;
    for _ in 0..32 {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let candidate = root.join(format!(
            ".submission.jsonl.tmp-{}-{sequence}",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&candidate) {
            Ok(file) => {
                temp_path = Some(candidate);
                temp_file = Some(file);
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(ExportError::new(format!(
                    "cannot create temporary submission manifest: {error}"
                )));
            }
        }
    }
    let temp_path = temp_path.ok_or_else(|| {
        ExportError::new("could not allocate a unique temporary manifest filename")
    })?;
    let mut file = temp_file.ok_or_else(|| {
        ExportError::new("temporary manifest file handle was not created")
    })?;

    let result = (|| {
        file.write_all(bytes)
            .map_err(|error| ExportError::new(format!("cannot write submission manifest: {error}")))?;
        file.sync_all()
            .map_err(|error| ExportError::new(format!("cannot sync submission manifest: {error}")))?;
        drop(file);
        fs::rename(&temp_path, destination).map_err(|error| {
            ExportError::new(format!("cannot atomically publish submission.jsonl: {error}"))
        })?;
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(std::path::PathBuf);

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn test_directory() -> TestDirectory {
        let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "mmae-export-test-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("audio")).unwrap();
        TestDirectory(root)
    }

    fn fixture() -> (TestDirectory, Vec<String>, Vec<OutputMappingV1>) {
        let directory = test_directory();
        let mut expected = Vec::new();
        let mut mappings = Vec::new();
        for index in 0..REQUIRED_SAMPLE_COUNT {
            let sample_id = format!("sample-{index:03}");
            let audio_path = format!("audio/{sample_id}.wav");
            let bytes = format!("synthetic-audio-payload-{sample_id}").into_bytes();
            fs::write(directory.0.join(&audio_path), &bytes).unwrap();
            let digest = sha256_bytes(&bytes);
            expected.push(sample_id.clone());
            mappings.push(OutputMappingV1 {
                sample_id,
                audio_path,
                audio_sha256: digest,
            });
        }
        (directory, expected, mappings)
    }

    #[test]
    fn exports_exact_two_field_jsonl_for_all_500_ids() {
        let (directory, expected, mappings) = fixture();
        let report =
            export_submission_manifest(&expected, &mappings, &directory.0).unwrap();
        assert_eq!(report.sample_count, REQUIRED_SAMPLE_COUNT);
        assert!(!report.already_present);
        let bytes = fs::read(directory.0.join(OFFICIAL_MANIFEST_NAME)).unwrap();
        let rows: Vec<OfficialManifestRowV1> = std::str::from_utf8(&bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows.len(), REQUIRED_SAMPLE_COUNT);
        assert_eq!(rows[0].id, "sample-000");
        assert_eq!(rows[0].audio_path, "audio/sample-000.wav");
        assert!(report.manifest_sha256.starts_with("sha256:"));
    }

    #[test]
    fn output_manifest_is_deterministic_and_idempotent() {
        let (directory, expected, mappings) = fixture();
        let first =
            export_submission_manifest(&expected, &mappings, &directory.0).unwrap();
        let second =
            export_submission_manifest(&expected, &mappings, &directory.0).unwrap();
        assert_eq!(first.manifest_sha256, second.manifest_sha256);
        assert!(!first.already_present);
        assert!(second.already_present);
    }

    #[test]
    fn rejects_non_500_expected_id_artifacts() {
        let (directory, mut expected, mappings) = fixture();
        expected.pop();
        assert!(export_submission_manifest(&expected, &mappings, &directory.0).is_err());
    }

    #[test]
    fn rejects_missing_output_ids() {
        let (directory, expected, mut mappings) = fixture();
        mappings.pop();
        assert!(export_submission_manifest(&expected, &mappings, &directory.0).is_err());
    }

    #[test]
    fn rejects_unknown_sample_ids() {
        let (directory, expected, mut mappings) = fixture();
        mappings[0].sample_id = "not-an-official-id".to_owned();
        assert!(export_submission_manifest(&expected, &mappings, &directory.0).is_err());
    }

    #[test]
    fn rejects_duplicate_sample_ids() {
        let (directory, expected, mut mappings) = fixture();
        mappings[1].sample_id = mappings[0].sample_id.clone();
        assert!(export_submission_manifest(&expected, &mappings, &directory.0).is_err());
    }

    #[test]
    fn rejects_duplicate_audio_paths() {
        let (directory, expected, mut mappings) = fixture();
        mappings[1].audio_path = mappings[0].audio_path.clone();
        assert!(export_submission_manifest(&expected, &mappings, &directory.0).is_err());
    }

    #[test]
    fn rejects_parent_traversal_absolute_paths_and_wrong_extensions() {
        let (directory, expected, mut mappings) = fixture();
        mappings[0].audio_path = "../outside.wav".to_owned();
        assert!(export_submission_manifest(&expected, &mappings, &directory.0).is_err());

        mappings[0].audio_path = "/tmp/outside.wav".to_owned();
        assert!(export_submission_manifest(&expected, &mappings, &directory.0).is_err());

        mappings[0].audio_path = "audio/sample-000.mp3".to_owned();
        assert!(export_submission_manifest(&expected, &mappings, &directory.0).is_err());
    }

    #[test]
    fn rejects_content_digest_mismatch() {
        let (directory, expected, mut mappings) = fixture();
        mappings[0].audio_sha256 = sha256_bytes(b"not-the-file");
        assert!(export_submission_manifest(&expected, &mappings, &directory.0).is_err());
    }

    #[test]
    fn rejects_missing_audio_file() {
        let (directory, expected, mappings) = fixture();
        fs::remove_file(directory.0.join(&mappings[0].audio_path)).unwrap();
        assert!(export_submission_manifest(&expected, &mappings, &directory.0).is_err());
    }

    #[test]
    fn rejects_existing_manifest_with_different_content() {
        let (directory, expected, mappings) = fixture();
        fs::write(directory.0.join(OFFICIAL_MANIFEST_NAME), b"stale manifest").unwrap();
        assert!(export_submission_manifest(&expected, &mappings, &directory.0).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_audio_path_components() {
        use std::os::unix::fs::symlink;
        let (directory, expected, mut mappings) = fixture();
        symlink(
            directory.0.join("audio"),
            directory.0.join("audio-link"),
        )
        .unwrap();
        mappings[0].audio_path = format!("audio-link/{}.wav", mappings[0].sample_id);
        assert!(export_submission_manifest(&expected, &mappings, &directory.0).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_submission_root() {
        use std::os::unix::fs::symlink;
        let (directory, expected, mappings) = fixture();
        let alias = directory.0.with_extension("alias");
        symlink(&directory.0, &alias).unwrap();
        assert!(export_submission_manifest(&expected, &mappings, &alias).is_err());
        let _ = fs::remove_file(alias);
    }

    #[test]
    fn rejects_symlinked_existing_submission_manifest() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let (directory, expected, mappings) = fixture();
            let target = directory.0.join("stale.jsonl");
            fs::write(&target, b"stale").unwrap();
            symlink(target, directory.0.join(OFFICIAL_MANIFEST_NAME)).unwrap();
            assert!(
                export_submission_manifest(&expected, &mappings, &directory.0).is_err()
            );
        }
    }

    #[test]
    fn accepts_uppercase_wav_extension_like_the_official_sdk() {
        let (directory, expected, mut mappings) = fixture();
        let original = directory.0.join(&mappings[0].audio_path);
        let renamed = directory.0.join("audio/sample-000.WAV");
        fs::rename(original, &renamed).unwrap();
        mappings[0].audio_path = "audio/sample-000.WAV".to_owned();
        assert!(export_submission_manifest(&expected, &mappings, &directory.0).is_ok());
    }
}
