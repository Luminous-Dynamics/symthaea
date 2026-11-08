use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

pub const REPORT_SCHEMA_VERSION: u16 = 1;
pub const MAX_MANIFEST_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_EXPECTED_IDS_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_RECORD_BYTES: usize = 1024 * 1024;
pub const DEFAULT_MANIFEST_NAME: &str = "submission.jsonl";

/// One strict row in the published reference JSONL layout.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SubmissionEntryV1 {
    /// Challenge test sample ID.
    pub id: String,
    /// Relative path to the edited WAV file.
    pub audio_path: String,
}

/// Byte-level evidence for one verified WAV.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct AudioFileEvidenceV1 {
    /// Challenge sample ID.
    pub id: String,
    /// Relative manifest path.
    pub audio_path: String,
    /// Exact byte length.
    pub byte_len: u64,
    /// SHA-256 of exact file bytes.
    pub sha256: String,
    /// Validated container name.
    pub wav_container: String,
}

/// One deterministic validation failure.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ValidationIssueV1 {
    /// Stable code.
    pub code: String,
    /// Human-readable explanation.
    pub message: String,
    /// 1-based manifest line, when known.
    pub line: Option<usize>,
    /// Sample ID, when known.
    pub sample_id: Option<String>,
    /// Relative path, when known.
    pub path: Option<String>,
}

/// Complete local preflight report.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ValidationReportV1 {
    /// Report schema version.
    pub schema_version: u16,
    /// Fixed profile identifier.
    pub profile: String,
    /// True only when every structural check passed.
    pub valid: bool,
    /// Caller-supplied expected sample count.
    pub expected_count: usize,
    /// Non-empty JSONL lines attempted.
    pub manifest_entries: usize,
    /// WAV files whose structure and digest were verified.
    pub validated_files: usize,
    /// File evidence in manifest order.
    pub files: Vec<AudioFileEvidenceV1>,
    /// All detected issues.
    pub issues: Vec<ValidationIssueV1>,
}

impl ValidationReportV1 {
    fn new(expected_count: usize) -> Self {
        Self {
            schema_version: REPORT_SCHEMA_VERSION,
            profile: "icasp-2027-mmae-reference-jsonl-v1".to_owned(),
            valid: false,
            expected_count,
            manifest_entries: 0,
            validated_files: 0,
            files: Vec::new(),
            issues: Vec::new(),
        }
    }

    fn issue(
        &mut self,
        code: &str,
        message: impl Into<String>,
        line: Option<usize>,
        sample_id: Option<&str>,
        path: Option<&str>,
    ) {
        self.issues.push(ValidationIssueV1 {
            code: code.to_owned(),
            message: message.into(),
            line,
            sample_id: sample_id.map(str::to_owned),
            path: path.map(str::to_owned),
        });
    }

    fn finish(&mut self) {
        self.validated_files = self.files.len();
        self.valid = self.issues.is_empty()
            && self.manifest_entries == self.expected_count
            && self.validated_files == self.expected_count;
    }
}

/// Read one expected sample ID per UTF-8 line. The caller supplies this list
/// from the released track inputs; IDs are never inferred from filenames.
pub fn read_expected_ids(path: &Path) -> Result<Vec<String>, String> {
    let metadata = fs::metadata(path).map_err(|e| format!("cannot stat expected-ID file: {e}"))?;
    if !metadata.is_file() || metadata.len() > MAX_EXPECTED_IDS_BYTES {
        return Err("expected-ID input must be a regular file within the size limit".to_owned());
    }
    let contents = fs::read_to_string(path).map_err(|e| format!("cannot read expected IDs: {e}"))?;
    let mut ids = Vec::new();
    let mut seen = BTreeSet::new();
    for (index, raw) in contents.lines().enumerate() {
        let id = raw.trim();
        if id.is_empty() {
            return Err(format!("expected-ID line {} is empty", index + 1));
        }
        validate_sample_id(id).map_err(|e| format!("expected-ID line {}: {e}", index + 1))?;
        if !seen.insert(id.to_owned()) {
            return Err(format!("expected-ID line {} duplicates {id:?}", index + 1));
        }
        ids.push(id.to_owned());
    }
    if ids.is_empty() {
        return Err("expected-ID file contains no sample IDs".to_owned());
    }
    Ok(ids)
}

/// Validate a package against caller-supplied expected IDs.
///
/// The profile follows the challenge website's reference row with exactly two
/// fields: id and audio_path. Paths must be audio/<id>.wav. The WAV check
/// verifies RIFF/WAVE chunk integrity, but does not constrain codec subtype,
/// sample rate, channels, bit depth, duration, loudness or perceptual quality.
pub fn validate_package(
    package_root: &Path,
    manifest_relative_path: &Path,
    expected_ids: &[String],
    expected_count: Option<usize>,
) -> ValidationReportV1 {
    let mut report = ValidationReportV1::new(expected_ids.len());
    let expected_set: BTreeSet<&str> = expected_ids.iter().map(String::as_str).collect();

    if expected_ids.is_empty() {
        report.issue("EXPECTED_IDS_EMPTY", "no expected IDs supplied", None, None, None);
    }
    if expected_set.len() != expected_ids.len() {
        report.issue("EXPECTED_IDS_DUPLICATE", "expected-ID list contains duplicates", None, None, None);
    }
    for id in expected_ids {
        if let Err(message) = validate_sample_id(id) {
            report.issue("EXPECTED_ID_INVALID", message, None, Some(id), None);
        }
    }
    if let Some(count) = expected_count {
        if expected_ids.len() != count {
            report.issue(
                "EXPECTED_COUNT_MISMATCH",
                format!("expected-ID list has {}, expected {count}", expected_ids.len()),
                None,
                None,
                None,
            );
        }
    }

    let root = match fs::canonicalize(package_root) {
        Ok(path) if path.is_dir() => path,
        Ok(_) => {
            report.issue("PACKAGE_ROOT_NOT_DIRECTORY", "package root is not a directory", None, None, None);
            report.finish();
            return report;
        }
        Err(error) => {
            report.issue("PACKAGE_ROOT_UNAVAILABLE", format!("cannot resolve package root: {error}"), None, None, None);
            report.finish();
            return report;
        }
    };

    let manifest = match resolve_regular_file(&root, manifest_relative_path) {
        Ok(path) => path,
        Err((code, message)) => {
            report.issue(code, message, None, None, Some(&manifest_relative_path.display().to_string()));
            report.finish();
            return report;
        }
    };
    let metadata = match fs::metadata(&manifest) {
        Ok(metadata) => metadata,
        Err(error) => {
            report.issue("MANIFEST_STAT_FAILED", format!("cannot stat manifest: {error}"), None, None, None);
            report.finish();
            return report;
        }
    };
    if metadata.len() > MAX_MANIFEST_BYTES {
        report.issue("MANIFEST_TOO_LARGE", "manifest exceeds the size limit", None, None, None);
        report.finish();
        return report;
    }
    let text = match fs::read_to_string(&manifest) {
        Ok(text) => text,
        Err(error) => {
            report.issue("MANIFEST_READ_FAILED", format!("cannot read manifest as UTF-8: {error}"), None, None, None);
            report.finish();
            return report;
        }
    };

    let mut seen_ids = BTreeSet::new();
    let mut seen_paths = BTreeSet::new();
    for (index, raw_line) in text.lines().enumerate() {
        let line = index + 1;
        if raw_line.len() > MAX_RECORD_BYTES {
            report.issue("RECORD_TOO_LARGE", "JSONL record exceeds the size limit", Some(line), None, None);
            continue;
        }
        if raw_line.trim().is_empty() {
            report.issue("EMPTY_JSONL_LINE", "blank records are not allowed", Some(line), None, None);
            continue;
        }
        report.manifest_entries += 1;

        let entry: SubmissionEntryV1 = match serde_json::from_str(raw_line) {
            Ok(entry) => entry,
            Err(error) => {
                report.issue("JSONL_RECORD_INVALID", format!("invalid reference JSONL object: {error}"), Some(line), None, None);
                continue;
            }
        };
        if let Err(message) = validate_sample_id(&entry.id) {
            report.issue("SAMPLE_ID_INVALID", message, Some(line), Some(&entry.id), Some(&entry.audio_path));
            continue;
        }
        if !expected_set.contains(entry.id.as_str()) {
            report.issue("UNEXPECTED_SAMPLE_ID", "ID is absent from caller-supplied expected list", Some(line), Some(&entry.id), Some(&entry.audio_path));
        }
        if !seen_ids.insert(entry.id.clone()) {
            report.issue("DUPLICATE_SAMPLE_ID", "sample ID appears more than once", Some(line), Some(&entry.id), Some(&entry.audio_path));
            continue;
        }

        let expected_path = format!("audio/{}.wav", entry.id);
        if entry.audio_path != expected_path {
            report.issue("AUDIO_PATH_MISMATCH", format!("reference profile requires {expected_path:?}"), Some(line), Some(&entry.id), Some(&entry.audio_path));
            continue;
        }
        if !seen_paths.insert(entry.audio_path.clone()) {
            report.issue("DUPLICATE_AUDIO_PATH", "multiple records reference the same path", Some(line), Some(&entry.id), Some(&entry.audio_path));
            continue;
        }

        let resolved = match resolve_regular_file(&root, Path::new(&entry.audio_path)) {
            Ok(path) => path,
            Err((code, message)) => {
                report.issue(code, message, Some(line), Some(&entry.id), Some(&entry.audio_path));
                continue;
            }
        };
        let audio_metadata = match fs::metadata(&resolved) {
            Ok(metadata) => metadata,
            Err(error) => {
                report.issue("AUDIO_STAT_FAILED", format!("cannot stat WAV: {error}"), Some(line), Some(&entry.id), Some(&entry.audio_path));
                continue;
            }
        };
        if audio_metadata.len() == 0 {
            report.issue("AUDIO_EMPTY", "audio file is empty", Some(line), Some(&entry.id), Some(&entry.audio_path));
            continue;
        }
        if let Err(message) = validate_wav_container(&resolved, audio_metadata.len()) {
            report.issue("WAV_CONTAINER_INVALID", message, Some(line), Some(&entry.id), Some(&entry.audio_path));
            continue;
        }
        match sha256_file(&resolved) {
            Ok(digest) => report.files.push(AudioFileEvidenceV1 {
                id: entry.id,
                audio_path: entry.audio_path,
                byte_len: audio_metadata.len(),
                sha256: digest,
                wav_container: "RIFF/WAVE".to_owned(),
            }),
            Err(error) => report.issue("AUDIO_HASH_FAILED", format!("cannot hash WAV bytes: {error}"), Some(line), Some(&entry.id), Some(&entry.audio_path)),
        }
    }

    for id in expected_ids {
        if !seen_ids.contains(id) {
            report.issue("EXPECTED_SAMPLE_MISSING", "expected ID does not appear in JSONL", None, Some(id), Some(&format!("audio/{id}.wav")));
        }
    }
    if report.manifest_entries != expected_ids.len() {
        report.issue(
            "MANIFEST_COUNT_MISMATCH",
            format!("manifest has {} records; expected {}", report.manifest_entries, expected_ids.len()),
            None,
            None,
            None,
        );
    }
    report.finish();
    report
}

fn validate_sample_id(id: &str) -> Result<(), String> {
    if id.trim().is_empty() {
        return Err("sample ID is empty".to_owned());
    }
    if id == "." || id == ".." {
        return Err("sample ID cannot be a dot path".to_owned());
    }
    if id.contains('/') || id.contains('\\') || id.contains(':') || id.chars().any(char::is_control) {
        return Err("sample ID contains a path separator, drive delimiter or control character".to_owned());
    }
    Ok(())
}

fn resolve_regular_file(root: &Path, relative: &Path) -> Result<PathBuf, (&'static str, String)> {
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative.to_string_lossy().contains('\\')
        || relative.to_string_lossy().contains(':')
    {
        return Err(("PATH_NOT_RELATIVE", "path must be relative and contain no backslash or drive prefix".to_owned()));
    }
    let mut current = root.to_path_buf();
    for component in relative.components() {
        match component {
            Component::Normal(part) => current.push(part),
            _ => return Err(("PATH_COMPONENT_INVALID", "absolute, parent and current-directory components are rejected".to_owned())),
        }
        let metadata = fs::symlink_metadata(&current)
            .map_err(|error| ("PATH_NOT_FOUND", format!("path component is unavailable: {error}")))?;
        if metadata.file_type().is_symlink() {
            return Err(("SYMLINK_REJECTED", "symlinked package paths are not accepted".to_owned()));
        }
    }
    let canonical = fs::canonicalize(&current)
        .map_err(|error| ("PATH_CANONICALIZE_FAILED", format!("cannot canonicalize package path: {error}")))?;
    if !canonical.starts_with(root) {
        return Err(("PATH_ESCAPES_PACKAGE", "resolved path escapes package root".to_owned()));
    }
    let metadata = fs::metadata(&canonical)
        .map_err(|error| ("PATH_STAT_FAILED", format!("cannot stat resolved path: {error}")))?;
    if !metadata.is_file() {
        return Err(("PATH_NOT_FILE", "resolved path is not a regular file".to_owned()));
    }
    Ok(canonical)
}

fn validate_wav_container(path: &Path, actual_len: u64) -> Result<(), String> {
    let mut file = File::open(path).map_err(|error| format!("cannot open WAV: {error}"))?;
    if actual_len < 44 {
        return Err("WAV is shorter than the minimum RIFF/WAVE structure".to_owned());
    }
    let mut header = [0u8; 12];
    file.read_exact(&mut header).map_err(|error| format!("cannot read WAV header: {error}"))?;
    if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        return Err("expected little-endian RIFF/WAVE".to_owned());
    }
    let declared_len = u64::from(u32::from_le_bytes([header[4], header[5], header[6], header[7]])) + 8;
    if declared_len != actual_len {
        return Err(format!("RIFF declares {declared_len} bytes but file has {actual_len}"));
    }

    let mut offset = 12u64;
    let mut has_fmt = false;
    let mut has_data = false;
    while offset < actual_len {
        if actual_len - offset < 8 {
            return Err("truncated WAV chunk header".to_owned());
        }
        file.seek(SeekFrom::Start(offset)).map_err(|error| format!("cannot seek WAV chunk: {error}"))?;
        let mut chunk = [0u8; 8];
        file.read_exact(&mut chunk).map_err(|error| format!("cannot read WAV chunk header: {error}"))?;
        let name = &chunk[0..4];
        let size = u64::from(u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]));
        let end = (offset + 8).checked_add(size).ok_or_else(|| "WAV chunk length overflow".to_owned())?;
        if end > actual_len {
            return Err("WAV chunk extends beyond file boundary".to_owned());
        }
        if name == b"fmt " {
            if size < 16 {
                return Err("WAV fmt chunk is shorter than 16 bytes".to_owned());
            }
            has_fmt = true;
        }
        if name == b"data" {
            if size == 0 {
                return Err("WAV data chunk is empty".to_owned());
            }
            has_data = true;
        }
        offset = end.checked_add(size % 2).ok_or_else(|| "WAV padding overflow".to_owned())?;
        if offset > actual_len {
            return Err("WAV odd-sized chunk is missing its pad byte".to_owned());
        }
    }
    if !has_fmt {
        return Err("WAV fmt chunk is missing".to_owned());
    }
    if !has_data {
        return Err("WAV data chunk is missing".to_owned());
    }
    Ok(())
}

/// Compute SHA-256 over exact file bytes with bounded memory.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let mut output = String::with_capacity(64);
    for byte in hasher.finalize() {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(1);

    struct TempDir(PathBuf);
    impl TempDir {
        fn new() -> Self {
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("mel-mmae-validator-{}-{n}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn path(&self) -> &Path { &self.0 }
    }
    impl Drop for TempDir {
        fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
    }

    fn wave(data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        let body_len = 4 + 8 + 16 + 8 + data.len() + data.len() % 2;
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(body_len as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&8_000u32.to_le_bytes());
        bytes.extend_from_slice(&16_000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(data);
        if data.len() % 2 != 0 { bytes.push(0); }
        bytes
    }

    fn pkg(ids: &[&str]) -> TempDir {
        let root = TempDir::new();
        fs::create_dir_all(root.path().join("audio")).unwrap();
        for id in ids {
            fs::write(root.path().join("audio").join(format!("{id}.wav")), wave(&[0, 0])).unwrap();
        }
        let lines = ids.iter().map(|id| format!(r#"{{"id":"{id}","audio_path":"audio/{id}.wav"}}"#)).collect::<Vec<_>>().join("\n");
        fs::write(root.path().join(DEFAULT_MANIFEST_NAME), format!("{lines}\n")).unwrap();
        root
    }
    fn ids(values: &[&str]) -> Vec<String> { values.iter().map(|x| (*x).to_owned()).collect() }
    fn run(root: &TempDir, expected: &[&str]) -> ValidationReportV1 {
        validate_package(root.path(), Path::new(DEFAULT_MANIFEST_NAME), &ids(expected), Some(expected.len()))
    }

    #[test]
    fn valid_reference_package_passes_and_emits_hashes() {
        let root = pkg(&["id-a", "id-b"]);
        let report = run(&root, &["id-a", "id-b"]);
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.validated_files, 2);
        assert_eq!(report.files[0].sha256.len(), 64);
        assert_eq!(report.files[0].wav_container, "RIFF/WAVE");
    }

    #[test]
    fn missing_and_extra_ids_are_rejected() {
        let missing = pkg(&["id-a"]);
        assert!(run(&missing, &["id-a", "id-b"]).issues.iter().any(|x| x.code == "EXPECTED_SAMPLE_MISSING"));
        let extra = pkg(&["id-a", "id-extra"]);
        assert!(run(&extra, &["id-a"]).issues.iter().any(|x| x.code == "UNEXPECTED_SAMPLE_ID"));
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let root = pkg(&["id-a"]);
        let row = r#"{"id":"id-a","audio_path":"audio/id-a.wav"}"#;
        fs::write(root.path().join(DEFAULT_MANIFEST_NAME), format!("{row}\n{row}\n")).unwrap();
        assert!(run(&root, &["id-a"]).issues.iter().any(|x| x.code == "DUPLICATE_SAMPLE_ID"));
    }

    #[test]
    fn duplicate_json_keys_and_unknown_fields_are_rejected() {
        let root = pkg(&["id-a"]);
        let path = root.path().join(DEFAULT_MANIFEST_NAME);
        fs::write(&path, r#"{"id":"id-a","id":"id-a","audio_path":"audio/id-a.wav"}"#).unwrap();
        assert!(run(&root, &["id-a"]).issues.iter().any(|x| x.code == "JSONL_RECORD_INVALID"));
        fs::write(&path, r#"{"id":"id-a","audio_path":"audio/id-a.wav","extra":1}"#).unwrap();
        assert!(run(&root, &["id-a"]).issues.iter().any(|x| x.code == "JSONL_RECORD_INVALID"));
    }

    #[test]
    fn path_must_match_documented_id_wav_mapping() {
        let root = pkg(&["id-a"]);
        fs::write(root.path().join(DEFAULT_MANIFEST_NAME), r#"{"id":"id-a","audio_path":"audio/other.wav"}"#).unwrap();
        assert!(run(&root, &["id-a"]).issues.iter().any(|x| x.code == "AUDIO_PATH_MISMATCH"));
        assert!(validate_sample_id("../escape").is_err());
        assert!(validate_sample_id(r"..\escape").is_err());
    }

    #[test]
    fn blank_lines_are_rejected() {
        let root = pkg(&["id-a"]);
        let row = r#"{"id":"id-a","audio_path":"audio/id-a.wav"}"#;
        fs::write(root.path().join(DEFAULT_MANIFEST_NAME), format!("{row}\n\n")).unwrap();
        assert!(run(&root, &["id-a"]).issues.iter().any(|x| x.code == "EMPTY_JSONL_LINE"));
    }

    #[test]
    fn zero_byte_and_non_wav_audio_are_rejected() {
        let root = pkg(&["id-a"]);
        let path = root.path().join("audio/id-a.wav");
        fs::write(&path, []).unwrap();
        assert!(run(&root, &["id-a"]).issues.iter().any(|x| x.code == "AUDIO_EMPTY"));
        fs::write(&path, b"not wav").unwrap();
        assert!(run(&root, &["id-a"]).issues.iter().any(|x| x.code == "WAV_CONTAINER_INVALID"));
    }

    #[test]
    fn riff_size_must_match_actual_file_length() {
        let root = pkg(&["id-a"]);
        let path = root.path().join("audio/id-a.wav");
        let mut bytes = wave(&[0, 0]);
        bytes[4..8].copy_from_slice(&0u32.to_le_bytes());
        fs::write(path, bytes).unwrap();
        assert!(run(&root, &["id-a"]).issues.iter().any(|x| x.code == "WAV_CONTAINER_INVALID"));
    }

    #[test]
    fn empty_data_chunk_is_rejected() {
        let root = pkg(&["id-a"]);
        fs::write(root.path().join("audio/id-a.wav"), wave(&[])).unwrap();
        assert!(run(&root, &["id-a"]).issues.iter().any(|x| x.code == "WAV_CONTAINER_INVALID"));
    }

    #[test]
    fn expected_count_mismatch_is_rejected() {
        let root = pkg(&["id-a"]);
        let report = validate_package(root.path(), Path::new(DEFAULT_MANIFEST_NAME), &ids(&["id-a"]), Some(500));
        assert!(report.issues.iter().any(|x| x.code == "EXPECTED_COUNT_MISMATCH"));
    }

    #[test]
    fn expected_id_file_rejects_blank_and_duplicate_values() {
        let root = TempDir::new();
        let ids_path = root.path().join("ids.txt");
        fs::write(&ids_path, "id-a\n\n").unwrap();
        assert!(read_expected_ids(&ids_path).is_err());
        fs::write(&ids_path, "id-a\nid-a\n").unwrap();
        assert!(read_expected_ids(&ids_path).is_err());
    }

    #[test]
    fn sha256_is_deterministic_for_same_bytes() {
        let root = pkg(&["id-a"]);
        let path = root.path().join("audio/id-a.wav");
        assert_eq!(sha256_file(&path).unwrap(), sha256_file(&path).unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_audio_file_is_rejected() {
        use std::os::unix::fs::symlink;
        let root = pkg(&["id-a"]);
        let outside = root.path().join("outside.wav");
        fs::write(&outside, wave(&[0, 0])).unwrap();
        fs::remove_file(root.path().join("audio/id-a.wav")).unwrap();
        symlink(&outside, root.path().join("audio/id-a.wav")).unwrap();
        assert!(run(&root, &["id-a"]).issues.iter().any(|x| x.code == "SYMLINK_REJECTED"));
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_audio_directory_is_rejected() {
        use std::os::unix::fs::symlink;
        let root = TempDir::new();
        let outside = root.path().join("outside");
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("id-a.wav"), wave(&[0, 0])).unwrap();
        symlink(&outside, root.path().join("audio")).unwrap();
        fs::write(root.path().join(DEFAULT_MANIFEST_NAME), r#"{"id":"id-a","audio_path":"audio/id-a.wav"}"#).unwrap();
        assert!(run(&root, &["id-a"]).issues.iter().any(|x| x.code == "SYMLINK_REJECTED"));
    }

    #[test]
    fn missing_manifest_is_rejected() {
        let root = pkg(&["id-a"]);
        fs::remove_file(root.path().join(DEFAULT_MANIFEST_NAME)).unwrap();
        assert!(run(&root, &["id-a"]).issues.iter().any(|x| x.code == "PATH_NOT_FOUND"));
    }
}
