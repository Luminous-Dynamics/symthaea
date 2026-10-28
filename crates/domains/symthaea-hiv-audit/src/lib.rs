// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Fail-closed dataset auditing for the Symthaea HIV Reservoir Discovery benchmark.
//!
//! This tool does not download data or infer biological conclusions. It captures
//! local file hashes and structural checks after the researcher has obtained data
//! from the registered public source. The receipt is a reproducibility aid, not
//! proof that a file is authentic or that its source publication is correct.

use csv::StringRecord;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const RECEIPT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
pub struct DatasetManifest {
    pub schema_version: String,
    pub entries: Vec<DatasetEntry>,
}

#[derive(Debug, Deserialize)]
pub struct DatasetEntry {
    pub id: String,
    pub title: String,
    pub source_url: String,
    #[serde(default)]
    pub source_version: Option<String>,
    #[serde(default)]
    pub described_cohort_size: Option<usize>,
    #[serde(default)]
    pub data_files_described: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetReceipt {
    pub schema_version: u32,
    pub dataset_id: String,
    pub dataset_title: String,
    pub source_url: String,
    pub source_version: Option<String>,
    pub cohort_size_reported_by_source: Option<usize>,
    pub captured_at_unix_seconds: u64,
    pub source_commit: String,
    pub source_archive: SourceArchiveAudit,
    pub files: Vec<FileAudit>,
    pub pairwise_uuid_joins: Vec<JoinAudit>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceArchiveAudit {
    /// Filename only; absolute local paths are deliberately not persisted.
    pub filename: String,
    pub byte_length: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileAudit {
    pub path: String,
    pub byte_length: u64,
    pub sha256: String,
    pub csv: Option<CsvAudit>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CsvAudit {
    /// Number of data records, excluding the header row.
    pub data_rows: usize,
    pub columns: usize,
    pub unique_uuid_count: usize,
    pub duplicate_uuid_rows: usize,
    /// Hash of sorted participant identifiers; identifiers themselves are not stored.
    pub uuid_set_sha256: String,
    pub header_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JoinAudit {
    pub left_file: String,
    pub right_file: String,
    pub left_unique_uuids: usize,
    pub right_unique_uuids: usize,
    pub shared_uuids: usize,
    pub only_left: usize,
    pub only_right: usize,
}

#[derive(Debug)]
struct ScannedFile {
    audit: FileAudit,
    uuids: Option<HashSet<String>>,
}

pub fn parse_manifest(path: &Path) -> Result<DatasetManifest, String> {
    let raw = fs::read(path).map_err(|e| format!("read manifest {}: {e}", path.display()))?;
    let manifest: DatasetManifest =
        serde_json::from_slice(&raw).map_err(|e| format!("parse manifest JSON: {e}"))?;
    if manifest.schema_version.trim().is_empty() {
        return Err("manifest schema_version is empty".to_owned());
    }
    if manifest.entries.is_empty() {
        return Err("manifest has no dataset entries".to_owned());
    }
    Ok(manifest)
}

pub fn select_dataset<'a>(
    manifest: &'a DatasetManifest,
    dataset_id: &str,
) -> Result<&'a DatasetEntry, String> {
    let mut matches = manifest.entries.iter().filter(|entry| entry.id == dataset_id);
    let entry = matches
        .next()
        .ok_or_else(|| format!("dataset id {dataset_id:?} not found in manifest"))?;
    if matches.next().is_some() {
        return Err(format!("dataset id {dataset_id:?} is duplicated in manifest"));
    }
    if !entry.source_url.starts_with("https://") {
        return Err(format!("dataset {} has a non-HTTPS source URL", entry.id));
    }
    if entry.data_files_described.is_empty() {
        return Err(format!("dataset {} has no registered data_files_described", entry.id));
    }
    let mut seen = HashSet::new();
    for name in &entry.data_files_described {
        validate_plain_filename(name)?;
        if !seen.insert(name) {
            return Err(format!("dataset {} repeats registered file {name:?}", entry.id));
        }
    }
    Ok(entry)
}

fn validate_plain_filename(name: &str) -> Result<(), String> {
    if name.trim().is_empty() {
        return Err("empty filename in dataset registry".to_owned());
    }
    let mut parts = Path::new(name).components();
    match (parts.next(), parts.next()) {
        (Some(Component::Normal(_)), None) if !name.contains('/') && !name.contains('\\') => Ok(()),
        _ => Err(format!("registered path must be a plain filename, got {name:?}")),
    }
}

pub fn scan_dataset(
    entry: &DatasetEntry,
    data_dir: &Path,
) -> Result<(Vec<FileAudit>, Vec<JoinAudit>), String> {
    if !data_dir.is_dir() {
        return Err(format!("data directory does not exist: {}", data_dir.display()));
    }

    let expected: HashSet<&str> = entry.data_files_described.iter().map(String::as_str).collect();
    for name in &entry.data_files_described {
        let path = data_dir.join(name);
        if !path.is_file() {
            return Err(format!("required source file missing: {}", path.display()));
        }
    }

    // Unregistered CSVs are a common source of accidental file/version confusion.
    for item in fs::read_dir(data_dir).map_err(|e| format!("read data directory: {e}"))? {
        let item = item.map_err(|e| format!("read data directory entry: {e}"))?;
        let path = item.path();
        if !path.is_file() {
            continue;
        }
        let is_csv = path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("csv"));
        if is_csv {
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or_default();
            if !expected.contains(name.as_str()) {
                return Err(format!("unregistered CSV file in data directory: {}", path.display()));
            }
        }
    }

    let mut scanned = Vec::with_capacity(entry.data_files_described.len());
    for name in &entry.data_files_described {
        let path = data_dir.join(name);
        let byte_length = fs::metadata(&path)
            .map_err(|e| format!("stat {}: {e}", path.display()))?
            .len();
        let sha256 = sha256_file(&path)?;
        let (csv, uuids) = if path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("csv"))
        {
            let (audit, ids) = audit_csv(&path)?;
            if audit.duplicate_uuid_rows != 0 {
                return Err(format!(
                    "{} has {} duplicate UUID row(s); investigate before capturing a receipt",
                    path.display(), audit.duplicate_uuid_rows
                ));
            }
            (Some(audit), Some(ids))
        } else {
            (None, None)
        };
        scanned.push(ScannedFile {
            audit: FileAudit { path: name.clone(), byte_length, sha256, csv },
            uuids,
        });
    }

    let mut joins = Vec::new();
    for left in 0..scanned.len() {
        let Some(left_ids) = scanned[left].uuids.as_ref() else {
            continue;
        };
        for right in (left + 1)..scanned.len() {
            let Some(right_ids) = scanned[right].uuids.as_ref() else {
                continue;
            };
            let shared = left_ids.intersection(right_ids).count();
            joins.push(JoinAudit {
                left_file: scanned[left].audit.path.clone(),
                right_file: scanned[right].audit.path.clone(),
                left_unique_uuids: left_ids.len(),
                right_unique_uuids: right_ids.len(),
                shared_uuids: shared,
                only_left: left_ids.len() - shared,
                only_right: right_ids.len() - shared,
            });
        }
    }

    Ok((scanned.into_iter().map(|file| file.audit).collect(), joins))
}

fn audit_csv(path: &Path) -> Result<(CsvAudit, HashSet<String>), String> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_path(path)
        .map_err(|e| format!("open CSV {}: {e}", path.display()))?;

    let headers: StringRecord = reader.headers()
        .map_err(|e| format!("read CSV header {}: {e}", path.display()))?
        .clone();
    if headers.len() < 2 {
        return Err(format!("{} has fewer than two columns", path.display()));
    }
    let first = headers.get(0).unwrap_or_default().trim_start_matches('\u{feff}');
    if !first.eq_ignore_ascii_case("uuid") {
        return Err(format!(
            "{} first column must be the de-identified participant key 'uuid', found {:?}",
            path.display(), first
        ));
    }

    let mut header_names = HashSet::with_capacity(headers.len());
    for header in headers.iter() {
        if header.trim().is_empty() {
            return Err(format!("{} has an empty column name", path.display()));
        }
        if !header_names.insert(header) {
            return Err(format!("{} has duplicate column name {:?}", path.display(), header));
        }
    }

    let mut ids = HashSet::new();
    let mut rows = 0usize;
    let mut duplicate_rows = 0usize;
    for row in reader.records() {
        let row = row.map_err(|e| format!("parse CSV record in {}: {e}", path.display()))?;
        rows += 1;
        let id = row.get(0).unwrap_or_default().trim();
        if id.is_empty() {
            return Err(format!("{} has an empty UUID at data row {}", path.display(), rows));
        }
        if !ids.insert(id.to_owned()) {
            duplicate_rows += 1;
        }
    }

    if rows == 0 || ids.is_empty() {
        return Err(format!("{} has no participant data rows", path.display()));
    }

    let header_serialized = headers.iter().collect::<Vec<_>>().join("\u{1f}");
    let mut sorted_ids: Vec<&str> = ids.iter().map(String::as_str).collect();
    sorted_ids.sort_unstable();
    let uuid_set_serialized = sorted_ids.join("\n");

    Ok((
        CsvAudit {
            data_rows: rows,
            columns: headers.len(),
            unique_uuid_count: ids.len(),
            duplicate_uuid_rows: duplicate_rows,
            uuid_set_sha256: sha256_bytes(uuid_set_serialized.as_bytes()),
            header_sha256: sha256_bytes(header_serialized.as_bytes()),
        },
        ids,
    ))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|e| format!("open {} for hashing: {e}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|e| format!("read {} for hashing: {e}", path.display()))?;
        if read == 0 { break; }
        digest.update(&buffer[..read]);
    }
    Ok(hex_digest(&digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    hex_digest(&digest.finalize())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

pub fn capture_receipt(
    manifest_path: &Path,
    dataset_id: &str,
    data_dir: &Path,
    source_commit: &str,
    source_archive_path: &Path,
) -> Result<DatasetReceipt, String> {
    validate_commit(source_commit)?;
    let manifest = parse_manifest(manifest_path)?;
    let entry = select_dataset(&manifest, dataset_id)?;
    let (files, pairwise_uuid_joins) = scan_dataset(entry, data_dir)?;
    let source_archive = audit_source_archive(source_archive_path)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH)
        .map_err(|e| format!("system clock is before Unix epoch: {e}"))?.as_secs();

    Ok(DatasetReceipt {
        schema_version: RECEIPT_SCHEMA_VERSION,
        dataset_id: entry.id.clone(),
        dataset_title: entry.title.clone(),
        source_url: entry.source_url.clone(),
        source_version: entry.source_version.clone(),
        cohort_size_reported_by_source: entry.described_cohort_size,
        captured_at_unix_seconds: now,
        source_commit: source_commit.to_owned(),
        source_archive,
        files,
        pairwise_uuid_joins,
    })
}

pub fn verify_receipt(
    manifest_path: &Path,
    dataset_id: &str,
    data_dir: &Path,
    receipt_path: &Path,
    source_archive_path: &Path,
) -> Result<DatasetReceipt, String> {
    let manifest = parse_manifest(manifest_path)?;
    let entry = select_dataset(&manifest, dataset_id)?;
    let raw = fs::read(receipt_path)
        .map_err(|e| format!("read receipt {}: {e}", receipt_path.display()))?;
    let receipt: DatasetReceipt = serde_json::from_slice(&raw)
        .map_err(|e| format!("parse receipt JSON: {e}"))?;

    if receipt.schema_version != RECEIPT_SCHEMA_VERSION {
        return Err(format!("unsupported receipt schema version {}", receipt.schema_version));
    }
    if receipt.dataset_id != entry.id || receipt.source_url != entry.source_url {
        return Err("receipt dataset id/source URL does not match the current registry".to_owned());
    }
    if receipt.dataset_title != entry.title
        || receipt.source_version != entry.source_version
        || receipt.cohort_size_reported_by_source != entry.described_cohort_size
    {
        return Err("receipt dataset metadata differs from the current registry".to_owned());
    }
    validate_commit(&receipt.source_commit)?;

    let actual_archive = audit_source_archive(source_archive_path)?;
    if receipt.source_archive != actual_archive {
        return Err(
            "source archive receipt mismatch: filename, byte length, or SHA-256 changed".to_owned(),
        );
    }
    let (actual_files, actual_joins) = scan_dataset(entry, data_dir)?;
    if receipt.files != actual_files {
        return Err(
            "file receipt mismatch: size, SHA-256, CSV schema, rows, or UUID-set hash changed"
                .to_owned(),
        );
    }
    if receipt.pairwise_uuid_joins != actual_joins {
        return Err("pairwise UUID join audit differs from the captured receipt".to_owned());
    }
    Ok(receipt)
}

fn audit_source_archive(path: &Path) -> Result<SourceArchiveAudit, String> {
    if !path.is_file() {
        return Err(format!("source archive does not exist: {}", path.display()));
    }
    let filename = path.file_name().and_then(|s| s.to_str())
        .ok_or_else(|| "source archive path has no UTF-8 filename".to_owned())?
        .to_owned();
    validate_plain_filename(&filename)?;
    if !path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("zip"))
    {
        return Err("source archive must have a .zip extension".to_owned());
    }
    let mut prefix = [0u8; 4];
    File::open(path)
        .and_then(|mut file| file.read_exact(&mut prefix))
        .map_err(|e| format!("read ZIP signature from {}: {e}", path.display()))?;
    if prefix != [0x50, 0x4b, 0x03, 0x04]
        && prefix != [0x50, 0x4b, 0x05, 0x06]
        && prefix != [0x50, 0x4b, 0x07, 0x08]
    {
        return Err("source archive does not have a recognized ZIP signature".to_owned());
    }
    let byte_length = fs::metadata(path)
        .map_err(|e| format!("stat source archive {}: {e}", path.display()))?.len();
    let sha256 = sha256_file(path)?;
    Ok(SourceArchiveAudit { filename, byte_length, sha256 })
}

fn validate_commit(commit: &str) -> Result<(), String> {
    if commit.len() < 7 || commit.len() > 64
        || !commit.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("source commit must be a 7-64 character hexadecimal Git commit id".to_owned());
    }
    Ok(())
}

pub fn write_receipt_atomic(path: &Path, receipt: &DatasetReceipt) -> Result<(), String> {
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|e| format!("create receipt directory {}: {e}", parent.display()))?;

    let temporary = temporary_path(path);
    let bytes = serde_json::to_vec_pretty(receipt).map_err(|e| format!("serialize receipt: {e}"))?;
    let mut file = File::create(&temporary)
        .map_err(|e| format!("create temporary receipt {}: {e}", temporary.display()))?;
    file.write_all(&bytes).map_err(|e| format!("write receipt: {e}"))?;
    file.write_all(b"\n").map_err(|e| format!("finish receipt: {e}"))?;
    file.sync_all().map_err(|e| format!("sync receipt: {e}"))?;
    fs::rename(&temporary, path).map_err(|e| {
        let _ = fs::remove_file(&temporary);
        format!("atomically publish receipt {}: {e}", path.display())
    })?;
    Ok(())
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp.{}", std::process::id()));
    path.with_file_name(name)
}

pub fn print_receipt_summary(receipt: &DatasetReceipt) {
    println!("dataset_id: {}", receipt.dataset_id);
    println!("source_url: {}", receipt.source_url);
    println!("source_version: {}", receipt.source_version.as_deref().unwrap_or("not registered"));
    println!("source_commit: {}", receipt.source_commit);
    println!("captured_at_unix_seconds: {}", receipt.captured_at_unix_seconds);
    println!("source_archive: {} ({} bytes, sha256={})",
        receipt.source_archive.filename,
        receipt.source_archive.byte_length,
        receipt.source_archive.sha256
    );
    println!("files: {}", receipt.files.len());
    for file in &receipt.files {
        print!("  {}: {} bytes sha256={}", file.path, file.byte_length, file.sha256);
        if let Some(csv) = &file.csv {
            print!(" rows={} columns={} unique_uuids={} duplicate_uuid_rows={}",
                csv.data_rows, csv.columns, csv.unique_uuid_count, csv.duplicate_uuid_rows);
        }
        println!();
    }
    println!("pairwise_uuid_joins: {}", receipt.pairwise_uuid_joins.len());
    for join in &receipt.pairwise_uuid_joins {
        println!("  {} <-> {}: shared={} only_left={} only_right={}",
            join.left_file, join.right_file, join.shared_uuids, join.only_left, join.only_right);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_uses_standard_known_vector() {
        assert_eq!(sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn source_archive_requires_zip_extension_and_signature() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-hiv-audit-source-{}.zip",
            std::process::id()
        ));
        fs::write(&path, &[0x50_u8, 0x4b, 0x03, 0x04]).expect("write ZIP-signature fixture");
        let audit = audit_source_archive(&path).expect("accept recognized ZIP signature");
        assert_eq!(audit.byte_length, 4);
        assert_eq!(
            audit.filename,
            path.file_name().unwrap().to_string_lossy().to_string()
        );
        let _ = fs::remove_file(&path);

        let not_zip = std::env::temp_dir().join(format!(
            "symthaea-hiv-audit-not-zip-{}.zip",
            std::process::id()
        ));
        fs::write(&not_zip, b"not a zip").expect("write non-ZIP fixture");
        assert!(audit_source_archive(&not_zip).is_err());
        let _ = fs::remove_file(not_zip);
    }

    #[test]
    fn csv_rows_with_wrong_width_fail_closed() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-hiv-audit-width-{}.csv",
            std::process::id()
        ));
        fs::write(&path, "uuid,gene_a,gene_b\nparticipant-a,2\n")
            .expect("write malformed CSV fixture");
        assert!(audit_csv(&path).is_err());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn commit_ids_must_be_hexadecimal() {
        assert!(validate_commit("c6355cc6ffe01843b841f6b9a8a09eabd2556deb").is_ok());
        assert!(validate_commit("not-a-commit").is_err());
        assert!(validate_commit("123").is_err());
    }

    #[test]
    fn paths_must_be_plain_filenames() {
        assert!(validate_plain_filename("data_clinical.csv").is_ok());
        assert!(validate_plain_filename("../data_clinical.csv").is_err());
        assert!(validate_plain_filename("/tmp/data.csv").is_err());
        assert!(validate_plain_filename("folder/data.csv").is_err());
    }

    #[test]
    fn parses_csv_and_hashes_uuid_set_without_emitting_identifiers() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-hiv-audit-unique-{}.csv",
            std::process::id()
        ));
        fs::write(&path, "uuid,gene_a,gene_b\nparticipant-a,2,3\nparticipant-b,4,5\n")
            .expect("write test CSV");
        let (audit, ids) = audit_csv(&path).expect("audit valid CSV");
        assert_eq!(audit.data_rows, 2);
        assert_eq!(audit.columns, 3);
        assert_eq!(audit.unique_uuid_count, 2);
        assert_eq!(audit.duplicate_uuid_rows, 0);
        assert_eq!(ids.len(), 2);
        assert!(!audit.uuid_set_sha256.contains("participant-a"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn duplicate_uuid_rows_are_counted_for_fail_closed_capture() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-hiv-audit-duplicate-{}.csv",
            std::process::id()
        ));
        fs::write(&path, "uuid,gene_a\nparticipant-a,2\nparticipant-a,9\n")
            .expect("write test CSV");
        let (audit, ids) = audit_csv(&path).expect("audit syntactically valid CSV");
        assert_eq!(audit.data_rows, 2);
        assert_eq!(audit.unique_uuid_count, 1);
        assert_eq!(audit.duplicate_uuid_rows, 1);
        assert_eq!(ids.len(), 1);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn dataset_registry_selection_fails_closed_on_unknown_ids() {
        let manifest = DatasetManifest {
            schema_version: "1.0".to_owned(),
            entries: vec![DatasetEntry {
                id: "dryad_test".to_owned(),
                title: "test".to_owned(),
                source_url: "https://example.org/data".to_owned(),
                described_cohort_size: Some(2),
                data_files_described: vec!["data.csv".to_owned()],
            }],
        };
        assert!(select_dataset(&manifest, "missing").is_err());
        assert!(select_dataset(&manifest, "dryad_test").is_ok());
    }
}
