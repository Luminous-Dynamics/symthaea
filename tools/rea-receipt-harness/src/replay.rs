//! Independent member-claim replay using zip-rs 9.3.0 and its zlib-rs Deflate backend.
//! This read-only verifier abstains on duplicate-name archives because zip-rs indexes unique names.

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use zip::CompressionMethod;

const HARD_ARCHIVE_LIMIT: u64 = 512 * 1024 * 1024;
const HARD_ENTRY_LIMIT: u64 = 100_000;
const HARD_BUFFER_LIMIT: usize = 1024 * 1024;
const HARD_CENTRAL_DIRECTORY_LIMIT: u64 = 64 * 1024 * 1024;
const EOCD_SIGNATURE: u32 = 0x0605_4b50;
const EOCD_FIXED_BYTES: usize = 22;
const EOCD_MAX_COMMENT_BYTES: usize = u16::MAX as usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    Unsupported(String),
    Rejected(String),
    Io(String),
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(s) => write!(f, "unsupported replay: {s}"),
            Self::Rejected(s) => write!(f, "replay rejected: {s}"),
            Self::Io(s) => write!(f, "replay I/O: {s}"),
        }
    }
}
impl std::error::Error for ReplayError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayReport {
    pub archive_sha256: String,
    pub payload_sha256: String,
    pub observed_bytes: u64,
    pub observed_crc32: u32,
    pub oracle: &'static str,
    pub codec: &'static str,
}

pub fn replay_member_claim(
    path: impl AsRef<Path>,
    receipt: &Value,
) -> Result<ReplayReport, ReplayError> {
    if string_at(receipt, &["outcome"])? != "verified" {
        return Err(ReplayError::Rejected(
            "only verified member receipts are replayable".into(),
        ));
    }
    let archive_limit =
        decimal_at(receipt, &["limits", "max_archive_bytes"])?.min(HARD_ARCHIVE_LIMIT);
    let archive_size = decimal_at(receipt, &["source", "archive_size_bytes"])?;
    // Keep one opened object for the entire operation. The parser receives a cloned handle
    // to this same object, not a second path open that could resolve to a replacement file.
    let mut source = File::open(path.as_ref()).map_err(io_error)?;
    let (archive_sha, actual_size) = hash_open_file(&mut source, archive_limit)?;
    if actual_size != archive_size {
        return Err(ReplayError::Rejected(
            "archive size differs from receipt".into(),
        ));
    }
    if archive_sha != string_at(receipt, &["source", "archive_sha256"])? {
        return Err(ReplayError::Rejected(
            "archive SHA-256 differs from receipt".into(),
        ));
    }

    let max_central_directory = decimal_at(receipt, &["limits", "max_central_directory_bytes"])?
        .min(HARD_CENTRAL_DIRECTORY_LIMIT);
    let max_entries = decimal_at(receipt, &["limits", "max_entries"])?.min(HARD_ENTRY_LIMIT);
    let preflight_entries =
        preflight_regular_zip(&mut source, actual_size, max_central_directory, max_entries)?;

    let parser_handle = source.try_clone().map_err(io_error)?;
    let mut archive = zip::ZipArchive::new(parser_handle)
        .map_err(|e| ReplayError::Unsupported(format!("zip-rs cannot parse this layout: {e}")))?;
    if archive.has_duplicate_names() {
        return Err(ReplayError::Unsupported(
            "zip-rs indexed view collapses duplicate raw names; occurrence cannot be proven".into(),
        ));
    }
    if archive.len() as u64 != preflight_entries {
        return Err(ReplayError::Rejected(
            "zip-rs unique-name count differs from EOCD entry count".into(),
        ));
    }
    archive.validate_header_consistency().map_err(|error| {
        ReplayError::Rejected(format!("zip-rs header consistency failure: {error}"))
    })?;
    if archive
        .has_overlapping_entries()
        .map_err(|error| ReplayError::Rejected(format!("zip-rs overlap check failed: {error}")))?
    {
        return Err(ReplayError::Rejected(
            "zip-rs reports overlapping member extents".into(),
        ));
    }
    let entry_limit = decimal_at(receipt, &["limits", "max_entries"])?.min(HARD_ENTRY_LIMIT);
    if archive.len() as u64 > entry_limit {
        return Err(ReplayError::Rejected(
            "archive entry count exceeds receipt limit".into(),
        ));
    }

    let index = usize::try_from(decimal_at(
        receipt,
        &["occurrence", "central_directory_index"],
    )?)
    .map_err(|_| ReplayError::Rejected("entry index does not fit platform".into()))?;
    if index >= archive.len() {
        return Err(ReplayError::Rejected(
            "receipt entry index is out of range".into(),
        ));
    }
    let raw_name = BASE64
        .decode(string_at(receipt, &["occurrence", "raw_filename_base64"])?)
        .map_err(|e| ReplayError::Rejected(format!("invalid raw filename base64: {e}")))?;
    if optional_decimal_at(receipt, &["occurrence", "duplicate_raw_name_count"])? != Some(1) {
        return Err(ReplayError::Unsupported(
            "receipt does not prove a unique raw filename".into(),
        ));
    }

    let declared_method = decimal_at(receipt, &["occurrence", "compression_method"])?;
    let declared_compressed = decimal_at(receipt, &["occurrence", "declared_compressed_bytes"])?;
    let declared_size = decimal_at(receipt, &["occurrence", "declared_uncompressed_bytes"])?;
    let declared_crc =
        u32::from_str_radix(string_at(receipt, &["occurrence", "declared_crc32"])?, 16)
            .map_err(|_| ReplayError::Rejected("invalid declared CRC-32".into()))?;
    let member_limit =
        decimal_at(receipt, &["limits", "max_member_output_bytes"])?.min(decimal_at(
            receipt,
            &["limits", "max_entry_declared_uncompressed_bytes"],
        )?);
    if declared_size > member_limit {
        return Err(ReplayError::Rejected(
            "declared size exceeds member output limit".into(),
        ));
    }
    let buffer_len = usize::try_from(integer_at(receipt, &["limits", "read_buffer_bytes"])?)
        .map_err(|_| ReplayError::Rejected("read buffer does not fit platform".into()))?;
    if buffer_len == 0 || buffer_len > HARD_BUFFER_LIMIT {
        return Err(ReplayError::Rejected(
            "read buffer exceeds replay bound".into(),
        ));
    }

    let mut member = archive
        .by_index(index)
        .map_err(|e| ReplayError::Rejected(format!("cannot open indexed entry: {e}")))?;
    if member.name_raw() != raw_name {
        return Err(ReplayError::Rejected(
            "raw filename differs from receipt".into(),
        ));
    }
    if member.header_start()
        != decimal_at(receipt, &["occurrence", "local_header_offset_absolute"])?
    {
        return Err(ReplayError::Rejected(
            "local-header offset differs from receipt".into(),
        ));
    }
    if member.data_start()
        != Some(decimal_at(
            receipt,
            &["occurrence", "data_offset_absolute"],
        )?)
        || member.compressed_size() != declared_compressed
        || member.size() != declared_size
        || member.crc32() != declared_crc
        || u64::from(member.flags().bits())
            != decimal_at(receipt, &["occurrence", "general_purpose_flags"])?
        || member.encrypted()
        || member.is_dir()
        || member.is_symlink()
    {
        return Err(ReplayError::Rejected(
            "zip-rs metadata differs from receipt".into(),
        ));
    }
    let method = match member.compression() {
        CompressionMethod::Stored => 0,
        CompressionMethod::Deflated => 8,
        _ => {
            return Err(ReplayError::Unsupported(
                "oracle method is outside Stored/Deflate".into(),
            ));
        }
    };
    if method as u64 != declared_method {
        return Err(ReplayError::Rejected(
            "compression method differs from receipt".into(),
        ));
    }

    let mut sha = Sha256::new();
    let mut crc = Crc32::new();
    let mut output = vec![0u8; buffer_len];
    let mut observed = 0u64;
    loop {
        let remaining = member_limit.saturating_sub(observed);
        let request = output
            .len()
            .min(usize::try_from(remaining.saturating_add(1)).unwrap_or(usize::MAX));
        let n = member.read(&mut output[..request]).map_err(|e| {
            ReplayError::Rejected(format!("zip-rs stream/CRC verification failed: {e}"))
        })?;
        if n == 0 {
            break;
        }
        if n as u64 > remaining {
            return Err(ReplayError::Rejected(
                "decompressed output exceeded member limit".into(),
            ));
        }
        observed = observed
            .checked_add(n as u64)
            .ok_or_else(|| ReplayError::Rejected("observed byte count overflow".into()))?;
        sha.update(&output[..n]);
        crc.update(&output[..n]);
    }
    if observed != declared_size {
        return Err(ReplayError::Rejected(
            "observed size differs from receipt".into(),
        ));
    }
    let payload_sha = format!("{:x}", sha.finalize());
    let observed_crc = crc.finish();
    if payload_sha != string_at(receipt, &["observations", "payload_sha256"])?
        || format!("{observed_crc:08x}") != string_at(receipt, &["observations", "observed_crc32"])?
    {
        return Err(ReplayError::Rejected(
            "independent payload digest or CRC differs from receipt".into(),
        ));
    }

    let (archive_sha_after, size_after) = hash_open_file(&mut source, archive_limit)?;
    if archive_sha_after != archive_sha || size_after != actual_size {
        return Err(ReplayError::Rejected(
            "archive changed during replay".into(),
        ));
    }
    Ok(ReplayReport {
        archive_sha256: archive_sha,
        payload_sha256: payload_sha,
        observed_bytes: observed,
        observed_crc32: observed_crc,
        oracle: "zip 9.3.0",
        codec: "flate2 / zlib-rs",
    })
}

/// Reject oversized or unsupported EOCD layouts before zip-rs allocates its index.
fn preflight_regular_zip(
    file: &mut File,
    file_length: u64,
    max_central_directory_bytes: u64,
    max_entries: u64,
) -> Result<u64, ReplayError> {
    let tail_length = file_length.min((EOCD_FIXED_BYTES + EOCD_MAX_COMMENT_BYTES) as u64);
    let tail_start = file_length
        .checked_sub(tail_length)
        .ok_or_else(|| ReplayError::Rejected("EOCD window underflow".into()))?;
    file.seek(SeekFrom::Start(tail_start)).map_err(io_error)?;
    let mut tail = vec![
        0u8;
        usize::try_from(tail_length).map_err(|_| ReplayError::Rejected(
            "EOCD window does not fit address space".into()
        ))?
    ];
    file.read_exact(&mut tail).map_err(io_error)?;

    let mut candidates = Vec::new();
    if tail.len() >= EOCD_FIXED_BYTES {
        for offset in (0..=tail.len() - EOCD_FIXED_BYTES).rev() {
            if read_u32(&tail, offset) != EOCD_SIGNATURE {
                continue;
            }
            let comment_len = u16::from_le_bytes([tail[offset + 20], tail[offset + 21]]) as usize;
            if offset + EOCD_FIXED_BYTES + comment_len == tail.len() {
                candidates.push(offset);
            }
        }
    }
    if candidates.len() != 1 {
        return Err(ReplayError::Unsupported(
            "EOCD is absent or ambiguous in the bounded tail".into(),
        ));
    }

    let eocd = candidates[0];
    let disk = read_u16(&tail, eocd + 4);
    let central_disk = read_u16(&tail, eocd + 6);
    let entries_on_disk = read_u16(&tail, eocd + 8);
    let entries_total = read_u16(&tail, eocd + 10);
    let central_bytes = u64::from(read_u32(&tail, eocd + 12));
    let central_offset = u64::from(read_u32(&tail, eocd + 16));

    if disk != 0 || central_disk != 0 || entries_on_disk != entries_total {
        return Err(ReplayError::Unsupported(
            "multi-disk or split ZIP archives are not in the independent replay subset".into(),
        ));
    }
    if entries_total == u16::MAX
        || central_bytes == u64::from(u32::MAX)
        || central_offset == u64::from(u32::MAX)
    {
        return Err(ReplayError::Unsupported(
            "ZIP64 is outside the first independent replay subset".into(),
        ));
    }
    if central_bytes > max_central_directory_bytes
        || central_bytes > HARD_CENTRAL_DIRECTORY_LIMIT
        || u64::from(entries_total) > max_entries
        || u64::from(entries_total) > HARD_ENTRY_LIMIT
    {
        return Err(ReplayError::Rejected(
            "central-directory bytes or entry count exceed replay limits".into(),
        ));
    }

    let eocd_absolute = tail_start + eocd as u64;
    let central_absolute = eocd_absolute
        .checked_sub(central_bytes)
        .ok_or_else(|| ReplayError::Rejected("central-directory range underflows source".into()))?;
    let archive_start = central_absolute
        .checked_sub(central_offset)
        .ok_or_else(|| {
            ReplayError::Rejected("ZIP-relative central-directory offset underflows source".into())
        })?;
    if archive_start != 0 {
        return Err(ReplayError::Unsupported(
            "prefixed/self-extracting carrier layouts are outside this replay subset".into(),
        ));
    }
    if central_absolute.checked_add(central_bytes) != Some(eocd_absolute) {
        return Err(ReplayError::Unsupported(
            "interstitial records between central directory and EOCD are not in replay subset"
                .into(),
        ));
    }
    Ok(u64::from(entries_total))
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn hash_open_file(file: &mut File, limit: u64) -> Result<(String, u64), ReplayError> {
    let initial_length = file.metadata().map_err(io_error)?.len();
    if initial_length > limit {
        return Err(ReplayError::Rejected("archive exceeds byte limit".into()));
    }
    file.seek(std::io::SeekFrom::Start(0)).map_err(io_error)?;
    let mut sha = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let n = file.read(&mut buf).map_err(io_error)?;
        if n == 0 {
            break;
        }
        total = total
            .checked_add(n as u64)
            .ok_or_else(|| ReplayError::Rejected("archive size overflow".into()))?;
        if total > limit {
            return Err(ReplayError::Rejected(
                "archive exceeded byte limit while hashing".into(),
            ));
        }
        sha.update(&buf[..n]);
    }
    let final_length = file.metadata().map_err(io_error)?.len();
    if total != initial_length || total != final_length {
        return Err(ReplayError::Rejected(
            "archive changed while hashing".into(),
        ));
    }
    Ok((format!("{:x}", sha.finalize()), total))
}

fn integer_at(value: &Value, path: &[&str]) -> Result<u64, ReplayError> {
    value_at(value, path)?.as_u64().ok_or_else(|| {
        ReplayError::Rejected(format!("{} must be a non-negative integer", path.join(".")))
    })
}

struct Crc32(u32);
impl Crc32 {
    fn new() -> Self {
        Self(u32::MAX)
    }
    fn update(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= *byte as u32;
            for _ in 0..8 {
                self.0 = if self.0 & 1 == 1 {
                    (self.0 >> 1) ^ 0xedb8_8320
                } else {
                    self.0 >> 1
                };
            }
        }
    }
    fn finish(self) -> u32 {
        !self.0
    }
}

fn decimal_at(value: &Value, path: &[&str]) -> Result<u64, ReplayError> {
    let s = string_at(value, path)?;
    if s.is_empty() || (s.len() > 1 && s.starts_with('0')) || !s.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(ReplayError::Rejected(format!(
            "{} is not a canonical u64 decimal string",
            path.join(".")
        )));
    }
    s.parse()
        .map_err(|_| ReplayError::Rejected(format!("{} exceeds u64", path.join("."))))
}
fn optional_decimal_at(value: &Value, path: &[&str]) -> Result<Option<u64>, ReplayError> {
    if value_at(value, path)?.is_null() {
        Ok(None)
    } else {
        decimal_at(value, path).map(Some)
    }
}
fn string_at<'a>(value: &'a Value, path: &[&str]) -> Result<&'a str, ReplayError> {
    value_at(value, path)?
        .as_str()
        .ok_or_else(|| ReplayError::Rejected(format!("{} must be a string", path.join("."))))
}
fn value_at<'a>(value: &'a Value, path: &[&str]) -> Result<&'a Value, ReplayError> {
    let mut current = value;
    for part in path {
        current = current
            .get(*part)
            .ok_or_else(|| ReplayError::Rejected(format!("missing {}", path.join("."))))?;
    }
    Ok(current)
}
fn io_error(error: std::io::Error) -> ReplayError {
    ReplayError::Io(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::{Cursor, Write};
    use std::sync::atomic::{AtomicU64, Ordering};
    use zip::write::SimpleFileOptions;

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct TempArchive(std::path::PathBuf);
    impl Drop for TempArchive {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn make_archive(
        method: CompressionMethod,
        entries: &[(&str, &[u8])],
    ) -> (TempArchive, Vec<u8>) {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, payload) in entries {
            let options = SimpleFileOptions::default().compression_method(method);
            writer.start_file(*name, options).unwrap();
            writer.write_all(payload).unwrap();
        }
        let bytes = writer.finish().unwrap().into_inner();
        let path = std::env::temp_dir().join(format!(
            "rea-replay-{}-{}.zip",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(&bytes).unwrap();
        drop(file);
        (TempArchive(path), bytes)
    }

    fn make_receipt(bytes: &[u8], index: usize, payload: &[u8]) -> Value {
        let mut oracle = zip::ZipArchive::new(Cursor::new(bytes.to_vec())).unwrap();
        let mut member = oracle.by_index(index).unwrap();
        let raw_name = member.name_raw().to_vec();
        let decoded_name = member.name().to_owned();
        let local_header_offset = member.header_start();
        let data_offset = member.data_start().unwrap();
        let compressed_size = member.compressed_size();
        let uncompressed_size = member.size();
        let flags = member.flags().bits();
        let crc = member.crc32();
        let method = match member.compression() {
            CompressionMethod::Stored => 0,
            CompressionMethod::Deflated => 8,
            _ => panic!("test fixture uses an unsupported compression method"),
        };
        drop(member);
        drop(oracle);

        json!({
            "schema_version": "0.1.0",
            "outcome": "verified",
            "source": {
                "archive_size_bytes": bytes.len().to_string(),
                "binding_strength": "qualified_identity",
                "identity_scheme": "unix_dev_ino_mode_mtime_ctime",
                "platform": {
                    "os": std::env::consts::OS,
                    "architecture": std::env::consts::ARCH,
                    "kernel_or_filesystem": "test fixture"
                },
                "held_descriptor_matches": true,
                "selected_path_matches": true,
                "identity_revalidated_after_attempt": true,
                "identity_commitment_sha256": null,
                "archive_sha256": format!("{:x}", Sha256::digest(bytes))
            },
            "occurrence": {
                "central_directory_index": index.to_string(),
                "raw_filename_base64": BASE64.encode(&raw_name),
                "decoded_filename": decoded_name,
                "duplicate_raw_name_count": "1",
                "local_header_offset_absolute": local_header_offset.to_string(),
                "data_offset_absolute": data_offset.to_string(),
                "entry_kind": "file",
                "encrypted": false,
                "general_purpose_flags": flags,
                "compression_method": method,
                "declared_compressed_bytes": compressed_size.to_string(),
                "declared_uncompressed_bytes": uncompressed_size.to_string(),
                "declared_crc32": format!("{crc:08x}")
            },
            "verifier": {
                "implementation": "production-member-verifier",
                "source_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "toolchain": "rustc 1.96.0",
                "target": "x86_64-unknown-linux-gnu",
                "cargo_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "sha256_implementation": "sha2 0.10.9",
                "independent_crc_implementation": "Symthaea bitwise CRC-32",
                "secondary_decoder": "zipcore 0.10.2 / miniz_oxide",
                "secondary_decoder_crc_implementation": "crc32fast 1.5.4",
                "differential_oracles": []
            },
            "limits": {
                "max_archive_bytes": (HARD_ARCHIVE_LIMIT).to_string(),
                "max_central_directory_bytes": (64 * 1024 * 1024u64).to_string(),
                "max_entries": HARD_ENTRY_LIMIT.to_string(),
                "max_total_declared_uncompressed_bytes": (4 * 1024 * 1024 * 1024u64).to_string(),
                "max_entry_declared_uncompressed_bytes": (2 * 1024 * 1024 * 1024u64).to_string(),
                "max_member_output_bytes": (128 * 1024 * 1024u64).to_string(),
                "read_buffer_bytes": 65536
            },
            "observations": {
                "observed_uncompressed_bytes": payload.len().to_string(),
                "observed_crc32": format!("{crc:08x}"),
                "payload_sha256": format!("{:x}", Sha256::digest(payload)),
                "decoder_reached_eof": true,
                "decoder_crc32_valid": true,
                "decoder_size_valid": true,
                "independent_crc32_valid": true,
                "metadata_parity_valid": true,
                "source_revalidated": true,
                "failure_kind": null,
                "failure_detail": null
            },
            "claim_scope": {
                "asserted_claim": "member_content_integrity_only",
                "not_established": [
                    "producer_authenticity", "producer_identity", "apk_signature_validity",
                    "ipa_code_signature_validity", "whole_archive_completeness",
                    "package_installability", "runtime_behavior", "masvs_control_pass",
                    "semantic_truth", "absence_of_malicious_writer"
                ]
            },
            "attestation": {
                "status": "not_provided",
                "envelope_reference": null
            }
        })
    }

    #[test]
    fn independently_replays_stored_member_receipt() {
        let payload = b"independent Stored replay payload";
        let (temp, bytes) = make_archive(CompressionMethod::Stored, &[("asset.bin", payload)]);
        let receipt = make_receipt(&bytes, 0, payload);
        let report = replay_member_claim(&temp.0, &receipt).unwrap();
        assert_eq!(
            report.payload_sha256,
            format!("{:x}", Sha256::digest(payload))
        );
        assert_eq!(report.observed_bytes, payload.len() as u64);
        assert_eq!(report.observed_crc32, crc32(payload));
        assert_eq!(report.oracle, "zip 9.3.0");
    }

    #[test]
    fn independently_replays_deflate_member_with_zlib_rs() {
        let payload = b"independent Deflate replay payload ".repeat(1024);
        let (temp, bytes) = make_archive(CompressionMethod::Deflated, &[("asset.bin", &payload)]);
        let receipt = make_receipt(&bytes, 0, &payload);
        let report = replay_member_claim(&temp.0, &receipt).unwrap();
        assert_eq!(report.codec, "flate2 / zlib-rs");
        assert_eq!(
            report.payload_sha256,
            format!("{:x}", Sha256::digest(&payload))
        );
    }

    #[test]
    fn wrong_archive_digest_is_rejected_before_member_replay() {
        let payload = b"archive digest negative control";
        let (temp, bytes) = make_archive(CompressionMethod::Stored, &[("asset.bin", payload)]);
        let mut receipt = make_receipt(&bytes, 0, payload);
        receipt["source"]["archive_sha256"] = json!("0".repeat(64));
        let error = replay_member_claim(&temp.0, &receipt).unwrap_err();
        assert!(matches!(error, ReplayError::Rejected(_)));
        assert!(error.to_string().contains("archive SHA-256"));
    }

    #[test]
    fn duplicate_name_archive_abstains_instead_of_collapsing_occurrences() {
        let (temp, bytes) = make_archive(
            CompressionMethod::Stored,
            &[("same.bin", b"first"), ("same.bin", b"second")],
        );
        let receipt = make_receipt(&bytes, 0, b"first");
        let error = replay_member_claim(&temp.0, &receipt).unwrap_err();
        assert!(matches!(error, ReplayError::Unsupported(_)));
        assert!(error.to_string().contains("duplicate raw names"));
    }

    #[test]
    fn metadata_mismatch_is_rejected() {
        let payload = b"metadata negative control";
        let (temp, bytes) = make_archive(CompressionMethod::Stored, &[("asset.bin", payload)]);
        let mut receipt = make_receipt(&bytes, 0, payload);
        receipt["occurrence"]["local_header_offset_absolute"] = json!("999");
        let error = replay_member_claim(&temp.0, &receipt).unwrap_err();
        assert!(matches!(error, ReplayError::Rejected(_)));
    }

    fn crc32(bytes: &[u8]) -> u32 {
        let mut state = Crc32::new();
        state.update(bytes);
        state.finish()
    }
}
