//! Independent member-claim replay using zip-rs 9.3.0 and its zlib-rs Deflate backend.
//! This read-only verifier abstains on duplicate-name archives because zip-rs indexes unique names.

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use zip::CompressionMethod;

const HARD_ARCHIVE_LIMIT: u64 = 512 * 1024 * 1024;
const HARD_ENTRY_LIMIT: u64 = 100_000;
const HARD_BUFFER_LIMIT: usize = 1024 * 1024;

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

pub fn replay_member_claim(path: impl AsRef<Path>, receipt: &Value) -> Result<ReplayReport, ReplayError> {
    if string_at(receipt, &["outcome"])? != "verified" {
        return Err(ReplayError::Rejected("only verified member receipts are replayable".into()));
    }
    let archive_limit = decimal_at(receipt, &["limits", "max_archive_bytes"])?
        .min(HARD_ARCHIVE_LIMIT);
    let archive_size = decimal_at(receipt, &["source", "archive_size_bytes"])?;
    // Keep one opened object for the entire operation. The parser receives a cloned handle
    // to this same object, not a second path open that could resolve to a replacement file.
    let mut source = File::open(path.as_ref()).map_err(io_error)?;
    let (archive_sha, actual_size) = hash_open_file(&mut source, archive_limit)?;
    if actual_size != archive_size {
        return Err(ReplayError::Rejected("archive size differs from receipt".into()));
    }
    if archive_sha != string_at(receipt, &["source", "archive_sha256"])? {
        return Err(ReplayError::Rejected("archive SHA-256 differs from receipt".into()));
    }

    let parser_handle = source.try_clone().map_err(io_error)?;
    let mut archive = zip::ZipArchive::new(parser_handle)
        .map_err(|e| ReplayError::Unsupported(format!("zip-rs cannot parse this layout: {e}")))?;
    if archive.has_duplicate_names() {
        return Err(ReplayError::Unsupported(
            "zip-rs indexed view collapses duplicate raw names; occurrence cannot be proven".into(),
        ));
    }
    let entry_limit = decimal_at(receipt, &["limits", "max_entries"])?.min(HARD_ENTRY_LIMIT);
    if archive.len() as u64 > entry_limit {
        return Err(ReplayError::Rejected("archive entry count exceeds receipt limit".into()));
    }

    let index = usize::try_from(decimal_at(receipt, &["occurrence", "central_directory_index"])?)
        .map_err(|_| ReplayError::Rejected("entry index does not fit platform".into()))?;
    if index >= archive.len() {
        return Err(ReplayError::Rejected("receipt entry index is out of range".into()));
    }
    let raw_name = BASE64.decode(string_at(receipt, &["occurrence", "raw_filename_base64"])?)
        .map_err(|e| ReplayError::Rejected(format!("invalid raw filename base64: {e}")))?;
    if optional_decimal_at(receipt, &["occurrence", "duplicate_raw_name_count"])? != Some(1) {
        return Err(ReplayError::Unsupported("receipt does not prove a unique raw filename".into()));
    }

    let declared_method = decimal_at(receipt, &["occurrence", "compression_method"])?;
    let declared_compressed = decimal_at(receipt, &["occurrence", "declared_compressed_bytes"])?;
    let declared_size = decimal_at(receipt, &["occurrence", "declared_uncompressed_bytes"])?;
    let declared_crc = u32::from_str_radix(string_at(receipt, &["occurrence", "declared_crc32"])?, 16)
        .map_err(|_| ReplayError::Rejected("invalid declared CRC-32".into()))?;
    let member_limit = decimal_at(receipt, &["limits", "max_member_output_bytes"])?
        .min(decimal_at(receipt, &["limits", "max_entry_declared_uncompressed_bytes"])?);
    if declared_size > member_limit {
        return Err(ReplayError::Rejected("declared size exceeds member output limit".into()));
    }
    let buffer_len = usize::try_from(decimal_at(receipt, &["limits", "read_buffer_bytes"])?)
        .map_err(|_| ReplayError::Rejected("read buffer does not fit platform".into()))?;
    if buffer_len == 0 || buffer_len > HARD_BUFFER_LIMIT {
        return Err(ReplayError::Rejected("read buffer exceeds replay bound".into()));
    }

    let mut member = archive.by_index(index)
        .map_err(|e| ReplayError::Rejected(format!("cannot open indexed entry: {e}")))?;
    if member.name_raw() != raw_name {
        return Err(ReplayError::Rejected("raw filename differs from receipt".into()));
    }
    if member.header_start() != decimal_at(receipt, &["occurrence", "local_header_offset_absolute"])? {
        return Err(ReplayError::Rejected("local-header offset differs from receipt".into()));
    }
    if member.data_start() != Some(decimal_at(receipt, &["occurrence", "data_offset_absolute"])?)
        || member.compressed_size() != declared_compressed
        || member.size() != declared_size
        || member.crc32() != declared_crc
        || u64::from(member.flags().bits()) != decimal_at(receipt, &["occurrence", "general_purpose_flags"])?
        || member.encrypted() || member.is_dir() || member.is_symlink()
    {
        return Err(ReplayError::Rejected("zip-rs metadata differs from receipt".into()));
    }
    let method = match member.compression() {
        CompressionMethod::Stored => 0,
        CompressionMethod::Deflated => 8,
        _ => return Err(ReplayError::Unsupported("oracle method is outside Stored/Deflate".into())),
    };
    if method as u64 != declared_method {
        return Err(ReplayError::Rejected("compression method differs from receipt".into()));
    }

    let mut sha = Sha256::new();
    let mut crc = Crc32::new();
    let mut output = vec![0u8; buffer_len];
    let mut observed = 0u64;
    loop {
        let remaining = member_limit.saturating_sub(observed);
        let request = output.len().min(usize::try_from(remaining.saturating_add(1)).unwrap_or(usize::MAX));
        let n = member.read(&mut output[..request])
            .map_err(|e| ReplayError::Rejected(format!("zip-rs stream/CRC verification failed: {e}")))?;
        if n == 0 { break; }
        if n as u64 > remaining {
            return Err(ReplayError::Rejected("decompressed output exceeded member limit".into()));
        }
        observed = observed.checked_add(n as u64)
            .ok_or_else(|| ReplayError::Rejected("observed byte count overflow".into()))?;
        sha.update(&output[..n]);
        crc.update(&output[..n]);
    }
    if observed != declared_size {
        return Err(ReplayError::Rejected("observed size differs from receipt".into()));
    }
    let payload_sha = format!("{:x}", sha.finalize());
    let observed_crc = crc.finish();
    if payload_sha != string_at(receipt, &["observations", "payload_sha256"])?
        || format!("{observed_crc:08x}") != string_at(receipt, &["observations", "observed_crc32"])?
    {
        return Err(ReplayError::Rejected("independent payload digest or CRC differs from receipt".into()));
    }

    let (archive_sha_after, size_after) = hash_open_file(&mut source, archive_limit)?;
    if archive_sha_after != archive_sha || size_after != actual_size {
        return Err(ReplayError::Rejected("archive changed during replay".into()));
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

fn hash_open_file(file: &mut File, limit: u64) -> Result<(String, u64), ReplayError> {
    let length = file.metadata().map_err(io_error)?.len();
    if length > limit {
        return Err(ReplayError::Rejected("archive exceeds byte limit".into()));
    }
    file.seek(std::io::SeekFrom::Start(0)).map_err(io_error)?;
    let mut sha = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let n = file.read(&mut buf).map_err(io_error)?;
        if n == 0 { break; }
        total = total.checked_add(n as u64)
            .ok_or_else(|| ReplayError::Rejected("archive size overflow".into()))?;
        if total > limit {
            return Err(ReplayError::Rejected("archive exceeded byte limit while hashing".into()));
        }
        sha.update(&buf[..n]);
    }
    if total != file.metadata().map_err(io_error)?.len() {
        return Err(ReplayError::Rejected("archive changed while hashing".into()));
    }
    Ok((format!("{:x}", sha.finalize()), total))
}

struct Crc32(u32);
impl Crc32 {
    fn new() -> Self { Self(u32::MAX) }
    fn update(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= *byte as u32;
            for _ in 0..8 {
                self.0 = if self.0 & 1 == 1 { (self.0 >> 1) ^ 0xedb8_8320 } else { self.0 >> 1 };
            }
        }
    }
    fn finish(self) -> u32 { !self.0 }
}

fn decimal_at(value: &Value, path: &[&str]) -> Result<u64, ReplayError> {
    let s = string_at(value, path)?;
    if s.is_empty() || (s.len() > 1 && s.starts_with('0')) || !s.bytes().all(|b| b.is_ascii_digit()) {
        return Err(ReplayError::Rejected(format!("{} is not a canonical u64 decimal string", path.join("."))));
    }
    s.parse().map_err(|_| ReplayError::Rejected(format!("{} exceeds u64", path.join("."))))
}
fn optional_decimal_at(value: &Value, path: &[&str]) -> Result<Option<u64>, ReplayError> {
    if value_at(value, path)?.is_null() { Ok(None) } else { decimal_at(value, path).map(Some) }
}
fn string_at<'a>(value: &'a Value, path: &[&str]) -> Result<&'a str, ReplayError> {
    value_at(value, path)?.as_str()
        .ok_or_else(|| ReplayError::Rejected(format!("{} must be a string", path.join("."))))
}
fn value_at<'a>(value: &'a Value, path: &[&str]) -> Result<&'a Value, ReplayError> {
    let mut current = value;
    for part in path {
        current = current.get(*part)
            .ok_or_else(|| ReplayError::Rejected(format!("missing {}", path.join("."))))?;
    }
    Ok(current)
}
fn io_error(error: std::io::Error) -> ReplayError { ReplayError::Io(error.to_string()) }
