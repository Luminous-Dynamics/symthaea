#![forbid(unsafe_code)]

//! Bounded, read-only ZIP metadata inventory for REA-compatible artifact work.
//!
//! This first slice does not extract, decrypt, decompress, hash member content,
//! interpret APK signing blocks, or prove that the bytes named by a central
//! directory entry are semantically trustworthy. It preserves archive
//! occurrences and keeps declared metadata separate from observed content.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use zipcore::error::Kind as ZipErrorKind;
use zipcore::{Archive as DecoderArchive, ReadOptions as DecoderReadOptions};

const EOCD_SIGNATURE: u32 = 0x0605_4b50;
const ZIP64_LOCATOR_SIGNATURE: u32 = 0x0706_4b50;
const ZIP64_EOCD_SIGNATURE: u32 = 0x0606_4b50;
const CENTRAL_HEADER_SIGNATURE: u32 = 0x0201_4b50;
const LOCAL_HEADER_SIGNATURE: u32 = 0x0403_4b50;
const CENTRAL_DIGITAL_SIGNATURE: u32 = 0x0505_4b50;
const DATA_DESCRIPTOR_SIGNATURE: u32 = 0x0807_4b50;
const EOCD_FIXED_BYTES: usize = 22;
const EOCD_MAX_COMMENT_BYTES: usize = u16::MAX as usize;
const ZIP64_LOCATOR_BYTES: u64 = 20;
const ZIP64_EOCD_FIXED_BYTES: u64 = 56;
const CENTRAL_HEADER_FIXED_BYTES: u64 = 46;
const LOCAL_HEADER_FIXED_BYTES: u64 = 30;
const ZIP64_EXTRA_ID: u16 = 0x0001;
const UNICODE_PATH_EXTRA_ID: u16 = 0x7075;
const UTF8_FLAG: u16 = 1 << 11;
const DATA_DESCRIPTOR_FLAG: u16 = 1 << 3;
const ENCRYPTED_FLAG: u16 = 1;
const UNIX_FILE_TYPE_MASK: u32 = 0o170000;
const UNIX_DIRECTORY: u32 = 0o040000;
const UNIX_SYMLINK: u32 = 0o120000;
const MAX_CONTENT_READ_BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReaderErrorKind {
    Io,
    InvalidArchive,
    Unsupported,
    LimitExceeded,
    SourceChanged,
    AmbiguousEndRecord,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReaderError {
    pub kind: ReaderErrorKind,
    pub message: String,
}

impl ReaderError {
    fn new(kind: ReaderErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    fn io(context: &str, error: io::Error) -> Self {
        Self::new(ReaderErrorKind::Io, format!("{context}: {error}"))
    }
}

impl std::fmt::Display for ReaderError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for ReaderError {}

fn map_zipcore_error(context: &str, error: zipcore::Error) -> ReaderError {
    let kind = match error.kind() {
        ZipErrorKind::LimitExceeded => ReaderErrorKind::LimitExceeded,
        ZipErrorKind::IntegrityFailure | ZipErrorKind::Malformed => {
            ReaderErrorKind::InvalidArchive
        }
        ZipErrorKind::Unsupported | ZipErrorKind::Password => ReaderErrorKind::Unsupported,
        _ => ReaderErrorKind::Io,
    };
    ReaderError::new(kind, format!("{context}: {error}"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReaderLimits {
    pub max_archive_bytes: u64,
    pub max_central_directory_bytes: u64,
    pub max_entries: u64,
    pub max_name_bytes: u64,
    pub max_extra_field_bytes: u64,
    pub max_total_declared_uncompressed_bytes: u64,
    pub max_entry_declared_uncompressed_bytes: u64,
    pub max_zip64_record_bytes: u64,
}

impl Default for ReaderLimits {
    fn default() -> Self {
        Self {
            max_archive_bytes: 512 * 1024 * 1024,
            max_central_directory_bytes: 64 * 1024 * 1024,
            max_entries: 100_000,
            max_name_bytes: 16 * 1024,
            max_extra_field_bytes: 1024 * 1024,
            max_total_declared_uncompressed_bytes: 4 * 1024 * 1024 * 1024,
            max_entry_declared_uncompressed_bytes: 2 * 1024 * 1024 * 1024,
            max_zip64_record_bytes: 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    File,
    Directory,
    Symlink,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathDisposition {
    Relative,
    Empty,
    Absolute,
    ParentTraversal,
    ContainsNul,
    Undecodable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactEntry {
    /// Stable within one inventory snapshot; never derived from the path.
    pub occurrence_index: u64,
    /// Exact bytes stored in the central directory.
    pub raw_name: Vec<u8>,
    /// Decoded name when a supported ZIP filename encoding is available.
    pub decoded_name: Option<String>,
    pub path_disposition: PathDisposition,
    pub kind: EntryKind,
    pub encrypted: bool,
    pub uses_data_descriptor: bool,
    pub general_purpose_flags: u16,
    pub compression_method: u16,
    pub declared_compressed_size: u64,
    pub declared_uncompressed_size: u64,
    pub declared_crc32: u32,
    pub unix_mode: Option<u32>,
    pub local_header_offset: u64,
    pub data_offset: u64,
    pub compressed_data_end: u64,
    /// Verified descriptor end, including the optional signature, when present.
    pub data_descriptor_end: Option<u64>,
    pub data_descriptor_has_signature: Option<bool>,
    pub limitations: Vec<String>,
}

/// Limits for verifying one admitted member occurrence without extracting it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentVerificationLimits {
    /// Hard ceiling on decompressed bytes. Verification fails rather than returning a partial hash.
    pub max_uncompressed_bytes: u64,
    /// Fixed streaming buffer, constrained to 1..=1 MiB.
    pub read_buffer_bytes: usize,
}

impl Default for ContentVerificationLimits {
    fn default() -> Self {
        Self {
            max_uncompressed_bytes: 128 * 1024 * 1024,
            read_buffer_bytes: 64 * 1024,
        }
    }
}

/// A per-occurrence result. declared_compressed_bytes is archive metadata, not a measured
/// count of bytes read from the compressed stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedEntryContent {
    pub occurrence_index: u64,
    pub sha256: String,
    pub declared_compressed_bytes: u64,
    pub declared_uncompressed_bytes: u64,
    pub observed_uncompressed_bytes: u64,
    pub verified_crc32: bool,
    pub compression_method: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderConsistency {
    LocalAndCentralHeaderCheckedFieldsAgree,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalStatus {
    Complete,
    Partial,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraversalCoverage {
    pub status: TraversalStatus,
    pub examined_entries: u64,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveRegions {
    /// Bytes before the ZIP structure. May be an executable stub.
    pub leading_bytes: u64,
    /// Absolute byte offset of the central directory in the selected file.
    pub central_directory_offset: u64,
    pub central_directory_bytes: u64,
    pub trailing_comment_bytes: u64,
    /// Bytes between the final listed member extent and central directory.
    /// May include data descriptors, APK Signing Block data, or other records.
    pub opaque_pre_central_directory_bytes: u64,
    pub zip64: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZipInventory {
    pub source_bytes: u64,
    pub entries: Vec<ArtifactEntry>,
    pub duplicate_raw_names_present: bool,
    pub header_consistency: HeaderConsistency,
    pub coverage: TraversalCoverage,
    pub regions: ArchiveRegions,
}

#[derive(Debug)]
pub struct ZipArtifactReader {
    path: PathBuf,
    file: File,
    initial_identity: SourceIdentity,
    inventory: ZipInventory,
}

impl ZipArtifactReader {
    /// Admit and inventory a regular ZIP/APK/IPA carrier without writing files.
    pub fn open(path: impl AsRef<Path>, limits: ReaderLimits) -> Result<Self, ReaderError> {
        let path = path.as_ref().to_path_buf();
        let selected = fs::symlink_metadata(&path)
            .map_err(|error| ReaderError::io("cannot inspect selected archive path", error))?;
        if selected.file_type().is_symlink() || !selected.is_file() {
            return Err(ReaderError::new(
                ReaderErrorKind::Unsupported,
                "selected archive path must name a regular file, not a symlink or special file",
            ));
        }

        let mut file = OpenOptions::new()
            .read(true)
            .open(&path)
            .map_err(|error| ReaderError::io("cannot open selected archive", error))?;
        let opened = file
            .metadata()
            .map_err(|error| ReaderError::io("cannot stat opened archive", error))?;
        if !opened.is_file() || !same_source_metadata(&selected, &opened) {
            return Err(ReaderError::new(
                ReaderErrorKind::SourceChanged,
                "selected archive changed while it was being admitted",
            ));
        }

        let source_bytes = opened.len();
        if source_bytes > limits.max_archive_bytes {
            return Err(ReaderError::new(
                ReaderErrorKind::LimitExceeded,
                format!(
                    "archive has {source_bytes} bytes; limit is {}",
                    limits.max_archive_bytes
                ),
            ));
        }

        let initial_identity = SourceIdentity::from_metadata(&opened);
        let preflight = preflight_end_record(&mut file, source_bytes, &limits)?;
        let (entries, duplicate_raw_names_present, total_declared_uncompressed) =
            read_central_directory(&mut file, &preflight, &limits)?;

        if total_declared_uncompressed > limits.max_total_declared_uncompressed_bytes {
            return Err(ReaderError::new(
                ReaderErrorKind::LimitExceeded,
                format!(
                    "entries declare {total_declared_uncompressed} uncompressed bytes; aggregate limit is {}",
                    limits.max_total_declared_uncompressed_bytes
                ),
            ));
        }

        let max_member_end = entries
            .iter()
            .map(|entry| {
                entry
                    .data_descriptor_end
                    .unwrap_or(entry.compressed_data_end)
            })
            .max()
            .unwrap_or(preflight.archive_start);
        let opaque_pre_central_directory_bytes = preflight
            .central_directory_offset
            .saturating_sub(max_member_end);

        let mut coverage_limitations = vec![
            "unlisted local-file-header scan is not implemented".to_owned(),
            "member payloads were not decompressed, hashed, or CRC-verified".to_owned(),
            "data descriptor metadata is compared with the central directory; payload CRC is not verified".to_owned(),
            "APK Signing Block and other non-entry regions are retained as opaque byte ranges".to_owned(),
        ];
        if entries.iter().any(|entry| entry.encrypted) {
            coverage_limitations
                .push("encrypted entries are metadata-only and were not decrypted".to_owned());
        }
        if entries.iter().any(|entry| entry.decoded_name.is_none()) {
            coverage_limitations.push("one or more member names could not be decoded".to_owned());
        }

        let inventory = ZipInventory {
            source_bytes,
            coverage: TraversalCoverage {
                status: TraversalStatus::Partial,
                examined_entries: entries.len() as u64,
                limitations: coverage_limitations,
            },
            duplicate_raw_names_present,
            header_consistency: HeaderConsistency::LocalAndCentralHeaderCheckedFieldsAgree,
            regions: ArchiveRegions {
                leading_bytes: preflight.archive_start,
                central_directory_offset: preflight.central_directory_offset,
                central_directory_bytes: preflight.central_directory_bytes,
                trailing_comment_bytes: preflight.trailing_comment_bytes,
                opaque_pre_central_directory_bytes,
                zip64: preflight.zip64,
            },
            entries,
        };

        let reader = Self {
            path,
            file,
            initial_identity,
            inventory,
        };
        reader.verify_source()?;
        Ok(reader)
    }

    pub fn inventory(&self) -> &ZipInventory {
        &self.inventory
    }

    /// Stream and hash one central-directory occurrence without extracting or writing it.
    ///
    /// The custom inventory remains the policy authority. A separately implemented decoder is
    /// used only for bounded streaming and CRC verification; its metadata must match our
    /// occurrence by index and exact fields before any bytes are consumed.
    pub fn verify_entry_content(
        &self,
        occurrence_index: u64,
        limits: ContentVerificationLimits,
    ) -> Result<VerifiedEntryContent, ReaderError> {
        self.verify_source()?;

        if limits.read_buffer_bytes == 0
            || limits.read_buffer_bytes > MAX_CONTENT_READ_BUFFER_BYTES
        {
            return Err(ReaderError::new(
                ReaderErrorKind::LimitExceeded,
                format!(
                    "content read buffer must be between 1 and {MAX_CONTENT_READ_BUFFER_BYTES} bytes"
                ),
            ));
        }

        let index = usize::try_from(occurrence_index).map_err(|_| {
            ReaderError::new(
                ReaderErrorKind::LimitExceeded,
                "member occurrence index does not fit this platform",
            )
        })?;
        let expected = self.inventory.entries.get(index).ok_or_else(|| {
            ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                format!("member occurrence {occurrence_index} is not present in this inventory"),
            )
        })?;
        if expected.occurrence_index != occurrence_index {
            return Err(ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "inventory occurrence index does not match its central-directory position",
            ));
        }
        if expected.encrypted {
            return Err(ReaderError::new(
                ReaderErrorKind::Unsupported,
                "encrypted members are not eligible for content verification",
            ));
        }
        if expected.kind != EntryKind::File {
            return Err(ReaderError::new(
                ReaderErrorKind::Unsupported,
                "directory and symbolic-link members are not eligible for content verification",
            ));
        }
        if !matches!(expected.compression_method, 0 | 8) {
            return Err(ReaderError::new(
                ReaderErrorKind::Unsupported,
                format!(
                    "compression method {} is not enabled for content verification",
                    expected.compression_method
                ),
            ));
        }
        if expected.declared_uncompressed_size > limits.max_uncompressed_bytes {
            return Err(ReaderError::new(
                ReaderErrorKind::LimitExceeded,
                format!(
                    "member declares {} uncompressed bytes; verification limit is {}",
                    expected.declared_uncompressed_size, limits.max_uncompressed_bytes
                ),
            ));
        }

        // Each call opens two independent descriptors: one for the decoder and one retained
        // solely as an identity witness. Do not use File::try_clone here; it can share a seek
        // cursor on Unix, and parallel verifications must never share file positions.
        let identity_witness = OpenOptions::new()
            .read(true)
            .open(&self.path)
            .map_err(|error| ReaderError::io("cannot open archive identity witness", error))?;
        let witness_identity = identity_witness
            .metadata()
            .map_err(|error| ReaderError::io("cannot stat archive identity witness", error))?;
        if !self.initial_identity.matches(&witness_identity) {
            return Err(ReaderError::new(
                ReaderErrorKind::SourceChanged,
                "identity witness does not identify the admitted archive source",
            ));
        }
        let decoder_file = OpenOptions::new()
            .read(true)
            .open(&self.path)
            .map_err(|error| ReaderError::io("cannot open archive for independent content verification", error))?;
        let decoder_identity = decoder_file
            .metadata()
            .map_err(|error| ReaderError::io("cannot stat content-verification descriptor", error))?;
        if !self.initial_identity.matches(&decoder_identity) {
            return Err(ReaderError::new(
                ReaderErrorKind::SourceChanged,
                "content-verification descriptor does not identify the admitted archive source",
            ));
        }
        self.verify_source()?;

        let decoder_limit = limits
            .max_uncompressed_bytes
            .min(expected.declared_uncompressed_size);
        let options = DecoderReadOptions {
            verify_crc: true,
            verify_size: true,
            max_entry_size: Some(decoder_limit),
            offset_correction: true,
            password: None,
        };

        // Keep all decoder setup and streaming in one result scope so the descriptors and
        // selected path are revalidated even when the secondary parser rejects metadata.
        let stream_result = (move || {
            let mut archive = DecoderArchive::new_with_options(decoder_file, options)
                .map_err(|error| map_zipcore_error("secondary ZIP parser rejected archive", error))?;

            if archive.len() != self.inventory.entries.len()
                || archive.central_directory_offset()
                    != self.inventory.regions.central_directory_offset
                || archive.is_zip64() != self.inventory.regions.zip64
            {
                return Err(ReaderError::new(
                    ReaderErrorKind::InvalidArchive,
                    "secondary ZIP parser archive-level metadata differs from the admitted inventory",
                ));
            }

            let decoder_entry = archive.entries().get(index).ok_or_else(|| {
                ReaderError::new(
                    ReaderErrorKind::InvalidArchive,
                    "secondary ZIP parser omitted the selected member occurrence",
                )
            })?;
            let corrected_header_offset = decoder_entry
                .header_offset()
                .checked_add(self.inventory.regions.leading_bytes);
            let decoder_kind_matches = match expected.kind {
                EntryKind::File => !decoder_entry.is_dir() && !decoder_entry.is_symlink(),
                EntryKind::Directory => decoder_entry.is_dir() && !decoder_entry.is_symlink(),
                EntryKind::Symlink => decoder_entry.is_symlink(),
            };
            if decoder_entry.name_raw() != expected.raw_name
                || corrected_header_offset != Some(expected.local_header_offset)
                || decoder_entry.flags() != expected.general_purpose_flags
                || decoder_entry.method() != expected.compression_method
                || decoder_entry.compressed_size() != expected.declared_compressed_size
                || decoder_entry.uncompressed_size() != expected.declared_uncompressed_size
                || decoder_entry.crc32() != expected.declared_crc32
                || decoder_entry.has_data_descriptor() != expected.uses_data_descriptor
                || decoder_entry.is_encrypted() != expected.encrypted
                || !decoder_kind_matches
            {
                return Err(ReaderError::new(
                    ReaderErrorKind::InvalidArchive,
                    "secondary ZIP parser metadata differs from the selected custom inventory occurrence",
                ));
            }

            let mut member = archive
                .by_index(index)
                .map_err(|error| map_zipcore_error("cannot open selected ZIP member stream", error))?;
            let mut buffer = vec![0u8; limits.read_buffer_bytes];
            let mut hasher = Sha256::new();
            let mut observed_uncompressed_bytes = 0u64;

            loop {
                let read = match member.read(&mut buffer) {
                    Ok(read) => read,
                    Err(io_error) => {
                        return Err(match member.take_terminal_error() {
                            Some(error) => map_zipcore_error("ZIP member stream verification failed", error),
                            None => ReaderError::io("cannot read ZIP member stream", io_error),
                        });
                    }
                };
                if read == 0 {
                    break;
                }
                observed_uncompressed_bytes = observed_uncompressed_bytes
                    .checked_add(read as u64)
                    .ok_or_else(|| {
                        ReaderError::new(
                            ReaderErrorKind::LimitExceeded,
                            "observed uncompressed byte count overflow",
                        )
                    })?;
                if observed_uncompressed_bytes > expected.declared_uncompressed_size
                    || observed_uncompressed_bytes > limits.max_uncompressed_bytes
                {
                    return Err(ReaderError::new(
                        ReaderErrorKind::LimitExceeded,
                        "member stream exceeded its declared size or caller byte budget",
                    ));
                }
                hasher.update(&buffer[..read]);
            }

            member.verify_to_end().map_err(|error| {
                map_zipcore_error("ZIP member did not reach a verified end state", error)
            })?;
            Ok((
                format!("{:x}", hasher.finalize()),
                observed_uncompressed_bytes,
            ))
        })();

        let final_witness_identity = identity_witness
            .metadata()
            .map_err(|error| ReaderError::io("cannot revalidate archive identity witness", error))?;
        if !self.initial_identity.matches(&final_witness_identity) {
            return Err(ReaderError::new(
                ReaderErrorKind::SourceChanged,
                "archive source identity changed during the member stream",
            ));
        }
        self.verify_source()?;
        let (sha256, observed_uncompressed_bytes) = stream_result?;

        Ok(VerifiedEntryContent {
            occurrence_index,
            sha256,
            declared_compressed_bytes: expected.declared_compressed_size,
            declared_uncompressed_bytes: expected.declared_uncompressed_size,
            observed_uncompressed_bytes,
            verified_crc32: true,
            compression_method: expected.compression_method,
        })
    }

    /// Revalidate both the held descriptor and the currently selected path.
    pub fn verify_source(&self) -> Result<(), ReaderError> {
        let opened = self
            .file
            .metadata()
            .map_err(|error| ReaderError::io("cannot stat held archive descriptor", error))?;
        let selected = fs::symlink_metadata(&self.path).map_err(|error| {
            ReaderError::io("selected archive path is no longer available", error)
        })?;

        if selected.file_type().is_symlink()
            || !selected.is_file()
            || !self.initial_identity.matches(&opened)
            || !self.initial_identity.matches(&selected)
        {
            return Err(ReaderError::new(
                ReaderErrorKind::SourceChanged,
                "archive source identity changed after inventory admission",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct EndRecord {
    archive_start: u64,
    central_directory_offset: u64,
    central_directory_bytes: u64,
    entry_count: u64,
    trailing_comment_bytes: u64,
    zip64: bool,
}

#[derive(Debug, Clone, Copy)]
struct EntryRange {
    start: u64,
    end: u64,
}

#[derive(Debug)]
struct SourceIdentity {
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    size: u64,
    #[cfg(unix)]
    mode: u32,
    #[cfg(unix)]
    mtime_seconds: i64,
    #[cfg(unix)]
    mtime_nanoseconds: i64,
    #[cfg(unix)]
    ctime_seconds: i64,
    #[cfg(unix)]
    ctime_nanoseconds: i64,
    #[cfg(not(unix))]
    modified: Option<std::time::SystemTime>,
    #[cfg(not(unix))]
    readonly: bool,
}

impl SourceIdentity {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Self {
                device: metadata.dev(),
                inode: metadata.ino(),
                size: metadata.size(),
                mode: metadata.mode(),
                mtime_seconds: metadata.mtime(),
                mtime_nanoseconds: metadata.mtime_nsec(),
                ctime_seconds: metadata.ctime(),
                ctime_nanoseconds: metadata.ctime_nsec(),
            }
        }
        #[cfg(not(unix))]
        {
            Self {
                size: metadata.len(),
                modified: metadata.modified().ok(),
                readonly: metadata.permissions().readonly(),
            }
        }
    }

    fn matches(&self, metadata: &fs::Metadata) -> bool {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            self.device == metadata.dev()
                && self.inode == metadata.ino()
                && self.size == metadata.size()
                && self.mode == metadata.mode()
                && self.mtime_seconds == metadata.mtime()
                && self.mtime_nanoseconds == metadata.mtime_nsec()
                && self.ctime_seconds == metadata.ctime()
                && self.ctime_nanoseconds == metadata.ctime_nsec()
        }
        #[cfg(not(unix))]
        {
            self.size == metadata.len()
                && self.modified == metadata.modified().ok()
                && self.readonly == metadata.permissions().readonly()
        }
    }
}

fn same_source_metadata(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    SourceIdentity::from_metadata(left).matches(right)
}

fn preflight_end_record(
    file: &mut File,
    file_len: u64,
    limits: &ReaderLimits,
) -> Result<EndRecord, ReaderError> {
    let tail_len = file_len.min((EOCD_FIXED_BYTES + EOCD_MAX_COMMENT_BYTES) as u64) as usize;
    let tail_start = file_len - tail_len as u64;
    file.seek(SeekFrom::Start(tail_start))
        .map_err(|error| ReaderError::io("cannot seek to ZIP end record window", error))?;
    let mut tail = vec![0u8; tail_len];
    file.read_exact(&mut tail)
        .map_err(|error| ReaderError::io("cannot read ZIP end record window", error))?;

    let mut candidates = Vec::new();
    let mut candidate_errors = Vec::new();
    if tail_len >= EOCD_FIXED_BYTES {
        for index in (0..=(tail_len - EOCD_FIXED_BYTES)).rev() {
            if read_u32(&tail[index..index + 4]) != EOCD_SIGNATURE {
                continue;
            }
            let comment_len = read_u16(&tail[index + 20..index + 22]) as usize;
            if index + EOCD_FIXED_BYTES + comment_len != tail_len {
                continue;
            }
            let absolute = tail_start + index as u64;
            match parse_end_record_candidate(
                file,
                &tail[index..index + EOCD_FIXED_BYTES],
                absolute,
                file_len,
                limits,
            ) {
                Ok(candidate) => candidates.push(candidate),
                Err(error) => candidate_errors.push(error),
            }
        }
    }

    match candidates.len() {
        0 => Err(candidate_errors
            .into_iter()
            .find(|error| error.kind != ReaderErrorKind::InvalidArchive)
            .unwrap_or_else(|| {
                ReaderError::new(
                    ReaderErrorKind::InvalidArchive,
                    "no unambiguous ZIP end-of-central-directory record was found",
                )
            })),
        1 => Ok(candidates[0]),
        _ => Err(ReaderError::new(
            ReaderErrorKind::AmbiguousEndRecord,
            "multiple plausible ZIP end-of-central-directory records were found",
        )),
    }
}

fn parse_end_record_candidate(
    file: &mut File,
    eocd: &[u8],
    eocd_offset: u64,
    file_len: u64,
    limits: &ReaderLimits,
) -> Result<EndRecord, ReaderError> {
    let disk_number = read_u16(&eocd[4..6]);
    let central_directory_disk = read_u16(&eocd[6..8]);
    let entries_on_disk = read_u16(&eocd[8..10]);
    let total_entries = read_u16(&eocd[10..12]);
    let central_size = read_u32(&eocd[12..16]);
    let central_offset = read_u32(&eocd[16..20]);
    let comment_len = read_u16(&eocd[20..22]) as u64;

    if disk_number != 0 || central_directory_disk != 0 {
        return Err(ReaderError::new(
            ReaderErrorKind::Unsupported,
            "multi-disk ZIP archives are not supported",
        ));
    }

    let uses_zip64 = entries_on_disk == u16::MAX
        || total_entries == u16::MAX
        || central_size == u32::MAX
        || central_offset == u32::MAX;

    let (count, directory_bytes, relative_directory_offset, directory_end, zip64) = if uses_zip64 {
        parse_zip64_end_record(file, eocd_offset, file_len, limits)?
    } else {
        if entries_on_disk != total_entries {
            return Err(ReaderError::new(
                ReaderErrorKind::Unsupported,
                "split ZIP archives are not supported",
            ));
        }
        let directory_bytes = central_size as u64;
        let directory_end = eocd_offset;
        if directory_bytes > directory_end {
            return Err(ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "central directory extends before the beginning of the file",
            ));
        }
        (
            total_entries as u64,
            directory_bytes,
            central_offset as u64,
            directory_end,
            false,
        )
    };

    if count > limits.max_entries {
        return Err(ReaderError::new(
            ReaderErrorKind::LimitExceeded,
            format!(
                "archive declares {count} entries; entry-count limit is {}",
                limits.max_entries
            ),
        ));
    }
    if directory_bytes > limits.max_central_directory_bytes {
        return Err(ReaderError::new(
            ReaderErrorKind::LimitExceeded,
            format!(
                "central directory declares {directory_bytes} bytes; limit is {}",
                limits.max_central_directory_bytes
            ),
        ));
    }
    if count.saturating_mul(CENTRAL_HEADER_FIXED_BYTES) > directory_bytes {
        return Err(ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "entry count cannot fit in the declared central-directory byte range",
        ));
    }

    let directory_offset = if zip64 {
        relative_directory_offset
    } else {
        central_offset as u64
    };
    let directory_start = directory_end.checked_sub(directory_bytes).ok_or_else(|| {
        ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "central-directory range underflows",
        )
    })?;
    let archive_start = directory_start
        .checked_sub(directory_offset)
        .ok_or_else(|| {
            ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "archive prefix and central-directory offset are inconsistent",
            )
        })?;
    if archive_start > file_len || directory_start > file_len || directory_end > file_len {
        return Err(ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "ZIP directory range is outside the selected file",
        ));
    }

    Ok(EndRecord {
        archive_start,
        central_directory_offset: directory_start,
        central_directory_bytes: directory_bytes,
        entry_count: count,
        trailing_comment_bytes: comment_len,
        zip64,
    })
}

fn parse_zip64_end_record(
    file: &mut File,
    eocd_offset: u64,
    file_len: u64,
    limits: &ReaderLimits,
) -> Result<(u64, u64, u64, u64, bool), ReaderError> {
    if eocd_offset < ZIP64_LOCATOR_BYTES {
        return Err(ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "ZIP64 end record is missing its locator",
        ));
    }
    let locator_offset = eocd_offset - ZIP64_LOCATOR_BYTES;
    file.seek(SeekFrom::Start(locator_offset))
        .map_err(|error| ReaderError::io("cannot seek to ZIP64 locator", error))?;
    let mut locator = [0u8; 20];
    file.read_exact(&mut locator)
        .map_err(|error| ReaderError::io("cannot read ZIP64 locator", error))?;
    if read_u32(&locator[0..4]) != ZIP64_LOCATOR_SIGNATURE {
        return Err(ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "ZIP64 end record is missing its locator signature",
        ));
    }
    let locator_disk = read_u32(&locator[4..8]);
    let locator_relative_offset = read_u64(&locator[8..16]);
    let disk_count = read_u32(&locator[16..20]);
    if locator_disk != 0 || disk_count != 1 {
        return Err(ReaderError::new(
            ReaderErrorKind::Unsupported,
            "multi-disk ZIP64 archives are not supported",
        ));
    }

    let search_start = locator_offset.saturating_sub(limits.max_zip64_record_bytes);
    let search_len = locator_offset - search_start;
    let search_len_usize = usize::try_from(search_len).map_err(|_| {
        ReaderError::new(
            ReaderErrorKind::LimitExceeded,
            "ZIP64 record window is too large",
        )
    })?;
    file.seek(SeekFrom::Start(search_start))
        .map_err(|error| ReaderError::io("cannot seek to ZIP64 record window", error))?;
    let mut search = vec![0u8; search_len_usize];
    file.read_exact(&mut search)
        .map_err(|error| ReaderError::io("cannot read ZIP64 record window", error))?;

    let mut records = Vec::new();
    if search.len() >= ZIP64_EOCD_FIXED_BYTES as usize {
        for index in (0..=(search.len() - ZIP64_EOCD_FIXED_BYTES as usize)).rev() {
            if read_u32(&search[index..index + 4]) != ZIP64_EOCD_SIGNATURE {
                continue;
            }
            let record_payload_bytes = read_u64(&search[index + 4..index + 12]);
            let record_total = 12u64.checked_add(record_payload_bytes).ok_or_else(|| {
                ReaderError::new(
                    ReaderErrorKind::InvalidArchive,
                    "ZIP64 record size overflow",
                )
            })?;
            let absolute_start = search_start + index as u64;
            if absolute_start.checked_add(record_total) != Some(locator_offset) {
                continue;
            }
            if record_total > limits.max_zip64_record_bytes {
                return Err(ReaderError::new(
                    ReaderErrorKind::LimitExceeded,
                    "ZIP64 extensible data sector exceeds its configured record limit",
                ));
            }
            records.push((absolute_start, record_payload_bytes));
        }
    }
    if records.len() != 1 {
        return Err(ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "ZIP64 end-of-central-directory record is missing or ambiguous",
        ));
    }

    let (record_start, record_payload_bytes) = records[0];
    if record_payload_bytes < 44 {
        return Err(ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "ZIP64 end-of-central-directory record is shorter than its fixed fields",
        ));
    }
    file.seek(SeekFrom::Start(record_start))
        .map_err(|error| ReaderError::io("cannot seek to ZIP64 end record", error))?;
    let mut record = [0u8; ZIP64_EOCD_FIXED_BYTES as usize];
    file.read_exact(&mut record)
        .map_err(|error| ReaderError::io("cannot read ZIP64 end record", error))?;

    let disk_number = read_u32(&record[16..20]);
    let central_directory_disk = read_u32(&record[20..24]);
    let entries_on_disk = read_u64(&record[24..32]);
    let total_entries = read_u64(&record[32..40]);
    let directory_bytes = read_u64(&record[40..48]);
    let directory_offset = read_u64(&record[48..56]);
    if disk_number != 0 || central_directory_disk != 0 || entries_on_disk != total_entries {
        return Err(ReaderError::new(
            ReaderErrorKind::Unsupported,
            "multi-disk ZIP64 archives are not supported",
        ));
    }

    let directory_start = record_start.checked_sub(directory_bytes).ok_or_else(|| {
        ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "ZIP64 central-directory size exceeds its physical position",
        )
    })?;
    let archive_start = directory_start
        .checked_sub(directory_offset)
        .ok_or_else(|| {
            ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "ZIP64 central-directory offset is inconsistent with the archive prefix",
            )
        })?;
    if archive_start.checked_add(locator_relative_offset) != Some(record_start) {
        return Err(ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "ZIP64 locator offset does not identify the ZIP64 end record",
        ));
    }
    if record_start > file_len {
        return Err(ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "ZIP64 end record lies outside the selected file",
        ));
    }
    Ok((
        total_entries,
        directory_bytes,
        directory_offset,
        record_start,
        true,
    ))
}

fn read_central_directory(
    file: &mut File,
    end_record: &EndRecord,
    limits: &ReaderLimits,
) -> Result<(Vec<ArtifactEntry>, bool, u64), ReaderError> {
    let directory_end = end_record
        .central_directory_offset
        .checked_add(end_record.central_directory_bytes)
        .ok_or_else(|| {
            ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "central-directory end overflow",
            )
        })?;
    file.seek(SeekFrom::Start(end_record.central_directory_offset))
        .map_err(|error| ReaderError::io("cannot seek to central directory", error))?;

    let mut entries = Vec::with_capacity(end_record.entry_count as usize);
    let mut ranges = Vec::with_capacity(end_record.entry_count as usize);
    let mut total_declared_uncompressed = 0u64;

    for index in 0..end_record.entry_count {
        let record_start = file
            .stream_position()
            .map_err(|error| ReaderError::io("cannot inspect central-directory position", error))?;
        if record_start.saturating_add(CENTRAL_HEADER_FIXED_BYTES) > directory_end {
            return Err(ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "central-directory record exceeds the declared byte range",
            ));
        }

        let mut fixed = [0u8; CENTRAL_HEADER_FIXED_BYTES as usize];
        file.read_exact(&mut fixed)
            .map_err(|error| ReaderError::io("cannot read central-directory header", error))?;
        if read_u32(&fixed[0..4]) != CENTRAL_HEADER_SIGNATURE {
            return Err(ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "central-directory entry has an invalid signature",
            ));
        }

        let version_made_by = read_u16(&fixed[4..6]);
        let flags = read_u16(&fixed[8..10]);
        let compression_method = read_u16(&fixed[10..12]);
        let declared_crc32 = read_u32(&fixed[16..20]);
        let compressed32 = read_u32(&fixed[20..24]);
        let uncompressed32 = read_u32(&fixed[24..28]);
        let name_len = read_u16(&fixed[28..30]) as u64;
        let extra_len = read_u16(&fixed[30..32]) as u64;
        let comment_len = read_u16(&fixed[32..34]) as u64;
        let disk_start16 = read_u16(&fixed[34..36]);
        let external_attributes = read_u32(&fixed[38..42]);
        let local_offset32 = read_u32(&fixed[42..46]);

        enforce_limit("member name", name_len, limits.max_name_bytes)?;
        enforce_limit(
            "member extra field",
            extra_len,
            limits.max_extra_field_bytes,
        )?;
        let variable_bytes = name_len
            .checked_add(extra_len)
            .and_then(|bytes| bytes.checked_add(comment_len))
            .ok_or_else(|| {
                ReaderError::new(
                    ReaderErrorKind::InvalidArchive,
                    "central record length overflow",
                )
            })?;
        let next_record = file
            .stream_position()
            .map_err(|error| ReaderError::io("cannot inspect central record position", error))?
            .checked_add(variable_bytes)
            .ok_or_else(|| {
                ReaderError::new(
                    ReaderErrorKind::InvalidArchive,
                    "central record end overflow",
                )
            })?;
        if next_record > directory_end {
            return Err(ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "central-directory variable fields exceed the declared byte range",
            ));
        }

        let name_len_usize = usize::try_from(name_len).map_err(|_| {
            ReaderError::new(ReaderErrorKind::LimitExceeded, "member name is too large")
        })?;
        let extra_len_usize = usize::try_from(extra_len).map_err(|_| {
            ReaderError::new(
                ReaderErrorKind::LimitExceeded,
                "member extra field is too large",
            )
        })?;
        let mut raw_name = vec![0u8; name_len_usize];
        let mut extra = vec![0u8; extra_len_usize];
        file.read_exact(&mut raw_name)
            .map_err(|error| ReaderError::io("cannot read member name", error))?;
        file.read_exact(&mut extra)
            .map_err(|error| ReaderError::io("cannot read member extra field", error))?;
        file.seek(SeekFrom::Current(comment_len as i64))
            .map_err(|error| ReaderError::io("cannot skip member comment", error))?;

        let needs_uncompressed = uncompressed32 == u32::MAX;
        let needs_compressed = compressed32 == u32::MAX;
        let needs_local_offset = local_offset32 == u32::MAX;
        let needs_disk_start = disk_start16 == u16::MAX;
        let zip64 = parse_zip64_extra(
            &extra,
            needs_uncompressed,
            needs_compressed,
            needs_local_offset,
            needs_disk_start,
        )?;

        let declared_uncompressed_size = if needs_uncompressed {
            zip64
                .uncompressed
                .ok_or_else(|| missing_zip64_value("uncompressed size"))?
        } else {
            uncompressed32 as u64
        };
        let declared_compressed_size = if needs_compressed {
            zip64
                .compressed
                .ok_or_else(|| missing_zip64_value("compressed size"))?
        } else {
            compressed32 as u64
        };
        let local_offset = if needs_local_offset {
            zip64
                .local_offset
                .ok_or_else(|| missing_zip64_value("local header offset"))?
        } else {
            local_offset32 as u64
        };
        let disk_start = if needs_disk_start {
            zip64
                .disk_start
                .ok_or_else(|| missing_zip64_value("disk start"))?
        } else {
            disk_start16 as u32
        };
        if disk_start != 0 {
            return Err(ReaderError::new(
                ReaderErrorKind::Unsupported,
                "multi-disk member entries are not supported",
            ));
        }

        enforce_limit(
            "member declared uncompressed size",
            declared_uncompressed_size,
            limits.max_entry_declared_uncompressed_bytes,
        )?;
        total_declared_uncompressed = total_declared_uncompressed
            .checked_add(declared_uncompressed_size)
            .ok_or_else(|| {
                ReaderError::new(
                    ReaderErrorKind::LimitExceeded,
                    "aggregate declared size overflow",
                )
            })?;
        if total_declared_uncompressed > limits.max_total_declared_uncompressed_bytes {
            return Err(ReaderError::new(
                ReaderErrorKind::LimitExceeded,
                format!(
                    "entries declare {total_declared_uncompressed} uncompressed bytes; aggregate limit is {}",
                    limits.max_total_declared_uncompressed_bytes
                ),
            ));
        }

        let decoded_name = decode_member_name(&raw_name, flags, &extra);
        let path_disposition = decoded_name
            .as_deref()
            .map(classify_path)
            .unwrap_or(PathDisposition::Undecodable);
        let unix_mode = if matches!(version_made_by >> 8, 3 | 19) {
            Some(external_attributes >> 16)
        } else {
            None
        };
        let file_type = unix_mode.unwrap_or(0) & UNIX_FILE_TYPE_MASK;
        let is_symlink = file_type == UNIX_SYMLINK;
        let is_directory = file_type == UNIX_DIRECTORY
            || raw_name.ends_with(b"/")
            || external_attributes & 0x10 != 0;
        let kind = if is_symlink {
            EntryKind::Symlink
        } else if is_directory {
            EntryKind::Directory
        } else {
            EntryKind::File
        };
        let encrypted = flags & ENCRYPTED_FLAG != 0;
        let uses_data_descriptor = flags & DATA_DESCRIPTOR_FLAG != 0;

        let local = verify_local_header(
            file,
            LocalHeaderExpectation {
                archive_start: end_record.archive_start,
                central_directory_offset: end_record.central_directory_offset,
                relative_local_offset: local_offset,
                expected_name: &raw_name,
                expected_flags: flags,
                expected_compression: compression_method,
                expected_crc32: declared_crc32,
                expected_compressed_size: declared_compressed_size,
                expected_uncompressed_size: declared_uncompressed_size,
                uses_data_descriptor,
            },
            limits,
        )?;
        let descriptor = if uses_data_descriptor {
            Some(verify_data_descriptor(
                file,
                local.compressed_data_end,
                end_record.central_directory_offset,
                declared_crc32,
                declared_compressed_size,
                declared_uncompressed_size,
                needs_compressed || needs_uncompressed,
            )?)
        } else {
            None
        };
        let member_extent_end = descriptor
            .map(|facts| facts.end)
            .unwrap_or(local.compressed_data_end);
        ranges.push(EntryRange {
            start: local.local_header_offset,
            end: member_extent_end,
        });

        let mut limitations = Vec::new();
        if uses_data_descriptor {
            limitations.push(
                "data descriptor matched central-directory metadata; payload CRC remains unverified"
                    .to_owned(),
            );
        }
        if encrypted {
            limitations
                .push("encrypted member is metadata-only; decryption is disabled".to_owned());
        }
        if path_disposition != PathDisposition::Relative {
            limitations.push(format!("member path disposition is {path_disposition:?}"));
        }

        file.seek(SeekFrom::Start(next_record))
            .map_err(|error| ReaderError::io("cannot resume central-directory traversal", error))?;

        entries.push(ArtifactEntry {
            occurrence_index: index,
            raw_name,
            decoded_name,
            path_disposition,
            kind,
            encrypted,
            uses_data_descriptor,
            general_purpose_flags: flags,
            compression_method,
            declared_compressed_size,
            declared_uncompressed_size,
            declared_crc32,
            unix_mode,
            local_header_offset: local.local_header_offset,
            data_offset: local.data_offset,
            compressed_data_end: local.compressed_data_end,
            data_descriptor_end: descriptor.map(|facts| facts.end),
            data_descriptor_has_signature: descriptor.map(|facts| facts.has_signature),
            limitations,
        });
    }

    let current = file
        .stream_position()
        .map_err(|error| ReaderError::io("cannot inspect end of central directory", error))?;
    if current != directory_end {
        if current + 6 > directory_end {
            return Err(ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "central-directory trailer is truncated",
            ));
        }
        let mut trailer = [0u8; 6];
        file.read_exact(&mut trailer)
            .map_err(|error| ReaderError::io("cannot read central-directory trailer", error))?;
        let signature_bytes = read_u32(&trailer[0..4]);
        let signature_len = read_u16(&trailer[4..6]) as u64;
        let trailer_end = current + 6 + signature_len;
        if signature_bytes != CENTRAL_DIGITAL_SIGNATURE || trailer_end != directory_end {
            return Err(ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "central directory contains an unsupported or malformed trailer",
            ));
        }
        file.seek(SeekFrom::Start(directory_end))
            .map_err(|error| ReaderError::io("cannot skip central-directory signature", error))?;
    }

    ranges.sort_by_key(|range| (range.start, range.end));
    for pair in ranges.windows(2) {
        if pair[1].start < pair[0].end {
            return Err(ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "ZIP entries have overlapping local-header or compressed-data extents",
            ));
        }
    }

    let mut name_counts: HashMap<Vec<u8>, u64> = HashMap::new();
    for entry in &entries {
        *name_counts.entry(entry.raw_name.clone()).or_default() += 1;
    }
    let duplicate_raw_names_present = name_counts.values().any(|count| *count > 1);
    if duplicate_raw_names_present {
        for entry in &mut entries {
            if name_counts.get(&entry.raw_name).copied().unwrap_or(0) > 1 {
                entry.limitations.push(
                    "duplicate raw member name; occurrence identity is the central-directory index"
                        .to_owned(),
                );
            }
        }
    }

    Ok((
        entries,
        duplicate_raw_names_present,
        total_declared_uncompressed,
    ))
}

#[derive(Debug, Clone, Copy)]
struct Zip64ExtraValues {
    uncompressed: Option<u64>,
    compressed: Option<u64>,
    local_offset: Option<u64>,
    disk_start: Option<u32>,
}

fn parse_zip64_extra(
    extra: &[u8],
    needs_uncompressed: bool,
    needs_compressed: bool,
    needs_local_offset: bool,
    needs_disk_start: bool,
) -> Result<Zip64ExtraValues, ReaderError> {
    let mut cursor = 0usize;
    let mut values = Zip64ExtraValues {
        uncompressed: None,
        compressed: None,
        local_offset: None,
        disk_start: None,
    };
    while cursor < extra.len() {
        if extra.len() - cursor < 4 {
            return Err(ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "truncated ZIP extra-field header",
            ));
        }
        let field_id = read_u16(&extra[cursor..cursor + 2]);
        let field_len = read_u16(&extra[cursor + 2..cursor + 4]) as usize;
        cursor += 4;
        let field_end = cursor.checked_add(field_len).ok_or_else(|| {
            ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "ZIP extra-field length overflow",
            )
        })?;
        if field_end > extra.len() {
            return Err(ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "ZIP extra field exceeds its containing record",
            ));
        }
        if field_id == ZIP64_EXTRA_ID {
            let data = &extra[cursor..field_end];
            let mut index = 0usize;
            if needs_uncompressed {
                values.uncompressed = Some(take_u64(data, &mut index)?);
            }
            if needs_compressed {
                values.compressed = Some(take_u64(data, &mut index)?);
            }
            if needs_local_offset {
                values.local_offset = Some(take_u64(data, &mut index)?);
            }
            if needs_disk_start {
                values.disk_start = Some(take_u32(data, &mut index)?);
            }
            break;
        }
        cursor = field_end;
    }
    Ok(values)
}

struct LocalHeaderExpectation<'a> {
    archive_start: u64,
    central_directory_offset: u64,
    relative_local_offset: u64,
    expected_name: &'a [u8],
    expected_flags: u16,
    expected_compression: u16,
    expected_crc32: u32,
    expected_compressed_size: u64,
    expected_uncompressed_size: u64,
    uses_data_descriptor: bool,
}

fn verify_local_header(
    file: &mut File,
    expected: LocalHeaderExpectation<'_>,
    limits: &ReaderLimits,
) -> Result<LocalEntryFacts, ReaderError> {
    let local_offset = expected
        .archive_start
        .checked_add(expected.relative_local_offset)
        .ok_or_else(|| {
            ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "local header offset overflow",
            )
        })?;
    if local_offset.saturating_add(LOCAL_HEADER_FIXED_BYTES) > expected.central_directory_offset {
        return Err(ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "local header lies outside the data region",
        ));
    }
    file.seek(SeekFrom::Start(local_offset))
        .map_err(|error| ReaderError::io("cannot seek to local member header", error))?;
    let mut header = [0u8; LOCAL_HEADER_FIXED_BYTES as usize];
    file.read_exact(&mut header)
        .map_err(|error| ReaderError::io("cannot read local member header", error))?;
    if read_u32(&header[0..4]) != LOCAL_HEADER_SIGNATURE {
        return Err(ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "central-directory entry does not point to a local-file-header signature",
        ));
    }

    let local_flags = read_u16(&header[6..8]);
    let local_compression = read_u16(&header[8..10]);
    let local_crc32 = read_u32(&header[14..18]);
    let local_compressed32 = read_u32(&header[18..22]);
    let local_uncompressed32 = read_u32(&header[22..26]);
    let local_name_len = read_u16(&header[26..28]) as u64;
    let local_extra_len = read_u16(&header[28..30]) as u64;

    if local_flags != expected.expected_flags || local_compression != expected.expected_compression
    {
        return Err(ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "local and central file headers disagree on flags or compression method",
        ));
    }
    enforce_limit("local member name", local_name_len, limits.max_name_bytes)?;
    enforce_limit(
        "local member extra field",
        local_extra_len,
        limits.max_extra_field_bytes,
    )?;

    let local_name_len_usize = usize::try_from(local_name_len).map_err(|_| {
        ReaderError::new(
            ReaderErrorKind::LimitExceeded,
            "local member name is too large",
        )
    })?;
    let local_extra_len_usize = usize::try_from(local_extra_len).map_err(|_| {
        ReaderError::new(
            ReaderErrorKind::LimitExceeded,
            "local extra field is too large",
        )
    })?;
    let mut local_name = vec![0u8; local_name_len_usize];
    let mut local_extra = vec![0u8; local_extra_len_usize];
    file.read_exact(&mut local_name)
        .map_err(|error| ReaderError::io("cannot read local member name", error))?;
    file.read_exact(&mut local_extra)
        .map_err(|error| ReaderError::io("cannot read local member extra field", error))?;
    if local_name != expected.expected_name {
        return Err(ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "local and central file headers disagree on raw member name",
        ));
    }

    if !expected.uses_data_descriptor {
        let local_zip64 = parse_zip64_extra(
            &local_extra,
            local_uncompressed32 == u32::MAX,
            local_compressed32 == u32::MAX,
            false,
            false,
        )?;
        let local_uncompressed = if local_uncompressed32 == u32::MAX {
            local_zip64
                .uncompressed
                .ok_or_else(|| missing_zip64_value("local uncompressed size"))?
        } else {
            local_uncompressed32 as u64
        };
        let local_compressed = if local_compressed32 == u32::MAX {
            local_zip64
                .compressed
                .ok_or_else(|| missing_zip64_value("local compressed size"))?
        } else {
            local_compressed32 as u64
        };
        if local_crc32 != expected.expected_crc32
            || local_compressed != expected.expected_compressed_size
            || local_uncompressed != expected.expected_uncompressed_size
        {
            return Err(ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "local and central file headers disagree on CRC or declared sizes",
            ));
        }
    }

    let data_offset = local_offset
        .checked_add(LOCAL_HEADER_FIXED_BYTES)
        .and_then(|value| value.checked_add(local_name_len))
        .and_then(|value| value.checked_add(local_extra_len))
        .ok_or_else(|| {
            ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "member data offset overflow",
            )
        })?;
    let compressed_data_end = data_offset
        .checked_add(expected.expected_compressed_size)
        .ok_or_else(|| {
            ReaderError::new(
                ReaderErrorKind::InvalidArchive,
                "compressed member end overflow",
            )
        })?;
    if compressed_data_end > expected.central_directory_offset {
        return Err(ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "compressed member extent overlaps the central directory",
        ));
    }
    Ok(LocalEntryFacts {
        local_header_offset: local_offset,
        data_offset,
        compressed_data_end,
    })
}

#[derive(Debug, Clone, Copy)]
struct LocalEntryFacts {
    local_header_offset: u64,
    data_offset: u64,
    compressed_data_end: u64,
}

#[derive(Debug, Clone, Copy)]
struct DataDescriptorFacts {
    end: u64,
    has_signature: bool,
}

/// Verify the optional-signature 32-bit or ZIP64 data descriptor against the
/// central-directory declaration. Ambiguous boundaries are rejected.
fn verify_data_descriptor(
    file: &mut File,
    descriptor_offset: u64,
    central_directory_offset: u64,
    expected_crc32: u32,
    expected_compressed_size: u64,
    expected_uncompressed_size: u64,
    zip64_sizes: bool,
) -> Result<DataDescriptorFacts, ReaderError> {
    let size_width = if zip64_sizes { 8usize } else { 4usize };
    let mut matches = Vec::new();

    for has_signature in [true, false] {
        let prefix_width = if has_signature { 4usize } else { 0usize };
        let descriptor_width = prefix_width + 4 + size_width * 2;
        let descriptor_end = descriptor_offset
            .checked_add(descriptor_width as u64)
            .ok_or_else(|| {
                ReaderError::new(
                    ReaderErrorKind::InvalidArchive,
                    "data descriptor end offset overflow",
                )
            })?;
        if descriptor_end > central_directory_offset {
            continue;
        }

        file.seek(SeekFrom::Start(descriptor_offset))
            .map_err(|error| ReaderError::io("cannot seek to data descriptor", error))?;
        let mut bytes = vec![0u8; descriptor_width];
        file.read_exact(&mut bytes)
            .map_err(|error| ReaderError::io("cannot read data descriptor", error))?;

        let mut cursor = 0usize;
        if has_signature {
            if read_u32(&bytes[0..4]) != DATA_DESCRIPTOR_SIGNATURE {
                continue;
            }
            cursor += 4;
        }
        let actual_crc32 = read_u32(&bytes[cursor..cursor + 4]);
        cursor += 4;
        let actual_compressed_size = if zip64_sizes {
            read_u64(&bytes[cursor..cursor + 8])
        } else {
            read_u32(&bytes[cursor..cursor + 4]) as u64
        };
        cursor += size_width;
        let actual_uncompressed_size = if zip64_sizes {
            read_u64(&bytes[cursor..cursor + 8])
        } else {
            read_u32(&bytes[cursor..cursor + 4]) as u64
        };

        if actual_crc32 == expected_crc32
            && actual_compressed_size == expected_compressed_size
            && actual_uncompressed_size == expected_uncompressed_size
        {
            matches.push(DataDescriptorFacts {
                end: descriptor_end,
                has_signature,
            });
        }
    }

    match matches.as_slice() {
        [facts] => Ok(*facts),
        [] => Err(ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "data descriptor is missing, truncated, or disagrees with central-directory CRC/sizes",
        )),
        _ => Err(ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "data descriptor boundary is ambiguous",
        )),
    }
}

fn decode_member_name(raw: &[u8], flags: u16, extra: &[u8]) -> Option<String> {
    if let Some(unicode_name) = valid_unicode_path_extra(raw, extra) {
        return Some(unicode_name);
    }
    if flags & UTF8_FLAG != 0 {
        return String::from_utf8(raw.to_vec()).ok();
    }
    Some(decode_cp437(raw))
}

fn valid_unicode_path_extra(raw_name: &[u8], extra: &[u8]) -> Option<String> {
    let mut cursor = 0usize;
    while cursor + 4 <= extra.len() {
        let field_id = read_u16(&extra[cursor..cursor + 2]);
        let field_len = read_u16(&extra[cursor + 2..cursor + 4]) as usize;
        cursor += 4;
        let end = cursor.checked_add(field_len)?;
        if end > extra.len() {
            return None;
        }
        if field_id == UNICODE_PATH_EXTRA_ID && field_len >= 5 {
            let field = &extra[cursor..end];
            if field[0] == 1 && read_u32(&field[1..5]) == crc32(raw_name) {
                return String::from_utf8(field[5..].to_vec()).ok();
            }
        }
        cursor = end;
    }
    None
}

fn decode_cp437(bytes: &[u8]) -> String {
    const HIGH: &str = "ÇüéâäàåçêëèïîìÄÅÉæÆôöòûùÿÖÜ¢£¥₧ƒáíóúñÑªº¿⌐¬½¼¡«»░▒▓│┤╡╢╖╕╣║╗╝╜╛┐└┴┬├─┼╞╟╚╔╩╦╠═╬╧╨╤╥╙╘╒╓╫╪┘┌█▄▌▐▀αßΓπΣσµτΦΘΩδ∞φε∩≡±≥≤⌠⌡÷≈°∙·√ⁿ²■\u{00A0}";
    bytes
        .iter()
        .map(|byte| {
            if *byte < 0x80 {
                char::from(*byte)
            } else {
                HIGH.chars()
                    .nth((*byte - 0x80) as usize)
                    .unwrap_or('\u{FFFD}')
            }
        })
        .collect()
}

fn classify_path(path: &str) -> PathDisposition {
    if path.is_empty() || path == "." {
        return PathDisposition::Empty;
    }
    if path.contains('\0') {
        return PathDisposition::ContainsNul;
    }
    let bytes = path.as_bytes();
    let windows_drive = bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':';
    if path.starts_with('/') || path.starts_with('\\') || windows_drive {
        return PathDisposition::Absolute;
    }
    if path.split(['/', '\\']).any(|part| part == "..") {
        return PathDisposition::ParentTraversal;
    }
    PathDisposition::Relative
}

fn enforce_limit(name: &str, value: u64, maximum: u64) -> Result<(), ReaderError> {
    if value > maximum {
        return Err(ReaderError::new(
            ReaderErrorKind::LimitExceeded,
            format!("{name} is {value} bytes; limit is {maximum}"),
        ));
    }
    Ok(())
}

fn missing_zip64_value(name: &str) -> ReaderError {
    ReaderError::new(
        ReaderErrorKind::InvalidArchive,
        format!("required ZIP64 extra-field value is missing: {name}"),
    )
}

fn take_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, ReaderError> {
    let end = cursor.checked_add(4).ok_or_else(|| {
        ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "ZIP64 extra-field cursor overflow",
        )
    })?;
    if end > bytes.len() {
        return Err(ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "ZIP64 extra field is truncated",
        ));
    }
    let value = read_u32(&bytes[*cursor..end]);
    *cursor = end;
    Ok(value)
}

fn take_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, ReaderError> {
    let end = cursor.checked_add(8).ok_or_else(|| {
        ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "ZIP64 extra-field cursor overflow",
        )
    })?;
    if end > bytes.len() {
        return Err(ReaderError::new(
            ReaderErrorKind::InvalidArchive,
            "ZIP64 extra field is truncated",
        ));
    }
    let value = read_u64(&bytes[*cursor..end]);
    *cursor = end;
    Ok(value)
}

fn read_u16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn read_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for byte in bytes {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct TestEntry<'a> {
        name: &'a [u8],
        data: &'a [u8],
        flags: u16,
        method: u16,
        external_attributes: u32,
    }

    fn build_zip(entries: &[TestEntry<'_>], comment: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut central = Vec::new();
        let mut local_offsets = Vec::new();
        for entry in entries {
            let local_offset = bytes.len() as u32;
            local_offsets.push(local_offset);
            let checksum = crc32(entry.data);
            push_u32(&mut bytes, LOCAL_HEADER_SIGNATURE);
            push_u16(&mut bytes, 20);
            push_u16(&mut bytes, entry.flags);
            push_u16(&mut bytes, entry.method);
            push_u16(&mut bytes, 0);
            push_u16(&mut bytes, 0);
            let uses_descriptor = entry.flags & DATA_DESCRIPTOR_FLAG != 0;
            push_u32(&mut bytes, if uses_descriptor { 0 } else { checksum });
            push_u32(
                &mut bytes,
                if uses_descriptor {
                    0
                } else {
                    entry.data.len() as u32
                },
            );
            push_u32(
                &mut bytes,
                if uses_descriptor {
                    0
                } else {
                    entry.data.len() as u32
                },
            );
            push_u16(&mut bytes, entry.name.len() as u16);
            push_u16(&mut bytes, 0);
            bytes.extend_from_slice(entry.name);
            bytes.extend_from_slice(entry.data);
            if uses_descriptor {
                push_u32(&mut bytes, DATA_DESCRIPTOR_SIGNATURE);
                push_u32(&mut bytes, checksum);
                push_u32(&mut bytes, entry.data.len() as u32);
                push_u32(&mut bytes, entry.data.len() as u32);
            }
        }
        let central_offset = bytes.len() as u32;
        for (entry, local_offset) in entries.iter().zip(local_offsets.iter()) {
            let checksum = crc32(entry.data);
            push_u32(&mut central, CENTRAL_HEADER_SIGNATURE);
            push_u16(&mut central, (3 << 8) | 20);
            push_u16(&mut central, 20);
            push_u16(&mut central, entry.flags);
            push_u16(&mut central, entry.method);
            push_u16(&mut central, 0);
            push_u16(&mut central, 0);
            push_u32(&mut central, checksum);
            push_u32(&mut central, entry.data.len() as u32);
            push_u32(&mut central, entry.data.len() as u32);
            push_u16(&mut central, entry.name.len() as u16);
            push_u16(&mut central, 0);
            push_u16(&mut central, 0);
            push_u16(&mut central, 0);
            push_u16(&mut central, 0);
            push_u32(&mut central, entry.external_attributes);
            push_u32(&mut central, *local_offset);
            central.extend_from_slice(entry.name);
        }
        let central_size = central.len() as u32;
        bytes.extend_from_slice(&central);
        push_u32(&mut bytes, EOCD_SIGNATURE);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, entries.len() as u16);
        push_u16(&mut bytes, entries.len() as u16);
        push_u32(&mut bytes, central_size);
        push_u32(&mut bytes, central_offset);
        push_u16(&mut bytes, comment.len() as u16);
        bytes.extend_from_slice(comment);
        bytes
    }

    fn push_u16(bytes: &mut Vec<u8>, value: u16) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn build_zip64(entries: &[TestEntry<'_>], comment: &[u8]) -> Vec<u8> {
        let mut bytes = build_zip(entries, &[]);
        let eocd_offset = bytes.len() - EOCD_FIXED_BYTES;
        let central_size = read_u32(&bytes[eocd_offset + 12..eocd_offset + 16]);
        let central_offset = read_u32(&bytes[eocd_offset + 16..eocd_offset + 20]);
        bytes.truncate(eocd_offset);
        let zip64_eocd_offset = bytes.len() as u64;

        push_u32(&mut bytes, ZIP64_EOCD_SIGNATURE);
        bytes.extend_from_slice(&44u64.to_le_bytes());
        push_u16(&mut bytes, (3 << 8) | 45);
        push_u16(&mut bytes, 45);
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, 0);
        bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(central_size as u64).to_le_bytes());
        bytes.extend_from_slice(&(central_offset as u64).to_le_bytes());

        push_u32(&mut bytes, ZIP64_LOCATOR_SIGNATURE);
        push_u32(&mut bytes, 0);
        bytes.extend_from_slice(&zip64_eocd_offset.to_le_bytes());
        push_u32(&mut bytes, 1);

        push_u32(&mut bytes, EOCD_SIGNATURE);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, u16::MAX);
        push_u16(&mut bytes, u16::MAX);
        push_u32(&mut bytes, u32::MAX);
        push_u32(&mut bytes, u32::MAX);
        push_u16(&mut bytes, comment.len() as u16);
        bytes.extend_from_slice(comment);
        bytes
    }

    fn write_temp(bytes: &[u8]) -> PathBuf {
        let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("symthaea-rea-zip-{}-{id}.zip", std::process::id()));
        let mut file = File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();
        path
    }

    fn single_entry(name: &[u8], data: &[u8]) -> Vec<u8> {
        build_zip(
            &[TestEntry {
                name,
                data,
                flags: 0,
                method: 0,
                external_attributes: 0,
            }],
            &[],
        )
    }

    #[test]
    fn verifies_stored_member_content_and_sha256_to_eof() {
        let path = write_temp(&single_entry(b"asset.bin", b"hello world"));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        let verified = reader
            .verify_entry_content(0, ContentVerificationLimits::default())
            .unwrap();

        assert_eq!(
            verified.sha256,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
        assert_eq!(verified.observed_uncompressed_bytes, 11);
        assert_eq!(verified.declared_uncompressed_bytes, 11);
        assert_eq!(verified.declared_compressed_bytes, 11);
        assert!(verified.verified_crc32);
        assert_eq!(verified.compression_method, 0);
        assert_eq!(reader.inventory().coverage.status, TraversalStatus::Partial);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn verifies_duplicate_names_by_distinct_central_directory_occurrence() {
        let path = write_temp(&build_zip(
            &[
                TestEntry {
                    name: b"same.bin",
                    data: b"first occurrence",
                    flags: 0,
                    method: 0,
                    external_attributes: 0,
                },
                TestEntry {
                    name: b"same.bin",
                    data: b"second occurrence",
                    flags: 0,
                    method: 0,
                    external_attributes: 0,
                },
            ],
            &[],
        ));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        assert!(reader.inventory().duplicate_raw_names_present);

        let first = reader
            .verify_entry_content(0, ContentVerificationLimits::default())
            .unwrap();
        let second = reader
            .verify_entry_content(1, ContentVerificationLimits::default())
            .unwrap();
        assert_ne!(first.sha256, second.sha256);
        assert_eq!(first.occurrence_index, 0);
        assert_eq!(second.occurrence_index, 1);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_crc_mismatch_in_member_payload() {
        let mut bytes = single_entry(b"asset.bin", b"hello world");
        let payload_offset = LOCAL_HEADER_FIXED_BYTES as usize + b"asset.bin".len();
        bytes[payload_offset] ^= 0x01;
        let path = write_temp(&bytes);
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        let error = reader
            .verify_entry_content(0, ContentVerificationLimits::default())
            .unwrap_err();
        assert_eq!(error.kind, ReaderErrorKind::InvalidArchive);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn enforces_content_budget_before_decompression() {
        let path = write_temp(&single_entry(b"asset.bin", b"hello world"));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        let error = reader
            .verify_entry_content(
                0,
                ContentVerificationLimits {
                    max_uncompressed_bytes: 10,
                    read_buffer_bytes: 64,
                },
            )
            .unwrap_err();
        assert_eq!(error.kind, ReaderErrorKind::LimitExceeded);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn refuses_encrypted_and_unsupported_compression_members() {
        let encrypted_path = write_temp(&build_zip(
            &[TestEntry {
                name: b"encrypted.bin",
                data: b"ciphertext",
                flags: ENCRYPTED_FLAG,
                method: 0,
                external_attributes: 0,
            }],
            &[],
        ));
        let encrypted_reader =
            ZipArtifactReader::open(&encrypted_path, ReaderLimits::default()).unwrap();
        let encrypted_error = encrypted_reader
            .verify_entry_content(0, ContentVerificationLimits::default())
            .unwrap_err();
        assert_eq!(encrypted_error.kind, ReaderErrorKind::Unsupported);
        fs::remove_file(encrypted_path).unwrap();

        let unsupported_path = write_temp(&build_zip(
            &[TestEntry {
                name: b"unsupported.bin",
                data: b"opaque",
                flags: 0,
                method: 99,
                external_attributes: 0,
            }],
            &[],
        ));
        let unsupported_reader =
            ZipArtifactReader::open(&unsupported_path, ReaderLimits::default()).unwrap();
        let unsupported_error = unsupported_reader
            .verify_entry_content(0, ContentVerificationLimits::default())
            .unwrap_err();
        assert_eq!(unsupported_error.kind, ReaderErrorKind::Unsupported);
        fs::remove_file(unsupported_path).unwrap();
    }

    #[test]
    fn verifies_data_descriptor_member_to_eof() {
        let path = write_temp(&build_zip(
            &[TestEntry {
                name: b"streamed.bin",
                data: b"data descriptor payload",
                flags: DATA_DESCRIPTOR_FLAG,
                method: 0,
                external_attributes: 0,
            }],
            &[],
        ));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        let verified = reader
            .verify_entry_content(0, ContentVerificationLimits::default())
            .unwrap();
        assert_eq!(verified.observed_uncompressed_bytes, 23);
        assert!(verified.verified_crc32);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn verifies_prepended_stub_zip64_member_content() {
        let archive = build_zip64(
            &[TestEntry {
                name: b"Payload/App",
                data: b"zip64 payload",
                flags: 0,
                method: 0,
                external_attributes: 0,
            }],
            &[],
        );
        let mut bytes = b"binary-stub".to_vec();
        bytes.extend_from_slice(&archive);
        let path = write_temp(&bytes);
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        let verified = reader
            .verify_entry_content(0, ContentVerificationLimits::default())
            .unwrap();
        assert_eq!(verified.observed_uncompressed_bytes, 13);
        assert!(verified.verified_crc32);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn verifies_deflate_member_stream() {
        let mut writer = zipcore::ZipWriter::new(std::io::Cursor::new(Vec::new())).unwrap();
        writer
            .add_file(
                "asset.bin",
                b"hello world",
                &zipcore::FileOptions::deflate(6),
            )
            .unwrap();
        let bytes = writer.finish().unwrap().into_inner();
        let path = write_temp(&bytes);
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        assert_eq!(reader.inventory().entries[0].compression_method, 8);
        let verified = reader
            .verify_entry_content(0, ContentVerificationLimits::default())
            .unwrap();
        assert_eq!(
            verified.sha256,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
        assert_eq!(verified.observed_uncompressed_bytes, 11);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_invalid_content_buffer_and_missing_occurrence() {
        let path = write_temp(&single_entry(b"asset.bin", b"hello"));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        let bad_buffer = reader
            .verify_entry_content(
                0,
                ContentVerificationLimits {
                    max_uncompressed_bytes: 16,
                    read_buffer_bytes: 0,
                },
            )
            .unwrap_err();
        assert_eq!(bad_buffer.kind, ReaderErrorKind::LimitExceeded);
        let missing = reader
            .verify_entry_content(1, ContentVerificationLimits::default())
            .unwrap_err();
        assert_eq!(missing.kind, ReaderErrorKind::InvalidArchive);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn deterministic_archive_mutations_never_panic_inventory_or_content_read() {
        let baseline = single_entry(b"mutation-target.bin", b"known payload for deterministic mutation");
        let mut state = 0x9e37_79b9_7f4a_7c15u64;

        for sample in 0..256u64 {
            let mut mutated = baseline.clone();
            for _ in 0..3 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let offset = (state as usize) % mutated.len();
                let bit = ((state >> 32) & 7) as u8;
                mutated[offset] ^= 1u8 << bit;
            }

            let path = write_temp(&mutated);
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if let Ok(reader) = ZipArtifactReader::open(&path, ReaderLimits::default()) {
                    let _ = reader
                        .verify_entry_content(0, ContentVerificationLimits::default());
                }
            }));
            fs::remove_file(path).unwrap();
            assert!(outcome.is_ok(), "archive mutation sample {sample} panicked");
        }
    }

    #[test]
    fn concurrent_content_verification_uses_independent_file_cursors() {
        let payload = vec![0x5au8; 2 * 1024 * 1024];
        let path = write_temp(&single_entry(b"concurrent.bin", &payload));
        let reader = std::sync::Arc::new(
            ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap(),
        );
        let workers: Vec<_> = (0..8)
            .map(|_| {
                let reader = std::sync::Arc::clone(&reader);
                std::thread::spawn(move || {
                    reader
                        .verify_entry_content(0, ContentVerificationLimits::default())
                        .unwrap()
                })
            })
            .collect();
        let results: Vec<_> = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect();

        assert_eq!(results.len(), 8);
        assert!(results.windows(2).all(|pair| pair[0].sha256 == pair[1].sha256));
        assert!(results.iter().all(|result| {
            result.observed_uncompressed_bytes == 2 * 1024 * 1024 && result.verified_crc32
        }));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn verifies_empty_stored_member() {
        let path = write_temp(&single_entry(b"empty.bin", b""));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        let verified = reader
            .verify_entry_content(0, ContentVerificationLimits::default())
            .unwrap();
        assert_eq!(
            verified.sha256,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(verified.observed_uncompressed_bytes, 0);
        assert!(verified.verified_crc32);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn refuses_directory_and_symlink_content_verification() {
        let path = write_temp(&build_zip(
            &[
                TestEntry {
                    name: b"folder/",
                    data: b"",
                    flags: 0,
                    method: 0,
                    external_attributes: (UNIX_DIRECTORY | 0o755) << 16,
                },
                TestEntry {
                    name: b"link",
                    data: b"target",
                    flags: 0,
                    method: 0,
                    external_attributes: (UNIX_SYMLINK | 0o777) << 16,
                },
            ],
            &[],
        ));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        let directory = reader
            .verify_entry_content(0, ContentVerificationLimits::default())
            .unwrap_err();
        let symlink = reader
            .verify_entry_content(1, ContentVerificationLimits::default())
            .unwrap_err();
        assert_eq!(directory.kind, ReaderErrorKind::Unsupported);
        assert_eq!(symlink.kind, ReaderErrorKind::Unsupported);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_deflate_output_that_exceeds_declared_uncompressed_size() {
        let payload = vec![0u8; 4096];
        let mut writer = zipcore::ZipWriter::new(std::io::Cursor::new(Vec::new())).unwrap();
        writer
            .add_file(
                "bomb.bin",
                &payload,
                &zipcore::FileOptions::deflate(6),
            )
            .unwrap();
        let mut bytes = writer.finish().unwrap().into_inner();

        let local_size = 10u32.to_le_bytes();
        bytes[22..26].copy_from_slice(&local_size);
        let central_start = bytes
            .windows(4)
            .position(|window| window == CENTRAL_HEADER_SIGNATURE.to_le_bytes())
            .unwrap();
        bytes[central_start + 24..central_start + 28].copy_from_slice(&local_size);

        let path = write_temp(&bytes);
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        let error = reader
            .verify_entry_content(0, ContentVerificationLimits::default())
            .unwrap_err();
        assert_eq!(error.kind, ReaderErrorKind::LimitExceeded);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn content_verification_refuses_a_source_changed_after_inventory() {
        let path = write_temp(&single_entry(b"asset.bin", b"payload"));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        let mut changed = OpenOptions::new().append(true).open(&path).unwrap();
        changed.write_all(b"mutated").unwrap();
        changed.sync_all().unwrap();

        let error = reader
            .verify_entry_content(0, ContentVerificationLimits::default())
            .unwrap_err();
        assert_eq!(error.kind, ReaderErrorKind::SourceChanged);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn parses_zip64_with_a_prepended_stub() {
        let archive = build_zip64(
            &[TestEntry {
                name: b"Payload/App",
                data: b"body",
                flags: 0,
                method: 0,
                external_attributes: 0,
            }],
            &[],
        );
        let mut bytes = b"binary-stub".to_vec();
        bytes.extend_from_slice(&archive);
        let path = write_temp(&bytes);
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        assert!(reader.inventory().regions.zip64);
        assert_eq!(reader.inventory().regions.leading_bytes, 11);
        assert_eq!(reader.inventory().entries.len(), 1);
        assert_eq!(
            reader.inventory().entries[0].decoded_name.as_deref(),
            Some("Payload/App")
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_multi_disk_end_records_with_an_explicit_unsupported_reason() {
        let mut bytes = single_entry(b"asset", b"data");
        let eocd = bytes.len() - EOCD_FIXED_BYTES;
        bytes[eocd + 4..eocd + 6].copy_from_slice(&1u16.to_le_bytes());
        let path = write_temp(&bytes);
        let error = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap_err();
        assert_eq!(error.kind, ReaderErrorKind::Unsupported);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_a_plausible_false_eocd_inside_the_archive_comment() {
        let fake_eocd = {
            let mut fake = Vec::new();
            push_u32(&mut fake, EOCD_SIGNATURE);
            push_u16(&mut fake, 0);
            push_u16(&mut fake, 0);
            push_u16(&mut fake, 0);
            push_u16(&mut fake, 0);
            push_u32(&mut fake, 0);
            push_u32(&mut fake, 0);
            push_u16(&mut fake, 0);
            fake
        };
        let path = write_temp(&build_zip(
            &[TestEntry {
                name: b"real-entry",
                data: b"payload",
                flags: 0,
                method: 0,
                external_attributes: 0,
            }],
            &fake_eocd,
        ));
        let error = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap_err();
        assert_eq!(error.kind, ReaderErrorKind::AmbiguousEndRecord);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn data_descriptor_entries_are_marked_without_treating_local_zeroes_as_final() {
        let path = write_temp(&build_zip(
            &[TestEntry {
                name: b"descriptor.bin",
                data: b"payload",
                flags: DATA_DESCRIPTOR_FLAG,
                method: 0,
                external_attributes: 0,
            }],
            &[],
        ));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        let entry = &reader.inventory().entries[0];
        assert!(entry.uses_data_descriptor);
        assert_eq!(entry.declared_crc32, crc32(b"payload"));
        assert_eq!(entry.declared_uncompressed_size, 7);
        assert!(entry.data_descriptor_end.is_some());
        assert_eq!(entry.data_descriptor_has_signature, Some(true));
        assert!(
            entry
                .limitations
                .iter()
                .any(|item| item.contains("payload CRC remains unverified"))
        );
        fs::remove_file(path).unwrap();

        let path = write_temp(&build_zip(
            &[TestEntry {
                name: b"descriptor.bin",
                data: b"payload",
                flags: DATA_DESCRIPTOR_FLAG,
                method: 0,
                external_attributes: 0,
            }],
            &[],
        ));
        let mut bytes = fs::read(&path).unwrap();
        fs::remove_file(&path).unwrap();
        let descriptor_start =
            LOCAL_HEADER_FIXED_BYTES as usize + b"descriptor.bin".len() + b"payload".len();
        bytes.drain(descriptor_start..descriptor_start + 4);
        let eocd = bytes.len() - EOCD_FIXED_BYTES;
        let old_directory_offset = read_u32(&bytes[eocd + 16..eocd + 20]);
        bytes[eocd + 16..eocd + 20].copy_from_slice(&(old_directory_offset - 4).to_le_bytes());
        let path = write_temp(&bytes);
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        assert_eq!(
            reader.inventory().entries[0].data_descriptor_has_signature,
            Some(false)
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_data_descriptor_crc_or_size_substitution() {
        let path = write_temp(&build_zip(
            &[TestEntry {
                name: b"descriptor.bin",
                data: b"payload",
                flags: DATA_DESCRIPTOR_FLAG,
                method: 0,
                external_attributes: 0,
            }],
            &[],
        ));
        let mut bytes = fs::read(&path).unwrap();
        fs::remove_file(&path).unwrap();
        let descriptor_start =
            LOCAL_HEADER_FIXED_BYTES as usize + b"descriptor.bin".len() + b"payload".len();
        bytes[descriptor_start + 4] ^= 0x01;
        let path = write_temp(&bytes);
        let error = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap_err();
        assert_eq!(error.kind, ReaderErrorKind::InvalidArchive);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn unsupported_compression_method_remains_visible_as_metadata() {
        let path = write_temp(&build_zip(
            &[TestEntry {
                name: b"packed.bin",
                data: b"opaque",
                flags: 0,
                method: 99,
                external_attributes: 0,
            }],
            &[],
        ));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        let entry = &reader.inventory().entries[0];
        assert_eq!(entry.compression_method, 99);
        assert!(!entry.encrypted);
        assert_eq!(reader.inventory().coverage.status, TraversalStatus::Partial);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn classifies_backslash_parent_traversal_on_unix_hosts() {
        let path = write_temp(&single_entry(b"..\\escape.txt", b"payload"));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        assert_eq!(
            reader.inventory().entries[0].path_disposition,
            PathDisposition::ParentTraversal
        );
        assert_eq!(
            reader.inventory().entries[0].decoded_name.as_deref(),
            Some("..\\escape.txt")
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn enforces_entry_count_and_aggregate_declared_expansion_budgets() {
        let bytes = build_zip(
            &[TestEntry {
                name: b"asset",
                data: b"payload",
                flags: 0,
                method: 0,
                external_attributes: 0,
            }],
            &[],
        );
        let path = write_temp(&bytes);
        let limits = ReaderLimits {
            max_entries: 0,
            ..ReaderLimits::default()
        };
        assert_eq!(
            ZipArtifactReader::open(&path, limits).unwrap_err().kind,
            ReaderErrorKind::LimitExceeded
        );
        fs::remove_file(&path).unwrap();

        let path = write_temp(&bytes);
        let limits = ReaderLimits {
            max_total_declared_uncompressed_bytes: 1,
            ..ReaderLimits::default()
        };
        assert_eq!(
            ZipArtifactReader::open(&path, limits).unwrap_err().kind,
            ReaderErrorKind::LimitExceeded
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn inventories_stored_member_without_extracting_it() {
        let path = write_temp(&single_entry(b"classes.dex", b"sample"));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        let inventory = reader.inventory();
        assert_eq!(inventory.entries.len(), 1);
        assert_eq!(inventory.entries[0].occurrence_index, 0);
        assert_eq!(
            inventory.entries[0].decoded_name.as_deref(),
            Some("classes.dex")
        );
        assert_eq!(inventory.entries[0].kind, EntryKind::File);
        assert_eq!(inventory.entries[0].declared_uncompressed_size, 6);
        assert_eq!(inventory.entries[0].declared_crc32, crc32(b"sample"));
        assert_eq!(inventory.coverage.status, TraversalStatus::Partial);
        reader.verify_source().unwrap();
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn duplicate_names_are_preserved_as_distinct_occurrences() {
        let path = write_temp(&build_zip(
            &[
                TestEntry {
                    name: b"classes.dex",
                    data: b"first",
                    flags: 0,
                    method: 0,
                    external_attributes: 0,
                },
                TestEntry {
                    name: b"classes.dex",
                    data: b"second",
                    flags: 0,
                    method: 0,
                    external_attributes: 0,
                },
            ],
            &[],
        ));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        let inventory = reader.inventory();
        assert!(inventory.duplicate_raw_names_present);
        assert_eq!(inventory.entries.len(), 2);
        assert_ne!(
            inventory.entries[0].occurrence_index,
            inventory.entries[1].occurrence_index
        );
        assert_eq!(
            inventory.entries[0].decoded_name,
            inventory.entries[1].decoded_name
        );
        assert!(inventory.entries.iter().all(|entry| {
            entry
                .limitations
                .iter()
                .any(|x| x.contains("duplicate raw member name"))
        }));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn preserves_leading_stub_and_trailing_comment_as_non_entry_regions() {
        let mut bytes = b"MZ-stub".to_vec();
        bytes.extend_from_slice(&single_entry(b"Payload/app.bin", b"body"));
        let comment = b"archive comment";
        let comment_start = bytes.len() - 2;
        bytes[comment_start..comment_start + 2]
            .copy_from_slice(&(comment.len() as u16).to_le_bytes());
        bytes.extend_from_slice(comment);
        // The fixture helper above already wrote an empty comment field. Add the
        // real comment and update the EOCD comment length after appending.
        let eocd = bytes.len() - comment.len() - EOCD_FIXED_BYTES;
        bytes[eocd + 20..eocd + 22].copy_from_slice(&(comment.len() as u16).to_le_bytes());
        let path = write_temp(&bytes);
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        assert_eq!(reader.inventory().regions.leading_bytes, 7);
        assert_eq!(
            reader.inventory().regions.trailing_comment_bytes,
            comment.len() as u64
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_local_and_central_header_disagreement() {
        let mut bytes = single_entry(b"classes.dex", b"payload");
        let name_offset = LOCAL_HEADER_FIXED_BYTES as usize;
        bytes[name_offset] = b'X';
        let path = write_temp(&bytes);
        let error = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap_err();
        assert_eq!(error.kind, ReaderErrorKind::InvalidArchive);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_overlapping_member_extents() {
        let mut bytes = build_zip(
            &[
                TestEntry {
                    name: b"one",
                    data: b"first",
                    flags: 0,
                    method: 0,
                    external_attributes: 0,
                },
                TestEntry {
                    name: b"two",
                    data: b"second",
                    flags: 0,
                    method: 0,
                    external_attributes: 0,
                },
            ],
            &[],
        );
        let central_start = bytes
            .windows(4)
            .position(|window| window == CENTRAL_HEADER_SIGNATURE.to_le_bytes())
            .unwrap();
        let second_header = central_start + CENTRAL_HEADER_FIXED_BYTES as usize + 3;
        let first_local_offset = read_u32(&bytes[central_start + 42..central_start + 46]);
        bytes[second_header + 42..second_header + 46]
            .copy_from_slice(&first_local_offset.to_le_bytes());
        let path = write_temp(&bytes);
        let error = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap_err();
        assert_eq!(error.kind, ReaderErrorKind::InvalidArchive);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn enforces_archive_and_central_directory_limits_before_entry_allocation() {
        let path = write_temp(&single_entry(b"a", b"x"));
        let limits = ReaderLimits {
            max_central_directory_bytes: 1,
            ..ReaderLimits::default()
        };
        let error = ZipArtifactReader::open(&path, limits).unwrap_err();
        assert_eq!(error.kind, ReaderErrorKind::LimitExceeded);
        fs::remove_file(path).unwrap();

        let path = write_temp(&single_entry(b"a", b"x"));
        let limits = ReaderLimits {
            max_archive_bytes: 1,
            ..ReaderLimits::default()
        };
        let error = ZipArtifactReader::open(&path, limits).unwrap_err();
        assert_eq!(error.kind, ReaderErrorKind::LimitExceeded);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn source_identity_detects_append_after_inventory() {
        let path = write_temp(&single_entry(b"asset", b"data"));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(b"changed").unwrap();
        file.sync_all().unwrap();
        let error = reader.verify_source().unwrap_err();
        assert_eq!(error.kind, ReaderErrorKind::SourceChanged);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn preserves_encrypted_and_symlink_metadata_without_reading_content() {
        let path = write_temp(&build_zip(
            &[
                TestEntry {
                    name: b"encrypted.bin",
                    data: b"x",
                    flags: ENCRYPTED_FLAG,
                    method: 99,
                    external_attributes: 0,
                },
                TestEntry {
                    name: b"link",
                    data: b"target",
                    flags: 0,
                    method: 0,
                    external_attributes: (0o120777u32) << 16,
                },
            ],
            &[],
        ));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        assert!(reader.inventory().entries[0].encrypted);
        assert_eq!(reader.inventory().entries[0].compression_method, 99);
        assert_eq!(reader.inventory().entries[1].kind, EntryKind::Symlink);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn retains_unsafe_paths_instead_of_sanitizing_them() {
        let path = write_temp(&single_entry(b"../escape.txt", b"payload"));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        assert_eq!(
            reader.inventory().entries[0].decoded_name.as_deref(),
            Some("../escape.txt")
        );
        assert_eq!(
            reader.inventory().entries[0].path_disposition,
            PathDisposition::ParentTraversal
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn retains_invalid_utf8_name_bytes_without_reinterpreting_them() {
        let path = write_temp(&build_zip(
            &[TestEntry {
                name: &[0xff, 0xfe],
                data: b"x",
                flags: UTF8_FLAG,
                method: 0,
                external_attributes: 0,
            }],
            &[],
        ));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        let entry = &reader.inventory().entries[0];
        assert_eq!(entry.raw_name, vec![0xff, 0xfe]);
        assert_eq!(entry.decoded_name, None);
        assert_eq!(entry.path_disposition, PathDisposition::Undecodable);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn decodes_cp437_fallback_names_without_utf8_flag() {
        assert_eq!(
            decode_member_name(b"caf\x82", 0, &[]).as_deref(),
            Some("café")
        );
        assert_eq!(decode_cp437(&[0x80, 0x82, 0x9c]), "Çé£");
    }

    #[test]
    fn accepts_unicode_path_extra_only_when_version_and_raw_name_crc_match() {
        let raw_name = b"caf\x82";
        let unicode_name = "café";
        let mut extra = Vec::new();
        push_u16(&mut extra, UNICODE_PATH_EXTRA_ID);
        push_u16(&mut extra, (5 + unicode_name.len()) as u16);
        extra.push(1);
        push_u32(&mut extra, crc32(raw_name));
        extra.extend_from_slice(unicode_name.as_bytes());

        assert_eq!(
            valid_unicode_path_extra(raw_name, &extra).as_deref(),
            Some(unicode_name)
        );
        assert_eq!(
            decode_member_name(raw_name, 0, &extra).as_deref(),
            Some(unicode_name)
        );

        let mut wrong_crc = extra.clone();
        wrong_crc[7] ^= 0x01;
        assert_eq!(valid_unicode_path_extra(raw_name, &wrong_crc), None);
        assert_eq!(
            decode_member_name(raw_name, 0, &wrong_crc).as_deref(),
            Some("café")
        );

        let mut wrong_version = extra;
        wrong_version[4] = 2;
        assert_eq!(valid_unicode_path_extra(raw_name, &wrong_version), None);
        assert_eq!(
            decode_member_name(raw_name, 0, &wrong_version).as_deref(),
            Some("café")
        );
    }

    #[test]
    fn classifies_directory_entries_and_preserves_occurrence() {
        let path = write_temp(&build_zip(
            &[TestEntry {
                name: b"assets/",
                data: b"",
                flags: 0,
                method: 0,
                external_attributes: (UNIX_DIRECTORY | 0o755) << 16,
            }],
            &[],
        ));
        let reader = ZipArtifactReader::open(&path, ReaderLimits::default()).unwrap();
        assert_eq!(reader.inventory().entries[0].kind, EntryKind::Directory);
        fs::remove_file(path).unwrap();
    }
}
