#![no_main]

//! Differential fuzzing for the supported shared ZIP subset.
//!
//! An admitted input is compared against zip-rs/zlib-rs only when both implementations are
//! expected to describe the same ordinary, single-disk, non-ZIP64, unprefixed, unique-name ZIP.
//! Outside that subset this target abstains. In-scope metadata or content disagreement panics so
//! libFuzzer minimizes the input into a reproducible finding.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use libfuzzer_sys::fuzz_target;
use sha2::{Digest, Sha256};
use symthaea_rea_artifact_reader::{
    ContentVerificationLimits, EntryKind, ReaderLimits, TraversalStatus, ZipArtifactReader,
};
use zip::{CompressionMethod, ZipArchive};

const MAX_INPUT_BYTES: usize = 1024 * 1024;
const MAX_ENTRIES_TO_COMPARE: usize = 256;
const MAX_MEMBER_OUTPUT_BYTES: u64 = 256 * 1024;
const READ_BUFFER_BYTES: usize = 4096;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct TempArchive(PathBuf);

impl Drop for TempArchive {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OracleContent {
    sha256: String,
    bytes: u64,
    crc32: u32,
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "symthaea-rea-zip-differential-{}-{id}.zip",
        std::process::id()
    ));
    let Ok(mut file) = OpenOptions::new().write(true).create_new(true).open(&path) else {
        return;
    };
    let _cleanup = TempArchive(path.clone());
    if file.write_all(data).is_err() {
        return;
    }
    drop(file);

    let reader_limits = ReaderLimits {
        max_archive_bytes: MAX_INPUT_BYTES as u64,
        max_central_directory_bytes: 256 * 1024,
        max_entries: MAX_ENTRIES_TO_COMPARE as u64,
        max_name_bytes: 4096,
        max_extra_field_bytes: 64 * 1024,
        max_total_declared_uncompressed_bytes: 4 * 1024 * 1024,
        max_entry_declared_uncompressed_bytes: MAX_MEMBER_OUTPUT_BYTES,
        max_zip64_record_bytes: 64 * 1024,
    };
    let Ok(reader) = ZipArtifactReader::open(&path, reader_limits) else {
        return;
    };
    let inventory = reader.inventory();

    // This first differential slice deliberately excludes ZIP64, prefixes, partial traversal,
    // duplicate-name mapping and archives above the small fuzz limits. Those are separate
    // oracle-compatibility families, not implicit parser failures.
    if inventory.coverage.status != TraversalStatus::Complete
        || inventory.duplicate_raw_names_present
        || inventory.regions.leading_bytes != 0
        || inventory.regions.zip64
        || inventory.entries.len() > MAX_ENTRIES_TO_COMPARE
    {
        return;
    }

    let Ok(mut oracle) = ZipArchive::new(File::open(&path).expect("fuzz temp archive exists")) else {
        // zip-rs rejecting an input admitted by the production reader is recorded as an
        // implementation disagreement for this otherwise shared-subset candidate.
        panic!("differential disagreement: production admitted ordinary unique-name ZIP but zip-rs rejected it");
    };
    if oracle.has_duplicate_names() {
        return;
    }
    if oracle.len() != inventory.entries.len() {
        panic!(
            "differential disagreement: entry count production={} zip-rs={}",
            inventory.entries.len(),
            oracle.len()
        );
    }
    if oracle.validate_header_consistency().is_err()
        || oracle.has_overlapping_entries().unwrap_or(true)
    {
        panic!("differential disagreement: production admitted archive rejected by zip-rs header/overlap checks");
    }

    for (index, expected) in inventory.entries.iter().enumerate() {
        let Ok(mut member) = oracle.by_index(index) else {
            panic!("differential disagreement: zip-rs cannot open inventory occurrence {index}");
        };

        if member.name_raw() != expected.raw_name.as_slice() {
            panic!("differential metadata disagreement: raw filename bytes at occurrence {index}");
        }
        if member.flags().as_u16() != expected.general_purpose_flags {
            panic!("differential metadata disagreement: general-purpose flags at occurrence {index}");
        }
        if member.compressed_size() != expected.declared_compressed_size {
            panic!("differential metadata disagreement: compressed size at occurrence {index}");
        }
        if member.size() != expected.declared_uncompressed_size {
            panic!("differential metadata disagreement: uncompressed size at occurrence {index}");
        }
        if member.crc32() != expected.declared_crc32 {
            panic!("differential metadata disagreement: declared CRC-32 at occurrence {index}");
        }
        if member.header_start() != expected.local_header_offset {
            panic!("differential metadata disagreement: local-header offset at occurrence {index}");
        }
        if member.data_start() != Some(expected.data_offset) {
            panic!("differential metadata disagreement: data offset at occurrence {index}");
        }

        let oracle_method = match member.compression() {
            CompressionMethod::Stored => 0,
            CompressionMethod::Deflated => 8,
            _ => continue,
        };
        if oracle_method != expected.compression_method {
            panic!("differential metadata disagreement: compression method at occurrence {index}");
        }

        let oracle_kind_matches = match expected.kind {
            EntryKind::File => !member.is_dir() && !member.is_symlink(),
            EntryKind::Directory => member.is_dir(),
            EntryKind::Symlink => member.is_symlink(),
        };
        if !oracle_kind_matches {
            panic!("differential metadata disagreement: entry kind at occurrence {index}");
        }

        if expected.kind != EntryKind::File
            || expected.encrypted
            || !matches!(expected.compression_method, 0 | 8)
            || expected.declared_uncompressed_size > MAX_MEMBER_OUTPUT_BYTES
        {
            continue;
        }

        let production = reader.verify_entry_content(
            expected.occurrence_index,
            ContentVerificationLimits {
                max_uncompressed_bytes: MAX_MEMBER_OUTPUT_BYTES,
                read_buffer_bytes: READ_BUFFER_BYTES,
            },
        );
        let oracle_content = stream_oracle_member(&mut member, MAX_MEMBER_OUTPUT_BYTES);
        match (production, oracle_content) {
            (Ok(production), Ok(oracle)) => {
                if production.sha256 != oracle.sha256
                    || production.observed_uncompressed_bytes != oracle.bytes
                    || !production.verified_crc32
                {
                    panic!(
                        "differential content disagreement at occurrence {index}: production sha/bytes/CRC-accepted={}/{}/{}; oracle sha/bytes/crc={}/{}/{:08x}",
                        production.sha256,
                        production.observed_uncompressed_bytes,
                        production.verified_crc32,
                        oracle.sha256,
                        oracle.bytes,
                        oracle.crc32
                    );
                }
                // The base reader API exposes the custom CRC result as a boolean rather than
                // a numeric field. The oracle independently checks its decoded CRC against the
                // central-directory CRC above; production must also accept its own CRC check.
            }
            (Err(_), Err(_)) => {
                // Both implementations rejected the member. Their error taxonomies differ and
                // are recorded in dedicated semantic fixtures; this is not a false agreement.
            }
            (Ok(_), Err(error)) => {
                panic!("differential one-sided acceptance at occurrence {index}: production verified, zip-rs rejected: {error}");
            }
            (Err(error), Ok(_)) => {
                panic!("differential one-sided acceptance at occurrence {index}: production rejected, zip-rs verified: {error}");
            }
        }
    }
});

fn stream_oracle_member<R: Read>(
    member: &mut R,
    max_output_bytes: u64,
) -> Result<OracleContent, String> {
    let mut sha = Sha256::new();
    let mut crc = Crc32::new();
    let mut buffer = [0u8; READ_BUFFER_BYTES];
    let mut observed = 0u64;
    loop {
        // One-byte sentinel makes an overrun observable even when the limit is reached exactly.
        let remaining = max_output_bytes.saturating_sub(observed);
        let request = usize::try_from(remaining.saturating_add(1))
            .unwrap_or(usize::MAX)
            .min(buffer.len());
        let read = member
            .read(&mut buffer[..request])
            .map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        if read as u64 > remaining {
            return Err("oracle output exceeded bounded member limit".into());
        }
        observed = observed
            .checked_add(read as u64)
            .ok_or_else(|| "oracle output byte count overflow".to_string())?;
        sha.update(&buffer[..read]);
        crc.update(&buffer[..read]);
    }
    Ok(OracleContent {
        sha256: format!("{:x}", sha.finalize()),
        bytes: observed,
        crc32: crc.finish(),
    })
}

struct Crc32(u32);

impl Crc32 {
    fn new() -> Self {
        Self(u32::MAX)
    }

    fn update(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u32::from(*byte);
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
