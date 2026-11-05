#![no_main]

//! Differential fuzzing across Symthaea's occurrence-preserving reader and zip-rs/zlib-rs.
//!
//! Only compare the intersection where both implementations claim support: non-prefixed,
//! non-ZIP64, single-disk archives with unique raw filenames and Stored/Deflate file entries.
//! Duplicate-name and layout differences are not parser bugs; they are outside this oracle's
//! occurrence-mapping contract and are deliberately skipped.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use libfuzzer_sys::fuzz_target;
use sha2::{Digest, Sha256};
use symthaea_rea_artifact_reader::{
    ContentVerificationLimits, EntryKind, ReaderLimits, ZipArtifactReader,
};
use zip::CompressionMethod;

const MAX_INPUT_BYTES: usize = 1024 * 1024;
const MAX_ENTRIES_TO_COMPARE: usize = 8;
const MAX_MEMBER_OUTPUT_BYTES: u64 = 256 * 1024;
const ORACLE_BUFFER_BYTES: usize = 16 * 1024;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct TempArchive(PathBuf);

impl Drop for TempArchive {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn mismatch(message: &str) -> ! {
    panic!("REA/zip-rs differential disagreement: {message}");
}

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "symthaea-rea-differential-fuzz-{}-{id}.zip",
        std::process::id()
    ));
    let Ok(mut temp_file) = OpenOptions::new().write(true).create_new(true).open(&path) else {
        return;
    };
    let _cleanup = TempArchive(path.clone());
    if temp_file.write_all(data).is_err() {
        return;
    }
    drop(temp_file);

    let limits = ReaderLimits {
        max_archive_bytes: MAX_INPUT_BYTES as u64,
        max_central_directory_bytes: 256 * 1024,
        max_entries: 256,
        max_name_bytes: 4096,
        max_extra_field_bytes: 64 * 1024,
        max_total_declared_uncompressed_bytes: 4 * 1024 * 1024,
        max_entry_declared_uncompressed_bytes: MAX_MEMBER_OUTPUT_BYTES,
        max_zip64_record_bytes: 64 * 1024,
    };
    let Ok(reader) = ZipArtifactReader::open(&path, limits) else {
        return;
    };
    let inventory = reader.inventory();

    // The independent oracle is intentionally not asked to map occurrence identity when names
    // are duplicated, nor to compensate for prefix or ZIP64 layouts.
    if inventory.duplicate_raw_names_present
        || inventory.regions.leading_bytes != 0
        || inventory.regions.zip64
        || inventory.entries.len() > 256
    {
        return;
    }

    let Ok(mut oracle) = zip::ZipArchive::new(File::open(&path).unwrap()) else {
        return;
    };
    if oracle.has_duplicate_names() || oracle.len() != inventory.entries.len() {
        return;
    }
    if oracle.validate_header_consistency().is_err()
        || oracle.has_overlapping_entries().unwrap_or(true)
    {
        return;
    }

    for index in 0..inventory.entries.len().min(MAX_ENTRIES_TO_COMPARE) {
        let expected = &inventory.entries[index];
        let Ok(mut member) = oracle.by_index(index) else {
            mismatch("zip-rs omitted an occurrence present in the custom inventory");
        };

        if member.name_raw() != expected.raw_name
            || member.header_start() != expected.local_header_offset
            || member.data_start() != Some(expected.data_offset)
            || member.flags().bits() != expected.general_purpose_flags
            || member.compressed_size() != expected.declared_compressed_size
            || member.size() != expected.declared_uncompressed_size
            || member.crc32() != expected.declared_crc32
        {
            mismatch("raw name, offset, flags, sizes or declared CRC differ");
        }

        let oracle_method = match member.compression() {
            CompressionMethod::Stored => 0,
            CompressionMethod::Deflated => 8,
            _ => u16::MAX,
        };
        if oracle_method != expected.compression_method {
            mismatch("compression method differs");
        }

        if expected.kind != EntryKind::File
            || expected.encrypted
            || !matches!(expected.compression_method, 0 | 8)
            || expected.declared_uncompressed_size > MAX_MEMBER_OUTPUT_BYTES
        {
            continue;
        }

        // Keep the independent decoder's output strictly bounded. The extra byte detects
        // whether a stream exceeds its declared/caller limit without buffering the full result.
        let mut oracle_stream = member.take(MAX_MEMBER_OUTPUT_BYTES + 1);
        let mut oracle_payload = Vec::with_capacity(
            usize::try_from(expected.declared_uncompressed_size.min(MAX_MEMBER_OUTPUT_BYTES))
                .unwrap_or(0),
        );
        if oracle_stream.read_to_end(&mut oracle_payload).is_err() {
            // A CRC/Deflate failure in the independent implementation is an expected
            // classification difference for malformed fuzz input, not a false panic.
            continue;
        }
        if oracle_payload.len() as u64 > MAX_MEMBER_OUTPUT_BYTES {
            continue;
        }
        let oracle_sha256 = format!("{:x}", Sha256::digest(&oracle_payload));

        let verification = reader.verify_entry_content(
            expected.occurrence_index,
            ContentVerificationLimits {
                max_uncompressed_bytes: MAX_MEMBER_OUTPUT_BYTES,
                read_buffer_bytes: ORACLE_BUFFER_BYTES,
            },
        );

        if let Ok(verified) = verification {
            if verified.observed_uncompressed_bytes != oracle_payload.len() as u64 {
                mismatch("successful decoders produced different uncompressed lengths");
            }
            if verified.sha256 != oracle_sha256 {
                mismatch("successful decoders produced different member SHA-256 digests");
            }
            if !verified.verified_crc32 {
                mismatch("custom reader reported success without independent CRC acceptance");
            }
            if verified.observed_crc32 != member.crc32() {
                mismatch("custom streaming CRC differs from zip-rs central-directory CRC");
            }
        }
    }
});
