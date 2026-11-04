#![no_main]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use libfuzzer_sys::fuzz_target;
use symthaea_rea_artifact_reader::{
    ContentVerificationLimits, EntryKind, ReaderLimits, ZipArtifactReader,
};

const MAX_INPUT_BYTES: usize = 1024 * 1024;
const MAX_ENTRIES_TO_VERIFY: usize = 8;
const MAX_MEMBER_OUTPUT_BYTES: u64 = 256 * 1024;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct TempArchive(PathBuf);

impl Drop for TempArchive {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "symthaea-rea-zip-fuzz-{}-{id}.zip",
        std::process::id()
    ));
    if fs::write(&path, data).is_err() {
        return;
    }
    let _cleanup = TempArchive(path.clone());

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

    let content_limits = ContentVerificationLimits {
        max_uncompressed_bytes: MAX_MEMBER_OUTPUT_BYTES,
        read_buffer_bytes: 4096,
    };

    for (index, entry) in reader
        .inventory()
        .entries
        .iter()
        .take(MAX_ENTRIES_TO_VERIFY)
        .enumerate()
    {
        if entry.kind != EntryKind::File
            || entry.encrypted
            || !matches!(entry.compression_method, 0 | 8)
            || entry.declared_uncompressed_size > MAX_MEMBER_OUTPUT_BYTES
        {
            continue;
        }

        // Structured CRC, size, codec, source-change, unsupported and limit errors are
        // expected outcomes for mutated inputs. A panic is a fuzz finding.
        let _ = reader.verify_entry_content(index as u64, content_limits.clone());
    }
});
