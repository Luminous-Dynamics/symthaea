use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use flate2::read::MultiGzDecoder;
use md5::{Digest, Md5};
use serde::Serialize;
use sha2::Sha256;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};
use tar::Archive;

const DATASET_NAME: &str = "Open Catalyst 2022 (OC22)";
const DATASET_TASK: &str = "IS2RE-Total / IS2RS combined archive";
const UPSTREAM_DOCS_COMMIT: &str = "57c71322d5a5be0a0fcdcf05592e1699b120dc79";
const UPSTREAM_DOCS_URL: &str = "https://github.com/facebookresearch/fairchem/blob/57c71322d5a5be0a0fcdcf05592e1699b120dc79/docs/catalysts/datasets/oc22.md";
const UPSTREAM_ARTIFACT_URL: &str = "https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/is2res_total_train_val_test_lmdbs.tar.gz";
const UPSTREAM_REPORTED_MD5: &str = "b35dc24e99ef3aeaee6c5c949903de94";
const MAX_COMPRESSED_ARCHIVE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_UNCOMPRESSED_ARCHIVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_TAR_MEMBERS: u64 = 100_000;
const MAX_MEMBER_PATH_BYTES: usize = 4 * 1024;
const MAX_TOTAL_MEMBER_PATH_BYTES: u64 = 32 * 1024 * 1024;
const UPSTREAM_REPORTED_LICENSE: &str = "CC BY 4.0 (as reported by upstream; verify the exact intended use and attribution)";

#[derive(Debug, Parser)]
#[command(
    name = "nano-admission-verifier",
    version,
    about = "Hash and inventory a local OC22 IS2RE-Total archive without extracting it"
)]
struct Cli {
    /// Path to the locally downloaded OC22 .tar.gz archive.
    #[arg(long, value_name = "PATH")]
    archive: PathBuf,

    /// Destination for the JSON receipt. Does not have to be inside the repository.
    #[arg(long, value_name = "PATH", default_value = "oc22-artifact-integrity.json")]
    output: PathBuf,
}

#[derive(Debug, Serialize)]
struct UpstreamSource {
    dataset: &'static str,
    task: &'static str,
    documentation_url: &'static str,
    documentation_commit: &'static str,
    artifact_url: &'static str,
    upstream_reported_md5: &'static str,
    upstream_reported_dataset_license: &'static str,
}

#[derive(Debug, Serialize)]
struct ArtifactIdentity {
    filename: String,
    byte_size_before_inspection: u64,
    measured_md5_before_inspection: String,
    upstream_md5_matches: bool,
    measured_sha256_before_inspection: String,
    byte_size_after_inspection: u64,
    measured_md5_after_inspection: String,
    measured_sha256_after_inspection: String,
    contents_unchanged_during_inspection: bool,
}

#[derive(Debug, Default, Serialize)]
struct ArchiveInventory {
    member_count: u64,
    regular_file_count: u64,
    directory_count: u64,
    declared_payload_bytes: u64,
    member_path_bytes_inspected: u64,
    top_level_member_counts: BTreeMap<String, u64>,
    duplicate_member_paths: Vec<String>,
    unsafe_member_paths: Vec<String>,
    special_member_entries: Vec<String>,
    payload_bytes_fully_streamed: bool,
}

impl ArchiveInventory {
    fn requires_review(&self) -> bool {
        !self.duplicate_member_paths.is_empty()
            || !self.unsafe_member_paths.is_empty()
            || !self.special_member_entries.is_empty()
            || !self.payload_bytes_fully_streamed
    }
}

#[derive(Debug, Serialize)]
struct AdmissionReceipt {
    schema_version: &'static str,
    status: &'static str,
    generated_at_unix_seconds: u64,
    upstream: UpstreamSource,
    artifact: ArtifactIdentity,
    archive_inventory: Option<ArchiveInventory>,
    archive_inspection_error: Option<String>,
    limitations: Vec<&'static str>,
}

#[derive(Debug, PartialEq, Eq)]
struct HashRecord {
    byte_size: u64,
    md5: String,
    sha256: String,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("nano-admission-verifier: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode> {
    let cli = Cli::parse();
    reject_same_input_and_output(&cli.archive, &cli.output)?;

    // Open once and validate the opened descriptor, rather than trusting a
    // pathname metadata preflight that can race a symlink/FIFO replacement.
    // On Unix, O_NONBLOCK ensures an unexpected FIFO cannot hang this open.
    let mut archive_file = open_regular_input(&cli.archive)?;
    let compressed_size = archive_file
        .metadata()
        .context("could not inspect opened archive metadata")?
        .len();
    if compressed_size > MAX_COMPRESSED_ARCHIVE_BYTES {
        bail!(
            "compressed archive exceeds configured byte limit of {MAX_COMPRESSED_ARCHIVE_BYTES}"
        );
    }
    let hashes_before = hash_open_file(&mut archive_file)
        .with_context(|| format!("could not hash archive {}", cli.archive.display()))?;
    let upstream_md5_matches = hashes_before.md5.eq_ignore_ascii_case(UPSTREAM_REPORTED_MD5);

    let (archive_inventory, archive_inspection_error) =
        inventory_if_checksum_matches(&mut archive_file, upstream_md5_matches);
    // Re-hash the same opened file after inventory so in-place content changes
    // during inspection block a clean receipt. This still is not an OS-level
    // immutable-file guarantee against a writer that mutates and restores bytes.
    let hashes_after = hash_open_file(&mut archive_file)
        .with_context(|| format!("could not re-hash archive {}", cli.archive.display()))?;
    let contents_unchanged = hashes_before == hashes_after;
    let status = status_for(upstream_md5_matches, archive_inventory.as_ref(), contents_unchanged);

    let generated_at_unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let receipt = AdmissionReceipt {
        schema_version: "nano-artifact-admission-v1",
        status,
        generated_at_unix_seconds,
        upstream: UpstreamSource {
            dataset: DATASET_NAME,
            task: DATASET_TASK,
            documentation_url: UPSTREAM_DOCS_URL,
            documentation_commit: UPSTREAM_DOCS_COMMIT,
            artifact_url: UPSTREAM_ARTIFACT_URL,
            upstream_reported_md5: UPSTREAM_REPORTED_MD5,
            upstream_reported_dataset_license: UPSTREAM_REPORTED_LICENSE,
        },
        artifact: ArtifactIdentity {
            filename: cli
                .archive
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<non-utf8-filename>")
                .to_owned(),
            byte_size_before_inspection: hashes_before.byte_size,
            measured_md5_before_inspection: hashes_before.md5,
            upstream_md5_matches,
            measured_sha256_before_inspection: hashes_before.sha256,
            byte_size_after_inspection: hashes_after.byte_size,
            measured_md5_after_inspection: hashes_after.md5,
            measured_sha256_after_inspection: hashes_after.sha256,
            contents_unchanged_during_inspection: contents_unchanged,
        },
        archive_inventory,
        archive_inspection_error,
        limitations: vec![
            "This tool performs no network access and does not download the artifact.",
            "This tool never extracts archive members and never executes archive content.",
            "A matching checksum and safe inventory do not establish dataset schema, split correctness, license sufficiency, scientific validity, or benchmark performance.",
            "The receipt status remains pending semantic schema/split review even when local artifact integrity checks pass.",
            "Upstream MD5 is checked for compatibility only; measured SHA-256 and byte size identify the local artifact more strongly.",
            "The same opened file handle is hashed before and after inventory; content changes detected by those reads block a clean integrity status.",
            "Compressed bytes, decompressed bytes, tar member counts, and member-path metadata are bounded to reduce resource-exhaustion risk from malformed archives. Archives failing the upstream MD5 compatibility check are not parsed; matching MD5 is not proof of authenticity.",
        ],
    };

    let output = serde_json::to_vec_pretty(&receipt).context("could not serialize receipt")?;
    write_receipt_new(&cli.output, &output)?;

    println!("status={}", receipt.status);
    println!("archive_bytes={}", receipt.artifact.byte_size_before_inspection);
    println!("upstream_md5_matches={}", receipt.artifact.upstream_md5_matches);
    println!("sha256={}", receipt.artifact.measured_sha256_before_inspection);
    println!("receipt={}", cli.output.display());

    match receipt.status {
        "artifact_integrity_verified_pending_schema" => Ok(ExitCode::SUCCESS),
        "checksum_mismatch" => Ok(ExitCode::from(2)),
        "archive_structure_review_required" => Ok(ExitCode::from(3)),
        "archive_inspection_failed" => Ok(ExitCode::from(4)),
        "artifact_changed_during_inspection" => Ok(ExitCode::from(5)),
        _ => {
            bail!("internal error: unexpected status {}", receipt.status)
        }
    }
}

fn open_regular_input(input: &Path) -> Result<File> {
    #[cfg(unix)]
    let file = {
        use std::os::unix::fs::OpenOptionsExt;
        OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(input)
            .with_context(|| format!("could not open archive {}", input.display()))?
    };
    #[cfg(not(unix))]
    let file = File::open(input)
        .with_context(|| format!("could not open archive {}", input.display()))?;

    let metadata = file
        .metadata()
        .with_context(|| format!("could not inspect opened archive {}", input.display()))?;
    if !metadata.is_file() {
        bail!("archive input must be a regular file: {}", input.display());
    }
    Ok(file)
}

fn reject_same_input_and_output(input: &Path, output: &Path) -> Result<()> {
    if input == output {
        bail!("input archive and receipt output cannot be the same path");
    }

    // Refuse every pre-existing destination, not just paths that canonicalize to
    // the same string. Distinct hard-link paths can identify the same file, and
    // canonicalize alone does not detect that alias. create_new in the writer
    // below also closes the check/write race.
    match fs::symlink_metadata(output) {
        Ok(_) => bail!(
            "receipt output already exists; refusing to overwrite it: {}",
            output.display()
        ),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("could not inspect receipt output {}", output.display()));
        }
    }
    Ok(())
}

fn write_receipt_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| {
            format!(
                "could not create receipt {} without overwriting an existing path",
                path.display()
            )
        })?;
    output
        .write_all(bytes)
        .with_context(|| format!("could not write receipt {}", path.display()))?;
    output
        .sync_all()
        .with_context(|| format!("could not sync receipt {}", path.display()))?;
    Ok(())
}

fn hash_open_file(file: &mut File) -> Result<HashRecord> {
    file.seek(SeekFrom::Start(0))
        .context("could not rewind archive before hashing")?;
    let mut reader = io::BufReader::new(&mut *file);
    let mut md5 = Md5::new();
    let mut sha256 = Sha256::new();
    let mut byte_size = 0_u64;
    let mut buffer = [0_u8; 1024 * 1024];

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        byte_size = next_compressed_byte_total(byte_size, read)?;
        md5.update(&buffer[..read]);
        sha256.update(&buffer[..read]);
    }

    Ok(HashRecord {
        byte_size,
        md5: lower_hex(&md5.finalize()),
        sha256: lower_hex(&sha256.finalize()),
    })
}

fn next_compressed_byte_total(current: u64, chunk_len: usize) -> Result<u64> {
    let next = current
        .checked_add(chunk_len as u64)
        .ok_or_else(|| anyhow!("archive byte size overflow"))?;
    if next > MAX_COMPRESSED_ARCHIVE_BYTES {
        bail!("compressed archive exceeds configured byte limit of {MAX_COMPRESSED_ARCHIVE_BYTES}");
    }
    Ok(next)
}

fn lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn validate_gzip_stream_open(file: &mut File) -> Result<()> {
    file.seek(SeekFrom::Start(0))
        .context("could not rewind archive before gzip validation")?;
    let mut decoder = MultiGzDecoder::new(&mut *file);
    // Drain the complete compressed stream so gzip CRC/length trailers are verified,
    // even if the tar reader below stops at tar's end-of-archive marker first. Bound
    // decompressed bytes to avoid an unbounded gzip-bomb scan.
    drain_with_limit(&mut decoder, MAX_UNCOMPRESSED_ARCHIVE_BYTES)
        .context("gzip stream failed full integrity read or exceeded the decompressed-byte budget")?;
    Ok(())
}

fn drain_with_limit<R: Read>(reader: &mut R, max_bytes: u64) -> Result<u64> {
    let mut consumed = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            return Ok(consumed);
        }
        let next = consumed
            .checked_add(read as u64)
            .ok_or_else(|| anyhow!("decompressed byte count overflow"))?;
        if next > max_bytes {
            bail!("decompressed data exceeds configured limit of {max_bytes} bytes");
        }
        consumed = next;
    }
}

fn inventory_if_checksum_matches(
    file: &mut File,
    upstream_md5_matches: bool,
) -> (Option<ArchiveInventory>, Option<String>) {
    // Do not parse an archive that fails the upstream-published MD5 compatibility
    // check. A matching MD5 is not a cryptographic authenticity proof; archive
    // parsing remains bounded and no member is extracted or executed.
    if !upstream_md5_matches {
        return (
            None,
            Some("Archive inspection skipped because the upstream-published MD5 did not match.".to_owned()),
        );
    }
    match inventory_tar_gz_open(file) {
        Ok(inventory) => (Some(inventory), None),
        Err(error) => (None, Some(format!("{error:#}"))),
    }
}

fn inventory_tar_gz(path: &Path) -> Result<ArchiveInventory> {
    let mut file = File::open(path)?;
    inventory_tar_gz_open(&mut file)
}

fn inventory_tar_gz_open(file: &mut File) -> Result<ArchiveInventory> {
    validate_gzip_stream_open(file)?;
    file.seek(SeekFrom::Start(0))
        .context("could not rewind archive before tar inventory")?;
    let decoder = MultiGzDecoder::new(&mut *file);
    let mut archive = Archive::new(decoder);
    let entries = archive.entries().context("not a readable tar archive inside gzip stream")?;
    let mut inventory = ArchiveInventory::default();
    let mut seen_paths = BTreeSet::new();

    for (index, result) in entries.enumerate() {
        if inventory.member_count >= MAX_TAR_MEMBERS {
            bail!("archive exceeds configured member-count limit of {MAX_TAR_MEMBERS}");
        }
        let mut entry = result.with_context(|| format!("invalid tar entry at index {index}"))?;
        // Keep the tar crate's Cow<Path> borrowed where possible; do not force
        // an allocation before checking an attacker-controlled member path length.
        // Convert only bounded diagnostic/index values to owned strings, then drop
        // the path borrow before reading the member body mutably.
        let (display_path, path_is_unsafe, normalized_path, top_level) = {
            let entry_path = entry.path()?;
            let path_text = entry_path.to_str();
            inventory.member_path_bytes_inspected = next_member_path_byte_total(
                inventory.member_path_bytes_inspected,
                member_path_metadata_len(&entry_path),
            )?;
            let display_path = path_text
                .map(bounded_member_display)
                .unwrap_or_else(|| format!("<non-utf8-path-{index}>"));
            let path_is_unsafe = path_is_unsafe(&entry_path, path_text)
                || path_text.is_some_and(|text| text.len() > MAX_MEMBER_PATH_BYTES);
            let normalized_path =
                path_text.filter(|text| text.len() <= MAX_MEMBER_PATH_BYTES)
                    .and_then(normalized_member_path);
            let top_level = path_text
                .map(bounded_top_level_component)
                .unwrap_or("<non-utf8>")
                .to_owned();
            (display_path, path_is_unsafe, normalized_path, top_level)
        };

        if path_is_unsafe {
            inventory.unsafe_member_paths.push(display_path.clone());
        }
        if let Some(normalized_path) = normalized_path {
            if !seen_paths.insert(normalized_path.clone()) {
                inventory.duplicate_member_paths.push(normalized_path);
            }
        }

        *inventory
            .top_level_member_counts
            .entry(top_level)
            .or_insert(0) += 1;

        let entry_type = entry.header().entry_type();
        if entry_type.is_file() {
            inventory.regular_file_count += 1;
        } else if entry_type.is_dir() {
            inventory.directory_count += 1;
        } else {
            inventory
                .special_member_entries
                .push(format!("{display_path} (tar type 0x{:02x})", entry_type.as_byte()));
        }

        let declared_size = entry.header().size()?;
        let previous_payload_bytes = inventory.declared_payload_bytes;
        inventory.declared_payload_bytes = inventory
            .declared_payload_bytes
            .checked_add(declared_size)
            .ok_or_else(|| anyhow!("declared archive payload size overflow"))?;
        if inventory.declared_payload_bytes > MAX_UNCOMPRESSED_ARCHIVE_BYTES {
            bail!("declared archive payload exceeds configured byte limit of {MAX_UNCOMPRESSED_ARCHIVE_BYTES}");
        }
        inventory.member_count += 1;

        // Read every member body through the gzip/tar stack to force decompression
        // and detect truncated member payloads. Data is discarded; nothing is extracted.
        let remaining_budget = MAX_UNCOMPRESSED_ARCHIVE_BYTES - previous_payload_bytes;
        let copied = drain_with_limit(&mut entry, remaining_budget)
            .with_context(|| format!("could not stream tar member {display_path}"))?;
        if copied != declared_size {
            bail!(
                "tar member {display_path} yielded {copied} bytes, header declared {declared_size}"
            );
        }
    }

    inventory.duplicate_member_paths.sort();
    inventory.unsafe_member_paths.sort();
    inventory.special_member_entries.sort();
    inventory.payload_bytes_fully_streamed = true;
    Ok(inventory)
}

fn path_is_unsafe(path: &Path, text: Option<&str>) -> bool {
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
    {
        return true;
    }

    let Some(text) = text else {
        return true;
    };
    let normalized = text.replace('\\', "/");
    let has_windows_drive_prefix = normalized.as_bytes().get(1) == Some(&b':')
        && normalized
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphabetic);

    normalized.starts_with('/')
        || has_windows_drive_prefix
        || normalized.split('/').any(|component| component == "..")
}

fn member_path_metadata_len(path: &Path) -> usize {
    path.as_os_str().as_encoded_bytes().len()
}

fn next_member_path_byte_total(current: u64, path_len: usize) -> Result<u64> {
    let next = current
        .checked_add(path_len as u64)
        .ok_or_else(|| anyhow!("member path byte count overflow"))?;
    if next > MAX_TOTAL_MEMBER_PATH_BYTES {
        bail!("member path metadata exceeds configured byte limit of {MAX_TOTAL_MEMBER_PATH_BYTES}");
    }
    Ok(next)
}

fn bounded_top_level_component(path: &str) -> &str {
    let component = top_level_component(path);
    if component.len() > MAX_MEMBER_PATH_BYTES {
        "<overlong>"
    } else {
        component
    }
}

fn bounded_member_display(path: &str) -> String {
    let mut chars = path.chars();
    let prefix: String = chars.by_ref().take(200).collect();
    if chars.next().is_some() {
        format!("{prefix}…<truncated>")
    } else {
        prefix
    }
}

fn normalized_member_path(path: &str) -> Option<String> {
    let normalized = path.replace('\\', "/");
    let has_windows_drive_prefix = normalized.as_bytes().get(1) == Some(&b':')
        && normalized
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphabetic);
    if normalized.starts_with('/') || has_windows_drive_prefix {
        return None;
    }

    let mut components = Vec::new();
    for component in normalized.split('/') {
        match component {
            "" | "." => {}
            ".." => return None,
            other => components.push(other),
        }
    }
    if components.is_empty() {
        Some(".".to_owned())
    } else {
        Some(components.join("/"))
    }
}

fn top_level_component(path: &str) -> &str {
    path.split(['/', '\\'])
        .find(|component| !component.is_empty() && *component != ".")
        .unwrap_or("<root>")
}

fn status_for(
    md5_matches: bool,
    inventory: Option<&ArchiveInventory>,
    contents_unchanged: bool,
) -> &'static str {
    if !contents_unchanged {
        return "artifact_changed_during_inspection";
    }
    if !md5_matches {
        return "checksum_mismatch";
    }
    let Some(inventory) = inventory else {
        return "archive_inspection_failed";
    };
    if inventory.requires_review() {
        return "archive_structure_review_required";
    }
    "artifact_integrity_verified_pending_schema"
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Cursor;
    use tar::{Builder, Header};

    fn write_fixture(path: &Path, members: &[(&str, &[u8])]) {
        let file = File::create(path).expect("create fixture file");
        let encoder = GzEncoder::new(file, Compression::default());
        let mut builder = Builder::new(encoder);

        for (name, bytes) in members {
            let mut header = Header::new_gnu();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_uid(0);
            header.set_gid(0);
            header.set_mtime(0);
            header.set_cksum();
            builder
                .append_data(&mut header, *name, Cursor::new(*bytes))
                .expect("append fixture member");
        }

        builder
            .into_inner()
            .expect("finish tar")
            .finish()
            .expect("finish gzip");
    }

    #[test]
    fn directory_is_rejected_as_an_archive_input() {
        let directory = tempfile::tempdir().expect("temporary directory");
        assert!(open_regular_input(directory.path()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn fifo_input_is_rejected_without_blocking() {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let directory = tempfile::tempdir().expect("temporary directory");
        let fifo_path = directory.path().join("input.fifo");
        let c_path = CString::new(fifo_path.as_os_str().as_bytes())
            .expect("temporary path contains no NUL");
        // SAFETY: c_path is a valid NUL-terminated path; the temporary directory
        // owns the FIFO and is removed when this test exits.
        let result = unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) };
        assert_eq!(result, 0, "mkfifo should create the fixture");
        assert!(open_regular_input(&fifo_path).is_err());
    }

    #[test]
    fn existing_hard_link_output_is_refused_without_modifying_the_archive() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let archive_path = directory.path().join("archive.tar.gz");
        let output_path = directory.path().join("receipt.json");
        fs::write(&archive_path, b"preserve these archive bytes").expect("write archive fixture");
        fs::hard_link(&archive_path, &output_path).expect("create output hard link");

        assert!(reject_same_input_and_output(&archive_path, &output_path).is_err());
        assert!(write_receipt_new(&output_path, b"{\"status\":\"unsafe\"}").is_err());
        assert_eq!(
            fs::read(&archive_path).expect("read archive after refusal"),
            b"preserve these archive bytes"
        );
    }

    #[test]
    fn hash_implementation_matches_published_test_vectors() {
        assert_eq!(lower_hex(&Md5::digest(b"abc")), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(
            lower_hex(&Sha256::digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn inventories_members_and_streams_payloads_without_extracting() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let archive_path = directory.path().join("fixture.tar.gz");
        write_fixture(
            &archive_path,
            &[("train/a.txt", b"hello"), ("valid/b.txt", b"world")],
        );

        let inventory = inventory_tar_gz(&archive_path).expect("inventory archive");
        assert_eq!(inventory.member_count, 2);
        assert_eq!(inventory.regular_file_count, 2);
        assert_eq!(inventory.declared_payload_bytes, 10);
        assert_eq!(inventory.top_level_member_counts.get("train"), Some(&1));
        assert_eq!(inventory.top_level_member_counts.get("valid"), Some(&1));
        assert!(inventory.payload_bytes_fully_streamed);
        assert!(!inventory.requires_review());
    }

    #[test]
    fn traversal_path_requires_review() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let archive_path = directory.path().join("traversal.tar.gz");
        write_fixture(&archive_path, &[("../outside.txt", b"no extraction")]);

        let inventory = inventory_tar_gz(&archive_path).expect("inventory archive");
        assert!(inventory.unsafe_member_paths.iter().any(|path| path.contains("..")));
        assert!(inventory.requires_review());
        assert_eq!(status_for(true, Some(&inventory), true), "archive_structure_review_required");
    }

    #[test]
    fn duplicate_paths_require_review() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let archive_path = directory.path().join("duplicate.tar.gz");
        write_fixture(
            &archive_path,
            &[("train/repeated.txt", b"first"), ("train/repeated.txt", b"second")],
        );

        let inventory = inventory_tar_gz(&archive_path).expect("inventory archive");
        assert_eq!(inventory.duplicate_member_paths, vec!["train/repeated.txt"]);
        assert_eq!(status_for(true, Some(&inventory), true), "archive_structure_review_required");
    }

    #[test]
    fn compressed_archive_size_budget_is_enforced() {
        assert_eq!(
            next_compressed_byte_total(MAX_COMPRESSED_ARCHIVE_BYTES - 8, 8)
                .expect("at limit"),
            MAX_COMPRESSED_ARCHIVE_BYTES
        );
        assert!(next_compressed_byte_total(MAX_COMPRESSED_ARCHIVE_BYTES, 1).is_err());
    }

    #[test]
    fn cumulative_member_path_metadata_is_bounded() {
        assert_eq!(next_member_path_byte_total(10, 20).expect("within path budget"), 30);
        assert!(next_member_path_byte_total(MAX_TOTAL_MEMBER_PATH_BYTES, 1).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_member_path_bytes_count_toward_cumulative_budget() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let path = PathBuf::from(OsString::from_vec(vec![b'x', 0xff, b'y']));
        assert_eq!(member_path_metadata_len(&path), 3);
        assert_eq!(
            next_member_path_byte_total(0, member_path_metadata_len(&path))
                .expect("count non-UTF-8 path bytes"),
            3
        );
    }

    #[test]
    fn overlong_top_level_component_is_not_copied_into_inventory_key() {
        let long_component = "x".repeat(MAX_MEMBER_PATH_BYTES + 1);
        assert_eq!(bounded_top_level_component(&long_component), "<overlong>");
    }

    #[test]
    fn long_path_diagnostic_is_bounded() {
        let long_path = "x".repeat(10_000);
        let display = bounded_member_display(&long_path);
        assert!(display.len() < 300);
        assert!(display.ends_with("<truncated>"));
    }

    #[test]
    fn path_aliases_are_detected_as_duplicate_members() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let archive_path = directory.path().join("aliases.tar.gz");
        write_fixture(
            &archive_path,
            &[("train/./a.txt", b"first"), ("train/a.txt", b"second")],
        );

        let inventory = inventory_tar_gz(&archive_path).expect("inventory archive");
        assert_eq!(inventory.duplicate_member_paths, vec!["train/a.txt"]);
        assert!(inventory.requires_review());
    }

    #[test]
    fn checksum_mismatch_never_promotes_to_admitted() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let archive_path = directory.path().join("fixture.tar.gz");
        write_fixture(&archive_path, &[("train/a.txt", b"hello")]);

        let inventory = inventory_tar_gz(&archive_path).expect("inventory archive");
        assert_eq!(status_for(false, Some(&inventory), true), "checksum_mismatch");
    }

    #[test]
    fn checksum_mismatch_skips_parsing_untrusted_archive() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let archive_path = directory.path().join("not-an-archive.tar.gz");
        fs::write(&archive_path, b"attacker-controlled bytes").expect("write invalid fixture");

        let mut archive_file = File::open(&archive_path).expect("open invalid fixture");
        let (inventory, error) = inventory_if_checksum_matches(&mut archive_file, false);
        assert!(inventory.is_none());
        assert!(error.expect("skip reason").contains("inspection skipped"));
        assert_eq!(status_for(false, None, true), "checksum_mismatch");
    }

    #[test]
    fn checksum_mismatch_is_reported_even_when_archive_inspection_fails() {
        assert_eq!(status_for(false, None, true), "checksum_mismatch");
    }

    #[test]
    fn valid_checksum_still_stops_before_schema_admission() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let archive_path = directory.path().join("fixture.tar.gz");
        write_fixture(&archive_path, &[("train/a.txt", b"hello")]);

        let inventory = inventory_tar_gz(&archive_path).expect("inventory archive");
        assert_eq!(
            status_for(true, Some(&inventory), true),
            "artifact_integrity_verified_pending_schema"
        );
        assert_ne!(status_for(true, Some(&inventory), true), "admitted");
    }

    #[test]
    fn decompressed_budget_is_enforced() {
        let mut too_large = Cursor::new(b"123456789");
        assert!(drain_with_limit(&mut too_large, 8).is_err());

        let mut within_limit = Cursor::new(b"12345678");
        assert_eq!(drain_with_limit(&mut within_limit, 8).expect("within budget"), 8);
    }

    #[test]
    fn changed_artifact_during_inspection_blocks_success() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let archive_path = directory.path().join("fixture.tar.gz");
        write_fixture(&archive_path, &[("train/a.txt", b"hello")]);

        let inventory = inventory_tar_gz(&archive_path).expect("inventory");
        assert_eq!(status_for(true, Some(&inventory), false), "artifact_changed_during_inspection");
    }

    #[test]
    fn gzip_footer_corruption_is_detected_by_full_stream_validation() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let archive_path = directory.path().join("corrupt-footer.tar.gz");
        write_fixture(&archive_path, &[("train/a.txt", b"payload for gzip trailer")]);

        let mut bytes = fs::read(&archive_path).expect("read fixture archive");
        assert!(bytes.len() >= 8, "gzip stream has a trailer");
        let crc_offset = bytes.len() - 8;
        bytes[crc_offset] ^= 0x01;
        fs::write(&archive_path, bytes).expect("write corrupted fixture");

        assert!(inventory_tar_gz(&archive_path).is_err());
    }

    #[test]
    fn invalid_gzip_or_tar_is_reported_as_inspection_failure() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let archive_path = directory.path().join("not-an-archive.tar.gz");
        fs::write(&archive_path, b"not a gzip archive").expect("write invalid fixture");

        assert!(inventory_tar_gz(&archive_path).is_err());
        assert_eq!(status_for(true, None, true), "archive_inspection_failed");
    }
}
