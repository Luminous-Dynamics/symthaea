//! HARDSEC-003 raw-corpus inventory.
//!
//! Creates a deterministic SHA-256 inventory of an already downloaded dataset
//! tree before any CSV parsing or sharding assumptions are made.
//!
//! cargo run --example hardsec_corpus_inventory --features hardsec-audit -- \
//!   --root /path/to/downloaded-dataset \
//!   --output /path/outside/dataset/corpus-inventory.json \
//!   --dataset-doi 10.34810/DATA2538 \
//!   --dataset-version 2.0 \
//!   --source-uri https://example.invalid/pinned-download \
//!   --stated-license "CC0 1.0 (catalogue statement; not independently audited)"
//!
//! This records the caller-supplied citation metadata and byte identities. It
//! does not download the source, interpret the licence, validate task labels,
//! prove corpus completeness, or guarantee an atomic filesystem snapshot.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

const SCHEMA_VERSION: &str = "hardsec-corpus-inventory-v1";
const CLAIM_SCOPE: &str = "raw_file_inventory_only";

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    root: PathBuf,
    output: PathBuf,
    dataset_doi: String,
    dataset_version: String,
    source_uri: String,
    stated_license: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct FileRecord {
    /// Relative path with '/' separators, independent of the host path style.
    path: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct Inventory {
    schema_version: &'static str,
    result: &'static str,
    claim_scope: &'static str,
    dataset_doi: String,
    dataset_version: String,
    source_uri: String,
    stated_license: String,
    limitations: Vec<&'static str>,
    file_count: usize,
    total_bytes: u64,
    files: Vec<FileRecord>,
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(Some(args)) => match run_inventory(&args) {
            Ok((inventory, receipt_path, receipt_sha256)) => {
                println!(
                    "HARDSEC_CORPUS_INVENTORY_OK dataset_doi={} version={} files={} total_bytes={} receipt_sha256={} receipt_path={}",
                    inventory.dataset_doi,
                    inventory.dataset_version,
                    inventory.file_count,
                    inventory.total_bytes,
                    receipt_sha256,
                    receipt_path.display()
                );
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("HARDSEC_CORPUS_INVENTORY_ERROR: {error}");
                ExitCode::from(2)
            }
        },
        Ok(None) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("HARDSEC_CORPUS_INVENTORY_ERROR: {error}");
            ExitCode::from(2)
        }
    }
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut args = env::args().skip(1);
    let mut root = None;
    let mut output = None;
    let mut dataset_doi = None;
    let mut dataset_version = None;
    let mut source_uri = None;
    let mut stated_license = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!(
                    r#"HARDSEC-003 raw-corpus inventory

Usage:
  hardsec_corpus_inventory --root DIR --output NEW_JSON \
    --dataset-doi DOI --dataset-version VERSION \
    --source-uri URI --stated-license TEXT

The output file must be outside the dataset root and must not already exist.
The inventory records exact file paths, sizes and SHA-256 hashes; it does not
interpret the dataset schema, confirm license coverage, or certify security."#
                );
                return Ok(None);
            }
            "--root" => {
                root = Some(PathBuf::from(args.next().ok_or("--root requires a path")?));
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.next().ok_or("--output requires a new path")?,
                ));
            }
            "--dataset-doi" => {
                dataset_doi = Some(
                    args.next()
                        .ok_or("--dataset-doi requires a value")?,
                );
            }
            "--dataset-version" => {
                dataset_version = Some(
                    args.next()
                        .ok_or("--dataset-version requires a value")?,
                );
            }
            "--source-uri" => {
                source_uri = Some(args.next().ok_or("--source-uri requires a value")?);
            }
            "--stated-license" => {
                stated_license = Some(
                    args.next()
                        .ok_or("--stated-license requires a value")?,
                );
            }
            other => return Err(format!("unknown argument {other:?}; use --help")),
        }
    }

    let required = |name: &str, value: Option<String>| -> Result<String, String> {
        let value = value.ok_or_else(|| format!("{name} is required"))?;
        if value.trim().is_empty() {
            return Err(format!("{name} must not be empty"));
        }
        Ok(value)
    };

    Ok(Some(Args {
        root: root.ok_or("--root is required")?,
        output: output.ok_or("--output is required")?,
        dataset_doi: required("--dataset-doi", dataset_doi)?,
        dataset_version: required("--dataset-version", dataset_version)?,
        source_uri: required("--source-uri", source_uri)?,
        stated_license: required("--stated-license", stated_license)?,
    }))
}

fn run_inventory(args: &Args) -> Result<(Inventory, PathBuf, String), String> {
    let root = fs::canonicalize(&args.root)
        .map_err(|error| format!("cannot resolve dataset root {}: {error}", args.root.display()))?;
    let root_metadata = fs::metadata(&root)
        .map_err(|error| format!("cannot inspect dataset root {}: {error}", root.display()))?;
    if !root_metadata.is_dir() {
        return Err(format!("dataset root is not a directory: {}", root.display()));
    }

    let receipt_path = resolve_new_output_path(&root, &args.output)?;
    let mut files = Vec::new();
    visit_directory(&root, &root, &mut files)?;
    files.sort_by(|left, right| left.path.cmp(&right.path));
    if files.is_empty() {
        return Err("dataset root contains no regular files".into());
    }

    let total_bytes = files.iter().try_fold(0u64, |total, file| {
        total.checked_add(file.size_bytes)
            .ok_or_else(|| "total corpus size overflowed u64".to_string())
    })?;

    let inventory = Inventory {
        schema_version: SCHEMA_VERSION,
        result: "inventory_created",
        claim_scope: CLAIM_SCOPE,
        dataset_doi: args.dataset_doi.clone(),
        dataset_version: args.dataset_version.clone(),
        source_uri: args.source_uri.clone(),
        stated_license: args.stated_license.clone(),
        limitations: vec![
            "source_uri, DOI, version and stated_license are caller-supplied metadata",
            "file hashes prove byte identity only, not source authenticity or factual labels",
            "this tool does not establish license coverage for embedded source code or binaries",
            "this tool does not parse schemas, validate completeness, or certify security",
            "the inventory is not an atomic snapshot if the source tree changes concurrently",
        ],
        file_count: files.len(),
        total_bytes,
        files,
    };

    write_new_receipt(&inventory, &receipt_path)?;
    let receipt_sha256 = sha256_file(&receipt_path)
        .map_err(|error| format!("cannot hash written inventory receipt: {error}"))?;
    Ok((inventory, receipt_path, receipt_sha256))
}

fn resolve_new_output_path(root: &Path, requested: &Path) -> Result<PathBuf, String> {
    let file_name = requested
        .file_name()
        .ok_or("output must name a file, not only a directory")?;
    if !matches!(file_name, name if !name.is_empty()) {
        return Err("output filename must not be empty".into());
    }

    let parent_requested = requested.parent().filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = fs::canonicalize(parent_requested).map_err(|error| {
        format!(
            "cannot resolve output parent {}: {error}",
            parent_requested.display()
        )
    })?;
    let target = parent.join(file_name);

    if target.starts_with(root) {
        return Err("output must be outside the dataset root to avoid self-inventory".into());
    }
    if fs::symlink_metadata(&target).is_ok() {
        return Err(format!(
            "output already exists; refusing to overwrite {}",
            target.display()
        ));
    }
    Ok(target)
}

fn visit_directory(root: &Path, directory: &Path, output: &mut Vec<FileRecord>) -> Result<(), String> {
    let directory_metadata = fs::symlink_metadata(directory).map_err(|error| {
        format!("cannot inspect directory {}: {error}", directory.display())
    })?;
    if directory_metadata.file_type().is_symlink() || !directory_metadata.is_dir() {
        return Err(format!(
            "source directory is a symlink or not a directory: {}",
            directory.display()
        ));
    }

    let mut entries = fs::read_dir(directory)
        .map_err(|error| format!("cannot read directory {}: {error}", directory.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("cannot enumerate {}: {error}", directory.display()))?;
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        let absolute = entry.path();
        let metadata = fs::symlink_metadata(&absolute)
            .map_err(|error| format!("cannot inspect {}: {error}", absolute.display()))?;
        if metadata.file_type().is_symlink() {
            return Err(format!("symlink is not allowed in dataset tree: {}", absolute.display()));
        }
        if metadata.is_dir() {
            visit_directory(root, &absolute, output)?;
            continue;
        }
        if !metadata.is_file() {
            return Err(format!(
                "unsupported filesystem object in dataset tree: {}",
                absolute.display()
            ));
        }

        let relative = absolute.strip_prefix(root).map_err(|error| {
            format!("cannot make {} relative to {}: {error}", absolute.display(), root.display())
        })?;
        if relative.components().any(|component| !matches!(component, Component::Normal(_))) {
            return Err(format!("unsafe relative path {}", relative.display()));
        }
        let size_before = metadata.len();
        let modified_before = metadata.modified().ok();
        let digest = sha256_file(&absolute)
            .map_err(|error| format!("cannot hash {}: {error}", absolute.display()))?;
        let after = fs::symlink_metadata(&absolute)
            .map_err(|error| format!("cannot re-inspect {}: {error}", absolute.display()))?;
        if after.file_type().is_symlink() || !after.is_file() {
            return Err(format!("source changed type during inventory: {}", absolute.display()));
        }
        if after.len() != size_before
            || (modified_before.is_some() && after.modified().ok() != modified_before)
        {
            return Err(format!("source changed while hashing: {}", absolute.display()));
        }

        let normalized_path = normalized_relative_path(relative)?;
        output.push(FileRecord {
            path: normalized_path,
            size_bytes: size_before,
            sha256: digest,
        });
    }
    Ok(())
}

fn normalized_relative_path(relative: &Path) -> Result<String, String> {
    let mut parts = Vec::new();
    for component in relative.components() {
        let Component::Normal(part) = component else {
            return Err(format!("unsafe relative path {}", relative.display()));
        };
        let part = part.to_str().ok_or_else(|| {
            format!(
                "non-UTF-8 path component is unsupported for deterministic inventory: {}",
                relative.display()
            )
        })?;
        parts.push(part);
    }
    if parts.is_empty() {
        return Err("empty relative path is not a file".into());
    }
    Ok(parts.join("/"))
}

fn sha256_file(path: &Path) -> Result<String, std::io::Error> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex::encode(digest.finalize()))
}

fn write_new_receipt(inventory: &Inventory, path: &Path) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(inventory)
        .map_err(|error| format!("cannot serialize inventory: {error}"))?;
    bytes.push(b'\n');

    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            format!(
                "cannot create new inventory receipt {}: {error}",
                path.display()
            )
        })?;
    if let Err(error) = file.write_all(&bytes) {
        drop(file);
        let _ = fs::remove_file(path);
        return Err(format!(
            "cannot write inventory receipt {}: {error}; removed incomplete output where possible",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        root: PathBuf,
        output_dir: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock before Unix epoch")
                .as_nanos();
            let base = env::temp_dir().join(format!(
                "hardsec-inventory-{}-{}-{}",
                std::process::id(),
                nonce,
                NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
            ));
            let root = base.join("dataset");
            let output_dir = base.join("receipts");
            fs::create_dir_all(&root).expect("create dataset root");
            fs::create_dir_all(&output_dir).expect("create output dir");
            fs::write(root.join("b.txt"), b"beta").expect("write b");
            fs::create_dir_all(root.join("nested")).expect("create nested dir");
            fs::write(root.join("nested").join("a.txt"), b"alpha").expect("write a");
            Self { root, output_dir }
        }

        fn args(&self, output_name: &str) -> Args {
            Args {
                root: self.root.clone(),
                output: self.output_dir.join(output_name),
                dataset_doi: "10.34810/DATA2538".into(),
                dataset_version: "2.0".into(),
                source_uri: "https://example.invalid/pinned-file".into(),
                stated_license: "CC0 1.0 (stated; not audited)".into(),
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            if let Some(parent) = self.root.parent() {
                let _ = fs::remove_dir_all(parent);
            }
        }
    }

    #[test]
    fn inventory_is_sorted_and_hashes_exact_bytes() {
        let fixture = Fixture::new();
        let (inventory, _, _) = run_inventory(&fixture.args("inventory.json"))
            .expect("valid dataset tree should inventory");
        assert_eq!(inventory.files.len(), 2);
        assert_eq!(inventory.files[0].path, "b.txt");
        assert_eq!(inventory.files[1].path, "nested/a.txt");
        assert_eq!(
            inventory.files[0].sha256,
            "f44e64e75f3948e9f73f8dfa94721c4ce8cbb4f265c4790c702b2d41cfbf2753"
        );
        assert_eq!(inventory.total_bytes, 9);
    }

    #[test]
    fn repeated_inventories_of_unchanged_tree_are_byte_deterministic() {
        let fixture = Fixture::new();
        let (first, _, _) =
            run_inventory(&fixture.args("first.json")).expect("first inventory succeeds");
        let (second, _, _) =
            run_inventory(&fixture.args("second.json")).expect("second inventory succeeds");
        assert_eq!(
            serde_json::to_vec_pretty(&first).unwrap(),
            serde_json::to_vec_pretty(&second).unwrap()
        );
    }

    #[test]
    fn output_inside_dataset_root_is_rejected() {
        let fixture = Fixture::new();
        let args = Args {
            output: fixture.root.join("inventory.json"),
            ..fixture.args("unused.json")
        };
        assert!(run_inventory(&args)
            .unwrap_err()
            .contains("outside the dataset root"));
    }

    #[test]
    fn preexisting_output_is_never_overwritten() {
        let fixture = Fixture::new();
        let output = fixture.output_dir.join("existing.json");
        fs::write(&output, b"keep-me").unwrap();
        assert!(write_new_receipt(
            &Inventory {
                schema_version: SCHEMA_VERSION,
                result: "inventory_created",
                claim_scope: CLAIM_SCOPE,
                dataset_doi: "10.34810/DATA2538".into(),
                dataset_version: "2.0".into(),
                source_uri: "fixture".into(),
                stated_license: "fixture".into(),
                limitations: vec!["fixture"],
                file_count: 0,
                total_bytes: 0,
                files: Vec::new(),
            },
            &output
        )
        .unwrap_err()
        .contains("cannot create new inventory receipt"));
        assert_eq!(fs::read(&output).unwrap(), b"keep-me");
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_in_input_tree_are_rejected() {
        let fixture = Fixture::new();
        let target = fixture.root.join("b.txt");
        std::os::unix::fs::symlink(&target, fixture.root.join("alias.txt")).unwrap();
        assert!(run_inventory(&fixture.args("inventory.json"))
            .unwrap_err()
            .contains("symlink is not allowed"));
    }

    #[test]
    fn empty_input_tree_is_rejected() {
        let fixture = Fixture::new();
        fs::remove_file(fixture.root.join("b.txt")).unwrap();
        fs::remove_file(fixture.root.join("nested").join("a.txt")).unwrap();
        assert!(run_inventory(&fixture.args("inventory.json"))
            .unwrap_err()
            .contains("contains no regular files"));
    }
}
