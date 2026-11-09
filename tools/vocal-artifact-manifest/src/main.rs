// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Content-addressed manifest generator for model and voicebank artifact trees.
//!
//! The manifest contains normalized relative paths only. Symlinks, special
//! files, non-UTF-8 paths, and files observed to mutate during hashing fail
//! closed. The root digest is independent of directory enumeration order.

use serde::{Deserialize, Serialize};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

const SCHEMA: &str = "symthaea-artifact-tree/v1";
const DOMAIN: &[u8] = b"symthaea-artifact-tree/v1\0";
const BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
struct FileEntry {
    path: String,
    size_bytes: u64,
    blake3: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
struct ArtifactManifest {
    schema: String,
    algorithm: String,
    root_blake3: String,
    total_files: u64,
    total_bytes: u64,
    entries: Vec<FileEntry>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("vocal-artifact-manifest: {error:#}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<OsString> = env::args_os().skip(1).collect();
    match args.as_slice() {
        [flag, root, output] if flag == OsStr::new("--create") => {
            let root = PathBuf::from(root);
            let output = PathBuf::from(output);
            ensure_output_outside_root(&root, &output)?;
            let manifest = build_manifest(&root)?;
            let json = serde_json::to_vec_pretty(&manifest)
                .map_err(|error| format!("serialize manifest: {error}"))?;
            let mut file = File::create(&output)
                .map_err(|error| format!("create {}: {error}", output.display()))?;
            file.write_all(&json)
                .map_err(|error| format!("write {}: {error}", output.display()))?;
            file.write_all(b"\n")
                .map_err(|error| format!("finish {}: {error}", output.display()))?;
            println!("root_blake3={}", manifest.root_blake3);
            println!("files={}", manifest.total_files);
            println!("bytes={}", manifest.total_bytes);
            Ok(())
        }
        [flag, root, manifest_path] if flag == OsStr::new("--verify") => {
            let expected_path = PathBuf::from(manifest_path);
            let expected_bytes = fs::read(&expected_path)
                .map_err(|error| format!("read {}: {error}", expected_path.display()))?;
            let expected: ArtifactManifest = serde_json::from_slice(&expected_bytes)
                .map_err(|error| format!("parse {}: {error}", expected_path.display()))?;
            validate_manifest_shape(&expected)?;
            let actual = build_manifest(Path::new(root))?;
            if actual != expected {
                return Err(describe_manifest_difference(&expected, &actual));
            }
            println!("verified_root_blake3={}", actual.root_blake3);
            println!("verified_files={}", actual.total_files);
            Ok(())
        }
        _ => Err(
            "usage: vocal-artifact-manifest --create <artifact-root> <output.json>\n\
             or: vocal-artifact-manifest --verify <artifact-root> <manifest.json>"
                .to_string(),
        ),
    }
}

fn build_manifest(root: &Path) -> Result<ArtifactManifest, String> {
    let root_metadata = fs::symlink_metadata(root)
        .map_err(|error| format!("inspect root {}: {error}", root.display()))?;
    if root_metadata.file_type().is_symlink() || !root_metadata.is_dir() {
        return Err(format!("artifact root must be a real directory: {}", root.display()));
    }
    let root = root
        .canonicalize()
        .map_err(|error| format!("canonicalize root {}: {error}", root.display()))?;

    let mut entries = Vec::new();
    walk_directory(&root, &root, &mut entries)?;
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    if entries.is_empty() {
        return Err("artifact root contains no files".to_string());
    }

    let total_files = u64::try_from(entries.len())
        .map_err(|_| "artifact file count exceeds u64".to_string())?;
    let total_bytes = entries.iter().try_fold(0u64, |total, entry| {
        total.checked_add(entry.size_bytes)
            .ok_or_else(|| "artifact total byte count overflow".to_string())
    })?;
    let root_blake3 = root_digest(&entries)?;

    Ok(ArtifactManifest {
        schema: SCHEMA.to_string(),
        algorithm: "BLAKE3-256".to_string(),
        root_blake3,
        total_files,
        total_bytes,
        entries,
    })
}

fn walk_directory(
    root: &Path,
    directory: &Path,
    entries: &mut Vec<FileEntry>,
) -> Result<(), String> {
    let mut children = fs::read_dir(directory)
        .map_err(|error| format!("read directory {}: {error}", directory.display()))?
        .map(|result| result.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("enumerate {}: {error}", directory.display()))?;
    children.sort();

    for path in children {
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| format!("inspect {}: {error}", path.display()))?;
        let kind = metadata.file_type();
        if kind.is_symlink() {
            return Err(format!("symlink is not allowed in artifact tree: {}", path.display()));
        }
        if kind.is_dir() {
            walk_directory(root, &path, entries)?;
            continue;
        }
        if !kind.is_file() {
            return Err(format!("special file is not allowed in artifact tree: {}", path.display()));
        }

        let relative = path
            .strip_prefix(root)
            .map_err(|_| format!("path escaped artifact root: {}", path.display()))?;
        let normalized = normalize_relative_path(relative)?;
        let (size_bytes, digest) = hash_stable_file(&path)?;
        entries.push(FileEntry {
            path: normalized,
            size_bytes,
            blake3: digest,
        });
    }
    Ok(())
}

fn normalize_relative_path(path: &Path) -> Result<String, String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => {
                let part = part
                    .to_str()
                    .ok_or_else(|| format!("non-UTF-8 artifact path: {}", path.display()))?;
                if part.is_empty() || part == "." || part == ".." || part.contains('\\') {
                    return Err(format!("non-canonical artifact path: {}", path.display()));
                }
                parts.push(part);
            }
            _ => return Err(format!("non-relative artifact path: {}", path.display())),
        }
    }
    if parts.is_empty() {
        return Err("empty relative artifact path".to_string());
    }
    Ok(parts.join("/"))
}

fn hash_stable_file(path: &Path) -> Result<(u64, String), String> {
    let before = fs::metadata(path)
        .map_err(|error| format!("metadata before hashing {}: {error}", path.display()))?;
    let mut file = File::open(path)
        .map_err(|error| format!("open {}: {error}", path.display()))?;
    let opened = file
        .metadata()
        .map_err(|error| format!("inspect opened file {}: {error}", path.display()))?;
    if !opened.is_file() || opened.len() != before.len() || modified(&opened) != modified(&before) {
        return Err(format!("artifact changed before hashing: {}", path.display()));
    }
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0u8; BUFFER_BYTES];
    let mut bytes_read = 0u64;

    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        bytes_read = bytes_read
            .checked_add(count as u64)
            .ok_or_else(|| format!("file length overflow while hashing {}", path.display()))?;
        hasher.update(&buffer[..count]);
    }

    let after_path = fs::symlink_metadata(path)
        .map_err(|error| format!("metadata after hashing {}: {error}", path.display()))?;
    let after = file
        .metadata()
        .map_err(|error| format!("inspect hashed file {}: {error}", path.display()))?;
    if after_path.file_type().is_symlink()
        || !after_path.is_file()
        || before.len() != after.len()
        || before.len() != after_path.len()
        || before.len() != bytes_read
        || modified(&before) != modified(&after)
        || modified(&before) != modified(&after_path)
    {
        return Err(format!("artifact changed while being hashed: {}", path.display()));
    }
    Ok((bytes_read, hasher.finalize().to_hex().to_string()))
}

fn modified(metadata: &fs::Metadata) -> Option<SystemTime> {
    metadata.modified().ok()
}

fn root_digest(entries: &[FileEntry]) -> Result<String, String> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DOMAIN);
    for entry in entries {
        let path = entry.path.as_bytes();
        let path_len = u64::try_from(path.len())
            .map_err(|_| format!("path too long to hash: {}", entry.path))?;
        let content_digest = decode_hex_32(&entry.blake3)?;
        hasher.update(&path_len.to_le_bytes());
        hasher.update(path);
        hasher.update(&entry.size_bytes.to_le_bytes());
        hasher.update(&content_digest);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn decode_hex_32(hex: &str) -> Result<[u8; 32], String> {
    if hex.len() != 64 {
        return Err(format!("invalid BLAKE3 digest length: {}", hex.len()));
    }
    let mut output = [0u8; 32];
    for (index, byte) in output.iter_mut().enumerate() {
        let start = index * 2;
        *byte = u8::from_str_radix(&hex[start..start + 2], 16)
            .map_err(|_| format!("invalid BLAKE3 hex digest: {hex}"))?;
    }
    Ok(output)
}

fn validate_manifest_shape(manifest: &ArtifactManifest) -> Result<(), String> {
    if manifest.schema != SCHEMA || manifest.algorithm != "BLAKE3-256" {
        return Err("unsupported artifact manifest schema or digest algorithm".to_string());
    }
    if manifest.entries.windows(2).any(|pair| pair[0].path >= pair[1].path) {
        return Err("manifest entries must be strictly sorted and unique".to_string());
    }
    for entry in &manifest.entries {
        if entry.path.starts_with('/')
            || entry.path.contains('\\')
            || entry.path.split('/').any(|part| part.is_empty() || part == "." || part == "..")
        {
            return Err(format!("manifest contains unsafe relative path: {}", entry.path));
        }
        decode_hex_32(&entry.blake3)?;
    }
    let expected_count = u64::try_from(manifest.entries.len())
        .map_err(|_| "manifest entry count exceeds u64".to_string())?;
    let expected_bytes = manifest.entries.iter().try_fold(0u64, |total, entry| {
        total.checked_add(entry.size_bytes)
            .ok_or_else(|| "manifest total byte count overflow".to_string())
    })?;
    if manifest.total_files != expected_count || manifest.total_bytes != expected_bytes {
        return Err("manifest aggregate counts do not match its entries".to_string());
    }
    if root_digest(&manifest.entries)? != manifest.root_blake3 {
        return Err("manifest root digest does not match its entries".to_string());
    }
    Ok(())
}

fn ensure_output_outside_root(root: &Path, output: &Path) -> Result<(), String> {
    let canonical_root = root
        .canonicalize()
        .map_err(|error| format!("canonicalize root {}: {error}", root.display()))?;
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let canonical_parent = parent
        .canonicalize()
        .map_err(|error| format!("canonicalize output parent {}: {error}", parent.display()))?;
    let filename = output
        .file_name()
        .ok_or_else(|| "output path must name a file".to_string())?;
    let candidate = canonical_parent.join(filename);
    if candidate.starts_with(&canonical_root) {
        return Err("manifest output must be outside the artifact root".to_string());
    }
    Ok(())
}

fn describe_manifest_difference(expected: &ArtifactManifest, actual: &ArtifactManifest) -> String {
    let expected_paths: std::collections::BTreeMap<_, _> =
        expected.entries.iter().map(|entry| (&entry.path, entry)).collect();
    let actual_paths: std::collections::BTreeMap<_, _> =
        actual.entries.iter().map(|entry| (&entry.path, entry)).collect();

    let missing: Vec<_> = expected_paths
        .keys()
        .filter(|path| !actual_paths.contains_key(*path))
        .take(8)
        .map(|path| path.as_str())
        .collect();
    let added: Vec<_> = actual_paths
        .keys()
        .filter(|path| !expected_paths.contains_key(*path))
        .take(8)
        .map(|path| path.as_str())
        .collect();
    let changed: Vec<_> = expected_paths
        .iter()
        .filter_map(|(path, expected_entry)| {
            actual_paths.get(*path).filter(|actual_entry| *actual_entry != *expected_entry)
                .map(|_| path.as_str())
        })
        .take(8)
        .collect();

    format!(
        "artifact tree mismatch: expected root {}, got {}; missing={missing:?}; added={added:?}; changed={changed:?}",
        expected.root_blake3, actual.root_blake3
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct TempTree(PathBuf);

    impl TempTree {
        fn new() -> Self {
            let path = env::temp_dir().join(format!(
                "symthaea-vocal-artifact-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).expect("create temp tree");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn digest_is_stable_and_entries_are_sorted() {
        let a = TempTree::new();
        fs::create_dir_all(a.path().join("nested")).unwrap();
        fs::write(a.path().join("z.bin"), b"z").unwrap();
        fs::write(a.path().join("nested/a.bin"), b"a").unwrap();

        let first = build_manifest(a.path()).unwrap();
        let second = build_manifest(a.path()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.entries[0].path, "nested/a.bin");
        assert_eq!(first.entries[1].path, "z.bin");
    }

    #[test]
    fn content_change_changes_root_digest() {
        let tree = TempTree::new();
        let file = tree.path().join("model.onnx");
        fs::write(&file, b"weights-a").unwrap();
        let first = build_manifest(tree.path()).unwrap();
        fs::write(&file, b"weights-b").unwrap();
        let second = build_manifest(tree.path()).unwrap();
        assert_ne!(first.root_blake3, second.root_blake3);
    }

    #[test]
    fn path_addition_and_deletion_change_root_digest() {
        let tree = TempTree::new();
        fs::write(tree.path().join("config.json"), b"{}").unwrap();
        let first = build_manifest(tree.path()).unwrap();
        fs::write(tree.path().join("speaker.bin"), b"speaker").unwrap();
        let second = build_manifest(tree.path()).unwrap();
        assert_ne!(first.root_blake3, second.root_blake3);
        fs::remove_file(tree.path().join("speaker.bin")).unwrap();
        let third = build_manifest(tree.path()).unwrap();
        assert_eq!(first.root_blake3, third.root_blake3);
    }

    #[test]
    fn manifest_verification_detects_tampered_entries() {
        let tree = TempTree::new();
        fs::write(tree.path().join("weights.bin"), b"weights").unwrap();
        let mut manifest = build_manifest(tree.path()).unwrap();
        manifest.entries[0].size_bytes += 1;
        assert!(validate_manifest_shape(&manifest).is_err());
    }

    #[test]
    fn verification_detects_missing_added_and_changed_files() {
        let tree = TempTree::new();
        fs::write(tree.path().join("voice.onnx"), b"weights").unwrap();
        let expected = build_manifest(tree.path()).unwrap();

        fs::write(tree.path().join("voice.onnx"), b"changed").unwrap();
        let error = describe_manifest_difference(&expected, &build_manifest(tree.path()).unwrap());
        assert!(error.contains("changed=[\"voice.onnx\"]"), "{error}");

        fs::remove_file(tree.path().join("voice.onnx")).unwrap();
        fs::write(tree.path().join("new.onnx"), b"new").unwrap();
        let error = describe_manifest_difference(&expected, &build_manifest(tree.path()).unwrap());
        assert!(error.contains("missing=[\"voice.onnx\"]"), "{error}");
        assert!(error.contains("added=[\"new.onnx\"]"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_are_rejected_instead_of_followed() {
        use std::os::unix::fs::symlink;

        let tree = TempTree::new();
        let outside = tree.path().with_extension("outside");
        fs::write(&outside, b"secret").unwrap();
        symlink(&outside, tree.path().join("external-link")).unwrap();
        let error = build_manifest(tree.path()).unwrap_err();
        assert!(error.contains("symlink is not allowed"), "{error}");
        let _ = fs::remove_file(outside);
    }
}
