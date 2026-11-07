// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic content-addressed manifests for voice/model artifact trees.
//!
//! Only regular files and directories are accepted. Symlinks, special files,
//! non-UTF8 paths, duplicate normalized paths, and path traversal are rejected.
//! The root digest covers the schema, artifact kind, sorted relative paths,
//! file sizes, and BLAKE3 content hashes. Absolute provisioning paths are
//! deliberately absent from the manifest.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

const SCHEMA: &str = "voice-artifact-tree/v1";
const BUFFER_SIZE: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ArtifactEntry {
    relative_path: String,
    size_bytes: u64,
    blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ArtifactManifest {
    schema: String,
    artifact_kind: String,
    entry_count: usize,
    entries: Vec<ArtifactEntry>,
    root_blake3: String,
}

#[derive(Serialize)]
struct RootPayload<'a> {
    schema: &'a str,
    artifact_kind: &'a str,
    entries: &'a [ArtifactEntry],
}

fn fail(message: impl Into<String>) -> String {
    message.into()
}

fn relative_path_string(root: &Path, path: &Path) -> Result<String, String> {
    let relative = path
        .strip_prefix(root)
        .map_err(|error| fail(format!("artifact path escaped root: {error}")))?;
    let mut components = Vec::new();
    for component in relative.components() {
        match component {
            Component::Normal(value) => {
                let value = value
                    .to_str()
                    .ok_or_else(|| fail("artifact path contains a non-UTF8 component"))?;
                if value.is_empty() {
                    return Err(fail("artifact path contains an empty component"));
                }
                components.push(value);
            }
            _ => return Err(fail("artifact path is not a normalized relative path")),
        }
    }
    if components.is_empty() {
        return Err(fail("artifact entry cannot refer to the root directory"));
    }
    Ok(components.join("/"))
}

fn hash_file(path: &Path) -> Result<(u64, String), String> {
    let mut file = fs::File::open(path)
        .map_err(|error| fail(format!("cannot open {}: {error}", path.display())))?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0u8; BUFFER_SIZE];
    let mut size_bytes = 0u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| fail(format!("cannot read {}: {error}", path.display())))?;
        if read == 0 {
            break;
        }
        size_bytes = size_bytes
            .checked_add(read as u64)
            .ok_or_else(|| fail("file size overflow while hashing artifact"))?;
        hasher.update(&buffer[..read]);
    }
    Ok((size_bytes, hasher.finalize().to_hex().to_string()))
}

fn walk(root: &Path, current: &Path, entries: &mut Vec<ArtifactEntry>) -> Result<(), String> {
    let mut children = fs::read_dir(current)
        .map_err(|error| fail(format!("cannot enumerate {}: {error}", current.display())))?
        .map(|entry| {
            entry.map_err(|error| fail(format!("cannot read directory entry: {error}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    children.sort_by_key(|entry| entry.file_name());

    for child in children {
        let path = child.path();
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| fail(format!("cannot stat {}: {error}", path.display())))?;
        let file_type = metadata.file_type();
        if file_type.is_symlink() {
            return Err(fail(format!(
                "symlink rejected in artifact tree: {}",
                relative_path_string(root, &path)?
            )));
        }
        if file_type.is_dir() {
            walk(root, &path, entries)?;
        } else if file_type.is_file() {
            let relative_path = relative_path_string(root, &path)?;
            let (size_bytes, blake3) = hash_file(&path)?;
            entries.push(ArtifactEntry {
                relative_path,
                size_bytes,
                blake3,
            });
        } else {
            return Err(fail(format!(
                "non-regular artifact entry rejected: {}",
                relative_path_string(root, &path)?
            )));
        }
    }
    Ok(())
}

fn normalized_root(path: &Path) -> Result<PathBuf, String> {
    // Inspect the caller-supplied root before canonicalization: canonicalize
    // follows symlinks, which would otherwise hide a symlink root.
    let supplied_metadata = fs::symlink_metadata(path)
        .map_err(|error| fail(format!("cannot stat artifact root: {error}")))?;
    if supplied_metadata.file_type().is_symlink() || !supplied_metadata.file_type().is_dir() {
        return Err(fail("artifact root must be a real directory, not a symlink"));
    }
    path.canonicalize()
        .map_err(|error| fail(format!("artifact root does not exist: {error}")))
}

fn ensure_manifest_outside_root(root: &Path, manifest_path: &Path) -> Result<(), String> {
    let root = normalized_root(root)?;
    let parent = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let parent = parent
        .canonicalize()
        .map_err(|error| fail(format!("manifest parent directory is unavailable: {error}")))?;
    let name = manifest_path
        .file_name()
        .ok_or_else(|| fail("manifest path must name a file"))?;
    let target = parent.join(name);
    if target.starts_with(&root) {
        return Err(fail("manifest file must be outside the artifact root"));
    }
    Ok(())
}

fn build_manifest(root: &Path, artifact_kind: &str) -> Result<ArtifactManifest, String> {
    if artifact_kind.trim().is_empty() {
        return Err(fail("artifact_kind must be non-empty"));
    }
    let root = normalized_root(root)?;
    let mut entries = Vec::new();
    walk(&root, &root, &mut entries)?;
    entries.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));

    let mut seen = BTreeSet::new();
    for entry in &entries {
        if !seen.insert(entry.relative_path.as_str()) {
            return Err(fail(format!(
                "duplicate normalized artifact path: {}",
                entry.relative_path
            )));
        }
    }
    if entries.is_empty() {
        return Err(fail("artifact tree contains no regular files"));
    }

    let payload = RootPayload {
        schema: SCHEMA,
        artifact_kind,
        entries: &entries,
    };
    let canonical = serde_json::to_vec(&payload)
        .map_err(|error| fail(format!("cannot encode canonical manifest payload: {error}")))?;
    let root_blake3 = blake3::hash(&canonical).to_hex().to_string();

    Ok(ArtifactManifest {
        schema: SCHEMA.to_string(),
        artifact_kind: artifact_kind.to_string(),
        entry_count: entries.len(),
        entries,
        root_blake3,
    })
}

fn verify_manifest(root: &Path, manifest: &ArtifactManifest) -> Result<(), String> {
    if manifest.schema != SCHEMA {
        return Err(fail(format!(
            "unsupported manifest schema {:?}",
            manifest.schema
        )));
    }
    if manifest.entry_count != manifest.entries.len() {
        return Err(fail("manifest entry_count does not match entries length"));
    }
    let actual = build_manifest(root, &manifest.artifact_kind)?;
    if actual != *manifest {
        return Err(fail(format!(
            "artifact tree mismatch: expected root {}, observed root {}",
            manifest.root_blake3, actual.root_blake3
        )));
    }
    Ok(())
}

fn arg(args: &[String], name: &str) -> Result<String, String> {
    let index = args
        .iter()
        .position(|argument| argument == name)
        .ok_or_else(|| fail(format!("missing required argument {name}")))?;
    args.get(index + 1)
        .cloned()
        .ok_or_else(|| fail(format!("missing value after {name}")))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let command = args
        .first()
        .ok_or_else(|| fail("expected create or verify command"))?
        .as_str();
    match command {
        "create" => {
            let root = PathBuf::from(arg(&args, "--root")?);
            let kind = arg(&args, "--kind")?;
            let output = PathBuf::from(arg(&args, "--output")?);
            ensure_manifest_outside_root(&root, &output)?;
            let manifest = build_manifest(&root, &kind)?;
            let bytes = serde_json::to_vec_pretty(&manifest)
                .map_err(|error| fail(format!("cannot serialize manifest: {error}")))?;
            fs::write(&output, bytes).map_err(|error| {
                fail(format!("cannot write manifest {}: {error}", output.display()))
            })?;
            println!(
                "artifact_kind={} entries={} root_blake3={}",
                manifest.artifact_kind, manifest.entry_count, manifest.root_blake3
            );
            Ok(())
        }
        "verify" => {
            let root = PathBuf::from(arg(&args, "--root")?);
            let path = PathBuf::from(arg(&args, "--manifest")?);
            let bytes = fs::read(&path).map_err(|error| {
                fail(format!("cannot read manifest {}: {error}", path.display()))
            })?;
            ensure_manifest_outside_root(&root, &path)?;
            let manifest: ArtifactManifest = serde_json::from_slice(&bytes)
                .map_err(|error| fail(format!("invalid artifact manifest: {error}")))?;
            verify_manifest(&root, &manifest)?;
            println!(
                "verified artifact_kind={} entries={} root_blake3={}",
                manifest.artifact_kind, manifest.entry_count, manifest.root_blake3
            );
            Ok(())
        }
        _ => Err(fail("expected create or verify command")),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("voice-artifact-manifest: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR: AtomicU64 = AtomicU64::new(1);

    struct TestTree(PathBuf);

    impl TestTree {
        fn new() -> Self {
            let id = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
            let path = env::temp_dir().join(format!(
                "symthaea-voice-artifact-test-{}-{id}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn write(&self, relative: &str, bytes: &[u8]) {
            let path = self.0.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, bytes).unwrap();
        }
    }

    impl Drop for TestTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn manifest_is_deterministic_and_paths_are_relative_and_sorted() {
        let tree = TestTree::new();
        tree.write("voices/en/voice.bin", b"voice-bytes");
        tree.write("model/config.json", b"{\\"version\\":1}");
        let first = build_manifest(&tree.0, "kokoro-voice-assets").unwrap();
        let second = build_manifest(&tree.0, "kokoro-voice-assets").unwrap();
        assert_eq!(first, second);
        assert_eq!(first.entry_count, 2);
        assert_eq!(first.entries[0].relative_path, "model/config.json");
        assert_eq!(first.entries[1].relative_path, "voices/en/voice.bin");
        assert!(!first.entries[0].relative_path.starts_with('/'));
        assert_eq!(first.schema, SCHEMA);
    }

    #[test]
    fn verify_passes_for_identical_tree_and_fails_for_modified_bytes() {
        let tree = TestTree::new();
        tree.write("acoustic.onnx", b"model-v1");
        tree.write("phonemes.json", b"{\\"a\\":1}");
        let manifest = build_manifest(&tree.0, "diffsinger-voicebank").unwrap();
        verify_manifest(&tree.0, &manifest).unwrap();

        tree.write("acoustic.onnx", b"model-v2");
        assert!(verify_manifest(&tree.0, &manifest).is_err());
    }

    #[test]
    fn extra_and_missing_files_change_the_root_digest() {
        let tree = TestTree::new();
        tree.write("weights.bin", b"weights");
        let manifest = build_manifest(&tree.0, "model").unwrap();

        tree.write("extra.bin", b"extra");
        assert!(verify_manifest(&tree.0, &manifest).is_err());

        fs::remove_file(tree.0.join("extra.bin")).unwrap();
        fs::remove_file(tree.0.join("weights.bin")).unwrap();
        assert!(verify_manifest(&tree.0, &manifest).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_are_rejected_instead_of_followed() {
        use std::os::unix::fs::symlink;
        let tree = TestTree::new();
        let external = TestTree::new();
        external.write("secret.bin", b"must-not-be-followed");
        symlink(external.0.join("secret.bin"), tree.0.join("link.bin")).unwrap();
        let error = build_manifest(&tree.0, "model").unwrap_err();
        assert!(error.contains("symlink rejected"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn symlink_root_is_rejected_before_canonicalization() {
        use std::os::unix::fs::symlink;
        let tree = TestTree::new();
        tree.write("weights.bin", b"weights");
        let parent = TestTree::new();
        let alias = parent.0.join("root-link");
        symlink(&tree.0, &alias).unwrap();
        let error = build_manifest(&alias, "model").unwrap_err();
        assert!(error.contains("real directory"), "{error}");
    }

    #[test]
    fn manifest_must_be_stored_outside_artifact_tree() {
        let tree = TestTree::new();
        tree.write("weights.bin", b"weights");
        assert!(
            ensure_manifest_outside_root(&tree.0, &tree.0.join("manifest.json")).is_err()
        );
    }

    #[test]
    fn empty_trees_and_empty_kinds_are_rejected() {
        let tree = TestTree::new();
        assert!(build_manifest(&tree.0, "model").is_err());
        tree.write("one.bin", b"x");
        assert!(build_manifest(&tree.0, " ").is_err());
    }
}
