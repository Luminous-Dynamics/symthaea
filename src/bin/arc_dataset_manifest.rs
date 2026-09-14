// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Canonical manifest oracle for an exact ARC-AGI-2 dataset checkout.
//!
//! This is evaluator-side provenance. The manifest may bind complete task bytes, including
//! held-out outputs. Those digests must never be reused as solver-visible features, policy seeds,
//! or action-order inputs.
//!
//! Required environment:
//! - `SYMTHAEA_ARC_DATASET_ROOT`: exact local ARC-AGI-2 checkout
//! - `SYMTHAEA_ARC_DATASET_REVISION`: exact 40-hex Git commit
//! - `SYMTHAEA_ARC_DATASET_TREE`: exact 40-hex Git tree
//! - `SYMTHAEA_ARC_SPLIT`: `training` or `evaluation`
//! - `SYMTHAEA_ARC_DATASET_MANIFEST_PATH`: output JSON path outside the dataset checkout
//!
//! Optional environment:
//! - `SYMTHAEA_ARC_DATASET_REPOSITORY`: defaults to `arcprize/ARC-AGI-2`
//! - `SYMTHAEA_ARC_EXPECTED_TASKS`: fail closed unless the split contains exactly this many tasks

use serde::Serialize;
use serde_json::Value;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const SCHEMA_VERSION: u32 = 1;
const MANIFEST_DOMAIN: &str = "symthaea/reasoning/arc-dataset-manifest/v1";
const CANONICAL_ENCODING: &str = "length-prefixed-le64-v1";
const DEFAULT_REPOSITORY: &str = "arcprize/ARC-AGI-2";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ManifestFile {
    relative_path: String,
    byte_length: u64,
    blake3: String,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct ArcDatasetManifest {
    schema_version: u32,
    manifest_domain: String,
    canonical_encoding: String,
    repository: String,
    revision: String,
    tree: String,
    split: String,
    task_count: usize,
    files: Vec<ManifestFile>,
    canonical_byte_length: usize,
    manifest_blake3: String,
    manifest_sha256: String,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("ARC dataset manifest qualification failed: {err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let root = PathBuf::from(required_env("SYMTHAEA_ARC_DATASET_ROOT")?);
    let revision = required_env("SYMTHAEA_ARC_DATASET_REVISION")?;
    let tree = required_env("SYMTHAEA_ARC_DATASET_TREE")?;
    let split = required_env("SYMTHAEA_ARC_SPLIT")?;
    let output_path = PathBuf::from(required_env("SYMTHAEA_ARC_DATASET_MANIFEST_PATH")?);
    let repository = env::var("SYMTHAEA_ARC_DATASET_REPOSITORY")
        .unwrap_or_else(|_| DEFAULT_REPOSITORY.to_string());
    let expected_tasks = optional_usize_env("SYMTHAEA_ARC_EXPECTED_TASKS")?;

    validate_full_git_hex("dataset revision", &revision)?;
    validate_full_git_hex("dataset tree", &tree)?;
    if split != "training" && split != "evaluation" {
        return Err("SYMTHAEA_ARC_SPLIT must be `training` or `evaluation`".into());
    }
    if repository.trim().is_empty() {
        return Err("dataset repository identity must not be empty".into());
    }
    if !root.is_dir() {
        return Err(format!("dataset root is not a directory: {}", root.display()));
    }

    verify_checkout(&root, &revision, &tree)?;
    let split_dir = root.join("data").join(&split);
    let files = collect_split_files(&root, &split_dir)?;
    if let Some(expected) = expected_tasks {
        if files.len() != expected {
            return Err(format!(
                "split {split} has {} tasks, expected exactly {expected}",
                files.len()
            ));
        }
    }

    let canonical = canonical_manifest_bytes(
        &repository,
        &revision,
        &tree,
        &split,
        &files,
    );
    let manifest_blake3 = blake3::hash(&canonical).to_hex().to_string();
    let manifest_sha256 = sha256_bytes(&canonical)?;

    let report = ArcDatasetManifest {
        schema_version: SCHEMA_VERSION,
        manifest_domain: MANIFEST_DOMAIN.into(),
        canonical_encoding: CANONICAL_ENCODING.into(),
        repository,
        revision: revision.clone(),
        tree: tree.clone(),
        split: split.clone(),
        task_count: files.len(),
        files,
        canonical_byte_length: canonical.len(),
        manifest_blake3,
        manifest_sha256,
    };

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let encoded = serde_json::to_string_pretty(&report)
        .map_err(|err| format!("failed to encode dataset manifest: {err}"))?;
    fs::write(&output_path, encoded)
        .map_err(|err| format!("failed to write {}: {err}", output_path.display()))?;

    // Writing the manifest must not mutate the dataset checkout. If the caller writes the output
    // inside the checkout this deliberately fails here.
    verify_checkout(&root, &revision, &tree)?;

    println!("ARC dataset manifest");
    println!("repository: {}", report.repository);
    println!("revision:   {}", report.revision);
    println!("tree:       {}", report.tree);
    println!("split:      {}", report.split);
    println!("tasks:      {}", report.task_count);
    println!("blake3:     {}", report.manifest_blake3);
    println!("sha256:     {}", report.manifest_sha256);
    println!("manifest:   {}", output_path.display());
    Ok(())
}

fn collect_split_files(root: &Path, split_dir: &Path) -> Result<Vec<ManifestFile>, String> {
    if !split_dir.is_dir() {
        return Err(format!("ARC split directory not found: {}", split_dir.display()));
    }

    let mut paths = Vec::new();
    for entry in fs::read_dir(split_dir)
        .map_err(|err| format!("failed to list {}: {err}", split_dir.display()))?
    {
        let entry = entry
            .map_err(|err| format!("failed to read entry in {}: {err}", split_dir.display()))?;
        let file_type = entry
            .file_type()
            .map_err(|err| format!("failed to inspect {}: {err}", entry.path().display()))?;
        if file_type.is_symlink() {
            return Err(format!("dataset split contains symlink: {}", entry.path().display()));
        }
        if !file_type.is_file() {
            return Err(format!(
                "dataset split contains non-file entry: {}",
                entry.path().display()
            ));
        }
        if entry.path().extension().and_then(|ext| ext.to_str()) != Some("json") {
            return Err(format!(
                "dataset split contains non-JSON file: {}",
                entry.path().display()
            ));
        }
        paths.push(entry.path());
    }
    paths.sort();
    if paths.is_empty() {
        return Err(format!("dataset split is empty: {}", split_dir.display()));
    }

    let mut files = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes = fs::read(&path)
            .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        validate_arc_task(&bytes)
            .map_err(|err| format!("invalid ARC task {}: {err}", path.display()))?;
        let relative = path
            .strip_prefix(root)
            .map_err(|_| format!("task escaped dataset root: {}", path.display()))?;
        let relative_path = relative
            .to_str()
            .ok_or_else(|| format!("task path is not valid UTF-8: {}", path.display()))?
            .replace('\\', "/");
        files.push(ManifestFile {
            relative_path,
            byte_length: bytes.len() as u64,
            blake3: blake3::hash(&bytes).to_hex().to_string(),
            sha256: sha256_bytes(&bytes)?,
        });
    }
    Ok(files)
}

fn validate_arc_task(bytes: &[u8]) -> Result<(), String> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|err| format!("invalid JSON: {err}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "task root must be an object".to_string())?;
    for section in ["train", "test"] {
        let pairs = object
            .get(section)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("task has no {section} array"))?;
        if pairs.is_empty() {
            return Err(format!("{section} array must not be empty"));
        }
        for (index, pair) in pairs.iter().enumerate() {
            let pair = pair
                .as_object()
                .ok_or_else(|| format!("{section}[{index}] must be an object"))?;
            validate_grid(
                pair.get("input")
                    .ok_or_else(|| format!("{section}[{index}] has no input grid"))?,
            )?;
            validate_grid(
                pair.get("output")
                    .ok_or_else(|| format!("{section}[{index}] has no output grid"))?,
            )?;
        }
    }
    Ok(())
}

fn validate_grid(value: &Value) -> Result<(), String> {
    let rows = value
        .as_array()
        .ok_or_else(|| "grid must be an array".to_string())?;
    if rows.is_empty() || rows.len() > 30 {
        return Err("grid height must be within 1..=30".into());
    }
    let mut width = None;
    for row in rows {
        let cells = row
            .as_array()
            .ok_or_else(|| "grid row must be an array".to_string())?;
        if cells.is_empty() || cells.len() > 30 {
            return Err("grid width must be within 1..=30".into());
        }
        if let Some(expected) = width {
            if cells.len() != expected {
                return Err("grid must be rectangular".into());
            }
        } else {
            width = Some(cells.len());
        }
        for cell in cells {
            let color = cell
                .as_u64()
                .ok_or_else(|| "grid cell must be an unsigned integer".to_string())?;
            if color > 9 {
                return Err("ARC colors must be within 0..=9".into());
            }
        }
    }
    Ok(())
}

fn canonical_manifest_bytes(
    repository: &str,
    revision: &str,
    tree: &str,
    split: &str,
    files: &[ManifestFile],
) -> Vec<u8> {
    let mut out = Vec::new();
    put_str(&mut out, MANIFEST_DOMAIN);
    put_u64(&mut out, u64::from(SCHEMA_VERSION));
    put_str(&mut out, CANONICAL_ENCODING);
    put_str(&mut out, repository);
    put_str(&mut out, revision);
    put_str(&mut out, tree);
    put_str(&mut out, split);
    put_u64(&mut out, files.len() as u64);
    for file in files {
        put_str(&mut out, &file.relative_path);
        put_u64(&mut out, file.byte_length);
        put_str(&mut out, &file.blake3);
        put_str(&mut out, &file.sha256);
    }
    out
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_bytes(out, value.as_bytes());
}

fn put_bytes(out: &mut Vec<u8>, value: &[u8]) {
    put_u64(out, value.len() as u64);
    out.extend_from_slice(value);
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn sha256_bytes(bytes: &[u8]) -> Result<String, String> {
    for (program, args) in [("sha256sum", Vec::<&str>::new()), ("shasum", vec!["-a", "256"])] {
        let mut child = match Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(_) => continue,
        };
        {
            let stdin = child
                .stdin
                .as_mut()
                .ok_or_else(|| format!("failed to open stdin for {program}"))?;
            stdin
                .write_all(bytes)
                .map_err(|err| format!("failed to stream bytes to {program}: {err}"))?;
        }
        let output = child
            .wait_with_output()
            .map_err(|err| format!("failed to wait for {program}: {err}"))?;
        if !output.status.success() {
            continue;
        }
        let stdout = String::from_utf8(output.stdout)
            .map_err(|err| format!("{program} returned non-UTF-8 output: {err}"))?;
        let digest = stdout
            .split_whitespace()
            .next()
            .ok_or_else(|| format!("{program} returned no digest"))?
            .to_ascii_lowercase();
        if digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Ok(digest);
        }
    }
    Err("no working SHA-256 provider found (`sha256sum` or `shasum -a 256`)".into())
}

fn verify_checkout(root: &Path, revision: &str, tree: &str) -> Result<(), String> {
    let head = git_output(root, &["rev-parse", "HEAD"])?;
    if head != revision {
        return Err(format!("dataset HEAD {head} does not equal declared revision {revision}"));
    }
    let actual_tree = git_output(root, &["rev-parse", "HEAD^{tree}"])?;
    if actual_tree != tree {
        return Err(format!(
            "dataset tree {actual_tree} does not equal declared tree {tree}"
        ));
    }
    let status = git_output(root, &["status", "--porcelain=v1", "--untracked-files=all"])?;
    if !status.is_empty() {
        return Err(format!("dataset checkout is dirty:\n{status}"));
    }
    Ok(())
}

fn git_output(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|err| format!("failed to execute git: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn validate_full_git_hex(label: &str, value: &str) -> Result<(), String> {
    if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{label} must be exactly 40 hexadecimal characters"));
    }
    Ok(())
}

fn required_env(name: &str) -> Result<String, String> {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        _ => Err(format!("required environment variable {name} is missing")),
    }
}

fn optional_usize_env(name: &str) -> Result<Option<usize>, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map(Some)
            .map_err(|err| format!("{name} must be an unsigned integer: {err}")),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(format!("failed to read {name}: {err}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_task() -> Vec<u8> {
        br#"{
          "train": [{"input": [[0,1],[2,3]], "output": [[1,0],[3,2]]}],
          "test": [{"input": [[4,5]], "output": [[5,4]]}]
        }"#
        .to_vec()
    }

    fn file(path: &str, byte_length: u64, blake3: &str, sha256: &str) -> ManifestFile {
        ManifestFile {
            relative_path: path.into(),
            byte_length,
            blake3: blake3.into(),
            sha256: sha256.into(),
        }
    }

    #[test]
    fn accepts_well_formed_arc_task() {
        validate_arc_task(&valid_task()).unwrap();
    }

    #[test]
    fn rejects_missing_test_output() {
        let raw = br#"{
          "train": [{"input": [[0]], "output": [[0]]}],
          "test": [{"input": [[1]]}]
        }"#;
        assert!(validate_arc_task(raw).is_err());
    }

    #[test]
    fn rejects_out_of_range_arc_color() {
        let raw = br#"{
          "train": [{"input": [[10]], "output": [[0]]}],
          "test": [{"input": [[1]], "output": [[1]]}]
        }"#;
        assert!(validate_arc_task(raw).is_err());
    }

    #[test]
    fn canonical_manifest_is_order_sensitive_and_content_bound() {
        let first = vec![
            file("data/training/a.json", 10, "aa", "11"),
            file("data/training/b.json", 20, "bb", "22"),
        ];
        let mut changed = first.clone();
        changed[1].sha256 = "33".into();
        let mut reordered = first.clone();
        reordered.reverse();

        let base = canonical_manifest_bytes("repo", &"a".repeat(40), &"b".repeat(40), "training", &first);
        let changed_bytes = canonical_manifest_bytes(
            "repo",
            &"a".repeat(40),
            &"b".repeat(40),
            "training",
            &changed,
        );
        let reordered_bytes = canonical_manifest_bytes(
            "repo",
            &"a".repeat(40),
            &"b".repeat(40),
            "training",
            &reordered,
        );
        assert_ne!(base, changed_bytes);
        assert_ne!(base, reordered_bytes);
    }

    #[test]
    fn full_git_identity_requires_40_hex_characters() {
        assert!(validate_full_git_hex("revision", &"a".repeat(40)).is_ok());
        assert!(validate_full_git_hex("revision", &"a".repeat(39)).is_err());
        assert!(validate_full_git_hex("revision", &"g".repeat(40)).is_err());
    }
}