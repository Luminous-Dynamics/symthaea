// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Evaluator-side ARC projection oracle for process-isolated policy execution.
//!
//! The solver view contains training demonstrations and test inputs, with a fixed public `[[0]]`
//! sentinel in the syntactic test-output slot required by the frozen policy parser. True test
//! outputs are written only to a separate evaluator-target bundle.

use serde::Serialize;
use serde_json::{Map, Value, json};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

type Grid = Vec<Vec<u8>>;

const SCHEMA_VERSION: u32 = 1;
const SOLVER_VIEW_DOMAIN: &str = "symthaea/reasoning/arc-solver-view/v1";
const TARGET_BUNDLE_DOMAIN: &str = "symthaea/reasoning/arc-evaluator-targets/v1";
const GRID_DOMAIN: &[u8] = b"symthaea/reasoning/arc-grid/v1";
const SENTINEL_ID: &str = "public-fixed-zero-grid-v1";

#[derive(Debug, Clone, Serialize)]
struct SolverViewFile {
    relative_path: String,
    byte_length: u64,
    blake3: String,
}

#[derive(Debug, Serialize)]
struct SolverViewManifest {
    schema_version: u32,
    domain: String,
    sentinel: String,
    split: String,
    task_count: usize,
    files: Vec<SolverViewFile>,
    commitment: String,
}

#[derive(Debug, Clone, Serialize)]
struct TargetRecord {
    problem_id: String,
    expected: Grid,
    target_digest: String,
}

#[derive(Debug, Serialize)]
struct EvaluatorTargetBundle {
    schema_version: u32,
    domain: String,
    dataset_repository: String,
    dataset_revision: String,
    dataset_tree: String,
    dataset_manifest_blake3: String,
    dataset_manifest_sha256: String,
    split: String,
    selected_task_paths: Vec<String>,
    targets: Vec<TargetRecord>,
    commitment: String,
}

#[derive(Debug)]
struct ProjectedTask {
    solver_bytes: Vec<u8>,
    targets: Vec<TargetRecord>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("ARC solver-view projection failed: {err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let dataset_root = PathBuf::from(required_env("SYMTHAEA_ARC_DATASET_ROOT")?);
    let manifest_path = PathBuf::from(required_env("SYMTHAEA_ARC_DATASET_MANIFEST_PATH")?);
    let solver_root = PathBuf::from(required_env("SYMTHAEA_ARC_SOLVER_VIEW_ROOT")?);
    let solver_manifest_path = PathBuf::from(required_env("SYMTHAEA_ARC_SOLVER_VIEW_MANIFEST_PATH")?);
    let target_bundle_path = PathBuf::from(required_env("SYMTHAEA_ARC_TARGET_BUNDLE_PATH")?);
    let task_limit = required_usize_env("SYMTHAEA_ARC_SMOKE_TASK_FILES")?;
    if task_limit == 0 {
        return Err("solver-view task limit must be positive".into());
    }

    let manifest_bytes = fs::read(&manifest_path)
        .map_err(|err| format!("failed to read {}: {err}", manifest_path.display()))?;
    let manifest_value: Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|err| format!("invalid dataset manifest: {err}"))?;
    let manifest = object(&manifest_value, "dataset manifest")?;
    require_eq_str(manifest, "split", "training", "dataset manifest")?;
    let repository = require_str(manifest, "repository", "dataset manifest")?.to_string();
    let revision = require_git_hex(manifest, "revision", "dataset manifest")?;
    let tree = require_git_hex(manifest, "tree", "dataset manifest")?;
    let manifest_blake3 = require_hex64(manifest, "manifest_blake3", "dataset manifest")?;
    let manifest_sha256 = require_hex64(manifest, "manifest_sha256", "dataset manifest")?;
    let files = require_array(manifest, "files", "dataset manifest")?;
    if files.len() < task_limit {
        return Err(format!(
            "dataset manifest has {} tasks, fewer than prefix {task_limit}",
            files.len()
        ));
    }

    if solver_root.exists() {
        fs::remove_dir_all(&solver_root)
            .map_err(|err| format!("failed to clear {}: {err}", solver_root.display()))?;
    }
    let training_dir = solver_root.join("training");
    fs::create_dir_all(&training_dir)
        .map_err(|err| format!("failed to create {}: {err}", training_dir.display()))?;

    let mut solver_files = Vec::with_capacity(task_limit);
    let mut selected_paths = Vec::with_capacity(task_limit);
    let mut targets = Vec::new();

    for file in files.iter().take(task_limit) {
        let file = object(file, "dataset manifest file")?;
        let relative = require_str(file, "relative_path", "dataset manifest file")?;
        if !relative.starts_with("data/training/") || !relative.ends_with(".json") {
            return Err(format!("unexpected training path {relative}"));
        }
        let source = dataset_root.join(relative);
        let source_bytes = fs::read(&source)
            .map_err(|err| format!("failed to read {}: {err}", source.display()))?;
        let expected_source = require_hex64(file, "blake3", "dataset manifest file")?;
        if blake3::hash(&source_bytes).to_hex().to_string() != expected_source {
            return Err(format!("dataset manifest/source mismatch for {relative}"));
        }
        let stem = source
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| format!("task filename is not UTF-8: {relative}"))?;
        let projected = project_task(&source_bytes, stem)?;
        let filename = source
            .file_name()
            .ok_or_else(|| format!("task path has no filename: {relative}"))?;
        let output = training_dir.join(filename);
        fs::write(&output, &projected.solver_bytes)
            .map_err(|err| format!("failed to write {}: {err}", output.display()))?;
        let solver_relative = format!(
            "training/{}",
            filename
                .to_str()
                .ok_or_else(|| "solver-view filename is not UTF-8".to_string())?
        );
        solver_files.push(SolverViewFile {
            relative_path: solver_relative,
            byte_length: projected.solver_bytes.len() as u64,
            blake3: blake3::hash(&projected.solver_bytes).to_hex().to_string(),
        });
        selected_paths.push(relative.to_string());
        targets.extend(projected.targets);
    }

    if selected_paths.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("selected dataset prefix is not strictly lexicographic".into());
    }

    let solver_commitment = solver_view_commitment(&solver_files);
    let solver_manifest = SolverViewManifest {
        schema_version: SCHEMA_VERSION,
        domain: SOLVER_VIEW_DOMAIN.into(),
        sentinel: SENTINEL_ID.into(),
        split: "training".into(),
        task_count: solver_files.len(),
        files: solver_files,
        commitment: solver_commitment,
    };
    write_json(&solver_manifest_path, &solver_manifest)?;

    let target_commitment = target_bundle_commitment(
        &repository,
        &revision,
        &tree,
        &manifest_blake3,
        &manifest_sha256,
        &selected_paths,
        &targets,
    );
    let target_bundle = EvaluatorTargetBundle {
        schema_version: SCHEMA_VERSION,
        domain: TARGET_BUNDLE_DOMAIN.into(),
        dataset_repository: repository,
        dataset_revision: revision,
        dataset_tree: tree,
        dataset_manifest_blake3: manifest_blake3,
        dataset_manifest_sha256: manifest_sha256,
        split: "training".into(),
        selected_task_paths: selected_paths,
        targets,
        commitment: target_commitment,
    };
    write_json(&target_bundle_path, &target_bundle)?;

    println!("ARC target-isolated projection");
    println!("tasks:         {}", solver_manifest.task_count);
    println!("solver view:   {}", solver_manifest.commitment);
    println!("target bundle: {}", target_bundle.commitment);
    Ok(())
}

fn project_task(source: &[u8], stem: &str) -> Result<ProjectedTask, String> {
    let value: Value = serde_json::from_slice(source)
        .map_err(|err| format!("invalid ARC task JSON: {err}"))?;
    let task = object(&value, "ARC task")?;
    let train = require_array(task, "train", "ARC task")?;
    let test = require_array(task, "test", "ARC task")?;
    if train.is_empty() || test.is_empty() {
        return Err("ARC task requires nonempty train and test arrays".into());
    }

    let mut visible_train = Vec::with_capacity(train.len());
    for (index, pair) in train.iter().enumerate() {
        let pair = object(pair, "training pair")?;
        let input = parse_grid(
            pair.get("input")
                .ok_or_else(|| format!("train[{index}] missing input"))?,
        )?;
        let output = parse_grid(
            pair.get("output")
                .ok_or_else(|| format!("train[{index}] missing output"))?,
        )?;
        visible_train.push(json!({"input": input, "output": output}));
    }

    let mut visible_test = Vec::with_capacity(test.len());
    let mut targets = Vec::with_capacity(test.len());
    for (index, pair) in test.iter().enumerate() {
        let pair = object(pair, "test pair")?;
        let input = parse_grid(
            pair.get("input")
                .ok_or_else(|| format!("test[{index}] missing input"))?,
        )?;
        let expected = parse_grid(
            pair.get("output")
                .ok_or_else(|| format!("test[{index}] missing output"))?,
        )?;
        visible_test.push(json!({"input": input, "output": [[0]]}));
        targets.push(TargetRecord {
            problem_id: format!("{stem}#test-{index}"),
            target_digest: grid_digest(&expected),
            expected,
        });
    }

    let solver_bytes = serde_json::to_vec(&json!({
        "train": visible_train,
        "test": visible_test,
    }))
    .map_err(|err| format!("failed to encode solver-view task: {err}"))?;
    Ok(ProjectedTask {
        solver_bytes,
        targets,
    })
}

fn parse_grid(value: &Value) -> Result<Grid, String> {
    let rows = value
        .as_array()
        .ok_or_else(|| "grid must be an array".to_string())?;
    if rows.is_empty() || rows.len() > 30 {
        return Err("grid height must be within 1..=30".into());
    }
    let mut grid = Vec::with_capacity(rows.len());
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
        let mut parsed = Vec::with_capacity(cells.len());
        for cell in cells {
            let color = cell
                .as_u64()
                .ok_or_else(|| "grid cell must be an unsigned integer".to_string())?;
            if color > 9 {
                return Err("ARC colors must be within 0..=9".into());
            }
            parsed.push(color as u8);
        }
        grid.push(parsed);
    }
    Ok(grid)
}

fn grid_digest(grid: &Grid) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, GRID_DOMAIN);
    hash_u64(&mut hasher, grid.len() as u64);
    for row in grid {
        hash_u64(&mut hasher, row.len() as u64);
        for cell in row {
            hasher.update(&[*cell]);
        }
    }
    hasher.finalize().to_hex().to_string()
}

fn solver_view_commitment(files: &[SolverViewFile]) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_str(&mut hasher, SOLVER_VIEW_DOMAIN);
    hash_str(&mut hasher, SENTINEL_ID);
    hash_u64(&mut hasher, files.len() as u64);
    for file in files {
        hash_str(&mut hasher, &file.relative_path);
        hash_u64(&mut hasher, file.byte_length);
        hash_str(&mut hasher, &file.blake3);
    }
    hasher.finalize().to_hex().to_string()
}

#[allow(clippy::too_many_arguments)]
fn target_bundle_commitment(
    repository: &str,
    revision: &str,
    tree: &str,
    manifest_blake3: &str,
    manifest_sha256: &str,
    paths: &[String],
    targets: &[TargetRecord],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_str(&mut hasher, TARGET_BUNDLE_DOMAIN);
    hash_str(&mut hasher, repository);
    hash_str(&mut hasher, revision);
    hash_str(&mut hasher, tree);
    hash_str(&mut hasher, manifest_blake3);
    hash_str(&mut hasher, manifest_sha256);
    hash_u64(&mut hasher, paths.len() as u64);
    for path in paths {
        hash_str(&mut hasher, path);
    }
    hash_u64(&mut hasher, targets.len() as u64);
    for target in targets {
        hash_str(&mut hasher, &target.problem_id);
        hash_str(&mut hasher, &target.target_digest);
    }
    hasher.finalize().to_hex().to_string()
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    fs::write(
        path,
        serde_json::to_string_pretty(value)
            .map_err(|err| format!("failed to encode {}: {err}", path.display()))?,
    )
    .map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn require_array<'a>(object: &'a Map<String, Value>, key: &str, label: &str) -> Result<&'a Vec<Value>, String> {
    object
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} field {key} must be an array"))
}

fn require_str<'a>(object: &'a Map<String, Value>, key: &str, label: &str) -> Result<&'a str, String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} field {key} must be a string"))
}

fn require_eq_str(object: &Map<String, Value>, key: &str, expected: &str, label: &str) -> Result<(), String> {
    let actual = require_str(object, key, label)?;
    if actual != expected {
        return Err(format!("{label} field {key} is {actual:?}, expected {expected:?}"));
    }
    Ok(())
}

fn require_hex64(object: &Map<String, Value>, key: &str, label: &str) -> Result<String, String> {
    let value = require_str(object, key, label)?;
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{label} field {key} must be a 64-hex digest"));
    }
    Ok(value.to_ascii_lowercase())
}

fn require_git_hex(object: &Map<String, Value>, key: &str, label: &str) -> Result<String, String> {
    let value = require_str(object, key, label)?;
    if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{label} field {key} must be a 40-hex Git id"));
    }
    Ok(value.to_ascii_lowercase())
}

fn required_env(name: &str) -> Result<String, String> {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => Ok(value.trim().to_string()),
        Ok(_) => Err(format!("required environment variable {name} is empty")),
        Err(err) => Err(format!("required environment variable {name} is missing: {err}")),
    }
}

fn required_usize_env(name: &str) -> Result<usize, String> {
    required_env(name)?
        .parse::<usize>()
        .map_err(|err| format!("{name} must be an unsigned integer: {err}"))
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    hash_bytes(hasher, value.as_bytes());
}

fn hash_bytes(hasher: &mut blake3::Hasher, value: &[u8]) {
    hash_u64(hasher, value.len() as u64);
    hasher.update(value);
}

fn hash_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(expected: &str) -> Vec<u8> {
        format!(
            r#"{{"train":[{{"input":[[0,1]],"output":[[1,0]]}}],"test":[{{"input":[[2,3]],"output":{expected}}}]}}"#
        )
        .into_bytes()
    }

    #[test]
    fn hidden_target_mutation_cannot_change_solver_view_bytes() {
        let a = project_task(&fixture("[[4,5]]"), "task").unwrap();
        let b = project_task(&fixture("[[5,4]]"), "task").unwrap();
        assert_eq!(a.solver_bytes, b.solver_bytes);
        assert_ne!(a.targets[0].target_digest, b.targets[0].target_digest);
    }

    #[test]
    fn target_digest_uses_frozen_policy_grid_encoding() {
        let projected = project_task(&fixture("[[4,5]]"), "task").unwrap();
        assert_eq!(projected.targets[0].target_digest, grid_digest(&vec![vec![4, 5]]));
    }

    #[test]
    fn solver_projection_uses_public_sentinel() {
        let projected = project_task(&fixture("[[4,5]]"), "task").unwrap();
        let value: Value = serde_json::from_slice(&projected.solver_bytes).unwrap();
        assert_eq!(value["test"][0]["output"], json!([[0]]));
    }
}