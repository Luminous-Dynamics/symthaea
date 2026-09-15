// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Content-addressed consistency receipts for transactional Psych-Bench reproduction.
//!
//! This helper does not execute benchmarks. It verifies the controller's completed
//! step journal and staged artifact census, binds those bytes to source/toolchain
//! identity, and emits a separate promotion manifest. Suite compatibility remains
//! explicitly unbound until the declarative/observed-suite theorem is qualified.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const RECEIPT_SCHEMA: &str = "psych-bench-reproduction-receipt-v1";
const PROMOTION_SCHEMA: &str = "psych-bench-reproduction-promotion-v1";
const RECEIPT_DOMAIN: &[u8] = b"symthaea.psych-bench.reproduction.receipt.v1\0";
const PROMOTION_DOMAIN: &[u8] = b"symthaea.psych-bench.reproduction.promotion.v1\0";
const DIGEST_ALGORITHM: &str = "blake3-256";
const STEP_IDS: [&str; 4] = [
    "core_runner",
    "paper_csv",
    "multi_seed_robustness",
    "qualia_confidence",
];
const TOP_LEVEL_ARTIFACTS: [&str; 3] = [
    "out_default.json",
    "out_stability.md",
    "out_qualia_confidence.json",
];
const PAPER_CSV_ARTIFACTS: [&str; 14] = [
    "ablation_domains.csv",
    "cognitive_profile.csv",
    "correlations.csv",
    "neuromod_curves.csv",
    "neuromod_profiles.csv",
    "normative_zscores.csv",
    "reliability.csv",
    "sat_arcfluid.csv",
    "sat_curves.csv",
    "sat_flanker.csv",
    "sat_nback.csv",
    "sat_stroop.csv",
    "sat_visualsearch.csv",
    "sat_wcst.csv",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct StepStatus {
    step_id: String,
    exit_code: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ArtifactIdentity {
    path: String,
    size_bytes: u64,
    blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SuiteBinding {
    status: String,
    reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ReproductionReceipt {
    schema_version: String,
    digest_algorithm: String,
    mode: String,
    source_subject: String,
    source_tree: String,
    reproduce_script_blake3: String,
    cargo_lock_blake3: String,
    rustc_version: String,
    cargo_version: String,
    suite_binding: SuiteBinding,
    steps: Vec<StepStatus>,
    artifacts: Vec<ArtifactIdentity>,
    receipt_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PromotionEntry {
    staged_path: String,
    canonical_path: String,
    blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PromotionManifest {
    schema_version: String,
    digest_algorithm: String,
    source_subject: String,
    reproduction_receipt_digest: String,
    entries: Vec<PromotionEntry>,
    manifest_digest: String,
}

struct Inputs {
    mode: String,
    artifacts: PathBuf,
    journal: PathBuf,
    source_subject: String,
    source_tree: String,
    script: PathBuf,
    cargo_lock: PathBuf,
    rustc_version_file: PathBuf,
    cargo_version_file: PathBuf,
    receipt_out: PathBuf,
    promotion_out: PathBuf,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("reproduction_receipt: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 12 {
        return Err(usage());
    }
    let command = args[0].as_str();
    let input = Inputs::from_args(&args[1..])?;
    match command {
        "build" => build(&input),
        "verify" => verify(&input),
        _ => Err(usage()),
    }
}

fn usage() -> String {
    "usage: reproduction_receipt <build|verify> <execute|self_test> <artifacts_dir> <journal> <source_sha> <source_tree> <reproduce_script> <Cargo.lock> <rustc_version_file> <cargo_version_file> <receipt_json> <promotion_json>".to_string()
}

impl Inputs {
    fn from_args(args: &[String]) -> Result<Self, String> {
        if args.len() != 11 {
            return Err(usage());
        }
        if args[0] != "execute" && args[0] != "self_test" {
            return Err("mode must be execute or self_test".to_string());
        }
        validate_git_oid("source_subject", &args[3])?;
        validate_git_oid("source_tree", &args[4])?;
        Ok(Self {
            mode: args[0].clone(),
            artifacts: PathBuf::from(&args[1]),
            journal: PathBuf::from(&args[2]),
            source_subject: args[3].clone(),
            source_tree: args[4].clone(),
            script: PathBuf::from(&args[5]),
            cargo_lock: PathBuf::from(&args[6]),
            rustc_version_file: PathBuf::from(&args[7]),
            cargo_version_file: PathBuf::from(&args[8]),
            receipt_out: PathBuf::from(&args[9]),
            promotion_out: PathBuf::from(&args[10]),
        })
    }
}

fn build(input: &Inputs) -> Result<(), String> {
    let receipt = reconstruct_receipt(input)?;
    let promotion = reconstruct_promotion(&receipt)?;
    write_json(&input.receipt_out, &receipt)?;
    write_json(&input.promotion_out, &promotion)?;
    verify(input)
}

fn verify(input: &Inputs) -> Result<(), String> {
    let expected_receipt = reconstruct_receipt(input)?;
    let actual_receipt: ReproductionReceipt = read_json(&input.receipt_out)?;
    if actual_receipt != expected_receipt {
        return Err("reproduction receipt does not match current subject/artifacts".to_string());
    }
    let expected_promotion = reconstruct_promotion(&expected_receipt)?;
    let actual_promotion: PromotionManifest = read_json(&input.promotion_out)?;
    if actual_promotion != expected_promotion {
        return Err("promotion manifest does not match verified reproduction receipt".to_string());
    }
    Ok(())
}

fn reconstruct_receipt(input: &Inputs) -> Result<ReproductionReceipt, String> {
    let mut receipt = ReproductionReceipt {
        schema_version: RECEIPT_SCHEMA.to_string(),
        digest_algorithm: DIGEST_ALGORITHM.to_string(),
        mode: input.mode.clone(),
        source_subject: input.source_subject.clone(),
        source_tree: input.source_tree.clone(),
        reproduce_script_blake3: hash_file(&input.script)?,
        cargo_lock_blake3: hash_file(&input.cargo_lock)?,
        rustc_version: read_nonempty_text(&input.rustc_version_file, "rustc version")?,
        cargo_version: read_nonempty_text(&input.cargo_version_file, "cargo version")?,
        suite_binding: SuiteBinding {
            status: "unbound".to_string(),
            reason: "declarative suite compatibility is intentionally deferred to #3334/#3386"
                .to_string(),
        },
        steps: read_journal(&input.journal)?,
        artifacts: census_artifacts(&input.artifacts)?,
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = compute_receipt_digest(&receipt);
    Ok(receipt)
}

fn reconstruct_promotion(receipt: &ReproductionReceipt) -> Result<PromotionManifest, String> {
    let mut entries = Vec::new();
    for name in PAPER_CSV_ARTIFACTS {
        let staged_path = format!("paper_csv/{name}");
        let artifact = receipt
            .artifacts
            .iter()
            .find(|artifact| artifact.path == staged_path)
            .ok_or_else(|| format!("verified receipt is missing {staged_path}"))?;
        entries.push(PromotionEntry {
            staged_path,
            canonical_path: format!("papers/data/psych_bench/{name}"),
            blake3: artifact.blake3.clone(),
        });
    }
    entries.sort_by(|left, right| left.canonical_path.cmp(&right.canonical_path));
    let mut manifest = PromotionManifest {
        schema_version: PROMOTION_SCHEMA.to_string(),
        digest_algorithm: DIGEST_ALGORITHM.to_string(),
        source_subject: receipt.source_subject.clone(),
        reproduction_receipt_digest: receipt.receipt_digest.clone(),
        entries,
        manifest_digest: String::new(),
    };
    manifest.manifest_digest = compute_promotion_digest(&manifest);
    Ok(manifest)
}

fn read_journal(path: &Path) -> Result<Vec<StepStatus>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("failed to read journal {}: {error}", path.display()))?;
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() != STEP_IDS.len() {
        return Err(format!(
            "step journal must contain exactly {} lines; found {}",
            STEP_IDS.len(),
            lines.len()
        ));
    }
    let mut statuses = Vec::new();
    for (index, expected_id) in STEP_IDS.iter().enumerate() {
        let mut fields = lines[index].split('\t');
        let step_id = fields.next().unwrap_or_default();
        let exit_code = fields.next().unwrap_or_default();
        if fields.next().is_some() || step_id != *expected_id || exit_code != "0" {
            return Err(format!(
                "invalid journal line {}: expected {expected_id}\\t0",
                index + 1
            ));
        }
        statuses.push(StepStatus {
            step_id: step_id.to_string(),
            exit_code: 0,
        });
    }
    Ok(statuses)
}

fn census_artifacts(root: &Path) -> Result<Vec<ArtifactIdentity>, String> {
    if !root.is_dir() {
        return Err(format!("artifact root is not a directory: {}", root.display()));
    }
    let expected_top: BTreeSet<String> = TOP_LEVEL_ARTIFACTS
        .iter()
        .map(|value| (*value).to_string())
        .chain(std::iter::once("paper_csv".to_string()))
        .collect();
    let actual_top = read_entry_names(root)?;
    if actual_top != expected_top {
        return Err(format!(
            "top-level artifact census mismatch: expected {expected_top:?}, found {actual_top:?}"
        ));
    }

    let paper_dir = root.join("paper_csv");
    let paper_metadata = fs::symlink_metadata(&paper_dir)
        .map_err(|error| format!("failed to inspect {}: {error}", paper_dir.display()))?;
    if paper_metadata.file_type().is_symlink() || !paper_metadata.is_dir() {
        return Err("paper_csv must be a real directory, not a symlink".to_string());
    }
    let expected_csv: BTreeSet<String> = PAPER_CSV_ARTIFACTS
        .iter()
        .map(|value| (*value).to_string())
        .collect();
    let actual_csv = read_entry_names(&paper_dir)?;
    if actual_csv != expected_csv {
        return Err(format!(
            "paper CSV census mismatch: expected {expected_csv:?}, found {actual_csv:?}"
        ));
    }

    let mut relative_paths: Vec<String> = TOP_LEVEL_ARTIFACTS
        .iter()
        .map(|value| (*value).to_string())
        .chain(
            PAPER_CSV_ARTIFACTS
                .iter()
                .map(|value| format!("paper_csv/{value}")),
        )
        .collect();
    relative_paths.sort();

    let mut artifacts = Vec::new();
    for relative in relative_paths {
        let path = root.join(&relative);
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(format!("artifact must be a regular non-symlink file: {relative}"));
        }
        if metadata.len() == 0 {
            return Err(format!("artifact must not be empty: {relative}"));
        }
        artifacts.push(ArtifactIdentity {
            path: relative,
            size_bytes: metadata.len(),
            blake3: hash_file(&path)?,
        });
    }
    Ok(artifacts)
}

fn read_entry_names(path: &Path) -> Result<BTreeSet<String>, String> {
    let mut names = BTreeSet::new();
    for entry in fs::read_dir(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?
    {
        let entry = entry.map_err(|error| format!("failed to read directory entry: {error}"))?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| format!("non-UTF-8 artifact name under {}", path.display()))?;
        names.insert(name);
    }
    Ok(names)
}

fn read_nonempty_text(path: &Path, label: &str) -> Result<String, String> {
    let value = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {label} {}: {error}", path.display()))?;
    let trimmed = value.trim().to_string();
    if trimmed.is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    Ok(trimmed)
}

fn hash_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("failed to read {} for hashing: {error}", path.display()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn validate_git_oid(label: &str, value: &str) -> Result<(), String> {
    let valid = value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if !valid {
        return Err(format!("{label} must be a lowercase 40-hex Git object id"));
    }
    Ok(())
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("output path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to serialize JSON: {error}"))?;
    fs::write(path, bytes)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))
}

fn compute_receipt_digest(receipt: &ReproductionReceipt) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECEIPT_DOMAIN);
    for value in [
        &receipt.schema_version,
        &receipt.digest_algorithm,
        &receipt.mode,
        &receipt.source_subject,
        &receipt.source_tree,
        &receipt.reproduce_script_blake3,
        &receipt.cargo_lock_blake3,
        &receipt.rustc_version,
        &receipt.cargo_version,
        &receipt.suite_binding.status,
        &receipt.suite_binding.reason,
    ] {
        push_str(&mut hasher, value);
    }
    hasher.update(&(receipt.steps.len() as u64).to_le_bytes());
    for step in &receipt.steps {
        push_str(&mut hasher, &step.step_id);
        hasher.update(&(step.exit_code as i64).to_le_bytes());
    }
    hasher.update(&(receipt.artifacts.len() as u64).to_le_bytes());
    for artifact in &receipt.artifacts {
        push_str(&mut hasher, &artifact.path);
        hasher.update(&artifact.size_bytes.to_le_bytes());
        push_str(&mut hasher, &artifact.blake3);
    }
    hasher.finalize().to_hex().to_string()
}

fn compute_promotion_digest(manifest: &PromotionManifest) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PROMOTION_DOMAIN);
    for value in [
        &manifest.schema_version,
        &manifest.digest_algorithm,
        &manifest.source_subject,
        &manifest.reproduction_receipt_digest,
    ] {
        push_str(&mut hasher, value);
    }
    hasher.update(&(manifest.entries.len() as u64).to_le_bytes());
    for entry in &manifest.entries {
        push_str(&mut hasher, &entry.staged_path);
        push_str(&mut hasher, &entry.canonical_path);
        push_str(&mut hasher, &entry.blake3);
    }
    hasher.finalize().to_hex().to_string()
}

fn push_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_artifact_names_are_unique_and_complete() {
        let top: BTreeSet<&str> = TOP_LEVEL_ARTIFACTS.into_iter().collect();
        let csv: BTreeSet<&str> = PAPER_CSV_ARTIFACTS.into_iter().collect();
        assert_eq!(top.len(), 3);
        assert_eq!(csv.len(), 14);
    }

    #[test]
    fn promotion_paths_cannot_escape_canonical_paper_data() {
        for name in PAPER_CSV_ARTIFACTS {
            let path = format!("papers/data/psych_bench/{name}");
            assert!(path.starts_with("papers/data/psych_bench/"));
            assert!(!path.contains(".."));
        }
    }

    #[test]
    fn git_object_ids_are_strict_lowercase_sha1_hex() {
        assert!(validate_git_oid("subject", "0123456789abcdef0123456789abcdef01234567").is_ok());
        assert!(validate_git_oid("subject", "0123456789ABCDEF0123456789ABCDEF01234567").is_err());
        assert!(validate_git_oid("subject", "0123456789abcdef").is_err());
        assert!(validate_git_oid("subject", "g123456789abcdef0123456789abcdef01234567").is_err());
    }
}
