// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Evaluator-side receipt for the first exact-manifest ARC public-training smoke run.
//!
//! This binary runs only after the budgeted policy report has been produced. It does not execute
//! policy actions and does not feed manifest/target provenance back into the solver. Its job is to
//! prove that a frozen policy report is structurally consistent and bind it to the exact public
//! dataset manifest used by the evaluator.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const SCHEMA_VERSION: u32 = 1;
const RECEIPT_DOMAIN: &str = "symthaea/reasoning/arc-public-training-smoke/v1";
const AUTHORITY: &str = "DevelopmentProbe";
const CONTAMINATION: &str = "Exposed";
const EXPECTED_MANIFEST_DOMAIN: &str = "symthaea/reasoning/arc-dataset-manifest/v1";
const EXPECTED_POLICY_SCHEMA: u64 = 2;
const EXPECTED_POLICY_CONFIGURATION: &str = "arc-native-budgeted-policy-v2";
const EXPECTED_ACTION_SPACE_VERSION: &str = "rq003-candidate-transform-v1";
const EXPECTED_GRAMMAR_SIZE: usize = 4_914;

#[derive(Debug)]
struct SmokeConfig {
    subject_revision: String,
    dataset_repository: String,
    dataset_revision: String,
    dataset_tree: String,
    task_files: usize,
    budget: usize,
    random_seed_root: u64,
}

#[derive(Debug)]
struct ManifestFacts {
    manifest_blake3: String,
    manifest_sha256: String,
    selected_task_paths: Vec<String>,
}

#[derive(Debug)]
struct PolicyFacts {
    action_space_commitment: String,
    test_cases_evaluated: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ArcSmokeReceipt {
    schema_version: u32,
    receipt_domain: String,
    authority: String,
    contamination_status: String,
    subject_revision: String,
    dataset_repository: String,
    dataset_revision: String,
    dataset_tree: String,
    dataset_split: String,
    dataset_manifest_domain: String,
    dataset_manifest_blake3: String,
    dataset_manifest_sha256: String,
    dataset_manifest_file_blake3: String,
    dataset_manifest_file_sha256: String,
    policy_report_schema_version: u64,
    policy_configuration_id: String,
    action_space_version: String,
    action_space_commitment: String,
    candidate_grammar_size: usize,
    task_limit: usize,
    test_cases_evaluated: usize,
    candidate_budget: usize,
    random_seed_root: u64,
    selected_task_paths: Vec<String>,
    policy_report_blake3: String,
    policy_report_sha256: String,
    receipt_blake3: String,
    receipt_sha256: String,
}

impl ArcSmokeReceipt {
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_str(&mut out, RECEIPT_DOMAIN);
        put_u64(&mut out, u64::from(self.schema_version));
        put_str(&mut out, &self.authority);
        put_str(&mut out, &self.contamination_status);
        put_str(&mut out, &self.subject_revision);
        put_str(&mut out, &self.dataset_repository);
        put_str(&mut out, &self.dataset_revision);
        put_str(&mut out, &self.dataset_tree);
        put_str(&mut out, &self.dataset_split);
        put_str(&mut out, &self.dataset_manifest_domain);
        put_str(&mut out, &self.dataset_manifest_blake3);
        put_str(&mut out, &self.dataset_manifest_sha256);
        put_str(&mut out, &self.dataset_manifest_file_blake3);
        put_str(&mut out, &self.dataset_manifest_file_sha256);
        put_u64(&mut out, self.policy_report_schema_version);
        put_str(&mut out, &self.policy_configuration_id);
        put_str(&mut out, &self.action_space_version);
        put_str(&mut out, &self.action_space_commitment);
        put_u64(&mut out, self.candidate_grammar_size as u64);
        put_u64(&mut out, self.task_limit as u64);
        put_u64(&mut out, self.test_cases_evaluated as u64);
        put_u64(&mut out, self.candidate_budget as u64);
        put_u64(&mut out, self.random_seed_root);
        put_u64(&mut out, self.selected_task_paths.len() as u64);
        for path in &self.selected_task_paths {
            put_str(&mut out, path);
        }
        put_str(&mut out, &self.policy_report_blake3);
        put_str(&mut out, &self.policy_report_sha256);
        out
    }

    fn commitment_valid(&self) -> Result<bool, String> {
        let canonical = self.canonical_bytes();
        let blake3 = blake3::hash(&canonical).to_hex().to_string();
        let sha256 = sha256_bytes(&canonical)?;
        Ok(self.receipt_blake3 == blake3 && self.receipt_sha256 == sha256)
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("ARC public-training smoke receipt failed: {err}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let config = SmokeConfig {
        subject_revision: required_full_revision("SYMTHAEA_SUBJECT_REVISION")?,
        dataset_repository: required_env("SYMTHAEA_ARC_DATASET_REPOSITORY")?,
        dataset_revision: required_full_revision("SYMTHAEA_ARC_DATASET_REVISION")?,
        dataset_tree: required_full_revision("SYMTHAEA_ARC_DATASET_TREE")?,
        task_files: required_usize_env("SYMTHAEA_ARC_SMOKE_TASK_FILES")?,
        budget: required_usize_env("SYMTHAEA_ARC_CANDIDATE_BUDGET")?,
        random_seed_root: required_u64_env("SYMTHAEA_ARC_POLICY_SEED")?,
    };
    if config.task_files == 0 {
        return Err("smoke task-file count must be greater than zero".into());
    }
    if config.budget == 0 || config.budget > EXPECTED_GRAMMAR_SIZE {
        return Err(format!(
            "smoke candidate budget must be within 1..={EXPECTED_GRAMMAR_SIZE}"
        ));
    }

    let manifest_path = PathBuf::from(required_env("SYMTHAEA_ARC_DATASET_MANIFEST_PATH")?);
    let report_path = PathBuf::from(required_env("SYMTHAEA_ARC_POLICY_RESULTS_PATH")?);
    let receipt_path = PathBuf::from(required_env("SYMTHAEA_ARC_SMOKE_RECEIPT_PATH")?);

    let manifest_bytes = fs::read(&manifest_path)
        .map_err(|err| format!("failed to read {}: {err}", manifest_path.display()))?;
    let report_bytes = fs::read(&report_path)
        .map_err(|err| format!("failed to read {}: {err}", report_path.display()))?;
    let manifest_value: Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|err| format!("invalid dataset manifest JSON: {err}"))?;
    let report_value: Value = serde_json::from_slice(&report_bytes)
        .map_err(|err| format!("invalid policy report JSON: {err}"))?;

    let manifest_facts = validate_manifest(&manifest_value, &config)?;
    let policy_facts = validate_policy_report(&report_value, &config, &manifest_facts)?;

    let mut receipt = ArcSmokeReceipt {
        schema_version: SCHEMA_VERSION,
        receipt_domain: RECEIPT_DOMAIN.into(),
        authority: AUTHORITY.into(),
        contamination_status: CONTAMINATION.into(),
        subject_revision: config.subject_revision,
        dataset_repository: config.dataset_repository,
        dataset_revision: config.dataset_revision,
        dataset_tree: config.dataset_tree,
        dataset_split: "training".into(),
        dataset_manifest_domain: EXPECTED_MANIFEST_DOMAIN.into(),
        dataset_manifest_blake3: manifest_facts.manifest_blake3,
        dataset_manifest_sha256: manifest_facts.manifest_sha256,
        dataset_manifest_file_blake3: blake3::hash(&manifest_bytes).to_hex().to_string(),
        dataset_manifest_file_sha256: sha256_bytes(&manifest_bytes)?,
        policy_report_schema_version: EXPECTED_POLICY_SCHEMA,
        policy_configuration_id: EXPECTED_POLICY_CONFIGURATION.into(),
        action_space_version: EXPECTED_ACTION_SPACE_VERSION.into(),
        action_space_commitment: policy_facts.action_space_commitment,
        candidate_grammar_size: EXPECTED_GRAMMAR_SIZE,
        task_limit: config.task_files,
        test_cases_evaluated: policy_facts.test_cases_evaluated,
        candidate_budget: config.budget,
        random_seed_root: config.random_seed_root,
        selected_task_paths: manifest_facts.selected_task_paths,
        policy_report_blake3: blake3::hash(&report_bytes).to_hex().to_string(),
        policy_report_sha256: sha256_bytes(&report_bytes)?,
        receipt_blake3: String::new(),
        receipt_sha256: String::new(),
    };
    let canonical = receipt.canonical_bytes();
    receipt.receipt_blake3 = blake3::hash(&canonical).to_hex().to_string();
    receipt.receipt_sha256 = sha256_bytes(&canonical)?;
    if !receipt.commitment_valid()? {
        return Err("new smoke receipt failed its own commitment validation".into());
    }

    if let Some(parent) = receipt_path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    fs::write(
        &receipt_path,
        serde_json::to_string_pretty(&receipt)
            .map_err(|err| format!("failed to encode smoke receipt: {err}"))?,
    )
    .map_err(|err| format!("failed to write {}: {err}", receipt_path.display()))?;

    println!("ARC public-training smoke receipt");
    println!("subject:       {}", receipt.subject_revision);
    println!("dataset:       {}", receipt.dataset_revision);
    println!("manifest:      {}", receipt.dataset_manifest_blake3);
    println!("policy report: {}", receipt.policy_report_blake3);
    println!("receipt:       {}", receipt.receipt_blake3);
    println!("path:          {}", receipt_path.display());
    Ok(())
}

fn validate_manifest(value: &Value, config: &SmokeConfig) -> Result<ManifestFacts, String> {
    let object = object(value, "dataset manifest")?;
    require_u64(object, "schema_version", "dataset manifest")?.eq(&1).then_some(()).ok_or_else(|| {
        "dataset manifest schema_version must equal 1".to_string()
    })?;
    require_eq_str(object, "manifest_domain", EXPECTED_MANIFEST_DOMAIN, "dataset manifest")?;
    require_eq_str(object, "repository", &config.dataset_repository, "dataset manifest")?;
    require_eq_str(object, "revision", &config.dataset_revision, "dataset manifest")?;
    require_eq_str(object, "tree", &config.dataset_tree, "dataset manifest")?;
    require_eq_str(object, "split", "training", "dataset manifest")?;

    let manifest_blake3 = require_hex64(object, "manifest_blake3", "dataset manifest")?;
    let manifest_sha256 = require_hex64(object, "manifest_sha256", "dataset manifest")?;
    let files = require_array(object, "files", "dataset manifest")?;
    if files.len() < config.task_files {
        return Err(format!(
            "dataset manifest has {} files, fewer than smoke prefix {}",
            files.len(), config.task_files
        ));
    }

    let mut selected_task_paths = Vec::with_capacity(config.task_files);
    for file in files.iter().take(config.task_files) {
        let file = object(file, "dataset manifest file")?;
        let path = require_str(file, "relative_path", "dataset manifest file")?;
        if !path.starts_with("data/training/") || !path.ends_with(".json") {
            return Err(format!("unexpected smoke task path: {path}"));
        }
        selected_task_paths.push(path.to_string());
    }
    if selected_task_paths.windows(2).any(|window| window[0] >= window[1]) {
        return Err("smoke task prefix is not strictly lexicographically ordered".into());
    }

    Ok(ManifestFacts {
        manifest_blake3,
        manifest_sha256,
        selected_task_paths,
    })
}

fn validate_policy_report(
    value: &Value,
    config: &SmokeConfig,
    manifest: &ManifestFacts,
) -> Result<PolicyFacts, String> {
    let object = object(value, "policy report")?;
    if require_u64(object, "schema_version", "policy report")? != EXPECTED_POLICY_SCHEMA {
        return Err(format!(
            "policy report schema must equal {EXPECTED_POLICY_SCHEMA}"
        ));
    }
    require_eq_str(object, "subject_revision", &config.subject_revision, "policy report")?;
    require_eq_str(object, "dataset_version", &config.dataset_revision, "policy report")?;
    require_eq_str(object, "split", "training", "policy report")?;
    require_eq_str(
        object,
        "configuration_id",
        EXPECTED_POLICY_CONFIGURATION,
        "policy report",
    )?;
    require_eq_str(
        object,
        "action_space_version",
        EXPECTED_ACTION_SPACE_VERSION,
        "policy report",
    )?;
    let action_space_commitment = require_hex64(object, "action_space_commitment", "policy report")?;
    if require_usize(object, "candidate_grammar_size", "policy report")? != EXPECTED_GRAMMAR_SIZE {
        return Err("unexpected ARC grammar size in policy report".into());
    }
    if require_usize(object, "budget", "policy report")? != config.budget {
        return Err("policy report candidate budget does not match smoke configuration".into());
    }
    if require_u64(object, "random_seed_root", "policy report")? != config.random_seed_root {
        return Err("policy report random seed root does not match smoke configuration".into());
    }
    if require_usize(object, "task_limit", "policy report")? != config.task_files {
        return Err("policy report task limit does not match smoke prefix".into());
    }
    if require_usize(object, "task_files_seen", "policy report")? != config.task_files {
        return Err("policy report did not evaluate the exact smoke task-file count".into());
    }
    let test_cases_evaluated = require_usize(object, "test_cases_evaluated", "policy report")?;
    if test_cases_evaluated == 0 {
        return Err("smoke policy report contains no test cases".into());
    }

    for aggregate_name in ["canonical", "uniform_random", "exhaustive_reference"] {
        let aggregate = object(
            object
                .get(aggregate_name)
                .ok_or_else(|| format!("policy report missing {aggregate_name}"))?,
            aggregate_name,
        )?;
        if require_usize(aggregate, "episodes", aggregate_name)? != test_cases_evaluated {
            return Err(format!("{aggregate_name} episode count does not match test cases"));
        }
    }

    let selected_stems = manifest
        .selected_task_paths
        .iter()
        .map(|path| {
            Path::new(path)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| format!("invalid selected task path: {path}"))
                .map(str::to_string)
        })
        .collect::<Result<BTreeSet<_>, _>>()?;

    let tasks = require_array(object, "tasks", "policy report")?;
    if tasks.len() != test_cases_evaluated.saturating_mul(2) {
        return Err("policy report must contain exactly two policy results per test case".into());
    }

    let mut policies_by_problem: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut visible_by_problem: BTreeMap<String, String> = BTreeMap::new();
    let mut target_by_problem: BTreeMap<String, String> = BTreeMap::new();
    let mut observed_stems = BTreeSet::new();

    for task in tasks {
        let task = object(task, "policy task result")?;
        let problem_id = require_str(task, "problem_id", "policy task result")?.to_string();
        let stem = problem_id
            .split_once("#test-")
            .map(|(stem, _)| stem)
            .ok_or_else(|| format!("invalid ARC problem id: {problem_id}"))?;
        if !selected_stems.contains(stem) {
            return Err(format!("policy report contains task outside smoke prefix: {problem_id}"));
        }
        observed_stems.insert(stem.to_string());

        let policy_id = require_str(task, "policy_id", "policy task result")?.to_string();
        if policy_id != "canonical-order-v1"
            && policy_id != "uniform-random-without-replacement-v1"
        {
            return Err(format!("unexpected policy id: {policy_id}"));
        }
        if !policies_by_problem
            .entry(problem_id.clone())
            .or_default()
            .insert(policy_id)
        {
            return Err(format!("duplicate policy result for {problem_id}"));
        }

        if require_usize(task, "budget", "policy task result")? != config.budget {
            return Err(format!("budget mismatch for {problem_id}"));
        }
        if require_usize(task, "grammar_size", "policy task result")? != EXPECTED_GRAMMAR_SIZE {
            return Err(format!("grammar-size mismatch for {problem_id}"));
        }
        require_eq_str(
            task,
            "action_space_commitment",
            &action_space_commitment,
            "policy task result",
        )?;
        let visible = require_hex64(task, "solver_visible_commitment", "policy task result")?;
        match visible_by_problem.get(&problem_id) {
            Some(existing) if existing != &visible => {
                return Err(format!("paired policies saw different visible problem for {problem_id}"));
            }
            None => {
                visible_by_problem.insert(problem_id.clone(), visible);
            }
            _ => {}
        }
        let target = require_hex64(task, "evaluator_target_digest", "policy task result")?;
        match target_by_problem.get(&problem_id) {
            Some(existing) if existing != &target => {
                return Err(format!("paired policies have different target digest for {problem_id}"));
            }
            None => {
                target_by_problem.insert(problem_id.clone(), target);
            }
            _ => {}
        }

        if require_usize(task, "candidates_evaluated", "policy task result")? != config.budget {
            return Err(format!("candidate-evaluation count mismatch for {problem_id}"));
        }
        let actions = require_array(task, "actions", "policy task result")?;
        if actions.len() != config.budget {
            return Err(format!("action count mismatch for {problem_id}"));
        }
        let mut ids = HashSet::with_capacity(actions.len());
        for (index, action) in actions.iter().enumerate() {
            let action = object(action, "native action")?;
            if require_usize(action, "step", "native action")? != index {
                return Err(format!("non-monotonic action step in {problem_id}"));
            }
            let candidate_id = require_usize(action, "candidate_id", "native action")?;
            if candidate_id >= EXPECTED_GRAMMAR_SIZE {
                return Err(format!("out-of-range candidate id in {problem_id}"));
            }
            if !ids.insert(candidate_id) {
                return Err(format!("repeated candidate id in {problem_id}"));
            }
            require_hex64(action, "action_commitment", "native action")?;
        }
        require_hex64(task, "policy_sealed_commitment", "policy task result")?;
    }

    if observed_stems != selected_stems {
        return Err("policy report did not cover every task in the exact smoke prefix".into());
    }
    if policies_by_problem.len() != test_cases_evaluated {
        return Err("unique policy problem count does not match test_cases_evaluated".into());
    }
    let expected_policies = BTreeSet::from([
        "canonical-order-v1".to_string(),
        "uniform-random-without-replacement-v1".to_string(),
    ]);
    for (problem_id, policies) in policies_by_problem {
        if policies != expected_policies {
            return Err(format!("incomplete paired policy set for {problem_id}"));
        }
    }

    Ok(PolicyFacts {
        action_space_commitment,
        test_cases_evaluated,
    })
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))
}

fn require_array<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a Vec<Value>, String> {
    object
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} field {key} must be an array"))
}

fn require_str<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a str, String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} field {key} must be a string"))
}

fn require_eq_str(
    object: &Map<String, Value>,
    key: &str,
    expected: &str,
    label: &str,
) -> Result<(), String> {
    let actual = require_str(object, key, label)?;
    if actual != expected {
        return Err(format!("{label} field {key} is {actual:?}, expected {expected:?}"));
    }
    Ok(())
}

fn require_u64(object: &Map<String, Value>, key: &str, label: &str) -> Result<u64, String> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label} field {key} must be an unsigned integer"))
}

fn require_usize(object: &Map<String, Value>, key: &str, label: &str) -> Result<usize, String> {
    usize::try_from(require_u64(object, key, label)?)
        .map_err(|_| format!("{label} field {key} does not fit usize"))
}

fn require_hex64(
    object: &Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<String, String> {
    let value = require_str(object, key, label)?;
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{label} field {key} must be a 64-hex digest"));
    }
    Ok(value.to_ascii_lowercase())
}

fn required_env(name: &str) -> Result<String, String> {
    match env::var(name) {
        Ok(value) if !value.trim().is_empty() => Ok(value),
        _ => Err(format!("required environment variable {name} is missing")),
    }
}

fn required_full_revision(name: &str) -> Result<String, String> {
    let value = required_env(name)?;
    if value.len() != 40 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{name} must be an exact 40-hex revision"));
    }
    Ok(value.to_ascii_lowercase())
}

fn required_usize_env(name: &str) -> Result<usize, String> {
    required_env(name)?
        .parse::<usize>()
        .map_err(|err| format!("{name} must be an unsigned integer: {err}"))
}

fn required_u64_env(name: &str) -> Result<u64, String> {
    required_env(name)?
        .parse::<u64>()
        .map_err(|err| format!("{name} must be an unsigned integer: {err}"))
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_u64(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt() -> ArcSmokeReceipt {
        ArcSmokeReceipt {
            schema_version: SCHEMA_VERSION,
            receipt_domain: RECEIPT_DOMAIN.into(),
            authority: AUTHORITY.into(),
            contamination_status: CONTAMINATION.into(),
            subject_revision: "a".repeat(40),
            dataset_repository: "arcprize/ARC-AGI-2".into(),
            dataset_revision: "b".repeat(40),
            dataset_tree: "c".repeat(40),
            dataset_split: "training".into(),
            dataset_manifest_domain: EXPECTED_MANIFEST_DOMAIN.into(),
            dataset_manifest_blake3: "1".repeat(64),
            dataset_manifest_sha256: "2".repeat(64),
            dataset_manifest_file_blake3: "3".repeat(64),
            dataset_manifest_file_sha256: "4".repeat(64),
            policy_report_schema_version: EXPECTED_POLICY_SCHEMA,
            policy_configuration_id: EXPECTED_POLICY_CONFIGURATION.into(),
            action_space_version: EXPECTED_ACTION_SPACE_VERSION.into(),
            action_space_commitment: "5".repeat(64),
            candidate_grammar_size: EXPECTED_GRAMMAR_SIZE,
            task_limit: 5,
            test_cases_evaluated: 5,
            candidate_budget: 128,
            random_seed_root: 2_797_608_998,
            selected_task_paths: vec!["data/training/a.json".into()],
            policy_report_blake3: "6".repeat(64),
            policy_report_sha256: "7".repeat(64),
            receipt_blake3: String::new(),
            receipt_sha256: String::new(),
        }
    }

    #[test]
    fn receipt_commitment_detects_tampering() {
        let mut receipt = receipt();
        let canonical = receipt.canonical_bytes();
        receipt.receipt_blake3 = blake3::hash(&canonical).to_hex().to_string();
        receipt.receipt_sha256 = sha256_bytes(&canonical).unwrap();
        assert!(receipt.commitment_valid().unwrap());

        receipt.candidate_budget = 129;
        assert!(!receipt.commitment_valid().unwrap());
    }

    #[test]
    fn selected_task_paths_are_commitment_bound() {
        let mut receipt = receipt();
        let before = receipt.canonical_bytes();
        receipt.selected_task_paths.push("data/training/b.json".into());
        assert_ne!(before, receipt.canonical_bytes());
    }
}