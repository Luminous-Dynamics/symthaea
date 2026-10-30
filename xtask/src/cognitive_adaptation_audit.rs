//! Independent raw-receipt-to-score audit for confirmatory cognitive-adaptation panels.
//!
//! A hash-linked score is not enough: this command re-opens the panel, protocol, every
//! per-cell scorer input, and every replay receipt; recomputes each score; and requires
//! exact equality with the score included in the panel. A successful audit establishes
//! artifact consistency under the declared protocol, not honest execution by the runner.
use anyhow::{ensure, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
};

use crate::{
    cognitive_adaptation_panel::{self, AdaptationPanelInput, AdaptationPanelResult},
    cognitive_adaptation_utility::{self, CounterfactualUtilityInput, CounterfactualUtilityScore},
};

type CellKey = (String, u64, String);

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuditBundle {
    schema_version: u32,
    /// Panel input containing the score records to independently reproduce.
    panel_path: PathBuf,
    panel_sha256: String,
    /// Preregistered protocol artifact. Its digest should be published before outcomes are seen.
    protocol_path: PathBuf,
    protocol_sha256: String,
    /// One unique mapping for every registered dataset × seed × scenario cell.
    cells: Vec<AuditCell>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuditCell {
    dataset_id: String,
    seed: u64,
    scenario_id: String,
    /// Raw input to score-counterfactual-adaptation-utility, before scoring.
    utility_input_path: PathBuf,
    utility_input_sha256: String,
    /// Raw receipt passed to --receipt for that score.
    replay_receipt_path: PathBuf,
    replay_receipt_sha256: String,
    /// Expected run ID from the pinned replay receipt.
    run_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditedCell {
    pub dataset_id: String,
    pub seed: u64,
    pub scenario_id: String,
    pub run_id: String,
    pub utility_input_sha256: String,
    pub replay_receipt_sha256: String,
    pub receipt_validation_sha256: String,
    pub utility_result_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdaptationPanelAuditReport {
    pub result_schema_version: u32,
    /// Hash of the exact bundle-manifest bytes used for the audit.
    pub bundle_manifest_sha256: String,
    pub protocol_sha256: String,
    pub panel_input_sha256: String,
    pub audited_cell_count: usize,
    pub distinct_run_count: usize,
    pub distinct_receipt_count: usize,
    pub audited_cells: Vec<AuditedCell>,
    pub independently_recomputed_panel: AdaptationPanelResult,
    /// SHA-256 of this report with result_sha256 set to null.
    pub result_sha256: Option<String>,
    pub interpretation_guard: String,
}

pub fn evaluate_file(
    path: &Path,
    expected_bundle_sha256: &str,
) -> anyhow::Result<AdaptationPanelAuditReport> {
    validate_sha256(expected_bundle_sha256, "expected_bundle_sha256")?;
    let manifest_path = fs::canonicalize(path)
        .with_context(|| format!("resolving audit bundle {}", path.display()))?;
    let manifest_bytes = fs::read(&manifest_path)
        .with_context(|| format!("reading audit bundle {}", manifest_path.display()))?;
    let actual_bundle_sha256 = sha256_hex(&manifest_bytes);
    validate_expected_bundle_digest(&actual_bundle_sha256, expected_bundle_sha256)?;
    let bundle: AuditBundle = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("parsing audit bundle {}", manifest_path.display()))?;
    ensure!(bundle.schema_version == 1, "audit bundle schema_version must be 1");

    let root = manifest_path
        .parent()
        .context("audit bundle has no parent directory")?;
    let root = fs::canonicalize(root).context("canonicalizing audit bundle root")?;

    validate_sha256(&bundle.panel_sha256, "panel_sha256")?;
    validate_sha256(&bundle.protocol_sha256, "protocol_sha256")?;
    let panel_path = resolve_bundle_file(&root, &bundle.panel_path, "panel_path")?;
    let protocol_path = resolve_bundle_file(&root, &bundle.protocol_path, "protocol_path")?;
    let panel_bytes = fs::read(&panel_path)
        .with_context(|| format!("reading panel input {}", panel_path.display()))?;
    let protocol_bytes = fs::read(&protocol_path)
        .with_context(|| format!("reading preregistered protocol {}", protocol_path.display()))?;
    let actual_panel_sha256 = sha256_hex(&panel_bytes);
    let actual_protocol_sha256 = sha256_hex(&protocol_bytes);
    ensure!(
        actual_panel_sha256 == bundle.panel_sha256,
        "panel_sha256 does not match the raw panel input"
    );
    ensure!(
        actual_protocol_sha256 == bundle.protocol_sha256,
        "protocol_sha256 does not match the preregistered protocol bytes"
    );
    let protocol: serde_json::Value = serde_json::from_slice(&protocol_bytes)
        .context("parsing preregistered protocol JSON")?;
    let metric_contract = protocol
        .get("counterfactual_metric_schema")
        .context("protocol is missing counterfactual_metric_schema")?;
    ensure!(
        metric_contract.get("command").and_then(serde_json::Value::as_str)
            == Some("score-counterfactual-adaptation-utility")
            && metric_contract.get("metric_version").and_then(serde_json::Value::as_str)
                == Some("counterfactual-adaptation-utility-v3"),
        "protocol metric contract does not match the v3 utility scorer"
    );
    let panel_contract = protocol
        .get("panel_aggregation")
        .context("protocol is missing panel_aggregation")?;
    let grid = panel_contract
        .get("complete_grid")
        .context("protocol panel_aggregation is missing complete_grid")?;
    ensure!(
        grid.get("dataset_count").and_then(serde_json::Value::as_u64) == Some(7)
            && grid.get("seeds_per_dataset").and_then(serde_json::Value::as_u64) == Some(50)
            && grid.get("expected_cell_count").and_then(serde_json::Value::as_u64) == Some(2800),
        "protocol must declare the registered 7 × 50 × 8 = 2800 panel"
    );
    let audit_contract = protocol
        .get("independent_audit")
        .context("protocol is missing the independent_audit contract")?;
    ensure!(
        audit_contract.get("schema_version").and_then(serde_json::Value::as_u64) == Some(1)
            && audit_contract.get("command").and_then(serde_json::Value::as_str)
                == Some("audit-cognitive-adaptation-panel"),
        "protocol independent_audit contract does not match this verifier"
    );

    let panel_input: AdaptationPanelInput = serde_json::from_slice(&panel_bytes)
        .context("parsing adaptation panel input")?;
    let protocol_datasets = grid
        .get("registered_dataset_ids")
        .and_then(serde_json::Value::as_array)
        .context("protocol complete_grid must declare registered_dataset_ids")?
        .iter()
        .map(|value| {
            value.as_str()
                .context("registered_dataset_ids must contain only strings")
                .map(str::to_owned)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let protocol_dataset_set: BTreeSet<String> = protocol_datasets.iter().cloned().collect();
    ensure!(
        protocol_datasets.len() == 7 && protocol_dataset_set.len() == 7,
        "protocol must declare exactly seven unique dataset IDs"
    );
    let panel_dataset_set: BTreeSet<String> =
        panel_input.expected_dataset_ids.iter().cloned().collect();
    ensure!(
        panel_input.expected_dataset_ids.len() == 7 && panel_dataset_set.len() == 7,
        "panel must declare exactly seven unique expected dataset IDs"
    );
    ensure!(
        protocol_dataset_set == panel_dataset_set,
        "panel dataset IDs differ from the preregistered protocol"
    );
    let protocol_seeds = grid
        .get("registered_seed_ids")
        .and_then(serde_json::Value::as_array)
        .context("protocol complete_grid must declare registered_seed_ids")?
        .iter()
        .map(|value| value.as_u64().context("registered_seed_ids must contain only non-negative integers"))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let protocol_seed_set: BTreeSet<u64> = protocol_seeds.iter().copied().collect();
    ensure!(
        protocol_seeds.len() == 50 && protocol_seed_set.len() == 50,
        "protocol must declare exactly fifty unique seed IDs"
    );
    let panel_seed_set: BTreeSet<u64> =
        panel_input.expected_seed_ids.iter().copied().collect();
    ensure!(
        panel_input.expected_seed_ids.len() == 50 && panel_seed_set.len() == 50,
        "panel must declare exactly fifty unique expected seed IDs"
    );
    ensure!(
        protocol_seed_set == panel_seed_set,
        "panel seed IDs differ from the preregistered protocol"
    );
    let protocol_scenario_ids = grid
        .get("registered_scenario_ids")
        .and_then(serde_json::Value::as_array)
        .context("protocol complete_grid must declare registered_scenario_ids")?
        .iter()
        .map(|value| {
            value.as_str()
                .context("registered_scenario_ids must contain only strings")
                .map(str::to_owned)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let protocol_scenarios: BTreeSet<String> =
        protocol_scenario_ids.iter().cloned().collect();
    let panel_scenarios: BTreeSet<String> =
        panel_input.expected_scenario_ids.iter().cloned().collect();
    ensure!(
        protocol_scenario_ids.len() == 8 && protocol_scenarios.len() == 8,
        "protocol must declare exactly eight unique scenario IDs"
    );
    ensure!(
        panel_input.expected_scenario_ids.len() == 8 && panel_scenarios.len() == 8,
        "panel must declare exactly eight unique expected scenario IDs"
    );
    ensure!(
        protocol_scenarios == panel_scenarios,
        "panel scenario IDs differ from the preregistered protocol"
    );
    let preview = cognitive_adaptation_panel::evaluate(panel_input.clone())
        .context("validating panel records before independent replay audit")?;
    ensure!(
        preview.complete_coverage_gate_pass,
        "cannot issue confirmatory audit: panel coverage is incomplete"
    );
    ensure!(
        preview.receipt_linkage_gate_pass && preview.primary_comparison.is_some(),
        "cannot issue confirmatory audit: panel lacks replay-linked score coverage"
    );
    ensure!(
        bundle.cells.len() == preview.expected_run_count,
        "audit bundle must map every registered panel cell exactly once"
    );
    ensure!(
        panel_input.runs.len() == preview.expected_run_count,
        "panel input does not contain the complete registered panel"
    );

    let mut scores_by_key: BTreeMap<CellKey, &CounterfactualUtilityScore> = BTreeMap::new();
    for score in &panel_input.runs {
        let key = (score.dataset_id.clone(), score.seed, score.scenario_id.clone());
        ensure!(
            scores_by_key.insert(key.clone(), score).is_none(),
            "panel contains a duplicate dataset/seed/scenario key: {:?}",
            key
        );
    }

    let mut seen_keys = BTreeSet::new();
    let mut seen_input_paths = BTreeSet::new();
    let mut seen_receipt_paths = BTreeSet::new();
    let mut seen_input_hashes = BTreeSet::new();
    let mut seen_receipt_hashes = BTreeSet::new();
    let mut seen_run_ids = BTreeSet::new();
    let mut audited_cells = Vec::with_capacity(bundle.cells.len());

    for (index, cell) in bundle.cells.iter().enumerate() {
        ensure!(!cell.dataset_id.trim().is_empty(), "cells[{index}].dataset_id must not be empty");
        ensure!(!cell.scenario_id.trim().is_empty(), "cells[{index}].scenario_id must not be empty");
        ensure!(!cell.run_id.trim().is_empty(), "cells[{index}].run_id must not be empty");
        validate_sha256(&cell.utility_input_sha256, "utility_input_sha256")?;
        validate_sha256(&cell.replay_receipt_sha256, "replay_receipt_sha256")?;

        let key = (cell.dataset_id.clone(), cell.seed, cell.scenario_id.clone());
        ensure!(seen_keys.insert(key.clone()), "audit bundle repeats panel cell {:?}", key);
        let stored_score = scores_by_key
            .get(&key)
            .copied()
            .with_context(|| format!("audit bundle cell {:?} has no matching panel score", key))?;

        let input_path = resolve_bundle_file(&root, &cell.utility_input_path, "utility_input_path")?;
        let receipt_path = resolve_bundle_file(&root, &cell.replay_receipt_path, "replay_receipt_path")?;
        ensure!(seen_input_paths.insert(input_path.clone()), "utility input path is reused across cells");
        ensure!(seen_receipt_paths.insert(receipt_path.clone()), "replay receipt path is reused across cells");

        let input_bytes = fs::read(&input_path)
            .with_context(|| format!("reading utility input {}", input_path.display()))?;
        let receipt_bytes = fs::read(&receipt_path)
            .with_context(|| format!("reading replay receipt {}", receipt_path.display()))?;
        let input_sha256 = sha256_hex(&input_bytes);
        let receipt_sha256 = sha256_hex(&receipt_bytes);
        ensure!(
            input_sha256 == cell.utility_input_sha256,
            "cells[{index}] utility input digest mismatch"
        );
        ensure!(
            receipt_sha256 == cell.replay_receipt_sha256,
            "cells[{index}] replay receipt digest mismatch"
        );
        ensure!(
            seen_input_hashes.insert(input_sha256.clone()),
            "utility input bytes are reused across distinct panel cells"
        );
        ensure!(
            seen_receipt_hashes.insert(receipt_sha256.clone()),
            "raw replay receipt is reused across distinct panel cells"
        );

        let raw_input: CounterfactualUtilityInput = serde_json::from_slice(&input_bytes)
            .with_context(|| format!("parsing utility input for cell {:?}", key))?;
        ensure!(
            raw_input.dataset_id == cell.dataset_id
                && raw_input.seed == cell.seed
                && raw_input.scenario_id == cell.scenario_id,
            "cells[{index}] manifest identity differs from its raw scoring input"
        );

        let recomputed = cognitive_adaptation_utility::evaluate_file_with_receipt(
            &input_path,
            &receipt_path,
        )
        .with_context(|| format!("recomputing score and validating receipt for cell {:?}", key))?;
        ensure!(
            recomputed.source_replay_run_kind.as_deref() == Some("replay"),
            "cells[{index}] is not backed by an executed replay receipt"
        );
        ensure!(
            recomputed.source_replay_run_id.as_deref() == Some(cell.run_id.as_str()),
            "cells[{index}] run_id differs from the raw replay receipt"
        );
        ensure!(
            seen_run_ids.insert(cell.run_id.clone()),
            "replay run_id is reused across distinct panel cells"
        );
        ensure!(
            &recomputed == stored_score,
            "cells[{index}] independently recomputed score does not exactly match panel score"
        );

        audited_cells.push(AuditedCell {
            dataset_id: cell.dataset_id.clone(),
            seed: cell.seed,
            scenario_id: cell.scenario_id.clone(),
            run_id: cell.run_id.clone(),
            utility_input_sha256: input_sha256,
            replay_receipt_sha256: receipt_sha256,
            receipt_validation_sha256: recomputed
                .source_replay_receipt_validation_sha256
                .clone()
                .context("recomputed score omitted receipt-validation digest")?,
            utility_result_sha256: recomputed
                .result_sha256
                .clone()
                .context("recomputed score omitted result digest")?,
        });
    }
    ensure!(
        seen_keys.len() == scores_by_key.len(),
        "audit bundle does not map every panel score"
    );

    audited_cells.sort_by(|left, right| {
        (&left.dataset_id, left.seed, &left.scenario_id)
            .cmp(&(&right.dataset_id, right.seed, &right.scenario_id))
    });
    let independently_recomputed_panel = cognitive_adaptation_panel::evaluate(panel_input)
        .context("recomputing confirmatory panel comparison")?;
    ensure!(
        independently_recomputed_panel.complete_coverage_gate_pass
            && independently_recomputed_panel.receipt_linkage_gate_pass
            && independently_recomputed_panel.primary_comparison.is_some(),
        "independently recomputed panel did not satisfy its confirmatory gates"
    );

    let mut report = AdaptationPanelAuditReport {
        result_schema_version: 1,
        bundle_manifest_sha256: actual_bundle_sha256,
        protocol_sha256: actual_protocol_sha256,
        panel_input_sha256: actual_panel_sha256,
        audited_cell_count: audited_cells.len(),
        distinct_run_count: seen_run_ids.len(),
        distinct_receipt_count: seen_receipt_hashes.len(),
        audited_cells,
        independently_recomputed_panel,
        result_sha256: None,
        interpretation_guard: "This report independently reopens the protocol, panel, per-cell scoring inputs, and raw replay receipts; validates declared artifact digests; re-runs the receipt-bound scorer for every cell; requires exact equality with each stored score; and recomputes the panel estimate. A pass establishes reproducibility and consistency of the supplied artifacts, not that the runner honestly executed, that oracle access was technically sandboxed, or that the dataset represents deployment. For confirmatory use, publish the audit-bundle digest before examining held-out outcomes and retain the exact repository commit and environment.".to_owned(),
    };
    report.result_sha256 = Some(report_digest(&report)?);
    Ok(report)
}

fn resolve_bundle_file(root: &Path, relative: &Path, label: &str) -> anyhow::Result<PathBuf> {
    validate_relative_path(relative, label)?;
    let candidate = root.join(relative);
    let canonical = fs::canonicalize(&candidate)
        .with_context(|| format!("resolving bundle file {label}: {}", relative.display()))?;
    ensure!(
        canonical.starts_with(root),
        "{label} resolves outside the audit bundle root"
    );
    ensure!(canonical.is_file(), "{label} must resolve to a regular file");
    Ok(canonical)
}

fn validate_relative_path(path: &Path, label: &str) -> anyhow::Result<()> {
    ensure!(!path.as_os_str().is_empty(), "{label} must not be empty");
    ensure!(!path.is_absolute(), "{label} must be relative to the audit bundle root");
    ensure!(
        path.components().all(|part| matches!(part, Component::Normal(_))),
        "{label} must not contain '.' or '..' path components, roots, or prefixes"
    );
    Ok(())
}

fn validate_expected_bundle_digest(actual: &str, expected: &str) -> anyhow::Result<()> {
    validate_sha256(expected, "expected_bundle_sha256")?;
    ensure!(
        actual == expected,
        "audit bundle digest does not match the independently supplied preregistration digest"
    );
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> anyhow::Result<()> {
    ensure!(
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{label} must be a 64-character hexadecimal SHA-256 digest"
    );
    ensure!(value == value.to_ascii_lowercase(), "{label} must use lowercase hexadecimal");
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn report_digest(report: &AdaptationPanelAuditReport) -> anyhow::Result<String> {
    let mut copy = report.clone();
    copy.result_sha256 = None;
    Ok(sha256_hex(&serde_json::to_vec(&copy)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_absolute_and_traversal_paths() {
        assert!(validate_relative_path(Path::new("/tmp/receipt.json"), "path").is_err());
        assert!(validate_relative_path(Path::new("../receipt.json"), "path").is_err());
        assert!(validate_relative_path(Path::new("runs/../receipt.json"), "path").is_err());
        assert!(validate_relative_path(Path::new("runs/receipt.json"), "path").is_ok());
    }

    #[test]
    fn rejects_malformed_or_noncanonical_digests() {
        assert!(validate_sha256("abc", "digest").is_err());
        assert!(validate_sha256(&"A".repeat(64), "digest").is_err());
        assert!(validate_sha256(&"a".repeat(64), "digest").is_ok());
    }

    #[test]
    fn requires_an_independently_pinned_bundle_digest() {
        let expected = "a".repeat(64);
        assert!(validate_expected_bundle_digest(&expected, &expected).is_ok());
        assert!(validate_expected_bundle_digest(&"b".repeat(64), &expected).is_err());
        assert!(validate_expected_bundle_digest(&expected, "not-pinned").is_err());
    }
}
